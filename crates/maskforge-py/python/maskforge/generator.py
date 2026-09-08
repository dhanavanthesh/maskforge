"""Reusable, batch-safe generator over a CPU Transformers model.

The schema is compiled and bound once, when the generator is constructed. Each call allocates
only per-sequence session state. torch, transformers and numpy are imported lazily.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, Sequence, TypeVar, overload

from .constraints import OutputInput, OutputSpec, normalize

if TYPE_CHECKING:
    from .compiler import BoundSchema, SchemaProgram, Session
    from .models import TransformersModel

T = TypeVar("T")

_JSON_WHITESPACE = frozenset(b" \t\r\n")
_OUTSIDE_STRING = 0
_INSIDE_STRING = 1
_AFTER_ESCAPE = 2


def _advance_json_lexical_state(
    state: int, whitespace_count: int, token: bytes, maximum: int
) -> tuple[int, int, bool]:
    """Tracks JSON strings and bounds insignificant whitespace without parsing the schema."""
    for byte in token:
        if state == _OUTSIDE_STRING:
            if byte in _JSON_WHITESPACE:
                whitespace_count += 1
                if whitespace_count > maximum:
                    return state, whitespace_count, False
                continue
            whitespace_count = 0
            if byte == ord('"'):
                state = _INSIDE_STRING
        elif state == _INSIDE_STRING:
            if byte == ord('"'):
                state = _OUTSIDE_STRING
            elif byte == ord("\\"):
                state = _AFTER_ESCAPE
        else:
            # The schema matcher validates the escape. For quote tracking, every legal one-byte
            # escape returns to the string body; later ``\\u`` hex digits cannot be quotes.
            state = _INSIDE_STRING
    return state, whitespace_count, True


class _BoundedJsonWhitespaceProcessor:
    """Generation policy that caps consecutive insignificant JSON whitespace characters."""

    def __init__(self, vocabulary: Any, maximum: int, prompt_width: int) -> None:
        self._vocabulary = vocabulary
        self._maximum = maximum
        self._prompt_width = prompt_width
        self._masks: dict[tuple[int, int], bytearray] = {}
        width = vocabulary.mask_vocab_size
        records = [vocabulary.token_bytes(token_id) for token_id in range(width)]
        for state in (_OUTSIDE_STRING, _INSIDE_STRING, _AFTER_ESCAPE):
            for count in range(maximum + 1):
                rejected = bytearray(width)
                for token_id, token in enumerate(records):
                    if token is not None and not _advance_json_lexical_state(
                        state, count, token, maximum
                    )[2]:
                        rejected[token_id] = 1
                self._masks[state, count] = rejected

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import torch

        for row_index in range(int(input_ids.shape[0])):
            state = _OUTSIDE_STRING
            count = 0
            for token_id in input_ids[row_index, self._prompt_width :].tolist():
                token = self._vocabulary.token_bytes(int(token_id))
                if token is not None:
                    state, count, _ = _advance_json_lexical_state(
                        state, count, token, self._maximum
                    )
            rejected = torch.frombuffer(self._masks[state, count], dtype=torch.bool)
            if rejected.device != scores.device:
                rejected = rejected.to(scores.device)
            scores[row_index].masked_fill_(rejected, float("-inf"))
        return scores

# Strategies that append multiple tokens or rewind cannot drive an append-only session.
# Each maps to the value meaning "not requested".
_UNSUPPORTED_STRATEGIES: dict[str, Any] = {
    "assistant_model": None,
    "prompt_lookup_num_tokens": None,
    "assistant_early_exit": None,
    "custom_generate": None,
}


class GenerationError(RuntimeError):
    """Generation stopped before the constrained value was complete."""


def _reject_unsupported(generate_kwargs: dict[str, Any]) -> None:
    """Rejects every generation mode incompatible with append-only sessions, before any work."""
    enabled = [
        name
        for name, unset in _UNSUPPORTED_STRATEGIES.items()
        if generate_kwargs.get(name, unset) is not unset
    ]
    if enabled:
        raise NotImplementedError(
            "maskforge.Generator drives append-only sessions and does not support: "
            + ", ".join(sorted(enabled))
        )
    if generate_kwargs.get("num_beams", 1) != 1:
        raise NotImplementedError("beam search is not supported by maskforge.Generator")
    if generate_kwargs.get("num_return_sequences", 1) != 1:
        raise NotImplementedError(
            "num_return_sequences != 1 is not supported by maskforge.Generator"
        )
    for reserved in ("eos_token_id", "pad_token_id"):
        if reserved in generate_kwargs:
            raise NotImplementedError(
                f"{reserved} is resolved from the vocabulary and cannot be overridden per call"
            )


class _BatchMaskProcessor:
    """Applies one allowed-token mask per row per step. Bound to a single `generate()` call.

    The persistent mask buffers are allocated once here and reused for every step. The framework
    still allocates per step - reading the newly generated ids back into Python, and torch's own
    unpack temporaries - and those are measured separately rather than claimed away.
    """

    def __init__(
        self,
        sessions: list["Session[T]"],
        mask_word_count: int,
        mask_vocab_size: int,
        prompt_width: int,
    ) -> None:
        import numpy as np
        import torch

        self._sessions = sessions
        self._mask_vocab_size = mask_vocab_size
        self._prompt_width = prompt_width
        self._committed_length = 0

        word_dtype = np.dtype("<i4")
        self._bufs = [bytearray(4 * mask_word_count) for _ in sessions]
        self._views = [np.frombuffer(buf, dtype=word_dtype) for buf in self._bufs]
        self._matrix = np.empty((len(sessions), mask_word_count), dtype=word_dtype)
        self._tensor = torch.from_numpy(self._matrix)

    def _advance(self, input_ids: Any) -> None:
        """Feeds the one newly generated token per row into its session."""
        generated_length = int(input_ids.shape[1]) - self._prompt_width
        if generated_length == self._committed_length:
            return
        if generated_length != self._committed_length + 1:
            raise NotImplementedError(
                "maskforge.Generator supports append-only decoding only "
                "(no rewind or multi-token jump)"
            )
        for session, token_id in zip(self._sessions, input_ids[:, -1].tolist()):
            if not session.is_stopped:
                session.advance(token_id)
        self._committed_length = generated_length

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        self._advance(input_ids)

        if scores.shape[1] != self._mask_vocab_size:
            raise ValueError(
                f"scores vocabulary width {scores.shape[1]} does not match the bound "
                f"maskforge width {self._mask_vocab_size}"
            )
        if scores.shape[0] != len(self._sessions):
            raise ValueError(
                f"scores batch {scores.shape[0]} does not match the session count "
                f"{len(self._sessions)}"
            )

        for session, buf, view, row in zip(
            self._sessions, self._bufs, self._views, self._matrix
        ):
            session.write_mask(buf)
            row[:] = view

        from .tensor_adapters.torch import apply_token_bitmask_inplace

        apply_token_bitmask_inplace(scores, self._tensor)
        return scores


class Generator(Generic[T]):
    """Constrains a model's output to one type. Compiles and binds once; reuse across calls.

    ```python
    generate = maskforge.Generator(model, Profile)
    first = generate("Create a profile", max_new_tokens=128)
    ```
    """

    def __init__(self, model: "TransformersModel", output_type: OutputInput[T]) -> None:
        self._model = model
        spec: "OutputSpec[T]" = normalize(output_type)
        self._program: "SchemaProgram[T]" = model.compiler.compile(spec)
        self._bound: "BoundSchema[T]" = self._program.bind(model.vocabulary)

    @property
    def output_spec(self):
        """The normalized output specification this generator constrains to."""
        return self._bound.output_spec

    def _encode(self, prompts: list[str]) -> tuple[Any, Any]:
        """Left-pads independently encoded prompts. Never mutates the caller's tokenizer."""
        import torch

        tokenizer = self._model.tokenizer
        rows = [
            tokenizer(prompt, add_special_tokens=True, return_tensors=None)["input_ids"]
            for prompt in prompts
        ]
        width = max(len(row) for row in rows)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._model.vocabulary.eos_token_id

        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            start = width - len(row)
            input_ids[index, start:] = torch.as_tensor(row, dtype=torch.long)
            attention_mask[index, start:] = 1
        return input_ids, attention_mask

    @overload
    def __call__(
        self, prompt: str, *, max_new_tokens: int = ..., do_sample: bool = ..., **kwargs: Any
    ) -> T: ...

    @overload
    def __call__(
        self,
        prompt: Sequence[str],
        *,
        max_new_tokens: int = ...,
        do_sample: bool = ...,
        **kwargs: Any,
    ) -> list[T]: ...

    def __call__(
        self,
        prompt: str | Sequence[str],
        *,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        max_json_whitespace: int | None = None,
        **generate_kwargs: Any,
    ) -> T | list[T]:
        import torch
        from transformers import LogitsProcessorList

        _reject_unsupported(generate_kwargs)

        single = isinstance(prompt, str)
        prompts: list[str] = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts or not all(isinstance(p, str) for p in prompts):
            raise TypeError("prompt must be a str or a non-empty sequence of str")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
            raise TypeError("max_new_tokens must be an integer")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if (
            max_json_whitespace is not None
            and (
                not isinstance(max_json_whitespace, int)
                or isinstance(max_json_whitespace, bool)
                or max_json_whitespace < 0
            )
        ):
            raise ValueError("max_json_whitespace must be a non-negative integer or None")

        input_ids, attention_mask = self._encode(prompts)
        prompt_width = int(input_ids.shape[1])

        bound = self._bound
        sessions = [bound.start_session() for _ in prompts]
        processor = _BatchMaskProcessor(
            sessions, bound.mask_word_count, bound.mask_vocab_size, prompt_width
        )

        supplied = list(generate_kwargs.pop("logits_processor", ()) or ())
        if max_json_whitespace is not None:
            supplied.append(
                _BoundedJsonWhitespaceProcessor(
                    self._model.vocabulary, max_json_whitespace, prompt_width
                )
            )
        supplied.append(processor)
        generate_kwargs["logits_processor"] = LogitsProcessorList(supplied)

        eos_id = self._model.vocabulary.eos_token_id
        tokenizer_pad = self._model.tokenizer.pad_token_id

        with torch.no_grad():
            output_ids = self._model.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                eos_token_id=eos_id,
                pad_token_id=eos_id if tokenizer_pad is None else tokenizer_pad,
                **generate_kwargs,
            )

        if int(output_ids.shape[0]) != len(prompts):
            raise GenerationError(
                f"model returned {int(output_ids.shape[0])} sequences for {len(prompts)} prompts"
            )

        results = [
            self._decode_row(output_ids[row, prompt_width:], row)
            for row in range(len(prompts))
        ]
        return results[0] if single else results

    def _decode_row(self, generated: Any, row: int) -> T:
        """Decodes one row and parses it, reporting an incomplete value as a maskforge error.

        Only EOS and trailing padding are dropped, by id. `skip_special_tokens` would also drop
        non-EOS special tokens the mask legitimately allowed, and tokenizer cleanup rewrites
        whitespace, so both are disabled: the decoded text must be exactly the constrained bytes.
        """
        eos_id = self._model.vocabulary.eos_token_id
        pad_id = self._model.tokenizer.pad_token_id
        ids = [int(token) for token in generated.tolist()]

        # Everything from the first EOS onward is padding for this row.
        if eos_id in ids:
            ids = ids[: ids.index(eos_id)]
        while ids and pad_id is not None and ids[-1] == pad_id:
            ids.pop()

        text = self._model.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        try:
            return self._bound.output_spec.parse_json(text)
        except Exception as exc:
            raise GenerationError(
                f"row {row} did not produce a complete value within max_new_tokens; "
                f"generated {len(text)} characters"
            ) from exc
