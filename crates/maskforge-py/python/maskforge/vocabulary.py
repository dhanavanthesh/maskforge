"""Vocabulary: a prepared vocabulary usable with `Compiler.compile(...).bind(vocabulary)`."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from . import _native


class Vocabulary:
    """A prepared, reusable vocabulary; build once and reuse across binds."""

    def __init__(self, native_compiled_vocabulary: Any) -> None:
        self._native = native_compiled_vocabulary

    @classmethod
    def from_tokens(
        cls,
        eos_token_id: int,
        tokens: Iterable[tuple[bytes, Sequence[int]]],
        logits_vocab_size: int | None = None,
    ) -> "Vocabulary":
        """Builds from `(token_bytes, [ids])` pairs, one entry per distinct byte string."""
        raw = _native.PyVocabulary(eos_token_id, tokens, logits_vocab_size)
        return cls(_native.CompiledVocabulary.from_vocabulary(raw))

    @classmethod
    def from_id_ordered_tokens(
        cls,
        decoded_tokens: Sequence[bytes | None],
        eos_token_id: int,
        logits_vocab_size: int | None = None,
    ) -> "Vocabulary":
        """Builds from `decoded_tokens[i]` = token id `i`'s bytes, or `None` for an absent id."""
        raw = _native.PyVocabulary.from_id_ordered_tokens(
            decoded_tokens, eos_token_id, logits_vocab_size
        )
        return cls(_native.CompiledVocabulary.from_vocabulary(raw))

    @property
    def mask_vocab_size(self) -> int:
        """This vocabulary's mask width, in tokens."""
        return self._native.mask_vocab_size()

    @property
    def eos_token_id(self) -> int:
        """This vocabulary's EOS token id."""
        return self._native.eos_token_id()

    def token_bytes(self, token_id: int) -> bytes | None:
        """Returns the exact raw bytes for one token id, or ``None`` for an absent id."""
        return self._native.token_bytes(int(token_id))
