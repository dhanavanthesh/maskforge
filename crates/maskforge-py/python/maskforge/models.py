"""CPU Transformers model adapter; imports torch and transformers only when called."""

from __future__ import annotations

from typing import Any

from . import tokenizers as _tokenizers
from .compiler import Compiler
from .vocabulary import Vocabulary


class TransformersModel:
    """A decoder-only causal model with its tokenizer, resolved vocabulary, and compiler."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        vocabulary: Vocabulary,
        compiler: Compiler,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.compiler = compiler


def _logits_vocab_size(model: Any) -> int:
    """Prefers the real output-embedding width; falls back to the documented config field."""
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        return int(output_embeddings.weight.shape[0])
    return int(model.config.vocab_size)


def _require_cpu(model: Any) -> None:
    """Rejects any placement this adapter cannot drive: accelerator, meta, or sharded."""
    device_map = getattr(model, "hf_device_map", None)
    if device_map and set(map(str, device_map.values())) - {"cpu"}:
        raise TypeError(
            "maskforge's Transformers integration runs on CPU only; this model is sharded or "
            f"offloaded across {sorted(set(map(str, device_map.values())))}"
        )
    devices = {parameter.device.type for parameter in model.parameters()}
    devices |= {buffer.device.type for buffer in model.buffers()}
    if devices - {"cpu"}:
        raise TypeError(
            "maskforge's Transformers integration runs on CPU only; this model has parameters "
            f"on {sorted(devices - {'cpu'})}. Call model.to('cpu') first"
        )


def from_transformers(
    model: Any,
    tokenizer: Any,
    *,
    compiler: Compiler | None = None,
    eos_token_id: Any = None,
) -> TransformersModel:
    """Builds a `TransformersModel`: resolves EOS, builds the vocabulary once, mutates nothing."""
    if getattr(model.config, "is_encoder_decoder", False):
        raise TypeError(
            "maskforge's Transformers integration supports decoder-only models only"
        )
    _require_cpu(model)

    resolved_eos = _tokenizers.resolve_eos_token_id(
        tokenizer, model=model, eos_token_id=eos_token_id
    )
    logits_vocab_size = _logits_vocab_size(model)
    vocabulary = _tokenizers.from_transformers(
        tokenizer,
        model=model,
        eos_token_id=resolved_eos,
        logits_vocab_size=logits_vocab_size,
    )
    return TransformersModel(model, tokenizer, vocabulary, compiler or Compiler())
