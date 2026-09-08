"""Bridges a Hugging Face fast tokenizer into a maskforge Vocabulary.

Imports no torch or transformers at module load.
"""

from __future__ import annotations

import operator
from typing import Any

from . import _native
from .vocabulary import Vocabulary


class TokenizerBridgeError(ValueError):
    """A tokenizer or model could not be turned into a maskforge Vocabulary."""


class EosResolutionError(TokenizerBridgeError):
    """EOS token id could not be resolved unambiguously; the message names the rule that failed."""


def _strict_token_id(value: Any, source: str) -> int:
    """Accepts only a genuine non-negative integer id; never coerces a bool, float or string."""
    if isinstance(value, bool):
        raise EosResolutionError(f"{source} must be an integer token id, not a bool")
    try:
        token_id = operator.index(value)
    except TypeError as exc:
        raise EosResolutionError(
            f"{source} must be an integer token id, got {type(value).__name__}"
        ) from exc
    if token_id < 0:
        raise EosResolutionError(f"{source} must be non-negative, got {token_id}")
    return token_id


def _normalize_eos_candidate(value: Any, source: str) -> int | None:
    """Reduces one EOS source (an id, or a sequence of ids) to a single id, or raises."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        distinct = sorted({_strict_token_id(v, source) for v in value})
        if not distinct:
            return None
        if len(distinct) > 1:
            raise EosResolutionError(
                f"{source} lists {len(distinct)} distinct EOS ids {distinct}; "
                "pass eos_token_id= explicitly"
            )
        return distinct[0]
    return _strict_token_id(value, source)


def resolve_eos_token_id(
    tokenizer: Any, model: Any = None, eos_token_id: Any = None
) -> int:
    """Explicit `eos_token_id` wins; else reconciles `model.generation_config` and the tokenizer."""
    if eos_token_id is not None:
        return _strict_token_id(eos_token_id, "eos_token_id")

    from_model = None
    if model is not None and getattr(model, "generation_config", None) is not None:
        from_model = _normalize_eos_candidate(
            model.generation_config.eos_token_id, "model.generation_config.eos_token_id"
        )

    from_tokenizer = _normalize_eos_candidate(
        getattr(tokenizer, "eos_token_id", None), "tokenizer.eos_token_id"
    )

    if from_model is not None and from_tokenizer is not None and from_model != from_tokenizer:
        raise EosResolutionError(
            f"model.generation_config.eos_token_id ({from_model}) disagrees with "
            f"tokenizer.eos_token_id ({from_tokenizer}); pass eos_token_id= explicitly to override"
        )
    resolved = from_model if from_model is not None else from_tokenizer
    if resolved is None:
        raise EosResolutionError(
            "no EOS token id found on the tokenizer or model; pass eos_token_id= explicitly"
        )
    return resolved


def _fast_tokenizer_json(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise TokenizerBridgeError(
            "maskforge requires a Hugging Face fast tokenizer (backend_tokenizer is None); "
            "use AutoTokenizer.from_pretrained(..., use_fast=True)"
        )
    return backend.to_str()


def from_transformers(
    tokenizer: Any,
    model: Any = None,
    eos_token_id: Any = None,
    logits_vocab_size: int | None = None,
) -> Vocabulary:
    """Builds a Vocabulary from a fast tokenizer, resolving EOS per `resolve_eos_token_id`."""
    resolved_eos = resolve_eos_token_id(tokenizer, model=model, eos_token_id=eos_token_id)
    tokenizer_json = _fast_tokenizer_json(tokenizer)
    if logits_vocab_size is not None and not 0 <= resolved_eos < logits_vocab_size:
        raise EosResolutionError(
            f"resolved EOS id {resolved_eos} is outside logits_vocab_size {logits_vocab_size}"
        )
    raw = _native.PyVocabulary.from_tokenizer_json(
        tokenizer_json, resolved_eos, logits_vocab_size
    )
    return Vocabulary(_native.CompiledVocabulary.from_vocabulary(raw))
