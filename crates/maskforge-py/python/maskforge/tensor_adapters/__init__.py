"""Tensor adapters: apply a packed allowed-token mask to a framework's logits tensor.

A ``PyIndex`` produces a packed ``u32`` mask; these adapters apply it, setting the logits of
disallowed tokens to negative infinity. Each framework backend is imported on demand so the base
package depends on none of them.
"""

_BACKENDS = ("numpy", "torch")


def get_adapter(framework):
    """Returns the ``apply_mask`` callable for ``framework`` ("numpy" or "torch")."""
    if framework not in _BACKENDS:
        raise KeyError(f"unknown tensor framework {framework!r}; expected one of {_BACKENDS}")
    if framework == "numpy":
        from .numpy import apply_mask
    else:
        from .torch import apply_mask
    return apply_mask


def mask_words_to_bool(mask_words, vocab_size):
    """Unpacks a packed ``u32`` mask into a list of ``vocab_size`` booleans (allowed per token).

    Raises ``ValueError`` if ``vocab_size`` is negative or ``mask_words`` is too short to hold it.
    """
    if vocab_size < 0:
        raise ValueError(f"vocab_size must be non-negative, got {vocab_size}")
    needed = (vocab_size + 31) // 32
    if len(mask_words) < needed:
        raise ValueError(
            f"mask has {len(mask_words)} words but vocab_size {vocab_size} needs {needed}"
        )
    return [bool((mask_words[i // 32] >> (i % 32)) & 1) for i in range(vocab_size)]


def check_logits_1d(shape, vocab_size):
    """Raises ``ValueError`` unless ``shape`` is 1-D of length ``vocab_size``."""
    if len(shape) != 1 or shape[0] != vocab_size:
        raise ValueError(f"logits must be 1-D of length {vocab_size}, got shape {tuple(shape)}")


def check_batch_shapes(mask_shape, logits_shape):
    """Raises ``ValueError`` unless the batched mask/logits shapes line up.

    ``mask`` must be 2-D ``(batch, words)``; ``logits`` must be 2-D with a matching batch size in
    dimension 0. ``logits`` may have more columns than ``32 * words``; the caller masks the tail.
    Dtype is checked separately, using each framework's own dtype objects (a string comparison
    like ``str(dtype)`` is not a stable cross-version contract).
    """
    if len(mask_shape) != 2:
        raise ValueError(f"mask must be 2-D (batch, words), got shape {tuple(mask_shape)}")
    if len(logits_shape) != 2:
        raise ValueError(f"logits must be 2-D (batch, vocab), got shape {tuple(logits_shape)}")
    if mask_shape[0] != logits_shape[0]:
        raise ValueError(
            f"batch size mismatch: mask has {mask_shape[0]} rows, logits has {logits_shape[0]}"
        )
