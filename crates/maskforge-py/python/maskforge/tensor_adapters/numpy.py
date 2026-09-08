"""NumPy tensor adapter: set disallowed logits to -inf from a packed mask."""

from . import check_batch_shapes, check_logits_1d, mask_words_to_bool


def apply_mask(logits, mask_words, vocab_size):
    """Returns ``logits`` with disallowed positions set to ``-inf``, preserving the input dtype.

    ``logits`` is a 1-D array of length ``vocab_size``; ``mask_words`` is the packed ``u32`` mask.
    """
    import numpy as np

    out = np.array(logits, copy=True)  # preserve the input dtype
    check_logits_1d(out.shape, vocab_size)
    allowed = np.array(mask_words_to_bool(mask_words, vocab_size), dtype=bool)
    out[~allowed] = -np.inf
    return out


def apply_token_bitmask_inplace(logits, mask):
    """Mutates ``logits`` in place: disallowed positions become ``-inf``, preserving dtype.

    ``logits`` is 2-D ``(batch, vocab)`` floating; ``mask`` is a 2-D ``int32`` packed mask
    ``(batch, ceil(vocab/32))`` with the same row count. Columns beyond ``32 * words`` are
    disallowed (matches the outlines_core kernel contract).
    """
    import numpy as np

    if mask.dtype != np.int32:
        raise ValueError(f"mask dtype must be int32, got {mask.dtype}")
    check_batch_shapes(mask.shape, logits.shape)
    if not np.issubdtype(logits.dtype, np.floating):
        raise ValueError(f"logits dtype must be floating-point, got {logits.dtype}")
    words = mask.view(np.uint32)
    batch, mask_len = words.shape
    n_cols = logits.shape[1]
    cutoff = 32 * mask_len
    cols = min(n_cols, cutoff)
    disallowed = np.ones((batch, n_cols), dtype=bool)
    if cols > 0:
        # unpackbits beats fancy-indexing a word per column: 4-5x less peak memory (PROVENANCE.md).
        byte_view = words.view(np.uint8).reshape(batch, mask_len, 4)
        allowed_full = np.unpackbits(byte_view, axis=2, bitorder="little").reshape(batch, cutoff)
        disallowed[:, :cols] = allowed_full[:, :cols] == 0
    logits[disallowed] = -np.inf
