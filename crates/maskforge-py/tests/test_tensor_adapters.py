"""Tensor-adapter tests: dtype preservation, correct masking, and input validation.

Only the NumPy backend is exercised here (torch is optional); the shared unpacking and
validation are backend-independent.
"""

import pytest

np = pytest.importorskip("numpy")

from maskforge.tensor_adapters import get_adapter, mask_words_to_bool


def packed(allowed_ids, vocab_size):
    words = [0] * ((vocab_size + 31) // 32)
    for i in allowed_ids:
        words[i // 32] |= 1 << (i % 32)
    return words


def test_unpack_matches_the_bits():
    words = packed([0, 5, 33], 40)
    flags = mask_words_to_bool(words, 40)
    assert flags[0] and flags[5] and flags[33]
    assert not flags[1] and not flags[32]


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_numpy_adapter_preserves_dtype_and_masks_only_disallowed(dtype):
    apply_mask = get_adapter("numpy")
    logits = np.arange(4, dtype=dtype)
    out = apply_mask(logits, packed([1, 3], 4), 4)
    assert out.dtype == dtype
    assert out[0] == -np.inf and out[2] == -np.inf
    assert out[1] == logits[1] and out[3] == logits[3]


def test_numpy_adapter_rejects_short_mask():
    apply_mask = get_adapter("numpy")
    with pytest.raises(ValueError):
        apply_mask(np.zeros(64, dtype=np.float32), [0], 64)


def test_numpy_adapter_rejects_wrong_logits_shape():
    apply_mask = get_adapter("numpy")
    with pytest.raises(ValueError):
        apply_mask(np.zeros((2, 4), dtype=np.float32), packed([0], 4), 4)
    with pytest.raises(ValueError):
        apply_mask(np.zeros(8, dtype=np.float32), packed([0], 4), 4)


def test_get_adapter_rejects_unknown_framework():
    with pytest.raises(KeyError):
        get_adapter("jax")


def packed_i32(allowed_ids, vocab_size):
    """Builds a packed mask as int32, safe for a set bit 31 (avoids the int32-overflow trap)."""
    return np.array(packed(allowed_ids, vocab_size), dtype=np.uint32).view(np.int32)


def test_numpy_batch_inplace_masks_each_row_independently_and_mutates_in_place():
    from maskforge.tensor_adapters.numpy import apply_token_bitmask_inplace

    vocab_size = 40
    mask = np.stack([packed_i32([0, 5, 33], vocab_size), packed_i32([1, 2], vocab_size)])
    logits = np.arange(2 * vocab_size, dtype=np.float32).reshape(2, vocab_size)
    buffer_id = id(logits)
    apply_token_bitmask_inplace(logits, mask)
    assert id(logits) == buffer_id, "must mutate the caller's array, not return a new one"
    assert logits[0, 0] != -np.inf and logits[0, 5] != -np.inf and logits[0, 33] != -np.inf
    assert logits[0, 1] == -np.inf
    assert logits[1, 1] != -np.inf and logits[1, 2] != -np.inf
    assert logits[1, 0] == -np.inf


def test_numpy_batch_inplace_masks_tail_columns_beyond_mask_width():
    from maskforge.tensor_adapters.numpy import apply_token_bitmask_inplace

    mask = packed_i32([0], 32).reshape(1, 1)
    logits = np.arange(40, dtype=np.float32).reshape(1, 40)
    apply_token_bitmask_inplace(logits, mask)
    assert logits[0, 0] != -np.inf
    assert np.all(logits[0, 32:] == -np.inf)


def test_numpy_batch_inplace_bit_31_is_read_correctly_not_sign_extended():
    from maskforge.tensor_adapters.numpy import apply_token_bitmask_inplace

    mask = packed_i32([31], 32).reshape(1, 1)
    logits = np.arange(32, dtype=np.float32).reshape(1, 32)
    apply_token_bitmask_inplace(logits, mask)
    assert logits[0, 31] != -np.inf
    assert np.all(logits[0, :31] == -np.inf)


def test_numpy_batch_inplace_rejects_dtype_ndim_and_batch_mismatch():
    from maskforge.tensor_adapters.numpy import apply_token_bitmask_inplace

    logits = np.zeros((2, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_token_bitmask_inplace(logits, np.zeros((2, 1), dtype=np.int64))
    with pytest.raises(ValueError):
        apply_token_bitmask_inplace(logits, np.zeros(1, dtype=np.int32))
    with pytest.raises(ValueError):
        apply_token_bitmask_inplace(logits, np.zeros((3, 1), dtype=np.int32))


def test_numpy_batch_inplace_rejects_integer_logits():
    from maskforge.tensor_adapters.numpy import apply_token_bitmask_inplace

    logits = np.zeros((2, 8), dtype=np.int32)
    mask = np.tile(packed_i32([0], 8), (2, 1))
    with pytest.raises(ValueError):
        apply_token_bitmask_inplace(logits, mask)


def test_numpy_batch_inplace_zero_rows_and_zero_vocab_columns():
    from maskforge.tensor_adapters.numpy import apply_token_bitmask_inplace

    zero_rows = np.zeros((0, 8), dtype=np.float32)
    apply_token_bitmask_inplace(zero_rows, np.zeros((0, 1), dtype=np.int32))
    assert zero_rows.shape == (0, 8)

    zero_cols = np.zeros((2, 0), dtype=np.float32)
    apply_token_bitmask_inplace(zero_cols, np.zeros((2, 1), dtype=np.int32))
    assert zero_cols.shape == (2, 0)


def test_numpy_batch_inplace_mask_wider_than_logits_extra_words_ignored():
    from maskforge.tensor_adapters.numpy import apply_token_bitmask_inplace

    mask = packed_i32([0, 1, 63], 64).reshape(1, 2)  # 2 words cover 64 bits
    logits = np.arange(8, dtype=np.float32).reshape(1, 8)  # only 8 columns exist
    apply_token_bitmask_inplace(logits, mask)
    assert logits[0, 0] != -np.inf and logits[0, 1] != -np.inf
    assert np.all(logits[0, 2:] == -np.inf)
