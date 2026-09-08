"""Every bit position of the packed mask must unpack correctly on both adapter paths.

Bit 31 is the sign bit of the int32 wire word, so it is the position a shift-based unpack gets
wrong most easily. Both the byte-lookup path and the shift fallback are checked against an
independent pure-Python unpack.
"""

import pytest

torch = pytest.importorskip("torch")

from maskforge.tensor_adapters.torch import (  # noqa: E402
    _disallowed_bits,
    apply_token_bitmask_inplace,
)


def _packed_int32(allowed_ids, words):
    """Packs `allowed_ids` into `words` little-endian int32 values, as the wire format does."""
    packed = [0] * words
    for token_id in allowed_ids:
        packed[token_id // 32] |= 1 << (token_id % 32)
    return [w - (1 << 32) if w >= (1 << 31) else w for w in packed]


def _finite_columns(logits):
    return [i for i in range(logits.shape[1]) if logits[0, i].item() != float("-inf")]


@pytest.mark.parametrize("vocab", [1, 31, 32, 33, 50, 63, 64, 96])
def test_each_single_allowed_token_survives_alone(vocab):
    words = (vocab + 31) // 32
    for allowed in range(vocab):
        mask = torch.tensor([_packed_int32([allowed], words)], dtype=torch.int32)
        logits = torch.zeros((1, vocab), dtype=torch.float32)
        apply_token_bitmask_inplace(logits, mask)
        assert _finite_columns(logits) == [allowed], f"vocab={vocab} allowed={allowed}"


def test_a_full_word_of_ones_is_the_int32_minus_one_pattern():
    mask = torch.tensor([[-1, 0]], dtype=torch.int32)
    logits = torch.zeros((1, 64), dtype=torch.float32)
    apply_token_bitmask_inplace(logits, mask)
    assert _finite_columns(logits) == list(range(32))


def test_columns_past_the_packed_width_are_disallowed():
    mask = torch.tensor([[-1]], dtype=torch.int32)
    logits = torch.zeros((1, 40), dtype=torch.float32)
    apply_token_bitmask_inplace(logits, mask)
    assert _finite_columns(logits) == list(range(32))


def test_the_shift_fallback_agrees_with_the_byte_lookup():
    """A non-contiguous mask takes the shift path; both must decide the same language."""
    words = 4
    allowed = [0, 1, 31, 32, 63, 64, 95, 127]
    packed = _packed_int32(allowed, words)

    contiguous = torch.tensor([packed], dtype=torch.int32)
    # Interleave with a filler column, then slice it away: the result is not contiguous.
    padded = torch.zeros((1, words * 2), dtype=torch.int32)
    padded[0, ::2] = contiguous[0]
    non_contiguous = padded[:, ::2]
    assert not non_contiguous.is_contiguous()

    from_lookup = _disallowed_bits(contiguous, 1, words * 32)
    from_shift = _disallowed_bits(non_contiguous, 1, words * 32)
    assert torch.equal(from_lookup, from_shift)

    expected = [bit not in allowed for bit in range(words * 32)]
    assert from_lookup[0].tolist() == expected


def test_batch_rows_are_masked_independently():
    words = 2
    mask = torch.tensor(
        [_packed_int32([3], words), _packed_int32([40], words)], dtype=torch.int32
    )
    logits = torch.zeros((2, 64), dtype=torch.float32)
    apply_token_bitmask_inplace(logits, mask)
    row0 = [i for i in range(64) if logits[0, i].item() != float("-inf")]
    row1 = [i for i in range(64) if logits[1, i].item() != float("-inf")]
    assert row0 == [3]
    assert row1 == [40]


def test_scratch_retention_is_bounded_by_the_largest_shape_not_their_sum():
    """A server seeing many shapes must retain one buffer per device, not one per shape."""
    from maskforge.tensor_adapters import torch as adapter

    # Start from a clean thread-local cache so the assertion is about this test's shapes.
    for name in ("bool_bits", "int64_indices"):
        if hasattr(adapter._LOCAL, name):
            delattr(adapter._LOCAL, name)

    widths = [8, 64, 16, 256, 32]
    for words in widths:
        mask = torch.zeros((1, words), dtype=torch.int32)
        adapter._disallowed_bits(mask, 1, words * 32)

    bool_cache = adapter._LOCAL.bool_bits
    index_cache = adapter._LOCAL.int64_indices
    assert len(bool_cache) == 1, f"one bool buffer per device, got {len(bool_cache)}"
    assert len(index_cache) == 1, f"one index buffer per device, got {len(index_cache)}"

    largest_bytes = max(widths) * 4
    assert bool_cache["cpu"].shape[0] == largest_bytes
    assert index_cache["cpu"].shape[0] == largest_bytes


def test_a_smaller_shape_after_a_larger_one_still_unpacks_correctly():
    """Slicing a grown buffer down must not leak the previous call's bits."""
    from maskforge.tensor_adapters import torch as adapter

    wide = torch.tensor([_packed_int32(list(range(64)), 4)], dtype=torch.int32)
    adapter._disallowed_bits(wide, 1, 128)

    narrow = torch.tensor([_packed_int32([5], 1)], dtype=torch.int32)
    result = adapter._disallowed_bits(narrow, 1, 32)
    assert result.shape == (1, 32)
    expected = [bit != 5 for bit in range(32)]
    assert result[0].tolist() == expected
