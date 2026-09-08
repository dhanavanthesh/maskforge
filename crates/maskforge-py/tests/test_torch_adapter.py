"""Torch backend parity, skipped where torch is not installed (e.g. the Windows dev host).

Compares the torch adapter against the documented outlines_core masking semantics (bit set -> keep,
else -inf) and checks dtype and device preservation over word-aligned and non-aligned vocab sizes.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch backend not installed")

from maskforge.tensor_adapters import get_adapter


def every_third(vocab_size):
    words = [0] * ((vocab_size + 31) // 32)
    for i in range(0, vocab_size, 3):
        words[i // 32] |= 1 << (i % 32)
    return words


def allowed_bool(words, vocab_size):
    return [bool((words[i // 32] >> (i % 32)) & 1) for i in range(vocab_size)]


@pytest.mark.parametrize("vocab_size", [0, 1, 31, 32, 33, 64, 65, 256, 257])
@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_torch_adapter_masks_and_preserves_dtype_device(vocab_size, dtype):
    tdtype = getattr(torch, dtype)
    logits = torch.arange(vocab_size, dtype=tdtype)
    words = every_third(vocab_size)
    out = get_adapter("torch")(logits, words, vocab_size)
    assert out.dtype == tdtype
    assert out.device == logits.device
    allowed = allowed_bool(words, vocab_size)
    for i in range(vocab_size):
        if allowed[i]:
            assert out[i].item() == logits[i].item()
        else:
            assert out[i].item() == float("-inf")


def test_torch_adapter_rejects_wrong_shape():
    with pytest.raises(ValueError):
        get_adapter("torch")(torch.zeros(2, 4), every_third(4), 4)


def packed_i32(allowed_ids, vocab_size):
    """Builds a packed mask as int32, safe for a set bit 31 (avoids the int32-overflow trap)."""
    words = every_third_ids(allowed_ids, vocab_size)
    return torch.tensor(words, dtype=torch.int64).to(torch.int32)


def every_third_ids(allowed_ids, vocab_size):
    words = [0] * ((vocab_size + 31) // 32)
    for i in allowed_ids:
        words[i // 32] |= 1 << (i % 32)
    return words


def test_torch_batch_inplace_masks_each_row_independently_and_mutates_in_place():
    from maskforge.tensor_adapters.torch import apply_token_bitmask_inplace

    vocab_size = 40
    mask = torch.stack(
        [packed_i32([0, 5, 33], vocab_size), packed_i32([1, 2], vocab_size)]
    )
    logits = torch.arange(2 * vocab_size, dtype=torch.float32).reshape(2, vocab_size)
    data_ptr = logits.data_ptr()
    apply_token_bitmask_inplace(logits, mask)
    assert logits.data_ptr() == data_ptr, "must mutate the caller's tensor, not return a new one"
    assert logits[0, 0] != float("-inf") and logits[0, 5] != float("-inf")
    assert logits[0, 33] != float("-inf")
    assert logits[0, 1] == float("-inf")
    assert logits[1, 1] != float("-inf") and logits[1, 2] != float("-inf")
    assert logits[1, 0] == float("-inf")


def test_torch_batch_inplace_masks_tail_columns_beyond_mask_width():
    from maskforge.tensor_adapters.torch import apply_token_bitmask_inplace

    mask = packed_i32([0], 32).reshape(1, 1)
    logits = torch.arange(40, dtype=torch.float32).reshape(1, 40)
    apply_token_bitmask_inplace(logits, mask)
    assert logits[0, 0] != float("-inf")
    assert torch.all(logits[0, 32:] == float("-inf"))


def test_torch_batch_inplace_bit_31_is_read_correctly_not_sign_extended():
    from maskforge.tensor_adapters.torch import apply_token_bitmask_inplace

    mask = packed_i32([31], 32).reshape(1, 1)
    logits = torch.arange(32, dtype=torch.float32).reshape(1, 32)
    apply_token_bitmask_inplace(logits, mask)
    assert logits[0, 31] != float("-inf")
    assert torch.all(logits[0, :31] == float("-inf"))


def test_torch_batch_inplace_rejects_dtype_ndim_and_batch_mismatch():
    from maskforge.tensor_adapters.torch import apply_token_bitmask_inplace

    logits = torch.zeros(2, 8, dtype=torch.float32)
    with pytest.raises(ValueError):
        apply_token_bitmask_inplace(logits, torch.zeros(2, 1, dtype=torch.int64))
    with pytest.raises(ValueError):
        apply_token_bitmask_inplace(logits, torch.zeros(1, dtype=torch.int32))
    with pytest.raises(ValueError):
        apply_token_bitmask_inplace(logits, torch.zeros(3, 1, dtype=torch.int32))


def test_torch_batch_inplace_rejects_integer_logits():
    from maskforge.tensor_adapters.torch import apply_token_bitmask_inplace

    logits = torch.zeros(2, 8, dtype=torch.int32)
    mask = packed_i32([0], 8).unsqueeze(0).expand(2, -1).clone()
    with pytest.raises(ValueError):
        apply_token_bitmask_inplace(logits, mask)


def test_torch_batch_inplace_device_mismatch_raises_before_any_op():
    from maskforge.tensor_adapters.torch import apply_token_bitmask_inplace

    logits = torch.zeros(2, 8, dtype=torch.float32)  # cpu
    mask = torch.zeros(2, 1, dtype=torch.int32, device="meta")  # a different device, no CUDA needed
    with pytest.raises(ValueError, match="device"):
        apply_token_bitmask_inplace(logits, mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device on this host")
def test_torch_batch_inplace_cuda_mask_and_cuda_logits_agree_with_cpu():
    from maskforge.tensor_adapters.torch import apply_token_bitmask_inplace

    vocab_size = 40
    mask_cpu = packed_i32([0, 5, 33], vocab_size).unsqueeze(0)
    logits_cpu = torch.arange(vocab_size, dtype=torch.float32).unsqueeze(0)
    apply_token_bitmask_inplace(logits_cpu, mask_cpu)

    mask_cuda = mask_cpu.clone().cuda()
    logits_cuda = torch.arange(vocab_size, dtype=torch.float32).unsqueeze(0).cuda()
    apply_token_bitmask_inplace(logits_cuda, mask_cuda)
    assert torch.equal(logits_cuda.cpu(), logits_cpu)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device on this host")
def test_torch_batch_inplace_cpu_mask_cuda_logits_rejected():
    from maskforge.tensor_adapters.torch import apply_token_bitmask_inplace

    logits = torch.zeros(1, 8, dtype=torch.float32).cuda()
    mask = torch.zeros(1, 1, dtype=torch.int32)  # cpu
    with pytest.raises(ValueError, match="device"):
        apply_token_bitmask_inplace(logits, mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device on this host")
def test_torch_batch_inplace_cuda_mask_cpu_logits_rejected():
    from maskforge.tensor_adapters.torch import apply_token_bitmask_inplace

    logits = torch.zeros(1, 8, dtype=torch.float32)  # cpu
    mask = torch.zeros(1, 1, dtype=torch.int32).cuda()
    with pytest.raises(ValueError, match="device"):
        apply_token_bitmask_inplace(logits, mask)


def test_torch_batch_inplace_zero_rows_and_zero_vocab_columns():
    from maskforge.tensor_adapters.torch import apply_token_bitmask_inplace

    zero_rows = torch.zeros(0, 8, dtype=torch.float32)
    apply_token_bitmask_inplace(zero_rows, torch.zeros(0, 1, dtype=torch.int32))
    assert zero_rows.shape == (0, 8)

    zero_cols = torch.zeros(2, 0, dtype=torch.float32)
    apply_token_bitmask_inplace(zero_cols, torch.zeros(2, 1, dtype=torch.int32))
    assert zero_cols.shape == (2, 0)


def test_torch_batch_inplace_mask_wider_than_logits_extra_words_ignored():
    from maskforge.tensor_adapters.torch import apply_token_bitmask_inplace

    mask = packed_i32([0, 1, 63], 64).reshape(1, 2)  # 2 words cover 64 bits
    logits = torch.arange(8, dtype=torch.float32).reshape(1, 8)  # only 8 columns exist
    apply_token_bitmask_inplace(logits, mask)
    assert logits[0, 0] != float("-inf") and logits[0, 1] != float("-inf")
    assert torch.all(logits[0, 2:] == float("-inf"))
