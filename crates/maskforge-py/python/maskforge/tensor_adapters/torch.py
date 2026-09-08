"""PyTorch tensor adapter: set disallowed logits to -inf from a packed mask."""

import sys
import threading

from . import check_batch_shapes, check_logits_1d, mask_words_to_bool

# Per-device constants, built once and reused; rebuilding them per step is pure overhead.
_SHIFT_CACHE: dict[object, object] = {}
_BYTE_TABLE_CACHE: dict[object, object] = {}
_LOCAL = threading.local()


def _bit_shifts(device):
    import torch

    shifts = _SHIFT_CACHE.get(device)
    if shifts is None:
        shifts = torch.arange(32, device=device, dtype=torch.int32)
        _SHIFT_CACHE[device] = shifts
    return shifts


def _byte_table(device):
    """Row `b` holds, for each bit of byte `b`, True when that bit is clear (disallowed)."""
    import torch

    table = _BYTE_TABLE_CACHE.get(device)
    if table is None:
        byte_values = torch.arange(256, device=device, dtype=torch.int32).unsqueeze(-1)
        bit_index = torch.arange(8, device=device, dtype=torch.int32)
        table = ((byte_values >> bit_index) & 1).eq(0)
        _BYTE_TABLE_CACHE[device] = table
    return table


def _grown(name, device, rows, dtype, shape):
    """One grow-only buffer per thread and device, sliced down for smaller calls.

    Keying by shape would retain every batch and vocabulary width a long-running server ever
    saw. Keying by device alone bounds retention at the largest request per device.
    """
    import torch

    cache = getattr(_LOCAL, name, None)
    if cache is None:
        cache = {}
        setattr(_LOCAL, name, cache)
    key = str(device)
    buffer = cache.get(key)
    if buffer is None or buffer.shape[0] < rows:
        buffer = cache[key] = torch.empty(shape(rows), dtype=dtype, device=device)
    return buffer[:rows]


def _scratch(device, rows):
    """Thread-local unpack destination, so concurrent callers never share a buffer."""
    import torch

    return _grown("bool_bits", device, rows, torch.bool, lambda n: (n, 8))


def _indices(device, count):
    """Thread-local int64 gather indices, so the uint8 view is not widened afresh each step."""
    import torch

    return _grown("int64_indices", device, count, torch.int64, lambda n: (n,))


def _disallowed_bits(mask, batch, cutoff):
    """Unpacks packed words into a `(batch, cutoff)` bool where True means the token is masked.

    The result may alias a thread-local scratch buffer that the next call overwrites; consume it
    before calling again, or clone it.
    """
    # A byte lookup touches a quarter of the bytes a 32-wide int32 shift does, but it reinterprets
    # word memory, so it only applies on a little-endian host with contiguous storage.
    if sys.byteorder == "little" and mask.is_contiguous():
        import torch

        byte_view = mask.view(torch.uint8).reshape(-1)
        count = byte_view.numel()
        # copy_ widens uint8 into the reused int64 buffer; `.to()` would allocate every step.
        indices = _indices(mask.device, count)
        indices.copy_(byte_view)
        out = _scratch(mask.device, count)
        torch.index_select(_byte_table(mask.device), 0, indices, out=out)
        return out.reshape(batch, cutoff)
    # `>>` on a signed int32 sign-extends, but `& 1` keeps only bit k, which the sign fill never
    # reaches, so the packed words need no widening.
    return ((mask.unsqueeze(-1) >> _bit_shifts(mask.device)) & 1).eq(0).reshape(batch, cutoff)


def apply_mask(logits, mask_words, vocab_size):
    """Returns ``logits`` (a 1-D tensor of length ``vocab_size``) with disallowed positions set to
    ``-inf``, preserving the input dtype and device."""
    import torch

    check_logits_1d(tuple(logits.shape), vocab_size)
    allowed = torch.tensor(
        mask_words_to_bool(mask_words, vocab_size),
        dtype=torch.bool,
        device=logits.device,
    )
    return logits.masked_fill(~allowed, float("-inf"))


def apply_token_bitmask_inplace(logits, mask):
    """Mutates ``logits`` in place: disallowed positions become ``-inf``, preserving dtype/device.

    ``logits`` is 2-D ``(batch, vocab)`` floating; ``mask`` is a 2-D ``int32`` packed mask
    ``(batch, ceil(vocab/32))`` with the same row count, on the SAME device as ``logits`` (raises
    ``ValueError`` otherwise; move it explicitly with ``mask.to(logits.device)``). Columns beyond
    ``32 * words`` are disallowed (matches the outlines_core contract).
    """
    import torch

    if mask.dtype != torch.int32:
        raise ValueError(f"mask dtype must be torch.int32, got {mask.dtype}")
    check_batch_shapes(tuple(mask.shape), tuple(logits.shape))
    if not logits.dtype.is_floating_point:
        raise ValueError(f"logits dtype must be floating-point, got {logits.dtype}")
    if mask.device != logits.device:
        raise ValueError(
            f"mask is on device {mask.device} but logits is on device {logits.device}; "
            "move the mask explicitly (mask.to(logits.device)) before calling"
        )
    batch, mask_len = mask.shape
    n_cols = logits.shape[1]
    cutoff = 32 * mask_len
    if cutoff == 0:
        logits.fill_(float("-inf"))
        return

    disallowed = _disallowed_bits(mask, batch, cutoff)

    if cutoff >= n_cols:
        logits.masked_fill_(disallowed[:, :n_cols], float("-inf"))
    else:
        # Columns past the packed width have no bit and are disallowed by contract.
        logits[:, :cutoff].masked_fill_(disallowed, float("-inf"))
        logits[:, cutoff:] = float("-inf")
