"""`compute_mask_into` takes a raw mutable view of a Python bytearray while holding the GIL.

The regular `mask_words_into` / `mask_word_into_state` paths already have length-rejection
coverage in test_mask_words_into.py; the structured path did not.
"""

import pytest

import maskforge

SCHEMA = '{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}'


def matcher():
    vocabulary = maskforge.PyVocabulary(
        2, [(b"true", [0]), (b"false", [1]), (b"{", [3]), (b"}", [4]), (b'"', [5])]
    )
    return maskforge.PyStructuredMatcher(maskforge.schema_to_ir(SCHEMA), vocabulary)


def exact_len(m):
    return 4 * ((m.mask_vocab_size() + 31) // 32)


@pytest.mark.parametrize("delta", [-4, -1, 1, 4, 64])
def test_a_wrong_sized_buffer_is_rejected(delta):
    m = matcher()
    size = exact_len(m) + delta
    if size < 0:
        pytest.skip("negative size is not expressible")
    with pytest.raises(maskforge.MaskforgeError):
        m.compute_mask_into(bytearray(size))


def test_an_empty_buffer_is_rejected_not_ignored():
    m = matcher()
    with pytest.raises(maskforge.MaskforgeError):
        m.compute_mask_into(bytearray())


def test_a_rejected_call_leaves_the_buffer_untouched():
    """The length check must run before any byte is written through the raw view."""
    m = matcher()
    out = bytearray(exact_len(m) + 8)
    with pytest.raises(maskforge.MaskforgeError):
        m.compute_mask_into(out)
    assert not any(out), "a rejected write must not have touched the buffer"


def test_the_exact_size_succeeds_and_fills():
    m = matcher()
    out = bytearray(exact_len(m))
    m.compute_mask_into(out)
    assert any(out), "the start position must allow at least one token"


def test_a_reused_buffer_is_fully_overwritten_between_calls():
    """Stale bits from a previous call must not survive into the next mask."""
    m = matcher()
    out = bytearray([0xFF] * exact_len(m))
    m.compute_mask_into(out)
    assert not all(byte == 0xFF for byte in out), "the buffer was not rewritten"
