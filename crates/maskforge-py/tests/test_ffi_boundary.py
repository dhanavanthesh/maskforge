"""FFI boundary tests: PyIndex.mask_words row arithmetic, the full schema-to-logits flow, and the
adapters against an independent reference masking with the documented outlines_core semantics.
"""

import json

import pytest

maskforge = pytest.importorskip("maskforge")
np = pytest.importorskip("numpy")

from maskforge.tensor_adapters import get_adapter, mask_words_to_bool

EOS = 256


def byte_vocab():
    return [(bytes([b]), [b]) for b in range(256)]


def index(vocab=None):
    return maskforge.corpus_index("suite-type-boolean", EOS, vocab or byte_vocab())



def test_mask_words_rejects_usize_max_row():
    idx = index()
    with pytest.raises(maskforge.MaskforgeError) as e:
        idx.mask_words([idx.start()], [2**64 - 1])
    assert e.value.args[0] == "ArtifactOutOfBounds"


def test_mask_words_rejects_row_whose_word_product_overflows():
    idx = index()
    # words_per_row is >= 1; a row near usize::MAX/wpr makes (row+1)*wpr overflow even though
    # row+1 alone does not.
    with pytest.raises(maskforge.MaskforgeError) as e:
        idx.mask_words([idx.start()], [2**63])
    assert e.value.args[0] == "ArtifactOutOfBounds"


def test_mask_words_rejects_mismatched_row_length():
    idx = index()
    with pytest.raises(maskforge.MaskforgeError) as e:
        idx.mask_words([idx.start()], [0, 1])
    assert e.value.args[0] == "ArtifactOutOfBounds"


def test_mask_words_allows_repeated_row_targets():
    idx = index()
    words = idx.mask_words([idx.start(), idx.start()], [0, 0])
    # Both states target row 0, so the buffer is exactly one row wide.
    assert len(words) == idx.words_per_row()


def test_mask_words_empty_states_and_empty_rows():
    idx = index()
    assert idx.mask_words([], None) == []
    assert idx.mask_words([], []) == []



def test_full_flow_schema_to_masked_logits():
    schema = json.loads(maskforge.corpus_json_schema("suite-type-boolean"))
    assert schema == {"type": "boolean"}
    idx = index()
    state = idx.start()
    for b in b"tru":  # a live, non-accepting prefix of "true"/"false"? only "true"
        state = idx.advance(state, b)
    words = idx.mask_words([state], None)
    vocab_size = idx.vocab_size()
    logits = np.zeros(vocab_size, dtype=np.float32)
    out = get_adapter("numpy")(logits, words, vocab_size)
    allowed = mask_words_to_bool(words, vocab_size)
    # After "tru", only "e" (byte 0x65) continues toward "true"; that id is allowed, others -inf.
    assert allowed[ord("e")]
    assert out[ord("e")] == 0.0
    assert out[ord("x")] == -np.inf



def reference_mask(logits, words, vocab_size):
    """The documented outlines_core kernel semantics: bit set -> keep the logit, else -inf."""
    out = np.array(logits, copy=True)
    for i in range(vocab_size):
        if not ((words[i // 32] >> (i % 32)) & 1):
            out[i] = -np.inf
    return out


@pytest.mark.parametrize("vocab_size", [0, 1, 31, 32, 33, 64, 65])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_numpy_adapter_matches_reference_over_sizes_and_dtypes(vocab_size, dtype):
    rng = np.random.default_rng(vocab_size)
    logits = rng.standard_normal(vocab_size).astype(dtype)
    # A mask allowing every third id.
    words = [0] * ((vocab_size + 31) // 32)
    for i in range(0, vocab_size, 3):
        words[i // 32] |= 1 << (i % 32)
    got = get_adapter("numpy")(logits, words, vocab_size)
    want = reference_mask(logits, words, vocab_size)
    assert got.dtype == dtype
    assert np.array_equal(got, want, equal_nan=True)
    # Final-word high bits above vocab_size never flip an in-range decision (only vocab_size ids).
    assert got.shape == (vocab_size,)
