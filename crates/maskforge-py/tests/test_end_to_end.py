"""Inspectable end-to-end tests with exact expected values.

The full flow under test:
    JSON schema -> reference byte engine -> vocabulary -> CompiledArtifact -> PyIndex
    -> packed u32 mask -> NumPy/Torch adapter -> final logits.

The `true` walk below asserts literal, hand-verified state/predicate/allowed-id/mask-word values at
every prefix, then applies the mask to a logits vector and checks the exact result. The remaining
cases assert engine behaviour against Python jsonschema (an independent recognizer) and against an
independent implementation of the outlines_core bit rule.
"""

import json

import numpy as np
import pytest

maskforge = pytest.importorskip("maskforge")
from jsonschema import Draft202012Validator
from maskforge.tensor_adapters import get_adapter

EOS = 256


def byte_vocab():
    return [(bytes([b]), [b]) for b in range(256)]


def boolean_index():
    return maskforge.corpus_index("suite-type-boolean", EOS, byte_vocab())


def reference_mask(logits, words, vocab_size):
    """The outlines_core bit rule: bit set -> keep the logit, bit clear -> -inf."""
    out = np.array(logits, copy=True)
    for i in range(vocab_size):
        if not ((words[i // 32] >> (i % 32)) & 1):
            out[i] = -np.inf
    return out


def engine_accepts(index, data: bytes) -> bool:
    state = index.start()
    for b in data:
        try:
            state = index.advance(state, b)
        except maskforge.MaskforgeError:
            return False
    return index.is_accepting(state)


# Hand-verified state walk for `true`, including mask words and allowed token ids.
# Only ids in 96..127 set the fourth mask word.
TRUE_WALK = [
    (None, 1, False, True, False, False, [102, 116], 1048640),
    (ord("t"), 3, False, True, False, False, [114], 262144),
    (ord("r"), 5, False, True, False, False, [117], 2097152),
    (ord("u"), 7, False, True, False, False, [101], 32),
    (ord("e"), 9, True, False, False, True, [], 0),
]


def test_true_walk_has_exact_expected_values():
    idx = boolean_index()
    assert idx.vocab_size() == 257
    assert idx.words_per_row() == 9
    state = idx.start()
    for byte, exp_state, acc, cont, dead, eos, allowed, word3 in TRUE_WALK:
        if byte is not None:
            state = idx.advance(state, byte)
        assert state == exp_state
        assert idx.is_accepting(state) == acc
        assert idx.can_continue(state) == cont
        assert idx.is_dead(state) == dead
        assert idx.eos_legal(state) == eos
        assert idx.allowed_ids(state) == allowed
        words = idx.mask_words([state], None)
        assert len(words) == 9
        expected = [0] * 9
        expected[3] = word3
        assert words == expected, f"prefix ending {byte!r}: {words} != {expected}"
        assert EOS not in allowed  # EOS is never an ordinary content token


def test_true_final_logits_are_exactly_masked():
    idx = boolean_index()
    state = idx.start()
    words = idx.mask_words([state], None)  # allows only 'f'(102) and 't'(116)
    vocab_size = idx.vocab_size()
    logits = np.arange(vocab_size, dtype=np.float32)
    out = get_adapter("numpy")(logits, words, vocab_size)
    assert out[102] == logits[102] and out[116] == logits[116]  # allowed unchanged
    disallowed = [i for i in range(vocab_size) if i not in (102, 116)]
    assert all(out[i] == -np.inf for i in disallowed)  # everything else -inf
    assert out[EOS] == -np.inf  # EOS never enabled
    # The full input reaches an accepting state, and jsonschema agrees.
    assert engine_accepts(idx, b"true")
    assert Draft202012Validator({"type": "boolean"}).is_valid(True)


@pytest.mark.parametrize(
    "data,accept",
    [
        (b"true", True),
        (b"false", True),
        (b"null", False),
        (b"xyz", False),
    ],
)
def test_boolean_accept_reject_matches_jsonschema(data, accept):
    idx = boolean_index()
    assert engine_accepts(idx, data) == accept
    if accept:
        assert Draft202012Validator({"type": "boolean"}).is_valid(json.loads(data))


def test_incomplete_prefix_is_continuable_not_accepting():
    idx = boolean_index()
    state = idx.start()
    for b in b"tru":
        state = idx.advance(state, b)
    assert not idx.is_accepting(state)
    assert idx.can_continue(state)
    assert not idx.is_dead(state)


def test_invalid_prefix_reaches_dead():
    idx = boolean_index()
    with pytest.raises(maskforge.MaskforgeError):
        idx.advance(idx.start(), ord("x"))  # 'x' starts neither true nor false


@pytest.mark.parametrize(
    "case,accepts,rejects",
    [
        ("syn-array-bounded", b"[true]", b"[]"),
        ("syn-string-pattern", b'"car"', b'"dog"'),
        ("syn-enum-prefix", b"12", b"2"),
    ],
)
def test_array_string_enum_cases(case, accepts, rejects):
    idx = maskforge.corpus_index(case, EOS, byte_vocab())
    assert engine_accepts(idx, accepts)
    assert not engine_accepts(idx, rejects)


def test_duplicate_byte_sequence_keeps_all_ids():
    idx = maskforge.corpus_index("suite-type-boolean", EOS, [(b"t", [10]), (b"t", [11])])
    allowed = idx.allowed_ids(idx.start())
    assert 10 in allowed and 11 in allowed


def test_raw_non_utf8_token_is_a_valid_payload():
    # 0xFF is a valid token byte; construction succeeds. It just does not start a boolean.
    idx = maskforge.corpus_index("suite-type-boolean", EOS, [(bytes([0xFF]), [200])])
    assert 200 not in idx.allowed_ids(idx.start())


def test_empty_token_is_rejected():
    with pytest.raises(maskforge.MaskforgeError) as e:
        maskforge.corpus_index("suite-type-boolean", 0, [(b"", [1])])
    assert e.value.args[0] == "EmptyToken"


SIZES = [0, 1, 31, 32, 33, 64, 65, 256, 257]
DTYPES = [np.float16, np.float32, np.float64]


def every_third_words(vocab_size):
    words = [0] * ((vocab_size + 31) // 32)
    for i in range(0, vocab_size, 3):
        words[i // 32] |= 1 << (i % 32)
    return words


@pytest.mark.parametrize("vocab_size", SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_numpy_adapter_matches_bit_rule(vocab_size, dtype):
    rng = np.random.default_rng(vocab_size)
    logits = rng.standard_normal(vocab_size).astype(dtype)
    words = every_third_words(vocab_size)
    got = get_adapter("numpy")(logits, words, vocab_size)
    want = reference_mask(logits, words, vocab_size)
    assert got.dtype == dtype
    assert np.array_equal(got, want, equal_nan=True)


def test_reference_comparison_is_not_vacuous():
    # A deliberately corrupted mask must be caught by the bit-rule comparison, proving the
    # adapter-vs-reference assertions above can actually fail.
    vocab_size = 65
    logits = np.zeros(vocab_size, dtype=np.float32)
    words = every_third_words(vocab_size)
    good = get_adapter("numpy")(logits, words, vocab_size)
    broken_words = list(words)
    broken_words[0] ^= 0b1  # flip token 0's allow bit
    broken = reference_mask(logits, broken_words, vocab_size)
    assert not np.array_equal(good, broken, equal_nan=True)
