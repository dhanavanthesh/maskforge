"""The Python jsonschema bridge: the complete-instance oracle, run in both directions.

For every corpus schema and every one of its instances, the verdict is computed two ways and must
agree:
  - the engine, by feeding the instance's bytes as single-byte tokens through the PyIndex, then
    asking `is_accepting`;
  - Python `jsonschema`, by validating the parsed instance against the equivalent schema.

Agreement in both directions rules out over-restriction (valid instances accepted) and
under-restriction (invalid instances rejected). The suite iterates every corpus case, and also
checks the tokenization-failure channel, duplicate-token-id rejection, and that EOS is never an
allowed content token.

Run with: uv run pytest crates/maskforge-py/tests -q
(The extension must be built first: uv run maturin develop --features python-bindings.)
"""

import json
from importlib.metadata import version

import pytest

maskforge = pytest.importorskip("maskforge")
jsonschema = pytest.importorskip("jsonschema")
from jsonschema import Draft202012Validator

EOS = 256


def byte_vocab():
    """One single-byte token per byte value 0..255; the byte value is its token id. EOS = 256."""
    return [(bytes([b]), [b]) for b in range(256)]


def engine_accepts(index, instance: bytes) -> bool:
    """Feeds `instance`'s bytes as single-byte tokens; True iff the end state accepts."""
    state = index.start()
    for b in instance:
        try:
            state = index.advance(state, b)
        except maskforge.MaskforgeError:
            return False
    return index.is_accepting(state)


def jsonschema_valid(validator, instance: bytes) -> bool:
    try:
        return validator.is_valid(json.loads(instance.decode("utf-8")))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False


@pytest.mark.parametrize("case_name", maskforge.corpus_names())
def test_engine_agrees_with_jsonschema_over_every_corpus_case(case_name):
    schema = json.loads(maskforge.corpus_json_schema(case_name))
    validator = Draft202012Validator(schema)
    index = maskforge.corpus_index(case_name, EOS, byte_vocab())
    samples = maskforge.corpus_samples(case_name)
    assert samples, f"{case_name}: corpus case has no samples"
    for instance in samples:
        raw = instance.encode("utf-8")
        engine = engine_accepts(index, raw)
        reference = jsonschema_valid(validator, raw)
        assert engine == reference, (
            f"{case_name}: instance {instance!r} engine={engine} jsonschema={reference}"
        )


def test_every_corpus_case_is_exercised():
    """The exact number of corpus cases the bridge covers, recorded for the evidence report."""
    assert len(maskforge.corpus_names()) == 14


def test_valid_instances_are_representable_and_accepted():
    """No over-restriction: a representative valid instance tokenizes into an accepted path."""
    index = maskforge.corpus_index("syn-enum-prefix", EOS, byte_vocab())
    assert engine_accepts(index, b"12"), "valid enum member '12' must be accepted"


def test_tokenization_failure_is_a_distinct_channel():
    """An empty-byte token is a VocabError (EmptyToken), never a schema rejection."""
    with pytest.raises(maskforge.MaskforgeError) as excinfo:
        maskforge.corpus_index("suite-type-boolean", 0, [(b"", [1])])
    assert excinfo.value.args[0] == "EmptyToken"


def test_duplicate_id_under_two_byte_sequences_is_rejected():
    """One id bound to two different byte sequences is ambiguous and rejected at construction."""
    with pytest.raises(maskforge.MaskforgeError) as excinfo:
        maskforge.corpus_index("suite-type-boolean", 9, [(b"a", [5]), (b"b", [5])])
    assert excinfo.value.args[0] == "MalformedTokenizer"


def test_repeated_byte_sequence_merges_its_ids():
    """The same byte sequence listed twice keeps all its ids; none is silently dropped."""
    # "tru" is a prefix of "true", so it is allowed from the boolean start state; ids 10 and 11
    # both share those bytes and must both be allowed.
    index = maskforge.corpus_index(
        "suite-type-boolean", EOS, [(b"tru", [10]), (b"tru", [11])]
    )
    allowed = index.allowed_ids(index.start())
    assert 10 in allowed and 11 in allowed


def test_eos_is_never_an_allowed_content_token():
    index = maskforge.corpus_index("suite-type-boolean", EOS, byte_vocab())
    state = index.start()
    for b in b"true":
        state = index.advance(state, b)
    assert EOS not in index.allowed_ids(state)
    assert index.eos_legal(state) == index.is_accepting(state)


def test_errors_key_on_a_stable_code_not_english_text():
    index = maskforge.corpus_index("suite-type-boolean", EOS, byte_vocab())
    with pytest.raises(maskforge.MaskforgeError) as excinfo:
        index.advance(index.start(), 999)  # 999 is not a token in the byte vocabulary
    assert excinfo.value.args[0] == "IllegalToken"


def test_records_jsonschema_version():
    assert version("jsonschema")
