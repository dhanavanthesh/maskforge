"""The compile-bind-session lifecycle: Compiler -> SchemaProgram -> BoundSchema -> Session.

Run with: uv run pytest crates/maskforge-py/tests -q
"""

import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

maskforge = pytest.importorskip("maskforge")

EOS = 256


def byte_vocab():
    return [(bytes([b]), [b]) for b in range(256)]


def mask_bit(mask, token_id):
    return bool(mask[token_id // 8] & (1 << (token_id % 8)))


def test_invalid_json_schema_text_raises_maskforge_error_not_json_decode_error():
    compiler = maskforge.Compiler()
    with pytest.raises(maskforge.MaskforgeError) as excinfo:
        compiler.compile("not json")
    assert excinfo.value.code
    assert excinfo.value.stage


def test_compile_bind_session_with_a_json_schema_string():
    compiler = maskforge.Compiler()
    program = compiler.compile(json.dumps({"type": "boolean"}))
    vocabulary = maskforge.Vocabulary.from_tokens(EOS, byte_vocab())
    bound = program.bind(vocabulary)
    session = bound.start_session()

    mask = bytearray(4 * session.mask_word_count)
    session.write_mask(mask)
    assert any(mask)

    for byte in b"true":
        session.advance(byte)
    assert session.is_accepting


def test_pydantic_knowledge_graph_root_mask_and_completion_witnesses():
    jsonschema = pytest.importorskip("jsonschema")
    pydantic = pytest.importorskip("pydantic")

    class Node(pydantic.BaseModel):
        """One entity in the graph."""

        model_config = pydantic.ConfigDict(extra="forbid")
        id: int = pydantic.Field(..., ge=1, le=9, description="Unique identifier of the node")
        label: str = pydantic.Field(..., max_length=10, description="Label of the node")

    class Edge(pydantic.BaseModel):
        """One directed relation between two entities."""

        model_config = pydantic.ConfigDict(extra="forbid")
        source: int = pydantic.Field(..., ge=1, le=9, description="Unique source of the edge")
        target: int = pydantic.Field(..., ge=1, le=9, description="Unique target of the edge")
        label: str = pydantic.Field(..., max_length=10, description="Label of the edge")

    class KnowledgeGraph(pydantic.BaseModel):
        """A graph of entities and the relations between them."""

        model_config = pydantic.ConfigDict(extra="forbid")
        nodes: list[Node] = pydantic.Field(..., min_length=1, max_length=3)
        edges: list[Edge] = pydantic.Field(..., min_length=1, max_length=2)

    schema = KnowledgeGraph.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["nodes"]["items"] == {"$ref": "#/$defs/Node"}
    assert schema["properties"]["edges"]["items"] == {"$ref": "#/$defs/Edge"}
    assert schema["required"] == ["nodes", "edges"]

    vocabulary = maskforge.Vocabulary.from_tokens(EOS, byte_vocab())
    bound = maskforge.Compiler().compile(schema).bind(vocabulary)
    root = bound.start_session()
    for byte in b'{ "':
        root.advance(byte)
    mask = bytearray(4 * root.mask_word_count)
    root.write_mask(mask)
    for byte in range(0x20, 0x7F):
        assert mask_bit(mask, byte) is (byte in b"ne\\")
    assert not mask_bit(mask, 0xEF)

    validator = jsonschema.Draft202012Validator(schema)
    witnesses = [
        b'{"nodes":[{"id":1,"label":"a"}],"edges":[{"source":1,"target":1,"label":"x"}]}',
        b'{"edges":[{"source":1,"target":1,"label":"x"}],"nodes":[{"label":"a","id":1}]}',
    ]
    for document in witnesses:
        value = json.loads(document)
        validator.validate(value)
        KnowledgeGraph.model_validate_json(document)
        session = bound.start_session()
        for byte in document:
            current = bytearray(4 * session.mask_word_count)
            session.write_mask(current)
            assert mask_bit(current, byte), (document, byte)
            session.advance(byte)
        assert session.is_accepting


def test_payment_escaped_unique_enum_has_public_completion_witnesses(monkeypatch):
    jsonschema = pytest.importorskip("jsonschema")
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = object
    transformers.AutoTokenizer = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    example = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "python"
        / "payment_instruction_showcase.py"
    )
    spec = importlib.util.spec_from_file_location("payment_instruction_showcase_test", example)
    assert spec is not None and spec.loader is not None
    payment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(payment)

    vocabulary = maskforge.Vocabulary.from_tokens(EOS, byte_vocab())
    bound = maskforge.Compiler().compile(payment.SCHEMA).bind(vocabulary)
    validator = jsonschema.Draft202012Validator(payment.SCHEMA)
    head = (
        b'{"rail":"wire","amountMinor":"500","currency":"EUR","iban":"DE1234",'
        b'"approvals":["'
    )
    spellings = [
        (b"treasury", b"risk"),
        (br"tr\u0065\u0061\u0073\u0075\u0072\u0079", b"risk"),
        (b"treasury", br"r\u0069\u0073\u006B"),
        (br"tr\u0065\u0061\u0073\u0075\u0072\u0079", br"r\u0069\u0073\u006B"),
    ]
    for treasury, risk in spellings:
        prefix = head + treasury + b'","' + risk + b'","'
        session = bound.start_session()
        for byte in prefix:
            session.advance(byte)
        mask = bytearray(4 * session.mask_word_count)
        session.write_mask(mask)
        assert mask_bit(mask, ord("o"))
        for byte in b"tr!":
            assert not mask_bit(mask, byte)

        document = prefix + b'ops"],"allocation":{"pct":"60"}}'
        validator.validate(json.loads(document))
        replay = bound.start_session()
        for index, byte in enumerate(document):
            current = bytearray(4 * replay.mask_word_count)
            replay.write_mask(current)
            assert mask_bit(current, byte), (index, byte, document[:index])
            replay.advance(byte)
        assert replay.is_accepting
    replay.advance(EOS)
    assert replay.is_stopped


def test_compile_once_bind_two_vocabularies():
    compiler = maskforge.Compiler()
    program = compiler.compile(json.dumps({"type": "boolean"}))
    vocab_a = maskforge.Vocabulary.from_tokens(EOS, byte_vocab())
    vocab_b = maskforge.Vocabulary.from_tokens(EOS + 1, byte_vocab())

    session_a = program.bind(vocab_a).start_session()
    session_b = program.bind(vocab_b).start_session()
    for byte in b"true":
        session_a.advance(byte)
    assert session_a.is_accepting
    assert not session_b.is_accepting


def test_one_binding_many_independent_sessions():
    compiler = maskforge.Compiler()
    program = compiler.compile(json.dumps({"type": "boolean"}))
    bound = program.bind(maskforge.Vocabulary.from_tokens(EOS, byte_vocab()))

    sessions = [bound.start_session() for _ in range(100)]
    for byte in b"true":
        sessions[0].advance(byte)
    assert sessions[0].is_accepting
    assert all(not s.is_accepting for s in sessions[1:])


def test_illegal_token_leaves_state_unchanged():
    compiler = maskforge.Compiler()
    program = compiler.compile(json.dumps({"type": "boolean"}))
    session = program.bind(maskforge.Vocabulary.from_tokens(EOS, byte_vocab())).start_session()
    with pytest.raises(maskforge.MaskforgeError):
        session.advance(ord("z"))
    # still usable: a legal token still advances normally afterward.
    session.advance(ord("t"))


def test_resource_aware_compile():
    compiler = maskforge.Compiler()
    program = compiler.compile(
        json.dumps({"$ref": "a.json#value"}),
        retrieval_uri="https://example.com/root.json",
        resources={"https://example.com/a.json": json.dumps({"$anchor": "value", "type": "integer"})},
    )
    session = program.bind(maskforge.Vocabulary.from_tokens(EOS, byte_vocab())).start_session()
    session.advance(ord("4"))
    assert session.is_accepting


def test_cache_clear_during_reuse_still_compiles_correctly():
    compiler = maskforge.Compiler()
    schema = json.dumps({"type": "boolean"})
    compiler.compile(schema)
    compiler.clear_caches()
    program = compiler.compile(schema)
    session = program.bind(maskforge.Vocabulary.from_tokens(EOS, byte_vocab())).start_session()
    for byte in b"false":
        session.advance(byte)
    assert session.is_accepting


def test_without_cache_still_compiles_correctly():
    compiler = maskforge.Compiler.without_cache()
    program = compiler.compile(json.dumps({"type": "boolean"}))
    assert compiler.cache_stats()[4] == 0  # entries
    session = program.bind(maskforge.Vocabulary.from_tokens(EOS, byte_vocab())).start_session()
    for byte in b"true":
        session.advance(byte)
    assert session.is_accepting


def test_sparse_wide_vocabulary():
    tokens = [(bytes([b % 256]), [b]) for b in range(0, 250_000, 997)]
    tokens.append((b"t", [ord("t")]))
    tokens.append((b"r", [ord("r")]))
    tokens.append((b"u", [ord("u")]))
    tokens.append((b"e", [ord("e")]))
    vocabulary = maskforge.Vocabulary.from_tokens(250_000, tokens, logits_vocab_size=250_001)
    compiler = maskforge.Compiler()
    program = compiler.compile(json.dumps({"type": "boolean"}))
    session = program.bind(vocabulary).start_session()
    mask = bytearray(4 * session.mask_word_count)
    session.write_mask(mask)
    for byte in b"true":
        session.advance(byte)
    assert session.is_accepting


def test_unevaluated_false_rejects_impossible_key_prefix_on_public_session_api():
    schema = {
        "type": "object",
        "required": ["method"],
        "properties": {"method": {"enum": ["card", "wallet"]}},
        "unevaluatedProperties": False,
    }
    vocabulary = maskforge.Vocabulary.from_tokens(EOS, byte_vocab())
    bound = maskforge.Compiler().compile(json.dumps(schema)).bind(vocabulary)

    session = bound.start_session()
    for byte in b'{"':
        session.advance(byte)
    mask = bytearray(4 * session.mask_word_count)
    session.write_mask(mask)
    assert mask[ord("m") // 8] & (1 << (ord("m") % 8))
    assert not mask[ord("b") // 8] & (1 << (ord("b") % 8))

    with pytest.raises(maskforge.MaskforgeError):
        session.advance(ord("b"))
    # Rejection is transactional: the same session remains at the open-key prefix.
    session.advance(ord("m"))

    for impossible in (b'{"bogus"', b'{"bogus":"x"'):
        rejected = bound.start_session()
        with pytest.raises(maskforge.MaskforgeError):
            for byte in impossible:
                rejected.advance(byte)

    accepted = bound.start_session()
    for byte in b'{"method":"card"}':
        accepted.advance(byte)
    assert accepted.is_accepting


def test_unevaluated_false_fixture_has_no_independent_bounded_bogus_completion():
    """Independent Draft 2020-12 ground truth for the finite R11 fixture.

    The candidate set is exhaustive for this schema: it requires exactly `method`, whose only
    possible valid values are the two enum members, and rejects every other property.
    """

    jsonschema = pytest.importorskip("jsonschema", reason="R11 independent oracle dependency")
    schema = {
        "type": "object",
        "required": ["method"],
        "properties": {"method": {"enum": ["card", "wallet"]}},
        "unevaluatedProperties": False,
    }
    validator = jsonschema.Draft202012Validator(schema)
    candidates = [
        {},
        {"method": "card"},
        {"method": "wallet"},
        {"method": "x"},
        {"bogus": "x"},
        {"method": "card", "bogus": "x"},
    ]
    encoded = [json.dumps(value, separators=(",", ":")) for value in candidates]
    valid = [text for value, text in zip(candidates, encoded, strict=True) if validator.is_valid(value)]
    assert valid == ['{"method":"card"}', '{"method":"wallet"}']
    for prefix, expected in [
        ('{"method', True),
        ('{"method":"card', True),
        ('{"bogus', False),
        ('{"bogus":"x', False),
    ]:
        assert any(document.startswith(prefix) for document in valid) is expected


@pytest.mark.parametrize("closure", ["additionalProperties", "unevaluatedProperties"])
@pytest.mark.parametrize("key_source", ["exact", "pattern"])
def test_public_session_rejects_duplicate_exact_and_pattern_keys(closure, key_source):
    """R11 four-cell regression: no backend may admit a duplicate key prefix."""

    schema = {"type": "object", closure: False}
    if key_source == "exact":
        schema["properties"] = {"amount": {"type": "string"}}
    else:
        schema.update(
            {
                "propertyNames": {"pattern": "^[a-z]{2,6}$"},
                "patternProperties": {"^[a-z]{2,6}$": {"type": "string"}},
                "maxProperties": 2,
            }
        )

    vocabulary = maskforge.Vocabulary.from_tokens(EOS, byte_vocab())
    bound = maskforge.Compiler().compile(json.dumps(schema)).bind(vocabulary)
    session = bound.start_session()
    for byte in b'{"amount":"x"':
        session.advance(byte)

    if key_source == "exact":
        # There is no unseen exact name, so the earliest impossible byte is the separator.
        with pytest.raises(maskforge.MaskforgeError):
            session.advance(ord(","))
        session.advance(ord("}"))
        assert session.is_accepting
        return

    for byte in b',"amoun':
        session.advance(byte)

    with pytest.raises(maskforge.MaskforgeError):
        session.advance(ord("t"))

    # Rejection is transactional: the shorter distinct key `amoun` can still close and finish.
    for byte in b'":"y"}':
        session.advance(byte)
    assert session.is_accepting


def test_public_session_rejects_impossible_pattern_key_unicode_escape():
    schema = {
        "type": "object",
        "propertyNames": {"pattern": "^[a-z]{2,6}$"},
        "patternProperties": {"^[a-z]{2,6}$": True},
        "additionalProperties": False,
    }
    vocabulary = maskforge.Vocabulary.from_tokens(EOS, byte_vocab())
    session = maskforge.Compiler().compile(json.dumps(schema)).bind(vocabulary).start_session()
    for byte in b'{"cosk\\u':
        session.advance(byte)
    with pytest.raises(maskforge.MaskforgeError):
        session.advance(ord("2"))


def test_public_session_rejects_member_separator_when_no_unseen_key_remains():
    schema = {
        "type": "object",
        "required": ["method"],
        "properties": {"method": {"enum": ["card", "wallet"]}},
        "unevaluatedProperties": False,
    }
    vocabulary = maskforge.Vocabulary.from_tokens(EOS, byte_vocab())
    bound = maskforge.Compiler().compile(json.dumps(schema)).bind(vocabulary)

    session = bound.start_session()
    for byte in b'{"method":"card"':
        session.advance(byte)
    with pytest.raises(maskforge.MaskforgeError):
        session.advance(ord(","))

    # The failed separator did not corrupt the committed state; closing remains valid.
    session.advance(ord("}"))
    assert session.is_accepting


@pytest.mark.parametrize(
    ("second_prefix", "rejected"),
    [(b'"', ord("t")), (b'"\\u007', ord("4"))],
    ids=["raw", "escaped"],
)
def test_public_session_rejects_duplicate_unique_item_before_dead_end(second_prefix, rejected):
    schema = {
        "type": "array",
        "items": {"enum": ["treasury", "ops", "risk"]},
        "contains": {"const": "treasury"},
        "uniqueItems": True,
        "minItems": 1,
        "maxItems": 3,
    }
    vocabulary = maskforge.Vocabulary.from_tokens(EOS, byte_vocab())
    session = maskforge.Compiler().compile(json.dumps(schema)).bind(vocabulary).start_session()
    prefix = b'["treasury",' + second_prefix
    for byte in prefix:
        session.advance(byte)
    with pytest.raises(maskforge.MaskforgeError):
        session.advance(rejected)


def test_public_session_complete_enum_member_cannot_start_an_extra_escape():
    schema = {
        "type": "array",
        "items": {"enum": ["treasury", "ops", "risk"]},
        "uniqueItems": True,
    }
    vocabulary = maskforge.Vocabulary.from_tokens(EOS, byte_vocab())
    session = maskforge.Compiler().compile(json.dumps(schema)).bind(vocabulary).start_session()
    for byte in b'["treasury':
        session.advance(byte)
    with pytest.raises(maskforge.MaskforgeError):
        session.advance(ord("\\"))
    session.advance(ord('"'))
    session.advance(ord("]"))
    assert session.is_accepting
