"""The typed-IR Python surface: building a frozen SchemaIR, crossing the wall as validated MFIR
bytes, and compiling it into a mask engine.

The IR is immutable across the boundary, its content address is stable through serialization, an
unsupported schema is reported (not silently compiled), and the raw-JSON path agrees with the
IR-bytes path. Generated acceptance is cross-checked against Python `jsonschema`.

Run with: uv run pytest crates/maskforge-py/tests -q
(Build first: uv run maturin develop --features python-bindings.)
"""

import json

import pytest

maskforge = pytest.importorskip("maskforge")

EOS = 256


def byte_vocab():
    return [(bytes([b]), [b]) for b in range(256)]


def engine_accepts(index, instance: bytes) -> bool:
    state = index.start()
    for b in instance:
        try:
            state = index.advance(state, b)
        except maskforge.MaskforgeError:
            return False
        if index.is_dead(state):
            return False
    return index.is_accepting(state)


def structured_accepts(schema: str, instance: bytes) -> bool:
    ir = maskforge.schema_to_ir(schema)
    matcher = maskforge.PyStructuredMatcher(ir, maskforge.PyVocabulary(EOS, byte_vocab()))
    return all(matcher.commit_token(byte) for byte in instance) and matcher.is_finished()


def session_accepts(schema: str, instance: bytes) -> bool:
    vocabulary = maskforge.PyVocabulary(EOS, byte_vocab())
    session = maskforge.ConstraintSession.from_json_schema(vocabulary, schema)
    try:
        for byte in instance:
            session.commit_token(byte)
    except maskforge.MaskforgeError:
        return False
    return session.is_finished


def test_schema_to_ir_builds_a_frozen_handle():
    ir = maskforge.schema_to_ir('{"type":"boolean"}')
    assert ir.ir_version() >= 1
    assert ir.is_supported()
    assert len(ir.canonical_hash()) == 32
    assert ir.node_count() >= 1


def test_schema_resource_api_preserves_retrieval_bases_and_is_order_independent():
    schema = '{"allOf":[{"$ref":"a.json#value"},{"$ref":"b.json#value"}]}'
    resources = {
        "https://example.com/a.json": '{"$anchor":"value","type":"integer"}',
        "https://example.com/b.json": '{"$anchor":"value","minimum":5}',
    }
    first = maskforge.schema_to_ir_with_resources(
        schema,
        retrieval_uri="https://example.com/root.json",
        resources=resources,
    )
    second = maskforge.schema_to_ir_with_resources(
        schema,
        retrieval_uri="https://example.com/root.json",
        resources=dict(reversed(list(resources.items()))),
    )
    assert first.canonical_hash() == second.canonical_hash()
    assert first.node_count() == second.node_count()


def test_schema_resource_api_reports_missing_resources_and_rejects_non_string_maps():
    with pytest.raises(maskforge.MaskforgeError) as excinfo:
        maskforge.schema_to_ir_with_resources(
            '{"$ref":"missing.json"}',
            retrieval_uri="https://example.com/root.json",
            resources={},
        )
    code, stage, _pointer, keyword, observed, _message, _limit = excinfo.value.args
    assert code == "ReferenceResolution"
    assert stage == "L1"
    assert keyword == "$ref"
    assert "missing.json" in observed

    with pytest.raises(TypeError):
        maskforge.schema_to_ir_with_resources("true", resources={1: "true"})
    with pytest.raises(TypeError):
        maskforge.schema_to_ir_with_resources("true", resources={"https://example.com/x": 1})


def test_ir_is_immutable_across_the_wall():
    ir = maskforge.schema_to_ir('{"type":"null"}')
    with pytest.raises(AttributeError):
        ir.node_count = 5  # frozen pyclass: no attribute assignment


def test_content_address_is_stable_through_serialization():
    ir = maskforge.schema_to_ir('{"type":"integer","minimum":0,"maximum":10}')
    before = ir.canonical_hash()
    back = maskforge.ir_from_wire(ir.to_wire())
    assert back.canonical_hash() == before


def test_two_builds_of_the_same_schema_share_the_content_address():
    a = maskforge.schema_to_ir('{"enum":[1,12,2]}')
    b = maskforge.schema_to_ir('{"enum":[2,1,12]}')  # enum is a set: order is normalized
    assert a.canonical_hash() == b.canonical_hash()


def test_unsupported_schema_is_reported_not_compiled():
    schema = '{"type":"string","pattern":"(?!reserved$)"}'
    ir = maskforge.schema_to_ir(schema)
    assert not ir.is_supported()
    reason, pointer = ir.diagnostics()[0]
    assert reason == "RegexAssertionUnsupported"
    assert pointer == "/pattern"
    with pytest.raises(maskforge.MaskforgeError):
        maskforge.compile_json_schema(schema, EOS, byte_vocab(), False)


def test_malformed_schema_raises():
    with pytest.raises(maskforge.MaskforgeError):
        maskforge.schema_to_ir("{")


def test_unsupported_compile_error_carries_the_full_structured_field_set():
    schema = '{"type":"string","pattern":"(?!reserved$)"}'
    with pytest.raises(maskforge.MaskforgeError) as excinfo:
        maskforge.compile_json_schema(schema, EOS, byte_vocab(), False)
    code, stage, pointer, keyword, observed, message, limit = excinfo.value.args
    assert code == "Unsupported"
    assert stage == "L2"  # compile_json_schema surfaces every Unsupported diagnostic at L2
    assert pointer == "/pattern"
    assert keyword == "pattern"
    assert isinstance(message, str) and message
    assert observed is None
    assert limit is None


def test_leading_negative_lookahead_is_supported_through_the_python_api():
    schema = '{"type":"string","pattern":"^(?!reserved$).+$"}'
    ir = maskforge.schema_to_ir(schema)
    assert ir.is_supported()
    assert structured_accepts(schema, b'"allowed"')
    assert not structured_accepts(schema, b'"reserved"')


def test_contradictory_integer_bounds_are_an_empty_language():
    schema = '{"type":"integer","minimum":5,"maximum":1}'
    ir = maskforge.schema_to_ir(schema)
    assert ir.is_supported()
    assert not structured_accepts(schema, b"1")
    assert not structured_accepts(schema, b"5")


def test_vocab_error_carries_the_offending_token_as_observed():
    with pytest.raises(maskforge.MaskforgeError) as excinfo:
        maskforge.compile_json_schema(
            '{"type":"boolean"}', EOS, [(b"", [0])] + byte_vocab(), False
        )
    code, stage, pointer, keyword, _observed, _message, _limit = excinfo.value.args
    assert code == "EmptyToken"
    assert stage == "VocabBuild"
    assert pointer is None
    assert keyword is None


def test_tampered_wire_bytes_fail_closed():
    ir = maskforge.schema_to_ir('{"type":"boolean"}')
    blob = bytearray(ir.to_wire())
    blob[-1] ^= 0x01  # corrupt the checksum
    with pytest.raises(maskforge.MaskforgeError):
        maskforge.ir_from_wire(bytes(blob))


def test_compile_ir_bytes_path_matches_raw_json_path():
    schema = '{"type":"boolean"}'
    ir = maskforge.schema_to_ir(schema)
    idx_bytes = maskforge.compile_ir(ir.to_wire(), EOS, byte_vocab())
    idx_json = maskforge.compile_json_schema(schema, EOS, byte_vocab(), False)
    for instance in (b"true", b"false", b"null", b"tru"):
        assert engine_accepts(idx_bytes, instance) == engine_accepts(idx_json, instance)


def test_end_to_end_generation_is_schema_valid():
    pytest.importorskip("jsonschema")
    from jsonschema import Draft202012Validator

    cases = [
        ('{"type":"boolean"}', [b"true", b"false"], [b"null", b"1"]),
        ('{"type":"integer","minimum":-2,"maximum":10}', [b"-2", b"0", b"10"], [b"11", b"007"]),
        ('{"enum":[1,12]}', [b"1", b"12"], [b"2", b"123"]),
    ]
    for schema, valid, invalid in cases:
        parsed = json.loads(schema)
        validator = Draft202012Validator(parsed)
        for good in valid:
            assert session_accepts(schema, good), f"{schema} should accept {good!r}"
            assert validator.is_valid(json.loads(good.decode())), "oracle disagreement"
        for bad in invalid:
            assert not session_accepts(schema, bad), f"{schema} should reject {bad!r}"


def test_type_union_is_supported_through_the_python_api():
    ir = maskforge.schema_to_ir('{"type":["string","null"]}')
    assert ir.is_supported()
    idx = maskforge.compile_json_schema('{"type":["string","null"]}', EOS, byte_vocab(), False)
    assert engine_accepts(idx, b'"x"')
    assert engine_accepts(idx, b"null")
    assert not engine_accepts(idx, b"true")


def test_type_union_with_unbounded_number_branch_is_supported_through_the_python_api():
    ir = maskforge.schema_to_ir('{"type":["number","string"]}')
    assert ir.is_supported()
    idx = maskforge.compile_json_schema('{"type":["number","string"]}', EOS, byte_vocab(), False)
    assert engine_accepts(idx, b"3.14")
    assert engine_accepts(idx, b'"x"')
    assert not engine_accepts(idx, b"true")


def test_anyof_is_supported_through_the_python_api():
    schema = '{"anyOf":[{"type":"string"},{"type":"null"}]}'
    ir = maskforge.schema_to_ir(schema)
    assert ir.is_supported()
    idx = maskforge.compile_json_schema(schema, EOS, byte_vocab(), False)
    assert engine_accepts(idx, b'"x"')
    assert engine_accepts(idx, b"null")
    assert not engine_accepts(idx, b"true")


def test_anyof_with_unsupported_pattern_branch_stays_unsupported_through_the_python_api():
    ir = maskforge.schema_to_ir(
        '{"anyOf":[{"type":"string","pattern":"(?!x)"},{"type":"null"}]}'
    )
    assert not ir.is_supported()
    reason, pointer = ir.diagnostics()[0]
    assert reason == "RegexAssertionUnsupported"
    assert pointer == "/anyOf/0/pattern"


def test_allof_is_real_intersection_through_the_python_api():
    schema = (
        '{"allOf":[{"type":"integer","minimum":0,"maximum":10},'
        '{"type":"integer","minimum":5,"maximum":20}]}'
    )
    ir = maskforge.schema_to_ir(schema)
    assert ir.is_supported()
    idx = maskforge.compile_json_schema(schema, EOS, byte_vocab(), False)
    assert engine_accepts(idx, b"7")
    assert not engine_accepts(idx, b"2")
    assert not engine_accepts(idx, b"15")


def test_allof_with_unsupported_pattern_branch_stays_unsupported_through_the_python_api():
    ir = maskforge.schema_to_ir(
        '{"allOf":[{"type":"string"},{"pattern":"(?!x)"}]}'
    )
    assert not ir.is_supported()
    reason, pointer = ir.diagnostics()[0]
    assert reason == "RegexAssertionUnsupported"
    assert pointer == "/allOf/1/pattern"


def test_oneof_is_real_exactly_one_through_the_python_api():
    schema = (
        '{"oneOf":[{"type":"integer","minimum":0,"maximum":10},'
        '{"type":"integer","minimum":5,"maximum":20}]}'
    )
    ir = maskforge.schema_to_ir(schema)
    assert ir.is_supported()
    idx = maskforge.compile_json_schema(schema, EOS, byte_vocab(), False)
    assert engine_accepts(idx, b"2")  # only the first branch matches
    assert engine_accepts(idx, b"15")  # only the second branch matches
    assert not engine_accepts(idx, b"7")  # both branches match: must be rejected


def test_oneof_with_unsupported_pattern_branch_stays_unsupported_through_the_python_api():
    ir = maskforge.schema_to_ir(
        '{"oneOf":[{"type":"string","pattern":"(?!x)"},{"type":"null"}]}'
    )
    assert not ir.is_supported()
    reason, pointer = ir.diagnostics()[0]
    assert reason == "RegexAssertionUnsupported"
    assert pointer == "/oneOf/0/pattern"


def test_not_is_supported_but_needs_the_structured_backend_through_the_python_api():
    ir = maskforge.schema_to_ir('{"not":{"type":"integer"}}')
    assert ir.is_supported()
    assert ir.requires_structured_backend()
    with pytest.raises(maskforge.MaskforgeError) as excinfo:
        maskforge.compile_json_schema(
            '{"not":{"type":"integer"}}', EOS, byte_vocab(), False
        )
    code, stage, *_ = excinfo.value.args
    assert code == "Unsupported"
    assert stage == "L3"  # a supported-but-structured schema, not an L2 diagnostic


def test_if_then_else_is_supported_but_needs_the_structured_backend():
    schema = '{"type":"integer","if":{"minimum":10},"then":{"maximum":20}}'
    ir = maskforge.schema_to_ir(schema)
    assert ir.is_supported()
    assert ir.requires_structured_backend()
    with pytest.raises(maskforge.MaskforgeError) as excinfo:
        maskforge.compile_json_schema(schema, EOS, byte_vocab(), False)
    code, stage, *_ = excinfo.value.args
    assert code == "Unsupported"
    assert stage == "L3"


def test_unbounded_number_is_supported_through_the_python_api():
    ir = maskforge.schema_to_ir('{"type":"number"}')
    assert ir.is_supported()
    idx = maskforge.compile_json_schema('{"type":"number"}', EOS, byte_vocab(), False)
    for good in (b"0", b"-3", b"3.14", b"1e9", b"-2.5e-3"):
        assert engine_accepts(idx, good), f"should accept {good!r}"
    for bad in (b"01", b".5", b"1.", b"+1", b"true"):
        assert not engine_accepts(idx, bad), f"should reject {bad!r}"


def test_integer_valued_number_bound_is_supported_through_the_python_api():
    ir = maskforge.schema_to_ir('{"type":"number","minimum":0,"maximum":10}')
    assert ir.is_supported()
    for good in (b"0", b"0.5", b"10", b"10.0", b"9.999"):
        assert structured_accepts('{"type":"number","minimum":0,"maximum":10}', good)
    for bad in (b"-0.1", b"10.1", b"11"):
        assert not structured_accepts('{"type":"number","minimum":0,"maximum":10}', bad)


def test_nonnegative_fractional_bound_is_now_supported_through_the_python_api():
    schema = '{"type":"number","minimum":0.5,"maximum":9.5}'
    ir = maskforge.schema_to_ir(schema)
    assert ir.is_supported()
    assert structured_accepts(schema, b"0.5")
    assert structured_accepts(schema, b"3.14")
    assert not structured_accepts(schema, b"0.4")
    assert not structured_accepts(schema, b"9.6")


def test_exclusive_fractional_bound_is_exact_through_the_python_api():
    schema = '{"type":"number","exclusiveMinimum":0.5}'
    ir = maskforge.schema_to_ir(schema)
    assert ir.is_supported()
    assert not structured_accepts(schema, b"0.5")
    assert structured_accepts(schema, b"0.5000000000000001")


def test_composite_const_object_accepts_whitespace_but_not_reordered_keys():
    schema = '{"const":{"foo":"bar","baz":"bax"}}'
    ir = maskforge.schema_to_ir(schema)
    assert ir.is_supported()
    idx = maskforge.compile_json_schema(schema, EOS, byte_vocab(), False)
    assert engine_accepts(idx, b'{"foo":"bar","baz":"bax"}')
    # Insignificant JSON whitespace around structural bytes is accepted; string content is not
    # touched by it.
    assert engine_accepts(idx, b'{"foo": "bar", "baz": "bax"}')
    # A reordered-key spelling of the same value is a documented regular-language boundary.
    assert not engine_accepts(idx, b'{"baz":"bax","foo":"bar"}')


def test_composite_enum_accepts_each_member_canonically():
    schema = '{"enum":[{"x":1},{"y":2},[3,4]]}'
    ir = maskforge.schema_to_ir(schema)
    assert ir.is_supported()
    idx = maskforge.compile_json_schema(schema, EOS, byte_vocab(), False)
    assert engine_accepts(idx, b'{"x":1}')
    assert engine_accepts(idx, b'{"y":2}')
    assert engine_accepts(idx, b"[3,4]")
    assert not engine_accepts(idx, b'{"x":2}')


def test_combinator_with_a_base_type_sibling_intersects_through_the_python_api():
    # oneOf intersected with a base type:integer: only integers reach the branches, and exactly
    # one branch must match. 5 is in [0,10] only; 8 is in both; "x" is not an integer.
    schema = (
        '{"type":"integer","oneOf":['
        '{"type":"integer","minimum":0,"maximum":10},'
        '{"type":"integer","minimum":6,"maximum":20}]}'
    )
    ir = maskforge.schema_to_ir(schema)
    assert ir.is_supported()
    idx = maskforge.compile_json_schema(schema, EOS, byte_vocab(), False)
    assert engine_accepts(idx, b"5")
    assert engine_accepts(idx, b"18")
    assert not engine_accepts(idx, b"8")
    assert not engine_accepts(idx, b'"x"')


def test_closed_tuple_is_supported_through_the_python_api():
    schema = '{"type":"array","prefixItems":[{"type":"integer"},{"type":"string"}],"items":false}'
    ir = maskforge.schema_to_ir(schema)
    assert ir.is_supported()
    idx = maskforge.compile_json_schema(schema, EOS, byte_vocab(), False)
    assert engine_accepts(idx, b'[1,"foo"]')
    assert engine_accepts(idx, b"[1]")  # a shorter array is valid
    assert engine_accepts(idx, b"[]")
    assert not engine_accepts(idx, b'["foo",1]')  # positions are type-checked
    assert not engine_accepts(idx, b'[1,"foo",true]')  # items:false forbids extra items


def test_tuple_with_uniform_tail_is_supported_through_the_python_api():
    schema = '{"type":"array","prefixItems":[{"type":"string"}],"items":{"type":"integer"}}'
    ir = maskforge.schema_to_ir(schema)
    assert ir.is_supported()
    idx = maskforge.compile_json_schema(schema, EOS, byte_vocab(), False)
    assert engine_accepts(idx, b'["a"]')
    assert engine_accepts(idx, b'["a",1,2,3]')
    assert not engine_accepts(idx, b'["a","b"]')  # the tail must be an integer


def test_bare_prefix_items_open_tail_needs_the_structured_backend_through_the_python_api():
    schema = '{"type":"array","prefixItems":[{"type":"integer"}]}'
    ir = maskforge.schema_to_ir(schema)
    assert ir.is_supported()
    assert ir.requires_structured_backend()
    with pytest.raises(maskforge.MaskforgeError) as excinfo:
        maskforge.compile_json_schema(schema, EOS, byte_vocab(), False)
    code, stage, *_ = excinfo.value.args
    assert code == "Unsupported"
    assert stage == "L3"


def test_typeless_schema_is_supported_but_needs_the_structured_backend_through_the_python_api():
    # No `type`/`const`/`enum`: dispatches to a union of all seven instance shapes, which always
    # includes an open object branch, so it always needs the structured backend, never the DFA.
    schema = '{"minLength":3}'
    ir = maskforge.schema_to_ir(schema)
    assert ir.is_supported()
    assert ir.requires_structured_backend()
    with pytest.raises(maskforge.MaskforgeError) as excinfo:
        maskforge.compile_json_schema(schema, EOS, byte_vocab(), False)
    code, stage, *_ = excinfo.value.args
    assert code == "Unsupported"
    assert stage == "L3"


def test_bare_boolean_schemas_are_supported_through_the_python_api():
    true_ir = maskforge.schema_to_ir("true")
    assert true_ir.is_supported()
    assert true_ir.requires_structured_backend()

    false_ir = maskforge.schema_to_ir("false")
    assert false_ir.is_supported()
    assert not false_ir.requires_structured_backend()
    idx = maskforge.compile_json_schema("false", EOS, byte_vocab(), False)
    assert not engine_accepts(idx, b"null")
    assert not engine_accepts(idx, b"1")


def test_regex_shim_matches_a_quoted_string_through_the_structured_matcher():
    ir = maskforge.regex_ir("(cat|car)")
    vocabulary = maskforge.PyVocabulary(EOS, byte_vocab())
    matcher = maskforge.PyStructuredMatcher(ir, vocabulary)
    for token in b'"cat"':
        assert matcher.commit_token(token)
    assert matcher.is_finished()

    matcher.reset()
    assert not matcher.commit_token(ord("c"))


@pytest.mark.parametrize("vocab_size", [0, 1, 31, 32, 33, 64, 65])
def test_typed_ir_path_compiles_at_vocab_size_boundaries(vocab_size):
    # Cover vocabulary sizes crossing the 32-bit packed-mask boundary.
    tokens = [(bytes([b % 256]), [b]) for b in range(vocab_size)]
    eos = vocab_size
    idx = maskforge.compile_json_schema('{"type":"boolean"}', eos, tokens, False)
    assert idx.vocab_size() >= eos + 1
    words = idx.mask_words([idx.start()], None)
    assert len(words) == idx.words_per_row()
