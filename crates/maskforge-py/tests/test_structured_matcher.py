"""Focused production-path checks for the incremental structured matcher."""

import maskforge


EOS = 256


def vocabulary():
    return maskforge.PyVocabulary(
        EOS,
        [
            (b"{", [0]),
            (b"}", [1]),
            (b'"x"', [2]),
            (b":", [3]),
            (b"true", [4, 8]),
            (b",", [5]),
        ],
    )


def byte_vocabulary():
    return maskforge.PyVocabulary(EOS, [(bytes([byte]), [byte]) for byte in range(256)])


def commit_bytes(matcher, document):
    return all(matcher.commit_token(byte) for byte in document)


def test_resource_dynamic_ref_compiles_once_and_runs_incrementally():
    tree = (
        '{"$id":"https://example.com/tree","$dynamicAnchor":"node",'
        '"type":"object","properties":{"data":true,"children":'
        '{"type":"array","items":{"$dynamicRef":"#node"}}}}'
    )
    strict = (
        '{"$id":"https://example.com/strict-tree","$dynamicAnchor":"node",'
        '"$ref":"https://example.com/tree","unevaluatedProperties":false}'
    )
    ir = maskforge.schema_to_ir_with_resources(
        strict,
        retrieval_uri="https://example.com/strict-tree",
        resources={"https://example.com/tree": tree},
    )
    vocab = byte_vocabulary()

    accepted = maskforge.PyStructuredMatcher(ir, vocab)
    assert commit_bytes(accepted, b'{"children":[{"data":"ok"}]}')
    assert accepted.is_finished()

    rejected = maskforge.PyStructuredMatcher(ir, vocab)
    assert not commit_bytes(rejected, b'{"children":[{"misspelled":true}]}')
    assert not rejected.is_dead()


def test_incremental_matcher_masks_and_commits_without_reparsing_prefix():
    ir = maskforge.schema_to_ir(
        '{"type":"object","properties":{"x":{"type":"boolean"}},'
        '"required":["x"],"additionalProperties":false}'
    )
    matcher = maskforge.PyStructuredMatcher(ir, vocabulary())
    matcher.commit_token(0)
    matcher.commit_token(2)
    matcher.commit_token(3)
    out = bytearray(4 * ((matcher.mask_vocab_size() + 31) // 32))
    matcher.compute_mask_into(out)
    assert out[0] & (1 << 4)
    assert out[1] & 1
    assert matcher.commit_token(4)
    assert matcher.commit_token(1)
    assert matcher.is_finished()
    matcher.reset()
    assert not matcher.is_finished()


def test_matchers_from_one_ir_keep_independent_incremental_positions():
    ir = maskforge.schema_to_ir(
        '{"type":"object","properties":{"x":{"type":"boolean"}},'
        '"required":["x"],"additionalProperties":false}'
    )
    first = maskforge.PyStructuredMatcher(ir, vocabulary())
    second = maskforge.PyStructuredMatcher(ir, vocabulary())

    assert first.commit_token(0)
    assert first.commit_token(2)
    assert first.commit_token(3)
    assert first.commit_token(4)
    assert first.commit_token(1)
    assert first.is_finished()

    assert not second.is_finished()
    assert second.commit_token(0)
    assert second.commit_token(1) is False
    assert not second.is_dead()


def test_mask_into_clears_dirty_bytes_and_preserves_duplicate_ids():
    ir = maskforge.schema_to_ir(
        '{"type":"object","properties":{"x":{"type":"boolean"}},'
        '"required":["x"],"additionalProperties":false}'
    )
    matcher = maskforge.PyStructuredMatcher(ir, vocabulary())
    assert matcher.commit_token(0)
    assert matcher.commit_token(2)
    assert matcher.commit_token(3)
    out = bytearray([0xFF] * (4 * ((matcher.mask_vocab_size() + 31) // 32)))
    matcher.compute_mask_into(out)
    assert out[0] & (1 << 4)
    assert out[1] & 1
    assert not out[0] & (1 << 0)
    assert not out[0] & (1 << 1)


def test_unique_items_contains_masks_whitespace_and_commits_after_probing():
    vocabulary = maskforge.PyVocabulary(
        EOS,
        [(b"[", [0]), (b"1", [1]), (b"]", [2]), (b" ", [3]), (b" ]", [4]), (b"x", [5])],
    )
    ir = maskforge.schema_to_ir(
        '{"type":"array","items":{"type":"number"},"uniqueItems":true,'
        '"contains":{"const":1}}'
    )
    matcher = maskforge.PyStructuredMatcher(ir, vocabulary)
    assert matcher.commit_token(0)
    assert matcher.commit_token(1)
    out = bytearray([0xFF] * (4 * ((matcher.mask_vocab_size() + 31) // 32)))
    for _ in range(10):
        matcher.compute_mask_into(out)
    assert matcher.commit_token(4)
    assert matcher.is_finished()


def test_local_recursive_reference_masks_and_commits_incrementally():
    recursive_vocabulary = maskforge.PyVocabulary(
        EOS,
        [
            (b"{", [0]),
            (b"}", [1, 8]),
            (b'"next"', [2]),
            (b":", [3]),
            (b",", [4]),
            (b'{"next":', [5]),
            (b"{}", [6]),
            (b"null", [7]),
        ],
    )
    ir = maskforge.schema_to_ir(
        '{"$defs":{"node":{"type":"object","properties":{'
        '"next":{"$ref":"#/$defs/node"}},"additionalProperties":false}},'
        '"$ref":"#/$defs/node"}'
    )
    matcher = maskforge.PyStructuredMatcher(ir, recursive_vocabulary)
    assert matcher.commit_token(5)
    out = bytearray(4 * ((matcher.mask_vocab_size() + 31) // 32))
    for _ in range(100):
        matcher.compute_mask_into(out)
        assert out[0] & (1 << 0)
        assert out[0] & (1 << 6)
        assert not out[0] & (1 << 7)
    assert matcher.commit_token(6)
    assert matcher.commit_token(1)
    assert matcher.is_finished()
