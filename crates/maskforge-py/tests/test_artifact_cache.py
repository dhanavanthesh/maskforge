"""The compiled-artifact cache: compile_json_schema_with_vocabulary/compile_ir_with_vocabulary
reuse a previously-compiled artifact for an identical (schema, vocabulary, mask width, bind mode)
instead of rebuilding one every call. Correctness (a hit and a miss must produce bit-identical
masks) is exercised alongside the cache-traffic counters, never counters alone.

Vocabulary-neutral executables live in the bounded process-wide L1 cache. Each PyVocabulary owns
its bounded L2 artifacts, so dropping a vocabulary releases its masks while L1 can bind the same
executable to another vocabulary."""

import json

import pytest

maskforge = pytest.importorskip("maskforge")
native = pytest.importorskip("maskforge._native")

EOS = 256


def byte_vocab():
    return [(bytes([b]), [b]) for b in range(256)]


SCHEMA = json.dumps({"type": "boolean"})
OTHER_SCHEMA = json.dumps({"type": "null"})


def test_repeated_identical_schema_is_a_cache_hit():
    handle = maskforge.PyVocabulary(EOS, byte_vocab())
    maskforge.compile_json_schema_with_vocabulary(handle, SCHEMA, False)  # warm: at least one miss
    _, _, hits0, misses0, _, _ = handle.artifact_cache_stats()
    maskforge.compile_json_schema_with_vocabulary(handle, SCHEMA, False)
    _, _, hits1, misses1, _, _ = handle.artifact_cache_stats()
    assert hits1 > hits0, "an identical (schema, vocabulary, mode) recompile must be a cache hit"
    assert misses1 == misses0, "a hit must not also count as a miss"


def test_same_schema_across_three_vocabularies_compiles_one_l1_executable_and_three_l2_artifacts():
    # A unique description keeps process-wide cache counters deterministic.
    schema = json.dumps({"description": "p5-cross-vocab-l1-identity", "type": "boolean"})
    first_vocab = maskforge.PyVocabulary(EOS, byte_vocab())
    second_vocab = maskforge.PyVocabulary(EOS + 1, byte_vocab() + [(b"extra", [EOS])])
    third_vocab = maskforge.PyVocabulary(EOS + 2, byte_vocab() + [(b"third", [EOS + 1])])
    _, misses0, _, _, entries0, _ = native.executable_cache_stats()
    first = maskforge.compile_json_schema_with_vocabulary(first_vocab, schema, False)
    hits1, misses1, _, _, entries1, _ = native.executable_cache_stats()
    second = maskforge.compile_json_schema_with_vocabulary(second_vocab, schema, False)
    hits2, misses2, _, _, entries2, _ = native.executable_cache_stats()
    third = maskforge.compile_json_schema_with_vocabulary(third_vocab, schema, False)
    hits3, misses3, _, _, entries3, _ = native.executable_cache_stats()
    assert misses1 == misses0 + 1
    assert entries1 == entries0 + 1
    assert misses2 == misses1, "the second vocabulary must reuse the L1 executable"
    assert hits2 == hits1 + 1
    assert entries2 == entries1
    assert misses3 == misses2
    assert hits3 == hits2 + 1
    assert entries3 == entries2
    assert first.start() == second.start() == third.start()
    assert first_vocab.artifact_cache_stats()[3] == 1
    assert second_vocab.artifact_cache_stats()[3] == 1
    assert third_vocab.artifact_cache_stats()[3] == 1


def test_a_hit_and_a_fresh_build_produce_the_same_masks():
    handle = maskforge.PyVocabulary(EOS, byte_vocab())
    first = maskforge.compile_json_schema_with_vocabulary(handle, SCHEMA, False)
    second = maskforge.compile_json_schema_with_vocabulary(handle, SCHEMA, False)  # cache hit
    s1, s2 = first.start(), second.start()
    assert first.mask_words([s1], None) == second.mask_words([s2], None)
    assert first.allowed_ids(s1) == second.allowed_ids(s2)


def test_a_different_schema_against_the_same_vocabulary_is_a_miss():
    handle = maskforge.PyVocabulary(EOS, byte_vocab())
    maskforge.compile_json_schema_with_vocabulary(handle, SCHEMA, False)
    _, _, _, misses0, _, _ = handle.artifact_cache_stats()
    maskforge.compile_json_schema_with_vocabulary(handle, OTHER_SCHEMA, False)
    _, _, _, misses1, _, _ = handle.artifact_cache_stats()
    assert misses1 > misses0, "a distinct schema must not hit the previous schema's entry"


def test_compile_ir_with_vocabulary_shares_the_semantic_l2_entry_with_json():
    handle = maskforge.PyVocabulary(EOS, byte_vocab())
    ir = maskforge.schema_to_ir(SCHEMA, False)
    from_json = maskforge.compile_json_schema_with_vocabulary(handle, SCHEMA, False)
    _, _, _hits0, misses0, _, _ = handle.artifact_cache_stats()
    from_ir = maskforge.compile_ir_with_vocabulary(handle, ir.to_wire())
    _, _, hits1, misses1, _, _ = handle.artifact_cache_stats()
    assert misses1 == misses0, "equivalent JSON and MFIR must share the executable-based L2 key"
    assert hits1 > _hits0
    from_ir_again = maskforge.compile_ir_with_vocabulary(handle, ir.to_wire())
    _, _, hits2, _, _, _ = handle.artifact_cache_stats()
    assert hits2 > hits1, "a repeat IR compile must hit the IR path's own entry"
    s1, s2, s3 = from_json.start(), from_ir.start(), from_ir_again.start()
    assert from_json.mask_words([s1], None) == from_ir.mask_words([s2], None)
    assert from_json.mask_words([s1], None) == from_ir_again.mask_words([s3], None)


def test_equivalent_structured_ir_objects_share_one_l1_program_across_vocabularies():
    schema = json.dumps({
        "type": "object",
        "properties": {"program5_unique_structured_key": {"type": "boolean"}},
    })
    first_ir = maskforge.schema_to_ir(schema)
    second_ir = maskforge.schema_to_ir(schema)
    first_vocab = maskforge.PyVocabulary(EOS, byte_vocab())
    second_vocab = maskforge.PyVocabulary(EOS + 1, byte_vocab() + [(b"extra", [EOS])])
    _, misses0, _, _, _, _ = native.executable_cache_stats()
    maskforge.PyStructuredMatcher(first_ir, first_vocab)
    hits1, misses1, _, _, _, _ = native.executable_cache_stats()
    maskforge.PyStructuredMatcher(second_ir, second_vocab)
    hits2, misses2, _, _, _, _ = native.executable_cache_stats()
    assert misses1 == misses0 + 1
    assert misses2 == misses1
    assert hits2 == hits1 + 1


def test_two_vocabularies_never_share_a_cache_entry_even_with_the_same_fingerprint():
    # Each vocabulary owns an independent artifact cache, even with identical contents.
    # The same schema is a miss for both handles.
    tokens = byte_vocab()
    a = maskforge.PyVocabulary(EOS, tokens)
    b = maskforge.PyVocabulary(EOS, tokens)
    assert a.fingerprint() == b.fingerprint()
    maskforge.compile_json_schema_with_vocabulary(a, SCHEMA, False)
    _, _, _, a_misses0, _, _ = a.artifact_cache_stats()
    _, _, _, b_misses0, _, _ = b.artifact_cache_stats()
    maskforge.compile_json_schema_with_vocabulary(b, SCHEMA, False)
    _, _, _, a_misses1, _, _ = a.artifact_cache_stats()
    _, _, _, b_misses1, _, _ = b.artifact_cache_stats()
    assert a_misses1 == a_misses0, "compiling against b must not touch a's own cache counters"
    assert b_misses1 > b_misses0, "b's first compile is a miss on its own, separate cache"


def test_dropping_a_vocabulary_does_not_affect_another_handles_cache_stats():
    # Dropping one handle must not affect another handle's cache.
    tokens = byte_vocab()
    keep = maskforge.PyVocabulary(EOS, tokens)
    maskforge.compile_json_schema_with_vocabulary(keep, SCHEMA, False)
    _, _, _, keep_misses0, _, _ = keep.artifact_cache_stats()

    scratch = maskforge.PyVocabulary(EOS, tokens)
    maskforge.compile_json_schema_with_vocabulary(scratch, SCHEMA, False)
    maskforge.compile_json_schema_with_vocabulary(scratch, OTHER_SCHEMA, False)
    del scratch

    _, _, _, keep_misses1, _, _ = keep.artifact_cache_stats()
    assert keep_misses1 == keep_misses0, "another handle's build/drop must not touch keep's stats"


def test_explicit_logits_vocab_size_changes_the_cache_key():
    tokens = byte_vocab()
    narrow = maskforge.PyVocabulary(EOS, tokens)
    wide = maskforge.PyVocabulary(EOS, tokens, logits_vocab_size=1000)
    assert narrow.fingerprint() == wide.fingerprint(), "fingerprint must ignore mask width"
    maskforge.compile_json_schema_with_vocabulary(narrow, SCHEMA, False)
    _, _, _, narrow_misses0, _, _ = narrow.artifact_cache_stats()
    maskforge.compile_json_schema_with_vocabulary(narrow, SCHEMA, False)  # repeat: a hit
    _, _, narrow_hits1, narrow_misses1, _, _ = narrow.artifact_cache_stats()
    assert narrow_misses1 == narrow_misses0, "a repeat compile on narrow must be a hit, not a miss"
    assert narrow_hits1 > 0

    maskforge.compile_json_schema_with_vocabulary(wide, SCHEMA, False)
    _, _, _, wide_misses, _, _ = wide.artifact_cache_stats()
    assert wide_misses > 0, "a different mask width, on its own cache, is still a fresh build"


def test_clear_artifact_cache_drops_retention_and_forces_a_fresh_build():
    handle = maskforge.PyVocabulary(EOS, byte_vocab())
    maskforge.compile_json_schema_with_vocabulary(handle, SCHEMA, False)
    entries0, bytes0, _, _, _, _ = handle.artifact_cache_stats()
    assert entries0 > 0 and bytes0 > 0
    handle.clear_artifact_cache()
    entries1, bytes1, _, _, _, _ = handle.artifact_cache_stats()
    assert entries1 == 0
    assert bytes1 == 0
    maskforge.compile_json_schema_with_vocabulary(handle, SCHEMA, False)
    entries2, _, _, _, _, _ = handle.artifact_cache_stats()
    assert entries2 > 0, "the cache must still work correctly after being cleared"


def test_clear_executable_cache_drops_retention_and_forces_a_fresh_build():
    handle = maskforge.PyVocabulary(EOS, byte_vocab())
    unique_schema = json.dumps({"description": "clear-l1-identity", "type": "boolean"})
    maskforge.compile_json_schema_with_vocabulary(handle, unique_schema, False)
    entries0 = maskforge.executable_cache_stats()[4]
    assert entries0 > 0
    maskforge.clear_executable_cache()
    entries1 = maskforge.executable_cache_stats()[4]
    assert entries1 == 0
    handle.clear_artifact_cache()  # avoid an L2 hit masking whether L1 actually rebuilt
    maskforge.compile_json_schema_with_vocabulary(handle, unique_schema, False)
    misses_after = maskforge.executable_cache_stats()[1]
    assert misses_after > 0, "a schema compiled again after clear must be a fresh L1 build"
