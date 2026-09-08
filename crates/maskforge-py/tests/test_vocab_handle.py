"""The reusable PyVocabulary handle: reuse produces the same masks as the rebuild path, the handle
is immutable and deterministic, and it rejects the same malformed tokenizer metadata the rebuild
path does. These cover the additive handle API without touching the existing compile_json_schema
behavior (also exercised, to prove it still works)."""

import json

import pytest

maskforge = pytest.importorskip("maskforge")

EOS = 256


def byte_vocab():
    return [(bytes([b]), [b]) for b in range(256)]


SCHEMA = json.dumps({"type": "boolean"})


def test_reuse_matches_rebuild_bit_for_bit():
    tokens = byte_vocab()
    handle = maskforge.PyVocabulary(EOS, tokens)
    reused = maskforge.compile_json_schema_with_vocabulary(handle, SCHEMA, False)
    rebuilt = maskforge.compile_json_schema(SCHEMA, EOS, tokens, False)
    assert reused.vocab_size() == rebuilt.vocab_size()
    assert reused.words_per_row() == rebuilt.words_per_row()
    s = reused.start()
    assert reused.mask_words([s], None) == rebuilt.mask_words([rebuilt.start()], None)
    assert reused.allowed_ids(s) == rebuilt.allowed_ids(rebuilt.start())


def test_one_handle_reused_across_many_schemas():
    handle = maskforge.PyVocabulary(EOS, byte_vocab())
    for schema in ({"type": "boolean"}, {"enum": ["yes", "no"]}, {"type": "null"}):
        idx = maskforge.compile_json_schema_with_vocabulary(handle, json.dumps(schema), True)
        assert idx.vocab_size() == EOS + 1


def test_handle_is_immutable_and_frozen():
    handle = maskforge.PyVocabulary(EOS, byte_vocab())
    # frozen pyclass: attribute assignment is rejected.
    with pytest.raises((AttributeError, TypeError)):
        handle.eos = 1
    # queries do not change identity
    assert handle.eos_token_id() == EOS
    assert handle.len() == 257
    assert not handle.is_empty()


def test_fingerprint_is_deterministic_and_order_independent():
    tokens = [(b"a", [1]), (b"bb", [2, 3]), (b"c", [4])]
    fp1 = maskforge.PyVocabulary(5, tokens).fingerprint()
    fp2 = maskforge.PyVocabulary(5, list(reversed(tokens))).fingerprint()
    assert isinstance(fp1, (bytes, bytearray))
    assert len(fp1) == 32
    assert fp1 == fp2, "same tokens in any order must fingerprint identically"
    # a different EOS or a different token set changes the fingerprint
    assert fp1 != maskforge.PyVocabulary(6, tokens).fingerprint()
    assert fp1 != maskforge.PyVocabulary(5, tokens + [(b"d", [7])]).fingerprint()


def test_handle_rejects_empty_token():
    with pytest.raises(maskforge.MaskforgeError) as e:
        maskforge.PyVocabulary(5, [(b"", [1])])
    assert e.value.args[0] == "EmptyToken"


def test_handle_rejects_eos_in_map():
    with pytest.raises(maskforge.MaskforgeError) as e:
        maskforge.PyVocabulary(3, [(b"x", [3])])
    assert e.value.args[0] == "MalformedTokenizer"


def test_handle_rejects_one_id_under_two_byte_sequences():
    with pytest.raises(maskforge.MaskforgeError) as e:
        maskforge.PyVocabulary(9, [(b"a", [5]), (b"b", [5])])
    assert e.value.args[0] == "MalformedTokenizer"


def test_compile_ir_with_vocabulary_matches_json_path():
    handle = maskforge.PyVocabulary(EOS, byte_vocab())
    ir = maskforge.schema_to_ir(SCHEMA, False)
    from_ir = maskforge.compile_ir_with_vocabulary(handle, ir.to_wire())
    from_json = maskforge.compile_json_schema_with_vocabulary(handle, SCHEMA, False)
    s = from_ir.start()
    assert from_ir.mask_words([s], None) == from_json.mask_words([from_json.start()], None)


def test_two_handles_for_the_same_vocabulary_share_the_process_wide_trie_cache():
    # Equivalent vocabulary handles share the process-wide byte-trie cache.
    # The second bind reuses the existing trie.
    tokens = byte_vocab()
    a = maskforge.PyVocabulary(EOS, tokens)
    b = maskforge.PyVocabulary(EOS, tokens)
    assert a.fingerprint() == b.fingerprint()
    ra = maskforge.compile_json_schema_with_vocabulary(a, SCHEMA, False, "byte_trie")
    rb = maskforge.compile_json_schema_with_vocabulary(b, SCHEMA, False, "byte_trie")
    s = ra.start()
    assert ra.mask_words([s], None) == rb.mask_words([rb.start()], None)


def test_many_handles_under_cache_pressure_stay_correct():
    # Cache pressure must not corrupt tries from other vocabulary handles.
    # Correctness does not depend on retaining a trie in cache.
    handles = []
    for i in range(50):
        toks = [(bytes([b]), [b]) for b in range(200)] + [(f"tag{i}".encode(), [200])]
        handles.append((i, maskforge.PyVocabulary(201, toks)))
    for i, h in handles:
        idx = maskforge.compile_json_schema_with_vocabulary(h, SCHEMA, False, "byte_trie")
        naive = maskforge.compile_json_schema_with_vocabulary(h, SCHEMA, False, "naive")
        assert idx.mask_words([idx.start()], None) == naive.mask_words([naive.start()], None)


def test_concurrent_first_use_across_many_handles_is_safe():
    import threading

    tokens = byte_vocab()
    errors = []

    def worker():
        try:
            h = maskforge.PyVocabulary(EOS, tokens)
            idx = maskforge.compile_json_schema_with_vocabulary(h, SCHEMA, False, "byte_trie")
            assert idx.mask_words([idx.start()], None)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


def test_trie_cache_stats_reflects_a_new_byte_trie():
    count0, bytes0, hits0, misses0, evict0, oversized0 = maskforge.trie_cache_stats()
    unique = [(f"stat-tag-{i}".encode(), [i]) for i in range(300)]
    h = maskforge.PyVocabulary(300, unique)
    maskforge.compile_json_schema_with_vocabulary(h, SCHEMA, False, "byte_trie")
    count1, bytes1, hits1, misses1, evict1, oversized1 = maskforge.trie_cache_stats()
    assert count1 >= count0
    assert bytes1 >= bytes0
    assert misses1 > misses0
    assert hits1 >= hits0
    assert evict1 >= evict0
    assert oversized1 >= oversized0


def test_trie_cache_stats_hit_on_repeated_bind_of_the_same_vocabulary():
    # Different schemas miss the artifact cache but reuse the vocabulary's byte trie.
    # The second compile therefore records a trie-cache hit.
    unique = [(f"hit-tag-{i}".encode(), [i]) for i in range(50)]
    h = maskforge.PyVocabulary(50, unique)
    maskforge.compile_json_schema_with_vocabulary(h, SCHEMA, False, "byte_trie")  # miss, builds
    _, _, hits_before, _, _, _ = maskforge.trie_cache_stats()
    other_schema = json.dumps({"type": "null"})
    maskforge.compile_json_schema_with_vocabulary(h, other_schema, False, "byte_trie")  # trie hit
    _, _, hits_after, _, _, _ = maskforge.trie_cache_stats()
    assert hits_after > hits_before


def test_many_distinct_small_vocabularies_stay_correct_under_the_shared_cache():
    for i in range(60):
        toks = [(f"vocab-{i}-tag-{b}".encode(), [b]) for b in range(20)]
        h = maskforge.PyVocabulary(20, toks)
        idx = maskforge.compile_json_schema_with_vocabulary(h, SCHEMA, False, "byte_trie")
        naive = maskforge.compile_json_schema_with_vocabulary(h, SCHEMA, False, "naive")
        assert idx.mask_words([idx.start()], None) == naive.mask_words([naive.start()], None)


def test_active_artifact_survives_after_the_handle_that_built_it_is_dropped():
    unique = [(f"survive-{i}".encode(), [i]) for i in range(30)]
    h = maskforge.PyVocabulary(30, unique)
    idx = maskforge.compile_json_schema_with_vocabulary(h, SCHEMA, False, "byte_trie")
    naive_before = maskforge.compile_json_schema_with_vocabulary(h, SCHEMA, False, "naive")
    expected = naive_before.mask_words([naive_before.start()], None)
    del h
    import gc

    gc.collect()
    assert idx.mask_words([idx.start()], None) == expected


def test_bytearray_and_bytes_token_components_produce_the_same_fingerprint():
    # Bytes, bytearray, and list token components must build equivalent vocabularies.
    as_bytes = [(bytes([b]), [b]) for b in range(64)]
    as_bytearray = [(bytearray([b]), [b]) for b in range(64)]
    h_bytes = maskforge.PyVocabulary(64, as_bytes)
    h_bytearray = maskforge.PyVocabulary(64, as_bytearray)
    assert h_bytes.fingerprint() == h_bytearray.fingerprint()


def test_default_mask_vocab_size_is_inferred_from_the_largest_id():
    tokens = [(b"a", [5])]
    h = maskforge.PyVocabulary(9, tokens)
    assert h.mask_vocab_size() == 10  # max(5, eos=9) + 1


def test_explicit_logits_vocab_size_overrides_the_inferred_width():
    tokens = [(b"a", [5])]
    h = maskforge.PyVocabulary(9, tokens, logits_vocab_size=50000)
    assert h.mask_vocab_size() == 50000


def test_explicit_logits_vocab_size_narrower_than_an_actual_id_is_rejected():
    tokens = [(b"a", [500])]
    with pytest.raises(maskforge.MaskforgeError):
        maskforge.PyVocabulary(9, tokens, logits_vocab_size=100)


def test_explicit_logits_vocab_size_does_not_change_the_fingerprint():
    tokens = [(b"a", [5])]
    narrow = maskforge.PyVocabulary(9, tokens)
    wide = maskforge.PyVocabulary(9, tokens, logits_vocab_size=50000)
    assert narrow.fingerprint() == wide.fingerprint()
    assert narrow.mask_vocab_size() != wide.mask_vocab_size()


def test_compile_json_schema_accepts_an_explicit_logits_vocab_size():
    tokens = [(b"a", [0]), (b"b", [1])]
    idx = maskforge.compile_json_schema(SCHEMA, 2, tokens, False, logits_vocab_size=1000)
    assert idx.vocab_size() == 1000


def test_prepare_returns_none_and_is_idempotent():
    handle = maskforge.PyVocabulary(EOS, byte_vocab())
    assert handle.prepare("byte_trie") is None
    assert handle.prepare("byte_trie") is None  # repeat call: a cache hit, not a rebuild


def test_prepare_defaults_to_byte_trie():
    handle = maskforge.PyVocabulary(EOS, byte_vocab())
    handle.prepare()  # no bind_mode arg
    count0, _, hits0, misses0, _, _ = maskforge.trie_cache_stats()
    handle.prepare("byte_trie")
    count1, _, hits1, misses1, _, _ = maskforge.trie_cache_stats()
    assert count1 == count0
    assert hits1 > hits0 or misses1 == misses0


def test_prepare_naive_is_a_no_op_success():
    handle = maskforge.PyVocabulary(EOS, byte_vocab())
    assert handle.prepare("naive") is None  # no trie concept for naive; succeeds trivially


def test_prepare_rejects_an_unknown_bind_mode():
    handle = maskforge.PyVocabulary(EOS, byte_vocab())
    with pytest.raises(maskforge.MaskforgeError):
        handle.prepare("not-a-real-mode")


def test_compiling_after_prepare_is_a_trie_cache_hit_not_a_miss():
    handle = maskforge.PyVocabulary(EOS, byte_vocab())
    handle.prepare("byte_trie")
    _, _, hits0, misses0, _, _ = maskforge.trie_cache_stats()
    # Use a fresh schema so an artifact-cache hit cannot bypass trie lookup.
    fresh_schema = json.dumps({"enum": ["trie-cache-hit-not-miss-probe"]})
    maskforge.compile_json_schema_with_vocabulary(handle, fresh_schema, False, "byte_trie")
    _, _, hits1, misses1, _, _ = maskforge.trie_cache_stats()
    assert hits1 > hits0, "the first compile after prepare() must reuse the prewarmed trie"
    assert misses1 == misses0, "prepare() must have already paid the one-time trie build"


def test_prepare_then_compile_matches_compile_without_prepare():
    tokens = byte_vocab()
    prepared = maskforge.PyVocabulary(EOS, tokens)
    prepared.prepare("byte_trie")
    unprepared = maskforge.PyVocabulary(EOS, tokens)
    a = maskforge.compile_json_schema_with_vocabulary(prepared, SCHEMA, False, "byte_trie")
    b = maskforge.compile_json_schema_with_vocabulary(unprepared, SCHEMA, False, "byte_trie")
    assert a.mask_words([a.start()], None) == b.mask_words([b.start()], None)
