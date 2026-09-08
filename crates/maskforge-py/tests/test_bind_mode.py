"""Phase-2 differential: the opt-in `byte_trie` bind mode must be byte-identical to the default
`naive` mode through the FFI, for every reachable state of every corpus schema."""

import maskforge
import pytest


def vocab64():
    """64 tokens (JSON literals, structural bytes, digits, letters); EOS = 64."""
    toks = [b"true", b"false", b"null"]
    toks += [bytes([b]) for b in b"[]{},:\""]
    toks += [bytes([b]) for b in range(ord("0"), ord("9") + 1)]
    toks += [bytes([b]) for b in range(ord("a"), ord("z") + 1)]
    for b in range(ord("A"), ord("Z") + 1):
        if len(toks) >= 64:
            break
        toks.append(bytes([b]))
    return 64, [(t, [i]) for i, t in enumerate(toks)]


def reachable_states(idx):
    seen = {idx.start()}
    frontier = [idx.start()]
    while frontier:
        s = frontier.pop()
        for tid in idx.allowed_ids(s):
            nxt = idx.advance(s, tid)
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


@pytest.mark.parametrize("name", maskforge.corpus_names())
def test_byte_trie_equals_naive_over_every_reachable_state(name):
    eos, tokens = vocab64()
    naive = maskforge.corpus_index(name, eos, tokens, bind_mode="naive")  # explicit oracle
    byte = maskforge.corpus_index(name, eos, tokens, bind_mode="byte_trie")

    states = reachable_states(naive)
    assert states  # at least the start state
    for s in states:
        assert byte.allowed_ids(s) == naive.allowed_ids(s), f"{name}: allowed_ids differ at {s}"
        assert byte.mask_words([s], None) == naive.mask_words([s], None), f"{name}: mask differs at {s}"
        assert byte.is_accepting(s) == naive.is_accepting(s)
        assert byte.can_continue(s) == naive.can_continue(s)
        assert byte.is_dead(s) == naive.is_dead(s)
        assert byte.eos_legal(s) == naive.eos_legal(s)
        for tid in naive.allowed_ids(s):
            assert byte.advance(s, tid) == naive.advance(s, tid), f"{name}: landing differs"


def test_dead_and_forged_states_agree():
    eos, tokens = vocab64()
    naive = maskforge.corpus_index("suite-type-boolean", eos, tokens, bind_mode="naive")
    byte = maskforge.corpus_index("suite-type-boolean", eos, tokens, bind_mode="byte_trie")
    for s in (naive.dead(), 4_000_000_000):
        assert byte.allowed_ids(s) == naive.allowed_ids(s) == []
        assert byte.mask_words([s], None) == naive.mask_words([s], None)


def test_lazy_byte_trie_equals_naive_over_every_reachable_state():
    eos, tokens = vocab64()
    handle = maskforge.PyVocabulary(eos, tokens)
    for name in maskforge.corpus_names():
        schema = maskforge.corpus_json_schema(name)
        try:
            naive = maskforge.compile_json_schema_with_vocabulary(handle, schema, False, "naive")
            lazy = maskforge.compile_json_schema_with_vocabulary(handle, schema, False, "byte_trie_lazy")
        except maskforge.MaskforgeError:
            continue  # a corpus schema the JSON frontend does not accept (e.g. anchors)
        for s in reachable_states(naive):
            assert lazy.allowed_ids(s) == naive.allowed_ids(s), f"{name}: lazy ids differ at {s}"
            assert lazy.mask_words([s], None) == naive.mask_words([s], None), f"{name}: lazy mask at {s}"
            for tid in naive.allowed_ids(s):
                assert lazy.advance(s, tid) == naive.advance(s, tid), f"{name}: lazy landing at {s}"


def test_packed_retained_bytes_are_per_artifact_and_sum_for_a_memory_budget():
    # The packed budget is per artifact; a server sums retained_bytes across live indexes to enforce
    # a total budget. Packed reports > 0 and scales with the number of live artifacts; lazy reports 0.
    eos, tokens = vocab64()
    handle = maskforge.PyVocabulary(eos, tokens)
    schema = '{"type":"boolean"}'
    one = maskforge.compile_json_schema_with_vocabulary(handle, schema, False, "byte_trie")
    assert one.retained_bytes() > 0
    lazy = maskforge.compile_json_schema_with_vocabulary(handle, schema, False, "byte_trie_lazy")
    assert lazy.retained_bytes() == 0, "lazy retains no per-state table"
    # Ten live packed artifacts retain ~10x one artifact's table (aggregate the app must budget).
    live = [maskforge.compile_json_schema_with_vocabulary(handle, schema, False, "byte_trie") for _ in range(10)]
    total = sum(i.retained_bytes() for i in live)
    assert total >= 10 * one.retained_bytes(), "aggregate retained bytes scale with live artifacts"


def test_public_compile_apis_default_to_packed_byte_trie():
    # Every public compile entry point, called WITHOUT bind_mode, must produce the packed byte_trie.
    eos, tokens = vocab64()
    handle = maskforge.PyVocabulary(eos, tokens)
    schema = '{"type":"boolean"}'
    ir_wire = maskforge.schema_to_ir(schema).to_wire()
    assert maskforge.compile_json_schema(schema, eos, tokens).bind_mode() == "byte_trie"
    assert maskforge.compile_ir(ir_wire, eos, tokens).bind_mode() == "byte_trie"
    assert maskforge.corpus_index("suite-type-boolean", eos, tokens).bind_mode() == "byte_trie"
    assert maskforge.compile_json_schema_with_vocabulary(handle, schema).bind_mode() == "byte_trie"
    assert maskforge.compile_ir_with_vocabulary(handle, ir_wire).bind_mode() == "byte_trie"
    assert maskforge.compile_json_schema(schema, eos, tokens, bind_mode="naive").bind_mode() == "naive"


def test_lazy_byte_trie_requires_a_handle_not_raw_tokens():
    eos, tokens = vocab64()
    with pytest.raises(maskforge.MaskforgeError) as exc:
        maskforge.corpus_index("suite-type-boolean", eos, tokens, bind_mode="byte_trie_lazy")
    assert exc.value.args[0] == "Unsupported"


def test_unknown_bind_mode_is_a_structured_error_not_a_silent_fallback():
    eos, tokens = vocab64()
    with pytest.raises(maskforge.MaskforgeError) as exc:
        maskforge.corpus_index("suite-type-boolean", eos, tokens, bind_mode="turbo")
    assert exc.value.args[0] == "Unsupported"  # stable code is args[0]


def test_reused_vocabulary_handle_binds_byte_trie_identically_to_naive():
    eos, tokens = vocab64()
    handle = maskforge.PyVocabulary(eos, tokens)
    schema = '{"type":"boolean"}'
    naive = maskforge.compile_json_schema_with_vocabulary(handle, schema, bind_mode="naive")
    # Repeated byte_trie binds on one handle reuse its cached trie; both must equal naive.
    a = maskforge.compile_json_schema_with_vocabulary(handle, schema, bind_mode="byte_trie")
    b = maskforge.compile_json_schema_with_vocabulary(handle, schema, bind_mode="byte_trie")
    for s in reachable_states(naive):
        assert a.allowed_ids(s) == naive.allowed_ids(s)
        assert b.mask_words([s], None) == naive.mask_words([s], None)


def test_concurrent_distinct_schema_compiles_on_one_handle_release_the_gil_and_agree():
    # Concurrent binds on one handle must be safe and match the naive oracle.
    import threading

    eos, tokens = vocab64()
    handle = maskforge.PyVocabulary(eos, tokens)
    schemas = [
        '{"type":"boolean"}',
        '{"type":"null"}',
        '{"enum":["1","12"]}',
        '{"enum":["1","12","123"]}',
    ]
    refs = {}
    for s in schemas:
        idx = maskforge.compile_json_schema_with_vocabulary(handle, s, False, "naive")
        refs[s] = {st: idx.allowed_ids(st) for st in reachable_states(idx)}
    errors = []

    def worker(schema):
        try:
            for _ in range(20):
                idx = maskforge.compile_json_schema_with_vocabulary(handle, schema)
                assert idx.bind_mode() == "byte_trie"
                for st, ids in refs[schema].items():
                    assert idx.allowed_ids(st) == ids, f"{schema}: mismatch at {st}"
        except Exception as e:  # noqa: BLE001 - record any thread failure
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(schemas[i % len(schemas)],)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


def test_concurrent_byte_trie_binds_on_one_handle_are_safe():
    import threading

    eos, tokens = vocab64()
    handle = maskforge.PyVocabulary(eos, tokens)
    schema = '{"enum":["1","12"]}'
    ref = maskforge.compile_json_schema_with_vocabulary(handle, schema, bind_mode="naive")
    ref_ids = {s: ref.allowed_ids(s) for s in reachable_states(ref)}
    errors = []

    def worker():
        try:
            idx = maskforge.compile_json_schema_with_vocabulary(handle, schema, bind_mode="byte_trie")
            for s, ids in ref_ids.items():
                assert idx.allowed_ids(s) == ids
        except Exception as e:  # noqa: BLE001 - record any thread failure for the assert below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
