"""PyVocabulary.from_packed_buffers - the packed-buffer vocab constructor must be byte-identical
to the tuple constructor and must reject every malformed buffer with a structured error, never a crash."""

import array

import maskforge
import pytest


def pack(items, eos):
    """Pack (bytes, [ids]) items into the four little-endian buffers from_packed_buffers expects."""
    token_bytes = b"".join(b for b, _ in items)
    byte_offsets, off = [], 0
    for b, _ in items:
        byte_offsets.append(off)
        off += len(b)
    byte_offsets.append(off)
    token_ids, id_offsets, ioff = [], [], 0
    for _, ids in items:
        id_offsets.append(ioff)
        ioff += len(ids)
        token_ids.extend(ids)
    id_offsets.append(ioff)
    le = lambda xs: array.array("I", xs).tobytes()  # noqa: E731 - x86 native == LE
    return token_bytes, le(byte_offsets), le(token_ids), le(id_offsets), eos


def vocab64_items():
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
    seen, frontier = {idx.start()}, [idx.start()]
    while frontier:
        s = frontier.pop()
        for tid in idx.allowed_ids(s):
            n = idx.advance(s, tid)
            if n not in seen:
                seen.add(n)
                frontier.append(n)
    return seen


def test_packed_fingerprint_equals_tuple_constructor():
    # A vocab with a multi-id token and a repeated byte sequence, to exercise the id CSR + merge path.
    items = [(b"cat", [17]), (b"car", [42, 99]), (b"c", [7]), (b"cat", [200])]
    eos = 1000
    h_tuple = maskforge.PyVocabulary(eos, items)
    h_packed = maskforge.PyVocabulary.from_packed_buffers(*pack(items, eos))
    assert bytes(h_tuple.fingerprint()) == bytes(h_packed.fingerprint())
    assert h_tuple.len() == h_packed.len()


def test_packed_compiles_byte_identical_to_tuple_over_every_corpus_schema():
    eos, items = vocab64_items()
    h_tuple = maskforge.PyVocabulary(eos, items)
    h_packed = maskforge.PyVocabulary.from_packed_buffers(*pack(items, eos))
    assert bytes(h_tuple.fingerprint()) == bytes(h_packed.fingerprint())
    for name in maskforge.corpus_names():
        try:
            schema = maskforge.corpus_json_schema(name)
            a = maskforge.compile_json_schema_with_vocabulary(h_tuple, schema, False, "byte_trie")
            b = maskforge.compile_json_schema_with_vocabulary(h_packed, schema, False, "byte_trie")
        except maskforge.MaskforgeError:
            continue  # schema the JSON frontend rejects (e.g. anchors)
        for s in reachable_states(a):
            assert a.mask_words([s], None) == b.mask_words([s], None), f"{name}: mask differs at {s}"
            assert a.allowed_ids(s) == b.allowed_ids(s)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda tb, bo, ti, io, e: (tb, bo[:-1], ti, io, e), id="byte_offsets_not_mult_of_4"),
        pytest.param(lambda tb, bo, ti, io, e: (tb, bo + b"\x00\x00\x00\x00", ti, io, e), id="offset_count_mismatch"),
        pytest.param(lambda tb, bo, ti, io, e: (tb[:-1], bo, ti, io, e), id="byte_offset_last_exceeds_len"),
        pytest.param(lambda tb, bo, ti, io, e: (tb, b"", ti, io, e), id="empty_byte_offsets"),
        pytest.param(lambda tb, bo, ti, io, e: (tb, bo, ti[:-4], io, e), id="token_ids_shorter_than_offsets_claim"),
    ],
)
def test_packed_rejects_malformed_buffers_with_a_structured_error(mutate):
    eos, items = vocab64_items()
    args = mutate(*pack(items, eos))
    with pytest.raises(maskforge.MaskforgeError):
        maskforge.PyVocabulary.from_packed_buffers(*args)


def test_packed_rejects_non_monotonic_byte_offsets():
    # Hand-craft a decreasing byte-offset array: [0, 3, 1, 6] over 6 bytes.
    tb = b"catcar"
    bo = array.array("I", [0, 3, 1, 6]).tobytes()  # 3 tokens, but 3 > 1 (decreasing)
    ti = array.array("I", [0, 1, 2]).tobytes()
    io = array.array("I", [0, 1, 2, 3]).tobytes()
    with pytest.raises(maskforge.MaskforgeError):
        maskforge.PyVocabulary.from_packed_buffers(tb, bo, ti, io, 1000)


def test_packed_rejects_eos_as_ordinary_id_and_empty_token():
    # EOS appearing as an ordinary id is rejected (same as the tuple path).
    tb, bo, ti, io, _ = pack([(b"a", [5])], eos=5)
    with pytest.raises(maskforge.MaskforgeError):
        maskforge.PyVocabulary.from_packed_buffers(tb, bo, ti, io, 5)
    # An empty token byte-sequence is rejected.
    tb2, bo2, ti2, io2, e2 = pack([(b"", [0])], eos=9)
    with pytest.raises(maskforge.MaskforgeError):
        maskforge.PyVocabulary.from_packed_buffers(tb2, bo2, ti2, io2, e2)


def test_packed_rejects_one_id_under_two_byte_sequences():
    # id 0 mapped to both "a" and "b" is a malformed tokenizer, rejected at handle build.
    tb, bo, ti, io, e = pack([(b"a", [0]), (b"b", [0])], eos=9)
    with pytest.raises(maskforge.MaskforgeError):
        maskforge.PyVocabulary.from_packed_buffers(tb, bo, ti, io, e)


def test_packed_handles_non_utf8_bytes_and_prefix_overlap():
    # Raw non-UTF-8 bytes and a prefix chain (a/ab/abc) - the packed path must marshal them exactly.
    items = [(bytes([0xFF, 0xFE, 0x80]), [0]), (b"a", [1]), (b"ab", [2]), (b"abc", [3])]
    eos = 1000
    h_tuple = maskforge.PyVocabulary(eos, items)
    h_packed = maskforge.PyVocabulary.from_packed_buffers(*pack(items, eos))
    assert bytes(h_tuple.fingerprint()) == bytes(h_packed.fingerprint())


def test_packed_construction_is_safe_under_concurrent_threads():
    # 16 threads build DISTINCT packed handles at once; the GIL is released during validation, so
    # this exercises the same concurrent-build path the trie cache serializes.
    import threading

    errors = []

    def worker(i):
        try:
            items = [(bytes([b"x"[0], i]), [0]), (bytes([b"y"[0], i]), [1])]
            h = maskforge.PyVocabulary.from_packed_buffers(*pack(items, 1000))
            assert h.len() == 3  # 2 ordinary ids + EOS
        except Exception as e:  # noqa: BLE001 - record, don't hide, thread failures
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


def test_packed_rejects_an_oversized_buffer_before_copying_it():
    # Verify that buffer-protocol ingestion rejects an oversized object before copying it.
    # An anonymous `mmap` avoids committing a large `bytes` allocation.
    import mmap

    huge = mmap.mmap(-1, (1 << 28) + 1)
    try:
        with pytest.raises(maskforge.MaskforgeError) as exc:
            maskforge.PyVocabulary.from_packed_buffers(huge, b"\x00" * 4, b"", b"\x00" * 4, 1000)
        assert exc.value.args[0] == "Unsupported"
        assert "exceeds the cap" in exc.value.args[5]
    finally:
        huge.close()


def test_packed_random_buffers_never_crash_only_raise_or_succeed():
    # Adversarial random buffers must either build a handle or raise `MaskforgeError`.
    # They must never crash or hang the process.
    import random

    rng = random.Random(0xC0FFEE)
    for _ in range(3000):
        tb = bytes(rng.getrandbits(8) for _ in range(rng.randrange(0, 40)))
        bo = bytes(rng.getrandbits(8) for _ in range(rng.randrange(0, 40)))
        ti = bytes(rng.getrandbits(8) for _ in range(rng.randrange(0, 40)))
        io = bytes(rng.getrandbits(8) for _ in range(rng.randrange(0, 40)))
        eos = rng.randrange(0, 1 << 32)
        try:
            maskforge.PyVocabulary.from_packed_buffers(tb, bo, ti, io, eos)
        except maskforge.MaskforgeError:
            pass  # the only acceptable failure mode


def test_packed_mutated_real_buffers_never_crash_only_raise_or_succeed():
    # Same idea, but seeded from a REAL well-formed packed vocabulary so most mutants pass the cheap
    # length/alignment checks and exercise the deeper offset/id-CSR validation logic.
    import random

    rng = random.Random(0xBADF00D)
    eos, items = vocab64_items()
    tb, bo, ti, io, _ = pack(items, eos)

    def mutate(buf):
        buf = bytearray(buf)
        for _ in range(1 + rng.randrange(0, 5)):
            if not buf:
                break
            buf[rng.randrange(0, len(buf))] = rng.getrandbits(8)
        if rng.randrange(0, 4) == 0 and buf:
            del buf[rng.randrange(0, len(buf)) :]
        return bytes(buf)

    for _ in range(3000):
        try:
            maskforge.PyVocabulary.from_packed_buffers(
                mutate(tb), mutate(bo), mutate(ti), mutate(io), eos
            )
        except maskforge.MaskforgeError:
            pass
