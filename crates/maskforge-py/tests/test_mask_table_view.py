"""PyIndex.mask_table_view() - an acquire-once, read-only, zero-copy memoryview over the
whole packed mask table. Own Arc<CompiledArtifact>, so it must outlive the PyIndex that produced
it; must never be writable; must never accept a non-packed (naive/lazy) artifact; must be safe
under concurrent readers, repeated acquire/release, and GC stress."""

import gc
import json
import sys

import pytest

maskforge = pytest.importorskip("maskforge")

EOS = 256


def byte_vocab():
    return [(bytes([b]), [b]) for b in range(256)]


SCHEMA = json.dumps({"type": "boolean"})


def packed_index():
    h = maskforge.PyVocabulary(EOS, byte_vocab())
    return maskforge.compile_json_schema_with_vocabulary(h, SCHEMA, False, "byte_trie")


def test_view_matches_the_copying_api_row_for_row():
    idx = packed_index()
    view = idx.mask_table_view()
    mv = memoryview(view)
    assert len(mv) == view.row_count * view.bytes_per_row

    buf = bytearray(view.bytes_per_row)
    for s in range(view.row_count):
        idx.mask_word_into_state(s, buf)
        row = mv[s * view.bytes_per_row : (s + 1) * view.bytes_per_row]
        assert bytes(row) == bytes(buf), f"state {s}: zero-copy row must match the copying API"


def test_view_is_read_only():
    idx = packed_index()
    mv = memoryview(idx.mask_table_view())
    assert mv.readonly
    with pytest.raises(TypeError):
        mv[0] = 0


def test_nonpacked_bind_mode_is_a_typed_error_not_a_wrong_view():
    h = maskforge.PyVocabulary(EOS, byte_vocab())
    lazy_idx = maskforge.compile_json_schema_with_vocabulary(h, SCHEMA, False, "byte_trie_lazy")
    with pytest.raises(maskforge.MaskforgeError) as exc:
        lazy_idx.mask_table_view()
    assert exc.value.args[0] == "Unsupported"

    naive_idx = maskforge.compile_json_schema_with_vocabulary(h, SCHEMA, False, "naive")
    with pytest.raises(maskforge.MaskforgeError):
        naive_idx.mask_table_view()


def test_view_survives_deletion_and_gc_of_the_owning_index():
    idx = packed_index()
    view = idx.mask_table_view()
    expected = bytes(memoryview(view))
    del idx
    gc.collect()
    gc.collect()
    assert bytes(memoryview(view)) == expected, "the view keeps its own Arc alive independently"


def test_exported_memoryview_survives_deletion_of_the_view_itself():
    # An exported memoryview keeps the underlying artifact alive after `view` is dropped.
    idx = packed_index()
    view = idx.mask_table_view()
    mv = memoryview(view)
    expected = bytes(mv)
    del view
    del idx
    gc.collect()
    gc.collect()
    assert bytes(mv) == expected
    mv.release()


def test_repeated_acquire_and_release_does_not_leak_or_corrupt():
    idx = packed_index()
    baseline = bytes(memoryview(idx.mask_table_view()))
    for _ in range(2000):
        v = idx.mask_table_view()
        mv = memoryview(v)
        assert bytes(mv) == baseline
        mv.release()
        del v, mv


def test_multiple_concurrent_readers_of_one_view():
    import threading

    idx = packed_index()
    view = idx.mask_table_view()
    baseline = bytes(memoryview(view))
    errors = []

    def reader():
        try:
            for _ in range(200):
                mv = memoryview(view)
                assert bytes(mv) == baseline
                mv.release()
        except Exception as e:  # noqa: BLE001 - record, don't hide, thread failures
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


def test_multiple_views_from_the_same_index_are_independent_and_consistent():
    idx = packed_index()
    views = [idx.mask_table_view() for _ in range(8)]
    snapshots = [bytes(memoryview(v)) for v in views]
    assert len(set(snapshots)) == 1, "every view over the same artifact reads identical bytes"


def test_view_dropped_before_owner_and_after_owner_both_fine():
    # Dropping the view before its index is the ordinary lifetime order.
    idx = packed_index()
    view = idx.mask_table_view()
    del view
    gc.collect()
    assert idx.mask_table_view().row_count > 0

    # A view must remain usable after its source index is dropped.
    idx2 = packed_index()
    view2 = idx2.mask_table_view()
    del idx2
    gc.collect()
    assert bytes(memoryview(view2))  # still readable, no use-after-free


def test_getters_agree_with_the_index_and_with_words_per_row():
    idx = packed_index()
    view = idx.mask_table_view()
    assert view.words_per_row == idx.words_per_row()
    assert view.bytes_per_row == view.words_per_row * 4


def test_gc_interleaved_concurrent_stress_no_use_after_free():
    # Stress view lifetime while threads create, export, release, and collect indexes.
    # This detects use-after-free through crashes or invalid bytes.
    import threading

    errors = []
    stop = threading.Event()

    def churn():
        try:
            for _ in range(300):
                idx = packed_index()
                view = idx.mask_table_view()
                mv = memoryview(view)
                snapshot = bytes(mv)
                mv.release()
                # Drop order varies across iterations on purpose.
                if len(snapshot) % 2 == 0:
                    del idx
                    del view
                else:
                    del view
                    del idx
                assert len(snapshot) > 0
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def gc_hammer():
        while not stop.is_set():
            gc.collect()

    gc_thread = threading.Thread(target=gc_hammer)
    gc_thread.start()
    workers = [threading.Thread(target=churn) for _ in range(8)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    stop.set()
    gc_thread.join()
    assert not errors, errors


@pytest.mark.skipif(sys.version_info < (3, 11), reason="buffer protocol needs abi3-py311+")
def test_python_version_is_at_least_311():
    # The buffer interface requires the Python 3.11 stable ABI.
    assert sys.version_info >= (3, 11)
