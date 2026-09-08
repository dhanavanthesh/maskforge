"""The shared packed-build/lazy-serving memory gates: configuration validation, stats shape, and
concurrent packed-buffer ingestion never deadlocks under the GIL-released permit wait."""

import array
import threading

import maskforge
import pytest


def pack(items, eos):
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


def test_configure_vocab_build_budget_rejects_zero_values():
    # A distinct, machine-matchable code (not the catch-all "Unsupported" every structural
    # failure collapses to), so a caller can branch on invalid-value vs already-initialized.
    with pytest.raises(maskforge.MaskforgeError) as exc:
        maskforge.configure_vocab_build_budget(0, 100)
    assert exc.value.args[0] == "InvalidGateConfig"
    with pytest.raises(maskforge.MaskforgeError) as exc:
        maskforge.configure_vocab_build_budget(4, 0)
    assert exc.value.args[0] == "InvalidGateConfig"


def test_configure_vocab_serving_budget_rejects_zero_values():
    with pytest.raises(maskforge.MaskforgeError) as exc:
        maskforge.configure_vocab_serving_budget(0, 100)
    assert exc.value.args[0] == "InvalidGateConfig"
    with pytest.raises(maskforge.MaskforgeError) as exc:
        maskforge.configure_vocab_serving_budget(4, 0)
    assert exc.value.args[0] == "InvalidGateConfig"


def test_configuring_an_already_initialized_gate_reports_a_distinct_code():
    # Prior budget access initializes the process-wide gate.
    # Reconfiguration must report its distinct failure code.
    maskforge.vocab_build_budget_stats()
    with pytest.raises(maskforge.MaskforgeError) as exc:
        maskforge.configure_vocab_build_budget(4, 100)
    assert exc.value.args[0] == "GateAlreadyInitialized"


def test_vocab_build_budget_stats_shape():
    active, peak, peak_concurrent, rejected = maskforge.vocab_build_budget_stats()
    for v in (active, peak, peak_concurrent, rejected):
        assert isinstance(v, int) and v >= 0


def test_vocab_serving_budget_stats_shape():
    active, peak, peak_concurrent, rejected = maskforge.vocab_serving_budget_stats()
    for v in (active, peak, peak_concurrent, rejected):
        assert isinstance(v, int) and v >= 0


def test_many_concurrent_packed_ingestions_never_deadlock_under_the_shared_gate():
    # Concurrent permit acquisition releases the GIL and must not deadlock.
    # The timeout turns a hang into a test failure.
    errors = []

    def worker(i):
        try:
            items = [(bytes([b"x"[0], i % 256]), [0]), (bytes([b"y"[0], i % 256]), [1])]
            h = maskforge.PyVocabulary.from_packed_buffers(*pack(items, 1000))
            assert h.len() == 3
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(64)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "a hung thread means the gate/GIL interaction deadlocked"
    assert not errors, errors
