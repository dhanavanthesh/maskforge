"""Resizing the output bytearray while Rust computes a mask must never crash the process.

`PySession.write_mask` releases the GIL for the mask computation, so another Python thread can
resize the caller's bytearray in that window. The only acceptable outcomes are a correct write
or a MaskforgeError; a panic, an abort, or a silent short write are all failures.
"""

import json
import threading

import pytest

import maskforge

# Wide enough that the GIL-released mask computation is a real window, not an instant.
VOCAB_SIZE = 250_000
ITERATIONS = 400


def _wide_vocabulary():
    tokens = [bytes([b]) for b in range(256)]
    index = 0
    while len(tokens) < VOCAB_SIZE:
        tokens.append(b"t" + str(index).encode())
        index += 1
    eos = VOCAB_SIZE - 1
    tokens[eos] = None
    return maskforge.Vocabulary.from_id_ordered_tokens(tokens, eos, VOCAB_SIZE)


@pytest.fixture(scope="module")
def bound():
    schema = json.dumps(
        {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
    )
    return maskforge.Compiler().compile(schema).bind(_wide_vocabulary())


def test_a_correctly_sized_buffer_still_works(bound):
    session = bound.start_session()
    buffer = bytearray(4 * bound.mask_word_count)
    session.write_mask(buffer)
    assert any(buffer)


def test_a_resized_buffer_is_rejected_not_written_past(bound):
    """The pre-detach length check alone would let this through."""
    session = bound.start_session()
    buffer = bytearray(4 * bound.mask_word_count + 1)
    with pytest.raises(maskforge.MaskforgeError):
        session.write_mask(buffer)


def test_resizing_the_output_mid_mask_never_crashes(bound):
    """A writer thread masks in a loop while a mutator resizes the same buffer underneath it."""
    session = bound.start_session()
    expected = 4 * bound.mask_word_count
    buffer = bytearray(expected)

    stop = threading.Event()
    failures = []
    outcomes = {"ok": 0, "rejected": 0}

    def writer():
        try:
            for _ in range(ITERATIONS):
                try:
                    session.write_mask(buffer)
                    outcomes["ok"] += 1
                except maskforge.MaskforgeError:
                    outcomes["rejected"] += 1
        except BaseException as exc:  # any other exception is a real defect
            failures.append(exc)
        finally:
            stop.set()

    def mutator():
        while not stop.is_set():
            try:
                # Shrink then restore, so the writer sometimes reacquires the GIL mid-change.
                del buffer[:]
                buffer.extend(b"\x00" * expected)
            except BufferError:
                # CPython refuses to resize while an exported buffer is live; that is a
                # legitimate outcome, not a defect.
                pass

    threads = [threading.Thread(target=writer), threading.Thread(target=mutator)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert not any(t.is_alive() for t in threads), "a thread hung"
    assert not failures, f"unexpected exception type: {failures[:1]}"
    assert outcomes["ok"] + outcomes["rejected"] == ITERATIONS
    # Whenever the write did succeed, the buffer must be exactly the right width.
    assert len(buffer) in (0, expected)


def test_many_sessions_masking_concurrently_stay_independent(bound):
    """Each thread owns its session and buffer; results must be identical across threads."""
    expected = 4 * bound.mask_word_count
    results = []
    errors = []
    lock = threading.Lock()

    def worker():
        try:
            session = bound.start_session()
            buffer = bytearray(expected)
            for _ in range(50):
                session.write_mask(buffer)
            with lock:
                results.append(bytes(buffer))
        except BaseException as exc:
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert not errors, f"concurrent masking raised: {errors[:1]}"
    assert len(results) == 8
    assert len(set(results)) == 1, "identical sessions produced different masks"
