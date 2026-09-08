"""`mask_words_into` must agree byte-for-byte with `mask_words`, for every reachable state of every
corpus schema, under both bind modes, and reject a wrong-sized or malformed output buffer."""

import struct

import maskforge
import pytest


def vocab64():
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


def words_to_bytearray(words):
    ba = bytearray(len(words) * 4)
    struct.pack_into("<" + "I" * len(words), ba, 0, *words)
    return ba


@pytest.mark.parametrize("mode", ["naive", "byte_trie"])
@pytest.mark.parametrize("name", maskforge.corpus_names())
def test_mask_words_into_matches_mask_words(name, mode):
    eos, tokens = vocab64()
    idx = maskforge.corpus_index(name, eos, tokens, mode)
    for s in reachable_states(idx):
        expected = idx.mask_words([s], None)
        out = bytearray(len(expected) * 4)
        idx.mask_words_into([s], out, None)
        assert list(struct.unpack("<" + "I" * len(expected), out)) == expected


def test_mask_words_into_batch_matches_mask_words():
    eos, tokens = vocab64()
    idx = maskforge.corpus_index("suite-type-boolean", eos, tokens, "byte_trie")
    states = list(reachable_states(idx))[:4]
    expected = idx.mask_words(states, None)
    out = bytearray(len(expected) * 4)
    idx.mask_words_into(states, out, None)
    assert list(struct.unpack("<" + "I" * len(expected), out)) == expected


@pytest.mark.parametrize("batch", [1, 8, 32])
@pytest.mark.parametrize("mode", ["naive", "byte_trie"])
def test_mask_words_into_batch_sizes_1_8_32(batch, mode):
    eos, tokens = vocab64()
    idx = maskforge.corpus_index("syn-object-nested", eos, tokens, mode)
    states = list(reachable_states(idx))
    schedule = [states[i % len(states)] for i in range(batch)]  # wraps -> forces duplicate rows
    row = len(idx.mask_words([schedule[0]], None))
    out = bytearray(row * 4 * batch)
    idx.mask_words_into(schedule, out, None)
    for i, s in enumerate(schedule):
        seg = out[i * row * 4:(i + 1) * row * 4]
        assert list(struct.unpack("<" + "I" * row, seg)) == idx.mask_words([s], None)


def test_mask_words_into_duplicate_rows_or_into_one_target():
    # rows=[0,0,...] must OR every state's mask into a single output row (documented duplicate policy).
    eos, tokens = vocab64()
    idx = maskforge.corpus_index("syn-object-nested", eos, tokens, "byte_trie")
    states = list(reachable_states(idx))[:4]
    row = len(idx.mask_words([states[0]], None))
    out = bytearray(row * 4)  # ONE row for all states
    idx.mask_words_into(states, out, [0] * len(states))
    got = list(struct.unpack("<" + "I" * row, out))
    expected = [0] * row
    for s in states:
        for i, w in enumerate(idx.mask_words([s], None)):
            expected[i] |= w
    assert got == expected


def test_mask_words_into_rejects_wrong_size_buffer():
    eos, tokens = vocab64()
    idx = maskforge.corpus_index("suite-type-boolean", eos, tokens, "byte_trie")
    s = idx.start()
    correct_len = len(idx.mask_words([s], None)) * 4
    with pytest.raises(maskforge.MaskforgeError) as exc:
        idx.mask_words_into([s], bytearray(correct_len + 4), None)
    assert exc.value.args[0] == "ArtifactOutOfBounds"
    with pytest.raises(maskforge.MaskforgeError):
        idx.mask_words_into([s], bytearray(correct_len - 4 if correct_len >= 4 else 0), None)


def test_mask_words_into_rejects_non_bytearray():
    eos, tokens = vocab64()
    idx = maskforge.corpus_index("suite-type-boolean", eos, tokens, "byte_trie")
    s = idx.start()
    with pytest.raises(TypeError):
        idx.mask_words_into([s], b"not a bytearray", None)


def test_mask_words_into_reused_buffer_across_calls():
    eos, tokens = vocab64()
    idx = maskforge.corpus_index("suite-type-boolean", eos, tokens, "byte_trie")
    states = list(reachable_states(idx))
    row_len = len(idx.mask_words([idx.start()], None))
    buf = bytearray(row_len * 4)
    for s in states:
        idx.mask_words_into([s], buf, None)
        expected = idx.mask_words([s], None)
        assert list(struct.unpack("<" + "I" * row_len, buf)) == expected


@pytest.mark.parametrize("mode", ["naive", "byte_trie"])
def test_mask_word_into_state_matches_mask_words(mode):
    eos, tokens = vocab64()
    idx = maskforge.corpus_index("suite-type-boolean", eos, tokens, mode)
    for s in reachable_states(idx):
        expected = idx.mask_words([s], None)
        out = bytearray(len(expected) * 4)
        idx.mask_word_into_state(s, out)
        assert list(struct.unpack("<" + "I" * len(expected), out)) == expected


def test_mask_word_into_state_rejects_wrong_size_buffer():
    eos, tokens = vocab64()
    idx = maskforge.corpus_index("suite-type-boolean", eos, tokens, "byte_trie")
    s = idx.start()
    correct_len = len(idx.mask_words([s], None)) * 4
    with pytest.raises(maskforge.MaskforgeError):
        idx.mask_word_into_state(s, bytearray(correct_len + 4))
