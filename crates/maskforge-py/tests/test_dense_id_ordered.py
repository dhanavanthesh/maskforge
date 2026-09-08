"""PyVocabulary.from_id_ordered_tokens: the dense, id-ordered ingestion path (decoded_tokens[i] is
token id i's bytes, or None for an absent id) must produce handles equivalent to the tuple
constructor for the same logical vocabulary, correctly skip absent slots, merge duplicate byte
strings across different ids, and reject the same malformed shapes the other constructors do."""

import json

import pytest

maskforge = pytest.importorskip("maskforge")

SCHEMA = json.dumps({"type": "boolean"})


def test_dense_matches_tuple_constructor_fingerprint_and_mask_width():
    # id0="cat", id1="car", id2 absent, id3="c" - same logical vocabulary as the tuple form below.
    dense = maskforge.PyVocabulary.from_id_ordered_tokens(
        [b"cat", b"car", None, b"c"], eos_token_id=9999
    )
    tuple_form = maskforge.PyVocabulary(
        9999, [(b"cat", [0]), (b"car", [1]), (b"c", [3])]
    )
    assert dense.fingerprint() == tuple_form.fingerprint()
    assert dense.mask_vocab_size() == tuple_form.mask_vocab_size()
    assert dense.len() == tuple_form.len()


def test_dense_and_tuple_produce_byte_identical_masks():
    dense = maskforge.PyVocabulary.from_id_ordered_tokens(
        [b"cat", b"car", None, b"c"], eos_token_id=9999
    )
    tuple_form = maskforge.PyVocabulary(
        9999, [(b"cat", [0]), (b"car", [1]), (b"c", [3])]
    )
    a = maskforge.compile_json_schema_with_vocabulary(dense, SCHEMA, False, "byte_trie")
    b = maskforge.compile_json_schema_with_vocabulary(tuple_form, SCHEMA, False, "byte_trie")
    assert a.mask_words([a.start()], None) == b.mask_words([b.start()], None)


def test_absent_slots_never_become_tokens():
    # len() = total ordinary ids + 1 for EOS; the absent slot must not count as an id.
    dense = maskforge.PyVocabulary.from_id_ordered_tokens([b"a", None, b"b"], eos_token_id=9999)
    assert dense.len() == 3  # 2 present ids (0, 2) + 1 for EOS


def test_duplicate_bytes_across_two_ids_merge_into_one_record_but_keep_both_ids():
    # id0 and id1 both decode to "a" - they merge into ONE record, but both ids individually
    # survive (len() counts total ids, not distinct records): 3 present ids (0, 1, 2) + 1 for EOS.
    dense = maskforge.PyVocabulary.from_id_ordered_tokens([b"a", b"a", b"b"], eos_token_id=9999)
    assert dense.len() == 4


def test_rejects_empty_bytes_on_a_present_slot():
    with pytest.raises(maskforge.MaskforgeError) as e:
        maskforge.PyVocabulary.from_id_ordered_tokens([b""], eos_token_id=5)
    assert e.value.args[0] == "EmptyToken"


def test_accepts_an_absent_slot_at_the_eos_position():
    # eos_token_id points at a None slot - the caller's own EOS placeholder, not a collision.
    h = maskforge.PyVocabulary.from_id_ordered_tokens([b"a", None], eos_token_id=1)
    assert h.eos_token_id() == 1
    assert h.len() == 2  # 1 present id (0) + 1 for EOS


def test_rejects_eos_id_present_as_an_ordinary_slot():
    with pytest.raises(maskforge.MaskforgeError) as e:
        maskforge.PyVocabulary.from_id_ordered_tokens([b"a", b"b"], eos_token_id=1)
    assert e.value.args[0] == "MalformedTokenizer"


def test_explicit_logits_vocab_size_overrides_the_inferred_width():
    h = maskforge.PyVocabulary.from_id_ordered_tokens(
        [b"a"], eos_token_id=9, logits_vocab_size=50000
    )
    assert h.mask_vocab_size() == 50000


def test_all_none_is_an_empty_vocabulary():
    h = maskforge.PyVocabulary.from_id_ordered_tokens([None, None, None], eos_token_id=9999)
    assert h.len() == 1  # 0 present ids + 1 for EOS
    assert h.is_empty()  # is_empty() counts ordinary tokens only, EOS excluded


def test_compiling_a_dense_handle_agrees_with_naive_on_a_richer_schema():
    dense = maskforge.PyVocabulary.from_id_ordered_tokens(
        [bytes([b]) for b in range(200)], eos_token_id=200
    )
    schema = json.dumps({"enum": [chr(b) for b in range(65, 91)]})
    fast = maskforge.compile_json_schema_with_vocabulary(dense, schema, False, "byte_trie")
    naive = maskforge.compile_json_schema_with_vocabulary(dense, schema, False, "naive")
    assert fast.mask_words([fast.start()], None) == naive.mask_words([naive.start()], None)
