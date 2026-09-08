//! Proves the exact bridge `maskforge-py` drives: a real `VocabularyHandle` feeding
//! `StructuredMatcher` through `iter_records`/`token_bytes`, not raw `(bytes, ids)` tuples.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use maskforge_core::index::VocabularyHandle;
use maskforge_core::{
    build_vocabulary, compile_ir, schema_to_ir, CompileOptions, StructuredMatcher, TokenId,
};

const SCHEMA: &str = r#"{
    "type": "object",
    "properties": {"name": {"type": "boolean"}},
    "required": ["name"],
    "additionalProperties": {"type": "boolean"}
}"#;

// Ids 5 and 10 deliberately share one byte string, so the mask must set both.
fn vocab_handle() -> VocabularyHandle {
    let pairs: &[(&[u8], &[u32])] = &[
        (b"{", &[0]),
        (b"}", &[1]),
        (b"\"name\"", &[2]),
        (b"\"extra\"", &[3]),
        (b":", &[4]),
        (b"true", &[5, 10]),
        (b"false", &[6]),
        (b",", &[7]),
        (b"42", &[9]),
        (b"\"oops\"", &[11]),
    ];
    let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
    for &(bytes, ids) in pairs {
        map.insert(bytes.to_vec(), ids.to_vec());
    }
    let vocab = build_vocabulary(99, map).expect("vocab");
    VocabularyHandle::new(Arc::new(vocab)).expect("handle")
}

fn matcher() -> (StructuredMatcher, VocabularyHandle) {
    let ir = schema_to_ir(SCHEMA, CompileOptions::default()).expect("frontend must accept it");
    assert_eq!(ir.diagnostics().len(), 0, "schema must be fully supported");
    assert!(
        ir.requires_structured_backend(),
        "additionalProperties schema needs OpenObject"
    );
    assert!(
        compile_ir(&ir).is_err(),
        "the byte-DFA compiler must refuse an OpenObject node"
    );
    (
        StructuredMatcher::new(
            maskforge_core::StructuredProgram::compile(Arc::new(ir)).expect("plan"),
        )
        .expect("matcher"),
        vocab_handle(),
    )
}

#[test]
fn a_duplicate_byte_string_under_two_ids_sets_both_bits_in_the_mask() {
    let (mut m, handle) = matcher();
    for tok in [b"{".as_slice(), b"\"name\"", b":"] {
        assert!(m.advance(tok).unwrap());
    }
    let mask = m
        .allowed_mask_from_records(handle.mask_vocab_size(), handle.iter_records())
        .unwrap();
    assert!(
        mask.get(TokenId(5)),
        "id 5 shares bytes \"true\" with id 10"
    );
    assert!(
        mask.get(TokenId(10)),
        "id 10 shares bytes \"true\" with id 5"
    );
    assert!(mask.get(TokenId(6)), "false is also a legal boolean value");
}

#[test]
fn known_and_open_fields_accept_both_orderings_through_the_real_vocabulary() {
    for order in [
        [b"\"name\"".as_slice(), b"\"extra\""],
        [b"\"extra\"", b"\"name\""],
    ] {
        let (mut m, _handle) = matcher();
        assert!(m.advance(b"{").unwrap());
        assert!(m.advance(order[0]).unwrap());
        assert!(m.advance(b":").unwrap());
        assert!(m.advance(b"true").unwrap());
        assert!(m.advance(b",").unwrap());
        assert!(m.advance(order[1]).unwrap());
        assert!(m.advance(b":").unwrap());
        assert!(m.advance(b"false").unwrap());
        assert!(m.advance(b"}").unwrap());
        assert!(
            m.is_accepting(),
            "order {order:?} must be a complete, valid document"
        );
    }
}

// A quoted string self-delimits, so a boolean-vs-string mismatch is visible immediately.
#[test]
fn a_self_delimited_type_mismatch_is_excluded_from_the_mask_immediately() {
    let (mut m, handle) = matcher();
    for tok in [b"{".as_slice(), b"\"name\"", b":"] {
        assert!(m.advance(tok).unwrap());
    }
    let mask = m
        .allowed_mask_from_records(handle.mask_vocab_size(), handle.iter_records())
        .unwrap();
    assert!(
        !mask.get(TokenId(11)),
        "\"oops\" must not be allowed where a boolean is required"
    );
    assert!(
        !m.advance(b"\"oops\"").unwrap(),
        "committing the excluded token must fail, not silently succeed"
    );
}

// A digit can never become "true"/"false": a known property's own compiled DFA rejects it as
// soon as it starts, without waiting for a later byte to close the number's span.
#[test]
fn a_non_self_delimited_type_mismatch_is_rejected_as_soon_as_it_starts() {
    let (mut m, handle) = matcher();
    for tok in [b"{".as_slice(), b"\"name\"", b":"] {
        assert!(m.advance(tok).unwrap());
    }
    let mask = m
        .allowed_mask_from_records(handle.mask_vocab_size(), handle.iter_records())
        .unwrap();
    assert!(
        !mask.get(TokenId(9)),
        "a digit is rejected immediately, not deferred to a later closing byte"
    );
    assert!(
        !m.advance(b"42").unwrap(),
        "commit must match the mask verdict"
    );
}

#[test]
fn a_duplicate_key_is_rejected_via_the_real_token_bytes_lookup() {
    let (mut m, handle) = matcher();
    let by_id = |id: u32, tok: &[u8]| {
        assert_eq!(handle.token_bytes(id), Some(tok));
    };
    for (id, tok) in [
        (0u32, b"{".as_slice()),
        (2, b"\"name\""),
        (4, b":"),
        (5, b"true"),
        (7, b","),
    ] {
        by_id(id, tok);
        assert!(m.advance(tok).unwrap(), "token {tok:?} must be accepted");
    }
    by_id(2, b"\"name\"");
    assert!(
        !m.advance(b"\"name\"").unwrap(),
        "the second \"name\" key closes as a duplicate, rejected on this exact token"
    );
}

#[test]
fn advance_never_moves_the_position_on_a_rejected_commit() {
    let (mut m, _handle) = matcher();
    assert!(m.advance(b"{").unwrap());
    assert!(
        !m.advance(b"}").unwrap(),
        "empty object is missing required \"name\""
    );
    assert!(
        m.advance(b"\"name\"").unwrap(),
        "the earlier rejection must not have moved the position"
    );
}

#[test]
fn an_unknown_token_id_is_rejected_never_a_panic() {
    let handle = vocab_handle();
    assert_eq!(handle.token_bytes(9999), None);
}
