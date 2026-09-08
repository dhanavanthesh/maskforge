//! The four-combination corpus plus the adversarial-vocabulary checks. Every combination of the
//! three predicates is demonstrably reached, DEAD is shown permanent, EOS stays a side predicate,
//! and the mask ABI holds at word boundaries.

mod common;

use common::{artifact, shared_vocab, EOS_ID};
use maskforge_core::correctness::oracle::{classify, walk_prefix, Classification};
use maskforge_core::error::ErrorCode;
use maskforge_core::error::HashState;
use maskforge_core::{build_vocabulary, CompiledArtifact, Matcher, Provenance, StateId, TokenId};
use rustc_hash::FxHashMap;

fn at(art: &CompiledArtifact, prefix: &[u8]) -> Classification {
    classify(art, walk_prefix(art, prefix))
}

#[test]
fn every_predicate_combination_is_reached() {
    let boolean = artifact("suite-type-boolean");
    let prefix_enum = artifact("syn-enum-prefix");

    // accept AND continue: after "1", "1" is a complete value and "12" extends it.
    let accept_continue = at(&prefix_enum, b"1");
    assert!(
        accept_continue.is_accepting && accept_continue.can_continue && !accept_continue.is_dead
    );

    // accept AND NOT continue: a closed, complete value.
    let accept_closed = at(&boolean, b"true");
    assert!(accept_closed.is_accepting && !accept_closed.can_continue && !accept_closed.is_dead);

    // NOT accept AND continue: ordinary mid-generation.
    let mid = at(&boolean, b"tru");
    assert!(!mid.is_accepting && mid.can_continue && !mid.is_dead);

    // NOT accept AND NOT continue: the single DEAD sentinel.
    let dead = classify(&boolean, boolean.dead());
    assert!(!dead.is_accepting && !dead.can_continue && dead.is_dead);
    assert_eq!(at(&boolean, b"xyz"), dead);
}

#[test]
fn dead_state_self_loops_on_all_256_bytes_and_stays_dead() {
    let boolean = artifact("suite-type-boolean");
    let dead = boolean.dead();
    assert!(boolean.is_dead(dead));
    for b in 0u8..=255 {
        assert!(
            boolean.walk(dead, &[b]).is_none(),
            "byte {b} from DEAD must stay DEAD"
        );
    }
}

#[test]
fn eos_is_a_side_predicate_and_never_a_mask_bit() {
    for name in [
        "suite-type-boolean",
        "syn-enum-prefix",
        "syn-array-unbounded",
    ] {
        let art = artifact(name);
        for raw in 0..art.state_count() as u32 {
            let s = StateId::try_from(raw as usize).unwrap();
            assert_eq!(
                art.eos_legal(s),
                art.is_accepting(s),
                "{name}: eos != accept"
            );
            let mask = art.allowed_mask(s).expect("mask");
            assert!(
                !mask.get(TokenId(EOS_ID)),
                "{name}: EOS bit set at state {raw}"
            );
        }
    }
}

#[test]
fn duplicate_byte_sequence_sets_all_its_ids_together() {
    // "tr" (ids 20 and 21) is consumable from the boolean engine's start.
    let boolean = artifact("suite-type-boolean");
    let mask = boolean.allowed_mask(boolean.start()).expect("mask");
    assert!(
        mask.get(TokenId(20)) && mask.get(TokenId(21)),
        "both ids sharing the bytes must be set together"
    );
}

#[test]
fn empty_byte_token_is_rejected_at_the_vocab_choke_point() {
    let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
    map.insert(Vec::new(), vec![1]);
    assert_eq!(
        build_vocabulary(0, map).unwrap_err().code,
        ErrorCode::EmptyToken
    );
}

#[test]
fn mask_abi_holds_at_word_boundaries_and_non_multiple_of_32_width() {
    let boolean = artifact("suite-type-boolean");
    // vocab width is EOS_ID + 1 = 65, not a multiple of 32.
    assert_eq!(boolean.vocab_size(), (EOS_ID + 1) as usize);
    assert_eq!(boolean.words_per_row(), 3);

    // The state after "1" allows the boundary ids 31 ("1") and 63 ("12") in the enum-prefix engine.
    let prefix_enum = artifact("syn-enum-prefix");
    let s1 = walk_prefix(&prefix_enum, b"1");
    let mut out = vec![0u32; prefix_enum.words_per_row()];
    prefix_enum
        .write_mask_into(&[s1], &mut out, None)
        .expect("fill");
    let bit = |id: u32| out[id as usize / 32] & (1u32 << (id % 32)) != 0;
    assert!(bit(32), "id 32 (\"2\") allowed after \"1\"");
    // The final word covers ids 64..95; only id 64 (EOS) exists there and is never set, so every
    // bit of the final word is zero.
    assert_eq!(out[2], 0, "unused high bits must be zero");
}

#[test]
fn write_mask_into_rejects_a_mismatched_buffer_length() {
    let boolean = artifact("suite-type-boolean");
    let s = boolean.start();
    let mut wrong = vec![0u32; boolean.words_per_row() + 1];
    assert_eq!(
        boolean
            .write_mask_into(&[s], &mut wrong, None)
            .unwrap_err()
            .code,
        ErrorCode::ArtifactOutOfBounds
    );
}

#[test]
fn a_rejected_token_does_not_advance_the_matcher() {
    // A rejected token leaves the live position unchanged; only an empty language reaches DEAD.
    let boolean = artifact("suite-type-boolean");
    let mut m = Matcher::new(boolean);
    let start = m.state();
    assert!(m.advance(TokenId(2)).is_err()); // "null" is not a boolean
    assert_eq!(
        m.state(),
        start,
        "a rejected token must not move the matcher"
    );
}

#[test]
fn artifact_validation_rejects_incomplete_provenance() {
    let engine = maskforge_core::correctness::corpus::corpus_cases()
        .expect("corpus")
        .into_iter()
        .find(|c| c.name == "suite-type-null")
        .unwrap()
        .engine;
    let err = CompiledArtifact::new(
        engine,
        std::sync::Arc::new(shared_vocab()),
        Provenance::reference(HashState::Unhashed),
    )
    .unwrap_err();
    assert_eq!(err.code, ErrorCode::ProvenanceIncomplete);
}
