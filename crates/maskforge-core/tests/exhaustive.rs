//! The exhaustive semantic-equivalence sweep. `#[ignore]` by default; run with
//! `cargo test -p maskforge-core --test exhaustive -- --include-ignored`.
//!
//! For every corpus engine, over every reachable state and every one of the 64 vocabulary tokens,
//! the end-to-end packed path must agree with the reference primitives: masks bit-for-bit, each
//! mask bit equal to whether that token has a byte path, the three predicates self-consistent, and
//! EOS legal exactly where the state accepts and never a mask bit.

use std::sync::Arc;

use maskforge_core::correctness::corpus::corpus_cases;
use maskforge_core::correctness::oracle::{all_states, first_state_mismatch};
use maskforge_core::error::HashState;
use maskforge_core::index::{BindMode, VocabTrie};
use maskforge_core::{build_vocabulary, CompiledArtifact, Provenance, StateId, TokenId};
use rustc_hash::FxHashMap;

const VOCAB: u32 = 64;

/// A vocabulary of exactly 64 ordinary tokens (ids 0..63) rich enough to traverse every corpus
/// grammar: the JSON literals, the structural bytes, digits, and letters. EOS = 64.
fn vocab64() -> maskforge_core::Vocabulary {
    let mut tokens: Vec<Vec<u8>> = vec![b"true".to_vec(), b"false".to_vec(), b"null".to_vec()];
    for &b in b"[]{},:\"".iter() {
        tokens.push(vec![b]);
    }
    for b in b'0'..=b'9' {
        tokens.push(vec![b]);
    }
    for b in b'a'..=b'z' {
        tokens.push(vec![b]);
    }
    for b in b'A'..=b'Z' {
        if tokens.len() >= VOCAB as usize {
            break;
        }
        tokens.push(vec![b]);
    }
    assert_eq!(tokens.len(), VOCAB as usize);
    let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
    for (id, bytes) in tokens.into_iter().enumerate() {
        map.insert(bytes, vec![id as u32]);
    }
    build_vocabulary(VOCAB, map).expect("vocab64 builds")
}

fn artifacts() -> Vec<(String, Arc<CompiledArtifact>)> {
    corpus_cases()
        .expect("corpus builds")
        .into_iter()
        .map(|c| {
            let artifact = CompiledArtifact::new(
                c.engine,
                std::sync::Arc::new(vocab64()),
                Provenance::reference(HashState::Hash([8; 32])),
            )
            .expect("artifact builds");
            (c.name.to_string(), Arc::new(artifact))
        })
        .collect()
}

#[test]
#[ignore = "exhaustive sweep; run with --include-ignored"]
fn every_reachable_state_by_every_token_agrees_with_the_reference() {
    let eos = TokenId(VOCAB);
    for (name, artifact) in artifacts() {
        assert_eq!(
            first_state_mismatch(&artifact).expect("sweep"),
            None,
            "{name}: a reachable state's packed mask diverged from the reference"
        );
        for s in all_states(&artifact) {
            let mask = artifact.allowed_mask(s).expect("mask");
            for id in 0..VOCAB {
                let token = TokenId(id);
                let has_path = artifact
                    .token_bytes(token)
                    .and_then(|b| artifact.walk(s, b))
                    .is_some();
                assert_eq!(
                    mask.get(token),
                    has_path,
                    "{name}: state {s:?} token {id} mask bit disagrees with byte successor"
                );
            }
            // Every one of the 256 byte successors lands in range or on DEAD.
            for b in 0u16..256 {
                if let Some(t) = artifact.walk(s, &[b as u8]) {
                    assert!(
                        (t.get() as usize) < artifact.state_count(),
                        "{name}: state {s:?} byte {b} left the state space"
                    );
                }
            }
            assert_eq!(
                artifact.is_dead(s),
                !artifact.is_accepting(s) && !artifact.can_continue(s),
                "{name}: state {s:?} breaks is_dead == !accept && !continue"
            );
            assert_eq!(
                artifact.eos_legal(s),
                artifact.is_accepting(s),
                "{name}: state {s:?} eos_legal != is_accepting"
            );
            assert!(!mask.get(eos), "{name}: EOS bit set at state {s:?}");
        }

        // DEAD and a forged out-of-range state reject every byte permanently and stay dead.
        let dead = artifact.dead();
        let forged = StateId(u32::MAX);
        assert!(
            artifact.is_dead(forged),
            "{name}: forged state must be dead-equivalent"
        );
        for b in 0u16..256 {
            let byte = [b as u8];
            assert!(
                artifact.walk(dead, &byte).is_none(),
                "{name}: DEAD escaped on byte {b}"
            );
            assert!(
                artifact.walk(forged, &byte).is_none(),
                "{name}: forged state escaped on byte {b}"
            );
        }
    }
}

/// Both trie routes bound to the same 64-token vocabulary, over every reachable state and every
/// token, must be byte-identical to the naive walk.
#[test]
#[ignore = "exhaustive sweep; run with --include-ignored"]
fn every_state_every_token_agrees_across_naive_byte_trie_and_class_trie() {
    let vocab = std::sync::Arc::new(vocab64());
    for case in corpus_cases().expect("corpus") {
        let name = case.name;
        let naive =
            CompiledArtifact::new(case.engine, vocab.clone(), reference_provenance()).unwrap();

        let byte_engine = corpus_engine(name);
        let byte_trie = VocabTrie::build_byte(&vocab).unwrap();
        let byte = CompiledArtifact::new_trie(
            byte_engine,
            vocab.clone(),
            reference_provenance(),
            BindMode::TrieJointByte,
            &byte_trie,
        )
        .unwrap();

        let class_engine = corpus_engine(name);
        let class_trie = VocabTrie::build_class(&vocab, &class_engine).unwrap();
        let class = CompiledArtifact::new_trie(
            class_engine,
            vocab.clone(),
            reference_provenance(),
            BindMode::TrieJointClass,
            &class_trie,
        )
        .unwrap();

        for s in all_states(&naive) {
            let a = naive.allowed_mask(s).unwrap();
            let c = byte.allowed_mask(s).unwrap();
            let b = class.allowed_mask(s).unwrap();
            assert_eq!(
                c.as_words(),
                a.as_words(),
                "{name}: byte-trie != naive at {s:?}"
            );
            assert_eq!(
                b.as_words(),
                a.as_words(),
                "{name}: class-trie != naive at {s:?}"
            );
        }
    }
}

fn reference_provenance() -> Provenance {
    Provenance::reference(HashState::Hash([8; 32]))
}

fn corpus_engine(name: &str) -> maskforge_core::RefEngine {
    corpus_cases()
        .expect("corpus")
        .into_iter()
        .find(|c| c.name == name)
        .expect("case")
        .engine
}
