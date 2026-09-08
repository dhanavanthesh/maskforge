//! The three-route vocabulary-binding differential, through the public API.
//!
//! Route A is the naive reference walk and the independent anchor; it never touches a trie. Route C
//! is the cached byte-trie walk, Route B the per-compile class-signature trie. For every reachable
//! state of every corpus schema, all three masks must be byte-identical.

use std::sync::Arc;

use maskforge_core::correctness::corpus::corpus_cases;
use maskforge_core::correctness::oracle::all_states;
use maskforge_core::error::HashState;
use maskforge_core::index::{BindMode, TrieCache, VocabTrie, VocabularyHandle};
use maskforge_core::{
    build_vocabulary, CompiledArtifact, Provenance, StateId, TokenId, Vocabulary,
};
use rustc_hash::FxHashMap;

const VOCAB: u32 = 64;

/// A 64-token vocabulary rich enough to traverse every corpus grammar: JSON literals, structural
/// bytes, digits, and letters. EOS = 64.
fn vocab64() -> Vocabulary {
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
    let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
    for (id, bytes) in tokens.into_iter().enumerate() {
        map.insert(bytes, vec![id as u32]);
    }
    build_vocabulary(VOCAB, map).expect("vocab64 builds")
}

fn prov() -> Provenance {
    Provenance::reference(HashState::Hash([8; 32]))
}

/// Builds the naive (Route A) artifact for the named corpus case.
fn naive_artifact(name: &str, vocab: &Arc<Vocabulary>) -> CompiledArtifact {
    let engine = corpus_cases()
        .expect("corpus")
        .into_iter()
        .find(|c| c.name == name)
        .expect("case")
        .engine;
    CompiledArtifact::new(engine, vocab.clone(), prov()).expect("artifact")
}

/// Builds a trie-bound artifact (Route B or C) for the named corpus case.
fn trie_artifact(name: &str, vocab: &Arc<Vocabulary>, mode: BindMode) -> CompiledArtifact {
    let engine = corpus_cases()
        .expect("corpus")
        .into_iter()
        .find(|c| c.name == name)
        .expect("case")
        .engine;
    let trie = match mode {
        BindMode::TrieJointByte => VocabTrie::build_byte(vocab).unwrap(),
        BindMode::TrieJointClass => VocabTrie::build_class(vocab, &engine).unwrap(),
        _ => unreachable!("trie_artifact needs a trie mode"),
    };
    CompiledArtifact::new_trie(engine, vocab.clone(), prov(), mode, &trie).expect("artifact")
}

/// Builds the lazy byte-trie artifact (rows walked on demand) for the named corpus case, through
/// the public opaque-handle API.
fn lazy_artifact(name: &str, handle: &VocabularyHandle, cache: &TrieCache) -> CompiledArtifact {
    let engine = corpus_cases()
        .expect("corpus")
        .into_iter()
        .find(|c| c.name == name)
        .expect("case")
        .engine;
    let bound = cache.bind(handle).expect("bind");
    CompiledArtifact::new_from_bound_trie(
        handle,
        engine,
        prov(),
        BindMode::TrieJointByteLazy,
        &bound,
    )
    .expect("lazy artifact")
}

#[test]
fn lazy_byte_trie_agrees_with_naive_and_eager_over_every_state() {
    let vocab = Arc::new(vocab64());
    let handle = VocabularyHandle::new(vocab.clone()).expect("handle");
    let cache = TrieCache::new();
    for name in case_names() {
        let a = naive_artifact(&name, &vocab);
        let c = trie_artifact(&name, &vocab, BindMode::TrieJointByte);
        let l = lazy_artifact(&name, &handle, &cache);
        assert_eq!(l.bind_mode(), BindMode::TrieJointByteLazy);
        for s in all_states(&a) {
            let ma = a.allowed_mask(s).expect("A");
            let ml = l.allowed_mask(s).expect("lazy");
            let mc = c.allowed_mask(s).expect("C");
            assert_eq!(
                ml.as_words(),
                ma.as_words(),
                "{name}: lazy != naive at {s:?}"
            );
            assert_eq!(
                ml.as_words(),
                mc.as_words(),
                "{name}: lazy != eager at {s:?}"
            );
        }
        // A repeated query on the same state (cache hit) must return the same mask.
        let s0 = a.start();
        assert_eq!(
            l.allowed_mask(s0).unwrap().as_words(),
            l.allowed_mask(s0).unwrap().as_words()
        );
        // Dead and forged states are empty in lazy mode too.
        for s in [a.dead(), StateId(u32::MAX)] {
            assert_eq!(l.allowed_mask(s).unwrap().count_ones(), 0);
        }
    }
}

/// Builds the fully-packed byte-trie artifact (a precomputed mask per state) for a case.
fn packed_artifact(name: &str, handle: &VocabularyHandle, cache: &TrieCache) -> CompiledArtifact {
    let engine = corpus_cases()
        .expect("corpus")
        .into_iter()
        .find(|c| c.name == name)
        .expect("case")
        .engine;
    let bound = cache.bind(handle).expect("bind");
    CompiledArtifact::new_from_bound_trie(
        handle,
        engine,
        prov(),
        BindMode::TrieJointBytePacked,
        &bound,
    )
    .expect("packed artifact")
}

#[test]
fn packed_byte_trie_agrees_with_naive_over_every_state() {
    let vocab = Arc::new(vocab64());
    let handle = VocabularyHandle::new(vocab.clone()).expect("handle");
    let cache = TrieCache::new();
    for name in case_names() {
        let a = naive_artifact(&name, &vocab);
        let p = packed_artifact(&name, &handle, &cache);
        assert_eq!(p.bind_mode(), BindMode::TrieJointBytePacked);
        for s in all_states(&a) {
            assert_eq!(
                p.allowed_mask(s).expect("packed").as_words(),
                a.allowed_mask(s).expect("naive").as_words(),
                "{name}: packed != naive at {s:?}"
            );
        }
        // Dead/forged states are empty; a repeated packed query is identical (flat lookup).
        for s in [a.dead(), StateId(u32::MAX)] {
            assert_eq!(p.allowed_mask(s).unwrap().count_ones(), 0);
        }
        let s0 = a.start();
        assert_eq!(
            p.allowed_mask(s0).unwrap().as_words(),
            p.allowed_mask(s0).unwrap().as_words()
        );
    }
}

/// Extends vocab64 with sparse, duplicate, non-UTF-8, and prefix-overlapping tokens.
/// EOS is 200; all routes must consistently exclude added tokens outside the corpus.
fn adversarial_vocab() -> Vocabulary {
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
    let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
    for (id, bytes) in tokens.into_iter().enumerate() {
        map.insert(bytes, vec![id as u32]);
    }
    // Prefix-overlap chain and a repeated byte sequence bound to two sparse ids.
    map.insert(b"ab".to_vec(), vec![100]);
    map.insert(b"abc".to_vec(), vec![101]);
    map.insert(b"abcd".to_vec(), vec![102, 150]); // one byte sequence, two ids (repeated seq)
                                                  // Non-UTF-8 byte payloads (valid tokens; only malformed tokenizer METADATA errors).
    map.insert(vec![0xFF], vec![160]);
    map.insert(vec![0x80, 0x81], vec![175]);
    build_vocabulary(200, map).expect("adversarial vocab")
}

#[test]
fn packed_lazy_eager_agree_with_naive_on_an_adversarial_vocabulary() {
    let vocab = Arc::new(adversarial_vocab());
    let handle = VocabularyHandle::new(vocab.clone()).expect("handle");
    let cache = TrieCache::new();
    for name in case_names() {
        let a = naive_artifact(&name, &vocab);
        let e = trie_artifact(&name, &vocab, BindMode::TrieJointByte);
        let l = lazy_artifact(&name, &handle, &cache);
        let p = packed_artifact(&name, &handle, &cache);
        for s in all_states(&a) {
            let ma = a.allowed_mask(s).expect("naive");
            assert_eq!(
                e.allowed_mask(s).unwrap().as_words(),
                ma.as_words(),
                "{name}: eager"
            );
            assert_eq!(
                l.allowed_mask(s).unwrap().as_words(),
                ma.as_words(),
                "{name}: lazy"
            );
            assert_eq!(
                p.allowed_mask(s).unwrap().as_words(),
                ma.as_words(),
                "{name}: packed"
            );
        }
    }
}

fn case_names() -> Vec<String> {
    corpus_cases()
        .expect("corpus")
        .into_iter()
        .map(|c| c.name.to_string())
        .collect()
}

#[test]
fn all_three_routes_agree_byte_for_byte_over_every_state() {
    let vocab = Arc::new(vocab64());
    for name in case_names() {
        let a = naive_artifact(&name, &vocab);
        let c = trie_artifact(&name, &vocab, BindMode::TrieJointByte);
        let b = trie_artifact(&name, &vocab, BindMode::TrieJointClass);
        assert_eq!(a.bind_mode(), BindMode::Naive);
        assert_eq!(c.bind_mode(), BindMode::TrieJointByte);
        assert_eq!(b.bind_mode(), BindMode::TrieJointClass);

        for s in all_states(&a) {
            let ma = a.allowed_mask(s).expect("A");
            let mc = c.allowed_mask(s).expect("C");
            let mb = b.allowed_mask(s).expect("B");
            assert_eq!(mc.as_words(), ma.as_words(), "{name}: C != A at {s:?}");
            assert_eq!(mb.as_words(), ma.as_words(), "{name}: B != A at {s:?}");
        }
    }
}

#[test]
fn exhaustive_small_vocab_every_state_every_token_matches_naive() {
    let vocab = Arc::new(vocab64());
    for name in case_names() {
        let a = naive_artifact(&name, &vocab);
        let c = trie_artifact(&name, &vocab, BindMode::TrieJointByte);
        for s in all_states(&a) {
            let ma = a.allowed_mask(s).expect("A");
            let mc = c.allowed_mask(s).expect("C");
            for id in 0..VOCAB {
                assert_eq!(
                    mc.get(TokenId(id)),
                    ma.get(TokenId(id)),
                    "{name}: token {id} at {s:?} disagrees"
                );
            }
        }
    }
}

#[test]
fn dead_and_forged_states_are_empty_both_ways() {
    let vocab = Arc::new(vocab64());
    let a = naive_artifact("suite-type-boolean", &vocab);
    let c = trie_artifact("suite-type-boolean", &vocab, BindMode::TrieJointByte);
    for s in [a.dead(), StateId(u32::MAX)] {
        assert_eq!(a.allowed_mask(s).unwrap().count_ones(), 0);
        assert_eq!(c.allowed_mask(s).unwrap().count_ones(), 0);
    }
}

#[test]
fn open_quote_then_ordinary_char_is_state_dependent_and_agrees() {
    // The string schema: only `"` is live at the start; ordinary characters become live only after
    // the quote. All three routes must reflect the state-dependent liveness identically.
    let vocab = Arc::new(vocab64());
    let a = naive_artifact("syn-string-pattern", &vocab);
    let c = trie_artifact("syn-string-pattern", &vocab, BindMode::TrieJointByte);
    let after_quote = a.walk(a.start(), b"\"").expect("live after opening quote");
    let ma = a.allowed_mask(after_quote).unwrap();
    let mc = c.allowed_mask(after_quote).unwrap();
    assert_eq!(mc.as_words(), ma.as_words());
    // An ordinary character ('c', a byte-token) is allowed after the quote.
    let c_id = vocab.token_ids(b"c").unwrap()[0];
    assert!(ma.get(TokenId(c_id)));
}

#[test]
fn byte_trie_cache_is_reused_across_schemas_on_one_vocabulary() {
    let vocab = Arc::new(vocab64());
    let cache = TrieCache::new();
    let first = cache.get_or_build_byte(&vocab).unwrap();
    let second = cache.get_or_build_byte(&vocab).unwrap();
    assert!(
        Arc::ptr_eq(&first, &second),
        "same vocabulary reuses the trie"
    );

    // The one cached byte-trie binds two different schemas, each byte-identical to its naive walk.
    for name in ["suite-type-boolean", "syn-enum-prefix"] {
        let engine = corpus_cases()
            .unwrap()
            .into_iter()
            .find(|x| x.name == name)
            .unwrap()
            .engine;
        let bound = CompiledArtifact::new_trie(
            engine,
            vocab.clone(),
            prov(),
            BindMode::TrieJointByte,
            &first,
        )
        .unwrap();
        let a = naive_artifact(name, &vocab);
        for s in all_states(&a) {
            assert_eq!(
                bound.allowed_mask(s).unwrap().as_words(),
                a.allowed_mask(s).unwrap().as_words(),
                "{name}: cached-trie mask diverged at {s:?}"
            );
        }
    }
}

#[test]
fn vocabulary_derives_are_untouched_by_binding() {
    // Building tries and a cache must not add interior mutable state to Vocabulary: its value
    // semantics (Clone + PartialEq) are unchanged.
    let vocab = vocab64();
    let clone = vocab.clone();
    let _ = VocabTrie::build_byte(&vocab).unwrap();
    let _ = TrieCache::new().get_or_build_byte(&vocab).unwrap();
    assert_eq!(vocab, clone, "binding must not mutate the vocabulary");
}

#[test]
fn a_large_sampled_vocabulary_agrees_across_routes() {
    // Stands in for a full tokenizer at the Rust layer: many multi-byte tokens with sparse ids.
    let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
    let mut id = 0u32;
    for a in b'a'..=b'z' {
        for b in b'a'..=b'z' {
            map.insert(vec![a, b], vec![id * 3]); // sparse ids
            id += 1;
        }
    }
    for b in b'a'..=b'z' {
        map.insert(vec![b], vec![id * 3]);
        id += 1;
    }
    let vocab = Arc::new(build_vocabulary(id * 3, map).expect("vocab"));

    let engine_a = compile_string_pattern();
    let engine_c = compile_string_pattern();
    let engine_b = compile_string_pattern();
    let trie_c = VocabTrie::build_byte(&vocab).unwrap();
    let trie_b = VocabTrie::build_class(&vocab, &engine_b).unwrap();

    let a = CompiledArtifact::new(engine_a, vocab.clone(), prov()).unwrap();
    let c = CompiledArtifact::new_trie(
        engine_c,
        vocab.clone(),
        prov(),
        BindMode::TrieJointByte,
        &trie_c,
    )
    .unwrap();
    let b = CompiledArtifact::new_trie(
        engine_b,
        vocab.clone(),
        prov(),
        BindMode::TrieJointClass,
        &trie_b,
    )
    .unwrap();
    for s in all_states(&a) {
        let ma = a.allowed_mask(s).unwrap();
        assert_eq!(
            c.allowed_mask(s).unwrap().as_words(),
            ma.as_words(),
            "C at {s:?}"
        );
        assert_eq!(
            b.allowed_mask(s).unwrap().as_words(),
            ma.as_words(),
            "B at {s:?}"
        );
    }
}

fn compile_string_pattern() -> maskforge_core::RefEngine {
    // User patterns use structured Unicode matching; length bounds remain on the byte-DFA path.
    // The generated lowercase tokens satisfy the length-bound case below.
    let ir = maskforge_core::schema_to_ir(
        r#"{"type":"string","minLength":1,"maxLength":2}"#,
        maskforge_core::CompileOptions::default(),
    )
    .expect("ir");
    maskforge_core::compile_ir(&ir).expect("engine")
}
