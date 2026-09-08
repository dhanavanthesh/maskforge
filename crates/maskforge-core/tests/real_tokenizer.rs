//! Real-tokenizer binding at scale. Loads a real GPT-2 (and, best-effort, a Llama-family)
//! vocabulary, proves the three binding routes agree byte-for-byte over every reachable state of
//! several hard schemas, and reports cold/warm bind timing.
//!
//! Network + heavy, so `#[ignore]` and behind the `huggingface-hub` feature. Run with:
//!   cargo test -p maskforge-core --features huggingface-hub --test real_tokenizer -- --ignored --nocapture
#![cfg(feature = "huggingface-hub")]

use std::sync::Arc;
use std::time::Instant;

use maskforge_core::correctness::oracle::all_states;
use maskforge_core::error::HashState;
use maskforge_core::index::{BindMode, TrieCache, VocabTrie, VocabularyHandle};
use maskforge_core::{
    compile_ir, schema_to_ir, CompileOptions, CompiledArtifact, Provenance, RefEngine, Vocabulary,
};

/// Hard schemas covering the shapes that stress the walk: literals, prefix-overlapping enums,
/// string bodies, arrays, and a closed nested object.
const SCHEMAS: &[(&str, &str)] = &[
    ("boolean", r#"{"type":"boolean"}"#),
    ("null", r#"{"type":"null"}"#),
    ("enum-prefix", r#"{"enum":[1,12]}"#),
    ("enum-triple", r#"{"enum":[1,12,123]}"#),
    ("string", r#"{"type":"string","pattern":"[a-z]+"}"#),
    (
        "array",
        r#"{"type":"array","items":{"type":"boolean"},"minItems":1,"maxItems":3}"#,
    ),
    (
        "object",
        r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"null"}},"required":["a","b"],"additionalProperties":false}"#,
    ),
];

fn engine(json: &str) -> RefEngine {
    compile_ir(&schema_to_ir(json, CompileOptions::default()).expect("ir")).expect("engine")
}

fn prov() -> Provenance {
    Provenance::reference(HashState::Hash([8; 32]))
}

/// Proves Route A == Route C == Route B over every reachable state of every schema, and prints
/// per-schema cold/warm byte-trie bind timing at the real vocabulary size.
fn check_routes_agree(label: &str, vocab: Arc<Vocabulary>) {
    let byte_trie = VocabTrie::build_byte(&vocab).expect("byte trie");
    let handle = VocabularyHandle::new(vocab.clone()).expect("handle");
    let cache = TrieCache::new();
    let bound = cache.bind(&handle).expect("bind");
    println!(
        "{label}: tokens={} byte_trie_nodes={} trie_heap_bytes={}",
        vocab.tokens().len(),
        byte_trie.node_count(),
        byte_trie.heap_bytes()
    );

    for (name, json) in SCHEMAS {
        let a = CompiledArtifact::new(engine(json), vocab.clone(), prov()).expect("naive");
        // The two public production modes on the real vocabulary: packed and lazy, both vs naive.
        let packed = CompiledArtifact::new_trie(
            engine(json),
            vocab.clone(),
            prov(),
            BindMode::TrieJointBytePacked,
            &byte_trie,
        )
        .expect("packed");
        let lazy = CompiledArtifact::new_from_bound_trie(
            &handle,
            engine(json),
            prov(),
            BindMode::TrieJointByteLazy,
            &bound,
        )
        .expect("lazy");

        let cold = Instant::now();
        let fresh_trie = VocabTrie::build_byte(&vocab).expect("byte trie");
        let c_cold = CompiledArtifact::new_trie(
            engine(json),
            vocab.clone(),
            prov(),
            BindMode::TrieJointByte,
            &fresh_trie,
        )
        .expect("route c cold");
        let cold_us = cold.elapsed().as_micros();

        let warm = Instant::now();
        let c_warm = CompiledArtifact::new_trie(
            engine(json),
            vocab.clone(),
            prov(),
            BindMode::TrieJointByte,
            &byte_trie,
        )
        .expect("route c warm");
        let warm_us = warm.elapsed().as_micros();

        let class_trie = VocabTrie::build_class(&vocab, &engine(json)).expect("class trie");
        let b = CompiledArtifact::new_trie(
            engine(json),
            vocab.clone(),
            prov(),
            BindMode::TrieJointClass,
            &class_trie,
        )
        .expect("route b");

        let mut states = 0usize;
        for s in all_states(&a) {
            states += 1;
            let ma = a.allowed_mask(s).expect("A");
            assert_eq!(
                c_cold.allowed_mask(s).unwrap().as_words(),
                ma.as_words(),
                "{label}/{name}: cold C != A at {s:?}"
            );
            assert_eq!(
                c_warm.allowed_mask(s).unwrap().as_words(),
                ma.as_words(),
                "{label}/{name}: warm C != A at {s:?}"
            );
            assert_eq!(
                b.allowed_mask(s).unwrap().as_words(),
                ma.as_words(),
                "{label}/{name}: B != A at {s:?}"
            );
            assert_eq!(
                packed.allowed_mask(s).unwrap().as_words(),
                ma.as_words(),
                "{label}/{name}: packed != A at {s:?}"
            );
            assert_eq!(
                lazy.allowed_mask(s).unwrap().as_words(),
                ma.as_words(),
                "{label}/{name}: lazy != A at {s:?}"
            );
        }
        println!(
            "  {name}: states={states} class_trie_nodes={} cold_c_us={cold_us} warm_c_us={warm_us}",
            class_trie.node_count()
        );
    }
}

#[test]
#[ignore = "network + real GPT-2 vocabulary; run with --features huggingface-hub -- --ignored --nocapture"]
fn gpt2_three_routes_agree_and_report_cold_warm() {
    let vocab = Vocabulary::from_pretrained("openai-community/gpt2", None).expect("gpt2 vocab");
    check_routes_agree("gpt2", Arc::new(vocab));
}

#[test]
#[ignore = "real Qwen2.5 vocabulary; run with --features huggingface-hub -- --ignored --nocapture"]
fn qwen2_5_three_routes_agree_and_report_cold_warm() {
    let vocab = Vocabulary::from_pretrained("Qwen/Qwen2.5-0.5B", None).expect("qwen2.5 vocab");
    check_routes_agree("qwen2.5", Arc::new(vocab));
}

#[test]
#[ignore = "network + a Llama-family tokenizer; run with --features huggingface-hub -- --ignored --nocapture"]
fn llama_family_three_routes_agree_or_skips_on_download_failure() {
    // Real Llama-3.1 (128k) is gated; a Llama-family tokenizer stands in and the test skips honestly
    // if the download is unavailable rather than failing the suite.
    match Vocabulary::from_pretrained("hf-internal-testing/llama-tokenizer", None) {
        Ok(vocab) => check_routes_agree("llama-family", Arc::new(vocab)),
        Err(e) => println!("llama-family: skipped (download unavailable: {e})"),
    }
}

#[test]
#[ignore = "network + a real BLOOM (250k-token) vocabulary; run with --features huggingface-hub -- --ignored --nocapture"]
fn bloom_250k_three_routes_agree_and_report_cold_warm() {
    // BLOOM approximates Gemma-3 scale without requiring its accepted-license access.
    // It is the largest ungated tokenizer used by this differential and timing gate.
    let vocab = Vocabulary::from_pretrained("bigscience/bloom", None).expect("bloom vocab");
    check_routes_agree("bloom-250k", Arc::new(vocab));
}
