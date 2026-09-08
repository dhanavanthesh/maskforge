//! Fuzz target: the same semantic oracle as the differential harness, driven by arbitrary bytes.
//! NIGHTLY toolchain, LINUX CI only: `cargo +nightly fuzz run diff_mask`.
//!
//! `SchemaIR` is crate-private and cannot be generated from this separate crate, so the fuzzer
//! selects a corpus schema and generates a token stream (bytes fed as single-byte tokens), then
//! asserts the end-to-end packed path never diverges from the reference primitives. Inputs are
//! bounded (stream length capped) to mirror the proptest bounds.

#![no_main]

use std::sync::Arc;

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use rustc_hash::FxHashMap;

use maskforge_core::correctness::corpus::corpus_cases;
use maskforge_core::correctness::oracle::{first_state_mismatch, matcher_path_agrees};
use maskforge_core::error::HashState;
use maskforge_core::{build_vocabulary, CompiledArtifact, Provenance, TokenId};

const MAX_STREAM: usize = 32;

#[derive(Debug)]
struct Input {
    case: u8,
    stream: Vec<u8>,
}

impl<'a> Arbitrary<'a> for Input {
    // Read the stream length first and cap it, so adversarial input cannot drive an unbounded
    // allocation before the bound is applied.
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let case = u8::arbitrary(u)?;
        let len = usize::from(u8::arbitrary(u)?) % (MAX_STREAM + 1);
        let mut stream = Vec::with_capacity(len);
        for _ in 0..len {
            stream.push(u8::arbitrary(u)?);
        }
        Ok(Self { case, stream })
    }
}

/// One single-byte token per byte value 0..=127 (id = byte value); EOS = 200.
fn byte_vocab() -> maskforge_core::Vocabulary {
    let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
    for b in 0u32..=127 {
        map.insert(vec![b as u8], vec![b]);
    }
    build_vocabulary(200, map).expect("byte vocab")
}

fuzz_target!(|input: Input| {
    let mut cases = corpus_cases().expect("corpus builds");
    if cases.is_empty() {
        return;
    }
    let idx = input.case as usize % cases.len();
    let engine = cases.swap_remove(idx).engine;
    let artifact = Arc::new(
        CompiledArtifact::new(
            engine,
            byte_vocab(),
            Provenance::reference(HashState::Hash([1; 32])),
        )
        .expect("artifact"),
    );

    assert_eq!(
        first_state_mismatch(&artifact).expect("sweep"),
        None,
        "a reachable state's packed mask diverged from the reference"
    );

    let tokens: Vec<TokenId> = input
        .stream
        .iter()
        .take(MAX_STREAM)
        .map(|&b| TokenId(u32::from(b)))
        .collect();
    assert!(
        matcher_path_agrees(&artifact, &tokens).expect("path"),
        "matcher path diverged from a fresh reference walk"
    );
});
