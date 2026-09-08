//! Compile one schema once, bind it to two different vocabularies.
//! Run with: cargo run --example cached_multi_vocabulary -p maskforge-core

use std::sync::Arc;

use maskforge_core::index::VocabularyHandle;
use maskforge_core::{build_vocabulary, compile_ir, schema_to_ir, CompileOptions};
use maskforge_core::{CompiledArtifact, HashState, Matcher, Provenance};
use rustc_hash::FxHashMap;

fn byte_vocabulary(extra: &[(&str, u32)]) -> FxHashMap<Vec<u8>, Vec<u32>> {
    let mut map: FxHashMap<Vec<u8>, Vec<u32>> =
        (0u32..256).map(|b| (vec![b as u8], vec![b])).collect();
    for (bytes, id) in extra {
        map.insert(bytes.as_bytes().to_vec(), vec![*id]);
    }
    map
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = r#"{"type":"boolean"}"#;

    let ir = schema_to_ir(schema, CompileOptions::default())?;
    let engine = Arc::new(compile_ir(&ir)?);
    let source_hash = HashState::Hash(*blake3::hash(schema.as_bytes()).as_bytes());

    let vocab_a = build_vocabulary(256, byte_vocabulary(&[("extra", 257)]))?;
    let handle_a = VocabularyHandle::new(Arc::new(vocab_a))?;
    let artifact_a = Arc::new(CompiledArtifact::new_from_handle_shared(
        &handle_a,
        Arc::clone(&engine),
        Provenance::reference(source_hash),
    )?);

    let vocab_b = build_vocabulary(257, byte_vocabulary(&[("third", 258)]))?;
    let handle_b = VocabularyHandle::new(Arc::new(vocab_b))?;
    let artifact_b = Arc::new(CompiledArtifact::new_from_handle_shared(
        &handle_b,
        Arc::clone(&engine),
        Provenance::reference(source_hash),
    )?);

    let mut matcher_a = Matcher::new(Arc::clone(&artifact_a));
    let mut matcher_b = Matcher::new(Arc::clone(&artifact_b));
    for byte in b"true" {
        matcher_a.advance(maskforge_core::TokenId::try_from(*byte as usize)?)?;
    }
    for byte in b"false" {
        matcher_b.advance(maskforge_core::TokenId::try_from(*byte as usize)?)?;
    }

    println!(
        "vocab A accepting after 'true': {}",
        matcher_a.is_accepting()
    );
    println!(
        "vocab B accepting after 'false': {}",
        matcher_b.is_accepting()
    );

    Ok(())
}
