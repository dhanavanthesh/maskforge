//! Shared fixtures for the integration tests: a public-API vocabulary and artifacts built from
//! the correctness corpus. Everything here uses only the crate's public surface.
#![allow(dead_code)] // each test binary links only the helpers it uses.

use std::sync::{Arc, OnceLock};

use maskforge_core::correctness::corpus::corpus_cases;
use maskforge_core::error::HashState;
use maskforge_core::{build_vocabulary, CompiledArtifact, Provenance, Vocabulary};
use rustc_hash::FxHashMap;

/// The EOS id. Its bit exists in the mask (width `EOS_ID + 1`) but is never set.
pub const EOS_ID: u32 = 64;

/// A shared vocabulary covering word-boundary ids (0, 31, 32, 63), a duplicate-byte token with
/// two ids, and a non-multiple-of-32 width (65). EOS = 64.
#[must_use]
pub fn shared_vocab() -> Vocabulary {
    let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
    let singles: &[(&[u8], u32)] = &[
        (b"true", 0),
        (b"false", 1),
        (b"null", 2),
        (b"[", 3),
        (b"]", 4),
        (b",", 5),
        (b"{", 6),
        (b"}", 7),
        (b"1", 31),
        (b"2", 32),
        (b"12", 63),
    ];
    for &(bytes, id) in singles {
        map.insert(bytes.to_vec(), vec![id]);
    }
    // A byte sequence bound to two distinct ids (both must be set together in a mask). "tr" is a
    // prefix of "true", so it is consumable from the boolean engine's start state.
    map.insert(b"tr".to_vec(), vec![20, 21]);
    build_vocabulary(EOS_ID, map).expect("shared vocab builds")
}

/// Builds the artifact for the named corpus case, bound to the shared vocabulary.
#[must_use]
pub fn artifact(case_name: &str) -> Arc<CompiledArtifact> {
    let engine = corpus_cases()
        .expect("corpus builds")
        .into_iter()
        .find(|c| c.name == case_name)
        .unwrap_or_else(|| panic!("no corpus case named {case_name}"))
        .engine;
    let artifact = CompiledArtifact::new(
        engine,
        Arc::new(shared_vocab()),
        Provenance::reference(HashState::Hash([5; 32])),
    )
    .expect("artifact builds");
    Arc::new(artifact)
}

/// Every corpus artifact, built once and cached (the corpus is deterministic and read-only).
#[must_use]
pub fn all_artifacts() -> &'static [Arc<CompiledArtifact>] {
    static ARTIFACTS: OnceLock<Vec<Arc<CompiledArtifact>>> = OnceLock::new();
    ARTIFACTS.get_or_init(|| {
        corpus_cases()
            .expect("corpus builds")
            .into_iter()
            .map(|c| {
                Arc::new(
                    CompiledArtifact::new(
                        c.engine,
                        Arc::new(shared_vocab()),
                        Provenance::reference(HashState::Hash([5; 32])),
                    )
                    .expect("artifact builds"),
                )
            })
            .collect()
    })
}
