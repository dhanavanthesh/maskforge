//! Proptest differential: feed generated token streams into every corpus artifact and assert the
//! end-to-end packed path never diverges from the reference primitives. At each step the matcher's
//! position must equal a fresh byte walk and the packed mask must equal the reference mask.
//!
//! The schema-generating side of the differential (a `prop_recursive` `SchemaIR` generator) lives
//! in the crate's own unit tests, because `SchemaIR` is intentionally crate-private and cannot be
//! named from an integration crate. This file exercises the same oracle over the public corpus.

mod common;

use common::all_artifacts;
use maskforge_core::correctness::oracle::{diff_state, walk_prefix};
use maskforge_core::{Matcher, TokenId};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn packed_path_matches_reference_along_token_streams(
        case_idx in any::<prop::sample::Index>(),
        stream in prop::collection::vec(0u32..66, 0..16),
    ) {
        let artifacts = all_artifacts();
        let artifact = &artifacts[case_idx.index(artifacts.len())];

        // The mask agrees at the start state before any token is consumed.
        prop_assert!(diff_state(artifact, artifact.start()).unwrap().agrees());

        let mut matcher = Matcher::new(artifact.clone());
        let mut seen: Vec<u8> = Vec::new();
        for id in stream {
            let token = TokenId(id);
            let bytes = artifact.token_bytes(token);
            match matcher.advance(token) {
                Ok(()) => {
                    seen.extend_from_slice(bytes.expect("advance ok implies a known token"));
                    prop_assert_eq!(matcher.state(), walk_prefix(artifact, &seen));
                    prop_assert!(diff_state(artifact, matcher.state()).unwrap().agrees());
                }
                Err(_) => {
                    // A rejected token: the reference must reject the same bytes, then the stream
                    // is finished (both engines reject).
                    let reference_rejects = match bytes {
                        None => true,
                        Some(b) => {
                            let mut probe = seen.clone();
                            probe.extend_from_slice(b);
                            artifact.walk(artifact.start(), &probe).is_none()
                        }
                    };
                    prop_assert!(reference_rejects);
                    break;
                }
            }
        }
    }
}
