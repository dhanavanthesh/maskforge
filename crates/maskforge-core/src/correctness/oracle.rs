//! Provides semantic checks for reference engines and matcher behavior.

use std::sync::Arc;

use crate::error::CompileError;
use crate::primitives::StateId;
use crate::runtime::{CompiledArtifact, Matcher};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Classification {
    pub is_accepting: bool,
    pub can_continue: bool,
    pub is_dead: bool,
    pub eos_legal: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ByteVerdict {
    Accept,
    Reject,
    Incomplete,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct StateDiff {
    pub state: StateId,
    pub packed_equals_bitmask: bool,
    pub packed_equals_independent: bool,
    pub mask_matches_successors: bool,
    pub predicates_self_consistent: bool,
    pub byte_successors_in_range: bool,
}

impl StateDiff {
    #[must_use]
    pub fn agrees(&self) -> bool {
        self.packed_equals_bitmask
            && self.packed_equals_independent
            && self.mask_matches_successors
            && self.predicates_self_consistent
            && self.byte_successors_in_range
    }
}

#[must_use]
pub fn classify(artifact: &CompiledArtifact, s: StateId) -> Classification {
    Classification {
        is_accepting: artifact.is_accepting(s),
        can_continue: artifact.can_continue(s),
        is_dead: artifact.is_dead(s),
        eos_legal: artifact.eos_legal(s),
    }
}

#[must_use]
pub fn walk_prefix(artifact: &CompiledArtifact, prefix: &[u8]) -> StateId {
    artifact
        .walk(artifact.start(), prefix)
        .unwrap_or_else(|| artifact.dead())
}

#[must_use]
pub fn byte_language_verdict(artifact: &CompiledArtifact, bytes: &[u8]) -> ByteVerdict {
    let s = walk_prefix(artifact, bytes);
    if artifact.is_dead(s) {
        ByteVerdict::Reject
    } else if artifact.is_accepting(s) {
        ByteVerdict::Accept
    } else {
        ByteVerdict::Incomplete
    }
}

#[must_use]
pub fn all_states(artifact: &CompiledArtifact) -> Vec<StateId> {
    (0..artifact.state_count())
        .filter_map(|raw| StateId::try_from(raw).ok())
        .collect()
}

#[must_use]
pub fn independent_mask_words(artifact: &CompiledArtifact, s: StateId) -> Vec<u32> {
    let mut words = vec![0u32; artifact.words_per_row()];
    for id in artifact.token_ids() {
        if artifact
            .token_bytes(id)
            .and_then(|b| artifact.walk(s, b))
            .is_some()
        {
            let i = id.get() as usize;
            words[i / 32] |= 1u32 << (i % 32);
        }
    }
    words
}

pub fn mask_matches_successors(
    artifact: &CompiledArtifact,
    s: StateId,
) -> Result<bool, CompileError> {
    let reference = artifact.allowed_mask(s)?;
    for id in artifact.token_ids() {
        let has_path = artifact
            .token_bytes(id)
            .and_then(|b| artifact.walk(s, b))
            .is_some();
        if reference.get(id) != has_path {
            return Ok(false);
        }
    }
    Ok(true)
}

#[must_use]
pub fn predicates_self_consistent(artifact: &CompiledArtifact, s: StateId) -> bool {
    let c = classify(artifact, s);
    c.is_dead == (!c.is_accepting && !c.can_continue) && c.eos_legal == c.is_accepting
}

#[must_use]
pub fn byte_successors_in_range(artifact: &CompiledArtifact, s: StateId) -> bool {
    (0u16..256).all(|b| {
        artifact
            .walk(s, &[b as u8])
            .is_none_or(|t| (t.get() as usize) < artifact.state_count())
    })
}

pub fn diff_state(artifact: &CompiledArtifact, s: StateId) -> Result<StateDiff, CompileError> {
    let bitmask = artifact.allowed_mask(s)?;
    let mut packed = vec![0u32; artifact.words_per_row()];
    artifact.write_mask_into(&[s], &mut packed, None)?;
    let independent = independent_mask_words(artifact, s);
    Ok(StateDiff {
        state: s,
        packed_equals_bitmask: packed.as_slice() == bitmask.as_words(),
        packed_equals_independent: packed == independent,
        mask_matches_successors: mask_matches_successors(artifact, s)?,
        predicates_self_consistent: predicates_self_consistent(artifact, s),
        byte_successors_in_range: byte_successors_in_range(artifact, s),
    })
}

pub fn first_state_mismatch(artifact: &CompiledArtifact) -> Result<Option<StateId>, CompileError> {
    for s in all_states(artifact) {
        if !diff_state(artifact, s)?.agrees() {
            return Ok(Some(s));
        }
    }
    Ok(None)
}

pub fn matcher_path_agrees(
    artifact: &Arc<CompiledArtifact>,
    tokens: &[crate::primitives::TokenId],
) -> Result<bool, CompileError> {
    let mut matcher = Matcher::new(artifact.clone());
    let mut seen: Vec<u8> = Vec::new();
    for &token in tokens {
        let bytes = artifact.token_bytes(token);
        match matcher.advance(token) {
            Ok(()) => {
                let b = bytes.expect("a successful advance implies a known token");
                seen.extend_from_slice(b);
                let expected = walk_prefix(artifact, &seen);
                if matcher.state() != expected {
                    return Ok(false);
                }
            }
            Err(_) => {
                let reference_rejects = match bytes {
                    None => true,
                    Some(b) => {
                        let mut probe = seen.clone();
                        probe.extend_from_slice(b);
                        artifact.walk(artifact.start(), &probe).is_none()
                    }
                };
                return Ok(reference_rejects);
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::super::reference_engine::build_reference;
    use super::super::schema_skeleton::{KeyOrder, ScalarLit, SchemaIR};
    use super::*;
    use crate::error::{HashState, Provenance};
    use crate::primitives::TokenId;
    use crate::vocab::{build_vocabulary, Vocabulary};
    use proptest::prelude::*;
    use rustc_hash::FxHashMap as Map;

    fn small_vocab() -> Vocabulary {
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        for (bytes, id) in [
            (b"true".as_slice(), 0u32),
            (b"false", 1),
            (b"null", 2),
            (b"[", 3),
            (b"]", 4),
            (b",", 5),
            (b"{", 6),
        ] {
            map.insert(bytes.to_vec(), vec![id]);
        }
        build_vocabulary(7, map).expect("vocab")
    }

    fn artifact_for(ir: &SchemaIR) -> Option<Arc<CompiledArtifact>> {
        let engine = build_reference(ir).ok()?;
        let artifact = CompiledArtifact::new(
            engine,
            Arc::new(small_vocab()),
            Provenance::reference(HashState::Hash([9; 32])),
        )
        .expect("artifact");
        Some(Arc::new(artifact))
    }

    fn arb_scalar() -> impl Strategy<Value = ScalarLit> {
        prop_oneof![
            Just(ScalarLit::Null),
            any::<bool>().prop_map(ScalarLit::Bool),
            (0i64..1000).prop_map(ScalarLit::Int),
            "[a-z]{0,3}".prop_map(ScalarLit::Str),
        ]
    }

    fn dedup_keys(fields: Vec<(String, SchemaIR)>) -> Vec<(String, SchemaIR)> {
        let mut seen: rustc_hash::FxHashSet<String> = rustc_hash::FxHashSet::default();
        fields
            .into_iter()
            .filter(|(k, _)| seen.insert(k.clone()))
            .collect()
    }

    fn arb_schema() -> impl Strategy<Value = SchemaIR> {
        let leaf = prop_oneof![
            Just(SchemaIR::Null),
            Just(SchemaIR::Boolean),
            "[a-z]{0,3}".prop_map(|value| SchemaIR::StringConst { value }),
            prop::collection::vec(arb_scalar(), 1..4).prop_map(|values| SchemaIR::Enum { values }),
        ];
        leaf.prop_recursive(4, 32, 5, |inner| {
            prop_oneof![
                (inner.clone(), 0u32..3, prop::option::of(0u32..3)).prop_map(
                    |(items, min, max)| SchemaIR::Array {
                        items: Box::new(items),
                        min,
                        max: max.map(|m| m.max(min)),
                    }
                ),
                prop::collection::vec(("[a-z]{1,3}", inner), 1..4).prop_map(|fields| {
                    SchemaIR::Object {
                        fields: dedup_keys(fields),
                        order: KeyOrder::AsDeclared,
                    }
                }),
            ]
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        #[test]
        fn generated_schema_packed_path_equals_reference(
            ir in arb_schema(),
            stream in prop::collection::vec(0u32..9, 0..12),
        ) {
            let Some(artifact) = artifact_for(&ir) else { return Ok(()); };
            for s in all_states(&artifact) {
                let diff = diff_state(&artifact, s).expect("diff");
                prop_assert!(diff.agrees(), "state {:?} diverged: {:?}", s, diff);
            }
            let tokens: Vec<TokenId> = stream.into_iter().map(TokenId).collect();
            prop_assert!(matcher_path_agrees(&artifact, &tokens).expect("path"));
        }
    }

    #[test]
    fn byte_language_verdict_classifies_accept_incomplete_reject() {
        let artifact = artifact_for(&SchemaIR::Boolean).expect("boolean");
        assert_eq!(
            byte_language_verdict(&artifact, b"true"),
            ByteVerdict::Accept
        );
        assert_eq!(
            byte_language_verdict(&artifact, b"tru"),
            ByteVerdict::Incomplete
        );
        assert_eq!(
            byte_language_verdict(&artifact, b"xyz"),
            ByteVerdict::Reject
        );
    }

    #[test]
    fn corpus_has_no_state_mismatch() {
        for case in crate::correctness::corpus::corpus_cases().expect("corpus") {
            let artifact = CompiledArtifact::new(
                case.engine,
                Arc::new(small_vocab()),
                Provenance::reference(HashState::Hash([3; 32])),
            )
            .expect("artifact");
            assert_eq!(
                first_state_mismatch(&artifact).expect("sweep"),
                None,
                "{} had a state mismatch",
                case.name
            );
        }
    }
}
