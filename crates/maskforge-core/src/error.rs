//! The error spine: stable machine-readable codes are the contract; the English `message` is advisory.

use crate::primitives::TokenId;

/// Stable, machine-readable error code. Callers (including Python) key on this, never on text.
#[non_exhaustive]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum ErrorCode {
    /// A keyword/shape the engine deliberately does not support (no silent downgrade).
    Unsupported,
    /// Structurally invalid input (e.g. empty enum, array `max < min`, duplicate object keys).
    Malformed,
    /// A `$ref`/recursion cycle was detected.
    RecursionDetected,
    /// A recursive semantic equation is non-monotone and has no defined bounded fixed point.
    NonMonotoneRecursion,
    /// A URI or URI reference is malformed or cannot be resolved against its active base.
    InvalidUri,
    /// A referenced schema resource, pointer, or anchor cannot be resolved.
    ReferenceResolution,
    /// Two schema resources claim the same canonical URI.
    DuplicateResource,
    /// Two static or dynamic anchors claim the same name in one resource.
    DuplicateAnchor,
    /// A resource cap was exceeded. The variant is a typed seam; it enforces no numeric bound.
    InternalLimitExceeded,
    /// A vocabulary token had zero bytes.
    EmptyToken,
    /// Tokenizer metadata (not a payload) was malformed.
    MalformedTokenizer,
    /// A packed mask access fell outside the buffer.
    ArtifactOutOfBounds,
    /// A provenance field required by its own profile was absent.
    ProvenanceIncomplete,
    /// A token was fed to a dead/non-advancing matcher.
    IllegalToken,
    /// A cached artifact did not match its expected fingerprint.
    ArtifactMismatch,
    /// The bounded diagnostics buffer filled before lowering finished.
    TooManyDiagnostics,
}

/// The compilation stage an error arose in.
#[non_exhaustive]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Stage {
    /// Schema-frontend parsing.
    L1,
    /// IR lowering and canonicalization.
    L2,
    /// Byte-level automaton construction.
    L3,
    /// Vocabulary binding.
    L4Bind,
    /// Vocabulary construction, the single validated entry point.
    VocabBuild,
    /// Compiled-artifact validation.
    ArtifactValidate,
}

/// Resource-limit kinds. The variants and error type are stable; no numeric bound is enforced
/// for any of them.
#[non_exhaustive]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum LimitKind {
    /// Number of members in an enum.
    EnumSize,
    /// Length of an array.
    ArrayLength,
    /// Number of IR nodes.
    NodeCount,
    /// Depth of nested recursion.
    RecursionDepth,
    /// Number of reference states.
    StateCount,
    /// Number of stored transitions.
    TransitionCount,
    /// Number of independently orderable shuffle fields.
    ShuffleFields,
    /// Number of states admitted by one shuffle construction.
    ShuffleStates,
    /// Number of edges admitted by one shuffle construction.
    ShuffleEdges,
    /// Number of branches admitted by one product construction.
    ProductBranches,
    /// Number of states admitted by one product construction.
    ProductStates,
    /// Number of edges admitted by one product construction.
    ProductEdges,
    /// Size of the vocabulary.
    VocabSize,
    /// Byte length of a single token.
    TokenByteLen,
    /// Size in bytes of a compiled artifact.
    ArtifactBytes,
    /// Number of properties observed on one object instance.
    PropertyCount,
    /// Byte length of the accumulated document a session has processed.
    DocumentBytes,
    /// Byte length of one object property key.
    KeyBytes,
    /// Byte length of one string value.
    StringBytes,
    /// Byte length of one number literal.
    NumberBytes,
    /// Number of simultaneously active nested validators (e.g. `dependentSchemas`/`unevaluated*`).
    ActiveValidators,
    /// Retained bytes in the `uniqueItems` canonical-value arena.
    CanonicalBytes,
    /// Number of entries in a session's undo log.
    UndoEntries,
    /// Accounted work units spent computing one mask.
    MaskWork,
    /// Retained bytes in one structured matcher traversal stack.
    MaskScratchBytes,
    /// Size in bytes of a compiled `StructuredPlan`.
    PlanBytes,
    /// Retained bytes of one session's mutable state.
    SessionBytes,
    /// Bytes requested from an allocator for a bounded runtime container.
    AllocationBytes,
    /// Number of registered schema documents.
    DocumentCount,
    /// Number of values retained by parsed schema ASTs.
    AstNodeCount,
    /// Heap bytes retained by parsed schema ASTs.
    AstBytes,
    /// Number of discovered schema resources.
    ResourceCount,
    /// Aggregate bytes of registered schema documents.
    InputBytes,
    /// Source bytes accepted by the canonical ECMA pattern lowering path.
    PatternInputBytes,
    /// Estimated syntax/HIR nodes produced by pattern parsing.
    PatternHirNodes,
    /// Thompson NFA bytes or states retained during pattern construction.
    PatternNfaBytes,
    /// Determinized DFA bytes or states retained during pattern construction.
    PatternDfaBytes,
    /// Bounded pattern parsing and automaton-construction work units.
    PatternCompileWork,
    /// URI bytes retained by the resource graph.
    UriBytes,
    /// Number of static and dynamic anchors.
    AnchorCount,
    /// Number of `$ref` and `$dynamicRef` occurrences.
    ReferenceCount,
    /// Number of indexed schema locations.
    SchemaLocationCount,
    /// Nested schema-resource depth established by `$id`.
    ResourceDepth,
    /// Number of schema resources retained in one runtime dynamic-scope chain.
    DynamicScopeDepth,
    /// Bounded resource-discovery and reference-resolution work.
    ResolutionWork,
    /// Retained bytes in the frozen schema-resource graph.
    ResourceGraphBytes,
    /// Peak bytes retained while constructing a schema-resource graph.
    ResourceBuildBytes,
}

/// A structured compile-time error. `Send + Sync`; the machine contract is the fields, not `message`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CompileError {
    /// Stable machine code.
    pub code: ErrorCode,
    /// The stage the error arose in.
    pub stage: Stage,
    /// JSON Pointer (RFC 6901) to the offending node, when known.
    pub json_pointer_path: Option<String>,
    /// The offending keyword, when applicable.
    pub keyword: Option<String>,
    /// The observed value/shape, when useful for diagnostics.
    pub observed: Option<String>,
    /// `(kind, observed, cap)` when a resource limit produced the error.
    pub limit: Option<(LimitKind, usize, usize)>,
    /// Whether a fallback path exists for this error.
    pub recoverable: bool,
    /// Advisory human-readable text. NEVER the machine contract.
    pub message: &'static str,
}

impl CompileError {
    /// Creates an error with the given code, stage, and advisory message; all optional fields empty.
    #[must_use]
    pub fn new(code: ErrorCode, stage: Stage, message: &'static str) -> Self {
        Self {
            code,
            stage,
            json_pointer_path: None,
            keyword: None,
            observed: None,
            limit: None,
            recoverable: false,
            message,
        }
    }

    /// Attaches a JSON Pointer path.
    #[must_use]
    pub fn with_pointer(mut self, path: impl Into<String>) -> Self {
        self.json_pointer_path = Some(path.into());
        self
    }

    /// Attaches the offending keyword.
    #[must_use]
    pub fn with_keyword(mut self, keyword: impl Into<String>) -> Self {
        self.keyword = Some(keyword.into());
        self
    }

    /// Attaches an observed-value description.
    #[must_use]
    pub fn with_observed(mut self, observed: impl Into<String>) -> Self {
        self.observed = Some(observed.into());
        self
    }
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}/{:?}] {}", self.code, self.stage, self.message)?;
        if let Some(path) = &self.json_pointer_path {
            write!(f, " (at {path})")?;
        }
        Ok(())
    }
}

impl std::error::Error for CompileError {}

/// A structured vocabulary-boundary error.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VocabError {
    /// Stable machine code (`EmptyToken` or `MalformedTokenizer`).
    pub code: ErrorCode,
    /// The offending token id, when one applies.
    pub token: Option<TokenId>,
    /// Advisory detail. Not the machine contract.
    pub detail: &'static str,
}

impl VocabError {
    /// An empty-byte token was rejected at the choke-point.
    #[must_use]
    pub fn empty_token() -> Self {
        Self {
            code: ErrorCode::EmptyToken,
            token: None,
            detail: "vocabulary token has zero bytes",
        }
    }

    /// Tokenizer metadata (not a byte payload) was malformed.
    #[must_use]
    pub fn malformed_tokenizer(detail: &'static str) -> Self {
        Self {
            code: ErrorCode::MalformedTokenizer,
            token: None,
            detail,
        }
    }

    /// Tokenizer metadata was malformed because of a specific, known token id.
    #[must_use]
    pub fn malformed_tokenizer_for(token: TokenId, detail: &'static str) -> Self {
        Self {
            code: ErrorCode::MalformedTokenizer,
            token: Some(token),
            detail,
        }
    }

    /// A vocabulary resource cap was exceeded; the input is well-formed but too large.
    #[must_use]
    pub fn limit_exceeded(detail: &'static str) -> Self {
        Self {
            code: ErrorCode::InternalLimitExceeded,
            token: None,
            detail,
        }
    }
}

impl std::fmt::Display for VocabError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}/VocabBuild] {}", self.code, self.detail)
    }
}

impl std::error::Error for VocabError {}

/// A runtime matcher error.
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum MatcherError {
    /// A token was fed that the matcher cannot legally consume (dead/no transition).
    IllegalToken {
        /// The rejected token id.
        token: TokenId,
    },
    /// A cached artifact did not match its expected fingerprint.
    ArtifactMismatch,
}

impl std::fmt::Display for MatcherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IllegalToken { token } => write!(f, "[IllegalToken] token {}", token.get()),
            Self::ArtifactMismatch => write!(f, "[ArtifactMismatch]"),
        }
    }
}

impl std::error::Error for MatcherError {}

/// Fingerprint state of a provenance field.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum HashState {
    /// No hash was computed; legal for the `Reference` profile.
    Unhashed,
    /// A 32-byte fingerprint.
    Hash([u8; 32]),
}

impl HashState {
    /// A present, non-zero fingerprint. An all-zero hash is never a valid fingerprint.
    #[must_use]
    pub fn is_valid_fingerprint(&self) -> bool {
        matches!(self, Self::Hash(h) if *h != [0u8; 32])
    }
}

/// Which provenance profile an artifact was produced under. Validation is BY PROFILE.
#[non_exhaustive]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ProvenanceProfile {
    /// A slow, unoptimized reference build; legal with `ir_hash = Unhashed`; never cache-valid.
    Reference,
    /// An artifact built from a hashed intermediate representation.
    Ir,
    /// A validated, cache-eligible artifact.
    CachedArtifact,
}

/// An artifact's provenance record, validated BY PROFILE.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Provenance {
    /// Which profile produced the artifact.
    pub profile: ProvenanceProfile,
    /// Hash of the source input.
    pub source_input_hash: HashState,
    /// Hash of the lowered IR (legal to be `Unhashed` for the `Reference` profile).
    pub ir_hash: HashState,
    /// Whether the artifact may be served from cache.
    pub cache_valid: bool,
}

impl Provenance {
    /// A fresh reference-build provenance: `ir_hash` unhashed, never cache-valid.
    #[must_use]
    pub fn reference(source_input_hash: HashState) -> Self {
        Self {
            profile: ProvenanceProfile::Reference,
            source_input_hash,
            ir_hash: HashState::Unhashed,
            cache_valid: false,
        }
    }

    /// Validates the record against its own profile. Every profile requires a real
    /// `source_input_hash`. `Reference` is legal with an unhashed IR; `Ir` and `CachedArtifact`
    /// require a real IR hash too. Only `CachedArtifact` may be `cache_valid`, and it must be.
    pub fn validate(&self) -> Result<(), CompileError> {
        let err = |msg| {
            Err(CompileError::new(
                ErrorCode::ProvenanceIncomplete,
                Stage::ArtifactValidate,
                msg,
            ))
        };
        if matches!(self.source_input_hash, HashState::Hash(h) if h == [0u8; 32])
            || matches!(self.ir_hash, HashState::Hash(h) if h == [0u8; 32])
        {
            return err("an all-zero hash is never a valid fingerprint");
        }
        if !self.source_input_hash.is_valid_fingerprint() {
            return err("every provenance record requires a source_input_hash");
        }
        match self.profile {
            ProvenanceProfile::Reference => {
                if self.cache_valid {
                    return err("a reference-build artifact is never cache-valid");
                }
                Ok(())
            }
            ProvenanceProfile::Ir => {
                if !self.ir_hash.is_valid_fingerprint() {
                    return err("an IR-profile artifact requires an ir_hash");
                }
                if self.cache_valid {
                    return err("an IR-profile artifact is never cache-valid");
                }
                Ok(())
            }
            ProvenanceProfile::CachedArtifact => {
                if !self.ir_hash.is_valid_fingerprint() {
                    return err("a cached artifact requires an ir_hash");
                }
                if !self.cache_valid {
                    return err("a cached artifact must be cache-valid");
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn error_types_are_send_sync() {
        assert_send_sync::<CompileError>();
        assert_send_sync::<VocabError>();
        assert_send_sync::<MatcherError>();
        assert_send_sync::<ErrorCode>();
    }

    #[test]
    fn machine_fields_survive_without_message() {
        let mut e = CompileError::new(ErrorCode::InternalLimitExceeded, Stage::L2, "cap")
            .with_pointer("/properties/name")
            .with_keyword("maxItems")
            .with_observed("5");
        e.limit = Some((LimitKind::ArrayLength, 5, 4));
        assert_eq!(e.code, ErrorCode::InternalLimitExceeded);
        assert_eq!(e.stage, Stage::L2);
        assert_eq!(e.json_pointer_path.as_deref(), Some("/properties/name"));
        assert_eq!(e.keyword.as_deref(), Some("maxItems"));
        assert_eq!(e.observed.as_deref(), Some("5"));
        assert_eq!(e.limit, Some((LimitKind::ArrayLength, 5, 4)));
        assert_eq!(VocabError::empty_token().code, ErrorCode::EmptyToken);
        match (MatcherError::IllegalToken { token: TokenId(9) }) {
            MatcherError::IllegalToken { token } => assert_eq!(token, TokenId(9)),
            MatcherError::ArtifactMismatch => unreachable!(),
        }
    }

    #[test]
    fn provenance_validated_by_profile() {
        assert!(Provenance::reference(HashState::Hash([2; 32]))
            .validate()
            .is_ok());
        assert_eq!(
            Provenance::reference(HashState::Unhashed)
                .validate()
                .unwrap_err()
                .code,
            ErrorCode::ProvenanceIncomplete
        );
        let mut cache_bad = Provenance::reference(HashState::Hash([2; 32]));
        cache_bad.cache_valid = true;
        assert_eq!(
            cache_bad.validate().unwrap_err().code,
            ErrorCode::ProvenanceIncomplete
        );
        let ir_profile = Provenance {
            profile: ProvenanceProfile::Ir,
            source_input_hash: HashState::Hash([2; 32]),
            ir_hash: HashState::Unhashed,
            cache_valid: false,
        };
        assert_eq!(
            ir_profile.validate().unwrap_err().code,
            ErrorCode::ProvenanceIncomplete
        );
        let cached = Provenance {
            profile: ProvenanceProfile::CachedArtifact,
            source_input_hash: HashState::Hash([2; 32]),
            ir_hash: HashState::Hash([1; 32]),
            cache_valid: true,
        };
        assert!(cached.validate().is_ok());
    }

    #[test]
    fn only_cached_artifact_may_be_cache_valid() {
        let ir_cache_valid = Provenance {
            profile: ProvenanceProfile::Ir,
            source_input_hash: HashState::Hash([2; 32]),
            ir_hash: HashState::Hash([3; 32]),
            cache_valid: true,
        };
        assert_eq!(
            ir_cache_valid.validate().unwrap_err().code,
            ErrorCode::ProvenanceIncomplete
        );
        let cached_not_valid = Provenance {
            profile: ProvenanceProfile::CachedArtifact,
            source_input_hash: HashState::Hash([2; 32]),
            ir_hash: HashState::Hash([3; 32]),
            cache_valid: false,
        };
        assert_eq!(
            cached_not_valid.validate().unwrap_err().code,
            ErrorCode::ProvenanceIncomplete
        );
    }

    #[test]
    fn every_profile_requires_a_source_input_hash() {
        for profile in [
            ProvenanceProfile::Reference,
            ProvenanceProfile::Ir,
            ProvenanceProfile::CachedArtifact,
        ] {
            let p = Provenance {
                profile,
                source_input_hash: HashState::Unhashed,
                ir_hash: HashState::Hash([1; 32]),
                cache_valid: matches!(profile, ProvenanceProfile::CachedArtifact),
            };
            assert_eq!(
                p.validate().unwrap_err().code,
                ErrorCode::ProvenanceIncomplete
            );
        }
    }

    #[test]
    fn zero_hash_is_never_a_valid_fingerprint() {
        assert!(!HashState::Hash([0; 32]).is_valid_fingerprint());
        assert!(!HashState::Unhashed.is_valid_fingerprint());
        assert!(HashState::Hash([1; 32]).is_valid_fingerprint());
        let zero_source = Provenance {
            profile: ProvenanceProfile::CachedArtifact,
            source_input_hash: HashState::Hash([0; 32]),
            ir_hash: HashState::Hash([1; 32]),
            cache_valid: true,
        };
        assert_eq!(
            zero_source.validate().unwrap_err().code,
            ErrorCode::ProvenanceIncomplete
        );
    }
}
