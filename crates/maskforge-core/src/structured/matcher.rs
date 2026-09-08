use std::fmt;
use std::sync::Arc;

use crate::error::{CompileError, ErrorCode, LimitKind};
use crate::index::{
    BoundByteTrie, BoundedMask, TokenSliceCatalog, VocabFingerprint, VocabTrie, VocabularyHandle,
};
use crate::ir::SchemaIR;
#[cfg(feature = "bench-internals")]
use crate::ir::{ContainsPolicy, Node};
use crate::mask::Bitmask;
use crate::primitives::{NodeId, StateId, TokenId, TrieNodeId};

use super::lexer::{DecodeStep, JsonStringDecoder};
use super::limits::{StructuredLimits, StructuredRuntimeError};
use super::plan::{PlanCompileFailure, StructuredPlan};
#[cfg(any(test, feature = "bench-internals"))]
use super::reference::ReferenceMatcher;
use super::state::{
    Checkpoint, ProofContext, SliceCertificate, SlicePreparation, SliceProof, StructuredState,
};

/// A compiled immutable structured program shared by sequence matchers.
pub struct StructuredProgram {
    #[cfg(feature = "bench-internals")]
    ir: Arc<SchemaIR>,
    backend: ProgramBackend,
    limits: StructuredLimits,
}

enum ProgramBackend {
    Incremental(Arc<StructuredPlan>),
    #[cfg(feature = "bench-internals")]
    ReferenceBench,
}

/// A mutable matcher with independent sequence state.
pub struct StructuredMatcher {
    program: Arc<StructuredProgram>,
    backend: MatcherBackend,
}

enum MatcherBackend {
    Incremental(IncrementalMatcher),
    #[cfg(feature = "bench-internals")]
    Reference(ReferenceMatcher),
}

struct IncrementalMatcher {
    state: Box<StructuredState<'static>>,
    events: Vec<TrieEvent>,
    /// Reused by the read-only lexical prefix walk before its first semantic boundary.
    lexical_path: Vec<u8>,
    slice_proofs: SliceProofCache,
    /// Reused across masks so a wrapper certificate allocates nothing on the warm path.
    certificate_proofs: Vec<SliceProof>,
    #[cfg(feature = "bench-internals")]
    mask_work_metrics: MaskWorkMetrics,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct SliceProofKey {
    vocabulary: VocabFingerprint,
    node: NodeId,
    state: StateId,
    body_prefix: bool,
    reject_invalid: bool,
}

#[derive(Clone, Copy)]
struct SliceProofEntry {
    key: SliceProofKey,
    subsumes: bool,
}

struct CandidateMaskEntry {
    key: SliceProofKey,
    mask: Arc<[u64]>,
}

#[derive(Default)]
struct SliceProofCache {
    entries: Vec<SliceProofEntry>,
    candidate: Option<CandidateMaskEntry>,
    next: usize,
    #[cfg(feature = "bench-internals")]
    hits: u64,
    #[cfg(feature = "bench-internals")]
    misses: u64,
}

impl SliceProofCache {
    fn get(&mut self, key: SliceProofKey) -> Option<bool> {
        let result = self
            .entries
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| entry.subsumes);
        #[cfg(feature = "bench-internals")]
        match result {
            Some(_) => self.hits = self.hits.saturating_add(1),
            None => self.misses = self.misses.saturating_add(1),
        }
        result
    }

    #[cfg(feature = "bench-internals")]
    fn reset_measurements(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }

    #[cfg(feature = "bench-internals")]
    fn measurements(&self) -> SliceProofMetrics {
        SliceProofMetrics {
            hits: self.hits,
            misses: self.misses,
            entries: self.entries.len() + usize::from(self.candidate.is_some()),
        }
    }

    fn insert(&mut self, key: SliceProofKey, subsumes: bool, byte_budget: usize) {
        let slots = byte_budget / std::mem::size_of::<SliceProofEntry>();
        if slots == 0 {
            return;
        }
        if self.entries.capacity() == 0
            && (self.entries.try_reserve_exact(slots).is_err()
                || self
                    .entries
                    .capacity()
                    .checked_mul(std::mem::size_of::<SliceProofEntry>())
                    .is_none_or(|bytes| bytes > byte_budget))
        {
            self.entries = Vec::new();
            return;
        }
        if self.entries.len() < self.entries.capacity() {
            self.entries.push(SliceProofEntry { key, subsumes });
            return;
        }
        self.entries[self.next] = SliceProofEntry { key, subsumes };
        self.next = (self.next + 1) % self.entries.len();
    }

    fn candidate(&mut self, key: SliceProofKey) -> Option<Arc<[u64]>> {
        let result = self
            .candidate
            .as_ref()
            .filter(|entry| entry.key == key)
            .map(|entry| entry.mask.clone());
        #[cfg(feature = "bench-internals")]
        match result {
            Some(_) => self.hits = self.hits.saturating_add(1),
            None => self.misses = self.misses.saturating_add(1),
        }
        result
    }

    fn insert_candidate(&mut self, key: SliceProofKey, mask: Arc<[u64]>, byte_budget: usize) {
        let Some(bytes) = mask.len().checked_mul(std::mem::size_of::<u64>()) else {
            return;
        };
        let Some(proof_bytes) = self
            .entries
            .capacity()
            .checked_mul(std::mem::size_of::<SliceProofEntry>())
        else {
            return;
        };
        if proof_bytes
            .checked_add(bytes)
            .is_none_or(|total| total > byte_budget)
        {
            return;
        }
        self.candidate = Some(CandidateMaskEntry { key, mask });
    }
}

/// Exact work performed by the most recent trie-mask operation.
#[cfg(feature = "bench-internals")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MaskWorkMetrics {
    /// Decoder/DFA edge transitions performed without mutating transactional matcher state.
    pub local_dfa_transitions: u64,
    /// Trie nodes whose visit event was processed.
    pub trie_nodes: u64,
    /// Trie edges whose byte was passed to the matcher.
    pub trie_edges: u64,
    /// Bytes passed to the matcher while walking trie edges.
    pub pushed_bytes: u64,
    /// State checkpoints created by the trie walk.
    pub checkpoints: u64,
    /// State rollbacks performed by the trie walk.
    pub rollbacks: u64,
    /// Dynamic references resolved while evaluating the mask.
    pub dynamic_resolutions: u64,
    /// Dynamic-scope slots inspected by those resolutions.
    pub dynamic_scope_steps: u64,
}

/// Per-mask proof-cache activity exposed only to benchmark builds.
#[cfg(feature = "bench-internals")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SliceProofMetrics {
    /// Cached proof lookups that found an equal entry.
    pub hits: u64,
    /// Cached proof lookups that required a new proof attempt.
    pub misses: u64,
    /// Entries retained after the current mask operation.
    pub entries: usize,
}

#[cfg(feature = "bench-internals")]
#[derive(Clone, Copy)]
enum MaskMetric {
    LocalDfaTransition,
    TrieNode,
    TrieEdge,
    PushedByte,
    Checkpoint,
    Rollback,
}

#[cfg(feature = "bench-internals")]
impl IncrementalMatcher {
    fn reset_mask_metrics(&mut self) {
        self.mask_work_metrics = MaskWorkMetrics::default();
        self.state.reset_dynamic_work_metrics();
        self.slice_proofs.reset_measurements();
    }

    fn increment_mask_metric(&mut self, metric: MaskMetric) {
        let counter = match metric {
            MaskMetric::LocalDfaTransition => &mut self.mask_work_metrics.local_dfa_transitions,
            MaskMetric::TrieNode => &mut self.mask_work_metrics.trie_nodes,
            MaskMetric::TrieEdge => &mut self.mask_work_metrics.trie_edges,
            MaskMetric::PushedByte => &mut self.mask_work_metrics.pushed_bytes,
            MaskMetric::Checkpoint => &mut self.mask_work_metrics.checkpoints,
            MaskMetric::Rollback => &mut self.mask_work_metrics.rollbacks,
        };
        *counter = counter.saturating_add(1);
    }

    fn finish_mask_metrics(&mut self) {
        let (resolutions, steps) = self.state.dynamic_work_metrics();
        self.mask_work_metrics.dynamic_resolutions = resolutions;
        self.mask_work_metrics.dynamic_scope_steps = steps;
    }
}

enum TrieEvent {
    Visit {
        node: TrieNodeId,
        incoming_byte: Option<u8>,
    },
    Restore(Checkpoint),
    LocalVisit {
        node: TrieNodeId,
        incoming_byte: Option<u8>,
        decoder: JsonStringDecoder,
        path_len: usize,
    },
    RestoreLocalPath(usize),
}

/// Errors from compilation, state advancement, and mask generation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum StructuredMatcherError {
    /// A schema or plan compilation error.
    Compile(CompileError),
    /// A bounded runtime resource was exhausted.
    ResourceLimit {
        /// Resource category that reached its cap.
        kind: LimitKind,
        /// Amount observed before rejection.
        observed: usize,
        /// Configured resource cap.
        limit: usize,
    },
}

impl fmt::Display for StructuredMatcherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(error) => error.fmt(f),
            Self::ResourceLimit {
                kind,
                observed,
                limit,
            } => write!(f, "structured resource limit {kind:?}: {observed}/{limit}"),
        }
    }
}

impl std::error::Error for StructuredMatcherError {}

impl From<CompileError> for StructuredMatcherError {
    fn from(error: CompileError) -> Self {
        Self::Compile(error)
    }
}

impl From<StructuredRuntimeError> for StructuredMatcherError {
    fn from(error: StructuredRuntimeError) -> Self {
        Self::ResourceLimit {
            kind: error.kind,
            observed: error.observed,
            limit: error.limit,
        }
    }
}

impl StructuredProgram {
    /// Compiles a schema using the default structured resource limits.
    pub fn compile(ir: Arc<SchemaIR>) -> Result<Arc<Self>, CompileError> {
        Self::compile_with_limits(ir, StructuredLimits::default())
    }

    pub(crate) fn compile_with_limits(
        ir: Arc<SchemaIR>,
        limits: StructuredLimits,
    ) -> Result<Arc<Self>, CompileError> {
        let backend = StructuredPlan::compile(ir.clone(), limits)
            .map(|plan| ProgramBackend::Incremental(Arc::new(plan)))
            .map_err(|PlanCompileFailure::Fatal(error)| error)?;
        Ok(Arc::new(Self {
            #[cfg(feature = "bench-internals")]
            ir,
            backend,
            limits,
        }))
    }

    /// Constructs the reference engine over the same IR for benchmark comparison.
    #[doc(hidden)]
    #[cfg(feature = "bench-internals")]
    pub fn reference_for_bench(ir: Arc<SchemaIR>) -> Arc<Self> {
        Arc::new(Self {
            ir,
            backend: ProgramBackend::ReferenceBench,
            limits: StructuredLimits::default(),
        })
    }

    /// Compiles a fresh structured plan and returns its isolated elapsed nanoseconds.
    #[doc(hidden)]
    #[cfg(feature = "bench-internals")]
    pub fn compile_timed_for_bench(ir: Arc<SchemaIR>) -> Result<(Arc<Self>, u64), CompileError> {
        let limits = StructuredLimits::default();
        let started = std::time::Instant::now();
        let plan = match StructuredPlan::compile(ir.clone(), limits) {
            Ok(plan) => plan,
            Err(PlanCompileFailure::Fatal(error)) => return Err(error),
        };
        let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let program = Arc::new(Self {
            #[cfg(feature = "bench-internals")]
            ir,
            backend: ProgramBackend::Incremental(Arc::new(plan)),
            limits,
        });
        Ok((program, elapsed))
    }

    /// Returns total plan-build and same-instance SCC-analysis nanoseconds separately.
    #[doc(hidden)]
    #[cfg(feature = "bench-internals")]
    pub fn compile_profiled_for_bench(
        ir: Arc<SchemaIR>,
    ) -> Result<(Arc<Self>, u64, u64), CompileError> {
        let limits = StructuredLimits::default();
        let (plan, total_ns, scc_ns) = match StructuredPlan::compile_profiled(ir.clone(), limits) {
            Ok(profile) => profile,
            Err(PlanCompileFailure::Fatal(error)) => return Err(error),
        };
        let program = Arc::new(Self {
            ir,
            backend: ProgramBackend::Incremental(Arc::new(plan)),
            limits,
        });
        Ok((program, total_ns, scc_ns))
    }

    /// Returns sorted reachable IR node-kind names for routing census builds.
    #[doc(hidden)]
    #[cfg(feature = "bench-internals")]
    pub fn reachable_node_kinds_for_bench(
        ir: &SchemaIR,
    ) -> Result<Vec<&'static str>, CompileError> {
        let mut seen = Vec::new();
        seen.try_reserve_exact(ir.node_count()).map_err(|_| {
            CompileError::new(
                ErrorCode::InternalLimitExceeded,
                crate::error::Stage::L3,
                "routing census allocation",
            )
        })?;
        seen.resize(ir.node_count(), false);
        let mut stack = Vec::new();
        stack.try_reserve_exact(ir.node_count()).map_err(|_| {
            CompileError::new(
                ErrorCode::InternalLimitExceeded,
                crate::error::Stage::L3,
                "routing census allocation",
            )
        })?;
        stack.push(ir.root());
        let mut kinds = Vec::new();
        kinds.try_reserve_exact(ir.node_count()).map_err(|_| {
            CompileError::new(
                ErrorCode::InternalLimitExceeded,
                crate::error::Stage::L3,
                "routing census allocation",
            )
        })?;
        while let Some(id) = stack.pop() {
            let index = usize::try_from(id.get()).map_err(|_| {
                CompileError::new(ErrorCode::Malformed, crate::error::Stage::L3, "node id")
            })?;
            let Some(slot) = seen.get_mut(index) else {
                return Err(CompileError::new(
                    ErrorCode::Malformed,
                    crate::error::Stage::L3,
                    "node id",
                ));
            };
            if *slot {
                continue;
            }
            *slot = true;
            let node = ir.node(id).ok_or_else(|| {
                CompileError::new(ErrorCode::Malformed, crate::error::Stage::L3, "node id")
            })?;
            kinds.push(node_kind_name(node));
            stack.extend(crate::compile::node_children(ir, node)?);
            match node {
                Node::Object { dependent, .. } | Node::OpenObject { dependent, .. } => {
                    stack.extend(
                        ir.props_at(*dependent)
                            .ok_or_else(|| {
                                CompileError::new(
                                    ErrorCode::Malformed,
                                    crate::error::Stage::L3,
                                    "dependent schema slice",
                                )
                            })?
                            .iter()
                            .map(|(_, child)| *child),
                    );
                }
                Node::Array {
                    contains:
                        Some(crate::ir::ContainsConstraint {
                            policy: ContainsPolicy::Schema(child),
                            ..
                        }),
                    ..
                }
                | Node::Tuple {
                    contains:
                        Some(crate::ir::ContainsConstraint {
                            policy: ContainsPolicy::Schema(child),
                            ..
                        }),
                    ..
                } => stack.push(*child),
                Node::Ref { def } => stack.push(ir.def_target(*def).ok_or_else(|| {
                    CompileError::new(
                        ErrorCode::Malformed,
                        crate::error::Stage::L3,
                        "reference definition slot",
                    )
                })?),
                _ => {}
            }
        }
        kinds.sort_unstable();
        kinds.dedup();
        Ok(kinds)
    }

    /// Constructs a fresh mutable sequence matcher sharing this program.
    pub fn new_matcher(self: &Arc<Self>) -> Result<StructuredMatcher, StructuredMatcherError> {
        let backend = match &self.backend {
            ProgramBackend::Incremental(plan) => MatcherBackend::Incremental(IncrementalMatcher {
                state: Box::new(StructuredState::try_new(plan.clone())?),
                events: Vec::new(),
                lexical_path: Vec::new(),
                slice_proofs: SliceProofCache::default(),
                certificate_proofs: Vec::new(),
                #[cfg(feature = "bench-internals")]
                mask_work_metrics: MaskWorkMetrics::default(),
            }),
            #[cfg(feature = "bench-internals")]
            ProgramBackend::ReferenceBench => MatcherBackend::Reference(
                ReferenceMatcher::new_with_limits(self.ir.clone(), self.limits),
            ),
        };
        Ok(StructuredMatcher {
            program: self.clone(),
            backend,
        })
    }

    #[cfg(test)]
    pub(crate) fn backend_kind(&self) -> &'static str {
        match &self.backend {
            ProgramBackend::Incremental(_) => "incremental",
            #[cfg(feature = "bench-internals")]
            ProgramBackend::ReferenceBench => "reference",
        }
    }

    /// Whether this program uses the incremental backend.
    #[must_use]
    pub fn uses_incremental_backend(&self) -> bool {
        matches!(self.backend, ProgramBackend::Incremental(_))
    }

    /// Benchmark diagnostic for the isolated oracle route.
    #[cfg(any(test, feature = "bench-internals"))]
    #[must_use]
    pub fn fallback_reason(&self) -> Option<&'static str> {
        None
    }

    /// Bytes retained by the compiled incremental plan; reference programs have no such plan.
    #[must_use]
    pub fn retained_plan_bytes(&self) -> Option<usize> {
        match &self.backend {
            ProgramBackend::Incremental(plan) => Some(plan.retained_bytes),
            #[cfg(feature = "bench-internals")]
            ProgramBackend::ReferenceBench => None,
        }
    }
}

/// Compiles and evaluates one complete document with the production incremental engine.
pub fn try_accepts(ir: Arc<SchemaIR>, bytes: &[u8]) -> Result<bool, StructuredMatcherError> {
    let program = StructuredProgram::compile(ir)?;
    let mut matcher = program.new_matcher()?;
    if !bytes.is_empty() && !matcher.advance(bytes)? {
        return Ok(false);
    }
    Ok(matcher.is_accepting() && matcher.eos_legal())
}

/// Compiles and evaluates whether a prefix can still reach a valid document.
pub fn try_can_continue(ir: Arc<SchemaIR>, bytes: &[u8]) -> Result<bool, StructuredMatcherError> {
    let program = StructuredProgram::compile(ir)?;
    let mut matcher = program.new_matcher()?;
    if !bytes.is_empty() && !matcher.advance(bytes)? {
        return Ok(false);
    }
    Ok(!matcher.is_dead())
}

#[cfg(feature = "bench-internals")]
fn node_kind_name(node: &Node) -> &'static str {
    match node {
        Node::Null => "Null",
        Node::Boolean => "Boolean",
        Node::Never => "Never",
        Node::StringConst { .. } => "StringConst",
        Node::StringPattern { .. } => "StringPattern",
        Node::Integer { .. } => "Integer",
        Node::Number { .. } | Node::LexicalNumber { .. } => "Number",
        Node::Enum { .. } => "Enum",
        Node::Array { .. } => "Array",
        Node::Tuple { .. } => "Tuple",
        Node::Object { .. } => "Object",
        Node::OpenObject { .. } => "OpenObject",
        Node::Union { .. } => "Union",
        Node::Intersection { .. } => "Intersection",
        Node::ExactlyOne { .. } => "ExactlyOne",
        Node::Not { .. } => "Not",
        Node::Ref { .. } => "Ref",
        Node::DynamicRef { .. } => "DynamicRef",
        Node::Unevaluated { .. } => "Unevaluated",
        Node::Unsupported { .. } => "Unsupported",
    }
}

impl StructuredMatcher {
    /// Returns work counters for the most recent trie-mask operation.
    #[cfg(feature = "bench-internals")]
    #[must_use]
    pub fn mask_work_metrics_for_bench(&self) -> MaskWorkMetrics {
        match &self.backend {
            MatcherBackend::Incremental(incremental) => incremental.mask_work_metrics,
            MatcherBackend::Reference(_) => MaskWorkMetrics::default(),
        }
    }

    /// Returns proof-cache activity for the most recent trie-mask operation.
    #[cfg(feature = "bench-internals")]
    #[must_use]
    pub fn slice_proof_metrics_for_bench(&self) -> SliceProofMetrics {
        match &self.backend {
            MatcherBackend::Incremental(incremental) => incremental.slice_proofs.measurements(),
            MatcherBackend::Reference(_) => SliceProofMetrics::default(),
        }
    }

    /// Bytes retained by the local slice-proof cache.
    #[cfg(feature = "bench-internals")]
    #[must_use]
    pub fn slice_proof_cache_bytes_for_bench(&self) -> usize {
        match &self.backend {
            MatcherBackend::Incremental(incremental) => incremental
                .slice_proofs
                .entries
                .capacity()
                .saturating_mul(std::mem::size_of::<SliceProofEntry>()),
            MatcherBackend::Reference(_) => 0,
        }
    }

    /// Constructs a matcher from an already compiled program.
    pub fn new(program: Arc<StructuredProgram>) -> Result<Self, StructuredMatcherError> {
        program.new_matcher()
    }

    /// Runtime backend selected for this matcher.
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        match &self.backend {
            MatcherBackend::Incremental(_) => "incremental",
            #[cfg(feature = "bench-internals")]
            MatcherBackend::Reference(_) => "reference",
        }
    }

    /// Whether the current state accepts a complete document.
    #[must_use]
    pub fn is_accepting(&self) -> bool {
        match &self.backend {
            MatcherBackend::Incremental(incremental) => incremental.state.is_accepting(),
            #[cfg(feature = "bench-internals")]
            MatcherBackend::Reference(reference) => reference.is_accepting(),
        }
    }

    /// Whether the current state is a permanent dead end.
    #[must_use]
    pub fn is_dead(&self) -> bool {
        match &self.backend {
            MatcherBackend::Incremental(incremental) => incremental.state.is_dead(),
            #[cfg(feature = "bench-internals")]
            MatcherBackend::Reference(reference) => reference.is_dead(),
        }
    }

    /// Whether end-of-stream is legal at the current state.
    #[must_use]
    pub fn eos_legal(&self) -> bool {
        match &self.backend {
            MatcherBackend::Incremental(incremental) => incremental.state.is_accepting(),
            #[cfg(feature = "bench-internals")]
            MatcherBackend::Reference(reference) => reference.eos_legal(),
        }
    }

    /// Bytes currently retained by this sequence's mutable backend state.
    #[must_use]
    pub fn retained_session_bytes(&self) -> usize {
        match &self.backend {
            MatcherBackend::Incremental(incremental) => incremental.state.retained_bytes(),
            #[cfg(feature = "bench-internals")]
            MatcherBackend::Reference(reference) => reference.retained_bytes(),
        }
    }

    /// Peak charged session bytes observed by this matcher.
    #[must_use]
    pub fn peak_session_bytes(&self) -> usize {
        match &self.backend {
            MatcherBackend::Incremental(incremental) => incremental.state.peak_session_bytes(),
            #[cfg(feature = "bench-internals")]
            MatcherBackend::Reference(reference) => reference.retained_bytes(),
        }
    }

    /// Number of validator cursors currently owned by this matcher tree.
    #[cfg(feature = "bench-internals")]
    #[must_use]
    pub fn active_validator_count_for_bench(&self) -> usize {
        match &self.backend {
            MatcherBackend::Incremental(incremental) => {
                incremental.state.active_validator_count_for_bench()
            }
            MatcherBackend::Reference(_) => 0,
        }
    }

    /// Returns `(frame depth, wrapper-frame depth)` for benchmark diagnostics.
    #[cfg(feature = "bench-internals")]
    #[must_use]
    pub fn frame_profile_for_bench(&self) -> (usize, usize) {
        match &self.backend {
            MatcherBackend::Incremental(incremental) => incremental.state.frame_profile_for_bench(),
            MatcherBackend::Reference(_) => (0, 0),
        }
    }

    /// Returns `(total, undo)` mutable retention for benchmark-only diagnostics.
    #[cfg(feature = "bench-internals")]
    #[must_use]
    pub fn retention_breakdown_for_bench(&self) -> (usize, usize) {
        match &self.backend {
            MatcherBackend::Incremental(incremental) => {
                incremental.state.retention_breakdown_for_bench()
            }
            MatcherBackend::Reference(reference) => (reference.retained_bytes(), 0),
        }
    }

    /// Atomically commits a whole tokenizer token.
    pub fn advance(&mut self, token_bytes: &[u8]) -> Result<bool, StructuredMatcherError> {
        if token_bytes.is_empty() {
            return Ok(false);
        }
        match &mut self.backend {
            MatcherBackend::Incremental(incremental) => {
                let state = &mut incremental.state;
                let mark = state.checkpoint();
                for &byte in token_bytes {
                    match state.try_push_byte(byte) {
                        Ok(true) => {}
                        Ok(false) => {
                            state.rollback(mark);
                            return Ok(false);
                        }
                        Err(error) => {
                            state.rollback(mark);
                            return Err(error.into());
                        }
                    }
                }
                if let Err(error) = state.commit_checkpoint() {
                    state.rollback(mark);
                    return Err(error.into());
                }
                Ok(true)
            }
            #[cfg(feature = "bench-internals")]
            MatcherBackend::Reference(reference) => reference
                .try_advance(token_bytes)
                .map_err(StructuredMatcherError::from),
        }
    }

    /// Computes a mask by speculatively feeding each candidate through one reusable state.
    pub fn allowed_mask_from_records<'a>(
        &mut self,
        mask_vocab_size: usize,
        records: impl Iterator<Item = (&'a [u8], &'a [u32])>,
    ) -> Result<Bitmask, StructuredMatcherError> {
        match &mut self.backend {
            MatcherBackend::Incremental(incremental) => {
                let state = &mut incremental.state;
                let mut mask = Bitmask::zeros(mask_vocab_size);
                let mut work = 0u64;
                for (bytes, ids) in records {
                    charge_mask_work(&mut work, bytes.len(), ids.len(), self.program.limits)?;
                    if candidate_is_allowed(state, bytes)? {
                        for &id in ids {
                            mask.set(TokenId(id))
                                .map_err(StructuredMatcherError::from)?;
                        }
                    }
                }
                Ok(mask)
            }
            #[cfg(feature = "bench-internals")]
            MatcherBackend::Reference(reference) => reference
                .allowed_mask_from_records(mask_vocab_size, records)
                .map_err(StructuredMatcherError::from),
        }
    }

    /// Clears and fills a packed `u32`-word mask by scanning vocabulary records, without an
    /// intermediate byte buffer. `O(1)` additional allocation beyond `out`.
    pub fn write_record_mask_words_into<'a>(
        &mut self,
        mask_vocab_size: usize,
        records: impl Iterator<Item = (&'a [u8], &'a [u32])>,
        out: &mut [u32],
    ) -> Result<(), StructuredMatcherError> {
        validate_mask_output_words(mask_vocab_size, out)?;
        out.fill(0);
        let result = match &mut self.backend {
            MatcherBackend::Incremental(incremental) => (|| {
                let mut work = 0u64;
                for (bytes, ids) in records {
                    charge_mask_work(&mut work, bytes.len(), ids.len(), self.program.limits)?;
                    if candidate_is_allowed(&mut incremental.state, bytes)? {
                        for &id in ids {
                            set_token_bit_word(out, mask_vocab_size, id)?;
                        }
                    }
                }
                Ok(())
            })(),
            #[cfg(feature = "bench-internals")]
            MatcherBackend::Reference(reference) => reference
                .allowed_mask_from_records(mask_vocab_size, records)
                .map(|mask| out.copy_from_slice(mask.as_words()))
                .map_err(StructuredMatcherError::from),
        };
        if result.is_err() {
            out.fill(0);
        }
        result
    }

    /// Clears and fills a little-endian packed mask by scanning vocabulary records.
    pub fn write_record_mask_le_bytes_into<'a>(
        &mut self,
        mask_vocab_size: usize,
        records: impl Iterator<Item = (&'a [u8], &'a [u32])>,
        out: &mut [u8],
    ) -> Result<(), StructuredMatcherError> {
        validate_mask_output(mask_vocab_size, out)?;
        out.fill(0);
        let result = match &mut self.backend {
            MatcherBackend::Incremental(incremental) => (|| {
                let mut work = 0u64;
                for (bytes, ids) in records {
                    charge_mask_work(&mut work, bytes.len(), ids.len(), self.program.limits)?;
                    if candidate_is_allowed(&mut incremental.state, bytes)? {
                        for &id in ids {
                            set_token_bit_le(out, mask_vocab_size, id)?;
                        }
                    }
                }
                Ok(())
            })(),
            #[cfg(feature = "bench-internals")]
            MatcherBackend::Reference(reference) => reference
                .write_mask_le_bytes_into(mask_vocab_size, records, out)
                .map_err(StructuredMatcherError::from),
        };
        if result.is_err() {
            out.fill(0);
        }
        result
    }

    /// Clears and fills a little-endian packed mask without allocating a temporary `Bitmask`.
    pub fn write_mask_le_bytes_into(
        &mut self,
        vocabulary: &VocabularyHandle,
        trie: Option<&BoundByteTrie>,
        out: &mut [u8],
    ) -> Result<(), StructuredMatcherError> {
        let width = vocabulary.mask_vocab_size();
        validate_mask_output(width, out)?;
        out.fill(0);
        let result = match &mut self.backend {
            MatcherBackend::Incremental(incremental) => {
                let trie = trie.ok_or_else(missing_trie_error)?;
                walk_trie_mask(incremental, vocabulary, trie, out, self.program.limits)
            }
            #[cfg(feature = "bench-internals")]
            MatcherBackend::Reference(reference) => reference
                .write_mask_le_bytes_into(width, vocabulary.iter_records(), out)
                .map_err(StructuredMatcherError::from),
        };
        if result.is_err() {
            out.fill(0);
        }
        result
    }

    /// Bypassing slice selection lets real-tokenizer benchmarks verify every emitted bit.
    #[cfg(feature = "bench-internals")]
    pub fn write_full_trie_mask_for_bench(
        &mut self,
        vocabulary: &VocabularyHandle,
        trie: &BoundByteTrie,
        out: &mut [u8],
    ) -> Result<(), StructuredMatcherError> {
        validate_mask_output(vocabulary.mask_vocab_size(), out)?;
        out.fill(0);
        let result = match &mut self.backend {
            MatcherBackend::Incremental(incremental) => {
                if trie.fingerprint() != vocabulary.fingerprint() {
                    return Err(missing_trie_error());
                }
                walk_trie_mask_inner(
                    incremental,
                    vocabulary,
                    trie.trie(),
                    out,
                    self.program.limits,
                    &mut 0,
                )
            }
            MatcherBackend::Reference(reference) => reference
                .write_mask_le_bytes_into(
                    vocabulary.mask_vocab_size(),
                    vocabulary.iter_records(),
                    out,
                )
                .map_err(StructuredMatcherError::from),
        };
        if result.is_err() {
            out.fill(0);
        }
        result
    }
}

fn validate_mask_output(width: usize, out: &[u8]) -> Result<(), StructuredMatcherError> {
    let bytes = width
        .div_ceil(32)
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(mask_output_error)?;
    if out.len() != bytes {
        return Err(mask_output_error());
    }
    Ok(())
}

fn mask_output_error() -> StructuredMatcherError {
    CompileError::new(
        ErrorCode::ArtifactOutOfBounds,
        crate::error::Stage::L4Bind,
        "structured mask output length does not match vocabulary width",
    )
    .into()
}

fn missing_trie_error() -> StructuredMatcherError {
    CompileError::new(
        ErrorCode::ArtifactMismatch,
        crate::error::Stage::L4Bind,
        "incremental structured matcher requires its bound byte trie",
    )
    .into()
}

fn set_token_bit_le(
    out: &mut [u8],
    width: usize,
    token: u32,
) -> Result<(), StructuredMatcherError> {
    let token = usize::try_from(token).map_err(|_| mask_output_error())?;
    if token >= width {
        return Err(mask_output_error());
    }
    let byte = token.checked_div(8).ok_or_else(mask_output_error)?;
    let bit = u32::try_from(token % 8).map_err(|_| mask_output_error())?;
    out[byte] |= 1u8 << bit;
    Ok(())
}

fn validate_mask_output_words(width: usize, out: &[u32]) -> Result<(), StructuredMatcherError> {
    if out.len() != width.div_ceil(32) {
        return Err(mask_output_error());
    }
    Ok(())
}

fn set_token_bit_word(
    out: &mut [u32],
    width: usize,
    token: u32,
) -> Result<(), StructuredMatcherError> {
    let token = usize::try_from(token).map_err(|_| mask_output_error())?;
    if token >= width {
        return Err(mask_output_error());
    }
    out[token / 32] |= 1u32 << (token % 32);
    Ok(())
}

fn candidate_is_allowed(
    state: &mut StructuredState<'static>,
    bytes: &[u8],
) -> Result<bool, StructuredMatcherError> {
    if bytes.is_empty() {
        return Ok(false);
    }
    state.validate_retained_budget()?;
    let mark = state.checkpoint();
    for &byte in bytes {
        match state.try_push_byte_for_mask(byte) {
            Ok(true) => {}
            Ok(false) => {
                state.rollback(mark);
                return Ok(false);
            }
            Err(error) => {
                state.rollback(mark);
                return Err(error.into());
            }
        }
    }
    state.rollback(mark);
    Ok(true)
}

fn walk_trie_mask(
    incremental: &mut IncrementalMatcher,
    vocabulary: &VocabularyHandle,
    bound: &BoundByteTrie,
    out: &mut [u8],
    limits: StructuredLimits,
) -> Result<(), StructuredMatcherError> {
    #[cfg(feature = "bench-internals")]
    incremental.reset_mask_metrics();
    if bound.fingerprint() != vocabulary.fingerprint() {
        return Err(missing_trie_error());
    }
    let (selected, mut proof_work) = select_adaptive_trie(incremental, vocabulary, limits)?;
    if let Some(slice) = selected {
        match slice.bounded {
            Some(mask) => or_bounded_mask(out, mask, slice.candidate.as_deref()),
            None => or_token_mask(
                out,
                slice.catalog.safe().token_mask(),
                slice.candidate.as_deref(),
            ),
        }
        if slice.body_prefix {
            or_token_mask(
                out,
                slice.catalog.body_extra().token_mask(),
                slice.candidate.as_deref(),
            );
        }
        let mut result = if slice.reject_invalid {
            walk_body_boundary_trie_mask_inner(
                incremental,
                vocabulary,
                slice.catalog.body_boundary_trie(),
                slice.catalog.body_boundary_max_bytes(),
                out,
                limits,
                &mut proof_work,
            )
        } else {
            walk_trie_mask_inner(
                incremental,
                vocabulary,
                if slice.body_prefix {
                    slice.catalog.body_uncertain_trie()
                } else {
                    slice.catalog.uncertain_trie()
                },
                out,
                limits,
                &mut proof_work,
            )
        };
        if result.is_ok() && slice.bounded.is_some_and(|mask| mask.needs_overflow_walk()) {
            result = walk_trie_mask_inner(
                incremental,
                vocabulary,
                slice.catalog.scalar_overflow_trie(),
                out,
                limits,
                &mut proof_work,
            );
        }
        #[cfg(feature = "bench-internals")]
        incremental.finish_mask_metrics();
        return result;
    }
    let result = walk_trie_mask_inner(
        incremental,
        vocabulary,
        bound.trie(),
        out,
        limits,
        &mut proof_work,
    );
    #[cfg(feature = "bench-internals")]
    incremental.finish_mask_metrics();
    result
}

fn walk_trie_mask_inner(
    incremental: &mut IncrementalMatcher,
    vocabulary: &VocabularyHandle,
    trie: &VocabTrie,
    out: &mut [u8],
    limits: StructuredLimits,
    work: &mut u64,
) -> Result<(), StructuredMatcherError> {
    incremental.state.validate_retained_budget()?;
    let width = vocabulary.mask_vocab_size();
    let root = incremental.state.checkpoint();
    #[cfg(feature = "bench-internals")]
    incremental.increment_mask_metric(MaskMetric::Checkpoint);
    incremental.events.clear();
    reserve_events(&mut incremental.events, 1, limits)?;
    incremental.events.push(TrieEvent::Visit {
        node: TrieNodeId(0),
        incoming_byte: None,
    });
    let result = (|| {
        while let Some(event) = incremental.events.pop() {
            match event {
                TrieEvent::Restore(mark) => {
                    incremental.state.rollback(mark);
                    #[cfg(feature = "bench-internals")]
                    incremental.increment_mask_metric(MaskMetric::Rollback);
                }
                TrieEvent::Visit {
                    node,
                    incoming_byte,
                } => {
                    #[cfg(feature = "bench-internals")]
                    incremental.increment_mask_metric(MaskMetric::TrieNode);
                    let mark = incremental.state.checkpoint();
                    #[cfg(feature = "bench-internals")]
                    incremental.increment_mask_metric(MaskMetric::Checkpoint);
                    if let Some(byte) = incoming_byte {
                        #[cfg(feature = "bench-internals")]
                        {
                            incremental.increment_mask_metric(MaskMetric::TrieEdge);
                            incremental.increment_mask_metric(MaskMetric::PushedByte);
                        }
                        match incremental.state.try_push_byte_for_mask(byte) {
                            Ok(true) => {}
                            Ok(false) => {
                                incremental.state.rollback(mark);
                                #[cfg(feature = "bench-internals")]
                                incremental.increment_mask_metric(MaskMetric::Rollback);
                                continue;
                            }
                            Err(error) => return Err(error.into()),
                        }
                    }
                    for &token in trie.leaf_tokens_of(node)? {
                        charge_mask_work(work, 0, 1, limits)?;
                        set_token_bit_le(out, width, token)?;
                    }
                    let (keys, children) = trie.children(node)?;
                    let event_count = children.len().checked_add(1).ok_or({
                        StructuredMatcherError::ResourceLimit {
                            kind: LimitKind::MaskScratchBytes,
                            observed: usize::MAX,
                            limit: limits.max_mask_scratch_bytes,
                        }
                    })?;
                    reserve_events(&mut incremental.events, event_count, limits)?;
                    incremental.events.push(TrieEvent::Restore(mark));
                    for (&key, &child) in keys.iter().zip(children.iter()).rev() {
                        let byte = u8::try_from(key).map_err(|_| missing_trie_error())?;
                        charge_mask_work(work, 1, 0, limits)?;
                        incremental.events.push(TrieEvent::Visit {
                            node: child,
                            incoming_byte: Some(byte),
                        });
                    }
                }
                TrieEvent::LocalVisit { .. } | TrieEvent::RestoreLocalPath(_) => {
                    unreachable!("local lexical event in exact trie walk")
                }
            }
        }
        Ok(())
    })();
    incremental.state.rollback(root);
    #[cfg(feature = "bench-internals")]
    incremental.increment_mask_metric(MaskMetric::Rollback);
    result
}

/// Walks tokens that cross a JSON-string closing quote.
/// Replays the shared prefix under one checkpoint before mutating matcher state.
fn walk_body_boundary_trie_mask_inner(
    incremental: &mut IncrementalMatcher,
    vocabulary: &VocabularyHandle,
    trie: &VocabTrie,
    max_path_bytes: usize,
    out: &mut [u8],
    limits: StructuredLimits,
    work: &mut u64,
) -> Result<(), StructuredMatcherError> {
    incremental.state.validate_retained_budget()?;
    reserve_lexical_path(incremental, max_path_bytes, limits)?;
    let width = vocabulary.mask_vocab_size();
    let root = incremental.state.checkpoint();
    incremental.events.clear();
    incremental.lexical_path.clear();
    reserve_events_with_extra(
        &mut incremental.events,
        1,
        incremental.lexical_path.capacity(),
        limits,
    )?;
    incremental.events.push(TrieEvent::LocalVisit {
        node: TrieNodeId(0),
        incoming_byte: None,
        decoder: JsonStringDecoder::new(),
        path_len: 0,
    });
    let result = (|| {
        while let Some(event) = incremental.events.pop() {
            match event {
                TrieEvent::Restore(mark) => {
                    incremental.state.rollback(mark);
                    #[cfg(feature = "bench-internals")]
                    incremental.increment_mask_metric(MaskMetric::Rollback);
                }
                TrieEvent::RestoreLocalPath(len) => incremental.lexical_path.truncate(len),
                TrieEvent::LocalVisit {
                    node,
                    incoming_byte,
                    mut decoder,
                    path_len,
                } => {
                    incremental.lexical_path.truncate(path_len);
                    let boundary = if let Some(byte) = incoming_byte {
                        if incremental.lexical_path.len() == max_path_bytes {
                            return Err(StructuredMatcherError::ResourceLimit {
                                kind: LimitKind::MaskScratchBytes,
                                observed: max_path_bytes.saturating_add(1),
                                limit: limits.max_mask_scratch_bytes,
                            });
                        }
                        incremental.lexical_path.push(byte);
                        if byte == b'"' && decoder.at_boundary() {
                            true
                        } else {
                            #[cfg(feature = "bench-internals")]
                            incremental.increment_mask_metric(MaskMetric::LocalDfaTransition);
                            if decoder.push(byte) == DecodeStep::Invalid {
                                incremental.lexical_path.truncate(path_len);
                                continue;
                            }
                            false
                        }
                    } else {
                        false
                    };

                    if boundary {
                        #[cfg(feature = "bench-internals")]
                        incremental.increment_mask_metric(MaskMetric::TrieNode);
                        let mark = incremental.state.checkpoint();
                        #[cfg(feature = "bench-internals")]
                        incremental.increment_mask_metric(MaskMetric::Checkpoint);
                        let mut accepted = true;
                        for index in 0..incremental.lexical_path.len() {
                            let byte = incremental.lexical_path[index];
                            #[cfg(feature = "bench-internals")]
                            {
                                incremental.increment_mask_metric(MaskMetric::TrieEdge);
                                incremental.increment_mask_metric(MaskMetric::PushedByte);
                            }
                            if !incremental.state.try_push_byte_for_mask(byte)? {
                                accepted = false;
                                break;
                            }
                        }
                        if !accepted {
                            incremental.state.rollback(mark);
                            #[cfg(feature = "bench-internals")]
                            incremental.increment_mask_metric(MaskMetric::Rollback);
                            incremental.lexical_path.truncate(path_len);
                            continue;
                        }
                        for &token in trie.leaf_tokens_of(node)? {
                            charge_mask_work(work, 0, 1, limits)?;
                            set_token_bit_le(out, width, token)?;
                        }
                        let (keys, children) = trie.children(node)?;
                        let event_count = children.len().checked_add(1).ok_or(
                            StructuredMatcherError::ResourceLimit {
                                kind: LimitKind::MaskScratchBytes,
                                observed: usize::MAX,
                                limit: limits.max_mask_scratch_bytes,
                            },
                        )?;
                        reserve_events_with_extra(
                            &mut incremental.events,
                            event_count,
                            incremental.lexical_path.capacity(),
                            limits,
                        )?;
                        incremental.events.push(TrieEvent::Restore(mark));
                        for (&key, &child) in keys.iter().zip(children.iter()).rev() {
                            let byte = u8::try_from(key).map_err(|_| missing_trie_error())?;
                            charge_mask_work(work, 1, 0, limits)?;
                            #[cfg(feature = "bench-internals")]
                            incremental.increment_mask_metric(MaskMetric::LocalDfaTransition);
                            if incremental.state.next_byte_is_lexically_impossible(byte) {
                                continue;
                            }
                            incremental.events.push(TrieEvent::Visit {
                                node: child,
                                incoming_byte: Some(byte),
                            });
                        }
                        incremental.lexical_path.truncate(path_len);
                        continue;
                    }

                    debug_assert!(trie.leaf_tokens_of(node)?.is_empty());
                    let (keys, children) = trie.children(node)?;
                    let event_count = children.len().checked_add(1).ok_or(
                        StructuredMatcherError::ResourceLimit {
                            kind: LimitKind::MaskScratchBytes,
                            observed: usize::MAX,
                            limit: limits.max_mask_scratch_bytes,
                        },
                    )?;
                    reserve_events_with_extra(
                        &mut incremental.events,
                        event_count,
                        incremental.lexical_path.capacity(),
                        limits,
                    )?;
                    incremental
                        .events
                        .push(TrieEvent::RestoreLocalPath(path_len));
                    let child_path_len = incremental.lexical_path.len();
                    for (&key, &child) in keys.iter().zip(children.iter()).rev() {
                        let byte = u8::try_from(key).map_err(|_| missing_trie_error())?;
                        charge_mask_work(work, 1, 0, limits)?;
                        incremental.events.push(TrieEvent::LocalVisit {
                            node: child,
                            incoming_byte: Some(byte),
                            decoder,
                            path_len: child_path_len,
                        });
                    }
                }
                TrieEvent::Visit {
                    node,
                    incoming_byte,
                } => {
                    #[cfg(feature = "bench-internals")]
                    incremental.increment_mask_metric(MaskMetric::TrieNode);
                    let mark = incremental.state.checkpoint();
                    #[cfg(feature = "bench-internals")]
                    incremental.increment_mask_metric(MaskMetric::Checkpoint);
                    if let Some(byte) = incoming_byte {
                        #[cfg(feature = "bench-internals")]
                        {
                            incremental.increment_mask_metric(MaskMetric::TrieEdge);
                            incremental.increment_mask_metric(MaskMetric::PushedByte);
                        }
                        match incremental.state.try_push_byte_for_mask(byte) {
                            Ok(true) => {}
                            Ok(false) => {
                                incremental.state.rollback(mark);
                                #[cfg(feature = "bench-internals")]
                                incremental.increment_mask_metric(MaskMetric::Rollback);
                                continue;
                            }
                            Err(error) => return Err(error.into()),
                        }
                    }
                    for &token in trie.leaf_tokens_of(node)? {
                        charge_mask_work(work, 0, 1, limits)?;
                        set_token_bit_le(out, width, token)?;
                    }
                    let (keys, children) = trie.children(node)?;
                    let event_count = children.len().checked_add(1).ok_or(
                        StructuredMatcherError::ResourceLimit {
                            kind: LimitKind::MaskScratchBytes,
                            observed: usize::MAX,
                            limit: limits.max_mask_scratch_bytes,
                        },
                    )?;
                    reserve_events_with_extra(
                        &mut incremental.events,
                        event_count,
                        incremental.lexical_path.capacity(),
                        limits,
                    )?;
                    incremental.events.push(TrieEvent::Restore(mark));
                    for (&key, &child) in keys.iter().zip(children.iter()).rev() {
                        let byte = u8::try_from(key).map_err(|_| missing_trie_error())?;
                        charge_mask_work(work, 1, 0, limits)?;
                        incremental.events.push(TrieEvent::Visit {
                            node: child,
                            incoming_byte: Some(byte),
                        });
                    }
                }
            }
        }
        Ok(())
    })();
    incremental.state.rollback(root);
    result
}

/// Computes slice eligibility when the innermost frame is a wrapper.
/// Wrapper cursors provide the lexical automata requiring trie-inclusion proof.
fn select_wrapped_slice<'a>(
    incremental: &mut IncrementalMatcher,
    vocabulary: &'a VocabularyHandle,
    limits: StructuredLimits,
    body_prefix: bool,
) -> Result<(Option<SelectedSlice<'a>>, u64), StructuredMatcherError> {
    let Some(catalog) = vocabulary.token_slices().map(AsRef::as_ref) else {
        return Ok((None, 0));
    };
    if catalog.safe().scalar_count().is_some() {
        return Ok((None, 0));
    }
    let mut proofs = std::mem::take(&mut incremental.certificate_proofs);
    proofs.clear();
    let max_bytes = catalog.safe().max_byte_len().max(if body_prefix {
        catalog
            .body_extra()
            .max_byte_len()
            .max(catalog.body_boundary_max_bytes())
            .max(catalog.body_invalid_max_bytes())
    } else {
        0
    });
    let Some(certificate_fuel) = incremental.state.certificate_fuel() else {
        incremental.certificate_proofs = proofs;
        return Ok((None, 0));
    };
    let mut fuel = ProofContext::for_preparation(
        max_bytes,
        incremental.state.retained_bytes(),
        limits.max_session_bytes,
        certificate_fuel,
    );
    fuel.body_prefix = body_prefix;
    let certificate = incremental.state.certify_slice(&mut proofs, &mut fuel);
    if certificate == SliceCertificate::Unknown {
        incremental.certificate_proofs = proofs;
        return Ok((None, 0));
    }
    let bounded = match fuel.remaining_scalars {
        Some(remaining) => {
            let Some(mask) = catalog.bounded_mask(max_bytes, Some(remaining)) else {
                incremental.certificate_proofs = proofs;
                return Ok((None, 0));
            };
            Some(mask)
        }
        None => None,
    };
    let mut work = 0u64;
    let mut proved = true;
    let mut reject_invalid = body_prefix;
    let mut candidate = None;
    for proof in proofs.iter().copied() {
        let (node, state, requires_live) = match proof {
            SliceProof::All(node, state) => (node, state, false),
            SliceProof::CandidateLive(node, state) => (node, state, true),
        };
        let key = SliceProofKey {
            vocabulary: vocabulary.fingerprint(),
            node,
            state,
            body_prefix,
            reject_invalid: false,
        };
        let Some(engine) = incremental.state.engine_for_node(node) else {
            proved = false;
            break;
        };
        if requires_live {
            if candidate.is_some() {
                proved = false;
                break;
            }
            let mask = if let Some(cached) = incremental.slice_proofs.candidate(key) {
                cached
            } else {
                let Some((mask, spent)) = candidate_live_mask(
                    engine,
                    state,
                    catalog.safe().filtered_trie(),
                    body_prefix.then_some(catalog.body_extra().filtered_trie()),
                    vocabulary.mask_vocab_size(),
                    limits,
                ) else {
                    proved = false;
                    break;
                };
                work = work.saturating_add(spent);
                incremental.slice_proofs.insert_candidate(
                    key,
                    mask.clone(),
                    limits.max_slice_cache_bytes,
                );
                mask
            };
            candidate = Some(mask);
            continue;
        }
        let ok = if let Some(cached) = incremental.slice_proofs.get(key) {
            cached
        } else {
            let (mut ok, mut spent) =
                safe_slice_subsumes(engine, state, catalog.safe().filtered_trie(), limits);
            if ok && body_prefix {
                let (extra_ok, extra_spent) =
                    slice_subsumes(engine, state, catalog.body_extra().filtered_trie(), limits);
                ok = extra_ok;
                spent = spent.saturating_add(extra_spent);
            }
            work = work.saturating_add(spent);
            incremental
                .slice_proofs
                .insert(key, ok, limits.max_slice_cache_bytes);
            ok
        };
        if !ok {
            proved = false;
            break;
        }
        if reject_invalid {
            let rejection_key = SliceProofKey {
                reject_invalid: true,
                ..key
            };
            reject_invalid = if let Some(cached) = incremental.slice_proofs.get(rejection_key) {
                cached
            } else {
                let (rejected, spent) = slice_rejects_at_lexical_error(
                    engine,
                    state,
                    catalog.body_invalid_trie(),
                    limits,
                );
                work = work.saturating_add(spent);
                incremental.slice_proofs.insert(
                    rejection_key,
                    rejected,
                    limits.max_slice_cache_bytes,
                );
                rejected
            };
        }
    }
    incremental.certificate_proofs = proofs;
    if proved {
        incremental
            .state
            .prepare_certified_slice(&mut ProofContext::with_fuel(max_bytes, certificate_fuel))?;
    }
    let selected = SelectedSlice {
        catalog,
        bounded,
        body_prefix,
        reject_invalid,
        candidate,
    };
    Ok((proved.then_some(selected), work))
}

/// The safe tokens a mask may take wholesale, plus the trie that still needs an exact walk.
struct SelectedSlice<'a> {
    catalog: &'a TokenSliceCatalog,
    /// `None` means every safe token fits the remaining budgets, so the whole safe mask applies.
    bounded: Option<BoundedMask<'a>>,
    body_prefix: bool,
    reject_invalid: bool,
    candidate: Option<Arc<[u64]>>,
}

fn select_adaptive_trie<'a>(
    incremental: &mut IncrementalMatcher,
    vocabulary: &'a VocabularyHandle,
    limits: StructuredLimits,
) -> Result<(Option<SelectedSlice<'a>>, u64), StructuredMatcherError> {
    let (body, body_work) = select_wrapped_slice(incremental, vocabulary, limits, true)?;
    if body.is_some() {
        return Ok((body, body_work));
    }
    let (slice, work) = select_plain_adaptive_trie(incremental, vocabulary, limits)?;
    Ok((slice, work.saturating_add(body_work)))
}

fn select_plain_adaptive_trie<'a>(
    incremental: &mut IncrementalMatcher,
    vocabulary: &'a VocabularyHandle,
    limits: StructuredLimits,
) -> Result<(Option<SelectedSlice<'a>>, u64), StructuredMatcherError> {
    if incremental.state.local_slice_residual().is_none() {
        return select_wrapped_slice(incremental, vocabulary, limits, false);
    }
    let (remaining_document_bytes, remaining_scalars, needs_proof) = {
        let Some(residual) = incremental.state.local_slice_residual() else {
            return Ok((None, 0));
        };
        let needs_proof = match (residual.engine, residual.pattern_state) {
            // No local automaton constrains this position, so nothing can accept and pop here.
            (None, None) => false,
            (Some(engine), Some(state))
                if engine.consume_token(state, b"a").is_some()
                    && engine.consume_token(state, b"z").is_some() =>
            {
                true
            }
            _ => return Ok((None, 0)),
        };
        (
            residual.remaining_document_bytes,
            residual.remaining_scalars,
            needs_proof,
        )
    };
    let Some(catalog) = vocabulary.token_slices().map(AsRef::as_ref) else {
        return Ok((None, 0));
    };
    if catalog.safe().scalar_count().is_some() {
        return Ok((None, 0));
    }
    // Length-bounded tables retain safe short tokens when the full mask exceeds a budget.
    // Resource overflow remains an error in the exact walker.
    if catalog.safe().max_byte_len() > remaining_document_bytes {
        return Ok((None, 0));
    }
    let unbounded =
        catalog.safe().max_byte_len() <= remaining_document_bytes && remaining_scalars.is_none();
    let (bounded, admitted_bytes) = if unbounded {
        (None, catalog.safe().max_byte_len())
    } else {
        let Some(mask) = catalog.bounded_mask(remaining_document_bytes, remaining_scalars) else {
            return Ok((None, 0));
        };
        (
            Some(mask),
            remaining_document_bytes
                .min(catalog.safe().max_byte_len())
                .min(remaining_scalars.map_or(usize::MAX, |scalars| scalars.saturating_mul(4))),
        )
    };
    if incremental.state.prepare_local_slice(admitted_bytes)? == SlicePreparation::Ineligible {
        return select_wrapped_slice(incremental, vocabulary, limits, false);
    }
    let Some(residual) = incremental.state.local_slice_residual() else {
        return Ok((None, 0));
    };
    let selected = SelectedSlice {
        catalog,
        bounded,
        body_prefix: false,
        reject_invalid: false,
        candidate: None,
    };
    if !needs_proof {
        return Ok((Some(selected), 0));
    }
    let Some(engine) = residual.engine else {
        return Ok((None, 0));
    };
    let Some(state) = residual.pattern_state else {
        return Ok((None, 0));
    };
    let key = SliceProofKey {
        vocabulary: vocabulary.fingerprint(),
        node: residual.node,
        state,
        body_prefix: false,
        reject_invalid: false,
    };
    if let Some(subsumes) = incremental.slice_proofs.get(key) {
        return Ok((subsumes.then_some(selected), 0));
    }
    let (proved, work) = safe_slice_subsumes(engine, state, catalog.safe().filtered_trie(), limits);
    incremental
        .slice_proofs
        .insert(key, proved, limits.max_slice_cache_bytes);
    Ok((proved.then_some(selected), work))
}

fn safe_slice_subsumes(
    engine: &crate::automaton::RefEngine,
    start: StateId,
    trie: &VocabTrie,
    limits: StructuredLimits,
) -> (bool, u64) {
    let (proved, work) = neutral_language_subsumes(engine, start, limits);
    if proved {
        return (true, work);
    }
    let mut remaining = limits;
    remaining.max_mask_work = remaining.max_mask_work.saturating_sub(work);
    let (proved, extra) = slice_subsumes(engine, start, trie, remaining);
    (proved, work.saturating_add(extra))
}

/// A closed product proves all neutral strings without visiting every vocabulary prefix.
fn neutral_language_subsumes(
    engine: &crate::automaton::RefEngine,
    start: StateId,
    limits: StructuredLimits,
) -> (bool, u64) {
    const CAPACITY: usize = 64;
    if std::mem::size_of::<[(StateId, u8); CAPACITY]>() > limits.max_mask_scratch_bytes {
        return (false, 0);
    }
    let mut pairs = [(start, 0u8); CAPACITY];
    let mut length = 1;
    let mut cursor = 0;
    let mut work = 0;
    while cursor < length {
        let (state, mode) = pairs[cursor];
        cursor += 1;
        for byte in 0..=u8::MAX {
            let Some(next_mode) = neutral_utf8_step(mode, byte) else {
                continue;
            };
            if work == limits.max_mask_work {
                return (false, work);
            }
            work += 1;
            let Some(next) = engine.consume_token(state, &[byte]) else {
                return (false, work);
            };
            let pair = (next, next_mode);
            if !pairs[..length].contains(&pair) {
                if length == CAPACITY {
                    return (false, work);
                }
                pairs[length] = pair;
                length += 1;
            }
        }
    }
    (true, work)
}

fn neutral_utf8_step(mode: u8, byte: u8) -> Option<u8> {
    match (mode, byte) {
        (0, b'"' | b'\\') => None,
        (0, 0x20..=0x7f) => Some(0),
        (0, 0xc2..=0xdf) => Some(1),
        (0, 0xe0) => Some(4),
        (0, 0xed) => Some(5),
        (0, 0xe1..=0xec | 0xee..=0xef) => Some(2),
        (0, 0xf0) => Some(6),
        (0, 0xf4) => Some(7),
        (0, 0xf1..=0xf3) => Some(3),
        (1..=3, 0x80..=0xbf) => Some(mode - 1),
        (4, 0xa0..=0xbf) | (5, 0x80..=0x9f) => Some(1),
        (6, 0x90..=0xbf) | (7, 0x80..=0x8f) => Some(2),
        _ => None,
    }
}

/// Proves every trie token has a live transition from `start`.
/// Acceptance does not pop a regular frame at a token boundary.
fn slice_subsumes(
    engine: &crate::automaton::RefEngine,
    start: StateId,
    trie: &VocabTrie,
    limits: StructuredLimits,
) -> (bool, u64) {
    let frontier = trie.max_dfs_frontier();
    let Some(bytes) = frontier.checked_mul(std::mem::size_of::<(TrieNodeId, StateId)>()) else {
        return (false, 0);
    };
    if bytes > limits.max_mask_scratch_bytes {
        return (false, 0);
    }
    let mut stack = Vec::new();
    if stack.try_reserve_exact(frontier.max(1)).is_err() {
        return (false, 0);
    }
    stack.push((TrieNodeId(0), start));
    let mut work = 0u64;
    while let Some((node, state)) = stack.pop() {
        let Ok((keys, children)) = trie.children(node) else {
            return (false, work);
        };
        for (&key, &child) in keys.iter().zip(children).rev() {
            let Some(next_work) = work.checked_add(1) else {
                return (false, limits.max_mask_work);
            };
            work = next_work;
            if work > limits.max_mask_work {
                return (false, limits.max_mask_work);
            }
            let Ok(byte) = u8::try_from(key) else {
                return (false, work);
            };
            let Some(next) = engine.consume_token(state, &[byte]) else {
                return (false, work);
            };
            stack.push((child, next));
        }
    }
    (true, work)
}

fn candidate_live_mask(
    engine: &crate::automaton::RefEngine,
    start: StateId,
    safe: &VocabTrie,
    extra: Option<&VocabTrie>,
    width: usize,
    limits: StructuredLimits,
) -> Option<(Arc<[u64]>, u64)> {
    let words = width.div_ceil(u64::BITS as usize);
    let bytes = words.checked_mul(std::mem::size_of::<u64>())?;
    if bytes > limits.max_slice_cache_bytes / 2 || bytes > limits.max_mask_scratch_bytes {
        return None;
    }
    let mut mask = Vec::new();
    mask.try_reserve_exact(words).ok()?;
    mask.resize(words, 0);
    let mut work = 0;
    candidate_live_mask_into(engine, start, safe, &mut mask, &mut work, limits)?;
    if let Some(extra) = extra {
        candidate_live_mask_into(engine, start, extra, &mut mask, &mut work, limits)?;
    }
    Some((Arc::from(mask), work))
}

fn candidate_live_mask_into(
    engine: &crate::automaton::RefEngine,
    start: StateId,
    trie: &VocabTrie,
    mask: &mut [u64],
    work: &mut u64,
    limits: StructuredLimits,
) -> Option<()> {
    let mut stack = Vec::new();
    stack
        .try_reserve_exact(trie.max_dfs_frontier().max(1))
        .ok()?;
    stack.push((TrieNodeId(0), start));
    while let Some((node, state)) = stack.pop() {
        for &token in trie.leaf_tokens_of(node).ok()? {
            let index = usize::try_from(token).ok()?;
            *mask.get_mut(index / u64::BITS as usize)? |= 1u64 << (index % u64::BITS as usize);
        }
        let (keys, children) = trie.children(node).ok()?;
        for (&key, &child) in keys.iter().zip(children).rev() {
            *work = work.checked_add(1)?;
            if *work > limits.max_mask_work {
                return None;
            }
            let byte = u8::try_from(key).ok()?;
            let Some(next) = engine.consume_token(state, &[byte]) else {
                continue;
            };
            if !engine.is_dead(next) {
                stack.push((child, next));
            }
        }
    }
    Some(())
}

fn slice_rejects_at_lexical_error(
    engine: &crate::automaton::RefEngine,
    start: StateId,
    trie: &VocabTrie,
    limits: StructuredLimits,
) -> (bool, u64) {
    use super::lexer::{DecodeStep, JsonStringDecoder};
    let frontier = trie.max_dfs_frontier().max(1);
    let Some(bytes) =
        frontier.checked_mul(std::mem::size_of::<(TrieNodeId, StateId, JsonStringDecoder)>())
    else {
        return (false, 0);
    };
    if bytes > limits.max_mask_scratch_bytes || engine.is_accepting(start) {
        return (false, 0);
    }
    let mut stack = Vec::new();
    if stack.try_reserve_exact(frontier).is_err() {
        return (false, 0);
    }
    stack.push((TrieNodeId(0), start, JsonStringDecoder::new()));
    let mut work = 0u64;
    while let Some((node, state, decoder)) = stack.pop() {
        if !trie
            .leaf_tokens_of(node)
            .is_ok_and(|tokens| tokens.is_empty())
        {
            return (false, work);
        }
        let Ok((keys, children)) = trie.children(node) else {
            return (false, work);
        };
        for (&key, &child) in keys.iter().zip(children).rev() {
            if work >= limits.max_mask_work {
                return (false, work);
            }
            work += 1;
            let Ok(byte) = u8::try_from(key) else {
                return (false, work);
            };
            let mut next_decoder = decoder;
            if byte == b'"' && decoder.at_boundary() {
                return (false, work);
            }
            let next = engine.consume_token(state, &[byte]);
            if next_decoder.push(byte) == DecodeStep::Invalid {
                // Rejection at the same nonaccepting prefix cannot replay into a parent.
                if next.is_some() || engine.is_accepting(state) {
                    return (false, work);
                }
                continue;
            }
            let Some(next) = next.filter(|&state| !engine.is_accepting(state)) else {
                return (false, work);
            };
            stack.push((child, next, next_decoder));
        }
    }
    (true, work)
}

fn or_token_mask(output: &mut [u8], mask: &[u64], filter: Option<&[u64]>) {
    for (index, (chunk, word)) in output.chunks_mut(8).zip(mask).enumerate() {
        let word =
            *word & filter.map_or(u64::MAX, |filter| filter.get(index).copied().unwrap_or(0));
        let bytes = word.to_le_bytes();
        for (dst, src) in chunk.iter_mut().zip(bytes) {
            *dst |= src;
        }
    }
}

/// Like `or_token_mask`, but each word is intersected on the fly, so no scratch buffer is needed.
fn or_bounded_mask(output: &mut [u8], mask: BoundedMask<'_>, filter: Option<&[u64]>) {
    for (index, chunk) in output.chunks_mut(8).take(mask.words()).enumerate() {
        let word = mask.word(index)
            & filter.map_or(u64::MAX, |filter| filter.get(index).copied().unwrap_or(0));
        let bytes = word.to_le_bytes();
        for (dst, src) in chunk.iter_mut().zip(bytes) {
            *dst |= src;
        }
    }
}

#[cfg(test)]
fn walk_full_trie_mask_for_test(
    incremental: &mut IncrementalMatcher,
    vocabulary: &VocabularyHandle,
    bound: &BoundByteTrie,
    out: &mut [u8],
    limits: StructuredLimits,
) -> Result<(), StructuredMatcherError> {
    if bound.fingerprint() != vocabulary.fingerprint() {
        return Err(missing_trie_error());
    }
    walk_trie_mask_inner(incremental, vocabulary, bound.trie(), out, limits, &mut 0)
}

fn reserve_events(
    events: &mut Vec<TrieEvent>,
    additional: usize,
    limits: StructuredLimits,
) -> Result<(), StructuredMatcherError> {
    reserve_events_with_extra(events, additional, 0, limits)
}

fn reserve_events_with_extra(
    events: &mut Vec<TrieEvent>,
    additional: usize,
    extra_bytes: usize,
    limits: StructuredLimits,
) -> Result<(), StructuredMatcherError> {
    let needed = events.len().checked_add(additional).ok_or({
        StructuredMatcherError::ResourceLimit {
            kind: LimitKind::MaskScratchBytes,
            observed: usize::MAX,
            limit: limits.max_mask_scratch_bytes,
        }
    })?;
    let bytes = needed
        .checked_mul(std::mem::size_of::<TrieEvent>())
        .and_then(|bytes| bytes.checked_add(extra_bytes))
        .ok_or(StructuredMatcherError::ResourceLimit {
            kind: LimitKind::MaskScratchBytes,
            observed: usize::MAX,
            limit: limits.max_mask_scratch_bytes,
        })?;
    if bytes > limits.max_mask_scratch_bytes {
        return Err(StructuredMatcherError::ResourceLimit {
            kind: LimitKind::MaskScratchBytes,
            observed: bytes,
            limit: limits.max_mask_scratch_bytes,
        });
    }
    if needed > events.capacity() {
        events
            .try_reserve_exact(needed - events.len())
            .map_err(|_| StructuredMatcherError::ResourceLimit {
                kind: LimitKind::MaskScratchBytes,
                observed: bytes,
                limit: limits.max_mask_scratch_bytes,
            })?;
    }
    let actual = events
        .capacity()
        .checked_mul(std::mem::size_of::<TrieEvent>())
        .and_then(|bytes| bytes.checked_add(extra_bytes))
        .ok_or(StructuredMatcherError::ResourceLimit {
            kind: LimitKind::MaskScratchBytes,
            observed: usize::MAX,
            limit: limits.max_mask_scratch_bytes,
        })?;
    if actual > limits.max_mask_scratch_bytes {
        return Err(StructuredMatcherError::ResourceLimit {
            kind: LimitKind::MaskScratchBytes,
            observed: actual,
            limit: limits.max_mask_scratch_bytes,
        });
    }
    Ok(())
}

fn reserve_lexical_path(
    incremental: &mut IncrementalMatcher,
    needed: usize,
    limits: StructuredLimits,
) -> Result<(), StructuredMatcherError> {
    let projected = incremental
        .events
        .capacity()
        .checked_mul(std::mem::size_of::<TrieEvent>())
        .and_then(|bytes| bytes.checked_add(needed))
        .ok_or(StructuredMatcherError::ResourceLimit {
            kind: LimitKind::MaskScratchBytes,
            observed: usize::MAX,
            limit: limits.max_mask_scratch_bytes,
        })?;
    if projected > limits.max_mask_scratch_bytes {
        return Err(StructuredMatcherError::ResourceLimit {
            kind: LimitKind::MaskScratchBytes,
            observed: projected,
            limit: limits.max_mask_scratch_bytes,
        });
    }
    if needed > incremental.lexical_path.capacity() {
        incremental
            .lexical_path
            .try_reserve_exact(needed - incremental.lexical_path.len())
            .map_err(|_| StructuredMatcherError::ResourceLimit {
                kind: LimitKind::MaskScratchBytes,
                observed: projected,
                limit: limits.max_mask_scratch_bytes,
            })?;
    }
    let actual = incremental
        .events
        .capacity()
        .checked_mul(std::mem::size_of::<TrieEvent>())
        .and_then(|bytes| bytes.checked_add(incremental.lexical_path.capacity()))
        .ok_or(StructuredMatcherError::ResourceLimit {
            kind: LimitKind::MaskScratchBytes,
            observed: usize::MAX,
            limit: limits.max_mask_scratch_bytes,
        })?;
    if actual > limits.max_mask_scratch_bytes {
        return Err(StructuredMatcherError::ResourceLimit {
            kind: LimitKind::MaskScratchBytes,
            observed: actual,
            limit: limits.max_mask_scratch_bytes,
        });
    }
    Ok(())
}

fn charge_mask_work(
    work: &mut u64,
    bytes: usize,
    ids: usize,
    limits: StructuredLimits,
) -> Result<(), StructuredMatcherError> {
    let delta = u64::try_from(bytes)
        .ok()
        .and_then(|value| value.checked_add(u64::try_from(ids).ok()?))
        .ok_or(StructuredMatcherError::ResourceLimit {
            kind: LimitKind::MaskWork,
            observed: usize::MAX,
            limit: usize::try_from(limits.max_mask_work).unwrap_or(usize::MAX),
        })?;
    let next = work
        .checked_add(delta)
        .ok_or(StructuredMatcherError::ResourceLimit {
            kind: LimitKind::MaskWork,
            observed: usize::MAX,
            limit: usize::try_from(limits.max_mask_work).unwrap_or(usize::MAX),
        })?;
    if next > limits.max_mask_work {
        return Err(StructuredMatcherError::ResourceLimit {
            kind: LimitKind::MaskWork,
            observed: usize::try_from(next).unwrap_or(usize::MAX),
            limit: usize::try_from(limits.max_mask_work).unwrap_or(usize::MAX),
        });
    }
    *work = next;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::schema_to_ir;
    use crate::index::TrieCache;
    use crate::ir::{Charset, CompileOptions, ScalarLit, MAX_UNROLLED_ENUM};
    use crate::vocab::build_vocabulary;
    use rustc_hash::FxHashMap;

    fn ir(schema: &str) -> Arc<SchemaIR> {
        Arc::new(schema_to_ir(schema, CompileOptions::default()).expect("schema"))
    }

    fn program(schema: &str) -> Arc<StructuredProgram> {
        StructuredProgram::compile(ir(schema)).expect("program")
    }

    fn large_string_enum_ir() -> Arc<SchemaIR> {
        let mut builder = crate::ir::Builder::new(CompileOptions::default());
        let values = (0..=MAX_UNROLLED_ENUM)
            .map(|index| ScalarLit::Str(format!("member-{index:04}")))
            .collect();
        let root = builder.enum_values(values).expect("enum");
        Arc::new(builder.finish(root).expect("ir"))
    }

    fn large_string_enum_schema() -> String {
        let members = (0..=MAX_UNROLLED_ENUM)
            .map(|index| format!(r#""member-{index:04}""#))
            .collect::<Vec<_>>()
            .join(",");
        format!(r#"{{"enum":["a","ab","abc","é","😀",{members}]}}"#)
    }

    fn mask(
        matcher: &mut StructuredMatcher,
        width: usize,
        records: &[(Vec<u8>, Vec<u32>)],
    ) -> Bitmask {
        matcher
            .allowed_mask_from_records(
                width,
                records
                    .iter()
                    .map(|(bytes, ids)| (bytes.as_slice(), ids.as_slice())),
            )
            .expect("mask")
    }

    #[cfg(feature = "bench-internals")]
    #[derive(Clone, Copy, Debug)]
    enum NearLimitTuning {
        Document(usize),
        Key(usize),
        String(usize),
        Session(usize),
        Undo(usize),
        Canonical(usize),
    }

    #[cfg(feature = "bench-internals")]
    fn tune_near_limit(matcher: &mut StructuredMatcher, tuning: NearLimitTuning) {
        let MatcherBackend::Incremental(inner) = &mut matcher.backend else {
            panic!("near-limit matrix requires the incremental backend")
        };
        match tuning {
            NearLimitTuning::Document(limit) => {
                inner.state.set_document_limit_for_test(limit);
            }
            NearLimitTuning::Key(limit) => inner.state.set_key_limit_for_test(limit),
            NearLimitTuning::String(limit) => inner.state.set_string_limit_for_test(limit),
            NearLimitTuning::Session(limit) => inner.state.set_session_limit_for_test(limit),
            NearLimitTuning::Undo(limit) => inner.state.set_undo_limit_for_test(limit),
            NearLimitTuning::Canonical(limit) => {
                inner.state.set_unique_canonical_limit_for_test(limit);
            }
        }
    }

    #[cfg(feature = "bench-internals")]
    fn assert_incremental_accounting(matcher: &StructuredMatcher, label: &str) {
        let MatcherBackend::Incremental(inner) = &matcher.backend else {
            panic!("{label}: expected incremental matcher")
        };
        assert_eq!(
            matcher.retained_session_bytes(),
            inner.state.recomputed_retained_bytes(),
            "{label}: retained-byte ledger diverged from actual capacities: {}",
            inner.state.accounting_breakdown_for_test()
        );
    }

    #[cfg(feature = "bench-internals")]
    fn assert_three_route_mask_and_advance(
        program: &Arc<StructuredProgram>,
        vocabulary: &VocabularyHandle,
        trie: &BoundByteTrie,
        prefix: &[u8],
        label: &str,
    ) -> (Vec<u8>, u64) {
        let output_len = vocabulary.mask_vocab_size().div_ceil(32) * 4;
        let mut adaptive = program.new_matcher().expect("adaptive matcher");
        let mut full = program.new_matcher().expect("full matcher");
        let mut record = program.new_matcher().expect("record matcher");
        for matcher in [&mut adaptive, &mut full, &mut record] {
            if !prefix.is_empty() {
                assert!(matcher.advance(prefix).unwrap(), "{label}: invalid prefix");
            }
        }
        let before = (
            adaptive.is_accepting(),
            adaptive.is_dead(),
            adaptive.eos_legal(),
        );
        let mut adaptive_out = vec![0xa5; output_len];
        let mut full_out = vec![0xa5; output_len];
        let mut record_out = vec![0xa5; output_len];
        adaptive
            .write_mask_le_bytes_into(vocabulary, Some(trie), &mut adaptive_out)
            .unwrap_or_else(|error| panic!("{label}: adaptive mask: {error}"));
        let nodes = adaptive.mask_work_metrics_for_bench().trie_nodes;
        full.write_full_trie_mask_for_bench(vocabulary, trie, &mut full_out)
            .unwrap_or_else(|error| panic!("{label}: full mask: {error}"));
        record
            .write_record_mask_le_bytes_into(
                vocabulary.mask_vocab_size(),
                vocabulary.iter_records(),
                &mut record_out,
            )
            .unwrap_or_else(|error| panic!("{label}: record mask: {error}"));
        assert_mask_bytes_equal(&adaptive_out, &full_out, vocabulary, label);
        assert_mask_bytes_equal(&adaptive_out, &record_out, vocabulary, label);
        assert_eq!(
            (
                adaptive.is_accepting(),
                adaptive.is_dead(),
                adaptive.eos_legal(),
            ),
            before,
            "{label}: mask changed committed logical state"
        );
        assert_incremental_accounting(&adaptive, label);
        assert_incremental_accounting(&full, label);
        assert_incremental_accounting(&record, label);
        let retained = adaptive.retained_session_bytes();
        let mut repeated = vec![0xa5; output_len];
        adaptive
            .write_mask_le_bytes_into(vocabulary, Some(trie), &mut repeated)
            .unwrap_or_else(|error| panic!("{label}: repeated mask: {error}"));
        assert_eq!(repeated, adaptive_out, "{label}: repeated mask drift");
        assert_eq!(
            adaptive.retained_session_bytes(),
            retained,
            "{label}: repeated mask retained additional capacity"
        );
        assert_incremental_accounting(&adaptive, label);

        for (token, ids) in vocabulary.iter_records() {
            let masked = ids
                .first()
                .is_some_and(|id| adaptive_out[*id as usize / 8] & (1 << (*id % 8)) != 0);
            assert!(
                ids.iter().all(|id| {
                    (adaptive_out[*id as usize / 8] & (1 << (*id % 8)) != 0) == masked
                }),
                "{label}: alias bits disagree for {token:?}"
            );
            let mut equivalent = program.new_matcher().expect("equivalent matcher");
            if !prefix.is_empty() {
                assert!(
                    equivalent.advance(prefix).unwrap(),
                    "{label}: replay prefix"
                );
            }
            let before_transition = (
                equivalent.is_accepting(),
                equivalent.is_dead(),
                equivalent.eos_legal(),
            );
            let accepted = equivalent
                .advance(token)
                .unwrap_or_else(|error| panic!("{label}: token transition error: {error}"));
            assert_eq!(
                masked, accepted,
                "{label}: mask/advance disagreement for {token:?}"
            );
            if !accepted {
                assert_eq!(
                    (
                        equivalent.is_accepting(),
                        equivalent.is_dead(),
                        equivalent.eos_legal(),
                    ),
                    before_transition,
                    "{label}: rejected transition mutated committed state for {token:?}"
                );
            }
            assert_incremental_accounting(&equivalent, label);
        }
        (adaptive_out, nodes)
    }

    #[cfg(feature = "bench-internals")]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum CompletionSearch {
        Found,
        ProvenTrap,
        Inconclusive,
    }

    /// Test-only independent liveness search. It rebuilds the matcher for every node, prioritizes
    /// JSON closers, and treats both depth and node-budget exhaustion as inconclusive.
    #[cfg(feature = "bench-internals")]
    fn closer_first_completion_search(
        program: &Arc<StructuredProgram>,
        prefix: &[u8],
        depth: usize,
        remaining_nodes: &mut usize,
    ) -> CompletionSearch {
        let mut matcher = program.new_matcher().expect("liveness matcher");
        if !matcher.advance(prefix).expect("liveness advance") {
            return CompletionSearch::ProvenTrap;
        }
        if matcher.is_accepting() {
            return CompletionSearch::Found;
        }
        if depth == 0 || *remaining_nodes == 0 {
            return CompletionSearch::Inconclusive;
        }
        const CLOSERS_FIRST: &[u8] =
            b"}\"],:0123456789truefalsnul abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-/\\.";
        let mut alphabet = Vec::with_capacity(98);
        for byte in CLOSERS_FIRST
            .iter()
            .copied()
            .chain(0x20u8..=0x7e)
            .chain([b'\t', b'\n'])
        {
            if !alphabet.contains(&byte) {
                alphabet.push(byte);
            }
        }
        let mut saw_inconclusive = false;
        for byte in alphabet {
            if *remaining_nodes == 0 {
                return CompletionSearch::Inconclusive;
            }
            *remaining_nodes -= 1;
            let mut candidate = Vec::with_capacity(prefix.len() + 1);
            candidate.extend_from_slice(prefix);
            candidate.push(byte);
            let mut probe = program.new_matcher().expect("liveness probe");
            if !probe.advance(&candidate).expect("liveness candidate") {
                continue;
            }
            match closer_first_completion_search(program, &candidate, depth - 1, remaining_nodes) {
                CompletionSearch::Found => return CompletionSearch::Found,
                CompletionSearch::Inconclusive => saw_inconclusive = true,
                CompletionSearch::ProvenTrap => {}
            }
        }
        if saw_inconclusive {
            CompletionSearch::Inconclusive
        } else {
            CompletionSearch::ProvenTrap
        }
    }

    #[cfg(feature = "bench-internals")]
    fn assert_near_limit_three_routes(
        program: &Arc<StructuredProgram>,
        vocabulary: &VocabularyHandle,
        trie: &BoundByteTrie,
        prefix: &[u8],
        tuning: NearLimitTuning,
        label: &str,
    ) {
        let mask_bytes = vocabulary.mask_vocab_size().div_ceil(32) * 4;
        let mut adaptive = program.new_matcher().expect("adaptive matcher");
        let mut full = program.new_matcher().expect("full matcher");
        let mut records = program.new_matcher().expect("record matcher");
        for matcher in [&mut adaptive, &mut full, &mut records] {
            assert!(matcher.advance(prefix).unwrap(), "{label}: prefix");
            tune_near_limit(matcher, tuning);
        }
        let before = (
            adaptive.is_accepting(),
            adaptive.is_dead(),
            adaptive.eos_legal(),
        );
        let mut adaptive_out = vec![0xa5; mask_bytes];
        let mut full_out = vec![0; mask_bytes];
        let mut record_out = vec![0xa5; mask_bytes];
        let adaptive_result =
            adaptive.write_mask_le_bytes_into(vocabulary, Some(trie), &mut adaptive_out);
        let full_result = {
            let MatcherBackend::Incremental(inner) = &mut full.backend else {
                panic!("{label}: incremental full matcher")
            };
            select_adaptive_trie(inner, vocabulary, program.limits)
                .map(|_| ())
                .and_then(|()| {
                    walk_full_trie_mask_for_test(
                        inner,
                        vocabulary,
                        trie,
                        &mut full_out,
                        program.limits,
                    )
                })
        };
        let record_preparation = {
            let MatcherBackend::Incremental(inner) = &mut records.backend else {
                panic!("{label}: incremental record matcher")
            };
            select_adaptive_trie(inner, vocabulary, program.limits).map(|_| ())
        };
        if record_preparation.is_err() {
            record_out.fill(0);
        }
        let record_result = record_preparation.and_then(|()| {
            records.write_record_mask_le_bytes_into(
                vocabulary.mask_vocab_size(),
                vocabulary.iter_records(),
                &mut record_out,
            )
        });
        assert_eq!(
            adaptive_result.is_ok(),
            full_result.is_ok(),
            "{label}: adaptive/full result"
        );
        assert_eq!(
            adaptive_result.is_ok(),
            record_result.is_ok(),
            "{label}: adaptive/record result"
        );
        assert_incremental_accounting(&adaptive, &format!("{label}/adaptive"));
        assert_incremental_accounting(&full, &format!("{label}/full"));
        assert_incremental_accounting(&records, &format!("{label}/record"));
        assert_eq!(
            (
                adaptive.is_accepting(),
                adaptive.is_dead(),
                adaptive.eos_legal()
            ),
            before,
            "{label}: mask changed committed logical state"
        );

        if adaptive_result.is_err() {
            assert_eq!(adaptive_out, vec![0; mask_bytes], "{label}: adaptive error");
            assert_eq!(record_out, vec![0; mask_bytes], "{label}: record error");
            let mut retry = vec![0xa5; mask_bytes];
            assert!(
                adaptive
                    .write_mask_le_bytes_into(vocabulary, Some(trie), &mut retry)
                    .is_err(),
                "{label}: retry must reproduce the resource error"
            );
            assert_eq!(retry, vec![0; mask_bytes], "{label}: retry error");
            assert_incremental_accounting(&adaptive, label);
            return;
        }

        assert_mask_bytes_equal(&adaptive_out, &full_out, vocabulary, label);
        assert_mask_bytes_equal(&adaptive_out, &record_out, vocabulary, label);
        let retained = adaptive.retained_session_bytes();
        let mut repeated = vec![0xa5; mask_bytes];
        adaptive
            .write_mask_le_bytes_into(vocabulary, Some(trie), &mut repeated)
            .unwrap_or_else(|error| {
                panic!("{label}: repeated mask failed after retaining {retained} bytes: {error}")
            });
        assert_eq!(adaptive_out, repeated, "{label}: repeated mask drift");
        assert_eq!(
            adaptive.retained_session_bytes(),
            retained,
            "{label}: repeated mask retained more capacity"
        );
        assert_incremental_accounting(&adaptive, label);

        for (token, ids) in vocabulary.iter_records() {
            if !ids
                .iter()
                .any(|id| adaptive_out[*id as usize / 8] & (1 << (*id % 8)) != 0)
            {
                continue;
            }
            let mut equivalent = program.new_matcher().expect("equivalent matcher");
            assert!(
                equivalent.advance(prefix).unwrap(),
                "{label}: equivalent prefix"
            );
            tune_near_limit(&mut equivalent, tuning);
            let mut equivalent_out = vec![0xa5; mask_bytes];
            equivalent
                .write_mask_le_bytes_into(vocabulary, Some(trie), &mut equivalent_out)
                .unwrap_or_else(|error| panic!("{label}: equivalent mask failed: {error}"));
            assert_eq!(equivalent_out, adaptive_out, "{label}: equivalent mask");
            assert!(
                equivalent.advance(token).unwrap_or_else(|error| {
                    panic!("{label}: emitted token raised during advance: {error}")
                }),
                "{label}: emitted token was rejected during advance"
            );
            assert_incremental_accounting(&equivalent, label);
        }
    }

    #[cfg(feature = "bench-internals")]
    fn adaptive_mask_succeeds_at(
        program: &Arc<StructuredProgram>,
        vocabulary: &VocabularyHandle,
        trie: &BoundByteTrie,
        prefix: &[u8],
        tuning: NearLimitTuning,
    ) -> bool {
        let mut matcher = program.new_matcher().expect("limit probe matcher");
        assert!(matcher.advance(prefix).unwrap());
        tune_near_limit(&mut matcher, tuning);
        let mut out = vec![0xa5; vocabulary.mask_vocab_size().div_ceil(32) * 4];
        let result = matcher.write_mask_le_bytes_into(vocabulary, Some(trie), &mut out);
        if result.is_err() {
            assert_eq!(out, vec![0; out.len()]);
        }
        assert_incremental_accounting(&matcher, "limit probe");
        result.is_ok()
    }

    #[cfg(feature = "bench-internals")]
    fn minimum_successful_limit(
        program: &Arc<StructuredProgram>,
        vocabulary: &VocabularyHandle,
        trie: &BoundByteTrie,
        prefix: &[u8],
        high: usize,
        tuning: fn(usize) -> NearLimitTuning,
    ) -> usize {
        assert!(adaptive_mask_succeeds_at(
            program,
            vocabulary,
            trie,
            prefix,
            tuning(high)
        ));
        let mut lo = 0usize;
        let mut hi = high;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if adaptive_mask_succeeds_at(program, vocabulary, trie, prefix, tuning(mid)) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }

    #[cfg(feature = "bench-internals")]
    fn around(limit: usize) -> [usize; 3] {
        [limit.saturating_sub(1), limit, limit.saturating_add(1)]
    }

    #[test]
    fn neutral_product_proof_is_bounded_and_preserves_constrained_fallback() {
        for (schema, expected) in [
            (
                r#"{"type":"object","properties":{"s":{"type":"string"}}}"#,
                true,
            ),
            (
                r#"{"type":"object","properties":{"s":{"type":"string","pattern":"^[a-z]*$"}}}"#,
                false,
            ),
        ] {
            let program = program(schema);
            let mut matcher = program.new_matcher().unwrap();
            assert!(matcher.advance(br#"{"s":"a"#).unwrap());
            #[cfg(not(feature = "bench-internals"))]
            let MatcherBackend::Incremental(inner) = &matcher.backend;
            #[cfg(feature = "bench-internals")]
            let inner = match &matcher.backend {
                MatcherBackend::Incremental(inner) => inner,
                MatcherBackend::Reference(_) => panic!("incremental program"),
            };
            let residual = inner.state.local_slice_residual().unwrap();
            let engine = residual.engine.unwrap();
            let state = residual.pattern_state.unwrap();
            let (proved, work) = neutral_language_subsumes(engine, state, program.limits);
            assert_eq!(proved, expected);
            assert!(work > 0 && work < 64 * 256);
            let mut limits = program.limits;
            limits.max_mask_work = 1;
            assert_eq!(neutral_language_subsumes(engine, state, limits), (false, 1));
        }
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn long_payment_key_mask_prepares_every_emitted_token_for_advance() {
        let schema = serde_json::json!({
            "type": "object", "unevaluatedProperties": false,
            "required": ["rail", "amountMinor", "currency", "approvals", "allocation"],
            "properties": {
                "rail": {"enum": ["ach", "wire", "internal"]},
                "amountMinor": {"type": "integer", "multipleOf": 100, "minimum": 100, "maximum": 900},
                "currency": {"enum": ["USD", "EUR", "RUB"], "not": {"const": "RUB"}},
                "iban": {"type": "string", "pattern": "^[A-Z]{2}[0-9]{4}$"},
                "approvals": {"type": "array", "items": {"enum": ["treasury", "ops", "risk"]},
                    "contains": {"const": "treasury"}, "uniqueItems": true, "minItems": 1, "maxItems": 3},
                "allocation": {"$ref": "#/$defs/split"}
            },
            "if": {"properties": {"rail": {"const": "wire"}}, "required": ["rail"]},
            "then": {"required": ["iban"]}, "else": {"not": {"required": ["iban"]}},
            "$defs": {"split": {"$dynamicAnchor": "split", "type": "object", "additionalProperties": false,
                "required": ["pct"], "properties": {"pct": {"type": "integer", "minimum": 1, "maximum": 99},
                    "sub": {"$dynamicRef": "#split"}}}}
        });
        let tokens = FxHashMap::from_iter([
            (b" .".repeat(300), vec![0, 1]),
            (b"a".to_vec(), vec![2]),
            (b"\"".to_vec(), vec![3]),
            (vec![0xf0, 0x9f], vec![4]),
        ]);
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(7, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let program = program(&schema.to_string());
        for (bytes, ids) in vocabulary.iter_records() {
            let mut matcher = program.new_matcher().unwrap();
            assert!(matcher.advance(br#"{""#).unwrap());
            let mut fast = [0; 4];
            matcher
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut fast)
                .unwrap();
            let retained_after_cold = matcher.retained_session_bytes();
            let mut repeated = [0; 4];
            matcher
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut repeated)
                .unwrap();
            assert_eq!(fast, repeated, "repeated mask changed token bits");
            assert_eq!(
                matcher.retained_session_bytes(),
                retained_after_cold,
                "repeated mask retained another key capacity"
            );
            let MatcherBackend::Incremental(inner) = &matcher.backend else {
                panic!("incremental payment program")
            };
            assert_eq!(
                inner.state.retained_bytes(),
                inner.state.recomputed_retained_bytes()
            );
            assert!(
                matcher.mask_work_metrics_for_bench().trie_nodes < trie.trie().node_count() as u64
            );
            let mut full = [0; 4];
            let mut scan = [0; 4];
            let MatcherBackend::Incremental(inner) = &mut matcher.backend else {
                panic!("incremental payment program")
            };
            walk_full_trie_mask_for_test(inner, &vocabulary, &trie, &mut full, program.limits)
                .unwrap();
            matcher
                .write_record_mask_le_bytes_into(
                    vocabulary.mask_vocab_size(),
                    vocabulary.iter_records(),
                    &mut scan,
                )
                .unwrap();
            assert_eq!(fast, full);
            assert_eq!(fast, scan);
            for (candidate, candidate_ids) in vocabulary.iter_records() {
                if !candidate_ids
                    .iter()
                    .any(|id| fast[*id as usize / 8] & (1 << (*id % 8)) != 0)
                {
                    continue;
                }
                let mut equivalent = program.new_matcher().unwrap();
                assert!(equivalent.advance(br#"{""#).unwrap());
                let mut equivalent_mask = [0; 4];
                equivalent
                    .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut equivalent_mask)
                    .unwrap();
                assert_eq!(equivalent_mask, fast);
                assert!(
                    equivalent.advance(candidate).unwrap(),
                    "emitted token {:?} must advance on an equivalent matcher",
                    candidate_ids
                );
            }
            for &id in ids {
                if fast[id as usize / 8] & (1 << (id % 8)) != 0 {
                    assert!(matcher.advance(bytes).unwrap(), "token {id}");
                    break;
                }
            }
        }
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn key_capacity_lease_transfers_releases_and_rearms_for_a_second_key() {
        let long = b" .".repeat(300);
        let tokens = FxHashMap::from_iter([
            (long.clone(), vec![0]),
            (b"a".to_vec(), vec![1]),
            (b"\"".to_vec(), vec![2]),
            (b":".to_vec(), vec![3]),
        ]);
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(4, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let program = program(r#"{"type":"object","additionalProperties":true}"#);
        let mut matcher = program.new_matcher().unwrap();
        assert!(matcher.advance(br#"{""#).unwrap());

        let mut adaptive = [0; 4];
        matcher
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut adaptive)
            .unwrap();
        let (phase, capacity, recomputed) = match &matcher.backend {
            MatcherBackend::Incremental(inner) => inner.state.key_lease_debug(),
            MatcherBackend::Reference(_) => panic!("incremental program"),
        };
        assert_eq!(phase, 1, "cold key mask arms a prepared lease");
        assert!(capacity >= long.len());
        assert_eq!(matcher.retained_session_bytes(), recomputed);

        let mut full = [0; 4];
        let mut records = [0; 4];
        let MatcherBackend::Incremental(inner) = &mut matcher.backend else {
            panic!("incremental program")
        };
        walk_full_trie_mask_for_test(inner, &vocabulary, &trie, &mut full, program.limits).unwrap();
        matcher
            .write_record_mask_le_bytes_into(
                vocabulary.mask_vocab_size(),
                vocabulary.iter_records(),
                &mut records,
            )
            .unwrap();
        assert_eq!(adaptive, full);
        assert_eq!(adaptive, records);

        assert!(
            matcher.advance(b"a").unwrap(),
            "short token uses long lease"
        );
        let (phase, _, recomputed) = match &matcher.backend {
            MatcherBackend::Incremental(inner) => inner.state.key_lease_debug(),
            MatcherBackend::Reference(_) => panic!("incremental program"),
        };
        assert_eq!(phase, 2, "successful commit transfers the lease");
        assert_eq!(matcher.retained_session_bytes(), recomputed);
        adaptive.fill(0);
        matcher
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut adaptive)
            .unwrap();
        let (phase, _, recomputed) = match &matcher.backend {
            MatcherBackend::Incremental(inner) => inner.state.key_lease_debug(),
            MatcherBackend::Reference(_) => panic!("incremental program"),
        };
        assert_eq!(
            phase, 1,
            "a later mask must rearm when the committed prefix increases required capacity"
        );
        assert_eq!(matcher.retained_session_bytes(), recomputed);
        assert!(matcher.advance(b"\"").unwrap());
        assert!(matcher.advance(b":").unwrap());
        let (phase, capacity, recomputed) = match &matcher.backend {
            MatcherBackend::Incremental(inner) => inner.state.key_lease_debug(),
            MatcherBackend::Reference(_) => panic!("incremental program"),
        };
        assert_eq!(phase, 0, "resolved key releases its transferred lease");
        assert_eq!(capacity, 0, "maximum token capacity is not retained");
        assert_eq!(matcher.retained_session_bytes(), recomputed);

        assert!(matcher.advance(br#"0,""#).unwrap());
        adaptive.fill(0);
        matcher
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut adaptive)
            .unwrap();
        let (phase, capacity, recomputed) = match &matcher.backend {
            MatcherBackend::Incremental(inner) => inner.state.key_lease_debug(),
            MatcherBackend::Reference(_) => panic!("incremental program"),
        };
        assert_eq!(phase, 1, "second key receives a fresh lease generation");
        assert!(capacity >= long.len());
        assert_eq!(matcher.retained_session_bytes(), recomputed);
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn wrapper_key_capacity_leases_are_distinct_per_structured_cursor() {
        let schema = r#"{"anyOf":[
            {"type":"object","properties":{"a":{"type":"integer"}}},
            {"type":"object","properties":{"b":{"type":"string"}}}
        ]}"#;
        let long = b"x".repeat(600);
        let vocabulary = VocabularyHandle::new(Arc::new(
            build_vocabulary(
                3,
                FxHashMap::from_iter([
                    (long, vec![0]),
                    (b"a".to_vec(), vec![1]),
                    (b"\"".to_vec(), vec![2]),
                ]),
            )
            .unwrap(),
        ))
        .unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let program = program(schema);
        let mut matcher = program.new_matcher().unwrap();
        assert!(matcher.advance(br#"{""#).unwrap());
        let mut out = [0; 4];
        matcher
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut out)
            .unwrap();
        let mut full = [0; 4];
        let mut records = [0; 4];
        let MatcherBackend::Incremental(inner) = &mut matcher.backend else {
            panic!("incremental program")
        };
        walk_full_trie_mask_for_test(inner, &vocabulary, &trie, &mut full, program.limits).unwrap();
        matcher
            .write_record_mask_le_bytes_into(
                vocabulary.mask_vocab_size(),
                vocabulary.iter_records(),
                &mut records,
            )
            .unwrap();
        assert_eq!(out, full);
        assert_eq!(out, records);
        for (candidate, ids) in vocabulary.iter_records() {
            if !ids
                .iter()
                .any(|id| out[*id as usize / 8] & (1 << (*id % 8)) != 0)
            {
                continue;
            }
            let mut equivalent = program.new_matcher().unwrap();
            assert!(equivalent.advance(br#"{""#).unwrap());
            let mut equivalent_mask = [0; 4];
            equivalent
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut equivalent_mask)
                .unwrap();
            assert_eq!(equivalent_mask, out);
            assert!(equivalent.advance(candidate).unwrap());
        }
        out.fill(0);
        matcher
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut out)
            .unwrap();
        let (owners, capacity) = match &matcher.backend {
            MatcherBackend::Incremental(inner) => inner.state.key_lease_tree_debug(),
            MatcherBackend::Reference(_) => panic!("incremental program"),
        };
        assert!(
            owners >= 2,
            "each simultaneously stepped wrapper cursor owns a lease"
        );
        assert!(capacity >= 2 * 600);
        let MatcherBackend::Incremental(inner) = &mut matcher.backend else {
            panic!("incremental program")
        };
        inner.state.discard_prepared_key_leases_tree();
        assert_eq!(
            inner.state.key_lease_tree_debug(),
            (0, 0),
            "recursive cleanup releases every cursor-owned lease"
        );
    }

    #[test]
    fn neutral_utf8_product_matches_scalar_boundaries() {
        for scalar in (0..=0x10ffff).filter_map(char::from_u32) {
            let mut buffer = [0; 4];
            let bytes = scalar.encode_utf8(&mut buffer).as_bytes();
            let result = bytes
                .iter()
                .try_fold(0, |mode, &byte| neutral_utf8_step(mode, byte));
            assert_eq!(
                result,
                (scalar >= ' ' && scalar != '"' && scalar != '\\').then_some(0)
            );
        }
        for bytes in [
            &[0xc0, 0x80][..],
            &[0xed, 0xa0, 0x80],
            &[0xf4, 0x90, 0x80, 0x80],
            &[0x80],
        ] {
            assert!(bytes
                .iter()
                .try_fold(0, |mode, &byte| neutral_utf8_step(mode, byte))
                .is_none());
        }
    }

    #[test]
    fn supported_structured_schema_uses_incremental_backend() {
        let structured = program(
            r#"{"type":"object","properties":{"x":{"type":"integer"}},"additionalProperties":true}"#,
        );
        assert_eq!(structured.backend_kind(), "incremental");
        let mut matcher = structured.new_matcher().expect("state");
        assert!(matcher.advance(br#"{"#).expect("advance"));
        assert!(!matcher.is_dead());
    }

    #[test]
    fn large_minimum_array_inside_an_object_routes_around_the_shuffle_cap() {
        let schema = r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"array","items":{"type":"null"},"minItems":500}}}"#;
        let started = std::time::Instant::now();
        let program = program(schema);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "structured routing must avoid constructing the oversized shuffle automaton"
        );
        assert!(program.uses_incremental_backend());

        let items = std::iter::repeat_n("null", 500)
            .collect::<Vec<_>>()
            .join(",");
        let document = format!(r#"{{"a":true,"b":[{items}]}}"#);
        let mut matcher = program.new_matcher().expect("state");
        assert!(matcher.advance(document.as_bytes()).expect("document"));
        assert!(matcher.is_accepting());

        let too_short = format!(r#"{{"a":true,"b":[{}]}}"#, items.replacen(",null", "", 1));
        let mut matcher = program.new_matcher().expect("state");
        assert!(!matcher.advance(too_short.as_bytes()).expect("document"));
        assert!(!matcher.is_accepting());
    }

    #[test]
    fn direct_shuffle_graph_keeps_the_closed_object_regular() {
        let schema = r#"{
            "type":"object",
            "properties":{
                "default":{"type":"string"},
                "field":{"enum":["year","organization","owners"]},
                "transform":{"enum":["upper"]}
            },
            "required":["field"],
            "additionalProperties":false
        }"#;
        let program = program(schema);
        #[cfg(not(feature = "bench-internals"))]
        let ProgramBackend::Incremental(plan) = &program.backend;
        #[cfg(feature = "bench-internals")]
        let plan = match &program.backend {
            ProgramBackend::Incremental(plan) => plan,
            ProgramBackend::ReferenceBench => panic!("production program must be incremental"),
        };
        assert!(matches!(
            plan.node(plan.root),
            crate::structured::plan::NodePlan::Regular(_)
        ));

        for document in [
            br#"{"field":"year"}"#.as_slice(),
            br#"{"transform":"upper","field":"owners","default":"x"}"#,
        ] {
            let mut matcher = program.new_matcher().expect("state");
            assert!(matcher.advance(document).expect("document"));
            assert!(matcher.is_accepting());
        }
    }

    #[test]
    fn negation_and_lowered_conditionals_use_the_incremental_backend() {
        type BackendCase<'a> = (&'a str, &'a [(&'a [u8], bool)]);
        let cases: &[BackendCase<'_>] = &[
            (
                r#"{"not":{"const":1}}"#,
                &[(b"1", false), (b"12", true), (b"2", true)],
            ),
            (
                r#"{"if":{"type":"integer"},"then":{"minimum":2},"else":{"type":"string"}}"#,
                &[
                    (b"1", false),
                    (b"2", true),
                    (br#""x""#, true),
                    (b"true", false),
                ],
            ),
        ];
        for (schema, documents) in cases {
            let program = program(schema);
            assert!(program.uses_incremental_backend(), "{schema}");
            for &(document, accepted) in *documents {
                let mut matcher = program.new_matcher().expect("state");
                let advanced = matcher.advance(document).expect("advance");
                assert!(advanced || !accepted, "{schema}: {document:?}");
                assert_eq!(matcher.is_accepting(), accepted, "{schema}: {document:?}");
            }
        }
    }

    #[test]
    fn local_recursive_root_reference_uses_incremental_backend() {
        let schema = r##"{
            "$defs": {
                "node": {
                    "type":"object",
                    "properties":{"next":{"$ref":"#/$defs/node"}},
                    "additionalProperties":false
                }
            },
            "$ref":"#/$defs/node"
        }"##;
        let program = program(schema);
        assert!(program.uses_incremental_backend());
        for document in [
            b"{}".as_slice(),
            br#"{"next":{}}"#,
            br#"{"next":{"next":{}}}"#,
        ] {
            let mut matcher = program.new_matcher().expect("state");
            assert!(matcher.advance(document).expect("document"), "{document:?}");
            assert!(matcher.is_accepting(), "{document:?}");
        }
        for document in [br#"{"next":1}"#.as_slice(), br#"{"next":{"extra":true}}"#] {
            let mut matcher = program.new_matcher().expect("state");
            assert!(
                !matcher.advance(document).expect("document"),
                "{document:?}"
            );
            assert!(!matcher.is_accepting(), "{document:?}");
        }
    }

    #[test]
    fn recursive_array_anchor_and_shared_definition_refs_are_incremental() {
        let cases: &[(&str, &[u8])] = &[
            (
                r##"{
                    "$defs":{"node":{"type":"object","properties":{"children":{"type":"array","items":{"$ref":"#/$defs/node"}}},"additionalProperties":false}},
                    "$ref":"#/$defs/node"
                }"##,
                br#"{"children":[{}, {"children":[]}]}"#,
            ),
            (
                r##"{
                    "$defs":{"node":{"$anchor":"node","type":"object","properties":{"next":{"$ref":"#node"}},"additionalProperties":false}},
                    "$ref":"#node"
                }"##,
                br#"{"next":{}}"#,
            ),
            (
                r##"{
                    "$defs":{"node":{"type":"object","properties":{"next":{"$ref":"#/$defs/node"}},"additionalProperties":false}},
                    "type":"object",
                    "properties":{"left":{"$ref":"#/$defs/node"},"right":{"$ref":"#/$defs/node"}},
                    "required":["left","right"],
                    "additionalProperties":false
                }"##,
                br#"{"left":{"next":{}},"right":{}}"#,
            ),
        ];
        for (schema, document) in cases {
            let program = program(schema);
            assert!(program.uses_incremental_backend(), "{schema}");
            let mut matcher = program.new_matcher().expect("state");
            assert!(matcher.advance(document).expect("document"), "{document:?}");
            assert!(matcher.is_accepting(), "{document:?}");
        }
    }

    #[test]
    fn productive_mutual_object_array_recursion_is_incremental() {
        let schema = r##"{
            "$defs":{
                "object":{"type":"object","properties":{"items":{"$ref":"#/$defs/list"}},"additionalProperties":false},
                "list":{"type":"array","items":{"$ref":"#/$defs/object"}}
            },
            "$ref":"#/$defs/object"
        }"##;
        let program = program(schema);
        assert!(program.uses_incremental_backend());
        let mut matcher = program.new_matcher().expect("state");
        assert!(matcher
            .advance(br#"{"items":[{}, {"items":[]}]}"#)
            .expect("document"));
        assert!(matcher.is_accepting());
    }

    #[test]
    fn pure_reference_cycles_are_incremental_empty_languages() {
        for schema in [
            r##"{"$ref":"#"}"##,
            r##"{"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"}"##,
        ] {
            let program = StructuredProgram::compile(ir(schema)).expect("positive reference SCC");
            assert!(program.uses_incremental_backend());
            let mut matcher = program.new_matcher().expect("matcher");
            assert!(!matcher.advance(b"null").expect("document"));
        }
    }

    #[test]
    fn positive_zero_consumption_cycles_use_bounded_incremental_fixed_points() {
        for schema in [
            r##"{"$defs":{"x":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/x"}]}},"$ref":"#/$defs/x"}"##,
            r##"{"$defs":{"x":{"$ref":"#/$defs/y"},"y":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/x"}]}},"$ref":"#/$defs/x"}"##,
        ] {
            let program = StructuredProgram::compile(ir(schema)).expect("positive union SCC");
            let mut matcher = program.new_matcher().expect("matcher");
            assert!(matcher.advance(b"null").expect("document"), "{schema}");
            assert!(matcher.is_accepting(), "{schema}");
        }

        let all = StructuredProgram::compile(ir(
            r##"{"$defs":{"x":{"allOf":[true,{"$ref":"#/$defs/x"}]}},"$ref":"#/$defs/x"}"##,
        ))
        .expect("positive intersection SCC");
        let mut matcher = all.new_matcher().expect("matcher");
        assert!(!matcher.advance(b"null").expect("document"));

        let mixed = StructuredProgram::compile(ir(r##"{
                "$defs":{
                    "x":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/y"}]},
                    "y":{"allOf":[true,{"$ref":"#/$defs/x"}]}
                },
                "$ref":"#/$defs/x"
            }"##))
        .expect("mixed positive SCC");
        let mut matcher = mixed.new_matcher().expect("matcher");
        assert!(matcher.advance(b"null").expect("document"));
        assert!(matcher.is_accepting());

        let error = match StructuredProgram::compile(ir(
            r##"{"$defs":{"x":{"oneOf":[{"type":"null"},{"$ref":"#/$defs/x"}]}},"$ref":"#/$defs/x"}"##,
        )) {
            Ok(_) => panic!("oneOf recursion compiled"),
            Err(error) => error,
        };
        assert_eq!(error.code, ErrorCode::NonMonotoneRecursion);

        for schema in [
            r##"{"$defs":{"x":{"type":"object","properties":{"next":{"$ref":"#/$defs/x"}},"additionalProperties":false}},"$ref":"#/$defs/x"}"##,
            r##"{"$defs":{"x":{"type":"array","items":{"$ref":"#/$defs/x"}}},"$ref":"#/$defs/x"}"##,
        ] {
            let program = program(schema);
            assert!(program.uses_incremental_backend(), "{schema}");
            assert_eq!(program.fallback_reason(), None, "{schema}");
        }
    }

    #[test]
    fn large_string_enum_selects_incremental_without_a_fallback_reason() {
        let program = StructuredProgram::compile(large_string_enum_ir()).expect("program");
        assert_eq!(program.backend_kind(), "incremental");
        assert_eq!(program.fallback_reason(), None);
        assert_eq!(
            program.new_matcher().expect("matcher").backend_name(),
            "incremental"
        );
    }

    #[test]
    fn large_string_enum_accepts_every_member_and_rejects_near_misses() {
        let program = StructuredProgram::compile(large_string_enum_ir()).expect("program");
        for index in 0..=MAX_UNROLLED_ENUM {
            let document = format!(r#""member-{index:04}""#);
            let mut matcher = program.new_matcher().expect("matcher");
            assert!(matcher.advance(document.as_bytes()).expect("member"));
            assert!(matcher.is_accepting(), "{document}");
        }
        for document in [
            r#""member-9999""#,
            r#""member-000""#,
            r#""member-00000""#,
            r#""member-0000-extra""#,
        ] {
            let mut matcher = program.new_matcher().expect("matcher");
            let accepted =
                matcher.advance(document.as_bytes()).expect("near miss") && matcher.is_accepting();
            assert!(!accepted, "{document}");
        }
    }

    #[test]
    fn large_string_enum_token_rejection_is_atomic_and_decoded_spelling_is_canonical() {
        let schema = large_string_enum_schema();
        for document in [
            b"\"a\"".as_slice(),
            b"\"\\u0061\"",
            "\"é\"".as_bytes(),
            b"\"\\u00e9\"",
            "\"😀\"".as_bytes(),
            b"\"\\uD83D\\uDE00\"",
        ] {
            let program = program(&schema);
            assert!(program.uses_incremental_backend());
            let mut matcher = program.new_matcher().expect("matcher");
            assert!(matcher.advance(document).expect("document"), "{document:?}");
            assert!(matcher.is_accepting(), "{document:?}");
        }

        let program = program(&schema);
        let mut matcher = program.new_matcher().expect("matcher");
        assert!(matcher.advance(b"\"ab").expect("prefix"));
        let retained = matcher.retained_session_bytes();
        assert!(!matcher.advance(b"z-invalid").expect("rejection"));
        assert_eq!(matcher.retained_session_bytes(), retained);
        assert!(matcher.advance(b"c\"").expect("alternative"));
        assert!(matcher.is_accepting());
    }

    #[test]
    fn large_string_enum_compositions_remain_incremental() {
        let value = large_string_enum_schema();
        let cases = [
            (
                format!(
                    r#"{{"type":"object","properties":{{"v":{value}}},"required":["v"],"additionalProperties":false}}"#
                ),
                br#"{"v":"a"}"#.as_slice(),
            ),
            (
                format!(
                    r#"{{"type":"object","patternProperties":{{"^v$":{value}}},"additionalProperties":false}}"#
                ),
                br#"{"v":"ab"}"#,
            ),
            (
                format!(r#"{{"type":"object","additionalProperties":{value}}}"#),
                br#"{"v":"abc"}"#,
            ),
            (
                format!(r#"{{"type":"array","items":{value}}}"#),
                br#"["a","ab"]"#,
            ),
            (
                format!(r#"{{"prefixItems":[{value}],"items":false}}"#),
                br#"["abc"]"#,
            ),
            (
                format!(r#"{{"type":"array","contains":{value},"minContains":1}}"#),
                br#"[null,"a"]"#,
            ),
            (
                format!(r#"{{"anyOf":[{value},{{"type":"null"}}]}}"#),
                b"\"a\"",
            ),
            (
                format!(r#"{{"allOf":[{value},{{"type":"string"}}]}}"#),
                b"\"ab\"",
            ),
            (
                format!(r#"{{"oneOf":[{value},{{"const":false}}]}}"#),
                b"\"abc\"",
            ),
            (format!(r#"{{"not":{value}}}"#), b"null"),
            (
                format!(r##"{{"$defs":{{"v":{value}}},"$ref":"#/$defs/v"}}"##),
                b"\"a\"",
            ),
            (
                format!(
                    r##"{{"$defs":{{"node":{{"type":"object","properties":{{"v":{value},"next":{{"$ref":"#/$defs/node"}}}},"additionalProperties":false}}}},"$ref":"#/$defs/node"}}"##
                ),
                br#"{"v":"a","next":{"v":"ab"}}"#,
            ),
            (
                format!(
                    r#"{{"type":"object","properties":{{"trigger":true,"v":true}},"dependentSchemas":{{"trigger":{{"properties":{{"v":{value}}},"required":["v"]}}}},"additionalProperties":true}}"#
                ),
                br#"{"trigger":true,"v":"a"}"#,
            ),
            (
                format!(r#"{{"properties":{{"known":true}},"unevaluatedProperties":{value}}}"#),
                br#"{"known":null,"extra":"a"}"#,
            ),
            (
                format!(r#"{{"prefixItems":[true],"unevaluatedItems":{value}}}"#),
                br#"[null,"a"]"#,
            ),
        ];
        for (schema, document) in cases {
            let program = program(&schema);
            assert!(program.uses_incremental_backend(), "{schema}");
            let mut matcher = program.new_matcher().expect("matcher");
            assert!(matcher.advance(document).expect("document"), "{schema}");
            assert!(matcher.is_accepting(), "{schema}");
        }
    }

    #[test]
    fn large_string_enum_prefixes_match_the_reference_oracle() {
        let schema = large_string_enum_schema();
        let schema_ir = ir(&schema);
        let program = StructuredProgram::compile(schema_ir.clone()).expect("program");
        assert!(program.uses_incremental_backend());
        for document in [b"\"abc\"".as_slice(), b"\"abd\"", b"\"\\u0061\""] {
            for end in 0..=document.len() {
                let prefix = &document[..end];
                for next in [
                    0u8, b'"', b'\\', b'a', b'b', b'c', b'z', 0x80, 0xc3, 0xe9, 0xff,
                ] {
                    let mut incremental = program.new_matcher().expect("matcher");
                    let mut reference = ReferenceMatcher::new(schema_ir.clone());
                    let incremental_prefix =
                        prefix.is_empty() || incremental.advance(prefix).expect("prefix");
                    let reference_prefix = prefix.is_empty() || reference.advance(prefix);
                    assert_eq!(incremental_prefix, reference_prefix, "{prefix:?}");
                    if incremental_prefix {
                        assert_eq!(
                            incremental.advance(&[next]).expect("next"),
                            reference.advance(&[next]),
                            "prefix={prefix:?}, next={next:#04x}"
                        );
                        assert_eq!(incremental.is_accepting(), reference.is_accepting());
                        assert_eq!(incremental.is_dead(), reference.is_dead());
                    }
                }
            }
        }
    }

    #[test]
    fn recursive_ref_every_prefix_matches_reference_oracle() {
        let schema = r##"{
            "$defs":{"node":{"type":"object","properties":{"next":{"$ref":"#/$defs/node"}},"additionalProperties":false}},
            "$ref":"#/$defs/node"
        }"##;
        let schema_ir = ir(schema);
        let program = StructuredProgram::compile(schema_ir.clone()).expect("program");
        let documents: &[&[u8]] = &[b"{}", br#"{"next":{}}"#, br#"{"next":{"next":{}}}"#];
        let deviations: &[&[u8]] = &[
            b"\0",
            b"{",
            b"}",
            b"[",
            b"]",
            b":",
            b",",
            b"\"",
            b"\\",
            b" ",
            b"\n",
            b"\xc2",
            b"\x80",
            b"\xff",
            br#"{"next":"#,
        ];

        for document in documents {
            let mut matcher = program.new_matcher().expect("matcher");
            for end in 0..=document.len() {
                let prefix = &document[..end];
                assert_eq!(
                    matcher.is_accepting(),
                    super::super::reference::accepts(&schema_ir, prefix),
                    "acceptance at {prefix:?}"
                );
                let mut records: Vec<(Vec<u8>, Vec<u32>)> = deviations
                    .iter()
                    .enumerate()
                    .map(|(id, bytes)| (bytes.to_vec(), vec![u32::try_from(id).unwrap()]))
                    .collect();
                if let Some(&next) = document.get(end) {
                    records.push((vec![next], vec![30]));
                    records.push((document[end..].to_vec(), vec![31]));
                }
                records.push((b"}".to_vec(), vec![32, 33]));
                let expected = matcher
                    .allowed_mask_from_records(
                        40,
                        records
                            .iter()
                            .map(|(bytes, ids)| (bytes.as_slice(), ids.as_slice())),
                    )
                    .expect("mask");
                for _ in 0..100 {
                    let repeated = matcher
                        .allowed_mask_from_records(
                            40,
                            records
                                .iter()
                                .map(|(bytes, ids)| (bytes.as_slice(), ids.as_slice())),
                        )
                        .expect("repeated mask");
                    assert_eq!(repeated, expected, "mask mutated state at {prefix:?}");
                }
                for (bytes, ids) in &records {
                    let mut candidate = prefix.to_vec();
                    candidate.extend_from_slice(bytes);
                    let oracle = super::super::reference::accepts(&schema_ir, &candidate)
                        || super::super::reference::can_continue(&schema_ir, &candidate);
                    for &id in ids {
                        assert_eq!(
                            expected.get(TokenId(id)),
                            oracle,
                            "prefix {prefix:?}, candidate {bytes:?}"
                        );
                    }
                }
                if let Some(&next) = document.get(end) {
                    assert!(matcher.advance(&[next]).expect("valid next byte"));
                }
            }
        }
    }

    #[test]
    fn recursive_ref_trie_record_and_reference_masks_are_identical() {
        let schema = r##"{
            "$defs":{"node":{"type":"object","properties":{"next":{"$ref":"#/$defs/node"}},"additionalProperties":false}},
            "$ref":"#/$defs/node"
        }"##;
        let schema_ir = ir(schema);
        let program = StructuredProgram::compile(schema_ir.clone()).expect("program");
        let mut tokens = FxHashMap::default();
        let pieces: &[&[u8]] = &[
            b"{",
            b"}",
            b":",
            b",",
            b"\"next\"",
            br#"{"next":"#,
            b"{}",
            b"}}",
            b" ",
            b" \n",
            b"null",
            b"[",
            b"\xff",
        ];
        for (id, piece) in pieces.iter().enumerate() {
            tokens.insert(piece.to_vec(), vec![u32::try_from(id).unwrap()]);
        }
        tokens.insert(b"}".to_vec(), vec![1, 20]);
        let width = 32usize;
        let handle = VocabularyHandle::new(Arc::new(build_vocabulary(31, tokens).expect("vocab")))
            .expect("handle");
        let trie = TrieCache::new().bind(&handle).expect("trie");
        let mut walked = program.new_matcher().expect("trie matcher");
        let mut scanned = program.new_matcher().expect("record matcher");
        let mut prefix = Vec::new();
        for step in [b"{".as_slice(), b"\"next\"", b":", b"{", b"}", b"}"] {
            let record = scanned
                .allowed_mask_from_records(width, handle.iter_records())
                .expect("record mask");
            let mut output = vec![0xff; width.div_ceil(32) * 4];
            for _ in 0..100 {
                walked
                    .write_mask_le_bytes_into(&handle, Some(&trie), &mut output)
                    .expect("trie mask");
                for id in 0..width {
                    let token = TokenId(u32::try_from(id).unwrap());
                    assert_eq!(
                        output[id / 8] & (1u8 << (id % 8)) != 0,
                        record.get(token),
                        "trie vs record at {prefix:?}, id {id}"
                    );
                }
            }
            for (bytes, ids) in handle.iter_records() {
                let mut candidate = prefix.clone();
                candidate.extend_from_slice(bytes);
                let oracle = super::super::reference::accepts(&schema_ir, &candidate)
                    || super::super::reference::can_continue(&schema_ir, &candidate);
                for &id in ids {
                    assert_eq!(record.get(TokenId(id)), oracle, "reference at id {id}");
                }
            }
            assert!(walked.advance(step).expect("trie step"));
            assert!(scanned.advance(step).expect("record step"));
            prefix.extend_from_slice(step);
        }
        assert!(walked.is_accepting() && scanned.is_accepting());
    }

    #[test]
    fn dependent_schemas_use_incremental_backend_and_validate_the_whole_object() {
        let program = program(
            r#"{"type":"object","dependentSchemas":{"x":{"type":"object","required":["y"],"additionalProperties":true}},"additionalProperties":true}"#,
        );
        assert_eq!(program.backend_kind(), "incremental");
        for (document, expected) in [
            (br#"{}"#.as_slice(), true),
            (br#"{"y":1}"#.as_slice(), true),
            (br#"{"x":1}"#.as_slice(), false),
            (br#"{"x":1,"y":2}"#.as_slice(), true),
            (br#"{"y":2,"x":1}"#.as_slice(), true),
            (br#"{"\u0078":1,"y":2}"#.as_slice(), true),
        ] {
            let mut matcher = program.new_matcher().expect("state");
            let actual = matcher.advance(document).expect("advance") && matcher.is_accepting();
            assert_eq!(actual, expected, "{}", String::from_utf8_lossy(document));
        }
    }

    #[test]
    fn dependent_schemas_hard_shapes_and_nested_locations_match_reference() {
        let cases: &[(&str, &[&[u8]])] = &[
            (
                r#"{"type":"object","dependentSchemas":{"a":{"required":["b"]},"c":{"properties":{"n":{"type":"integer"}}}},"additionalProperties":true}"#,
                &[
                    br#"{}"#,
                    br#"{"b":1}"#,
                    br#"{"a":1}"#,
                    br#"{"b":1,"a":2}"#,
                    br#"{"a":2,"b":1}"#,
                    br#"{"c":1,"n":"x"}"#,
                ],
            ),
            (
                r#"{"type":"object","dependentSchemas":{"a":{"patternProperties":{"^x":{"type":"integer"}},"propertyNames":{"pattern":"^[ax]+$"},"minProperties":2,"maxProperties":3}},"additionalProperties":true}"#,
                &[
                    br#"{"a":1,"x":2}"#,
                    br#"{"x":2,"a":1}"#,
                    br#"{"a":1,"x":"bad"}"#,
                    br#"{"a":1,"other":2}"#,
                ],
            ),
            (
                r#"{"type":"object","dependentSchemas":{"a":{"properties":{"items":{"type":"array","items":{"type":"integer"}}}}},"additionalProperties":true}"#,
                &[
                    br#"{"items":[1,2],"a":true}"#,
                    br#"{"a":true,"items":[1,"x"]}"#,
                ],
            ),
            (
                r#"{"type":"object","dependentSchemas":{"a":{"allOf":[{"required":["x"]},{"anyOf":[{"required":["y"]},{"not":{"required":["z"]}}]}]}},"additionalProperties":true}"#,
                &[
                    br#"{"a":1,"x":1,"y":1}"#,
                    br#"{"x":1,"a":1}"#,
                    br#"{"a":1,"y":1}"#,
                ],
            ),
            (
                r#"{"type":"object","properties":{"child":{"type":"object","dependentSchemas":{"a":{"required":["b"]}},"additionalProperties":true}},"dependentSchemas":{"p":{"required":["child"]}},"additionalProperties":true}"#,
                &[
                    br#"{"p":1,"child":{"b":1,"a":1}}"#,
                    br#"{"child":{"a":1},"p":1}"#,
                ],
            ),
            (
                r#"{"type":"array","items":{"type":"object","dependentSchemas":{"a":{"required":["b"]}},"additionalProperties":true}}"#,
                &[br#"[{"b":1,"a":1},{}]"#, br#"[{"a":1}]"#],
            ),
            (
                r#"{"anyOf":[{"type":"object","dependentSchemas":{"a":false}},{"type":"array"}]}"#,
                &[br#"{}"#, br#"{"a":1}"#, br#"[]"#],
            ),
            (
                r#"{"not":{"type":"object","dependentSchemas":{"a":false},"required":["a"]}}"#,
                &[br#"{}"#, br#"{"a":1}"#, b"null"],
            ),
            (
                r#"{"dependentSchemas":{"a":false}}"#,
                &[b"null", b"true", br#""text""#, b"12", b"[]", br#"{"a":1}"#],
            ),
        ];
        for (schema, documents) in cases {
            let schema_ir = ir(schema);
            let program = StructuredProgram::compile(schema_ir.clone()).expect("program");
            assert_eq!(program.backend_kind(), "incremental", "{schema}");
            for &document in *documents {
                let expected = super::super::reference::accepts(&schema_ir, document);
                let mut matcher = program.new_matcher().expect("matcher");
                let actual = matcher.advance(document).expect("advance") && matcher.is_accepting();
                assert_eq!(actual, expected, "{schema}: {document:?}");
            }
        }
    }

    #[test]
    fn dependent_schemas_prefixes_match_the_reference() {
        type Case<'a> = (&'a str, &'a [(&'a [u8], bool)]);
        let cases: &[Case<'_>] = &[
            (
                r#"{"type":"object","dependentSchemas":{"a":{"required":["b"]}},"additionalProperties":true}"#,
                &[
                    (br#"{}"#, true),
                    (br#"{"b":1}"#, true),
                    (br#"{"a":1}"#, false),
                    (br#"{"a":1,"b":2}"#, true),
                    (br#"{"b":2,"a":1}"#, true),
                ],
            ),
            (
                r#"{"type":"object","dependentSchemas":{"a":{"required":["b"]}},"additionalProperties":true}"#,
                &[(br#"{"a":1,"b":2}"#, true), (br#"{"a":1}"#, false)],
            ),
            (
                r#"{"type":"object","dependentSchemas":{"a":{"required":["y"]},"b":{"required":["z"]}},"additionalProperties":true}"#,
                &[
                    (br#"{"a":1,"b":2,"y":1,"z":2}"#, true),
                    (br#"{"a":1,"b":2,"y":1}"#, false),
                    (br#"{"a":1,"y":1}"#, true),
                ],
            ),
            (
                r#"{"type":"object","dependentSchemas":{"a":{"not":{"required":["z"]}}},"additionalProperties":true}"#,
                &[(br#"{"a":1}"#, true), (br#"{"a":1,"z":1}"#, false)],
            ),
        ];
        let mut document_total = 0usize;
        let mut prefix_total = 0usize;
        for (schema, documents) in cases {
            for &(document, valid) in *documents {
                document_total += 1;
                prefix_total += assert_combinator_prefixes(schema, document, valid);
            }
        }
        println!("dependentSchemas documents={document_total} byte_prefixes={prefix_total}");
    }

    /// Closing key `a` triggers a false dependent schema, so every completion fails.
    /// The reference continuation check evaluates this dependency only when the object closes.
    #[test]
    fn a_false_triggered_dependency_dies_before_reference_can_continue_notices() {
        let schema =
            r#"{"type":"object","dependentSchemas":{"a":false},"additionalProperties":true}"#;
        let schema_ir = ir(schema);
        let program = StructuredProgram::compile(schema_ir.clone()).expect("program");
        assert!(program.uses_incremental_backend());
        let mut matcher = program.new_matcher().expect("matcher");
        assert!(matcher.advance(br#"{"a"#).expect("key body still open"));
        let before = matcher.is_accepting();
        assert!(
            !matcher
                .advance(b"\"")
                .expect("closing quote fixes the doomed trigger"),
            "the closing quote that commits key `a` must be rejected outright"
        );
        assert_eq!(
            matcher.is_accepting(),
            before,
            "rejection must not mutate state"
        );

        let vocab_tail: &[&[u8]] = &[b":1}", b":1,\"z\":2}", b":null}", b":{}}", b":[1,2]}"];
        for &tail in vocab_tail {
            let mut completed = b"{\"a\"".to_vec();
            completed.extend_from_slice(tail);
            assert!(
                !super::super::reference::accepts(&schema_ir, &completed),
                "{completed:?} should never validate once trigger `a` is fixed"
            );
        }

        let mut record = program.new_matcher().expect("record matcher");
        assert!(record.advance(br#"{"a"#).expect("key body still open"));
        // The mask at this exact position must exclude the closing quote (it would fix the doomed
        // trigger) while a token that instead extends the key spelling remains a live continuation.
        let candidates: &[&[u8]] = &[b"\"", b"b", b"a"];
        let mut tokens = FxHashMap::default();
        for (id, bytes) in candidates.iter().enumerate() {
            tokens.insert(bytes.to_vec(), vec![u32::try_from(id).unwrap()]);
        }
        let handle = VocabularyHandle::new(Arc::new(
            build_vocabulary(u32::try_from(candidates.len()).unwrap(), tokens).unwrap(),
        ))
        .unwrap();
        let mask = record
            .allowed_mask_from_records(handle.mask_vocab_size(), handle.iter_records())
            .unwrap();
        for (id, &candidate) in candidates.iter().enumerate() {
            let allowed = mask.get(TokenId(u32::try_from(id).unwrap()));
            assert_eq!(
                allowed,
                candidate != b"\"",
                "candidate {candidate:?} mask bit"
            );
        }
    }

    #[test]
    fn dependent_schema_cycles_are_typed_errors_and_annotation_consumers_are_incremental() {
        for schema in [
            r##"{"$defs":{"x":{"type":"object","dependentSchemas":{"a":{"anyOf":[true,{"$ref":"#/$defs/x"}]}}}},"$ref":"#/$defs/x"}"##,
            r##"{"$defs":{"x":{"type":"object","dependentSchemas":{"a":{"not":{"$ref":"#/$defs/x"}}}}},"$ref":"#/$defs/x"}"##,
            r##"{"$defs":{"x":{"type":"object","dependentSchemas":{"a":{"$ref":"#/$defs/y"}}},"y":{"$ref":"#/$defs/x"}},"$ref":"#/$defs/x"}"##,
        ] {
            let error = match StructuredProgram::compile(ir(schema)) {
                Ok(_) => panic!("dependent recursion compiled: {schema}"),
                Err(error) => error,
            };
            assert_eq!(error.code, ErrorCode::NonMonotoneRecursion, "{schema}");
            assert!(!error.recoverable, "{schema}");
        }
        for schema in [
            r#"{"type":"object","dependentSchemas":{"a":{"properties":{"x":true}}},"unevaluatedProperties":false}"#,
            r#"{"dependentSchemas":{"a":{"items":true}},"unevaluatedItems":false}"#,
        ] {
            let compiled = StructuredProgram::compile(ir(schema))
                .unwrap_or_else(|error| panic!("{schema}: {error:?}"));
            assert_eq!(compiled.backend_kind(), "incremental", "{schema}");
        }
        let dependent_required = ir(r#"{"type":"object","dependentRequired":{"a":["b"]}}"#);
        assert_eq!(dependent_required.diagnostics().len(), 0);
        let program = StructuredProgram::compile(dependent_required).unwrap();
        assert!(program.uses_incremental_backend());
    }

    #[test]
    fn dependent_required_is_non_annotating_under_unevaluated_properties() {
        let only_presence =
            ir(r#"{"dependentRequired":{"a":["b"]},"unevaluatedProperties":false}"#);
        let program = StructuredProgram::compile(only_presence).unwrap();
        assert_eq!(program.backend_kind(), "incremental");
        let mut matcher = program.new_matcher().unwrap();
        assert!(!matcher.advance(br#"{"a":1,"b":2}"#).unwrap() || !matcher.is_accepting());

        let annotated = ir(r#"{
            "properties":{"a":true,"b":true},
            "dependentRequired":{"a":["b"]},
            "unevaluatedProperties":false
        }"#);
        let program = StructuredProgram::compile(annotated).unwrap();
        assert_eq!(program.backend_kind(), "incremental");
        let mut matcher = program.new_matcher().unwrap();
        assert!(matcher.advance(br#"{"a":1,"b":2}"#).unwrap());
        assert!(matcher.is_accepting());
    }

    #[test]
    fn dependent_required_combinations_remain_incremental_and_correct() {
        let cases = [
            (
                r#"{"type":"object","properties":{"a":true,"b":true},"dependentRequired":{"a":["b"]},"additionalProperties":false}"#,
                br#"{"a":1,"b":2}"#.as_slice(),
                true,
            ),
            (
                r#"{"type":"object","required":["a"],"dependentRequired":{"a":["b"]},"additionalProperties":true}"#,
                br#"{"a":1}"#,
                false,
            ),
            (
                r#"{"type":"object","patternProperties":{"^b$":true},"dependentRequired":{"a":["b"]},"additionalProperties":true}"#,
                br#"{"b":2,"a":1}"#,
                true,
            ),
            (
                r#"{"type":"object","propertyNames":{"minLength":1},"minProperties":1,"maxProperties":2,"dependentRequired":{"a":["b"]},"additionalProperties":true}"#,
                br#"{"a":1,"b":2}"#,
                true,
            ),
            (
                r#"{"type":"object","dependentSchemas":{"a":{"required":["c"]}},"dependentRequired":{"a":["b"]},"additionalProperties":true}"#,
                br#"{"a":1,"b":2,"c":3}"#,
                true,
            ),
            (
                r#"{"allOf":[{"dependentRequired":{"a":["b"]}},{"type":"object"}]}"#,
                br#"{"a":1}"#,
                false,
            ),
            (
                r#"{"anyOf":[{"dependentRequired":{"a":["b"]}},{"type":"array"}]}"#,
                br#"[]"#,
                true,
            ),
            (
                r#"{"oneOf":[{"dependentRequired":{"a":["b"]}},{"type":"array"}]}"#,
                br#"{"a":1,"b":2}"#,
                true,
            ),
            (
                r#"{"not":{"dependentRequired":{"a":["b"]}}}"#,
                br#"{"a":1}"#,
                true,
            ),
            (
                r#"{"if":{"required":["a"]},"then":{"dependentRequired":{"a":["b"]}},"else":true}"#,
                br#"{"a":1,"b":2}"#,
                true,
            ),
        ];
        for (schema, document, expected) in cases {
            let schema_ir = ir(schema);
            let program = StructuredProgram::compile(schema_ir.clone()).unwrap();
            assert!(program.uses_incremental_backend(), "{schema}");
            let mut matcher = program.new_matcher().unwrap();
            let actual = matcher.advance(document).unwrap() && matcher.is_accepting();
            assert_eq!(actual, expected, "{schema} / {document:?}");
            assert_eq!(
                actual,
                super::super::reference::accepts(&schema_ir, document)
            );
        }
    }

    #[test]
    fn dependent_required_hard_presence_graphs_nested_and_recursive_cases() {
        let cases = [
            (
                r#"{"dependentRequired":{"a":["shared"],"x":["shared"]}}"#,
                br#"{"a":1,"x":2,"shared":3}"#.as_slice(),
                true,
            ),
            (
                r#"{"dependentRequired":{"a":["b"],"b":["a"]}}"#,
                br#"{"a":1}"#,
                false,
            ),
            (
                r#"{"dependentRequired":{"a":["b"],"b":["c"]}}"#,
                br#"{"a":1,"b":2}"#,
                false,
            ),
            (
                r#"{"dependentRequired":{"a":["b"],"b":["c"]}}"#,
                br#"{"c":3,"b":2,"a":1}"#,
                true,
            ),
            (
                r#"{"type":"object","properties":{"nested":{"type":"object","dependentRequired":{"a":["b"]},"additionalProperties":true}},"additionalProperties":true}"#,
                br#"{"nested":{"a":1}}"#,
                false,
            ),
            (
                r#"{"type":"array","items":{"type":"object","dependentRequired":{"a":["b"]},"additionalProperties":true}}"#,
                br#"[{"a":1,"b":2},{"x":3}]"#,
                true,
            ),
            (
                r##"{"$defs":{"node":{"type":"object","dependentRequired":{"a":["b"]},"properties":{"next":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/node"}]}},"additionalProperties":true}},"$ref":"#/$defs/node"}"##,
                br#"{"a":1,"b":2,"next":{"a":3,"b":4,"next":null}}"#,
                true,
            ),
        ];
        for (schema, document, expected) in cases {
            let schema_ir = ir(schema);
            let program = StructuredProgram::compile(schema_ir.clone()).unwrap();
            assert!(program.uses_incremental_backend(), "{schema}");
            let mut matcher = program.new_matcher().unwrap();
            let actual = matcher.advance(document).unwrap() && matcher.is_accepting();
            assert_eq!(actual, expected, "{schema} / {document:?}");
            assert_eq!(
                actual,
                super::super::reference::accepts(&schema_ir, document)
            );
        }
    }

    #[test]
    fn dependent_required_remaining_keyword_routing_is_explicit() {
        for (schema, document, expected) in [
            (
                r#"{"type":"object","dependentRequired":{"a":["b"]},"additionalProperties":{"type":"integer"}}"#,
                br#"{"a":1,"b":2}"#.as_slice(),
                true,
            ),
            (
                r#"{"dependentRequired":{"a":["b"]},"unevaluatedItems":false}"#,
                br#"{"a":1,"b":2}"#,
                true,
            ),
        ] {
            let schema_ir = ir(schema);
            let program = StructuredProgram::compile(schema_ir.clone()).unwrap();
            assert!(program.uses_incremental_backend());
            let mut matcher = program.new_matcher().unwrap();
            let actual = matcher.advance(document).unwrap() && matcher.is_accepting();
            assert_eq!(actual, expected);
            assert_eq!(
                actual,
                super::super::reference::accepts(&schema_ir, document)
            );
        }
    }

    #[test]
    fn dependent_object_presence_record_preallocated_trie_and_reference_masks_are_identical() {
        for schema in [
            r#"{"type":"object","dependentSchemas":{"a":{"required":["b"]},"never":false},"additionalProperties":true}"#,
            r#"{"type":"object","dependentRequired":{"a":["b"]},"additionalProperties":true}"#,
        ] {
            let mut mask_comparisons = 0usize;
            let schema_ir = ir(schema);
            let program = StructuredProgram::compile(schema_ir.clone()).expect("program");
            let mut tokens = FxHashMap::default();
            for (id, bytes) in [
                (0, b"{".as_slice()),
                (1, b"\"a\""),
                (2, b"\"\\u0061\""),
                (3, b":"),
                (4, b"1"),
                (5, b","),
                (6, b"\"b\":"),
                (7, b"2}"),
                (8, b"}"),
                (9, b" \t"),
                (10, b"\"other\":1"),
                (11, b"1,"),
                (12, b"}\n,"),
                (13, b"\"a"),
                (14, b"\\u00"),
                (15, b"61\""),
            ] {
                tokens.insert(bytes.to_vec(), vec![id]);
            }
            let handle =
                VocabularyHandle::new(Arc::new(build_vocabulary(16, tokens).unwrap())).unwrap();
            let trie = TrieCache::new().bind(&handle).unwrap();
            let mut record = program.new_matcher().unwrap();
            let mut walked = program.new_matcher().unwrap();
            let mut prefix = Vec::new();
            for token in [
                b"{".as_slice(),
                b"\"b\":",
                b"2",
                b",",
                b"\"\\u0061\"",
                b":",
                b"1",
                b"}",
            ] {
                let allocating = record
                    .allowed_mask_from_records(handle.mask_vocab_size(), handle.iter_records())
                    .unwrap();
                let mut preallocated = [0xff; 4];
                record
                    .write_record_mask_le_bytes_into(
                        handle.mask_vocab_size(),
                        handle.iter_records(),
                        &mut preallocated,
                    )
                    .unwrap();
                let mut trie_output = [0xff; 4];
                walked
                    .write_mask_le_bytes_into(&handle, Some(&trie), &mut trie_output)
                    .unwrap();
                for (bytes, ids) in handle.iter_records() {
                    let mut candidate = prefix.clone();
                    candidate.extend_from_slice(bytes);
                    let oracle = super::super::reference::accepts(&schema_ir, &candidate)
                        || super::super::reference::can_continue(&schema_ir, &candidate);
                    for &id in ids {
                        let id = usize::try_from(id).unwrap();
                        assert_eq!(
                            allocating.get(TokenId(id as u32)),
                            oracle,
                            "{prefix:?} + {bytes:?}"
                        );
                        assert_eq!(preallocated[id / 8] & (1 << (id % 8)) != 0, oracle);
                        assert_eq!(trie_output[id / 8] & (1 << (id % 8)) != 0, oracle);
                        mask_comparisons += 3;
                    }
                }
                assert!(record.advance(token).unwrap(), "{token:?}");
                assert!(walked.advance(token).unwrap(), "{token:?}");
                prefix.extend_from_slice(token);
            }
            assert!(record.is_accepting() && walked.is_accepting());
            println!("dependent-object token-mask comparisons={mask_comparisons} schema={schema}");
        }
    }

    #[test]
    fn rejected_candidate_rolls_back_incremental_state() {
        let program = program(
            r#"{"type":"object","required":["x"],"properties":{"x":{"type":"integer"}},"additionalProperties":false}"#,
        );
        let mut matcher = program.new_matcher().expect("state");
        assert!(matcher.advance(br#"{"#).expect("advance"));
        let before = matcher.is_accepting();
        assert!(!matcher.advance(b"}").expect("advance"));
        assert_eq!(matcher.is_accepting(), before);
        assert!(matcher.advance(b"\"x\"").expect("advance"));
    }

    #[test]
    fn mask_evaluation_rolls_back_and_preserves_duplicate_ids() {
        let program = program(
            r#"{"type":"object","properties":{"x":{"type":"boolean"}},"required":["x"],"additionalProperties":false}"#,
        );
        let mut matcher = program.new_matcher().expect("state");
        assert!(matcher.advance(br#"{"x":"#).expect("advance"));
        let records = [
            (b"true".as_slice(), [3u32, 7].as_slice()),
            (b"0".as_slice(), [9u32].as_slice()),
        ];
        let mask = matcher
            .allowed_mask_from_records(10, records.into_iter())
            .expect("mask");
        assert!(mask.get(TokenId(3)) && mask.get(TokenId(7)));
        assert!(!mask.get(TokenId(9)));
        assert!(matcher.advance(b"true").expect("advance"));
    }

    #[test]
    fn committed_object_keys_remain_unique() {
        let schema = r#"{"type":"object","properties":{"name":{"type":"boolean"}},"required":["name"],"additionalProperties":{"type":"boolean"}}"#;
        let ir = Arc::new(schema_to_ir(schema, CompileOptions::default()).expect("schema"));
        let program = StructuredProgram::compile(ir).expect("program");
        let mut matcher = program.new_matcher().expect("state");
        for token in [b"{".as_slice(), b"\"name\"", b":", b"true", b","] {
            assert!(matcher.advance(token).expect("advance"));
        }
        assert!(!matcher.advance(b"\"name\"").expect("duplicate"));
    }

    #[test]
    fn supported_structured_shapes_select_incremental_backend() {
        let schemas = [
            r#"{"type":"object","properties":{"x":{"type":"integer"}},"additionalProperties":{"type":"string"}}"#,
            r#"{"type":"object","patternProperties":{"^x+$":{"type":"integer"}},"additionalProperties":false}"#,
            r#"{"type":"object","propertyNames":{"type":"string","pattern":"^a+$"},"minProperties":1,"maxProperties":2,"additionalProperties":true}"#,
            r#"{"type":"array","items":{"type":"integer"},"contains":{"type":"integer","minimum":1},"minContains":1,"maxContains":2,"uniqueItems":true}"#,
            r#"{"type":"object","properties":{"\u0061":{"type":"array","items":{"type":"object","properties":{"n":{"type":"number"}},"required":["n"],"additionalProperties":false},"uniqueItems":true}},"additionalProperties":false}"#,
        ];
        for schema in schemas {
            assert_eq!(program(schema).backend_kind(), "incremental", "{schema}");
        }
    }

    #[test]
    fn plan_limit_is_an_error_not_a_reference_fallback() {
        let limits = StructuredLimits {
            max_plan_bytes: 0,
            ..StructuredLimits::default()
        };
        let result = StructuredProgram::compile_with_limits(
            ir(r#"{"type":"string","pattern":"^a+$"}"#),
            limits,
        );
        let error = match result {
            Ok(_) => panic!("zero plan budget compiled"),
            Err(error) => error,
        };
        assert_eq!(error.code, ErrorCode::InternalLimitExceeded);
        assert!(!error.recoverable);
    }

    #[test]
    fn incremental_program_has_no_fallback_reason() {
        let program = program(r#"{"type":"string"}"#);
        assert!(program.uses_incremental_backend());
        assert_eq!(program.fallback_reason(), None);
    }

    #[test]
    fn every_supported_node_shape_has_an_explicit_runtime_route() {
        let large_enum = large_string_enum_schema();
        let schemas = [
            r#"{"type":"null"}"#.to_owned(),
            r#"{"type":"boolean"}"#.to_owned(),
            "false".to_owned(),
            r#"{"const":"x"}"#.to_owned(),
            r#"{"type":"string","pattern":"^x+$"}"#.to_owned(),
            r#"{"type":"integer"}"#.to_owned(),
            r#"{"type":"number"}"#.to_owned(),
            r#"{"enum":["a","b"]}"#.to_owned(),
            large_enum,
            r#"{"type":"array","items":{"type":"integer"},"maxItems":100}"#.to_owned(),
            r#"{"prefixItems":[{"type":"integer"}],"items":false}"#.to_owned(),
            r#"{"prefixItems":[{"type":"integer"}],"items":true}"#.to_owned(),
            r#"{"type":"array","items":{"type":"object"},"uniqueItems":true,"contains":{"properties":{"k":{"const":1}}}}"#.to_owned(),
            r#"{"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":false}"#.to_owned(),
            r#"{"type":"object","patternProperties":{"^x$":{"type":"integer"}},"additionalProperties":true}"#.to_owned(),
            r#"{"anyOf":[{"type":"null"},{"type":"boolean"}]}"#.to_owned(),
            r#"{"allOf":[{"type":"integer"},{"minimum":0}]}"#.to_owned(),
            r#"{"oneOf":[{"type":"null"},{"type":"boolean"}]}"#.to_owned(),
            r#"{"not":{"type":"null"}}"#.to_owned(),
            r##"{"$defs":{"x":{"type":"string"}},"$ref":"#/$defs/x"}"##.to_owned(),
            r#"{"properties":{"a":true},"unevaluatedProperties":false}"#.to_owned(),
        ];
        for schema in schemas {
            let program = program(&schema);
            assert_ne!(
                program.fallback_reason(),
                Some("unsupported-node-shape"),
                "{schema}"
            );
            assert!(program.uses_incremental_backend(), "{schema}");
        }
    }

    #[test]
    fn former_fallbacks_are_incremental_or_typed_errors() {
        let mut builder = crate::ir::Builder::new(CompileOptions::default());
        let lower = builder
            .string_pattern("a+", Some(5), None, Charset::Utf8Any)
            .expect("lower");
        let upper = builder
            .string_pattern("a+", None, Some(2), Charset::Utf8Any)
            .expect("upper");
        let root = builder
            .intersection_of(vec![lower, upper])
            .expect("intersection");
        let intersection = StructuredProgram::compile(Arc::new(builder.finish(root).expect("ir")))
            .expect("program");
        assert!(intersection.uses_incremental_backend());
        let mut matcher = intersection.new_matcher().expect("matcher");
        assert!(!matcher.advance(br#""aaaaa""#).expect("document"));

        let mut builder = crate::ir::Builder::new(CompileOptions::default());
        let value = builder.string_const("x").expect("value");
        let fields = (0..5).map(|index| (format!("p{index}"), value)).collect();
        let root = builder
            .object(
                fields,
                &[false; 5],
                crate::ir::ObjectClosure::RejectOpenObjects,
            )
            .expect("object");
        let error = match StructuredProgram::compile(Arc::new(builder.finish(root).expect("ir"))) {
            Ok(_) => panic!("RejectOpenObjects compiled"),
            Err(error) => error,
        };
        assert_eq!(error.code, ErrorCode::Unsupported);
        assert!(!error.recoverable);
    }

    #[test]
    fn resource_errors_roll_back_token_and_mask_state() {
        let limits = StructuredLimits {
            max_document_bytes: 1,
            max_mask_work: 1,
            ..StructuredLimits::default()
        };
        let program = StructuredProgram::compile_with_limits(ir(r#"{"type":"string"}"#), limits)
            .expect("program");
        let mut matcher = program.new_matcher().expect("matcher");
        let error = matcher.advance(b"\"a").expect_err("document cap");
        assert!(matches!(
            error,
            StructuredMatcherError::ResourceLimit {
                kind: LimitKind::DocumentBytes,
                observed: 2,
                limit: 1,
            }
        ));
        assert!(!matcher.is_dead());

        let records = [(b"\"".to_vec(), vec![1u32, 2])];
        let error = matcher
            .allowed_mask_from_records(
                3,
                records
                    .iter()
                    .map(|(bytes, ids)| (bytes.as_slice(), ids.as_slice())),
            )
            .expect_err("mask work cap");
        assert!(matches!(
            error,
            StructuredMatcherError::ResourceLimit {
                kind: LimitKind::MaskWork,
                ..
            }
        ));
        assert!(matcher.advance(b"\"").expect("rollback kept empty state"));
    }

    #[test]
    fn masks_are_order_independent_and_match_the_reference_oracle() {
        let schema = r#"{"type":"object","properties":{"n":{"type":"number","minimum":-1,"maximum":1,"multipleOf":0.5},"s":{"type":"string","pattern":"^a+$","minLength":1,"maxLength":2}},"required":["n","s"],"additionalProperties":false}"#;
        let ir = ir(schema);
        let program = StructuredProgram::compile(ir.clone()).expect("program");
        assert_eq!(program.backend_kind(), "incremental");
        let prefix = br#"{"n":1,"s":"a"#;
        let records = vec![
            (b"\"}".to_vec(), vec![0, 6]),
            (b"a\"}".to_vec(), vec![1]),
            (b"aa\"}".to_vec(), vec![2]),
            (b"b\"}".to_vec(), vec![3]),
        ];
        let mut forward = program.new_matcher().expect("matcher");
        let mut reverse = program.new_matcher().expect("matcher");
        for &byte in prefix {
            assert!(forward.advance(&[byte]).expect("prefix"));
            assert!(reverse.advance(&[byte]).expect("prefix"));
        }
        let forward_mask = mask(&mut forward, 7, &records);
        let mut reversed = records.clone();
        reversed.reverse();
        let reverse_mask = mask(&mut reverse, 7, &reversed);
        assert_eq!(forward_mask, reverse_mask);

        for (bytes, ids) in &records {
            let mut candidate = prefix.to_vec();
            candidate.extend_from_slice(bytes);
            let expected = super::super::reference::accepts(&ir, &candidate)
                || super::super::reference::can_continue(&ir, &candidate);
            for &id in ids {
                assert_eq!(forward_mask.get(TokenId(id)), expected, "{bytes:?}");
            }
        }
        assert!(forward.advance(b"\"}").expect("mask rolled back"));
        assert!(forward.is_accepting());
    }

    #[test]
    fn shared_program_matchers_are_independent() {
        let program = program(r#"{"type":"array","items":{"type":"integer"},"uniqueItems":true}"#);
        let mut first = program.new_matcher().expect("first");
        let second = program.new_matcher().expect("second");
        assert!(first.advance(b"[").expect("advance"));
        assert!(!first.is_dead());
        assert!(!second.is_accepting());
        assert!(!second.is_dead());
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn mask_work_metrics_count_actual_walk_and_reset_per_operation() {
        let program = program(r#"{"type":"string","pattern":"^a+$"}"#);
        let mut tokens = FxHashMap::default();
        for (id, bytes) in [
            (0u32, b"a".as_slice()),
            (1, b"aa"),
            (2, b"b"),
            (3, b"\""),
            (4, b"\\"),
        ] {
            tokens.insert(bytes.to_vec(), vec![id]);
        }
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(5, tokens).expect("vocabulary")))
                .expect("handle");
        let trie = TrieCache::new().bind(&vocabulary).expect("trie");
        let mut matcher = program.new_matcher().expect("matcher");
        assert!(matcher.advance(b"\"").expect("open string"));
        let mut output = vec![0u8; vocabulary.mask_vocab_size().div_ceil(32) * 4];

        matcher
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
            .expect("first mask");
        let first = matcher.mask_work_metrics_for_bench();
        assert!(first.trie_nodes > 0);
        assert!(first.trie_edges > 0);
        assert_eq!(first.pushed_bytes, first.trie_edges);
        assert_eq!(first.checkpoints, first.trie_nodes.saturating_add(1));
        assert_eq!(first.rollbacks, first.checkpoints);

        matcher
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
            .expect("second mask");
        assert_eq!(matcher.mask_work_metrics_for_bench(), first);
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn mask_work_metrics_are_isolated_between_matchers() {
        let program = program(
            r##"{"$dynamicAnchor":"node","type":"object","properties":{"child":{"$dynamicRef":"#node"}}}"##,
        );
        let mut tokens = FxHashMap::default();
        tokens.insert(b"{".to_vec(), vec![0]);
        tokens.insert(b"}".to_vec(), vec![1]);
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(2, tokens).expect("vocabulary")))
                .expect("handle");
        let trie = TrieCache::new().bind(&vocabulary).expect("trie");
        let mut first = program.new_matcher().expect("first");
        let second = program.new_matcher().expect("second");
        assert!(first.advance(br#"{"child":"#).expect("child prefix"));
        let mut output = vec![0u8; vocabulary.mask_vocab_size().div_ceil(32) * 4];

        first
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
            .expect("dynamic mask");
        let first_metrics = first.mask_work_metrics_for_bench();
        assert!(first_metrics.dynamic_resolutions > 0);
        assert!(first_metrics.dynamic_scope_steps > 0);
        first
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
            .expect("warm dynamic mask");
        let warm_metrics = first.mask_work_metrics_for_bench();
        assert!(warm_metrics.dynamic_resolutions > 0);
        assert_eq!(warm_metrics.dynamic_scope_steps, 0);
        assert_eq!(
            second.mask_work_metrics_for_bench(),
            MaskWorkMetrics::default()
        );
    }

    /// Every byte prefix of a valid document vs the reference oracle: exact completeness agreement,
    /// no over-acceptance, and the real next byte always allowed (no under-acceptance on valid input).
    #[test]
    fn incremental_matcher_agrees_with_the_reference_oracle_across_every_prefix() {
        let cases: &[(&str, &[u8])] = &[
            (
                r#"{"type":"object","additionalProperties":{"type":"object","additionalProperties":{"type":"boolean"}}}"#,
                br#"{"a":{"b":true},"c":{"d":false}}"#,
            ),
            (
                r#"{"type":"object","patternProperties":{"^x":{"type":"integer"}},"additionalProperties":{"type":"string"}}"#,
                br#"{"x1":5,"other":"ok"}"#,
            ),
            (
                r#"{"type":"object","propertyNames":{"type":"string","pattern":"^a$"},"additionalProperties":true}"#,
                br#"{"a":1}"#,
            ),
            (
                r#"{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":4}"#,
                br#"[1,22,333]"#,
            ),
            (r#"{"type":"string","pattern":"^a.c$"}"#, br#""abc""#),
            (
                r#"{"type":"object","minProperties":1,"maxProperties":2,"additionalProperties":{"type":"boolean"}}"#,
                br#"{"p":true,"q":false}"#,
            ),
            (
                r#"{"type":"object","dependentSchemas":{"a":{"type":"object","required":["b"],"additionalProperties":true}},"additionalProperties":true}"#,
                br#"{"b":2,"\u0061":1}"#,
            ),
            (
                r#"{"type":"object","dependentRequired":{"a":["b"]},"additionalProperties":true}"#,
                br#"{"b":2,"\u0061":1}"#,
            ),
            (
                r#"{"dependentSchemas":{"a":false}}"#,
                br#"[null,true,{"a":1}]"#,
            ),
        ];
        let alphabet: Vec<u8> = (0u8..=127).collect();
        let mut prefix_count = 0usize;
        let mut candidate_byte_count = 0usize;
        for (schema, doc) in cases {
            let ir = ir(schema);
            let program = StructuredProgram::compile(ir.clone()).expect("program");
            assert_eq!(
                program.backend_kind(),
                "incremental",
                "{schema} must be incremental"
            );
            let records: Vec<(Vec<u8>, Vec<u32>)> = alphabet
                .iter()
                .enumerate()
                .map(|(i, &b)| (vec![b], vec![i as u32]))
                .collect();
            let mut m = program.new_matcher().expect("matcher");
            for end in 0..doc.len() {
                prefix_count += 1;
                candidate_byte_count += alphabet.len();
                let prefix = &doc[..end];
                assert_eq!(
                    m.is_accepting(),
                    super::super::reference::accepts(&ir, prefix),
                    "is_accepting diverged at {prefix:?} for {schema}"
                );
                let msk = mask(&mut m, alphabet.len(), &records);
                for (i, &b) in alphabet.iter().enumerate() {
                    if msk.get(TokenId(i as u32)) {
                        let mut cand = prefix.to_vec();
                        cand.push(b);
                        let oracle = super::super::reference::accepts(&ir, &cand)
                            || super::super::reference::can_continue(&ir, &cand);
                        assert!(
                            oracle,
                            "over-acceptance: byte {b:#04x} allowed at {prefix:?} but the oracle forbids it, {schema}"
                        );
                    }
                }
                let next = doc[end];
                assert!(
                    msk.get(TokenId(usize::from(next) as u32)),
                    "under-acceptance: valid next byte {next:#04x} masked out at {prefix:?}, {schema}"
                );
                assert!(
                    m.advance(&[next]).expect("advance"),
                    "valid byte {next:#04x} rejected at {prefix:?}, {schema}"
                );
            }
            assert!(
                m.is_accepting(),
                "complete valid document not accepting: {schema}"
            );
        }
        println!(
            "every-prefix differential: prefixes={prefix_count} candidate_bytes={candidate_byte_count}"
        );
    }

    #[test]
    fn uniqueitems_speculative_mask_must_not_crash() {
        let program = program(
            r#"{"type":"array","items":{"type":"number"},"uniqueItems":true,"contains":{"const":1}}"#,
        );
        let mut m = program.new_matcher().expect("matcher");
        assert!(m.advance(b"[").expect("advance"));
        assert!(m.advance(b"1").expect("advance"));
        let mut records: Vec<(Vec<u8>, Vec<u32>)> = (0u8..=127)
            .map(|byte| (vec![byte], vec![u32::from(byte)]))
            .collect();
        records.push((b" ]".to_vec(), vec![127]));
        for _ in 0..100 {
            m.allowed_mask_from_records(
                128,
                records
                    .iter()
                    .map(|(bytes, ids)| (bytes.as_slice(), ids.as_slice())),
            )
            .expect("speculative mask");
        }
        assert!(!m.advance(b"x").expect("invalid token"));
        assert!(m.advance(b"\t").expect("whitespace"));
        assert!(m.advance(b"]").expect("close"));
        assert!(m.is_accepting());
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn duplicate_finite_string_item_is_rejected_at_its_first_impossible_byte() {
        let base_program = program(
            r#"{"type":"array","items":{"enum":["treasury","ops","risk"]},"contains":{"const":"treasury"},"uniqueItems":true,"minItems":1,"maxItems":3}"#,
        );
        let tokens = (0u8..=255)
            .map(|byte| (vec![byte], vec![u32::from(byte)]))
            .collect();
        let vocabulary = VocabularyHandle::new(Arc::new(
            build_vocabulary(256, tokens).expect("byte vocabulary"),
        ))
        .expect("vocabulary handle");
        let trie = TrieCache::new().bind(&vocabulary).expect("trie");

        for (name, second_prefix, rejected) in [
            ("raw", b"\"".as_slice(), b't'),
            ("escaped", b"\"\\u007".as_slice(), b'4'),
        ] {
            let mut prefix = b"[\"treasury\",".to_vec();
            prefix.extend_from_slice(second_prefix);
            let (mask, _) = assert_three_route_mask_and_advance(
                &base_program,
                &vocabulary,
                &trie,
                &prefix,
                name,
            );
            assert_eq!(
                mask[usize::from(rejected) / 8] & (1 << (rejected % 8)),
                0,
                "{name}: first byte completing the duplicate must be masked"
            );

            let mut direct = base_program.new_matcher().expect("direct matcher");
            assert!(direct.advance(&prefix).expect("prefix"));
            assert!(!direct.advance(&[rejected]).expect("duplicate byte"));
            assert!(!direct.is_dead(), "rejection must preserve committed state");
        }

        let distinct = b"[\"treasury\",\"risk";
        let (mask, _) = assert_three_route_mask_and_advance(
            &base_program,
            &vocabulary,
            &trie,
            distinct,
            "distinct",
        );
        assert_ne!(
            mask[usize::from(b'"') / 8] & (1 << (b'"' % 8)),
            0,
            "distinct item must remain closable"
        );

        let nested = program(
            r#"{"type":"object","properties":{"approvals":{"type":"array","items":{"enum":["treasury","ops","risk"]},"contains":{"const":"treasury"},"uniqueItems":true,"minItems":1,"maxItems":3}},"required":["approvals"],"unevaluatedProperties":false}"#,
        );
        let nested_prefix = b"{\"approvals\":[\"tr\\u0065\\u0061\\u0073\\u0075\\u0072\\u0079\",\"";
        let (mask, _) = assert_three_route_mask_and_advance(
            &nested,
            &vocabulary,
            &trie,
            nested_prefix,
            "nested-payment-shape",
        );
        assert_eq!(
            mask[usize::from(b't') / 8] & (1 << (b't' % 8)),
            0,
            "nested payment-shaped array must reject the duplicate at t"
        );

        let escape_program =
            program(r#"{"type":"array","items":{"enum":["tram","tree"]},"uniqueItems":true}"#);
        let escape_prefix = b"[\"tree\",\"tr\\u006";
        let (mask, _) = assert_three_route_mask_and_advance(
            &escape_program,
            &vocabulary,
            &trie,
            escape_prefix,
            "pending-enum-escape",
        );
        assert_eq!(
            mask[usize::from(b'5') / 8] & (1 << (b'5' % 8)),
            0,
            "escape resolving only to the seen `tree` must be rejected"
        );
        assert_ne!(
            mask[usize::from(b'1') / 8] & (1 << (b'1' % 8)),
            0,
            "escape resolving to unseen `tram` must remain available"
        );

        let completed_member = b"[\"treasury";
        let (mask, _) = assert_three_route_mask_and_advance(
            &base_program,
            &vocabulary,
            &trie,
            completed_member,
            "completed-enum-member",
        );
        assert_ne!(
            mask[usize::from(b'"') / 8] & (1 << (b'"' % 8)),
            0,
            "a complete first item must retain its closing quote"
        );
        assert_eq!(
            mask[usize::from(b'\\') / 8] & (1 << (b'\\' % 8)),
            0,
            "an enum member with no longer completion must reject a new escape"
        );

        let before_third_item = b"[\"treasury\",\"risk\",";
        let (mask, _) = assert_three_route_mask_and_advance(
            &base_program,
            &vocabulary,
            &trie,
            before_third_item,
            "before-third-enum-item",
        );
        assert_ne!(
            mask[usize::from(b'"') / 8] & (1 << (b'"' % 8)),
            0,
            "an unseen member must permit the next opening quote"
        );

        let third_item = b"[\"treasury\",\"risk\",\"";
        let (mask, _) = assert_three_route_mask_and_advance(
            &base_program,
            &vocabulary,
            &trie,
            third_item,
            "third-enum-item",
        );
        assert_eq!(
            mask[usize::from(b'!') / 8] & (1 << (b'!' % 8)),
            0,
            "a third enum item must reject a byte outside every remaining enum member"
        );
        assert_ne!(
            mask[usize::from(b'o') / 8] & (1 << (b'o' % 8)),
            0,
            "a third enum item must retain the unseen `ops` prefix"
        );
        for byte in [b'r', b't'] {
            assert_eq!(
                mask[usize::from(byte) / 8] & (1 << (byte % 8)),
                0,
                "a third enum item must reject seen member prefix {byte:?}"
            );
        }

        let exhausted = program(
            r#"{"type":"array","items":{"enum":["treasury","ops","risk"]},"uniqueItems":true,"maxItems":4}"#,
        );
        let mut exhausted_matcher = exhausted.new_matcher().expect("exhausted matcher");
        assert!(exhausted_matcher
            .advance(br#"["treasury","risk","ops""#)
            .unwrap());
        assert!(!exhausted_matcher.advance(b",").unwrap());
        assert!(!exhausted_matcher.is_dead());
        assert!(exhausted_matcher.advance(b"]").unwrap());
        assert!(exhausted_matcher.is_accepting());

        let wrapped = program(
            r#"{
                "type":"object",
                "properties":{
                    "rail":{"enum":["ach","wire","internal"]},
                    "approvals":{
                        "type":"array",
                        "items":{"enum":["treasury","ops","risk"]},
                        "contains":{"const":"treasury"},
                        "uniqueItems":true,
                        "minItems":1,
                        "maxItems":3
                    }
                },
                "required":["approvals"],
                "if":{"properties":{"rail":{"const":"wire"}},"required":["rail"]},
                "then":{"required":["rail"]},
                "else":{"not":{"required":["rail"]}},
                "allOf":[
                    {"if":{"properties":{"rail":{"const":"internal"}},"required":["rail"]},"then":{"required":["rail"]}},
                    {"if":{"properties":{"rail":{"const":"ach"}},"required":["rail"]},"then":{"required":["rail"]}}
                ],
                "unevaluatedProperties":false
            }"#,
        );
        for (spelling, treasury, risk) in [
            ("raw-raw", b"treasury".as_slice(), b"risk".as_slice()),
            (
                "escaped-raw",
                br"tr\u0065\u0061\u0073\u0075\u0072\u0079".as_slice(),
                b"risk".as_slice(),
            ),
            (
                "raw-escaped",
                b"treasury".as_slice(),
                br"r\u0069\u0073\u006B".as_slice(),
            ),
            (
                "escaped-escaped",
                br"tr\u0065\u0061\u0073\u0075\u0072\u0079".as_slice(),
                br"r\u0069\u0073\u006B".as_slice(),
            ),
        ] {
            for (scope, case, head, tail) in [
                (
                    "direct",
                    &base_program,
                    b"[\"".as_slice(),
                    br#"ops"]"#.as_slice(),
                ),
                (
                    "wrapper",
                    &wrapped,
                    b"{\"approvals\":[\"".as_slice(),
                    br#"ops"]}"#.as_slice(),
                ),
            ] {
                let label = format!("{scope}-{spelling}");
                let mut prefix = head.to_vec();
                prefix.extend_from_slice(treasury);
                prefix.extend_from_slice(b"\",\"");
                prefix.extend_from_slice(risk);
                prefix.extend_from_slice(b"\",\"");
                let (mask, _) =
                    assert_three_route_mask_and_advance(case, &vocabulary, &trie, &prefix, &label);
                let bit = |byte: u8| mask[usize::from(byte) / 8] & (1 << (byte % 8)) != 0;
                assert!(bit(b'o'), "{label}: `ops` must remain unseen");
                for rejected in [b't', b'r', b'!'] {
                    assert!(!bit(rejected), "{label}: rejected byte {rejected:?}");
                }

                let mut witness = prefix;
                witness.extend_from_slice(tail);
                let mut one_call = case.new_matcher().expect("one-call matcher");
                assert!(one_call.advance(&witness).unwrap(), "{label}: witness");
                assert!(one_call.is_accepting(), "{label}: accepting");

                let mut bytewise = case.new_matcher().expect("bytewise matcher");
                for (index, &byte) in witness.iter().enumerate() {
                    assert!(
                        bytewise.advance(&[byte]).unwrap(),
                        "{label}: bytewise witness rejected {byte:?} at {index}"
                    );
                }
                assert!(bytewise.is_accepting(), "{label}: bytewise accepting");

                // The release gate exercises the expensive mask-before-every-byte schedule. The
                // debug-focused test keeps the same logical matrix without multiplying runtime.
                #[cfg(not(debug_assertions))]
                {
                    let mut masked = case.new_matcher().expect("masked byte matcher");
                    for (index, &byte) in witness.iter().enumerate() {
                        let mut next = vec![0xa5; vocabulary.mask_vocab_size().div_ceil(32) * 4];
                        masked
                            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut next)
                            .unwrap();
                        assert!(
                            next[usize::from(byte) / 8] & (1 << (byte % 8)) != 0,
                            "{label}: witness byte {byte:?} masked at {index}"
                        );
                        assert!(masked.advance(&[byte]).unwrap());
                    }
                    assert!(masked.is_accepting(), "{label}: masked replay accepting");
                }
            }
        }
        let wrapped_prefix = b"{\"approvals\":[\"treasury\",\"risk\",\"";
        let (mask, _) = assert_three_route_mask_and_advance(
            &wrapped,
            &vocabulary,
            &trie,
            wrapped_prefix,
            "wrapped-third-enum-item",
        );
        assert_ne!(
            mask[usize::from(b'o') / 8] & (1 << (b'o' % 8)),
            0,
            "wrappers must retain the remaining unseen `ops` member"
        );
        assert_eq!(
            mask[usize::from(b'!') / 8] & (1 << (b'!' % 8)),
            0,
            "wrappers must reject a prefix outside the remaining enum language"
        );
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn payment_approval_enum_retains_unseen_member_after_escaped_items() {
        let payment = program(
            r##"{
                "$schema":"https://json-schema.org/draft/2020-12/schema",
                "$id":"https://example.test/payment-instruction",
                "type":"object",
                "unevaluatedProperties":false,
                "required":["rail","amountMinor","currency","approvals","allocation"],
                "properties":{
                    "rail":{"enum":["ach","wire","internal"]},
                    "amountMinor":{"type":"integer","multipleOf":100,"minimum":100,"maximum":900},
                    "currency":{"enum":["USD","EUR","RUB"],"not":{"const":"RUB"}},
                    "routingNumber":{"type":"string","pattern":"^[0-9]{4}$"},
                    "iban":{"type":"string","pattern":"^[A-Z]{2}[0-9]{4}$"},
                    "accountRef":{"type":"string","pattern":"^acct-[0-9]{2}$"},
                    "fxRate":{"type":"integer","minimum":1,"maximum":9},
                    "settlementDate":{"type":"string","pattern":"^20[0-9]{2}$"},
                    "approvals":{
                        "type":"array",
                        "items":{"enum":["treasury","ops","risk"]},
                        "contains":{"const":"treasury"},
                        "uniqueItems":true,
                        "minItems":1,
                        "maxItems":3
                    },
                    "allocation":{"$ref":"#/$defs/split"},
                    "tags":{
                        "type":"object",
                        "propertyNames":{"pattern":"^[a-z]{2,6}$"},
                        "patternProperties":{"^[a-z]{2,6}$":{"type":"string","maxLength":6}},
                        "additionalProperties":false,
                        "maxProperties":2
                    }
                },
                "dependentRequired":{"fxRate":["settlementDate"]},
                "if":{"properties":{"rail":{"const":"wire"}},"required":["rail"]},
                "then":{"required":["iban"]},
                "else":{"not":{"required":["iban"]}},
                "allOf":[
                    {"if":{"properties":{"rail":{"const":"internal"}},"required":["rail"]},"then":{"required":["accountRef"]}},
                    {"if":{"properties":{"rail":{"const":"ach"}},"required":["rail"]},"then":{"required":["routingNumber"]}}
                ],
                "$defs":{
                    "split":{
                        "$dynamicAnchor":"split",
                        "type":"object",
                        "additionalProperties":false,
                        "required":["pct"],
                        "properties":{
                            "pct":{"type":"integer","minimum":1,"maximum":99},
                            "sub":{"$dynamicRef":"#split"}
                        }
                    }
                }
            }"##,
        );
        let records = (0u8..=255)
            .map(|byte| (vec![byte], vec![u32::from(byte)]))
            .collect();
        let vocabulary = VocabularyHandle::new(Arc::new(
            build_vocabulary(256, records).expect("byte vocabulary"),
        ))
        .expect("vocabulary handle");
        let trie = TrieCache::new().bind(&vocabulary).expect("trie");
        let document_head = b"{\n\n\"amountMinor\":100,\n\n\"currency\":\"USD\",\n\n\"accountRef\":\"acct-11\",\n\n\"fxRate\":1,\n\n\"routingNumber\":\"1123\",\n\n\"rail\":\"ach\",\n\n\"settlementDate\":\"2016\",\n\n\"tags\":{\n\n\"amount\":\"USD\",\n\n\"routin\":\"acct-1\"\n\n},\n\n\"approvals\":[\n\n\"";

        for (label, treasury, risk) in [
            ("raw-raw", b"treasury".as_slice(), b"risk".as_slice()),
            (
                "escaped-raw",
                br"tr\u0065\u0061\u0073\u0075\u0072\u0079".as_slice(),
                b"risk".as_slice(),
            ),
            (
                "raw-escaped",
                b"treasury".as_slice(),
                br"r\u0069\u0073\u006B".as_slice(),
            ),
            (
                "escaped-escaped",
                br"tr\u0065\u0061\u0073\u0075\u0072\u0079".as_slice(),
                br"r\u0069\u0073\u006B".as_slice(),
            ),
        ] {
            let mut prefix = document_head.to_vec();
            prefix.extend_from_slice(treasury);
            prefix.extend_from_slice(b"\",\"");
            prefix.extend_from_slice(risk);
            prefix.extend_from_slice(b"\",\"");
            let (mask, _) =
                assert_three_route_mask_and_advance(&payment, &vocabulary, &trie, &prefix, label);
            let bit = |byte: u8| mask[usize::from(byte) / 8] & (1 << (byte % 8)) != 0;
            assert!(bit(b'o'), "{label}: unseen `ops` must remain reachable");
            for rejected in [b't', b'r', b'!'] {
                assert!(!bit(rejected), "{label}: rejected byte {rejected:?}");
            }

            let mut bytewise = payment.new_matcher().expect("bytewise payment matcher");
            for (index, &byte) in prefix.iter().enumerate() {
                assert!(
                    bytewise.advance(&[byte]).unwrap(),
                    "{label}: bytewise prefix rejected index {index}, byte {byte:?}, after {}",
                    String::from_utf8_lossy(&prefix[..index])
                );
            }

            let mut witness = prefix;
            witness.extend_from_slice(br#"ops"],"allocation":{"pct":60}}"#);
            let mut matcher = payment.new_matcher().expect("payment matcher");
            assert!(matcher.advance(&witness).unwrap(), "{label}: witness");
            assert!(matcher.is_accepting(), "{label}: final payment");
        }
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn pending_scalar_bytes_preserve_pattern_liveness() {
        let records = (0u8..=255)
            .map(|byte| (vec![byte], vec![u32::from(byte)]))
            .collect();
        let vocabulary = VocabularyHandle::new(Arc::new(
            build_vocabulary(256, records).expect("byte vocabulary"),
        ))
        .expect("vocabulary handle");
        let trie = TrieCache::new().bind(&vocabulary).expect("trie");
        let bit = |mask: &[u8], byte: u8| mask[usize::from(byte) / 8] & (1 << (byte % 8)) != 0;
        let digits = *b"0123456789abcdefABCDEF";
        let escaped = program(r#"{"type":"string","pattern":"^ac[0-9]{2}$"}"#);
        for (label, prefix, expected) in [
            ("escape-start", br#""a\u"#.as_slice(), b"0".as_slice()),
            ("escape-zero", br#""a\u0"#.as_slice(), b"0".as_slice()),
            ("escape-byte", br#""a\u00"#.as_slice(), b"6".as_slice()),
            ("escape-nibble", br#""a\u006"#.as_slice(), b"3".as_slice()),
        ] {
            let (mask, _) =
                assert_three_route_mask_and_advance(&escaped, &vocabulary, &trie, prefix, label);
            for digit in digits {
                assert_eq!(
                    bit(&mask, digit),
                    expected.contains(&digit),
                    "{label}: hex byte {digit:?}"
                );
            }
            assert!(mask.iter().any(|byte| *byte != 0), "{label}: empty mask");
        }
        let mut dead_escape = escaped.new_matcher().expect("escaped matcher");
        assert!(dead_escape.advance(br#""a\u00"#).unwrap());
        assert!(!dead_escape.advance(b"a").unwrap());
        assert!(!dead_escape.is_dead());

        let already_complete = program(r#"{"type":"string","pattern":"^ac[0-9]{2}$"}"#);
        let prefix = br#""ac11"#;
        let (mask, _) = assert_three_route_mask_and_advance(
            &already_complete,
            &vocabulary,
            &trie,
            prefix,
            "complete-pattern-cannot-open-an-extra-escape",
        );
        assert!(bit(&mask, b'"'));
        assert!(!bit(&mask, b'\\'));
        let mut rejected_surrogate = already_complete
            .new_matcher()
            .expect("surrogate rejection matcher");
        assert!(rejected_surrogate.advance(prefix).unwrap());
        assert!(!rejected_surrogate.advance(b"\\").unwrap());
        assert!(!rejected_surrogate.is_dead());

        let supplementary = program(r#"{"type":"string","pattern":"^a😀$"}"#);
        let (mask, _) = assert_three_route_mask_and_advance(
            &supplementary,
            &vocabulary,
            &trie,
            br#""a\u"#,
            "live-surrogate-start",
        );
        assert!(bit(&mask, b'D') && bit(&mask, b'd'));
        assert!(!bit(&mask, b'0'));
        let mut supplementary_witness = supplementary
            .new_matcher()
            .expect("supplementary witness matcher");
        assert!(supplementary_witness
            .advance(br#""a\uD83D\uDE00""#)
            .unwrap());
        assert!(supplementary_witness.is_accepting());

        for (label, schema, prefix) in [
            (
                "unconstrained-string-escape",
                r#"{"type":"string"}"#,
                br#""a\u"#.as_slice(),
            ),
            (
                "length-headroom-escape",
                r#"{"type":"string","maxLength":5}"#,
                br#""ab\u"#.as_slice(),
            ),
            (
                "single-scalar-pattern-escape",
                r#"{"type":"string","pattern":"^a.$"}"#,
                br#""a\u"#.as_slice(),
            ),
            (
                "open-object-key-escape",
                r#"{"type":"object"}"#,
                br#"{"a\u"#.as_slice(),
            ),
            (
                "plain-array-string-escape",
                r#"{"type":"array","items":{"type":"string"}}"#,
                br#"["a\u"#.as_slice(),
            ),
        ] {
            let control = program(schema);
            let (mask, _) =
                assert_three_route_mask_and_advance(&control, &vocabulary, &trie, prefix, label);
            for digit in digits {
                assert!(bit(&mask, digit), "{label}: rejected {digit:?}");
            }
        }

        let bmp_enum = program(r#"{"type":"array","items":{"enum":["c","d"]},"uniqueItems":true}"#);
        let (mask, _) = assert_three_route_mask_and_advance(
            &bmp_enum,
            &vocabulary,
            &trie,
            br#"["\u"#,
            "bmp-enum-surrogate-rejection",
        );
        for digit in digits {
            assert_eq!(bit(&mask, digit), digit == b'0', "BMP enum: {digit:?}");
        }

        let supplementary_enum =
            program(r#"{"type":"array","items":{"enum":["\ud83d\ude00","x"]},"uniqueItems":true}"#);
        let (mask, _) = assert_three_route_mask_and_advance(
            &supplementary_enum,
            &vocabulary,
            &trie,
            br#"["\u"#,
            "supplementary-enum-surrogate-reachable",
        );
        assert!(bit(&mask, b'D') && bit(&mask, b'd'));

        let raw = program(r#"{"type":"string","pattern":"^aé$"}"#);
        let (mask, _) =
            assert_three_route_mask_and_advance(&raw, &vocabulary, &trie, b"\"a", "raw-utf8-lead");
        assert!(bit(&mask, 0xc3));
        assert!(!bit(&mask, 0xc2));
        let (mask, _) = assert_three_route_mask_and_advance(
            &raw,
            &vocabulary,
            &trie,
            b"\"a\xc3",
            "raw-utf8-continuation",
        );
        assert!(bit(&mask, 0xa9));
        assert!(!bit(&mask, 0xa8));
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn object_key_partial_scalars_and_pattern_unions_preserve_liveness() {
        let records = (0u8..=255)
            .map(|byte| (vec![byte], vec![u32::from(byte)]))
            .collect();
        let vocabulary = VocabularyHandle::new(Arc::new(
            build_vocabulary(256, records).expect("byte vocabulary"),
        ))
        .expect("vocabulary handle");
        let trie = TrieCache::new().bind(&vocabulary).expect("trie");
        let bit = |mask: &[u8], byte: u8| mask[usize::from(byte) / 8] & (1 << (byte % 8)) != 0;
        let digits = *b"0123456789abcdefABCDEF";

        let exact = program(
            r#"{"type":"object","properties":{"edges":{"type":"integer"},"force":{"type":"array","uniqueItems":true}},"additionalProperties":false}"#,
        );
        for (label, prefix) in [
            ("exact-first-escape", br#"{"e\u"#.as_slice()),
            ("exact-second-escape", br#"{"ed\u"#.as_slice()),
        ] {
            let (mask, _) =
                assert_three_route_mask_and_advance(&exact, &vocabulary, &trie, prefix, label);
            for digit in digits {
                assert_eq!(bit(&mask, digit), digit == b'0', "{label}: {digit:?}");
            }
        }

        let named_pattern = program(
            r#"{"type":"object","propertyNames":{"pattern":"^[a-z]{2}$"},"patternProperties":{"^[a-z]{2}$":{"type":"integer"}},"additionalProperties":false}"#,
        );
        let (mask, _) = assert_three_route_mask_and_advance(
            &named_pattern,
            &vocabulary,
            &trie,
            br#"{"a\u"#,
            "property-name-and-pattern",
        );
        for digit in digits {
            assert_eq!(bit(&mask, digit), digit == b'0', "pattern key: {digit:?}");
        }

        let no_keys = program(
            r#"{"type":"object","propertyNames":{"pattern":"^[a-z]{2}$"},"additionalProperties":false}"#,
        );
        let (mask, _) = assert_three_route_mask_and_advance(
            &no_keys,
            &vocabulary,
            &trie,
            b"{",
            "empty-key-language",
        );
        assert!(!bit(&mask, b'"'));
        assert!(bit(&mask, b'}'));

        let two_patterns = program(
            r#"{"type":"object","patternProperties":{"^a[a-z]$":{"type":"integer"},"^b[a-z]$":{"type":"integer"}},"additionalProperties":false}"#,
        );
        let (mask, _) = assert_three_route_mask_and_advance(
            &two_patterns,
            &vocabulary,
            &trie,
            br#"{""#,
            "two-pattern-union",
        );
        assert!(bit(&mask, b'a') && bit(&mask, b'b') && bit(&mask, b'\\'));
        assert!(!bit(&mask, b'z'));

        let unevaluated_patterns = program(
            r#"{"type":"object","patternProperties":{"^a[a-z]$":{"type":"integer"},"^b[a-z]$":{"type":"integer"}},"unevaluatedProperties":false}"#,
        );
        let (mask, _) = assert_three_route_mask_and_advance(
            &unevaluated_patterns,
            &vocabulary,
            &trie,
            br#"{""#,
            "unevaluated-two-pattern-union",
        );
        assert!(bit(&mask, b'a') && bit(&mask, b'b'));
        assert!(!bit(&mask, b'z'));

        let covered_by_false_pattern = program(
            r#"{"type":"object","patternProperties":{"^a[a-z]$":true,"^a.$":false},"additionalProperties":false}"#,
        );
        let (mask, _) = assert_three_route_mask_and_advance(
            &covered_by_false_pattern,
            &vocabulary,
            &trie,
            b"{",
            "positive-language-covered-by-false-pattern",
        );
        assert!(!bit(&mask, b'"'));
        assert!(bit(&mask, b'}'));

        let partially_subtracted_pattern = program(
            r#"{"type":"object","patternProperties":{"^a[a-z]$":true,"^ab$":false},"additionalProperties":false}"#,
        );
        let (mask, _) = assert_three_route_mask_and_advance(
            &partially_subtracted_pattern,
            &vocabulary,
            &trie,
            br#"{"a"#,
            "positive-language-minus-one-false-key",
        );
        assert!(!bit(&mask, b'b'));
        assert!(bit(&mask, b'c'));
        let (mask, _) = assert_three_route_mask_and_advance(
            &partially_subtracted_pattern,
            &vocabulary,
            &trie,
            br#"{"aa"#,
            "complete-key-cannot-open-an-extra-escape",
        );
        assert!(bit(&mask, b'"'));
        assert!(!bit(&mask, b'\\'));

        let unevaluated_items =
            program(r#"{"type":"array","prefixItems":[{"enum":["p"]}],"unevaluatedItems":false}"#);
        let (mask, _) = assert_three_route_mask_and_advance(
            &unevaluated_items,
            &vocabulary,
            &trie,
            br#"["p""#,
            "unevaluated-items-exhausted",
        );
        assert!(bit(&mask, b']'));
        assert!(!bit(&mask, b','));

        let ascii_records: FxHashMap<Vec<u8>, Vec<u32>> = (0x20u8..=0x7e)
            .chain([b'\t', b'\n', b'\r'])
            .map(|byte| (vec![byte], vec![u32::from(byte)]))
            .collect();
        let ascii_vocabulary = VocabularyHandle::new(Arc::new(
            build_vocabulary(256, ascii_records).expect("ASCII vocabulary"),
        ))
        .expect("ASCII vocabulary handle");
        let ascii_trie = TrieCache::new()
            .bind(&ascii_vocabulary)
            .expect("ASCII trie");
        let (ascii_mask, _) = assert_three_route_mask_and_advance(
            &unevaluated_items,
            &ascii_vocabulary,
            &ascii_trie,
            br#"["p""#,
            "unevaluated-items-sparse-vocabulary",
        );
        assert!(!bit(&ascii_mask, b','));
        let mut whole = unevaluated_items
            .new_matcher()
            .expect("whole-token unevaluatedItems matcher");
        assert!(!whole.advance(br#"["p","#).unwrap());
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn arrays_numbers_and_negation_refuse_known_dead_prefixes() {
        let records = (0u8..=255)
            .map(|byte| (vec![byte], vec![u32::from(byte)]))
            .collect();
        let vocabulary = VocabularyHandle::new(Arc::new(
            build_vocabulary(256, records).expect("byte vocabulary"),
        ))
        .expect("vocabulary handle");
        let trie = TrieCache::new().bind(&vocabulary).expect("trie");
        let bit = |mask: &[u8], byte: u8| mask[usize::from(byte) / 8] & (1 << (byte % 8)) != 0;

        let max_contains = program(
            r#"{"type":"array","items":{"enum":["p","q"]},"contains":{"const":"p"},"maxContains":1,"maxItems":3}"#,
        );
        let (mask, _) = assert_three_route_mask_and_advance(
            &max_contains,
            &vocabulary,
            &trie,
            br#"["p",""#,
            "max-contains-second-item-prefix",
        );
        assert!(!bit(&mask, b'p'));
        assert!(bit(&mask, b'q'));
        let mut direct = max_contains.new_matcher().expect("maxContains matcher");
        assert!(direct.advance(br#"["p",""#).unwrap());
        assert!(!direct.advance(b"p").unwrap());

        let exclusive = program(r#"{"type":"number","exclusiveMinimum":0,"exclusiveMaximum":1}"#);
        let (mask, _) = assert_three_route_mask_and_advance(
            &exclusive,
            &vocabulary,
            &trie,
            b" ",
            "exclusive-unit-interval",
        );
        assert!(bit(&mask, b'0'));
        assert!(bit(&mask, b'1'));
        let mut exponent_witness = exclusive.new_matcher().expect("number witness");
        assert!(exponent_witness.advance(b"1e-1").unwrap());
        assert!(exponent_witness.is_accepting());

        let negated_limit = program(r#"{"type":"string","maxLength":3,"not":{"const":"abc"}}"#);
        let (mask, _) = assert_three_route_mask_and_advance(
            &negated_limit,
            &vocabulary,
            &trie,
            br#""ab"#,
            "negated-max-length",
        );
        assert!(!bit(&mask, b'c'));

        let impossible_unique = program(
            r#"{"type":"array","items":{"enum":["p","q"]},"uniqueItems":true,"minItems":3,"maxItems":3}"#,
        );
        let (mask, _) = assert_three_route_mask_and_advance(
            &impossible_unique,
            &vocabulary,
            &trie,
            b"",
            "impossible-finite-unique-array",
        );
        assert!(!bit(&mask, b'['));
        for whitespace in b" \t\r\n" {
            assert!(!bit(&mask, *whitespace));
        }
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn realistic_depth_mask_advance_and_witness_reachability_corpus_gate() {
        struct Fixture {
            name: &'static str,
            schema: &'static str,
            witness: &'static [u8],
            selected_prefix: &'static [u8],
        }

        let fixtures = [
            Fixture {
                name: "closed-exact-object",
                schema: r#"{"type":"object","properties":{"nodes":{"type":"array"},"edges":{"type":"array"}},"required":["nodes","edges"],"additionalProperties":false}"#,
                witness: br#"{"nodes":[],"edges":[]}"#,
                selected_prefix: br#"{""#,
            },
            Fixture {
                name: "pydantic-defs-ref-nested-arrays",
                schema: r##"{"$defs":{"Node":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":false},"Edge":{"type":"object","properties":{"source":{"type":"integer"},"target":{"type":"integer"}},"required":["source","target"],"additionalProperties":false}},"type":"object","properties":{"nodes":{"type":"array","items":{"$ref":"#/$defs/Node"}},"edges":{"type":"array","items":{"$ref":"#/$defs/Edge"}}},"required":["nodes","edges"],"additionalProperties":false}"##,
                witness: br#"{"nodes":[{"id":1}],"edges":[{"source":1,"target":2}]}"#,
                selected_prefix: br#"{"nodes":[{"id":1}],"edges":[{"source":1,"target":"#,
            },
            Fixture {
                name: "property-name-two-pattern-union",
                schema: r#"{"type":"object","propertyNames":{"pattern":"^[a-z]{2,6}$"},"patternProperties":{"^a[a-z]{1,5}$":{"type":"string"},"^r[a-z]{1,5}$":{"type":"string"}},"additionalProperties":false,"maxProperties":2}"#,
                witness: br#"{"amount":"USD","routin":"acct"}"#,
                selected_prefix: br#"{"amount":"USD","routin"#,
            },
            Fixture {
                name: "conditional-dependent-unevaluated-oneof",
                schema: r#"{"type":"object","properties":{"kind":{"enum":["a","b"]},"value":{"oneOf":[{"type":"integer"},{"type":"string","pattern":"^ok$"}]},"flag":{"type":"boolean"}},"required":["kind","value"],"dependentSchemas":{"flag":{"required":["value"]}},"allOf":[{"if":{"properties":{"kind":{"const":"a"}},"required":["kind"]},"then":{"not":{"properties":{"value":{"const":0}},"required":["value"]}},"else":{"anyOf":[{"required":["flag"]},{"not":{"required":["flag"]}}]}}],"unevaluatedProperties":false}"#,
                witness: br#"{"kind":"a","value":1,"flag":true}"#,
                selected_prefix: br#"{"kind":"a","value":1,"flag":t"#,
            },
            Fixture {
                name: "unique-contains-third-item",
                schema: r#"{"type":"array","items":{"enum":["treasury","ops","risk"]},"contains":{"const":"treasury"},"uniqueItems":true,"minItems":1,"maxItems":3}"#,
                witness: br#"["treasury","risk","ops"]"#,
                selected_prefix: br#"["treasury","risk",""#,
            },
            Fixture {
                name: "unevaluated-items-boundary",
                schema: r#"{"prefixItems":[{"enum":["p"]}],"contains":{"const":"q"},"unevaluatedItems":false}"#,
                witness: br#"["p","q"]"#,
                selected_prefix: br#"["p","q"#,
            },
            Fixture {
                name: "dynamic-reference-recursion",
                schema: r##"{"$dynamicAnchor":"node","type":"object","properties":{"value":{"type":"integer"},"child":{"$dynamicRef":"#node"}},"required":["value"],"additionalProperties":false}"##,
                witness: br#"{"value":1,"child":{"value":2,"child":{"value":3}}}"#,
                selected_prefix: br#"{"value":1,"child":{"value":2,"child":{"value":"#,
            },
            Fixture {
                name: "exclusive-decimal-multiple",
                schema: r#"{"type":"number","exclusiveMinimum":0,"exclusiveMaximum":1,"multipleOf":0.1}"#,
                witness: b"0.5",
                selected_prefix: b"0.",
            },
            Fixture {
                name: "max-contains-cap",
                schema: r#"{"type":"array","items":{"enum":["p","q"]},"contains":{"const":"p"},"maxContains":1,"maxItems":3}"#,
                witness: br#"["p","q"]"#,
                selected_prefix: br#"["p",""#,
            },
            Fixture {
                name: "negated-constant-at-max-length",
                schema: r#"{"type":"string","maxLength":3,"not":{"const":"abc"}}"#,
                witness: br#""abd""#,
                selected_prefix: br#""ab"#,
            },
        ];

        let mut records: FxHashMap<Vec<u8>, Vec<u32>> = (0u8..=255)
            .map(|byte| (vec![byte], vec![u32::from(byte)]))
            .collect();
        records.insert(b"e".to_vec(), vec![u32::from(b'e'), 256]);
        records.insert(b"name".to_vec(), vec![257, 258]);
        records.insert(b"ops".to_vec(), vec![259, 260]);
        let vocabulary = VocabularyHandle::new(Arc::new(
            build_vocabulary(261, records).expect("corpus vocabulary"),
        ))
        .expect("corpus vocabulary handle");
        let trie = TrieCache::new().bind(&vocabulary).expect("corpus trie");
        let output_len = vocabulary.mask_vocab_size().div_ceil(32) * 4;

        for fixture in fixtures {
            assert!(fixture.witness.starts_with(fixture.selected_prefix));
            let case = program(fixture.schema);
            assert_three_route_mask_and_advance(
                &case,
                &vocabulary,
                &trie,
                fixture.selected_prefix,
                fixture.name,
            );

            let mut matcher = case.new_matcher().expect("corpus matcher");
            for (index, &next) in fixture.witness.iter().enumerate() {
                let mut first = vec![0xa5; output_len];
                matcher
                    .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut first)
                    .unwrap_or_else(|error| panic!("{} prefix {index}: {error}", fixture.name));
                assert!(
                    first.iter().any(|byte| *byte != 0),
                    "{} produced an empty mask at valid prefix {:?}",
                    fixture.name,
                    &fixture.witness[..index]
                );
                assert!(
                    first[usize::from(next) / 8] & (1 << (next % 8)) != 0,
                    "{} masked witness byte {next:?} at prefix {:?}",
                    fixture.name,
                    &fixture.witness[..index]
                );
                assert!(
                    matcher.advance(&[next]).unwrap(),
                    "{} rejected witness byte {next:?} at {index}",
                    fixture.name
                );
                assert_incremental_accounting(&matcher, fixture.name);
            }
            assert!(matcher.is_accepting(), "{} witness", fixture.name);
        }
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn finite_admitted_frontier_successors_are_completable() {
        use std::collections::{HashSet, VecDeque};

        let fixtures = [
            (
                "max-contains-frontier",
                r#"{"type":"array","items":{"enum":["p","q"]},"contains":{"const":"p"},"maxContains":1,"maxItems":3}"#,
                br#"["p",""#.as_slice(),
            ),
            (
                "unique-contains-frontier",
                r#"{"type":"array","items":{"enum":["treasury","ops","risk"]},"contains":{"const":"treasury"},"uniqueItems":true,"minItems":1,"maxItems":3}"#,
                br#"["treasury","risk",""#.as_slice(),
            ),
            (
                "pattern-difference-frontier",
                r#"{"type":"object","patternProperties":{"^a[a-z]$":true,"^ab$":false},"additionalProperties":false}"#,
                br#"{"a"#.as_slice(),
            ),
            (
                "negated-max-length-frontier",
                r#"{"type":"string","maxLength":3,"not":{"const":"abc"}}"#,
                br#""ab"#.as_slice(),
            ),
        ];
        let records: FxHashMap<Vec<u8>, Vec<u32>> = (0x20u8..=0x7e)
            .chain([b'\t', b'\n', b'\r'])
            .map(|byte| (vec![byte], vec![u32::from(byte)]))
            .collect();
        let vocabulary = VocabularyHandle::new(Arc::new(
            build_vocabulary(256, records).expect("ASCII liveness vocabulary"),
        ))
        .expect("ASCII liveness vocabulary handle");
        let trie = TrieCache::new()
            .bind(&vocabulary)
            .expect("ASCII liveness trie");
        let output_len = vocabulary.mask_vocab_size().div_ceil(32) * 4;
        let fixture_filter = std::env::var("MF_LIVENESS_FIXTURE").ok();

        for (label, schema, initial) in fixtures {
            if fixture_filter
                .as_deref()
                .is_some_and(|wanted| wanted != label)
            {
                continue;
            }
            let case = program(schema);
            let mut frontier = VecDeque::from([(initial.to_vec(), 0usize)]);
            let mut visited = HashSet::from([initial.to_vec()]);
            while let Some((prefix, frontier_depth)) = frontier.pop_front() {
                let mut matcher = case.new_matcher().expect("frontier matcher");
                assert!(matcher.advance(&prefix).unwrap(), "{label}: {prefix:?}");
                let mut output = vec![0xa5; output_len];
                matcher
                    .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
                    .unwrap_or_else(|error| panic!("{label} {prefix:?}: {error}"));
                for (token, ids) in vocabulary.iter_records() {
                    let admitted = ids
                        .iter()
                        .any(|id| output[*id as usize / 8] & (1 << (*id % 8)) != 0);
                    let mut candidate = prefix.clone();
                    candidate.extend_from_slice(token);
                    let mut exact = case.new_matcher().expect("frontier transition matcher");
                    assert_eq!(
                        admitted,
                        exact.advance(&candidate).unwrap(),
                        "{label}: mask/whole-token advance disagreement: {candidate:?}"
                    );
                }
                if std::env::var_os("MF_LIVENESS_TRANSITIONS_ONLY").is_some() {
                    continue;
                }
                for (token, ids) in vocabulary.iter_records() {
                    let admitted = ids
                        .iter()
                        .any(|id| output[*id as usize / 8] & (1 << (*id % 8)) != 0);
                    if !admitted {
                        continue;
                    }
                    let mut candidate = prefix.clone();
                    candidate.extend_from_slice(token);
                    let mut budget = 50_000;
                    assert_eq!(
                        closer_first_completion_search(&case, &candidate, 24, &mut budget),
                        CompletionSearch::Found,
                        "{label}: admitted successor has no proven completion: {candidate:?}"
                    );
                    if frontier_depth < 1 && visited.insert(candidate.clone()) {
                        frontier.push_back((candidate, frontier_depth + 1));
                    }
                }
                assert!(visited.len() <= 512, "{label}: unbounded test frontier");
            }
        }
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn closed_object_key_slice_never_emits_a_token_that_exact_advance_rejects() {
        // Leading whitespace covers a GPT-2 prefix with no declared property match.
        // `uniqueItems` elsewhere still selects the structured backend.
        let program = program(
            r#"{
                "type":"object",
                "properties":{
                    "productId":{"type":"integer"},
                    "tags":{"type":"array","items":{"type":"string"},"uniqueItems":true}
                },
                "required":["productId"],
                "additionalProperties":false
            }"#,
        );
        let mut tokens: FxHashMap<Vec<u8>, Vec<u32>> = (0u8..=255)
            .map(|byte| (vec![byte], vec![u32::from(byte)]))
            .collect();
        tokens.insert(b"name".to_vec(), vec![256]);
        let vocabulary = VocabularyHandle::new(Arc::new(
            build_vocabulary(257, tokens).expect("key-slice vocabulary"),
        ))
        .expect("vocabulary handle");
        let trie = TrieCache::new().bind(&vocabulary).expect("trie");
        let (mask, _) = assert_three_route_mask_and_advance(
            &program,
            &vocabulary,
            &trie,
            b"\n{ \"",
            "closed-key-after-whitespace",
        );
        assert_eq!(
            mask[256 / 8] & (1 << (256 % 8)),
            0,
            "an impossible multi-byte key prefix must not be emitted"
        );
        assert_ne!(
            mask[usize::from(b'p') / 8] & (1 << (b'p' % 8)),
            0,
            "the viable `productId` prefix must remain"
        );
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn pydantic_knowledge_graph_root_key_mask_matches_every_transition() {
        let schema = program(
            r##"{
                "$defs":{
                    "Edge":{
                        "additionalProperties":false,
                        "description":"One directed relation between two entities.",
                        "properties":{
                            "source":{"description":"Unique source of the edge","maximum":9,"minimum":1,"title":"Source","type":"integer"},
                            "target":{"description":"Unique target of the edge","maximum":9,"minimum":1,"title":"Target","type":"integer"},
                            "label":{"description":"Label of the edge","maxLength":10,"title":"Label","type":"string"}
                        },
                        "required":["source","target","label"],
                        "title":"Edge",
                        "type":"object"
                    },
                    "Node":{
                        "additionalProperties":false,
                        "description":"One entity in the graph.",
                        "properties":{
                            "id":{"description":"Unique identifier of the node","maximum":9,"minimum":1,"title":"Id","type":"integer"},
                            "label":{"description":"Label of the node","maxLength":10,"title":"Label","type":"string"}
                        },
                        "required":["id","label"],
                        "title":"Node",
                        "type":"object"
                    }
                },
                "additionalProperties":false,
                "description":"A graph of entities and the relations between them.",
                "properties":{
                    "nodes":{"items":{"$ref":"#/$defs/Node"},"maxItems":3,"minItems":1,"title":"Nodes","type":"array"},
                    "edges":{"items":{"$ref":"#/$defs/Edge"},"maxItems":2,"minItems":1,"title":"Edges","type":"array"}
                },
                "required":["nodes","edges"],
                "title":"KnowledgeGraph",
                "type":"object"
            }"##,
        );
        let mut records: FxHashMap<Vec<u8>, Vec<u32>> = (0u8..=255)
            .map(|byte| (vec![byte], vec![u32::from(byte)]))
            .collect();
        for (id, token) in [
            (256, b"name".as_slice()),
            (257, b"nodes"),
            (259, b"edges"),
            (260, b"node"),
            (261, b"edge"),
            (262, b"nodes\""),
            (263, b"nodes\":"),
            (264, b"edges\":["),
            (265, b"\\u006e"),
            (266, b"\\u0065"),
            (267, b"\\u2"),
            (268, b"\":"),
            (269, b"nodes\":["),
            (270, b"n\\u006fdes\":"),
            (271, b"\xc3\xa9"),
        ] {
            records.insert(token.to_vec(), vec![id]);
        }
        records.get_mut(b"nodes".as_slice()).unwrap().push(258);
        records.get_mut(&[0xef][..]).unwrap().push(272);
        let vocabulary = VocabularyHandle::new(Arc::new(
            build_vocabulary(280, records).expect("knowledge graph vocabulary"),
        ))
        .expect("vocabulary handle");
        let trie = TrieCache::new().bind(&vocabulary).expect("trie");
        let prefix = b"{ \"";
        let (mask, _) = assert_three_route_mask_and_advance(
            &schema,
            &vocabulary,
            &trie,
            prefix,
            "knowledge-graph-root-key",
        );
        let bit = |id: u32| mask[id as usize / 8] & (1 << (id % 8)) != 0;

        for byte in 0x20u8..=0x7e {
            assert_eq!(
                bit(u32::from(byte)),
                matches!(byte, b'n' | b'e' | b'\\'),
                "printable root key byte {byte:?}"
            );
        }
        for byte in 0u8..0x20 {
            assert!(!bit(u32::from(byte)), "control key byte {byte:#04x}");
        }
        assert!(!bit(256), "raw token `name` must be rejected");
        assert!(bit(257) && bit(258), "all aliases for `nodes` must agree");
        assert!(bit(259));

        let mut decoder = super::super::lexer::JsonStringDecoder::new();
        assert_eq!(
            decoder.push(0xef),
            super::super::lexer::DecodeStep::Continue,
            "0xEF is a lexically pending UTF-8 lead byte"
        );
        assert!(!bit(0xef), "0xEF cannot reach either finite decoded key");
        assert!(!bit(272), "the 0xEF alias must agree");

        for (token, ids) in vocabulary.iter_records() {
            let mut matcher = schema.new_matcher().expect("transition matcher");
            assert!(matcher.advance(prefix).unwrap());
            let before = (
                matcher.is_accepting(),
                matcher.is_dead(),
                matcher.eos_legal(),
            );
            let accepted = matcher.advance(token).unwrap();
            for id in ids {
                assert_eq!(bit(*id), accepted, "token {token:?}, id {id}");
            }
            if !accepted {
                assert_eq!(
                    (
                        matcher.is_accepting(),
                        matcher.is_dead(),
                        matcher.eos_legal(),
                    ),
                    before,
                    "rejected token mutated committed state: {token:?}"
                );
            }
            assert_incremental_accounting(&matcher, "knowledge-graph-transition");
        }

        for document in [
            br#"{"nodes":[{"id":1,"label":"a"}],"edges":[{"source":1,"target":1,"label":"x"}]}"#
                .as_slice(),
            br#"{"edges":[{"source":1,"target":1,"label":"x"}],"nodes":[{"label":"a","id":1}]}"#
                .as_slice(),
        ] {
            let mut matcher = schema.new_matcher().expect("witness matcher");
            assert!(matcher.advance(document).unwrap());
            assert!(matcher.is_accepting());
            assert_incremental_accounting(&matcher, "knowledge-graph-witness");
        }
    }

    #[test]
    fn unique_contains_trie_masks_are_transactional_after_each_committed_item() {
        let ir = ir(
            r#"{"type":"array","items":{"type":"integer"},"uniqueItems":true,"contains":{"const":0}}"#,
        );
        let program = StructuredProgram::compile(ir.clone()).expect("program");
        let mut tokens = FxHashMap::default();
        for (id, bytes) in [
            (0, b"[".as_slice()),
            (1, b"0"),
            (2, b",1"),
            (3, b",2"),
            (4, b" "),
            (5, b","),
            (6, b"]"),
            (7, b",1]"),
            (8, b",3]"),
            (9, b"x"),
        ] {
            tokens.insert(bytes.to_vec(), vec![id]);
        }
        let handle = VocabularyHandle::new(Arc::new(build_vocabulary(10, tokens).expect("vocab")))
            .expect("handle");
        let trie = TrieCache::new().bind(&handle).expect("trie");
        let mut trie_matcher = program.new_matcher().expect("trie matcher");
        let mut record_matcher = program.new_matcher().expect("record matcher");
        let mut output = [0u8; 4];

        for token in [b"[".as_slice(), b"0", b",1", b",2"] {
            assert!(trie_matcher.advance(token).expect("trie commit"));
            assert!(record_matcher.advance(token).expect("record commit"));
            for (bytes, ids) in handle.iter_records() {
                record_matcher
                    .allowed_mask_from_records(handle.mask_vocab_size(), [(bytes, ids)].into_iter())
                    .unwrap_or_else(|error| {
                        panic!("candidate {bytes:?} after {token:?}: {error:?}")
                    });
            }
            let expected = record_matcher
                .allowed_mask_from_records(handle.mask_vocab_size(), handle.iter_records())
                .unwrap_or_else(|error| panic!("record mask after {token:?}: {error:?}"));
            for _ in 0..128 {
                trie_matcher
                    .write_mask_le_bytes_into(&handle, Some(&trie), &mut output)
                    .expect("trie mask");
            }
            for id in 0..handle.mask_vocab_size() {
                let bit = output[id / 8] & (1u8 << (id % 8)) != 0;
                assert_eq!(bit, expected.get(TokenId(u32::try_from(id).expect("id"))));
            }
        }
        // Speculative probes: a completed duplicate (",1]") stays rejected, a distinct value
        // (",3]") stays accepted, agreeing between the trie walk and the record scan.
        let dup_records = [(b",1]".as_slice(), [7u32].as_slice())];
        let distinct_records = [(b",3]".as_slice(), [8u32].as_slice())];
        let dup_mask = record_matcher
            .allowed_mask_from_records(handle.mask_vocab_size(), dup_records.into_iter())
            .expect("dup mask");
        assert!(!dup_mask.get(TokenId(7)), "duplicate 1 must stay rejected");
        let distinct_mask = record_matcher
            .allowed_mask_from_records(handle.mask_vocab_size(), distinct_records.into_iter())
            .expect("distinct mask");
        assert!(
            distinct_mask.get(TokenId(8)),
            "distinct value 3 must stay accepted"
        );
        let mut trie_out = [0u8; 4];
        trie_matcher
            .write_mask_le_bytes_into(&handle, Some(&trie), &mut trie_out)
            .expect("trie mask after probes");
        let trie_bit = |id: u32| trie_out[id as usize / 8] & (1u8 << (id % 8)) != 0;
        assert!(!trie_bit(7), "trie: duplicate 1 must stay rejected");
        assert!(trie_bit(8), "trie: distinct 3 must stay accepted");
        // A properly-terminated distinct value still closes the array cleanly afterward.
        assert!(trie_matcher
            .advance(b",3]")
            .expect("distinct value 3 accepted"));
        assert!(trie_matcher.is_accepting());
        assert!(record_matcher
            .advance(b",3]")
            .expect("distinct value 3 accepted"));
        assert!(record_matcher.is_accepting());
    }

    #[test]
    fn trie_output_matches_record_scan_and_clears_dirty_output() {
        let program = program(
            r#"{"type":"object","properties":{"x":{"type":"boolean"}},"required":["x"],"additionalProperties":false}"#,
        );
        let mut tokens = FxHashMap::default();
        tokens.insert(b"{".to_vec(), vec![0]);
        tokens.insert(b"\"x\"".to_vec(), vec![1]);
        tokens.insert(b":".to_vec(), vec![2]);
        tokens.insert(b"true".to_vec(), vec![3, 7]);
        tokens.insert(b"false".to_vec(), vec![4]);
        tokens.insert(b"}".to_vec(), vec![5]);
        let vocabulary = Arc::new(build_vocabulary(8, tokens).expect("vocabulary"));
        let handle = VocabularyHandle::new(vocabulary).expect("handle");
        let trie = TrieCache::new().bind(&handle).expect("trie");
        let mut records = program.new_matcher().expect("records matcher");
        let mut walked = program.new_matcher().expect("trie matcher");
        for token in [b"{".as_slice(), b"\"x\"", b":"] {
            assert!(records.advance(token).expect("advance"));
            assert!(walked.advance(token).expect("advance"));
        }
        let expected = records
            .allowed_mask_from_records(handle.mask_vocab_size(), handle.iter_records())
            .expect("record mask");
        let mut output =
            vec![0xff; handle.mask_vocab_size().div_ceil(32) * std::mem::size_of::<u32>()];
        walked
            .write_mask_le_bytes_into(&handle, Some(&trie), &mut output)
            .expect("trie mask");
        for id in 0..handle.mask_vocab_size() {
            let token = TokenId(u32::try_from(id).expect("test token id"));
            let observed = output[id / 8] & (1u8 << (id % 8)) != 0;
            assert_eq!(observed, expected.get(token), "token {id}");
        }
        assert!(walked.advance(b"true").expect("traversal rolled back"));
    }

    #[test]
    fn trie_record_scan_and_reference_masks_are_bit_identical_at_a_committed_prefix() {
        let schema = r#"{"type":"object","additionalProperties":{"type":"boolean"}}"#;
        let ir = ir(schema);
        let program = StructuredProgram::compile(ir.clone()).expect("program");
        assert_eq!(program.backend_kind(), "incremental");
        let mut tokens = FxHashMap::default();
        let pieces: &[&[u8]] = &[
            b"{", b"}", b"\"", b":", b",", b"k", b"0", b"1", b"true", b"false", b"tru", b"e",
        ];
        for (i, p) in pieces.iter().enumerate() {
            tokens.insert(p.to_vec(), vec![i as u32]);
        }
        tokens.insert(b"}".to_vec(), vec![1, 12]);
        let width = 16usize;
        let handle = VocabularyHandle::new(Arc::new(build_vocabulary(15, tokens).expect("vocab")))
            .expect("handle");
        let trie = TrieCache::new().bind(&handle).expect("trie");
        let prefix: &[u8] = br#"{"k0":true,"k1":false"#;

        let mut walked = program.new_matcher().expect("walked");
        let mut scanned = program.new_matcher().expect("scanned");
        for &b in prefix {
            assert!(walked.advance(&[b]).expect("advance"));
            assert!(scanned.advance(&[b]).expect("advance"));
        }
        let mut trie_out = vec![0u8; width.div_ceil(32) * 4];
        walked
            .write_mask_le_bytes_into(&handle, Some(&trie), &mut trie_out)
            .expect("trie mask");
        let record_mask = scanned
            .allowed_mask_from_records(width, handle.iter_records())
            .expect("record mask");

        let mut id_bytes: std::collections::HashMap<usize, Vec<u8>> =
            std::collections::HashMap::new();
        for (i, p) in pieces.iter().enumerate() {
            id_bytes.insert(i, p.to_vec());
        }
        id_bytes.insert(12, b"}".to_vec());
        let mut trie_sum = 0u64;
        let mut record_sum = 0u64;
        let mut ref_sum = 0u64;
        for id in 0..width {
            let token = TokenId(u32::try_from(id).expect("id"));
            let trie_bit = trie_out[id / 8] & (1u8 << (id % 8)) != 0;
            let record_bit = record_mask.get(token);
            let ref_bit = match id_bytes.get(&id) {
                Some(piece) => {
                    let mut cand = prefix.to_vec();
                    cand.extend_from_slice(piece);
                    super::super::reference::accepts(&ir, &cand)
                        || super::super::reference::can_continue(&ir, &cand)
                }
                None => false,
            };
            assert_eq!(trie_bit, record_bit, "trie vs record at id {id}");
            assert_eq!(trie_bit, ref_bit, "trie vs reference at id {id}");
            trie_sum |= u64::from(trie_bit) << id;
            record_sum |= u64::from(record_bit) << id;
            ref_sum |= u64::from(ref_bit) << id;
        }
        assert_eq!(trie_sum, record_sum, "checksum trie==record");
        assert_eq!(trie_sum, ref_sum, "checksum trie==reference");
    }

    #[test]
    fn adaptive_full_trie_and_record_scan_match_on_string_boundaries() {
        let schemas_and_prefixes: &[(&str, &[&[u8]])] = &[
            (
                r#"{"anyOf":[{"type":"object","properties":{"s":{"type":"string","maxLength":1}}},{"type":"object","properties":{"s":{"type":"string","maxLength":3}}}]}"#,
                &[br#"{"s":"a"#, br#"{"s":"aa"#],
            ),
            (
                r#"{"type":"object","unevaluatedProperties":{"type":"string","maxLength":1}}"#,
                &[br#"{"s":"a"#],
            ),
            (
                r#"{"type":"string","maxLength":40}"#,
                &[b"\"", b"\"abcdefgh"],
            ),
            (
                r#"{"type":"string","maxLength":2}"#,
                &[b"\"", b"\"a", b"\"aa"],
            ),
            (
                r#"{"type":"object","properties":{"s":{"type":"string","maxLength":2}}}"#,
                &[br#"{"s":""#, br#"{"s":"a"#, br#"{"s":"aa"#],
            ),
            (
                r#"{"type":"object","properties":{"s":{"type":"string","maxLength":2}},"unevaluatedProperties":false}"#,
                &[br#"{"s":""#, br#"{"s":"a"#, br#"{"s":"aa"#],
            ),
            (
                r#"{"allOf":[{"type":"object"},{"propertyNames":{"maxLength":2}}]}"#,
                &[br#"{""#, br#"{"a"#, br#"{"aa"#],
            ),
            (r#"{"type":"string"}"#, &[b"\"", b"\"a", br#""\u00"#]),
            (
                r#"{"allOf":[{"type":"string","pattern":"^a"},{"type":"string","pattern":"z$"}]}"#,
                &[b"\"a"],
            ),
            (
                r#"{"type":"string","minLength":1,"maxLength":2,"pattern":"^a"}"#,
                &[b"\"", b"\"a"],
            ),
            (
                r#"{"allOf":[{"type":"string","pattern":"^a"},{"type":"string","format":"email"}]}"#,
                &[b"\"a"],
            ),
        ];
        let mut tokens = FxHashMap::default();
        tokens.insert(b"a".to_vec(), vec![0, 1]);
        tokens.insert(br#"\u0061"#.to_vec(), vec![2]);
        tokens.insert(br#"\uD83D"#.to_vec(), vec![9]);
        tokens.insert(br#"\uDE00"#.to_vec(), vec![10]);
        tokens.insert(vec![0xf0, 0x9f], vec![3]);
        tokens.insert(vec![0x98, 0x80], vec![4]);
        tokens.insert(b"\xf0\x9f\x98\x80".to_vec(), vec![5]);
        tokens.insert(b"\"".to_vec(), vec![6]);
        tokens.insert(b"\\".to_vec(), vec![7]);
        tokens.insert(vec![0x1f], vec![8]);
        tokens.insert(b"z".to_vec(), vec![100]);
        tokens.insert(b"zz".to_vec(), vec![101]);
        tokens.insert("\u{e9}".as_bytes().to_vec(), vec![102]);
        tokens.insert("\u{20ac}".as_bytes().to_vec(), vec![103]);
        tokens.insert("e\u{301}".as_bytes().to_vec(), vec![104]);
        tokens.insert(vec![b'a'; 40], vec![105]);
        tokens.insert(br#"\uD83D\uDE00"#.to_vec(), vec![106]);
        tokens.insert(vec![b'a'; 33], vec![107]);
        tokens.insert(vec![b'a'; 65], vec![108]);
        tokens.insert(vec![b'a'; 129], vec![109]);
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(127, tokens).expect("vocabulary")))
                .expect("handle");
        let trie = TrieCache::new().bind(&vocabulary).expect("trie");
        let mask_bytes = vocabulary.mask_vocab_size().div_ceil(32) * 4;

        for &(schema, prefixes) in schemas_and_prefixes {
            let program = program(schema);
            for &prefix in prefixes {
                let mut adaptive = program.new_matcher().expect("adaptive matcher");
                let mut full = program.new_matcher().expect("full matcher");
                let mut records = program.new_matcher().expect("record matcher");
                assert!(adaptive.advance(prefix).expect("adaptive prefix"));
                assert!(full.advance(prefix).expect("full prefix"));
                assert!(records.advance(prefix).expect("record prefix"));
                let mut adaptive_out = vec![0u8; mask_bytes];
                let mut full_out = vec![0u8; mask_bytes];
                let mut record_out = vec![0u8; mask_bytes];
                adaptive
                    .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut adaptive_out)
                    .expect("adaptive mask");
                #[cfg(not(feature = "bench-internals"))]
                let MatcherBackend::Incremental(incremental) = &mut full.backend;
                #[cfg(feature = "bench-internals")]
                let incremental = match &mut full.backend {
                    MatcherBackend::Incremental(incremental) => incremental,
                    MatcherBackend::Reference(_) => panic!("expected incremental matcher"),
                };
                walk_full_trie_mask_for_test(
                    incremental,
                    &vocabulary,
                    &trie,
                    &mut full_out,
                    program.limits,
                )
                .expect("full trie mask");
                records
                    .write_record_mask_le_bytes_into(
                        vocabulary.mask_vocab_size(),
                        vocabulary.iter_records(),
                        &mut record_out,
                    )
                    .expect("record mask");
                assert_mask_bytes_equal(&adaptive_out, &full_out, &vocabulary, "adaptive/full");
                assert_mask_bytes_equal(&adaptive_out, &record_out, &vocabulary, "adaptive/record");
            }
        }
    }

    #[test]
    fn slice_proof_cache_is_exact_bounded_and_evicts() {
        let mut cache = SliceProofCache::default();
        let bytes = 2 * std::mem::size_of::<SliceProofEntry>();
        let key = |tag, state| SliceProofKey {
            vocabulary: VocabFingerprint::from_bytes([tag; 32]),
            node: NodeId(3),
            state: StateId(state),
            body_prefix: false,
            reject_invalid: false,
        };
        cache.insert(key(1, 4), true, bytes);
        cache.insert(key(2, 4), false, bytes);
        assert_eq!(cache.get(key(1, 4)), Some(true));
        assert_eq!(cache.get(key(2, 4)), Some(false));
        assert_eq!(cache.get(key(1, 5)), None);
        assert_eq!(
            cache.get(SliceProofKey {
                body_prefix: true,
                ..key(1, 4)
            }),
            None
        );
        cache.insert(key(3, 4), true, bytes);
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.get(key(1, 4)), None);
        let mut disabled = SliceProofCache::default();
        disabled.insert(key(1, 4), true, 0);
        assert!(disabled.entries.is_empty());
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn slice_proof_cache_measurements_track_hits_misses_and_entries() {
        let mut cache = SliceProofCache::default();
        let key = SliceProofKey {
            vocabulary: VocabFingerprint::from_bytes([1; 32]),
            node: NodeId(3),
            state: StateId(4),
            body_prefix: false,
            reject_invalid: false,
        };
        cache.insert(key, true, std::mem::size_of::<SliceProofEntry>());
        cache.reset_measurements();
        assert_eq!(cache.get(key), Some(true));
        assert_eq!(
            cache.get(SliceProofKey {
                state: StateId(5),
                ..key
            }),
            None
        );
        assert_eq!(
            cache.measurements(),
            SliceProofMetrics {
                hits: 1,
                misses: 1,
                entries: 1,
            }
        );
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn near_limit_scalar_escape_boundary_matrix_matches_all_routes_and_advances() {
        struct TokenCase {
            name: &'static str,
            bytes: Vec<u8>,
            decoded_key_bytes: Option<usize>,
            string_content_bytes: usize,
        }

        let cases = vec![
            TokenCase {
                name: "ascii",
                bytes: b"a".to_vec(),
                decoded_key_bytes: Some(1),
                string_content_bytes: 1,
            },
            TokenCase {
                name: "utf8-2",
                bytes: "¢".as_bytes().to_vec(),
                decoded_key_bytes: Some(2),
                string_content_bytes: 2,
            },
            TokenCase {
                name: "utf8-3",
                bytes: "€".as_bytes().to_vec(),
                decoded_key_bytes: Some(3),
                string_content_bytes: 3,
            },
            TokenCase {
                name: "utf8-4",
                bytes: "😀".as_bytes().to_vec(),
                decoded_key_bytes: Some(4),
                string_content_bytes: 4,
            },
            TokenCase {
                name: "combining-mark",
                bytes: "\u{301}".as_bytes().to_vec(),
                decoded_key_bytes: Some(2),
                string_content_bytes: 2,
            },
            TokenCase {
                name: "json-escape",
                bytes: br#"\n"#.to_vec(),
                decoded_key_bytes: Some(1),
                string_content_bytes: 2,
            },
            TokenCase {
                name: "surrogate-pair",
                bytes: br#"\uD83D\uDE00"#.to_vec(),
                decoded_key_bytes: Some(4),
                string_content_bytes: 12,
            },
            TokenCase {
                name: "partial-utf8",
                bytes: vec![0xf0, 0x9f],
                decoded_key_bytes: None,
                string_content_bytes: 2,
            },
            TokenCase {
                name: "cross-quote",
                bytes: b"a\"".to_vec(),
                decoded_key_bytes: Some(1),
                string_content_bytes: 1,
            },
            TokenCase {
                name: "cross-quote-colon",
                bytes: br#"a":"#.to_vec(),
                decoded_key_bytes: Some(1),
                string_content_bytes: 1,
            },
            TokenCase {
                name: "cross-quote-colon-value",
                bytes: br#"a":0"#.to_vec(),
                decoded_key_bytes: Some(1),
                string_content_bytes: 1,
            },
        ];
        let key_program = program(r#"{"type":"object","additionalProperties":true}"#);
        let any_string_program = program(r#"{"type":"object","unevaluatedProperties":true}"#);
        let canonical_program =
            program(r#"{"type":"array","uniqueItems":true,"items":{"type":"string"}}"#);
        let dependent_program = program(
            r#"{"type":"object","dependentSchemas":{"pa":{"type":"object"}},"additionalProperties":true}"#,
        );
        let property_names_program = program(
            r#"{"type":"object","propertyNames":{"type":"string","maxLength":64},"additionalProperties":true}"#,
        );
        let key_prefix = br#"{"p"#;
        let string_prefix = br#"{"s":"p"#;
        let canonical_prefix = br#"["p"#;

        for case in cases {
            let vocabulary = VocabularyHandle::new(Arc::new(
                build_vocabulary(7, FxHashMap::from_iter([(case.bytes.clone(), vec![0])])).unwrap(),
            ))
            .unwrap();
            let trie = TrieCache::new().bind(&vocabulary).unwrap();

            let document_exact = key_prefix.len() + case.bytes.len();
            for limit in around(document_exact) {
                assert_near_limit_three_routes(
                    &key_program,
                    &vocabulary,
                    &trie,
                    key_prefix,
                    NearLimitTuning::Document(limit),
                    &format!("max_document_bytes/{}/{limit}", case.name),
                );
            }

            if let Some(decoded) = case.decoded_key_bytes {
                let key_exact = 1 + decoded;
                for limit in around(key_exact) {
                    assert_near_limit_three_routes(
                        &key_program,
                        &vocabulary,
                        &trie,
                        key_prefix,
                        NearLimitTuning::Key(limit),
                        &format!("max_key_bytes/{}/{limit}", case.name),
                    );
                }
            }

            let string_exact = 1 + case.string_content_bytes;
            for limit in around(string_exact) {
                assert_near_limit_three_routes(
                    &any_string_program,
                    &vocabulary,
                    &trie,
                    string_prefix,
                    NearLimitTuning::String(limit),
                    &format!("max_string_bytes/{}/{limit}", case.name),
                );
            }

            let undo_exact = minimum_successful_limit(
                &key_program,
                &vocabulary,
                &trie,
                key_prefix,
                StructuredLimits::default().max_undo_entries,
                NearLimitTuning::Undo,
            );
            for limit in around(undo_exact) {
                assert_near_limit_three_routes(
                    &key_program,
                    &vocabulary,
                    &trie,
                    key_prefix,
                    NearLimitTuning::Undo(limit),
                    &format!("max_undo_entries/{}/{limit}", case.name),
                );
            }

            let session_exact = minimum_successful_limit(
                &key_program,
                &vocabulary,
                &trie,
                key_prefix,
                StructuredLimits::default().max_session_bytes,
                NearLimitTuning::Session,
            );
            for limit in around(session_exact) {
                assert_near_limit_three_routes(
                    &key_program,
                    &vocabulary,
                    &trie,
                    key_prefix,
                    NearLimitTuning::Session(limit),
                    &format!("max_session_bytes/{}/{limit}", case.name),
                );
            }

            let canonical_exact = minimum_successful_limit(
                &canonical_program,
                &vocabulary,
                &trie,
                canonical_prefix,
                StructuredLimits::default().max_unique_canonical_bytes,
                NearLimitTuning::Canonical,
            );
            for limit in around(canonical_exact) {
                assert_near_limit_three_routes(
                    &canonical_program,
                    &vocabulary,
                    &trie,
                    canonical_prefix,
                    NearLimitTuning::Canonical(limit),
                    &format!("canonical_uniqueItems_storage/{}/{limit}", case.name),
                );
            }

            for (storage_name, storage_program) in [
                ("dependent_observer_storage", &dependent_program),
                ("propertyNames_cursor_storage", &property_names_program),
            ] {
                let exact = minimum_successful_limit(
                    storage_program,
                    &vocabulary,
                    &trie,
                    key_prefix,
                    StructuredLimits::default().max_session_bytes,
                    NearLimitTuning::Session,
                );
                for limit in around(exact) {
                    assert_near_limit_three_routes(
                        storage_program,
                        &vocabulary,
                        &trie,
                        key_prefix,
                        NearLimitTuning::Session(limit),
                        &format!("{storage_name}/{}/{limit}", case.name),
                    );
                }
            }
        }
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn active_unique_items_canonical_builder_forces_exact_fallback() {
        let program = program(r#"{"type":"array","uniqueItems":true,"items":{"type":"string"}}"#);
        let vocabulary = VocabularyHandle::new(Arc::new(
            build_vocabulary(3, FxHashMap::from_iter([(b"abc".to_vec(), vec![0])])).unwrap(),
        ))
        .unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let mut matcher = program.new_matcher().unwrap();
        assert!(matcher.advance(br#"["p"#).unwrap());
        {
            let MatcherBackend::Incremental(inner) = &mut matcher.backend else {
                panic!("incremental matcher")
            };
            let (selected, _) = select_adaptive_trie(inner, &vocabulary, program.limits).unwrap();
            assert!(
                selected.is_none(),
                "an active canonical builder must conservatively use the exact walker"
            );
        }
        let mut adaptive = [0; 4];
        let mut full = [0; 4];
        let mut records = [0; 4];
        matcher
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut adaptive)
            .unwrap();
        matcher
            .write_full_trie_mask_for_bench(&vocabulary, &trie, &mut full)
            .unwrap();
        matcher
            .write_record_mask_le_bytes_into(
                vocabulary.mask_vocab_size(),
                vocabulary.iter_records(),
                &mut records,
            )
            .unwrap();
        assert_eq!(adaptive, full);
        assert_eq!(adaptive, records);
        assert_incremental_accounting(&matcher, "canonical exact fallback");
    }

    #[test]
    fn sliced_key_resource_errors_match_record_scan_and_clear_output() {
        let schemas = [
            r#"{"type":"object","additionalProperties":true}"#,
            r#"{"type":"object","unevaluatedProperties":true}"#,
            r#"{"allOf":[{"type":"object"},{"additionalProperties":true}]}"#,
            r#"{"type":"array","uniqueItems":true,"items":{"type":"object"}}"#,
            r#"{"type":"array","contains":{"type":"object"},"items":{"type":"object"}}"#,
        ];
        for schema in schemas {
            let prefix: &[u8] = if schema.contains("array") {
                br#"[{"a"#
            } else {
                br#"{"a"#
            };
            for budget in [2, 3, 4] {
                for document_limit in [false, true] {
                    let mut limits = StructuredLimits::default();
                    if document_limit {
                        limits.max_document_bytes = prefix.len() + budget - 1;
                    } else {
                        limits.max_key_bytes = budget;
                    }
                    let program =
                        StructuredProgram::compile_with_limits(ir(schema), limits).unwrap();
                    let mut tokens = FxHashMap::default();
                    tokens.insert(b"abc".to_vec(), vec![0, 1]);
                    let vocabulary =
                        VocabularyHandle::new(Arc::new(build_vocabulary(7, tokens).unwrap()))
                            .unwrap();
                    let trie = TrieCache::new().bind(&vocabulary).unwrap();
                    let mut fast = program.new_matcher().unwrap();
                    let mut scan = program.new_matcher().unwrap();
                    assert!(fast.advance(prefix).unwrap());
                    assert!(scan.advance(prefix).unwrap());
                    let mut fast_out = [0xa5; 4];
                    let mut scan_out = [0xa5; 4];
                    let observed =
                        fast.write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut fast_out);
                    let expected = scan.write_record_mask_le_bytes_into(
                        vocabulary.mask_vocab_size(),
                        vocabulary.iter_records(),
                        &mut scan_out,
                    );
                    assert_eq!(
                        observed.is_ok(),
                        expected.is_ok(),
                        "{schema}, budget={budget}, document={document_limit}"
                    );
                    assert_eq!(fast_out, scan_out);
                    if observed.is_err() {
                        assert_eq!(fast_out, [0; 4]);
                    } else if fast_out[0] & 1 != 0 {
                        assert!(fast.advance(b"abc").unwrap());
                    }
                }
            }
        }
    }

    #[test]
    fn wrapped_any_string_byte_limits_match_record_errors_and_advance() {
        for limit in [3, 4, 5] {
            let limits = StructuredLimits {
                max_string_bytes: limit,
                ..StructuredLimits::default()
            };
            let program = StructuredProgram::compile_with_limits(
                ir(r#"{"type":"object","unevaluatedProperties":true}"#),
                limits,
            )
            .unwrap();
            let mut tokens = FxHashMap::default();
            tokens.insert(b"abc".to_vec(), vec![0, 1]);
            let vocabulary =
                VocabularyHandle::new(Arc::new(build_vocabulary(7, tokens).unwrap())).unwrap();
            let trie = TrieCache::new().bind(&vocabulary).unwrap();
            let mut fast = program.new_matcher().unwrap();
            let mut scan = program.new_matcher().unwrap();
            assert!(fast.advance(br#"{"s":"a"#).unwrap());
            assert!(scan.advance(br#"{"s":"a"#).unwrap());
            let mut fast_out = [0xa5; 4];
            let mut scan_out = [0xa5; 4];
            let observed = fast.write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut fast_out);
            let expected = scan.write_record_mask_le_bytes_into(
                vocabulary.mask_vocab_size(),
                vocabulary.iter_records(),
                &mut scan_out,
            );
            assert_eq!(observed.is_ok(), expected.is_ok());
            assert_eq!(observed.is_ok(), limit >= 4);
            assert_eq!(fast_out, scan_out);
            if observed.is_ok() {
                assert!(fast_out[0] & 1 != 0);
                assert!(fast.advance(b"abc").unwrap());
            } else {
                assert_eq!(fast_out, [0; 4]);
                assert!(!fast.is_dead());
            }
        }
    }

    #[test]
    fn unique_item_canonical_limits_match_record_masks_and_advance() {
        let mut tokens = FxHashMap::default();
        tokens.insert(b"abc".to_vec(), vec![0, 1]);
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(7, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let mut observed_success = false;
        let mut observed_limit = false;
        for canonical_limit in [16, 32, 64, 128, 1024] {
            let program =
                program(r#"{"type":"array","uniqueItems":true,"items":{"type":"string"}}"#);
            let mut fast = program.new_matcher().unwrap();
            let mut scan = program.new_matcher().unwrap();
            assert!(fast.advance(b"[\"").unwrap());
            assert!(scan.advance(b"[\"").unwrap());
            match &mut fast.backend {
                MatcherBackend::Incremental(inner) => {
                    inner
                        .state
                        .set_unique_canonical_limit_for_test(canonical_limit);
                }
                #[cfg(feature = "bench-internals")]
                MatcherBackend::Reference(_) => panic!("incremental matcher"),
            }
            match &mut scan.backend {
                MatcherBackend::Incremental(inner) => {
                    inner
                        .state
                        .set_unique_canonical_limit_for_test(canonical_limit);
                }
                #[cfg(feature = "bench-internals")]
                MatcherBackend::Reference(_) => panic!("incremental matcher"),
            }
            let mut fast_out = [0xa5; 4];
            let mut scan_out = [0xa5; 4];
            let observed = fast.write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut fast_out);
            let expected = scan.write_record_mask_le_bytes_into(
                vocabulary.mask_vocab_size(),
                vocabulary.iter_records(),
                &mut scan_out,
            );
            assert_eq!(
                observed.is_ok(),
                expected.is_ok(),
                "limit={canonical_limit}"
            );
            assert_eq!(fast_out, scan_out, "limit={canonical_limit}");
            if observed.is_ok() && fast_out[0] & 1 != 0 {
                observed_success = true;
                assert!(fast.advance(b"abc").unwrap());
            }
            if observed.is_err() {
                observed_limit = true;
                assert_eq!(fast_out, [0; 4]);
                assert!(!fast.is_dead());
            }
        }
        assert!(observed_success);
        assert!(observed_limit);
    }

    #[test]
    fn session_limits_match_record_masks_and_advance() {
        let mut tokens = FxHashMap::default();
        tokens.insert(b"abcdefghijklmnopqrstuvwxyz0123456789".to_vec(), vec![0, 1]);
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(7, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let program = program(r#"{"type":"object","additionalProperties":true}"#);
        let mut observed_success = false;
        let mut observed_limit = false;
        for additional in [0, 1, 32, 65_536] {
            let mut fast = program.new_matcher().unwrap();
            let mut scan = program.new_matcher().unwrap();
            assert!(fast.advance(b"{\"").unwrap());
            assert!(scan.advance(b"{\"").unwrap());
            let limit = fast
                .retained_session_bytes()
                .checked_add(additional)
                .unwrap();
            match &mut fast.backend {
                MatcherBackend::Incremental(inner) => {
                    inner.state.set_session_limit_for_test(limit);
                }
                #[cfg(feature = "bench-internals")]
                MatcherBackend::Reference(_) => panic!("incremental matcher"),
            }
            match &mut scan.backend {
                MatcherBackend::Incremental(inner) => {
                    inner.state.set_session_limit_for_test(limit);
                }
                #[cfg(feature = "bench-internals")]
                MatcherBackend::Reference(_) => panic!("incremental matcher"),
            }
            let mut fast_out = [0xa5; 4];
            let mut scan_out = [0xa5; 4];
            let observed = fast.write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut fast_out);
            let expected = scan.write_record_mask_le_bytes_into(
                vocabulary.mask_vocab_size(),
                vocabulary.iter_records(),
                &mut scan_out,
            );
            assert_eq!(
                observed.is_ok(),
                expected.is_ok(),
                "additional={additional}"
            );
            assert_eq!(fast_out, scan_out, "additional={additional}");
            if observed.is_ok() && fast_out[0] & 1 != 0 {
                observed_success = true;
                assert!(fast
                    .advance(b"abcdefghijklmnopqrstuvwxyz0123456789")
                    .unwrap());
            }
            if observed.is_err() {
                observed_limit = true;
                assert_eq!(fast_out, [0; 4]);
                assert!(!fast.is_dead());
            }
        }
        assert!(observed_success);
        assert!(observed_limit);
    }

    #[test]
    #[cfg(feature = "bench-internals")]
    fn bounded_string_slice_fires_and_warm_retention_stabilizes() {
        let program =
            program(r#"{"type":"object","properties":{"s":{"type":"string","maxLength":3}}}"#);
        let tokens = (0..100u32)
            .map(|id| (format!("word{id}").into_bytes(), vec![id]))
            .chain([
                (b"a".to_vec(), vec![100]),
                (b"\"".to_vec(), vec![101]),
                ("\u{1f600}".as_bytes().to_vec(), vec![102]),
            ])
            .collect();
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(127, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let mut matcher = program.new_matcher().unwrap();
        assert!(matcher.advance(br#"{"s":"a"#).unwrap());
        let mut output = [0; 16];
        matcher
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
            .unwrap();
        let retained = match &matcher.backend {
            MatcherBackend::Incremental(inner) => inner.state.retained_bytes(),
            _ => panic!("incremental program"),
        };
        for _ in 0..8 {
            matcher
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
                .unwrap();
            assert!(
                matcher.mask_work_metrics_for_bench().trie_nodes
                    < trie.trie().node_count() as u64 / 2
            );
            let MatcherBackend::Incremental(inner) = &matcher.backend else {
                panic!("incremental program")
            };
            assert_eq!(inner.state.retained_bytes(), retained);
        }
    }

    #[test]
    #[cfg(feature = "bench-internals")]
    fn wrapper_slices_fire_and_match_both_exact_walks_after_warmup() {
        let cases = [
            (
                r#"{"type":"object","properties":{"mode":{"type":"string"},"text":{"type":"string"}},"dependentSchemas":{"mode":{"required":["text"],"properties":{"text":{"maxLength":12}}}}}"#,
                br#"{"mo"#.as_slice(),
            ),
            (
                r#"{"type":"object","properties":{"mode":{"type":"string"},"text":{"type":"string"}},"dependentSchemas":{"mode":{"required":["text"],"properties":{"text":{"maxLength":12}}}}}"#,
                br#"{"mode":"text","text":"rea"#.as_slice(),
            ),
            (
                r#"{"type":"object","properties":{"owner":{"type":"string","maxLength":8}},"allOf":[{"properties":{"team":{"type":"string","maxLength":8}}}],"unevaluatedProperties":false}"#,
                br#"{"owner":"ops","team":""#.as_slice(),
            ),
            (
                r#"{"type":"object","properties":{"owner":{"type":"string","maxLength":8}},"allOf":[{"properties":{"team":{"type":"string","maxLength":8}}}],"unevaluatedProperties":false}"#,
                br#"{"owner":"ops","team":"core"#.as_slice(),
            ),
            (
                r#"{"type":"object","properties":{"owner":{"type":"string","maxLength":8}},"allOf":[{"properties":{"team":{"type":"string","maxLength":8}}}],"unevaluatedProperties":false}"#,
                br#"{"owner":"o"#.as_slice(),
            ),
            (
                r#"{"type":"object","properties":{"owner":{"type":"string","maxLength":8}},"allOf":[{"properties":{"team":{"type":"string","maxLength":8}}}],"unevaluatedProperties":false}"#,
                br#"{"owner":"1234567"#.as_slice(),
            ),
            (
                r#"{"type":"object","unevaluatedProperties":true}"#,
                br#"{"a"#.as_slice(),
            ),
            (
                r#"{"allOf":[{"type":"object"},{"additionalProperties":true}]}"#,
                br#"{"a"#,
            ),
            (
                r##"{"$dynamicAnchor":"node","type":"object","properties":{"child":{"$dynamicRef":"#node"}},"additionalProperties":true}"##,
                br#"{"child":{"a"#,
            ),
            (
                r#"{"type":"array","items":{"type":"string"},"contains":{"type":"string"},"minContains":1}"#,
                br#"["rea"#,
            ),
            (
                r#"{"type":"array","items":{"type":"string"},"contains":{"const":"ok"},"minContains":2,"maxItems":2}"#,
                br#"["o"#,
            ),
            (
                r#"{"type":"array","contains":{"type":"string"},"unevaluatedItems":false}"#,
                br#"["rea"#,
            ),
            (
                r#"{"type":"array","items":{"type":"string"},"contains":{"type":"string","pattern":"^ok"},"unevaluatedItems":false}"#,
                br#"["#,
            ),
            (
                r#"{"type":"array","prefixItems":[{"type":"string","maxLength":12}],"contains":{"type":"string","pattern":"^ok"},"unevaluatedItems":false}"#,
                br#"["#,
            ),
        ];
        let tokens = (0..100u32)
            .map(|id| (format!("word{id}").into_bytes(), vec![id]))
            .chain([
                (b"\"".to_vec(), vec![100]),
                (br#"\u0061"#.to_vec(), vec![101]),
                (vec![0xf0, 0x9f], vec![102]),
            ])
            .collect();
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(127, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        for (schema, prefix) in cases {
            let program = program(schema);
            let mut matcher = program.new_matcher().unwrap();
            assert!(matcher.advance(prefix).unwrap());
            let mut fast = [0; 16];
            let mut full = [0; 16];
            let mut scan = [0; 16];
            matcher
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut fast)
                .unwrap();
            assert!(
                matcher.mask_work_metrics_for_bench().trie_nodes
                    < trie.trie().node_count() as u64 / 2,
                "cold {schema}"
            );
            let cold = fast;
            matcher
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut fast)
                .unwrap();
            assert!(
                matcher.mask_work_metrics_for_bench().trie_nodes
                    < trie.trie().node_count() as u64 / 2,
                "{schema}"
            );
            let MatcherBackend::Incremental(inner) = &mut matcher.backend else {
                panic!("incremental program")
            };
            walk_full_trie_mask_for_test(inner, &vocabulary, &trie, &mut full, program.limits)
                .unwrap();
            matcher
                .write_record_mask_le_bytes_into(
                    vocabulary.mask_vocab_size(),
                    vocabulary.iter_records(),
                    &mut scan,
                )
                .unwrap();
            assert_eq!(fast, full, "{schema}");
            assert_eq!(fast, scan, "{schema}");
            assert_eq!(cold, full, "cold {schema}");
        }
    }

    #[test]
    #[cfg(feature = "bench-internals")]
    fn safe_interior_array_certificate_matrix_matches_all_routes_and_advances() {
        #[derive(Clone, Copy)]
        struct Case {
            name: &'static str,
            schema: &'static str,
            prefix: &'static [u8],
            fast: bool,
        }
        let cases = [
            Case {
                name: "empty-array-structural-position",
                schema: r#"{"type":"array","items":{"type":"string"}}"#,
                prefix: b"[",
                fast: false,
            },
            Case {
                name: "prefix-item",
                schema: r#"{"prefixItems":[{"type":"string"}],"items":false}"#,
                prefix: br#"["pre"#,
                fast: true,
            },
            Case {
                name: "prefix-to-tail",
                schema: r#"{"prefixItems":[{"type":"string"}],"items":{"type":"string"}}"#,
                prefix: br#"["a","ta"#,
                fast: true,
            },
            Case {
                name: "nested-array-item",
                schema: r#"{"type":"array","items":{"type":"array","items":{"type":"string"}}}"#,
                prefix: br#"[["ne"#,
                fast: true,
            },
            Case {
                name: "contains-observer",
                schema: r#"{"type":"array","items":{"type":"string"},"contains":{"type":"string","pattern":"^ok"},"minContains":1}"#,
                prefix: br#"["ok"#,
                fast: true,
            },
            Case {
                name: "unevaluated-items-observer",
                schema: r#"{"prefixItems":[true],"contains":{"type":"string"},"unevaluatedItems":false}"#,
                prefix: br#"[1,"ok"#,
                fast: true,
            },
            Case {
                name: "maximum-items-interior",
                schema: r#"{"type":"array","items":{"type":"string"},"maxItems":2}"#,
                prefix: br#"["a"#,
                fast: true,
            },
            Case {
                name: "canonical-unique-items-fallback",
                schema: r#"{"type":"array","items":{"type":"string"},"uniqueItems":true}"#,
                prefix: br#"["a"#,
                fast: false,
            },
        ];
        let mut tokens = (0..100u32)
            .map(|id| (format!("word{id}").into_bytes(), vec![id]))
            .collect::<FxHashMap<_, _>>();
        for (id, bytes) in [
            (100, b"\"".as_slice()),
            (101, b"\","),
            (102, b"\"]"),
            (103, b","),
            (104, b"]"),
            (105, br#"\u0061"#),
            (106, b"\xf0\x9f"),
        ] {
            tokens.insert(bytes.to_vec(), vec![id]);
        }
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(127, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let output_len = vocabulary.mask_vocab_size().div_ceil(32) * 4;

        for case in cases {
            let program = program(case.schema);
            let mut adaptive = program.new_matcher().unwrap();
            let mut full = program.new_matcher().unwrap();
            let mut record = program.new_matcher().unwrap();
            for matcher in [&mut adaptive, &mut full, &mut record] {
                assert!(matcher.advance(case.prefix).unwrap(), "{}", case.name);
            }
            {
                let MatcherBackend::Incremental(inner) = &mut adaptive.backend else {
                    panic!("incremental matcher")
                };
                let (selected, _) =
                    select_adaptive_trie(inner, &vocabulary, program.limits).unwrap();
                assert_eq!(selected.is_some(), case.fast, "{}", case.name);
            }
            let mut adaptive_out = vec![0; output_len];
            let mut full_out = vec![0; output_len];
            let mut record_out = vec![0; output_len];
            adaptive
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut adaptive_out)
                .unwrap();
            if case.fast {
                assert!(
                    adaptive.mask_work_metrics_for_bench().trie_nodes
                        < trie.trie().node_count() as u64,
                    "{} fast path did not fire",
                    case.name
                );
            }
            {
                let MatcherBackend::Incremental(inner) = &mut full.backend else {
                    panic!("incremental matcher")
                };
                select_adaptive_trie(inner, &vocabulary, program.limits).unwrap();
                walk_full_trie_mask_for_test(
                    inner,
                    &vocabulary,
                    &trie,
                    &mut full_out,
                    program.limits,
                )
                .unwrap();
            }
            {
                let MatcherBackend::Incremental(inner) = &mut record.backend else {
                    panic!("incremental matcher")
                };
                select_adaptive_trie(inner, &vocabulary, program.limits).unwrap();
            }
            record
                .write_record_mask_le_bytes_into(
                    vocabulary.mask_vocab_size(),
                    vocabulary.iter_records(),
                    &mut record_out,
                )
                .unwrap();
            assert_mask_bytes_equal(&adaptive_out, &full_out, &vocabulary, case.name);
            assert_mask_bytes_equal(&adaptive_out, &record_out, &vocabulary, case.name);

            for (candidate, ids) in vocabulary.iter_records() {
                if !ids
                    .iter()
                    .any(|id| adaptive_out[*id as usize / 8] & (1 << (*id % 8)) != 0)
                {
                    continue;
                }
                let mut equivalent = program.new_matcher().unwrap();
                assert!(equivalent.advance(case.prefix).unwrap(), "{}", case.name);
                let mut prepared = vec![0; output_len];
                equivalent
                    .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut prepared)
                    .unwrap();
                assert_eq!(prepared, adaptive_out, "{}", case.name);
                assert!(
                    equivalent.advance(candidate).unwrap(),
                    "{} emitted {ids:?}",
                    case.name
                );
                assert_incremental_accounting(&equivalent, case.name);
            }
            for matcher in [&adaptive, &full, &record] {
                assert_incremental_accounting(matcher, case.name);
            }
        }

        let tail = cases
            .iter()
            .find(|case| case.name == "prefix-to-tail")
            .unwrap();
        let program = program(tail.schema);
        let mut matcher = program.new_matcher().unwrap();
        assert!(matcher.advance(tail.prefix).unwrap());
        let mut output = vec![0; output_len];
        matcher
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
            .unwrap();
        assert_ne!(output[101 / 8] & (1 << (101 % 8)), 0);
    }

    #[test]
    #[cfg(feature = "bench-internals")]
    fn combined_unevaluated_certificate_matches_all_routes_and_advances() {
        let schema = r#"{
            "type":"object",
            "properties":{
                "mode":{"enum":["fast","slow"]},
                "count":{"type":"integer","minimum":0,"maximum":9},
                "code":{"type":"string","pattern":"^[A-Z]{2}[0-9]{2}$"},
                "items":{"type":"array","items":{"type":"string"},"contains":{"const":"required"},"uniqueItems":true,"minItems":1,"maxItems":3},
                "meta":{"type":"object","propertyNames":{"pattern":"^[a-z]+$"},"patternProperties":{"^[a-z]+$":{"type":"string"}},"additionalProperties":false}
            },
            "required":["mode","count","code","items","meta"],
            "dependentRequired":{"mode":["code"]},
            "if":{"properties":{"mode":{"const":"fast"}},"required":["mode"]},
            "then":{"properties":{"count":{"maximum":5}}},
            "else":{"properties":{"count":{"minimum":6}}},
            "allOf":[
                {"if":{"properties":{"code":{"pattern":"^OK"}},"required":["code"]},"then":{"properties":{"meta":{"minProperties":1}}},"else":{"properties":{"meta":{"maxProperties":2}}}},
                {"if":{"properties":{"count":{"maximum":5}},"required":["count"]},"then":{"properties":{"items":{"maxItems":3}}},"else":{"properties":{"items":{"minItems":2}}}}
            ],
            "unevaluatedProperties":false
        }"#;
        let tokens = (0..100u32)
            .map(|id| (format!("word{id}").into_bytes(), vec![id]))
            .chain([(b"\"".to_vec(), vec![100])])
            .collect();
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(127, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let program = program(schema);
        let prefix = br#"{"mo"#;
        let mut adaptive = program.new_matcher().unwrap();
        let mut full = program.new_matcher().unwrap();
        let mut record = program.new_matcher().unwrap();
        for matcher in [&mut adaptive, &mut full, &mut record] {
            assert!(matcher.advance(prefix).unwrap());
        }
        let mut adaptive_out = [0; 16];
        let mut full_out = [0; 16];
        let mut record_out = [0; 16];
        adaptive
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut adaptive_out)
            .unwrap();
        assert!(
            adaptive.mask_work_metrics_for_bench().trie_nodes < trie.trie().node_count() as u64 / 2,
            "combined unevaluated certificate did not activate"
        );
        {
            let MatcherBackend::Incremental(inner) = &mut full.backend else {
                panic!("incremental matcher")
            };
            select_adaptive_trie(inner, &vocabulary, program.limits).unwrap();
            walk_full_trie_mask_for_test(inner, &vocabulary, &trie, &mut full_out, program.limits)
                .unwrap();
        }
        {
            let MatcherBackend::Incremental(inner) = &mut record.backend else {
                panic!("incremental matcher")
            };
            select_adaptive_trie(inner, &vocabulary, program.limits).unwrap();
        }
        record
            .write_record_mask_le_bytes_into(
                vocabulary.mask_vocab_size(),
                vocabulary.iter_records(),
                &mut record_out,
            )
            .unwrap();
        assert_mask_bytes_equal(
            &adaptive_out,
            &full_out,
            &vocabulary,
            "combined unevaluated",
        );
        assert_mask_bytes_equal(
            &adaptive_out,
            &record_out,
            &vocabulary,
            "combined unevaluated",
        );

        let first = adaptive_out;
        adaptive
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut adaptive_out)
            .unwrap();
        assert_eq!(adaptive_out, first, "speculative rollback changed the mask");
        for (candidate, ids) in vocabulary.iter_records() {
            if !ids
                .iter()
                .any(|id| adaptive_out[*id as usize / 8] & (1 << (*id % 8)) != 0)
            {
                continue;
            }
            let mut equivalent = program.new_matcher().unwrap();
            assert!(equivalent.advance(prefix).unwrap());
            let mut prepared = [0; 16];
            equivalent
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut prepared)
                .unwrap();
            assert_eq!(prepared, adaptive_out);
            assert!(equivalent.advance(candidate).unwrap(), "emitted {ids:?}");
            assert_incremental_accounting(&equivalent, "combined unevaluated advance");
        }
        for matcher in [&adaptive, &full, &record] {
            assert_incremental_accounting(matcher, "combined unevaluated route");
        }
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn lexical_prefix_walk_matches_exact_routes_and_every_emitted_token_advances() {
        let tokens = FxHashMap::from_iter([
            (b"abc\"".to_vec(), vec![0]),
            (b"abc\"}".to_vec(), vec![1]),
            (b"ab\\u0063\"}".to_vec(), vec![2]),
            ("é\"}".as_bytes().to_vec(), vec![3]),
            (vec![0xf0, 0x9f], vec![4]),
            (vec![0xf0, 0x9f, b'"'], vec![5]),
            (b"ab\n\"".to_vec(), vec![6]),
            (b"\\\"still".to_vec(), vec![7]),
            (b"\"}".to_vec(), vec![8]),
        ]);
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(31, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let program = program(
            r#"{"type":"object","properties":{"note":{"type":"string"}},"required":["note"]}"#,
        );
        let prefix = br#"{"note":""#;
        let mut adaptive = program.new_matcher().unwrap();
        let mut full = program.new_matcher().unwrap();
        let mut record = program.new_matcher().unwrap();
        for matcher in [&mut adaptive, &mut full, &mut record] {
            assert!(matcher.advance(prefix).unwrap());
        }
        if let MatcherBackend::Incremental(inner) = &mut adaptive.backend {
            let mark = inner.state.checkpoint();
            for &byte in b"abc\"" {
                assert!(inner.state.try_push_byte_for_mask(byte).unwrap());
            }
            assert!(inner.state.next_byte_is_lexically_impossible(b'a'));
            assert!(!inner.state.next_byte_is_lexically_impossible(b'}'));
            inner.state.rollback(mark);
        }
        let mut adaptive_out = [0u8; 4];
        let mut full_out = [0u8; 4];
        let mut record_out = [0u8; 4];
        adaptive
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut adaptive_out)
            .unwrap();
        let metrics = adaptive.mask_work_metrics_for_bench();
        assert!(metrics.local_dfa_transitions > 0);
        let (event_capacity, path_capacity) = match &adaptive.backend {
            MatcherBackend::Incremental(inner) => {
                (inner.events.capacity(), inner.lexical_path.capacity())
            }
            MatcherBackend::Reference(_) => panic!("incremental matcher"),
        };
        let MatcherBackend::Incremental(inner) = &mut full.backend else {
            panic!("incremental matcher")
        };
        walk_full_trie_mask_for_test(inner, &vocabulary, &trie, &mut full_out, program.limits)
            .unwrap();
        record
            .write_record_mask_le_bytes_into(
                vocabulary.mask_vocab_size(),
                vocabulary.iter_records(),
                &mut record_out,
            )
            .unwrap();
        assert_eq!(adaptive_out, full_out);
        assert_eq!(adaptive_out, record_out);

        adaptive
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut adaptive_out)
            .unwrap();
        let MatcherBackend::Incremental(inner) = &adaptive.backend else {
            panic!("incremental matcher")
        };
        assert_eq!(inner.events.capacity(), event_capacity);
        assert_eq!(inner.lexical_path.capacity(), path_capacity);

        for (bytes, ids) in vocabulary.iter_records() {
            if !ids
                .iter()
                .any(|id| adaptive_out[*id as usize / 8] & (1 << (*id % 8)) != 0)
            {
                continue;
            }
            let mut equivalent = program.new_matcher().unwrap();
            assert!(equivalent.advance(prefix).unwrap());
            let mut prepared = [0u8; 4];
            equivalent
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut prepared)
                .unwrap();
            assert_eq!(prepared, adaptive_out);
            assert!(equivalent.advance(bytes).unwrap(), "emitted token {ids:?}");
        }
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn contains_product_uncertain_states_use_the_exact_walker() {
        let cases = [
            (
                "fresh-item",
                r#"{"type":"array","items":{"type":"string"},"contains":{"type":"string"},"minContains":1}"#,
                b"[".as_slice(),
            ),
            (
                "at-maximum",
                r#"{"type":"array","items":true,"contains":{"type":"string"},"minContains":0,"maxContains":1}"#,
                br#"["ok","a"#,
            ),
            (
                "multiple-structured-candidates",
                r#"{"type":"array","maxItems":1,"items":{"type":"array","maxItems":1,"items":{"type":"string"},"contains":{"type":"string","pattern":"^a"},"minContains":1},"contains":{"type":"array","contains":{"type":"string","pattern":"^a"},"minContains":1},"minContains":1}"#,
                br#"[["a"#,
            ),
        ];
        let vocabulary = VocabularyHandle::new(Arc::new(
            build_vocabulary(
                8,
                FxHashMap::from_iter([
                    (b"word".to_vec(), vec![0]),
                    (b"x".to_vec(), vec![1]),
                    (b"\"".to_vec(), vec![2]),
                ]),
            )
            .unwrap(),
        ))
        .unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        for (name, schema, prefix) in cases {
            let program = program(schema);
            let mut matcher = program.new_matcher().unwrap();
            assert!(matcher.advance(prefix).unwrap(), "{name}");
            {
                let MatcherBackend::Incremental(inner) = &mut matcher.backend else {
                    panic!("incremental matcher")
                };
                let (selected, _) =
                    select_adaptive_trie(inner, &vocabulary, program.limits).unwrap();
                assert!(selected.is_none(), "{name} must remain exact");
            }
            let mut adaptive = [0; 4];
            let mut full = [0; 4];
            let mut record = [0; 4];
            matcher
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut adaptive)
                .unwrap();
            matcher
                .write_full_trie_mask_for_bench(&vocabulary, &trie, &mut full)
                .unwrap();
            matcher
                .write_record_mask_le_bytes_into(
                    vocabulary.mask_vocab_size(),
                    vocabulary.iter_records(),
                    &mut record,
                )
                .unwrap();
            assert_eq!(adaptive, full, "{name}");
            assert_eq!(adaptive, record, "{name}");
            assert_incremental_accounting(&matcher, name);
        }
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn invalid_body_prefixes_skip_the_parser_only_with_a_cold_rejection_proof() {
        let tokens = (0..100u32)
            .map(|id| (format!("prefix{id}\n").into_bytes(), vec![id]))
            .chain([
                (b"a".to_vec(), vec![100]),
                (b"a\"\n".to_vec(), vec![101]),
                (b"\xf0\x9f".to_vec(), vec![102]),
            ])
            .collect();
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(127, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        for (schema, prefix) in [
            (
                r#"{"type":"object","unevaluatedProperties":true}"#,
                br#"{"a"#.as_slice(),
            ),
            (
                r#"{"type":"object","properties":{"s":{"type":"string"}}}"#,
                br#"{"s":"a"#.as_slice(),
            ),
        ] {
            let program = program(schema);
            let mut matcher = program.new_matcher().unwrap();
            assert!(matcher.advance(prefix).unwrap());
            let mut fast = [0; 16];
            let mut full = [0; 16];
            let mut scan = [0; 16];
            matcher
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut fast)
                .unwrap();
            assert!(
                matcher.mask_work_metrics_for_bench().trie_nodes < 10,
                "{schema}"
            );
            assert!(
                matcher.mask_work_metrics_for_bench().trie_nodes < trie.trie().node_count() as u64
            );
            let MatcherBackend::Incremental(inner) = &mut matcher.backend else {
                panic!("incremental program")
            };
            walk_full_trie_mask_for_test(inner, &vocabulary, &trie, &mut full, program.limits)
                .unwrap();
            matcher
                .write_record_mask_le_bytes_into(
                    vocabulary.mask_vocab_size(),
                    vocabulary.iter_records(),
                    &mut scan,
                )
                .unwrap();
            assert_eq!(fast, full, "{schema}");
            assert_eq!(fast, scan, "{schema}");
            assert_ne!(
                fast[101 / 8] & (1 << (101 % 8)),
                0,
                "whitespace after closing quote"
            );
        }
    }

    #[test]
    fn invalid_body_filter_preserves_errors_before_lexical_rejection() {
        let tokens = [(b"abc\n".to_vec(), vec![0])].into_iter().collect();
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(7, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let prefix = br#"{"a"#;
        for remaining in 1..=5 {
            let limits = StructuredLimits {
                max_document_bytes: prefix.len() + remaining,
                ..StructuredLimits::default()
            };
            let program = StructuredProgram::compile_with_limits(
                ir(r#"{"type":"object","unevaluatedProperties":true}"#),
                limits,
            )
            .unwrap();
            let mut fast = program.new_matcher().unwrap();
            let mut scan = program.new_matcher().unwrap();
            assert!(fast.advance(prefix).unwrap());
            assert!(scan.advance(prefix).unwrap());
            let mut fast_out = [0xff; 4];
            let mut scan_out = [0xff; 4];
            let observed = fast.write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut fast_out);
            let expected = scan.write_record_mask_le_bytes_into(
                vocabulary.mask_vocab_size(),
                vocabulary.iter_records(),
                &mut scan_out,
            );
            assert_eq!(observed.is_ok(), expected.is_ok(), "remaining={remaining}");
            assert_eq!(fast_out, scan_out, "remaining={remaining}");
            assert_eq!(fast_out, [0; 4]);
        }
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn escaped_and_partial_body_slice_fires_on_a_cold_wrapper() {
        let tokens = (0..100u32)
            .map(|id| (format!("\\u0061{id}").into_bytes(), vec![id]))
            .chain([
                (b"\"".to_vec(), vec![100]),
                (vec![0xf0, 0x9f], vec![101]),
                (b"\"\n".to_vec(), vec![102]),
                (b"\xff".to_vec(), vec![103]),
                (b"plain".to_vec(), vec![104]),
            ])
            .collect();
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(127, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let program = program(r#"{"type":"object","unevaluatedProperties":true}"#);
        let mut matcher = program.new_matcher().unwrap();
        assert!(matcher.advance(br#"{"a"#).unwrap());
        let mut fast = [0; 16];
        let mut full = [0; 16];
        let mut scan = [0; 16];
        matcher
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut fast)
            .unwrap();
        assert!(matcher.mask_work_metrics_for_bench().trie_nodes < 20);
        assert!(matcher.mask_work_metrics_for_bench().trie_nodes < trie.trie().node_count() as u64);
        let MatcherBackend::Incremental(inner) = &mut matcher.backend else {
            panic!("incremental program")
        };
        walk_full_trie_mask_for_test(inner, &vocabulary, &trie, &mut full, program.limits).unwrap();
        matcher
            .write_record_mask_le_bytes_into(
                vocabulary.mask_vocab_size(),
                vocabulary.iter_records(),
                &mut scan,
            )
            .unwrap();
        assert_eq!(fast, full);
        assert_eq!(fast, scan);
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn adaptive_slice_removes_at_least_eighty_percent_of_string_edges() {
        let program = program(
            r#"{"allOf":[{"type":"string","pattern":"^a"},{"type":"string","pattern":"z$"}]}"#,
        );
        let mut tokens = FxHashMap::default();
        for id in 0..512u32 {
            tokens.insert(format!("token{id:04}").into_bytes(), vec![id]);
        }
        tokens.insert(b"\"".to_vec(), vec![600]);
        tokens.insert(b"\\".to_vec(), vec![601]);
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(700, tokens).expect("vocabulary")))
                .expect("handle");
        let trie = TrieCache::new().bind(&vocabulary).expect("trie");
        let mut adaptive = program.new_matcher().expect("adaptive");
        let mut full = program.new_matcher().expect("full");
        assert!(adaptive.advance(b"\"a").expect("prefix"));
        assert!(full.advance(b"\"a").expect("prefix"));
        let bytes = vocabulary.mask_vocab_size().div_ceil(32) * 4;
        let mut adaptive_out = vec![0u8; bytes];
        let mut full_out = vec![0u8; bytes];
        adaptive
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut adaptive_out)
            .expect("adaptive warmup");
        adaptive_out.fill(0);
        adaptive
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut adaptive_out)
            .expect("adaptive mask");
        let adaptive_metrics = adaptive.mask_work_metrics_for_bench();
        let incremental = match &mut full.backend {
            MatcherBackend::Incremental(incremental) => incremental,
            MatcherBackend::Reference(_) => panic!("expected incremental matcher"),
        };
        incremental.reset_mask_metrics();
        walk_full_trie_mask_for_test(
            incremental,
            &vocabulary,
            &trie,
            &mut full_out,
            program.limits,
        )
        .expect("full mask");
        incremental.finish_mask_metrics();
        let full_metrics = incremental.mask_work_metrics;
        assert_mask_bytes_equal(&adaptive_out, &full_out, &vocabulary, "adaptive/full");
        assert!(
            adaptive_metrics.trie_edges.saturating_mul(5) <= full_metrics.trie_edges,
            "adaptive={} full={}",
            adaptive_metrics.trie_edges,
            full_metrics.trie_edges
        );
    }

    #[test]
    fn concurrent_matchers_share_catalogue_without_sharing_state() {
        let program = program(r#"{"type":"string"}"#);
        let mut tokens = FxHashMap::default();
        tokens.insert(b"a".to_vec(), vec![0]);
        tokens.insert(b"b".to_vec(), vec![1]);
        tokens.insert(b"\"".to_vec(), vec![2]);
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(3, tokens).expect("vocabulary")))
                .expect("handle");
        let trie = TrieCache::new().bind(&vocabulary).expect("trie");
        let expected = {
            let mut matcher = program.new_matcher().expect("expected matcher");
            assert!(matcher.advance(b"\"").expect("prefix"));
            let mut output = vec![0u8; vocabulary.mask_vocab_size().div_ceil(32) * 4];
            matcher
                .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
                .expect("expected mask");
            output
        };
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let program = program.clone();
                let vocabulary = vocabulary.clone();
                let trie = trie.clone();
                std::thread::spawn(move || {
                    let mut matcher = program.new_matcher().expect("thread matcher");
                    assert!(matcher.advance(b"\"").expect("thread prefix"));
                    let mut output = vec![0u8; vocabulary.mask_vocab_size().div_ceil(32) * 4];
                    matcher
                        .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
                        .expect("thread mask");
                    output
                })
            })
            .collect();
        for thread in threads {
            assert_eq!(thread.join().expect("thread"), expected);
        }
    }

    fn assert_mask_bytes_equal(
        left: &[u8],
        right: &[u8],
        vocabulary: &VocabularyHandle,
        label: &str,
    ) {
        if left == right {
            return;
        }
        let differences: Vec<_> = (0..vocabulary.mask_vocab_size())
            .filter(|id| left[id / 8] & (1 << (id % 8)) != right[id / 8] & (1 << (id % 8)))
            .map(|id| (id, vocabulary.token_bytes(u32::try_from(id).unwrap())))
            .collect();
        panic!("{label} mismatch: {differences:?}");
    }

    #[test]
    fn vocabulary_a_with_trie_b_is_artifact_mismatch_and_leaves_output_zeroed() {
        let program = program(r#"{"type":"object","additionalProperties":{"type":"boolean"}}"#);
        let mut tokens_a = FxHashMap::default();
        tokens_a.insert(b"{".to_vec(), vec![0]);
        tokens_a.insert(b"}".to_vec(), vec![1]);
        let mut tokens_b = FxHashMap::default();
        tokens_b.insert(b"{".to_vec(), vec![0]);
        tokens_b.insert(b"\"".to_vec(), vec![1]);
        tokens_b.insert(b"}".to_vec(), vec![2]);
        let handle_a =
            VocabularyHandle::new(Arc::new(build_vocabulary(3, tokens_a).expect("vocab a")))
                .expect("handle a");
        let handle_b =
            VocabularyHandle::new(Arc::new(build_vocabulary(4, tokens_b).expect("vocab b")))
                .expect("handle b");
        let trie_b = TrieCache::new().bind(&handle_b).expect("trie b");
        let mut matcher = program.new_matcher().expect("matcher");
        let mut output = vec![0xff; handle_a.mask_vocab_size().div_ceil(32) * 4];
        let error = matcher
            .write_mask_le_bytes_into(&handle_a, Some(&trie_b), &mut output)
            .expect_err("vocab a paired with trie b must mismatch");
        assert!(matches!(
            error,
            StructuredMatcherError::Compile(CompileError {
                code: ErrorCode::ArtifactMismatch,
                ..
            })
        ));
        assert!(
            output.iter().all(|&b| b == 0),
            "output must be cleared on mismatch"
        );
    }

    #[test]
    fn direct_output_rejects_a_wrong_length_before_mutation() {
        let program = program(r#"{"type":"string"}"#);
        let vocabulary = Arc::new(build_vocabulary(1, FxHashMap::default()).expect("vocabulary"));
        let handle = VocabularyHandle::new(vocabulary).expect("handle");
        let trie = TrieCache::new().bind(&handle).expect("trie");
        let mut matcher = program.new_matcher().expect("matcher");
        let mut output = [0xa5u8; 2];
        let error = matcher
            .write_mask_le_bytes_into(&handle, Some(&trie), &mut output)
            .expect_err("wrong output length");
        assert!(matches!(
            error,
            StructuredMatcherError::Compile(CompileError {
                code: ErrorCode::ArtifactOutOfBounds,
                ..
            })
        ));
        assert_eq!(output, [0xa5; 2]);
    }

    fn assert_combinator_prefixes(schema: &str, document: &[u8], valid: bool) -> usize {
        let schema_ir = ir(schema);
        let program = StructuredProgram::compile(schema_ir.clone()).expect("program");
        assert!(program.uses_incremental_backend(), "{schema}");
        let mut matcher = program.new_matcher().expect("matcher");
        let mut prefix = Vec::new();
        for &byte in document {
            let before_accepting = matcher.is_accepting();
            let before_dead = matcher.is_dead();
            let mut candidate = prefix.clone();
            candidate.push(byte);
            let allowed = matcher.advance(&[byte]).expect("incremental byte");
            let expected = super::super::reference::accepts(&schema_ir, &candidate)
                || super::super::reference::can_continue(&schema_ir, &candidate);
            assert_eq!(allowed, expected, "prefix {candidate:?}, schema {schema}");
            if !allowed {
                assert_eq!(matcher.is_accepting(), before_accepting);
                assert_eq!(matcher.is_dead(), before_dead);
                let records = [([byte], [0u32])];
                let mask = matcher
                    .allowed_mask_from_records(
                        1,
                        records
                            .iter()
                            .map(|(bytes, ids)| (bytes.as_slice(), ids.as_slice())),
                    )
                    .expect("rejected-byte mask");
                assert!(!mask.get(TokenId(0)));
                return candidate.len();
            }
            prefix.push(byte);
            assert_eq!(
                matcher.is_accepting(),
                super::super::reference::accepts(&schema_ir, &prefix),
                "acceptance at {prefix:?}, schema {schema}"
            );
        }
        assert_eq!(matcher.is_accepting(), valid, "{document:?}, {schema}");
        prefix.len()
    }

    #[test]
    fn non_regular_combinator_hard_correctness_matrix_matches_reference_at_every_prefix() {
        type CombinatorCase<'a> = (&'a str, &'a [(&'a [u8], bool)]);
        let cases: &[CombinatorCase<'_>] = &[
            (
                r#"{"allOf":[{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":true},{"type":"object","patternProperties":{"^x":{"type":"string"}},"additionalProperties":true}]}"#,
                &[
                    (r#"{"xé":"λ","id":1}"#.as_bytes(), true),
                    (br#"{"id":1,"x":"\u03bb"}"#, true),
                ],
            ),
            (
                r#"{"allOf":[{"type":"array","items":{"type":"integer"},"contains":{"const":2},"minContains":1},{"type":"array","uniqueItems":true,"minItems":2}]}"#,
                &[(b"[1,2]", true), (b"[1,3]", false)],
            ),
            (
                r#"{"anyOf":[{"enum":["aaaaaaaaab","aaaaaaaaac"]},{"type":"string","pattern":"^aaaaaaaaad+$"},{"type":"array","items":{"type":"integer"},"uniqueItems":true}]}"#,
                &[
                    (br#""aaaaaaaaac""#, true),
                    (br#""aaaaaaaaadd""#, true),
                    (b"[1,2]", true),
                    (br#""aaaaaaaaa""#, false),
                ],
            ),
            (
                r#"{"oneOf":[{"const":1},{"type":"number"},{"type":"object","additionalProperties":{"type":"integer"}}]}"#,
                &[(b"1", false), (b"12", true), (br#"{"x":1}"#, true)],
            ),
            (
                r#"{"oneOf":[{"type":"string","pattern":"^.*$"},{"type":"string","pattern":"^.*$"},{"type":"array","uniqueItems":true}]}"#,
                &[(br#""a""#, false), (b"[]", true)],
            ),
            (
                r#"{"allOf":[{"anyOf":[{"oneOf":[{"type":"object","required":["a"],"additionalProperties":true},{"type":"array","uniqueItems":true}]},{"type":"null"}]},{"anyOf":[{"type":"object","additionalProperties":true},{"type":"array","contains":{"const":1}},{"type":"null"}]}]}"#,
                &[
                    (r#" { "a" : "λ" } "#.as_bytes(), true),
                    (b" [ 1 ] ", true),
                    (b"null", true),
                    (br#"{}"#, false),
                ],
            ),
            (
                r##"{"$defs":{"node":{"anyOf":[{"type":"null"},{"type":"object","properties":{"value":{"type":"integer"},"next":{"$ref":"#/$defs/node"}},"required":["value","next"],"additionalProperties":false}]}},"$ref":"#/$defs/node"}"##,
                &[
                    (b"null", true),
                    (br#"{"value":1,"next":{"value":2,"next":null}}"#, true),
                    (br#"{"value":1,"next":]"#, false),
                ],
            ),
        ];
        for (schema, documents) in cases {
            for (document, valid) in *documents {
                assert_combinator_prefixes(schema, document, *valid);
            }
        }
    }

    #[test]
    fn negation_and_conditional_prefixes_match_the_reference() {
        type Case<'a> = (&'a str, &'a [(&'a [u8], bool)]);
        let cases: &[Case<'_>] = &[
            (
                r#"{"not":{"const":1}}"#,
                &[(b"1", false), (b"12", true), (b"2", true)],
            ),
            (
                r#"{"not":{"enum":[null,true,"x"]}}"#,
                &[(b"null", false), (b"false", true), (br#""x""#, false)],
            ),
            (
                r#"{"not":{"type":"object","required":["x"],"additionalProperties":true}}"#,
                &[(br#"{"x":1}"#, false), (br#"{"y":1}"#, true), (b"[]", true)],
            ),
            (
                r#"{"not":{"type":"array","uniqueItems":true}}"#,
                &[(b"[1,2]", false), (b"[1,1]", true), (b"{}", true)],
            ),
            (
                r#"{"not":{"type":"array","contains":{"const":1}}}"#,
                &[(b"[1]", false), (b"[2]", true), (b"{}", true)],
            ),
            (
                r#"{"not":{"type":"number","minimum":-1,"maximum":2}}"#,
                &[(b"-0", false), (b"2.0", false), (b"3", true)],
            ),
            (
                r#"{"not":{"type":"number"}}"#,
                &[(b"1e2", false), (b"-0e+9", false), (br#""1e2""#, true)],
            ),
            (
                r#"{"not":{"type":"string","pattern":"^λ+$"}}"#,
                &[
                    (r#""λ""#.as_bytes(), false),
                    (br#""\u03bb""#, false),
                    (br#""x""#, true),
                ],
            ),
            (
                r#"{"not":{"allOf":[{"type":"object"},{"required":["x"]}]}}"#,
                &[(br#"{"x":1}"#, false), (b"{}", true), (b"[]", true)],
            ),
            (
                r#"{"not":{"anyOf":[{"type":"null"},{"const":1}]}}"#,
                &[(b"null", false), (b"1", false), (b"2", true)],
            ),
            (
                r#"{"not":{"oneOf":[{"const":1},{"type":"number"}]}}"#,
                &[(b"1", true), (b"2", false), (b"null", true)],
            ),
            (
                r#"{"not":{"not":{"type":"string"}}}"#,
                &[(br#""a""#, true), (b"1", false)],
            ),
            (
                r#"{"if":{"type":"object","required":["kind"],"properties":{"kind":{"const":"a"}},"additionalProperties":true},"then":{"required":["a"]},"else":{"required":["b"]}}"#,
                &[
                    (br#"{"kind":"a","a":1}"#, true),
                    (br#"{"a":1,"kind":"a"}"#, true),
                    (br#"{"kind":"b","b":1}"#, true),
                    (br#"{"kind":"a","b":1}"#, false),
                ],
            ),
            (
                r#"{"if":{"type":"integer"},"then":{"minimum":2}}"#,
                &[(b"1", false), (b"2", true), (br#""x""#, true)],
            ),
            (
                r#"{"if":{"type":"integer"},"else":{"type":"string"}}"#,
                &[(b"1", true), (br#""x""#, true)],
            ),
            (
                r#"{"if":true,"then":{"type":"integer"},"else":false}"#,
                &[(b"1", true)],
            ),
            (
                r#"{"if":false,"then":false,"else":{"type":"string"}}"#,
                &[(br#""x""#, true)],
            ),
            (
                r#"{"type":"array","items":{"if":{"type":"integer"},"then":{"minimum":0},"else":{"type":"string"}}}"#,
                &[(br#"[0,"x"]"#, true)],
            ),
            (
                r##"{"$defs":{"condition":{"type":"object","required":["kind"],"properties":{"kind":{"const":"a"}},"additionalProperties":true},"branch":{"required":["a"]}},"if":{"$ref":"#/$defs/condition"},"then":{"$ref":"#/$defs/branch"},"else":{"required":["b"]}}"##,
                &[
                    (br#"{"kind":"a","a":1}"#, true),
                    (br#"{"kind":"b","b":1}"#, true),
                ],
            ),
            (
                r#"{"if":{"allOf":[{"type":"object"},{"anyOf":[{"required":["x"]},{"required":["y"]}]}]},"then":{"oneOf":[{"required":["a"]},{"required":["b"]}]},"else":false}"#,
                &[
                    (br#"{"x":1,"a":1}"#, true),
                    (br#"{"y":1,"a":1,"b":1}"#, false),
                ],
            ),
        ];
        let mut document_total = 0usize;
        let mut prefix_total = 0usize;
        for (schema, documents) in cases {
            for &(document, valid) in *documents {
                document_total += 1;
                prefix_total += assert_combinator_prefixes(schema, document, valid);
            }
        }
        println!("documents={document_total} byte_prefixes={prefix_total}");
    }

    #[test]
    fn combinator_trie_record_and_reference_masks_are_identical() {
        let schema = r#"{"oneOf":[{"const":1},{"type":"number"},{"type":"object","additionalProperties":{"type":"integer"}}]}"#;
        let schema_ir = ir(schema);
        let program = StructuredProgram::compile(schema_ir.clone()).expect("program");
        assert!(program.uses_incremental_backend());
        let mut tokens = FxHashMap::default();
        for (id, bytes) in [
            (0u32, b"1".as_slice()),
            (1, b"12"),
            (2, br#"{"x":1}"#),
            (3, br#""x""#),
        ] {
            tokens.insert(bytes.to_vec(), vec![id]);
        }
        let vocabulary = crate::index::VocabularyHandle::new(Arc::new(
            build_vocabulary(4, tokens).expect("vocabulary"),
        ))
        .expect("handle");
        let trie = TrieCache::new().bind(&vocabulary).expect("trie");
        let mut record_matcher = program.new_matcher().expect("record matcher");
        let mut trie_matcher = program.new_matcher().expect("trie matcher");
        let record = record_matcher
            .allowed_mask_from_records(vocabulary.mask_vocab_size(), vocabulary.iter_records())
            .expect("record mask");
        let mut output = vec![0u8; vocabulary.mask_vocab_size().div_ceil(32) * 4];
        trie_matcher
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
            .expect("trie mask");
        for id in 0..vocabulary.mask_vocab_size() {
            let token = TokenId(u32::try_from(id).unwrap());
            let trie_bit = output[id / 8] & (1 << (id % 8)) != 0;
            assert_eq!(trie_bit, record.get(token), "trie/record id {id}");
            let bytes = vocabulary
                .iter_records()
                .find(|(_, ids)| ids.contains(&(id as u32)))
                .map(|(bytes, _)| bytes);
            let oracle = bytes.is_some_and(|bytes| {
                super::super::reference::accepts(&schema_ir, bytes)
                    || super::super::reference::can_continue(&schema_ir, bytes)
            });
            assert_eq!(trie_bit, oracle, "reference id {id}");
        }
    }

    #[test]
    fn negation_trie_record_and_reference_masks_are_identical() {
        let schema = r#"{"not":{"const":1}}"#;
        let schema_ir = ir(schema);
        let program = StructuredProgram::compile(schema_ir.clone()).expect("program");
        assert!(program.uses_incremental_backend());
        let mut tokens = FxHashMap::default();
        for (id, bytes) in [
            (0u32, b"1".as_slice()),
            (1, b"12"),
            (2, b"1 "),
            (3, b"2"),
            (4, br#""x""#),
            (5, b"[1]"),
        ] {
            tokens.insert(bytes.to_vec(), vec![id]);
        }
        let vocabulary = crate::index::VocabularyHandle::new(Arc::new(
            build_vocabulary(6, tokens).expect("vocabulary"),
        ))
        .expect("handle");
        let trie = TrieCache::new().bind(&vocabulary).expect("trie");
        let mut records = program.new_matcher().expect("record matcher");
        let mut walked = program.new_matcher().expect("trie matcher");
        let record = records
            .allowed_mask_from_records(vocabulary.mask_vocab_size(), vocabulary.iter_records())
            .expect("record mask");
        let mut direct_output = vec![0u8; vocabulary.mask_vocab_size().div_ceil(32) * 4];
        records
            .write_record_mask_le_bytes_into(
                vocabulary.mask_vocab_size(),
                vocabulary.iter_records(),
                &mut direct_output,
            )
            .expect("direct record output");
        let mut output = vec![0u8; vocabulary.mask_vocab_size().div_ceil(32) * 4];
        walked
            .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut output)
            .expect("trie mask");
        for id in 0..vocabulary.mask_vocab_size() {
            let token = TokenId(u32::try_from(id).unwrap());
            let trie_bit = output[id / 8] & (1 << (id % 8)) != 0;
            let direct_bit = direct_output[id / 8] & (1 << (id % 8)) != 0;
            assert_eq!(direct_bit, record.get(token), "direct/record id {id}");
            assert_eq!(trie_bit, record.get(token), "trie/record id {id}");
            let bytes = vocabulary
                .iter_records()
                .find(|(_, ids)| ids.contains(&u32::try_from(id).unwrap()))
                .map(|(bytes, _)| bytes);
            let oracle = bytes.is_some_and(|bytes| {
                super::super::reference::accepts(&schema_ir, bytes)
                    || super::super::reference::can_continue(&schema_ir, bytes)
            });
            assert_eq!(trie_bit, oracle, "reference id {id}");
        }
    }

    #[test]
    fn trie_scratch_limit_rejects_without_changing_matcher_state() {
        let limits = StructuredLimits {
            max_mask_scratch_bytes: 0,
            ..StructuredLimits::default()
        };
        let program = StructuredProgram::compile_with_limits(ir(r#"{"type":"string"}"#), limits)
            .expect("program");
        let mut tokens = FxHashMap::default();
        tokens.insert(b"\"".to_vec(), vec![0]);
        let handle =
            VocabularyHandle::new(Arc::new(build_vocabulary(1, tokens).expect("vocabulary")))
                .expect("handle");
        let trie = TrieCache::new().bind(&handle).expect("trie");
        let mut matcher = program.new_matcher().expect("matcher");
        let mut output = [0xa5; 4];
        let error = matcher
            .write_mask_le_bytes_into(&handle, Some(&trie), &mut output)
            .expect_err("scratch limit");
        assert!(matches!(
            error,
            StructuredMatcherError::ResourceLimit {
                kind: LimitKind::MaskScratchBytes,
                ..
            }
        ));
        assert_eq!(output, [0; 4]);
        assert!(!matcher.is_dead());
    }

    #[test]
    fn trie_scratch_capacity_stabilizes_after_warm_masks() {
        let program = program(r#"{"type":"string"}"#);
        let mut tokens = FxHashMap::default();
        tokens.insert(b"\"a".to_vec(), vec![0]);
        tokens.insert(b"\"b".to_vec(), vec![1]);
        let handle =
            VocabularyHandle::new(Arc::new(build_vocabulary(2, tokens).expect("vocabulary")))
                .expect("handle");
        let trie = TrieCache::new().bind(&handle).expect("trie");
        let mut matcher = program.new_matcher().expect("matcher");
        let mut output = [0; 4];
        matcher
            .write_mask_le_bytes_into(&handle, Some(&trie), &mut output)
            .expect("warmup");
        let capacity = match &matcher.backend {
            MatcherBackend::Incremental(incremental) => incremental.events.capacity(),
            #[cfg(feature = "bench-internals")]
            MatcherBackend::Reference(_) => panic!("production program selected reference backend"),
        };
        for _ in 0..32 {
            matcher
                .write_mask_le_bytes_into(&handle, Some(&trie), &mut output)
                .expect("warm mask");
        }
        let after = match &matcher.backend {
            MatcherBackend::Incremental(incremental) => incremental.events.capacity(),
            #[cfg(feature = "bench-internals")]
            MatcherBackend::Reference(_) => panic!("production program selected reference backend"),
        };
        assert_eq!(after, capacity);
    }

    /// Exercises full object generation with an overlapping byte-level vocabulary.
    /// Direct trie masks must match record scans at every commit step.
    #[test]
    fn byte_level_vocabulary_trie_mask_matches_record_scan_across_a_full_generation() {
        let program = program(
            r#"{"type":"object","properties":{"id":{"type":"integer"},"ok":{"type":"boolean"}},"required":["id","ok"],"additionalProperties":false}"#,
        );
        let mut tokens = FxHashMap::default();
        let pieces: &[&[u8]] = &[
            b"{", b"}", b"\"", b":", b",", b" ", b"id", b"ok", b"i", b"d", b"o", b"k", b"\"id\"",
            b"\"ok\"", b"true", b"false", b"tru", b"e", b"0", b"1", b"42", b"-",
        ];
        for (i, piece) in pieces.iter().enumerate() {
            tokens.insert(piece.to_vec(), vec![i as u32]);
        }
        // Duplicate byte string mapped to two ids: both must be set together by the trie walk.
        tokens.insert(b"}".to_vec(), vec![1, 40]);
        let width = 64usize;
        let handle =
            VocabularyHandle::new(Arc::new(build_vocabulary(63, tokens).expect("vocabulary")))
                .expect("handle");
        let trie = TrieCache::new().bind(&handle).expect("trie");

        let doc: &[&[u8]] = &[
            b"{", b"\"id\"", b":", b"42", b",", b"\"ok\"", b":", b"true", b"}",
        ];
        let mut records = program.new_matcher().expect("records");
        let mut walked = program.new_matcher().expect("walked");
        let byte_len = width.div_ceil(32) * std::mem::size_of::<u32>();
        for step in doc {
            let expected = records
                .allowed_mask_from_records(handle.mask_vocab_size(), handle.iter_records())
                .expect("record mask");
            let mut output = vec![0xff; byte_len];
            walked
                .write_mask_le_bytes_into(&handle, Some(&trie), &mut output)
                .expect("trie mask");
            for id in 0..handle.mask_vocab_size() {
                let observed = output[id / 8] & (1u8 << (id % 8)) != 0;
                let token = TokenId(u32::try_from(id).expect("token id"));
                assert_eq!(observed, expected.get(token), "id {id} at step {step:?}");
            }
            assert!(records.advance(step).expect("records advance"), "{step:?}");
            assert!(walked.advance(step).expect("walked advance"), "{step:?}");
        }
        assert!(walked.is_accepting());
        assert!(records.is_accepting());
    }

    /// A token may end in an accepting nested regular frame.
    /// The frame remains active until a later byte closes it.
    #[test]
    fn nested_string_token_ending_accepting_keeps_the_frame_and_matches_the_record_scan() {
        let program = program(
            r#"{"type":"object","properties":{"n":{"type":"string"}},"required":["n"],"additionalProperties":false}"#,
        );
        let mut tokens = FxHashMap::default();
        // `ab` and `abc` both end accepting inside the string body; the quote is a separate token,
        // so the frame must survive every one of them.
        let pieces: &[&[u8]] = &[
            b"{", b"}", b"\"", b":", b",", b"n", b"\"n\"", b"ab", b"abc", b"c", b"z", b" ",
        ];
        for (i, piece) in pieces.iter().enumerate() {
            tokens.insert(piece.to_vec(), vec![i as u32]);
        }
        let width = 64usize;
        let handle =
            VocabularyHandle::new(Arc::new(build_vocabulary(63, tokens).expect("vocabulary")))
                .expect("handle");
        let trie = TrieCache::new().bind(&handle).expect("trie");

        let doc: &[&[u8]] = &[
            b"{", b"\"n\"", b":", b"\"", b"ab", b"abc", b"c", b"\"", b"}",
        ];
        let mut records = program.new_matcher().expect("records");
        let mut walked = program.new_matcher().expect("walked");
        let byte_len = width.div_ceil(32) * std::mem::size_of::<u32>();
        for step in doc {
            let expected = records
                .allowed_mask_from_records(handle.mask_vocab_size(), handle.iter_records())
                .expect("record mask");
            let mut output = vec![0xff; byte_len];
            walked
                .write_mask_le_bytes_into(&handle, Some(&trie), &mut output)
                .expect("trie mask");
            for id in 0..handle.mask_vocab_size() {
                let observed = output[id / 8] & (1u8 << (id % 8)) != 0;
                let token = TokenId(u32::try_from(id).expect("token id"));
                assert_eq!(observed, expected.get(token), "id {id} at step {step:?}");
            }
            assert!(records.advance(step).expect("records advance"), "{step:?}");
            assert!(walked.advance(step).expect("walked advance"), "{step:?}");
        }
        assert!(walked.is_accepting());
        assert!(records.is_accepting());
    }

    /// The nested slice path must actually be taken, or the test above proves only that two slow
    /// paths agree. Inside a nested string body the walk must stay off the full trie.
    #[test]
    #[cfg(feature = "bench-internals")]
    fn nested_string_body_mask_walks_only_the_uncertain_trie() {
        let program = program(
            r#"{"type":"object","properties":{"n":{"type":"string"}},"required":["n"],"additionalProperties":false}"#,
        );
        let mut tokens = FxHashMap::default();
        let pieces: &[&[u8]] = &[
            b"{", b"}", b"\"", b":", b"n", b"\"n\"", b"ab", b"cd", b"ef", b"gh", b"ij", b"kl",
        ];
        for (i, piece) in pieces.iter().enumerate() {
            tokens.insert(piece.to_vec(), vec![i as u32]);
        }
        let handle =
            VocabularyHandle::new(Arc::new(build_vocabulary(63, tokens).expect("vocabulary")))
                .expect("handle");
        let trie = TrieCache::new().bind(&handle).expect("trie");
        let mut matcher = program.new_matcher().expect("matcher");
        let mut output = vec![0u8; 64usize.div_ceil(32) * std::mem::size_of::<u32>()];

        for step in [b"{".as_slice(), b"\"n\"", b":", b"\""] {
            matcher
                .write_mask_le_bytes_into(&handle, Some(&trie), &mut output)
                .expect("mask");
            assert!(matcher.advance(step).expect("advance"), "{step:?}");
        }
        matcher
            .write_mask_le_bytes_into(&handle, Some(&trie), &mut output)
            .expect("string body mask");
        let nodes = matcher.mask_work_metrics_for_bench().trie_nodes;
        let full = trie.trie().node_count() as u64;
        assert!(
            nodes < full,
            "string body walked {nodes} of {full} nodes; the slice never fired"
        );
    }

    #[test]
    fn trie_record_and_reference_masks_agree_after_every_uniqueitems_prefix() {
        let schema = r#"{"type":"array","items":{"type":"integer"},"uniqueItems":true,"contains":{"const":0}}"#;
        let doc_ir = ir(schema);
        let program = StructuredProgram::compile(doc_ir.clone()).expect("program");
        assert_eq!(program.backend_kind(), "incremental");
        let mut tokens = FxHashMap::default();
        for (id, bytes) in [
            (0u32, b"[".as_slice()),
            (1, b"0"),
            (2, b","),
            (3, b"1"),
            (4, b"2"),
            (5, b"]"),
        ] {
            tokens.insert(bytes.to_vec(), vec![id]);
        }
        let handle = VocabularyHandle::new(Arc::new(build_vocabulary(6, tokens).expect("vocab")))
            .expect("handle");
        let trie = TrieCache::new().bind(&handle).expect("trie");
        let mut walked = program.new_matcher().expect("trie matcher");
        let mut recorded = program.new_matcher().expect("record matcher");
        let prefixes: &[&[u8]] = &[b"[", b"[0", b"[0,1", b"[0,1,2"];
        let steps: &[&[u8]] = &[b"[", b"0", b",1", b",2"];
        let mut committed = Vec::new();
        for (prefix, step) in prefixes.iter().zip(steps.iter()) {
            for &b in *step {
                assert!(walked.advance(&[b]).expect("advance"));
                assert!(recorded.advance(&[b]).expect("advance"));
            }
            committed.extend_from_slice(step);
            assert_eq!(committed.as_slice(), *prefix);

            let record_mask = recorded
                .allowed_mask_from_records(handle.mask_vocab_size(), handle.iter_records())
                .expect("record mask");
            let mut trie_out = vec![0u8; handle.mask_vocab_size().div_ceil(32) * 4];
            walked
                .write_mask_le_bytes_into(&handle, Some(&trie), &mut trie_out)
                .expect("trie mask");
            for id in 0..handle.mask_vocab_size() {
                let token = TokenId(u32::try_from(id).expect("id"));
                let trie_bit = trie_out[id / 8] & (1u8 << (id % 8)) != 0;
                let record_bit = record_mask.get(token);
                assert_eq!(trie_bit, record_bit, "prefix {prefix:?} id {id}");
                let id32 = u32::try_from(id).expect("id");
                let Some((bytes, _)) = handle.iter_records().find(|(_, ids)| ids.contains(&id32))
                else {
                    continue;
                };
                let mut candidate = prefix.to_vec();
                candidate.extend_from_slice(bytes);
                let ref_bit = super::super::reference::accepts(&doc_ir, &candidate)
                    || super::super::reference::can_continue(&doc_ir, &candidate);
                assert_eq!(trie_bit, ref_bit, "prefix {prefix:?} id {id} vs reference");
            }
        }
    }

    #[test]
    fn unevaluated_prefix_masks_match_record_preallocated_trie_and_reference() {
        let cases = [
            (
                r#"{"anyOf":[{"properties":{"a":true}},{"properties":{"b":true}}],"unevaluatedProperties":false}"#,
                br#"{"a":1,"b":2}"#.as_slice(),
            ),
            (
                r#"{"prefixItems":[true],"contains":{"type":"string"},"unevaluatedItems":false}"#,
                br#"[1,"foo"]"#.as_slice(),
            ),
            (
                r#"{"properties":{"v":true},"unevaluatedProperties":true}"#,
                br#"{"v":[{"deep":[1,2]}]}"#.as_slice(),
            ),
        ];
        let mut token_map = FxHashMap::default();
        for byte in 0u8..=127 {
            token_map.insert(vec![byte], vec![u32::from(byte)]);
        }
        let handle = VocabularyHandle::new(Arc::new(
            build_vocabulary(128, token_map).expect("ASCII vocabulary"),
        ))
        .expect("handle");
        let trie = TrieCache::new().bind(&handle).expect("trie");
        let output_bytes = handle.mask_vocab_size().div_ceil(32) * 4;
        let mut bytes_by_id = vec![None; handle.mask_vocab_size()];
        for (bytes, ids) in handle.iter_records() {
            for &id in ids {
                bytes_by_id[usize::try_from(id).unwrap()] = Some(bytes.to_vec());
            }
        }

        for (schema, document) in cases {
            let schema_ir = ir(schema);
            let program = StructuredProgram::compile(schema_ir.clone()).expect("program");
            let mut record = program.new_matcher().expect("record matcher");
            let mut preallocated = program.new_matcher().expect("preallocated matcher");
            let mut walked = program.new_matcher().expect("trie matcher");
            assert_eq!(record.backend_name(), "incremental", "{schema}");
            assert_eq!(preallocated.backend_name(), "incremental", "{schema}");
            assert_eq!(walked.backend_name(), "incremental", "{schema}");
            let mut whitespace = program.new_matcher().unwrap();
            assert!(
                whitespace.advance(b"\t").unwrap(),
                "leading whitespace {schema}"
            );
            let direct = record
                .allowed_mask_from_records(
                    128,
                    std::iter::once((b"\t".as_slice(), [9u32].as_slice())),
                )
                .unwrap();
            assert!(direct.get(TokenId(9)), "direct tab mask {schema}");

            for prefix_len in 0..=document.len() {
                let prefix = &document[..prefix_len];
                let record_mask = record
                    .allowed_mask_from_records(handle.mask_vocab_size(), handle.iter_records())
                    .expect("record mask");
                let mut preallocated_out = vec![0xff; output_bytes];
                preallocated
                    .write_record_mask_le_bytes_into(
                        handle.mask_vocab_size(),
                        handle.iter_records(),
                        &mut preallocated_out,
                    )
                    .expect("preallocated record mask");
                let mut trie_out = vec![0xff; output_bytes];
                walked
                    .write_mask_le_bytes_into(&handle, Some(&trie), &mut trie_out)
                    .expect("trie mask");

                for id in 0..handle.mask_vocab_size() {
                    let token = TokenId(u32::try_from(id).unwrap());
                    let record_bit = record_mask.get(token);
                    let preallocated_bit = preallocated_out[id / 8] & (1u8 << (id % 8)) != 0;
                    let trie_bit = trie_out[id / 8] & (1u8 << (id % 8)) != 0;
                    assert_eq!(
                        record_bit, preallocated_bit,
                        "{schema} prefix={prefix:?} id={id}"
                    );
                    assert_eq!(record_bit, trie_bit, "{schema} prefix={prefix:?} id={id}");
                    if let Some(bytes) = &bytes_by_id[id] {
                        let mut candidate = prefix.to_vec();
                        candidate.extend_from_slice(bytes);
                        let reference_bit =
                            super::super::reference::accepts(&schema_ir, &candidate)
                                || super::super::reference::can_continue(&schema_ir, &candidate);
                        assert_eq!(
                            record_bit, reference_bit,
                            "{schema} prefix={prefix:?} id={id} bytes={bytes:?}"
                        );
                    }
                }
                assert_eq!(
                    record.eos_legal(),
                    super::super::reference::accepts(&schema_ir, prefix)
                );
                if prefix_len < document.len() {
                    let byte = &document[prefix_len..prefix_len + 1];
                    assert!(record.advance(byte).unwrap(), "{schema} prefix={prefix:?}");
                    assert!(preallocated.advance(byte).unwrap());
                    assert!(walked.advance(byte).unwrap());
                }
            }
        }
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn finite_unevaluated_key_prefix_rejects_dead_end_and_matches_all_routes() {
        let schema = r#"{"type":"object","required":["method"],"properties":{"method":{"enum":["card","wallet"]}},"unevaluatedProperties":false}"#;
        let structured = program(schema);
        assert_eq!(structured.backend_kind(), "incremental");
        let mut tokens = FxHashMap::default();
        for byte in 0u8..=255 {
            tokens.insert(vec![byte], vec![u32::from(byte)]);
        }
        for (id, bytes) in [
            (256, b"method".as_slice()),
            (257, b"bogus-dead-branch-one"),
            (258, b"bogus-dead-branch-two"),
        ] {
            tokens.insert(bytes.to_vec(), vec![id]);
        }
        let handle =
            VocabularyHandle::new(Arc::new(build_vocabulary(259, tokens).expect("vocabulary")))
                .expect("handle");
        let trie = TrieCache::new().bind(&handle).expect("trie");
        let output_len = handle.mask_vocab_size().div_ceil(32) * 4;

        for prefix in [br#"{""#.as_slice(), br#"{"method""#, br#"{"method":"card""#] {
            let mut adaptive = structured.new_matcher().expect("adaptive");
            let mut full = structured.new_matcher().expect("full");
            let mut record = structured.new_matcher().expect("record");
            for matcher in [&mut adaptive, &mut full, &mut record] {
                assert!(matcher.advance(prefix).expect("valid prefix"), "{prefix:?}");
            }
            let mut adaptive_out = vec![0xa5; output_len];
            let mut full_out = vec![0xa5; output_len];
            let mut record_out = vec![0xa5; output_len];
            adaptive
                .write_mask_le_bytes_into(&handle, Some(&trie), &mut adaptive_out)
                .expect("adaptive mask");
            let adaptive_nodes = adaptive.mask_work_metrics_for_bench().trie_nodes;
            full.write_full_trie_mask_for_bench(&handle, &trie, &mut full_out)
                .expect("full mask");
            record
                .write_record_mask_le_bytes_into(
                    handle.mask_vocab_size(),
                    handle.iter_records(),
                    &mut record_out,
                )
                .expect("record mask");
            assert_eq!(adaptive_out, full_out, "adaptive/full at {prefix:?}");
            assert_eq!(adaptive_out, record_out, "adaptive/record at {prefix:?}");
            if prefix == br#"{""# {
                assert!(
                    adaptive_nodes < trie.trie().node_count() as u64,
                    "finite key pruning did not activate: {adaptive_nodes} nodes"
                );
            }

            for (bytes, ids) in handle.iter_records() {
                if !ids
                    .iter()
                    .any(|id| adaptive_out[*id as usize / 8] & (1 << (*id % 8)) != 0)
                {
                    continue;
                }
                let mut equivalent = structured.new_matcher().expect("equivalent");
                assert!(equivalent.advance(prefix).expect("equivalent prefix"));
                let mut prepared = vec![0; output_len];
                equivalent
                    .write_mask_le_bytes_into(&handle, Some(&trie), &mut prepared)
                    .expect("prepare equivalent");
                assert_eq!(prepared, adaptive_out);
                assert!(
                    equivalent.advance(bytes).expect("emitted token"),
                    "emitted {bytes:?} at {prefix:?}"
                );
            }

            let bit = |byte: u8| adaptive_out[byte as usize / 8] & (1 << (byte % 8)) != 0;
            match prefix {
                br#"{""# => {
                    assert!(bit(b'm'));
                    assert!(!bit(b'b'));
                }
                br#"{"method""# => {
                    assert!(bit(b':'));
                    assert!(!bit(b'}'));
                }
                br#"{"method":"card""# => {
                    assert!(bit(b'}'));
                    assert!(!bit(b','));
                }
                _ => unreachable!(),
            }
        }

        for schema in [
            schema,
            r#"{"type":"object","required":["method"],"properties":{"method":{"enum":["card","wallet"]}},"additionalProperties":false}"#,
        ] {
            let program = program(schema);
            let mut matcher = program.new_matcher().expect("matcher");
            assert!(matcher.advance(br#"{""#).expect("opening key"));
            assert!(!matcher.advance(b"b").expect("dead key prefix"), "{schema}");
            assert!(
                matcher.advance(b"m").expect("rollback preserved state"),
                "{schema}"
            );

            for impossible in [br#"{"bogus""#.as_slice(), br#"{"bogus":"x""#] {
                let mut matcher = program.new_matcher().expect("impossible matcher");
                assert!(
                    !matcher.advance(impossible).expect("impossible prefix"),
                    "{schema} admitted {impossible:?}"
                );
            }

            let mut no_trailing_member = program.new_matcher().expect("trailing member matcher");
            assert!(no_trailing_member.advance(br#"{"method":"card""#).unwrap());
            assert!(
                !no_trailing_member.advance(b",").unwrap(),
                "{schema} admitted a comma with no viable unseen member"
            );
            assert!(
                no_trailing_member.advance(b"}").unwrap(),
                "{schema} did not roll the rejected comma back"
            );
        }
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn finite_unevaluated_key_viability_matrix_preserves_branches_intersections_and_seen_keys() {
        let mut tokens = FxHashMap::default();
        for byte in 0u8..=255 {
            tokens.insert(vec![byte], vec![u32::from(byte)]);
        }
        for (id, token) in [
            (256, b"method\":\"card\"".as_slice()),
            (257, b"a\\/b\""),
            (258, b"\\u00e9\""),
        ] {
            tokens.insert(token.to_vec(), vec![id]);
        }
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(259, tokens).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let bit = |mask: &[u8], id: usize| mask[id / 8] & (1 << (id % 8)) != 0;

        let conditional = program(
            r#"{"allOf":[
                {"properties":{"alpha":true}},
                {"if":{"properties":{"kind":{"const":"x"}},"required":["kind"]},
                 "then":{"properties":{"beta":true}},
                 "else":{"properties":{"gamma":true}}}
            ],"unevaluatedProperties":false}"#,
        );
        let (mask, nodes) = assert_three_route_mask_and_advance(
            &conditional,
            &vocabulary,
            &trie,
            br#"{""#,
            "conditional-union",
        );
        for allowed in [b'a', b'b', b'g', b'k'] {
            assert!(
                bit(&mask, allowed as usize),
                "missing branch prefix {allowed}"
            );
        }
        assert!(!bit(&mask, b'z' as usize));
        assert!(nodes < trie.trie().node_count() as u64);

        let property_names = program(
            r#"{"properties":{"apple":true,"beta":true},"propertyNames":{"pattern":"^a"},"unevaluatedProperties":false}"#,
        );
        let (mask, _) = assert_three_route_mask_and_advance(
            &property_names,
            &vocabulary,
            &trie,
            br#"{""#,
            "propertyNames-intersection",
        );
        assert!(bit(&mask, b'a' as usize));
        assert!(!bit(&mask, b'b' as usize));

        let value_certificate = program(
            r#"{"properties":{"impossible":false,"method":{"enum":["card","wallet"]}},"unevaluatedProperties":false}"#,
        );
        let (mask, _) = assert_three_route_mask_and_advance(
            &value_certificate,
            &vocabulary,
            &trie,
            br#"{""#,
            "proven-empty-value",
        );
        assert!(!bit(&mask, b'i' as usize));
        assert!(bit(&mask, b'm' as usize));

        let shared_prefix =
            program(r#"{"properties":{"a":true,"ab":true},"unevaluatedProperties":false}"#);
        let (mask, _) = assert_three_route_mask_and_advance(
            &shared_prefix,
            &vocabulary,
            &trie,
            br#"{"a":1,"#,
            "seen-terminal-unseen-descendant",
        );
        assert!(bit(&mask, b'"' as usize), "unseen `ab` still permits a key");
        let (mask, _) = assert_three_route_mask_and_advance(
            &shared_prefix,
            &vocabulary,
            &trie,
            br#"{"a":1,"a"#,
            "seen-terminal-prefix",
        );
        assert!(bit(&mask, b'b' as usize));
        assert!(
            !bit(&mask, b'"' as usize),
            "duplicate exact `a` must not close"
        );
        let mut exhausted = shared_prefix.new_matcher().unwrap();
        assert!(exhausted.advance(br#"{"a":1,"ab":2"#).unwrap());
        assert!(
            !exhausted.advance(b",").unwrap(),
            "all finite names are seen, so another member cannot begin"
        );
        assert!(exhausted.advance(b"}").unwrap());

        let dependent = program(
            r#"{"properties":{"trigger":true},"dependentSchemas":{"trigger":{"properties":{"extra":true}}},"unevaluatedProperties":false}"#,
        );
        let (mask, _) = assert_three_route_mask_and_advance(
            &dependent,
            &vocabulary,
            &trie,
            br#"{""#,
            "dependent-schema-union",
        );
        assert!(bit(&mask, b'e' as usize));
        assert!(bit(&mask, b't' as usize));
        assert!(!bit(&mask, b'z' as usize));

        let encoded_names =
            program(r#"{"properties":{"a/b":true,"é":true},"unevaluatedProperties":false}"#);
        let (mask, _) = assert_three_route_mask_and_advance(
            &encoded_names,
            &vocabulary,
            &trie,
            br#"{"a\/"#,
            "escaped-key",
        );
        assert!(bit(&mask, b'b' as usize));
        assert!(!bit(&mask, b'"' as usize));
        let (mask, _) = assert_three_route_mask_and_advance(
            &encoded_names,
            &vocabulary,
            &trie,
            br#"{"\u00e9"#,
            "unicode-escape-key",
        );
        assert!(bit(&mask, b'"' as usize));
        let (mask, _) = assert_three_route_mask_and_advance(
            &encoded_names,
            &vocabulary,
            &trie,
            b"{\"\xc3",
            "partial-utf8-key",
        );
        assert!(bit(&mask, 0xa9));

        let method = program(
            r#"{"required":["method"],"properties":{"method":{"enum":["card","wallet"]}},"unevaluatedProperties":false}"#,
        );
        let (mask, _) = assert_three_route_mask_and_advance(
            &method,
            &vocabulary,
            &trie,
            br#"{""#,
            "quote-colon-value-crossing",
        );
        assert!(bit(&mask, 256));

        // One pattern is a bounded one-DFA product; interacting patterns remain on the exact walker.
        let patterned = program(
            r#"{"patternProperties":{"^x$":{"type":"string"}},"additionalProperties":false,"unevaluatedProperties":false}"#,
        );
        let (mask, nodes) = assert_three_route_mask_and_advance(
            &patterned,
            &vocabulary,
            &trie,
            br#"{""#,
            "single-pattern-bounded-product",
        );
        assert!(bit(&mask, b'x' as usize));
        assert!(!bit(&mask, b'y' as usize));
        assert!(nodes < trie.trie().node_count() as u64);
        let (mask, _) = assert_three_route_mask_and_advance(
            &patterned,
            &vocabulary,
            &trie,
            br#"{"x"#,
            "single-pattern-complete-rejection",
        );
        assert!(bit(&mask, b'"' as usize));
        assert!(!bit(&mask, b'y' as usize));

        let bounded_pattern = program(
            r#"{"patternProperties":{"^[a-z]{2,6}$":{"type":"string"}},"additionalProperties":false,"unevaluatedProperties":false}"#,
        );
        let (mask, _) = assert_three_route_mask_and_advance(
            &bounded_pattern,
            &vocabulary,
            &trie,
            br#"{"amount":"USD","amoun"#,
            "seen-pattern-key-terminal-prefix",
        );
        assert!(
            !bit(&mask, b't' as usize),
            "a duplicate maximal-length pattern key must not become reachable"
        );
        let mut duplicate = bounded_pattern.new_matcher().unwrap();
        assert!(duplicate.advance(br#"{"amount":"USD","amoun"#).unwrap());
        assert!(!duplicate.advance(b"t").unwrap());
        assert!(duplicate.advance(b"z").unwrap());

        // Exact and pattern-defined keys must make the same duplicate decision under both
        // closed-object keywords. This is the permanent public-reproduction four-cell matrix.
        for (source, keyword) in [
            ("exact", "additionalProperties"),
            ("exact", "unevaluatedProperties"),
            ("pattern", "additionalProperties"),
            ("pattern", "unevaluatedProperties"),
        ] {
            let declaration = if source == "exact" {
                r#""properties":{"amount":{"type":"string"}}"#
            } else {
                r#""patternProperties":{"^[a-z]{2,6}$":{"type":"string"}}"#
            };
            let schema = format!(r#"{{"type":"object",{declaration},"{keyword}":false}}"#);
            let case = program(&schema);
            let mut direct = case.new_matcher().unwrap();
            assert!(
                !direct.advance(br#"{"amount":"x","amount"#).unwrap(),
                "{source}/{keyword} admitted a duplicate key prefix"
            );
        }

        let finite_pattern = program(
            r#"{"patternProperties":{"^[ab]$":true},"additionalProperties":false,"unevaluatedProperties":false}"#,
        );
        let (mask, _) = assert_three_route_mask_and_advance(
            &finite_pattern,
            &vocabulary,
            &trie,
            br#"{"a":1,""#,
            "finite-pattern-one-unseen-key",
        );
        assert!(!bit(&mask, b'a' as usize));
        assert!(bit(&mask, b'b' as usize));
        let mut exhausted_pattern = finite_pattern.new_matcher().unwrap();
        assert!(exhausted_pattern.advance(br#"{"a":1,"b":2"#).unwrap());
        assert!(!exhausted_pattern.advance(b",").unwrap());
        assert!(exhausted_pattern.advance(b"}").unwrap());

        let false_exact_intersection = program(
            r#"{"properties":{"x":false},"patternProperties":{"^x$":true},"additionalProperties":false}"#,
        );
        let mut false_exact = false_exact_intersection.new_matcher().unwrap();
        assert!(false_exact.advance(br#"{"#).unwrap());
        assert!(
            !false_exact.advance(br#""x"#).unwrap(),
            "a pattern must not revive an exact property whose intersected value schema is false"
        );

        let escaped_pattern = program(
            r#"{"patternProperties":{"^cosk[a-z]$":true},"additionalProperties":false,"unevaluatedProperties":false}"#,
        );
        let (mask, _) = assert_three_route_mask_and_advance(
            &escaped_pattern,
            &vocabulary,
            &trie,
            br#"{"cosk\u"#,
            "partial-unicode-escape-key-language",
        );
        assert!(
            bit(&mask, b'0' as usize),
            "U+0000..U+0FFF still contains ASCII lowercase completions"
        );
        assert!(
            !bit(&mask, b'2' as usize),
            "U+2000..U+2FFF cannot complete an ASCII lowercase pattern key"
        );
        let mut escaped = escaped_pattern.new_matcher().unwrap();
        assert!(escaped.advance(br#"{"cosk\u"#).unwrap());
        assert!(!escaped.advance(b"2").unwrap());
        assert!(escaped.advance(b"0").unwrap());
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn admitted_bounded_pattern_key_successors_retain_a_path_to_object_close() {
        let schema = program(
            r#"{"type":"object","propertyNames":{"pattern":"^[a-z]{2,6}$"},"patternProperties":{"^[a-z]{2,6}$":{"type":"string","maxLength":6}},"additionalProperties":false,"maxProperties":2}"#,
        );
        let prefix = br#"{"amount":"USD","amoun"#;
        let records = (0u8..=255)
            .map(|byte| (vec![byte], vec![u32::from(byte)]))
            .collect::<Vec<_>>();
        let mut matcher = schema.new_matcher().unwrap();
        assert!(matcher.advance(prefix).unwrap());
        let allowed = mask(&mut matcher, 256, &records);
        for byte in (0x20u8..=0x7e).chain([b'\t', b'\n']) {
            if !allowed.get(crate::primitives::TokenId(u32::from(byte))) {
                continue;
            }
            let tail: &[u8] = match byte {
                b'"' => br#":"x"}"#,
                b'\\' => br#"u007a":"x"}"#,
                b'a'..=b'z' => br#"":"x"}"#,
                _ => panic!("unexpected admitted key byte {byte:#04x}"),
            };
            let mut document = prefix.to_vec();
            document.push(byte);
            document.extend_from_slice(tail);
            let mut completed = schema.new_matcher().unwrap();
            assert!(
                completed.advance(&document).unwrap() && completed.is_accepting(),
                "admitted byte {byte:#04x} left no direct path to object close: {document:?}"
            );
        }
        let mut search_budget = 5_000;
        assert_eq!(
            closer_first_completion_search(
                &schema,
                br#"{"amount":"USD","amounz"#,
                8,
                &mut search_budget,
            ),
            CompletionSearch::Found,
            "the independent closer-first search must find the short closing continuation"
        );

        // The partially decoded escape witness has a concrete live branch and a proven-dead one.
        let escaped =
            program(r#"{"patternProperties":{"^cosk[a-z]$":true},"additionalProperties":false}"#);
        let mut dead = escaped.new_matcher().unwrap();
        assert!(dead.advance(br#"{"cosk\u"#).unwrap());
        assert!(!dead.advance(b"2").unwrap());
        let mut live = escaped.new_matcher().unwrap();
        assert!(live.advance(br#"{"cosk\u0061":1}"#).unwrap());
        assert!(live.is_accepting());
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn finite_unevaluated_key_session_boundary_clears_errors_and_preserves_accounting() {
        let program = program(
            r#"{"required":["method"],"properties":{"method":{"enum":["card","wallet"]}},"unevaluatedProperties":false}"#,
        );
        let records = [b"m".as_slice(), b"b", b"\"", b":", b"method", b"bogus"]
            .into_iter()
            .enumerate()
            .map(|(id, bytes)| (bytes.to_vec(), vec![u32::try_from(id).unwrap()]))
            .collect();
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(8, records).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let prefix = br#"{""#;
        let exact = minimum_successful_limit(
            &program,
            &vocabulary,
            &trie,
            prefix,
            program.limits.max_session_bytes,
            NearLimitTuning::Session,
        );
        for limit in around(exact) {
            assert_near_limit_three_routes(
                &program,
                &vocabulary,
                &trie,
                prefix,
                NearLimitTuning::Session(limit),
                &format!("finite-key-session-{limit}"),
            );
        }
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn contains_unevaluated_product_matrix_matches_all_routes_and_advances() {
        struct Case {
            name: &'static str,
            schema: &'static str,
            document: &'static [u8],
        }
        let cases = [
            Case {
                name: "min-contains-0",
                schema: r#"{"type":"array","items":{"type":"integer"},"contains":{"type":"string"},"minContains":0}"#,
                document: b"[1]",
            },
            Case {
                name: "min-contains-1",
                schema: r#"{"type":"array","items":{"type":"string"},"contains":{"type":"string","pattern":"^ok$"},"minContains":1}"#,
                document: br#"["ok"]"#,
            },
            Case {
                name: "min-contains-2-candidate-dies-then-later-satisfies",
                schema: r#"{"type":"array","items":{"type":"string"},"contains":{"type":"string","pattern":"^ok"},"minContains":2,"maxItems":3}"#,
                document: br#"["no","okay","ok"]"#,
            },
            Case {
                name: "max-contains-0",
                schema: r#"{"type":"array","items":{"type":"integer"},"contains":{"type":"string"},"minContains":0,"maxContains":0}"#,
                document: b"[1]",
            },
            Case {
                name: "max-contains-1",
                schema: r#"{"type":"array","items":true,"contains":{"type":"string"},"minContains":0,"maxContains":1}"#,
                document: br#"["a",1]"#,
            },
            Case {
                name: "max-contains-2",
                schema: r#"{"type":"array","items":true,"contains":{"type":"string"},"minContains":0,"maxContains":2}"#,
                document: br#"["a","b",1]"#,
            },
            Case {
                name: "prefix-items-to-tail",
                schema: r#"{"type":"array","prefixItems":[{"type":"integer"}],"items":{"type":"string"},"contains":{"type":"string"},"minContains":1}"#,
                document: br#"[1,"x"]"#,
            },
            Case {
                name: "contains-produces-unevaluated-annotation",
                schema: r#"{"prefixItems":[true],"contains":{"type":"string","pattern":"^ok$"},"unevaluatedItems":false}"#,
                document: br#"[1,"ok"]"#,
            },
            Case {
                name: "item-accepts-contains-rejects",
                schema: r#"{"type":"array","items":{"type":"string"},"contains":{"type":"string","pattern":"^ok$"},"minContains":0}"#,
                document: br#"["no"]"#,
            },
            Case {
                name: "contains-accepts-item-rejects-empty-array",
                schema: r#"{"type":"array","items":{"type":"integer"},"contains":{"type":"string"},"minContains":0}"#,
                document: b"[]",
            },
            Case {
                name: "maximum-length-array",
                schema: r#"{"type":"array","items":{"type":"integer"},"contains":{"const":2},"minContains":1,"maxItems":2}"#,
                document: b"[1,2]",
            },
        ];
        let token_bytes: &[&[u8]] = &[
            b"[", b"]", b",", b"0", b"1", b"2", b"\"", b"a", b"b", b"k", b"n", b"o", b"x", b"y",
            b"okay", b"\"ok\"", b"\"no\"",
        ];
        let records = token_bytes
            .iter()
            .enumerate()
            .map(|(id, bytes)| (bytes.to_vec(), vec![u32::try_from(id).unwrap()]))
            .collect();
        let vocabulary =
            VocabularyHandle::new(Arc::new(build_vocabulary(32, records).unwrap())).unwrap();
        let trie = TrieCache::new().bind(&vocabulary).unwrap();
        let output_len = vocabulary.mask_vocab_size().div_ceil(32) * 4;

        for case in cases {
            let program = program(case.schema);
            let mut adaptive = program.new_matcher().unwrap();
            let mut full = program.new_matcher().unwrap();
            let mut record = program.new_matcher().unwrap();
            for prefix_len in 0..=case.document.len() {
                let prefix = &case.document[..prefix_len];
                let mut adaptive_out = vec![0; output_len];
                let mut full_out = vec![0; output_len];
                let mut record_out = vec![0; output_len];
                adaptive
                    .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut adaptive_out)
                    .unwrap_or_else(|error| panic!("{} {prefix:?}: {error}", case.name));
                let MatcherBackend::Incremental(inner) = &mut full.backend else {
                    panic!("incremental full matcher")
                };
                select_adaptive_trie(inner, &vocabulary, program.limits).unwrap();
                walk_full_trie_mask_for_test(
                    inner,
                    &vocabulary,
                    &trie,
                    &mut full_out,
                    program.limits,
                )
                .unwrap();
                let MatcherBackend::Incremental(inner) = &mut record.backend else {
                    panic!("incremental record matcher")
                };
                select_adaptive_trie(inner, &vocabulary, program.limits).unwrap();
                record
                    .write_record_mask_le_bytes_into(
                        vocabulary.mask_vocab_size(),
                        vocabulary.iter_records(),
                        &mut record_out,
                    )
                    .unwrap();
                assert_mask_bytes_equal(&adaptive_out, &full_out, &vocabulary, case.name);
                assert_mask_bytes_equal(&adaptive_out, &record_out, &vocabulary, case.name);

                for (candidate, ids) in vocabulary.iter_records() {
                    if !ids
                        .iter()
                        .any(|id| adaptive_out[*id as usize / 8] & (1 << (*id % 8)) != 0)
                    {
                        continue;
                    }
                    let mut equivalent = program.new_matcher().unwrap();
                    if !prefix.is_empty() {
                        assert!(
                            equivalent.advance(prefix).unwrap(),
                            "{} {prefix:?}",
                            case.name
                        );
                    }
                    let mut prepared = vec![0; output_len];
                    equivalent
                        .write_mask_le_bytes_into(&vocabulary, Some(&trie), &mut prepared)
                        .unwrap();
                    assert_eq!(prepared, adaptive_out, "{} {prefix:?}", case.name);
                    assert!(
                        equivalent.advance(candidate).unwrap(),
                        "{} emitted {candidate:?} at {prefix:?}",
                        case.name
                    );
                    assert_incremental_accounting(&equivalent, case.name);
                }
                for matcher in [&adaptive, &full, &record] {
                    assert_incremental_accounting(matcher, case.name);
                }
                if prefix_len < case.document.len() {
                    let next = &case.document[prefix_len..prefix_len + 1];
                    let (_, ids) = vocabulary
                        .iter_records()
                        .find(|(bytes, _)| *bytes == next)
                        .expect("each document byte is a vocabulary token");
                    assert!(ids
                        .iter()
                        .any(|id| { adaptive_out[*id as usize / 8] & (1 << (*id % 8)) != 0 }));
                    assert!(adaptive.advance(next).unwrap(), "{} {prefix:?}", case.name);
                    assert!(full.advance(next).unwrap(), "{} {prefix:?}", case.name);
                    assert!(record.advance(next).unwrap(), "{} {prefix:?}", case.name);
                }
            }
        }
    }

    #[test]
    fn unevaluated_accepts_tokens_spanning_multiple_and_nested_boundaries() {
        let program = StructuredProgram::compile(ir(
            r#"{"properties":{"known":true},"unevaluatedProperties":true}"#,
        ))
        .expect("program");
        let mut whole = program.new_matcher().expect("matcher");
        assert_eq!(whole.backend_name(), "incremental");
        assert!(whole
            .advance(br#"{"known":[1,{"deep":[2,3]}],"extra":{"x":true}}"#)
            .unwrap());
        assert!(whole.is_accepting());

        let mut split = program.new_matcher().expect("matcher");
        assert!(split.advance(br#"{"known":"#).unwrap());
        assert!(split.advance(br#"[{"deep":[1,{"x":2}]}]"#).unwrap());
        assert!(split.advance(br#", "extra":false}"#).unwrap());
        assert!(split.is_accepting());

        let mut rollback = program.new_matcher().expect("matcher");
        assert!(rollback.advance(br#"{"known":[1,2]"#).unwrap());
        assert!(!rollback.advance(br#",]}}"#).unwrap());
        assert!(rollback.advance(br#", "extra":3}"#).unwrap());
        assert!(rollback.is_accepting());
    }

    fn boolean_vocabulary(eos: u32) -> VocabularyHandle {
        let mut tokens = FxHashMap::default();
        for b in 0u32..256 {
            tokens.insert(vec![b as u8], vec![b]);
        }
        VocabularyHandle::new(Arc::new(build_vocabulary(eos, tokens).unwrap())).unwrap()
    }

    #[test]
    fn word_mask_is_bit_identical_to_the_byte_mask_including_the_eos_word() {
        let program = program(r#"{"not":{"type":"string"}}"#);
        let handle = boolean_vocabulary(256);
        let mut byte_matcher = program.new_matcher().unwrap();
        let mut word_matcher = program.new_matcher().unwrap();

        let mut byte_out = vec![0u8; 4 * handle.mask_vocab_size().div_ceil(32)];
        byte_matcher
            .write_record_mask_le_bytes_into(
                handle.mask_vocab_size(),
                handle.iter_records(),
                &mut byte_out,
            )
            .unwrap();
        let mut word_out = vec![0u32; handle.mask_vocab_size().div_ceil(32)];
        word_matcher
            .write_record_mask_words_into(
                handle.mask_vocab_size(),
                handle.iter_records(),
                &mut word_out,
            )
            .unwrap();

        let expected: Vec<u32> = byte_out
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(word_out, expected);
    }

    #[test]
    fn word_mask_sets_bit_31_correctly_and_clears_a_dirty_buffer() {
        // No JSON value starts with a raw byte, so test bit 31 (byte 63='?') inside a string.
        let program = program(r#"{"not":{"type":"integer"}}"#);
        let handle = boolean_vocabulary(256);
        let mut matcher = program.new_matcher().unwrap();
        assert!(matcher.advance(b"\"").unwrap());
        let words = handle.mask_vocab_size().div_ceil(32);

        let mut out = vec![0xFFFF_FFFFu32; words]; // dirty: every bit set before the call
        matcher
            .write_record_mask_words_into(handle.mask_vocab_size(), handle.iter_records(), &mut out)
            .unwrap();

        assert_eq!(
            out[1] >> 31 & 1,
            1,
            "byte 63 ('?') must be legal and land on word 1 bit 31"
        );
        let vocab_size = handle.mask_vocab_size() as u32;
        for (word_index, word) in out.iter().enumerate() {
            for bit in 0..32 {
                let token_id = word_index as u32 * 32 + bit;
                if token_id >= vocab_size {
                    assert_eq!(word >> bit & 1, 0, "padding bit {token_id} must stay clear");
                }
            }
        }
    }

    #[test]
    fn word_mask_rejects_wrong_length_without_touching_the_buffer() {
        let program = program(r#"{"not":{"type":"string"}}"#);
        let handle = boolean_vocabulary(256);
        let mut matcher = program.new_matcher().unwrap();

        let mut wrong = vec![0xFFFF_FFFFu32; 3];
        matcher
            .write_record_mask_words_into(
                handle.mask_vocab_size(),
                handle.iter_records(),
                &mut wrong,
            )
            .unwrap_err();
        assert_eq!(
            wrong,
            vec![0xFFFF_FFFFu32; 3],
            "a length-mismatch error must not touch the buffer"
        );
    }

    #[test]
    fn word_mask_zero_allocation_hot_path_matches_records_across_many_steps() {
        let program = program(
            r#"{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}"#,
        );
        let handle = boolean_vocabulary(256);
        let mut matcher = program.new_matcher().unwrap();
        let mut out = vec![0u32; handle.mask_vocab_size().div_ceil(32)];

        for byte in b"{\"ok\":tru" {
            matcher
                .write_record_mask_words_into(
                    handle.mask_vocab_size(),
                    handle.iter_records(),
                    &mut out,
                )
                .unwrap();
            let bit_set = |id: u32| out[(id / 32) as usize] >> (id % 32) & 1 == 1;
            assert!(
                bit_set(*byte as u32),
                "byte {byte} must be legal at this prefix"
            );
            assert!(matcher.advance(&[*byte]).unwrap());
        }
    }
}
