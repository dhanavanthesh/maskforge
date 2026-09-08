//! The immutable, schema-derived compiled plan the incremental engine walks. Everything reusable
//! across every session (compiled child engines, property indices) lives here, built exactly once.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::automaton::{
    build_from_regex, build_from_unicode_regex, Combinator, ProductAutomaton, RefEngine,
};
use crate::compile::{compile_node, number_syntax_engine};
use crate::error::{CompileError, ErrorCode, LimitKind, Stage};
use crate::ir::{
    AdditionalPolicy, AnchorId, LitSlice, Node, ResourceId, ScalarLit, SchemaIR, UnevaluatedKind,
};
use crate::primitives::NodeId;

use super::limits::{charge, resource_limit, StructuredLimits};

#[derive(Debug)]
pub(crate) enum PlanCompileFailure {
    Fatal(CompileError),
}

impl From<CompileError> for PlanCompileFailure {
    fn from(error: CompileError) -> Self {
        Self::Fatal(error)
    }
}

impl std::ops::Deref for PlanCompileFailure {
    type Target = CompileError;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Fatal(error) => error,
        }
    }
}

/// Index into `ObjectPlan::known`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct PropertyId(pub(crate) u32);

pub(crate) struct PropertyPlan {
    pub(crate) value: NodeId,
    pub(crate) maybe_satisfiable: bool,
}

/// One compiled `patternProperties` entry: `engine` matches candidate keys against the pattern.
pub(crate) struct CompiledPropertyPattern {
    pub(crate) engine: RefEngine,
    pub(crate) value: NodeId,
    pub(crate) maybe_satisfiable: bool,
}

pub(crate) enum AdditionalPlan {
    Forbid,
    Open,
    AllowAny,
    Schema(NodeId),
}

#[derive(Clone, Copy)]
pub(crate) struct DependentSchemaPlan {
    pub(crate) schema: NodeId,
}

#[derive(Clone, Copy)]
pub(crate) struct DependentRequiredRule {
    pub(crate) trigger: u32,
    pub(crate) required_start: u32,
    pub(crate) required_len: u32,
}

pub(crate) struct DependentRequiredPlan {
    pub(crate) name_by_text: FxHashMap<String, u32>,
    pub(crate) rules: Box<[DependentRequiredRule]>,
    pub(crate) required_ids: Box<[u32]>,
    pub(crate) name_count: u32,
}

pub(crate) struct ObjectPlan {
    pub(crate) known_by_name: FxHashMap<String, PropertyId>,
    pub(crate) known: Vec<PropertyPlan>,
    pub(crate) patterns: Vec<CompiledPropertyPattern>,
    /// Complete dynamic-key language: union of satisfiable patterns minus every false pattern.
    pub(crate) viable_pattern: Option<Box<RefEngine>>,
    /// One bit per `known` property, set iff required - a session seeds its "still missing"
    /// bitset from this and clears bits as properties are observed.
    pub(crate) required_template: Vec<u64>,
    pub(crate) additional: AdditionalPlan,
    pub(crate) property_names: Option<NodeId>,
    pub(crate) min_properties: Option<u32>,
    pub(crate) max_properties: Option<u32>,
    pub(crate) dependent_by_name: FxHashMap<String, u32>,
    pub(crate) dependent_schemas: Box<[DependentSchemaPlan]>,
    pub(crate) dependent_required: Option<DependentRequiredPlan>,
}

/// What an element past `prefix` must satisfy.
#[derive(Clone, Copy)]
pub(crate) enum TailPlan {
    /// A closed tuple: no element is allowed past `prefix`.
    Closed,
    Schema(NodeId),
    /// `items: true`: any syntactically valid JSON value, matched via `AnyJson`.
    AllowAny,
}

/// What counts as a `contains` match.
#[derive(Clone, Copy)]
pub(crate) enum ContainsMatch {
    Never,
    Always,
    Schema(NodeId),
}

pub(crate) struct ContainsPlan {
    pub(crate) matches: ContainsMatch,
    pub(crate) min: u32,
    pub(crate) max: Option<u32>,
}

pub(crate) struct StringPlan {
    pub(crate) min_scalars: u32,
    pub(crate) max_scalars: Option<u32>,
    pub(crate) pattern: Option<Box<RefEngine>>,
}

#[derive(Clone, Copy)]
pub(crate) struct NumberPlan {
    pub(crate) integer_only: bool,
    pub(crate) minimum: Option<(i128, u32, bool)>,
    pub(crate) maximum: Option<(i128, u32, bool)>,
    pub(crate) multiple_of: Option<(u64, u32)>,
}

pub(crate) struct StringEnumPlan {
    bytes: Box<[u8]>,
    offsets: Box<[u32]>,
}

impl StringEnumPlan {
    pub(crate) fn member_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub(crate) fn member(&self, index: usize) -> Option<&[u8]> {
        let start = usize::try_from(*self.offsets.get(index)?).ok()?;
        let end = usize::try_from(*self.offsets.get(index.checked_add(1)?)?).ok()?;
        self.bytes.get(start..end)
    }

    pub(crate) fn byte_at(&self, index: usize, offset: usize) -> Option<u8> {
        self.member(index)?.get(offset).copied()
    }

    pub(crate) fn narrow(&self, lo: u32, hi: u32, position: u32, byte: u8) -> Option<(u32, u32)> {
        let lo = usize::try_from(lo).ok()?;
        let hi = usize::try_from(hi).ok()?;
        let position = usize::try_from(position).ok()?;
        if lo >= hi || hi > self.member_count() {
            return None;
        }
        if hi - lo == 1 {
            return (self.byte_at(lo, position) == Some(byte)).then(|| {
                (
                    u32::try_from(lo).expect("validated enum index fits u32"),
                    u32::try_from(hi).expect("validated enum index fits u32"),
                )
            });
        }
        let lower = self.lower_bound_byte(lo, hi, position, byte, false);
        let upper = self.lower_bound_byte(lower, hi, position, byte, true);
        (lower < upper).then(|| {
            (
                u32::try_from(lower).expect("validated enum index fits u32"),
                u32::try_from(upper).expect("validated enum index fits u32"),
            )
        })
    }

    pub(crate) fn contains_exact(&self, lo: u32, hi: u32, len: u32) -> bool {
        let (Ok(lo), Ok(hi), Ok(len)) = (
            usize::try_from(lo),
            usize::try_from(hi),
            usize::try_from(len),
        ) else {
            return false;
        };
        lo < hi
            && hi <= self.member_count()
            && self.member(lo).is_some_and(|value| value.len() == len)
    }

    pub(crate) fn can_continue(&self, lo: u32, hi: u32, position: u32) -> bool {
        let (Ok(lo), Ok(hi), Ok(position)) = (
            usize::try_from(lo),
            usize::try_from(hi),
            usize::try_from(position),
        ) else {
            return false;
        };
        lo < hi
            && hi <= self.member_count()
            && self
                .member(hi - 1)
                .is_some_and(|value| value.len() > position)
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.bytes
            .len()
            .checked_add(
                self.offsets
                    .len()
                    .checked_mul(size_of::<u32>())
                    .unwrap_or(usize::MAX),
            )
            .unwrap_or(usize::MAX)
    }

    fn lower_bound_byte(
        &self,
        mut lo: usize,
        mut hi: usize,
        position: usize,
        target: u8,
        include_equal: bool,
    ) -> usize {
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let advances = match self.byte_at(mid, position) {
                None => true,
                Some(value) => value < target || (include_equal && value == target),
            };
            if advances {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CombinatorKind {
    All,
    Any,
    ExactlyOne,
}

pub(crate) struct CombinatorPlan {
    pub(crate) kind: CombinatorKind,
    pub(crate) branches: Box<[NodeId]>,
}

/// A schema-bounded array too large to unroll into one byte-DFA.
pub(crate) struct ArrayPlan {
    /// Positional element schemas (`prefixItems`), validated in order.
    pub(crate) prefix: Vec<NodeId>,
    /// What every element past `prefix` must satisfy.
    pub(crate) tail: TailPlan,
    pub(crate) min_items: u32,
    pub(crate) max_items: Option<u32>,
    pub(crate) unique_items: bool,
    /// Exact capacity when every homogeneous item comes from one finite string language.
    pub(crate) finite_unique_capacity: Option<u32>,
    pub(crate) contains: Option<ContainsPlan>,
    pub(crate) tail_annotates: bool,
}

pub(crate) struct UnevaluatedPlan {
    pub(crate) kind: UnevaluatedKind,
    pub(crate) scope: NodeId,
    pub(crate) unevaluated: NodeId,
    /// `false` makes an unannotated member terminal rather than merely constrained.
    pub(crate) rejects_unevaluated: bool,
    /// Exact decoded key language contributed by the scope's in-place applicators. `None` means
    /// the compiler could not prove that language finite, so runtime must remain conservative.
    pub(crate) finite_evaluated_keys: Option<FiniteKeyTrie>,
}

#[derive(Clone, Copy)]
struct FiniteKeyNode {
    first_edge: u32,
    edge_len: u32,
    terminal: bool,
}

#[derive(Clone, Copy)]
struct FiniteKeyEdge {
    byte: u8,
    target: u32,
}

pub(crate) struct FiniteKeyTrie {
    nodes: Box<[FiniteKeyNode]>,
    edges: Box<[FiniteKeyEdge]>,
    /// UTF-8 names indexed by the subtree bitsets. Offsets has one sentinel entry.
    name_bytes: Box<[u8]>,
    name_offsets: Box<[u32]>,
    words_per_node: u32,
    subtree_ids: Box<[u64]>,
    /// Bounded pattern languages whose union contributes evaluated property names. False-valued
    /// or otherwise ambiguous interactions still omit the optional viability proof at plan time.
    patterns: Vec<RefEngine>,
}

impl FiniteKeyTrie {
    fn name(&self, id: usize) -> Option<&str> {
        let start = usize::try_from(*self.name_offsets.get(id)?).ok()?;
        let end = usize::try_from(*self.name_offsets.get(id.checked_add(1)?)?).ok()?;
        std::str::from_utf8(self.name_bytes.get(start..end)?).ok()
    }

    pub(crate) fn permits(
        &self,
        prefix: &str,
        complete: bool,
        seen_pattern_completions: usize,
        mut is_seen: impl FnMut(&str) -> bool,
    ) -> bool {
        let pattern_viable = self.patterns.iter().any(|engine| {
            let Some(state) = engine.consume_token(engine.start(), prefix.as_bytes()) else {
                return false;
            };
            if complete {
                engine.is_accepting(state) && !is_seen(prefix)
            } else if self.patterns.len() == 1 {
                engine.residual_cardinality(state) > seen_pattern_completions as u64
            } else {
                // The union can overlap. Its per-pattern cardinality remains a proof when there
                // are no exclusions; otherwise retain the conservative exact-walker fallback.
                seen_pattern_completions == 0 && engine.residual_cardinality(state) > 0
            }
        });
        let mut node_index = 0usize;
        for &byte in prefix.as_bytes() {
            let Some(node) = self.nodes.get(node_index) else {
                return pattern_viable;
            };
            let start = node.first_edge as usize;
            let Some(end) = start.checked_add(node.edge_len as usize) else {
                return pattern_viable;
            };
            let Some(edges) = self.edges.get(start..end) else {
                return pattern_viable;
            };
            let Ok(offset) = edges.binary_search_by_key(&byte, |edge| edge.byte) else {
                return pattern_viable;
            };
            node_index = edges[offset].target as usize;
        }
        let Some(node) = self.nodes.get(node_index) else {
            return pattern_viable;
        };
        if complete {
            return (node.terminal && !is_seen(prefix)) || pattern_viable;
        }
        let words = self.words_per_node as usize;
        let Some(start) = node_index.checked_mul(words) else {
            return false;
        };
        let Some(end) = start.checked_add(words) else {
            return false;
        };
        let Some(bits) = self.subtree_ids.get(start..end) else {
            return false;
        };
        pattern_viable
            || bits.iter().enumerate().any(|(word_index, &word)| {
                let mut remaining = word;
                while remaining != 0 {
                    let bit = remaining.trailing_zeros() as usize;
                    let Some(id) = word_index
                        .checked_mul(BITSET_WORD_BITS)
                        .and_then(|base| base.checked_add(bit))
                    else {
                        return false;
                    };
                    if self.name(id).is_some_and(|name| !is_seen(name)) {
                        return true;
                    }
                    remaining &= remaining - 1;
                }
                false
            })
    }

    /// Checks viability after a pending decoder finishes one scalar.
    /// The scalar is walked as UTF-8 without allocating a candidate key.
    pub(crate) fn permits_after_scalar(
        &self,
        prefix: &str,
        scalar: char,
        seen_pattern_completions: usize,
        mut is_seen: impl FnMut(&str) -> bool,
    ) -> bool {
        let mut encoded = [0u8; 4];
        let scalar_bytes = scalar.encode_utf8(&mut encoded).as_bytes();
        let pattern_viable = self.patterns.iter().any(|engine| {
            let Some(state) = engine
                .consume_token(engine.start(), prefix.as_bytes())
                .and_then(|state| engine.consume_token(state, scalar_bytes))
            else {
                return false;
            };
            if self.patterns.len() == 1 {
                engine.residual_cardinality(state) > seen_pattern_completions as u64
            } else {
                seen_pattern_completions == 0 && engine.residual_cardinality(state) > 0
            }
        });

        let mut node_index = 0usize;
        for &byte in prefix.as_bytes().iter().chain(scalar_bytes) {
            let Some(node) = self.nodes.get(node_index) else {
                return pattern_viable;
            };
            let start = node.first_edge as usize;
            let Some(end) = start.checked_add(node.edge_len as usize) else {
                return pattern_viable;
            };
            let Some(edges) = self.edges.get(start..end) else {
                return pattern_viable;
            };
            let Ok(offset) = edges.binary_search_by_key(&byte, |edge| edge.byte) else {
                return pattern_viable;
            };
            node_index = edges[offset].target as usize;
        }
        let words = self.words_per_node as usize;
        let Some(start) = node_index.checked_mul(words) else {
            return pattern_viable;
        };
        let Some(end) = start.checked_add(words) else {
            return pattern_viable;
        };
        let Some(bits) = self.subtree_ids.get(start..end) else {
            return pattern_viable;
        };
        pattern_viable
            || bits.iter().enumerate().any(|(word_index, &word)| {
                let mut remaining = word;
                while remaining != 0 {
                    let bit = remaining.trailing_zeros() as usize;
                    let Some(id) = word_index
                        .checked_mul(BITSET_WORD_BITS)
                        .and_then(|base| base.checked_add(bit))
                    else {
                        return false;
                    };
                    if self.name(id).is_some_and(|name| !is_seen(name)) {
                        return true;
                    }
                    remaining &= remaining - 1;
                }
                false
            })
    }

    /// Checks viability after a high surrogate resolves within `start..=end`.
    /// Exact names use direct lookup; patterns use a symbolic UTF-8 interval walk.
    pub(crate) fn permits_after_supplementary_range(
        &self,
        prefix: &str,
        start: u32,
        end: u32,
        mut is_seen: impl FnMut(&str) -> bool,
    ) -> Option<bool> {
        let exact_viable = (0..self.name_offsets.len().saturating_sub(1)).any(|id| {
            self.name(id).is_some_and(|name| {
                !is_seen(name)
                    && name.strip_prefix(prefix).is_some_and(|rest| {
                        rest.chars()
                            .next()
                            .is_some_and(|scalar| (start..=end).contains(&u32::from(scalar)))
                    })
            })
        });
        if exact_viable {
            return Some(true);
        }
        if self.patterns.is_empty() {
            return Some(false);
        }
        let mut unknown = false;
        for engine in &self.patterns {
            let Some(state) = engine.consume_token(engine.start(), prefix.as_bytes()) else {
                continue;
            };
            match engine.can_consume_supplementary_range(state, start, end) {
                Some(true) => return Some(true),
                Some(false) => {}
                None => unknown = true,
            }
        }
        if unknown {
            None
        } else {
            Some(false)
        }
    }

    /// Viability after a just-opened JSON escape, which may decode to any Unicode scalar.
    pub(crate) fn permits_after_any_scalar(
        &self,
        prefix: &str,
        seen_pattern_completions: usize,
        mut is_seen: impl FnMut(&str) -> bool,
    ) -> Option<bool> {
        if (0..self.name_offsets.len().saturating_sub(1)).any(|id| {
            self.name(id).is_some_and(|name| {
                !is_seen(name)
                    && name
                        .strip_prefix(prefix)
                        .is_some_and(|rest| !rest.is_empty())
            })
        }) {
            return Some(true);
        }
        let mut unknown = false;
        for engine in &self.patterns {
            let Some(state) = engine.consume_token(engine.start(), prefix.as_bytes()) else {
                continue;
            };
            match engine.can_consume_any_scalar(state) {
                Some(true) if seen_pattern_completions == 0 => return Some(true),
                Some(true) => unknown = true,
                Some(false) => {}
                None => unknown = true,
            }
        }
        (!unknown).then_some(false)
    }

    pub(crate) fn pattern_accepts(&self, name: &str) -> bool {
        self.patterns
            .iter()
            .any(|engine| engine.accepts(name.as_bytes()))
    }

    fn retained_bytes(&self) -> usize {
        self.nodes
            .len()
            .saturating_mul(size_of::<FiniteKeyNode>())
            .saturating_add(self.edges.len().saturating_mul(size_of::<FiniteKeyEdge>()))
            .saturating_add(self.name_bytes.len())
            .saturating_add(self.name_offsets.len().saturating_mul(size_of::<u32>()))
            .saturating_add(self.subtree_ids.len().saturating_mul(size_of::<u64>()))
            .saturating_add(
                self.patterns
                    .iter()
                    .map(engine_retained_bytes)
                    .sum::<usize>(),
            )
    }
}

/// The per-node compiled plan, densely indexed by `NodeId` (never an `FxHashMap<NodeId, _>` walk).
pub(crate) enum NodePlan {
    /// A subtree compilable as one byte-DFA - the incremental walk delegates to it directly.
    Regular(Box<RefEngine>),
    /// An `OpenObject` (or shuffle-capacity-exceeding closed `Object`): incremental key/value state.
    Object(ObjectPlan),
    /// An array too large to unroll into a byte-DFA: incremental element-count state.
    Array(ArrayPlan),
    String(StringPlan),
    StringEnum(StringEnumPlan),
    Number {
        engine: Box<RefEngine>,
        constraint: NumberPlan,
    },
    /// A transparent local reference edge; the target owns all runtime state.
    Ref {
        target: NodeId,
    },
    DynamicRef {
        initial_target: NodeId,
        anchor: AnchorId,
    },
    Combinator(CombinatorPlan),
    Negation {
        inner: NodeId,
    },
    Unevaluated(UnevaluatedPlan),
    /// Defensive sentinel rejected by production plan construction.
    Unsupported,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct DynamicBinding {
    pub(crate) name: u32,
    pub(crate) resource: ResourceId,
    pub(crate) target: NodeId,
}

pub(crate) struct StructuredPlan {
    pub(crate) nodes: Vec<NodePlan>,
    pub(crate) root: NodeId,
    pub(crate) annotation_required: Box<[u64]>,
    pub(crate) node_resources: Box<[ResourceId]>,
    dynamic_anchor_names: Box<[u32]>,
    dynamic_bindings: Box<[DynamicBinding]>,
    pub(crate) dynamic_scope_enabled: bool,
    /// Bounds session key scratch by object cursors that can parse one key together.
    pub(crate) max_concurrent_key_cursors: usize,
    pub(crate) retained_bytes: usize,
    pub(crate) limits: StructuredLimits,
}

#[derive(Clone, Copy)]
struct VisitFrame {
    node: NodeId,
    next_child: usize,
}

#[derive(Clone, Copy)]
enum SameInstanceEdge {
    Reference(NodeId),
    Semantic(NodeId),
}

impl SameInstanceEdge {
    fn target(self) -> NodeId {
        match self {
            Self::Reference(node) | Self::Semantic(node) => node,
        }
    }
}

/// Word bits per `required_template`/session bitset entry.
pub(crate) const BITSET_WORD_BITS: usize = u64::BITS as usize;

fn words_for(count: usize) -> usize {
    count.div_ceil(BITSET_WORD_BITS)
}

fn compile_scratch_bytes(node_count: usize) -> Result<usize, CompileError> {
    let node_table = node_count
        .checked_mul(size_of::<NodePlan>())
        .ok_or_else(plan_allocation_error)?;
    let compiled = node_count
        .checked_mul(size_of::<bool>())
        .ok_or_else(plan_allocation_error)?;
    let regular = compiled;
    let visit_state = node_count
        .checked_mul(size_of::<u8>())
        .ok_or_else(plan_allocation_error)?;
    let traversal = node_count
        .checked_mul(size_of::<(NodeId, bool)>())
        .ok_or_else(plan_allocation_error)?;
    let queued = compiled;
    let worklist = node_count
        .checked_mul(size_of::<NodeId>())
        .ok_or_else(plan_allocation_error)?;
    let cycle_colors = node_count
        .checked_mul(size_of::<u8>())
        .ok_or_else(plan_allocation_error)?;
    let cycle_stack = node_count
        .checked_mul(size_of::<VisitFrame>())
        .ok_or_else(plan_allocation_error)?;
    let cycle_positions = node_count
        .checked_mul(size_of::<usize>())
        .ok_or_else(plan_allocation_error)?;
    [
        node_table,
        compiled,
        regular,
        visit_state,
        traversal,
        queued,
        worklist,
        cycle_colors,
        cycle_stack,
        cycle_positions,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| total.checked_add(bytes))
    .ok_or_else(plan_allocation_error)
}

impl StructuredPlan {
    #[cfg(test)]
    pub(crate) fn unsupported_for_test() -> Self {
        Self {
            nodes: vec![NodePlan::Unsupported],
            root: NodeId(0),
            annotation_required: Box::new([]),
            node_resources: Box::new([ResourceId(0)]),
            dynamic_anchor_names: Box::new([]),
            dynamic_bindings: Box::new([]),
            dynamic_scope_enabled: false,
            max_concurrent_key_cursors: 0,
            retained_bytes: 0,
            limits: StructuredLimits::default(),
        }
    }

    /// Compiles `ir` into a complete production plan or returns the exact fatal error.
    pub(crate) fn compile(
        ir: Arc<SchemaIR>,
        limits: StructuredLimits,
    ) -> Result<Self, PlanCompileFailure> {
        Self::compile_impl(ir, limits, None)
    }

    #[cfg(feature = "bench-internals")]
    pub(crate) fn compile_profiled(
        ir: Arc<SchemaIR>,
        limits: StructuredLimits,
    ) -> Result<(Self, u64, u64), PlanCompileFailure> {
        let started = std::time::Instant::now();
        let mut scc_ns = 0u64;
        let plan = Self::compile_impl(ir, limits, Some(&mut scc_ns))?;
        let total_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        Ok((plan, total_ns, scc_ns))
    }

    fn compile_impl(
        ir: Arc<SchemaIR>,
        limits: StructuredLimits,
        scc_ns: Option<&mut u64>,
    ) -> Result<Self, PlanCompileFailure> {
        if ir.diagnostics().next().is_some() {
            return Err(CompileError::new(
                ErrorCode::Unsupported,
                Stage::L3,
                "schema has unsupported diagnostics",
            )
            .into());
        }
        let node_count = ir.node_count();
        let dynamic_scope_enabled = ir
            .nodes()
            .any(|node| matches!(node, Node::DynamicRef { .. }));
        let has_unevaluated = ir
            .nodes()
            .any(|node| matches!(node, Node::Unevaluated { .. }));
        let compile_peak = compile_scratch_bytes(node_count)?;
        let node_table_bytes = node_count
            .checked_mul(size_of::<NodePlan>())
            .ok_or_else(plan_allocation_error)?;
        let scratch_bytes = compile_peak
            .checked_sub(node_table_bytes)
            .ok_or_else(plan_allocation_error)?;
        let annotation_bytes = (if has_unevaluated {
            words_for(node_count)
        } else {
            0
        })
        .checked_mul(size_of::<u64>())
        .ok_or_else(plan_allocation_error)?;
        let initial_peak = node_table_bytes
            .checked_add(scratch_bytes)
            .and_then(|bytes| bytes.checked_add(annotation_bytes))
            .ok_or_else(plan_allocation_error)?;
        if initial_peak > limits.max_plan_bytes {
            return Err(resource_limit(
                LimitKind::PlanBytes,
                initial_peak,
                limits.max_plan_bytes,
                Stage::L3,
            )
            .into());
        }
        let mut retained_bytes = 0usize;
        charge(
            &mut retained_bytes,
            node_table_bytes,
            limits.max_plan_bytes,
            LimitKind::PlanBytes,
            Stage::L3,
        )?;
        charge(
            &mut retained_bytes,
            annotation_bytes,
            limits.max_plan_bytes,
            LimitKind::PlanBytes,
            Stage::L3,
        )?;
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(node_count)
            .map_err(|_| plan_allocation_error())?;
        nodes.resize_with(node_count, || NodePlan::Unsupported);
        let mut compiled = Vec::new();
        compiled
            .try_reserve_exact(node_count)
            .map_err(|_| plan_allocation_error())?;
        compiled.resize(node_count, false);
        let (dynamic_anchor_names, dynamic_bindings, dynamic_bytes) = if dynamic_scope_enabled {
            build_dynamic_bindings(&ir)?
        } else {
            (
                Vec::<u32>::new().into_boxed_slice(),
                Vec::<DynamicBinding>::new().into_boxed_slice(),
                0,
            )
        };
        charge(
            &mut retained_bytes,
            dynamic_bytes,
            limits.max_plan_bytes,
            LimitKind::PlanBytes,
            Stage::L3,
        )?;
        let annotation_required =
            compute_annotation_required(&ir, node_count, &dynamic_anchor_names, &dynamic_bindings)?;
        let regular = classify_regular(&ir, node_count, &annotation_required)?;
        let mut worklist = Vec::new();
        worklist
            .try_reserve_exact(node_count)
            .map_err(|_| plan_allocation_error())?;
        worklist.push(ir.root());
        let mut queued = Vec::new();
        queued
            .try_reserve_exact(node_count)
            .map_err(|_| plan_allocation_error())?;
        queued.resize(node_count, false);
        let root_index = usize::try_from(ir.root().get()).map_err(|_| plan_allocation_error())?;
        queued[root_index] = true;
        for binding in dynamic_bindings.iter().copied() {
            queue_child(binding.target, &mut worklist, &mut queued);
        }
        while let Some(id) = worklist.pop() {
            let idx = usize::try_from(id.get()).map_err(|_| plan_allocation_error())?;
            if compiled[idx] {
                continue;
            }
            let mut context = PlanBuild {
                regular: &regular,
                limits: &limits,
                retained_bytes: &mut retained_bytes,
                scratch_bytes,
                temporary_bytes: 0,
                worklist: &mut worklist,
                queued: &mut queued,
            };
            let plan = build_node_plan(&ir, id, &mut context)?;
            nodes[idx] = plan;
            compiled[idx] = true;
        }
        #[cfg(feature = "bench-internals")]
        let scc_started = scc_ns.as_ref().map(|_| std::time::Instant::now());
        resolve_same_instance_cycles(&mut nodes, &queued, &mut retained_bytes, limits)?;
        let max_concurrent_key_cursors =
            max_concurrent_key_cursors(&nodes, ir.root(), limits.max_active_validators)?;
        #[cfg(feature = "bench-internals")]
        if let (Some(output), Some(started)) = (scc_ns, scc_started) {
            *output = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        }
        #[cfg(not(feature = "bench-internals"))]
        let _ = scc_ns;
        let root = ir.root();
        let root_resource = ir
            .node_resource(root)
            .ok_or_else(|| invalid_ir("root resource is invalid"))?;
        let mut node_resources = Vec::new();
        node_resources
            .try_reserve_exact(nodes.len())
            .map_err(|_| plan_allocation_error())?;
        for index in 0..nodes.len() {
            let resource = if index < node_count {
                ir.node_resource(NodeId(
                    u32::try_from(index).map_err(|_| plan_allocation_error())?,
                ))
                .ok_or_else(|| invalid_ir("node resource is invalid"))?
            } else {
                root_resource
            };
            node_resources.push(resource);
        }
        charge(
            &mut retained_bytes,
            node_resources
                .capacity()
                .saturating_mul(size_of::<ResourceId>()),
            limits.max_plan_bytes,
            LimitKind::PlanBytes,
            Stage::L3,
        )?;
        Ok(Self {
            nodes,
            root,
            annotation_required,
            node_resources: node_resources.into_boxed_slice(),
            dynamic_anchor_names,
            dynamic_bindings,
            dynamic_scope_enabled,
            max_concurrent_key_cursors,
            retained_bytes,
            limits,
        })
    }

    pub(crate) fn node(&self, id: NodeId) -> &NodePlan {
        self.nodes
            .get(usize::try_from(id.get()).unwrap_or(usize::MAX))
            .unwrap_or(&NodePlan::Unsupported)
    }

    pub(crate) fn annotations_required(&self, id: NodeId) -> bool {
        usize::try_from(id.get())
            .ok()
            .is_some_and(|index| bit_is_set(&self.annotation_required, index))
    }

    pub(crate) fn node_resource(&self, id: NodeId) -> Option<ResourceId> {
        self.node_resources.get(id.get() as usize).copied()
    }

    pub(crate) fn dynamic_target(&self, anchor: AnchorId, resource: ResourceId) -> Option<NodeId> {
        let name = *self.dynamic_anchor_names.get(anchor.get() as usize)?;
        if name == u32::MAX {
            return None;
        }
        self.dynamic_bindings
            .binary_search_by_key(&(name, resource), |binding| {
                (binding.name, binding.resource)
            })
            .ok()
            .and_then(|index| self.dynamic_bindings.get(index))
            .map(|binding| binding.target)
    }

    pub(crate) fn dynamic_anchor_capacity(&self) -> usize {
        self.dynamic_anchor_names.len()
    }
}

type DynamicBindings = (Box<[u32]>, Box<[DynamicBinding]>, usize);

fn build_dynamic_bindings(ir: &SchemaIR) -> Result<DynamicBindings, CompileError> {
    let mut anchor_names = Vec::new();
    anchor_names
        .try_reserve_exact(ir.anchors().len())
        .map_err(|_| plan_allocation_error())?;
    anchor_names.resize(ir.anchors().len(), u32::MAX);
    let mut names: FxHashMap<&str, u32> = FxHashMap::default();
    names
        .try_reserve(ir.anchors().len())
        .map_err(|_| plan_allocation_error())?;
    let mut bindings = Vec::new();
    bindings
        .try_reserve_exact(ir.anchors().len())
        .map_err(|_| plan_allocation_error())?;
    for (resource_index, resource) in ir.resources().iter().enumerate() {
        let resource_id =
            ResourceId(u32::try_from(resource_index).map_err(|_| plan_allocation_error())?);
        let start = resource.dynamic_anchors.off as usize;
        let end = start
            .checked_add(resource.dynamic_anchors.len as usize)
            .ok_or_else(plan_allocation_error)?;
        let anchors = ir
            .anchors()
            .get(start..end)
            .ok_or_else(|| invalid_ir("dynamic anchor run is invalid"))?;
        for (offset, anchor) in anchors.iter().enumerate() {
            let name = ir
                .str_at(anchor.name)
                .ok_or_else(|| invalid_ir("dynamic anchor name is invalid"))?;
            let next = u32::try_from(names.len()).map_err(|_| plan_allocation_error())?;
            let name_id = *names.entry(name).or_insert(next);
            let anchor_index = start
                .checked_add(offset)
                .ok_or_else(plan_allocation_error)?;
            anchor_names[anchor_index] = name_id;
            bindings.push(DynamicBinding {
                name: name_id,
                resource: resource_id,
                target: anchor.target,
            });
        }
    }
    bindings.sort_unstable();
    let retained = anchor_names
        .capacity()
        .checked_mul(size_of::<u32>())
        .and_then(|bytes| {
            bytes.checked_add(
                bindings
                    .capacity()
                    .checked_mul(size_of::<DynamicBinding>())?,
            )
        })
        .ok_or_else(plan_allocation_error)?;
    Ok((
        anchor_names.into_boxed_slice(),
        bindings.into_boxed_slice(),
        retained,
    ))
}

pub(crate) fn object_plan_of(plan: &StructuredPlan, node: NodeId) -> &ObjectPlan {
    match plan.node(node) {
        NodePlan::Object(op) => op,
        _ => unreachable!("object_plan_of called on a non-object node"),
    }
}

pub(crate) fn array_plan_of(plan: &StructuredPlan, node: NodeId) -> &ArrayPlan {
    match plan.node(node) {
        NodePlan::Array(ap) => ap,
        _ => unreachable!("array_plan_of called on a non-array node"),
    }
}

fn classify_regular(
    ir: &SchemaIR,
    node_count: usize,
    annotation_required: &[u64],
) -> Result<Vec<bool>, PlanCompileFailure> {
    let mut regular = Vec::new();
    regular
        .try_reserve_exact(node_count)
        .map_err(|_| plan_allocation_error())?;
    for (index, route) in ir.route_table().routes().iter().enumerate() {
        regular.push(
            route.kind == crate::routing::ExecutionKind::Regular
                && !annotation_shape_required(
                    ir.node(NodeId(index as u32)),
                    annotation_required,
                    index,
                ),
        );
    }
    Ok(regular)
}

fn annotation_shape_required(
    node: Option<&Node>,
    annotation_required: &[u64],
    index: usize,
) -> bool {
    bit_is_set(annotation_required, index)
        && matches!(
            node,
            Some(
                Node::Object { .. }
                    | Node::OpenObject { .. }
                    | Node::Array { .. }
                    | Node::Tuple { .. }
                    | Node::Union { .. }
                    | Node::Intersection { .. }
                    | Node::ExactlyOne { .. }
                    | Node::Ref { .. }
                    | Node::DynamicRef { .. }
                    | Node::Unevaluated { .. }
            )
        )
}

fn bit_is_set(words: &[u64], index: usize) -> bool {
    words
        .get(index / BITSET_WORD_BITS)
        .is_some_and(|word| word & (1 << (index % BITSET_WORD_BITS)) != 0)
}

fn compute_annotation_required(
    ir: &SchemaIR,
    node_count: usize,
    dynamic_anchor_names: &[u32],
    dynamic_bindings: &[DynamicBinding],
) -> Result<Box<[u64]>, CompileError> {
    if !ir
        .nodes()
        .any(|node| matches!(node, Node::Unevaluated { .. }))
    {
        return Ok(Box::new([]));
    }
    let word_count = words_for(node_count);
    let mut required = Vec::new();
    required
        .try_reserve_exact(word_count)
        .map_err(|_| plan_allocation_error())?;
    required.resize(word_count, 0u64);
    let mut work = Vec::new();
    work.try_reserve_exact(node_count)
        .map_err(|_| plan_allocation_error())?;

    let mut mark = |id: NodeId, work: &mut Vec<NodeId>| -> Result<(), CompileError> {
        let index = usize::try_from(id.get()).map_err(|_| plan_allocation_error())?;
        if index >= node_count {
            return Err(invalid_ir("annotation node id out of range"));
        }
        let word = &mut required[index / BITSET_WORD_BITS];
        let mask = 1u64 << (index % BITSET_WORD_BITS);
        if *word & mask == 0 {
            *word |= mask;
            work.push(id);
        }
        Ok(())
    };

    for node in ir.nodes() {
        if let Node::Unevaluated { scope, .. } = node {
            mark(*scope, &mut work)?;
        }
    }
    while let Some(id) = work.pop() {
        let node = ir
            .node(id)
            .ok_or_else(|| invalid_ir("annotation node id out of range"))?;
        match node {
            Node::Ref { def } => {
                let target = ir
                    .def_target(*def)
                    .ok_or_else(|| invalid_ir("reference definition slot is invalid"))?;
                mark(target, &mut work)?;
            }
            Node::DynamicRef {
                initial_target,
                anchor,
            } => {
                mark(*initial_target, &mut work)?;
                let name = dynamic_anchor_names
                    .get(anchor.get() as usize)
                    .copied()
                    .filter(|name| *name != u32::MAX)
                    .ok_or_else(|| invalid_ir("dynamic annotation anchor is invalid"))?;
                let start = dynamic_bindings.partition_point(|binding| binding.name < name);
                let end = dynamic_bindings.partition_point(|binding| binding.name <= name);
                for binding in &dynamic_bindings[start..end] {
                    mark(binding.target, &mut work)?;
                }
            }
            Node::Intersection { branches }
            | Node::Union { branches }
            | Node::ExactlyOne { branches } => {
                for &branch in ir
                    .refs_at(*branches)
                    .ok_or_else(|| invalid_ir("annotation branch slice"))?
                {
                    mark(branch, &mut work)?;
                }
            }
            Node::Object { dependent, .. } | Node::OpenObject { dependent, .. } => {
                for &(_, child) in ir
                    .props_at(*dependent)
                    .ok_or_else(|| invalid_ir("annotation dependentSchemas slice"))?
                {
                    mark(child, &mut work)?;
                }
            }
            Node::Unevaluated { scope, .. } => mark(*scope, &mut work)?,
            _ => {}
        }
    }
    Ok(required.into_boxed_slice())
}

fn regular_for(id: NodeId, regular: &[bool]) -> bool {
    usize::try_from(id.get())
        .ok()
        .and_then(|index| regular.get(index))
        .copied()
        .unwrap_or(false)
}

fn build_contains_plan(
    contains: &Option<crate::ir::ContainsConstraint>,
    worklist: &mut Vec<NodeId>,
    queued: &mut [bool],
) -> Result<Option<ContainsPlan>, CompileError> {
    let Some(c) = contains else {
        return Ok(None);
    };
    let matches = match c.policy {
        crate::ir::ContainsPolicy::Never => ContainsMatch::Never,
        crate::ir::ContainsPolicy::Always => ContainsMatch::Always,
        crate::ir::ContainsPolicy::Schema(id) => {
            queue_child(id, worklist, queued);
            ContainsMatch::Schema(id)
        }
    };
    Ok(Some(ContainsPlan {
        matches,
        min: c.min,
        max: c.max,
    }))
}

struct PlanBuild<'a> {
    regular: &'a [bool],
    limits: &'a StructuredLimits,
    retained_bytes: &'a mut usize,
    scratch_bytes: usize,
    temporary_bytes: usize,
    worklist: &'a mut Vec<NodeId>,
    queued: &'a mut [bool],
}

fn regex_compile_peak(source_len: usize) -> Result<usize, CompileError> {
    source_len
        .checked_mul(128)
        .and_then(|bytes| bytes.checked_add(size_of::<RefEngine>()))
        .ok_or_else(plan_allocation_error)
}

fn ensure_compile_budget(context: &PlanBuild<'_>, additional: usize) -> Result<(), CompileError> {
    let projected = compile_budget_projection(context, additional)?;
    if projected > context.limits.max_plan_bytes {
        return Err(resource_limit(
            LimitKind::PlanBytes,
            projected,
            context.limits.max_plan_bytes,
            Stage::L3,
        ));
    }
    Ok(())
}

fn compile_budget_projection(
    context: &PlanBuild<'_>,
    additional: usize,
) -> Result<usize, CompileError> {
    context
        .retained_bytes
        .checked_add(context.scratch_bytes)
        .and_then(|n| n.checked_add(context.temporary_bytes))
        .and_then(|n| n.checked_add(additional))
        .ok_or_else(plan_allocation_error)
}

fn reserve_temporary(context: &mut PlanBuild<'_>, bytes: usize) -> Result<(), CompileError> {
    ensure_compile_budget(context, bytes)?;
    context.temporary_bytes = context
        .temporary_bytes
        .checked_add(bytes)
        .ok_or_else(plan_allocation_error)?;
    Ok(())
}

fn release_temporary(context: &mut PlanBuild<'_>, bytes: usize) -> Result<(), CompileError> {
    context.temporary_bytes = context.temporary_bytes.checked_sub(bytes).ok_or_else(|| {
        resource_limit(
            LimitKind::PlanBytes,
            usize::MAX,
            context.limits.max_plan_bytes,
            Stage::L3,
        )
    })?;
    Ok(())
}

fn charge_retained(context: &mut PlanBuild<'_>, bytes: usize) -> Result<(), CompileError> {
    ensure_compile_budget(context, bytes)?;
    charge(
        context.retained_bytes,
        bytes,
        context.limits.max_plan_bytes,
        LimitKind::PlanBytes,
        Stage::L3,
    )
}

/// Builds an incremental object plan shared by `OpenObject` and closed `Object` nodes: known
/// properties, `patternProperties`, and an additional-key policy, all validated order-independently.
#[allow(clippy::too_many_arguments)]
fn build_open_object_plan(
    ir: &SchemaIR,
    context: &mut PlanBuild<'_>,
    known_src: &[(crate::ir::StrRef, NodeId)],
    known_required: crate::ir::BitSlice,
    patterns_src: &[(crate::ir::StrRef, NodeId)],
    additional: &AdditionalPolicy,
    property_names: Option<NodeId>,
    min_properties: Option<u32>,
    max_properties: Option<u32>,
    dependent: crate::ir::PropSlice,
    dependent_required: crate::ir::DependentRequiredSlice,
) -> Result<NodePlan, CompileError> {
    let dependent_src = ir
        .props_at(dependent)
        .ok_or_else(|| invalid_ir("dependentSchemas property slice"))?;
    let dependent_required_src = ir
        .dependent_required_at(dependent_required)
        .ok_or_else(|| invalid_ir("dependentRequired pair slice"))?;
    let minimum = object_plan_minimum_bytes(ir, known_src, patterns_src.len(), dependent_src)?;
    let current = *context.retained_bytes;
    let projected = current
        .checked_add(minimum)
        .ok_or_else(plan_allocation_error)?;
    if projected > context.limits.max_plan_bytes {
        return Err(resource_limit(
            LimitKind::PlanBytes,
            projected,
            context.limits.max_plan_bytes,
            Stage::L3,
        ));
    }
    let mut known_by_name: FxHashMap<String, PropertyId> = FxHashMap::default();
    known_by_name
        .try_reserve(known_src.len())
        .map_err(|_| plan_allocation_error())?;
    let mut known_plans = Vec::new();
    known_plans
        .try_reserve_exact(known_src.len())
        .map_err(|_| plan_allocation_error())?;
    for (i, (name_ref, value)) in known_src.iter().enumerate() {
        let name = ir
            .str_at(*name_ref)
            .ok_or_else(|| invalid_ir("property name string reference"))?;
        let property_id = PropertyId(u32::try_from(i).map_err(|_| plan_allocation_error())?);
        let owned_name = fallible_string(name)?;
        known_by_name.insert(owned_name, property_id);
        known_plans.push(PropertyPlan {
            value: *value,
            maybe_satisfiable: !matches!(ir.node(*value), Some(Node::Never)),
        });
        queue_child(*value, context.worklist, context.queued);
    }
    let mut required_template = Vec::new();
    required_template
        .try_reserve_exact(words_for(known_src.len()))
        .map_err(|_| plan_allocation_error())?;
    required_template.resize(words_for(known_src.len()), 0);
    for (i, _) in known_src.iter().enumerate() {
        let property_index = u32::try_from(i).map_err(|_| plan_allocation_error())?;
        if ir.is_required(known_required, property_index) {
            required_template[i / BITSET_WORD_BITS] |= 1 << (i % BITSET_WORD_BITS);
        }
    }
    let mut pattern_plans = Vec::new();
    pattern_plans
        .try_reserve_exact(patterns_src.len())
        .map_err(|_| plan_allocation_error())?;
    for (regex_ref, value) in patterns_src {
        let regex_src = ir
            .str_at(*regex_ref)
            .ok_or_else(|| invalid_ir("pattern property regex string reference"))?;
        let temporary = regex_compile_peak(regex_src.len())?;
        reserve_temporary(context, temporary)?;
        let engine = super::pattern::build_property_search_engine(regex_src)?;
        release_temporary(context, temporary)?;
        charge_retained(context, engine_retained_bytes(&engine))?;
        pattern_plans.push(CompiledPropertyPattern {
            engine,
            value: *value,
            maybe_satisfiable: !matches!(ir.node(*value), Some(Node::Never)),
        });
        queue_child(*value, context.worklist, context.queued);
    }
    let viable_pattern = if matches!(additional, AdditionalPolicy::Forbid) {
        build_viable_property_pattern(&pattern_plans, context)?
    } else {
        None
    };
    let additional_plan = match additional {
        AdditionalPolicy::Forbid => AdditionalPlan::Forbid,
        AdditionalPolicy::Open => AdditionalPlan::Open,
        AdditionalPolicy::AllowAny => AdditionalPlan::AllowAny,
        AdditionalPolicy::Schema(target) => {
            queue_child(*target, context.worklist, context.queued);
            AdditionalPlan::Schema(*target)
        }
    };
    if let Some(names) = property_names {
        queue_child(names, context.worklist, context.queued);
    }
    let mut dependent_by_name = FxHashMap::default();
    dependent_by_name
        .try_reserve(dependent_src.len())
        .map_err(|_| plan_allocation_error())?;
    let dedup_temporary = dependent_src
        .len()
        .checked_mul(size_of::<(NodeId, u32)>() + 16)
        .ok_or_else(plan_allocation_error)?;
    reserve_temporary(context, dedup_temporary)?;
    let mut unique_by_node: FxHashMap<NodeId, u32> = FxHashMap::default();
    unique_by_node
        .try_reserve(dependent_src.len())
        .map_err(|_| plan_allocation_error())?;
    let mut dependent_schemas = Vec::new();
    dependent_schemas
        .try_reserve_exact(dependent_src.len())
        .map_err(|_| plan_allocation_error())?;
    for &(name_ref, schema) in dependent_src {
        let name = ir
            .str_at(name_ref)
            .ok_or_else(|| invalid_ir("dependentSchemas trigger string reference"))?;
        if ir.node(schema).is_none() {
            return Err(invalid_ir("dependentSchemas child node"));
        }
        let dependency = if let Some(&index) = unique_by_node.get(&schema) {
            index
        } else {
            let index =
                u32::try_from(dependent_schemas.len()).map_err(|_| plan_allocation_error())?;
            unique_by_node.insert(schema, index);
            dependent_schemas.push(DependentSchemaPlan { schema });
            queue_child(schema, context.worklist, context.queued);
            index
        };
        let owned_name = fallible_string(name)?;
        if dependent_by_name.insert(owned_name, dependency).is_some() {
            return Err(invalid_ir("duplicate dependentSchemas trigger"));
        }
    }
    drop(unique_by_node);
    release_temporary(context, dedup_temporary)?;
    let dependent_required = build_dependent_required_plan(ir, dependent_required_src, context)?;
    let object_bytes = known_plans
        .capacity()
        .checked_mul(std::mem::size_of::<PropertyPlan>())
        .and_then(|n| {
            n.checked_add(
                known_by_name
                    .capacity()
                    .checked_mul(std::mem::size_of::<(String, PropertyId)>() + 16)?,
            )
        })
        .and_then(|n| {
            n.checked_add(
                dependent_by_name
                    .capacity()
                    .checked_mul(std::mem::size_of::<(String, u32)>() + 16)?,
            )
        })
        .and_then(|n| {
            n.checked_add(
                dependent_by_name
                    .keys()
                    .map(|name| name.len())
                    .try_fold(0usize, usize::checked_add)?,
            )
        })
        .and_then(|n| {
            n.checked_add(
                dependent_schemas
                    .capacity()
                    .checked_mul(std::mem::size_of::<DependentSchemaPlan>())?,
            )
        })
        .and_then(|n| n.checked_add(required_template.capacity() * std::mem::size_of::<u64>()))
        .and_then(|n| {
            n.checked_add(pattern_plans.capacity() * std::mem::size_of::<CompiledPropertyPattern>())
        })
        .and_then(|n| {
            n.checked_add(
                known_by_name
                    .keys()
                    .map(|name| name.len())
                    .try_fold(0usize, usize::checked_add)?,
            )
        })
        .ok_or_else(plan_allocation_error)?;
    charge_retained(context, object_bytes)?;
    Ok(NodePlan::Object(ObjectPlan {
        known_by_name,
        known: known_plans,
        patterns: pattern_plans,
        viable_pattern,
        required_template,
        additional: additional_plan,
        property_names,
        min_properties,
        max_properties,
        dependent_by_name,
        dependent_schemas: dependent_schemas.into_boxed_slice(),
        dependent_required,
    }))
}

fn build_viable_property_pattern(
    patterns: &[CompiledPropertyPattern],
    context: &mut PlanBuild<'_>,
) -> Result<Option<Box<RefEngine>>, CompileError> {
    let reference_bytes = patterns
        .len()
        .checked_mul(2)
        .and_then(|count| count.checked_mul(size_of::<&RefEngine>()))
        .ok_or_else(plan_allocation_error)?;
    reserve_temporary(context, reference_bytes)?;
    let mut live = Vec::new();
    let mut false_patterns = Vec::new();
    live.try_reserve_exact(patterns.len())
        .map_err(|_| plan_allocation_error())?;
    false_patterns
        .try_reserve_exact(patterns.len())
        .map_err(|_| plan_allocation_error())?;
    for pattern in patterns {
        if pattern.maybe_satisfiable {
            live.push(&pattern.engine);
        } else {
            false_patterns.push(&pattern.engine);
        }
    }
    if live.is_empty() {
        release_temporary(context, reference_bytes)?;
        return Ok(None);
    }
    let allowed_union = ProductAutomaton::build_reachable(&live, Combinator::Any)?.into_engine()?;
    let result = if false_patterns.is_empty() {
        allowed_union
    } else {
        let forbidden_union =
            ProductAutomaton::build_reachable(&false_patterns, Combinator::Any)?.into_engine()?;
        let transient = engine_retained_bytes(&allowed_union)
            .checked_add(engine_retained_bytes(&forbidden_union))
            .ok_or_else(plan_allocation_error)?;
        reserve_temporary(context, transient)?;
        let result = ProductAutomaton::build_reachable(
            &[&allowed_union, &forbidden_union],
            Combinator::Difference,
        )?
        .into_engine()?;
        release_temporary(context, transient)?;
        result
    };
    release_temporary(context, reference_bytes)?;
    charge_retained(context, engine_retained_bytes(&result))?;
    Ok(Some(Box::new(result)))
}

fn build_dependent_required_plan(
    ir: &SchemaIR,
    pairs: &[crate::ir::DependentRequiredPair],
    context: &mut PlanBuild<'_>,
) -> Result<Option<DependentRequiredPlan>, CompileError> {
    if pairs.is_empty() {
        return Ok(None);
    }
    let temporary = pairs
        .len()
        .checked_mul(size_of::<(u32, u32)>() + size_of::<u32>() + 32)
        .ok_or_else(plan_allocation_error)?;
    reserve_temporary(context, temporary)?;
    let mut name_by_text = FxHashMap::default();
    name_by_text
        .try_reserve(pairs.len().saturating_mul(2))
        .map_err(|_| plan_allocation_error())?;
    // One flat, fallibly-reserved edge vector - no per-trigger heap Vec, so a schema with many
    // triggers each holding one requirement costs one allocation, not one per trigger.
    let mut edges: Vec<(u32, u32)> = Vec::new();
    edges
        .try_reserve_exact(pairs.len())
        .map_err(|_| plan_allocation_error())?;
    for pair in pairs {
        let trigger = ir
            .str_at(pair.trigger)
            .ok_or_else(|| invalid_ir("dependentRequired trigger string reference"))?;
        let required = ir
            .str_at(pair.required)
            .ok_or_else(|| invalid_ir("dependentRequired name string reference"))?;
        let trigger_id = intern_presence_name(&mut name_by_text, trigger)?;
        let required_id = intern_presence_name(&mut name_by_text, required)?;
        edges.push((trigger_id, required_id));
    }
    edges.sort_unstable();
    edges.dedup();
    let mut rules = Vec::new();
    // At most one rule per distinct trigger - `edges.len()` is a safe upper bound.
    rules
        .try_reserve_exact(edges.len())
        .map_err(|_| plan_allocation_error())?;
    let mut required_ids = Vec::new();
    required_ids
        .try_reserve_exact(edges.len())
        .map_err(|_| plan_allocation_error())?;
    let mut index = 0usize;
    while index < edges.len() {
        let trigger = edges[index].0;
        let run_end = edges[index..].partition_point(|&(t, _)| t == trigger) + index;
        let required_start =
            u32::try_from(required_ids.len()).map_err(|_| plan_allocation_error())?;
        let required_len = u32::try_from(run_end - index).map_err(|_| plan_allocation_error())?;
        required_ids.extend(edges[index..run_end].iter().map(|&(_, required)| required));
        rules.push(DependentRequiredRule {
            trigger,
            required_start,
            required_len,
        });
        index = run_end;
    }
    let name_count = u32::try_from(name_by_text.len()).map_err(|_| plan_allocation_error())?;
    let retained = name_by_text
        .capacity()
        .checked_mul(size_of::<(String, u32)>() + 16)
        .and_then(|n| {
            n.checked_add(
                name_by_text
                    .keys()
                    .map(String::len)
                    .try_fold(0usize, usize::checked_add)?,
            )
        })
        .and_then(|n| n.checked_add(rules.capacity() * size_of::<DependentRequiredRule>()))
        .and_then(|n| n.checked_add(required_ids.capacity() * size_of::<u32>()))
        .ok_or_else(plan_allocation_error)?;
    release_temporary(context, temporary)?;
    charge_retained(context, retained)?;
    Ok(Some(DependentRequiredPlan {
        name_by_text,
        rules: rules.into_boxed_slice(),
        required_ids: required_ids.into_boxed_slice(),
        name_count,
    }))
}

fn intern_presence_name(
    names: &mut FxHashMap<String, u32>,
    text: &str,
) -> Result<u32, CompileError> {
    if let Some(id) = names.get(text) {
        return Ok(*id);
    }
    let id = u32::try_from(names.len()).map_err(|_| plan_allocation_error())?;
    names.insert(fallible_string(text)?, id);
    Ok(id)
}

fn string_enum_retained_bytes(
    member_count: usize,
    total_bytes: usize,
) -> Result<(usize, usize), CompileError> {
    let offset_count = member_count
        .checked_add(1)
        .ok_or_else(plan_allocation_error)?;
    u32::try_from(offset_count).map_err(|_| plan_allocation_error())?;
    u32::try_from(total_bytes).map_err(|_| plan_allocation_error())?;
    let offset_bytes = offset_count
        .checked_mul(size_of::<u32>())
        .ok_or_else(plan_allocation_error)?;
    let retained = total_bytes
        .checked_add(offset_bytes)
        .ok_or_else(plan_allocation_error)?;
    Ok((offset_count, retained))
}

fn string_enum_layout(literals: &[ScalarLit]) -> Result<(usize, usize), CompileError> {
    if literals.is_empty() {
        return Err(invalid_ir("string enum is empty"));
    }
    let mut total_bytes = 0usize;
    let mut previous: Option<&str> = None;
    for literal in literals {
        let ScalarLit::Str(value) = literal else {
            return Err(invalid_ir("string enum contains a non-string literal"));
        };
        if previous.is_some_and(|prior| prior >= value.as_str()) {
            return Err(invalid_ir("string enum is not strictly sorted"));
        }
        total_bytes = total_bytes
            .checked_add(value.len())
            .ok_or_else(plan_allocation_error)?;
        previous = Some(value);
    }
    let (_, retained) = string_enum_retained_bytes(literals.len(), total_bytes)?;
    Ok((total_bytes, retained))
}

fn build_string_enum_plan(
    ir: &SchemaIR,
    values: LitSlice,
    context: &mut PlanBuild<'_>,
) -> Result<NodePlan, CompileError> {
    let literals = ir
        .lits_at(values)
        .ok_or_else(|| invalid_ir("string enum literal run"))?;
    let (total_bytes, retained) = string_enum_layout(literals)?;
    ensure_compile_budget(context, retained)?;
    let (offset_count, _) = string_enum_retained_bytes(literals.len(), total_bytes)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(total_bytes)
        .map_err(|_| plan_allocation_error())?;
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(offset_count)
        .map_err(|_| plan_allocation_error())?;
    offsets.push(0u32);
    for literal in literals {
        let ScalarLit::Str(value) = literal else {
            return Err(invalid_ir("string enum contains a non-string literal"));
        };
        bytes.extend_from_slice(value.as_bytes());
        offsets.push(u32::try_from(bytes.len()).map_err(|_| plan_allocation_error())?);
    }
    let plan = StringEnumPlan {
        bytes: bytes.into_boxed_slice(),
        offsets: offsets.into_boxed_slice(),
    };
    if plan.retained_bytes() != retained {
        return Err(plan_allocation_error());
    }
    charge_retained(context, retained)?;
    Ok(NodePlan::StringEnum(plan))
}

fn build_string_const_plan(
    ir: &SchemaIR,
    value: crate::ir::StrRef,
    context: &mut PlanBuild<'_>,
) -> Result<NodePlan, CompileError> {
    let value = ir
        .str_at(value)
        .ok_or_else(|| invalid_ir("string const literal"))?;
    let (_, retained) = string_enum_retained_bytes(1, value.len())?;
    ensure_compile_budget(context, retained)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(value.len())
        .map_err(|_| plan_allocation_error())?;
    bytes.extend_from_slice(value.as_bytes());
    let end = u32::try_from(bytes.len()).map_err(|_| plan_allocation_error())?;
    let plan = StringEnumPlan {
        bytes: bytes.into_boxed_slice(),
        offsets: Box::new([0, end]),
    };
    charge_retained(context, retained)?;
    Ok(NodePlan::StringEnum(plan))
}

fn finite_string_member_count(ir: &SchemaIR, start: NodeId) -> Option<u32> {
    let mut node = start;
    for _ in 0..=ir.node_count() {
        match ir.node(node)? {
            Node::StringConst { .. } => return Some(1),
            Node::Enum { values } => {
                let members = ir.lits_at(*values)?;
                if members
                    .iter()
                    .all(|member| matches!(member, ScalarLit::Str(_)))
                {
                    return u32::try_from(members.len()).ok();
                }
                return None;
            }
            Node::Ref { def } => node = ir.def_target(*def)?,
            _ => return None,
        }
    }
    None
}

struct MutableFiniteKeyNode {
    edges: Vec<(u8, u32)>,
    terminal: bool,
}

fn queue_finite_key_node(
    id: NodeId,
    visited: &mut [bool],
    work: &mut Vec<NodeId>,
) -> Result<(), CompileError> {
    let index = usize::try_from(id.get()).map_err(|_| plan_allocation_error())?;
    let seen = visited
        .get_mut(index)
        .ok_or_else(|| invalid_ir("finite-key scope node"))?;
    if !*seen {
        *seen = true;
        // `visited` guarantees at most `ir.node_count()` pushes, matching the preflighted capacity.
        work.push(id);
    }
    Ok(())
}

fn collect_maybe_satisfiable_property_names(
    ir: &SchemaIR,
    properties: &[(crate::ir::StrRef, NodeId)],
    names: &mut Vec<crate::ir::StrRef>,
) -> Result<(), CompileError> {
    for &(name, value) in properties {
        let value_node = ir
            .node(value)
            .ok_or_else(|| invalid_ir("finite-key property value node"))?;
        // This is deliberately a one-sided certificate. `Never` is independently proven empty;
        // every other value language remains possible/Unknown and therefore stays in the trie.
        if !matches!(value_node, Node::Never) {
            names.push(name);
        }
    }
    Ok(())
}

fn build_finite_evaluated_key_trie(
    ir: &SchemaIR,
    scope: NodeId,
    context: &mut PlanBuild<'_>,
) -> Result<Option<FiniteKeyTrie>, CompileError> {
    let mark = context.temporary_bytes;
    let result = (|| {
        let node_count = ir.node_count();
        let mut property_count = 0usize;
        let mut pattern_count = 0usize;
        for node in ir.nodes() {
            let names = match node {
                Node::Object { fields, .. } => ir
                    .props_at(*fields)
                    .ok_or_else(|| invalid_ir("finite-key object property slice"))?,
                Node::OpenObject { known, .. } => ir
                    .props_at(*known)
                    .ok_or_else(|| invalid_ir("finite-key open-object property slice"))?,
                _ => &[],
            };
            property_count = property_count
                .checked_add(names.len())
                .ok_or_else(plan_allocation_error)?;
            if let Node::OpenObject { patterns, .. } = node {
                pattern_count = pattern_count
                    .checked_add(
                        ir.props_at(*patterns)
                            .ok_or_else(|| invalid_ir("finite-key pattern property slice"))?
                            .len(),
                    )
                    .ok_or_else(plan_allocation_error)?;
            }
        }
        let collection_bytes = node_count
            .checked_mul(size_of::<u8>() + size_of::<NodeId>())
            .and_then(|bytes| {
                bytes.checked_add(property_count.checked_mul(size_of::<crate::ir::StrRef>())?)
            })
            .and_then(|bytes| {
                bytes.checked_add(pattern_count.checked_mul(size_of::<crate::ir::StrRef>())?)
            })
            .ok_or_else(plan_allocation_error)?;
        if compile_budget_projection(context, collection_bytes)? > context.limits.max_plan_bytes {
            return Ok(None);
        }
        reserve_temporary(context, collection_bytes)?;

        let mut visited = Vec::new();
        visited
            .try_reserve_exact(node_count)
            .map_err(|_| plan_allocation_error())?;
        visited.resize(node_count, false);
        let mut work = Vec::new();
        work.try_reserve_exact(node_count)
            .map_err(|_| plan_allocation_error())?;
        queue_finite_key_node(scope, &mut visited, &mut work)?;
        let mut names = Vec::new();
        names
            .try_reserve_exact(property_count)
            .map_err(|_| plan_allocation_error())?;
        let mut pattern_regexes = Vec::new();
        pattern_regexes
            .try_reserve_exact(pattern_count)
            .map_err(|_| plan_allocation_error())?;
        let mut finite = true;
        while let Some(id) = work.pop() {
            let node = ir
                .node(id)
                .ok_or_else(|| invalid_ir("finite-key scope node"))?;
            match node {
                Node::Object {
                    fields, dependent, ..
                } => {
                    collect_maybe_satisfiable_property_names(
                        ir,
                        ir.props_at(*fields)
                            .ok_or_else(|| invalid_ir("finite-key object property slice"))?,
                        &mut names,
                    )?;
                    for &(_, schema) in ir
                        .props_at(*dependent)
                        .ok_or_else(|| invalid_ir("finite-key dependent schema slice"))?
                    {
                        queue_finite_key_node(schema, &mut visited, &mut work)?;
                    }
                }
                Node::OpenObject {
                    known,
                    patterns,
                    additional,
                    dependent,
                    ..
                } => {
                    let patterns = ir
                        .props_at(*patterns)
                        .ok_or_else(|| invalid_ir("finite-key pattern property slice"))?;
                    for &(regex, value) in patterns {
                        // A false-valued pattern can intersect exact properties and invalidate
                        // them. The one-DFA union proof does not model that product yet.
                        if matches!(ir.node(value), Some(Node::Never)) {
                            finite = false;
                            break;
                        }
                        pattern_regexes.push(regex);
                    }
                    if !finite
                        || matches!(
                            additional,
                            AdditionalPolicy::AllowAny | AdditionalPolicy::Schema(_)
                        )
                    {
                        finite = false;
                        break;
                    }
                    collect_maybe_satisfiable_property_names(
                        ir,
                        ir.props_at(*known)
                            .ok_or_else(|| invalid_ir("finite-key known property slice"))?,
                        &mut names,
                    )?;
                    for &(_, schema) in ir
                        .props_at(*dependent)
                        .ok_or_else(|| invalid_ir("finite-key dependent schema slice"))?
                    {
                        queue_finite_key_node(schema, &mut visited, &mut work)?;
                    }
                }
                Node::Intersection { branches }
                | Node::Union { branches }
                | Node::ExactlyOne { branches } => {
                    for &child in ir
                        .refs_at(*branches)
                        .ok_or_else(|| invalid_ir("finite-key combinator branch slice"))?
                    {
                        queue_finite_key_node(child, &mut visited, &mut work)?;
                    }
                }
                // Negated subschemas do not contribute successful evaluated-property annotations.
                Node::Not { .. } => {}
                // Static and dynamic references conservatively disable this first finite proof;
                // in particular, no recursive graph is narrowed from incomplete evidence.
                Node::Ref { .. } | Node::DynamicRef { .. } | Node::Unsupported { .. } => {
                    finite = false;
                    break;
                }
                Node::Unevaluated {
                    kind,
                    scope,
                    unevaluated,
                } => {
                    if *kind == UnevaluatedKind::Properties
                        && !matches!(ir.node(*unevaluated), Some(Node::Never))
                    {
                        finite = false;
                        break;
                    }
                    queue_finite_key_node(*scope, &mut visited, &mut work)?;
                }
                Node::Null
                | Node::Boolean
                | Node::Never
                | Node::StringConst { .. }
                | Node::StringPattern { .. }
                | Node::Integer { .. }
                | Node::Number { .. }
                | Node::LexicalNumber { .. }
                | Node::Enum { .. }
                | Node::Array { .. }
                | Node::Tuple { .. } => {}
            }
        }
        if !finite {
            return Ok(None);
        }

        let mut compiled_patterns = Vec::new();
        compiled_patterns
            .try_reserve_exact(pattern_regexes.len())
            .map_err(|_| plan_allocation_error())?;
        for &regex in &pattern_regexes {
            let source = ir
                .str_at(regex)
                .ok_or_else(|| invalid_ir("finite-key pattern regex"))?;
            let temporary = regex_compile_peak(source.len())?;
            if compile_budget_projection(context, temporary)? > context.limits.max_plan_bytes {
                return Ok(None);
            }
            reserve_temporary(context, temporary)?;
            let engine = super::pattern::build_property_search_engine(source)?;
            release_temporary(context, temporary)?;
            let retained = engine_retained_bytes(&engine);
            if compile_budget_projection(context, retained)? > context.limits.max_plan_bytes {
                return Ok(None);
            }
            // The engine remains live while the trie is constructed, so include its retained
            // storage in the temporary peak. The final charge occurs once through `retained_bytes`.
            reserve_temporary(context, retained)?;
            compiled_patterns.push(engine);
        }

        let total_name_bytes = names.iter().try_fold(0usize, |total, name| {
            let text = ir
                .str_at(*name)
                .ok_or_else(|| invalid_ir("finite-key property name"))?;
            total
                .checked_add(text.len())
                .ok_or_else(plan_allocation_error)
        })?;
        let max_nodes = total_name_bytes
            .checked_add(1)
            .ok_or_else(plan_allocation_error)?;
        let words_per_node = words_for(names.len());
        let subtree_word_count = max_nodes
            .checked_mul(words_per_node)
            .ok_or_else(plan_allocation_error)?;
        let offset_count = names
            .len()
            .checked_add(1)
            .ok_or_else(plan_allocation_error)?;
        let build_bytes = max_nodes
            .checked_mul(size_of::<MutableFiniteKeyNode>() + size_of::<FiniteKeyNode>())
            .and_then(|bytes| {
                bytes.checked_add(
                    total_name_bytes
                        .checked_mul(size_of::<(u8, u32)>() + size_of::<FiniteKeyEdge>())?,
                )
            })
            .and_then(|bytes| bytes.checked_add(total_name_bytes))
            .and_then(|bytes| bytes.checked_add(offset_count.checked_mul(size_of::<u32>())?))
            .and_then(|bytes| bytes.checked_add(subtree_word_count.checked_mul(size_of::<u64>())?))
            .ok_or_else(plan_allocation_error)?;
        if compile_budget_projection(context, build_bytes)? > context.limits.max_plan_bytes {
            return Ok(None);
        }
        reserve_temporary(context, build_bytes)?;
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(max_nodes)
            .map_err(|_| plan_allocation_error())?;
        nodes.push(MutableFiniteKeyNode {
            edges: Vec::new(),
            terminal: false,
        });
        for &name in &names {
            let text = ir
                .str_at(name)
                .ok_or_else(|| invalid_ir("finite-key property name"))?;
            let mut current = 0usize;
            for &byte in text.as_bytes() {
                let next = nodes[current]
                    .edges
                    .iter()
                    .find_map(|&(edge, target)| (edge == byte).then_some(target));
                current = if let Some(target) = next {
                    usize::try_from(target).map_err(|_| plan_allocation_error())?
                } else {
                    let target = u32::try_from(nodes.len()).map_err(|_| plan_allocation_error())?;
                    nodes[current]
                        .edges
                        .try_reserve_exact(1)
                        .map_err(|_| plan_allocation_error())?;
                    nodes[current].edges.push((byte, target));
                    nodes.push(MutableFiniteKeyNode {
                        edges: Vec::new(),
                        terminal: false,
                    });
                    target as usize
                };
            }
            nodes[current].terminal = true;
        }
        for node in &mut nodes {
            node.edges.sort_unstable_by_key(|&(byte, _)| byte);
        }
        let edge_count = nodes.iter().try_fold(0usize, |total, node| {
            total
                .checked_add(node.edges.len())
                .ok_or_else(plan_allocation_error)
        })?;
        let mut flat_nodes = Vec::new();
        flat_nodes
            .try_reserve_exact(nodes.len())
            .map_err(|_| plan_allocation_error())?;
        let mut flat_edges = Vec::new();
        flat_edges
            .try_reserve_exact(edge_count)
            .map_err(|_| plan_allocation_error())?;
        for node in nodes {
            let first_edge =
                u32::try_from(flat_edges.len()).map_err(|_| plan_allocation_error())?;
            let edge_len = u32::try_from(node.edges.len()).map_err(|_| plan_allocation_error())?;
            flat_edges.extend(
                node.edges
                    .into_iter()
                    .map(|(byte, target)| FiniteKeyEdge { byte, target }),
            );
            flat_nodes.push(FiniteKeyNode {
                first_edge,
                edge_len,
                terminal: node.terminal,
            });
        }
        let mut name_bytes = Vec::new();
        name_bytes
            .try_reserve_exact(total_name_bytes)
            .map_err(|_| plan_allocation_error())?;
        let mut name_offsets = Vec::new();
        name_offsets
            .try_reserve_exact(offset_count)
            .map_err(|_| plan_allocation_error())?;
        name_offsets.push(0);
        for &name in &names {
            let text = ir
                .str_at(name)
                .ok_or_else(|| invalid_ir("finite-key property name"))?;
            name_bytes.extend_from_slice(text.as_bytes());
            name_offsets
                .push(u32::try_from(name_bytes.len()).map_err(|_| plan_allocation_error())?);
        }
        let actual_word_count = flat_nodes
            .len()
            .checked_mul(words_per_node)
            .ok_or_else(plan_allocation_error)?;
        let mut subtree_ids = Vec::new();
        subtree_ids
            .try_reserve_exact(actual_word_count)
            .map_err(|_| plan_allocation_error())?;
        subtree_ids.resize(actual_word_count, 0);
        for (id, &name) in names.iter().enumerate() {
            let word = id / BITSET_WORD_BITS;
            let bit = id % BITSET_WORD_BITS;
            let mask = 1u64 << bit;
            let mut current = 0usize;
            let root_bit = current
                .checked_mul(words_per_node)
                .and_then(|base| base.checked_add(word))
                .ok_or_else(plan_allocation_error)?;
            *subtree_ids
                .get_mut(root_bit)
                .ok_or_else(plan_allocation_error)? |= mask;
            let text = ir
                .str_at(name)
                .ok_or_else(|| invalid_ir("finite-key property name"))?;
            for &byte in text.as_bytes() {
                let node = flat_nodes.get(current).ok_or_else(plan_allocation_error)?;
                let start = node.first_edge as usize;
                let end = start
                    .checked_add(node.edge_len as usize)
                    .ok_or_else(plan_allocation_error)?;
                let edges = flat_edges
                    .get(start..end)
                    .ok_or_else(plan_allocation_error)?;
                let offset = edges
                    .binary_search_by_key(&byte, |edge| edge.byte)
                    .map_err(|_| plan_allocation_error())?;
                current = edges[offset].target as usize;
                let index = current
                    .checked_mul(words_per_node)
                    .and_then(|base| base.checked_add(word))
                    .ok_or_else(plan_allocation_error)?;
                *subtree_ids
                    .get_mut(index)
                    .ok_or_else(plan_allocation_error)? |= mask;
            }
        }
        Ok(Some(FiniteKeyTrie {
            nodes: flat_nodes.into_boxed_slice(),
            edges: flat_edges.into_boxed_slice(),
            name_bytes: name_bytes.into_boxed_slice(),
            name_offsets: name_offsets.into_boxed_slice(),
            words_per_node: u32::try_from(words_per_node).map_err(|_| plan_allocation_error())?,
            subtree_ids: subtree_ids.into_boxed_slice(),
            patterns: compiled_patterns,
        }))
    })();
    context.temporary_bytes = mark;
    let trie = result?;
    if let Some(trie) = &trie {
        charge_retained(context, trie.retained_bytes())?;
    }
    Ok(trie)
}

fn build_node_plan(
    ir: &SchemaIR,
    id: NodeId,
    context: &mut PlanBuild<'_>,
) -> Result<NodePlan, PlanCompileFailure> {
    let Some(node) = ir.node(id) else {
        return Err(invalid_ir("node id out of range").into());
    };
    // Array equality keywords compare decoded strings, not their JSON source spelling.
    if needs_decoded_string_semantics(ir, id) {
        if let Node::Enum { values } = node {
            if ir.lits_at(*values).is_some_and(|literals| {
                !literals.is_empty()
                    && literals
                        .iter()
                        .all(|literal| matches!(literal, ScalarLit::Str(_)))
            }) {
                return Ok(build_string_enum_plan(ir, *values, context)?);
            }
        }
        if let Node::StringConst { value } = node {
            return Ok(build_string_const_plan(ir, *value, context)?);
        }
    }
    if let Node::Number {
        integer_only,
        minimum,
        maximum,
        multiple_of,
    } = node
    {
        let temporary = regex_compile_peak(64)?;
        reserve_temporary(context, temporary)?;
        let engine = number_syntax_engine()?;
        release_temporary(context, temporary)?;
        charge_retained(context, engine_retained_bytes(&engine))?;
        return Ok(NodePlan::Number {
            engine: Box::new(engine),
            constraint: NumberPlan {
                integer_only: *integer_only,
                minimum: *minimum,
                maximum: *maximum,
                multiple_of: *multiple_of,
            },
        });
    }
    // Explicit counters let bounded strings reuse vocabulary masks as their remaining length changes.
    let bounded_string = matches!(
        node,
        Node::StringPattern {
            max_len: Some(_),
            ..
        }
    );
    if regular_for(id, context.regular) && !bounded_string {
        let temporary = regex_compile_peak(64)?;
        reserve_temporary(context, temporary)?;
        let compiled = compile_node(ir, id);
        release_temporary(context, temporary)?;
        match compiled {
            Ok(engine) => {
                charge_retained(context, engine_retained_bytes(&engine))?;
                return Ok(NodePlan::Regular(Box::new(engine)));
            }
            Err(error) if automaton_limit_has_structured_route(node, &error) => {}
            Err(error) => return Err(error.into()),
        }
    }
    match node {
        Node::Enum { values } if ir.is_large_string_enum(node) => {
            Ok(build_string_enum_plan(ir, *values, context)?)
        }
        Node::OpenObject {
            known,
            known_required,
            patterns,
            additional,
            property_names,
            min_properties,
            max_properties,
            dependent,
            dependent_required,
        } => {
            let known_src = ir.props_at(*known).unwrap_or(&[]);
            let patterns_src = ir.props_at(*patterns).unwrap_or(&[]);
            Ok(build_open_object_plan(
                ir,
                context,
                known_src,
                *known_required,
                patterns_src,
                additional,
                *property_names,
                *min_properties,
                *max_properties,
                *dependent,
                *dependent_required,
            )?)
        }
        Node::Object {
            fields,
            required,
            closure,
            dependent,
            dependent_required,
        } => {
            // A closed object too large for the byte-DFA shuffle validates incrementally here: its
            // fields become known properties and its closure maps to an additional-key policy.
            let additional = match closure {
                crate::ir::ObjectClosure::Forbidden
                | crate::ir::ObjectClosure::AssumeClosedProfile => AdditionalPolicy::Forbid,
                crate::ir::ObjectClosure::AllowOpenProfile => AdditionalPolicy::AllowAny,
                crate::ir::ObjectClosure::RejectOpenObjects => {
                    return Err(unsupported_plan("open object").into())
                }
            };
            let fields_src = ir.props_at(*fields).unwrap_or(&[]);
            Ok(build_open_object_plan(
                ir,
                context,
                fields_src,
                *required,
                &[],
                &additional,
                None,
                None,
                None,
                *dependent,
                *dependent_required,
            )?)
        }
        Node::Array {
            items,
            min_items,
            max_items,
            unique_items,
            contains,
            items_annotates,
        } => {
            let finite_unique_capacity = if *unique_items {
                match items {
                    crate::ir::ItemsPolicy::Schema(id) => finite_string_member_count(ir, *id),
                    crate::ir::ItemsPolicy::AllowAny => None,
                }
            } else {
                None
            };
            let tail = match items {
                crate::ir::ItemsPolicy::Schema(id) => {
                    queue_child(*id, context.worklist, context.queued);
                    TailPlan::Schema(*id)
                }
                crate::ir::ItemsPolicy::AllowAny => TailPlan::AllowAny,
            };
            let contains_plan = build_contains_plan(contains, context.worklist, context.queued)?;
            Ok(NodePlan::Array(ArrayPlan {
                prefix: Vec::new(),
                tail,
                min_items: *min_items,
                max_items: *max_items,
                unique_items: *unique_items,
                finite_unique_capacity,
                contains: contains_plan,
                tail_annotates: *items_annotates,
            }))
        }
        Node::Tuple {
            prefix,
            tail,
            min_items,
            max_items,
            unique_items,
            tail_annotates,
            contains,
        } => {
            let prefix_src = ir.refs_at(*prefix).unwrap_or(&[]);
            let mut prefix_vec = Vec::new();
            prefix_vec
                .try_reserve_exact(prefix_src.len())
                .map_err(|_| plan_allocation_error())?;
            prefix_vec.extend_from_slice(prefix_src);
            for &id in &prefix_vec {
                queue_child(id, context.worklist, context.queued);
            }
            let tail = match tail {
                Some(t) => {
                    queue_child(*t, context.worklist, context.queued);
                    TailPlan::Schema(*t)
                }
                None => TailPlan::Closed,
            };
            let contains_plan = build_contains_plan(contains, context.worklist, context.queued)?;
            charge_retained(context, prefix_vec.len() * std::mem::size_of::<NodeId>())?;
            Ok(NodePlan::Array(ArrayPlan {
                prefix: prefix_vec,
                tail,
                min_items: *min_items,
                max_items: *max_items,
                unique_items: *unique_items,
                finite_unique_capacity: None,
                contains: contains_plan,
                tail_annotates: *tail_annotates,
            }))
        }
        Node::StringPattern {
            regex,
            min_len,
            max_len,
            ..
        } => {
            let pattern = if ir.is_user_pattern(node) {
                let source = ir
                    .str_at(*regex)
                    .ok_or_else(|| invalid_ir("string pattern string reference"))?;
                let temporary = regex_compile_peak(source.len())?;
                reserve_temporary(context, temporary)?;
                let engine = build_from_unicode_regex(source)?;
                release_temporary(context, temporary)?;
                charge_retained(context, engine_retained_bytes(&engine))?;
                Some(Box::new(engine))
            } else {
                None
            };
            Ok(NodePlan::String(StringPlan {
                min_scalars: min_len.unwrap_or(0),
                max_scalars: *max_len,
                pattern,
            }))
        }
        Node::Intersection { branches }
            if ir
                .refs_at(*branches)
                .is_some_and(|ids| fast_string_intersection(ir, ids)) =>
        {
            let branch_ids = ir.refs_at(*branches).unwrap_or(&[]);
            let mark = context.temporary_bytes;
            let result = build_intersection_pattern(ir, branch_ids, context);
            context.temporary_bytes = mark;
            let (min_scalars, max_scalars, pattern) = match result {
                Ok(pattern) => pattern,
                Err(error)
                    if error.code == ErrorCode::InternalLimitExceeded
                        && error.message == "product cap" =>
                {
                    return Ok(build_combinator_plan(
                        ir,
                        CombinatorKind::All,
                        Some(branch_ids),
                        context,
                    )?);
                }
                Err(error) => return Err(error),
            };
            if let Some(engine) = &pattern {
                charge_retained(context, engine_retained_bytes(engine))?;
            }
            Ok(NodePlan::String(StringPlan {
                min_scalars,
                max_scalars,
                pattern,
            }))
        }
        Node::Intersection { branches } => Ok(build_combinator_plan(
            ir,
            CombinatorKind::All,
            ir.refs_at(*branches),
            context,
        )?),
        Node::Union { branches } => Ok(build_combinator_plan(
            ir,
            CombinatorKind::Any,
            ir.refs_at(*branches),
            context,
        )?),
        Node::ExactlyOne { branches } => Ok(build_combinator_plan(
            ir,
            CombinatorKind::ExactlyOne,
            ir.refs_at(*branches),
            context,
        )?),
        Node::Not { inner } => {
            if ir.node(*inner).is_none() {
                return Err(invalid_ir("not target node").into());
            }
            queue_child(*inner, context.worklist, context.queued);
            Ok(NodePlan::Negation { inner: *inner })
        }
        Node::Ref { def } => {
            let target = ir
                .def_target(*def)
                .ok_or_else(|| invalid_ir("reference definition slot is invalid"))?;
            queue_child(target, context.worklist, context.queued);
            Ok(NodePlan::Ref { target })
        }
        Node::DynamicRef {
            initial_target,
            anchor,
        } => {
            if ir.node(*initial_target).is_none() {
                return Err(invalid_ir("dynamic reference initial target is invalid").into());
            }
            queue_child(*initial_target, context.worklist, context.queued);
            Ok(NodePlan::DynamicRef {
                initial_target: *initial_target,
                anchor: *anchor,
            })
        }
        Node::Unevaluated {
            kind,
            scope,
            unevaluated,
        } => {
            if ir.node(*scope).is_none() || ir.node(*unevaluated).is_none() {
                return Err(invalid_ir("unevaluated child node").into());
            }
            queue_child(*scope, context.worklist, context.queued);
            queue_child(*unevaluated, context.worklist, context.queued);
            let rejects_unevaluated = matches!(ir.node(*unevaluated), Some(Node::Never));
            let finite_evaluated_keys =
                if *kind == UnevaluatedKind::Properties && rejects_unevaluated {
                    build_finite_evaluated_key_trie(ir, *scope, context)?
                } else {
                    None
                };
            Ok(NodePlan::Unevaluated(UnevaluatedPlan {
                kind: *kind,
                scope: *scope,
                unevaluated: *unevaluated,
                rejects_unevaluated,
                finite_evaluated_keys,
            }))
        }
        _ => Err(unsupported_plan("this node shape").into()),
    }
}

fn needs_decoded_string_semantics(ir: &SchemaIR, candidate: NodeId) -> bool {
    ir.nodes().any(|node| match node {
        Node::Array {
            items: crate::ir::ItemsPolicy::Schema(item),
            unique_items,
            contains,
            ..
        } => {
            *item == candidate
                && (*unique_items
                    || contains.as_ref().is_some_and(|constraint| {
                        constraint.max.is_some()
                            && finite_string_member_count(ir, candidate).is_some()
                    }))
                || contains.as_ref().is_some_and(|constraint| {
                    matches!(constraint.policy, crate::ir::ContainsPolicy::Schema(id) if id == candidate)
                })
        }
        Node::Tuple {
            prefix,
            tail,
            unique_items,
            contains,
            ..
        } => {
            (*unique_items
                && (tail == &Some(candidate)
                    || ir
                        .refs_at(*prefix)
                        .is_some_and(|items| items.contains(&candidate))))
                || (contains.as_ref().is_some_and(|constraint| constraint.max.is_some())
                    && finite_string_member_count(ir, candidate).is_some()
                    && (tail == &Some(candidate)
                        || ir
                            .refs_at(*prefix)
                            .is_some_and(|items| items.contains(&candidate))))
                || contains.as_ref().is_some_and(|constraint| {
                    matches!(constraint.policy, crate::ir::ContainsPolicy::Schema(id) if id == candidate)
                })
        }
        _ => false,
    })
}

fn automaton_limit_has_structured_route(node: &Node, error: &CompileError) -> bool {
    error.code == ErrorCode::InternalLimitExceeded
        && error.stage == Stage::L3
        && matches!(
            node,
            Node::Object { .. }
                | Node::Array { .. }
                | Node::Tuple { .. }
                | Node::Union { .. }
                | Node::Intersection { .. }
                | Node::ExactlyOne { .. }
        )
}

fn build_combinator_plan(
    ir: &SchemaIR,
    kind: CombinatorKind,
    branches: Option<&[NodeId]>,
    context: &mut PlanBuild<'_>,
) -> Result<NodePlan, CompileError> {
    let source = branches.ok_or_else(|| invalid_ir("combinator branch slice"))?;
    let bytes = source
        .len()
        .checked_mul(size_of::<NodeId>())
        .ok_or_else(plan_allocation_error)?;
    ensure_compile_budget(context, bytes)?;
    let mut owned = Vec::new();
    owned
        .try_reserve_exact(source.len())
        .map_err(|_| plan_allocation_error())?;
    for &branch in source {
        if ir.node(branch).is_none() {
            return Err(invalid_ir("combinator branch node"));
        }
        queue_child(branch, context.worklist, context.queued);
        owned.push(branch);
    }
    charge_retained(
        context,
        owned.capacity().saturating_mul(size_of::<NodeId>()),
    )?;
    Ok(NodePlan::Combinator(CombinatorPlan {
        kind,
        branches: owned.into_boxed_slice(),
    }))
}

fn same_instance_child_count(node: &NodePlan) -> usize {
    match node {
        NodePlan::Ref { .. } | NodePlan::DynamicRef { .. } => 1,
        NodePlan::Negation { .. } => 1,
        NodePlan::Combinator(plan) => plan.branches.len(),
        NodePlan::Object(plan) => plan.dependent_schemas.len(),
        NodePlan::Unevaluated(_) => 1,
        NodePlan::Regular(_)
        | NodePlan::Number { .. }
        | NodePlan::Array(_)
        | NodePlan::String(_)
        | NodePlan::StringEnum(_)
        | NodePlan::Unsupported => 0,
    }
}

fn same_instance_child(node: &NodePlan, index: usize) -> Option<SameInstanceEdge> {
    match node {
        NodePlan::Ref { target } => (index == 0).then_some(SameInstanceEdge::Reference(*target)),
        NodePlan::DynamicRef { initial_target, .. } => {
            (index == 0).then_some(SameInstanceEdge::Reference(*initial_target))
        }
        NodePlan::Negation { inner } => (index == 0).then_some(SameInstanceEdge::Semantic(*inner)),
        NodePlan::Combinator(plan) => plan
            .branches
            .get(index)
            .copied()
            .map(SameInstanceEdge::Semantic),
        NodePlan::Object(plan) => plan
            .dependent_schemas
            .get(index)
            .map(|dependency| SameInstanceEdge::Semantic(dependency.schema)),
        NodePlan::Unevaluated(plan) => {
            (index == 0).then_some(SameInstanceEdge::Semantic(plan.scope))
        }
        NodePlan::Regular(_)
        | NodePlan::Number { .. }
        | NodePlan::Array(_)
        | NodePlan::String(_)
        | NodePlan::StringEnum(_)
        | NodePlan::Unsupported => None,
    }
}

fn max_concurrent_key_cursors(
    nodes: &[NodePlan],
    root: NodeId,
    limit: usize,
) -> Result<usize, PlanCompileFailure> {
    fn visit(
        nodes: &[NodePlan],
        node: NodeId,
        limit: usize,
        memo: &mut [Option<usize>],
        visiting: &mut [bool],
    ) -> Result<usize, PlanCompileFailure> {
        let index = usize::try_from(node.get()).map_err(|_| plan_allocation_error())?;
        let Some(plan) = nodes.get(index) else {
            return Err(invalid_ir("key-cursor node is invalid").into());
        };
        if let Some(count) = memo[index] {
            return Ok(count);
        }
        if visiting[index] {
            return Err(invalid_ir("same-instance cycle was not resolved").into());
        }
        visiting[index] = true;
        let mut add = |total: &mut usize, child| -> Result<(), PlanCompileFailure> {
            *total = total
                .checked_add(visit(nodes, child, limit, memo, visiting)?)
                .ok_or_else(plan_allocation_error)?;
            if *total > limit {
                return Err(
                    resource_limit(LimitKind::ActiveValidators, *total, limit, Stage::L3).into(),
                );
            }
            Ok(())
        };
        let mut total = usize::from(matches!(plan, NodePlan::Object(_)));
        match plan {
            NodePlan::Ref { target }
            | NodePlan::DynamicRef {
                initial_target: target,
                ..
            }
            | NodePlan::Negation { inner: target } => add(&mut total, *target)?,
            NodePlan::Combinator(plan) => {
                for &child in plan.branches.iter() {
                    add(&mut total, child)?;
                }
            }
            NodePlan::Object(plan) => {
                for dependency in plan.dependent_schemas.iter() {
                    add(&mut total, dependency.schema)?;
                }
            }
            NodePlan::Unevaluated(plan) => add(&mut total, plan.scope)?,
            NodePlan::Regular(_)
            | NodePlan::Array(_)
            | NodePlan::String(_)
            | NodePlan::StringEnum(_)
            | NodePlan::Number { .. }
            | NodePlan::Unsupported => {}
        }
        visiting[index] = false;
        memo[index] = Some(total);
        Ok(total)
    }

    let mut memo = Vec::new();
    memo.try_reserve_exact(nodes.len())
        .map_err(|_| plan_allocation_error())?;
    memo.resize(nodes.len(), None);
    let mut visiting = Vec::new();
    visiting
        .try_reserve_exact(nodes.len())
        .map_err(|_| plan_allocation_error())?;
    visiting.resize(nodes.len(), false);
    visit(nodes, root, limit, &mut memo, &mut visiting)
}

fn resolve_same_instance_cycles(
    nodes: &mut Vec<NodePlan>,
    reachable: &[bool],
    retained_bytes: &mut usize,
    limits: StructuredLimits,
) -> Result<(), PlanCompileFailure> {
    let node_count = reachable.len();
    let mut reverse_counts = Vec::new();
    reverse_counts
        .try_reserve_exact(node_count)
        .map_err(|_| plan_allocation_error())?;
    reverse_counts.resize(node_count, 0usize);
    let mut edge_count = 0usize;
    for index in 0..node_count {
        if !reachable[index] {
            continue;
        }
        let node = nodes
            .get(index)
            .ok_or_else(|| invalid_ir("same-instance node is invalid"))?;
        for child_index in 0..same_instance_child_count(node) {
            let target = same_instance_child(node, child_index)
                .ok_or_else(|| invalid_ir("same-instance edge is invalid"))?
                .target();
            let target = usize::try_from(target.get())
                .map_err(|_| invalid_ir("same-instance child is invalid"))?;
            if !reachable.get(target).copied().unwrap_or(false) {
                return Err(invalid_ir("reachable same-instance child is not compiled").into());
            }
            reverse_counts[target] = reverse_counts[target]
                .checked_add(1)
                .ok_or_else(plan_allocation_error)?;
            edge_count = edge_count
                .checked_add(1)
                .ok_or_else(plan_allocation_error)?;
        }
    }
    let scc_temporary = node_count
        .checked_mul(
            size_of::<usize>()
                .checked_mul(6)
                .and_then(|bytes| bytes.checked_add(size_of::<VisitFrame>()))
                .and_then(|bytes| bytes.checked_add(size_of::<bool>() * 2))
                .ok_or_else(plan_allocation_error)?,
        )
        .and_then(|bytes| bytes.checked_add(size_of::<usize>()))
        .and_then(|bytes| bytes.checked_add(edge_count.checked_mul(size_of::<usize>())?))
        .and_then(|bytes| bytes.checked_add(edge_count.checked_mul(size_of::<NodeId>())?))
        .ok_or_else(plan_allocation_error)?;
    let observed = retained_bytes
        .checked_add(scc_temporary)
        .ok_or_else(plan_allocation_error)?;
    if observed > limits.max_plan_bytes {
        return Err(resource_limit(
            LimitKind::PlanBytes,
            observed,
            limits.max_plan_bytes,
            Stage::L3,
        )
        .into());
    }

    let mut reverse_offsets = Vec::new();
    reverse_offsets
        .try_reserve_exact(node_count.saturating_add(1))
        .map_err(|_| plan_allocation_error())?;
    reverse_offsets.push(0usize);
    for count in reverse_counts.iter().copied() {
        let next = reverse_offsets
            .last()
            .copied()
            .and_then(|offset| offset.checked_add(count))
            .ok_or_else(plan_allocation_error)?;
        reverse_offsets.push(next);
    }
    let mut reverse_edges = Vec::new();
    reverse_edges
        .try_reserve_exact(edge_count)
        .map_err(|_| plan_allocation_error())?;
    reverse_edges.resize(edge_count, 0usize);
    let mut reverse_cursor = reverse_offsets[..node_count].to_vec();
    for index in 0..node_count {
        if !reachable[index] {
            continue;
        }
        let child_count = same_instance_child_count(&nodes[index]);
        for child_index in 0..child_count {
            let target = same_instance_child(&nodes[index], child_index)
                .ok_or_else(|| invalid_ir("same-instance edge is invalid"))?
                .target();
            let target = usize::try_from(target.get())
                .map_err(|_| invalid_ir("same-instance child is invalid"))?;
            let slot = reverse_cursor[target];
            reverse_edges[slot] = index;
            reverse_cursor[target] = slot.checked_add(1).ok_or_else(plan_allocation_error)?;
        }
    }

    let mut seen = Vec::new();
    seen.try_reserve_exact(node_count)
        .map_err(|_| plan_allocation_error())?;
    seen.resize(node_count, false);
    let mut stack = Vec::new();
    stack
        .try_reserve_exact(node_count)
        .map_err(|_| plan_allocation_error())?;
    let mut order = Vec::new();
    order
        .try_reserve_exact(node_count)
        .map_err(|_| plan_allocation_error())?;
    for start in 0..node_count {
        if !reachable[start] || seen[start] {
            continue;
        }
        let start_node = NodeId(u32::try_from(start).map_err(|_| plan_allocation_error())?);
        seen[start] = true;
        stack.push(VisitFrame {
            node: start_node,
            next_child: 0,
        });
        while let Some(frame) = stack.last_mut() {
            let index = usize::try_from(frame.node.get())
                .map_err(|_| invalid_ir("same-instance node is invalid"))?;
            let child_count = same_instance_child_count(
                nodes
                    .get(index)
                    .ok_or_else(|| invalid_ir("same-instance node is invalid"))?,
            );
            if frame.next_child == child_count {
                order.push(index);
                stack.pop();
                continue;
            }
            let edge = same_instance_child(&nodes[index], frame.next_child)
                .ok_or_else(|| invalid_ir("same-instance edge is invalid"))?;
            frame.next_child += 1;
            let child_index = usize::try_from(edge.target().get())
                .map_err(|_| invalid_ir("same-instance child is invalid"))?;
            if !seen[child_index] {
                seen[child_index] = true;
                let child =
                    NodeId(u32::try_from(child_index).map_err(|_| plan_allocation_error())?);
                stack.push(VisitFrame {
                    node: child,
                    next_child: 0,
                });
            }
        }
    }

    seen.fill(false);
    let mut component = Vec::new();
    component
        .try_reserve_exact(node_count)
        .map_err(|_| plan_allocation_error())?;
    let mut member = Vec::new();
    member
        .try_reserve_exact(node_count)
        .map_err(|_| plan_allocation_error())?;
    member.resize(node_count, false);
    let mut never = None;
    while let Some(start) = order.pop() {
        if seen[start] {
            continue;
        }
        component.clear();
        stack.clear();
        seen[start] = true;
        component.push(start);
        stack.push(VisitFrame {
            node: NodeId(u32::try_from(start).map_err(|_| plan_allocation_error())?),
            next_child: 0,
        });
        while let Some(frame) = stack.pop() {
            let index = usize::try_from(frame.node.get()).map_err(|_| plan_allocation_error())?;
            let begin = reverse_offsets[index];
            let end = reverse_offsets[index + 1];
            for &parent in &reverse_edges[begin..end] {
                if !seen[parent] {
                    seen[parent] = true;
                    component.push(parent);
                    stack.push(VisitFrame {
                        node: NodeId(u32::try_from(parent).map_err(|_| plan_allocation_error())?),
                        next_child: 0,
                    });
                }
            }
        }
        let cyclic = component.len() > 1 || {
            let only = component[0];
            (0..same_instance_child_count(&nodes[only])).any(|edge| {
                same_instance_child(&nodes[only], edge)
                    .is_some_and(|edge| usize::try_from(edge.target().get()) == Ok(only))
            })
        };
        if !cyclic {
            continue;
        }
        for &index in &component {
            member[index] = true;
        }
        let result = lower_positive_component(
            nodes,
            &component,
            &member,
            retained_bytes,
            limits,
            &mut never,
        );
        for &index in &component {
            member[index] = false;
        }
        result?;
    }
    Ok(())
}

fn lower_positive_component(
    nodes: &mut Vec<NodePlan>,
    component: &[usize],
    member: &[bool],
    retained_bytes: &mut usize,
    limits: StructuredLimits,
    never: &mut Option<NodeId>,
) -> Result<(), PlanCompileFailure> {
    let mut has_union = false;
    let mut has_intersection = false;
    for &index in component {
        match &nodes[index] {
            NodePlan::Ref { .. } | NodePlan::DynamicRef { .. } => {}
            NodePlan::Combinator(plan) => match plan.kind {
                CombinatorKind::Any => has_union = true,
                CombinatorKind::All => has_intersection = true,
                CombinatorKind::ExactlyOne => {
                    return Err(non_monotone_recursion("oneOf recursion"))
                }
            },
            NodePlan::Negation { .. } => return Err(non_monotone_recursion("not recursion")),
            NodePlan::Object(_) => {
                return Err(non_monotone_recursion("dependentSchemas recursion"))
            }
            NodePlan::Unevaluated(_) => {
                return Err(non_monotone_recursion("unevaluated recursion"))
            }
            _ => return Err(invalid_ir("zero-consumption SCC contains a consuming node").into()),
        }
    }
    if has_union && has_intersection {
        return lower_mixed_positive_component(
            nodes,
            component,
            member,
            retained_bytes,
            limits,
            never,
        );
    }

    let target = if has_union {
        let mut external = Vec::new();
        for &index in component {
            let child_count = same_instance_child_count(&nodes[index]);
            external
                .try_reserve(child_count)
                .map_err(|_| plan_allocation_error())?;
            for child_index in 0..child_count {
                let child = same_instance_child(&nodes[index], child_index)
                    .ok_or_else(|| invalid_ir("same-instance edge is invalid"))?
                    .target();
                let child_index = usize::try_from(child.get())
                    .map_err(|_| invalid_ir("same-instance child is invalid"))?;
                if !member.get(child_index).copied().unwrap_or(false) {
                    external.push(child);
                }
            }
        }
        external.sort_unstable_by_key(|node| node.get());
        external.dedup();
        match external.len() {
            0 => ensure_never_node(nodes, retained_bytes, limits, never)?,
            1 => external[0],
            _ => {
                let branch_bytes = external
                    .len()
                    .checked_mul(size_of::<NodeId>())
                    .ok_or_else(plan_allocation_error)?;
                charge(
                    retained_bytes,
                    branch_bytes,
                    limits.max_plan_bytes,
                    LimitKind::PlanBytes,
                    Stage::L3,
                )?;
                append_synthetic_node(
                    nodes,
                    NodePlan::Combinator(CombinatorPlan {
                        kind: CombinatorKind::Any,
                        branches: external.into_boxed_slice(),
                    }),
                    retained_bytes,
                    limits,
                )?
            }
        }
    } else {
        ensure_never_node(nodes, retained_bytes, limits, never)?
    };

    for &index in component {
        if let NodePlan::Combinator(plan) = &nodes[index] {
            let released = plan.branches.len().saturating_mul(size_of::<NodeId>());
            *retained_bytes = retained_bytes.saturating_sub(released);
        }
        nodes[index] = NodePlan::Ref { target };
    }
    Ok(())
}

struct PositiveCycleExpr {
    kind: Option<CombinatorKind>,
    children: Box<[NodeId]>,
}

fn lower_mixed_positive_component(
    nodes: &mut Vec<NodePlan>,
    component: &[usize],
    member: &[bool],
    retained_bytes: &mut usize,
    limits: StructuredLimits,
    never: &mut Option<NodeId>,
) -> Result<(), PlanCompileFailure> {
    let bottom = ensure_never_node(nodes, retained_bytes, limits, never)?;
    let mut local_index = Vec::new();
    local_index
        .try_reserve_exact(member.len())
        .map_err(|_| plan_allocation_error())?;
    local_index.resize(member.len(), usize::MAX);
    for (local, &index) in component.iter().enumerate() {
        local_index[index] = local;
    }

    let mut expressions = Vec::new();
    expressions
        .try_reserve_exact(component.len())
        .map_err(|_| plan_allocation_error())?;
    for &index in component {
        let expression = match &nodes[index] {
            NodePlan::Ref { target } => PositiveCycleExpr {
                kind: None,
                children: Box::new([*target]),
            },
            NodePlan::DynamicRef { initial_target, .. } => PositiveCycleExpr {
                kind: None,
                children: Box::new([*initial_target]),
            },
            NodePlan::Combinator(plan)
                if matches!(plan.kind, CombinatorKind::All | CombinatorKind::Any) =>
            {
                let mut children = Vec::new();
                children
                    .try_reserve_exact(plan.branches.len())
                    .map_err(|_| plan_allocation_error())?;
                children.extend_from_slice(&plan.branches);
                PositiveCycleExpr {
                    kind: Some(plan.kind),
                    children: children.into_boxed_slice(),
                }
            }
            _ => return Err(invalid_ir("positive SCC contains a non-positive node").into()),
        };
        expressions.push(expression);
    }

    let mut previous = Vec::new();
    previous
        .try_reserve_exact(component.len())
        .map_err(|_| plan_allocation_error())?;
    previous.resize(component.len(), bottom);
    let mut current = Vec::new();
    current
        .try_reserve_exact(component.len())
        .map_err(|_| plan_allocation_error())?;

    // A monotone Boolean system of N members reaches its least fixed point in at most N rises.
    for _ in 0..component.len() {
        current.clear();
        for expression in &expressions {
            let mut resolved = Vec::new();
            resolved
                .try_reserve_exact(expression.children.len())
                .map_err(|_| plan_allocation_error())?;
            for &child in expression.children.iter() {
                let child_index = usize::try_from(child.get())
                    .map_err(|_| invalid_ir("positive SCC child is invalid"))?;
                if member.get(child_index).copied().unwrap_or(false) {
                    let local = *local_index
                        .get(child_index)
                        .ok_or_else(|| invalid_ir("positive SCC child is invalid"))?;
                    let resolved_child = previous
                        .get(local)
                        .copied()
                        .ok_or_else(|| invalid_ir("positive SCC member is missing"))?;
                    resolved.push(resolved_child);
                } else {
                    resolved.push(child);
                }
            }
            let node = match (expression.kind, resolved.as_slice()) {
                (_, [only]) => *only,
                (Some(kind), _) => {
                    let branch_bytes = resolved
                        .len()
                        .checked_mul(size_of::<NodeId>())
                        .ok_or_else(plan_allocation_error)?;
                    charge(
                        retained_bytes,
                        branch_bytes,
                        limits.max_plan_bytes,
                        LimitKind::PlanBytes,
                        Stage::L3,
                    )?;
                    append_synthetic_node(
                        nodes,
                        NodePlan::Combinator(CombinatorPlan {
                            kind,
                            branches: resolved.into_boxed_slice(),
                        }),
                        retained_bytes,
                        limits,
                    )?
                }
                (None, _) => return Err(invalid_ir("reference has multiple targets").into()),
            };
            current.push(node);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    for (local, &index) in component.iter().enumerate() {
        if let NodePlan::Combinator(plan) = &nodes[index] {
            let released = plan.branches.len().saturating_mul(size_of::<NodeId>());
            *retained_bytes = retained_bytes.saturating_sub(released);
        }
        nodes[index] = NodePlan::Ref {
            target: previous[local],
        };
    }
    Ok(())
}

fn ensure_never_node(
    nodes: &mut Vec<NodePlan>,
    retained_bytes: &mut usize,
    limits: StructuredLimits,
    never: &mut Option<NodeId>,
) -> Result<NodeId, PlanCompileFailure> {
    if let Some(node) = *never {
        return Ok(node);
    }
    let engine = build_from_regex(r"[^\x00-\xff]")?;
    charge(
        retained_bytes,
        engine_retained_bytes(&engine),
        limits.max_plan_bytes,
        LimitKind::PlanBytes,
        Stage::L3,
    )?;
    let node = append_synthetic_node(
        nodes,
        NodePlan::Regular(Box::new(engine)),
        retained_bytes,
        limits,
    )?;
    *never = Some(node);
    Ok(node)
}

fn append_synthetic_node(
    nodes: &mut Vec<NodePlan>,
    node: NodePlan,
    retained_bytes: &mut usize,
    limits: StructuredLimits,
) -> Result<NodeId, PlanCompileFailure> {
    let index = nodes.len();
    let id = NodeId(u32::try_from(index).map_err(|_| plan_allocation_error())?);
    let before = nodes.capacity();
    nodes
        .try_reserve_exact(1)
        .map_err(|_| plan_allocation_error())?;
    let added = nodes
        .capacity()
        .checked_sub(before)
        .and_then(|slots| slots.checked_mul(size_of::<NodePlan>()))
        .ok_or_else(plan_allocation_error)?;
    charge(
        retained_bytes,
        added,
        limits.max_plan_bytes,
        LimitKind::PlanBytes,
        Stage::L3,
    )?;
    nodes.push(node);
    Ok(id)
}

type IntersectionPattern = (u32, Option<u32>, Option<Box<RefEngine>>);

fn fast_string_intersection(ir: &SchemaIR, branch_ids: &[NodeId]) -> bool {
    const MAX_PRODUCT_PATTERNS: usize = 9;
    const MAX_PRODUCT_SOURCE_BYTES: usize = 2 << 10;
    let mut user_patterns = 0usize;
    let mut source_bytes = 0usize;
    let mut has_negative = false;
    for &branch in branch_ids {
        let Some(candidate) = string_intersection_branch(ir, branch) else {
            return false;
        };
        match candidate {
            StringIntersectionBranch::Pattern {
                node,
                regex,
                negative,
            } => {
                has_negative |= negative;
                if ir.is_user_pattern(node) {
                    user_patterns += 1;
                    let Some(source) = ir.str_at(regex) else {
                        return false;
                    };
                    let Some(total) = source_bytes.checked_add(source.len()) else {
                        return false;
                    };
                    source_bytes = total;
                    if user_patterns > MAX_PRODUCT_PATTERNS
                        || source_bytes > MAX_PRODUCT_SOURCE_BYTES
                    {
                        return false;
                    }
                }
            }
            StringIntersectionBranch::NegativeConst { value } => {
                has_negative = true;
                let Some(value) = ir.str_at(value) else {
                    return false;
                };
                let Some(total) = source_bytes.checked_add(value.len().saturating_mul(10)) else {
                    return false;
                };
                source_bytes = total;
            }
        }
    }
    has_negative || (user_patterns <= 2 && source_bytes <= 128)
}

enum StringIntersectionBranch<'a> {
    Pattern {
        node: &'a Node,
        regex: crate::ir::StrRef,
        negative: bool,
    },
    NegativeConst {
        value: crate::ir::StrRef,
    },
}

fn string_intersection_branch(
    ir: &SchemaIR,
    branch: NodeId,
) -> Option<StringIntersectionBranch<'_>> {
    match ir.node(branch)? {
        node @ Node::StringPattern { regex, .. } => Some(StringIntersectionBranch::Pattern {
            node,
            regex: *regex,
            negative: false,
        }),
        Node::Not { inner } => match ir.node(*inner)? {
            node @ Node::StringPattern {
                regex,
                min_len: None,
                max_len: None,
                ..
            } => Some(StringIntersectionBranch::Pattern {
                node,
                regex: *regex,
                negative: true,
            }),
            Node::StringConst { value } => {
                Some(StringIntersectionBranch::NegativeConst { value: *value })
            }
            _ => None,
        },
        _ => None,
    }
}

fn decoded_literal_pattern(value: &str) -> Result<String, CompileError> {
    let capacity = value
        .chars()
        .count()
        .checked_mul(10)
        .ok_or_else(plan_allocation_error)?;
    let mut pattern = String::new();
    pattern
        .try_reserve_exact(capacity)
        .map_err(|_| plan_allocation_error())?;
    for scalar in value.chars() {
        pattern.push_str("\\x{");
        let mut value = u32::from(scalar);
        let mut digits = [0u8; 6];
        let mut used = 0usize;
        loop {
            let digit = (value & 0xf) as u8;
            digits[used] = if digit < 10 {
                b'0' + digit
            } else {
                b'a' + digit - 10
            };
            used += 1;
            value >>= 4;
            if value == 0 {
                break;
            }
        }
        for digit in digits[..used].iter().rev() {
            pattern.push(char::from(*digit));
        }
        pattern.push('}');
    }
    Ok(pattern)
}

fn build_intersection_pattern(
    ir: &SchemaIR,
    branch_ids: &[NodeId],
    context: &mut PlanBuild<'_>,
) -> Result<IntersectionPattern, PlanCompileFailure> {
    let mut min_scalars = 0u32;
    let mut max_scalars: Option<u32> = None;
    let mut positive_engines = Vec::new();
    let mut negative_engines = Vec::new();
    let engine_vec_bytes = branch_ids
        .len()
        .checked_mul(size_of::<RefEngine>())
        .ok_or_else(plan_allocation_error)?;
    reserve_temporary(context, engine_vec_bytes)?;
    positive_engines
        .try_reserve_exact(branch_ids.len())
        .map_err(|_| plan_allocation_error())?;
    negative_engines
        .try_reserve_exact(branch_ids.len())
        .map_err(|_| plan_allocation_error())?;
    for &branch in branch_ids {
        let Some(candidate) = string_intersection_branch(ir, branch) else {
            return Err(unsupported_plan("non-string intersection branch").into());
        };
        match candidate {
            StringIntersectionBranch::Pattern {
                node,
                regex,
                negative,
            } => {
                let Node::StringPattern {
                    min_len, max_len, ..
                } = node
                else {
                    unreachable!("pattern intersection branch")
                };
                if !negative {
                    min_scalars = min_scalars.max(min_len.unwrap_or(0));
                    if let Some(max_len) = max_len {
                        max_scalars = Some(max_scalars.map_or(*max_len, |old| old.min(*max_len)));
                    }
                }
                if ir.is_user_pattern(node) {
                    let source = ir
                        .str_at(regex)
                        .ok_or_else(|| invalid_ir("string pattern string reference"))?;
                    reserve_temporary(context, regex_compile_peak(source.len())?)?;
                    let engine = build_from_unicode_regex(source)?;
                    if negative {
                        negative_engines.push(engine);
                    } else {
                        positive_engines.push(engine);
                    }
                }
            }
            StringIntersectionBranch::NegativeConst { value } => {
                let literal = ir
                    .str_at(value)
                    .ok_or_else(|| invalid_ir("negative string const literal"))?;
                let source = decoded_literal_pattern(literal)?;
                reserve_temporary(context, regex_compile_peak(source.len())?)?;
                negative_engines.push(build_from_unicode_regex(&source)?);
            }
        }
    }
    let mut accepted = combine_pattern_engines(positive_engines, Combinator::All, context)?;
    let pattern = if negative_engines.is_empty() {
        accepted.map(Box::new)
    } else {
        if accepted.is_none() {
            const ANY_SCALARS: &str = "(?s:.)*";
            reserve_temporary(context, regex_compile_peak(ANY_SCALARS.len())?)?;
            accepted = Some(build_from_unicode_regex(ANY_SCALARS)?);
        }
        let accepted = accepted.expect("universal decoded-string engine was installed");
        negative_engines.insert(0, accepted);
        combine_pattern_engines(negative_engines, Combinator::Difference, context)?.map(Box::new)
    };
    Ok((min_scalars, max_scalars, pattern))
}

fn combine_pattern_engines(
    mut engines: Vec<RefEngine>,
    combinator: Combinator,
    context: &mut PlanBuild<'_>,
) -> Result<Option<RefEngine>, PlanCompileFailure> {
    const PRODUCT_WORK_BYTES: usize = (1 << 16) * 64;
    if engines.len() < 2 {
        return Ok(engines.pop());
    }
    let refs_bytes = engines
        .len()
        .checked_mul(size_of::<&RefEngine>())
        .ok_or_else(plan_allocation_error)?;
    reserve_temporary(context, refs_bytes)?;
    let retained = engines
        .iter()
        .map(engine_retained_bytes)
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(plan_allocation_error)?;
    reserve_temporary(context, retained)?;
    reserve_temporary(context, PRODUCT_WORK_BYTES)?;
    let mut refs = Vec::new();
    refs.try_reserve_exact(engines.len())
        .map_err(|_| plan_allocation_error())?;
    refs.extend(engines.iter());
    Ok(Some(
        ProductAutomaton::build_reachable(&refs, combinator)?.into_engine()?,
    ))
}

fn fallible_string(value: &str) -> Result<String, CompileError> {
    let mut owned = String::new();
    owned
        .try_reserve_exact(value.len())
        .map_err(|_| plan_allocation_error())?;
    owned.push_str(value);
    Ok(owned)
}

fn object_plan_minimum_bytes(
    ir: &SchemaIR,
    known: &[(crate::ir::StrRef, NodeId)],
    pattern_count: usize,
    dependent: &[(crate::ir::StrRef, NodeId)],
) -> Result<usize, CompileError> {
    let names = known.iter().try_fold(0usize, |total, (name, _)| {
        let bytes = ir
            .str_at(*name)
            .ok_or_else(|| invalid_ir("property name string reference"))?
            .len();
        total.checked_add(bytes).ok_or_else(plan_allocation_error)
    })?;
    let properties = known
        .len()
        .checked_mul(size_of::<PropertyPlan>())
        .ok_or_else(plan_allocation_error)?;
    let bitset = words_for(known.len())
        .checked_mul(size_of::<u64>())
        .ok_or_else(plan_allocation_error)?;
    let patterns = pattern_count
        .checked_mul(size_of::<CompiledPropertyPattern>())
        .ok_or_else(plan_allocation_error)?;
    let buckets = known
        .len()
        .checked_mul(size_of::<(String, PropertyId)>() + 16)
        .ok_or_else(plan_allocation_error)?;
    let dependent_names = dependent.iter().try_fold(0usize, |total, (name, _)| {
        let bytes = ir
            .str_at(*name)
            .ok_or_else(|| invalid_ir("dependentSchemas trigger string reference"))?
            .len();
        total.checked_add(bytes).ok_or_else(plan_allocation_error)
    })?;
    let dependent_entries = dependent
        .len()
        .checked_mul(size_of::<DependentSchemaPlan>())
        .ok_or_else(plan_allocation_error)?;
    let dependent_buckets = dependent
        .len()
        .checked_mul(size_of::<(String, u32)>() + 16)
        .ok_or_else(plan_allocation_error)?;
    [
        names,
        properties,
        bitset,
        patterns,
        buckets,
        dependent_names,
        dependent_entries,
        dependent_buckets,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| total.checked_add(bytes))
    .ok_or_else(plan_allocation_error)
}

fn plan_allocation_error() -> CompileError {
    CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L3,
        "structured plan allocation",
    )
}

fn queue_child(id: NodeId, worklist: &mut Vec<NodeId>, queued: &mut [bool]) {
    let idx = usize::try_from(id.get()).unwrap_or(usize::MAX);
    if let Some(slot) = queued.get_mut(idx) {
        if !*slot {
            *slot = true;
            worklist.push(id);
        }
    }
}

fn unsupported_plan(what: &'static str) -> CompileError {
    CompileError::new(
        ErrorCode::Unsupported,
        Stage::L3,
        "construct not yet supported by the incremental structured engine",
    )
    .with_keyword(what)
}

fn non_monotone_recursion(what: &'static str) -> PlanCompileFailure {
    CompileError::new(
        ErrorCode::NonMonotoneRecursion,
        Stage::L3,
        "non-monotone zero-consumption recursion",
    )
    .with_keyword(what)
    .into()
}

fn invalid_ir(what: &'static str) -> CompileError {
    CompileError::new(ErrorCode::Malformed, Stage::L3, "invalid structured IR").with_observed(what)
}

fn engine_retained_bytes(engine: &RefEngine) -> usize {
    size_of::<RefEngine>()
        .checked_add(engine.heap_bytes())
        .unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Builder, CompileOptions, ScalarLit, MAX_UNROLLED_ENUM};

    fn plan(schema: &str) -> StructuredPlan {
        let ir =
            Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
        StructuredPlan::compile(ir, StructuredLimits::default()).unwrap()
    }

    #[test]
    fn unevaluated_plan_marks_false_as_terminal() {
        for (schema, expected) in [
            (r#"{"type":"array","unevaluatedItems":false}"#, true),
            (r#"{"type":"array","unevaluatedItems":true}"#, false),
        ] {
            let compiled = plan(schema);
            let NodePlan::Unevaluated(plan) = compiled.node(compiled.root) else {
                panic!("expected an unevaluated plan")
            };
            assert_eq!(plan.rejects_unevaluated, expected);
        }
    }

    #[test]
    fn key_cursor_bound_counts_same_instance_object_branches() {
        let compiled = plan(
            r#"{"allOf":[
                {"type":"object","dependentRequired":{"a":["b"]}},
                {"type":"object","dependentRequired":{"c":["d"]}}
            ]}"#,
        );
        assert_eq!(compiled.max_concurrent_key_cursors, 2);
    }

    fn string_enum_ir(values: impl IntoIterator<Item = String>) -> Arc<SchemaIR> {
        let mut builder = Builder::new(CompileOptions::default());
        let values = values.into_iter().map(ScalarLit::Str).collect();
        let root = builder.enum_values(values).expect("enum");
        Arc::new(builder.finish(root).expect("ir"))
    }

    #[test]
    fn two_small_user_patterns_use_the_bounded_product_route() {
        let compiled = plan(
            r#"{"allOf":[{"type":"string","pattern":"^a"},{"type":"string","pattern":"z$"}]}"#,
        );
        assert!(matches!(compiled.node(compiled.root), NodePlan::String(_)));
    }

    #[test]
    fn three_user_patterns_keep_the_general_parallel_route() {
        let compiled = plan(
            r#"{"allOf":[{"type":"string","pattern":"^a"},{"type":"string","pattern":"z$"},{"type":"string","pattern":"b"}]}"#,
        );
        assert!(matches!(
            compiled.node(compiled.root),
            NodePlan::Combinator(CombinatorPlan {
                kind: CombinatorKind::All,
                ..
            })
        ));
    }

    #[test]
    fn static_dynamic_anchor_reference_keeps_dynamic_plan_storage_empty() {
        let compiled =
            plan(r##"{"$dynamicRef":"#x","$defs":{"target":{"$anchor":"x","type":"integer"}}}"##);
        assert!(!compiled.dynamic_scope_enabled);
        assert!(compiled.dynamic_anchor_names.is_empty());
        assert!(compiled.dynamic_bindings.is_empty());
    }

    #[test]
    fn large_string_enum_uses_the_shared_regular_route() {
        let mut values = (0..MAX_UNROLLED_ENUM)
            .map(|index| format!("member-{index:04}"))
            .collect::<Vec<_>>();
        values.extend(["a".to_owned(), "ab".to_owned(), "abc".to_owned()]);
        let ir = string_enum_ir(values);
        let compiled = StructuredPlan::compile(ir, StructuredLimits::default()).expect("plan");
        let NodePlan::Regular(engine) = compiled.node(compiled.root) else {
            panic!("expected regular enum plan")
        };
        assert!(engine.accepts(br#""a""#));
        assert!(engine.accepts(br#""member-0000""#));
        assert!(!engine.accepts(br#""missing""#));
    }

    #[test]
    fn frontend_large_string_enum_literals_are_strictly_sorted() {
        let members = (0..=MAX_UNROLLED_ENUM)
            .rev()
            .map(|index| format!(r#""member-{index:04}""#))
            .collect::<Vec<_>>()
            .join(",");
        let ir = crate::frontend::schema_to_ir(
            &format!(r#"{{"enum":[{members}]}}"#),
            CompileOptions::default(),
        )
        .expect("ir");
        let Node::Enum { values } = ir.node(ir.root()).expect("root") else {
            panic!("expected enum root")
        };
        let literals = ir.lits_at(*values).expect("literals");
        assert!(literals.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn small_string_enum_keeps_the_regular_engine() {
        let compiled = plan(r#"{"enum":["a","ab","abc"]}"#);
        assert!(matches!(compiled.node(compiled.root), NodePlan::Regular(_)));
    }

    #[test]
    fn string_enum_threshold_and_checked_layout_boundaries_are_exact() {
        for count in [63usize, 64, 65, MAX_UNROLLED_ENUM] {
            let ir = string_enum_ir((0..count).map(|index| format!("member-{index:04}")));
            let compiled = StructuredPlan::compile(ir, StructuredLimits::default()).expect("plan");
            assert!(matches!(compiled.node(compiled.root), NodePlan::Regular(_)));
        }
        let ir = string_enum_ir((0..=MAX_UNROLLED_ENUM).map(|index| format!("member-{index:04}")));
        let compiled = StructuredPlan::compile(ir, StructuredLimits::default()).expect("plan");
        assert!(matches!(compiled.node(compiled.root), NodePlan::Regular(_)));

        if usize::BITS > u32::BITS {
            let largest = usize::try_from(u32::MAX).unwrap();
            assert!(string_enum_retained_bytes(1, largest).is_ok());
            assert!(string_enum_retained_bytes(1, largest + 1).is_err());
            assert!(string_enum_retained_bytes(largest - 1, 0).is_ok());
            assert!(string_enum_retained_bytes(largest, 0).is_err());
        }
        assert!(string_enum_retained_bytes(usize::MAX, 0).is_err());
        assert!(string_enum_retained_bytes(1, usize::MAX).is_err());
    }

    #[test]
    fn large_string_enum_plan_limit_is_fatal_at_the_exact_boundary() {
        let ir = string_enum_ir((0..=MAX_UNROLLED_ENUM).map(|index| format!("member-{index:04}")));
        let mut low = 1usize;
        let mut high = StructuredLimits::default().max_plan_bytes;
        while low < high {
            let mid = low + (high - low) / 2;
            let limits = StructuredLimits {
                max_plan_bytes: mid,
                ..StructuredLimits::default()
            };
            if StructuredPlan::compile(ir.clone(), limits).is_ok() {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        let exact = low;
        assert!(StructuredPlan::compile(
            ir.clone(),
            StructuredLimits {
                max_plan_bytes: exact,
                ..StructuredLimits::default()
            }
        )
        .is_ok());
        let failure = match StructuredPlan::compile(
            ir,
            StructuredLimits {
                max_plan_bytes: exact - 1,
                ..StructuredLimits::default()
            },
        ) {
            Ok(_) => panic!("one byte below the exact cap compiled"),
            Err(error) => error,
        };
        assert!(matches!(failure, PlanCompileFailure::Fatal(_)));
        assert_eq!(failure.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn large_string_enum_malformed_metadata_is_rejected_without_allocation() {
        assert_eq!(
            string_enum_layout(&[ScalarLit::Str("a".to_owned()), ScalarLit::Int(1)])
                .unwrap_err()
                .code,
            ErrorCode::Malformed
        );
        assert_eq!(
            string_enum_layout(&[
                ScalarLit::Str("a".to_owned()),
                ScalarLit::Str("a".to_owned()),
            ])
            .unwrap_err()
            .code,
            ErrorCode::Malformed
        );

        let valid =
            string_enum_ir((0..=MAX_UNROLLED_ENUM).map(|index| format!("member-{index:04}")));
        let mut wire = valid.to_wire_struct();
        let root = usize::try_from(wire.root.get()).expect("root index");
        let Node::Enum { values } = &mut wire.nodes[root] else {
            panic!("expected enum root")
        };
        values.off = u32::MAX;
        assert_eq!(
            SchemaIR::assemble(wire).unwrap_err().code,
            ErrorCode::Malformed
        );
    }

    #[test]
    fn ten_thousand_member_string_enum_uses_the_shared_regular_engine() {
        let values = (0..10_000)
            .map(|index| format!("member-{index:05}"))
            .collect::<Vec<_>>();
        let started = std::time::Instant::now();
        let ir = string_enum_ir(values);
        let compiled = StructuredPlan::compile(ir, StructuredLimits::default()).expect("plan");
        let elapsed = started.elapsed();
        let NodePlan::Regular(engine) = compiled.node(compiled.root) else {
            panic!("expected regular enum plan")
        };
        assert!(engine.accepts(br#""member-00000""#));
        assert!(engine.accepts(br#""member-09999""#));
        assert!(!engine.accepts(br#""member-10000""#));
        eprintln!("10,000-member string enum frontend+plan: {elapsed:?}");
    }

    #[test]
    fn a_closed_object_with_scalar_fields_compiles_as_one_regular_engine_at_root() {
        let p = plan(
            r#"{"type":"object","properties":{"a":{"type":"boolean"}},"additionalProperties":false}"#,
        );
        assert!(matches!(p.node(p.root), NodePlan::Regular(_)));
    }

    #[test]
    fn an_open_object_gets_an_object_plan() {
        let p = plan(
            r#"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":{"type":"boolean"}}"#,
        );
        match p.node(p.root) {
            NodePlan::Object(op) => {
                assert_eq!(op.known.len(), 1);
                assert_eq!(
                    op.known_by_name.keys().next().map(|name| name.as_ref()),
                    Some("a")
                );
                assert!(matches!(op.additional, AdditionalPlan::Schema(_)));
                assert_eq!(op.required_template[0] & 1, 1);
            }
            _ => panic!("expected an object plan"),
        }
    }

    #[test]
    fn local_recursive_reference_compiles_as_one_shared_graph_edge() {
        let ir = Arc::new(
            crate::frontend::schema_to_ir(
                r##"{
                    "$defs":{"node":{"type":"object","properties":{"next":{"$ref":"#/$defs/node"}},"additionalProperties":false}},
                    "$ref":"#/$defs/node"
                }"##,
                CompileOptions::default(),
            )
            .unwrap(),
        );
        let expected = match ir.node(ir.root()) {
            Some(Node::Ref { def }) => ir.def_target(*def).unwrap(),
            _ => panic!("recursive root must lower to a reference"),
        };
        let p = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
        let NodePlan::Ref { target } = p.node(p.root) else {
            panic!("recursive root must compile as a reference plan");
        };
        assert_eq!(*target, expected);
        assert_eq!(
            p.nodes
                .iter()
                .filter(|node| matches!(node, NodePlan::Object(_)))
                .count(),
            1,
            "the definition body is compiled once"
        );
        assert!(p.retained_bytes >= p.nodes.len() * size_of::<NodePlan>());
    }

    #[test]
    fn invalid_reference_definition_index_is_malformed_ir() {
        let valid = crate::frontend::schema_to_ir(
            r##"{"$defs":{"node":{"type":"object","properties":{"next":{"$ref":"#/$defs/node"}}}},"$ref":"#/$defs/node"}"##,
            CompileOptions::default(),
        )
        .unwrap();
        let mut wire = valid.to_wire_struct();
        let root = usize::try_from(wire.root.get()).unwrap();
        wire.nodes[root] = Node::Ref { def: u32::MAX };
        let error = SchemaIR::assemble(wire).unwrap_err();
        assert_eq!(error.code, ErrorCode::Malformed);
    }

    #[test]
    fn invalid_dependent_schema_child_node_id_is_malformed_ir() {
        let valid = crate::frontend::schema_to_ir(
            r#"{"type":"object","dependentSchemas":{"trig":{"type":"null"}},"additionalProperties":true}"#,
            CompileOptions::default(),
        )
        .unwrap();
        let mut wire = valid.to_wire_struct();
        assert_eq!(
            wire.props.len(),
            1,
            "one trigger, one dependent-schema prop"
        );
        wire.props[0].1 = NodeId(u32::MAX);
        let error = SchemaIR::assemble(wire).unwrap_err();
        assert_eq!(error.code, ErrorCode::Malformed);
    }

    #[test]
    fn invalid_dependent_schema_trigger_str_ref_is_malformed_ir() {
        let valid = crate::frontend::schema_to_ir(
            r#"{"type":"object","dependentSchemas":{"trig":{"type":"null"}},"additionalProperties":true}"#,
            CompileOptions::default(),
        )
        .unwrap();
        let mut wire = valid.to_wire_struct();
        assert_eq!(
            wire.props.len(),
            1,
            "one trigger, one dependent-schema prop"
        );
        wire.props[0].0 = crate::ir::StrRef {
            off: u32::MAX,
            len: 1,
        };
        let error = SchemaIR::assemble(wire).unwrap_err();
        assert_eq!(error.code, ErrorCode::Malformed);
    }

    #[test]
    fn pure_reference_cycles_compile_to_the_least_fixed_point() {
        for schema in [
            r##"{"$ref":"#"}"##,
            r##"{"$defs":{"a":{"$ref":"#/$defs/b"},"b":{"$ref":"#/$defs/a"}},"$ref":"#/$defs/a"}"##,
        ] {
            let ir =
                Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
            let compiled = StructuredPlan::compile(ir, StructuredLimits::default())
                .expect("positive reference SCC");
            assert!(matches!(compiled.node(compiled.root), NodePlan::Ref { .. }));
        }
    }

    #[test]
    fn long_reference_chain_compiles_iteratively() {
        let mut builder = crate::ir::Builder::new(CompileOptions::default());
        let mut slots = Vec::new();
        for _ in 0..1024 {
            slots.push(builder.alloc_def_slot().unwrap());
        }
        let mut refs = Vec::new();
        for &slot in &slots {
            refs.push(builder.ref_node(slot).unwrap());
        }
        let terminal = builder.boolean().unwrap();
        for (index, &slot) in slots.iter().enumerate() {
            let target = refs.get(index + 1).copied().unwrap_or(terminal);
            builder.set_def_target(slot, target).unwrap();
        }
        let ir = Arc::new(builder.finish(refs[0]).unwrap());
        let plan = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
        assert!(matches!(
            plan.node(resolve_chain_for_test(&plan, plan.root)),
            NodePlan::Regular(_)
        ));
    }

    fn resolve_chain_for_test(plan: &StructuredPlan, mut node: NodeId) -> NodeId {
        for _ in 0..=plan.nodes.len() {
            match plan.node(node) {
                NodePlan::Ref { target } => node = *target,
                _ => return node,
            }
        }
        panic!("test chain must terminate")
    }

    #[test]
    fn dependent_schemas_compile_incrementally() {
        let ir = Arc::new(
            crate::frontend::schema_to_ir(
                r#"{"type":"object","additionalProperties":true,"dependentSchemas":{"a":{"type":"object","required":["b"],"additionalProperties":true}}}"#,
                CompileOptions::default(),
            )
            .unwrap(),
        );
        let plan = StructuredPlan::compile(ir, StructuredLimits::default()).expect("plan");
        let object = object_plan_of(&plan, plan.root);
        assert_eq!(object.dependent_by_name.get("a"), Some(&0));
        assert_eq!(object.dependent_schemas.len(), 1);
    }

    #[test]
    fn dependent_schema_node_ids_are_deduplicated() {
        let ir = Arc::new(
            crate::frontend::schema_to_ir(
                r#"{"type":"object","dependentSchemas":{"a":false,"b":false},"additionalProperties":true}"#,
                CompileOptions::default(),
            )
            .unwrap(),
        );
        let plan = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
        let object = object_plan_of(&plan, plan.root);
        assert_eq!(object.dependent_schemas.len(), 1);
        assert_eq!(object.dependent_by_name["a"], object.dependent_by_name["b"]);
    }

    #[test]
    fn empty_and_distinct_dependent_schema_plans_are_exact() {
        let empty = plan(r#"{"type":"object","dependentSchemas":{},"additionalProperties":true}"#);
        let empty_object = object_plan_of(&empty, empty.root);
        assert!(empty_object.dependent_by_name.is_empty());
        assert!(empty_object.dependent_schemas.is_empty());

        let distinct = plan(
            r#"{"type":"object","dependentSchemas":{"a":true,"b":false},"additionalProperties":true}"#,
        );
        let object = object_plan_of(&distinct, distinct.root);
        assert_eq!(object.dependent_by_name.len(), 2);
        assert_eq!(object.dependent_schemas.len(), 2);
        assert_ne!(object.dependent_by_name["a"], object.dependent_by_name["b"]);
    }

    #[test]
    fn dependent_schema_retained_bytes_scale_with_unique_children_and_names() {
        let retained = |count: usize| {
            let entries = (0..count)
                .map(|index| format!(r#""trigger-{index}":{{"const":{index}}}"#))
                .collect::<Vec<_>>()
                .join(",");
            plan(&format!(
                r#"{{"type":"object","dependentSchemas":{{{entries}}},"additionalProperties":true}}"#
            ))
            .retained_bytes
        };
        let one = retained(1);
        let eight = retained(8);
        let many = retained(64);
        assert!(one < eight, "{one} !< {eight}");
        assert!(eight < many, "{eight} !< {many}");
    }

    #[test]
    fn deep_acyclic_dependent_schema_graph_compiles_iteratively() {
        let mut builder = crate::ir::Builder::new(CompileOptions::default());
        let mut current = builder.boolean().unwrap();
        for index in 0..2048 {
            current = builder
                .open_object_full(
                    Vec::new(),
                    &[],
                    Vec::new(),
                    crate::ir::AdditionalPolicy::AllowAny,
                    None,
                    None,
                    None,
                    vec![(format!("trigger-{index}"), current)],
                    Vec::new(),
                )
                .unwrap();
        }
        let ir = Arc::new(builder.finish(current).unwrap());
        let compiled = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
        assert!(matches!(compiled.node(compiled.root), NodePlan::Object(_)));
        assert_eq!(
            compiled
                .nodes
                .iter()
                .filter(|node| matches!(node, NodePlan::Object(_)))
                .count(),
            2048
        );
    }

    #[test]
    fn dependent_schema_same_instance_cycle_is_a_typed_compile_error() {
        let ir = Arc::new(
            crate::frontend::schema_to_ir(
                r##"{"$defs":{"x":{"type":"object","dependentSchemas":{"a":{"$ref":"#/$defs/x"}},"additionalProperties":true}},"$ref":"#/$defs/x"}"##,
                CompileOptions::default(),
            )
            .unwrap(),
        );
        let error = match StructuredPlan::compile(ir, StructuredLimits::default()) {
            Ok(_) => panic!("same-instance dependency cycle compiled"),
            Err(error) => error,
        };
        assert_eq!(error.code, ErrorCode::NonMonotoneRecursion);
        assert!(!error.recoverable);
    }

    #[test]
    fn unreachable_recursive_definition_does_not_reject_reachable_root() {
        let ir = Arc::new(
            crate::frontend::schema_to_ir(
                r##"{"$defs":{"loop":{"$ref":"#/$defs/loop"}},"type":"string"}"##,
                CompileOptions::default(),
            )
            .unwrap(),
        );
        let plan = StructuredPlan::compile(ir, StructuredLimits::default());
        assert!(plan.is_ok());
    }

    #[test]
    fn tiny_plan_budget_returns_a_typed_error() {
        let ir = Arc::new(
            crate::frontend::schema_to_ir(
                r#"{"type":"object","properties":{"long-property":{"type":"string"}}}"#,
                CompileOptions::default(),
            )
            .unwrap(),
        );
        let limits = StructuredLimits {
            max_plan_bytes: 1,
            ..StructuredLimits::default()
        };
        let result = StructuredPlan::compile(ir, limits);
        assert!(matches!(
            result,
            Err(PlanCompileFailure::Fatal(CompileError {
                code: ErrorCode::InternalLimitExceeded,
                ..
            }))
        ));
    }

    #[test]
    fn finite_key_viability_is_optional_under_a_predictable_plan_budget() {
        let properties = (0..256)
            .map(|index| format!("\"property-{index:03}\":{{\"type\":\"string\"}}"))
            .collect::<Vec<_>>()
            .join(",");
        let schema = format!(
            r#"{{"type":"object","properties":{{{properties}}},"unevaluatedProperties":false}}"#
        );
        let ir =
            Arc::new(crate::frontend::schema_to_ir(&schema, CompileOptions::default()).unwrap());
        let default_plan = StructuredPlan::compile(ir.clone(), StructuredLimits::default())
            .expect("default finite-key plan");
        let NodePlan::Unevaluated(default_unevaluated) = default_plan.node(default_plan.root)
        else {
            panic!("expected unevaluated root plan");
        };
        assert!(default_unevaluated.finite_evaluated_keys.is_some());

        let mut lo = 1usize;
        let mut hi = StructuredLimits::default().max_plan_bytes;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let limits = StructuredLimits {
                max_plan_bytes: mid,
                ..StructuredLimits::default()
            };
            if StructuredPlan::compile(ir.clone(), limits).is_ok() {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let limits = StructuredLimits {
            max_plan_bytes: lo,
            ..StructuredLimits::default()
        };
        let constrained = StructuredPlan::compile(ir, limits)
            .expect("base plan should remain compilable when optional trie is omitted");
        let NodePlan::Unevaluated(constrained_unevaluated) = constrained.node(constrained.root)
        else {
            panic!("expected unevaluated constrained root plan");
        };
        assert!(constrained_unevaluated.finite_evaluated_keys.is_none());
    }

    #[test]
    fn finite_key_pattern_union_is_bounded_and_false_patterns_fall_back() {
        let single = plan(
            r#"{"patternProperties":{"^x$":{"type":"string"}},"additionalProperties":false,"unevaluatedProperties":false}"#,
        );
        let NodePlan::Unevaluated(single_root) = single.node(single.root) else {
            panic!("single pattern unevaluated root");
        };
        let single_trie = single_root
            .finite_evaluated_keys
            .as_ref()
            .expect("one pattern has a bounded one-DFA proof");
        assert_eq!(single_trie.patterns.len(), 1);
        assert!(single_trie.permits("x", true, 0, |_| false));
        assert!(!single_trie.permits("xy", true, 0, |_| false));

        let union = plan(
            r#"{"patternProperties":{"^x":true,"^y":true},"additionalProperties":false,"unevaluatedProperties":false}"#,
        );
        let NodePlan::Unevaluated(union_root) = union.node(union.root) else {
            panic!("pattern union unevaluated root");
        };
        let union_trie = union_root
            .finite_evaluated_keys
            .as_ref()
            .expect("positive pattern union has a bounded proof");
        assert_eq!(union_trie.patterns.len(), 2);
        assert!(union_trie.permits("x", false, 0, |_| false));
        assert!(union_trie.permits("y", false, 0, |_| false));
        assert!(!union_trie.permits("z", false, 0, |_| false));

        let false_pattern = plan(
            r#"{"properties":{"x":true},"patternProperties":{"^x$":false},"additionalProperties":false,"unevaluatedProperties":false}"#,
        );
        let NodePlan::Unevaluated(root) = false_pattern.node(false_pattern.root) else {
            panic!("false-pattern fallback unevaluated root");
        };
        assert!(root.finite_evaluated_keys.is_none());
    }

    #[test]
    fn plan_peak_includes_scratch_retained_and_intersection_temporaries() {
        let schema = r#"{"allOf":[
            {"type":"string","pattern":"a+"},
            {"type":"string","pattern":"a*"},
            {"type":"string","pattern":"aa*"}
        ]}"#;
        let ir =
            Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
        let sufficient = StructuredPlan::compile(ir.clone(), StructuredLimits::default())
            .expect("intersection fits the default plan budget");
        assert!(sufficient.retained_bytes > 0);
        let limits = StructuredLimits {
            max_plan_bytes: sufficient.retained_bytes,
            ..StructuredLimits::default()
        };
        let result = StructuredPlan::compile(ir, limits);
        assert!(matches!(
            result,
            Err(PlanCompileFailure::Fatal(CompileError {
                code: ErrorCode::InternalLimitExceeded,
                ..
            }))
        ));
    }

    #[test]
    fn a_nested_open_object_property_gets_its_own_plan_node() {
        let p = plan(
            r#"{"type":"object","additionalProperties":{"type":"boolean"},"properties":{"a":{"type":"object","additionalProperties":{"type":"boolean"},"properties":{"b":{"type":"boolean"}}}}}"#,
        );
        let NodePlan::Object(outer) = p.node(p.root) else {
            panic!("expected object plan");
        };
        let inner_id = outer.known[0].value;
        assert!(matches!(p.node(inner_id), NodePlan::Object(_)));
    }

    #[test]
    fn non_regular_combinators_keep_dense_branch_ids_in_the_plan() {
        let cases = [
            (
                r#"{"allOf":[{"type":"object","additionalProperties":{"type":"integer"}},{"type":"object","patternProperties":{"^x":{"type":"integer"}}}]}"#,
                CombinatorKind::All,
            ),
            (
                r#"{"anyOf":[{"type":"object","additionalProperties":{"type":"integer"}},{"type":"array","items":{"type":"integer"},"uniqueItems":true}]}"#,
                CombinatorKind::Any,
            ),
            (
                r#"{"oneOf":[{"type":"object","additionalProperties":{"type":"integer"}},{"type":"array","items":{"type":"integer"},"contains":{"const":1}}]}"#,
                CombinatorKind::ExactlyOne,
            ),
        ];
        for (schema, expected) in cases {
            let p = plan(schema);
            let NodePlan::Combinator(combinator) = p.node(p.root) else {
                panic!("expected incremental combinator for {schema}");
            };
            assert_eq!(combinator.kind, expected);
            assert_eq!(combinator.branches.len(), 2);
            assert!(combinator
                .branches
                .iter()
                .all(|branch| usize::try_from(branch.get()).unwrap() < p.nodes.len()));
            assert!(p.retained_bytes >= combinator.branches.len() * size_of::<NodeId>());
        }
    }

    #[test]
    fn cheap_regular_combinators_stay_on_the_regular_path() {
        for schema in [
            r#"{"allOf":[{"type":"boolean"},{"const":true}]}"#,
            r#"{"anyOf":[{"const":"a"},{"const":"b"}]}"#,
            r#"{"oneOf":[{"const":"a"},{"const":"b"}]}"#,
        ] {
            let p = plan(schema);
            assert!(matches!(p.node(p.root), NodePlan::Regular(_)), "{schema}");
        }
    }

    #[test]
    fn negation_has_a_dedicated_plan_and_compiles_its_inner() {
        for schema in [
            r#"{"not":{"const":1}}"#,
            r#"{"not":{"type":"object","additionalProperties":{"type":"integer"}}}"#,
            r#"{"not":{"anyOf":[{"type":"null"},{"type":"object","additionalProperties":{"type":"integer"}}]}}"#,
            r#"{"not":{"not":{"const":1}}}"#,
        ] {
            let p = plan(schema);
            let NodePlan::Negation { inner } = p.node(p.root) else {
                panic!("expected negation plan for {schema}");
            };
            assert!(!matches!(p.node(*inner), NodePlan::Unsupported), "{schema}");
        }
    }

    #[test]
    fn negation_same_instance_cycles_are_nonrecoverable() {
        for schema in [
            r##"{"not":{"$ref":"#"}}"##,
            r##"{"$defs":{"x":{"not":{"$ref":"#/$defs/x"}}},"$ref":"#/$defs/x"}"##,
            r##"{"$defs":{"x":{"not":{"anyOf":[{"$ref":"#/$defs/x"},{"type":"null"}]}}},"$ref":"#/$defs/x"}"##,
        ] {
            let ir =
                Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
            let error = match StructuredPlan::compile(ir, StructuredLimits::default()) {
                Ok(_) => panic!("same-instance negation cycle compiled"),
                Err(error) => error,
            };
            assert_eq!(error.code, ErrorCode::NonMonotoneRecursion, "{schema}");
            assert!(!error.recoverable, "{schema}");
        }
    }

    #[test]
    fn productive_recursion_with_negation_remains_incremental() {
        for schema in [
            r##"{"$defs":{"node":{"type":"object","properties":{"value":{"not":{"const":0}},"next":{"$ref":"#/$defs/node"}},"additionalProperties":false}},"$ref":"#/$defs/node"}"##,
            r##"{"$defs":{"node":{"type":"array","prefixItems":[{"not":{"const":0}}],"items":{"$ref":"#/$defs/node"}}},"$ref":"#/$defs/node"}"##,
        ] {
            let compiled = plan(schema);
            assert!(
                compiled
                    .nodes
                    .iter()
                    .any(|node| matches!(node, NodePlan::Negation { .. })),
                "{schema}"
            );
        }
    }

    #[test]
    fn deep_acyclic_negation_reference_chain_compiles_iteratively() {
        let mut builder = crate::ir::Builder::new(CompileOptions::default());
        let mut current = builder.boolean().unwrap();
        for _ in 0..2048 {
            let slot = builder.alloc_def_slot().unwrap();
            let reference = builder.ref_node(slot).unwrap();
            builder.set_def_target(slot, current).unwrap();
            current = builder.not(reference).unwrap();
        }
        let ir = Arc::new(builder.finish(current).unwrap());
        let compiled = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
        assert!(matches!(
            compiled.node(compiled.root),
            NodePlan::Negation { .. }
        ));
    }

    #[test]
    fn positive_same_instance_union_and_intersection_cycles_compile() {
        for schema in [
            r##"{"$defs":{"x":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/x"}]}},"$ref":"#/$defs/x"}"##,
            r##"{"$defs":{"x":{"allOf":[true,{"$ref":"#/$defs/x"}]}},"$ref":"#/$defs/x"}"##,
            r##"{"$defs":{"x":{"$ref":"#/$defs/y"},"y":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/x"}]}},"$ref":"#/$defs/x"}"##,
        ] {
            let ir =
                Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
            StructuredPlan::compile(ir, StructuredLimits::default())
                .unwrap_or_else(|error| panic!("{schema}: {error:?}"));
        }
    }

    #[test]
    fn exactly_one_recursion_is_a_typed_non_monotone_error() {
        let schema = r##"{"$defs":{"x":{"oneOf":[{"type":"null"},{"$ref":"#/$defs/x"}]}},"$ref":"#/$defs/x"}"##;
        let ir =
            Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).expect("ir"));
        let error = match StructuredPlan::compile(ir, StructuredLimits::default()) {
            Ok(_) => panic!("oneOf recursion compiled"),
            Err(error) => error,
        };
        assert_eq!(error.code, ErrorCode::NonMonotoneRecursion);
        assert!(!error.recoverable);
    }

    #[test]
    fn unreachable_same_instance_combinator_cycle_does_not_reject_the_root() {
        let p = plan(
            r##"{"$defs":{"x":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/x"}]}},"type":"string"}"##,
        );
        assert!(matches!(p.node(p.root), NodePlan::Regular(_)));
    }

    #[test]
    fn dependent_required_plan_groups_unique_names_and_edges_without_known_properties() {
        let p = plan(
            r#"{"type":"object","dependentRequired":{"a":["b","c"],"x":["c"]},"additionalProperties":true}"#,
        );
        let object = object_plan_of(&p, p.root);
        assert!(object.known_by_name.is_empty());
        assert!(object.known.is_empty());
        assert!(object.dependent_schemas.is_empty());
        let presence = object.dependent_required.as_ref().unwrap();
        assert_eq!(presence.name_count, 4);
        assert_eq!(presence.name_by_text.len(), 4);
        assert_eq!(presence.rules.len(), 2);
        assert_eq!(presence.required_ids.len(), 3);
    }

    /// Exercises large frontend and nested-vector grouping at 4,096 edges.
    /// Correctness relies on structural assertions rather than wall-clock timing.
    #[test]
    fn dependent_required_large_shapes_compile_bounded_and_stay_order_independent() {
        use std::fmt::Write as _;
        const N: usize = 4096;

        // Shape 1: one trigger, N required names.
        let mut one_by_many = String::from(r#"{"type":"object","dependentRequired":{"a":["#);
        for i in 0..N {
            if i > 0 {
                one_by_many.push(',');
            }
            write!(one_by_many, "\"r{i}\"").unwrap();
        }
        one_by_many.push_str(r#"]},"additionalProperties":true}"#);

        // Shape 2: N triggers, each with its own single required name.
        let mut many_by_one = String::from(r#"{"type":"object","dependentRequired":{"#);
        for i in 0..N {
            if i > 0 {
                many_by_one.push(',');
            }
            write!(many_by_one, "\"t{i}\":[\"shared\"]").unwrap();
        }
        many_by_one.push_str(r#"},"additionalProperties":true}"#);

        // Shape 3: N triggers, each with its own unique required name (no name sharing at all).
        let mut many_unique = String::from(r#"{"type":"object","dependentRequired":{"#);
        for i in 0..N {
            if i > 0 {
                many_unique.push(',');
            }
            write!(many_unique, "\"t{i}\":[\"r{i}\"]").unwrap();
        }
        many_unique.push_str(r#"},"additionalProperties":true}"#);

        for (label, schema, expected_names, expected_rules, expected_edges) in [
            ("one_by_many", one_by_many.as_str(), N + 1, 1, N),
            ("many_by_one", many_by_one.as_str(), N + 1, N, N),
            ("many_unique", many_unique.as_str(), 2 * N, N, N),
        ] {
            let t0 = std::time::Instant::now();
            let ir =
                Arc::new(crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap());
            let frontend_us = t0.elapsed().as_micros();
            let t0 = std::time::Instant::now();
            let compiled = StructuredPlan::compile(ir, StructuredLimits::default()).unwrap();
            let plan_us = t0.elapsed().as_micros();
            println!(
                "{label}: frontend={frontend_us}us plan={plan_us}us retained={}",
                compiled.retained_bytes
            );
            let object = object_plan_of(&compiled, compiled.root);
            let presence = object.dependent_required.as_ref().unwrap();
            assert_eq!(presence.name_count as usize, expected_names, "{label}");
            assert_eq!(presence.rules.len(), expected_rules, "{label}");
            assert_eq!(presence.required_ids.len(), expected_edges, "{label}");
        }

        // Order independence still holds at scale: reversed trigger declaration order and a
        // shuffled required-name array must produce the same canonical hash.
        let forward = crate::frontend::schema_to_ir(
            r#"{"dependentRequired":{"a":["p","q","r"],"b":["s"]}}"#,
            CompileOptions::default(),
        )
        .unwrap();
        let reordered = crate::frontend::schema_to_ir(
            r#"{"dependentRequired":{"b":["s"],"a":["r","p","q"]}}"#,
            CompileOptions::default(),
        )
        .unwrap();
        assert_eq!(forward.canonical_hash(), reordered.canonical_hash());
    }

    #[test]
    fn dependent_required_exact_plan_limit_succeeds_and_one_below_fails() {
        let source = Arc::new(
            crate::frontend::schema_to_ir(
                r#"{"type":"object","dependentRequired":{"a":["b","c"],"x":["y"]},"additionalProperties":true}"#,
                CompileOptions::default(),
            )
            .unwrap(),
        );
        let mut low = 1usize;
        let mut high = StructuredLimits::default().max_plan_bytes;
        while low < high {
            let mid = low + (high - low) / 2;
            let result = StructuredPlan::compile(
                source.clone(),
                StructuredLimits {
                    max_plan_bytes: mid,
                    ..StructuredLimits::default()
                },
            );
            if result.is_ok() {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        let exact = low;
        println!("dependentRequired exact plan budget bytes={exact}");
        assert!(StructuredPlan::compile(
            source.clone(),
            StructuredLimits {
                max_plan_bytes: exact,
                ..StructuredLimits::default()
            }
        )
        .is_ok());
        assert!(StructuredPlan::compile(
            source,
            StructuredLimits {
                max_plan_bytes: exact - 1,
                ..StructuredLimits::default()
            }
        )
        .is_err());
    }
}
