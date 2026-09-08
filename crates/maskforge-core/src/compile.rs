//! The production compiler: a validated arena `SchemaIR` is lowered directly to a byte-level
//! pattern and handed to the shared anchored-DFA builder. It consumes the typed nodes; it never
//! re-parses JSON and never transports a regex string across the FFI.

use crate::automaton::{
    build_from_regex, concatenate, multiple_of_engine, repeat, string_set, union, Combinator,
    ProductAutomaton, RefEngine, ShuffleAutomaton,
};
#[cfg(test)]
use crate::automaton::{decimal_multiple_of_regex, multiple_of_regex};
use crate::diagnostics::{Diagnostic, UnsupportedReason};
use crate::error::{CompileError, ErrorCode, Stage};
use crate::ir::{AdditionalPolicy, Charset, ItemsPolicy, Node, ScalarLit, SchemaIR};
use crate::primitives::NodeId;

/// Named compilation phases reported by the feature-gated profiling API.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum CompilePhase {
    /// Read schema and resource bytes.
    InputLoad,
    /// Parse JSON text.
    JsonParse,
    /// Index registered resources.
    ResourceIndex,
    /// Index static and dynamic anchors.
    AnchorIndex,
    /// Resolve references.
    ReferenceResolution,
    /// Lower the frontend representation to IR.
    IrLowering,
    /// Compute the shared route table.
    RouteAnalysis,
    /// Parse regular-expression syntax.
    RegexParse,
    /// Build Thompson NFAs.
    ThompsonNfa,
    /// Determinize NFAs.
    Determinization,
    /// Materialize the MaskForge DFA representation.
    DfaMaterialization,
    /// Construct byte equivalence classes.
    ByteClassConstruction,
    /// Analyze structured semantics and annotations.
    StructuredAnalysis,
    /// Build structured plan nodes.
    StructuredNodeBuild,
    /// Resolve plan SCCs.
    SccResolution,
    /// Freeze the immutable plan.
    PlanFreeze,
    /// Look up a vocabulary-neutral executable.
    SchemaCacheLookup,
    /// Look up or build a vocabulary trie.
    TrieLookup,
    /// Bind an executable to a vocabulary.
    VocabularyBind,
    /// Fill packed mask rows.
    PackedMaskFill,
}

impl CompilePhase {
    /// Number of named phases.
    pub const COUNT: usize = 20;
}

/// Nanoseconds indexed by [`CompilePhase`]. Unobserved phases remain zero.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PhaseDurations {
    values: [u64; CompilePhase::COUNT],
}

impl PhaseDurations {
    /// Duration recorded for `phase`.
    #[must_use]
    pub fn get(&self, phase: CompilePhase) -> u64 {
        self.values[phase as usize]
    }

    #[cfg(feature = "bench-internals")]
    fn set(&mut self, phase: CompilePhase, value: u64) {
        self.values[phase as usize] = value;
    }
}

/// Aggregate compiler work counts. Counters that a backend cannot expose remain zero.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CompileCounters {
    /// Input schema bytes.
    pub schema_bytes: u64,
    /// Registered resource bytes.
    pub resource_bytes: u64,
    /// Frozen IR nodes.
    pub ir_nodes: u64,
    /// Frozen IR child edges.
    pub ir_edges: u64,
    /// Nodes routed through regular execution.
    pub regular_nodes: u64,
    /// Nodes routed through structured or hybrid execution.
    pub structured_nodes: u64,
    /// Materialized DFA states.
    pub dfa_states: u64,
    /// Materialized byte classes.
    pub byte_classes: u64,
    /// Stored non-dead DFA edges.
    pub graph_edges: u64,
    /// Theoretical DFA transition cells.
    pub transition_cells: u64,
    /// Structured plan nodes.
    pub structured_plan_nodes: u64,
    /// SCC nodes examined.
    pub scc_nodes: u64,
}

/// Retained and temporary byte accounting for one compile.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RetainedBytes {
    /// Frozen IR bytes when supplied by the caller, if measured.
    pub ir: u64,
    /// Regular automaton bytes.
    pub automaton: u64,
    /// Structured plan bytes.
    pub plan: u64,
    /// Conservative peak temporary bytes.
    pub estimated_peak_temporary: u64,
}

/// Named summary profile for regular IR compilation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompileProfile {
    /// Per-phase wall durations.
    pub durations_ns: PhaseDurations,
    /// Backend work counters.
    pub counters: CompileCounters,
    /// Retained and temporary byte estimates.
    pub retained: RetainedBytes,
    /// Selected root route.
    pub route: crate::routing::ExecutionKind,
}

// Array/tuple wrapping duplicates its element text; an already-large child (e.g. a oneOf's own
// eliminated regex) can double past a safe re-compile size without either side alone looking big.
const MAX_ARRAY_BODY_BYTES: usize = 1 << 16;

fn require_regex_fits(len: usize) -> Result<(), CompileError> {
    if len > MAX_ARRAY_BODY_BYTES {
        return Err(CompileError::new(
            ErrorCode::Unsupported,
            Stage::L3,
            UnsupportedReason::StructuredBackendRequired.advisory(),
        ));
    }
    Ok(())
}

/// Compiles an IR into the anchored byte engine. An `Unsupported` occurrence fails the compile
/// with its diagnostic; a structured-backend schema fails fast with an actionable message instead of
/// a slow, cryptic elimination blow-up; otherwise the arena is lowered and built into a DFA.
pub fn compile_ir(ir: &SchemaIR) -> Result<RefEngine, CompileError> {
    if let Some(d) = ir.diagnostics().next() {
        return Err(diagnostic_error(ir, d));
    }
    if ir.requires_structured_backend() {
        return Err(CompileError::new(
            ErrorCode::Unsupported,
            Stage::L3,
            UnsupportedReason::StructuredBackendRequired.advisory(),
        ));
    }
    compile_node(ir, ir.root())
}

/// Compiles a regular IR and returns named phase, work, route, and memory measurements.
#[cfg(feature = "bench-internals")]
pub fn compile_ir_profiled(ir: &SchemaIR) -> Result<(RefEngine, CompileProfile), CompileError> {
    if let Some(d) = ir.diagnostics().next() {
        return Err(diagnostic_error(ir, d));
    }
    let mut durations_ns = PhaseDurations::default();
    let route_started = std::time::Instant::now();
    let route_table = ir.route_table();
    let route = route_table
        .get(ir.root())
        .map_or(crate::routing::ExecutionKind::Structured, |route| {
            route.kind
        });
    durations_ns.set(
        CompilePhase::RouteAnalysis,
        u64::try_from(route_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    );
    if route != crate::routing::ExecutionKind::Regular {
        return Err(CompileError::new(
            ErrorCode::Unsupported,
            Stage::L3,
            UnsupportedReason::StructuredBackendRequired.advisory(),
        ));
    }
    let started = std::time::Instant::now();
    let engine = compile_node(ir, ir.root())?;
    durations_ns.set(
        CompilePhase::DfaMaterialization,
        u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    );
    let regular_nodes = route_table
        .routes()
        .iter()
        .filter(|node| node.kind == crate::routing::ExecutionKind::Regular)
        .count();
    let counters = CompileCounters {
        ir_nodes: u64::try_from(ir.node_count()).unwrap_or(u64::MAX),
        regular_nodes: u64::try_from(regular_nodes).unwrap_or(u64::MAX),
        structured_nodes: u64::try_from(ir.node_count().saturating_sub(regular_nodes))
            .unwrap_or(u64::MAX),
        dfa_states: u64::try_from(engine.state_count()).unwrap_or(u64::MAX),
        byte_classes: u64::try_from(engine.byte_class_count()).unwrap_or(u64::MAX),
        graph_edges: u64::try_from(engine.live_transition_count()).unwrap_or(u64::MAX),
        transition_cells: u64::try_from(engine.transition_cells()).unwrap_or(u64::MAX),
        ..CompileCounters::default()
    };
    let retained = RetainedBytes {
        automaton: u64::try_from(engine.heap_bytes()).unwrap_or(u64::MAX),
        ..RetainedBytes::default()
    };
    Ok((
        engine,
        CompileProfile {
            durations_ns,
            counters,
            retained,
            route,
        },
    ))
}

/// Compatibility timing wrapper returning `(route_analysis_ns, materialization_ns)`.
pub fn compile_ir_timed(ir: &SchemaIR) -> Result<(RefEngine, u64, u64), CompileError> {
    if let Some(d) = ir.diagnostics().next() {
        return Err(diagnostic_error(ir, d));
    }
    let route_started = std::time::Instant::now();
    let route = ir.route_table().get(ir.root()).map(|route| route.kind);
    let route_analysis_ns = u64::try_from(route_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if route != Some(crate::routing::ExecutionKind::Regular) {
        return Err(CompileError::new(
            ErrorCode::Unsupported,
            Stage::L3,
            UnsupportedReason::StructuredBackendRequired.advisory(),
        ));
    }
    let started = std::time::Instant::now();
    let engine = compile_node(ir, ir.root())?;
    let materialization_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    Ok((engine, route_analysis_ns, materialization_ns))
}

/// Compiles the subtree rooted at `id` alone into its own anchored byte engine; used by the
/// structured backend to fast-path a sub-schema that has no `OpenObject` anywhere beneath it.
pub(crate) fn compile_node(ir: &SchemaIR, id: NodeId) -> Result<RefEngine, CompileError> {
    let node = ir
        .node(id)
        .ok_or_else(|| internal("node id out of range"))?;
    if !uses_direct_compilation(ir, node) {
        let body = lower_reachable(ir, id)?;
        return build_from_regex(&wrapped_regex(&body)?);
    }
    let body = compile_unwrapped(ir, id)?;
    surround_engine(&body, JSON_WS_RE, JSON_WS_RE)
}

fn uses_direct_compilation(ir: &SchemaIR, node: &Node) -> bool {
    match node {
        Node::Integer {
            multiple_of: Some(_),
            ..
        }
        | Node::Intersection { .. }
        | Node::ExactlyOne { .. }
        | Node::Union { .. } => true,
        Node::Enum { values } => ir
            .lits_at(*values)
            .is_some_and(|items| items.iter().all(|item| matches!(item, ScalarLit::Str(_)))),
        Node::Object {
            dependent,
            dependent_required,
            ..
        } => {
            ir.props_at(*dependent).is_some_and(<[_]>::is_empty)
                && ir
                    .dependent_required_at(*dependent_required)
                    .is_some_and(<[_]>::is_empty)
        }
        Node::Array {
            items: ItemsPolicy::Schema(_),
            unique_items: false,
            contains: None,
            ..
        }
        | Node::Tuple {
            unique_items: false,
            contains: None,
            ..
        } => true,
        _ => false,
    }
}

fn wrapped_regex(body: &str) -> Result<String, CompileError> {
    let capacity = JSON_WS_RE
        .len()
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(body.len()))
        .ok_or_else(|| compile_allocation("wrapped regex length"))?;
    let mut pattern = String::new();
    pattern
        .try_reserve_exact(capacity)
        .map_err(|_| compile_allocation("wrapped regex allocation"))?;
    pattern.push_str(JSON_WS_RE);
    pattern.push_str(body);
    pattern.push_str(JSON_WS_RE);
    Ok(pattern)
}

fn compile_unwrapped(ir: &SchemaIR, id: NodeId) -> Result<RefEngine, CompileError> {
    let node = ir
        .node(id)
        .ok_or_else(|| internal("node id out of range"))?;
    match node {
        Node::Integer {
            minimum,
            maximum,
            multiple_of: Some(modulus),
        } => compile_integer_multiple(*minimum, *maximum, *modulus),
        Node::Intersection { branches } => compile_product_node(ir, *branches, Combinator::All),
        Node::ExactlyOne { branches } => {
            compile_product_node(ir, *branches, Combinator::ExactlyOne)
        }
        Node::Union { branches } => compile_union_node(ir, *branches),
        Node::Enum { values }
            if ir.lits_at(*values).is_some_and(|literals| {
                literals
                    .iter()
                    .all(|item| matches!(item, ScalarLit::Str(_)))
            }) =>
        {
            compile_string_enum(ir, *values)
        }
        Node::Object {
            fields,
            required,
            dependent,
            dependent_required,
            ..
        } if ir
            .props_at(*dependent)
            .is_some_and(|items| items.is_empty())
            && ir
                .dependent_required_at(*dependent_required)
                .is_some_and(|items| items.is_empty()) =>
        {
            compile_object_node(ir, *fields, *required)
        }
        Node::Array {
            items: ItemsPolicy::Schema(item),
            min_items,
            max_items,
            unique_items: false,
            contains: None,
            ..
        } => compile_array_node(ir, *item, *min_items, *max_items),
        Node::Tuple {
            prefix,
            tail,
            min_items,
            max_items,
            unique_items: false,
            contains: None,
            ..
        } => compile_tuple_node(ir, *prefix, *tail, *min_items, *max_items),
        _ => build_from_regex(&lower_reachable(ir, id)?),
    }
}

fn compile_integer_multiple(
    minimum: Option<i64>,
    maximum: Option<i64>,
    modulus: u64,
) -> Result<RefEngine, CompileError> {
    let range = build_from_regex(&integer_regex(minimum, maximum))?;
    let multiple = multiple_of_engine(modulus)?;
    ProductAutomaton::build(&[&range, &multiple], Combinator::All)?.into_engine()
}

fn compile_string_enum(
    ir: &SchemaIR,
    values: crate::ir::LitSlice,
) -> Result<RefEngine, CompileError> {
    let literals = ir.lits_at(values).ok_or_else(|| internal("enum ref"))?;
    let mut strings = Vec::new();
    strings
        .try_reserve_exact(literals.len())
        .map_err(|_| compile_allocation("enum string reference allocation"))?;
    for literal in literals {
        if let ScalarLit::Str(value) = literal {
            strings.push(value.as_str());
        }
    }
    RefEngine::from_graph(&string_set(&strings)?)
}

fn compile_product_node(
    ir: &SchemaIR,
    branches: crate::ir::RefSlice,
    combinator: Combinator,
) -> Result<RefEngine, CompileError> {
    let ids = ir.refs_at(branches).ok_or_else(|| internal("branch ref"))?;
    let mut engines = Vec::new();
    engines
        .try_reserve_exact(ids.len())
        .map_err(|_| compile_allocation("product engine allocation"))?;
    for &id in ids {
        engines.push(compile_unwrapped(ir, id)?);
    }
    let refs = engine_refs(&engines)?;
    ProductAutomaton::build(&refs, combinator)?.into_engine()
}

fn compile_union_node(
    ir: &SchemaIR,
    branches: crate::ir::RefSlice,
) -> Result<RefEngine, CompileError> {
    let ids = ir.refs_at(branches).ok_or_else(|| internal("branch ref"))?;
    let mut engines = Vec::new();
    engines
        .try_reserve_exact(ids.len())
        .map_err(|_| compile_allocation("union engine allocation"))?;
    for &id in ids {
        engines.push(compile_unwrapped(ir, id)?);
    }
    let refs = engine_refs(&engines)?;
    RefEngine::from_graph(&union(&refs)?)
}

fn compile_object_node(
    ir: &SchemaIR,
    fields: crate::ir::PropSlice,
    required: crate::ir::BitSlice,
) -> Result<RefEngine, CompileError> {
    let properties = ir.props_at(fields).ok_or_else(|| internal("object ref"))?;
    let mut field_engines = Vec::new();
    let mut required_flags = Vec::new();
    field_engines
        .try_reserve_exact(properties.len())
        .map_err(|_| compile_allocation("field engine allocation"))?;
    required_flags
        .try_reserve_exact(properties.len())
        .map_err(|_| compile_allocation("field flag allocation"))?;
    for (index, (name, child_id)) in properties.iter().enumerate() {
        let name = ir.str_at(*name).ok_or_else(|| internal("field name"))?;
        let prefix = build_from_regex(&format!(
            "{}{}",
            escape_regex(&json_encode(name)),
            colon_sep()
        ))?;
        let child = compile_unwrapped(ir, *child_id)?;
        field_engines.push(concat_engine(&[&prefix, &child])?);
        let index = u32::try_from(index).map_err(|_| internal("field index overflow"))?;
        required_flags.push(ir.is_required(required, index));
    }
    let refs = engine_refs(&field_engines)?;
    let separator = build_from_regex(&comma_sep())?;
    let body = ShuffleAutomaton::build(&refs, &required_flags, &separator)?.into_engine()?;
    surround_engine(
        &body,
        &format!(r"\{{{JSON_WS_RE}"),
        &format!(r"{JSON_WS_RE}\}}"),
    )
}

fn compile_array_node(
    ir: &SchemaIR,
    item: NodeId,
    min_items: u32,
    max_items: Option<u32>,
) -> Result<RefEngine, CompileError> {
    let item = compile_unwrapped(ir, item)?;
    let comma = build_from_regex(&comma_sep())?;
    let comma_item = concat_engine(&[&comma, &item])?;
    let body = if max_items == Some(0) {
        build_from_regex("")?
    } else if min_items == 0 {
        let tail_max = max_items.map(|max| max.saturating_sub(1));
        let tail = RefEngine::from_graph(&repeat(&comma_item, 0, tail_max)?)?;
        let nonempty = concat_engine(&[&item, &tail])?;
        let empty = build_from_regex("")?;
        RefEngine::from_graph(&union(&[&empty, &nonempty])?)?
    } else {
        let tail = RefEngine::from_graph(&repeat(
            &comma_item,
            min_items - 1,
            max_items.map(|max| max - 1),
        )?)?;
        concat_engine(&[&item, &tail])?
    };
    surround_engine(
        &body,
        &format!(r"\[{JSON_WS_RE}"),
        &format!(r"{JSON_WS_RE}\]"),
    )
}

fn compile_tuple_node(
    ir: &SchemaIR,
    prefix: crate::ir::RefSlice,
    tail: Option<NodeId>,
    min_items: u32,
    max_items: Option<u32>,
) -> Result<RefEngine, CompileError> {
    let ids = ir.refs_at(prefix).ok_or_else(|| internal("tuple ref"))?;
    let mut prefix_engines = Vec::new();
    prefix_engines
        .try_reserve_exact(ids.len())
        .map_err(|_| compile_allocation("tuple engine allocation"))?;
    for &id in ids {
        prefix_engines.push(compile_unwrapped(ir, id)?);
    }
    let tail_engine = tail.map(|id| compile_unwrapped(ir, id)).transpose()?;
    let body = tuple_body_engine(&prefix_engines, tail_engine.as_ref(), min_items, max_items)?;
    surround_engine(
        &body,
        &format!(r"\[{JSON_WS_RE}"),
        &format!(r"{JSON_WS_RE}\]"),
    )
}

fn tuple_body_engine(
    prefix: &[RefEngine],
    tail: Option<&RefEngine>,
    min_items: u32,
    max_items: Option<u32>,
) -> Result<RefEngine, CompileError> {
    let prefix_len = u32::try_from(prefix.len()).map_err(|_| internal("tuple length"))?;
    let upper = max_items.unwrap_or(prefix_len);
    let finite_upper = if tail.is_some() {
        upper
    } else {
        upper.min(prefix_len)
    };
    let mut alternatives = Vec::new();
    let finite_count = finite_upper
        .checked_sub(min_items)
        .and_then(|count| count.checked_add(1))
        .unwrap_or(0);
    alternatives
        .try_reserve_exact(
            usize::try_from(finite_count)
                .map_err(|_| compile_allocation("tuple alternative count"))?
                .saturating_add(usize::from(max_items.is_none() && tail.is_some())),
        )
        .map_err(|_| compile_allocation("tuple alternative allocation"))?;
    for count in min_items..=finite_upper {
        alternatives.push(tuple_exact_engine(prefix, tail, count)?);
    }
    if max_items.is_none() {
        if let Some(tail) = tail {
            let base_count = min_items.max(prefix_len);
            let base = tuple_exact_engine(prefix, Some(tail), base_count)?;
            let comma = build_from_regex(&comma_sep())?;
            let comma_tail = concat_engine(&[&comma, tail])?;
            let rest = RefEngine::from_graph(&repeat(&comma_tail, 0, None)?)?;
            alternatives.push(concat_engine(&[&base, &rest])?);
        }
    }
    let refs = engine_refs(&alternatives)?;
    RefEngine::from_graph(&union(&refs)?)
}

fn tuple_exact_engine(
    prefix: &[RefEngine],
    tail: Option<&RefEngine>,
    count: u32,
) -> Result<RefEngine, CompileError> {
    if count == 0 {
        return build_from_regex("");
    }
    let comma = build_from_regex(&comma_sep())?;
    let count = usize::try_from(count).map_err(|_| internal("tuple count"))?;
    let mut sequence: Vec<&RefEngine> = Vec::new();
    sequence
        .try_reserve_exact(count.saturating_mul(2).saturating_sub(1))
        .map_err(|_| compile_allocation("tuple sequence allocation"))?;
    for index in 0..count {
        if index > 0 {
            sequence.push(&comma);
        }
        let item = prefix.get(index).or(tail).ok_or_else(|| {
            CompileError::new(
                ErrorCode::Malformed,
                Stage::L3,
                "tuple count exceeds closed tail",
            )
        })?;
        sequence.push(item);
    }
    concat_engine(&sequence)
}

fn concat_engine(engines: &[&RefEngine]) -> Result<RefEngine, CompileError> {
    RefEngine::from_graph(&concatenate(engines)?)
}

fn engine_refs(engines: &[RefEngine]) -> Result<Vec<&RefEngine>, CompileError> {
    let mut refs = Vec::new();
    refs.try_reserve_exact(engines.len())
        .map_err(|_| compile_allocation("engine reference allocation"))?;
    refs.extend(engines.iter());
    Ok(refs)
}

fn compile_allocation(what: &'static str) -> CompileError {
    CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L3,
        "compiler allocation limit",
    )
    .with_observed(what)
}

fn surround_engine(
    engine: &RefEngine,
    prefix: &str,
    suffix: &str,
) -> Result<RefEngine, CompileError> {
    let prefix = build_from_regex(prefix)?;
    let suffix = build_from_regex(suffix)?;
    concat_engine(&[&prefix, engine, &suffix])
}

pub(crate) fn number_syntax_engine() -> Result<RefEngine, CompileError> {
    build_from_regex(&format!("{JSON_WS_RE}{JSON_NUMBER_RE}{JSON_WS_RE}"))
}

/// Lowers only the nodes reachable from `root`, bottom-up (children precede parents): an
/// `OpenObject` elsewhere in the arena, unreachable from `root`, must not fail this compile.
fn lower_reachable(ir: &SchemaIR, root: NodeId) -> Result<String, CompileError> {
    let reachable = reachable_from(ir, root)?;
    let mut frag: Vec<String> = vec![String::new(); ir.node_count()];
    for (i, node) in ir.nodes().enumerate() {
        if reachable[i] {
            frag[i] = lower_node(ir, node, &frag)?;
        }
    }
    frag.into_iter()
        .nth(root.get() as usize)
        .ok_or_else(|| internal("root fragment missing"))
}

/// Every node id reachable from `root` via child edges (the same edges `lower_node` follows).
fn reachable_from(ir: &SchemaIR, root: NodeId) -> Result<Vec<bool>, CompileError> {
    let mut reached = vec![false; ir.node_count()];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        let idx = id.get() as usize;
        if reached.get(idx).copied().unwrap_or(true) {
            continue;
        }
        reached[idx] = true;
        let node = ir
            .node(id)
            .ok_or_else(|| internal("node id out of range"))?;
        stack.extend(node_children(ir, node)?);
    }
    Ok(reached)
}

/// The immediate child node ids of `node`, per its own variant's back-edges.
pub(crate) fn node_children(ir: &SchemaIR, node: &Node) -> Result<Vec<NodeId>, CompileError> {
    Ok(match node {
        Node::Array {
            items: ItemsPolicy::Schema(id),
            ..
        } => vec![*id],
        Node::Array {
            items: ItemsPolicy::AllowAny,
            ..
        } => Vec::new(),
        Node::Tuple { prefix, tail, .. } => {
            let mut ids = ir
                .refs_at(*prefix)
                .ok_or_else(|| internal("tuple ref"))?
                .to_vec();
            ids.extend(*tail);
            ids
        }
        Node::Object { fields, .. } => ir
            .props_at(*fields)
            .ok_or_else(|| internal("object ref"))?
            .iter()
            .map(|(_, c)| *c)
            .collect(),
        Node::OpenObject {
            known,
            patterns,
            additional,
            property_names,
            ..
        } => {
            let mut ids: Vec<NodeId> = ir
                .props_at(*known)
                .ok_or_else(|| internal("open object known ref"))?
                .iter()
                .map(|(_, c)| *c)
                .collect();
            ids.extend(
                ir.props_at(*patterns)
                    .ok_or_else(|| internal("open object pattern ref"))?
                    .iter()
                    .map(|(_, c)| *c),
            );
            if let AdditionalPolicy::Schema(id) = additional {
                ids.push(*id);
            }
            ids.extend(*property_names);
            ids
        }
        Node::Union { branches }
        | Node::Intersection { branches }
        | Node::ExactlyOne { branches } => ir
            .refs_at(*branches)
            .ok_or_else(|| internal("branch ref"))?
            .to_vec(),
        Node::Not { inner } => vec![*inner],
        Node::DynamicRef { initial_target, .. } => vec![*initial_target],
        Node::Unevaluated {
            scope, unevaluated, ..
        } => vec![*scope, *unevaluated],
        Node::Null
        | Node::Boolean
        | Node::Never
        | Node::StringConst { .. }
        | Node::StringPattern { .. }
        | Node::Integer { .. }
        | Node::Number { .. }
        | Node::LexicalNumber { .. }
        | Node::Enum { .. }
        | Node::Ref { .. }
        | Node::Unsupported { .. } => Vec::new(),
    })
}

/// The full JSON number grammar: optional sign, an integer part with no leading zeros, an optional
/// fractional part, and an optional exponent. A regular language, unlike a bounded real range.
const JSON_NUMBER_RE: &str = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?";

/// RFC 8259 insignificant whitespace, zero or more. The only whitespace fragment in this file.
const JSON_WS_RE: &str = r"[\x20\x09\x0A\x0D]*";

/// A JSON `:` with optional whitespace on both sides.
fn colon_sep() -> String {
    format!("{JSON_WS_RE}:{JSON_WS_RE}")
}

/// A JSON `,` with optional whitespace on both sides.
fn comma_sep() -> String {
    format!("{JSON_WS_RE},{JSON_WS_RE}")
}

fn lower_node(ir: &SchemaIR, node: &Node, frag: &[String]) -> Result<String, CompileError> {
    Ok(match node {
        Node::Null => "null".to_string(),
        Node::Boolean => "(?:true|false)".to_string(),
        Node::Never => r"[^\x00-\xff]".to_string(),
        Node::StringConst { value } => {
            let s = ir.str_at(*value).ok_or_else(|| internal("string ref"))?;
            escape_regex(&json_encode(s))
        }
        Node::StringPattern {
            regex,
            min_len,
            max_len,
            charset,
        } => {
            let base = ir.str_at(*regex).ok_or_else(|| internal("pattern ref"))?;
            let body = match charset {
                Charset::AsciiPrintableNoQuoteBackslash | Charset::Utf8CountedCodepoints => {
                    format!("(?:{base}){}", quantifier(*min_len, *max_len))
                }
                Charset::Utf8Any => format!("(?:{base})"),
            };
            format!("\"{body}\"")
        }
        Node::Integer {
            minimum,
            maximum,
            multiple_of,
        } => {
            if multiple_of.is_some() {
                return Err(internal("multipleOf must use direct graph compilation"));
            }
            integer_regex(*minimum, *maximum)
        }
        Node::Number {
            integer_only,
            minimum,
            maximum,
            multiple_of,
        } => {
            if *integer_only || minimum.is_some() || maximum.is_some() || multiple_of.is_some() {
                return Err(CompileError::new(
                    ErrorCode::Unsupported,
                    Stage::L3,
                    UnsupportedReason::StructuredBackendRequired.advisory(),
                ));
            }
            JSON_NUMBER_RE.to_string()
        }
        Node::LexicalNumber { regex } => ir
            .str_at(*regex)
            .ok_or_else(|| internal("lexical number pattern"))?
            .to_owned(),
        Node::Enum { values } => {
            let lits = ir.lits_at(*values).ok_or_else(|| internal("enum ref"))?;
            let mut alts: Vec<String> = lits.iter().map(scalar_regex).collect();
            // Longest-first so a shorter byte-prefix alternative cannot pre-empt a longer one.
            alts.sort_by_key(|a| std::cmp::Reverse(a.len()));
            format!("(?:{})", alts.join("|"))
        }
        Node::Array {
            unique_items: true, ..
        }
        | Node::Array {
            items: ItemsPolicy::AllowAny,
            ..
        }
        | Node::Array {
            contains: Some(_), ..
        } => {
            return Err(CompileError::new(
                ErrorCode::Unsupported,
                Stage::L3,
                UnsupportedReason::StructuredBackendRequired.advisory(),
            ));
        }
        Node::Array {
            items: ItemsPolicy::Schema(id),
            min_items,
            max_items,
            ..
        } => {
            let elem = child(frag, *id)?;
            let body = array_body(elem, *min_items, *max_items);
            require_regex_fits(body.len())?;
            format!("\\[{JSON_WS_RE}{body}{JSON_WS_RE}\\]")
        }
        Node::Tuple {
            unique_items: true, ..
        }
        | Node::Tuple {
            contains: Some(_), ..
        } => {
            return Err(CompileError::new(
                ErrorCode::Unsupported,
                Stage::L3,
                UnsupportedReason::StructuredBackendRequired.advisory(),
            ));
        }
        Node::Tuple {
            prefix,
            tail,
            min_items,
            max_items,
            ..
        } => {
            let ids = ir.refs_at(*prefix).ok_or_else(|| internal("tuple ref"))?;
            let mut elems = Vec::with_capacity(ids.len());
            for id in ids {
                elems.push(child(frag, *id)?);
            }
            let tail = match tail {
                Some(t) => Some(child(frag, *t)?),
                None => None,
            };
            let body = tuple_body(&elems, tail, *min_items, *max_items);
            require_regex_fits(body.len())?;
            format!("\\[{JSON_WS_RE}{body}{JSON_WS_RE}\\]")
        }
        Node::Object { dependent: dep, .. } if ir.props_at(*dep).is_some_and(|d| !d.is_empty()) => {
            return Err(CompileError::new(
                ErrorCode::Unsupported,
                Stage::L3,
                UnsupportedReason::StructuredBackendRequired.advisory(),
            ));
        }
        Node::Object {
            dependent_required, ..
        } if ir
            .dependent_required_at(*dependent_required)
            .is_some_and(|pairs| !pairs.is_empty()) =>
        {
            return Err(CompileError::new(
                ErrorCode::Unsupported,
                Stage::L3,
                UnsupportedReason::StructuredBackendRequired.advisory(),
            ));
        }
        Node::Object {
            fields, required, ..
        } => {
            let _ = (fields, required, frag);
            return Err(internal("object must use direct graph compilation"));
        }
        Node::OpenObject { .. } => {
            return Err(CompileError::new(
                ErrorCode::Unsupported,
                Stage::L3,
                UnsupportedReason::StructuredBackendRequired.advisory(),
            ));
        }
        Node::Union { branches } => {
            let _ = branches;
            return Err(internal("union must use direct graph compilation"));
        }
        Node::Intersection { branches } => {
            let _ = branches;
            return Err(internal("intersection must use direct graph compilation"));
        }
        Node::ExactlyOne { branches } => {
            let _ = branches;
            return Err(internal("oneOf must use direct graph compilation"));
        }
        Node::Not { .. }
        | Node::Ref { .. }
        | Node::DynamicRef { .. }
        | Node::Unevaluated { .. } => {
            return Err(CompileError::new(
                ErrorCode::Unsupported,
                Stage::L3,
                UnsupportedReason::StructuredBackendRequired.advisory(),
            ));
        }
        Node::Unsupported { .. } => return Err(internal("unsupported node reached the compiler")),
    })
}

/// Intersects two independent regex fragments via the product automaton (language intersection).
#[cfg(test)]
fn intersect_regex(a: &str, b: &str) -> Result<String, CompileError> {
    let engines = [build_from_regex(a)?, build_from_regex(b)?];
    let refs: Vec<&RefEngine> = engines.iter().collect();
    ProductAutomaton::build(&refs, Combinator::All)?.to_regex()
}

fn child(frag: &[String], id: NodeId) -> Result<&str, CompileError> {
    frag.get(id.get() as usize)
        .map(String::as_str)
        .ok_or_else(|| internal("child fragment missing"))
}

fn scalar_regex(lit: &ScalarLit) -> String {
    match lit {
        ScalarLit::Null => "null".to_string(),
        ScalarLit::Bool(true) => "true".to_string(),
        ScalarLit::Bool(false) => "false".to_string(),
        ScalarLit::Int(i) => i.to_string(),
        ScalarLit::Str(s) => escape_regex(&json_encode(s)),
        // A composite literal is already its canonical JSON bytes; match them structurally (own
        // brackets/colons/commas get whitespace, string content passes through untouched).
        ScalarLit::Json(s) => json_literal_regex(s),
    }
}

/// Rebuilds a canonical-JSON literal with `JSON_WS_RE` around structural bytes outside strings;
/// tracks string/escape state so a `:` or `,` inside a string is left alone.
fn json_literal_regex(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    let mut in_string = false;
    let mut escaped = false;
    for c in s.chars() {
        if in_string {
            push_escaped(&mut out, c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                push_escaped(&mut out, c);
            }
            '{' | '}' | '[' | ']' | ':' | ',' => {
                out.push_str(JSON_WS_RE);
                push_escaped(&mut out, c);
                out.push_str(JSON_WS_RE);
            }
            c => push_escaped(&mut out, c),
        }
    }
    out
}

/// The comma-separated repetition body for `[min,max]` array cardinality.
fn array_body(element: &str, min: u32, max: Option<u32>) -> String {
    let e = format!("(?:{element})");
    let c = comma_sep();
    match max {
        None if min == 0 => format!("(?:{e}(?:{c}{e})*)?"),
        None => format!("{e}(?:{c}{e}){{{},}}", min - 1),
        Some(0) => String::new(),
        Some(max) if min == 0 => format!("(?:{e}(?:{c}{e}){{0,{}}})?", max - 1),
        Some(max) => format!("{e}(?:{c}{e}){{{},{}}}", min - 1, max - 1),
    }
}

/// The comma-joined tuple body: `elems` match the first positions, `tail` the rest (none = closed);
/// present-item counts down to `min` are branches, the unbounded tail folds into one repetition.
fn tuple_body(elems: &[&str], tail: Option<&str>, min: u32, max: Option<u32>) -> String {
    // Prefix length is bounded by the IR ref-slice caps enforced at build/decode, so it fits u32.
    let n = u32::try_from(elems.len()).expect("tuple prefix length fits u32");
    let at = |i: u32| -> String {
        let e = if i < n {
            elems[i as usize]
        } else {
            tail.unwrap_or("")
        };
        format!("(?:{e})")
    };
    let c = comma_sep();
    // Comma-joins positions [0, k), which is an array body of exactly k items.
    let join = |k: u32| -> String { (0..k).map(&at).collect::<Vec<_>>().join(&c) };

    let mut alts: Vec<String> = Vec::new();
    match (tail, max) {
        // Closed tuple: at most `n` items; enumerate every valid length in [min, n].
        (None, _) => {
            let hi = max.map_or(n, |m| m.min(n));
            for k in min..=hi {
                alts.push(join(k));
            }
        }
        // Open tail, bounded max: exact lengths below the prefix boundary, then the full prefix
        // followed by the tail repeated a bounded number of times. O(prefix length) branches, never
        // proportional to `max`, so a large `maxItems` cannot inflate the pattern string.
        (Some(_), Some(hi)) if hi < n => {
            // max sits below the prefix: every valid length stays within the prefix positions.
            for k in min..=hi {
                alts.push(join(k));
            }
        }
        (Some(t), Some(hi)) => {
            for k in min..n {
                alts.push(join(k));
            }
            let base = join(n);
            let tail_lo = min.saturating_sub(n);
            let tail_hi = hi - n;
            let rep = if tail_hi == 0 {
                String::new()
            } else {
                format!("(?:{c}(?:{t})){{{tail_lo},{tail_hi}}}")
            };
            alts.push(format!("{base}{rep}"));
        }
        // Open tail, unbounded max: enumerate lengths up to the prefix boundary, then one branch
        // holding the full prefix followed by a repeated tail for every longer length.
        (Some(t), None) => {
            let hi = min.max(n);
            for k in min..hi {
                alts.push(join(k));
            }
            let base = join(n);
            let tail_lo = min.saturating_sub(n);
            let rep = if tail_lo == 0 {
                format!("(?:{c}(?:{t}))*")
            } else {
                format!("(?:{c}(?:{t})){{{tail_lo},}}")
            };
            alts.push(format!("{base}{rep}"));
        }
    }
    // Longest branch first so a shorter length cannot pre-empt a longer one under leftmost-first.
    alts.sort_by_key(|a| std::cmp::Reverse(a.len()));
    format!("(?:{})", alts.join("|"))
}

fn quantifier(min: Option<u32>, max: Option<u32>) -> String {
    match (min, max) {
        (Some(a), Some(b)) => format!("{{{a},{b}}}"),
        (Some(a), None) => format!("{{{a},}}"),
        (None, Some(b)) => format!("{{0,{b}}}"),
        (None, None) => "*".to_string(),
    }
}

/// Encodes a string as a JSON string body with RFC 8259 escapes (no surrounding quotes).
fn json_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

const REGEX_META: &[char] = &[
    '\\', '.', '+', '*', '?', '(', ')', '[', ']', '{', '}', '^', '$', '|',
];

/// Pushes `c` onto `out`, backslash-escaping it first if it is a regex metacharacter.
fn push_escaped(out: &mut String, c: char) {
    if REGEX_META.contains(&c) {
        out.push('\\');
    }
    out.push(c);
}

/// Escapes regex metacharacters so a literal is matched byte-for-byte.
fn escape_regex(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        push_escaped(&mut out, c);
    }
    out.push('"');
    out
}

fn diagnostic_error(ir: &SchemaIR, d: &Diagnostic) -> CompileError {
    let mut e = CompileError::new(ErrorCode::Unsupported, Stage::L2, d.reason.advisory());
    if let Some(k) = ir.str_at(d.keyword) {
        e = e.with_keyword(k.to_string());
    }
    if let Some(p) = ir.str_at(d.json_pointer) {
        e = e.with_pointer(p.to_string());
    }
    e
}

fn internal(what: &'static str) -> CompileError {
    CompileError::new(ErrorCode::InternalLimitExceeded, Stage::L2, what)
}

// --- exact decimal-range regex for signed integers ---------------------------------------------

/// A byte regex matching exactly the JSON integers `n` with `lo <= n <= hi` (a `None` bound is
/// unbounded). Never enumerates: it emits digit-range patterns whose size is proportional to the
/// number of digits, not the range width. Brute-force verified; shared with the reference lowering.
pub(crate) fn integer_regex(lo: Option<i64>, hi: Option<i64>) -> String {
    let lo = lo.map(i128::from);
    let hi = hi.map(i128::from);
    let mut alts: Vec<String> = Vec::new();

    // Negative side: n in [lo, min(hi, -1)], mapped to m = -n which is positive.
    let neg_hi = hi.map_or(-1, |h| h.min(-1));
    let neg_present = neg_hi <= -1 && lo.is_none_or(|l| l <= neg_hi);
    if neg_present {
        let a = to_u64(-neg_hi);
        match lo {
            Some(l) => alts.push(format!("-{}", pos_range(a, to_u64(-l)))),
            None => alts.push(format!("-{}", pos_ge(a))),
        }
    }

    if lo.is_none_or(|l| l <= 0) && hi.is_none_or(|h| h >= 0) {
        alts.push("0".to_string());
    }

    // Positive side: n in [max(lo, 1), hi].
    let pos_lo = lo.map_or(1, |l| l.max(1));
    if hi.is_none_or(|h| pos_lo <= h) {
        match hi {
            Some(h) => alts.push(pos_range(to_u64(pos_lo), to_u64(h))),
            None => alts.push(pos_ge(to_u64(pos_lo))),
        }
    }

    format!("(?:{})", alts.join("|"))
}

/// Whether the fractional digits after the integer part are unconstrained, forced to all-zero
/// (an exact integer value), or forced to contain a nonzero digit (strictly past the integer).
#[derive(Copy, Clone, PartialEq, Eq)]
#[cfg(test)]
enum FracRule {
    Any,
    Zero,
    NonZero,
}

/// The fraction rule that keeps a value at exactly the boundary magnitude `v` satisfying `>= v`
/// (inclusive) or `> v` (exclusive); `None` means no fraction can satisfy it (the whole magnitude
/// is excluded). Increasing digits after a negative integer part decreases the value, so the rule
/// flips by sign.
#[cfg(test)]
fn frac_rule_for_min(v: i64, exclusive: bool) -> Option<FracRule> {
    if v < 0 {
        if exclusive {
            None
        } else {
            Some(FracRule::Zero)
        }
    } else if exclusive {
        Some(FracRule::NonZero)
    } else {
        Some(FracRule::Any)
    }
}

/// Mirror of `frac_rule_for_min` for a `<= v` (inclusive) or `< v` (exclusive) upper bound.
#[cfg(test)]
fn frac_rule_for_max(v: i64, exclusive: bool) -> Option<FracRule> {
    if v >= 0 {
        if exclusive {
            None
        } else {
            Some(FracRule::Zero)
        }
    } else if exclusive {
        Some(FracRule::NonZero)
    } else {
        Some(FracRule::Any)
    }
}

/// Intersects two fraction rules for a magnitude that is both the minimum and the maximum bound.
#[cfg(test)]
fn combine_frac_rules(a: FracRule, b: FracRule) -> Option<FracRule> {
    match (a, b) {
        (FracRule::Zero, FracRule::NonZero) | (FracRule::NonZero, FracRule::Zero) => None,
        (FracRule::Zero, _) | (_, FracRule::Zero) => Some(FracRule::Zero),
        (FracRule::NonZero, _) | (_, FracRule::NonZero) => Some(FracRule::NonZero),
        (FracRule::Any, FracRule::Any) => Some(FracRule::Any),
    }
}

#[cfg(test)]
fn frac_pattern(rule: FracRule) -> &'static str {
    match rule {
        FracRule::Any => r"(?:\.[0-9]+)?",
        FracRule::Zero => r"(?:\.0+)?",
        FracRule::NonZero => r"\.[0-9]*[1-9][0-9]*",
    }
}

/// Converts integer-valued number bounds to inclusive integer bounds (for an integer `multipleOf`,
/// whose multiples are integers): an exclusive edge shifts inward by one.
#[cfg(test)]
pub(crate) fn number_bounds_to_integer(
    minimum: Option<(i64, bool)>,
    maximum: Option<(i64, bool)>,
) -> (Option<i64>, Option<i64>) {
    let lo = minimum.map(|(v, excl)| if excl { v.saturating_add(1) } else { v });
    let hi = maximum.map(|(v, excl)| if excl { v.saturating_sub(1) } else { v });
    (lo, hi)
}

/// The full `type:number` + `multipleOf` regex (bounds intersected with divisibility). An integer
/// divisor reuses the well-formed integer path with a `.0*` tail; a fractional divisor intersects the
/// numeric range with the decimal-multiple DFA. Errs (limit) when a divisor's DFA is too large to
/// combine, so lowering can validate buildability up front rather than fail late.
#[cfg(test)]
pub(crate) fn number_multiple_of_regex(
    minimum: Option<(i64, bool)>,
    maximum: Option<(i64, bool)>,
    coef: u64,
    exp: u32,
) -> Result<String, CompileError> {
    if exp == 0 {
        let (lo, hi) = number_bounds_to_integer(minimum, maximum);
        let ints = intersect_regex(&integer_regex(lo, hi), &multiple_of_regex(coef)?)?;
        return Ok(format!("(?:{ints})(?:\\.0+)?"));
    }
    let range = number_range_regex(minimum, maximum);
    intersect_regex(&range, &decimal_multiple_of_regex(coef, exp)?)
}

/// A byte regex matching JSON numbers (sign, integer part, optional fraction, no exponent) with an
/// integer-valued inclusive/exclusive minimum and/or maximum. Every value satisfying such a bound
/// has a non-exponent spelling, so dropping exponent forms loses no language a generator needs.
/// `None`/`None` (fully unbounded) returns the full numeric grammar, exponent included.
#[cfg(test)]
pub(crate) fn number_range_regex(
    minimum: Option<(i64, bool)>,
    maximum: Option<(i64, bool)>,
) -> String {
    if minimum.is_none() && maximum.is_none() {
        return JSON_NUMBER_RE.to_string();
    }
    let mut alts: Vec<String> = Vec::new();
    let zero_reachable = minimum
        .is_none_or(|(value, exclusive)| value < 0 || value == 0 && !exclusive)
        && maximum.is_none_or(|(value, exclusive)| value > 0 || value == 0 && !exclusive);
    if zero_reachable {
        alts.push(r"-0(?:\.0+)?".to_string());
    }
    let interior_lo = minimum.map(|(v, _)| v.checked_add(1));
    let interior_hi = maximum.map(|(v, _)| v.checked_sub(1));
    let interior_ok = match (interior_lo, interior_hi) {
        (Some(None), _) | (_, Some(None)) => false,
        (Some(Some(lo)), Some(Some(hi))) => lo <= hi,
        _ => true,
    };
    if interior_ok {
        let lo = interior_lo.flatten();
        let hi = interior_hi.flatten();
        alts.push(format!(
            "{}{}",
            integer_regex(lo, hi),
            frac_pattern(FracRule::Any)
        ));
    }
    let mut push_boundary = |v: i64, rule: Option<FracRule>| {
        if let Some(rule) = rule {
            alts.push(format!(
                "{}{}",
                integer_regex(Some(v), Some(v)),
                frac_pattern(rule)
            ));
        }
    };
    match (minimum, maximum) {
        (Some((mv, mex)), Some((xv, xex))) if mv == xv => {
            let rule = frac_rule_for_min(mv, mex).zip(frac_rule_for_max(xv, xex));
            if let Some((a, b)) = rule {
                push_boundary(mv, combine_frac_rules(a, b));
            }
        }
        _ => {
            if let Some((mv, mex)) = minimum {
                push_boundary(mv, frac_rule_for_min(mv, mex));
            }
            if let Some((xv, xex)) = maximum {
                push_boundary(xv, frac_rule_for_max(xv, xex));
            }
        }
    }
    // `integer_regex` never emits "-0": its negative side starts at magnitude 1, and its zero
    // alternative is unsigned. Values in (-1, 0), e.g. -0.5, need this hand-added slot.
    let neg_zero_reachable =
        minimum.is_none_or(|(v, _)| v <= -1) && maximum.is_none_or(|(v, _)| v >= 0);
    if neg_zero_reachable {
        alts.push(format!("-0{}", frac_pattern(FracRule::NonZero)));
    }
    format!("(?:{})", alts.join("|"))
}

/// A bound's magnitude is at most `i64::MIN` negated, which fits `u64`.
fn to_u64(v: i128) -> u64 {
    u64::try_from(v).expect("integer bound magnitude fits u64")
}

fn pow10(k: u32) -> u64 {
    let mut v = 1u64;
    for _ in 0..k {
        v *= 10;
    }
    v
}

fn num_digits(n: u64) -> u32 {
    let mut d = 1;
    let mut v = n;
    while v >= 10 {
        v /= 10;
        d += 1;
    }
    d
}

/// Renders fractional digits (values `0..9`) as a literal string.
#[cfg(test)]
fn frac_str(a: &[u8]) -> String {
    a.iter().map(|d| (b'0' + d) as char).collect()
}

/// Drops trailing zeros: `0.a` equals `0.<stripped>` numerically.
#[cfg(test)]
fn strip_trailing_zeros(a: &[u8]) -> &[u8] {
    let mut end = a.len();
    while end > 0 && a[end - 1] == 0 {
        end -= 1;
    }
    &a[..end]
}

/// Regex for a non-empty fractional digit run `F` with `0.F >= 0.a`. Empty `F` (value 0) satisfies
/// this only when `a` is all zeros; the caller decides whether to make the fraction optional.
#[cfg(test)]
fn frac_ge_body(a: &[u8]) -> String {
    let a = strip_trailing_zeros(a);
    if a.is_empty() {
        return "[0-9]+".to_string();
    }
    let mut alts: Vec<String> = Vec::new();
    for i in 0..a.len() {
        if a[i] < 9 {
            alts.push(format!("{}[{}-9][0-9]*", frac_str(&a[..i]), a[i] + 1));
        }
    }
    alts.push(format!("{}[0-9]*", frac_str(a)));
    format!("(?:{})", alts.join("|"))
}

/// Regex for a non-empty fractional digit run `F` with `0.F <= 0.a`.
#[cfg(test)]
fn frac_le_body(a: &[u8]) -> String {
    let a = strip_trailing_zeros(a);
    if a.is_empty() {
        return "0+".to_string();
    }
    let mut alts: Vec<String> = Vec::new();
    for i in 0..a.len() {
        if a[i] > 0 {
            alts.push(format!("{}[0-{}][0-9]*", frac_str(&a[..i]), a[i] - 1));
        }
    }
    for i in 1..a.len() {
        alts.push(frac_str(&a[..i]));
    }
    alts.push(format!("{}0*", frac_str(a)));
    format!("(?:{})", alts.join("|"))
}

/// The fractional suffix (including the leading `.`, or empty) for `x`'s integer part equal to a
/// lower bound whose fractional digits are `lf`: `>= lf`, with the fraction optional iff `lf` is 0.
#[cfg(test)]
fn lower_frac_suffix(lf: &[u8]) -> String {
    if strip_trailing_zeros(lf).is_empty() {
        return format!("(?:\\.{})?", frac_ge_body(lf));
    }
    format!("\\.{}", frac_ge_body(lf))
}

/// The fractional suffix for `x`'s integer part equal to an upper bound whose fractional digits are
/// `hf`: `<= hf`, always optional (no fraction means `.0`, which is `<=` any bound).
#[cfg(test)]
fn upper_frac_suffix(hf: &[u8]) -> String {
    format!("(?:\\.{})?", frac_le_body(hf))
}

/// Non-negative decimals `x` with `lo <= x <= hi` (`hi = None` is `+inf`); `lo`/`hi` are
/// `(integer_part, fractional_digits)`. Interior integer parts take any fraction; the two boundary
/// integer parts constrain the fraction via `frac_ge`/`frac_le`, intersected when they coincide.
#[cfg(test)]
fn nonneg_decimal_range(
    lo: (u64, &[u8]),
    hi: Option<(u64, &[u8])>,
) -> Result<String, CompileError> {
    let (li, lf) = lo;
    let li = li as i64;
    let mut alts: Vec<String> = Vec::new();
    match hi {
        None => {
            alts.push(format!(
                "{}{}",
                integer_regex(Some(li), None),
                lower_frac_suffix(lf)
            ));
            alts.push(format!(
                "{}(?:\\.[0-9]+)?",
                integer_regex(Some(li + 1), None)
            ));
        }
        Some((hi_i, hf)) => {
            let hi_i = hi_i as i64;
            if li == hi_i {
                // `intersect_regex` returns an ungrouped alternation, so wrap it before composing.
                let both = intersect_regex(&frac_ge_body(lf), &frac_le_body(hf))?;
                let frac = if strip_trailing_zeros(lf).is_empty() {
                    format!("(?:\\.(?:{both}))?")
                } else {
                    format!("\\.(?:{both})")
                };
                alts.push(format!("{}{}", integer_regex(Some(li), Some(li)), frac));
            } else {
                alts.push(format!(
                    "{}{}",
                    integer_regex(Some(li), Some(li)),
                    lower_frac_suffix(lf)
                ));
                alts.push(format!(
                    "{}{}",
                    integer_regex(Some(hi_i), Some(hi_i)),
                    upper_frac_suffix(hf)
                ));
                if hi_i - li >= 2 {
                    alts.push(format!(
                        "{}(?:\\.[0-9]+)?",
                        integer_regex(Some(li + 1), Some(hi_i - 1))
                    ));
                }
            }
        }
    }
    Ok(format!("(?:{})", alts.join("|")))
}

/// Splits a decimal bound into `(negative, integer_part, fractional_digits)`.
#[cfg(test)]
fn to_decimal(bound: (i64, u32, bool)) -> (bool, u64, Vec<u8>) {
    let (value, scale, _) = bound;
    let mag = value.unsigned_abs();
    let div = 10u64.pow(scale);
    let frac_val = mag % div;
    let frac = format!("{frac_val:0width$}", width = scale as usize)
        .bytes()
        .map(|b| b - b'0')
        .collect();
    (value < 0, mag / div, frac)
}

/// Every negative number (well-formed): `-` followed by a positive magnitude.
#[cfg(test)]
fn all_negatives() -> &'static str {
    r"-(?:[1-9][0-9]*(?:\.[0-9]+)?|0\.[0-9]*[1-9][0-9]*)"
}

#[cfg(test)]
fn unsupported_decimal() -> CompileError {
    CompileError::new(
        ErrorCode::Unsupported,
        Stage::L3,
        "this decimal bound shape is not yet supported",
    )
}

/// Inclusive decimal range as a regex, for the supported sign cases: a non-negative lower bound
/// (upper `>= 0` or none), or an upper bound `>= 0` with no lower bound (all negatives plus
/// `[0, max]`). Negative bounds, ranges crossing zero, and exclusive bounds return an error.
#[cfg(test)]
fn decimal_range_regex(
    minimum: Option<(i64, u32, bool)>,
    maximum: Option<(i64, u32, bool)>,
) -> Result<String, CompileError> {
    if minimum.is_some_and(|(_, _, e)| e) || maximum.is_some_and(|(_, _, e)| e) {
        return Err(unsupported_decimal());
    }
    let lo = minimum.map(to_decimal);
    let hi = maximum.map(to_decimal);
    match (&lo, &hi) {
        (Some((false, li, lf)), _) if hi.as_ref().is_none_or(|(hn, ..)| !hn) => {
            let hi_ref = hi.as_ref().map(|(_, hi_i, hf)| (*hi_i, hf.as_slice()));
            nonneg_decimal_range((*li, lf), hi_ref)
        }
        (None, Some((false, hi_i, hf))) => {
            let pos = nonneg_decimal_range((0, &[]), Some((*hi_i, hf.as_slice())))?;
            Ok(format!("(?:{}|{})", all_negatives(), pos))
        }
        _ => Err(unsupported_decimal()),
    }
}

/// The `type:number` regex: integer-valued bounds (all `scale == 0`) reuse the integer path and
/// permit `multipleOf`; a fractional bound routes to the decimal range and forbids `multipleOf`.
#[cfg(test)]
pub(crate) fn number_node_regex(
    minimum: Option<(i64, u32, bool)>,
    maximum: Option<(i64, u32, bool)>,
    multiple_of: Option<(u64, u32)>,
) -> Result<String, CompileError> {
    let as_int = |b: Option<(i64, u32, bool)>| match b {
        None => Some(None),
        Some((v, 0, e)) => Some(Some((v, e))),
        Some(_) => None,
    };
    if let (Some(lo), Some(hi)) = (as_int(minimum), as_int(maximum)) {
        return match multiple_of {
            None => Ok(number_range_regex(lo, hi)),
            Some((coef, exp)) => number_multiple_of_regex(lo, hi, coef, exp),
        };
    }
    if multiple_of.is_some() {
        return Err(unsupported_decimal());
    }
    decimal_range_regex(minimum, maximum)
}

/// Positive integers `>= a` (no leading zeros). Longer (more-digit) alternatives come first, so a
/// shorter prefix cannot pre-empt a longer match under leftmost-first alternation.
fn pos_ge(a: u64) -> String {
    let d = num_digits(a);
    let same = flrange(a, pow10(d) - 1, d);
    let more = if d == 1 {
        "[1-9][0-9]+".to_string()
    } else {
        format!("[1-9][0-9]{{{d},}}")
    };
    format!("(?:{more}|{same})")
}

/// Positive integers in `[a, b]` (no leading zeros), `1 <= a <= b`. Digit-length bands are emitted
/// longest-first to avoid leftmost-first prefix pre-emption.
fn pos_range(a: u64, b: u64) -> String {
    let (da, db) = (num_digits(a), num_digits(b));
    if da == db {
        return flrange(a, b, da);
    }
    let mut parts = vec![flrange(a, pow10(da) - 1, da)];
    for d in (da + 1)..db {
        parts.push(all_digits(d));
    }
    parts.push(flrange(pow10(db - 1), b, db));
    parts.reverse(); // longest digit-length band first
    alt(parts)
}

/// Every `d`-digit positive integer (no leading zeros).
fn all_digits(d: u32) -> String {
    if d == 1 {
        "[1-9]".to_string()
    } else {
        format!("[1-9][0-9]{{{}}}", d - 1)
    }
}

/// Fixed-length decimal strings (leading zeros allowed) in `[lo, hi]`, `len` digits.
fn flrange(lo: u64, hi: u64, len: u32) -> String {
    // A full band collapses to `[0-9]{len}`; without this, full sub-ranges expand digit by digit
    // and the pattern blows up to megabytes for wide bounds.
    if lo == 0 && hi == pow10(len) - 1 {
        return any_digits(len);
    }
    if len == 1 {
        return digit_class(lo, hi);
    }
    let p = pow10(len - 1);
    let (ld, hd) = (lo / p, hi / p);
    let (lr, hr) = (lo % p, hi % p);
    if ld == hd {
        return format!("{}{}", digit(ld), flrange(lr, hr, len - 1));
    }
    let mut parts = vec![format!("{}{}", digit(ld), flrange(lr, p - 1, len - 1))];
    if hd >= ld + 2 {
        parts.push(format!(
            "{}{}",
            digit_class(ld + 1, hd - 1),
            any_digits(len - 1)
        ));
    }
    parts.push(format!("{}{}", digit(hd), flrange(0, hr, len - 1)));
    alt(parts)
}

fn digit(d: u64) -> char {
    // Every caller passes a single decimal digit, so this conversion cannot fail.
    char::from(b'0' + u8::try_from(d).expect("a decimal digit is < 10"))
}

fn digit_class(lo: u64, hi: u64) -> String {
    if lo == hi {
        digit(lo).to_string()
    } else {
        format!("[{}-{}]", digit(lo), digit(hi))
    }
}

fn any_digits(k: u32) -> String {
    match k {
        0 => String::new(),
        1 => "[0-9]".to_string(),
        _ => format!("[0-9]{{{k}}}"),
    }
}

fn alt(parts: Vec<String>) -> String {
    if parts.len() == 1 {
        parts.into_iter().next().unwrap_or_default()
    } else {
        format!("(?:{})", parts.join("|"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::build_from_regex;
    use crate::ir::{Builder, CompileOptions};

    #[test]
    fn compile_ir_timed_fails_fast_on_a_structured_backend_schema_like_compile_ir() {
        let ir = crate::frontend::schema_to_ir(
            r#"{"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":true}"#,
            CompileOptions::default(),
        )
        .unwrap();
        assert!(ir.requires_structured_backend());
        assert!(compile_ir(&ir).is_err());
        assert!(compile_ir_timed(&ir).is_err());
    }

    // 0.f vs 0.a as rationals: pad both to equal length with trailing zeros, compare as integers.
    fn frac_cmp(f: &[u8], a: &[u8]) -> std::cmp::Ordering {
        let n = f.len().max(a.len());
        let val = |d: &[u8]| -> u128 {
            let mut v = 0u128;
            for i in 0..n {
                v = v * 10 + u128::from(*d.get(i).unwrap_or(&0));
            }
            v
        };
        val(f).cmp(&val(a))
    }

    fn digits(s: &str) -> Vec<u8> {
        s.bytes().map(|b| b - b'0').collect()
    }

    #[test]
    fn frac_ge_le_bodies_agree_with_the_exact_rational_oracle() {
        let anchored = |body: &str| build_from_regex(&format!("^{body}$")).unwrap();
        for a_len in 1..=3usize {
            for a_code in 0..10u32.pow(a_len as u32) {
                let a = digits(&format!("{a_code:0a_len$}"));
                let ge = anchored(&frac_ge_body(&a));
                let le = anchored(&frac_le_body(&a));
                for f_len in 1..=4usize {
                    for f_code in 0..10u32.pow(f_len as u32) {
                        let f_str = format!("{f_code:0f_len$}");
                        let f = digits(&f_str);
                        let ord = frac_cmp(&f, &a);
                        assert_eq!(
                            ge.accepts(f_str.as_bytes()),
                            ord != std::cmp::Ordering::Less,
                            "ge a=0.{} f=0.{f_str}",
                            frac_str(&a)
                        );
                        assert_eq!(
                            le.accepts(f_str.as_bytes()),
                            ord != std::cmp::Ordering::Greater,
                            "le a=0.{} f=0.{f_str}",
                            frac_str(&a)
                        );
                    }
                }
            }
        }
    }

    // Exact value of a decimal string scaled to `s` fractional places (assumes <= s frac digits).
    fn scaled(x: &str, s: u32) -> i128 {
        let (i, f) = x.split_once('.').unwrap_or((x, ""));
        let mut digits = i.to_string();
        for _ in 0..s {
            digits.push('0');
        }
        let base: i128 = digits.parse().unwrap();
        let frac_shift: i128 = f
            .chars()
            .fold(0i128, |a, c| a * 10 + (c as u8 - b'0') as i128)
            * 10i128.pow(s - f.len() as u32);
        base + frac_shift
    }

    #[test]
    fn nonneg_decimal_range_agrees_with_the_exact_rational_oracle() {
        let cands = [
            "0", "0.0", "0.5", "1", "1.0", "1.5", "1.50", "1.499", "1.7", "2", "2.5", "3", "3.7",
            "3.70", "3.700", "3.71", "0.25", "0.3", "2.999", "10", "10.4", "9.99", "0.1", "0.01",
            "1.25", "2.3", "2.7", "2.35",
        ];
        #[allow(clippy::type_complexity)]
        let bounds: &[((u64, &str), Option<(u64, &str)>)] = &[
            ((1, "5"), Some((3, "7"))),
            ((0, "0"), Some((1, "5"))),
            ((2, "3"), Some((2, "7"))),
            ((1, "5"), None),
            ((0, "25"), Some((10, "4"))),
            ((3, "0"), Some((3, "0"))),
        ];
        for (lo, hi) in bounds {
            let (li, lf) = (lo.0, digits(lo.1));
            let hi_pair = hi.map(|(i, f)| (i, digits(f)));
            let regex =
                nonneg_decimal_range((li, &lf), hi_pair.as_ref().map(|(i, f)| (*i, f.as_slice())))
                    .unwrap();
            let e = build_from_regex(&format!("^{regex}$")).unwrap();
            let s = 6u32;
            let lo_v = scaled(&format!("{}.{}", lo.0, lo.1), s);
            let hi_v = hi.map(|(i, f)| scaled(&format!("{i}.{f}"), s));
            for c in cands {
                let v = scaled(c, s);
                let want = v >= lo_v && hi_v.is_none_or(|h| v <= h);
                assert_eq!(e.accepts(c.as_bytes()), want, "range={lo:?}..{hi:?} c={c}");
            }
        }
    }

    #[test]
    fn never_is_dead_from_the_very_first_byte() {
        let mut b = Builder::new(CompileOptions::default());
        let n = b.never().unwrap();
        let ir = b.finish(n).unwrap();
        let e = compile_ir(&ir).unwrap();
        assert!(
            e.is_dead(e.start()),
            "never must be dead before consuming any byte"
        );
        assert!(!e.accepts(b"null"));
        assert!(!e.accepts(b"1"));
        assert!(!e.accepts(b""));
    }

    /// A single-field object has one possible key order: no shuffle automaton is built for it.
    #[test]
    fn single_required_field_object_skips_the_shuffle_round_trip() {
        let schema = r#"{"type":"object","additionalProperties":false,"required":["a"],
            "properties":{"a":{"type":"string"}}}"#;
        let ir = crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap();
        let e = compile_ir(&ir).unwrap();
        assert!(e.accepts(br#"{"a":"x"}"#));
        assert!(!e.accepts(b"{}"), "the required field is missing");
        assert!(!e.accepts(br#"{"a":1}"#), "a is a string, not a number");
    }

    #[test]
    fn single_optional_field_object_accepts_empty_or_present() {
        let schema = r#"{"type":"object","additionalProperties":false,
            "properties":{"a":{"type":"string"}}}"#;
        let ir = crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap();
        let e = compile_ir(&ir).unwrap();
        assert!(e.accepts(br#"{"a":"x"}"#));
        assert!(e.accepts(b"{}"), "an optional field may be absent");
    }

    // Regression for a real jsonschemabench schema that used to take 30+ seconds: an array of a
    // tiny 2-field object whose string-pattern child, once eliminated back to regex text by the
    // parent shuffle, blew up to 163KB - then got re-embedded and re-compiled twice more.
    #[test]
    fn a_real_world_nested_array_of_objects_schema_compiles_or_falls_back_in_under_a_second() {
        let schema = r#"{
            "properties": {
                "builders": {
                    "items": {
                        "properties": {
                            "builder": {"pattern": ".*:.*", "type": "string"},
                            "options": {"type": "object", "additionalProperties": true}
                        },
                        "required": ["builder"],
                        "type": "object",
                        "additionalProperties": true
                    },
                    "minItems": 1,
                    "type": "array"
                }
            },
            "type": "object",
            "additionalProperties": true
        }"#;
        let opts = CompileOptions {
            object_closure: crate::ir::ObjectClosure::AssumeClosedProfile,
            ..CompileOptions::default()
        };
        let ir = crate::frontend::schema_to_ir(schema, opts).unwrap();
        assert_eq!(ir.diagnostics().next(), None);
        let start = std::time::Instant::now();
        let result = compile_ir(&ir);
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(1),
            "compile_ir must fail fast (or succeed fast), took {elapsed:?}"
        );
        // Either outcome is fine: the point is speed, not which backend wins this schema.
        if let Ok(engine) = result {
            assert!(engine.accepts(br#"{"builders":[{"builder":"a:b"}]}"#));
        } else {
            assert!(crate::structured::try_accepts(
                std::sync::Arc::new(ir),
                br#"{"builders":[{"builder":"a:b"}]}"#,
            )
            .expect("structured program"));
        }
    }

    // Regression for a real jsonschemabench schema (Github_easy/o67463.json) that used to take
    // 20+ seconds: array-of-array-of-oneOf, whose eliminated regex text doubled per array wrap.
    #[test]
    fn a_real_world_nested_array_of_array_of_one_of_schema_compiles_or_falls_back_fast() {
        // The jsonschemabench corpus is an optional local checkout, not part of this repository.
        let path = format!(
            "{}/../../integration/jsonschemabench/data/Github_easy/o67463.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let Ok(schema) = std::fs::read_to_string(&path) else {
            eprintln!("skipping: corpus not present at {path}");
            return;
        };
        let opts = CompileOptions {
            object_closure: crate::ir::ObjectClosure::AssumeClosedProfile,
            ..CompileOptions::default()
        };
        let ir = crate::frontend::schema_to_ir(&schema, opts).unwrap();
        assert_eq!(ir.diagnostics().next(), None);
        let start = std::time::Instant::now();
        let result = compile_ir(&ir);
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "compile_ir must fail fast (or succeed fast), took {elapsed:?}"
        );
        let data = br#"[[{"field":"year"}]]"#;
        if let Ok(engine) = result {
            assert!(engine.accepts(data));
        } else {
            assert!(
                crate::structured::try_accepts(std::sync::Arc::new(ir), data)
                    .expect("structured program")
            );
        }
    }

    fn engine_for_integer(lo: Option<i64>, hi: Option<i64>) -> RefEngine {
        let mut b = Builder::new(CompileOptions::default());
        let n = b.integer(lo, hi).unwrap();
        let ir = b.finish(n).unwrap();
        compile_ir(&ir).unwrap()
    }

    #[test]
    fn integer_range_matches_membership_by_brute_force() {
        let bounds = [
            (Some(0), Some(0)),
            (Some(0), Some(9)),
            (Some(1), Some(255)),
            (Some(-5), Some(5)),
            (Some(-128), Some(127)),
            (Some(10), Some(1000)),
            (Some(-1000), Some(-7)),
            (Some(-50), Some(50)),
            (Some(99), Some(101)),
            (Some(7), Some(7)),
            (None, Some(3)),
            (Some(-3), None),
            (None, None),
        ];
        for (lo, hi) in bounds {
            let e = engine_for_integer(lo, hi);
            for n in -2000i64..=2000 {
                let want = lo.is_none_or(|l| n >= l) && hi.is_none_or(|h| n <= h);
                let got = e.accepts(n.to_string().as_bytes());
                assert_eq!(got, want, "n={n} bounds=({lo:?},{hi:?})");
            }
        }
    }

    fn engine_for_tuple(
        prefix: Vec<(Option<i64>, Option<i64>)>,
        tail: Option<(Option<i64>, Option<i64>)>,
        min: u32,
        max: Option<u32>,
    ) -> RefEngine {
        let mut b = Builder::new(CompileOptions::default());
        let prefix_ids = prefix
            .iter()
            .map(|&(lo, hi)| b.integer(lo, hi).unwrap())
            .collect();
        let tail_id = tail.map(|(lo, hi)| b.integer(lo, hi).unwrap());
        let node = b.tuple(prefix_ids, tail_id, min, max).unwrap();
        let ir = b.finish(node).unwrap();
        compile_ir(&ir).unwrap()
    }

    #[test]
    fn tuple_bounded_open_tail_accepts_exactly_the_valid_lengths() {
        // prefix [int], uniform int tail, minItems 2, maxItems 4: arrays of 2..=4 single-digit ints.
        let e = engine_for_tuple(
            vec![(Some(0), Some(9))],
            Some((Some(0), Some(9))),
            2,
            Some(4),
        );
        assert!(!e.accepts(b"[1]"), "below minItems");
        assert!(e.accepts(b"[1,2]"));
        assert!(e.accepts(b"[1,2,3]"));
        assert!(e.accepts(b"[1,2,3,4]"));
        assert!(!e.accepts(b"[1,2,3,4,5]"), "above maxItems");
    }

    #[test]
    fn tuple_max_items_below_the_prefix_length_caps_within_the_prefix() {
        // prefix of 3 positions, uniform tail, but maxItems 2 forces every valid length <= 2.
        let e = engine_for_tuple(
            vec![(Some(0), Some(9)), (Some(0), Some(9)), (Some(0), Some(9))],
            Some((Some(0), Some(9))),
            1,
            Some(2),
        );
        assert!(e.accepts(b"[1]"));
        assert!(e.accepts(b"[1,2]"));
        assert!(
            !e.accepts(b"[1,2,3]"),
            "maxItems 2 is below the 3-position prefix"
        );
    }

    #[test]
    fn tuple_moderate_max_items_compiles_without_a_quadratic_alternation() {
        // A moderate maxItems that would be O(max^2) as an enumerated alternation compiles fine as a
        // bounded repetition; a length in range accepts and a length past it rejects.
        let e = engine_for_tuple(
            vec![(Some(0), Some(9))],
            Some((Some(0), Some(9))),
            1,
            Some(64),
        );
        assert!(e.accepts(b"[1,2,3,4,5]"));
        assert!(e.accepts(b"[0]"));
    }

    #[test]
    fn integer_extremes_compile_and_accept_the_boundary() {
        let e = engine_for_integer(Some(i64::MIN), Some(i64::MAX));
        assert!(e.accepts(i64::MIN.to_string().as_bytes()));
        assert!(e.accepts(i64::MAX.to_string().as_bytes()));
        assert!(e.accepts(b"0"));
        assert!(!e.accepts(b"-0"));
        assert!(!e.accepts(b"007"));
    }

    #[test]
    fn rejects_leading_zero_and_plus() {
        let e = engine_for_integer(Some(0), Some(100));
        assert!(e.accepts(b"7"));
        assert!(!e.accepts(b"07"));
        assert!(!e.accepts(b"+7"));
    }

    #[test]
    fn compiling_an_unsupported_ir_fails_with_the_diagnostic() {
        use crate::diagnostics::UnsupportedReason;
        let mut b = Builder::new(CompileOptions::default());
        let u = b
            .unsupported("anyOf", UnsupportedReason::UnionType, "/anyOf")
            .unwrap();
        let ir = b.finish(u).unwrap();
        let e = compile_ir(&ir).unwrap_err();
        assert_eq!(e.code, ErrorCode::Unsupported);
        assert_eq!(e.json_pointer_path.as_deref(), Some("/anyOf"));
    }

    #[test]
    fn boolean_and_null_and_enum_compile() {
        let mut b = Builder::new(CompileOptions::default());
        let n = b.boolean().unwrap();
        let ir = b.finish(n).unwrap();
        let e = compile_ir(&ir).unwrap();
        assert!(e.accepts(b"true") && e.accepts(b"false") && !e.accepts(b"null"));
    }

    #[test]
    fn large_string_enum_compiles_as_a_decoded_value_trie() {
        let mut values = (0..=crate::ir::MAX_UNROLLED_ENUM)
            .map(|index| ScalarLit::Str(format!("member-{index:04}")))
            .collect::<Vec<_>>();
        values.extend([
            ScalarLit::Str("a".to_owned()),
            ScalarLit::Str("😀".to_owned()),
        ]);
        let mut builder = Builder::new(CompileOptions::default());
        let root = builder.enum_values(values).unwrap();
        let ir = builder.finish(root).unwrap();
        assert!(!ir.requires_structured_backend());
        let engine = compile_ir(&ir).unwrap();
        assert!(engine.accepts(br#""member-0042""#));
        assert!(engine.accepts(br#""\u0061""#));
        assert!(engine.accepts(br#""\uD83D\uDE00""#));
        assert!(!engine.accepts(br#""member-9999""#));
    }

    #[test]
    fn small_string_enum_uses_decoded_json_string_semantics() {
        let mut builder = Builder::new(CompileOptions::default());
        let root = builder
            .enum_values(vec![
                ScalarLit::Str("a".to_owned()),
                ScalarLit::Str("😀".to_owned()),
            ])
            .unwrap();
        let engine = compile_ir(&builder.finish(root).unwrap()).unwrap();
        for accepted in [br#""a""#.as_slice(), br#""\u0061""#, br#""\ud83d\ude00""#] {
            assert!(engine.accepts(accepted), "rejected {accepted:?}");
        }
    }

    fn engine_for_union(ranges: &[(Option<i64>, Option<i64>)]) -> RefEngine {
        let mut b = Builder::new(CompileOptions::default());
        let branches = ranges
            .iter()
            .map(|&(lo, hi)| b.integer(lo, hi).unwrap())
            .collect();
        let u = b.union_of(branches).unwrap();
        let ir = b.finish(u).unwrap();
        compile_ir(&ir).unwrap()
    }

    /// A byte-prefix collision (one branch's shorter accepted string is a textual prefix of
    /// another branch's longer one, e.g. "1" from `[0,5]` prefixing "10" from `[3,10]`) must not
    /// cause the union to drop the longer match: see `automaton::byte_dfa::build_from_regex` for
    /// why `MatchKind::All` is required for a union DFA.
    #[test]
    fn union_of_colliding_integer_ranges_matches_membership_by_brute_force() {
        let cases: &[&[(Option<i64>, Option<i64>)]] = &[
            &[(Some(0), Some(5)), (Some(3), Some(10))],
            &[(Some(0), Some(10)), (Some(5), Some(15))],
            &[(Some(-5), Some(5)), (Some(0), Some(10))],
            &[(Some(-100), Some(-7)), (Some(-20), Some(20))],
        ];
        for ranges in cases {
            let e = engine_for_union(ranges);
            for n in -200i64..=200 {
                let want = ranges
                    .iter()
                    .any(|&(lo, hi)| lo.is_none_or(|l| n >= l) && hi.is_none_or(|h| n <= h));
                let got = e.accepts(n.to_string().as_bytes());
                assert_eq!(got, want, "n={n} ranges={ranges:?}");
            }
        }
    }

    #[test]
    fn union_of_0_5_and_3_10_accepts_10() {
        let e = engine_for_union(&[(Some(0), Some(5)), (Some(3), Some(10))]);
        assert!(
            e.accepts(b"10"),
            "10 is in [3,10] so the union must accept it"
        );
        assert!(e.accepts(b"0"));
        assert!(e.accepts(b"5"));
        assert!(!e.accepts(b"11"));
    }

    #[test]
    fn direct_intersection_preserves_acceptance_and_outer_whitespace() {
        let schema = r#"{"allOf":[{"type":"integer","minimum":0,"maximum":10},{"type":"integer","minimum":5,"maximum":20}]}"#;
        let ir = crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap();
        let engine = compile_ir(&ir).unwrap();
        for accepted in [b"5".as_slice(), b"10", b" \t5\r\n"] {
            assert!(engine.accepts(accepted), "rejected {accepted:?}");
        }
        for rejected in [b"4".as_slice(), b"11"] {
            assert!(!engine.accepts(rejected), "accepted {rejected:?}");
        }
    }

    /// Not a correctness check: prints where compile time actually goes (frontend, IR-to-regex
    /// lowering, regex-to-engine) on the real production-corpus schemas. `cargo test --release
    /// -p maskforge-core --lib compile::tests::stage_breakdown -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn stage_breakdown_on_production_corpus_schemas() {
        use crate::frontend::schema_to_ir;
        use std::time::Instant;

        let schemas: [(&str, &str); 6] = [
            ("bool", r#"{"type":"boolean"}"#),
            (
                "flat_object",
                r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false}"#,
            ),
            (
                "nested_object",
                r#"{"type":"object","properties":{"user":{"type":"object","properties":{"id":{"type":"integer"},"active":{"type":"boolean"}},"required":["id","active"],"additionalProperties":false}},"required":["user"],"additionalProperties":false}"#,
            ),
            (
                "bounded_array_of_objects",
                r#"{"type":"array","items":{"type":"object","properties":{"x":{"type":"number"}},"required":["x"],"additionalProperties":false},"minItems":1,"maxItems":5}"#,
            ),
            (
                "deeply_nested",
                r#"{"type":"object","properties":{"l1":{"type":"object","properties":{"l2":{"type":"object","properties":{"l3":{"type":"boolean"}},"required":["l3"],"additionalProperties":false}},"required":["l2"],"additionalProperties":false}},"required":["l1"],"additionalProperties":false}"#,
            ),
            (
                "mixed_object_array",
                r#"{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"},"minItems":0,"maxItems":3},"count":{"type":"integer"}},"required":["tags","count"],"additionalProperties":false}"#,
            ),
        ];

        for (name, text) in schemas {
            let n = 200;
            let mut frontend_ns = 0u128;
            let mut lower_ns = 0u128;
            let mut engine_ns = 0u128;
            for _ in 0..n {
                let t0 = Instant::now();
                let ir = schema_to_ir(text, crate::ir::CompileOptions::default()).unwrap();
                frontend_ns += t0.elapsed().as_nanos();

                let t0 = Instant::now();
                let pattern = lower_reachable(&ir, ir.root()).unwrap();
                lower_ns += t0.elapsed().as_nanos();

                let t0 = Instant::now();
                build_from_regex(&pattern).unwrap();
                engine_ns += t0.elapsed().as_nanos();
            }
            println!(
                "{name:28} frontend={:>8.1}us  lower={:>8.1}us  regex_to_engine={:>8.1}us",
                frontend_ns as f64 / n as f64 / 1000.0,
                lower_ns as f64 / n as f64 / 1000.0,
                engine_ns as f64 / n as f64 / 1000.0,
            );
        }
    }

    /// Exhaustive sweep: every pair of range endpoints within 0..8, checked against integer
    /// membership by brute force. Building one DFA per pair dominates the cost, so the sweep is
    /// kept small (9*10/2 = 45 ranges, 45*45 = 2025 pairs) to stay in the low seconds.
    #[test]
    fn exhaustive_small_range_pair_sweep_agrees_with_brute_force_membership() {
        let ranges: Vec<(i64, i64)> = (0i64..=8)
            .flat_map(|lo| (lo..=8).map(move |hi| (lo, hi)))
            .collect();
        for &(a_lo, a_hi) in &ranges {
            for &(b_lo, b_hi) in &ranges {
                let branches = [(Some(a_lo), Some(a_hi)), (Some(b_lo), Some(b_hi))];
                let e = engine_for_union(&branches);
                for n in 0i64..=15 {
                    let want = (n >= a_lo && n <= a_hi) || (n >= b_lo && n <= b_hi);
                    let got = e.accepts(n.to_string().as_bytes());
                    assert_eq!(got, want, "n={n} a=[{a_lo},{a_hi}] b=[{b_lo},{b_hi}]");
                }
            }
        }
    }

    /// Sweeps every min/max bound pair (with inclusive/exclusive) in -3..=3 against candidates
    /// covering both signs, magnitudes 0..=5, and with/without a fraction, checked against an
    /// independent f64 comparison - not the regex construction under test.
    #[test]
    fn number_range_brute_force_matches_independent_decimal_comparison() {
        for min_v in -3i64..=3 {
            for min_excl in [false, true] {
                for max_v in -3i64..=3 {
                    for max_excl in [false, true] {
                        if max_v < min_v || (max_v == min_v && (min_excl || max_excl)) {
                            continue;
                        }
                        let pattern =
                            number_range_regex(Some((min_v, min_excl)), Some((max_v, max_excl)));
                        let e = build_from_regex(&pattern).unwrap();
                        for sign in [1i64, -1i64] {
                            for int_mag in 0i64..=5 {
                                for has_frac in [false, true] {
                                    if sign < 0 && int_mag == 0 && !has_frac {
                                        continue;
                                    }
                                    let text = format!(
                                        "{}{int_mag}{}",
                                        if sign < 0 { "-" } else { "" },
                                        if has_frac { ".1" } else { "" }
                                    );
                                    let value = sign as f64
                                        * (int_mag as f64 + if has_frac { 0.1 } else { 0.0 });
                                    let ge_min = if min_excl {
                                        value > min_v as f64
                                    } else {
                                        value >= min_v as f64
                                    };
                                    let le_max = if max_excl {
                                        value < max_v as f64
                                    } else {
                                        value <= max_v as f64
                                    };
                                    let want = ge_min && le_max;
                                    let got = e.accepts(text.as_bytes());
                                    assert_eq!(
                                        got, want,
                                        "text={text} min=({min_v},{min_excl}) max=({max_v},{max_excl})"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn engine_from_schema(schema: &str) -> RefEngine {
        let ir = crate::frontend::schema_to_ir(schema, CompileOptions::default()).unwrap();
        compile_ir(&ir).unwrap()
    }

    #[test]
    fn object_whitespace_variants_are_all_accepted() {
        let e = engine_from_schema(
            r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false}"#,
        );
        assert!(e.accepts(br#"{"a":1}"#));
        assert!(e.accepts(br#"{"a" : 1}"#));
        assert!(e.accepts(br#"{ "a" : 1 }"#));
        assert!(e.accepts(b"{\n\"a\"\t:\r1\n}"));
    }

    #[test]
    fn array_whitespace_variants_are_all_accepted() {
        let e = engine_from_schema(r#"{"type":"array","items":{"type":"integer"}}"#);
        assert!(e.accepts(b"[1,2]"));
        assert!(e.accepts(b"[ 1 , 2 ]"));
    }

    #[test]
    fn root_leading_and_trailing_whitespace_is_accepted() {
        let e = engine_from_schema(r#"{"type":"boolean"}"#);
        assert!(e.accepts(b" true"));
        assert!(e.accepts(b"true "));
        assert!(e.accepts(b" \t\ntrue\r\n "));
    }

    #[test]
    fn nested_object_and_array_whitespace_is_accepted() {
        let e = engine_from_schema(
            r#"{"type":"object","properties":{"xs":{"type":"array","items":{"type":"boolean"}}},"required":["xs"],"additionalProperties":false}"#,
        );
        assert!(e.accepts(br#"{ "xs" : [ true , false ] }"#));
    }

    #[test]
    fn punctuation_inside_a_string_const_is_never_treated_as_structure() {
        let e = engine_from_schema(r#"{"const":"a:b,c"}"#);
        assert!(e.accepts(br#""a:b,c""#));
        assert!(
            !e.accepts(br#""a: b, c""#),
            "string content is literal, not JSON structure"
        );
    }

    #[test]
    fn composite_object_const_keeps_string_content_literal_but_own_structure_spaced() {
        let e = engine_from_schema(r#"{"const":{"a":"x:y,z"}}"#);
        assert!(e.accepts(br#"{"a":"x:y,z"}"#));
        assert!(e.accepts(br#"{ "a" : "x:y,z" }"#));
        assert!(
            !e.accepts(br#"{"a":"x: y, z"}"#),
            "the : and , inside the string are literal"
        );
    }

    #[test]
    fn composite_literal_handles_escaped_quotes_and_backslashes() {
        let e = engine_from_schema(r#"{"const":{"k":"a\"b\\c"}}"#);
        assert!(e.accepts(br#"{"k":"a\"b\\c"}"#));
        assert!(e.accepts(br#"{ "k" : "a\"b\\c" }"#));
    }

    #[test]
    fn composite_array_and_enum_const_accept_whitespace_around_their_own_structure() {
        let arr = engine_from_schema(r#"{"const":[1,{"a":true},[2,3]]}"#);
        assert!(arr.accepts(br#"[1,{"a":true},[2,3]]"#));
        assert!(arr.accepts(br#"[ 1 , { "a" : true } , [ 2 , 3 ] ]"#));

        let en = engine_from_schema(r#"{"enum":[{"x":1},[2,3]]}"#);
        assert!(en.accepts(br#"{ "x" : 1 }"#));
        assert!(en.accepts(b"[ 2 , 3 ]"));
    }
}
