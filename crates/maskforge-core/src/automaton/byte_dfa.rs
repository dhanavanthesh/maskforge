//! Anchored byte-DFA built via `regex-automata`, materialized into MaskForge's canonical-DEAD reference graph.

use std::collections::VecDeque;

use regex_automata::dfa::{dense, Automaton, StartKind};
use regex_automata::nfa::thompson;
use regex_automata::util::primitives::StateID as RaStateId;
use regex_automata::util::{start, syntax};
use regex_automata::{Anchored, MatchKind};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::error::{CompileError, ErrorCode, Stage};
use crate::mem_gate::acquire_automaton_build_bytes;
use crate::primitives::{ByteClassId, StateId};

use super::graph::{graph_limit, AutomatonGraph};

use crate::mask::Bitmask;
use crate::primitives::TokenId;
use crate::vocab::Vocabulary;

const DFA_SIZE_LIMIT: usize = 64 << 20;
const DETERMINIZE_SIZE_LIMIT: usize = 64 << 20;
const NFA_SIZE_LIMIT: usize = 16 << 20;
const NEST_LIMIT: u32 = 4096;

const MAX_MATERIALIZE_STATES: usize = 1 << 16;
const MAX_TRANSITION_CELLS: usize = 1 << 22;

const DISCOVERY_SCRATCH_BYTES_PER_STATE: usize = 256
    * (std::mem::size_of::<u16>() + std::mem::size_of::<u32>())
    + 6 * (std::mem::size_of::<u32>() + 16);

fn materialize_worst_case_bytes() -> usize {
    MAX_MATERIALIZE_STATES * std::mem::size_of::<u32>()
        + MAX_TRANSITION_CELLS * std::mem::size_of::<(ByteClassId, StateId)>()
        + MAX_MATERIALIZE_STATES
            * (std::mem::size_of::<bool>()
                + std::mem::size_of::<ClassMask>()
                + std::mem::size_of::<u64>())
        + MAX_TRANSITION_CELLS * std::mem::size_of::<u32>()
        + MAX_MATERIALIZE_STATES
            * (2 * std::mem::size_of::<u32>()
                + 4 * std::mem::size_of::<usize>()
                + std::mem::size_of::<u64>())
}

fn automaton_build_worst_case_bytes(
    dfa_size_limit: usize,
    determinize_size_limit: usize,
    nfa_size_limit: usize,
) -> usize {
    dfa_size_limit
        .saturating_add(determinize_size_limit)
        .saturating_add(nfa_size_limit)
        .saturating_add(MAX_MATERIALIZE_STATES.saturating_mul(DISCOVERY_SCRATCH_BYTES_PER_STATE))
        .saturating_add(materialize_worst_case_bytes())
}

pub(crate) const DEAD: StateId = StateId(0);

#[derive(Debug)]
pub(crate) struct ClassTable {
    pub(crate) members: Vec<Vec<u8>>,
    pub(crate) of_byte: [ByteClassId; 256],
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct ClassMask {
    words: [u64; 4],
}

impl ClassMask {
    fn empty() -> Self {
        Self { words: [0; 4] }
    }

    fn set(&mut self, c: usize) {
        self.words[c >> 6] |= 1u64 << (c & 63);
    }

    pub(crate) fn contains(&self, c: usize) -> bool {
        self.words[c >> 6] & (1u64 << (c & 63)) != 0
    }

    fn is_empty(&self) -> bool {
        self.words == [0; 4]
    }
}

#[derive(Debug)]
struct Transitions {
    row_offsets: Box<[u32]>,
    edges: Box<[(ByteClassId, StateId)]>,
}

#[derive(Debug)]
pub struct RefEngine {
    transitions: Transitions,
    accepting: Box<[bool]>,
    residual_cardinality: Box<[u64]>,
    live_classes: Box<[ClassMask]>,
    pub(crate) class_table: ClassTable,
    class_count: usize,
    state_count: usize,
    live_transition_count: usize,
    start: StateId,
    dead: StateId,
}

impl RefEngine {
    pub(crate) fn from_graph(graph: &AutomatonGraph) -> Result<Self, CompileError> {
        graph_to_engine(graph)
    }

    #[must_use]
    pub fn state_count(&self) -> usize {
        self.state_count
    }

    #[must_use]
    pub fn transition_cells(&self) -> usize {
        self.state_count.saturating_mul(self.class_count)
    }

    #[must_use]
    pub fn byte_class_count(&self) -> usize {
        self.class_count
    }

    #[must_use]
    pub fn live_transition_count(&self) -> usize {
        self.live_transition_count
    }

    pub(crate) fn raw_byte_transition_count(&self) -> Option<usize> {
        self.transitions
            .edges
            .iter()
            .try_fold(0usize, |total, (class, _)| {
                self.class_table
                    .members
                    .get(usize::from(class.get()))
                    .and_then(|bytes| total.checked_add(bytes.len()))
            })
    }

    #[must_use]
    pub fn transition_density(&self) -> f64 {
        let cells = self.transition_cells();
        if cells == 0 {
            return 0.0;
        }
        self.live_transition_count as f64 / cells as f64
    }

    #[must_use]
    #[deprecated(note = "use live_transition_count() (or transition_cells() for the cell count)")]
    pub fn transition_count(&self) -> usize {
        self.live_transition_count
    }

    #[inline]
    pub(crate) fn step(&self, state: StateId, class: ByteClassId) -> StateId {
        let Ok(s) = usize::try_from(state.get()) else {
            return self.dead;
        };
        let c = usize::from(class.get());
        if s >= self.state_count || c >= self.class_count {
            return self.dead;
        }
        let (Some(&lo), Some(&hi)) = (
            self.transitions.row_offsets.get(s),
            self.transitions.row_offsets.get(s + 1),
        ) else {
            return self.dead;
        };
        for (cc, t) in &self.transitions.edges[lo as usize..hi as usize] {
            let cc = usize::from(cc.get());
            if cc == c {
                return *t;
            }
            if cc > c {
                break;
            }
        }
        self.dead
    }

    pub(crate) fn live_classes(&self, state: StateId) -> ClassMask {
        self.live_classes
            .get(state.get() as usize)
            .copied()
            .unwrap_or_else(ClassMask::empty)
    }

    #[must_use]
    pub fn start(&self) -> StateId {
        self.start
    }

    #[must_use]
    pub fn dead(&self) -> StateId {
        self.dead
    }

    #[must_use]
    pub fn is_accepting(&self, s: StateId) -> bool {
        self.accepting
            .get(s.get() as usize)
            .copied()
            .unwrap_or(false)
    }

    #[must_use]
    pub fn is_dead(&self, s: StateId) -> bool {
        s == self.dead || s.get() as usize >= self.state_count
    }

    #[must_use]
    pub fn can_continue(&self, s: StateId) -> bool {
        !self.live_classes(s).is_empty()
    }

    #[must_use]
    pub(crate) fn residual_cardinality(&self, s: StateId) -> u64 {
        self.residual_cardinality
            .get(s.get() as usize)
            .copied()
            .unwrap_or(0)
    }

    #[must_use]
    pub fn heap_bytes(&self) -> usize {
        let transitions = self
            .transitions
            .row_offsets
            .len()
            .saturating_mul(std::mem::size_of::<u32>())
            .saturating_add(
                self.transitions
                    .edges
                    .len()
                    .saturating_mul(std::mem::size_of::<(ByteClassId, StateId)>()),
            );
        let accepting = self
            .accepting
            .len()
            .saturating_mul(std::mem::size_of::<bool>());
        let residual_cardinality = self
            .residual_cardinality
            .len()
            .saturating_mul(std::mem::size_of::<u64>());
        let live_classes = self
            .live_classes
            .len()
            .saturating_mul(std::mem::size_of::<ClassMask>());
        let class_bytes: usize = self
            .class_table
            .members
            .iter()
            .map(Vec::capacity)
            .sum::<usize>()
            .saturating_add(
                self.class_table
                    .members
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Vec<u8>>()),
            );
        transitions
            .saturating_add(accepting)
            .saturating_add(residual_cardinality)
            .saturating_add(live_classes)
            .saturating_add(class_bytes)
    }
}

fn residual_cardinalities(
    transitions: &Transitions,
    accepting: &[bool],
    class_table: &ClassTable,
) -> Result<Box<[u64]>, CompileError> {
    let state_count = accepting.len();
    let edge_count = transitions.edges.len();
    let mut predecessor_counts = Vec::new();
    predecessor_counts
        .try_reserve_exact(state_count)
        .map_err(|_| limit_err())?;
    predecessor_counts.resize(state_count, 0u32);
    let mut remaining_outdegree = Vec::new();
    remaining_outdegree
        .try_reserve_exact(state_count)
        .map_err(|_| limit_err())?;
    remaining_outdegree.resize(state_count, 0u32);
    for (state, outdegree) in remaining_outdegree.iter_mut().enumerate() {
        let lo = transitions.row_offsets[state] as usize;
        let hi = transitions.row_offsets[state + 1] as usize;
        *outdegree = u32::try_from(hi - lo).map_err(|_| limit_err())?;
        for &(_, target) in &transitions.edges[lo..hi] {
            let target = target.get() as usize;
            predecessor_counts[target] = predecessor_counts[target]
                .checked_add(1)
                .ok_or_else(limit_err)?;
        }
    }

    let mut predecessor_offsets = Vec::new();
    predecessor_offsets
        .try_reserve_exact(state_count + 1)
        .map_err(|_| limit_err())?;
    predecessor_offsets.push(0usize);
    for count in predecessor_counts {
        let next = predecessor_offsets
            .last()
            .copied()
            .and_then(|offset| offset.checked_add(count as usize))
            .ok_or_else(limit_err)?;
        predecessor_offsets.push(next);
    }
    let mut predecessors = Vec::new();
    predecessors
        .try_reserve_exact(edge_count)
        .map_err(|_| limit_err())?;
    predecessors.resize(edge_count, 0u32);
    let mut cursors = Vec::new();
    cursors
        .try_reserve_exact(state_count)
        .map_err(|_| limit_err())?;
    cursors.extend_from_slice(&predecessor_offsets[..state_count]);
    for source in 0..state_count {
        let lo = transitions.row_offsets[source] as usize;
        let hi = transitions.row_offsets[source + 1] as usize;
        for &(_, target) in &transitions.edges[lo..hi] {
            let target = target.get() as usize;
            predecessors[cursors[target]] = u32::try_from(source).map_err(|_| limit_err())?;
            cursors[target] += 1;
        }
    }

    let mut queue = VecDeque::new();
    queue.try_reserve(state_count).map_err(|_| limit_err())?;
    for (state, &outdegree) in remaining_outdegree.iter().enumerate().skip(1) {
        if outdegree == 0 {
            queue.push_back(state);
        }
    }
    let mut finite_order = Vec::new();
    finite_order
        .try_reserve_exact(state_count)
        .map_err(|_| limit_err())?;
    while let Some(target) = queue.pop_front() {
        finite_order.push(target);
        for &source in &predecessors[predecessor_offsets[target]..predecessor_offsets[target + 1]] {
            let source = source as usize;
            remaining_outdegree[source] = remaining_outdegree[source]
                .checked_sub(1)
                .ok_or_else(limit_err)?;
            if source != DEAD.get() as usize && remaining_outdegree[source] == 0 {
                queue.push_back(source);
            }
        }
    }

    let mut cardinalities = Vec::new();
    cardinalities
        .try_reserve_exact(state_count)
        .map_err(|_| limit_err())?;
    cardinalities.resize(state_count, u64::MAX);
    cardinalities[DEAD.get() as usize] = 0;
    for &state in &finite_order {
        let mut count: u64 = if accepting[state] { 1 } else { 0 };
        let lo = transitions.row_offsets[state] as usize;
        let hi = transitions.row_offsets[state + 1] as usize;
        for &(class, target) in &transitions.edges[lo..hi] {
            let multiplicity = class_table.members[class.get() as usize].len() as u64;
            count = count
                .saturating_add(cardinalities[target.get() as usize].saturating_mul(multiplicity));
        }
        cardinalities[state] = count;
    }
    Ok(cardinalities.into_boxed_slice())
}

fn graph_to_engine(graph: &AutomatonGraph) -> Result<RefEngine, CompileError> {
    let state_count = graph.states.len();
    let cells = state_count.checked_mul(256).ok_or_else(limit_err)?;
    if state_count > MAX_MATERIALIZE_STATES || cells > (1 << 24) {
        return Err(graph_limit("graph byte-matrix cap"));
    }
    let estimate = cells
        .checked_mul(std::mem::size_of::<StateId>())
        .and_then(|bytes| bytes.checked_add(materialize_worst_case_bytes()))
        .ok_or_else(limit_err)?;
    let _permit = acquire_automaton_build_bytes(estimate)?;
    let live = graph_live_states(graph)?;

    let mut remap = Vec::new();
    remap
        .try_reserve_exact(state_count)
        .map_err(|_| graph_limit("graph remap allocation"))?;
    remap.resize(state_count, DEAD);
    let mut next = 1u32;
    for (index, is_live) in live.iter().copied().enumerate() {
        if is_live {
            remap[index] = StateId(next);
            next = next
                .checked_add(1)
                .ok_or_else(|| graph_limit("graph state id"))?;
        }
    }
    let final_state_count = usize::try_from(next).map_err(|_| limit_err())?;
    let final_cells = final_state_count.checked_mul(256).ok_or_else(limit_err)?;

    let mut matrix = Vec::new();
    matrix
        .try_reserve_exact(final_cells)
        .map_err(|_| graph_limit("graph transition matrix allocation"))?;
    matrix.resize(final_cells, DEAD);
    for (source, mapped) in remap.iter().copied().enumerate() {
        if mapped == DEAD {
            continue;
        }
        let row = usize::try_from(mapped.get()).map_err(|_| limit_err())?;
        for byte in 0u8..=255 {
            let target = graph
                .transition(source, byte)
                .and_then(|target| remap.get(target).copied())
                .unwrap_or(DEAD);
            matrix[row * 256 + usize::from(byte)] = target;
        }
    }
    let (class_table, representatives) = graph_class_table(&matrix, final_state_count)?;
    let transition_cells = final_state_count
        .checked_mul(class_table.members.len())
        .ok_or_else(limit_err)?;
    if transition_cells > MAX_TRANSITION_CELLS {
        return Err(graph_limit("live graph transition-cell cap"));
    }
    materialize_graph_rows(graph, &remap, matrix, class_table, &representatives)
}

fn graph_live_states(graph: &AutomatonGraph) -> Result<Vec<bool>, CompileError> {
    let state_count = graph.states.len();
    let mut predecessor_count = Vec::new();
    predecessor_count
        .try_reserve_exact(state_count)
        .map_err(|_| graph_limit("graph predecessor allocation"))?;
    predecessor_count.resize(state_count, 0usize);
    for edge in &graph.edges {
        let target = usize::try_from(edge.target.0).map_err(|_| limit_err())?;
        let count = predecessor_count
            .get_mut(target)
            .ok_or_else(|| graph_limit("graph target"))?;
        *count = count
            .checked_add(1)
            .ok_or_else(|| graph_limit("graph predecessor count"))?;
    }
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(state_count + 1)
        .map_err(|_| graph_limit("graph reverse offsets allocation"))?;
    offsets.push(0usize);
    for count in predecessor_count {
        let next = offsets
            .last()
            .copied()
            .and_then(|offset| offset.checked_add(count))
            .ok_or_else(|| graph_limit("graph reverse offset"))?;
        offsets.push(next);
    }
    let mut predecessors = Vec::new();
    predecessors
        .try_reserve_exact(graph.edges.len())
        .map_err(|_| graph_limit("graph reverse edges allocation"))?;
    predecessors.resize(graph.edges.len(), 0usize);
    let mut cursor = Vec::new();
    cursor
        .try_reserve_exact(state_count)
        .map_err(|_| graph_limit("graph reverse cursor allocation"))?;
    cursor.extend_from_slice(&offsets[..state_count]);
    for (source, state) in graph.states.iter().enumerate() {
        let lo = usize::try_from(state.edge_start).map_err(|_| limit_err())?;
        let len = usize::try_from(state.edge_len).map_err(|_| limit_err())?;
        let hi = lo.checked_add(len).ok_or_else(limit_err)?;
        for edge in graph.edges.get(lo..hi).ok_or_else(limit_err)? {
            let target = usize::try_from(edge.target.0).map_err(|_| limit_err())?;
            let at = cursor.get_mut(target).ok_or_else(limit_err)?;
            predecessors[*at] = source;
            *at = at.checked_add(1).ok_or_else(limit_err)?;
        }
    }
    let mut live = Vec::new();
    live.try_reserve_exact(state_count)
        .map_err(|_| graph_limit("graph live-set allocation"))?;
    live.resize(state_count, false);
    let mut queue = VecDeque::new();
    queue
        .try_reserve(state_count)
        .map_err(|_| graph_limit("graph live queue allocation"))?;
    for (index, state) in graph.states.iter().enumerate() {
        if state.accepting {
            live[index] = true;
            queue.push_back(index);
        }
    }
    while let Some(target) = queue.pop_front() {
        for &source in &predecessors[offsets[target]..offsets[target + 1]] {
            if !live[source] {
                live[source] = true;
                queue.push_back(source);
            }
        }
    }
    Ok(live)
}

fn graph_class_table(
    matrix: &[StateId],
    state_count: usize,
) -> Result<(ClassTable, Vec<u8>), CompileError> {
    let mut signatures: FxHashMap<Box<[StateId]>, u16> = FxHashMap::default();
    signatures
        .try_reserve(256)
        .map_err(|_| graph_limit("graph class map allocation"))?;
    let mut members: Vec<Vec<u8>> = Vec::new();
    members
        .try_reserve_exact(256)
        .map_err(|_| graph_limit("graph class allocation"))?;
    let mut representatives = Vec::new();
    representatives
        .try_reserve_exact(256)
        .map_err(|_| graph_limit("graph class representative allocation"))?;
    let mut of_byte = [ByteClassId(0); 256];
    for byte in 0u8..=255 {
        let mut signature = Vec::new();
        signature
            .try_reserve_exact(state_count)
            .map_err(|_| graph_limit("graph class signature allocation"))?;
        for state in 0..state_count {
            signature.push(matrix[state * 256 + usize::from(byte)]);
        }
        let class = if let Some(class) = signatures.get(signature.as_slice()) {
            *class
        } else {
            let class = u16::try_from(members.len()).map_err(|_| limit_err())?;
            signatures.insert(signature.into_boxed_slice(), class);
            members.push(Vec::new());
            representatives.push(byte);
            class
        };
        members[usize::from(class)]
            .try_reserve(1)
            .map_err(|_| graph_limit("graph class member allocation"))?;
        members[usize::from(class)].push(byte);
        of_byte[usize::from(byte)] = ByteClassId(class);
    }
    Ok((ClassTable { members, of_byte }, representatives))
}

fn materialize_graph_rows(
    graph: &AutomatonGraph,
    remap: &[StateId],
    matrix: Vec<StateId>,
    class_table: ClassTable,
    representatives: &[u8],
) -> Result<RefEngine, CompileError> {
    let state_count = matrix.len() / 256;
    let mut accepting = Vec::new();
    accepting
        .try_reserve_exact(state_count)
        .map_err(|_| graph_limit("graph accepting allocation"))?;
    accepting.resize(state_count, false);
    for (source, mapped) in remap.iter().copied().enumerate() {
        if mapped != DEAD {
            accepting[usize::try_from(mapped.get()).map_err(|_| limit_err())?] =
                graph.states[source].accepting;
        }
    }
    let mut live_classes = Vec::new();
    live_classes
        .try_reserve_exact(state_count)
        .map_err(|_| graph_limit("graph live-class allocation"))?;
    live_classes.resize(state_count, ClassMask::empty());
    let mut row_offsets = Vec::new();
    row_offsets
        .try_reserve_exact(state_count + 1)
        .map_err(|_| graph_limit("graph row allocation"))?;
    let mut edges = Vec::new();
    edges
        .try_reserve(matrix.iter().filter(|&&target| target != DEAD).count())
        .map_err(|_| graph_limit("graph transition allocation"))?;
    for state in 0..state_count {
        row_offsets.push(u32::try_from(edges.len()).map_err(|_| limit_err())?);
        for (class, byte) in representatives.iter().copied().enumerate() {
            let target = matrix[state * 256 + usize::from(byte)];
            if target != DEAD {
                let class_id = ByteClassId(u16::try_from(class).map_err(|_| limit_err())?);
                edges.push((class_id, target));
                live_classes[state].set(class);
            }
        }
    }
    row_offsets.push(u32::try_from(edges.len()).map_err(|_| limit_err())?);
    let start_index = usize::try_from(graph.start.0).map_err(|_| limit_err())?;
    let start = remap.get(start_index).copied().unwrap_or(DEAD);
    let live_transition_count = edges.len();
    let transitions = Transitions {
        row_offsets: row_offsets.into_boxed_slice(),
        edges: edges.into_boxed_slice(),
    };
    let accepting = accepting.into_boxed_slice();
    let residual_cardinality = residual_cardinalities(&transitions, &accepting, &class_table)?;
    Ok(RefEngine {
        transitions,
        accepting,
        residual_cardinality,
        live_classes: live_classes.into_boxed_slice(),
        class_count: class_table.members.len(),
        class_table,
        state_count,
        live_transition_count,
        start,
        dead: DEAD,
    })
}

pub(crate) fn build_from_regex(pattern: &str) -> Result<RefEngine, CompileError> {
    build_from_regex_with_limits(
        pattern,
        DFA_SIZE_LIMIT,
        DETERMINIZE_SIZE_LIMIT,
        NFA_SIZE_LIMIT,
        NEST_LIMIT,
    )
}

pub(crate) fn build_from_unicode_regex(pattern: &str) -> Result<RefEngine, CompileError> {
    let _permit = acquire_automaton_build_bytes(automaton_build_worst_case_bytes(
        DFA_SIZE_LIMIT,
        DETERMINIZE_SIZE_LIMIT,
        NFA_SIZE_LIMIT,
    ))?;
    let dfa = dense::Builder::new()
        .configure(
            dense::Config::new()
                .minimize(false)
                .start_kind(StartKind::Anchored)
                .match_kind(MatchKind::All)
                .dfa_size_limit(Some(DFA_SIZE_LIMIT))
                .determinize_size_limit(Some(DETERMINIZE_SIZE_LIMIT)),
        )
        .syntax(
            syntax::Config::new()
                .unicode(true)
                .utf8(true)
                .nest_limit(NEST_LIMIT),
        )
        .thompson(thompson::Config::new().nfa_size_limit(Some(NFA_SIZE_LIMIT)))
        .build(pattern)
        .map_err(map_build_error)?;
    bfs_materialize(&dfa)
}

fn build_from_regex_with_limits(
    pattern: &str,
    dfa_size_limit: usize,
    determinize_size_limit: usize,
    nfa_size_limit: usize,
    nest_limit: u32,
) -> Result<RefEngine, CompileError> {
    let _permit = acquire_automaton_build_bytes(automaton_build_worst_case_bytes(
        dfa_size_limit,
        determinize_size_limit,
        nfa_size_limit,
    ))?;
    let dfa = dense::Builder::new()
        .configure(
            dense::Config::new()
                .minimize(false)
                .start_kind(StartKind::Anchored)
                .match_kind(MatchKind::All)
                .dfa_size_limit(Some(dfa_size_limit))
                .determinize_size_limit(Some(determinize_size_limit)),
        )
        .syntax(
            syntax::Config::new()
                .unicode(false)
                .utf8(false)
                .nest_limit(nest_limit),
        )
        .thompson(thompson::Config::new().nfa_size_limit(Some(nfa_size_limit)))
        .build(pattern)
        .map_err(map_build_error)?;
    bfs_materialize(&dfa)
}

fn map_build_error(e: dense::BuildError) -> CompileError {
    let nfa_size_limit_hit = std::error::Error::source(&e)
        .and_then(|src| src.downcast_ref::<thompson::BuildError>())
        .is_some_and(|nfa_err| nfa_err.size_limit().is_some());
    let code = if e.is_size_limit_exceeded() || nfa_size_limit_hit {
        ErrorCode::InternalLimitExceeded
    } else {
        ErrorCode::Malformed
    };
    let mut observed = e.to_string();
    let mut source = std::error::Error::source(&e);
    for _ in 0..4 {
        let Some(error) = source else {
            break;
        };
        observed.push_str(": ");
        observed.push_str(&error.to_string());
        source = error.source();
    }
    CompileError::new(
        code,
        Stage::L3,
        "regex-automata failed to build the byte DFA",
    )
    .with_observed(observed)
}

fn build_class_table(dfa: &dense::DFA<Vec<u32>>) -> ClassTable {
    let classes = dfa.byte_classes();
    let mut raw: FxHashMap<u8, Vec<u8>> = FxHashMap::default();
    for b in 0u8..=255 {
        raw.entry(classes.get(b)).or_default().push(b);
    }
    let mut ordered: Vec<(u8, Vec<u8>)> = raw.into_iter().collect();
    ordered.sort_by_key(|(_, bytes)| bytes[0]);
    let members: Vec<Vec<u8>> = ordered.into_iter().map(|(_, bytes)| bytes).collect();

    let mut of_byte = [ByteClassId(0); 256];
    for (cid, bytes) in members.iter().enumerate() {
        let class = ByteClassId(u16::try_from(cid).expect("byte class count <= 256"));
        for &b in bytes {
            of_byte[b as usize] = class;
        }
    }
    ClassTable { members, of_byte }
}

struct RawState {
    edges: Vec<(ByteClassId, RaStateId)>,
    accepting: bool,
}

fn bfs_materialize(dfa: &dense::DFA<Vec<u32>>) -> Result<RefEngine, CompileError> {
    let ct = build_class_table(dfa);
    let num_classes = ct.members.len();

    let ext_start = dfa
        .start_state(&start::Config::new().anchored(Anchored::Yes))
        .map_err(|e| {
            CompileError::new(
                ErrorCode::Malformed,
                Stage::L3,
                "anchored start state does not exist",
            )
            .with_observed(e.to_string())
        })?;
    if dfa.is_quit_state(ext_start) {
        return Err(quit_err());
    }

    let mut disco_order: Vec<RaStateId> = Vec::new();
    let mut seen: FxHashSet<RaStateId> = FxHashSet::default();
    let mut raw: FxHashMap<RaStateId, RawState> = FxHashMap::default();
    let mut queue: VecDeque<RaStateId> = VecDeque::new();
    queue.push_back(ext_start);
    seen.insert(ext_start);
    while let Some(e) = queue.pop_front() {
        if disco_order.len() >= MAX_MATERIALIZE_STATES {
            return Err(limit_err());
        }
        disco_order.push(e);
        let mut edges = Vec::new(); // reserving num_classes wastes 90%+ on typical schemas (measured)
        for (cid, bytes) in ct.members.iter().enumerate() {
            let t = dfa.next_state(e, bytes[0]);
            if dfa.is_quit_state(t) {
                return Err(quit_err());
            }
            if !dfa.is_dead_state(t) {
                let class = ByteClassId(u16::try_from(cid).expect("byte class count <= 256"));
                edges.push((class, t));
                if seen.insert(t) {
                    queue.push_back(t);
                }
            }
        }
        let accepting = dfa.is_match_state(dfa.next_eoi_state(e));
        raw.insert(e, RawState { edges, accepting });
    }

    let mut rev: FxHashMap<RaStateId, Vec<RaStateId>> = FxHashMap::default();
    for e in &disco_order {
        for (_c, t) in &raw[e].edges {
            rev.entry(*t).or_default().push(*e);
        }
    }
    let mut live: FxHashSet<RaStateId> = FxHashSet::default();
    let mut lq: VecDeque<RaStateId> = VecDeque::new();
    for e in &disco_order {
        if raw[e].accepting && live.insert(*e) {
            lq.push_back(*e);
        }
    }
    while let Some(t) = lq.pop_front() {
        if let Some(preds) = rev.get(&t) {
            for p in preds {
                if live.insert(*p) {
                    lq.push_back(*p);
                }
            }
        }
    }

    let mut final_remap: FxHashMap<RaStateId, StateId> = FxHashMap::default();
    for e in &disco_order {
        if live.contains(e) && !final_remap.contains_key(e) {
            let id = StateId(u32::try_from(final_remap.len() + 1).map_err(|_| limit_err())?);
            final_remap.insert(*e, id);
        }
    }
    let state_count = final_remap.len() + 1; // +1 for DEAD

    let cells = state_count.checked_mul(num_classes).ok_or_else(limit_err)?;
    if cells > MAX_TRANSITION_CELLS {
        return Err(limit_err());
    }

    let mut accepting = vec![false; state_count];
    let mut live_classes = vec![ClassMask::empty(); state_count];
    let mut row_offsets = Vec::new();
    row_offsets
        .try_reserve_exact(state_count + 1)
        .map_err(|_| limit_err())?;
    row_offsets.push(0u32); // DEAD (row 0) is never in disco_order/final_remap and always empty.
    let mut edges_flat = Vec::new();
    for e in &disco_order {
        let Some(&id) = final_remap.get(e) else {
            continue; // non-live folds into DEAD, no row.
        };
        let raw_state = &raw[e];
        let row = id.get() as usize;
        accepting[row] = raw_state.accepting;
        row_offsets.push(u32::try_from(edges_flat.len()).map_err(|_| limit_err())?);
        let mut lc = ClassMask::empty();
        for (class, t) in &raw_state.edges {
            let target = final_remap.get(t).copied().unwrap_or(DEAD);
            if target == DEAD {
                continue; // a co-accessibility-pruned edge is exactly "no edge": drop it.
            }
            if edges_flat.len() >= MAX_TRANSITION_CELLS {
                return Err(limit_err());
            }
            edges_flat.push((*class, target));
            lc.set(usize::from(class.get()));
        }
        live_classes[row] = lc;
    }
    row_offsets.push(u32::try_from(edges_flat.len()).map_err(|_| limit_err())?);
    let live_transition_count = edges_flat.len();

    let start = final_remap.get(&ext_start).copied().unwrap_or(DEAD);
    let transitions = Transitions {
        row_offsets: row_offsets.into_boxed_slice(),
        edges: edges_flat.into_boxed_slice(),
    };
    let accepting = accepting.into_boxed_slice();
    let residual_cardinality = residual_cardinalities(&transitions, &accepting, &ct)?;
    Ok(RefEngine {
        transitions,
        accepting,
        residual_cardinality,
        live_classes: live_classes.into_boxed_slice(),
        class_table: ct,
        class_count: num_classes,
        state_count,
        live_transition_count,
        start,
        dead: DEAD,
    })
}

fn quit_err() -> CompileError {
    CompileError::new(
        ErrorCode::Unsupported,
        Stage::L3,
        "byte DFA reached a quit state",
    )
}

fn limit_err() -> CompileError {
    let mut error = CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L3,
        "reference DFA state count exceeds the materialization limit",
    );
    error.limit = Some((
        crate::error::LimitKind::StateCount,
        MAX_MATERIALIZE_STATES.saturating_add(1),
        MAX_MATERIALIZE_STATES,
    ));
    error
}

impl RefEngine {
    #[must_use]
    pub(crate) fn class_of_byte(&self, b: u8) -> ByteClassId {
        self.class_table.of_byte[b as usize]
    }

    #[must_use]
    pub fn consume_token(&self, s: StateId, token_bytes: &[u8]) -> Option<StateId> {
        let mut cur = s;
        for &b in token_bytes {
            if cur == self.dead() {
                return None;
            }
            cur = self.step(cur, self.class_table.of_byte[b as usize]);
            if cur == self.dead() {
                return None;
            }
        }
        Some(cur)
    }

    fn can_consume_scalar_interval(&self, s: StateId, start: u32, end: u32) -> Option<bool> {
        if start > end || end > 0x10_ffff || self.is_dead(s) {
            return Some(false);
        }

        fn encoded(value: u32) -> Option<([u8; 4], usize)> {
            let scalar = char::from_u32(value)?;
            let mut storage = [0u8; 4];
            let len = scalar.encode_utf8(&mut storage).len();
            Some((storage, len))
        }

        let (lower, width) = encoded(start)?;
        let (upper, upper_width) = encoded(end)?;
        if width != upper_width {
            return None;
        }
        let mut frontier = Vec::new();
        frontier.try_reserve_exact(1).ok()?;
        frontier.push((s, true, true));

        for position in 0..width {
            let mut next = Vec::new();
            next.try_reserve(self.state_count.min(256).saturating_mul(4))
                .ok()?;
            for &(state, lower_tight, upper_tight) in &frontier {
                let lo = if lower_tight {
                    lower[position]
                } else if position == 0 {
                    lower[0]
                } else {
                    0x80
                };
                let hi = if upper_tight {
                    upper[position]
                } else if position == 0 {
                    upper[0]
                } else {
                    0xbf
                };
                for byte in lo..=hi {
                    let target = self.step(state, self.class_of_byte(byte));
                    if self.is_dead(target) {
                        continue;
                    }
                    let candidate = (
                        target,
                        lower_tight && byte == lower[position],
                        upper_tight && byte == upper[position],
                    );
                    if !next.contains(&candidate) {
                        next.try_reserve(1).ok()?;
                        next.push(candidate);
                    }
                }
            }
            if next.is_empty() {
                return Some(false);
            }
            frontier = next;
        }
        Some(true)
    }

    pub(crate) fn can_consume_supplementary_range(
        &self,
        s: StateId,
        start: u32,
        end: u32,
    ) -> Option<bool> {
        if start < 0x1_0000 || end > 0x10_ffff {
            return Some(false);
        }
        self.can_consume_scalar_interval(s, start, end)
    }

    pub(crate) fn can_consume_any_scalar(&self, s: StateId) -> Option<bool> {
        let intervals = [
            (0, 0x7f),
            (0x80, 0x7ff),
            (0x800, 0xd7ff),
            (0xe000, 0xffff),
            (0x1_0000, 0x10_ffff),
        ];
        let mut unknown = false;
        for (start, end) in intervals {
            match self.can_consume_scalar_interval(s, start, end) {
                Some(true) => return Some(true),
                Some(false) => {}
                None => unknown = true,
            }
        }
        (!unknown).then_some(false)
    }

    pub fn allowed_tokens(&self, s: StateId, vocab: &Vocabulary) -> Result<Bitmask, CompileError> {
        self.allowed_tokens_from_records(
            s,
            mask_width(vocab)?,
            vocab
                .tokens()
                .iter()
                .map(|(b, i)| (b.as_slice(), i.as_slice())),
        )
    }

    pub(crate) fn allowed_tokens_from_records<'a>(
        &self,
        s: StateId,
        mask_vocab_size: usize,
        records: impl Iterator<Item = (&'a [u8], &'a [u32])>,
    ) -> Result<Bitmask, CompileError> {
        let mut out = Bitmask::zeros(mask_vocab_size);
        for (bytes, ids) in records {
            if self.consume_token(s, bytes).is_some() {
                for &id in ids {
                    out.set(TokenId(id))?;
                }
            }
        }
        Ok(out)
    }

    #[must_use]
    pub fn eos_legal(&self, s: StateId) -> bool {
        self.is_accepting(s)
    }

    #[must_use]
    pub fn accepts(&self, instance: &[u8]) -> bool {
        let mut cur = self.start();
        for &b in instance {
            if cur == self.dead() {
                return false;
            }
            cur = self.step(cur, self.class_table.of_byte[b as usize]);
        }
        self.is_accepting(cur)
    }
}

pub(crate) fn mask_width(vocab: &Vocabulary) -> Result<usize, CompileError> {
    let max_token = vocab.tokens().values().flatten().copied().max();
    let hi = max_token.map_or(vocab.eos_token_id(), |m| m.max(vocab.eos_token_id()));
    usize::try_from(hi)
        .ok()
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            CompileError::new(
                ErrorCode::InternalLimitExceeded,
                Stage::L3,
                "max token id overflows the mask width",
            )
        })
}

#[cfg(test)]
type StatePair = (StateId, StateId);
#[cfg(test)]
type PairPredecessors = FxHashMap<StatePair, Option<(StatePair, u8)>>;

#[cfg(test)]
pub(crate) fn assert_exact_equivalence(old: &RefEngine, new: &RefEngine) {
    let start = (old.start(), new.start());
    let mut queue = VecDeque::from([start]);
    let mut predecessor = PairPredecessors::default();
    predecessor.insert(start, None);
    while let Some(pair) = queue.pop_front() {
        let classification_matches = old.is_dead(pair.0) == new.is_dead(pair.1)
            && old.can_continue(pair.0) == new.can_continue(pair.1)
            && old.is_accepting(pair.0) == new.is_accepting(pair.1);
        if !classification_matches {
            panic!(
                "automata differ after bytes {:?}: old(dead={}, continue={}, accept={}) new(dead={}, continue={}, accept={})",
                equivalence_counterexample(pair, &predecessor),
                old.is_dead(pair.0),
                old.can_continue(pair.0),
                old.is_accepting(pair.0),
                new.is_dead(pair.1),
                new.can_continue(pair.1),
                new.is_accepting(pair.1),
            );
        }
        for byte in 0u8..=255 {
            let next = (
                old.step(pair.0, old.class_table.of_byte[usize::from(byte)]),
                new.step(pair.1, new.class_table.of_byte[usize::from(byte)]),
            );
            if let std::collections::hash_map::Entry::Vacant(entry) = predecessor.entry(next) {
                entry.insert(Some((pair, byte)));
                queue.push_back(next);
            }
        }
    }
}

#[cfg(test)]
fn equivalence_counterexample(mut pair: StatePair, predecessor: &PairPredecessors) -> Vec<u8> {
    let mut bytes = Vec::new();
    while let Some(Some((previous, byte))) = predecessor.get(&pair) {
        bytes.push(*byte);
        pair = *previous;
    }
    bytes.reverse();
    bytes
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;
    use crate::mem_gate::BuildGate;

    #[test]
    #[ignore]
    fn engine_build_split_on_production_corpus_patterns() {
        use std::time::Instant;
        let patterns = [
            ("bool", r#"(?:true|false)"#),
            (
                "flat_object",
                r#"\{(?:"age":-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?,"name":"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"|"name":"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*","age":-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\}"#,
            ),
            (
                "bounded_array_of_objects",
                r#"\[(?:\{"x":-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\}(?:,\{"x":-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\}){0,4})\]"#,
            ),
        ];
        for (name, pattern) in patterns {
            let n = 200;
            let mut dense_ns = 0u128;
            let mut materialize_ns = 0u128;
            for _ in 0..n {
                let t0 = Instant::now();
                let dfa = dense::Builder::new()
                    .configure(
                        dense::Config::new()
                            .minimize(false)
                            .start_kind(StartKind::Anchored)
                            .match_kind(MatchKind::All),
                    )
                    .syntax(syntax::Config::new().unicode(false).utf8(false))
                    .build(pattern)
                    .unwrap();
                dense_ns += t0.elapsed().as_nanos();

                let t0 = Instant::now();
                bfs_materialize(&dfa).unwrap();
                materialize_ns += t0.elapsed().as_nanos();
            }
            println!(
                "{name:28} dense_dfa_build={:>8.1}us  bfs_materialize={:>8.1}us",
                dense_ns as f64 / n as f64 / 1000.0,
                materialize_ns as f64 / n as f64 / 1000.0,
            );
        }
    }

    #[test]
    fn class_table_covers_all_256_bytes_disjointly() {
        let e = build_from_regex("(cat|car|carbon)").unwrap();
        let ct = &e.class_table;
        let mut seen = [false; 256];
        for (cid, bytes) in ct.members.iter().enumerate() {
            for &b in bytes {
                assert!(!seen[b as usize], "byte {b} appears in two classes");
                seen[b as usize] = true;
                assert_eq!(ct.of_byte[b as usize].0 as usize, cid);
            }
        }
        assert!(seen.iter().all(|&x| x), "every byte 0..=255 is covered");
    }

    #[test]
    fn quit_byte_on_any_reachable_transition_is_a_structured_error() {
        let dfa = dense::Builder::new()
            .configure(
                dense::Config::new()
                    .minimize(false)
                    .start_kind(StartKind::Anchored)
                    .quit(b'x', true),
            )
            .syntax(syntax::Config::new().unicode(false).utf8(false))
            .build("a")
            .expect("dfa builds with a quit byte configured");
        let err = bfs_materialize(&dfa).expect_err("a reachable quit byte must be rejected");
        assert_eq!(err.code, ErrorCode::Unsupported);
    }

    #[test]
    fn dfa_size_limit_exceeded_is_a_typed_internal_limit_not_malformed() {
        let err = build_from_regex_with_limits("[a-z]{50}", 1, 1 << 20, 1 << 20, NEST_LIMIT)
            .expect_err("1 byte cannot hold any real DFA");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
        assert_eq!(
            err.stage,
            Stage::L3,
            "automaton admission must report the compile stage"
        );
    }

    #[test]
    fn determinize_size_limit_exceeded_is_a_typed_internal_limit_not_malformed() {
        let err = build_from_regex_with_limits("[a-z]{50}", 1 << 30, 1, 1 << 20, NEST_LIMIT)
            .expect_err("1 byte cannot hold any determinization scratch");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
        assert_eq!(err.stage, Stage::L3);
    }

    #[test]
    fn nfa_size_limit_exceeded_is_a_typed_internal_limit_not_malformed() {
        let err = build_from_regex_with_limits("[a-z]{50}", 1 << 30, 1 << 30, 1, NEST_LIMIT)
            .expect_err("1 byte cannot hold any Thompson NFA");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
        assert_eq!(err.stage, Stage::L3);
    }

    #[test]
    fn wrapped_nfa_build_error_under_a_realistic_size_limit_is_still_typed_internal_limit() {
        let pattern = format!(".{{1,{}}}", 1 << 20);
        let err = build_from_regex_with_limits(&pattern, 1 << 30, 1 << 30, 1 << 10, NEST_LIMIT)
            .expect_err("a million-repetition class must exceed a 1KiB NFA heap");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    #[ignore = "builds a ~65k-state DFA; run manually with --ignored --nocapture"]
    fn discovery_state_count_cap_rejects_a_pattern_with_too_many_states() {
        let pattern = format!("a{{1,{}}}", MAX_MATERIALIZE_STATES + 10);
        let err = build_from_regex(&pattern).expect_err("must exceed the discovery state cap");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
        assert_eq!(err.stage, Stage::L3);
    }

    #[test]
    fn automaton_gate_is_charged_and_the_permit_is_released_on_success() {
        let gate = BuildGate::new(4, 1 << 20, Stage::L3);
        let before_peak = gate.peak_bytes();
        {
            let _permit = gate.acquire(1024).unwrap();
            assert_eq!(gate.active_bytes(), 1024);
        }
        assert_eq!(gate.active_bytes(), 0, "the permit must release on drop");
        assert!(gate.peak_bytes() >= before_peak.max(1024));
    }

    #[test]
    fn nest_limit_exceeded_is_a_structured_error() {
        let deep = "(?:".repeat(10) + "a" + &")?".repeat(10);
        let err = build_from_regex_with_limits(&deep, 1 << 30, 1 << 30, 1 << 30, 5)
            .expect_err("10 nested groups must exceed a nest limit of 5");
        assert!(matches!(
            err.code,
            ErrorCode::Malformed | ErrorCode::InternalLimitExceeded
        ));
    }

    #[test]
    fn dead_self_loops_on_all_256_bytes_and_never_accepts() {
        let e = build_from_regex("true").unwrap();
        let dead = e.dead();
        assert!(!e.is_accepting(dead));
        for b in 0u8..=255 {
            let class = e.class_table.of_byte[b as usize];
            assert_eq!(
                e.step(dead, class),
                dead,
                "byte {b} from DEAD must stay DEAD"
            );
        }
    }

    #[test]
    fn dead_row_has_zero_stored_edges() {
        let e = build_from_regex("true").unwrap();
        assert_eq!(e.transitions.row_offsets[0], e.transitions.row_offsets[1]);
    }

    #[test]
    fn residual_cardinality_is_exact_for_finite_languages_and_marks_infinite_ones() {
        let finite = build_from_regex("[a-c]{1,2}").unwrap();
        assert_eq!(finite.residual_cardinality(finite.start()), 12);
        let after_a = finite.consume_token(finite.start(), b"a").unwrap();
        assert_eq!(finite.residual_cardinality(after_a), 4);
        assert_eq!(finite.residual_cardinality(finite.dead()), 0);

        let infinite = build_from_regex("[a-z]+").unwrap();
        assert_eq!(infinite.residual_cardinality(infinite.start()), u64::MAX);
        let after_a = infinite.consume_token(infinite.start(), b"a").unwrap();
        assert_eq!(infinite.residual_cardinality(after_a), u64::MAX);
    }

    #[test]
    fn wide_out_degree_row_accepts_every_member_at_first_middle_last_and_rejects_gaps() {
        let alphabet: Vec<u8> = (1u8..=120).collect();
        let pattern = format!(
            "(?:{})",
            alphabet
                .iter()
                .map(|&b| format!("\\x{b:02x}"))
                .collect::<Vec<_>>()
                .join("|")
        );
        let e = build_from_regex(&pattern).unwrap();
        for &b in &alphabet {
            assert!(e.accepts(&[b]), "byte {b:#04x} must be accepted");
        }
        for b in [0u8, 121, 200, 255] {
            assert!(
                !e.accepts(&[b]),
                "byte {b:#04x} was never a branch and must reject"
            );
        }
    }

    #[test]
    fn independent_walk_against_the_raw_regex_automata_dfa_agrees_on_random_inputs() {
        let patterns = [
            "(cat|car|carbon)",
            "\\{\"a\":(?:true|false)\\}",
            "[a-z]{1,5}",
            "(?:0|[1-9][0-9]{0,4})",
            "(?:foo|foobar|foobaz|bar)",
        ];
        for pattern in patterns {
            let raw_dfa = dense::Builder::new()
                .configure(
                    dense::Config::new()
                        .minimize(false)
                        .start_kind(StartKind::Anchored)
                        .match_kind(MatchKind::All),
                )
                .syntax(syntax::Config::new().unicode(false).utf8(false))
                .build(pattern)
                .unwrap();
            let engine = build_from_regex(pattern).unwrap();
            let start = raw_dfa
                .start_state(&start::Config::new().anchored(Anchored::Yes))
                .unwrap();
            let mut rng = 0x2545F4914F6CDD1Du64;
            let mut next = || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            for _ in 0..200 {
                let len = (next() % 6) as usize;
                let bytes: Vec<u8> = (0..len).map(|_| (next() % 256) as u8).collect();
                let mut raw_state = start;
                let mut raw_dead = false;
                for &b in &bytes {
                    raw_state = raw_dfa.next_state(raw_state, b);
                    if raw_dfa.is_dead_state(raw_state) {
                        raw_dead = true;
                        break;
                    }
                }
                let raw_accepts =
                    !raw_dead && raw_dfa.is_match_state(raw_dfa.next_eoi_state(raw_state));
                assert_eq!(
                    engine.accepts(&bytes),
                    raw_accepts,
                    "pattern {pattern:?} bytes {bytes:?}"
                );
            }
        }
    }

    #[test]
    fn deterministic_build_is_byte_identical() {
        let a = build_from_regex("(cat|car|carbon)").unwrap();
        let b = build_from_regex("(cat|car|carbon)").unwrap();
        assert_eq!(a.state_count(), b.state_count());
        assert_eq!(a.start(), b.start());
        assert_eq!(a.class_table.members, b.class_table.members);
        let num_classes = a.class_table.members.len();
        for raw in 0..a.state_count() {
            let s = StateId(raw as u32);
            assert_eq!(a.is_accepting(s), b.is_accepting(s));
            for c in 0..num_classes {
                let class = ByteClassId(u16::try_from(c).unwrap());
                assert_eq!(a.step(s, class), b.step(s, class));
            }
        }
    }

    #[test]
    fn every_live_state_reaches_acceptance() {
        let e = build_from_regex("(cat|car|carbon)").unwrap();
        let num_classes = e.class_table.members.len();
        for start in 1..e.state_count() {
            let mut seen = vec![false; e.state_count()];
            let mut q = VecDeque::from([start]);
            seen[start] = true;
            let mut reaches_accept = false;
            while let Some(s) = q.pop_front() {
                if e.is_accepting(StateId(s as u32)) {
                    reaches_accept = true;
                    break;
                }
                for c in 0..num_classes {
                    let class = ByteClassId(u16::try_from(c).unwrap());
                    let tgt = e.step(StateId(s as u32), class).get() as usize;
                    if tgt != 0 && !seen[tgt] {
                        seen[tgt] = true;
                        q.push_back(tgt);
                    }
                }
            }
            assert!(
                reaches_accept,
                "state {start} cannot reach an accepting state"
            );
        }
    }

    #[test]
    fn empty_language_collapses_to_dead_start() {
        let e = build_from_regex("[a&&b]").unwrap();
        assert_eq!(e.start(), e.dead());
    }

    #[test]
    fn empty_pattern_matches_only_the_empty_string() {
        let e = build_from_regex("").unwrap();
        assert!(e.is_accepting(e.start()));
        assert!(!build_from_regex("").unwrap().accepts(b"a"));
    }

    #[test]
    fn escaped_special_bytes_match_the_literal_byte() {
        let e = build_from_regex(r"a\.b").unwrap();
        assert!(e.accepts(b"a.b"));
        assert!(!e.accepts(b"axb"));
    }

    #[test]
    fn anchored_matching_rejects_a_value_embedded_in_trailing_bytes() {
        let e = build_from_regex("cat").unwrap(); // anchored + EOI-only: not a substring search
        assert!(e.accepts(b"cat"));
        assert!(!e.accepts(b"cats"));
        assert!(!e.accepts(b"xcat"));
        assert!(!e.accepts(b"xcaty"));
    }

    mod crossref_vs_outlines_core_index_tests {
        use super::*;

        fn accepts(pattern: &str, s: &str) -> bool {
            build_from_regex(pattern).unwrap().accepts(s.as_bytes())
        }

        #[test]
        fn matches_index_from_regex_language() {
            let p = "0|[1-9][0-9]*";
            assert!(accepts(p, "0"));
            assert!(accepts(p, "2"));
            assert!(accepts(p, "20"));
            assert!(!accepts(p, "02"));
            assert!(!accepts(p, "1a"));
            assert!(!accepts(p, ""));
        }

        #[test]
        fn matches_index_from_regex_completeness_language() {
            let p = "(ac|[^a])+";
            assert!(accepts(p, "acac"));
            assert!(accepts(p, "b"));
            assert!(!accepts(p, "a"));
            assert!(!accepts(p, ""));
        }

        #[test]
        fn multibyte_char_class_range_is_a_structured_unicode_mode_error() {
            let p = "😇| [😈-😍][😇-😎]*";
            let code = build_from_regex(p).unwrap_err().code;
            assert!(matches!(
                code,
                crate::error::ErrorCode::Malformed | crate::error::ErrorCode::Unsupported
            ));
        }

        #[test]
        fn multibyte_literal_alternation_still_works_byte_for_byte() {
            let p = "😇|😈";
            assert!(accepts(p, "😇"));
            assert!(accepts(p, "😈"));
            assert!(!accepts(p, "😍"));
            assert!(!accepts(p, ""));
        }

        #[test]
        fn multibyte_bracket_set_of_discrete_chars_also_fails() {
            assert!(build_from_regex("[😈😇]").is_err());
        }

        #[test]
        fn ascii_bracket_range_is_unaffected() {
            assert!(accepts("[a-c]", "b"));
            assert!(!accepts("[a-c]", "d"));
        }

        #[test]
        fn matches_index_from_regex_initial_in_allowed_language() {
            let p = "`\n(\\.\n)?`\n";
            assert!(accepts(p, "`\n`\n"));
            assert!(accepts(p, "`\n.\n`\n"));
            assert!(!accepts(p, "`\n"));
            assert!(!accepts(p, ""));
        }

        #[test]
        fn matches_index_incompatible_vocabulary_error_regex() {
            let p = "0 1";
            assert!(accepts(p, "0 1"));
            assert!(!accepts(p, "01"));
            assert!(!accepts(p, "0  1"));
        }
    }

    #[test]
    fn discovery_capacity_utilization_for_representative_patterns() {
        for (name, pattern) in [
            ("dense_alternation", "(cat|car|carbon|dog|doghouse|do)"),
            ("dense_repetition", "[a-z0-9]+"),
            (
                "sparse_literal_chain",
                "hello world this is a long fixed literal string",
            ),
            (
                "sparse_json_object",
                r#"\{"type":"(cat|dog)","age":[0-9]{1,3}\}"#,
            ),
        ] {
            let e = build_from_regex(pattern).unwrap();
            let classes = e.class_table.members.len();
            let live_states = e.state_count() - 1; // exclude DEAD
            let live_transitions = e.live_transition_count();
            let avg_out_degree = f64::from(u32::try_from(live_transitions).unwrap())
                / f64::from(u32::try_from(live_states.max(1)).unwrap());
            let utilization = avg_out_degree / f64::from(u32::try_from(classes).unwrap());
            eprintln!(
                "{name}: classes={classes} states={live_states} avg_out_degree={avg_out_degree:.1} \
                 utilization={utilization:.2}"
            );
        }
    }
}
