//! Bounded deterministic byte graph used by composed automata before `RefEngine` materialization.

use crate::error::{CompileError, ErrorCode, Stage};
use crate::mem_gate::acquire_automaton_build_bytes;
use crate::primitives::StateId;
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

use super::byte_dfa::RefEngine;

pub(crate) const MAX_GRAPH_STATES: usize = 1 << 16;
pub(crate) const MAX_GRAPH_EDGES: usize = 1 << 20;
const GRAPH_BUILD_BUDGET: usize = 64 << 20;
pub(crate) type ByteState = (bool, Vec<(u8, u32)>);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GraphStateId(pub(crate) u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ByteClass {
    words: [u64; 4],
}

impl ByteClass {
    pub(crate) fn singleton(byte: u8) -> Self {
        let mut words = [0u64; 4];
        let byte = usize::from(byte);
        words[byte >> 6] = 1u64 << (byte & 63);
        Self { words }
    }

    pub(crate) fn contains(self, byte: u8) -> bool {
        let byte = usize::from(byte);
        self.words[byte >> 6] & (1u64 << (byte & 63)) != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GraphState {
    pub(crate) edge_start: u32,
    pub(crate) edge_len: u32,
    pub(crate) accepting: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GraphEdge {
    pub(crate) class: ByteClass,
    pub(crate) target: GraphStateId,
}

#[derive(Debug)]
pub(crate) struct AutomatonGraph {
    pub(crate) states: Vec<GraphState>,
    pub(crate) edges: Vec<GraphEdge>,
    pub(crate) start: GraphStateId,
}

impl AutomatonGraph {
    pub(crate) fn from_byte_states(states: &[ByteState], start: u32) -> Result<Self, CompileError> {
        let _permit = acquire_automaton_build_bytes(graph_retained_estimate(states)?)?;
        if states.is_empty() || usize::try_from(start).map_or(true, |s| s >= states.len()) {
            return Err(graph_error("graph start state is out of range"));
        }
        if states.len() > MAX_GRAPH_STATES {
            return Err(graph_limit("graph state cap"));
        }
        let edge_count = states
            .iter()
            .try_fold(0usize, |total, (_, edges)| total.checked_add(edges.len()));
        let Some(edge_count) = edge_count else {
            return Err(graph_limit("graph edge count overflow"));
        };
        if edge_count > MAX_GRAPH_EDGES {
            return Err(graph_limit("graph edge cap"));
        }

        let mut graph_states = Vec::new();
        graph_states
            .try_reserve_exact(states.len())
            .map_err(|_| graph_limit("graph state allocation"))?;
        let mut graph_edges = Vec::new();
        graph_edges
            .try_reserve_exact(edge_count)
            .map_err(|_| graph_limit("graph edge allocation"))?;
        for (accepting, edges) in states {
            let edge_start =
                u32::try_from(graph_edges.len()).map_err(|_| graph_limit("graph edge index"))?;
            let mut seen = [false; 256];
            for &(byte, target) in edges {
                let target_index = usize::try_from(target)
                    .map_err(|_| graph_error("graph target is out of range"))?;
                if target_index >= states.len() || seen[usize::from(byte)] {
                    return Err(graph_error("graph is not deterministic"));
                }
                seen[usize::from(byte)] = true;
                graph_edges.push(GraphEdge {
                    class: ByteClass::singleton(byte),
                    target: GraphStateId(target),
                });
            }
            graph_states.push(GraphState {
                edge_start,
                edge_len: u32::try_from(edges.len())
                    .map_err(|_| graph_limit("graph row length"))?,
                accepting: *accepting,
            });
        }
        Ok(Self {
            states: graph_states,
            edges: graph_edges,
            start: GraphStateId(start),
        })
    }

    pub(crate) fn transition(&self, state: usize, byte: u8) -> Option<usize> {
        let row = self.states.get(state)?;
        let lo = usize::try_from(row.edge_start).ok()?;
        let len = usize::try_from(row.edge_len).ok()?;
        self.edges
            .get(lo..lo.checked_add(len)?)?
            .iter()
            .find_map(|edge| {
                edge.class
                    .contains(byte)
                    .then(|| usize::try_from(edge.target.0).ok())
                    .flatten()
            })
    }
}

pub(crate) fn string_set(values: &[&str]) -> Result<AutomatonGraph, CompileError> {
    let _permit = acquire_automaton_build_bytes(GRAPH_BUILD_BUDGET)?;
    let mut decoded = DecodedTrie::new()?;
    for value in values {
        decoded.insert(value)?;
    }
    decoded.into_json_graph()
}

pub(crate) fn byte_string_set(values: &[Vec<u8>]) -> Result<AutomatonGraph, CompileError> {
    let _permit = acquire_automaton_build_bytes(GRAPH_BUILD_BUDGET)?;
    let mut trie = ByteTrie::new()?;
    for value in values {
        trie.insert(value)?;
    }
    trie.into_graph()
}

#[derive(Debug)]
struct ByteTrieNode {
    edges: Vec<(u8, u32)>,
    accepting: bool,
}

#[derive(Debug)]
struct ByteTrie {
    nodes: Vec<ByteTrieNode>,
}

impl ByteTrie {
    fn new() -> Result<Self, CompileError> {
        let mut nodes = Vec::new();
        nodes
            .try_reserve(1)
            .map_err(|_| graph_limit("byte trie root allocation"))?;
        nodes.push(ByteTrieNode {
            edges: Vec::new(),
            accepting: false,
        });
        Ok(Self { nodes })
    }

    fn insert(&mut self, value: &[u8]) -> Result<(), CompileError> {
        let mut node = 0usize;
        for &byte in value {
            node = self.next_or_insert(node, byte)?;
        }
        self.nodes[node].accepting = true;
        Ok(())
    }

    fn next_or_insert(&mut self, node: usize, byte: u8) -> Result<usize, CompileError> {
        if let Some((_, target)) = self.nodes[node]
            .edges
            .iter()
            .find(|(candidate, _)| *candidate == byte)
        {
            return usize::try_from(*target).map_err(|_| graph_limit("byte trie state id"));
        }
        if self.nodes.len() >= MAX_GRAPH_STATES {
            return Err(graph_limit("byte trie state cap"));
        }
        let target =
            u32::try_from(self.nodes.len()).map_err(|_| graph_limit("byte trie state id"))?;
        self.nodes
            .try_reserve(1)
            .map_err(|_| graph_limit("byte trie state allocation"))?;
        self.nodes.push(ByteTrieNode {
            edges: Vec::new(),
            accepting: false,
        });
        self.nodes[node]
            .edges
            .try_reserve(1)
            .map_err(|_| graph_limit("byte trie edge allocation"))?;
        self.nodes[node].edges.push((byte, target));
        usize::try_from(target).map_err(|_| graph_limit("byte trie state id"))
    }

    fn into_graph(self) -> Result<AutomatonGraph, CompileError> {
        let mut states = Vec::new();
        states
            .try_reserve_exact(self.nodes.len())
            .map_err(|_| graph_limit("byte trie graph allocation"))?;
        for node in self.nodes {
            states.push((node.accepting, node.edges));
        }
        AutomatonGraph::from_byte_states(&states, 0)
    }
}

#[derive(Debug)]
struct DecodedNode {
    edges: Vec<(char, u32)>,
    accepting: bool,
}

#[derive(Debug)]
struct DecodedTrie {
    nodes: Vec<DecodedNode>,
}

impl DecodedTrie {
    fn new() -> Result<Self, CompileError> {
        let mut nodes = Vec::new();
        nodes
            .try_reserve(1)
            .map_err(|_| graph_limit("string trie root allocation"))?;
        nodes.push(DecodedNode {
            edges: Vec::new(),
            accepting: false,
        });
        Ok(Self { nodes })
    }

    fn insert(&mut self, value: &str) -> Result<(), CompileError> {
        let mut node = 0usize;
        for scalar in value.chars() {
            let next = if let Some((_, target)) = self.nodes[node]
                .edges
                .iter()
                .find(|(candidate, _)| *candidate == scalar)
            {
                usize::try_from(*target).map_err(|_| graph_limit("string trie state id"))?
            } else {
                if self.nodes.len() >= MAX_GRAPH_STATES {
                    return Err(graph_limit("string trie state cap"));
                }
                let target = u32::try_from(self.nodes.len())
                    .map_err(|_| graph_limit("string trie state id"))?;
                self.nodes
                    .try_reserve(1)
                    .map_err(|_| graph_limit("string trie state allocation"))?;
                self.nodes.push(DecodedNode {
                    edges: Vec::new(),
                    accepting: false,
                });
                self.nodes[node]
                    .edges
                    .try_reserve(1)
                    .map_err(|_| graph_limit("string trie edge allocation"))?;
                self.nodes[node].edges.push((scalar, target));
                usize::try_from(target).map_err(|_| graph_limit("string trie state id"))?
            };
            node = next;
        }
        self.nodes[node].accepting = true;
        Ok(())
    }

    fn into_json_graph(self) -> Result<AutomatonGraph, CompileError> {
        let mut states = Vec::new();
        states
            .try_reserve(self.nodes.len().saturating_add(2))
            .map_err(|_| graph_limit("JSON string graph allocation"))?;
        states.push((false, Vec::new()));
        for _ in &self.nodes {
            states.push((false, Vec::new()));
        }
        states.push((true, Vec::new()));
        add_path(&mut states, 0, 1, b"\"")?;
        let accept =
            u32::try_from(states.len() - 1).map_err(|_| graph_limit("JSON string accept id"))?;
        for (source, node) in self.nodes.iter().enumerate() {
            let source =
                u32::try_from(source + 1).map_err(|_| graph_limit("JSON string state id"))?;
            if node.accepting {
                add_path(&mut states, source, accept, b"\"")?;
            }
            for &(scalar, target) in &node.edges {
                let target = target
                    .checked_add(1)
                    .ok_or_else(|| graph_limit("JSON string target id"))?;
                add_scalar_spellings(&mut states, source, target, scalar)?;
            }
        }
        AutomatonGraph::from_byte_states(&states, 0)
    }
}

fn add_scalar_spellings(
    states: &mut Vec<(bool, Vec<(u8, u32)>)>,
    source: u32,
    target: u32,
    scalar: char,
) -> Result<(), CompileError> {
    if u32::from(scalar) >= 0x20 && scalar != '"' && scalar != '\\' {
        let mut utf8 = [0u8; 4];
        add_path(
            states,
            source,
            target,
            scalar.encode_utf8(&mut utf8).as_bytes(),
        )?;
    }
    match scalar {
        '"' => add_path(states, source, target, br#"\""#)?,
        '\\' => add_path(states, source, target, br"\\")?,
        '/' => add_path(states, source, target, br"\/")?,
        '\u{08}' => add_path(states, source, target, br"\b")?,
        '\u{0c}' => add_path(states, source, target, br"\f")?,
        '\n' => add_path(states, source, target, br"\n")?,
        '\r' => add_path(states, source, target, br"\r")?,
        '\t' => add_path(states, source, target, br"\t")?,
        _ => {}
    }
    let code = u32::from(scalar);
    if code <= 0xffff {
        add_unicode_escape(
            states,
            source,
            target,
            u16::try_from(code).map_err(|_| graph_limit("scalar code"))?,
        )?;
    } else {
        let adjusted = code - 0x1_0000;
        let high =
            0xd800u16 + u16::try_from(adjusted >> 10).map_err(|_| graph_limit("surrogate"))?;
        let low =
            0xdc00u16 + u16::try_from(adjusted & 0x3ff).map_err(|_| graph_limit("surrogate"))?;
        let high_paths = unicode_escape_paths(high)?;
        let low_paths = unicode_escape_paths(low)?;
        for high_path in &high_paths {
            for low_path in &low_paths {
                let mut path = Vec::new();
                path.try_reserve_exact(high_path.len().saturating_add(low_path.len()))
                    .map_err(|_| graph_limit("surrogate path allocation"))?;
                path.extend_from_slice(high_path);
                path.extend_from_slice(low_path);
                add_path(states, source, target, &path)?;
            }
        }
    }
    Ok(())
}

fn add_unicode_escape(
    states: &mut Vec<(bool, Vec<(u8, u32)>)>,
    source: u32,
    target: u32,
    code: u16,
) -> Result<(), CompileError> {
    for path in unicode_escape_paths(code)? {
        add_path(states, source, target, &path)?;
    }
    Ok(())
}

fn unicode_escape_paths(code: u16) -> Result<Vec<Vec<u8>>, CompileError> {
    let digits = format!("{code:04x}");
    let mut paths = vec![Vec::from(br"\u")];
    for byte in digits.bytes() {
        if byte.is_ascii_alphabetic() {
            let existing = paths.len();
            paths
                .try_reserve(existing)
                .map_err(|_| graph_limit("unicode escape alternatives"))?;
            for index in 0..existing {
                let mut upper = Vec::new();
                upper
                    .try_reserve_exact(paths[index].len().saturating_add(1))
                    .map_err(|_| graph_limit("unicode escape path allocation"))?;
                upper.extend_from_slice(&paths[index]);
                upper.push(byte.to_ascii_uppercase());
                paths.push(upper);
                paths[index].push(byte);
            }
        } else {
            for path in &mut paths {
                path.push(byte);
            }
        }
    }
    Ok(paths)
}

fn add_path(
    states: &mut Vec<(bool, Vec<(u8, u32)>)>,
    source: u32,
    target: u32,
    bytes: &[u8],
) -> Result<(), CompileError> {
    let mut current = source;
    for (index, &byte) in bytes.iter().enumerate() {
        let last = index + 1 == bytes.len();
        let current_index = usize::try_from(current).map_err(|_| graph_limit("path state id"))?;
        if let Some((_, existing)) = states[current_index]
            .1
            .iter()
            .find(|(candidate, _)| *candidate == byte)
        {
            if last && *existing != target {
                return Err(graph_error("JSON scalar spellings are not deterministic"));
            }
            current = *existing;
            continue;
        }
        let next = if last {
            target
        } else {
            new_byte_state(states)?
        };
        states[current_index]
            .1
            .try_reserve(1)
            .map_err(|_| graph_limit("path edge allocation"))?;
        states[current_index].1.push((byte, next));
        current = next;
    }
    Ok(())
}

fn new_byte_state(states: &mut Vec<(bool, Vec<(u8, u32)>)>) -> Result<u32, CompileError> {
    if states.len() >= MAX_GRAPH_STATES {
        return Err(graph_limit("JSON string graph state cap"));
    }
    states
        .try_reserve(1)
        .map_err(|_| graph_limit("JSON string graph state allocation"))?;
    let id = u32::try_from(states.len()).map_err(|_| graph_limit("JSON string graph state id"))?;
    states.push((false, Vec::new()));
    Ok(id)
}

#[derive(Debug)]
struct NfaState {
    edges: Vec<(u8, u32)>,
    epsilon: Vec<u32>,
    accepting: bool,
}

#[derive(Debug)]
struct ByteNfa {
    states: Vec<NfaState>,
    start: u32,
}

pub(crate) fn concatenate(engines: &[&RefEngine]) -> Result<AutomatonGraph, CompileError> {
    let _permit = acquire_automaton_build_bytes(GRAPH_BUILD_BUDGET)?;
    if engines.is_empty() {
        return AutomatonGraph::from_byte_states(&[(true, Vec::new())], 0);
    }
    if engines.iter().any(|engine| engine.start() == engine.dead()) {
        return AutomatonGraph::from_byte_states(&[(false, Vec::new())], 0);
    }
    admit_engine_copies(engines.iter().copied())?;
    let mut nfa = ByteNfa {
        states: Vec::new(),
        start: 0,
    };
    let mut starts = Vec::new();
    starts
        .try_reserve_exact(engines.len())
        .map_err(|_| graph_limit("concatenation starts allocation"))?;
    let mut accepting_runs: Vec<Vec<u32>> = Vec::new();
    accepting_runs
        .try_reserve_exact(engines.len())
        .map_err(|_| graph_limit("concatenation accepts allocation"))?;
    for engine in engines {
        let (start, accepting) = append_engine(&mut nfa.states, engine)?;
        starts.push(start);
        accepting_runs.push(accepting);
    }
    nfa.start = starts[0];
    for index in 0..engines.len().saturating_sub(1) {
        for &state in &accepting_runs[index] {
            let state = usize::try_from(state).map_err(|_| graph_limit("NFA state index"))?;
            nfa.states[state]
                .epsilon
                .try_reserve(1)
                .map_err(|_| graph_limit("NFA epsilon allocation"))?;
            nfa.states[state].epsilon.push(starts[index + 1]);
            nfa.states[state].accepting = false;
        }
    }
    nfa.determinize()
}

pub(crate) fn union(engines: &[&RefEngine]) -> Result<AutomatonGraph, CompileError> {
    let _permit = acquire_automaton_build_bytes(GRAPH_BUILD_BUDGET)?;
    if engines.is_empty() {
        return AutomatonGraph::from_byte_states(&[(false, Vec::new())], 0);
    }
    admit_engine_copies(engines.iter().copied())?;
    let mut nfa = ByteNfa {
        states: Vec::new(),
        start: 0,
    };
    nfa.states
        .try_reserve(1)
        .map_err(|_| graph_limit("union start allocation"))?;
    nfa.states.push(NfaState {
        edges: Vec::new(),
        epsilon: Vec::new(),
        accepting: false,
    });
    for engine in engines {
        if engine.start() == engine.dead() {
            continue;
        }
        let (start, _) = append_engine(&mut nfa.states, engine)?;
        nfa.states[0]
            .epsilon
            .try_reserve(1)
            .map_err(|_| graph_limit("union epsilon allocation"))?;
        nfa.states[0].epsilon.push(start);
    }
    nfa.determinize()
}

pub(crate) fn repeat(
    engine: &RefEngine,
    min: u32,
    max: Option<u32>,
) -> Result<AutomatonGraph, CompileError> {
    let _permit = acquire_automaton_build_bytes(GRAPH_BUILD_BUDGET)?;
    if max.is_some_and(|max| max < min) {
        return Err(graph_error("repeat maximum is below minimum"));
    }
    if max == Some(0) {
        return AutomatonGraph::from_byte_states(&[(true, Vec::new())], 0);
    }
    if engine.start() == engine.dead() {
        let accepts_empty = min == 0;
        return AutomatonGraph::from_byte_states(&[(accepts_empty, Vec::new())], 0);
    }
    let copies = max.unwrap_or_else(|| min.saturating_add(1)).max(1);
    let copies = usize::try_from(copies).map_err(|_| graph_limit("repeat count"))?;
    admit_engine_copies(std::iter::repeat_n(engine, copies))?;
    let mut nfa = repeat_nfa_start(min == 0)?;
    let mut starts = Vec::new();
    let mut accepts = Vec::new();
    starts
        .try_reserve_exact(copies)
        .map_err(|_| graph_limit("repeat starts allocation"))?;
    accepts
        .try_reserve_exact(copies)
        .map_err(|_| graph_limit("repeat accepts allocation"))?;
    for _ in 0..copies {
        let (start, run) = append_engine(&mut nfa.states, engine)?;
        starts.push(start);
        accepts.push(run);
    }
    nfa.states[0].epsilon.push(starts[0]);
    link_repeat_runs(&mut nfa, &starts, &accepts, min, max)?;
    nfa.determinize()
}

fn repeat_nfa_start(accepting: bool) -> Result<ByteNfa, CompileError> {
    let mut states = Vec::new();
    states
        .try_reserve(1)
        .map_err(|_| graph_limit("repeat start allocation"))?;
    let mut epsilon = Vec::new();
    epsilon
        .try_reserve(1)
        .map_err(|_| graph_limit("repeat epsilon allocation"))?;
    states.push(NfaState {
        edges: Vec::new(),
        epsilon,
        accepting,
    });
    Ok(ByteNfa { states, start: 0 })
}

fn admit_engine_copies<'a>(
    mut engines: impl Iterator<Item = &'a RefEngine>,
) -> Result<(), CompileError> {
    let (states, byte_edges) = engines
        .try_fold((0usize, 0usize), |(states, edges), engine| {
            let engine_states = engine.state_count().saturating_sub(1);
            let states = states.checked_add(engine_states)?;
            let edges = edges.checked_add(engine.raw_byte_transition_count()?)?;
            Some((states, edges))
        })
        .ok_or_else(|| graph_limit("NFA admission overflow"))?;
    if states > MAX_GRAPH_STATES || byte_edges > MAX_GRAPH_EDGES {
        return Err(graph_limit("NFA construction cap"));
    }
    Ok(())
}

fn link_repeat_runs(
    nfa: &mut ByteNfa,
    starts: &[u32],
    accepts: &[Vec<u32>],
    min: u32,
    max: Option<u32>,
) -> Result<(), CompileError> {
    for (index, run) in accepts.iter().enumerate() {
        let completed = u32::try_from(index + 1).map_err(|_| graph_limit("repeat count"))?;
        for &state_id in run {
            let state = nfa
                .states
                .get_mut(usize::try_from(state_id).map_err(|_| graph_limit("NFA state id"))?)
                .ok_or_else(|| graph_limit("NFA state id"))?;
            state.accepting = completed >= min;
            let next = if index + 1 < starts.len() {
                Some(starts[index + 1])
            } else if max.is_none() {
                starts.last().copied()
            } else {
                None
            };
            if let Some(next) = next {
                state
                    .epsilon
                    .try_reserve(1)
                    .map_err(|_| graph_limit("repeat epsilon growth"))?;
                state.epsilon.push(next);
            }
        }
    }
    Ok(())
}

fn append_engine(
    states: &mut Vec<NfaState>,
    engine: &RefEngine,
) -> Result<(u32, Vec<u32>), CompileError> {
    let base = u32::try_from(states.len()).map_err(|_| graph_limit("NFA state index"))?;
    let live_count = engine.state_count().saturating_sub(1);
    let new_len = states
        .len()
        .checked_add(live_count)
        .ok_or_else(|| graph_limit("NFA state count"))?;
    if new_len > MAX_GRAPH_STATES {
        return Err(graph_limit("NFA state cap"));
    }
    states
        .try_reserve_exact(live_count)
        .map_err(|_| graph_limit("NFA state allocation"))?;
    let mut accepting = Vec::new();
    accepting
        .try_reserve(live_count)
        .map_err(|_| graph_limit("NFA accepting allocation"))?;
    for raw in 1..engine.state_count() {
        let source = StateId(u32::try_from(raw).map_err(|_| graph_limit("engine state id"))?);
        let mut edges = Vec::new();
        for byte in 0u8..=255 {
            let target = engine.step(source, engine.class_table.of_byte[usize::from(byte)]);
            if target != engine.dead() {
                let offset = target
                    .get()
                    .checked_sub(1)
                    .and_then(|target| base.checked_add(target))
                    .ok_or_else(|| graph_limit("NFA target id"))?;
                edges
                    .try_reserve(1)
                    .map_err(|_| graph_limit("NFA edge growth"))?;
                edges.push((byte, offset));
            }
        }
        let is_accepting = engine.is_accepting(source);
        if is_accepting {
            accepting.push(
                base.checked_add(u32::try_from(raw - 1).map_err(|_| graph_limit("NFA state id"))?)
                    .ok_or_else(|| graph_limit("NFA state id"))?,
            );
        }
        states.push(NfaState {
            edges,
            epsilon: Vec::new(),
            accepting: is_accepting,
        });
    }
    let start = base
        .checked_add(engine.start().get() - 1)
        .ok_or_else(|| graph_limit("NFA start id"))?;
    Ok((start, accepting))
}

impl ByteNfa {
    fn determinize(&self) -> Result<AutomatonGraph, CompileError> {
        let start = epsilon_closure(self, &[self.start])?;
        let mut subsets = Vec::new();
        let mut ids: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
        ids.try_reserve(MAX_GRAPH_STATES)
            .map_err(|_| graph_limit("subset map allocation"))?;
        let mut queue = VecDeque::new();
        let start_id = intern_subset(start, &mut subsets, &mut ids, &mut queue)?;
        let mut output = Vec::new();
        output
            .try_reserve(MAX_GRAPH_STATES.min(self.states.len().saturating_mul(2)))
            .map_err(|_| graph_limit("subset output allocation"))?;
        let mut edge_count = 0usize;
        while let Some(id) = queue.pop_front() {
            let index = usize::try_from(id).map_err(|_| graph_limit("subset id"))?;
            let subset = std::mem::take(
                subsets
                    .get_mut(index)
                    .ok_or_else(|| graph_limit("subset id"))?,
            );
            let accepting = subset.iter().any(|&state| {
                usize::try_from(state)
                    .ok()
                    .and_then(|state| self.states.get(state))
                    .is_some_and(|state| state.accepting)
            });
            let edges = self.subset_edges(&subset, &mut subsets, &mut ids, &mut queue)?;
            subsets[index] = subset;
            edge_count = edge_count
                .checked_add(edges.len())
                .ok_or_else(|| graph_limit("subset edge count"))?;
            if edge_count > MAX_GRAPH_EDGES {
                return Err(graph_limit("subset edge cap"));
            }
            output.push((accepting, edges));
        }
        AutomatonGraph::from_byte_states(&output, start_id)
    }

    fn subset_edges(
        &self,
        subset: &[u32],
        subsets: &mut Vec<Vec<u32>>,
        ids: &mut FxHashMap<Vec<u32>, u32>,
        queue: &mut VecDeque<u32>,
    ) -> Result<Vec<(u8, u32)>, CompileError> {
        let mut edges = Vec::new();
        for byte in 0u8..=255 {
            let mut raw = Vec::new();
            for &state in subset {
                let state = self
                    .states
                    .get(usize::try_from(state).map_err(|_| graph_limit("NFA state id"))?)
                    .ok_or_else(|| graph_limit("NFA state id"))?;
                for &(edge_byte, target) in &state.edges {
                    if edge_byte == byte {
                        raw.try_reserve(1)
                            .map_err(|_| graph_limit("subset target allocation"))?;
                        raw.push(target);
                    }
                }
            }
            if raw.is_empty() {
                continue;
            }
            let closed = epsilon_closure(self, &raw)?;
            let target = intern_subset(closed, subsets, ids, queue)?;
            edges
                .try_reserve(1)
                .map_err(|_| graph_limit("subset row growth"))?;
            edges.push((byte, target));
        }
        Ok(edges)
    }
}

fn epsilon_closure(nfa: &ByteNfa, seeds: &[u32]) -> Result<Vec<u32>, CompileError> {
    let mut closure = Vec::new();
    closure
        .try_reserve(seeds.len())
        .map_err(|_| graph_limit("epsilon closure allocation"))?;
    closure.extend_from_slice(seeds);
    let mut cursor = 0usize;
    while cursor < closure.len() {
        let state = nfa
            .states
            .get(usize::try_from(closure[cursor]).map_err(|_| graph_limit("NFA state id"))?)
            .ok_or_else(|| graph_limit("NFA state id"))?;
        for &target in &state.epsilon {
            if !closure.contains(&target) {
                closure
                    .try_reserve(1)
                    .map_err(|_| graph_limit("epsilon closure growth"))?;
                closure.push(target);
            }
        }
        cursor += 1;
    }
    closure.sort_unstable();
    closure.dedup();
    Ok(closure)
}

fn intern_subset(
    subset: Vec<u32>,
    subsets: &mut Vec<Vec<u32>>,
    ids: &mut FxHashMap<Vec<u32>, u32>,
    queue: &mut VecDeque<u32>,
) -> Result<u32, CompileError> {
    if let Some(id) = ids.get(&subset) {
        return Ok(*id);
    }
    if subsets.len() >= MAX_GRAPH_STATES {
        return Err(graph_limit("subset state cap"));
    }
    subsets
        .try_reserve(1)
        .map_err(|_| graph_limit("subset state allocation"))?;
    let id = u32::try_from(subsets.len()).map_err(|_| graph_limit("subset id"))?;
    let mut key = Vec::new();
    key.try_reserve_exact(subset.len())
        .map_err(|_| graph_limit("subset key allocation"))?;
    key.extend_from_slice(&subset);
    ids.insert(key, id);
    subsets.push(subset);
    queue.push_back(id);
    Ok(id)
}

pub(crate) fn graph_limit(what: &'static str) -> CompileError {
    CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L3,
        "automaton graph cap",
    )
    .with_observed(what)
}

fn graph_error(message: &'static str) -> CompileError {
    CompileError::new(ErrorCode::Malformed, Stage::L3, message)
}

fn graph_retained_estimate(states: &[ByteState]) -> Result<usize, CompileError> {
    states.iter().try_fold(
        states
            .len()
            .checked_mul(std::mem::size_of::<GraphState>())
            .ok_or_else(|| graph_limit("graph retained state bytes"))?,
        |bytes, (_, edges)| {
            edges
                .len()
                .checked_mul(std::mem::size_of::<GraphEdge>())
                .and_then(|edge_bytes| bytes.checked_add(edge_bytes))
                .ok_or_else(|| graph_limit("graph retained edge bytes"))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::build_from_regex;

    fn string_engine(values: &[&str]) -> RefEngine {
        RefEngine::from_graph(&string_set(values).unwrap()).unwrap()
    }

    #[test]
    fn string_set_accepts_raw_escaped_unicode_and_surrogates() {
        let engine = string_engine(&["a", "\"\\/\n", "😀", "😁"]);
        for accepted in [
            br#""a""#.as_slice(),
            br#""\u0061""#,
            br#""\"\\\/\n""#,
            br#""\u0022\u005C\u002f\u000A""#,
            "\"😀\"".as_bytes(),
            br#""\uD83D\uDE00""#,
            br#""\ud83d\ude01""#,
        ] {
            assert!(engine.accepts(accepted), "rejected {accepted:?}");
        }
        for rejected in [br#""b""#.as_slice(), br#""\u0062""#, "\"😀a\"".as_bytes()] {
            assert!(!engine.accepts(rejected), "accepted {rejected:?}");
        }
    }

    #[test]
    fn string_set_shares_prefixes_and_is_deterministic() {
        let first = string_engine(&["", "a", "ab", "alpha", "alpine"]);
        let second = string_engine(&["", "a", "ab", "alpha", "alpine"]);
        assert_exact_engine(&first, &second);
        for value in [b"\"\"".as_slice(), br#""a""#, br#""ab""#, br#""alpha""#] {
            assert!(first.accepts(value), "rejected {value:?}");
        }
        assert!(!first.accepts(br#""alp""#));
    }

    #[test]
    fn graph_composition_matches_regex_oracles_exactly() {
        let a = build_from_regex("a*").unwrap();
        let b = build_from_regex("b?").unwrap();
        let concatenated = RefEngine::from_graph(&concatenate(&[&a, &b]).unwrap()).unwrap();
        assert_exact_engine(&build_from_regex("(?:a*)(?:b?)").unwrap(), &concatenated);

        let unioned = RefEngine::from_graph(&union(&[&a, &b]).unwrap()).unwrap();
        assert_exact_engine(&build_from_regex("(?:a*|b?)").unwrap(), &unioned);

        let atom = build_from_regex("ab").unwrap();
        let bounded = RefEngine::from_graph(&repeat(&atom, 1, Some(3)).unwrap()).unwrap();
        assert_exact_engine(&build_from_regex("(?:ab){1,3}").unwrap(), &bounded);
        let unbounded = RefEngine::from_graph(&repeat(&atom, 2, None).unwrap()).unwrap();
        assert_exact_engine(&build_from_regex("(?:ab){2,}").unwrap(), &unbounded);
    }

    #[test]
    fn graph_validation_rejects_duplicate_bytes_and_out_of_range_targets() {
        let duplicate = vec![(false, vec![(b'a', 0), (b'a', 0)])];
        assert_eq!(
            AutomatonGraph::from_byte_states(&duplicate, 0)
                .unwrap_err()
                .code,
            ErrorCode::Malformed
        );
        let invalid = vec![(false, vec![(b'a', 1)])];
        assert_eq!(
            AutomatonGraph::from_byte_states(&invalid, 0)
                .unwrap_err()
                .code,
            ErrorCode::Malformed
        );
    }

    #[test]
    #[ignore = "release-only finite-set compile profile"]
    fn finite_string_set_compile_profile() {
        for count in [10usize, 100, 1_000, 10_000] {
            let values = (0..count)
                .map(|index| format!("value-{index:05}"))
                .collect::<Vec<_>>();
            let refs = values.iter().map(String::as_str).collect::<Vec<_>>();
            let started = std::time::Instant::now();
            let direct = string_engine(&refs);
            let direct_elapsed = started.elapsed();
            let pattern = format!(
                "(?:{})",
                values
                    .iter()
                    .map(|value| format!(r#""{value}""#))
                    .collect::<Vec<_>>()
                    .join("|")
            );
            let started = std::time::Instant::now();
            let regex = build_from_regex(&pattern);
            let regex_elapsed = started.elapsed();
            for value in &values {
                let document = format!(r#""{value}""#);
                assert!(direct.accepts(document.as_bytes()));
                if let Ok(regex) = &regex {
                    assert!(regex.accepts(document.as_bytes()));
                }
            }
            println!(
                "finite_set count={count} direct_us={} regex_us={} regex_ok={} direct_bytes={}",
                direct_elapsed.as_micros(),
                regex_elapsed.as_micros(),
                regex.is_ok(),
                direct.heap_bytes()
            );
        }
    }

    fn assert_exact_engine(left: &RefEngine, right: &RefEngine) {
        crate::automaton::byte_dfa::assert_exact_equivalence(left, right);
    }
}
