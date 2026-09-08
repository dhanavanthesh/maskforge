//! Builds automata for unordered collections of independent field fragments.

use rustc_hash::FxHashMap;
use std::collections::VecDeque;

use super::byte_dfa::RefEngine;
#[cfg(test)]
use super::elimination::{to_regex_via_state_elimination, EliminationState, MAX_ELIMINATION_BYTES};
use super::graph::AutomatonGraph;
use crate::error::{CompileError, ErrorCode, LimitKind, Stage};
use crate::mem_gate::acquire_automaton_build_bytes;
use crate::primitives::StateId;

const MAX_SHUFFLE_FIELDS: usize = 32;
const MAX_SHUFFLE_STATES: usize = 1 << 16;
const MAX_SHUFFLE_EDGES: usize = 1 << 20;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
enum Config {
    Ready(u32),
    InComma(u32, StateId),
    AfterComma(u32),
    Active(u32, u8, StateId),
}

#[derive(Debug)]
pub(crate) struct ShuffleAutomaton {
    states: Vec<(bool, Vec<(u8, u32)>)>,
    start: u32,
}

impl ShuffleAutomaton {
    pub(crate) fn build(
        fields: &[&RefEngine],
        required: &[bool],
        separator: &RefEngine,
    ) -> Result<Self, CompileError> {
        let _permit = acquire_automaton_build_bytes(64 << 20)?;
        if fields.len() != required.len() {
            return Err(malformed(
                "shuffle field count does not match required flags",
            ));
        }
        if fields.len() > MAX_SHUFFLE_FIELDS {
            return Err(limit_value(
                LimitKind::ShuffleFields,
                fields.len(),
                MAX_SHUFFLE_FIELDS,
                "shuffle field count",
            ));
        }
        let max_field_states = fields.iter().map(|f| f.state_count()).max().unwrap_or(0);
        let worst_case_states = max_field_states.saturating_mul(1usize << fields.len());
        if worst_case_states > MAX_SHUFFLE_STATES {
            return Err(limit_value(
                LimitKind::ShuffleStates,
                worst_case_states,
                MAX_SHUFFLE_STATES,
                "shuffle state estimate",
            ));
        }
        let worst_case_edges = worst_case_states.saturating_mul(256);
        if worst_case_edges > MAX_SHUFFLE_EDGES {
            return Err(limit_value(
                LimitKind::ShuffleEdges,
                worst_case_edges,
                MAX_SHUFFLE_EDGES,
                "shuffle edge estimate",
            ));
        }
        let admitted_states = worst_case_states.max(1);
        let required_mask =
            required
                .iter()
                .enumerate()
                .fold(0u32, |m, (i, &r)| if r { m | (1 << i) } else { m });

        let mut disco_order: Vec<Vec<Config>> = Vec::new();
        disco_order
            .try_reserve(admitted_states)
            .map_err(|_| limit("shuffle state allocation"))?;
        let mut index_of: FxHashMap<Vec<Config>, u32> = FxHashMap::default();
        index_of
            .try_reserve(admitted_states)
            .map_err(|_| limit("shuffle index allocation"))?;
        let mut queue: VecDeque<u32> = VecDeque::new();
        queue
            .try_reserve(admitted_states)
            .map_err(|_| limit("shuffle queue allocation"))?;
        let mut initial = Vec::new();
        initial
            .try_reserve_exact(1)
            .map_err(|_| limit("shuffle start allocation"))?;
        initial.push(Config::Ready(0));
        let start_configs = epsilon_closure(initial, fields, separator)?;
        let start_idx = intern(start_configs, &mut disco_order, &mut index_of, &mut queue)?;

        let mut states: Vec<(bool, Vec<(u8, u32)>)> = Vec::new();
        states
            .try_reserve(admitted_states)
            .map_err(|_| limit("shuffle output allocation"))?;
        let mut total_edges = 0usize;
        while let Some(idx) = queue.pop_front() {
            let index = usize::try_from(idx).map_err(|_| limit("shuffle state index"))?;
            let configs = std::mem::take(
                disco_order
                    .get_mut(index)
                    .ok_or_else(|| limit("shuffle state index"))?,
            );
            let accepting = configs
                .iter()
                .any(|c| matches!(c, Config::Ready(b) if b & required_mask == required_mask));
            let mut edges = Vec::new();
            for byte in 0u8..=255 {
                let raw = step_configs(&configs, fields, separator, byte)?;
                if raw.is_empty() {
                    continue;
                }
                let next = epsilon_closure(raw, fields, separator)?;
                let target = intern(next, &mut disco_order, &mut index_of, &mut queue)?;
                if disco_order.len() > MAX_SHUFFLE_STATES {
                    return Err(limit_value(
                        LimitKind::ShuffleStates,
                        disco_order.len(),
                        MAX_SHUFFLE_STATES,
                        "shuffle state cap",
                    ));
                }
                total_edges += 1;
                if total_edges > MAX_SHUFFLE_EDGES {
                    return Err(limit_value(
                        LimitKind::ShuffleEdges,
                        total_edges,
                        MAX_SHUFFLE_EDGES,
                        "shuffle edge budget",
                    ));
                }
                edges
                    .try_reserve(1)
                    .map_err(|_| limit("shuffle row growth"))?;
                edges.push((byte, target));
            }
            disco_order[index] = configs;
            debug_assert_eq!(states.len(), index);
            states.push((accepting, edges));
        }
        Ok(Self {
            states,
            start: start_idx,
        })
    }

    pub(crate) fn into_graph(self) -> Result<AutomatonGraph, CompileError> {
        AutomatonGraph::from_byte_states(&self.states, self.start)
    }

    pub(crate) fn into_engine(self) -> Result<RefEngine, CompileError> {
        RefEngine::from_graph(&self.into_graph()?)
    }

    #[cfg(test)]
    pub(crate) fn to_regex(&self) -> Result<String, CompileError> {
        let states: Vec<EliminationState> = self
            .states
            .iter()
            .map(|(accepting, edges)| EliminationState {
                accepting: *accepting,
                edges: edges.clone(),
            })
            .collect();
        to_regex_via_state_elimination(&states, self.start, MAX_ELIMINATION_BYTES)
    }
}

fn epsilon_closure(
    mut configs: Vec<Config>,
    fields: &[&RefEngine],
    separator: &RefEngine,
) -> Result<Vec<Config>, CompileError> {
    loop {
        let mut additions = Vec::new();
        additions
            .try_reserve(configs.len())
            .map_err(|_| limit("shuffle closure allocation"))?;
        for c in &configs {
            let added = match *c {
                Config::InComma(bitset, s) if separator.is_accepting(s) => {
                    Some(Config::AfterComma(bitset))
                }
                Config::Active(bitset, i, s) if fields[usize::from(i)].is_accepting(s) => {
                    Some(Config::Ready(bitset | (1 << i)))
                }
                _ => None,
            };
            if let Some(a) = added {
                if !configs.contains(&a) && !additions.contains(&a) {
                    additions.push(a);
                }
            }
        }
        if additions.is_empty() {
            break;
        }
        configs
            .try_reserve(additions.len())
            .map_err(|_| limit("shuffle closure growth"))?;
        configs.extend(additions);
    }
    configs.sort_unstable();
    configs.dedup();
    Ok(configs)
}

fn step_configs(
    configs: &[Config],
    fields: &[&RefEngine],
    separator: &RefEngine,
    byte: u8,
) -> Result<Vec<Config>, CompileError> {
    let mut next = Vec::new();
    let capacity = configs
        .len()
        .checked_mul(fields.len().max(1))
        .ok_or_else(|| limit("shuffle step capacity"))?;
    next.try_reserve(capacity)
        .map_err(|_| limit("shuffle step allocation"))?;
    for c in configs {
        match *c {
            Config::Ready(0) => {
                for (i, engine) in fields.iter().enumerate() {
                    if let Some(ns) = live_step(engine, engine.start(), byte) {
                        let idx = u8::try_from(i).map_err(|_| limit("shuffle field index"))?;
                        next.push(Config::Active(0, idx, ns));
                    }
                }
            }
            Config::Ready(bitset) => {
                if let Some(ns) = live_step(separator, separator.start(), byte) {
                    next.push(Config::InComma(bitset, ns));
                }
            }
            Config::InComma(bitset, s) => {
                if let Some(ns) = live_step(separator, s, byte) {
                    next.push(Config::InComma(bitset, ns));
                }
            }
            Config::AfterComma(bitset) => {
                for (i, engine) in fields.iter().enumerate() {
                    if bitset & (1 << i) != 0 {
                        continue;
                    }
                    if let Some(ns) = live_step(engine, engine.start(), byte) {
                        let idx = u8::try_from(i).map_err(|_| limit("shuffle field index"))?;
                        next.push(Config::Active(bitset, idx, ns));
                    }
                }
            }
            Config::Active(bitset, i, s) => {
                let engine = fields[usize::from(i)];
                if let Some(ns) = live_step(engine, s, byte) {
                    next.push(Config::Active(bitset, i, ns));
                }
            }
        }
    }
    Ok(next)
}

fn live_step(engine: &RefEngine, from: StateId, byte: u8) -> Option<StateId> {
    engine
        .consume_token(from, &[byte])
        .filter(|&t| !engine.is_dead(t))
}

fn intern(
    configs: Vec<Config>,
    disco_order: &mut Vec<Vec<Config>>,
    index_of: &mut FxHashMap<Vec<Config>, u32>,
    queue: &mut VecDeque<u32>,
) -> Result<u32, CompileError> {
    if let Some(&id) = index_of.get(&configs) {
        return Ok(id);
    }
    let id = u32::try_from(disco_order.len()).map_err(|_| limit("shuffle state index"))?;
    let mut stored = Vec::new();
    stored
        .try_reserve_exact(configs.len())
        .map_err(|_| limit("shuffle config storage"))?;
    stored.extend_from_slice(&configs);
    disco_order.push(stored);
    index_of.insert(configs, id);
    queue.push_back(id);
    Ok(id)
}

fn malformed(message: &'static str) -> CompileError {
    CompileError::new(ErrorCode::Malformed, Stage::L3, message)
}

fn limit(what: &'static str) -> CompileError {
    let mut e = CompileError::new(ErrorCode::InternalLimitExceeded, Stage::L3, "shuffle cap");
    e.observed = Some(what.to_string());
    e
}

fn limit_value(kind: LimitKind, observed: usize, cap: usize, what: &'static str) -> CompileError {
    let mut error = limit(what);
    error.limit = Some((kind, observed, cap));
    error
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::build_from_regex;
    use crate::automaton::byte_dfa::assert_exact_equivalence;

    mod itertools_lite {
        pub(crate) fn permutations(n: usize) -> Vec<Vec<usize>> {
            if n == 0 {
                return vec![Vec::new()];
            }
            let mut out = Vec::new();
            for p in permutations(n - 1) {
                for pos in 0..=p.len() {
                    let mut np = p.clone();
                    np.insert(pos, n - 1);
                    out.push(np);
                }
            }
            out
        }
    }

    fn field(pattern: &str) -> RefEngine {
        build_from_regex(pattern).unwrap()
    }

    fn shuffle_engine(fields: &[RefEngine], required: &[bool]) -> RefEngine {
        let refs: Vec<&RefEngine> = fields.iter().collect();
        let sep = field(r",");
        ShuffleAutomaton::build(&refs, required, &sep)
            .unwrap()
            .into_engine()
            .unwrap()
    }

    #[test]
    fn direct_shuffle_is_exactly_equivalent_to_elimination_oracle() {
        let fields = [field(r#""a":1"#), field(r#""b":2"#), field(r#""c":3"#)];
        let refs: Vec<&RefEngine> = fields.iter().collect();
        let separator = field(",");
        let old_graph = ShuffleAutomaton::build(&refs, &[true, false, true], &separator).unwrap();
        let old = build_from_regex(&old_graph.to_regex().unwrap()).unwrap();
        let new = ShuffleAutomaton::build(&refs, &[true, false, true], &separator)
            .unwrap()
            .into_engine()
            .unwrap();
        assert_exact_equivalence(&old, &new);
    }

    #[test]
    fn two_required_fields_accept_both_orders() {
        let a = field(r#""a":1"#);
        let b = field(r#""b":2"#);
        let e = shuffle_engine(&[a, b], &[true, true]);
        assert!(e.accepts(br#""a":1,"b":2"#));
        assert!(e.accepts(br#""b":2,"a":1"#));
        assert!(
            !e.accepts(br#""a":1"#),
            "missing a required field must reject"
        );
        assert!(
            !e.accepts(br#""a":1,"a":1"#),
            "a field consumed twice must reject"
        );
    }

    #[test]
    fn optional_field_may_be_omitted_in_either_position() {
        let a = field(r#""a":1"#);
        let b = field(r#""b":2"#);
        let e = shuffle_engine(&[a, b], &[true, false]);
        assert!(e.accepts(br#""a":1"#));
        assert!(e.accepts(br#""a":1,"b":2"#));
        assert!(e.accepts(br#""b":2,"a":1"#));
        assert!(
            !e.accepts(br#""b":2"#),
            "the required field must still be present"
        );
    }

    #[test]
    fn five_fields_accept_every_one_of_the_120_permutations() {
        let names = ["a", "b", "c", "d", "e"];
        let fields: Vec<RefEngine> = names
            .iter()
            .map(|n| field(&format!(r#""{n}":1"#)))
            .collect();
        let e = shuffle_engine(&fields, &[true; 5]);
        for perm in itertools_lite::permutations(5) {
            let body = perm
                .iter()
                .map(|&i| format!(r#""{}":1"#, names[i]))
                .collect::<Vec<_>>()
                .join(",");
            assert!(
                e.accepts(body.as_bytes()),
                "permutation {perm:?} must accept"
            );
        }
        assert!(
            !e.accepts(br#""a":1,"b":1,"c":1,"d":1"#),
            "missing one field must reject"
        );
    }

    #[test]
    fn shared_key_prefix_fields_are_disambiguated_by_later_bytes() {
        let t1 = field(r#""type":1"#);
        let t2 = field(r#""typeName":2"#);
        let e = shuffle_engine(&[t1, t2], &[true, true]);
        assert!(e.accepts(br#""type":1,"typeName":2"#));
        assert!(e.accepts(br#""typeName":2,"type":1"#));
        assert!(!e.accepts(br#""type":1,"type":1"#));
    }

    #[test]
    fn zero_fields_accepts_only_the_empty_body() {
        let e = shuffle_engine(&[], &[]);
        assert!(e.accepts(b""));
        assert!(!e.accepts(b"x"));
    }

    #[test]
    fn field_count_over_the_cap_is_a_structured_error() {
        let sep = field(",");
        let fields: Vec<RefEngine> = (0..MAX_SHUFFLE_FIELDS + 1)
            .map(|i| field(&format!(r#""f{i}":1"#)))
            .collect();
        let refs: Vec<&RefEngine> = fields.iter().collect();
        let required = vec![true; refs.len()];
        let err = ShuffleAutomaton::build(&refs, &required, &sep).unwrap_err();
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    fn wide_alternation_field(key: &str, n: usize) -> RefEngine {
        let alts: Vec<String> = (0..n).map(|i| format!("\"v{i}\"")).collect();
        field(&format!("\"{key}\":(?:{})", alts.join("|")))
    }

    #[test]
    fn two_large_field_automatons_are_rejected_immediately_not_after_a_slow_walk() {
        let a = wide_alternation_field("a", 8000);
        let b = wide_alternation_field("b", 8000);
        assert!(a.state_count() > 1000, "fixture must actually be large");
        let sep = field(",");
        let refs = [&a, &b];
        let required = [true, true];
        let start = std::time::Instant::now();
        let err = ShuffleAutomaton::build(&refs, &required, &sep).unwrap_err();
        let elapsed = start.elapsed();
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
        assert!(
            elapsed < std::time::Duration::from_millis(200),
            "the size pre-check must reject before the discovery walk runs, took {elapsed:?}"
        );
    }

    #[test]
    fn one_moderately_wide_field_alone_still_shuffles_fine() {
        let a = wide_alternation_field("a", 50);
        let b = field(r#""b":1"#);
        let e = shuffle_engine(&[a, b], &[true, true]);
        assert!(e.accepts(br#""a":"v7","b":1"#));
        assert!(e.accepts(br#""b":1,"a":"v49""#));
        assert!(!e.accepts(br#""a":"v50","b":1"#), "v50 is out of range");
    }
}
