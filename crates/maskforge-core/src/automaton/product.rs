//! Builds product automata over independently compiled reference engines.

use std::collections::VecDeque;

use rustc_hash::FxHashMap;

use super::byte_dfa::RefEngine;
#[cfg(test)]
use super::elimination::{to_regex_via_state_elimination, EliminationState, MAX_ELIMINATION_BYTES};
use super::graph::AutomatonGraph;
use crate::error::{CompileError, ErrorCode, LimitKind, Stage};
use crate::mem_gate::acquire_automaton_build_bytes;
use crate::primitives::StateId;

const MAX_PRODUCT_STATES: usize = 1 << 16;

const MAX_PRODUCT_BRANCHES: usize = 256;

const MAX_PRODUCT_EDGES: usize = 1 << 20;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum Combinator {
    All,
    Any,
    ExactlyOne,
    Difference,
}

impl Combinator {
    fn accepts_states(self, states: &[StateId], engines: &[&RefEngine]) -> bool {
        match self {
            Self::All => states
                .iter()
                .zip(engines)
                .all(|(&state, engine)| engine.is_accepting(state)),
            Self::Any => states
                .iter()
                .zip(engines)
                .any(|(&state, engine)| engine.is_accepting(state)),
            Self::ExactlyOne => {
                states
                    .iter()
                    .zip(engines)
                    .filter(|&(state, engine)| engine.is_accepting(*state))
                    .count()
                    == 1
            }
            Self::Difference => states.split_first().zip(engines.split_first()).is_some_and(
                |((&first_state, rest_states), (&first_engine, rest_engines))| {
                    first_engine.is_accepting(first_state)
                        && rest_states
                            .iter()
                            .zip(rest_engines)
                            .all(|(&state, engine)| !engine.is_accepting(state))
                },
            ),
        }
    }
}

#[derive(Debug)]
pub(crate) struct ProductAutomaton {
    states: Vec<(bool, Vec<(u8, u32)>)>,
    start: u32,
}

impl ProductAutomaton {
    pub(crate) fn build(
        branches: &[&RefEngine],
        combinator: Combinator,
    ) -> Result<Self, CompileError> {
        Self::build_capped(
            branches,
            combinator,
            MAX_PRODUCT_STATES,
            MAX_PRODUCT_EDGES,
            true,
        )
    }

    pub(crate) fn build_reachable(
        branches: &[&RefEngine],
        combinator: Combinator,
    ) -> Result<Self, CompileError> {
        Self::build_capped(
            branches,
            combinator,
            MAX_PRODUCT_STATES,
            MAX_PRODUCT_EDGES,
            false,
        )
    }

    fn build_capped(
        branches: &[&RefEngine],
        combinator: Combinator,
        max_states: usize,
        max_edges: usize,
        reject_cartesian_over_cap: bool,
    ) -> Result<Self, CompileError> {
        let _permit = acquire_automaton_build_bytes(64 << 20)?;
        if branches.is_empty() {
            return Err(malformed("a product automaton needs at least one branch"));
        }
        if branches.len() > MAX_PRODUCT_BRANCHES {
            return Err(limit_value(
                LimitKind::ProductBranches,
                branches.len(),
                MAX_PRODUCT_BRANCHES,
                "product branch count",
            ));
        }
        let worst_case_states = branches
            .iter()
            .try_fold(1usize, |acc, e| acc.checked_mul(e.state_count()))
            .unwrap_or(usize::MAX);
        if reject_cartesian_over_cap && worst_case_states > max_states {
            return Err(limit_value(
                LimitKind::ProductStates,
                worst_case_states,
                max_states,
                "product state estimate",
            ));
        }
        let worst_case_edges = worst_case_states.saturating_mul(256);
        if reject_cartesian_over_cap && worst_case_edges > max_edges {
            return Err(limit_value(
                LimitKind::ProductEdges,
                worst_case_edges,
                max_edges,
                "product edge estimate",
            ));
        }
        let admitted_states = worst_case_states.min(max_states).max(1);
        let mut start_tuple = Vec::new();
        start_tuple
            .try_reserve_exact(branches.len())
            .map_err(|_| limit("product start allocation"))?;
        start_tuple.extend(branches.iter().map(|engine| engine.start()));

        let mut disco_order: Vec<Vec<StateId>> = Vec::new();
        disco_order
            .try_reserve(admitted_states)
            .map_err(|_| limit("product state allocation"))?;
        let mut index_of: FxHashMap<Vec<StateId>, u32> = FxHashMap::default();
        index_of
            .try_reserve(admitted_states)
            .map_err(|_| limit("product index allocation"))?;
        let mut queue: VecDeque<u32> = VecDeque::new();
        queue
            .try_reserve(admitted_states)
            .map_err(|_| limit("product queue allocation"))?;

        let start_idx = intern_tuple(&start_tuple, &mut disco_order, &mut index_of, &mut queue)?;

        let mut raw_accepting: Vec<bool> = Vec::new();
        let mut raw_edges: Vec<Vec<(u8, u32)>> = Vec::new();
        raw_accepting
            .try_reserve(admitted_states)
            .map_err(|_| limit("product acceptance allocation"))?;
        raw_edges
            .try_reserve(admitted_states)
            .map_err(|_| limit("product edge-row allocation"))?;
        let mut total_edges = 0usize;

        while let Some(idx) = queue.pop_front() {
            let tuple_index = usize::try_from(idx).map_err(|_| limit("product state index"))?;
            let tuple = std::mem::take(
                disco_order
                    .get_mut(tuple_index)
                    .ok_or_else(|| limit("product state index"))?,
            );
            let accepting = combinator.accepts_states(&tuple, branches);

            let mut edges = Vec::new();
            for byte in 0u8..=255 {
                let mut next = Vec::new();
                next.try_reserve_exact(tuple.len())
                    .map_err(|_| limit("product tuple allocation"))?;
                let mut all_dead = true;
                for (&s, e) in tuple.iter().zip(branches) {
                    let t = if e.is_dead(s) {
                        e.dead()
                    } else {
                        e.consume_token(s, &[byte]).unwrap_or_else(|| e.dead())
                    };
                    if !e.is_dead(t) {
                        all_dead = false;
                    }
                    next.push(t);
                }
                if all_dead {
                    continue;
                }
                let target_idx = intern_tuple(&next, &mut disco_order, &mut index_of, &mut queue)?;
                if disco_order.len() > max_states {
                    return Err(limit_value(
                        LimitKind::ProductStates,
                        disco_order.len(),
                        max_states,
                        "product state cap",
                    ));
                }
                total_edges += 1;
                if total_edges > max_edges {
                    return Err(limit_value(
                        LimitKind::ProductEdges,
                        total_edges,
                        max_edges,
                        "product edge budget",
                    ));
                }
                edges
                    .try_reserve(1)
                    .map_err(|_| limit("product row growth"))?;
                edges.push((byte, target_idx));
            }
            disco_order[tuple_index] = tuple;
            raw_accepting.push(accepting);
            raw_edges.push(edges);
        }

        let n = disco_order.len();
        let mut predecessor_counts = Vec::new();
        predecessor_counts
            .try_reserve_exact(n)
            .map_err(|_| limit("product predecessor allocation"))?;
        predecessor_counts.resize(n, 0usize);
        for edges in &raw_edges {
            for &(_, target) in edges {
                let target =
                    usize::try_from(target).map_err(|_| limit("product predecessor index"))?;
                predecessor_counts[target] = predecessor_counts[target]
                    .checked_add(1)
                    .ok_or_else(|| limit("product predecessor count"))?;
            }
        }
        let mut rev = Vec::new();
        rev.try_reserve_exact(n)
            .map_err(|_| limit("product reverse allocation"))?;
        for count in predecessor_counts {
            let mut row = Vec::new();
            row.try_reserve_exact(count)
                .map_err(|_| limit("product reverse row allocation"))?;
            rev.push(row);
        }
        for (i, edges) in raw_edges.iter().enumerate() {
            for &(_, t) in edges {
                rev[t as usize].push(u32::try_from(i).map_err(|_| limit("product state index"))?);
            }
        }
        let mut live = Vec::new();
        live.try_reserve_exact(n)
            .map_err(|_| limit("product live allocation"))?;
        live.resize(n, false);
        let mut lq: VecDeque<u32> = VecDeque::new();
        lq.try_reserve(n)
            .map_err(|_| limit("product live queue allocation"))?;
        for i in 0..n {
            if raw_accepting[i] {
                live[i] = true;
                lq.push_back(u32::try_from(i).map_err(|_| limit("product state index"))?);
            }
        }
        while let Some(t) = lq.pop_front() {
            for &p in &rev[t as usize] {
                if !live[p as usize] {
                    live[p as usize] = true;
                    lq.push_back(p);
                }
            }
        }

        let mut final_id = Vec::new();
        final_id
            .try_reserve_exact(n)
            .map_err(|_| limit("product remap allocation"))?;
        final_id.resize(n, None);
        let mut next_id = 0u32;
        for i in 0..n {
            if live[i] {
                final_id[i] = Some(next_id);
                next_id += 1;
            }
        }

        let mut states = Vec::new();
        states
            .try_reserve_exact(usize::try_from(next_id).map_err(|_| limit("product state count"))?)
            .map_err(|_| limit("product final allocation"))?;
        for i in 0..n {
            if !live[i] {
                continue;
            }
            let mut edges = Vec::new();
            edges
                .try_reserve_exact(raw_edges[i].len())
                .map_err(|_| limit("product final row allocation"))?;
            edges.extend(
                raw_edges[i]
                    .iter()
                    .filter_map(|&(b, t)| final_id[t as usize].map(|ft| (b, ft))),
            );
            states.push((raw_accepting[i], edges));
        }

        let start = final_id[start_idx as usize].unwrap_or(0);
        if states.is_empty() {
            return Ok(Self {
                states: vec![(false, Vec::new())],
                start: 0,
            });
        }
        Ok(Self { states, start })
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

fn intern_tuple(
    tuple: &[StateId],
    disco_order: &mut Vec<Vec<StateId>>,
    index_of: &mut FxHashMap<Vec<StateId>, u32>,
    queue: &mut VecDeque<u32>,
) -> Result<u32, CompileError> {
    if let Some(&id) = index_of.get(tuple) {
        return Ok(id);
    }
    let id = u32::try_from(disco_order.len()).map_err(|_| limit("product state index"))?;
    let mut stored = Vec::new();
    stored
        .try_reserve_exact(tuple.len())
        .map_err(|_| limit("product tuple storage"))?;
    stored.extend_from_slice(tuple);
    let mut key = Vec::new();
    key.try_reserve_exact(tuple.len())
        .map_err(|_| limit("product tuple key"))?;
    key.extend_from_slice(tuple);
    disco_order.push(stored);
    index_of.insert(key, id);
    queue.push_back(id);
    Ok(id)
}

fn malformed(message: &'static str) -> CompileError {
    CompileError::new(ErrorCode::Malformed, Stage::L3, message)
}

fn limit(what: &'static str) -> CompileError {
    let mut e = CompileError::new(ErrorCode::InternalLimitExceeded, Stage::L3, "product cap");
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

    fn engine(pattern: &str) -> RefEngine {
        build_from_regex(pattern).unwrap()
    }

    #[test]
    fn all_of_two_integer_ranges_intersects_membership() {
        let a = engine(&crate::compile::integer_regex(Some(0), Some(10)));
        let b = engine(&crate::compile::integer_regex(Some(5), Some(20)));
        let product = ProductAutomaton::build(&[&a, &b], Combinator::All).unwrap();
        let regex = product.to_regex().unwrap();
        let e = build_from_regex(&regex).unwrap();
        for n in -5i64..=25 {
            let want = (0..=10).contains(&n) && (5..=20).contains(&n);
            assert_eq!(e.accepts(n.to_string().as_bytes()), want, "n={n}");
        }
    }

    #[test]
    fn direct_product_is_exactly_equivalent_to_elimination_oracle() {
        for combinator in [
            Combinator::All,
            Combinator::Any,
            Combinator::ExactlyOne,
            Combinator::Difference,
        ] {
            let a = engine(&crate::compile::integer_regex(Some(-20), Some(15)));
            let b = engine(&crate::compile::integer_regex(Some(5), Some(30)));
            let old_product = ProductAutomaton::build(&[&a, &b], combinator).unwrap();
            let old = build_from_regex(&old_product.to_regex().unwrap()).unwrap();
            let new = ProductAutomaton::build(&[&a, &b], combinator)
                .unwrap()
                .into_engine()
                .unwrap();
            assert_exact_equivalence(&old, &new);
        }
    }

    #[test]
    fn all_of_self_intersection_is_the_same_language() {
        let a = engine(&crate::compile::integer_regex(Some(0), Some(9)));
        let b = engine(&crate::compile::integer_regex(Some(0), Some(9)));
        let product = ProductAutomaton::build(&[&a, &b], Combinator::All).unwrap();
        let e = build_from_regex(&product.to_regex().unwrap()).unwrap();
        for n in -2i64..=12 {
            assert_eq!(e.accepts(n.to_string().as_bytes()), (0..=9).contains(&n));
        }
    }

    #[test]
    fn state_cap_is_enforced_not_unbounded_allocation() {
        let a = engine(&crate::compile::integer_regex(Some(100), Some(199)));
        let b = engine(&crate::compile::integer_regex(Some(150), Some(249)));
        let err =
            ProductAutomaton::build_capped(&[&a, &b], Combinator::All, 2, MAX_PRODUCT_EDGES, true)
                .unwrap_err();
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn edge_budget_is_enforced_during_the_bfs() {
        let a = engine(&crate::compile::integer_regex(Some(100), Some(199)));
        let b = engine(&crate::compile::integer_regex(Some(150), Some(249)));
        let err =
            ProductAutomaton::build_capped(&[&a, &b], Combinator::All, MAX_PRODUCT_STATES, 4, true)
                .unwrap_err();
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn all_of_disjoint_ranges_is_the_empty_language() {
        let a = engine(&crate::compile::integer_regex(Some(0), Some(5)));
        let b = engine(&crate::compile::integer_regex(Some(10), Some(20)));
        let product = ProductAutomaton::build(&[&a, &b], Combinator::All).unwrap();
        let e = build_from_regex(&product.to_regex().unwrap()).unwrap();
        for n in -5i64..=25 {
            assert!(!e.accepts(n.to_string().as_bytes()), "n={n}");
        }
        assert!(
            e.is_dead(e.start()),
            "empty language collapses to a dead start"
        );
    }

    #[test]
    fn exactly_one_of_two_overlapping_ranges_rejects_the_overlap() {
        let a = engine(&crate::compile::integer_regex(Some(0), Some(10)));
        let b = engine(&crate::compile::integer_regex(Some(5), Some(20)));
        let product = ProductAutomaton::build(&[&a, &b], Combinator::ExactlyOne).unwrap();
        let e = build_from_regex(&product.to_regex().unwrap()).unwrap();
        for n in -5i64..=25 {
            let in_a = (0..=10).contains(&n);
            let in_b = (5..=20).contains(&n);
            let want = in_a ^ in_b;
            assert_eq!(e.accepts(n.to_string().as_bytes()), want, "n={n}");
        }
    }

    #[test]
    fn difference_accepts_the_left_language_without_the_right_language() {
        let left = engine("[a-z]+");
        let right = engine("(?:bad|blocked)");
        let product = ProductAutomaton::build(&[&left, &right], Combinator::Difference)
            .unwrap()
            .into_engine()
            .unwrap();
        assert!(product.accepts(b"good"));
        assert!(!product.accepts(b"bad"));
        assert!(!product.accepts(b"blocked"));
        assert!(!product.accepts(b"123"));
    }

    #[test]
    fn exactly_one_of_three_never_folds_pairwise() {
        let a = engine(&crate::compile::integer_regex(Some(0), Some(15)));
        let b = engine(&crate::compile::integer_regex(Some(5), Some(20)));
        let c = engine(&crate::compile::integer_regex(Some(10), Some(25)));
        let product = ProductAutomaton::build(&[&a, &b, &c], Combinator::ExactlyOne).unwrap();
        let e = build_from_regex(&product.to_regex().unwrap()).unwrap();
        for n in -5i64..=30 {
            let flags = [
                (0..=15).contains(&n),
                (5..=20).contains(&n),
                (10..=25).contains(&n),
            ];
            let want = flags.iter().filter(|&&f| f).count() == 1;
            assert_eq!(e.accepts(n.to_string().as_bytes()), want, "n={n}");
        }
        assert!(!e.accepts(b"12"));
    }

    #[test]
    fn exactly_one_of_identical_branches_rejects_every_match() {
        let a = engine(&crate::compile::integer_regex(Some(0), Some(9)));
        let b = engine(&crate::compile::integer_regex(Some(0), Some(9)));
        let product = ProductAutomaton::build(&[&a, &b], Combinator::ExactlyOne).unwrap();
        let e = build_from_regex(&product.to_regex().unwrap()).unwrap();
        for n in 0i64..=9 {
            assert!(!e.accepts(n.to_string().as_bytes()), "n={n}");
        }
    }

    #[test]
    fn an_empty_branch_list_is_a_structured_error() {
        for combinator in [
            Combinator::All,
            Combinator::Any,
            Combinator::ExactlyOne,
            Combinator::Difference,
        ] {
            let err = ProductAutomaton::build(&[], combinator).unwrap_err();
            assert_eq!(err.code, ErrorCode::Malformed);
        }
    }

    #[test]
    fn a_single_branch_reduces_to_that_branch_under_every_combinator() {
        let only = engine(&crate::compile::integer_regex(Some(0), Some(9)));
        for combinator in [
            Combinator::All,
            Combinator::Any,
            Combinator::ExactlyOne,
            Combinator::Difference,
        ] {
            let product = ProductAutomaton::build(&[&only], combinator).unwrap();
            let e = build_from_regex(&product.to_regex().unwrap()).unwrap();
            for n in -2i64..=12 {
                assert_eq!(e.accepts(n.to_string().as_bytes()), (0..=9).contains(&n));
            }
        }
    }

    #[test]
    fn too_many_branches_fails_before_construction() {
        let e = engine("(?:true|false)");
        let refs: Vec<&RefEngine> = std::iter::repeat_n(&e, MAX_PRODUCT_BRANCHES + 1).collect();
        let err = ProductAutomaton::build(&refs, Combinator::All).unwrap_err();
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn state_elimination_size_budget_fails_closed() {
        let a = engine(&crate::compile::integer_regex(Some(100), Some(199)));
        let b = engine(&crate::compile::integer_regex(Some(150), Some(249)));
        let product = ProductAutomaton::build(&[&a, &b], Combinator::All).unwrap();
        let states: Vec<EliminationState> = product
            .states
            .iter()
            .map(|(accepting, edges)| EliminationState {
                accepting: *accepting,
                edges: edges.clone(),
            })
            .collect();
        let err = to_regex_via_state_elimination(&states, product.start, 8).unwrap_err();
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn repeated_build_produces_an_identical_regex() {
        let a = engine(&crate::compile::integer_regex(Some(0), Some(10)));
        let b = engine(&crate::compile::integer_regex(Some(5), Some(20)));
        let first = ProductAutomaton::build(&[&a, &b], Combinator::All)
            .unwrap()
            .to_regex()
            .unwrap();
        let second = ProductAutomaton::build(&[&a, &b], Combinator::All)
            .unwrap()
            .to_regex()
            .unwrap();
        assert_eq!(first, second, "product construction must be deterministic");
    }

    #[test]
    fn a_wide_branch_paired_with_a_small_one_is_rejected_immediately() {
        let alts: Vec<String> = (0..8000).map(|i| format!("\"v{i}\"")).collect();
        let wide = engine(&format!("(?:{})", alts.join("|")));
        let small = engine(r#"(?:true|false)"#);
        assert!(wide.state_count() > 1000, "fixture must actually be large");
        let start = std::time::Instant::now();
        let err = ProductAutomaton::build(&[&wide, &small], Combinator::ExactlyOne).unwrap_err();
        let elapsed = start.elapsed();
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
        assert!(
            elapsed < std::time::Duration::from_millis(500),
            "the size cap must reject before the BFS runs, took {elapsed:?}"
        );
    }
}
