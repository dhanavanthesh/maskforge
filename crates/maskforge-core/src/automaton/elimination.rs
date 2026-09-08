//! Shared byte-level state elimination: turns any small edge-labeled automaton (product or
//! shuffle) into a regex fragment via classic Kleene/Brzozowski-McCluskey elimination.

use rustc_hash::FxHashMap;

use crate::error::{CompileError, ErrorCode, Stage};

#[derive(Debug)]
pub(crate) struct EliminationState {
    pub(crate) accepting: bool,
    pub(crate) edges: Vec<(u8, u32)>,
}

pub(crate) const MAX_ELIMINATION_BYTES: usize = 1 << 20;

type EdgeMap = FxHashMap<usize, String>;

pub(crate) fn to_regex_via_state_elimination(
    states: &[EliminationState],
    start: u32,
    max_bytes: usize,
) -> Result<String, CompileError> {
    let n = states.len();
    let mut work_bytes = 0usize;
    let sink = n;
    let mut out: Vec<EdgeMap> = vec![EdgeMap::default(); n + 1];
    for (i, st) in states.iter().enumerate() {
        for &(byte, target) in &st.edges {
            let label = literal_byte_regex(byte);
            union_edge(
                &mut out[i],
                target as usize,
                &label,
                &mut work_bytes,
                max_bytes,
            )?;
        }
        if st.accepting {
            union_edge(&mut out[i], sink, "", &mut work_bytes, max_bytes)?;
        }
    }
    let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for (i, edges) in out.iter().enumerate() {
        for &t in edges.keys() {
            incoming[t].push(i);
        }
    }

    let mut removed = vec![false; n + 1];
    loop {
        let elim = (0..n)
            .filter(|&i| !removed[i] && u32::try_from(i).is_ok_and(|x| x != start))
            .min_by_key(|&i| {
                let preds = incoming[i]
                    .iter()
                    .filter(|&&p| !removed[p] && p != i)
                    .count();
                preds * out[i].len()
            });
        let Some(elim) = elim else { break };
        removed[elim] = true;
        let self_loop = out[elim].remove(&elim);
        let self_star = self_loop.map(|l| format!("(?:{})*", wrap(&l)));

        let mut preds: Vec<usize> = incoming[elim]
            .iter()
            .copied()
            .filter(|&p| !removed[p] && p != elim)
            .collect();
        preds.sort_unstable();
        preds.dedup();
        let succs: Vec<(usize, String)> = out[elim]
            .iter()
            .map(|(&t, l)| (t, l.clone()))
            .filter(|&(t, _)| !removed[t])
            .collect();

        for &p in &preds {
            let Some(in_label) = out[p].get(&elim).cloned() else {
                continue;
            };
            for (t, out_label) in &succs {
                let mut piece = wrap(&in_label);
                if let Some(star) = &self_star {
                    piece.push_str(star);
                }
                piece.push_str(&wrap(out_label));
                union_edge(&mut out[p], *t, &piece, &mut work_bytes, max_bytes)?;
                incoming[*t].push(p);
            }
        }
        for p in &preds {
            out[*p].remove(&elim);
        }
        out[elim].clear();
    }

    let start_row = &out[start as usize];
    let Some(to_sink) = start_row.get(&sink) else {
        return Ok("[^\\x00-\\xff]".to_string());
    };
    Ok(match start_row.get(&(start as usize)) {
        None => to_sink.clone(),
        Some(self_loop) => format!("(?:{})*{}", wrap(self_loop), wrap(to_sink)),
    })
}

fn union_edge(
    map: &mut EdgeMap,
    target: usize,
    label: &str,
    work: &mut usize,
    max: usize,
) -> Result<(), CompileError> {
    let (old_len, new_len) = match map.get(&target) {
        Some(existing) => (existing.len(), existing.len() + 1 + label.len()),
        None => (0, label.len()),
    };
    *work = work.saturating_add(new_len).saturating_sub(old_len);
    if *work > max {
        return Err(limit("state elimination size"));
    }
    map.entry(target)
        .and_modify(|existing| *existing = format!("{existing}|{label}"))
        .or_insert_with(|| label.to_string());
    Ok(())
}

fn wrap(s: &str) -> String {
    if s.is_empty() || is_single_atom(s) {
        s.to_string()
    } else {
        format!("(?:{s})")
    }
}

fn is_single_atom(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() == 1 {
        return true;
    }
    if bytes.len() == 4 && bytes[0] == b'\\' && bytes[1] == b'x' {
        return true;
    }
    false
}

fn literal_byte_regex(byte: u8) -> String {
    format!("\\x{byte:02x}")
}

fn limit(what: &'static str) -> CompileError {
    let mut e = CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L3,
        "elimination cap",
    );
    e.observed = Some(what.to_string());
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::build_from_regex;

    #[test]
    fn a_self_loop_directly_on_the_never_eliminated_start_state_is_not_dropped() {
        let states = vec![EliminationState {
            accepting: true,
            edges: vec![(b'a', 0)],
        }];
        let regex = to_regex_via_state_elimination(&states, 0, MAX_ELIMINATION_BYTES).unwrap();
        let e = build_from_regex(&regex).unwrap();
        assert!(e.accepts(b""));
        assert!(e.accepts(b"a"));
        assert!(e.accepts(b"aaaa"));
        assert!(!e.accepts(b"b"));
    }

    #[test]
    fn a_non_accepting_start_with_only_a_self_loop_is_the_empty_language() {
        let states = vec![EliminationState {
            accepting: false,
            edges: vec![(b'a', 0)],
        }];
        let regex = to_regex_via_state_elimination(&states, 0, MAX_ELIMINATION_BYTES).unwrap();
        let e = build_from_regex(&regex).unwrap();
        assert!(e.is_dead(e.start()));
    }
}
