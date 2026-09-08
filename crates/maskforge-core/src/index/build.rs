//! Builds token transitions through a joint trie and DFA walk.

use super::artifact::CompiledIndex;
use super::trie::{TrieKind, VocabTrie};
use super::BindMode;
use crate::automaton::RefEngine;
use crate::error::{CompileError, ErrorCode, Provenance, Stage};
use crate::mask::Bitmask;
use crate::primitives::{ByteClassId, StateId, TokenId, TrieNodeId};

/// The tokens allowed from one DFA state, each with the state its byte string lands in.
/// Diagnostic-only (see `build_delta`); kept `pub` for `benches/harness.rs`.
#[derive(Clone, Debug, Default)]
pub struct DeltaRow {
    entries: Vec<(TokenId, StateId)>,
}

impl DeltaRow {
    fn push(&mut self, token: TokenId, landing: StateId) {
        self.entries.push((token, landing));
    }

    /// The `(token, landing state)` pairs allowed from this state. Order is unspecified.
    pub fn iter(&self) -> impl Iterator<Item = (TokenId, StateId)> + '_ {
        self.entries.iter().copied()
    }

    /// The number of allowed tokens (counting each id once per byte sequence it belongs to).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no token is allowed from this state.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The row's allocated (not just live) byte footprint, by `Vec::capacity`.
    #[must_use]
    pub(crate) fn heap_bytes(&self) -> usize {
        self.entries.capacity() * std::mem::size_of::<(TokenId, StateId)>()
    }
}

/// Bounds on eager index construction, so a permissive DFA over a large vocabulary cannot build an
/// unbounded `Delta`. Crate-internal defaults only; there is no public configuration API yet, so a
#[derive(Copy, Clone, Debug)]
pub(crate) struct IndexBuildLimits {
    /// Maximum reachable states.
    pub max_states: usize,
    /// Maximum `(state, token)` incidences in ONE state's row (bounds a single permissive state).
    pub max_row_pairs: usize,
    /// Maximum total incidences across the whole `Delta`.
    pub max_delta_pairs: usize,
}

impl Default for IndexBuildLimits {
    fn default() -> Self {
        Self {
            max_states: 1 << 22,
            max_row_pairs: 1 << 21,
            max_delta_pairs: 1 << 26,
        }
    }
}

/// Builds the full per-state transition relation by walking every reachable state's live subtree.
/// Only ever builds `TrieJointByte`/`TrieJointClass` - differential-tested correctness paths, not
pub fn build_delta(
    engine: &RefEngine,
    trie: &VocabTrie,
    mode: BindMode,
    provenance: Provenance,
) -> Result<CompiledIndex, CompileError> {
    build_delta_limited(engine, trie, mode, provenance, IndexBuildLimits::default())
}

fn build_delta_limited(
    engine: &RefEngine,
    trie: &VocabTrie,
    mode: BindMode,
    provenance: Provenance,
    limits: IndexBuildLimits,
) -> Result<CompiledIndex, CompileError> {
    match (mode, trie.kind()) {
        (BindMode::TrieJointByte, TrieKind::Byte) | (BindMode::TrieJointClass, TrieKind::Class) => {
        }
        _ => {
            return Err(CompileError::new(
                ErrorCode::ArtifactOutOfBounds,
                Stage::L4Bind,
                "bind mode does not match the trie alphabet",
            ))
        }
    }
    if trie.kind() == TrieKind::Class
        && trie.partition() != Some(super::trie::partition_fingerprint(engine))
    {
        return Err(CompileError::new(
            ErrorCode::ArtifactMismatch,
            Stage::L4Bind,
            "class-trie partition does not match the engine byte-class partition",
        ));
    }
    let count = engine.state_count();
    if count > limits.max_states {
        return Err(index_limit(
            "reachable state count exceeds the index build limit",
        ));
    }
    let mut scratch = WalkScratch::new();
    let mut delta = Vec::with_capacity(count);
    let mut budget = DeltaBudget::new(limits.max_row_pairs, limits.max_delta_pairs);
    for raw in 0..count {
        let s = StateId::try_from(raw).map_err(|_| state_overflow())?;
        delta.push(walk_one_state(trie, engine, &mut scratch, s, &mut budget)?);
    }
    Ok(CompiledIndex::new(delta, provenance, mode))
}

/// A `Delta` budget checked before each emitted pair: a per-row cap bounds one permissive state, a
/// total cap bounds the whole index. Both use checked arithmetic.
struct DeltaBudget {
    row_used: usize,
    row_max: usize,
    total_used: usize,
    total_max: usize,
}

impl DeltaBudget {
    fn new(row_max: usize, total_max: usize) -> Self {
        Self {
            row_used: 0,
            row_max,
            total_used: 0,
            total_max,
        }
    }

    fn start_row(&mut self) {
        self.row_used = 0;
    }

    fn charge_one(&mut self) -> Result<(), CompileError> {
        self.row_used = self
            .row_used
            .checked_add(1)
            .ok_or_else(|| index_limit("delta row pair count overflow"))?;
        if self.row_used > self.row_max {
            return Err(index_limit(
                "a state's row exceeds the index build row limit",
            ));
        }
        self.total_used = self
            .total_used
            .checked_add(1)
            .ok_or_else(|| index_limit("delta pair count overflow"))?;
        if self.total_used > self.total_max {
            return Err(index_limit(
                "delta pair count exceeds the index build limit",
            ));
        }
        Ok(())
    }
}

fn index_limit(msg: &'static str) -> CompileError {
    CompileError::new(ErrorCode::InternalLimitExceeded, Stage::L4Bind, msg)
}

/// Cap on one lazy query's transient scratch (just the DFS stack now - the livemask moved to
/// `RefEngine::live_classes`, charged once at engine-build time, not per walk), checked before any
const MAX_LAZY_TRANSIENT_BYTES: usize = 1 << 26;

/// Upper bound on one lazy walk's transient bytes: the DFS stack (never exceeds `max_dfs_frontier`
/// entries, the trie's own precomputed exact peak) plus the fixed step memo and header allowance.
fn lazy_walk_transient_bytes(max_dfs_frontier: usize) -> Option<usize> {
    let stack = packed_stack_bytes(max_dfs_frontier)?;
    stack
        .checked_add(STEP_MEMO_BYTES)?
        .checked_add(SCRATCH_HEADER_ALLOWANCE)
}

/// Writes `s`'s allowed-token mask DIRECTLY into `mask` (the lazy hot path): no `DeltaRow`, no
/// `(TokenId, StateId)` vector, landing states never materialized - bit-set is idempotent, so the
pub(crate) fn walk_state_mask_into(
    trie: &VocabTrie,
    engine: &RefEngine,
    s: StateId,
    mask: &mut Bitmask,
) -> Result<(), CompileError> {
    walk_state_mask_into_budgeted(trie, engine, s, mask, MAX_LAZY_TRANSIENT_BYTES)
}

/// `walk_state_mask_into` with an explicit transient-byte budget (tests can inject a tiny one
/// instead of needing a real oversized trie). Checked BEFORE any write to `mask`, so a rejection
fn walk_state_mask_into_budgeted(
    trie: &VocabTrie,
    engine: &RefEngine,
    s: StateId,
    mask: &mut Bitmask,
    max_transient_bytes: usize,
) -> Result<(), CompileError> {
    let max_dfs_frontier = trie.max_dfs_frontier();
    let estimated = lazy_walk_transient_bytes(max_dfs_frontier)
        .ok_or_else(|| index_limit("lazy walk transient estimate overflows"))?;
    if estimated > max_transient_bytes {
        return Err(index_limit("lazy walk transient memory exceeds the budget"));
    }
    let _permit = crate::mem_gate::try_acquire_vocab_serving_bytes(estimated)?;
    let stack_fuel = max_dfs_frontier;
    let mut scratch = WalkScratch::with_stack_capacity(stack_fuel)?;
    walk_live_subtree(trie, engine, &mut scratch, s, stack_fuel, |tok, _q| {
        mask.set(TokenId(tok))
    })
}

/// Fills a packed mask matrix DIRECTLY from the trie walk, one contiguous `wpr`-word row per state,
/// with NO intermediate `Delta` and NO per-row allocation. `rows.len()` must be `state_count * wpr`.
pub(crate) fn fill_packed_rows(
    trie: &VocabTrie,
    engine: &RefEngine,
    rows: &mut [u32],
    wpr: usize,
    vocab_size: usize,
    stack_fuel: usize,
) -> Result<(), CompileError> {
    let n = engine.state_count();
    if wpr != vocab_size.div_ceil(32) {
        return Err(CompileError::new(
            ErrorCode::ArtifactOutOfBounds,
            Stage::L4Bind,
            "packed words_per_row does not match ceil(vocab_size / 32)",
        ));
    }
    let expected = n
        .checked_mul(wpr)
        .ok_or_else(|| index_limit("packed rows length overflows"))?;
    if rows.len() != expected {
        return Err(CompileError::new(
            ErrorCode::ArtifactOutOfBounds,
            Stage::L4Bind,
            "packed rows length does not match state_count * words_per_row",
        ));
    }
    let mut scratch = WalkScratch::with_stack_capacity(trie.max_dfs_frontier())?;
    for raw in 0..n {
        let s = StateId::try_from(raw).map_err(|_| state_overflow())?;
        let base = raw * wpr; // base + wpr <= rows.len() (caller sized rows as n * wpr)
        let row = &mut rows[base..base + wpr];
        walk_live_subtree(trie, engine, &mut scratch, s, stack_fuel, |tok, _q| {
            let id = tok as usize;
            if id >= vocab_size {
                return Err(CompileError::new(
                    ErrorCode::ArtifactOutOfBounds,
                    Stage::L4Bind,
                    "a token id exceeds the mask vocab size",
                ));
            }
            row[id / 32] |= 1u32 << (id % 32);
            Ok(())
        })?;
    }
    Ok(())
}

/// Walks the live subtree under `s`, emitting every allowed token with its landing state.
/// Iterative (never recursive) so a deep trie cannot overflow the stack.
fn walk_one_state(
    trie: &VocabTrie,
    engine: &RefEngine,
    scratch: &mut WalkScratch,
    s: StateId,
    budget: &mut DeltaBudget,
) -> Result<DeltaRow, CompileError> {
    let mut out = DeltaRow::default();
    budget.start_row();
    walk_live_subtree(trie, engine, scratch, s, usize::MAX, |tok, q| {
        budget.charge_one()?; // bound the row DURING construction, not after
        out.push(TokenId(tok), q);
        Ok(())
    })?;
    Ok(out)
}

/// The one dead-pruned joint DFS, shared by the row and direct-mask consumers so they can never
/// diverge. `emit(token, landing_state)` is called for every allowed token; the caller decides
fn walk_live_subtree<F>(
    trie: &VocabTrie,
    engine: &RefEngine,
    scratch: &mut WalkScratch,
    s: StateId,
    stack_fuel: usize,
    mut emit: F,
) -> Result<(), CompileError>
where
    F: FnMut(u32, StateId) -> Result<(), CompileError>,
{
    let dead = engine.dead();
    let start_live = start_livemask_bug().then(|| engine.live_classes(s));
    scratch.stack.clear();
    scratch.stack.push((TrieNodeId(0), s));
    while let Some((n, q)) = scratch.stack.pop() {
        record_pop();
        if q == dead {
            continue; // reached only with pruning disabled; skipping keeps the output correct
        }
        for &tok in trie.leaf_tokens_of(n)? {
            emit(tok, q)?;
        }
        let live = match start_live {
            Some(buggy) => buggy,
            None => engine.live_classes(q), // keyed by the CURRENT state q, never the start
        };
        scratch.memo.bump();
        let (keys, kids) = trie.children(n)?;
        for (&key, &child) in keys.iter().zip(kids) {
            let class = trie.class_of_edge(key, engine);
            let ci = usize::from(class.get());
            if !livemask_all_ones() && !live.contains(ci) {
                continue; // dead-from-q classes cannot survive; a pure pre-filter (T-F1)
            }
            let q2 = scratch.memo.get_or(ci, || edge_step(engine, q, class));
            if q2 != dead || dead_prune_disabled() {
                if scratch.stack.len() >= stack_fuel {
                    return Err(index_limit("joint-walk stack exceeds the build fuel"));
                }
                scratch.stack.push((child, q2));
            }
        }
    }
    Ok(())
}

/// Per-walk scratch: a per-node step memo and the reusable DFS stack. The livemask that used to
/// live here (SPARSE hashmap or DENSE table, recomputed/memoized per walk) moved to
struct WalkScratch {
    memo: StepMemo,
    stack: Vec<(TrieNodeId, StateId)>,
}

impl WalkScratch {
    fn new() -> Self {
        Self {
            memo: StepMemo::new(),
            stack: Vec::new(),
        }
    }

    /// A scratch whose DFS stack is fallibly reserved to `stack_capacity` up front, so an
    /// allocator failure is a typed error, never an abort - used by both the lazy per-query path
    fn with_stack_capacity(stack_capacity: usize) -> Result<Self, CompileError> {
        let mut stack = Vec::new();
        stack
            .try_reserve_exact(stack_capacity)
            .map_err(|_| index_limit("walk stack allocation failed"))?;
        Ok(Self {
            memo: StepMemo::new(),
            stack,
        })
    }
}

/// Fixed scratch bytes the packed fill retains, EXCLUDING the DFS stack (the caller sizes the stack
/// from the remaining budget and passes it as fuel): just the fixed step memo and a small header
pub(crate) fn packed_scratch_fixed_bytes() -> Option<usize> {
    Some(STEP_MEMO_BYTES + SCRATCH_HEADER_ALLOWANCE)
}

/// EXACT DFS-stack bytes for a walk over a trie whose precomputed exact peak is
/// `max_dfs_frontier` (`VocabTrie::max_dfs_frontier`): the dense scratch reserves the stack to that
pub(crate) fn packed_stack_bytes(max_dfs_frontier: usize) -> Option<usize> {
    max_dfs_frontier.checked_mul(std::mem::size_of::<(TrieNodeId, StateId)>())
}

/// The step memo's fixed footprint: a 256-entry generation table plus a 256-entry value table.
const STEP_MEMO_BYTES: usize = 256 * (std::mem::size_of::<u32>() + std::mem::size_of::<StateId>());

/// Small fixed allowance for the scratch `Vec` headers and allocator rounding.
const SCRATCH_HEADER_ALLOWANCE: usize = 4096;

/// A versioned per-node cache of `step(q, class)`, avoiding a re-step for byte children sharing a
/// class. Sized to the whole 256-class space so a foreign class id can never index out of range.
struct StepMemo {
    generation: Vec<u32>,
    value: Vec<StateId>,
    current: u32,
}

impl StepMemo {
    fn new() -> Self {
        Self {
            generation: vec![0; 256],
            value: vec![StateId(0); 256],
            current: 0,
        }
    }

    /// Starts a fresh node, invalidating the previous node's cached steps in O(1).
    fn bump(&mut self) {
        self.current = self.current.wrapping_add(1);
        if self.current == 0 {
            self.generation.iter_mut().for_each(|g| *g = 0);
            self.current = 1;
        }
    }

    fn get_or(&mut self, c: usize, compute: impl FnOnce() -> StateId) -> StateId {
        if self.generation[c] == self.current {
            return self.value[c];
        }
        let v = compute();
        self.generation[c] = self.current;
        self.value[c] = v;
        v
    }
}

#[cfg(test)]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[cfg(test)]
static EDGE_STEPS: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
static NODES_POPPED: AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
static LIVEMASK_ALL_ONES: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static DEAD_PRUNE_DISABLED: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static START_LIVEMASK_BUG: AtomicBool = AtomicBool::new(false);

#[inline]
fn edge_step(engine: &RefEngine, q: StateId, class: ByteClassId) -> StateId {
    #[cfg(test)]
    EDGE_STEPS.fetch_add(1, Ordering::Relaxed);
    engine.step(q, class)
}

#[inline]
fn record_pop() {
    #[cfg(test)]
    NODES_POPPED.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
fn livemask_all_ones() -> bool {
    LIVEMASK_ALL_ONES.load(Ordering::Relaxed)
}
#[cfg(not(test))]
fn livemask_all_ones() -> bool {
    false
}

#[cfg(test)]
fn dead_prune_disabled() -> bool {
    DEAD_PRUNE_DISABLED.load(Ordering::Relaxed)
}
#[cfg(not(test))]
fn dead_prune_disabled() -> bool {
    false
}

#[cfg(test)]
fn start_livemask_bug() -> bool {
    START_LIVEMASK_BUG.load(Ordering::Relaxed)
}
#[cfg(not(test))]
fn start_livemask_bug() -> bool {
    false
}

fn state_overflow() -> CompileError {
    CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L4Bind,
        "reference state count exceeds the StateId width",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::build_from_regex;
    use crate::mask::Bitmask;
    use crate::vocab::{build_vocabulary, Vocabulary};
    use rustc_hash::FxHashMap;

    static GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());
    fn guard() -> std::sync::MutexGuard<'static, ()> {
        GUARD.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn vocab(pairs: &[(&[u8], &[u32])]) -> Vocabulary {
        let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
        for &(bytes, ids) in pairs {
            map.insert(bytes.to_vec(), ids.to_vec());
        }
        build_vocabulary(9999, map).expect("vocab")
    }

    /// The independent reference: the naive per-token walk, sharing no code with the trie walk.
    fn naive_mask(engine: &RefEngine, v: &Vocabulary, s: StateId, width: usize) -> Bitmask {
        let mut m = Bitmask::zeros(width);
        for (bytes, ids) in v.tokens() {
            if engine.consume_token(s, bytes).is_some() {
                for &id in ids {
                    m.set(TokenId(id)).unwrap();
                }
            }
        }
        m
    }

    fn row_mask(row: &DeltaRow, width: usize) -> Bitmask {
        let mut m = Bitmask::zeros(width);
        for (tok, _land) in row.iter() {
            m.set(tok).unwrap();
        }
        m
    }

    /// The mask width covering every token id plus EOS.
    fn width(v: &Vocabulary) -> usize {
        v.tokens()
            .values()
            .flatten()
            .copied()
            .max()
            .map_or(v.eos_token_id(), |m| m.max(v.eos_token_id())) as usize
            + 1
    }

    fn byte_index(engine: &RefEngine, v: &Vocabulary) -> CompiledIndex {
        let trie = VocabTrie::build_byte(v).unwrap();
        build_delta(engine, &trie, BindMode::TrieJointByte, prov()).unwrap()
    }

    fn class_index(engine: &RefEngine, v: &Vocabulary) -> CompiledIndex {
        let trie = VocabTrie::build_class(v, engine).unwrap();
        build_delta(engine, &trie, BindMode::TrieJointClass, prov()).unwrap()
    }

    fn prov() -> Provenance {
        Provenance::reference(crate::error::HashState::Hash([5; 32]))
    }

    /// Asserts a trie index agrees with the independent naive walk on every reachable state.
    fn assert_matches_naive(engine: &RefEngine, v: &Vocabulary, idx: &CompiledIndex) {
        let w = width(v);
        for raw in 0..engine.state_count() {
            let s = StateId(raw as u32);
            let naive = naive_mask(engine, v, s, w);
            let trie = row_mask(idx.row(s).unwrap(), w);
            assert_eq!(
                trie.as_words(),
                naive.as_words(),
                "state {raw} mask diverged from naive"
            );
        }
    }

    #[test]
    fn byte_route_equals_naive_on_string_pattern() {
        let _g = guard();
        let engine = build_from_regex("\"(cat|car|carbon)\"").unwrap();
        let v = vocab(&[
            (b"\"", &[0]),
            (b"c", &[1]),
            (b"a", &[2]),
            (b"t", &[3]),
            (b"r", &[4]),
            (b"b", &[5]),
            (b"o", &[6]),
            (b"n", &[7]),
            (b"cat", &[8]),
            (b"x", &[9]),
        ]);
        assert_matches_naive(&engine, &v, &byte_index(&engine, &v));
    }

    #[test]
    fn class_trie_equals_byte_trie_and_naive() {
        let _g = guard();
        let engine = build_from_regex("[0-9]{1,3}").unwrap();
        let v = vocab(&[
            (b"1", &[0]),
            (b"2", &[1]),
            (b"3", &[2]),
            (b"12", &[3]),
            (b"123", &[4]),
            (b"a", &[5]),
        ]);
        let byte = byte_index(&engine, &v);
        let class = class_index(&engine, &v);
        assert_matches_naive(&engine, &v, &byte);
        assert_matches_naive(&engine, &v, &class);
        let w = width(&v);
        for raw in 0..engine.state_count() {
            let s = StateId(raw as u32);
            assert_eq!(
                row_mask(byte.row(s).unwrap(), w).as_words(),
                row_mask(class.row(s).unwrap(), w).as_words(),
                "byte and class routes diverged at state {raw}"
            );
        }
    }

    #[test]
    fn state_dependent_livemask_open_quote_then_ordinary_char() {
        let _g = guard();
        let engine = build_from_regex("\"[a-z]\"").unwrap();
        let v = vocab(&[(b"\"", &[0]), (b"a", &[1]), (b"\"a", &[2])]);
        let idx = byte_index(&engine, &v);
        assert_matches_naive(&engine, &v, &idx);
        let after_quote = engine.consume_token(engine.start(), b"\"").unwrap();
        let w = width(&v);
        assert!(row_mask(idx.row(after_quote).unwrap(), w).get(TokenId(1)));
    }

    #[test]
    fn buggy_start_livemask_drops_a_multibyte_token_whose_tail_needs_a_late_class() {
        let _g = guard();
        let engine = build_from_regex("\"[a-z]\"").unwrap();
        let v = vocab(&[(b"\"", &[0]), (b"\"a", &[1])]);
        let start = engine.start();
        let w = width(&v);

        let correct = byte_index(&engine, &v);
        assert!(row_mask(correct.row(start).unwrap(), w).get(TokenId(1)));

        START_LIVEMASK_BUG.store(true, Ordering::Relaxed);
        let buggy = byte_index(&engine, &v);
        START_LIVEMASK_BUG.store(false, Ordering::Relaxed);
        assert!(
            !row_mask(buggy.row(start).unwrap(), w).get(TokenId(1)),
            "the start-livemask bug must drop the multibyte token (proving the test is not vacuous)"
        );
        assert!(row_mask(byte_index(&engine, &v).row(start).unwrap(), w).get(TokenId(1)));
    }

    #[test]
    fn dead_prune_disabled_stays_correct_but_pops_more_nodes() {
        let _g = guard();
        let engine = build_from_regex("(cat|car|carbon)").unwrap();
        let v = vocab(&[(b"cat", &[0]), (b"car", &[1]), (b"dog", &[2]), (b"c", &[3])]);

        LIVEMASK_ALL_ONES.store(true, Ordering::Relaxed);

        NODES_POPPED.store(0, Ordering::Relaxed);
        let pruned = byte_index(&engine, &v);
        let pruned_pops = NODES_POPPED.load(Ordering::Relaxed);

        DEAD_PRUNE_DISABLED.store(true, Ordering::Relaxed);
        NODES_POPPED.store(0, Ordering::Relaxed);
        let unpruned = byte_index(&engine, &v);
        let unpruned_pops = NODES_POPPED.load(Ordering::Relaxed);
        DEAD_PRUNE_DISABLED.store(false, Ordering::Relaxed);
        LIVEMASK_ALL_ONES.store(false, Ordering::Relaxed);

        assert_matches_naive(&engine, &v, &pruned);
        assert_matches_naive(&engine, &v, &unpruned); // still correct
        assert!(
            unpruned_pops > pruned_pops,
            "disabling the dead-prune must visit strictly more nodes ({unpruned_pops} vs {pruned_pops})"
        );
    }

    #[test]
    fn livemask_all_ones_stays_correct_but_steps_more() {
        let _g = guard();
        let engine = build_from_regex("(cat|car|carbon)").unwrap();
        let v = vocab(&[(b"cat", &[0]), (b"car", &[1]), (b"z", &[2])]);

        EDGE_STEPS.store(0, Ordering::Relaxed);
        let filtered = byte_index(&engine, &v);
        let filtered_steps = EDGE_STEPS.load(Ordering::Relaxed);

        LIVEMASK_ALL_ONES.store(true, Ordering::Relaxed);
        EDGE_STEPS.store(0, Ordering::Relaxed);
        let unfiltered = byte_index(&engine, &v);
        let unfiltered_steps = EDGE_STEPS.load(Ordering::Relaxed);
        LIVEMASK_ALL_ONES.store(false, Ordering::Relaxed);

        assert_matches_naive(&engine, &v, &filtered);
        assert_matches_naive(&engine, &v, &unfiltered);
        assert!(
            unfiltered_steps > filtered_steps,
            "the livemask AND must remove edge steps ({unfiltered_steps} vs {filtered_steps})"
        );
    }

    #[test]
    fn shared_prefix_walk_steps_fewer_than_the_naive_restart() {
        let _g = guard();
        let engine = build_from_regex("(ca|cat|car|carbon)").unwrap();
        let v = vocab(&[
            (b"ca", &[0]),
            (b"cat", &[1]),
            (b"car", &[2]),
            (b"carbon", &[3]),
        ]);
        let trie = VocabTrie::build_byte(&v).unwrap();

        EDGE_STEPS.store(0, Ordering::Relaxed);
        let mut scratch = WalkScratch::new();
        let mut budget = DeltaBudget::new(usize::MAX, usize::MAX);
        let _ = walk_one_state(&trie, &engine, &mut scratch, engine.start(), &mut budget).unwrap();
        let trie_steps = EDGE_STEPS.load(Ordering::Relaxed);

        EDGE_STEPS.store(0, Ordering::Relaxed);
        for bytes in v.tokens().keys() {
            let mut cur = engine.start();
            for &b in bytes {
                if cur == engine.dead() {
                    break;
                }
                cur = edge_step(&engine, cur, engine.class_of_byte(b));
            }
        }
        let naive_steps = EDGE_STEPS.load(Ordering::Relaxed);
        assert!(
            trie_steps < naive_steps,
            "the shared prefix must be walked once ({trie_steps} trie steps vs {naive_steps} naive)"
        );
    }

    #[test]
    fn empty_language_and_dead_state_have_empty_rows() {
        let _g = guard();
        let engine = build_from_regex("[a&&b]").unwrap(); // empty language: start is DEAD
        let v = vocab(&[(b"a", &[0]), (b"b", &[1])]);
        let idx = byte_index(&engine, &v);
        for raw in 0..engine.state_count() {
            assert!(idx.row(StateId(raw as u32)).unwrap().is_empty());
        }
    }

    #[test]
    fn stack_scratch_rejects_an_unsatisfiable_allocation_instead_of_aborting() {
        match WalkScratch::with_stack_capacity(usize::MAX / 2) {
            Err(e) => assert_eq!(e.code, ErrorCode::InternalLimitExceeded),
            Ok(_) => panic!("an allocation this large must not succeed"),
        }
    }

    #[test]
    fn deep_trie_uses_the_iterative_stack_without_overflow() {
        let _g = guard();
        let engine = build_from_regex("a+").unwrap();
        let long: Vec<u8> = vec![b'a'; 20_000];
        let v = vocab(&[(&long, &[0]), (b"a", &[1])]);
        let idx = byte_index(&engine, &v);
        assert_matches_naive(&engine, &v, &idx);
    }

    #[test]
    fn every_delta_landing_equals_the_naive_consume_token_landing() {
        let _g = guard();
        let engine = build_from_regex("(cat|car|carbon)").unwrap();
        let v = vocab(&[
            (b"c", &[0]),
            (b"ca", &[1]),
            (b"car", &[2]),
            (b"carbon", &[3]),
        ]);
        let idx = byte_index(&engine, &v);
        for raw in 0..engine.state_count() {
            let s = StateId(raw as u32);
            for (tok, landing) in idx.row(s).unwrap().iter() {
                let bytes = v
                    .tokens()
                    .iter()
                    .find(|(_, ids)| ids.contains(&tok.get()))
                    .map(|(b, _)| b.clone())
                    .unwrap();
                let expected = engine.consume_token(s, &bytes).unwrap();
                assert_eq!(landing, expected, "token {tok:?} landing at state {raw}");
                assert_ne!(
                    landing,
                    engine.dead(),
                    "an emitted token never lands on DEAD"
                );
            }
        }
    }

    #[test]
    fn a_perturbed_delta_landing_is_caught_by_the_naive_landing_diff() {
        let _g = guard();
        let engine = build_from_regex("(cat|car|carbon)").unwrap();
        let v = vocab(&[
            (b"c", &[0]),
            (b"ca", &[1]),
            (b"car", &[2]),
            (b"carbon", &[3]),
        ]);
        let idx = byte_index(&engine, &v);

        let mut observed: Vec<(u32, u32, u32)> = Vec::new();
        for raw in 0..engine.state_count() {
            let s = StateId(raw as u32);
            for (tok, landing) in idx.row(s).unwrap().iter() {
                observed.push((raw as u32, tok.get(), landing.get()));
            }
        }

        let landing_matches = |set: &[(u32, u32, u32)]| {
            set.iter().all(|&(raw, tok, landing)| {
                let bytes = v
                    .tokens()
                    .iter()
                    .find(|(_, ids)| ids.contains(&tok))
                    .map(|(b, _)| b.clone())
                    .unwrap();
                engine.consume_token(StateId(raw), &bytes).unwrap().get() == landing
            })
        };
        assert!(
            landing_matches(&observed),
            "unperturbed landings must match naive"
        );

        let mut perturbed = observed.clone();
        let (_, _, landing) = &mut perturbed[0];
        *landing = landing.wrapping_add(1) % engine.state_count() as u32;
        assert!(
            !landing_matches(&perturbed),
            "a single perturbed landing must be rejected (the diff is not vacuous)"
        );
    }

    #[test]
    fn count_conservation_dropping_one_signature_token_would_be_caught() {
        let _g = guard();
        let engine = build_from_regex("[0-9]+").unwrap();
        let v = vocab(&[(b"1", &[0]), (b"2", &[1]), (b"3", &[2]), (b"12", &[3])]);
        let idx = class_index(&engine, &v);
        let s = engine.start();
        let mut allowed: Vec<u32> = idx.row(s).unwrap().iter().map(|(t, _)| t.get()).collect();
        allowed.sort_unstable();
        let w = width(&v);
        let naive = naive_mask(&engine, &v, s, w);
        let naive_ids: Vec<u32> = (0..w as u32).filter(|&i| naive.get(TokenId(i))).collect();
        assert_eq!(allowed, naive_ids);
        let mut dropped = allowed.clone();
        dropped.pop();
        assert_ne!(dropped, naive_ids);
    }

    #[test]
    fn delta_pair_cap_trips_during_row_construction_no_partial_artifact() {
        let _g = guard();
        let engine = build_from_regex("[a-z]+").unwrap();
        let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
        for (i, b) in (b'a'..=b'z').enumerate() {
            map.insert(vec![b], vec![i as u32]);
        }
        let v = build_vocabulary(9999, map).unwrap();
        let trie = VocabTrie::build_byte(&v).unwrap();

        let mut scratch = WalkScratch::new();
        let mut budget = DeltaBudget::new(usize::MAX, 1); // total cap 1, row cap unlimited
        let err = walk_one_state(&trie, &engine, &mut scratch, engine.start(), &mut budget)
            .expect_err("the second emitted pair must trip the cap");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
        assert!(
            budget.total_used <= 2,
            "the walk stopped at the cap, not after emitting the whole row (used={})",
            budget.total_used
        );
        assert!(build_delta(&engine, &trie, BindMode::TrieJointByte, prov()).is_ok());
    }

    #[test]
    fn delta_pair_cap_boundaries_zero_exact_and_one_under() {
        let _g = guard();
        let engine = build_from_regex("(true|false)").unwrap();
        let v = vocab(&[
            (b"t", &[0]),
            (b"true", &[1]),
            (b"f", &[2]),
            (b"false", &[3]),
            (b"x", &[4]),
        ]);
        let trie = VocabTrie::build_byte(&v).unwrap();

        let total = build_delta(&engine, &trie, BindMode::TrieJointByte, prov())
            .unwrap()
            .live_pairs();
        assert!(total >= 1);

        let with = |cap: usize| {
            build_delta_limited(
                &engine,
                &trie,
                BindMode::TrieJointByte,
                prov(),
                IndexBuildLimits {
                    max_states: 1 << 20,
                    max_row_pairs: usize::MAX,
                    max_delta_pairs: cap,
                },
            )
        };
        assert!(with(total).is_ok(), "the exact total must fit");
        assert_eq!(
            with(total - 1).unwrap_err().code,
            ErrorCode::InternalLimitExceeded
        );
        assert_eq!(with(0).unwrap_err().code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn delta_row_cap_boundaries_and_global_interaction() {
        let _g = guard();
        let engine = build_from_regex("[a-z]+").unwrap();
        let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
        for (i, b) in (b'a'..=b'z').enumerate() {
            map.insert(vec![b], vec![i as u32]);
        }
        let v = build_vocabulary(9999, map).unwrap();
        let trie = VocabTrie::build_byte(&v).unwrap();

        let start_row = build_delta(&engine, &trie, BindMode::TrieJointByte, prov())
            .unwrap()
            .max_row_len();
        assert_eq!(start_row, 26);

        let with = |row: usize, total: usize| {
            build_delta_limited(
                &engine,
                &trie,
                BindMode::TrieJointByte,
                prov(),
                IndexBuildLimits {
                    max_states: 1 << 20,
                    max_row_pairs: row,
                    max_delta_pairs: total,
                },
            )
        };
        assert!(with(start_row, usize::MAX).is_ok(), "exact row must fit");
        assert_eq!(
            with(start_row - 1, usize::MAX).unwrap_err().code,
            ErrorCode::InternalLimitExceeded
        );
        assert_eq!(
            with(0, usize::MAX).unwrap_err().code,
            ErrorCode::InternalLimitExceeded
        );
        assert_eq!(
            with(usize::MAX, 1).unwrap_err().code,
            ErrorCode::InternalLimitExceeded
        );
    }

    #[test]
    fn fill_packed_rows_rejects_a_mismatched_buffer_shape() {
        let _g = guard();
        let engine = build_from_regex("(true|false)").unwrap();
        let v = vocab(&[
            (b"t", &[0]),
            (b"true", &[1]),
            (b"f", &[2]),
            (b"false", &[3]),
        ]);
        let trie = VocabTrie::build_byte(&v).unwrap();
        let n = engine.state_count();
        let vocab_size = width(&v);
        let wpr = vocab_size.div_ceil(32);

        let mut good = vec![0u32; n * wpr];
        fill_packed_rows(&trie, &engine, &mut good, wpr, vocab_size, usize::MAX).unwrap();
        let mut wrong_len = vec![0u32; n * wpr + 1];
        assert_eq!(
            fill_packed_rows(&trie, &engine, &mut wrong_len, wpr, vocab_size, usize::MAX)
                .unwrap_err()
                .code,
            ErrorCode::ArtifactOutOfBounds
        );
        let mut right_len = vec![0u32; n * wpr];
        assert_eq!(
            fill_packed_rows(
                &trie,
                &engine,
                &mut right_len,
                wpr + 1,
                vocab_size,
                usize::MAX
            )
            .unwrap_err()
            .code,
            ErrorCode::ArtifactOutOfBounds
        );
    }

    #[test]
    fn wrong_class_map_is_rejected_structurally_not_walked() {
        let _g = guard();
        let digit_engine = build_from_regex("[0-9]+").unwrap();
        let letter_engine = build_from_regex("[a-z]+").unwrap();
        let v = vocab(&[(b"a", &[0]), (b"b", &[1]), (b"1", &[2])]);
        let trie = VocabTrie::build_class(&v, &digit_engine).unwrap();
        let err = build_delta(&letter_engine, &trie, BindMode::TrieJointClass, prov())
            .expect_err("a mismatched class partition must be rejected");
        assert_eq!(err.code, ErrorCode::ArtifactMismatch);
    }

    #[test]
    fn lazy_walk_matches_naive_below_the_default_budget() {
        let _g = guard();
        let engine = build_from_regex("(cat|car|carbon)").unwrap();
        let v = vocab(&[
            (b"c", &[0]),
            (b"ca", &[1]),
            (b"car", &[2]),
            (b"carbon", &[3]),
        ]);
        let trie = VocabTrie::build_byte(&v).unwrap();
        let w = width(&v);
        for raw in 0..engine.state_count() {
            let s = StateId(raw as u32);
            let mut mask = Bitmask::zeros(w);
            walk_state_mask_into(&trie, &engine, s, &mut mask).unwrap();
            assert_eq!(mask, naive_mask(&engine, &v, s, w), "state {raw}");
        }
    }

    #[test]
    fn lazy_walk_matches_naive_on_a_wide_trie() {
        let _g = guard();
        let engine = build_from_regex("[\\x00-\\xc7]").unwrap();
        let pairs: Vec<(Vec<u8>, Vec<u32>)> =
            (0u8..=199).map(|b| (vec![b], vec![u32::from(b)])).collect();
        let refs: Vec<(&[u8], &[u32])> = pairs
            .iter()
            .map(|(b, i)| (b.as_slice(), i.as_slice()))
            .collect();
        let v = vocab(&refs);
        let trie = VocabTrie::build_byte(&v).unwrap();
        let w = width(&v);
        let mut mask = Bitmask::zeros(w);
        walk_state_mask_into(&trie, &engine, engine.start(), &mut mask).unwrap();
        assert_eq!(mask, naive_mask(&engine, &v, engine.start(), w));
    }

    #[test]
    fn lazy_walk_matches_naive_on_a_deep_trie() {
        let _g = guard();
        let engine = build_from_regex("a+").unwrap();
        let long: Vec<u8> = vec![b'a'; 20_000];
        let v = vocab(&[(&long, &[0]), (b"a", &[1])]);
        let trie = VocabTrie::build_byte(&v).unwrap();
        let w = width(&v);
        let mut mask = Bitmask::zeros(w);
        walk_state_mask_into(&trie, &engine, engine.start(), &mut mask).unwrap();
        assert_eq!(mask, naive_mask(&engine, &v, engine.start(), w));
    }

    #[test]
    fn lazy_walk_rejects_under_a_tiny_injected_budget() {
        let _g = guard();
        let engine = build_from_regex("a+").unwrap();
        let v = vocab(&[(b"a", &[0]), (b"b", &[1])]);
        let trie = VocabTrie::build_byte(&v).unwrap();
        let mut mask = Bitmask::zeros(width(&v));
        let err = walk_state_mask_into_budgeted(&trie, &engine, engine.start(), &mut mask, 4)
            .expect_err("a 4-byte budget cannot hold even the smallest real walk");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn lazy_walk_budget_rejection_leaves_the_mask_untouched() {
        let _g = guard();
        let engine = build_from_regex("a+").unwrap();
        let v = vocab(&[(b"a", &[0]), (b"b", &[1])]);
        let trie = VocabTrie::build_byte(&v).unwrap();
        let mut mask = Bitmask::zeros(width(&v));
        walk_state_mask_into_budgeted(&trie, &engine, engine.start(), &mut mask, 4).unwrap_err();
        assert_eq!(
            mask.count_ones(),
            0,
            "a rejected admission must not have written any bit"
        );
    }

    #[test]
    fn lazy_walk_binds_once_the_budget_covers_the_real_estimate() {
        let _g = guard();
        let engine = build_from_regex("a+").unwrap();
        let v = vocab(&[(b"a", &[0]), (b"b", &[1])]);
        let trie = VocabTrie::build_byte(&v).unwrap();
        let generous = lazy_walk_transient_bytes(trie.max_dfs_frontier()).unwrap() + 1;
        let mut mask = Bitmask::zeros(width(&v));
        assert!(
            walk_state_mask_into_budgeted(&trie, &engine, engine.start(), &mut mask, generous)
                .is_ok()
        );
    }

    #[test]
    fn lazy_walk_is_safe_under_concurrent_cold_calls_on_one_trie() {
        let _g = guard();
        let engine = std::sync::Arc::new(build_from_regex("(cat|car|carbon)").unwrap());
        let v = vocab(&[
            (b"c", &[0]),
            (b"ca", &[1]),
            (b"car", &[2]),
            (b"carbon", &[3]),
        ]);
        let trie = std::sync::Arc::new(VocabTrie::build_byte(&v).unwrap());
        let w = width(&v);
        let handles: Vec<_> = (0..engine.state_count())
            .map(|raw| {
                let engine = engine.clone();
                let trie = trie.clone();
                std::thread::spawn(move || {
                    let s = StateId(raw as u32);
                    let mut mask = Bitmask::zeros(w);
                    walk_state_mask_into(&trie, &engine, s, &mut mask).unwrap();
                    mask
                })
            })
            .collect();
        for (raw, h) in handles.into_iter().enumerate() {
            let mask = h.join().unwrap();
            let s = StateId(raw as u32);
            assert_eq!(mask, naive_mask(&engine, &v, s, w), "state {raw}");
        }
    }

    #[test]
    fn lazy_serving_does_not_wait_behind_a_full_build_gate() {
        let _g = guard();
        let _build_hog = crate::mem_gate::acquire_vocab_build_bytes(1).unwrap();
        let engine = build_from_regex("a+").unwrap();
        let v = vocab(&[(b"a", &[0]), (b"b", &[1])]);
        let trie = VocabTrie::build_byte(&v).unwrap();
        let mut mask = Bitmask::zeros(width(&v));
        let t = std::time::Instant::now();
        walk_state_mask_into(&trie, &engine, engine.start(), &mut mask).unwrap();
        assert!(
            t.elapsed() < std::time::Duration::from_secs(1),
            "a lazy query must not queue behind the build gate"
        );
    }
}
