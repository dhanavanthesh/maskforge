//! Provides compiled artifacts and stateful matchers.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use crate::automaton::RefEngine;
use crate::error::{CompileError, ErrorCode, MatcherError, Provenance, Stage};
use crate::index::{
    build_delta, BindMode, BoundByteTrie, CompiledIndex, VocabFingerprint, VocabTrie,
    VocabularyHandle,
};
use crate::mask::Bitmask;
use crate::primitives::{StateId, TokenId};
use crate::vocab::{CanonicalVocabulary, Vocabulary};

mod artifact_cache;
mod executable_cache;
pub use artifact_cache::{ArtifactCache, COMPILER_SEMANTICS_VERSION};
pub use executable_cache::{
    global_executable_cache, CachedExecutable, ExecutableCache, ExecutableCacheStats,
    ExecutableSchema, EXECUTABLE_SEMANTICS_VERSION,
};

/// Total mask-buffer cap in `u32` words: `rows * words_per_row` above this is rejected before
/// allocation, so a large row index cannot force a multi-gigabyte buffer. `1 << 26` words = 256 MiB.
const MAX_MASK_WORDS: usize = 1 << 26;

/// Byte budget for the per-state mask cache. Once this many bytes are retained, further states are
/// computed but not cached, so a large DFA plus a wide vocabulary cannot retain unbounded memory.
const MAX_MASK_CACHE_BYTES: usize = 1 << 26;

/// Conservative per-entry overhead charged on top of a cached mask's words: the `Arc<Bitmask>`
/// control block, the `Vec` header, and the hash-map slot. Keeps the byte budget from being
const MASK_CACHE_ENTRY_OVERHEAD: usize = 64;

/// Budget for the INCREMENTAL packed-build peak of the `TrieJointBytePacked` mode: the payload
/// table, a fixed header allowance, the dense walk scratch, and the DFS stack's exact peak (see
const MAX_PACKED_MASK_BYTES: usize = 1 << 28;

/// Measured packed row-fill cost per `u32` word: the delta between packed and forced-lazy
/// `ir_to_index` time divided by `state_count * words_per_row`, calibrated on gpt2 (wpr=1571,
const PACKED_ROW_FILL_NS_PER_WORD: usize = 2;

/// Target ceiling on the packed row-fill contribution to compile latency - a policy choice, not a
/// measured fact: 20ms keeps an eager packed compile from becoming visibly slow relative to a
const PACKED_COMPILE_LATENCY_BUDGET_US: usize = 20_000;

/// Fixed allowance added to the incremental packed-build peak estimate for the boxed-slice header, the
/// artifact struct growth, and allocator rounding, so the declared budget is not silently exceeded by
const PACKED_HEADER_ALLOWANCE: usize = 1 << 16;

/// One contiguous packed mask matrix: `rows` holds `state_count * words_per_row` little-endian `u32`
/// words, row `s` at `[s*wpr .. (s+1)*wpr]`. A `Box<[u32]>` (NOT `Arc<[u32]>`): `CompiledArtifact` is
#[derive(Debug)]
struct PackedMaskTable {
    words_per_row: usize,
    rows: Box<[u32]>,
}

impl PackedMaskTable {
    /// The packed words for `s`, or `None` for an out-of-range (forged) state. Borrowed, no copy.
    fn row(&self, s: StateId) -> Option<&[u32]> {
        let wpr = self.words_per_row;
        let start = (s.get() as usize).checked_mul(wpr)?;
        let end = start.checked_add(wpr)?;
        self.rows.get(start..end)
    }

    /// Retained payload bytes (the table words; metadata is a fixed small struct).
    fn retained_bytes(&self) -> usize {
        self.rows.len().saturating_mul(4)
    }
}

/// A slot's outcome once decided. `Uncached` is permanent: production budgets never grow after
/// construction, so a state rejected once stays rejected for the artifact's lifetime.
#[derive(Clone, Debug)]
enum SlotValue {
    Cached(Result<Arc<Bitmask>, Arc<CompileError>>),
    Uncached,
}

type MaskSlot = OnceLock<SlotValue>;

/// Cap on the `LazyRows` slot array itself (not the masks it later retains): `1 << 28` = 256 MiB,
/// mirroring the packed budget.
const MAX_LAZY_ROWS_BYTES: usize = 1 << 28;

/// One `OnceLock` per state: a query for `s` never contends with a query for a different state.
#[derive(Debug)]
struct LazyRows {
    rows: Box<[MaskSlot]>,
    retained_bytes: AtomicUsize,
    budget_bytes: AtomicUsize,
}

impl LazyRows {
    fn try_new(state_count: usize, budget_bytes: usize) -> Result<Self, CompileError> {
        state_count
            .checked_mul(std::mem::size_of::<MaskSlot>())
            .filter(|&b| b <= MAX_LAZY_ROWS_BYTES)
            .ok_or_else(|| {
                CompileError::new(
                    ErrorCode::InternalLimitExceeded,
                    Stage::L4Bind,
                    "lazy row slot array exceeds the size cap",
                )
            })?;
        let mut rows = Vec::new();
        rows.try_reserve_exact(state_count).map_err(|_| {
            CompileError::new(
                ErrorCode::InternalLimitExceeded,
                Stage::L4Bind,
                "lazy row slot array allocation failed",
            )
        })?;
        rows.resize_with(state_count, OnceLock::new);
        Ok(Self {
            rows: rows.into_boxed_slice(),
            retained_bytes: AtomicUsize::new(0),
            budget_bytes: AtomicUsize::new(budget_bytes),
        })
    }

    fn try_reserve(&self, bytes: usize) -> bool {
        let budget = self.budget_bytes.load(Ordering::Acquire);
        self.retained_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(bytes).filter(|&total| total <= budget)
            })
            .is_ok()
    }

    /// `state` must be `< state_count` (checked by the caller). Only the `OnceLock`-winning
    /// closure ever reserves; losing racers read its published result and never reserve.
    fn get_or_compute(
        &self,
        state: usize,
        entry_bytes: usize,
        compute: impl FnOnce() -> Result<Bitmask, CompileError>,
    ) -> Result<Arc<Bitmask>, CompileError> {
        let mut compute = Some(compute);
        let value = match self.rows[state].get() {
            Some(v) => v,
            None => self.rows[state].get_or_init(|| {
                if !self.try_reserve(entry_bytes) {
                    return SlotValue::Uncached;
                }
                let result = compute.take().expect("reserved path runs once")().map(Arc::new);
                if result.is_err() {
                    self.retained_bytes.fetch_sub(entry_bytes, Ordering::AcqRel);
                }
                SlotValue::Cached(result.map_err(Arc::new))
            }),
        };
        match value {
            SlotValue::Cached(Ok(mask)) => Ok(mask.clone()),
            SlotValue::Cached(Err(e)) => Err((**e).clone()),
            SlotValue::Uncached => {
                compute.take().expect("untouched on the uncached path")().map(Arc::new)
            }
        }
    }

    /// Conservative upper bound on what this structure could ever retain: the slot array itself
    /// plus its configured mask-byte budget.
    fn max_charge_bytes(&self) -> usize {
        self.rows
            .len()
            .saturating_mul(std::mem::size_of::<MaskSlot>())
            .saturating_add(self.budget_bytes.load(Ordering::Acquire))
    }

    #[cfg(test)]
    fn filled_count(&self) -> usize {
        self.rows
            .iter()
            .filter(|r| matches!(r.get(), Some(SlotValue::Cached(_))))
            .count()
    }

    #[cfg(test)]
    fn retained_bytes(&self) -> usize {
        self.retained_bytes.load(Ordering::Acquire)
    }

    #[cfg(test)]
    fn set_budget_for_test(&self, budget: usize) {
        self.budget_bytes.store(budget, Ordering::Release);
    }
}

/// Mask storage backing `cached_mask`. Packed mode needs none: its table is already fully
/// materialized, so allocating per-state slots would be pure waste.
#[derive(Debug)]
enum MaskRows {
    Disabled,
    Lazy(LazyRows),
}

/// A byte engine bound to a vocabulary, shared by [`Arc`] across sequences and the FFI. Only
/// `mask_cache` mutates after construction.
#[derive(Debug)]
pub struct CompiledArtifact {
    engine: Arc<RefEngine>,
    vocab: Arc<CanonicalVocabulary>,
    provenance: Provenance,
    start_is_dead: bool,
    bind_mode: BindMode,
    index: Option<Arc<CompiledIndex>>,
    lazy_trie: Option<Arc<VocabTrie>>,
    packed: Option<PackedMaskTable>,
    mask_cache: MaskRows,
}

impl CompiledArtifact {
    /// Binds `engine` to `vocab` under `provenance`, validating the result. Returns a structured
    /// error (never a partial artifact) if provenance, start state, or token-id bounds are invalid.
    pub fn new(
        engine: RefEngine,
        vocab: Arc<Vocabulary>,
        provenance: Provenance,
    ) -> Result<Self, CompileError> {
        let canonical = Arc::new(CanonicalVocabulary::from_vocabulary(&vocab)?);
        Self::new_with_canonical(engine, canonical, provenance)
    }

    /// The safe public bind (naive mode): takes the canonical vocabulary from ONE `VocabularyHandle`,
    /// so the mask width and the id lookup always agree with each other by construction.
    pub fn new_from_handle(
        handle: &VocabularyHandle,
        engine: RefEngine,
        provenance: Provenance,
    ) -> Result<Self, CompileError> {
        Self::new_with_canonical(engine, handle.canonical().clone(), provenance)
    }

    /// Naive binding from a vocabulary-neutral shared engine.
    pub fn new_from_handle_shared(
        handle: &VocabularyHandle,
        engine: Arc<RefEngine>,
        provenance: Provenance,
    ) -> Result<Self, CompileError> {
        Self::new_full(
            engine,
            handle.canonical().clone(),
            provenance,
            BindMode::Naive,
            None,
            None,
            None,
        )
    }

    /// Binds using an already-built canonical vocabulary instead of rebuilding it (the warm-bind fast
    /// path). Crate-internal: `vocab` is trusted to genuinely be canonical, so only a caller that
    pub(crate) fn new_with_canonical(
        engine: RefEngine,
        vocab: Arc<CanonicalVocabulary>,
        provenance: Provenance,
    ) -> Result<Self, CompileError> {
        Self::new_full(
            Arc::new(engine),
            vocab,
            provenance,
            BindMode::Naive,
            None,
            None,
            None,
        )
    }

    /// Builds the artifact with its FINAL bind mode decided up front - no field is mutated after.
    /// Packed gets no `LazyRows` (its table is already complete); every other mode gets one.
    fn new_full(
        engine: Arc<RefEngine>,
        vocab: Arc<CanonicalVocabulary>,
        provenance: Provenance,
        bind_mode: BindMode,
        index: Option<Arc<CompiledIndex>>,
        lazy_trie: Option<Arc<VocabTrie>>,
        packed: Option<PackedMaskTable>,
    ) -> Result<Self, CompileError> {
        let start_is_dead = engine.is_dead(engine.start());
        let mask_cache = if bind_mode == BindMode::TrieJointBytePacked {
            MaskRows::Disabled
        } else {
            MaskRows::Lazy(LazyRows::try_new(
                engine.state_count(),
                MAX_MASK_CACHE_BYTES,
            )?)
        };
        let artifact = Self {
            engine,
            vocab,
            provenance,
            start_is_dead,
            bind_mode,
            index,
            lazy_trie,
            packed,
            mask_cache,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    /// Binds `engine` to `vocab` from a borrowed `trie`: the eager delta walks
    /// (`TrieJointByte`/`TrieJointClass`) and the packed table (`TrieJointBytePacked`).
    pub fn new_trie(
        engine: RefEngine,
        vocab: Arc<Vocabulary>,
        provenance: Provenance,
        mode: BindMode,
        trie: &VocabTrie,
    ) -> Result<Self, CompileError> {
        let canonical = Arc::new(CanonicalVocabulary::from_vocabulary(&vocab)?);
        let fingerprint = canonical.fingerprint();
        match mode {
            BindMode::TrieJointByte | BindMode::TrieJointClass => {
                Self::new_trie_prevalidated(engine, canonical, provenance, mode, trie, fingerprint)
            }
            BindMode::TrieJointBytePacked => {
                Self::new_packed_from_trie(engine, canonical, provenance, trie, fingerprint)
            }
            BindMode::TrieJointByteLazy | BindMode::Naive => Err(CompileError::new(
                ErrorCode::ArtifactOutOfBounds,
                Stage::L4Bind,
                "this bind mode is not constructible from a borrowed trie; use new_from_bound_trie \
                 (byte_trie_lazy) or new/new_from_handle (naive)",
            )),
        }
    }

    /// Same check as `new_trie`, but skips rebuilding `vocab`. Not public: a caller-suppliable
    /// fingerprint here is never re-verified against `vocab`'s real content, so only
    pub(crate) fn new_trie_prevalidated(
        engine: RefEngine,
        vocab: Arc<CanonicalVocabulary>,
        provenance: Provenance,
        mode: BindMode,
        trie: &VocabTrie,
        vocab_fingerprint: VocabFingerprint,
    ) -> Result<Self, CompileError> {
        Self::new_trie_prevalidated_shared(
            Arc::new(engine),
            vocab,
            provenance,
            mode,
            trie,
            vocab_fingerprint,
        )
    }

    fn new_trie_prevalidated_shared(
        engine: Arc<RefEngine>,
        vocab: Arc<CanonicalVocabulary>,
        provenance: Provenance,
        mode: BindMode,
        trie: &VocabTrie,
        vocab_fingerprint: VocabFingerprint,
    ) -> Result<Self, CompileError> {
        if trie.vocab_fingerprint() != vocab_fingerprint {
            return Err(CompileError::new(
                ErrorCode::ArtifactMismatch,
                Stage::L4Bind,
                "trie was built from a different vocabulary than the one being bound",
            ));
        }
        let index = build_delta(&engine, trie, mode, provenance)?;
        Self::new_full(
            engine,
            vocab,
            provenance,
            mode,
            Some(Arc::new(index)),
            None,
            None,
        )
    }

    /// Binds `engine` to `handle`'s vocabulary using `bound`, a trie the trie cache already proved
    /// paired with SOME `VocabularyHandle`. Rejects `bound` if it was not proven paired with THIS
    pub fn new_from_bound_trie(
        handle: &VocabularyHandle,
        engine: RefEngine,
        provenance: Provenance,
        mode: BindMode,
        bound: &BoundByteTrie,
    ) -> Result<Self, CompileError> {
        Self::new_from_bound_trie_shared(handle, Arc::new(engine), provenance, mode, bound)
    }

    /// Binds a vocabulary-neutral shared engine without cloning its states or transition tables.
    pub fn new_from_bound_trie_shared(
        handle: &VocabularyHandle,
        engine: Arc<RefEngine>,
        provenance: Provenance,
        mode: BindMode,
        bound: &BoundByteTrie,
    ) -> Result<Self, CompileError> {
        if bound.fingerprint() != handle.fingerprint() {
            return Err(CompileError::new(
                ErrorCode::ArtifactMismatch,
                Stage::L4Bind,
                "bound trie does not match this vocabulary handle",
            ));
        }
        if mode == BindMode::TrieJointByteLazy {
            return Self::new_lazy_trie_shared(
                engine,
                handle.canonical().clone(),
                provenance,
                bound.trie_arc(),
                handle.fingerprint(),
            );
        }
        if mode == BindMode::TrieJointBytePacked {
            return Self::new_packed_from_trie_shared(
                engine,
                handle.canonical().clone(),
                provenance,
                bound.trie(),
                handle.fingerprint(),
            );
        }
        Self::new_trie_prevalidated_shared(
            engine,
            handle.canonical().clone(),
            provenance,
            mode,
            bound.trie(),
            handle.fingerprint(),
        )
    }

    /// Recommends `requested` as-is unless it is `TrieJointBytePacked` and either the packed byte
    /// budget (a safety limit) or the packed compile-latency budget (a policy target, see
    #[must_use]
    pub fn recommended_bind_mode(
        state_count: usize,
        mask_vocab_size: usize,
        bound: &BoundByteTrie,
        requested: BindMode,
    ) -> BindMode {
        if requested != BindMode::TrieJointBytePacked {
            return requested;
        }
        let wpr = mask_vocab_size.div_ceil(32);
        let over_budget = !matches!(
            Self::packed_build_peak_bytes(state_count, wpr, bound.trie()),
            Some(peak) if peak <= MAX_PACKED_MASK_BYTES
        );
        let over_latency = state_count
            .checked_mul(wpr)
            .and_then(|words| words.checked_mul(PACKED_ROW_FILL_NS_PER_WORD))
            .is_none_or(|ns| ns / 1000 > PACKED_COMPILE_LATENCY_BUDGET_US);
        if over_budget || over_latency {
            BindMode::TrieJointByteLazy
        } else {
            requested
        }
    }

    /// Binds `engine` to `vocab` in packed mode. Packed needs NO `Delta`: the packed payload is
    /// `state_count * words_per_row * 4`, exact and known before any walk, so each state's row is
    fn new_packed_from_trie(
        engine: RefEngine,
        vocab: Arc<CanonicalVocabulary>,
        provenance: Provenance,
        trie: &VocabTrie,
        fingerprint: VocabFingerprint,
    ) -> Result<Self, CompileError> {
        Self::new_packed_from_trie_shared(Arc::new(engine), vocab, provenance, trie, fingerprint)
    }

    fn new_packed_from_trie_shared(
        engine: Arc<RefEngine>,
        vocab: Arc<CanonicalVocabulary>,
        provenance: Provenance,
        trie: &VocabTrie,
        fingerprint: VocabFingerprint,
    ) -> Result<Self, CompileError> {
        Self::new_packed_from_trie_budgeted_shared(
            engine,
            vocab,
            provenance,
            trie,
            fingerprint,
            MAX_PACKED_MASK_BYTES,
        )
    }

    /// `new_packed_from_trie` with an explicit `budget` (production passes `MAX_PACKED_MASK_BYTES`;
    /// tests pass a small budget). Admission is EXACT: the only heap allocation the packed build makes
    #[cfg(test)]
    fn new_packed_from_trie_budgeted(
        engine: RefEngine,
        vocab: Arc<CanonicalVocabulary>,
        provenance: Provenance,
        trie: &VocabTrie,
        fingerprint: VocabFingerprint,
        budget: usize,
    ) -> Result<Self, CompileError> {
        Self::new_packed_from_trie_budgeted_shared(
            Arc::new(engine),
            vocab,
            provenance,
            trie,
            fingerprint,
            budget,
        )
    }

    fn new_packed_from_trie_budgeted_shared(
        engine: Arc<RefEngine>,
        vocab: Arc<CanonicalVocabulary>,
        provenance: Provenance,
        trie: &VocabTrie,
        fingerprint: VocabFingerprint,
        budget: usize,
    ) -> Result<Self, CompileError> {
        if trie.vocab_fingerprint() != fingerprint {
            return Err(CompileError::new(
                ErrorCode::ArtifactMismatch,
                Stage::L4Bind,
                "trie was built from a different vocabulary than the one being bound",
            ));
        }
        let n = engine.state_count();
        let wpr = vocab.mask_width_words();
        let payload = Self::packed_payload_bytes(n, wpr).ok_or_else(Self::packed_overflow)?;
        let peak = Self::packed_build_peak_bytes(n, wpr, trie).ok_or_else(Self::packed_overflow)?;
        if peak > budget {
            return Err(CompileError::new(
                ErrorCode::InternalLimitExceeded,
                Stage::L4Bind,
                "packed mask table plus walk scratch exceeds the memory budget; select byte_trie_lazy",
            ));
        }
        let _permit = crate::mem_gate::acquire_vocab_build_bytes(peak)?;
        let stack_fuel = trie.max_dfs_frontier();
        let word_count = payload / 4;
        let mut rows: Vec<u32> = Vec::new();
        rows.try_reserve_exact(word_count)
            .map_err(|_| Self::packed_alloc_failed())?;
        rows.resize(word_count, 0);
        crate::index::fill_packed_rows(
            trie,
            &engine,
            &mut rows,
            wpr,
            vocab.mask_vocab_size(),
            stack_fuel,
        )?;
        let packed = PackedMaskTable {
            words_per_row: wpr,
            rows: rows.into_boxed_slice(),
        };
        Self::new_full(
            engine,
            vocab,
            provenance,
            BindMode::TrieJointBytePacked,
            None,
            None,
            Some(packed),
        )
    }

    /// Retained packed payload bytes for `state_count` states of `wpr` words, checked (`None` on
    /// overflow). The exact admission size (the packed table is the packed build's only allocation).
    #[must_use]
    fn packed_payload_bytes(state_count: usize, wpr: usize) -> Option<usize> {
        state_count.checked_mul(wpr)?.checked_mul(4)
    }

    /// The exact incremental packed-build peak: payload table, fixed header allowance, dense walk
    /// scratch, and the DFS stack's precomputed exact peak - the same total `new_packed_from_trie_
    fn packed_build_peak_bytes(state_count: usize, wpr: usize, trie: &VocabTrie) -> Option<usize> {
        let payload = Self::packed_payload_bytes(state_count, wpr)?;
        let scratch_fixed = crate::index::packed_scratch_fixed_bytes()?;
        let stack_bytes = crate::index::packed_stack_bytes(trie.max_dfs_frontier())?;
        payload
            .checked_add(PACKED_HEADER_ALLOWANCE)?
            .checked_add(scratch_fixed)?
            .checked_add(stack_bytes)
    }

    fn packed_overflow() -> CompileError {
        CompileError::new(
            ErrorCode::InternalLimitExceeded,
            Stage::L4Bind,
            "packed mask matrix size overflows",
        )
    }

    fn packed_alloc_failed() -> CompileError {
        CompileError::new(
            ErrorCode::InternalLimitExceeded,
            Stage::L4Bind,
            "packed mask matrix allocation failed",
        )
    }

    /// The borrowed packed row for `s` in packed mode (no allocation), else `None`. The native
    /// zero-copy serving view; a Python view must hold an `Arc` to keep the artifact alive.
    #[must_use]
    pub fn mask_row_words(&self, s: StateId) -> Option<&[u32]> {
        self.packed.as_ref()?.row(s)
    }

    /// The WHOLE packed mask matrix, borrowed flat as `state_count() * words_per_row()` little-endian
    /// `u32` words (row `s` at `[s*words_per_row .. (s+1)*words_per_row]`), or `None` outside packed
    #[must_use]
    pub fn packed_mask_table_words(&self) -> Option<&[u32]> {
        self.packed.as_ref().map(|p| &p.rows[..])
    }

    /// Retained bytes of the packed mask matrix (0 when not in packed mode), for memory reporting.
    #[must_use]
    pub fn packed_retained_bytes(&self) -> usize {
        self.packed
            .as_ref()
            .map_or(0, PackedMaskTable::retained_bytes)
    }

    /// Retained bytes of the eager delta index (0 outside `TrieJointByte`/`TrieJointClass` modes),
    /// for the compiled-artifact cache's byte-budget accounting.
    #[must_use]
    pub fn index_retained_bytes(&self) -> usize {
        self.index.as_ref().map_or(0, |i| i.heap_bytes())
    }

    /// Retained bytes of the byte engine itself (states, transitions, class table) - present in
    /// every bind mode, so the compiled-artifact cache charges it alongside the packed/index bytes
    #[must_use]
    pub fn engine_retained_bytes(&self) -> usize {
        self.engine.heap_bytes()
    }

    /// Conservative upper bound on total retained bytes, valid for ANY bind mode and stable for
    /// the artifact's whole lifetime - safe to charge once at cache-admission time.
    #[must_use]
    pub fn artifact_cache_charge_upper_bound(&self) -> usize {
        let lazy_bytes = match &self.mask_cache {
            MaskRows::Disabled => 0,
            MaskRows::Lazy(rows) => rows.max_charge_bytes(),
        };
        let trie_bytes = self.lazy_trie.as_ref().map_or(0, |t| t.heap_bytes());
        self.engine_retained_bytes()
            .saturating_add(self.packed_retained_bytes())
            .saturating_add(self.index_retained_bytes())
            .saturating_add(lazy_bytes)
            .saturating_add(trie_bytes)
    }

    /// Binds `engine` to `vocab` in lazy byte-trie mode: no eager `Delta`; each state's row is
    /// walked on first query and cached. Rejects a trie built from a different vocabulary.
    #[cfg(test)]
    pub(crate) fn new_lazy_trie(
        engine: RefEngine,
        vocab: Arc<CanonicalVocabulary>,
        provenance: Provenance,
        trie: Arc<VocabTrie>,
        vocab_fingerprint: VocabFingerprint,
    ) -> Result<Self, CompileError> {
        Self::new_lazy_trie_shared(Arc::new(engine), vocab, provenance, trie, vocab_fingerprint)
    }

    fn new_lazy_trie_shared(
        engine: Arc<RefEngine>,
        vocab: Arc<CanonicalVocabulary>,
        provenance: Provenance,
        trie: Arc<VocabTrie>,
        vocab_fingerprint: VocabFingerprint,
    ) -> Result<Self, CompileError> {
        if trie.vocab_fingerprint() != vocab_fingerprint {
            return Err(CompileError::new(
                ErrorCode::ArtifactMismatch,
                Stage::L4Bind,
                "trie was built from a different vocabulary than the one being bound",
            ));
        }
        Self::new_full(
            engine,
            vocab,
            provenance,
            BindMode::TrieJointByteLazy,
            None,
            Some(trie),
            None,
        )
    }

    /// Validates the bindings this layer owns: provenance, the start state in range, every bound
    /// token id inside the mask, and (unless the language is empty) at least one accepting state.
    pub fn validate(&self) -> Result<(), CompileError> {
        self.provenance.validate()?;
        let out_of_bounds = |msg| {
            Err(CompileError::new(
                ErrorCode::ArtifactOutOfBounds,
                Stage::ArtifactValidate,
                msg,
            ))
        };
        if (self.engine.start().get() as usize) >= self.engine.state_count() {
            return out_of_bounds("artifact start state is out of range");
        }
        if !self.start_is_dead
            && !(0..self.engine.state_count())
                .filter_map(|raw| StateId::try_from(raw).ok())
                .any(|s| self.engine.is_accepting(s))
        {
            return Err(CompileError::new(
                ErrorCode::Malformed,
                Stage::ArtifactValidate,
                "a non-empty-language artifact has no accepting state",
            ));
        }
        Ok(())
    }

    /// The anchored start state.
    #[must_use]
    pub fn start(&self) -> StateId {
        self.engine.start()
    }

    /// The canonical DEAD state.
    #[must_use]
    pub fn dead(&self) -> StateId {
        self.engine.dead()
    }

    /// Whether the start state is DEAD (the language accepts nothing). Diagnostic only.
    #[must_use]
    pub fn start_is_dead(&self) -> bool {
        self.start_is_dead
    }

    /// The artifact's provenance record.
    #[must_use]
    pub fn provenance(&self) -> Provenance {
        self.provenance
    }

    /// The token-id space the mask covers (`max_token_id + 1`; EOS occupies a bit, never set).
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.mask_vocab_size()
    }

    /// The number of reference states, including DEAD. State ids span `0..state_count`.
    #[must_use]
    pub fn state_count(&self) -> usize {
        self.engine.state_count()
    }

    /// Walks `bytes` from `from`; `None` once DEAD (or an out-of-range state) is reached.
    #[must_use]
    pub fn walk(&self, from: StateId, bytes: &[u8]) -> Option<StateId> {
        self.engine.consume_token(from, bytes)
    }

    /// The number of packed `u32` words in one mask row: `ceil(vocab_size / 32)`.
    #[must_use]
    pub fn words_per_row(&self) -> usize {
        self.vocab.mask_width_words()
    }

    /// Stopping at `s` yields a complete, valid instance.
    #[must_use]
    pub fn is_accepting(&self, s: StateId) -> bool {
        self.engine.is_accepting(s)
    }

    /// Some byte leads out of `s` toward an accepting state.
    #[must_use]
    pub fn can_continue(&self, s: StateId) -> bool {
        self.engine.can_continue(s)
    }

    /// `s` is the canonical dead sink.
    #[must_use]
    pub fn is_dead(&self, s: StateId) -> bool {
        self.engine.is_dead(s)
    }

    /// EOS is legal exactly where `s` accepts. EOS is a side predicate, never a byte or a mask bit.
    #[must_use]
    pub fn eos_legal(&self, s: StateId) -> bool {
        self.engine.eos_legal(s)
    }

    /// The bytes of `token`, or `None` if the id is not in the vocabulary.
    #[must_use]
    pub fn token_bytes(&self, token: TokenId) -> Option<&[u8]> {
        self.vocab.token_bytes(token.get())
    }

    /// The ordinary token ids bound to this artifact (EOS excluded). Order is unspecified.
    #[must_use]
    pub fn token_ids(&self) -> Vec<TokenId> {
        self.vocab
            .iter()
            .flat_map(|(_, ids)| ids.iter().copied())
            .map(TokenId)
            .collect()
    }

    /// How this artifact produces masks: the naive walk or a prebuilt trie index.
    #[must_use]
    pub fn bind_mode(&self) -> BindMode {
        self.bind_mode
    }

    /// The allowed ordinary-token mask at `s` (EOS excluded). Routes through the selected bind mode;
    /// every route is byte-identical to `Naive`.
    pub fn allowed_mask(&self, s: StateId) -> Result<Bitmask, CompileError> {
        match self.bind_mode {
            BindMode::Naive => self.engine.allowed_tokens_from_records(
                s,
                self.vocab.mask_vocab_size(),
                self.vocab.iter(),
            ),
            BindMode::TrieJointByte | BindMode::TrieJointClass => self.mask_from_index(s),
            BindMode::TrieJointByteLazy => self.mask_from_lazy_walk(s),
            BindMode::TrieJointBytePacked => Ok(self.packed_mask(s)),
        }
    }

    /// The compatibility copying view of `s`'s packed mask (an out-of-range state is empty). Native
    /// callers should use `mask_row_words`/the direct-buffer path, which borrow the row without a copy.
    fn packed_mask(&self, s: StateId) -> Bitmask {
        match self.packed.as_ref().and_then(|p| p.row(s)) {
            Some(words) => Bitmask::from_words(words, self.vocab.mask_vocab_size()),
            None => Bitmask::zeros(self.vocab.mask_vocab_size()),
        }
    }

    /// Walks `s`'s row on demand from the shared byte-trie and packs it into a mask. An out-of-range
    /// (forged) state is empty, exactly as the eager index and naive walk report it.
    fn mask_from_lazy_walk(&self, s: StateId) -> Result<Bitmask, CompileError> {
        let mut mask = Bitmask::zeros(self.vocab.mask_vocab_size());
        if (s.get() as usize) >= self.engine.state_count() {
            return Ok(mask);
        }
        let trie = self.lazy_trie.as_ref().ok_or_else(|| {
            CompileError::new(
                ErrorCode::ArtifactOutOfBounds,
                Stage::L4Bind,
                "the lazy bind mode requires a byte-trie",
            )
        })?;
        crate::index::walk_state_mask_into(trie, &self.engine, s, &mut mask)?;
        Ok(mask)
    }

    /// Reads the allowed-token mask for `s` from the trie index (an out-of-range state is empty).
    fn mask_from_index(&self, s: StateId) -> Result<Bitmask, CompileError> {
        let index = self.index.as_ref().ok_or_else(|| {
            CompileError::new(
                ErrorCode::ArtifactOutOfBounds,
                Stage::L4Bind,
                "a trie bind mode requires a compiled index",
            )
        })?;
        let mut mask = Bitmask::zeros(self.vocab.mask_vocab_size());
        if let Some(row) = index.row(s) {
            for (token, _landing) in row.iter() {
                mask.set(token)?;
            }
        }
        Ok(mask)
    }

    /// The ordinary token ids allowed at `s` (EOS excluded), read from the per-state mask cache so
    /// repeated queries on one state never re-walk the vocabulary. Ids are ascending.
    pub fn allowed_ids(&self, s: StateId) -> Result<Vec<TokenId>, CompileError> {
        let mask = self.cached_mask(s)?;
        let mut out = Vec::with_capacity(mask.count_ones());
        for (word_index, &word) in mask.as_words().iter().enumerate() {
            let base = u32::try_from(word_index)
                .expect("mask word count stays far below u32::MAX (bounded by MAX_MASK_TOKENS)")
                * 32;
            let mut bits = word;
            while bits != 0 {
                out.push(TokenId(base + bits.trailing_zeros()));
                bits &= bits - 1;
            }
        }
        Ok(out)
    }

    /// `allowed_mask(s)`, cached per state id. Out-of-range ids (DEAD-equivalent) never touch the
    /// cache - only `0..state_count` has a slot.
    fn cached_mask(&self, s: StateId) -> Result<Arc<Bitmask>, CompileError> {
        if self.packed.is_some() {
            return Ok(Arc::new(self.packed_mask(s)));
        }
        if (s.get() as usize) >= self.engine.state_count() {
            return Ok(Arc::new(self.allowed_mask(s)?));
        }
        let MaskRows::Lazy(rows) = &self.mask_cache else {
            unreachable!("non-packed artifact without a lazy mask cache");
        };
        let entry_bytes = self.words_per_row() * 4 + MASK_CACHE_ENTRY_OVERHEAD;
        rows.get_or_compute(s.get() as usize, entry_bytes, || self.allowed_mask(s))
    }

    #[cfg(test)]
    fn set_mask_cache_budget(&self, budget: usize) {
        let MaskRows::Lazy(rows) = &self.mask_cache else {
            panic!("set_mask_cache_budget called on a packed artifact");
        };
        rows.set_budget_for_test(budget);
    }

    #[cfg(test)]
    fn mask_cache_filled_count(&self) -> usize {
        match &self.mask_cache {
            MaskRows::Lazy(rows) => rows.filled_count(),
            MaskRows::Disabled => 0,
        }
    }

    #[cfg(test)]
    fn mask_cache_retained_bytes(&self) -> usize {
        match &self.mask_cache {
            MaskRows::Lazy(rows) => rows.retained_bytes(),
            MaskRows::Disabled => 0,
        }
    }

    #[cfg(test)]
    fn mask_cache_is_disabled(&self) -> bool {
        matches!(self.mask_cache, MaskRows::Disabled)
    }

    /// The exact packed `u32` length a mask buffer needs, all arithmetic checked (a length mismatch,
    /// a boundary row index, or an overflowing product is a structured error). Callers size from here.
    pub fn mask_buffer_len(
        &self,
        states_len: usize,
        rows: Option<&[usize]>,
    ) -> Result<usize, CompileError> {
        let rows_count = match rows {
            Some(r) => {
                if r.len() != states_len {
                    return Err(self.mask_err("rows length does not match states length"));
                }
                match r.iter().copied().max() {
                    Some(m) => m
                        .checked_add(1)
                        .ok_or_else(|| self.mask_err("row index overflows"))?,
                    None => 0,
                }
            }
            None => states_len,
        };
        let words = rows_count
            .checked_mul(self.words_per_row())
            .ok_or_else(|| self.mask_err("mask buffer length overflows"))?;
        if words > MAX_MASK_WORDS {
            return Err(self.mask_err("mask buffer length exceeds the resource cap"));
        }
        Ok(words)
    }

    /// Fills caller-owned `out` with the packed masks for `states` (`rows` selects each target row,
    /// identity when `None`). `out.len()` must equal [`Self::mask_buffer_len`]; unused bits stay zero.
    pub fn write_mask_into(
        &self,
        states: &[StateId],
        out: &mut [u32],
        rows: Option<&[usize]>,
    ) -> Result<(), CompileError> {
        let wpr = self.words_per_row();
        let expected = self.mask_buffer_len(states.len(), rows)?;
        if out.len() != expected {
            return Err(self.mask_err("mask buffer length does not match rows * words_per_row"));
        }
        let overwrite = rows.is_none();
        if !overwrite {
            out.iter_mut().for_each(|w| *w = 0);
        }
        for (i, &s) in states.iter().enumerate() {
            let row = rows.map_or(i, |r| r[i]);
            let base = row * wpr;
            if let Some(words) = self.mask_row_words(s) {
                write_words_u32(&mut out[base..base + wpr], words, overwrite);
            } else {
                let mask = self.cached_mask(s)?;
                write_words_u32(&mut out[base..base + wpr], mask.as_words(), overwrite);
            }
        }
        Ok(())
    }

    /// Like `write_mask_into`, but writes little-endian bytes straight from the cached mask into
    /// `out` - no `Vec<u32>` intermediate. `out.len()` must equal `4 * mask_buffer_len(...)`.
    pub fn write_mask_le_bytes_into(
        &self,
        states: &[StateId],
        out: &mut [u8],
        rows: Option<&[usize]>,
    ) -> Result<(), CompileError> {
        let wpr = self.words_per_row();
        let expected_words = self.mask_buffer_len(states.len(), rows)?;
        let expected_bytes = expected_words
            .checked_mul(4)
            .ok_or_else(|| self.mask_err("mask buffer length overflows"))?;
        if out.len() != expected_bytes {
            return Err(self.mask_err("mask buffer length does not match 4 * rows * words_per_row"));
        }
        let overwrite = rows.is_none();
        if !overwrite {
            out.iter_mut().for_each(|b| *b = 0);
        }
        for (i, &s) in states.iter().enumerate() {
            let row = rows.map_or(i, |r| r[i]);
            let base = row * wpr * 4;
            if let Some(words) = self.mask_row_words(s) {
                write_words_le(&mut out[base..base + wpr * 4], words, overwrite);
            } else {
                let mask = self.cached_mask(s)?;
                write_words_le(&mut out[base..base + wpr * 4], mask.as_words(), overwrite);
            }
        }
        Ok(())
    }

    fn mask_err(&self, msg: &'static str) -> CompileError {
        CompileError::new(ErrorCode::ArtifactOutOfBounds, Stage::ArtifactValidate, msg)
    }
}

/// Writes `words` into a `u32` row: overwrite (bulk copy) for a one-to-one row, else OR-accumulate.
fn write_words_u32(row: &mut [u32], words: &[u32], overwrite: bool) {
    if overwrite {
        row.copy_from_slice(words);
    } else {
        for (dst, &word) in row.iter_mut().zip(words) {
            *dst |= word;
        }
    }
}

/// Writes `words` as little-endian bytes into a byte row: overwrite for a one-to-one row (no
/// read-back), else OR-accumulate. `row.len()` must be `4 * words.len()`.
fn write_words_le(row: &mut [u8], words: &[u32], overwrite: bool) {
    for (i, &word) in words.iter().enumerate() {
        let at = i * 4;
        let value = if overwrite {
            word
        } else {
            u32::from_le_bytes([row[at], row[at + 1], row[at + 2], row[at + 3]]) | word
        };
        row[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
}

/// Validates a vocabulary binds without ambiguity (the same id -> bytes indexing an artifact runs),
/// so a handle checked here reuses across compilations without re-discovering a conflict.
pub fn validate_token_bytes(vocab: &Vocabulary) -> Result<(), CompileError> {
    CanonicalVocabulary::from_vocabulary(vocab)?;
    Ok(())
}

/// A position inside a [`CompiledArtifact`]. Owns the position, shares the artifact by [`Arc`].
#[derive(Clone, Debug)]
pub struct Matcher {
    state: StateId,
    artifact: Arc<CompiledArtifact>,
    bytes_seen: u64,
}

impl Matcher {
    /// Starts a matcher at the artifact's anchored start state.
    #[must_use]
    pub fn new(artifact: Arc<CompiledArtifact>) -> Self {
        let state = artifact.start();
        Self {
            state,
            artifact,
            bytes_seen: 0,
        }
    }

    /// The current position.
    #[must_use]
    pub fn state(&self) -> StateId {
        self.state
    }

    /// The number of bytes consumed so far.
    #[must_use]
    pub fn bytes_seen(&self) -> u64 {
        self.bytes_seen
    }

    /// The shared artifact.
    #[must_use]
    pub fn artifact(&self) -> &Arc<CompiledArtifact> {
        &self.artifact
    }

    /// Consumes `token`, advancing the position. Fails with `IllegalToken` if the id is unknown or
    /// no path exists (a DEAD matcher rejects every token forever; the position never moves on
    pub fn advance(&mut self, token: TokenId) -> Result<(), MatcherError> {
        let bytes = self
            .artifact
            .token_bytes(token)
            .ok_or(MatcherError::IllegalToken { token })?;
        match self.artifact.engine.consume_token(self.state, bytes) {
            Some(next) => {
                self.state = next;
                self.bytes_seen += bytes.len() as u64;
                Ok(())
            }
            None => Err(MatcherError::IllegalToken { token }),
        }
    }

    /// Stopping now yields a complete, valid instance.
    #[must_use]
    pub fn is_accepting(&self) -> bool {
        self.artifact.is_accepting(self.state)
    }

    /// Some token can still extend this position.
    #[must_use]
    pub fn can_continue(&self) -> bool {
        self.artifact.can_continue(self.state)
    }

    /// The matcher has reached the permanent dead sink.
    #[must_use]
    pub fn is_dead(&self) -> bool {
        self.artifact.is_dead(self.state)
    }

    /// EOS is legal here (equivalently, the position accepts).
    #[must_use]
    pub fn eos_legal(&self) -> bool {
        self.artifact.eos_legal(self.state)
    }

    /// The allowed ordinary-token mask at the current position.
    pub fn allowed_mask(&self) -> Result<Bitmask, CompileError> {
        self.artifact.allowed_mask(self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::build_from_regex;
    use crate::correctness::corpus::corpus_cases;
    use crate::error::HashState;
    use crate::vocab::build_vocabulary;
    use rustc_hash::FxHashMap as Map;

    #[cfg(feature = "huggingface-hub")]
    #[test]
    #[ignore = "timing; run with --release --features huggingface-hub -- --ignored --nocapture"]
    fn packed_bind_stage_isolated_gate_ab_on_real_tokenizers() {
        use std::time::Instant;

        fn stats(mut xs: Vec<f64>) -> (f64, f64, f64, f64, f64, f64) {
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = xs.len();
            let median = xs[n / 2];
            let p95 = xs[((n as f64 * 0.95) as usize).min(n - 1)];
            let p99 = xs[((n as f64 * 0.99) as usize).min(n - 1)];
            let mean = xs.iter().sum::<f64>() / n as f64;
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            (median, p95, p99, var.sqrt(), xs[0], xs[n - 1])
        }
        fn report(label: &str, xs: Vec<f64>) {
            let (median, p95, p99, stddev, min, max) = stats(xs.clone());
            println!(
                "    {label}: n={} median={median:.2} p95={p95:.2} p99={p99:.2} stddev={stddev:.2} min={min:.2} max={max:.2} us",
                xs.len()
            );
        }

        const N: usize = 200;
        for repo in [
            "openai-community/gpt2",
            "Qwen/Qwen2.5-0.5B",
            "bigscience/bloom",
        ] {
            let vocab = Vocabulary::from_pretrained(repo, None).expect("vocab");
            let canonical =
                Arc::new(CanonicalVocabulary::from_vocabulary(&vocab).expect("canonical"));
            let trie = VocabTrie::build_byte(&vocab).expect("trie");
            let fresh_engine = || {
                corpus_cases()
                    .expect("corpus")
                    .into_iter()
                    .find(|c| c.name == "syn-object-nested")
                    .expect("case")
                    .engine
            };
            println!("=== {repo} ===");

            let mut buckets: [[Vec<f64>; 5]; 2] = Default::default();
            let mut checksums: [Option<Vec<u32>>; 2] = [None, None];
            let order: Vec<bool> = (0..N / 2)
                .flat_map(|i| {
                    if i % 2 == 0 {
                        [true, false]
                    } else {
                        [false, true]
                    }
                })
                .collect();
            const WARMUP: usize = 20;
            for &use_gate in std::iter::repeat_n(&true, WARMUP).chain(order.iter()) {
                let gi = usize::from(!use_gate);
                let engine = fresh_engine();
                let n = engine.state_count();
                let wpr = canonical.mask_width_words();
                let payload = n.checked_mul(wpr).unwrap().checked_mul(4).unwrap();
                let scratch_fixed = crate::index::packed_scratch_fixed_bytes().unwrap();
                let stack_bytes =
                    crate::index::packed_stack_bytes(trie.max_dfs_frontier()).unwrap();
                let peak = payload + PACKED_HEADER_ALLOWANCE + scratch_fixed + stack_bytes;

                let t_total = Instant::now();
                let t = Instant::now();
                let _permit = if use_gate {
                    Some(crate::mem_gate::acquire_vocab_build_bytes(peak).expect("permit"))
                } else {
                    None
                };
                let gate_dt = t.elapsed().as_secs_f64() * 1e6;

                let t = Instant::now();
                let stack_fuel = trie.max_dfs_frontier();
                let word_count = payload / 4;
                let mut rows: Vec<u32> = Vec::new();
                rows.try_reserve_exact(word_count).expect("reserve");
                rows.resize(word_count, 0);
                let alloc_dt = t.elapsed().as_secs_f64() * 1e6;

                let t = Instant::now();
                crate::index::fill_packed_rows(
                    &trie,
                    &engine,
                    &mut rows,
                    wpr,
                    canonical.mask_vocab_size(),
                    stack_fuel,
                )
                .expect("fill");
                let fill_dt = t.elapsed().as_secs_f64() * 1e6;

                let t = Instant::now();
                let mut artifact = CompiledArtifact::new_with_canonical(
                    engine,
                    canonical.clone(),
                    Provenance::reference(HashState::Hash([7; 32])),
                )
                .expect("artifact");
                artifact.bind_mode = BindMode::TrieJointBytePacked;
                artifact.packed = Some(PackedMaskTable {
                    words_per_row: wpr,
                    rows: rows.into_boxed_slice(),
                });
                let construct_dt = t.elapsed().as_secs_f64() * 1e6;
                let total_dt = t_total.elapsed().as_secs_f64() * 1e6;

                if checksums[gi].is_none() {
                    checksums[gi] = Some(artifact.packed_mask_table_words().unwrap().to_vec());
                }
                std::hint::black_box(&artifact);

                buckets[gi][0].push(gate_dt);
                buckets[gi][1].push(alloc_dt);
                buckets[gi][2].push(fill_dt);
                buckets[gi][3].push(construct_dt);
                buckets[gi][4].push(total_dt);
            }
            for (gi, use_gate) in [(0, true), (1, false)] {
                println!("  use_gate={use_gate}:");
                let [gate_us, alloc_us, fill_us, construct_us, total_us] = buckets[gi].clone();
                report("gate_acquire", gate_us);
                report("payload_alloc", alloc_us);
                report("fill_packed_rows", fill_us);
                report("artifact_construct", construct_us);
                report("total", total_us);
            }
            assert_eq!(
                checksums[0], checksums[1],
                "gate on/off must produce byte-identical packed tables ({repo})"
            );
        }
    }

    #[cfg(feature = "huggingface-hub")]
    #[test]
    #[ignore = "timing; run with --release --features huggingface-hub -- --ignored --nocapture"]
    fn warm_bind_stage_breakdown_on_real_tokenizers() {
        use std::time::Instant;
        let median = |mut xs: Vec<u128>| {
            xs.sort_unstable();
            xs[xs.len() / 2]
        };
        let fresh_engine = || {
            corpus_cases()
                .expect("corpus")
                .into_iter()
                .find(|c| c.name == "syn-object-nested")
                .expect("case")
                .engine
        };
        for repo in ["openai-community/gpt2", "Qwen/Qwen2.5-0.5B"] {
            let vocab = Arc::new(Vocabulary::from_pretrained(repo, None).expect("vocab"));
            let mut meta_us = Vec::new();
            let mut new_us = Vec::new();
            for _ in 0..11 {
                let t = Instant::now();
                let idx = CanonicalVocabulary::from_vocabulary(&vocab).expect("canonical");
                meta_us.push(t.elapsed().as_micros());
                std::hint::black_box(&idx);
                let t = Instant::now();
                let a = CompiledArtifact::new(
                    fresh_engine(),
                    vocab.clone(),
                    Provenance::reference(HashState::Hash([7; 32])),
                )
                .expect("artifact");
                new_us.push(t.elapsed().as_micros());
                std::hint::black_box(&a);
            }
            println!(
                "{repo}: canonical_build_us={} CompiledArtifact_new_us={} tokens={}",
                median(meta_us),
                median(new_us),
                vocab.len()
            );
        }
    }

    /// Packs `vocab` into the four CSR buffers `VocabularyHandle::from_packed` expects.
    #[cfg(feature = "huggingface-hub")]
    fn pack_vocab(vocab: &Vocabulary) -> (Vec<u8>, Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut token_bytes = Vec::new();
        let mut byte_offsets = vec![0u32];
        let mut token_ids = Vec::new();
        let mut id_offsets = vec![0u32];
        for (bytes, ids) in vocab.tokens() {
            token_bytes.extend_from_slice(bytes);
            byte_offsets.push(token_bytes.len() as u32);
            token_ids.extend_from_slice(ids);
            id_offsets.push(token_ids.len() as u32);
        }
        (token_bytes, byte_offsets, token_ids, id_offsets)
    }

    /// Stage-isolated: tuple-vs-packed CONSTRUCTION (sort/flatten vs full handle build), then a
    /// cold BIND breakdown (schema IR, DFA compile, trie build, artifact assembly, first-vs-repeated
    #[cfg(feature = "huggingface-hub")]
    #[test]
    #[ignore = "timing; run with --release --features huggingface-hub -- --ignored --nocapture"]
    fn stage_isolated_construction_and_bind_on_real_tokenizers() {
        use std::time::Instant;

        use crate::index::TrieCache;

        fn stats(mut xs: Vec<f64>) -> (f64, f64, f64) {
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = xs.len();
            (
                xs[n / 2],
                xs[((n as f64 * 0.95) as usize).min(n - 1)],
                xs[n - 1],
            )
        }
        fn report(label: &str, xs: Vec<f64>) {
            let (median, p95, max) = stats(xs.clone());
            println!(
                "    {label}: n={} median={median:.2} p95={p95:.2} max={max:.2} us",
                xs.len()
            );
        }

        const N: usize = 30;
        for repo in [
            "openai-community/gpt2",
            "Qwen/Qwen2.5-0.5B",
            "bigscience/bloom",
        ] {
            let vocab = Vocabulary::from_pretrained(repo, None).expect("vocab");
            let (tb, bo, ti, io) = pack_vocab(&vocab);
            println!("=== {repo} ({} tokens) ===", vocab.len());

            let mut tuple_sort_us = Vec::new();
            let mut tuple_total_us = Vec::new();
            let mut packed_sort_us = Vec::new();
            let mut packed_total_us = Vec::new();
            for _ in 0..N {
                let t = Instant::now();
                let prepared = crate::vocab::PreparedVocabulary::build(&vocab).expect("prepared");
                tuple_sort_us.push(t.elapsed().as_secs_f64() * 1e6);
                std::hint::black_box(&prepared);

                let t = Instant::now();
                let canonical = CanonicalVocabulary::from_vocabulary(&vocab).expect("canonical");
                tuple_total_us.push(t.elapsed().as_secs_f64() * 1e6);
                std::hint::black_box(&canonical);

                let t = Instant::now();
                let prepared = crate::vocab::PreparedVocabulary::build_from_packed(
                    &tb,
                    &bo,
                    &ti,
                    &io,
                    vocab.eos_token_id(),
                )
                .expect("prepared");
                packed_sort_us.push(t.elapsed().as_secs_f64() * 1e6);
                std::hint::black_box(&prepared);

                let t = Instant::now();
                let handle =
                    VocabularyHandle::from_packed(&tb, &bo, &ti, &io, vocab.eos_token_id())
                        .expect("handle");
                packed_total_us.push(t.elapsed().as_secs_f64() * 1e6);
                std::hint::black_box(&handle);
            }
            println!("  construction (sort/flatten only, then the whole handle build):");
            report("tuple_sort_flatten", tuple_sort_us);
            report("tuple_total_handle_build", tuple_total_us);
            report("packed_sort_flatten", packed_sort_us);
            report("packed_total_handle_build", packed_total_us);

            let schema = crate::correctness::corpus::corpus_cases()
                .expect("corpus")
                .into_iter()
                .find(|c| c.name == "syn-object-nested")
                .expect("case")
                .json_schema
                .to_string();
            let handle = VocabularyHandle::new(Arc::new(vocab.clone())).expect("handle");

            let mut ir_us = Vec::new();
            let mut dfa_us = Vec::new();
            let mut trie_build_us = Vec::new();
            let mut artifact_us = Vec::new();
            let mut first_mask_us = Vec::new();
            let mut repeated_mask_us = Vec::new();
            for _ in 0..N {
                let t = Instant::now();
                let ir =
                    crate::schema_to_ir(&schema, crate::CompileOptions::default()).expect("ir");
                ir_us.push(t.elapsed().as_secs_f64() * 1e6);

                let t = Instant::now();
                let engine = crate::compile_ir(&ir).expect("engine");
                dfa_us.push(t.elapsed().as_secs_f64() * 1e6);

                let t = Instant::now();
                let cache = TrieCache::new();
                let bound = cache.bind(&handle).expect("bound");
                trie_build_us.push(t.elapsed().as_secs_f64() * 1e6);

                let t = Instant::now();
                let artifact = CompiledArtifact::new_from_bound_trie(
                    &handle,
                    engine,
                    Provenance::reference(HashState::Hash([7; 32])),
                    BindMode::TrieJointBytePacked,
                    &bound,
                )
                .expect("artifact");
                artifact_us.push(t.elapsed().as_secs_f64() * 1e6);

                let start = artifact.start();
                let wpr = artifact.words_per_row();
                let mut buf = vec![0u8; wpr * 4];
                let t = Instant::now();
                artifact
                    .write_mask_le_bytes_into(&[start], &mut buf, None)
                    .expect("mask");
                first_mask_us.push(t.elapsed().as_secs_f64() * 1e6);

                let t = Instant::now();
                artifact
                    .write_mask_le_bytes_into(&[start], &mut buf, None)
                    .expect("mask");
                repeated_mask_us.push(t.elapsed().as_secs_f64() * 1e6);
            }
            println!("  cold bind (fresh trie every iteration):");
            report("schema_to_ir", ir_us);
            report("compile_ir (DFA)", dfa_us);
            report(
                "trie_build (cold, via fresh TrieCache::bind)",
                trie_build_us,
            );
            report("artifact_assembly (packed table fill)", artifact_us);
            report("first_direct_mask (same API as repeated)", first_mask_us);
            report("repeated_direct_mask", repeated_mask_us);
        }
    }

    fn boolean_artifact() -> CompiledArtifact {
        let engine = corpus_cases()
            .expect("corpus")
            .into_iter()
            .find(|c| c.name == "suite-type-boolean")
            .expect("boolean case")
            .engine;
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        map.insert(b"t".to_vec(), vec![0]);
        map.insert(b"true".to_vec(), vec![1]);
        map.insert(b"z".to_vec(), vec![2, 3]);
        let vocab = build_vocabulary(9, map).expect("vocab");
        CompiledArtifact::new(
            engine,
            Arc::new(vocab),
            Provenance::reference(HashState::Hash([7; 32])),
        )
        .expect("artifact")
    }

    fn boolean_lazy_artifact() -> CompiledArtifact {
        let engine = corpus_cases()
            .expect("corpus")
            .into_iter()
            .find(|c| c.name == "suite-type-boolean")
            .expect("boolean case")
            .engine;
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        map.insert(b"t".to_vec(), vec![0]);
        map.insert(b"true".to_vec(), vec![1]);
        map.insert(b"z".to_vec(), vec![2, 3]);
        let vocab = Arc::new(build_vocabulary(9, map).expect("vocab"));
        let fp = VocabFingerprint::of(&vocab).unwrap();
        let canonical = Arc::new(CanonicalVocabulary::from_vocabulary(&vocab).expect("canonical"));
        let trie = Arc::new(VocabTrie::build_byte(&vocab).expect("trie"));
        CompiledArtifact::new_lazy_trie(
            engine,
            canonical,
            Provenance::reference(HashState::Hash([7; 32])),
            trie,
            fp,
        )
        .expect("lazy artifact")
    }

    #[test]
    fn packed_payload_reservation_is_fallible_not_an_abort() {
        let mut rows: Vec<u32> = Vec::new();
        assert!(rows.try_reserve_exact(usize::MAX / 2).is_err());
    }

    #[test]
    fn packed_admission_charges_table_scratch_and_stack_and_rejects_before_allocating() {
        let vocab = Arc::new({
            let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
            map.insert(b"t".to_vec(), vec![0]);
            build_vocabulary(9, map).expect("vocab")
        });
        let handle = crate::index::VocabularyHandle::new(vocab.clone()).expect("handle");
        let trie = VocabTrie::build_byte(&vocab).expect("trie");
        let n = boolean_engine().state_count();
        let wpr = handle.canonical().mask_width_words();
        let scratch = crate::index::packed_scratch_fixed_bytes().unwrap();
        let stack = crate::index::packed_stack_bytes(trie.max_dfs_frontier()).unwrap();
        assert!(stack > 0, "the DFS stack charge is load-bearing");
        let payload_header_scratch = n * wpr * 4 + PACKED_HEADER_ALLOWANCE + scratch;
        let peak = payload_header_scratch + stack;

        let budget = |b: usize| {
            CompiledArtifact::new_packed_from_trie_budgeted(
                boolean_engine(),
                handle.canonical().clone(),
                prov_bytes([4; 32]),
                &trie,
                handle.fingerprint(),
                b,
            )
        };
        assert_eq!(
            budget(payload_header_scratch).unwrap_err().code,
            ErrorCode::InternalLimitExceeded
        );
        assert_eq!(
            budget(peak - 1).unwrap_err().code,
            ErrorCode::InternalLimitExceeded
        );
        assert_eq!(
            budget(peak + 1).expect("peak + 1 binds").bind_mode(),
            BindMode::TrieJointBytePacked
        );
    }

    #[test]
    fn packed_admission_rejects_when_scratch_alone_overflows_an_otherwise_fitting_payload() {
        let vocab = Arc::new({
            let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
            map.insert(b"a".to_vec(), vec![0]);
            build_vocabulary(1, map).expect("vocab")
        });
        let handle = crate::index::VocabularyHandle::new(vocab.clone()).expect("handle");
        let trie = VocabTrie::build_byte(&vocab).expect("trie");
        let n = build_from_regex("a{1,200}").expect("engine").state_count();
        let wpr = handle.canonical().mask_width_words();
        let scratch = crate::index::packed_scratch_fixed_bytes().unwrap();
        assert!(
            scratch > 4096,
            "the dense scratch is the load-bearing charge ({scratch} bytes)"
        );
        let budget = n * wpr * 4 + PACKED_HEADER_ALLOWANCE + (scratch / 2);
        let err = CompiledArtifact::new_packed_from_trie_budgeted(
            build_from_regex("a{1,200}").unwrap(),
            handle.canonical().clone(),
            prov_bytes([4; 32]),
            &trie,
            handle.fingerprint(),
            budget,
        )
        .expect_err("scratch charged on top of a fitting payload must reject");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn packed_admission_holds_for_many_sparse_states_one_token_per_row() {
        let vocab = Arc::new({
            let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
            map.insert(b"a".to_vec(), vec![0]);
            build_vocabulary(1, map).expect("vocab")
        });
        let handle = crate::index::VocabularyHandle::new(vocab.clone()).expect("handle");
        let trie = VocabTrie::build_byte(&vocab).expect("trie");
        let n = build_from_regex("a{1,200}").expect("engine").state_count();
        assert!(n > 100, "the chain must have many states ({n})");
        let wpr = handle.canonical().mask_width_words();
        let scratch = crate::index::packed_scratch_fixed_bytes().unwrap();
        let fixed = n * wpr * 4 + PACKED_HEADER_ALLOWANCE + scratch;

        let under = CompiledArtifact::new_packed_from_trie_budgeted(
            build_from_regex("a{1,200}").unwrap(),
            handle.canonical().clone(),
            prov_bytes([4; 32]),
            &trie,
            handle.fingerprint(),
            fixed - 1,
        )
        .expect_err("one byte under the table+scratch total must reject");
        assert_eq!(under.code, ErrorCode::InternalLimitExceeded);

        let packed = CompiledArtifact::new_packed_from_trie(
            build_from_regex("a{1,200}").unwrap(),
            handle.canonical().clone(),
            prov_bytes([4; 32]),
            &trie,
            handle.fingerprint(),
        )
        .expect("generous budget binds the many-state chain");
        let naive = CompiledArtifact::new(
            build_from_regex("a{1,200}").unwrap(),
            vocab,
            prov_bytes([4; 32]),
        )
        .expect("naive");
        for raw in 0..naive.engine.state_count() {
            let s = StateId(raw as u32);
            assert_eq!(
                packed.allowed_mask(s).unwrap().as_words(),
                naive.allowed_mask(s).unwrap().as_words(),
                "packed != naive at state {raw}"
            );
        }
    }

    #[test]
    fn recommended_bind_mode_is_a_no_op_for_every_non_packed_request() {
        let vocab = Arc::new({
            let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
            map.insert(b"a".to_vec(), vec![0]);
            build_vocabulary(1, map).expect("vocab")
        });
        let handle = crate::index::VocabularyHandle::new(vocab).expect("handle");
        let bound = crate::index::TrieCache::new().bind(&handle).expect("bound");
        let n = boolean_engine().state_count();
        for requested in [
            BindMode::Naive,
            BindMode::TrieJointByte,
            BindMode::TrieJointByteLazy,
            BindMode::TrieJointClass,
        ] {
            assert_eq!(
                CompiledArtifact::recommended_bind_mode(
                    n,
                    handle.mask_vocab_size(),
                    &bound,
                    requested
                ),
                requested,
                "{requested:?} must pass through unchanged"
            );
        }
    }

    #[test]
    fn recommended_bind_mode_keeps_packed_when_the_estimate_fits_the_real_budget() {
        let vocab = Arc::new({
            let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
            map.insert(b"a".to_vec(), vec![0]);
            build_vocabulary(1, map).expect("vocab")
        });
        let handle = crate::index::VocabularyHandle::new(vocab).expect("handle");
        let bound = crate::index::TrieCache::new().bind(&handle).expect("bound");
        let n = boolean_engine().state_count();
        assert_eq!(
            CompiledArtifact::recommended_bind_mode(
                n,
                handle.mask_vocab_size(),
                &bound,
                BindMode::TrieJointBytePacked
            ),
            BindMode::TrieJointBytePacked,
            "a tiny grammar/vocabulary must stay packed"
        );
    }

    #[test]
    fn recommended_bind_mode_downgrades_on_latency_alone_well_under_the_byte_budget() {
        let vocab = Arc::new({
            let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
            map.insert(b"a".to_vec(), vec![0]);
            build_vocabulary(1, map).expect("vocab")
        });
        let handle = crate::index::VocabularyHandle::new(vocab).expect("handle");
        let bound = crate::index::TrieCache::new().bind(&handle).expect("bound");
        let state_count =
            (PACKED_COMPILE_LATENCY_BUDGET_US * 1000) / PACKED_ROW_FILL_NS_PER_WORD + 1000;
        assert!(
            state_count * 4 < MAX_PACKED_MASK_BYTES,
            "fixture sanity: byte budget not the cause"
        );
        assert_eq!(
            CompiledArtifact::recommended_bind_mode(
                state_count,
                handle.mask_vocab_size(),
                &bound,
                BindMode::TrieJointBytePacked
            ),
            BindMode::TrieJointByteLazy,
            "latency alone must downgrade even when bytes are nowhere near the budget"
        );
    }

    #[test]
    fn recommended_bind_mode_downgrades_to_lazy_when_the_payload_alone_exceeds_the_real_budget() {
        let vocab = Arc::new({
            let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
            map.insert(b"a".to_vec(), vec![0]);
            build_vocabulary(1, map).expect("vocab")
        });
        let handle = crate::index::VocabularyHandle::new(vocab).expect("handle");
        let bound = crate::index::TrieCache::new().bind(&handle).expect("bound");
        let wpr = handle.mask_vocab_size().div_ceil(32);
        assert_eq!(
            wpr, 1,
            "fixture sanity: one row word, so payload = state_count * 4"
        );
        let huge_state_count = MAX_PACKED_MASK_BYTES / 4 + 1; // payload alone > MAX_PACKED_MASK_BYTES

        assert_eq!(
            CompiledArtifact::recommended_bind_mode(
                huge_state_count,
                handle.mask_vocab_size(),
                &bound,
                BindMode::TrieJointBytePacked,
            ),
            BindMode::TrieJointByteLazy,
            "a payload alone past the fixed packed budget must downgrade to lazy"
        );
    }

    #[test]
    fn recommended_bind_mode_never_recommends_packed_when_the_real_build_would_reject() {
        let vocab = Arc::new({
            let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
            map.insert(b"a".to_vec(), vec![0]);
            build_vocabulary(1, map).expect("vocab")
        });
        let handle = crate::index::VocabularyHandle::new(vocab.clone()).expect("handle");
        let trie = VocabTrie::build_byte(&vocab).expect("trie");
        let n = build_from_regex("a{1,200}").expect("engine").state_count();
        let wpr = handle.canonical().mask_width_words();
        let peak = CompiledArtifact::packed_build_peak_bytes(n, wpr, &trie).expect("peak");

        CompiledArtifact::new_packed_from_trie_budgeted(
            build_from_regex("a{1,200}").unwrap(),
            handle.canonical().clone(),
            prov_bytes([4; 32]),
            &trie,
            handle.fingerprint(),
            peak,
        )
        .expect("the real build must bind at exactly `peak`");
        CompiledArtifact::new_packed_from_trie_budgeted(
            build_from_regex("a{1,200}").unwrap(),
            handle.canonical().clone(),
            prov_bytes([4; 32]),
            &trie,
            handle.fingerprint(),
            peak - 1,
        )
        .expect_err("the real build must reject one byte under `peak`");
        assert!(
            peak <= MAX_PACKED_MASK_BYTES,
            "fixture sanity: this small chain must fit the real production budget"
        );

        let bound = crate::index::TrieCache::new().bind(&handle).expect("bound");
        assert_eq!(
            CompiledArtifact::recommended_bind_mode(
                n,
                handle.mask_vocab_size(),
                &bound,
                BindMode::TrieJointBytePacked,
            ),
            BindMode::TrieJointBytePacked
        );
    }

    #[test]
    fn new_trie_rejects_lazy_and_naive_with_a_typed_error() {
        let vocab = Arc::new({
            let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
            map.insert(b"t".to_vec(), vec![0]);
            build_vocabulary(9, map).expect("vocab")
        });
        let trie = VocabTrie::build_byte(&vocab).expect("trie");
        for mode in [BindMode::TrieJointByteLazy, BindMode::Naive] {
            let err = CompiledArtifact::new_trie(
                boolean_engine(),
                vocab.clone(),
                prov_bytes([2; 32]),
                mode,
                &trie,
            )
            .expect_err("lazy/naive are not constructible from a borrowed trie");
            assert_eq!(err.code, ErrorCode::ArtifactOutOfBounds);
        }
    }

    #[test]
    fn new_trie_builds_packed_equal_to_naive() {
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        map.insert(b"t".to_vec(), vec![0]);
        map.insert(b"true".to_vec(), vec![1]);
        map.insert(b"z".to_vec(), vec![2, 3]);
        let vocab = Arc::new(build_vocabulary(9, map).expect("vocab"));
        let trie = VocabTrie::build_byte(&vocab).expect("trie");
        let packed = CompiledArtifact::new_trie(
            boolean_engine(),
            vocab.clone(),
            prov_bytes([6; 32]),
            BindMode::TrieJointBytePacked,
            &trie,
        )
        .expect("packed via new_trie");
        let naive = boolean_artifact();
        assert_eq!(packed.bind_mode(), BindMode::TrieJointBytePacked);
        for raw in 0..packed.engine.state_count() {
            let s = StateId(raw as u32);
            assert_eq!(
                packed.allowed_mask(s).unwrap().as_words(),
                naive.allowed_mask(s).unwrap().as_words()
            );
        }
    }

    fn object_with_ws_engine() -> crate::automaton::RefEngine {
        use crate::compile::compile_ir;
        use crate::ir::{Builder, CompileOptions, ObjectClosure};
        let mut b = Builder::new(CompileOptions::default());
        let boolean = b.boolean().unwrap();
        let obj = b
            .object(
                vec![("a".to_string(), boolean)],
                &[true],
                ObjectClosure::AssumeClosedProfile,
            )
            .unwrap();
        let ir = b.finish(obj).unwrap();
        compile_ir(&ir).unwrap()
    }

    /// The Task 2 acceptance gate: a spaced instance must be accepted, and every bind mode must
    /// compute the identical mask, since whitespace lives in the shared engine, not the bind layer.
    #[test]
    fn whitespace_variants_agree_across_naive_packed_and_lazy_bind_modes() {
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        for (bytes, id) in [
            (b"{".as_slice(), 0u32),
            (b"\"a\"".as_slice(), 1),
            (b":".as_slice(), 2),
            (b" ".as_slice(), 3),
            (b"true".as_slice(), 4),
            (b"}".as_slice(), 5),
        ] {
            map.insert(bytes.to_vec(), vec![id]);
        }
        let vocab = Arc::new(build_vocabulary(6, map).expect("vocab"));

        let naive =
            CompiledArtifact::new(object_with_ws_engine(), vocab.clone(), prov_bytes([11; 32]))
                .expect("naive");
        let trie = VocabTrie::build_byte(&vocab).expect("trie");
        let packed = CompiledArtifact::new_trie(
            object_with_ws_engine(),
            vocab.clone(),
            prov_bytes([12; 32]),
            BindMode::TrieJointBytePacked,
            &trie,
        )
        .expect("packed");
        let fp = VocabFingerprint::of(&vocab).unwrap();
        let canonical = Arc::new(CanonicalVocabulary::from_vocabulary(&vocab).expect("canonical"));
        let lazy = CompiledArtifact::new_lazy_trie(
            object_with_ws_engine(),
            canonical,
            prov_bytes([13; 32]),
            Arc::new(VocabTrie::build_byte(&vocab).expect("trie")),
            fp,
        )
        .expect("lazy");

        for raw in 0..naive.engine.state_count() {
            let s = StateId(raw as u32);
            let n = naive.allowed_mask(s).expect("naive mask");
            assert_eq!(
                n.as_words(),
                packed.allowed_mask(s).expect("packed mask").as_words()
            );
            assert_eq!(
                n.as_words(),
                lazy.allowed_mask(s).expect("lazy mask").as_words()
            );
        }

        for artifact in [&naive, &packed, &lazy] {
            let mut s = artifact.start();
            for tok in [b"{".as_slice(), b"\"a\"", b":", b" ", b"true", b"}"] {
                s = artifact.walk(s, tok).expect("spaced token sequence walks");
            }
            assert!(
                artifact.is_accepting(s),
                "{:?} must accept {{\"a\": true}}",
                artifact.bind_mode()
            );
        }
    }

    #[test]
    fn packed_mask_table_words_is_the_whole_table_row_major_matching_mask_row_words() {
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        map.insert(b"t".to_vec(), vec![0]);
        map.insert(b"true".to_vec(), vec![1]);
        map.insert(b"z".to_vec(), vec![2, 3]);
        let vocab = Arc::new(build_vocabulary(9, map).expect("vocab"));
        let trie = VocabTrie::build_byte(&vocab).expect("trie");
        let packed = CompiledArtifact::new_trie(
            boolean_engine(),
            vocab,
            prov_bytes([7; 32]),
            BindMode::TrieJointBytePacked,
            &trie,
        )
        .expect("packed via new_trie");

        let wpr = packed.words_per_row();
        let table = packed
            .packed_mask_table_words()
            .expect("packed mode always has a flat table");
        assert_eq!(table.len(), packed.state_count() * wpr);

        for raw in 0..packed.state_count() {
            let s = StateId(raw as u32);
            let row_via_lookup = packed.mask_row_words(s).unwrap();
            let row_via_table = &table[raw * wpr..(raw + 1) * wpr];
            assert_eq!(
                row_via_table, row_via_lookup,
                "the flat table's row {raw} must match the single-row accessor byte for byte"
            );
        }
    }

    #[test]
    fn packed_mask_table_words_is_none_outside_packed_mode() {
        let naive = boolean_artifact();
        assert_eq!(naive.bind_mode(), BindMode::Naive);
        assert!(naive.packed_mask_table_words().is_none());
    }

    #[test]
    fn new_from_handle_equals_new_and_cannot_mispair() {
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        map.insert(b"t".to_vec(), vec![0]);
        map.insert(b"true".to_vec(), vec![1]);
        map.insert(b"z".to_vec(), vec![2]);
        let vocab = Arc::new(build_vocabulary(9, map).expect("vocab"));
        let handle = crate::index::VocabularyHandle::new(vocab.clone()).expect("handle");
        let engine = boolean_engine();
        let shared = CompiledArtifact::new_from_handle(&handle, engine, prov_bytes([5; 32]))
            .expect("handle");
        let fresh =
            CompiledArtifact::new(boolean_engine(), vocab, prov_bytes([5; 32])).expect("fresh");
        for raw in 0..shared.engine.state_count() {
            let s = StateId(raw as u32);
            assert_eq!(
                shared.allowed_mask(s).unwrap().as_words(),
                fresh.allowed_mask(s).unwrap().as_words(),
                "handle-shared metadata must equal freshly-built metadata at {raw}"
            );
        }
    }

    fn prov_bytes(h: [u8; 32]) -> Provenance {
        Provenance::reference(HashState::Hash(h))
    }

    #[test]
    fn lazy_cached_mask_walks_once_then_reuses_the_arc_and_equals_naive() {
        let lazy = boolean_lazy_artifact();
        let naive = boolean_artifact();
        for raw in 0..lazy.engine.state_count() {
            let s = StateId(raw as u32);
            let first = lazy.cached_mask(s).expect("miss walks the trie");
            let second = lazy.cached_mask(s).expect("hit reuses the cache");
            assert!(
                Arc::ptr_eq(&first, &second),
                "a repeated lazy query must reuse the cached mask, not re-walk"
            );
            assert_eq!(
                first.as_words(),
                naive.allowed_mask(s).expect("naive").as_words(),
                "lazy direct-mask walk must equal naive at state {raw}"
            );
        }
    }

    #[test]
    fn growing_lazy_rows_never_exceed_the_charged_upper_bound() {
        let lazy = boolean_lazy_artifact();
        let upper_bound = lazy.artifact_cache_charge_upper_bound();
        for raw in 0..lazy.engine.state_count() {
            let s = StateId(raw as u32);
            let _ = lazy.cached_mask(s).expect("mask");
            assert_eq!(
                lazy.artifact_cache_charge_upper_bound(),
                upper_bound,
                "the charged upper bound is fixed at build time, not measured live"
            );
            assert!(
                lazy.mask_cache_retained_bytes() <= upper_bound,
                "live retained bytes must never exceed what was charged"
            );
        }
    }

    #[test]
    fn write_mask_into_matches_the_reference_mask_bit_for_bit() {
        let artifact = boolean_artifact();
        let s = artifact.start();
        let reference = artifact.allowed_mask(s).expect("reference mask");
        let mut out = vec![0u32; artifact.words_per_row()];
        artifact
            .write_mask_into(&[s], &mut out, None)
            .expect("fill");
        assert_eq!(out.as_slice(), reference.as_words());
    }

    #[test]
    fn allowed_ids_matches_a_per_id_mask_scan_over_every_state() {
        let artifact = boolean_artifact();
        for raw in 0..artifact.state_count() {
            let s = StateId(raw as u32);
            let mask = artifact.allowed_mask(s).expect("mask");
            let reference: Vec<TokenId> = (0..mask.vocab_size())
                .map(|id| TokenId(id as u32))
                .filter(|&id| mask.get(id))
                .collect();
            assert_eq!(artifact.allowed_ids(s).expect("allowed_ids"), reference);
        }
    }

    #[test]
    fn write_mask_le_bytes_into_matches_write_mask_into_byte_for_byte() {
        let artifact = boolean_artifact();
        let s = artifact.start();
        let mut words = vec![0u32; artifact.words_per_row()];
        artifact.write_mask_into(&[s], &mut words, None).unwrap();
        let mut bytes = vec![0u8; artifact.words_per_row() * 4];
        artifact
            .write_mask_le_bytes_into(&[s], &mut bytes, None)
            .unwrap();
        let expected: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(bytes, expected);
    }

    #[test]
    fn write_mask_le_bytes_into_clears_a_dirty_reused_buffer() {
        let artifact = boolean_artifact();
        let s = artifact.start();
        let mut fresh = vec![0u8; artifact.words_per_row() * 4];
        artifact
            .write_mask_le_bytes_into(&[s], &mut fresh, None)
            .unwrap();
        let mut dirty = vec![0xFFu8; artifact.words_per_row() * 4];
        artifact
            .write_mask_le_bytes_into(&[s], &mut dirty, None)
            .unwrap();
        assert_eq!(dirty, fresh, "stale bits must be cleared, not OR-ed in");
    }

    #[test]
    fn write_mask_le_bytes_into_rejects_a_mismatched_buffer_length() {
        let artifact = boolean_artifact();
        let s = artifact.start();
        let mut too_short = vec![0u8; artifact.words_per_row() * 4 - 1];
        let err = artifact
            .write_mask_le_bytes_into(&[s], &mut too_short, None)
            .expect_err("length mismatch is a structured error");
        assert_eq!(err.code, ErrorCode::ArtifactOutOfBounds);
    }

    #[test]
    fn cached_mask_hit_returns_the_same_content_as_a_fresh_computation() {
        let artifact = boolean_artifact();
        let s = artifact.start();
        let fresh = artifact.allowed_mask(s).expect("uncached");
        let first = artifact.cached_mask(s).expect("miss, populates cache");
        let second = artifact.cached_mask(s).expect("hit, reads cache");
        assert_eq!(first.as_words(), fresh.as_words());
        assert_eq!(second.as_words(), fresh.as_words());
        assert!(
            Arc::ptr_eq(&first, &second),
            "a hit must reuse the cached Arc, not recompute"
        );
    }

    #[test]
    fn cached_mask_does_not_confuse_two_different_states() {
        let artifact = boolean_artifact();
        let start = artifact.start();
        let dead = artifact.dead();
        let start_mask = artifact.cached_mask(start).expect("start");
        let dead_mask = artifact.cached_mask(dead).expect("dead");
        assert_ne!(
            start_mask.as_words(),
            dead_mask.as_words(),
            "a live start state must not share the dead state's (empty) mask"
        );
        assert_eq!(
            dead_mask.as_words(),
            artifact.allowed_mask(dead).unwrap().as_words()
        );
    }

    #[test]
    fn cached_mask_is_isolated_per_artifact_not_shared_globally() {
        let a = boolean_artifact();
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        map.insert(b"t".to_vec(), vec![0]);
        map.insert(b"true".to_vec(), vec![1]);
        let vocab_b = build_vocabulary(5, map).expect("vocab");
        let engine_b = corpus_cases()
            .expect("corpus")
            .into_iter()
            .find(|c| c.name == "suite-type-boolean")
            .expect("boolean case")
            .engine;
        let b = CompiledArtifact::new(
            engine_b,
            Arc::new(vocab_b),
            Provenance::reference(HashState::Hash([9; 32])),
        )
        .expect("artifact b");

        let s = a.start();
        let mask_a = a.cached_mask(s).expect("a");
        let mask_b = b.cached_mask(s).expect("b");
        assert_eq!(mask_a.as_words().len(), a.words_per_row());
        assert_eq!(mask_b.as_words().len(), b.words_per_row());
        assert_ne!(
            a.vocab_size(),
            b.vocab_size(),
            "the two artifacts have different vocab sizes (EOS 9 vs EOS 5)"
        );
        assert_eq!(mask_a.as_words(), a.allowed_mask(s).unwrap().as_words());
        assert_eq!(mask_b.as_words(), b.allowed_mask(s).unwrap().as_words());
    }

    #[test]
    fn cached_mask_agrees_with_uncached_mask_at_vocab_boundary_sizes() {
        for vocab_size in [0usize, 1, 31, 32, 33, 64, 65] {
            let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
            for i in 0..vocab_size {
                map.insert(vec![b'a', i as u8], vec![i as u32]);
            }
            let eos = vocab_size as u32;
            let vocab = build_vocabulary(eos, map).expect("vocab");
            let engine = corpus_cases()
                .expect("corpus")
                .into_iter()
                .find(|c| c.name == "suite-type-boolean")
                .expect("boolean case")
                .engine;
            let artifact = CompiledArtifact::new(
                engine,
                Arc::new(vocab),
                Provenance::reference(HashState::Hash([3; 32])),
            )
            .expect("artifact");
            let s = artifact.start();
            let cached = artifact.cached_mask(s).expect("cached");
            let uncached = artifact.allowed_mask(s).expect("uncached");
            assert_eq!(
                cached.as_words(),
                uncached.as_words(),
                "vocab_size={vocab_size}"
            );
        }
    }

    #[test]
    fn write_mask_into_rejects_a_mismatched_buffer_length() {
        let artifact = boolean_artifact();
        let s = artifact.start();
        let mut too_short = vec![0u32; artifact.words_per_row().saturating_sub(1)];
        let err = artifact
            .write_mask_into(&[s], &mut too_short, None)
            .expect_err("length mismatch is a structured error");
        assert_eq!(err.code, ErrorCode::ArtifactOutOfBounds);
    }

    #[test]
    fn write_mask_into_rejects_an_overflowing_row_index() {
        let artifact = boolean_artifact();
        let s = artifact.start();
        let mut out = vec![0u32; artifact.words_per_row()];
        let err = artifact
            .write_mask_into(&[s], &mut out, Some(&[usize::MAX]))
            .expect_err("a row index of usize::MAX must not overflow");
        assert_eq!(err.code, ErrorCode::ArtifactOutOfBounds);
    }

    #[test]
    fn advance_tracks_consume_token_and_leaves_state_on_rejection() {
        let artifact = Arc::new(boolean_artifact());
        let mut m = Matcher::new(artifact);
        m.advance(TokenId(1)).expect("`true` is legal from start");
        assert!(m.is_accepting());
        assert!(!m.is_dead());
        assert_eq!(m.bytes_seen(), 4);
        let before = m.state();
        assert!(m.advance(TokenId(2)).is_err());
        assert_eq!(m.state(), before);
    }

    #[test]
    fn empty_language_matcher_is_dead_and_rejects_every_token_forever() {
        let engine = build_from_regex("[a&&b]").expect("empty-language engine");
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        map.insert(b"a".to_vec(), vec![0]);
        map.insert(b"b".to_vec(), vec![1]);
        let vocab = build_vocabulary(5, map).expect("vocab");
        let artifact = Arc::new(
            CompiledArtifact::new(
                engine,
                Arc::new(vocab),
                Provenance::reference(HashState::Hash([7; 32])),
            )
            .expect("artifact"),
        );
        assert!(artifact.start_is_dead());
        let mut m = Matcher::new(artifact);
        assert!(m.is_dead());
        for id in [0u32, 1, 2, 99] {
            assert!(m.advance(TokenId(id)).is_err());
            assert!(m.is_dead(), "is_dead must stay true after every attempt");
        }
    }

    #[test]
    fn unknown_token_id_is_illegal() {
        let artifact = Arc::new(boolean_artifact());
        let mut m = Matcher::new(artifact);
        match m.advance(TokenId(999)) {
            Err(MatcherError::IllegalToken { token }) => assert_eq!(token, TokenId(999)),
            other => panic!("expected IllegalToken, got {other:?}"),
        }
    }

    #[test]
    fn artifact_rejects_incomplete_provenance() {
        let engine = corpus_cases()
            .expect("corpus")
            .into_iter()
            .find(|c| c.name == "suite-type-null")
            .expect("null case")
            .engine;
        let vocab = build_vocabulary(1, Map::default()).expect("vocab");
        let err = CompiledArtifact::new(
            engine,
            Arc::new(vocab),
            Provenance::reference(HashState::Unhashed),
        )
        .expect_err("unhashed reference provenance is incomplete");
        assert_eq!(err.code, ErrorCode::ProvenanceIncomplete);
    }

    #[test]
    fn token_bytes_returns_the_stored_bytes_for_multi_id_and_repeated_sequences() {
        let artifact = boolean_artifact();
        assert_eq!(artifact.token_bytes(TokenId(0)), Some(b"t".as_slice()));
        assert_eq!(artifact.token_bytes(TokenId(1)), Some(b"true".as_slice()));
        assert_eq!(artifact.token_bytes(TokenId(2)), Some(b"z".as_slice()));
        assert_eq!(artifact.token_bytes(TokenId(3)), Some(b"z".as_slice()));
        assert_eq!(
            artifact.token_bytes(TokenId(2)),
            artifact.token_bytes(TokenId(3)),
            "two ids sharing one byte sequence must return the same bytes"
        );
        assert_eq!(artifact.token_bytes(TokenId(42)), None);
    }

    #[test]
    fn validate_token_bytes_accepts_a_clean_map_and_rejects_a_conflicting_id() {
        let mut ok: Map<Vec<u8>, Vec<u32>> = Map::default();
        ok.insert(b"a".to_vec(), vec![0]);
        ok.insert(b"bb".to_vec(), vec![1, 2]);
        let clean = build_vocabulary(9, ok).expect("vocab");
        validate_token_bytes(&clean).expect("a clean map has no id conflict");

        let mut bad: Map<Vec<u8>, Vec<u32>> = Map::default();
        bad.insert(b"a".to_vec(), vec![5]);
        bad.insert(b"b".to_vec(), vec![5]);
        let conflicting = build_vocabulary(9, bad).expect("vocab");
        let err = validate_token_bytes(&conflicting)
            .expect_err("one id under two byte sequences is ambiguous");
        assert_eq!(err.code, ErrorCode::MalformedTokenizer);
    }

    #[test]
    fn artifact_rejects_one_id_under_two_byte_sequences() {
        let engine = corpus_cases()
            .expect("corpus")
            .into_iter()
            .find(|c| c.name == "suite-type-boolean")
            .expect("boolean case")
            .engine;
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        map.insert(b"a".to_vec(), vec![5]);
        map.insert(b"b".to_vec(), vec![5]);
        let vocab = build_vocabulary(9, map).expect("vocab");
        let err = CompiledArtifact::new(
            engine,
            Arc::new(vocab),
            Provenance::reference(HashState::Hash([7; 32])),
        )
        .expect_err("one id under two byte sequences is ambiguous");
        assert_eq!(err.code, ErrorCode::MalformedTokenizer);
    }

    fn boolean_engine() -> crate::automaton::RefEngine {
        corpus_cases()
            .expect("corpus")
            .into_iter()
            .find(|c| c.name == "suite-type-boolean")
            .expect("boolean case")
            .engine
    }

    #[test]
    fn a_forged_huge_token_id_is_rejected_before_allocating_a_mask() {
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        map.insert(b"t".to_vec(), vec![u32::MAX - 1]);
        let vocab = build_vocabulary(9, map).expect("vocab");
        let err = CompiledArtifact::new(
            boolean_engine(),
            Arc::new(vocab),
            Provenance::reference(HashState::Hash([7; 32])),
        )
        .expect_err("an oversized mask width must be rejected");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn forged_out_of_range_state_ids_do_not_grow_the_mask_cache() {
        let artifact = boolean_artifact();
        let n = artifact.state_count() as u32;
        for id in n..n + 1000 {
            let _ = artifact
                .cached_mask(StateId(id))
                .expect("out-of-range mask");
        }
        assert_eq!(
            artifact.mask_cache_filled_count(),
            0,
            "a forged out-of-range id must never be cached"
        );
        artifact
            .cached_mask(artifact.start())
            .expect("in-range mask");
        let cached = artifact.mask_cache_filled_count();
        assert!(
            (1..=artifact.state_count()).contains(&cached),
            "only in-range states are cached, bounded by state_count"
        );
    }

    #[test]
    fn the_mask_cache_respects_its_byte_budget() {
        let artifact = boolean_artifact();
        artifact.set_mask_cache_budget(0);
        artifact
            .cached_mask(artifact.start())
            .expect("mask still computed");
        assert_eq!(
            artifact.mask_cache_filled_count(),
            0,
            "a zero budget caches nothing, but the mask is still returned"
        );
        let one_entry =
            artifact.words_per_row() * std::mem::size_of::<u32>() + MASK_CACHE_ENTRY_OVERHEAD;
        artifact.set_mask_cache_budget(one_entry);
        artifact.cached_mask(artifact.start()).expect("first fits");
        artifact
            .cached_mask(artifact.dead())
            .expect("second does not fit");
        assert!(
            artifact.mask_cache_retained_bytes() <= one_entry,
            "cached bytes never exceed the budget"
        );
        assert!(
            artifact.mask_cache_filled_count() <= 1,
            "only what fits in the budget is retained"
        );
    }

    #[test]
    fn concurrent_lazy_queries_across_many_threads_agree_with_naive() {
        let lazy = boolean_lazy_artifact();
        let naive = boolean_artifact();
        let n = lazy.engine.state_count();
        std::thread::scope(|scope| {
            for t in 0..32 {
                let lazy = &lazy;
                let naive = &naive;
                scope.spawn(move || {
                    for raw in 0..n {
                        let s = StateId(((raw + t) % n) as u32);
                        assert_eq!(
                            lazy.cached_mask(s).unwrap().as_words(),
                            naive.cached_mask(s).unwrap().as_words(),
                            "thread {t} state {raw} diverged from naive"
                        );
                    }
                });
            }
        });
    }

    #[test]
    fn concurrent_race_on_one_state_charges_exactly_one_entry() {
        let artifact = boolean_lazy_artifact();
        let entry_bytes = artifact.words_per_row() * 4 + MASK_CACHE_ENTRY_OVERHEAD;
        std::thread::scope(|scope| {
            for _ in 0..32 {
                let artifact = &artifact;
                scope.spawn(move || {
                    artifact.cached_mask(artifact.start()).unwrap();
                });
            }
        });
        assert_eq!(artifact.mask_cache_retained_bytes(), entry_bytes);
    }

    #[test]
    fn concurrent_race_across_m_states_charges_exactly_m_entries() {
        let artifact = boolean_lazy_artifact();
        let m = artifact.engine.state_count();
        let entry_bytes = artifact.words_per_row() * 4 + MASK_CACHE_ENTRY_OVERHEAD;
        std::thread::scope(|scope| {
            for _ in 0..32 {
                let artifact = &artifact;
                scope.spawn(move || {
                    for raw in 0..m {
                        artifact.cached_mask(StateId(raw as u32)).unwrap();
                    }
                });
            }
        });
        assert_eq!(artifact.mask_cache_retained_bytes(), m * entry_bytes);
    }

    #[test]
    fn a_failed_compute_refunds_exactly_one_reservation() {
        let rows = LazyRows::try_new(4, 1024).expect("rows");
        let err = || {
            Err(CompileError::new(
                ErrorCode::Unsupported,
                Stage::L4Bind,
                "injected",
            ))
        };
        let result = rows.get_or_compute(0, 100, err);
        assert!(result.is_err());
        assert_eq!(
            rows.retained_bytes(),
            0,
            "a failed compute must refund its reservation"
        );
    }

    #[test]
    fn packed_artifact_allocates_no_lazy_rows() {
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        map.insert(b"t".to_vec(), vec![0]);
        let vocab = Arc::new(build_vocabulary(1, map).expect("vocab"));
        let trie = VocabTrie::build_byte(&vocab).expect("trie");
        let packed = CompiledArtifact::new_trie(
            boolean_engine(),
            vocab,
            prov_bytes([9; 32]),
            BindMode::TrieJointBytePacked,
            &trie,
        )
        .expect("packed");
        assert!(packed.mask_cache_is_disabled());
        let lazy_share = packed
            .artifact_cache_charge_upper_bound()
            .saturating_sub(packed.engine_retained_bytes())
            .saturating_sub(packed.packed_retained_bytes())
            .saturating_sub(packed.index_retained_bytes());
        assert_eq!(
            lazy_share, 0,
            "packed's charge must carry zero lazy-row bytes"
        );
    }

    #[test]
    fn oversized_lazy_row_array_fails_structurally_before_allocating() {
        let huge_state_count = MAX_LAZY_ROWS_BYTES / std::mem::size_of::<MaskSlot>() + 1;
        let err =
            LazyRows::try_new(huge_state_count, 1024).expect_err("must reject before allocating");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn mask_slot_stays_compact_via_arc_compile_error() {
        let slot = std::mem::size_of::<MaskSlot>();
        let inline = std::mem::size_of::<OnceLock<Result<Arc<Bitmask>, CompileError>>>();
        assert!(
            slot <= 24,
            "MaskSlot grew past 24 bytes: {slot} - CompileError may be inlined again"
        );
        assert!(
            slot < inline,
            "Arc<CompileError> ({slot}B) must stay smaller than inlining it ({inline}B)"
        );
    }

    #[test]
    fn packed_forced_lazy_and_adaptive_masks_all_equal_naive() {
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        map.insert(b"t".to_vec(), vec![0]);
        map.insert(b"true".to_vec(), vec![1]);
        map.insert(b"z".to_vec(), vec![2, 3]);
        let vocab = Arc::new(build_vocabulary(9, map).expect("vocab"));
        let handle = crate::index::VocabularyHandle::new(vocab.clone()).expect("handle");
        let bound = crate::index::TrieCache::new().bind(&handle).expect("bound");
        let naive = boolean_artifact();
        let n = boolean_engine().state_count();

        let packed = CompiledArtifact::new_from_bound_trie(
            &handle,
            boolean_engine(),
            prov_bytes([8; 32]),
            BindMode::TrieJointBytePacked,
            &bound,
        )
        .expect("packed");
        let lazy = CompiledArtifact::new_from_bound_trie(
            &handle,
            boolean_engine(),
            prov_bytes([8; 32]),
            BindMode::TrieJointByteLazy,
            &bound,
        )
        .expect("lazy");
        let adaptive_mode = CompiledArtifact::recommended_bind_mode(
            n,
            handle.mask_vocab_size(),
            &bound,
            BindMode::TrieJointBytePacked,
        );
        let adaptive = CompiledArtifact::new_from_bound_trie(
            &handle,
            boolean_engine(),
            prov_bytes([8; 32]),
            adaptive_mode,
            &bound,
        )
        .expect("adaptive");

        for raw in 0..n {
            let s = StateId(raw as u32);
            let expected = naive.allowed_mask(s).unwrap();
            assert_eq!(
                packed.allowed_mask(s).unwrap().as_words(),
                expected.as_words()
            );
            assert_eq!(
                lazy.allowed_mask(s).unwrap().as_words(),
                expected.as_words()
            );
            assert_eq!(
                adaptive.allowed_mask(s).unwrap().as_words(),
                expected.as_words()
            );
        }
    }

    #[test]
    fn a_shared_canonical_vocabulary_is_not_rebuilt_per_artifact() {
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        map.insert(b"t".to_vec(), vec![0]);
        let vocab = Arc::new(build_vocabulary(9, map).expect("vocab"));
        let handle = crate::index::VocabularyHandle::new(vocab).expect("handle");
        let before = Arc::strong_count(handle.canonical());
        let a = CompiledArtifact::new_from_handle(
            &handle,
            boolean_engine(),
            Provenance::reference(HashState::Hash([7; 32])),
        )
        .expect("artifact a");
        let b = CompiledArtifact::new_from_handle(
            &handle,
            boolean_engine(),
            Provenance::reference(HashState::Hash([8; 32])),
        )
        .expect("artifact b");
        assert_eq!(
            Arc::strong_count(handle.canonical()),
            before + 2,
            "each artifact shares the canonical Arc, never rebuilds it"
        );
        assert_eq!(a.vocab_size(), b.vocab_size());
    }

    #[test]
    fn new_trie_rejects_a_trie_built_from_a_different_vocabulary() {
        use crate::index::{BindMode, VocabTrie};
        let mut map_a: Map<Vec<u8>, Vec<u32>> = Map::default();
        map_a.insert(b"a".to_vec(), vec![0]);
        let vocab_a = build_vocabulary(9, map_a).expect("vocab a");

        let mut map_b: Map<Vec<u8>, Vec<u32>> = Map::default();
        map_b.insert(b"b".to_vec(), vec![0]);
        let vocab_b = Arc::new(build_vocabulary(9, map_b).expect("vocab b"));

        let trie_a = VocabTrie::build_byte(&vocab_a).expect("trie a");
        let err = CompiledArtifact::new_trie(
            boolean_engine(),
            vocab_b,
            Provenance::reference(HashState::Hash([7; 32])),
            BindMode::TrieJointByte,
            &trie_a,
        )
        .expect_err("a trie from a different vocabulary must be rejected");
        assert_eq!(err.code, ErrorCode::ArtifactMismatch);
    }

    #[test]
    fn new_trie_prevalidated_still_rejects_a_genuine_mismatch() {
        use crate::index::{BindMode, VocabFingerprint, VocabTrie};
        let mut map_a: Map<Vec<u8>, Vec<u32>> = Map::default();
        map_a.insert(b"a".to_vec(), vec![0]);
        let vocab_a = build_vocabulary(9, map_a).expect("vocab a");
        let fingerprint_a = VocabFingerprint::of(&vocab_a).unwrap();

        let mut map_b: Map<Vec<u8>, Vec<u32>> = Map::default();
        map_b.insert(b"b".to_vec(), vec![0]);
        let vocab_b = Arc::new(build_vocabulary(9, map_b).expect("vocab b"));
        let trie_b = VocabTrie::build_byte(&vocab_b).expect("trie b");
        let canonical_b =
            Arc::new(CanonicalVocabulary::from_vocabulary(&vocab_b).expect("canonical b"));

        let err = CompiledArtifact::new_trie_prevalidated(
            boolean_engine(),
            canonical_b,
            Provenance::reference(HashState::Hash([7; 32])),
            BindMode::TrieJointByte,
            &trie_b,
            fingerprint_a, // wrong fingerprint for this trie
        )
        .expect_err(
            "a trie whose OWN fingerprint disagrees with the supplied one must be rejected",
        );
        assert_eq!(err.code, ErrorCode::ArtifactMismatch);
    }

    #[test]
    fn new_trie_prevalidated_agrees_with_new_trie_on_a_genuine_pairing() {
        use crate::index::{BindMode, VocabFingerprint, VocabTrie};
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        map.insert(b"true".to_vec(), vec![0]);
        map.insert(b"false".to_vec(), vec![1]);
        let vocab = Arc::new(build_vocabulary(9, map).expect("vocab"));
        let trie = VocabTrie::build_byte(&vocab).expect("trie");

        let a = CompiledArtifact::new_trie(
            boolean_engine(),
            vocab.clone(),
            Provenance::reference(HashState::Hash([3; 32])),
            BindMode::TrieJointByte,
            &trie,
        )
        .expect("generic new_trie");

        let fingerprint = VocabFingerprint::of(&vocab).unwrap();
        let canonical = Arc::new(CanonicalVocabulary::from_vocabulary(&vocab).expect("canonical"));
        let b = CompiledArtifact::new_trie_prevalidated(
            boolean_engine(),
            canonical,
            Provenance::reference(HashState::Hash([3; 32])),
            BindMode::TrieJointByte,
            &trie,
            fingerprint,
        )
        .expect("prevalidated new_trie");

        let s = a.start();
        assert_eq!(
            a.allowed_mask(s).unwrap().as_words(),
            b.allowed_mask(s).unwrap().as_words(),
            "the fast and generic constructors must produce byte-identical masks"
        );
    }

    fn handle_of(pairs: &[(&[u8], u32)], eos: u32) -> crate::index::VocabularyHandle {
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        for &(bytes, id) in pairs {
            map.insert(bytes.to_vec(), vec![id]);
        }
        let vocab = Arc::new(build_vocabulary(eos, map).expect("vocab"));
        crate::index::VocabularyHandle::new(vocab).expect("handle")
    }

    #[test]
    fn new_from_bound_trie_rejects_a_trie_bound_to_a_different_handle() {
        use crate::index::{BindMode, TrieCache};
        let cache = TrieCache::new();
        let a = handle_of(&[(b"a", 0)], 9);
        let b = handle_of(&[(b"b", 0)], 9);
        let bound_a = cache.bind(&a).expect("bind a");
        let err = CompiledArtifact::new_from_bound_trie(
            &b,
            boolean_engine(),
            Provenance::reference(HashState::Hash([7; 32])),
            BindMode::TrieJointByte,
            &bound_a,
        )
        .expect_err("a trie bound to handle a must not bind under handle b");
        assert_eq!(err.code, ErrorCode::ArtifactMismatch);
    }

    #[test]
    fn new_from_bound_trie_rejects_same_bytes_different_eos() {
        use crate::index::{BindMode, TrieCache};
        let cache = TrieCache::new();
        let a = handle_of(&[(b"a", 0), (b"b", 1)], 9);
        let b = handle_of(&[(b"a", 0), (b"b", 1)], 10);
        let bound_a = cache.bind(&a).expect("bind a");
        let err = CompiledArtifact::new_from_bound_trie(
            &b,
            boolean_engine(),
            Provenance::reference(HashState::Hash([7; 32])),
            BindMode::TrieJointByte,
            &bound_a,
        )
        .expect_err("identical token bytes under a different EOS must not bind");
        assert_eq!(err.code, ErrorCode::ArtifactMismatch);
    }

    #[test]
    fn new_from_bound_trie_rejects_same_ids_different_bytes() {
        use crate::index::{BindMode, TrieCache};
        let cache = TrieCache::new();
        let a = handle_of(&[(b"a", 0), (b"b", 1)], 9);
        let b = handle_of(&[(b"x", 0), (b"y", 1)], 9);
        let bound_a = cache.bind(&a).expect("bind a");
        let err = CompiledArtifact::new_from_bound_trie(
            &b,
            boolean_engine(),
            Provenance::reference(HashState::Hash([7; 32])),
            BindMode::TrieJointByte,
            &bound_a,
        )
        .expect_err("same token ids under different byte payloads must not bind");
        assert_eq!(err.code, ErrorCode::ArtifactMismatch);
    }

    #[test]
    fn new_from_bound_trie_accepts_a_genuinely_matching_pair() {
        use crate::index::{BindMode, TrieCache};
        let cache = TrieCache::new();
        let handle = handle_of(&[(b"true", 0), (b"false", 1)], 9);
        let bound = cache.bind(&handle).expect("bind");
        let artifact = CompiledArtifact::new_from_bound_trie(
            &handle,
            boolean_engine(),
            Provenance::reference(HashState::Hash([7; 32])),
            BindMode::TrieJointByte,
            &bound,
        )
        .expect("a genuinely matching handle and bound trie must bind");
        assert!(artifact.allowed_mask(artifact.start()).is_ok());
    }

    #[test]
    fn cache_bind_reuses_the_same_arc_across_two_handles_of_equal_content() {
        use crate::index::TrieCache;
        let cache = TrieCache::new();
        let a = handle_of(&[(b"a", 0), (b"b", 1)], 9);
        let b = handle_of(&[(b"a", 0), (b"b", 1)], 9);
        let bound_a = cache.bind(&a).expect("bind a");
        let bound_b = cache.bind(&b).expect("bind b");
        assert_eq!(
            bound_a.fingerprint(),
            bound_b.fingerprint(),
            "equal content under two distinct handles must share one cache entry"
        );
    }
}
