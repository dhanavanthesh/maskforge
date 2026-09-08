//! Caches schema-independent byte tries by vocabulary fingerprint.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use rustc_hash::FxHashMap;

use super::slice::TokenSliceCatalog;
use super::trie::VocabTrie;
use crate::error::{CompileError, Stage};
use crate::mem_gate::{acquire_vocab_build_bytes, BuildGate, PermitGuard};
use crate::vocab::{
    projected_peak_bytes, try_alloc, CanonicalVocabulary, PreparedVocabulary, Vocabulary,
};

/// A content address of a vocabulary's token map. Equal fingerprints denote the same token->bytes
/// relation, so they share a byte-trie.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct VocabFingerprint([u8; 32]);

impl VocabFingerprint {
    /// Computes the fingerprint from the token map SORTED by bytes then id (never the `HashMap`'s
    /// iteration order) plus the EOS id. Every variable-length field is length-prefixed - the byte
    pub fn of(vocab: &Vocabulary) -> Result<Self, CompileError> {
        match PreparedVocabulary::build(vocab) {
            Ok(prepared) => Ok(Self::of_prepared(&prepared, vocab.eos_token_id())),
            Err(_) => Self::try_of_unprepared(vocab),
        }
    }

    pub(crate) fn of_prepared(prepared: &PreparedVocabulary, eos: u32) -> Self {
        Self::of_prepared_with_ordinary_max(prepared, eos).0
    }

    /// Same digest as `of_prepared`, plus the largest non-EOS id seen along the way - one pass
    /// instead of two when a caller (canonical construction) needs both.
    pub(crate) fn of_prepared_with_ordinary_max(
        prepared: &PreparedVocabulary,
        eos: u32,
    ) -> (Self, Option<u32>) {
        let mut hasher = blake3::Hasher::new();
        let mut ordinary_max = None;
        for (bytes, ids) in prepared.iter() {
            hasher.update(&(bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
            hasher.update(&(ids.len() as u64).to_le_bytes());
            for &id in ids {
                hasher.update(&id.to_le_bytes());
                ordinary_max = Some(ordinary_max.map_or(id, |m: u32| m.max(id)));
            }
        }
        hasher.update(&eos.to_le_bytes());
        (Self(*hasher.finalize().as_bytes()), ordinary_max)
    }

    /// The same digest as `of_prepared`, computed directly from `vocab` without going through a
    /// `PreparedVocabulary` (the fallback when preparing one is not possible). Every scratch
    fn try_of_unprepared(vocab: &Vocabulary) -> Result<Self, CompileError> {
        let mut entries: Vec<(&[u8], Vec<u32>)> =
            try_alloc(vocab.tokens().len(), "fingerprint fallback entry scratch")?;
        for (bytes, ids) in vocab.tokens() {
            let mut sorted: Vec<u32> = try_alloc(ids.len(), "fingerprint fallback id scratch")?;
            sorted.extend_from_slice(ids);
            sorted.sort_unstable();
            entries.push((bytes.as_slice(), sorted));
        }
        entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
        let mut hasher = blake3::Hasher::new();
        for (bytes, ids) in &entries {
            hasher.update(&(bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
            hasher.update(&(ids.len() as u64).to_le_bytes());
            for &id in ids {
                hasher.update(&id.to_le_bytes());
            }
        }
        hasher.update(&vocab.eos_token_id().to_le_bytes());
        Ok(Self(*hasher.finalize().as_bytes()))
    }

    /// A fixed fingerprint for tests that build tries from synthetic key streams, not a vocabulary.
    #[cfg(test)]
    pub(crate) const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The raw 32-byte digest, for exposing the content address outside this crate.
    #[must_use]
    pub fn as_bytes(&self) -> [u8; 32] {
        self.0
    }
}

/// Default byte budget for retained tries: `1 << 28` = 256 MiB. Above it, a trie is still built and
/// returned but not retained, so compiling many distinct vocabularies cannot grow the cache forever.
const DEFAULT_TRIE_CACHE_BYTES: usize = 1 << 28;

/// Conservative per-entry overhead charged ON TOP of a trie's heap arrays: the `Arc` control block
/// (two counters) and the boxed `VocabTrie` struct, the `FxHashMap` slot (a 32-byte key plus the
const CACHE_ENTRY_OVERHEAD: usize = 2 * std::mem::size_of::<usize>()
    + std::mem::size_of::<VocabTrie>()
    + std::mem::size_of::<VocabFingerprint>()
    + std::mem::size_of::<usize>()
    + 64;

/// Default cap on SIMULTANEOUS distinct-vocabulary trie builds. The retained-byte budget bounds
/// CACHED tries, not TRANSIENT builds; without this, a burst of distinct large vocabularies (reachable
const DEFAULT_MAX_CONCURRENT_BUILDS: usize = 8;

/// Default budget for the ESTIMATED transient memory of concurrent trie builds. A count cap alone
/// does not bound memory (8 huge builds can still exhaust the process), so builds are also admitted
const DEFAULT_MAX_TRANSIENT_BUILD_BYTES: usize = 1 << 30;

/// A conservative upper bound on the transient bytes one byte-trie build needs, from token count and
/// total token bytes. The byte sum is a checked fold; overflow saturates to `usize::MAX` (over budget).
fn estimate_transient_build_bytes(vocab: &Vocabulary) -> usize {
    let total_bytes = vocab
        .tokens()
        .keys()
        .try_fold(0usize, |acc, k| acc.checked_add(k.len()))
        .unwrap_or(usize::MAX);
    let count = vocab.len();
    total_bytes
        .saturating_mul(48)
        .saturating_add(count.saturating_mul(12))
}

/// The same estimate as `estimate_transient_build_bytes`, read directly off an already-built
/// `PreparedVocabulary` (its total bytes are already known, so this needs no scan). Uses
fn estimate_transient_build_bytes_prepared(prepared: &PreparedVocabulary) -> usize {
    prepared
        .total_bytes()
        .saturating_mul(48)
        .saturating_add(prepared.total_ids().saturating_add(1).saturating_mul(12))
}

/// Selects how a cache miss builds its trie: from a raw `Vocabulary` (rehashes/resorts) or from an
/// already-canonical `PreparedVocabulary` (skips both).
enum BuildSource<'a> {
    Vocab(&'a Vocabulary),
    Prepared(&'a PreparedVocabulary),
}

#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(test)]
static DELAY_BUILDS: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static DELAY_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
thread_local! {
    static PANIC_ON_BUILD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The one builder's result for a fingerprint, read by every waiter, so a build the budget does NOT
/// retain is still not rebuilt once per waiter. `CompileError` is `Clone`, so the error is shared too.
#[derive(Debug)]
struct BuildSlot {
    result: Mutex<Option<Result<Arc<VocabTrie>, CompileError>>>,
    ready: Condvar,
}

/// A retained trie with its charged byte size and an LRU access stamp.
#[derive(Debug)]
struct CacheEntry {
    trie: Arc<VocabTrie>,
    bytes: usize,
    last_used: u64,
}

#[derive(Debug, Default)]
struct CacheState {
    map: FxHashMap<VocabFingerprint, CacheEntry>,
    bytes: usize,
    tick: u64,
    lru: VecDeque<(VocabFingerprint, u64)>,
    in_flight: FxHashMap<VocabFingerprint, Arc<BuildSlot>>,
}

impl CacheState {
    /// Records a touch for LRU: bumps the clock, appends `(fp, tick)` to the log, and returns the
    /// tick to store as the entry's `last_used`. Compacts the log once it outgrows the live entries.
    fn touch(&mut self, fp: VocabFingerprint) -> u64 {
        self.tick += 1;
        let now = self.tick;
        self.lru.push_back((fp, now));
        if self.lru.len() > 4 * self.map.len().max(1) + 64 {
            self.compact_lru();
        }
        now
    }

    /// Drops every log record that is no longer the live touch for its fingerprint.
    fn compact_lru(&mut self) {
        let map = &self.map;
        self.lru
            .retain(|(fp, tick)| map.get(fp).is_some_and(|e| e.last_used == *tick));
    }

    /// Evicts least-recently-used retained entries until `need` more bytes fit under `budget`. Only
    /// the cache's `Arc` is dropped; an artifact still holding its own `Arc` stays valid.
    fn evict_until_fits(&mut self, need: usize, budget: usize, stats: &mut CacheStats) {
        while self.bytes.saturating_add(need) > budget {
            let Some((fp, tick)) = self.lru.pop_front() else {
                break; // log exhausted; map empty (the new entry alone exceeds the budget: oversized)
            };
            if self.map.get(&fp).is_some_and(|e| e.last_used == tick) {
                if let Some(e) = self.map.remove(&fp) {
                    self.bytes -= e.bytes;
                    stats.evictions += 1;
                }
            } // else: a stale record from an earlier touch that was since superseded - skip it
        }
    }
}

/// Cumulative cache-traffic counters, since creation (not reset by `clear`).
#[derive(Debug, Default)]
struct CacheStats {
    hits: u64,
    misses: u64,
    evictions: u64,
    oversized_bypasses: u64,
}

/// A thread-safe, byte-bounded cache of byte-tries keyed by vocabulary fingerprint. Read-mostly
/// after warm-up; once the budget is reached, further tries are built and returned but not retained.
#[derive(Debug)]
pub struct TrieCache {
    state: Mutex<CacheState>,
    stats: Mutex<CacheStats>,
    budget: usize,
    gate: BuildGate,
}

impl Default for TrieCache {
    fn default() -> Self {
        Self::with_build_limit(DEFAULT_TRIE_CACHE_BYTES, DEFAULT_MAX_CONCURRENT_BUILDS)
    }
}

/// Publishes an error to the build slot and clears the in-flight mark on drop UNLESS disarmed, so a
/// panicking builder never leaves waiters hung on a slot that will never be filled, nor a permanently
struct SlotGuard<'a> {
    cache: &'a TrieCache,
    fp: VocabFingerprint,
    slot: Arc<BuildSlot>,
    armed: bool,
}

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            let mut r = self.slot.result.lock().unwrap_or_else(|p| p.into_inner());
            if r.is_none() {
                *r = Some(Err(CompileError::new(
                    crate::error::ErrorCode::InternalLimitExceeded,
                    crate::error::Stage::L4Bind,
                    "trie build failed to publish a result",
                )));
            }
            drop(r);
            self.slot.ready.notify_all();
            self.cache
                .state
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .in_flight
                .remove(&self.fp);
        }
    }
}

impl TrieCache {
    /// An empty cache with the default byte budget and concurrent-build cap.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty cache with an explicit retained-byte budget.
    #[must_use]
    pub fn with_budget(budget: usize) -> Self {
        Self::with_build_limit(budget, DEFAULT_MAX_CONCURRENT_BUILDS)
    }

    /// An empty cache with an explicit retained-byte budget and the default build gate.
    #[must_use]
    pub fn with_build_limit(budget: usize, max_concurrent_builds: usize) -> Self {
        Self::with_build_gate(
            budget,
            max_concurrent_builds,
            DEFAULT_MAX_TRANSIENT_BUILD_BYTES,
        )
    }

    /// An empty cache with an explicit retained-byte budget, a concurrent-build count cap, AND a
    /// transient-build byte budget (so many large concurrent builds cannot exhaust memory).
    #[must_use]
    pub fn with_build_gate(
        budget: usize,
        max_concurrent_builds: usize,
        max_transient_build_bytes: usize,
    ) -> Self {
        Self {
            state: Mutex::new(CacheState::default()),
            stats: Mutex::new(CacheStats::default()),
            budget,
            gate: BuildGate::new(
                max_concurrent_builds,
                max_transient_build_bytes,
                Stage::L4Bind,
            ),
        }
    }

    /// The peak number of simultaneous distinct-vocabulary builds observed (for tests/telemetry).
    #[must_use]
    pub fn peak_concurrent_builds(&self) -> usize {
        self.gate.peak_concurrent()
    }

    /// The peak concurrent transient-build bytes charged (for tests/telemetry).
    #[must_use]
    pub fn peak_concurrent_build_bytes(&self) -> usize {
        self.gate.peak_bytes()
    }

    /// The cached byte-trie for `vocab`, building and inserting it on a miss. Two vocabularies with
    /// equal fingerprints get the same `Arc`; the trie is built outside the lock, so a rare double
    pub fn get_or_build_byte(&self, vocab: &Vocabulary) -> Result<Arc<VocabTrie>, CompileError> {
        self.get_or_build_byte_prevalidated(vocab, VocabFingerprint::of(vocab)?)
    }

    /// Same lookup as `get_or_build_byte` but skips rehashing `vocab` on a hit too. Not exposed
    /// publicly: a mismatched `(vocab, fingerprint)` pair here is never re-verified against
    pub(crate) fn get_or_build_byte_prevalidated(
        &self,
        vocab: &Vocabulary,
        fingerprint: VocabFingerprint,
    ) -> Result<Arc<VocabTrie>, CompileError> {
        self.get_or_build_byte_impl(BuildSource::Vocab(vocab), fingerprint)
    }

    /// Same lookup, building from an ALREADY-canonical `PreparedVocabulary` on a miss - the entry
    /// point `bind` uses, so a `VocabularyHandle`'s one sort is never repeated at first-bind time.
    pub(crate) fn get_or_build_byte_from_prepared(
        &self,
        prepared: &PreparedVocabulary,
        fingerprint: VocabFingerprint,
    ) -> Result<Arc<VocabTrie>, CompileError> {
        self.get_or_build_byte_impl(BuildSource::Prepared(prepared), fingerprint)
    }

    /// The shared cache/single-flight/build-gate logic; `source` only decides HOW a miss builds.
    fn get_or_build_byte_impl(
        &self,
        source: BuildSource<'_>,
        fingerprint: VocabFingerprint,
    ) -> Result<Arc<VocabTrie>, CompileError> {
        let slot = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.map.contains_key(&fingerprint) {
                let now = state.touch(fingerprint);
                let entry = state
                    .map
                    .get_mut(&fingerprint)
                    .expect("just confirmed present under the same lock hold");
                entry.last_used = now;
                let trie = entry.trie.clone();
                self.stats.lock().unwrap_or_else(|p| p.into_inner()).hits += 1;
                return Ok(trie);
            }
            if let Some(slot) = state.in_flight.get(&fingerprint).cloned() {
                drop(state);
                let result = {
                    let mut r = slot.result.lock().unwrap_or_else(|p| p.into_inner());
                    while r.is_none() {
                        r = slot.ready.wait(r).unwrap_or_else(|p| p.into_inner());
                    }
                    r.clone().expect("slot is ready")
                };
                if result.is_ok() {
                    self.stats.lock().unwrap_or_else(|p| p.into_inner()).hits += 1;
                }
                return result;
            }
            let slot = Arc::new(BuildSlot {
                result: Mutex::new(None),
                ready: Condvar::new(),
            });
            state.in_flight.insert(fingerprint, slot.clone());
            slot
        };

        let mut guard = SlotGuard {
            cache: self,
            fp: fingerprint,
            slot: slot.clone(),
            armed: true,
        };
        let built: Result<Arc<VocabTrie>, CompileError> = (|| {
            let est = match source {
                BuildSource::Vocab(v) => estimate_transient_build_bytes(v),
                BuildSource::Prepared(p) => estimate_transient_build_bytes_prepared(p),
            };
            let _permit = self.gate.acquire(est)?;
            #[cfg(test)]
            if DELAY_BUILDS.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            #[cfg(test)]
            if PANIC_ON_BUILD.with(|p| p.get()) {
                panic!("injected test panic mid-build");
            }
            let trie = match source {
                BuildSource::Vocab(v) => VocabTrie::build_byte_with_fingerprint(v, fingerprint)?,
                BuildSource::Prepared(p) => VocabTrie::build_byte_from_prepared(p, fingerprint)?,
            };
            Ok(Arc::new(trie))
        })();

        {
            let mut r = slot.result.lock().unwrap_or_else(|p| p.into_inner());
            *r = Some(built.clone());
            slot.ready.notify_all();
        }
        {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            state.in_flight.remove(&fingerprint);
            if let Ok(ref arc) = built {
                let mut stats = self.stats.lock().unwrap_or_else(|p| p.into_inner());
                stats.misses += 1;
                let entry_bytes = arc.heap_bytes().saturating_add(CACHE_ENTRY_OVERHEAD);
                if entry_bytes > self.budget {
                    stats.oversized_bypasses += 1; // one trie bigger than the whole budget: never cache
                } else {
                    state.evict_until_fits(entry_bytes, self.budget, &mut stats);
                    let now = state.touch(fingerprint);
                    state.bytes += entry_bytes;
                    state.map.insert(
                        fingerprint,
                        CacheEntry {
                            trie: arc.clone(),
                            bytes: entry_bytes,
                            last_used: now,
                        },
                    );
                }
            }
        }
        guard.armed = false; // published + cleared cleanly; the guard has nothing to repair
        built
    }

    /// The byte-trie for `handle`'s vocabulary, wrapped as an opaque `BoundByteTrie` whose
    /// fingerprint is guaranteed (by construction, not by caller honesty) to be `handle`'s own -
    pub fn bind(&self, handle: &VocabularyHandle) -> Result<BoundByteTrie, CompileError> {
        let trie = self.get_or_build_byte_from_prepared(handle.prepared(), handle.fingerprint)?;
        Ok(BoundByteTrie {
            trie,
            fingerprint: handle.fingerprint,
        })
    }

    /// Drops every retained trie. Also clears the LRU touch-log, so repeated fill/clear cycles
    /// cannot accumulate stale log records indefinitely. Traffic counters (`stats()`) are NOT reset -
    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.map.clear();
        state.bytes = 0;
        state.lru.clear();
        state.tick = 0;
    }

    /// The number of retained tries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .map
            .len()
    }

    /// Whether the cache retains no trie.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retained trie bytes. Bounds the CACHE only - a trie an active `PyIndex`/`CompiledArtifact`
    /// holds its own `Arc` to stays alive via that reference even if evicted or never cached here.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).bytes
    }

    /// `(hits, misses, evictions, oversized_bypasses)` since creation - a lifetime traffic total,
    /// NOT reset by `clear()` (which only drops currently-retained content). Single-flight caps
    #[must_use]
    pub fn stats(&self) -> (u64, u64, u64, u64) {
        let s = self.stats.lock().unwrap_or_else(|p| p.into_inner());
        (s.hits, s.misses, s.evictions, s.oversized_bypasses)
    }
}

/// An immutable, opaque handle owning ONE canonical vocabulary representation and its content
/// fingerprint - always computed here, never caller-suppliable.
#[derive(Clone, Debug)]
pub struct VocabularyHandle {
    canonical: Arc<CanonicalVocabulary>,
    fingerprint: VocabFingerprint,
    token_slices: Arc<OnceLock<Option<Arc<TokenSliceCatalog>>>>,
}

/// One `(&[u8], &[u32])` sort-scratch entry: two fat pointers, structurally fatter than the packed
/// path's one `u32` sort index per token, so it is its own term, not folded into that estimate.
const PAIR_SCRATCH_BYTES_PER_TOKEN: usize = 4 * std::mem::size_of::<usize>();

/// One `(Vec<u8>, Vec<u32>)` outer-descriptor entry: two 3-word `Vec` headers (ptr/len/cap each).
const TUPLE_DESCRIPTOR_BYTES_PER_TOKEN: usize = 2 * 3 * std::mem::size_of::<usize>();

/// Per-token `FxHashMap<Vec<u8>, Vec<u32>>` slot cost: the key+value payload plus hashbrown
/// control/load-factor slack. An over-estimate, not an exact bound on the opaque allocation.
const HASHMAP_SLOT_BYTES_PER_TOKEN: usize = std::mem::size_of::<(Vec<u8>, Vec<u32>)>() + 16;

/// A conservative estimated peak (not a structural bound - hashbrown's real allocation is opaque)
/// for `VocabularyHandle::new`'s tuple/`HashMap` construction: max of copy/map-build/canonical-build.
#[must_use]
pub(crate) fn handle_construction_peak_bytes_from_totals(
    total_bytes: usize,
    total_ids: usize,
    token_count: usize,
) -> usize {
    let ids_bytes = total_ids.saturating_mul(4);
    let descriptors = token_count.saturating_mul(TUPLE_DESCRIPTOR_BYTES_PER_TOKEN);
    let hashmap_buckets = token_count.saturating_mul(HASHMAP_SLOT_BYTES_PER_TOKEN);

    let copy_peak = descriptors
        .saturating_add(total_bytes)
        .saturating_add(ids_bytes);
    let map_peak = copy_peak.saturating_add(hashmap_buckets);

    let source_map_retained = total_bytes
        .saturating_add(ids_bytes)
        .saturating_add(hashmap_buckets);
    let canonical_build = projected_peak_bytes(total_bytes, total_ids, token_count)
        .saturating_add(token_count.saturating_mul(PAIR_SCRATCH_BYTES_PER_TOKEN));
    let canonical_peak = source_map_retained.saturating_add(canonical_build);

    copy_peak.max(map_peak).max(canonical_peak)
}

/// `handle_construction_peak_bytes_from_totals`, from an already-built `vocab`'s own totals.
fn handle_construction_peak_bytes(vocab: &Vocabulary) -> usize {
    let total_bytes = vocab
        .tokens()
        .keys()
        .try_fold(0usize, |acc, k| acc.checked_add(k.len()))
        .unwrap_or(usize::MAX);
    let total_ids = vocab
        .tokens()
        .values()
        .try_fold(0usize, |acc, v| acc.checked_add(v.len()))
        .unwrap_or(usize::MAX);
    handle_construction_peak_bytes_from_totals(total_bytes, total_ids, vocab.tokens().len())
}

impl VocabularyHandle {
    /// Wraps `vocab`, building the canonical representation once, admitted against the shared
    /// build-memory gate so concurrent large constructions cannot each exceed the process budget.
    pub fn new(vocab: Arc<Vocabulary>) -> Result<Self, crate::error::CompileError> {
        Self::new_with_logits_vocab_size(vocab, None)
    }

    /// `new`, with an explicit mask width instead of the inferred one. Rejected if `Some` and
    /// smaller than the largest token/EOS id actually present.
    pub fn new_with_logits_vocab_size(
        vocab: Arc<Vocabulary>,
        logits_vocab_size: Option<usize>,
    ) -> Result<Self, crate::error::CompileError> {
        let permit = acquire_vocab_build_bytes(handle_construction_peak_bytes(&vocab))?;
        Self::new_admitted_with_logits_vocab_size(vocab, logits_vocab_size, &permit)
    }

    /// `new`, but the caller already holds a `PermitGuard` for this whole operation - re-verified
    /// against this route's own estimate first, so an undersized permit is rejected, not trusted.
    pub fn new_admitted(
        vocab: Arc<Vocabulary>,
        permit: &PermitGuard<'_>,
    ) -> Result<Self, crate::error::CompileError> {
        Self::new_admitted_with_logits_vocab_size(vocab, None, permit)
    }

    /// `new_admitted`, with the same explicit-mask-width contract as `new_with_logits_vocab_size`.
    pub fn new_admitted_with_logits_vocab_size(
        vocab: Arc<Vocabulary>,
        logits_vocab_size: Option<usize>,
        permit: &PermitGuard<'_>,
    ) -> Result<Self, crate::error::CompileError> {
        permit.ensure_covers(handle_construction_peak_bytes(&vocab))?;
        let canonical = Arc::new(CanonicalVocabulary::from_vocabulary_with_logits_vocab_size(
            &vocab,
            logits_vocab_size,
        )?);
        let fingerprint = canonical.fingerprint();
        Ok(Self {
            canonical,
            fingerprint,
            token_slices: Arc::new(OnceLock::new()),
        })
    }

    /// Builds directly from four packed CSR buffers (the tuple-free fast path): no intermediate
    /// `Vocabulary`/`HashMap`. Same validation as `new`, admitted before any construction runs.
    pub fn from_packed(
        token_bytes: &[u8],
        byte_offsets: &[u32],
        token_ids: &[u32],
        id_offsets: &[u32],
        eos_token_id: u32,
    ) -> Result<Self, crate::error::CompileError> {
        Self::from_packed_with_logits_vocab_size(
            token_bytes,
            byte_offsets,
            token_ids,
            id_offsets,
            eos_token_id,
            None,
        )
    }

    /// `from_packed`, with the same explicit-mask-width contract as `new_with_logits_vocab_size`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_packed_with_logits_vocab_size(
        token_bytes: &[u8],
        byte_offsets: &[u32],
        token_ids: &[u32],
        id_offsets: &[u32],
        eos_token_id: u32,
        logits_vocab_size: Option<usize>,
    ) -> Result<Self, crate::error::CompileError> {
        let token_count = byte_offsets.len().saturating_sub(1);
        let peak = projected_peak_bytes(token_bytes.len(), token_ids.len(), token_count);
        let permit = acquire_vocab_build_bytes(peak)?;
        Self::from_packed_admitted_with_logits_vocab_size(
            token_bytes,
            byte_offsets,
            token_ids,
            id_offsets,
            eos_token_id,
            logits_vocab_size,
            &permit,
        )
    }

    /// `from_packed`, but the caller already holds a `PermitGuard` covering the whole ingestion -
    /// re-verified the same way as `new_admitted`, from the actual slices.
    pub fn from_packed_admitted(
        token_bytes: &[u8],
        byte_offsets: &[u32],
        token_ids: &[u32],
        id_offsets: &[u32],
        eos_token_id: u32,
        permit: &PermitGuard<'_>,
    ) -> Result<Self, crate::error::CompileError> {
        Self::from_packed_admitted_with_logits_vocab_size(
            token_bytes,
            byte_offsets,
            token_ids,
            id_offsets,
            eos_token_id,
            None,
            permit,
        )
    }

    /// `from_packed_admitted`, with the same explicit-mask-width contract as
    /// `new_with_logits_vocab_size`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_packed_admitted_with_logits_vocab_size(
        token_bytes: &[u8],
        byte_offsets: &[u32],
        token_ids: &[u32],
        id_offsets: &[u32],
        eos_token_id: u32,
        logits_vocab_size: Option<usize>,
        permit: &PermitGuard<'_>,
    ) -> Result<Self, crate::error::CompileError> {
        let token_count = byte_offsets.len().saturating_sub(1);
        let required = projected_peak_bytes(token_bytes.len(), token_ids.len(), token_count);
        permit.ensure_covers(required)?;
        let canonical = Arc::new(CanonicalVocabulary::from_packed_with_logits_vocab_size(
            token_bytes,
            byte_offsets,
            token_ids,
            id_offsets,
            eos_token_id,
            logits_vocab_size,
        )?);
        let fingerprint = canonical.fingerprint();
        Ok(Self {
            canonical,
            fingerprint,
            token_slices: Arc::new(OnceLock::new()),
        })
    }

    /// Builds directly from a DENSE, ID-ORDERED tokenizer decode: `decoded[i]` (via `byte_offsets`)
    /// is token id `i`'s bytes, `present[i]` says whether slot `i` exists. No id list, no
    pub fn from_dense_id_ordered(
        token_bytes: &[u8],
        byte_offsets: &[u32],
        present: &[bool],
        eos_token_id: u32,
    ) -> Result<Self, crate::error::CompileError> {
        Self::from_dense_id_ordered_with_logits_vocab_size(
            token_bytes,
            byte_offsets,
            present,
            eos_token_id,
            None,
        )
    }

    /// `from_dense_id_ordered`, with the same explicit-mask-width contract as
    /// `new_with_logits_vocab_size`.
    pub fn from_dense_id_ordered_with_logits_vocab_size(
        token_bytes: &[u8],
        byte_offsets: &[u32],
        present: &[bool],
        eos_token_id: u32,
        logits_vocab_size: Option<usize>,
    ) -> Result<Self, crate::error::CompileError> {
        let present_count = present.iter().filter(|&&p| p).count();
        let peak = projected_peak_bytes(token_bytes.len(), present_count, present_count);
        let permit = acquire_vocab_build_bytes(peak)?;
        Self::from_dense_id_ordered_admitted_with_logits_vocab_size(
            token_bytes,
            byte_offsets,
            present,
            eos_token_id,
            logits_vocab_size,
            &permit,
        )
    }

    /// `from_dense_id_ordered`, but the caller already holds a `PermitGuard` covering the whole
    /// ingestion - re-verified from the actual slices, same as `from_packed_admitted`.
    pub fn from_dense_id_ordered_admitted(
        token_bytes: &[u8],
        byte_offsets: &[u32],
        present: &[bool],
        eos_token_id: u32,
        permit: &PermitGuard<'_>,
    ) -> Result<Self, crate::error::CompileError> {
        Self::from_dense_id_ordered_admitted_with_logits_vocab_size(
            token_bytes,
            byte_offsets,
            present,
            eos_token_id,
            None,
            permit,
        )
    }

    /// `from_dense_id_ordered_admitted`, with the same explicit-mask-width contract as
    /// `new_with_logits_vocab_size`.
    pub fn from_dense_id_ordered_admitted_with_logits_vocab_size(
        token_bytes: &[u8],
        byte_offsets: &[u32],
        present: &[bool],
        eos_token_id: u32,
        logits_vocab_size: Option<usize>,
        permit: &PermitGuard<'_>,
    ) -> Result<Self, crate::error::CompileError> {
        let present_count = present.iter().filter(|&&p| p).count();
        let required = projected_peak_bytes(token_bytes.len(), present_count, present_count);
        permit.ensure_covers(required)?;
        let canonical = Arc::new(
            CanonicalVocabulary::from_dense_id_ordered_with_logits_vocab_size(
                token_bytes,
                byte_offsets,
                present,
                eos_token_id,
                logits_vocab_size,
            )?,
        );
        let fingerprint = canonical.fingerprint();
        Ok(Self {
            canonical,
            fingerprint,
            token_slices: Arc::new(OnceLock::new()),
        })
    }

    /// The one canonical vocabulary representation.
    #[must_use]
    pub(crate) fn canonical(&self) -> &Arc<CanonicalVocabulary> {
        &self.canonical
    }

    /// The vocabulary's content fingerprint, computed once at construction.
    #[must_use]
    pub fn fingerprint(&self) -> VocabFingerprint {
        self.fingerprint
    }

    /// The mask width (`max(largest token id, EOS id) + 1`, or the explicit `logits_vocab_size`
    /// this handle was built with) - the number of bits one mask row covers.
    #[must_use]
    pub fn mask_vocab_size(&self) -> usize {
        self.canonical.mask_vocab_size()
    }

    /// The EOS token id (a side value, never a mask bit).
    #[must_use]
    pub fn eos_token_id(&self) -> u32 {
        self.canonical.eos_token_id()
    }

    /// The number of stored ordinary token ids plus one for EOS (NOT the distinct byte-string
    /// count and NOT the mask width).
    #[must_use]
    pub fn len(&self) -> usize {
        self.canonical.len()
    }

    /// True when the token map has no records (matches `Vocabulary::is_empty()`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.canonical.is_empty()
    }

    /// `id`'s byte string, or `None` if `id` names no ordinary token (including EOS).
    #[must_use]
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.canonical.token_bytes(id)
    }

    /// Every record as `(bytes, ids)`, byte-sorted (for a caller with no precomputed mask table).
    pub fn iter_records(&self) -> impl Iterator<Item = (&[u8], &[u32])> {
        self.canonical.iter()
    }

    /// The canonical form's shared byte-sorted view, reused by `TrieCache::bind` so a first trie
    /// build never re-sorts the token map fingerprinting already sorted.
    pub(crate) fn prepared(&self) -> &PreparedVocabulary {
        self.canonical.prepared()
    }

    pub(crate) fn token_slices(&self) -> Option<&Arc<TokenSliceCatalog>> {
        self.token_slices
            .get_or_init(|| TokenSliceCatalog::try_build(&self.canonical).ok().flatten())
            .as_ref()
    }

    /// Bytes retained by the optional vocabulary-derived token-slice catalogue.
    #[must_use]
    pub fn token_slice_retained_bytes(&self) -> usize {
        self.token_slices()
            .map_or(0, |catalog| catalog.retained_bytes())
    }
}

/// A byte-trie proven paired with a specific `VocabularyHandle`. Private fields: the only public
/// constructor is `TrieCache::bind`, which always derives the fingerprint from the SAME handle the
#[derive(Clone, Debug)]
pub struct BoundByteTrie {
    trie: Arc<VocabTrie>,
    fingerprint: VocabFingerprint,
}

impl BoundByteTrie {
    pub(crate) fn trie(&self) -> &VocabTrie {
        &self.trie
    }

    /// The shared trie `Arc`, for the lazy bind path to retain without rebuilding.
    pub(crate) fn trie_arc(&self) -> Arc<VocabTrie> {
        self.trie.clone()
    }

    pub(crate) fn fingerprint(&self) -> VocabFingerprint {
        self.fingerprint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorCode;
    use crate::vocab::build_vocabulary;
    use rustc_hash::FxHashMap as Map;

    fn vocab(pairs: &[(&[u8], u32)], eos: u32) -> Vocabulary {
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        for &(bytes, id) in pairs {
            map.insert(bytes.to_vec(), vec![id]);
        }
        build_vocabulary(eos, map).expect("vocab")
    }

    #[test]
    fn fingerprint_is_injective_over_marker_byte_id_lists() {
        let fp = |pairs: &[(&[u8], &[u32])], eos: u32| {
            let mut m: Map<Vec<u8>, Vec<u32>> = Map::default();
            for &(b, ids) in pairs {
                m.insert(b.to_vec(), ids.to_vec());
            }
            VocabFingerprint::of(&build_vocabulary(eos, m).expect("vocab")).expect("fingerprint")
        };
        let one_token = fp(&[(b"a", &[0x0000_00fe, 0x0000_00ff])], 9);
        let two_tokens = fp(&[(b"a", &[0x0000_00fe]), (b"b", &[0x0000_00ff])], 9);
        assert_ne!(
            one_token, two_tokens,
            "id-list grouping must change the fingerprint"
        );
        let reordered = fp(&[(b"a", &[0x0000_00ff, 0x0000_00fe])], 9);
        assert_eq!(
            one_token, reordered,
            "id order within a token must not matter"
        );
        assert_ne!(
            one_token,
            fp(&[(b"a", &[0x0000_00fe, 0x0000_00ff])], 10),
            "eos must change the fingerprint"
        );
    }

    #[test]
    fn same_vocabulary_returns_the_same_arc() {
        let cache = TrieCache::new();
        let v = vocab(&[(b"a", 0), (b"b", 1)], 9);
        let first = cache.get_or_build_byte(&v).unwrap();
        let second = cache.get_or_build_byte(&v).unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "a cache hit must reuse the Arc"
        );
    }

    #[test]
    fn prevalidated_lookup_agrees_with_the_generic_one() {
        let cache = TrieCache::new();
        let v = vocab(&[(b"a", 0), (b"b", 1)], 9);
        let generic = cache.get_or_build_byte(&v).unwrap();
        let fp = VocabFingerprint::of(&v).unwrap();
        let fast = cache.get_or_build_byte_prevalidated(&v, fp).unwrap();
        assert!(
            Arc::ptr_eq(&generic, &fast),
            "same vocabulary, same cached Arc"
        );
    }

    #[test]
    fn a_handles_fingerprint_matches_the_direct_unprepared_computation() {
        let cases = [
            vocab(&[(b"a", 0), (b"b", 1)], 9),
            vocab(&[(b"cat", 17), (b"car", 42)], 1000),
            vocab(&[(&[0xff, 0xfe], 3), (&[0x00], 4)], 9999),
        ];
        for v in cases {
            let direct = VocabFingerprint::of(&v).unwrap();
            let via_handle = VocabularyHandle::new(Arc::new(v)).unwrap().fingerprint();
            assert_eq!(
                direct, via_handle,
                "handle fingerprint must match the direct one"
            );
        }
    }

    #[test]
    fn dense_id_ordered_handle_matches_the_tuple_handle_on_equivalent_content() {
        let dense_v = vocab(&[(b"cat", 0), (b"car", 1), (b"c", 3)], 9999);
        let via_new = VocabularyHandle::new(Arc::new(dense_v)).unwrap();

        let token_bytes = b"catcarc";
        let byte_offsets = [0u32, 3, 6, 6, 7]; // cat, car, (absent), c
        let present = [true, true, false, true];
        let via_dense =
            VocabularyHandle::from_dense_id_ordered(token_bytes, &byte_offsets, &present, 9999)
                .unwrap();

        assert_eq!(via_new.fingerprint(), via_dense.fingerprint());
        assert_eq!(via_new.mask_vocab_size(), via_dense.mask_vocab_size());
    }

    #[test]
    fn dense_id_ordered_handle_rejects_eos_present_as_an_ordinary_slot() {
        let token_bytes = b"ab";
        let byte_offsets = [0u32, 1, 2];
        let present = [true, true];
        let err = VocabularyHandle::from_dense_id_ordered(token_bytes, &byte_offsets, &present, 1)
            .expect_err("eos id 1 collides with the present 'b' slot");
        assert_eq!(err.code, ErrorCode::MalformedTokenizer);
    }

    #[test]
    fn a_bind_hit_does_not_rebuild_and_only_the_miss_counts() {
        let cache = TrieCache::new();
        let handle = VocabularyHandle::new(Arc::new(vocab(&[(b"a", 0), (b"b", 1)], 9))).unwrap();
        let (h0, m0, _, _) = cache.stats();
        let _first = cache.bind(&handle).unwrap(); // miss: builds once
        let (_h1, m1, _, _) = cache.stats();
        let _second = cache.bind(&handle).unwrap(); // hit: no rebuild, no rehash
        let (h2, m2, _, _) = cache.stats();
        assert_eq!(m1, m0 + 1, "the first bind is the only miss");
        assert_eq!(m2, m1, "a hit must not add a miss");
        assert_eq!(h2, h0 + 1, "the second bind is a hit");
        assert_eq!(cache.len(), 1, "one trie retained for one vocabulary");
    }

    #[test]
    fn different_vocabularies_are_different_keys() {
        let cache = TrieCache::new();
        let a = vocab(&[(b"a", 0)], 9);
        let b = vocab(&[(b"b", 0)], 9);
        let ta = cache.get_or_build_byte(&a).unwrap();
        let tb = cache.get_or_build_byte(&b).unwrap();
        assert!(
            !Arc::ptr_eq(&ta, &tb),
            "distinct token maps must not share a trie"
        );
        assert_ne!(
            VocabFingerprint::of(&a).unwrap(),
            VocabFingerprint::of(&b).unwrap()
        );
    }

    #[test]
    fn fingerprint_ignores_hashmap_order_but_tracks_eos() {
        let a = vocab(&[(b"a", 0), (b"b", 1)], 9);
        let b = vocab(&[(b"b", 1), (b"a", 0)], 9);
        assert_eq!(
            VocabFingerprint::of(&a).unwrap(),
            VocabFingerprint::of(&b).unwrap()
        );
        let eos_changed = vocab(&[(b"a", 0), (b"b", 1)], 8);
        assert_ne!(
            VocabFingerprint::of(&a).unwrap(),
            VocabFingerprint::of(&eos_changed).unwrap()
        );
    }

    #[test]
    fn boundary_ambiguity_is_separated_by_domain_markers() {
        let one = vocab(&[(b"ab", 0)], 9);
        let two = vocab(&[(b"a", 0), (b"b", 1)], 9);
        assert_ne!(
            VocabFingerprint::of(&one).unwrap(),
            VocabFingerprint::of(&two).unwrap()
        );
    }

    #[test]
    fn a_zero_budget_builds_but_retains_nothing() {
        let cache = TrieCache::with_budget(0);
        let v = vocab(&[(b"a", 0)], 9);
        let built = cache.get_or_build_byte(&v).expect("still built");
        assert_eq!(built.node_count(), 2);
        assert!(cache.is_empty(), "a zero budget retains no trie");
    }

    #[test]
    fn a_budget_at_one_trie_retains_one_not_two() {
        let a = vocab(&[(b"a", 0)], 9);
        let budget = VocabTrie::build_byte(&a).unwrap().heap_bytes() + CACHE_ENTRY_OVERHEAD;
        let cache = TrieCache::with_budget(budget);
        cache.get_or_build_byte(&a).unwrap(); // fits exactly
        cache.get_or_build_byte(&vocab(&[(b"bb", 1)], 9)).unwrap(); // does not fit
        assert_eq!(cache.len(), 1, "the budget retains exactly one trie");
    }

    #[test]
    fn entry_overhead_is_charged_so_many_tiny_tries_cannot_exceed_the_budget() {
        let payload = VocabTrie::build_byte(&vocab(&[(b"a", 0)], 1000))
            .unwrap()
            .heap_bytes();
        let cache = TrieCache::with_budget(payload * 10);
        for i in 0..10u32 {
            cache
                .get_or_build_byte(&vocab(&[(&[b'a' + i as u8], i)], 1000))
                .unwrap();
        }
        assert!(
            cache.len() < 10,
            "charging per-entry overhead must retain fewer than a payload-only budget would ({})",
            cache.len()
        );
        assert!(
            cache.bytes() <= payload * 10,
            "retained bytes stay within budget"
        );
    }

    #[test]
    fn single_flight_builds_a_cold_trie_exactly_once_under_concurrency() {
        use std::thread;
        let cache = Arc::new(TrieCache::new());
        let v = Arc::new(vocab(
            &[(b"a", 0), (b"bb", 1), (b"ccc", 2), (b"dddd", 3)],
            1000,
        ));
        let ptrs: Vec<usize> = (0..32)
            .map(|_| {
                let c = cache.clone();
                let v = v.clone();
                thread::spawn(move || Arc::as_ptr(&c.get_or_build_byte(&v).unwrap()) as usize)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        assert!(
            ptrs.iter().all(|&p| p == ptrs[0]),
            "all callers share one trie"
        );
        let (hits, misses, _, _) = cache.stats();
        assert_eq!(misses, 1, "single-flight builds the cold trie exactly once");
        assert_eq!(hits, 31, "the other 31 callers hit the one build");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn single_flight_shares_one_build_even_when_the_budget_retains_nothing() {
        use std::sync::Barrier;
        use std::thread;
        let _serialize = DELAY_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let cache = Arc::new(TrieCache::with_budget(0));
        let v = Arc::new(vocab(
            &[(b"a", 0), (b"bb", 1), (b"ccc", 2), (b"dddd", 3)],
            1000,
        ));
        let barrier = Arc::new(Barrier::new(32));
        struct ResetDelay;
        impl Drop for ResetDelay {
            fn drop(&mut self) {
                DELAY_BUILDS.store(false, Ordering::Relaxed);
            }
        }
        DELAY_BUILDS.store(true, Ordering::Relaxed);
        let _reset = ResetDelay;
        let ptrs: Vec<usize> = (0..32)
            .map(|_| {
                let c = cache.clone();
                let v = v.clone();
                let b = barrier.clone();
                thread::spawn(move || {
                    b.wait();
                    Arc::as_ptr(&c.get_or_build_byte(&v).unwrap()) as usize
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        assert!(
            ptrs.iter().all(|&p| p == ptrs[0]),
            "all callers share the one build"
        );
        let (_h, misses, _e, _o) = cache.stats();
        assert_eq!(misses, 1, "exactly one build even with nothing retained");
        assert!(cache.is_empty(), "a zero budget retains no trie");
    }

    #[test]
    fn concurrent_distinct_builds_are_bounded_by_the_gate() {
        use std::thread;
        let cache = Arc::new(TrieCache::with_build_limit(1 << 28, 3));
        let handles: Vec<_> = (0..24u32)
            .map(|i| {
                let c = cache.clone();
                thread::spawn(move || {
                    c.get_or_build_byte(&vocab(
                        &[(&[b'a', i as u8], 0), (&[b'b', i as u8, 0], 1)],
                        1000,
                    ))
                    .unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert!(
            cache.peak_concurrent_builds() <= 3,
            "simultaneous distinct builds must stay within the gate ({})",
            cache.peak_concurrent_builds()
        );
        assert_eq!(cache.len(), 24, "all 24 distinct tries built and retained");
    }

    #[test]
    fn concurrent_builds_are_bounded_by_the_transient_byte_budget() {
        use std::sync::Barrier;
        use std::thread;
        let _serialize = DELAY_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let sample = vocab(&[(b"xa", 0), (b"xb", 1), (b"xc", 2)], 1000);
        let est = estimate_transient_build_bytes(&sample);
        let byte_budget = est * 2;
        let cache = Arc::new(TrieCache::with_build_gate(1 << 28, 8, byte_budget));
        let barrier = Arc::new(Barrier::new(8));
        DELAY_BUILDS.store(true, Ordering::Relaxed);
        struct ResetDelay;
        impl Drop for ResetDelay {
            fn drop(&mut self) {
                DELAY_BUILDS.store(false, Ordering::Relaxed);
            }
        }
        let _reset = ResetDelay;
        let handles: Vec<_> = (0..8u32)
            .map(|i| {
                let c = cache.clone();
                let b = barrier.clone();
                thread::spawn(move || {
                    b.wait();
                    c.get_or_build_byte(&vocab(
                        &[
                            (&[b'x', i as u8], 0),
                            (&[b'y', i as u8], 1),
                            (&[b'z', i as u8], 2),
                        ],
                        1000,
                    ))
                    .unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(
            cache.peak_concurrent_builds(),
            2,
            "the byte budget, not the count cap of 8, must be the active constraint"
        );
        assert!(cache.peak_concurrent_build_bytes() <= byte_budget);
        assert_eq!(
            cache.len(),
            8,
            "all 8 distinct tries eventually built and retained"
        );
    }

    #[test]
    fn an_oversized_build_estimate_is_rejected_and_shared_with_every_waiter() {
        use std::thread;
        let big = vocab(
            &[
                (b"big-token-one", 0),
                (b"big-token-two", 1),
                (b"big-token-three", 2),
            ],
            1000,
        );
        let est = estimate_transient_build_bytes(&big);
        let cache = Arc::new(TrieCache::with_build_gate(1 << 28, 8, est - 1));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let c = cache.clone();
                let v = big.clone();
                thread::spawn(move || c.get_or_build_byte(&v).map_err(|e| e.code))
            })
            .collect();
        for h in handles {
            let result = h.join().unwrap();
            assert_eq!(
                result.unwrap_err(),
                ErrorCode::InternalLimitExceeded,
                "an over-budget build must be rejected, not silently clamped and run"
            );
        }
        assert!(cache.is_empty(), "a rejected build is never cached");
    }

    #[test]
    fn a_hot_vocabulary_becomes_and_stays_cached_after_a_junk_burst_fills_the_cache() {
        let sample = vocab(&[(&[b'j', 0, 0], 0)], 1000);
        let entry_bytes =
            VocabTrie::build_byte(&sample).unwrap().heap_bytes() + CACHE_ENTRY_OVERHEAD;
        let cache = TrieCache::with_budget(entry_bytes * 3);
        for i in 0..50u32 {
            let junk = vocab(&[(&[b'j', (i % 251) as u8, (i / 251) as u8], i)], 1000);
            cache.get_or_build_byte(&junk).unwrap();
        }
        let (_, _, evictions_from_junk, _) = cache.stats();
        assert!(
            evictions_from_junk > 0,
            "the junk burst overflowed the budget"
        );

        let hot = vocab(&[(b"hot-token-aaaa", 0), (b"hot-token-bbbb", 1)], 1000);
        let (_, misses0, _, _) = cache.stats();
        cache.get_or_build_byte(&hot).unwrap();
        let (_, misses1, _, _) = cache.stats();
        assert_eq!(
            misses1,
            misses0 + 1,
            "the first access to a new hot vocab is a miss"
        );

        cache.get_or_build_byte(&hot).unwrap();
        let (_, misses2, _, _) = cache.stats();
        assert_eq!(misses2, misses1, "the second access hits the cached entry");
    }

    #[test]
    fn old_bypass_only_policy_would_fail_the_hot_vocabulary_regression() {
        let cache = TrieCache::with_budget(0);
        let hot = vocab(&[(b"hot-token-aaaa", 0), (b"hot-token-bbbb", 1)], 1000);
        cache.get_or_build_byte(&hot).unwrap();
        let (_, misses1, _, _) = cache.stats();
        cache.get_or_build_byte(&hot).unwrap();
        let (_, misses2, _, _) = cache.stats();
        assert_eq!(
            misses2,
            misses1 + 1,
            "a bypass-only cache never turns a repeat into a hit"
        );
    }

    #[test]
    fn concurrent_native_threads_share_one_trie_and_never_corrupt_the_cache() {
        use std::thread;
        let cache = Arc::new(TrieCache::new());
        let same = Arc::new(vocab(&[(b"a", 0), (b"bb", 1), (b"ccc", 2)], 1000));
        let mut handles = Vec::new();
        for _ in 0..16 {
            let c = cache.clone();
            let v = same.clone();
            handles.push(thread::spawn(move || {
                c.get_or_build_byte(&v).map(|t| Arc::as_ptr(&t) as usize)
            }));
        }
        for i in 0..16u32 {
            let c = cache.clone();
            handles.push(thread::spawn(move || {
                c.get_or_build_byte(&vocab(&[(&[b'z', i as u8], i)], 1000))
                    .map(|t| Arc::as_ptr(&t) as usize)
            }));
        }
        let ptrs: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().unwrap().unwrap())
            .collect();
        let shared = Arc::as_ptr(&cache.get_or_build_byte(&same).unwrap()) as usize;
        for p in &ptrs[..16] {
            assert_eq!(*p, shared, "same vocab must resolve to one cached trie");
        }
        assert_eq!(
            cache.len(),
            17,
            "one shared trie plus 16 distinct tries retained"
        );
    }

    #[test]
    fn clear_drops_retained_tries() {
        let cache = TrieCache::new();
        cache.get_or_build_byte(&vocab(&[(b"a", 0)], 9)).unwrap();
        cache.get_or_build_byte(&vocab(&[(b"b", 0)], 9)).unwrap();
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn clear_also_drops_the_lru_log_so_repeated_fill_clear_does_not_leak() {
        let cache = TrieCache::new();
        for cycle in 0..50u32 {
            cache
                .get_or_build_byte(&vocab(&[(&[cycle as u8], 0)], 9))
                .unwrap();
            cache.clear();
        }
        let lru_len = cache
            .state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .lru
            .len();
        assert_eq!(
            lru_len, 0,
            "clear() must drop the LRU log, not just the map"
        );
    }

    #[test]
    fn stats_are_a_lifetime_total_not_reset_by_clear() {
        let cache = TrieCache::new();
        cache.get_or_build_byte(&vocab(&[(b"a", 0)], 9)).unwrap();
        cache.get_or_build_byte(&vocab(&[(b"a", 0)], 9)).unwrap(); // hit
        let (hits, misses, ..) = cache.stats();
        assert_eq!((hits, misses), (1, 1));
        cache.clear();
        let (hits_after, misses_after, ..) = cache.stats();
        assert_eq!(
            (hits_after, misses_after),
            (1, 1),
            "clear() must not reset traffic counters"
        );
    }

    #[test]
    fn an_evicted_trie_stays_valid_and_correct_while_a_caller_still_holds_its_arc() {
        let a = vocab(&[(b"a", 0), (b"ab", 1)], 9);
        let budget = VocabTrie::build_byte(&a).unwrap().heap_bytes() + CACHE_ENTRY_OVERHEAD;
        let cache = TrieCache::with_budget(budget);

        let held = cache.get_or_build_byte(&a).unwrap();
        assert_eq!(cache.len(), 1);

        cache.get_or_build_byte(&vocab(&[(b"z", 0)], 9)).unwrap();
        assert_eq!(cache.len(), 1, "the budget still retains exactly one entry");

        assert_eq!(
            held.node_count(),
            3,
            "evicted trie's structure is untouched"
        );

        let rebuilt = cache.get_or_build_byte(&a).unwrap();
        assert_eq!(
            rebuilt.node_count(),
            held.node_count(),
            "a rebuild after eviction reproduces the identical structure"
        );
    }

    #[test]
    fn a_panicking_build_releases_its_permit_and_slot_so_the_next_build_is_not_stuck() {
        let cache = TrieCache::with_build_limit(usize::MAX, 1); // only one concurrent build allowed
        let v = vocab(&[(b"a", 0)], 9);

        PANIC_ON_BUILD.with(|p| p.set(true));
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cache.get_or_build_byte(&v)));
        PANIC_ON_BUILD.with(|p| p.set(false));
        assert!(result.is_err(), "the injected panic must actually unwind");

        let recovered = cache
            .get_or_build_byte(&v)
            .expect("a fresh build must succeed");
        assert_eq!(recovered.node_count(), 2);
    }

    #[test]
    fn prepared_transient_estimate_charges_many_ids_under_one_record_like_the_vocab_estimate() {
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        map.insert(b"a".to_vec(), (0..5000).collect());
        let v = build_vocabulary(6000, map).expect("vocab");
        let prepared = PreparedVocabulary::build(&v).expect("prepared");
        assert_eq!(
            estimate_transient_build_bytes_prepared(&prepared),
            estimate_transient_build_bytes(&v)
        );
    }
}
