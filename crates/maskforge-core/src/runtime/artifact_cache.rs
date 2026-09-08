//! Caches compiled artifacts by input and vocabulary identity.

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use rustc_hash::FxHashMap;

use super::CompiledArtifact;
use crate::error::CompileError;
use crate::index::{BindMode, VocabFingerprint};

/// Bumped whenever a change to IR-compile or bind semantics could make a previously-cached artifact
/// for the same `(ir_hash, vocab_fingerprint, mask_vocab_size, bind_mode)` no longer equivalent to
pub const COMPILER_SEMANTICS_VERSION: u32 = 2;

/// Which raw byte representation an [`ArtifactKey`]'s `input_hash` was computed over. Keeps a
/// JSON-schema-keyed entry and an MFIR-wire-keyed entry from ever comparing equal even if the two
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
enum InputKind {
    JsonSchema,
    MfirWire,
    SemanticExecutable,
}

/// Content address of one compiled artifact: two vocabularies/schemas that produce equal keys are
/// guaranteed (by construction, not caller honesty - every field is read from validated state) to
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct ArtifactKey {
    input_hash: [u8; 32],
    exact_input: Arc<[u8]>,
    input_kind: InputKind,
    vocab_fingerprint: VocabFingerprint,
    mask_vocab_size: usize,
    bind_mode: BindMode,
    compiler_semantics_version: u32,
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct ArtifactCandidate {
    input_hash: [u8; 32],
    input_kind: InputKind,
    vocab_fingerprint: VocabFingerprint,
    mask_vocab_size: usize,
    bind_mode: BindMode,
    compiler_semantics_version: u32,
}

fn hash_input(domain: &[u8], bytes: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(domain);
    h.update(&u64::try_from(bytes.len()).unwrap_or(u64::MAX).to_le_bytes());
    h.update(bytes);
    *h.finalize().as_bytes()
}

impl Hash for ArtifactKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.candidate().hash(state);
    }
}

impl Hash for ArtifactCandidate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input_hash.hash(state);
        self.input_kind.hash(state);
        self.vocab_fingerprint.hash(state);
        self.mask_vocab_size.hash(state);
        self.bind_mode.hash(state);
        self.compiler_semantics_version.hash(state);
    }
}

impl ArtifactKey {
    /// A key over the exact JSON Schema text and `assume_closed` flag, hashed before any parsing -
    /// a repeat call with identical bytes is a cache hit that skips schema-to-IR lowering
    #[must_use]
    pub fn for_json_schema(
        schema: &str,
        assume_closed: bool,
        vocab_fingerprint: VocabFingerprint,
        mask_vocab_size: usize,
        bind_mode: BindMode,
    ) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"MFIR-artifact-key-json-schema-v2");
        hasher.update(
            &u64::try_from(schema.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        hasher.update(schema.as_bytes());
        hasher.update(&[u8::from(assume_closed)]);
        Self {
            input_hash: *hasher.finalize().as_bytes(),
            exact_input: Arc::from(schema.as_bytes()),
            input_kind: InputKind::JsonSchema,
            vocab_fingerprint,
            mask_vocab_size,
            bind_mode,
            compiler_semantics_version: COMPILER_SEMANTICS_VERSION,
        }
    }

    /// A key over the exact validated MFIR wire bytes, hashed before decoding - a repeat call
    /// skips `SchemaIR::from_wire`'s decode and revalidation on a hit.
    #[must_use]
    pub fn for_mfir_wire(
        wire: &[u8],
        vocab_fingerprint: VocabFingerprint,
        mask_vocab_size: usize,
        bind_mode: BindMode,
    ) -> Self {
        Self {
            input_hash: hash_input(b"MFIR-artifact-key-mfir-wire-v1", wire),
            exact_input: Arc::from(wire),
            input_kind: InputKind::MfirWire,
            vocab_fingerprint,
            mask_vocab_size,
            bind_mode,
            compiler_semantics_version: COMPILER_SEMANTICS_VERSION,
        }
    }

    /// A vocabulary binding key over the executable's exact canonical IR identity.
    #[must_use]
    pub fn for_executable(
        semantic_identity: &[u8],
        vocab_fingerprint: VocabFingerprint,
        mask_vocab_size: usize,
        bind_mode: BindMode,
    ) -> Self {
        Self {
            input_hash: hash_input(
                b"MFIR-artifact-key-semantic-executable-v1",
                semantic_identity,
            ),
            exact_input: Arc::from(semantic_identity),
            input_kind: InputKind::SemanticExecutable,
            vocab_fingerprint,
            mask_vocab_size,
            bind_mode,
            compiler_semantics_version: COMPILER_SEMANTICS_VERSION,
        }
    }

    fn candidate(&self) -> ArtifactCandidate {
        ArtifactCandidate {
            input_hash: self.input_hash,
            input_kind: self.input_kind,
            vocab_fingerprint: self.vocab_fingerprint,
            mask_vocab_size: self.mask_vocab_size,
            bind_mode: self.bind_mode,
            compiler_semantics_version: self.compiler_semantics_version,
        }
    }

    fn retained_bytes(&self) -> usize {
        self.exact_input
            .len()
            .saturating_add(std::mem::size_of::<Self>())
    }
}

/// Default byte budget for retained artifacts: `1 << 28` = 256 MiB. Above it, an artifact is still
/// built and returned but not retained, so compiling many distinct large schemas cannot grow the
const DEFAULT_ARTIFACT_CACHE_BYTES: usize = 1 << 28;

/// Conservative per-entry overhead charged ON TOP of an artifact's own retained bytes (the engine's
/// heap, the packed table, and the eager delta index): the `Arc` control block, the
const ARTIFACT_CACHE_ENTRY_OVERHEAD: usize = 2 * std::mem::size_of::<usize>()
    + std::mem::size_of::<CompiledArtifact>()
    + std::mem::size_of::<ArtifactKey>()
    + 64;

/// The one builder's result for a key, read by every waiter, so a build the budget does NOT retain
/// is still not rebuilt once per waiter. `CompileError` is `Clone`, so the error is shared too.
struct BuildSlot {
    result: Mutex<Option<Result<Arc<CompiledArtifact>, CompileError>>>,
    ready: Condvar,
}

/// A retained artifact with its charged byte size and an LRU access stamp.
struct CacheEntry {
    artifact: Arc<CompiledArtifact>,
    bytes: usize,
    last_used: u64,
}

#[derive(Default)]
struct CacheState {
    map: FxHashMap<ArtifactKey, CacheEntry>,
    bytes: usize,
    tick: u64,
    lru: VecDeque<(ArtifactCandidate, u64)>,
    in_flight: FxHashMap<ArtifactKey, Arc<BuildSlot>>,
}

impl CacheState {
    fn touch(&mut self, key: &ArtifactKey) -> u64 {
        self.tick = self.tick.saturating_add(1);
        let now = self.tick;
        self.lru.push_back((key.candidate(), now));
        let compact_at = self.map.len().max(1).saturating_mul(4).saturating_add(64);
        if self.lru.len() > compact_at {
            self.compact_lru();
        }
        now
    }

    fn compact_lru(&mut self) {
        let map = &self.map;
        self.lru.retain(|(candidate, tick)| {
            map.iter()
                .any(|(key, entry)| key.candidate() == *candidate && entry.last_used == *tick)
        });
    }

    /// Evicts least-recently-used retained entries until `need` more bytes fit under `budget`. Only
    /// the cache's `Arc` is dropped; a `Matcher`/caller still holding its own `Arc` stays valid.
    fn evict_until_fits(&mut self, need: usize, budget: usize, stats: &CacheStats) {
        while self.bytes.saturating_add(need) > budget {
            let Some((candidate, tick)) = self.lru.pop_front() else {
                break; // log exhausted; map empty (the new entry alone exceeds the budget: oversized)
            };
            let key = self.map.iter().find_map(|(key, entry)| {
                (key.candidate() == candidate && entry.last_used == tick).then(|| key.clone())
            });
            if let Some(key) = key {
                if let Some(entry) = self.map.remove(&key) {
                    self.bytes = self.bytes.saturating_sub(entry.bytes);
                    stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }
}

/// Cumulative cache-traffic counters, since creation (not reset by `clear`). Plain atomics, not a
/// `Mutex`: measured lock contention (`benches/artifact_cache_contention.rs`) showed a second mutex
#[derive(Default)]
struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    oversized_bypasses: AtomicU64,
}

/// Publishes an error to the build slot and clears the in-flight mark on drop UNLESS disarmed, so a
/// panicking builder never leaves waiters hung on a slot that will never be filled, nor a permanently
struct SlotGuard<'a> {
    cache: &'a ArtifactCache,
    key: ArtifactKey,
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
                    "artifact build failed to publish a result",
                )));
            }
            drop(r);
            self.slot.ready.notify_all();
            self.cache
                .state
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .in_flight
                .remove(&self.key);
        }
    }
}

/// A thread-safe, byte-bounded cache of compiled artifacts keyed by [`ArtifactKey`]. Read-mostly
/// after warm-up; once the budget is reached, further artifacts are built and returned but not
pub struct ArtifactCache {
    state: Mutex<CacheState>,
    stats: CacheStats,
    budget: usize,
}

impl Default for ArtifactCache {
    fn default() -> Self {
        Self::with_budget(DEFAULT_ARTIFACT_CACHE_BYTES)
    }
}

impl ArtifactCache {
    /// An empty cache with the default byte budget.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty cache with an explicit retained-byte budget.
    #[must_use]
    pub fn with_budget(budget: usize) -> Self {
        Self {
            state: Mutex::new(CacheState::default()),
            stats: CacheStats::default(),
            budget,
        }
    }

    /// The cached artifact for the exact JSON Schema text and options, calling `build` and
    /// inserting its result on a miss. The key is derived from `schema`/`assume_closed` INSIDE this
    #[allow(clippy::too_many_arguments)]
    pub fn get_or_build_json_schema(
        &self,
        schema: &str,
        assume_closed: bool,
        vocab_fingerprint: VocabFingerprint,
        mask_vocab_size: usize,
        bind_mode: BindMode,
        build: impl FnOnce() -> Result<CompiledArtifact, CompileError>,
    ) -> Result<Arc<CompiledArtifact>, CompileError> {
        let key = ArtifactKey::for_json_schema(
            schema,
            assume_closed,
            vocab_fingerprint,
            mask_vocab_size,
            bind_mode,
        );
        self.get_or_build(key, build)
    }

    /// Same contract as [`Self::get_or_build_json_schema`], keyed on the exact MFIR wire bytes
    /// instead of schema text.
    pub fn get_or_build_mfir_wire(
        &self,
        wire: &[u8],
        vocab_fingerprint: VocabFingerprint,
        mask_vocab_size: usize,
        bind_mode: BindMode,
        build: impl FnOnce() -> Result<CompiledArtifact, CompileError>,
    ) -> Result<Arc<CompiledArtifact>, CompileError> {
        let key = ArtifactKey::for_mfir_wire(wire, vocab_fingerprint, mask_vocab_size, bind_mode);
        self.get_or_build(key, build)
    }

    /// Returns or binds one exact vocabulary-neutral executable identity.
    pub fn get_or_build_executable(
        &self,
        semantic_identity: &[u8],
        vocab_fingerprint: VocabFingerprint,
        mask_vocab_size: usize,
        bind_mode: BindMode,
        build: impl FnOnce() -> Result<CompiledArtifact, CompileError>,
    ) -> Result<Arc<CompiledArtifact>, CompileError> {
        let key = ArtifactKey::for_executable(
            semantic_identity,
            vocab_fingerprint,
            mask_vocab_size,
            bind_mode,
        );
        self.get_or_build(key, build)
    }

    /// The cached artifact for `key`, calling `build` and inserting its result on a miss. Every
    /// bind mode is retained (charged via `artifact_cache_charge_upper_bound`, valid for any mode
    fn get_or_build(
        &self,
        key: ArtifactKey,
        build: impl FnOnce() -> Result<CompiledArtifact, CompileError>,
    ) -> Result<Arc<CompiledArtifact>, CompileError> {
        let slot = {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            if state.map.contains_key(&key) {
                let now = state.touch(&key);
                let entry = state
                    .map
                    .get_mut(&key)
                    .expect("just confirmed present under the same lock hold");
                entry.last_used = now;
                let artifact = entry.artifact.clone();
                drop(state);
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(artifact);
            }
            if let Some(slot) = state.in_flight.get(&key).cloned() {
                drop(state);
                let result = {
                    let mut r = slot.result.lock().unwrap_or_else(|p| p.into_inner());
                    while r.is_none() {
                        r = slot.ready.wait(r).unwrap_or_else(|p| p.into_inner());
                    }
                    r.clone().expect("slot is ready")
                };
                if result.is_ok() {
                    self.stats.hits.fetch_add(1, Ordering::Relaxed);
                }
                return result;
            }
            let slot = Arc::new(BuildSlot {
                result: Mutex::new(None),
                ready: Condvar::new(),
            });
            state.in_flight.insert(key.clone(), slot.clone());
            slot
        };

        let mut guard = SlotGuard {
            cache: self,
            key: key.clone(),
            slot: slot.clone(),
            armed: true,
        };
        let built: Result<Arc<CompiledArtifact>, CompileError> = build().map(Arc::new);

        {
            let mut r = slot.result.lock().unwrap_or_else(|p| p.into_inner());
            *r = Some(built.clone());
            slot.ready.notify_all();
        }
        {
            let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
            state.in_flight.remove(&key);
            if let Ok(ref arc) = built {
                self.stats.misses.fetch_add(1, Ordering::Relaxed);
                let entry_bytes = arc
                    .artifact_cache_charge_upper_bound()
                    .saturating_add(key.retained_bytes())
                    .saturating_add(ARTIFACT_CACHE_ENTRY_OVERHEAD);
                if entry_bytes > self.budget {
                    self.stats
                        .oversized_bypasses
                        .fetch_add(1, Ordering::Relaxed);
                } else {
                    state.evict_until_fits(entry_bytes, self.budget, &self.stats);
                    let now = state.touch(&key);
                    state.bytes = state.bytes.saturating_add(entry_bytes);
                    state.map.insert(
                        key,
                        CacheEntry {
                            artifact: arc.clone(),
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

    /// Drops every retained artifact. Also clears the LRU touch-log. Traffic counters (`stats()`)
    /// are NOT reset - they are a lifetime total, not scoped to content currently retained.
    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.map.clear();
        state.bytes = 0;
        state.lru.clear();
        state.tick = 0;
    }

    /// The number of retained artifacts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .map
            .len()
    }

    /// Whether the cache retains no artifact.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retained artifact bytes. Bounds the CACHE only - an artifact a `Matcher` still holds its own
    /// `Arc` to stays alive via that reference even if evicted or never cached here.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).bytes
    }

    /// `(hits, misses, evictions, oversized_bypasses)` since creation - a lifetime traffic total,
    /// NOT reset by `clear()`. `oversized_bypasses` is a built artifact whose conservative upper
    #[must_use]
    pub fn stats(&self) -> (u64, u64, u64, u64) {
        (
            self.stats.hits.load(Ordering::Relaxed),
            self.stats.misses.load(Ordering::Relaxed),
            self.stats.evictions.load(Ordering::Relaxed),
            self.stats.oversized_bypasses.load(Ordering::Relaxed),
        )
    }
}

#[cfg(test)]
mod key_tests {
    use super::*;

    fn fp() -> VocabFingerprint {
        let mut m: rustc_hash::FxHashMap<Vec<u8>, Vec<u32>> = rustc_hash::FxHashMap::default();
        m.insert(b"a".to_vec(), vec![0]);
        let v = crate::vocab::build_vocabulary(9, m).expect("vocab");
        VocabFingerprint::of(&v).expect("fingerprint")
    }

    #[test]
    fn json_schema_key_is_deterministic() {
        let a = ArtifactKey::for_json_schema("true", false, fp(), 10, BindMode::Naive);
        let b = ArtifactKey::for_json_schema("true", false, fp(), 10, BindMode::Naive);
        assert_eq!(a, b);
    }

    #[test]
    fn json_schema_key_distinguishes_schema_text() {
        let a = ArtifactKey::for_json_schema("true", false, fp(), 10, BindMode::Naive);
        let b = ArtifactKey::for_json_schema("false", false, fp(), 10, BindMode::Naive);
        assert_ne!(a, b);
    }

    #[test]
    fn json_schema_key_distinguishes_assume_closed() {
        let a = ArtifactKey::for_json_schema("true", false, fp(), 10, BindMode::Naive);
        let b = ArtifactKey::for_json_schema("true", true, fp(), 10, BindMode::Naive);
        assert_ne!(
            a, b,
            "assume_closed changes IR semantics; the key must not collapse it"
        );
    }

    #[test]
    fn mfir_wire_key_is_deterministic_and_distinguishes_bytes() {
        let a = ArtifactKey::for_mfir_wire(b"one", fp(), 10, BindMode::Naive);
        let b = ArtifactKey::for_mfir_wire(b"one", fp(), 10, BindMode::Naive);
        let c = ArtifactKey::for_mfir_wire(b"two", fp(), 10, BindMode::Naive);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn json_schema_and_mfir_wire_keys_never_collide_on_the_same_bytes() {
        let json_key = ArtifactKey::for_json_schema("true", false, fp(), 10, BindMode::Naive);
        let wire_key = ArtifactKey::for_mfir_wire(b"true\0", fp(), 10, BindMode::Naive);
        assert_ne!(json_key, wire_key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::RefEngine;
    use crate::correctness::corpus::corpus_cases;
    use crate::error::{HashState, Provenance};
    use crate::index::{VocabTrie, VocabularyHandle};
    use crate::vocab::{build_vocabulary, Vocabulary};
    use rustc_hash::FxHashMap as Map;

    fn boolean_engine() -> RefEngine {
        corpus_cases()
            .expect("corpus")
            .into_iter()
            .find(|c| c.name == "suite-type-boolean")
            .expect("boolean case")
            .engine
    }

    fn vocab(pairs: &[(&[u8], u32)], eos: u32) -> Vocabulary {
        let mut m: Map<Vec<u8>, Vec<u32>> = Map::default();
        for &(b, id) in pairs {
            m.entry(b.to_vec()).or_default().push(id);
        }
        build_vocabulary(eos, m).expect("vocab")
    }

    fn key_for(v: &Vocabulary, ir_hash: [u8; 32]) -> ArtifactKey {
        let fp = VocabFingerprint::of(v).expect("fingerprint");
        ArtifactKey::for_mfir_wire(&ir_hash, fp, v.eos_token_id() as usize + 1, BindMode::Naive)
    }

    /// A key matching what `packed_artifact` below actually produces.
    fn packed_key_for(v: &Vocabulary, ir_hash: [u8; 32]) -> ArtifactKey {
        let fp = VocabFingerprint::of(v).expect("fingerprint");
        ArtifactKey::for_mfir_wire(
            &ir_hash,
            fp,
            v.eos_token_id() as usize + 1,
            BindMode::TrieJointBytePacked,
        )
    }

    fn packed_artifact(v: &Vocabulary, ir_hash: [u8; 32]) -> CompiledArtifact {
        let handle = VocabularyHandle::new(Arc::new(v.clone())).expect("handle");
        let trie = VocabTrie::build_byte(v).expect("trie");
        CompiledArtifact::new_packed_from_trie(
            boolean_engine(),
            handle.canonical().clone(),
            Provenance::reference(HashState::Hash(ir_hash)),
            &trie,
            handle.fingerprint(),
        )
        .expect("packed artifact")
    }

    fn lazy_key_for(v: &Vocabulary, ir_hash: [u8; 32]) -> ArtifactKey {
        let fp = VocabFingerprint::of(v).expect("fingerprint");
        ArtifactKey::for_mfir_wire(
            &ir_hash,
            fp,
            v.eos_token_id() as usize + 1,
            BindMode::TrieJointByteLazy,
        )
    }

    fn lazy_artifact(v: &Vocabulary, ir_hash: [u8; 32]) -> CompiledArtifact {
        let handle = VocabularyHandle::new(Arc::new(v.clone())).expect("handle");
        let trie = VocabTrie::build_byte(v).expect("trie");
        CompiledArtifact::new_lazy_trie(
            boolean_engine(),
            handle.canonical().clone(),
            Provenance::reference(HashState::Hash(ir_hash)),
            Arc::new(trie),
            handle.fingerprint(),
        )
        .expect("lazy artifact")
    }

    #[test]
    fn a_second_identical_key_is_a_hit_not_a_rebuild() {
        let cache = ArtifactCache::new();
        let v = vocab(&[(b"a", 0), (b"b", 1)], 9);
        let key = packed_key_for(&v, [1; 32]);
        let mut builds = 0;
        let a = cache
            .get_or_build(key.clone(), || {
                builds += 1;
                Ok(packed_artifact(&v, [1; 32]))
            })
            .unwrap();
        let b = cache
            .get_or_build(key.clone(), || {
                builds += 1;
                Ok(packed_artifact(&v, [1; 32]))
            })
            .unwrap();
        assert!(Arc::ptr_eq(&a, &b), "a cache hit must reuse the Arc");
        assert_eq!(builds, 1, "the second call must not rebuild");
        let (hits, misses, _, _) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[test]
    fn distinct_ir_hashes_are_distinct_entries() {
        let cache = ArtifactCache::new();
        let v = vocab(&[(b"a", 0), (b"b", 1)], 9);
        let key_a = packed_key_for(&v, [1; 32]);
        let key_b = packed_key_for(&v, [2; 32]);
        let a = cache
            .get_or_build(key_a, || Ok(packed_artifact(&v, [1; 32])))
            .unwrap();
        let b = cache
            .get_or_build(key_b, || Ok(packed_artifact(&v, [2; 32])))
            .unwrap();
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn a_non_packed_result_is_retained_under_its_conservative_upper_bound() {
        let cache = ArtifactCache::new();
        let v = vocab(&[(b"a", 0), (b"b", 1)], 9);
        let key = key_for(&v, [1; 32]);
        let mut builds = 0;
        let a = cache
            .get_or_build(key.clone(), || {
                builds += 1;
                CompiledArtifact::new(
                    boolean_engine(),
                    Arc::new(v.clone()),
                    Provenance::reference(HashState::Hash([1; 32])),
                )
            })
            .unwrap();
        let b = cache
            .get_or_build(key.clone(), || {
                builds += 1;
                CompiledArtifact::new(
                    boolean_engine(),
                    Arc::new(v.clone()),
                    Provenance::reference(HashState::Hash([1; 32])),
                )
            })
            .unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "a repeat naive-mode build must be a real cache hit"
        );
        assert_eq!(builds, 1);
        assert_eq!(cache.len(), 1);
        let (hits, misses, _, _) = cache.stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[test]
    fn an_adaptively_lazy_result_is_retained_and_hit_on_repeat() {
        let cache = ArtifactCache::new();
        let v = vocab(&[(b"a", 0), (b"b", 1)], 9);
        let key = lazy_key_for(&v, [6; 32]);
        let mut builds = 0;
        let a = cache
            .get_or_build(key.clone(), || {
                builds += 1;
                Ok(lazy_artifact(&v, [6; 32]))
            })
            .unwrap();
        let b = cache
            .get_or_build(key, || {
                builds += 1;
                Ok(lazy_artifact(&v, [6; 32]))
            })
            .unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "an adaptive-lazy result must hit on repeat, not rebuild"
        );
        assert_eq!(
            builds, 1,
            "an identical deep schema must not silently rebuild forever"
        );
    }

    #[test]
    fn concurrent_identical_lazy_misses_build_exactly_once() {
        let cache = ArtifactCache::new();
        let v = vocab(&[(b"a", 0), (b"b", 1)], 9);
        let key = lazy_key_for(&v, [7; 32]);
        let builds = AtomicU64::new(0);
        let results: Vec<Arc<CompiledArtifact>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let cache = &cache;
                    let v = &v;
                    let builds = &builds;
                    let key = key.clone();
                    scope.spawn(move || {
                        cache
                            .get_or_build(key, || {
                                builds.fetch_add(1, Ordering::SeqCst);
                                std::thread::sleep(std::time::Duration::from_millis(5));
                                Ok(lazy_artifact(v, [7; 32]))
                            })
                            .unwrap()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(
            builds.load(Ordering::SeqCst),
            1,
            "8 concurrent identical lazy misses must build exactly once"
        );
        for r in &results[1..] {
            assert!(
                Arc::ptr_eq(&results[0], r),
                "every waiter must get the same Arc"
            );
        }
    }

    #[test]
    fn a_build_error_is_not_cached_and_a_later_call_can_still_succeed() {
        let cache = ArtifactCache::new();
        let v = vocab(&[(b"a", 0), (b"b", 1)], 9);
        let key = packed_key_for(&v, [3; 32]);
        let err = cache
            .get_or_build(key.clone(), || {
                Err(CompileError::new(
                    crate::error::ErrorCode::Unsupported,
                    crate::error::Stage::L4Bind,
                    "injected",
                ))
            })
            .unwrap_err();
        assert_eq!(err.code, crate::error::ErrorCode::Unsupported);
        assert_eq!(cache.len(), 0, "a failed build must not be retained");
        let ok = cache
            .get_or_build(key, || Ok(packed_artifact(&v, [3; 32])))
            .unwrap();
        assert_eq!(cache.len(), 1);
        let _ = ok;
    }

    #[test]
    fn an_oversized_artifact_is_returned_but_not_retained() {
        let cache = ArtifactCache::with_budget(1); // impossibly small: every entry is oversized
        let v = vocab(&[(b"a", 0), (b"b", 1)], 9);
        let key = packed_key_for(&v, [4; 32]);
        let a = cache
            .get_or_build(key, || Ok(packed_artifact(&v, [4; 32])))
            .unwrap();
        drop(a);
        assert_eq!(cache.len(), 0);
        let (_, _, _, oversized) = cache.stats();
        assert_eq!(oversized, 1);
    }

    #[test]
    fn clear_drops_retained_entries_but_not_lifetime_stats() {
        let cache = ArtifactCache::new();
        let v = vocab(&[(b"a", 0), (b"b", 1)], 9);
        let key = packed_key_for(&v, [5; 32]);
        cache
            .get_or_build(key, || Ok(packed_artifact(&v, [5; 32])))
            .unwrap();
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert_eq!(cache.len(), 0);
        let (_, misses, _, _) = cache.stats();
        assert_eq!(misses, 1, "clear must not reset lifetime traffic counters");
    }
}
