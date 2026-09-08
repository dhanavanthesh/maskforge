//! Bounded single-flight cache for vocabulary-independent schema executables.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};

use rustc_hash::FxHashMap;

use crate::automaton::RefEngine;
use crate::compile::compile_ir;
use crate::error::{CompileError, ErrorCode, Stage};
use crate::frontend::{
    schema_to_ir, schema_to_ir_with_resources, SchemaRegistry, SchemaResourceLimits,
};
use crate::ir::{CompileOptions, SchemaIR};
use crate::routing::{ExecutionKind, ROUTE_POLICY_VERSION};
use crate::structured::StructuredProgram;

/// Increment when vocabulary-neutral executable semantics change.
pub const EXECUTABLE_SEMANTICS_VERSION: u32 = 1;

const DEFAULT_EXECUTABLE_CACHE_BYTES: usize = 256 << 20;
const ENTRY_OVERHEAD: usize = 160;

fn cache_key_limit() -> CompileError {
    CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L1,
        "schema cache exact-key allocation",
    )
}

fn push_exact_usize(output: &mut Vec<u8>, value: usize) -> Result<(), CompileError> {
    let value = u64::try_from(value).map_err(|_| cache_key_limit())?;
    output
        .try_reserve(std::mem::size_of::<u64>())
        .map_err(|_| cache_key_limit())?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn push_exact_part(output: &mut Vec<u8>, value: &[u8]) -> Result<(), CompileError> {
    push_exact_usize(output, value.len())?;
    output
        .try_reserve(value.len())
        .map_err(|_| cache_key_limit())?;
    output.extend_from_slice(value);
    Ok(())
}

fn hash_resource_limits(hasher: &mut blake3::Hasher, limits: SchemaResourceLimits) {
    for value in [
        limits.max_documents,
        limits.max_resources,
        limits.max_total_input_bytes,
        limits.max_document_bytes,
        limits.max_ast_nodes,
        limits.max_ast_bytes,
        limits.max_uri_bytes,
        limits.max_total_uri_bytes,
        limits.max_anchors_per_resource,
        limits.max_total_anchors,
        limits.max_references,
        limits.max_schema_locations,
        limits.max_resolution_work,
        limits.max_retained_graph_bytes,
        limits.max_peak_build_bytes,
    ] {
        hasher.update(&u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes());
    }
    hasher.update(&limits.max_resource_depth.to_le_bytes());
}

/// Immutable compiled schema semantics with no vocabulary or mutable session state.
pub enum ExecutableSchema {
    /// A complete regular byte automaton.
    Regular(Arc<RefEngine>),
    /// A complete structured program with no reusable regular descendants.
    Structured(Arc<StructuredProgram>),
    /// A structured program whose plan embeds reusable regular descendants.
    Hybrid(Arc<StructuredProgram>),
}

/// Cached executable plus its canonical IR identity for provenance and L2 keys.
pub struct CachedExecutable {
    executable: Arc<ExecutableSchema>,
    ir_hash: [u8; 32],
    semantic_identity: Arc<[u8]>,
}

impl CachedExecutable {
    /// Vocabulary-neutral executable.
    #[must_use]
    pub fn executable(&self) -> &Arc<ExecutableSchema> {
        &self.executable
    }

    /// Canonical semantic IR hash produced once on the L1 miss.
    #[must_use]
    pub fn ir_hash(&self) -> [u8; 32] {
        self.ir_hash
    }

    /// Exact canonical IR bytes used by vocabulary-dependent cache identity.
    #[must_use]
    pub fn semantic_identity(&self) -> &[u8] {
        &self.semantic_identity
    }

    fn retained_bytes(&self) -> usize {
        self.executable
            .retained_bytes()
            .saturating_add(std::mem::size_of::<Self>())
    }
}

impl ExecutableSchema {
    /// Compile one already-frozen IR according to its shared root route.
    pub fn compile(ir: Arc<SchemaIR>) -> Result<Arc<Self>, CompileError> {
        let kind = ir
            .route_table()
            .get(ir.root())
            .map_or(ExecutionKind::Structured, |route| route.kind);
        let executable = match kind {
            ExecutionKind::Regular => Self::Regular(Arc::new(compile_ir(&ir)?)),
            ExecutionKind::Structured => Self::Structured(StructuredProgram::compile(ir)?),
            ExecutionKind::Hybrid => Self::Hybrid(StructuredProgram::compile(ir)?),
        };
        Ok(Arc::new(executable))
    }

    /// Regular engine when this executable is fully regular.
    #[must_use]
    pub fn regular_engine(&self) -> Option<&Arc<RefEngine>> {
        match self {
            Self::Regular(engine) => Some(engine),
            Self::Structured(_) | Self::Hybrid(_) => None,
        }
    }

    /// Structured program when this executable uses structured or hybrid execution.
    #[must_use]
    pub fn structured_program(&self) -> Option<&Arc<StructuredProgram>> {
        match self {
            Self::Regular(_) => None,
            Self::Structured(program) | Self::Hybrid(program) => Some(program),
        }
    }

    /// Conservative retained bytes charged to the schema cache.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        match self {
            Self::Regular(engine) => engine.heap_bytes(),
            Self::Structured(program) | Self::Hybrid(program) => {
                program.retained_plan_bytes().unwrap_or(0)
            }
        }
    }
}

#[derive(Clone, Eq, Hash, PartialEq)]
struct ExactInputKey {
    kind: InputKeyKind,
    bytes: Arc<[u8]>,
    options: CompileOptions,
    resource_limits: Option<SchemaResourceLimits>,
    route_policy_version: u32,
    executable_semantics_version: u32,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
enum InputKeyKind {
    Json,
    MfirWire,
    SchemaSet,
    SemanticIr,
}

impl ExactInputKey {
    fn json(schema: &str, options: CompileOptions) -> Self {
        Self {
            kind: InputKeyKind::Json,
            bytes: Arc::from(schema.as_bytes()),
            options,
            resource_limits: None,
            route_policy_version: ROUTE_POLICY_VERSION,
            executable_semantics_version: EXECUTABLE_SEMANTICS_VERSION,
        }
    }

    fn schema_set(
        schema: &str,
        retrieval_uri: Option<&str>,
        resources: &[(String, String)],
        options: CompileOptions,
        limits: SchemaResourceLimits,
    ) -> Result<Self, CompileError> {
        let mut bytes = Vec::new();
        push_exact_part(&mut bytes, schema.as_bytes())?;
        match retrieval_uri {
            Some(uri) => {
                bytes.push(1);
                push_exact_part(&mut bytes, uri.as_bytes())?;
            }
            None => bytes.push(0),
        }
        let mut ordered = Vec::new();
        ordered
            .try_reserve_exact(resources.len())
            .map_err(|_| cache_key_limit())?;
        ordered.extend(resources.iter());
        ordered.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        push_exact_usize(&mut bytes, ordered.len())?;
        for (uri, resource) in ordered {
            push_exact_part(&mut bytes, uri.as_bytes())?;
            push_exact_part(&mut bytes, resource.as_bytes())?;
        }
        Ok(Self {
            kind: InputKeyKind::SchemaSet,
            bytes: Arc::from(bytes),
            options,
            resource_limits: Some(limits),
            route_policy_version: ROUTE_POLICY_VERSION,
            executable_semantics_version: EXECUTABLE_SEMANTICS_VERSION,
        })
    }

    fn mfir_wire(wire: &[u8]) -> Self {
        Self {
            kind: InputKeyKind::MfirWire,
            bytes: Arc::from(wire),
            options: CompileOptions::default(),
            resource_limits: None,
            route_policy_version: ROUTE_POLICY_VERSION,
            executable_semantics_version: EXECUTABLE_SEMANTICS_VERSION,
        }
    }

    fn semantic(ir: &SchemaIR, bytes: Arc<[u8]>) -> Self {
        Self {
            kind: InputKeyKind::SemanticIr,
            bytes,
            options: ir.options().clone(),
            resource_limits: None,
            route_policy_version: ROUTE_POLICY_VERSION,
            executable_semantics_version: EXECUTABLE_SEMANTICS_VERSION,
        }
    }

    fn digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"maskforge-executable-input-v1");
        hasher.update(&[match self.kind {
            InputKeyKind::Json => 0,
            InputKeyKind::MfirWire => 1,
            InputKeyKind::SchemaSet => 2,
            InputKeyKind::SemanticIr => 3,
        }]);
        hasher.update(
            &u64::try_from(self.bytes.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        hasher.update(&self.bytes);
        hasher.update(&[self.options.object_closure as u8]);
        hasher.update(&[u8::from(self.options.format_assertion)]);
        hasher.update(&[self.options.unsupported_policy as u8]);
        hasher.update(&self.options.max_diagnostics.to_le_bytes());
        if let Some(limits) = self.resource_limits {
            hasher.update(&[1]);
            hash_resource_limits(&mut hasher, limits);
        } else {
            hasher.update(&[0]);
        }
        hasher.update(&self.route_policy_version.to_le_bytes());
        hasher.update(&self.executable_semantics_version.to_le_bytes());
        *hasher.finalize().as_bytes()
    }

    fn retained_bytes(&self) -> usize {
        self.bytes.len().saturating_add(std::mem::size_of::<Self>())
    }
}

struct Entry {
    key: ExactInputKey,
    executable: EntryValue,
    bytes: usize,
    stamp: u64,
}

enum EntryValue {
    Strong(Arc<CachedExecutable>),
    Alias(Weak<CachedExecutable>),
}

impl EntryValue {
    fn get(&self) -> Option<Arc<CachedExecutable>> {
        match self {
            Self::Strong(executable) => Some(executable.clone()),
            Self::Alias(executable) => executable.upgrade(),
        }
    }
}

struct BuildSlot {
    result: Mutex<Option<Result<Arc<CachedExecutable>, CompileError>>>,
    ready: Condvar,
}

#[derive(Default)]
struct State {
    buckets: FxHashMap<[u8; 32], Vec<Entry>>,
    in_flight: FxHashMap<ExactInputKey, Arc<BuildSlot>>,
    lru: VecDeque<([u8; 32], u64)>,
    bytes: usize,
    entries: usize,
    tick: u64,
    /// Bumped by `clear()`; a build started under an earlier generation is not retained.
    generation: u64,
}

/// Snapshot of vocabulary-neutral cache traffic and retention.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExecutableCacheStats {
    /// Exact-key ready hits, including single-flight waiters.
    pub hits: u64,
    /// Builders that claimed a vacant exact key.
    pub misses: u64,
    /// Ready entries removed to meet the byte budget.
    pub evictions: u64,
    /// Built executables returned but too large to retain.
    pub oversized_bypasses: u64,
    /// Current retained entry count.
    pub entries: usize,
    /// Current charged bytes.
    pub bytes: usize,
}

#[derive(Default)]
struct Traffic {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    oversized_bypasses: AtomicU64,
}

/// Exact-input, byte-bounded, single-flight cache of vocabulary-neutral executables.
pub struct ExecutableCache {
    state: Mutex<State>,
    traffic: Traffic,
    budget: usize,
}

impl Default for ExecutableCache {
    fn default() -> Self {
        Self::with_budget(DEFAULT_EXECUTABLE_CACHE_BYTES)
    }
}

impl ExecutableCache {
    /// Empty cache with the production byte budget.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Empty cache with an explicit byte budget.
    #[must_use]
    pub fn with_budget(budget: usize) -> Self {
        Self {
            state: Mutex::new(State::default()),
            traffic: Traffic::default(),
            budget,
        }
    }

    /// Returns or compiles the exact JSON input and options. Digest equality only selects a bucket;
    /// `ExactInputKey` equality establishes identity.
    pub fn get_or_compile_json(
        &self,
        schema: &str,
        options: CompileOptions,
    ) -> Result<Arc<CachedExecutable>, CompileError> {
        let key = ExactInputKey::json(schema, options.clone());
        self.get_or_build_alias(key, || {
            let ir = Arc::new(schema_to_ir(schema, options)?);
            self.get_or_compile_ir(ir)
        })
    }

    /// Returns or compiles an exact root, resource registry, options, and limit set.
    pub fn get_or_compile_schema_set(
        &self,
        schema: &str,
        retrieval_uri: Option<&str>,
        resources: &[(String, String)],
        options: CompileOptions,
        limits: SchemaResourceLimits,
    ) -> Result<Arc<CachedExecutable>, CompileError> {
        let key =
            ExactInputKey::schema_set(schema, retrieval_uri, resources, options.clone(), limits)?;
        self.get_or_build_alias(key, || {
            let mut registry = SchemaRegistry::new(limits);
            for (uri, resource) in resources {
                registry.insert(uri, Arc::<str>::from(resource.as_str()))?;
            }
            let ir = Arc::new(schema_to_ir_with_resources(
                schema,
                retrieval_uri,
                options,
                limits,
                registry,
            )?);
            self.get_or_compile_ir(ir)
        })
    }

    /// Returns or decodes one exact MFIR wire input before semantic executable lookup.
    pub fn get_or_compile_wire(&self, wire: &[u8]) -> Result<Arc<CachedExecutable>, CompileError> {
        let key = ExactInputKey::mfir_wire(wire);
        self.get_or_build_alias(key, || {
            let ir = Arc::new(SchemaIR::from_wire(wire)?);
            self.get_or_compile_ir(ir)
        })
    }

    /// Returns or compiles one exact canonical IR independently of its source spelling.
    pub fn get_or_compile_ir(
        &self,
        ir: Arc<SchemaIR>,
    ) -> Result<Arc<CachedExecutable>, CompileError> {
        let semantic_identity: Arc<[u8]> = Arc::from(ir.to_wire());
        let key = ExactInputKey::semantic(&ir, semantic_identity.clone());
        self.get_or_build(key, || {
            let ir_hash = ir.canonical_hash();
            let executable = ExecutableSchema::compile(ir)?;
            Ok(Arc::new(CachedExecutable {
                executable,
                ir_hash,
                semantic_identity,
            }))
        })
    }

    fn get_or_build_alias(
        &self,
        key: ExactInputKey,
        build: impl FnOnce() -> Result<Arc<CachedExecutable>, CompileError>,
    ) -> Result<Arc<CachedExecutable>, CompileError> {
        self.get_or_build_with_charge(key, false, build)
    }

    fn get_or_build(
        &self,
        key: ExactInputKey,
        build: impl FnOnce() -> Result<Arc<CachedExecutable>, CompileError>,
    ) -> Result<Arc<CachedExecutable>, CompileError> {
        self.get_or_build_with_charge(key, true, build)
    }

    fn get_or_build_with_charge(
        &self,
        key: ExactInputKey,
        charge_executable: bool,
        build: impl FnOnce() -> Result<Arc<CachedExecutable>, CompileError>,
    ) -> Result<Arc<CachedExecutable>, CompileError> {
        let digest = key.digest();
        let (slot, build_generation) = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if let Some(bucket) = state.buckets.get(&digest) {
                if let Some(index) = bucket.iter().position(|entry| entry.key == key) {
                    if let Some(executable) = bucket[index].executable.get() {
                        state.tick = state.tick.saturating_add(1);
                        let stamp = state.tick;
                        state.buckets.get_mut(&digest).expect("same lock")[index].stamp = stamp;
                        state.lru.push_back((digest, stamp));
                        self.traffic.hits.fetch_add(1, Ordering::Relaxed);
                        return Ok(executable);
                    }
                    let removed = state
                        .buckets
                        .get_mut(&digest)
                        .expect("same lock")
                        .swap_remove(index);
                    state.bytes = state.bytes.saturating_sub(removed.bytes);
                    state.entries = state.entries.saturating_sub(1);
                    if state.buckets.get(&digest).is_some_and(Vec::is_empty) {
                        state.buckets.remove(&digest);
                    }
                }
            }
            if let Some(slot) = state.in_flight.get(&key).cloned() {
                drop(state);
                let mut result = slot
                    .result
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                while result.is_none() {
                    result = slot
                        .ready
                        .wait(result)
                        .unwrap_or_else(|poison| poison.into_inner());
                }
                let result = result.clone().expect("notified slot has a result");
                if result.is_ok() {
                    self.traffic.hits.fetch_add(1, Ordering::Relaxed);
                }
                return result;
            }
            let slot = Arc::new(BuildSlot {
                result: Mutex::new(None),
                ready: Condvar::new(),
            });
            state.in_flight.insert(key.clone(), slot.clone());
            (slot, state.generation)
        };

        let guard = PublishGuard {
            cache: self,
            key: key.clone(),
            slot: slot.clone(),
            published: false,
        };
        let built = build();
        {
            let mut result = slot
                .result
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            *result = Some(built.clone());
            slot.ready.notify_all();
        }
        {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            state.in_flight.remove(&key);
            self.traffic.misses.fetch_add(1, Ordering::Relaxed);
            if let Ok(executable) = &built {
                if state.generation == build_generation {
                    let executable_bytes = if charge_executable {
                        executable.retained_bytes()
                    } else {
                        std::mem::size_of::<Arc<CachedExecutable>>()
                    };
                    let bytes = key
                        .retained_bytes()
                        .saturating_add(executable_bytes)
                        .saturating_add(ENTRY_OVERHEAD);
                    if bytes > self.budget {
                        self.traffic
                            .oversized_bypasses
                            .fetch_add(1, Ordering::Relaxed);
                    } else {
                        self.evict_until_fits(&mut state, bytes);
                        state.tick = state.tick.saturating_add(1);
                        let stamp = state.tick;
                        state.buckets.entry(digest).or_default().push(Entry {
                            key,
                            executable: if charge_executable {
                                EntryValue::Strong(executable.clone())
                            } else {
                                EntryValue::Alias(Arc::downgrade(executable))
                            },
                            bytes,
                            stamp,
                        });
                        state.entries = state.entries.saturating_add(1);
                        state.bytes = state.bytes.saturating_add(bytes);
                        state.lru.push_back((digest, stamp));
                    }
                }
            }
        }
        let mut guard = guard;
        guard.published = true;
        built
    }

    fn evict_until_fits(&self, state: &mut State, need: usize) {
        while state.bytes.saturating_add(need) > self.budget {
            let Some((digest, stamp)) = state.lru.pop_front() else {
                break;
            };
            let mut removed = None;
            let mut empty = false;
            if let Some(bucket) = state.buckets.get_mut(&digest) {
                if let Some(index) = bucket.iter().position(|entry| entry.stamp == stamp) {
                    removed = Some(bucket.swap_remove(index).bytes);
                    empty = bucket.is_empty();
                }
            }
            if empty {
                state.buckets.remove(&digest);
            }
            if let Some(bytes) = removed {
                state.bytes = state.bytes.saturating_sub(bytes);
                state.entries = state.entries.saturating_sub(1);
                self.traffic.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Drops every ready entry, releasing their retained bytes; leaves in-flight builds running.
    pub fn clear(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.buckets.clear();
        state.lru.clear();
        state.bytes = 0;
        state.entries = 0;
        state.generation = state.generation.wrapping_add(1);
    }

    /// Current retention plus cumulative traffic.
    #[must_use]
    pub fn stats(&self) -> ExecutableCacheStats {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        ExecutableCacheStats {
            hits: self.traffic.hits.load(Ordering::Relaxed),
            misses: self.traffic.misses.load(Ordering::Relaxed),
            evictions: self.traffic.evictions.load(Ordering::Relaxed),
            oversized_bypasses: self.traffic.oversized_bypasses.load(Ordering::Relaxed),
            entries: state.entries,
            bytes: state.bytes,
        }
    }
}

struct PublishGuard<'a> {
    cache: &'a ExecutableCache,
    key: ExactInputKey,
    slot: Arc<BuildSlot>,
    published: bool,
}

impl Drop for PublishGuard<'_> {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        let mut result = self
            .slot
            .result
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if result.is_none() {
            *result = Some(Err(CompileError::new(
                ErrorCode::InternalLimitExceeded,
                Stage::L3,
                "schema executable builder failed to publish",
            )));
        }
        drop(result);
        self.slot.ready.notify_all();
        self.cache
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .in_flight
            .remove(&self.key);
    }
}

/// Process-wide bounded L1 cache. It contains no vocabulary or session state.
#[must_use]
pub fn global_executable_cache() -> &'static ExecutableCache {
    static CACHE: OnceLock<ExecutableCache> = OnceLock::new();
    CACHE.get_or_init(ExecutableCache::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_options_are_part_of_identity() {
        let cache = ExecutableCache::new();
        let schema = r#"{"type":"string"}"#;
        let first = cache
            .get_or_compile_json(schema, CompileOptions::default())
            .unwrap();
        let second = cache
            .get_or_compile_json(schema, CompileOptions::default())
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        let mut changed = CompileOptions::default();
        changed.format_assertion = !changed.format_assertion;
        let third = cache.get_or_compile_json(schema, changed).unwrap();
        assert!(!Arc::ptr_eq(&first, &third));
        assert_eq!(cache.stats().entries, 4);
    }

    #[test]
    fn clear_drops_retention_but_a_later_lookup_still_compiles_correctly() {
        let cache = ExecutableCache::new();
        let schema = r#"{"type":"string"}"#;
        let first = cache
            .get_or_compile_json(schema, CompileOptions::default())
            .unwrap();
        let entries_before = cache.stats().entries;
        assert!(entries_before > 0);
        cache.clear();
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().bytes, 0);
        let second = cache
            .get_or_compile_json(schema, CompileOptions::default())
            .unwrap();
        assert!(
            !Arc::ptr_eq(&first, &second),
            "clear must force a fresh build, not alias the old Arc"
        );
        assert_eq!(first.ir_hash(), second.ir_hash());
        assert_eq!(cache.stats().entries, entries_before);
    }

    #[test]
    fn clear_during_an_in_flight_build_still_satisfies_the_waiter_but_is_not_retained() {
        let cache = Arc::new(ExecutableCache::new());
        let key = ExactInputKey::json(r#"{"type":"boolean"}"#, CompileOptions::default());
        let ir =
            Arc::new(schema_to_ir(r#"{"type":"boolean"}"#, CompileOptions::default()).unwrap());
        let ir_hash = ir.canonical_hash();
        let prepared = Arc::new(CachedExecutable {
            executable: ExecutableSchema::compile(ir).unwrap(),
            ir_hash,
            semantic_identity: Arc::from([9u8]),
        });

        let (claimed_tx, claimed_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let builder_cache = cache.clone();
        let builder_key = key.clone();
        let builder_prepared = prepared.clone();
        let builder = std::thread::spawn(move || {
            builder_cache.get_or_build(builder_key, || {
                claimed_tx.send(()).unwrap();
                release_rx.recv().unwrap(); // blocks until the test explicitly releases it
                Ok(builder_prepared)
            })
        });

        claimed_rx.recv().unwrap(); // the build has claimed its in-flight slot
        cache.clear(); // bumps the generation while that build is still running
        release_tx.send(()).unwrap();

        let built = builder.join().unwrap().unwrap();
        assert!(
            Arc::ptr_eq(&built, &prepared),
            "a build in flight when clear() runs must still satisfy its own waiter"
        );
        assert_eq!(
            cache.stats().entries,
            0,
            "a build started before clear() must not be retained once it completes"
        );
        assert_eq!(cache.stats().bytes, 0);
    }

    #[test]
    fn distinct_json_spellings_share_one_semantic_executable() {
        let cache = ExecutableCache::new();
        let compact = cache
            .get_or_compile_json(r#"{"type":"boolean"}"#, CompileOptions::default())
            .unwrap();
        let spaced = cache
            .get_or_compile_json(r#"{ "type": "boolean" }"#, CompileOptions::default())
            .unwrap();
        assert!(Arc::ptr_eq(&compact, &spaced));
        assert_eq!(cache.stats().entries, 3);
    }

    #[test]
    fn resource_bytes_and_limits_are_exact_input_identity() {
        let cache = ExecutableCache::new();
        let root = r#"{"$ref":"https://example.test/child"}"#;
        let first_resources = vec![(
            "https://example.test/child".to_string(),
            r#"{"type":"boolean"}"#.to_string(),
        )];
        let second_resources = vec![(
            "https://example.test/child".to_string(),
            r#"{"type":"null"}"#.to_string(),
        )];
        let limits = SchemaResourceLimits::default();
        let first = cache
            .get_or_compile_schema_set(
                root,
                Some("https://example.test/root"),
                &first_resources,
                CompileOptions::default(),
                limits,
            )
            .unwrap();
        let second = cache
            .get_or_compile_schema_set(
                root,
                Some("https://example.test/root"),
                &second_resources,
                CompileOptions::default(),
                limits,
            )
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &second));

        let mut larger_limits = limits;
        larger_limits.max_document_bytes = larger_limits.max_document_bytes.saturating_add(1);
        let limited = cache
            .get_or_compile_schema_set(
                root,
                Some("https://example.test/root"),
                &first_resources,
                CompileOptions::default(),
                larger_limits,
            )
            .unwrap();
        assert!(Arc::ptr_eq(&first, &limited));
        assert_eq!(cache.stats().entries, 5);
    }

    #[test]
    fn digest_bucket_collision_still_requires_exact_key_equality() {
        let cache = ExecutableCache::new();
        let first_key = ExactInputKey::json("true", CompileOptions::default());
        let second_key = ExactInputKey::json("false", CompileOptions::default());
        let ir = Arc::new(schema_to_ir("true", CompileOptions::default()).unwrap());
        let prepared = Arc::new(CachedExecutable {
            ir_hash: ir.canonical_hash(),
            executable: ExecutableSchema::compile(ir).unwrap(),
            semantic_identity: Arc::from([1u8]),
        });
        let forced_digest = second_key.digest();
        cache
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .buckets
            .entry(forced_digest)
            .or_default()
            .push(Entry {
                key: first_key,
                executable: EntryValue::Strong(prepared.clone()),
                bytes: 1,
                stamp: 1,
            });
        let builds = AtomicU64::new(0);
        let result = cache
            .get_or_build(second_key, || {
                builds.fetch_add(1, Ordering::SeqCst);
                Ok(prepared.clone())
            })
            .unwrap();
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert!(Arc::ptr_eq(&result, &prepared));
        assert_eq!(
            cache
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .buckets[&forced_digest]
                .len(),
            2
        );
    }

    #[test]
    fn thirty_two_callers_single_flight_one_compile() {
        let cache = Arc::new(ExecutableCache::new());
        let key = ExactInputKey::json(r#"{"type":"boolean"}"#, CompileOptions::default());
        let ir =
            Arc::new(schema_to_ir(r#"{"type":"boolean"}"#, CompileOptions::default()).unwrap());
        let ir_hash = ir.canonical_hash();
        let prepared = Arc::new(CachedExecutable {
            executable: ExecutableSchema::compile(ir).unwrap(),
            ir_hash,
            semantic_identity: Arc::from([2u8]),
        });
        let builds = Arc::new(AtomicU64::new(0));
        let results: Vec<_> = (0..32)
            .map(|_| {
                let cache = cache.clone();
                let key = key.clone();
                let prepared = prepared.clone();
                let builds = builds.clone();
                std::thread::spawn(move || {
                    cache
                        .get_or_build(key, || {
                            builds.fetch_add(1, Ordering::SeqCst);
                            std::thread::sleep(std::time::Duration::from_millis(20));
                            Ok(prepared)
                        })
                        .unwrap()
                })
            })
            .map(|thread| thread.join().unwrap())
            .collect();
        for executable in &results[1..] {
            assert!(Arc::ptr_eq(&results[0], executable));
        }
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 31);
    }

    #[test]
    fn oversized_executable_is_returned_but_not_retained() {
        let cache = ExecutableCache::with_budget(1);
        let executable = cache
            .get_or_compile_json("true", CompileOptions::default())
            .unwrap();
        drop(executable);
        assert_eq!(cache.stats().entries, 0);
        assert_eq!(cache.stats().oversized_bypasses, 2);
    }

    #[test]
    fn builder_panic_wakes_waiter_and_clears_the_claim() {
        let cache = Arc::new(ExecutableCache::new());
        let key = ExactInputKey::json(
            r#"{"description":"panic-test","type":"boolean"}"#,
            CompileOptions::default(),
        );
        let ir = Arc::new(schema_to_ir("true", CompileOptions::default()).unwrap());
        let ir_hash = ir.canonical_hash();
        let prepared = Arc::new(CachedExecutable {
            executable: ExecutableSchema::compile(ir).unwrap(),
            ir_hash,
            semantic_identity: Arc::from([3u8]),
        });
        let (claimed_tx, claimed_rx) = std::sync::mpsc::channel();
        let builder_cache = cache.clone();
        let builder_key = key.clone();
        let builder = std::thread::spawn(move || {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = builder_cache.get_or_build(builder_key, || {
                    claimed_tx.send(()).unwrap();
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    panic!("injected builder panic");
                });
            }))
        });
        claimed_rx.recv().unwrap();
        let waiter_cache = cache.clone();
        let waiter_key = key.clone();
        let waiter = std::thread::spawn(move || {
            waiter_cache.get_or_build(waiter_key, || panic!("waiter must not build"))
        });
        assert!(builder.join().unwrap().is_err());
        let waiter_error = match waiter.join().unwrap() {
            Ok(_) => panic!("a waiter must observe the panic guard's failure"),
            Err(error) => error,
        };
        assert_eq!(waiter_error.code, ErrorCode::InternalLimitExceeded);
        let rebuilt = cache
            .get_or_build(key, || Ok(prepared.clone()))
            .expect("a panic must not poison the exact key permanently");
        assert!(Arc::ptr_eq(&rebuilt, &prepared));
    }

    #[test]
    fn builder_error_wakes_waiters_and_is_not_retained() {
        let cache = Arc::new(ExecutableCache::new());
        let key = ExactInputKey::json(
            r#"{"description":"error-test","type":"boolean"}"#,
            CompileOptions::default(),
        );
        let (claimed_tx, claimed_rx) = std::sync::mpsc::channel();
        let builder_cache = cache.clone();
        let builder_key = key.clone();
        let builder = std::thread::spawn(move || {
            builder_cache.get_or_build(builder_key, || {
                claimed_tx.send(()).unwrap();
                std::thread::sleep(std::time::Duration::from_millis(50));
                Err(CompileError::new(
                    ErrorCode::Unsupported,
                    Stage::L3,
                    "injected",
                ))
            })
        });
        claimed_rx.recv().unwrap();
        let waiter_cache = cache.clone();
        let waiter_key = key.clone();
        let waiter = std::thread::spawn(move || {
            waiter_cache.get_or_build(waiter_key, || panic!("waiter must not build"))
        });
        let builder_error = match builder.join().unwrap() {
            Ok(_) => panic!("injected builder failure must be returned"),
            Err(error) => error,
        };
        let waiter_error = match waiter.join().unwrap() {
            Ok(_) => panic!("waiter must observe the builder failure"),
            Err(error) => error,
        };
        assert_eq!(builder_error.code, ErrorCode::Unsupported);
        assert_eq!(waiter_error.code, ErrorCode::Unsupported);
        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn evicted_executable_remains_usable_through_the_callers_arc() {
        let probe = ExecutableCache::new();
        let probe_ir = Arc::new(schema_to_ir("true", CompileOptions::default()).unwrap());
        let _ = probe.get_or_compile_ir(probe_ir).unwrap();
        let budget = probe.stats().bytes.saturating_add(1);

        let cache = ExecutableCache::with_budget(budget);
        let first_ir = Arc::new(schema_to_ir("true", CompileOptions::default()).unwrap());
        let held = cache.get_or_compile_ir(first_ir).unwrap();
        let second_ir = Arc::new(schema_to_ir("false", CompileOptions::default()).unwrap());
        let _ = cache.get_or_compile_ir(second_ir).unwrap();

        assert!(cache.stats().evictions >= 1);
        assert!(!held.semantic_identity().is_empty());
        assert_ne!(held.ir_hash(), [0; 32]);
    }
}
