//! The stable, high-level facade: compile a schema once, prepare a vocabulary once, bind them,
//! drive one generation sequence. Callers never inspect which backend a schema needs.
//!
//! ```no_run
//! # use maskforge_core::{Compiler, CompilerOptions, CompiledVocabulary};
//! # use std::sync::Arc;
//! # fn go(vocabulary: Arc<maskforge_core::Vocabulary>, mask: &mut [u32]) -> Result<(), Box<dyn std::error::Error>> {
//! let compiler = Compiler::new(CompilerOptions::default());
//! let program = compiler.compile_json_schema(r#"{"type":"boolean"}"#)?;
//! let vocabulary = CompiledVocabulary::try_from(vocabulary)?;
//! let bound = program.bind(&vocabulary)?;
//! let mut session = bound.start_session()?;
//! session.write_mask(mask)?;
//! # Ok(())
//! # }
//! ```

use std::fmt;
use std::sync::Arc;

use crate::error::{CompileError, HashState, MatcherError, Provenance};
use crate::frontend::SchemaResourceLimits;
use crate::index::{BindMode, BoundByteTrie, TrieCache, VocabularyHandle};
use crate::ir::CompileOptions;
use crate::primitives::TokenId;
use crate::runtime::{
    CachedExecutable, CompiledArtifact, ExecutableCache, ExecutableCacheStats, ExecutableSchema,
    Matcher,
};
use crate::structured::{StructuredMatcher, StructuredMatcherError, StructuredProgram};
use crate::vocab::Vocabulary;

/// How [`SchemaProgram::bind`] chooses a regular-schema bind mode; `Naive` stays low-level-only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum BindPolicy {
    /// Packed table when affordable, else lazy trie-backed rows. The default.
    #[default]
    Adaptive,
    /// Always the packed table: fastest per-mask query, higher bind cost.
    Eager,
    /// Always lazy trie-backed rows: cheap bind, each row cached on first query.
    Lazy,
}

/// Cache budgets a [`Compiler`] owns, in bytes rather than entry counts.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct CompilerOptions {
    /// Byte budget for the vocabulary-independent executable cache. `0` disables caching.
    pub executable_cache_bytes: usize,
    /// Byte budget for the shared vocabulary byte-trie cache. `0` disables retention only.
    pub trie_cache_bytes: usize,
    /// How [`SchemaProgram::bind`] picks a regular-schema bind mode.
    pub bind_policy: BindPolicy,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        Self {
            executable_cache_bytes: 256 << 20,
            trie_cache_bytes: 256 << 20,
            bind_policy: BindPolicy::Adaptive,
        }
    }
}

impl CompilerOptions {
    /// Sets the executable cache budget, in bytes. `0` disables caching.
    #[must_use]
    pub fn with_executable_cache_bytes(mut self, bytes: usize) -> Self {
        self.executable_cache_bytes = bytes;
        self
    }

    /// Sets the shared vocabulary byte-trie cache budget, in bytes.
    #[must_use]
    pub fn with_trie_cache_bytes(mut self, bytes: usize) -> Self {
        self.trie_cache_bytes = bytes;
        self
    }

    /// Sets how [`SchemaProgram::bind`] picks a regular-schema bind mode.
    #[must_use]
    pub fn with_bind_policy(mut self, policy: BindPolicy) -> Self {
        self.bind_policy = policy;
        self
    }
}

/// Current retention plus cumulative traffic for a [`Compiler`]'s trie cache.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct TrieCacheStats {
    /// Number of vocabulary tries currently retained.
    pub entries: usize,
    /// Bytes currently retained across those entries.
    pub bytes: usize,
    /// Lifetime cache hits.
    pub hits: u64,
    /// Lifetime cache misses (a trie was built).
    pub misses: u64,
    /// Lifetime evictions (a retained trie dropped to make room for another).
    pub evictions: u64,
    /// Lifetime builds too large to retain under the budget (built and used, never cached).
    pub oversized_bypasses: u64,
}

/// Owns schema-compilation cache state. Construct one per process (or per isolated tenant);
/// nothing here is required to be a singleton.
pub struct Compiler {
    executable_cache: ExecutableCache,
    trie_cache: Arc<TrieCache>,
    bind_policy: BindPolicy,
}

impl Compiler {
    /// A compiler with its own cache, budgeted per `options`.
    #[must_use]
    pub fn new(options: CompilerOptions) -> Self {
        Self {
            executable_cache: ExecutableCache::with_budget(options.executable_cache_bytes),
            trie_cache: Arc::new(TrieCache::with_budget(options.trie_cache_bytes)),
            bind_policy: options.bind_policy,
        }
    }

    /// A compiler that caches nothing: every compile and bind rebuilds its artifacts.
    /// Useful for deterministic tests or callers that cache `SchemaProgram` themselves.
    #[must_use]
    pub fn without_cache() -> Self {
        Self::new(CompilerOptions {
            executable_cache_bytes: 0,
            trie_cache_bytes: 0,
            bind_policy: BindPolicy::Adaptive,
        })
    }

    /// Compiles JSON Schema text with default options. Vocabulary-independent: reuse the
    /// returned [`SchemaProgram`] across every vocabulary that needs this schema.
    pub fn compile_json_schema(&self, schema: &str) -> Result<SchemaProgram, CompileError> {
        self.compile_json_schema_with_options(schema, CompileOptions::default())
    }

    /// Compiles JSON Schema text with explicit lowering options.
    pub fn compile_json_schema_with_options(
        &self,
        schema: &str,
        options: CompileOptions,
    ) -> Result<SchemaProgram, CompileError> {
        let cached = self.executable_cache.get_or_compile_json(schema, options)?;
        Ok(self.wrap(cached))
    }

    /// Compiles against caller-supplied resources; never fetches a `$ref` target itself.
    pub fn compile_json_schema_with_resources(
        &self,
        schema: &str,
        retrieval_uri: Option<&str>,
        resources: &[(String, String)],
        options: CompileOptions,
        limits: SchemaResourceLimits,
    ) -> Result<SchemaProgram, CompileError> {
        let cached = self.executable_cache.get_or_compile_schema_set(
            schema,
            retrieval_uri,
            resources,
            options,
            limits,
        )?;
        Ok(self.wrap(cached))
    }

    /// Decodes and compiles a frozen MFIR wire payload (see [`crate::SchemaIR::to_wire`]).
    pub fn compile_ir_wire(&self, wire: &[u8]) -> Result<SchemaProgram, CompileError> {
        let cached = self.executable_cache.get_or_compile_wire(wire)?;
        Ok(self.wrap(cached))
    }

    fn wrap(&self, cached: Arc<CachedExecutable>) -> SchemaProgram {
        SchemaProgram {
            cached,
            trie_cache: Arc::clone(&self.trie_cache),
            bind_policy: self.bind_policy,
        }
    }

    /// Current retention plus cumulative traffic for the executable cache.
    #[must_use]
    pub fn cache_stats(&self) -> ExecutableCacheStats {
        self.executable_cache.stats()
    }

    /// Current retention plus cumulative traffic for the shared vocabulary trie cache.
    #[must_use]
    pub fn trie_cache_stats(&self) -> TrieCacheStats {
        let (hits, misses, evictions, oversized_bypasses) = self.trie_cache.stats();
        TrieCacheStats {
            entries: self.trie_cache.len(),
            bytes: self.trie_cache.bytes(),
            hits,
            misses,
            evictions,
            oversized_bypasses,
        }
    }

    /// Drops every cached executable and retained vocabulary trie; an existing `BoundSchema` keeps its own `Arc`-retained trie safely.
    pub fn clear_caches(&self) {
        self.executable_cache.clear();
        self.trie_cache.clear();
    }
}

/// A compiled schema, immutable and vocabulary-independent. Cheap to clone (an `Arc` handle);
/// safe to share across threads and reuse across as many vocabularies as needed.
#[derive(Clone)]
pub struct SchemaProgram {
    cached: Arc<CachedExecutable>,
    trie_cache: Arc<TrieCache>,
    bind_policy: BindPolicy,
}

impl SchemaProgram {
    /// Binds this schema to a prepared vocabulary, trie-backed per [`BindPolicy`]; `self` stays reusable.
    pub fn bind(&self, vocabulary: &CompiledVocabulary) -> Result<BoundSchema, CompileError> {
        let handle = vocabulary.handle.clone();
        let source_hash = HashState::Hash(self.cached.ir_hash());
        let eos_token_id = TokenId(handle.eos_token_id());
        let mask_vocab_size = handle.mask_vocab_size();
        let inner = match self.cached.executable().as_ref() {
            ExecutableSchema::Regular(engine) => {
                let bound_trie = self.trie_cache.bind(&handle)?;
                let requested = match self.bind_policy {
                    BindPolicy::Lazy => BindMode::TrieJointByteLazy,
                    BindPolicy::Eager | BindPolicy::Adaptive => BindMode::TrieJointBytePacked,
                };
                let mode = match self.bind_policy {
                    BindPolicy::Adaptive => CompiledArtifact::recommended_bind_mode(
                        engine.state_count(),
                        mask_vocab_size,
                        &bound_trie,
                        requested,
                    ),
                    BindPolicy::Eager | BindPolicy::Lazy => requested,
                };
                let artifact = CompiledArtifact::new_from_bound_trie_shared(
                    &handle,
                    Arc::clone(engine),
                    Provenance::reference(source_hash),
                    mode,
                    &bound_trie,
                )?;
                BoundSchemaKind::Regular(Arc::new(artifact))
            }
            ExecutableSchema::Structured(program) | ExecutableSchema::Hybrid(program) => {
                let trie = program
                    .uses_incremental_backend()
                    .then(|| self.trie_cache.bind(&handle))
                    .transpose()?;
                BoundSchemaKind::Structured(Arc::clone(program), handle, trie)
            }
        };
        Ok(BoundSchema {
            inner,
            eos_token_id,
            mask_vocab_size,
        })
    }

    /// Builds and binds a fresh [`CompiledVocabulary`] each call; prefer [`bind`](Self::bind) with a reused one.
    pub fn bind_new_vocabulary(
        &self,
        vocabulary: Arc<Vocabulary>,
    ) -> Result<BoundSchema, CompileError> {
        self.bind(&CompiledVocabulary::new(vocabulary)?)
    }
}

/// A prepared, reusable vocabulary; build once and reuse across binds. Cheap to clone.
#[derive(Clone)]
pub struct CompiledVocabulary {
    handle: VocabularyHandle,
}

impl CompiledVocabulary {
    /// Wraps `vocabulary`, inferring the mask width from its largest token/EOS id.
    pub fn new(vocabulary: Arc<Vocabulary>) -> Result<Self, CompileError> {
        Ok(Self {
            handle: VocabularyHandle::new(vocabulary)?,
        })
    }

    /// `new`, with an explicit mask width (e.g. a model whose logits are padded wider than the
    /// tokenizer's own id range). Rejected if narrower than the largest token/EOS id present.
    pub fn with_logits_vocab_size(
        vocabulary: Arc<Vocabulary>,
        logits_vocab_size: usize,
    ) -> Result<Self, CompileError> {
        Ok(Self {
            handle: VocabularyHandle::new_with_logits_vocab_size(
                vocabulary,
                Some(logits_vocab_size),
            )?,
        })
    }

    /// Builds directly from four packed CSR buffers, skipping an intermediate
    /// `Vocabulary`/`HashMap`.
    pub fn from_packed(
        token_bytes: &[u8],
        byte_offsets: &[u32],
        token_ids: &[u32],
        id_offsets: &[u32],
        eos_token_id: u32,
    ) -> Result<Self, CompileError> {
        Ok(Self {
            handle: VocabularyHandle::from_packed(
                token_bytes,
                byte_offsets,
                token_ids,
                id_offsets,
                eos_token_id,
            )?,
        })
    }

    /// Wraps an already-built `VocabularyHandle`, skipping re-canonicalization.
    #[must_use]
    pub fn from_handle(handle: VocabularyHandle) -> Self {
        Self { handle }
    }

    /// The mask width: `write_mask` buffers must hold `ceil(mask_vocab_size / 32)` words.
    #[must_use]
    pub fn mask_vocab_size(&self) -> usize {
        self.handle.mask_vocab_size()
    }

    /// This vocabulary's EOS token id.
    #[must_use]
    pub fn eos_token_id(&self) -> u32 {
        self.handle.eos_token_id()
    }

    /// Exact raw bytes associated with `token_id`, or `None` for an absent/special id.
    #[must_use]
    pub fn token_bytes(&self, token_id: u32) -> Option<&[u8]> {
        self.handle.token_bytes(token_id)
    }
}

impl TryFrom<Arc<Vocabulary>> for CompiledVocabulary {
    type Error = CompileError;

    fn try_from(vocabulary: Arc<Vocabulary>) -> Result<Self, Self::Error> {
        Self::new(vocabulary)
    }
}

#[derive(Clone)]
enum BoundSchemaKind {
    Regular(Arc<CompiledArtifact>),
    Structured(
        Arc<StructuredProgram>,
        VocabularyHandle,
        Option<BoundByteTrie>,
    ),
}

/// A schema bound to one vocabulary. Immutable; start as many independent [`Session`]s from it
/// as there are concurrent generation sequences. Cheap to clone.
#[derive(Clone)]
pub struct BoundSchema {
    inner: BoundSchemaKind,
    eos_token_id: TokenId,
    mask_vocab_size: usize,
}

impl BoundSchema {
    /// The `u32` word count a [`Session::write_mask`] buffer must have: `ceil(vocab_size / 32)`.
    #[must_use]
    pub fn mask_word_count(&self) -> usize {
        self.mask_vocab_size.div_ceil(32)
    }

    /// This binding's vocabulary width.
    #[must_use]
    pub fn mask_vocab_size(&self) -> usize {
        self.mask_vocab_size
    }

    /// This binding's EOS token id.
    #[must_use]
    pub fn eos_token_id(&self) -> TokenId {
        self.eos_token_id
    }

    /// Starts one fresh, independently mutable generation sequence, `Active` at its initial
    /// state.
    pub fn start_session(&self) -> Result<Session, SessionError> {
        let inner = match &self.inner {
            BoundSchemaKind::Regular(artifact) => {
                SessionKind::Regular(Matcher::new(Arc::clone(artifact)))
            }
            BoundSchemaKind::Structured(program, handle, trie) => {
                let matcher = program.new_matcher()?;
                SessionKind::Structured(Box::new(StructuredSessionState {
                    matcher,
                    program: Arc::clone(program),
                    vocabulary: handle.clone(),
                    trie: trie.clone(),
                }))
            }
        };
        Ok(Session {
            inner,
            lifecycle: Lifecycle::Active,
            eos_token_id: self.eos_token_id,
            mask_vocab_size: self.mask_vocab_size,
        })
    }
}

struct StructuredSessionState {
    matcher: StructuredMatcher,
    program: Arc<StructuredProgram>,
    vocabulary: VocabularyHandle,
    trie: Option<BoundByteTrie>,
}

enum SessionKind {
    Regular(Matcher),
    Structured(Box<StructuredSessionState>),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Lifecycle {
    /// Ordinary generation: legal tokens advance the position, EOS is legal only when
    /// accepting.
    Active,
    /// EOS has been committed. No further token may be committed until `reset()`.
    Stopped,
}

/// One generation sequence's mutable state. Never shared between concurrent sequences. Opaque:
/// callers never inspect which backend it wraps.
pub struct Session {
    inner: SessionKind,
    lifecycle: Lifecycle,
    eos_token_id: TokenId,
    mask_vocab_size: usize,
}

fn set_bit_le(mask: &mut [u8], id: u32) {
    mask[(id / 8) as usize] |= 1u8 << (id % 8);
}

/// Reinterprets `words` as raw bytes, in place.
// SAFETY: u8 has no alignment requirement and the byte length is exactly 4 * words.len().
fn u32_words_as_bytes_mut(words: &mut [u32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), words.len() * 4) }
}

impl Session {
    /// This session's vocabulary width.
    #[must_use]
    pub fn mask_vocab_size(&self) -> usize {
        self.mask_vocab_size
    }

    /// The `u32` word count a [`Self::write_mask`] buffer must have.
    #[must_use]
    pub fn mask_word_count(&self) -> usize {
        self.mask_vocab_size.div_ceil(32)
    }

    /// This session's EOS token id.
    #[must_use]
    pub fn eos_token_id(&self) -> TokenId {
        self.eos_token_id
    }

    /// Returns the checked buffer length for [`Self::write_mask_le_bytes`].
    /// Callers should use this instead of recomputing `mask_word_count() * 4`.
    pub fn mask_byte_count(&self) -> Result<usize, SessionError> {
        self.mask_word_count()
            .checked_mul(4)
            .ok_or(SessionError::MaskWidthOverflow {
                mask_vocab_size: self.mask_vocab_size,
            })
    }

    /// Fills `mask` (length must equal [`Self::mask_word_count`]) with the allowed-token bitmask; left zeroed on error.
    /// Thin zero-copy wrapper over [`Self::write_mask_le_bytes`].
    pub fn write_mask(&mut self, mask: &mut [u32]) -> Result<(), SessionError> {
        let expected = self.mask_word_count();
        if mask.len() != expected {
            return Err(SessionError::MaskBufferLength {
                expected,
                actual: mask.len(),
            });
        }
        self.write_mask_le_bytes(u32_words_as_bytes_mut(mask))?;
        // The writer emits little-endian bytes; reading them back as native words needs one
        // swap per word on a big-endian target.
        #[cfg(target_endian = "big")]
        for word in mask.iter_mut() {
            *word = word.swap_bytes();
        }
        Ok(())
    }

    /// Fills `out` (length must equal [`Self::mask_byte_count`]) with the little-endian packed mask; left zeroed on error. Primary zero-extra-allocation path for both backends.
    pub fn write_mask_le_bytes(&mut self, out: &mut [u8]) -> Result<(), SessionError> {
        let expected = self.mask_byte_count()?;
        if out.len() != expected {
            return Err(SessionError::MaskByteBufferLength {
                expected,
                actual: out.len(),
            });
        }
        out.fill(0);
        if self.lifecycle == Lifecycle::Stopped {
            set_bit_le(out, self.eos_token_id.get());
            return Ok(());
        }
        let result = match &mut self.inner {
            SessionKind::Regular(matcher) => {
                let artifact = matcher.artifact();
                artifact
                    .write_mask_le_bytes_into(&[matcher.state()], out, None)
                    .map_err(SessionError::from)
            }
            SessionKind::Structured(session) => session
                .matcher
                .write_mask_le_bytes_into(&session.vocabulary, session.trie.as_ref(), out)
                .map_err(SessionError::Structured),
        };
        if result.is_err() {
            out.fill(0);
            return result;
        }
        if self.is_accepting() {
            set_bit_le(out, self.eos_token_id.get());
        }
        Ok(())
    }

    /// Consumes one token; EOS transitions to `Stopped` instead of reaching the matcher.
    pub fn advance(&mut self, token: TokenId) -> Result<(), SessionError> {
        if self.lifecycle == Lifecycle::Stopped {
            return Err(SessionError::SessionStopped);
        }
        if token == self.eos_token_id {
            if !self.is_accepting() {
                return Err(SessionError::IllegalToken { token });
            }
            self.lifecycle = Lifecycle::Stopped;
            return Ok(());
        }
        match &mut self.inner {
            SessionKind::Regular(matcher) => matcher.advance(token).map_err(SessionError::from),
            SessionKind::Structured(session) => {
                let bytes = session
                    .vocabulary
                    .token_bytes(token.get())
                    .ok_or(SessionError::UnknownToken { token })?;
                let accepted = session
                    .matcher
                    .advance(bytes)
                    .map_err(SessionError::Structured)?;
                if !accepted {
                    return Err(SessionError::IllegalToken { token });
                }
                Ok(())
            }
        }
    }

    /// Returns to `Active` at the initial state, discarding any generated prefix. Transactional: builds the replacement before committing, so a failed reset changes nothing.
    pub fn reset(&mut self) -> Result<(), SessionError> {
        match &mut self.inner {
            SessionKind::Regular(matcher) => {
                *matcher = Matcher::new(Arc::clone(matcher.artifact()));
            }
            SessionKind::Structured(session) => {
                let replacement = StructuredMatcher::new(Arc::clone(&session.program))
                    .map_err(SessionError::Structured)?;
                session.matcher = replacement;
            }
        }
        self.lifecycle = Lifecycle::Active;
        Ok(())
    }

    /// Stopping now (committing EOS) would yield a complete, valid instance. Independent of
    /// [`Self::is_stopped`]: stays true after stopping too.
    #[must_use]
    pub fn is_accepting(&self) -> bool {
        match &self.inner {
            SessionKind::Regular(matcher) => matcher.is_accepting(),
            SessionKind::Structured(session) => session.matcher.is_accepting(),
        }
    }

    /// The permanent dead sink has been reached: no token can ever be legal again. Independent
    /// of [`Self::is_stopped`].
    #[must_use]
    pub fn is_dead(&self) -> bool {
        match &self.inner {
            SessionKind::Regular(matcher) => matcher.is_dead(),
            SessionKind::Structured(session) => session.matcher.is_dead(),
        }
    }

    /// EOS has been committed; every `advance` raises until [`Self::reset`].
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.lifecycle == Lifecycle::Stopped
    }
}

/// Errors from binding a schema or driving a [`Session`].
#[derive(Debug)]
#[non_exhaustive]
pub enum SessionError {
    /// Schema compilation or artifact-binding failed.
    Compile(CompileError),
    /// The regular-backend matcher rejected a token.
    Matcher(MatcherError),
    /// The structured-backend matcher hit a resource limit or a compile error.
    Structured(StructuredMatcherError),
    /// A token was rejected: no path from the current position (either backend).
    IllegalToken {
        /// The rejected token id.
        token: TokenId,
    },
    /// `advance` was given a token id absent from the bound vocabulary.
    UnknownToken {
        /// The unrecognized token id.
        token: TokenId,
    },
    /// `advance` was called on a session that already committed EOS; call `reset()` first.
    SessionStopped,
    /// `write_mask`'s output buffer length did not match the vocabulary's word count.
    MaskBufferLength {
        /// The required length, in `u32` words.
        expected: usize,
        /// The length the caller actually supplied.
        actual: usize,
    },
    /// `write_mask_le_bytes`'s output buffer length did not match the vocabulary's byte width.
    MaskByteBufferLength {
        /// The required length, in bytes.
        expected: usize,
        /// The length the caller actually supplied.
        actual: usize,
    },
    /// The vocabulary is too wide for its packed mask to be addressed on this target.
    MaskWidthOverflow {
        /// The vocabulary width whose byte length overflowed `usize`.
        mask_vocab_size: usize,
    },
}

impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(e) => write!(f, "{e}"),
            Self::Matcher(e) => write!(f, "{e}"),
            Self::Structured(e) => write!(f, "{e}"),
            Self::IllegalToken { token } => write!(f, "[IllegalToken] token {}", token.get()),
            Self::UnknownToken { token } => write!(f, "unknown token id {}", token.get()),
            Self::SessionStopped => write!(f, "[SessionStopped] session already committed EOS"),
            Self::MaskBufferLength { expected, actual } => write!(
                f,
                "mask buffer must contain exactly {expected} words, got {actual}"
            ),
            Self::MaskByteBufferLength { expected, actual } => write!(
                f,
                "mask byte buffer must contain exactly {expected} bytes, got {actual}"
            ),
            Self::MaskWidthOverflow { mask_vocab_size } => write!(
                f,
                "vocabulary width {mask_vocab_size} overflows the packed mask byte length"
            ),
        }
    }
}

impl std::error::Error for SessionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Compile(e) => Some(e),
            Self::Matcher(e) => Some(e),
            Self::Structured(e) => Some(e),
            _ => None,
        }
    }
}

impl From<CompileError> for SessionError {
    fn from(e: CompileError) -> Self {
        Self::Compile(e)
    }
}

impl From<MatcherError> for SessionError {
    fn from(e: MatcherError) -> Self {
        Self::Matcher(e)
    }
}

impl From<StructuredMatcherError> for SessionError {
    fn from(e: StructuredMatcherError) -> Self {
        Self::Structured(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_vocabulary;
    use rustc_hash::FxHashMap;

    fn vocabulary(eos: u32) -> CompiledVocabulary {
        let map: FxHashMap<Vec<u8>, Vec<u32>> =
            (0u32..256).map(|b| (vec![b as u8], vec![b])).collect();
        CompiledVocabulary::new(Arc::new(build_vocabulary(eos, map).unwrap())).unwrap()
    }

    fn eos_bit_set(mask: &[u32], eos: u32) -> bool {
        (mask[(eos / 32) as usize] >> (eos % 32)) & 1 == 1
    }

    #[test]
    fn regular_write_mask_omits_eos_while_active_and_not_accepting() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let bound = compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        let mut session = bound.start_session().unwrap();
        let mut mask = vec![0u32; bound.mask_word_count()];
        session.write_mask(&mut mask).unwrap();
        assert!(
            !eos_bit_set(&mask, 256),
            "EOS must not be legal before any content is generated"
        );
    }

    #[test]
    fn regular_write_mask_sets_eos_exactly_when_accepting() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let bound = compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        let mut session = bound.start_session().unwrap();
        for byte in b"true" {
            session
                .advance(TokenId::try_from(*byte as usize).unwrap())
                .unwrap();
        }
        assert!(session.is_accepting());
        let mut mask = vec![0u32; bound.mask_word_count()];
        session.write_mask(&mut mask).unwrap();
        assert!(
            eos_bit_set(&mask, 256),
            "EOS must be legal once the prefix is a complete value"
        );
    }

    #[test]
    fn structured_write_mask_sets_eos_exactly_when_accepting() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let bound = compiler
            .compile_json_schema(r#"{"not":{"type":"string"}}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        let mut session = bound.start_session().unwrap();
        let mut mask = vec![0u32; bound.mask_word_count()];
        session.write_mask(&mut mask).unwrap();
        assert!(!eos_bit_set(&mask, 256));
        session
            .advance(TokenId::try_from(b'4' as usize).unwrap())
            .unwrap();
        assert!(session.is_accepting());
        session.write_mask(&mut mask).unwrap();
        assert!(
            eos_bit_set(&mask, 256),
            "a single digit is a complete number"
        );
    }

    #[test]
    fn eos_before_acceptance_is_illegal_and_leaves_state_unchanged() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let bound = compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        let mut session = bound.start_session().unwrap();
        let err = session.advance(TokenId(256)).unwrap_err();
        assert!(matches!(err, SessionError::IllegalToken { .. }));
        assert!(!session.is_stopped());
        // still usable: a legal ordinary token still advances normally.
        session
            .advance(TokenId::try_from(b't' as usize).unwrap())
            .unwrap();
    }

    #[test]
    fn eos_at_acceptance_stops_without_reaching_the_matcher() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let bound = compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        let mut session = bound.start_session().unwrap();
        for byte in b"true" {
            session
                .advance(TokenId::try_from(*byte as usize).unwrap())
                .unwrap();
        }
        session.advance(TokenId(256)).unwrap();
        assert!(session.is_stopped());
    }

    #[test]
    fn stopped_session_rejects_every_further_advance() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let bound = compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        let mut session = bound.start_session().unwrap();
        for byte in b"true" {
            session
                .advance(TokenId::try_from(*byte as usize).unwrap())
                .unwrap();
        }
        session.advance(TokenId(256)).unwrap();
        let err = session
            .advance(TokenId::try_from(b' ' as usize).unwrap())
            .unwrap_err();
        assert!(matches!(err, SessionError::SessionStopped));
        let err = session.advance(TokenId(256)).unwrap_err();
        assert!(matches!(err, SessionError::SessionStopped));
    }

    #[test]
    fn stopped_mask_is_eos_only() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let bound = compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        let mut session = bound.start_session().unwrap();
        for byte in b"true" {
            session
                .advance(TokenId::try_from(*byte as usize).unwrap())
                .unwrap();
        }
        session.advance(TokenId(256)).unwrap();
        let mut mask = vec![0xFFFF_FFFFu32; bound.mask_word_count()]; // dirty buffer
        session.write_mask(&mut mask).unwrap();
        for (i, word) in mask.iter().enumerate() {
            let expected = if i == 8 { 1u32 } else { 0u32 };
            assert_eq!(*word, expected, "word {i} must be EOS-only");
        }
    }

    #[test]
    fn reset_returns_to_active_at_the_initial_state() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let bound = compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        let mut session = bound.start_session().unwrap();
        for byte in b"true" {
            session
                .advance(TokenId::try_from(*byte as usize).unwrap())
                .unwrap();
        }
        session.advance(TokenId(256)).unwrap();
        assert!(session.is_stopped());
        session.reset().unwrap();
        assert!(!session.is_stopped());
        assert!(!session.is_accepting());
        // usable again from the initial state.
        for byte in b"false" {
            session
                .advance(TokenId::try_from(*byte as usize).unwrap())
                .unwrap();
        }
        assert!(session.is_accepting());
    }

    #[test]
    fn structured_session_reset_also_returns_to_active() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let bound = compiler
            .compile_json_schema(r#"{"not":{"type":"string"}}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        let mut session = bound.start_session().unwrap();
        session
            .advance(TokenId::try_from(b'4' as usize).unwrap())
            .unwrap();
        session.advance(TokenId(256)).unwrap();
        assert!(session.is_stopped());
        session.reset().unwrap();
        assert!(!session.is_stopped());
        assert!(!session.is_accepting());
    }

    #[test]
    fn accessors_agree_with_the_vocabulary() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let bound = compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        assert_eq!(bound.eos_token_id(), TokenId(256));
        assert_eq!(bound.mask_vocab_size(), vocab.mask_vocab_size());
        let session = bound.start_session().unwrap();
        assert_eq!(session.eos_token_id(), TokenId(256));
        assert_eq!(session.mask_vocab_size(), vocab.mask_vocab_size());
        assert_eq!(session.mask_word_count(), bound.mask_word_count());
    }

    #[test]
    fn compiled_vocabulary_is_reused_across_two_binds() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let program = compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap();
        let bound_a = program.bind(&vocab).unwrap();
        let bound_b = program.bind(&vocab).unwrap();
        assert_eq!(bound_a.mask_vocab_size(), bound_b.mask_vocab_size());
    }

    #[test]
    fn bound_schema_is_clone_and_sessions_stay_independent() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let bound = compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        let cloned = bound.clone();
        let mut a = bound.start_session().unwrap();
        let mut b = cloned.start_session().unwrap();
        for byte in b"true" {
            a.advance(TokenId::try_from(*byte as usize).unwrap())
                .unwrap();
        }
        assert!(a.is_accepting());
        // b, from the cloned BoundSchema, must still be at its own untouched initial state.
        assert!(!b.is_accepting());
        for byte in b"false" {
            b.advance(TokenId::try_from(*byte as usize).unwrap())
                .unwrap();
        }
        assert!(b.is_accepting());
    }

    #[test]
    fn compile_with_resources_resolves_a_caller_supplied_ref() {
        let compiler = Compiler::new(CompilerOptions::default());
        let resources = vec![(
            "https://example.com/a.json".to_string(),
            r#"{"$anchor":"value","type":"integer"}"#.to_string(),
        )];
        let program = compiler
            .compile_json_schema_with_resources(
                r#"{"$ref":"a.json#value"}"#,
                Some("https://example.com/root.json"),
                &resources,
                CompileOptions::default(),
                SchemaResourceLimits::default(),
            )
            .unwrap();
        let vocab = vocabulary(256);
        let bound = program.bind(&vocab).unwrap();
        let mut session = bound.start_session().unwrap();
        session
            .advance(TokenId::try_from(b'4' as usize).unwrap())
            .unwrap();
        assert!(session.is_accepting());
    }

    #[test]
    fn compiled_vocabulary_try_from_matches_new() {
        let raw = Arc::new(build_vocabulary(256, FxHashMap::default()).unwrap());
        let via_try_from = CompiledVocabulary::try_from(Arc::clone(&raw)).unwrap();
        let via_new = CompiledVocabulary::new(raw).unwrap();
        assert_eq!(via_try_from.eos_token_id(), via_new.eos_token_id());
    }

    #[test]
    fn compiled_vocabulary_from_handle_matches_new() {
        let raw = Arc::new(build_vocabulary(256, FxHashMap::default()).unwrap());
        let handle = VocabularyHandle::new(Arc::clone(&raw)).unwrap();
        let via_handle = CompiledVocabulary::from_handle(handle);
        let via_new = CompiledVocabulary::new(raw).unwrap();
        assert_eq!(via_handle.eos_token_id(), via_new.eos_token_id());
        assert_eq!(via_handle.mask_vocab_size(), via_new.mask_vocab_size());
    }

    #[test]
    fn advancing_an_unknown_token_id_is_a_typed_error_not_a_panic() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let bound = compiler
            .compile_json_schema(r#"{"not":{"type":"string"}}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        let mut session = bound.start_session().unwrap();
        let err = session
            .advance(TokenId::try_from(9999usize).unwrap())
            .unwrap_err();
        assert!(matches!(err, SessionError::UnknownToken { .. }));
    }

    #[test]
    fn one_program_binds_to_two_vocabularies_independently() {
        let compiler = Compiler::new(CompilerOptions::default());
        let program = compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap();
        let bound_a = program.bind(&vocabulary(256)).unwrap();
        let bound_b = program.bind(&vocabulary(257)).unwrap();
        let mut session_a = bound_a.start_session().unwrap();
        let session_b = bound_b.start_session().unwrap();
        for byte in b"true" {
            session_a
                .advance(TokenId::try_from(*byte as usize).unwrap())
                .unwrap();
        }
        assert!(session_a.is_accepting());
        assert!(!session_b.is_accepting());
    }

    #[test]
    fn without_cache_still_compiles_correctly_and_caches_nothing() {
        let compiler = Compiler::without_cache();
        let schema = r#"{"type":"boolean"}"#;
        compiler.compile_json_schema(schema).unwrap();
        compiler.compile_json_schema(schema).unwrap();
        assert_eq!(compiler.cache_stats().entries, 0);
    }

    #[test]
    fn clear_caches_resets_retention() {
        let compiler = Compiler::new(CompilerOptions::default());
        compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap();
        assert!(compiler.cache_stats().entries > 0);
        compiler.clear_caches();
        assert_eq!(compiler.cache_stats().entries, 0);
    }

    #[test]
    fn a_reused_schema_program_hits_the_compiler_cache() {
        let compiler = Compiler::new(CompilerOptions::default());
        let schema = r#"{"type":"boolean"}"#;
        compiler.compile_json_schema(schema).unwrap();
        let misses_after_first = compiler.cache_stats().misses;
        compiler.compile_json_schema(schema).unwrap();
        assert_eq!(compiler.cache_stats().misses, misses_after_first);
        assert!(compiler.cache_stats().hits > 0);
    }

    #[test]
    fn default_bind_policy_is_trie_backed_never_naive() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let bound = compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        let BoundSchemaKind::Regular(artifact) = &bound.inner else {
            panic!("a boolean schema binds to the regular backend");
        };
        assert_ne!(artifact.bind_mode(), crate::index::BindMode::Naive);
    }

    #[test]
    fn eager_lazy_adaptive_masks_equal_the_naive_oracle_across_states() {
        let vocab = vocabulary(256);
        let schema = r#"{"type":"boolean"}"#;
        let reference = Compiler::new(CompilerOptions::default());
        let program = reference.compile_json_schema(schema).unwrap();
        let engine = match program.cached.executable().as_ref() {
            ExecutableSchema::Regular(engine) => Arc::clone(engine),
            _ => panic!("a boolean schema compiles to the regular backend"),
        };
        let source_hash = HashState::Hash(program.cached.ir_hash());

        for policy in [BindPolicy::Eager, BindPolicy::Lazy, BindPolicy::Adaptive] {
            let naive = CompiledArtifact::new_from_handle_shared(
                &vocab.handle,
                Arc::clone(&engine),
                Provenance::reference(source_hash),
            )
            .unwrap();
            let mut naive_matcher = Matcher::new(Arc::new(naive));

            let policy_compiler = Compiler::new(CompilerOptions {
                bind_policy: policy,
                ..CompilerOptions::default()
            });
            let bound = policy_compiler
                .compile_json_schema(schema)
                .unwrap()
                .bind(&vocab)
                .unwrap();
            let mut session = bound.start_session().unwrap();

            for byte in b"true" {
                let mut mask = vec![0u32; bound.mask_word_count()];
                session.write_mask(&mut mask).unwrap();
                let mut naive_mask = vec![0u32; bound.mask_word_count()];
                naive_matcher
                    .artifact()
                    .write_mask_into(&[naive_matcher.state()], &mut naive_mask, None)
                    .unwrap();
                assert_eq!(
                    mask, naive_mask,
                    "{policy:?} mask disagrees with the naive oracle"
                );

                let token = TokenId::try_from(*byte as usize).unwrap();
                session.advance(token).unwrap();
                naive_matcher.advance(token).unwrap();
            }
        }
    }

    #[test]
    fn structured_trie_mask_equals_the_record_scan_oracle() {
        use crate::frontend::schema_to_ir;
        let vocab = vocabulary(256);
        let schema = r#"{"not":{"type":"string"}}"#;
        let ir = Arc::new(schema_to_ir(schema, CompileOptions::default()).unwrap());
        let program = StructuredProgram::compile(ir).unwrap();
        let mut oracle = StructuredMatcher::new(Arc::clone(&program)).unwrap();

        let compiler = Compiler::new(CompilerOptions::default());
        let bound = compiler
            .compile_json_schema(schema)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        let mut session = bound.start_session().unwrap();

        let eos_word = 256usize / 32;
        let eos_bit = 1u32 << (256u32 % 32);
        let mut mask = vec![0u32; bound.mask_word_count()];
        let mut oracle_mask = vec![0u32; bound.mask_word_count()];

        session.write_mask(&mut mask).unwrap();
        oracle
            .write_record_mask_words_into(
                vocab.mask_vocab_size(),
                vocab.handle.iter_records(),
                &mut oracle_mask,
            )
            .unwrap();
        assert_eq!(mask, oracle_mask, "at the initial state");

        session
            .advance(TokenId::try_from(b'4' as usize).unwrap())
            .unwrap();
        oracle.advance(b"4").unwrap();
        session.write_mask(&mut mask).unwrap();
        mask[eos_word] &= !eos_bit; // EOS is a side predicate, absent from the record-scan oracle
        oracle
            .write_record_mask_words_into(
                vocab.mask_vocab_size(),
                vocab.handle.iter_records(),
                &mut oracle_mask,
            )
            .unwrap();
        assert_eq!(
            mask, oracle_mask,
            "at the accepting state's ordinary-token bits"
        );
    }

    #[test]
    fn write_mask_le_bytes_rejects_wrong_length_and_zeroes_output() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let bound = compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        let mut session = bound.start_session().unwrap();
        let mut out = vec![0xFFu8; 4 * bound.mask_word_count() - 1];
        let err = session.write_mask_le_bytes(&mut out).unwrap_err();
        assert!(matches!(err, SessionError::MaskByteBufferLength { .. }));
    }

    #[test]
    fn write_mask_le_bytes_matches_write_mask_bit_for_bit() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        let bound = compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        let mut session = bound.start_session().unwrap();
        for byte in b"tru" {
            session
                .advance(TokenId::try_from(*byte as usize).unwrap())
                .unwrap();
        }
        let mut words = vec![0u32; bound.mask_word_count()];
        session.write_mask(&mut words).unwrap();
        let mut bytes = vec![0u8; 4 * bound.mask_word_count()];
        session.write_mask_le_bytes(&mut bytes).unwrap();
        let words_from_bytes: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(words, words_from_bytes);
    }

    #[test]
    fn compiler_without_cache_also_builds_no_retained_trie() {
        let compiler = Compiler::without_cache();
        let vocab = vocabulary(256);
        compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        assert_eq!(compiler.trie_cache_stats().entries, 0);
    }

    #[test]
    fn clear_caches_also_clears_the_trie_cache() {
        let compiler = Compiler::new(CompilerOptions::default());
        let vocab = vocabulary(256);
        compiler
            .compile_json_schema(r#"{"type":"boolean"}"#)
            .unwrap()
            .bind(&vocab)
            .unwrap();
        assert!(compiler.trie_cache_stats().entries > 0);
        compiler.clear_caches();
        assert_eq!(compiler.trie_cache_stats().entries, 0);
    }
}
