//! The PyO3 wall. `PyIndex` shares a read-only [`CompiledArtifact`] by `Arc` and exposes mask fill
//! plus state queries; errors cross as the structured `MaskforgeError` 7-tuple (`message` advisory).

use std::sync::{Arc, OnceLock};

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyByteArrayMethods, PyBytes, PyBytesMethods};
#[cfg(feature = "test-utils")]
use rustc_hash::FxHashMap;

#[cfg(feature = "test-utils")]
use maskforge_core::build_vocabulary;
#[cfg(feature = "test-utils")]
use maskforge_core::correctness::corpus::corpus_cases;
use maskforge_core::error::{
    CompileError, ErrorCode, HashState, MatcherError, Provenance, Stage, VocabError,
};
#[cfg(feature = "bench-internals")]
use maskforge_core::frontend::schema_to_ir_with_resources_profiled;
use maskforge_core::index::{BindMode as CoreBindMode, BoundByteTrie, TrieCache, VocabularyHandle};
use maskforge_core::runtime::{global_executable_cache, ArtifactCache, CachedExecutable};
use maskforge_core::{
    compile_ir as core_compile_ir, regex_to_ir, schema_to_ir as core_schema_to_ir,
    schema_to_ir_with_resources as core_schema_to_ir_with_resources, BindPolicy, BoundSchema,
    CompileOptions, CompiledArtifact, CompiledVocabulary, Compiler, CompilerOptions, ObjectClosure,
    SchemaIR, SchemaProgram, SchemaRegistry, SchemaResourceLimits, Session, SessionError, StateId,
    StructuredMatcher, StructuredMatcherError, StructuredProgram, TokenId, Vocabulary,
};

pyo3::create_exception!(
    _native,
    MaskforgeError,
    PyException,
    "A MaskForge error. `args` is always the 7-tuple `(code, stage, pointer, keyword, observed, \
     message, limit)`; every field but `code`, `stage`, and `message` may be `None`. `code` is \
     the stable machine contract; `message` is advisory text."
);

/// Builds the stable `(code, stage, pointer, keyword, observed, message, limit)` error tuple every
/// raise site uses; absent fields are `None`.
#[allow(clippy::too_many_arguments)]
fn structured_err(
    code: String,
    stage: String,
    pointer: Option<String>,
    keyword: Option<String>,
    observed: Option<String>,
    message: String,
    limit: Option<String>,
) -> PyErr {
    MaskforgeError::new_err((code, stage, pointer, keyword, observed, message, limit))
}

fn compile_err(e: &CompileError) -> PyErr {
    let limit = e
        .limit
        .map(|(kind, observed, cap)| format!("{kind:?}:{observed}/{cap}"));
    structured_err(
        format!("{:?}", e.code),
        format!("{:?}", e.stage),
        e.json_pointer_path.clone(),
        e.keyword.clone(),
        e.observed.clone(),
        e.message.to_string(),
        limit,
    )
}

fn structured_matcher_err(e: &StructuredMatcherError) -> PyErr {
    match e {
        StructuredMatcherError::Compile(error) => compile_err(error),
        StructuredMatcherError::ResourceLimit {
            kind,
            observed,
            limit,
        } => structured_err(
            format!("{:?}", ErrorCode::InternalLimitExceeded),
            format!("{:?}", Stage::L4Bind),
            None,
            None,
            Some(observed.to_string()),
            "structured matcher resource limit exceeded".to_string(),
            Some(format!("{kind:?}:{observed}/{limit}")),
        ),
    }
}

fn matcher_err(e: &MatcherError) -> PyErr {
    match e {
        MatcherError::IllegalToken { token } => illegal_token(token.get()),
        MatcherError::ArtifactMismatch => structured_err(
            "ArtifactMismatch".to_string(),
            format!("{:?}", Stage::ArtifactValidate),
            None,
            None,
            None,
            "artifact mismatch".to_string(),
            None,
        ),
        _ => structured_err(
            "Unsupported".to_string(),
            format!("{:?}", Stage::ArtifactValidate),
            None,
            None,
            None,
            e.to_string(),
            None,
        ),
    }
}

/// `SessionError` is `#[non_exhaustive]`, so this match needs a wildcard even though every
/// variant known today is listed.
fn session_err(e: &SessionError) -> PyErr {
    match e {
        SessionError::Compile(err) => compile_err(err),
        SessionError::Matcher(err) => matcher_err(err),
        SessionError::Structured(err) => structured_matcher_err(err),
        SessionError::IllegalToken { token } => illegal_token(token.get()),
        SessionError::UnknownToken { token } => structured_err(
            "UnknownToken".to_string(),
            format!("{:?}", Stage::ArtifactValidate),
            None,
            None,
            Some(token.get().to_string()),
            format!("token {} is absent from the bound vocabulary", token.get()),
            None,
        ),
        SessionError::SessionStopped => structured_err(
            "SessionStopped".to_string(),
            format!("{:?}", Stage::ArtifactValidate),
            None,
            None,
            None,
            "session already committed EOS; call reset() first".to_string(),
            None,
        ),
        SessionError::MaskBufferLength { expected, actual } => structured_err(
            "MaskBufferLength".to_string(),
            format!("{:?}", Stage::ArtifactValidate),
            None,
            None,
            Some(actual.to_string()),
            format!("mask buffer must contain exactly {expected} words, got {actual}"),
            Some(expected.to_string()),
        ),
        SessionError::MaskByteBufferLength { expected, actual } => structured_err(
            "MaskBufferLength".to_string(),
            format!("{:?}", Stage::ArtifactValidate),
            None,
            None,
            Some(actual.to_string()),
            format!("mask buffer must contain exactly {expected} bytes, got {actual}"),
            Some(expected.to_string()),
        ),
        _ => structured_err(
            "Unsupported".to_string(),
            format!("{:?}", Stage::ArtifactValidate),
            None,
            None,
            None,
            e.to_string(),
            None,
        ),
    }
}

fn vocab_err(e: &VocabError) -> PyErr {
    structured_err(
        format!("{:?}", e.code),
        format!("{:?}", Stage::VocabBuild),
        None,
        None,
        e.token.map(|t| t.get().to_string()),
        e.detail.to_string(),
        None,
    )
}

fn illegal_token(token: u32) -> PyErr {
    structured_err(
        "IllegalToken".to_string(),
        format!("{:?}", Stage::ArtifactValidate),
        None,
        None,
        Some(token.to_string()),
        format!("token {token}"),
        None,
    )
}

/// Pure-Rust bind errors cross `Python::detach` and become Python errors after reacquiring the GIL.
#[derive(Debug)]
enum CoreError {
    Compile(CompileError),
    Vocab(VocabError),
    Unsupported(String),
}

impl From<CompileError> for CoreError {
    fn from(e: CompileError) -> Self {
        Self::Compile(e)
    }
}

impl From<VocabError> for CoreError {
    fn from(e: VocabError) -> Self {
        Self::Vocab(e)
    }
}

fn core_err(e: &CoreError) -> PyErr {
    match e {
        CoreError::Compile(c) => compile_err(c),
        CoreError::Vocab(v) => vocab_err(v),
        CoreError::Unsupported(msg) => structured_err(
            "Unsupported".to_string(),
            "L4Bind".to_string(),
            None,
            None,
            None,
            msg.clone(),
            None,
        ),
    }
}

/// Narrows FFI errors to the core cache's error type.
fn core_error_to_compile_error(e: CoreError) -> CompileError {
    match e {
        CoreError::Compile(c) => c,
        CoreError::Vocab(v) => CompileError::new(v.code, Stage::VocabBuild, v.detail),
        CoreError::Unsupported(_) => {
            CompileError::new(ErrorCode::Unsupported, Stage::L4Bind, "unsupported bind")
        }
    }
}

/// A read-only, frozen handle to a compiled artifact. State and token ids cross the wall as `u32`;
/// a forged state id is DEAD-equivalent, never a panic.
#[pyclass(frozen, module = "maskforge._native")]
pub struct PyIndex {
    inner: Arc<CompiledArtifact>,
}

#[pymethods]
impl PyIndex {
    /// The anchored start state.
    fn start(&self) -> u32 {
        self.inner.start().get()
    }

    /// The canonical DEAD state.
    fn dead(&self) -> u32 {
        self.inner.dead().get()
    }

    /// Whether the language accepts nothing (the start state is DEAD).
    fn start_is_dead(&self) -> bool {
        self.inner.start_is_dead()
    }

    /// The token-id space the mask covers (`max_token_id + 1`).
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// The number of packed `u32` words per mask row.
    fn words_per_row(&self) -> usize {
        self.inner.words_per_row()
    }

    /// The bind mode this index was compiled under, as its public mode string (`byte_trie`,
    /// `byte_trie_lazy`, or `naive`). Lets a caller confirm the default path is the packed `byte_trie`.
    fn bind_mode(&self) -> &'static str {
        match self.inner.bind_mode() {
            CoreBindMode::TrieJointBytePacked => "byte_trie",
            CoreBindMode::TrieJointByteLazy => "byte_trie_lazy",
            CoreBindMode::Naive => "naive",
            _ => "internal",
        }
    }

    /// Packed mask bytes retained by this artifact. Lazy and naive artifacts return zero.
    fn retained_bytes(&self) -> usize {
        self.inner.packed_retained_bytes()
    }

    /// Stopping at `state` yields a complete, valid instance.
    fn is_accepting(&self, state: u32) -> bool {
        self.inner.is_accepting(StateId(state))
    }

    /// Some token can extend `state`.
    fn can_continue(&self, state: u32) -> bool {
        self.inner.can_continue(StateId(state))
    }

    /// `state` is the permanent dead sink.
    fn is_dead(&self, state: u32) -> bool {
        self.inner.is_dead(StateId(state))
    }

    /// EOS is legal at `state` (equivalently, it accepts). EOS is never a mask bit.
    fn eos_legal(&self, state: u32) -> bool {
        self.inner.eos_legal(StateId(state))
    }

    /// Consumes `token` from `state`, returning the next state. Raises `MaskforgeError` with code
    /// `IllegalToken` if the id is unknown or no byte path exists.
    fn advance(&self, state: u32, token: u32) -> PyResult<u32> {
        let bytes = self
            .inner
            .token_bytes(TokenId(token))
            .ok_or_else(|| illegal_token(token))?;
        self.inner
            .walk(StateId(state), bytes)
            .map(StateId::get)
            .ok_or_else(|| illegal_token(token))
    }

    /// The ascending list of token ids allowed at `state` (EOS excluded). Reads the per-state mask
    /// cache, so repeated calls on one state do not re-walk the vocabulary.
    fn allowed_ids(&self, state: u32) -> PyResult<Vec<u32>> {
        Ok(self
            .inner
            .allowed_ids(StateId(state))
            .map_err(|e| compile_err(&e))?
            .into_iter()
            .map(TokenId::get)
            .collect())
    }

    /// Returns packed little-endian mask words. `rows` optionally selects output rows.
    fn mask_words(&self, states: Vec<u32>, rows: Option<Vec<usize>>) -> PyResult<Vec<u32>> {
        let state_ids = StateId::cast_slice(&states);
        let len = self
            .inner
            .mask_buffer_len(state_ids.len(), rows.as_deref())
            .map_err(|e| compile_err(&e))?;
        let mut buf = vec![0u32; len];
        self.inner
            .write_mask_into(state_ids, &mut buf, rows.as_deref())
            .map_err(|e| compile_err(&e))?;
        Ok(buf)
    }

    /// Like `mask_words`, but fills a caller-owned little-endian `out: bytearray` (sized
    /// `4 * rows * words_per_row`, e.g. viewable as `numpy.frombuffer(out, "<u4")`) - no list of ints.
    fn mask_words_into(
        &self,
        states: Vec<u32>,
        out: &Bound<'_, PyByteArray>,
        rows: Option<Vec<usize>>,
    ) -> PyResult<()> {
        let state_ids = StateId::cast_slice(&states);
        // SAFETY: GIL held throughout and the writer is pure Rust; Python cannot resize `out`.
        let dst = unsafe { out.as_bytes_mut() };
        self.inner
            .write_mask_le_bytes_into(state_ids, dst, rows.as_deref())
            .map_err(|e| compile_err(&e))
    }

    /// Single-state fast path for `mask_words_into` - skips the `Vec<StateId>` allocation for the
    /// common one-state-per-step decoding loop.
    fn mask_word_into_state(&self, state: u32, out: &Bound<'_, PyByteArray>) -> PyResult<()> {
        // SAFETY: GIL held throughout and the writer is pure Rust; Python cannot resize `out`.
        let dst = unsafe { out.as_bytes_mut() };
        self.inner
            .write_mask_le_bytes_into(&[StateId(state)], dst, None)
            .map_err(|e| compile_err(&e))
    }

    /// Acquires a read-only packed-table view that retains the artifact independently.
    fn mask_table_view(&self) -> PyResult<PyMaskTableView> {
        if self.inner.packed_mask_table_words().is_none() {
            return Err(structured_err(
                "Unsupported".to_string(),
                "L4Bind".to_string(),
                None,
                None,
                None,
                "mask_table_view requires byte_trie (packed) bind mode".to_string(),
                None,
            ));
        }
        #[cfg(not(target_endian = "little"))]
        return Err(structured_err(
            "Unsupported".to_string(),
            "L4Bind".to_string(),
            None,
            None,
            None,
            "mask_table_view is unavailable on big-endian targets (native u32 storage would not \
             match the documented little-endian byte contract)"
                .to_string(),
            None,
        ));
        #[cfg(target_endian = "little")]
        Ok(PyMaskTableView {
            artifact: self.inner.clone(),
            row_count: self.inner.state_count(),
            words_per_row: self.inner.words_per_row(),
        })
    }
}

/// Read-only zero-copy view of a packed table. Its Arc keeps the artifact alive for the buffer.
#[pyclass(frozen, module = "maskforge._native")]
pub struct PyMaskTableView {
    artifact: Arc<CompiledArtifact>,
    row_count: usize,
    words_per_row: usize,
}

#[pymethods]
impl PyMaskTableView {
    /// The number of mask rows (states) the table covers.
    #[getter]
    fn row_count(&self) -> usize {
        self.row_count
    }

    /// Packed `u32` words per row (`ceil(vocab_size / 32)`).
    #[getter]
    fn words_per_row(&self) -> usize {
        self.words_per_row
    }

    /// Bytes per row (`4 * words_per_row`) - the stride to slice the exported buffer by.
    #[getter]
    fn bytes_per_row(&self) -> usize {
        self.words_per_row * 4
    }

    /// # Safety
    /// Standard PyO3 buffer-protocol slot; `view` is supplied by the CPython buffer machinery.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        unsafe { fill_readonly_table_view(view, flags, slf) }
    }

    /// # Safety
    /// Standard PyO3 buffer-protocol slot; frees only the `format` string this exporter allocated.
    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        unsafe {
            if !(*view).format.is_null() {
                drop(std::ffi::CString::from_raw((*view).format));
            }
        }
    }
}

/// Fills a read-only packed-table buffer view.
/// # Safety: `view` is checked and owns `slf` until release.
unsafe fn fill_readonly_table_view(
    view: *mut pyo3::ffi::Py_buffer,
    flags: std::os::raw::c_int,
    slf: Bound<'_, PyMaskTableView>,
) -> PyResult<()> {
    use pyo3::exceptions::PyBufferError;
    if view.is_null() {
        return Err(PyBufferError::new_err("view is null"));
    }
    if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE {
        return Err(PyBufferError::new_err("mask table view is read-only"));
    }
    let data: &[u8] = {
        let this = slf.get();
        let words = this
            .artifact
            .packed_mask_table_words()
            .ok_or_else(|| PyBufferError::new_err("artifact is no longer in packed bind mode"))?;
        let len = words
            .len()
            .checked_mul(4)
            .ok_or_else(|| PyBufferError::new_err("mask table byte length overflows"))?;
        // SAFETY: u8 needs no alignment and `len` is exactly 4 * words.len(); the artifact that
        // owns `words` is kept alive by the reference transferred into `view.obj` below.
        unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), len) }
    };
    let len_isize = isize::try_from(data.len())
        .map_err(|_| PyBufferError::new_err("mask table byte length exceeds isize::MAX"))?;
    // SAFETY: `view` was null-checked above and CPython guarantees it points at a live Py_buffer.
    unsafe {
        (*view).obj = slf.into_any().into_ptr(); // ref transferred to the buffer; owner stays alive
        (*view).buf = data.as_ptr() as *mut std::ffi::c_void;
        (*view).len = len_isize;
        (*view).readonly = 1;
        (*view).itemsize = 1;
        // Owned here and freed by `__releasebuffer__`, which calls `CString::from_raw`.
        (*view).format = if (flags & pyo3::ffi::PyBUF_FORMAT) == pyo3::ffi::PyBUF_FORMAT {
            std::ffi::CString::new("B").unwrap().into_raw()
        } else {
            std::ptr::null_mut()
        };
        (*view).ndim = 1;
        (*view).shape = if (flags & pyo3::ffi::PyBUF_ND) == pyo3::ffi::PyBUF_ND {
            &mut (*view).len
        } else {
            std::ptr::null_mut()
        };
        (*view).strides = if (flags & pyo3::ffi::PyBUF_STRIDES) == pyo3::ffi::PyBUF_STRIDES {
            &mut (*view).itemsize
        } else {
            std::ptr::null_mut()
        };
        (*view).suboffsets = std::ptr::null_mut();
        (*view).internal = std::ptr::null_mut();
    }
    Ok(())
}

fn options(assume_closed: bool) -> CompileOptions {
    CompileOptions {
        object_closure: if assume_closed {
            ObjectClosure::AssumeClosedProfile
        } else {
            ObjectClosure::AllowOpenProfile
        },
        ..CompileOptions::default()
    }
}

pub(crate) fn resource_limits_with_overrides(
    max_document_bytes: Option<usize>,
    max_total_input_bytes: Option<usize>,
) -> SchemaResourceLimits {
    let mut limits = SchemaResourceLimits::default();
    if let Some(value) = max_document_bytes {
        limits.max_document_bytes = value;
    }
    if let Some(value) = max_total_input_bytes {
        limits.max_total_input_bytes = value;
    }
    limits
}

fn schema_set_to_ir_owned(
    schema: String,
    retrieval_uri: Option<String>,
    mut resources: Vec<(String, String)>,
    assume_closed: bool,
    max_document_bytes: Option<usize>,
    max_total_input_bytes: Option<usize>,
) -> Result<SchemaIR, CompileError> {
    resources.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    let limits = resource_limits_with_overrides(max_document_bytes, max_total_input_bytes);
    let mut registry = SchemaRegistry::new(limits);
    for (uri, text) in resources {
        registry.insert(uri, Arc::<str>::from(text))?;
    }
    core_schema_to_ir_with_resources(
        &schema,
        retrieval_uri.as_deref(),
        options(assume_closed),
        limits,
        registry,
    )
}

#[cfg(feature = "bench-internals")]
type ProfiledSchemaIr = (SchemaIR, Vec<(&'static str, u64)>);

#[cfg(feature = "bench-internals")]
fn schema_set_to_ir_profiled_owned(
    schema: String,
    retrieval_uri: Option<String>,
    mut resources: Vec<(String, String)>,
    assume_closed: bool,
    format_assertion: bool,
    max_document_bytes: Option<usize>,
    max_total_input_bytes: Option<usize>,
) -> Result<ProfiledSchemaIr, CompileError> {
    resources.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    let limits = resource_limits_with_overrides(max_document_bytes, max_total_input_bytes);
    let mut registry = SchemaRegistry::new(limits);
    for (uri, text) in resources {
        registry.insert(uri, Arc::<str>::from(text))?;
    }
    let mut compile_options = options(assume_closed);
    compile_options.format_assertion = format_assertion;
    let (ir, profile) = schema_to_ir_with_resources_profiled(
        &schema,
        retrieval_uri.as_deref(),
        compile_options,
        limits,
        registry,
    )?;
    Ok((
        ir,
        vec![
            ("registry_insert", profile.registry_insert_ns),
            ("document_parse", profile.document_parse_ns),
            (
                "resource_uri_anchor_index",
                profile.resource_uri_anchor_index_ns,
            ),
            ("resource_canonicalize", profile.resource_canonicalize_ns),
            ("reference_resolution", profile.reference_resolution_ns),
            ("graph_freeze", profile.graph_freeze_ns),
            (
                "ir_lowering_and_construction_validation",
                profile.ir_lowering_and_construction_validation_ns,
            ),
            ("ir_revalidation", profile.ir_revalidation_ns),
            (
                "retained_graph_bytes",
                u64::try_from(profile.retained_graph_bytes).unwrap_or(u64::MAX),
            ),
            (
                "peak_resource_build_bytes",
                u64::try_from(profile.peak_resource_build_bytes).unwrap_or(u64::MAX),
            ),
        ],
    ))
}

/// Marshals owned `(bytes, [ids])` pairs into the validated core `Vocabulary`. The vacant case MOVES
/// the caller's `ids` allocation into the map instead of `or_default() + extend`'s copy.
#[cfg(feature = "test-utils")]
fn build_vocab_core(eos: u32, tokens: Vec<(Vec<u8>, Vec<u32>)>) -> Result<Vocabulary, CoreError> {
    use std::collections::hash_map::Entry;
    let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
    map.try_reserve(tokens.len())
        .map_err(|_| CoreError::Unsupported("vocabulary map allocation failed".to_string()))?;
    for (bytes, ids) in tokens {
        match map.entry(bytes) {
            Entry::Vacant(slot) => {
                slot.insert(ids); // move the caller's allocation; no fresh Vec, no copy
            }
            Entry::Occupied(mut slot) => slot.get_mut().extend(ids),
        }
    }
    Ok(build_vocabulary(eos, map)?)
}

/// Length-only pre-scan of a `(bytes, ids)` sequence: `(total_bytes, total_ids, token_count)`,
/// without copying content - lets an admission peak be computed before any Rust allocation.
fn prescan_token_totals(tokens: &Bound<'_, PyAny>) -> PyResult<(usize, usize, usize)> {
    let token_count = tokens.len()?;
    let mut total_bytes = 0usize;
    let mut total_ids = 0usize;
    for i in 0..token_count {
        let item = tokens.get_item(i)?;
        total_bytes = total_bytes
            .checked_add(item.get_item(0)?.len()?)
            .ok_or_else(|| {
                core_err(&CoreError::Unsupported(
                    "token byte total overflows".to_string(),
                ))
            })?;
        total_ids = total_ids
            .checked_add(item.get_item(1)?.len()?)
            .ok_or_else(|| {
                core_err(&CoreError::Unsupported(
                    "token id total overflows".to_string(),
                ))
            })?;
    }
    Ok((total_bytes, total_ids, token_count))
}

/// The four packed CSR buffers (see `from_packed_buffers`' wire contract) plus the permit admitted
/// for their whole build + canonical-construction peak.
type PackedTokens = (
    Vec<u8>,
    Vec<u32>,
    Vec<u32>,
    Vec<u32>,
    maskforge_core::index::VocabBuildPermit<'static>,
);

fn alloc_err(what: &'static str) -> PyErr {
    core_err(&CoreError::Unsupported(format!("{what} allocation failed")))
}

fn checked_u32(n: usize) -> PyResult<u32> {
    u32::try_from(n).map_err(|_| {
        core_err(&CoreError::Unsupported(
            "packed offset exceeds the addressable range".to_string(),
        ))
    })
}

/// Appends bytes only while the observed length stays within the admitted capacity.
fn append_bytes_component(
    obj: &Bound<'_, PyAny>,
    out: &mut Vec<u8>,
    budget: usize,
) -> PyResult<()> {
    let remaining = budget.saturating_sub(out.len());
    if let Ok(b) = obj.cast::<PyBytes>() {
        let slice = b.as_bytes();
        if slice.len() > remaining {
            return Err(core_err(&CoreError::Unsupported(TOCTOU_MSG.to_string())));
        }
        out.extend_from_slice(slice);
    } else {
        let owned: Vec<u8> = obj.extract()?;
        if owned.len() > remaining {
            return Err(core_err(&CoreError::Unsupported(TOCTOU_MSG.to_string())));
        }
        out.extend_from_slice(&owned);
    }
    Ok(())
}

/// Appends ids without an intermediate Vec and rechecks the admitted capacity before every push.
fn append_ids_component(obj: &Bound<'_, PyAny>, out: &mut Vec<u32>, budget: usize) -> PyResult<()> {
    let declared = obj.len()?;
    if declared > budget.saturating_sub(out.len()) {
        return Err(core_err(&CoreError::Unsupported(TOCTOU_MSG.to_string())));
    }
    for i in 0..declared {
        if out.len() >= budget {
            return Err(core_err(&CoreError::Unsupported(TOCTOU_MSG.to_string())));
        }
        out.push(obj.get_item(i)?.extract()?);
    }
    Ok(())
}

/// Admits the complete packed build before copying token data into CSR buffers.
fn admit_and_pack_tokens(py: Python<'_>, tokens: &Bound<'_, PyAny>) -> PyResult<PackedTokens> {
    let (total_bytes, total_ids, token_count) = prescan_token_totals(tokens)?;
    let peak =
        maskforge_core::index::direct_csr_ingestion_peak_bytes(total_bytes, total_ids, token_count)
            .ok_or_else(|| {
                core_err(&CoreError::Unsupported(
                    "packed ingestion peak overflows".to_string(),
                ))
            })?;
    let permit = py
        .detach(|| maskforge_core::index::acquire_vocab_build_permit(peak))
        .map_err(|e| core_err(&e.into()))?;

    let mut token_bytes: Vec<u8> = Vec::new();
    token_bytes
        .try_reserve_exact(total_bytes)
        .map_err(|_| alloc_err("token byte buffer"))?;
    let mut byte_offsets: Vec<u32> = Vec::new();
    byte_offsets
        .try_reserve_exact(token_count + 1)
        .map_err(|_| alloc_err("byte offsets"))?;
    let mut token_ids: Vec<u32> = Vec::new();
    token_ids
        .try_reserve_exact(total_ids)
        .map_err(|_| alloc_err("token id buffer"))?;
    let mut id_offsets: Vec<u32> = Vec::new();
    id_offsets
        .try_reserve_exact(token_count + 1)
        .map_err(|_| alloc_err("id offsets"))?;
    byte_offsets.push(0);
    id_offsets.push(0);

    for i in 0..token_count {
        let item = tokens.get_item(i)?;
        append_bytes_component(&item.get_item(0)?, &mut token_bytes, total_bytes)?;
        byte_offsets.push(checked_u32(token_bytes.len())?);

        append_ids_component(&item.get_item(1)?, &mut token_ids, total_ids)?;
        id_offsets.push(checked_u32(token_ids.len())?);
    }
    Ok((token_bytes, byte_offsets, token_ids, id_offsets, permit))
}

/// Pre-scans a dense, id-ordered decode (`decoded[i]` is `bytes` or `None`), returning
/// `(total_bytes, present_count, token_count)`.
fn prescan_dense_totals(decoded: &Bound<'_, PyAny>) -> PyResult<(usize, usize, usize)> {
    let token_count = decoded.len()?;
    let mut total_bytes = 0usize;
    let mut present_count = 0usize;
    for i in 0..token_count {
        let item = decoded.get_item(i)?;
        if !item.is_none() {
            total_bytes = total_bytes.checked_add(item.len()?).ok_or_else(|| {
                core_err(&CoreError::Unsupported(
                    "token byte total overflows".to_string(),
                ))
            })?;
            present_count += 1;
        }
    }
    Ok((total_bytes, present_count, token_count))
}

/// The dense `(token_bytes, byte_offsets, present)` triple plus the permit admitted for the whole
/// build.
type DenseTokens = (
    Vec<u8>,
    Vec<u32>,
    Vec<bool>,
    maskforge_core::index::VocabBuildPermit<'static>,
);

/// Admits the complete dense build before copying decoded tokens into CSR buffers.
fn admit_and_pack_dense_tokens(
    py: Python<'_>,
    decoded: &Bound<'_, PyAny>,
) -> PyResult<DenseTokens> {
    let (total_bytes, present_count, token_count) = prescan_dense_totals(decoded)?;
    let peak = maskforge_core::index::dense_id_ordered_ingestion_peak_bytes(
        total_bytes,
        present_count,
        token_count,
    )
    .ok_or_else(|| {
        core_err(&CoreError::Unsupported(
            "dense ingestion peak overflows".to_string(),
        ))
    })?;
    let permit = py
        .detach(|| maskforge_core::index::acquire_vocab_build_permit(peak))
        .map_err(|e| core_err(&e.into()))?;

    let mut token_bytes: Vec<u8> = Vec::new();
    token_bytes
        .try_reserve_exact(total_bytes)
        .map_err(|_| alloc_err("token byte buffer"))?;
    let mut byte_offsets: Vec<u32> = Vec::new();
    byte_offsets
        .try_reserve_exact(token_count + 1)
        .map_err(|_| alloc_err("byte offsets"))?;
    let mut present: Vec<bool> = Vec::new();
    present
        .try_reserve_exact(token_count)
        .map_err(|_| alloc_err("present flags"))?;
    byte_offsets.push(0);

    for i in 0..token_count {
        let item = decoded.get_item(i)?;
        if item.is_none() {
            present.push(false);
        } else {
            append_bytes_component(&item, &mut token_bytes, total_bytes)?;
            present.push(true);
        }
        byte_offsets.push(checked_u32(token_bytes.len())?);
    }
    Ok((token_bytes, byte_offsets, present, permit))
}

/// The input sequence grew between the admission scan and the copy - concurrent mutation of a
/// token list passed across the FFI wall is not a supported usage.
const TOCTOU_MSG: &str = "token content changed between admission and copy";

/// Hard cap on any ONE packed buffer's byte length.
const MAX_PACKED_BUFFER_BYTES: usize = 1 << 28;

/// Hard cap on the sum of all four packed buffers before decoding.
const MAX_PACKED_AGGREGATE_BYTES: usize = 1 << 28;

/// Pure length check, no buffer/allocation involved - each length against the per-buffer cap, the
/// sum against the aggregate cap. Testable with synthetic sizes, not real multi-hundred-MB objects.
fn check_packed_lengths(lens: [usize; 4]) -> Result<(), CoreError> {
    let mut total = 0usize;
    for &len in &lens {
        if len > MAX_PACKED_BUFFER_BYTES {
            return Err(CoreError::Unsupported(format!(
                "packed buffer of {len} bytes exceeds the cap of {MAX_PACKED_BUFFER_BYTES} (per buffer)"
            )));
        }
        total = total.saturating_add(len);
    }
    if total > MAX_PACKED_AGGREGATE_BYTES {
        return Err(CoreError::Unsupported(format!(
            "packed buffers total {total} bytes, which exceeds the cap of \
             {MAX_PACKED_AGGREGATE_BYTES} (aggregate)"
        )));
    }
    Ok(())
}

/// Decodes little-endian u32 data directly when contiguous, with one checked fallback copy.
fn buffer_to_u32_vec_le(
    py: Python<'_>,
    buf: &PyBuffer<u8>,
    what: &'static str,
) -> PyResult<Vec<u32>> {
    let len = buf.len_bytes();
    if len % 4 != 0 {
        return Err(core_err(&CoreError::Unsupported(format!(
            "{what} byte length {len} is not a multiple of 4"
        ))));
    }
    let mut out: Vec<u32> = Vec::new();
    out.try_reserve_exact(len / 4)
        .map_err(|_| alloc_err(what))?;
    if let Some(slice) = buf.as_slice(py) {
        out.extend(
            slice
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0].get(), c[1].get(), c[2].get(), c[3].get()])),
        );
    } else {
        let bytes = buf.to_vec(py)?;
        out.extend(
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
    }
    Ok(out)
}

/// Packed token bytes and decoded CSR arrays covered by one build permit.
type PackedBuffers = (
    (Vec<u8>, Vec<u32>, Vec<u32>, Vec<u32>),
    maskforge_core::index::VocabBuildPermit<'static>,
);

/// Admits the full four-buffer ingestion peak before any buffer is copied or decoded.
fn bounded_buffers_to_vecs(
    py: Python<'_>,
    objs: [&Bound<'_, PyAny>; 4],
) -> PyResult<PackedBuffers> {
    let bufs: Vec<PyBuffer<u8>> = objs
        .iter()
        .map(|o| PyBuffer::<u8>::get(o))
        .collect::<PyResult<_>>()?;
    let lens: [usize; 4] = bufs
        .iter()
        .map(|b| b.len_bytes())
        .collect::<Vec<_>>()
        .try_into()
        .expect("exactly 4 buffers");
    check_packed_lengths(lens).map_err(|e| core_err(&e))?;
    let [tb_len, bo_len, ti_len, io_len] = lens;
    let peak = maskforge_core::index::packed_ingestion_peak_bytes(tb_len, bo_len, ti_len, io_len)
        .ok_or_else(|| {
        core_err(&CoreError::Unsupported(
            "packed ingestion peak overflows".to_string(),
        ))
    })?;
    let permit = py
        .detach(|| maskforge_core::index::acquire_vocab_build_permit(peak))
        .map_err(|e| core_err(&e.into()))?;

    let token_bytes = bufs[0].to_vec(py)?;
    let byte_offsets = buffer_to_u32_vec_le(py, &bufs[1], "byte_offsets")?;
    let token_ids = buffer_to_u32_vec_le(py, &bufs[2], "token_ids")?;
    let id_offsets = buffer_to_u32_vec_le(py, &bufs[3], "id_offsets")?;
    Ok(((token_bytes, byte_offsets, token_ids, id_offsets), permit))
}

#[cfg(test)]
mod packed_length_tests {
    use super::{check_packed_lengths, MAX_PACKED_AGGREGATE_BYTES, MAX_PACKED_BUFFER_BYTES};

    #[test]
    fn accepts_lengths_under_both_caps() {
        assert!(check_packed_lengths([100, 100, 100, 100]).is_ok());
    }

    #[test]
    fn rejects_a_single_buffer_over_the_per_buffer_cap() {
        let err = check_packed_lengths([MAX_PACKED_BUFFER_BYTES + 1, 0, 0, 0]).unwrap_err();
        match err {
            super::CoreError::Unsupported(msg) => assert!(msg.contains("per buffer")),
            _ => panic!("expected Unsupported"),
        }
    }

    #[test]
    fn rejects_four_individually_legal_buffers_that_exceed_the_aggregate_cap() {
        let each = MAX_PACKED_AGGREGATE_BYTES / 4 + 1;
        assert!(each <= MAX_PACKED_BUFFER_BYTES, "fixture sanity");
        assert!(check_packed_lengths([each, each, each, each]).is_err());
    }

    #[test]
    fn accepts_four_buffers_exactly_at_the_aggregate_boundary() {
        let each = MAX_PACKED_AGGREGATE_BYTES / 4;
        assert!(check_packed_lengths([each, each, each, each]).is_ok());
    }
}

/// Builds a handle from decoded CSR arrays under the caller's existing permit.
#[allow(clippy::too_many_arguments)]
fn build_handle_from_packed_core(
    token_bytes: &[u8],
    byte_offsets: &[u32],
    token_ids: &[u32],
    id_offsets: &[u32],
    eos: u32,
    logits_vocab_size: Option<usize>,
    permit: &maskforge_core::index::VocabBuildPermit<'_>,
) -> Result<VocabularyHandle, CoreError> {
    Ok(
        VocabularyHandle::from_packed_admitted_with_logits_vocab_size(
            token_bytes,
            byte_offsets,
            token_ids,
            id_offsets,
            eos,
            logits_vocab_size,
            permit,
        )?,
    )
}

/// Parses supported Python bind modes.
/// Test-only modes are unavailable in release builds.
fn parse_bind_mode(mode: &str) -> PyResult<CoreBindMode> {
    match mode {
        "byte_trie" => Ok(CoreBindMode::TrieJointBytePacked),
        "byte_trie_lazy" => Ok(CoreBindMode::TrieJointByteLazy),
        #[cfg(any(feature = "test-utils", feature = "bench-internals"))]
        "naive" => Ok(CoreBindMode::Naive),
        other => Err(structured_err(
            "Unsupported".to_string(),
            "L4Bind".to_string(),
            None,
            None,
            Some(other.to_string()),
            format!("unknown bind mode {other:?}; expected \"byte_trie\" or \"byte_trie_lazy\""),
            None,
        )),
    }
}

/// Binds an engine to a vocabulary. Core constructors validate handle and trie pairing.
fn build_artifact_core(
    engine: maskforge_core::RefEngine,
    vocab: Option<Arc<Vocabulary>>,
    provenance: Provenance,
    mode: CoreBindMode,
    handle: Option<&VocabularyHandle>,
    bound: Option<&BoundByteTrie>,
) -> Result<CompiledArtifact, CoreError> {
    let need_vocab = || {
        vocab.clone().ok_or_else(|| {
            CoreError::Unsupported("bind needs a vocabulary or a handle".to_string())
        })
    };
    let artifact = match mode {
        CoreBindMode::Naive => match handle {
            Some(handle) => CompiledArtifact::new_from_handle(handle, engine, provenance)?,
            None => CompiledArtifact::new(engine, need_vocab()?, provenance)?,
        },
        CoreBindMode::TrieJointByte | CoreBindMode::TrieJointBytePacked => match (handle, bound) {
            (Some(handle), Some(bound)) => {
                CompiledArtifact::new_from_bound_trie(handle, engine, provenance, mode, bound)?
            }
            _ => {
                let handle = VocabularyHandle::new(need_vocab()?)?;
                let bound = bind_via_configured_cache(&handle)?;
                CompiledArtifact::new_from_bound_trie(&handle, engine, provenance, mode, &bound)?
            }
        },
        CoreBindMode::TrieJointByteLazy => match (handle, bound) {
            (Some(handle), Some(bound)) => {
                CompiledArtifact::new_from_bound_trie(handle, engine, provenance, mode, bound)?
            }
            _ => {
                return Err(CoreError::Unsupported(
                    "byte_trie_lazy requires a PyVocabulary handle".to_string(),
                ))
            }
        },
        _ => {
            return Err(CoreError::Unsupported(
                "bind mode not supported through the FFI".to_string(),
            ))
        }
    };
    Ok(artifact)
}

/// The handle's cached byte-trie when `mode` needs it, else `None` - the pure-Rust core of
/// `trie_for_mode`, callable under `detach`.
fn trie_for_mode_core(
    handle: &VocabularyHandle,
    mode: CoreBindMode,
) -> Result<Option<BoundByteTrie>, CoreError> {
    match mode {
        CoreBindMode::TrieJointByte
        | CoreBindMode::TrieJointByteLazy
        | CoreBindMode::TrieJointBytePacked => Ok(Some(bind_via_configured_cache(handle)?)),
        _ => Ok(None),
    }
}

/// Downgrades an oversized packed request to lazy row construction.
fn adapt_packed_mode(
    engine: &maskforge_core::RefEngine,
    handle: &VocabularyHandle,
    mode: CoreBindMode,
    bound: Option<&BoundByteTrie>,
) -> CoreBindMode {
    match bound {
        Some(bound) => CompiledArtifact::recommended_bind_mode(
            engine.state_count(),
            handle.mask_vocab_size(),
            bound,
            mode,
        ),
        None => mode,
    }
}

/// Compiles against an existing vocabulary handle without rebuilding canonical token data.
fn bind_artifact_with_handle(
    ir: &SchemaIR,
    handle: &VocabularyHandle,
    mode: CoreBindMode,
    bound: Option<&BoundByteTrie>,
) -> Result<Arc<CompiledArtifact>, CoreError> {
    let engine = core_compile_ir(ir)?;
    let mode = adapt_packed_mode(&engine, handle, mode, bound);
    let provenance = Provenance::reference(HashState::Hash(ir.canonical_hash()));
    build_artifact_core(engine, None, provenance, mode, Some(handle), bound).map(Arc::new)
}

fn bind_cached_executable(
    cached: &CachedExecutable,
    handle: &VocabularyHandle,
    mode: CoreBindMode,
) -> Result<CompiledArtifact, CompileError> {
    let engine = cached
        .executable()
        .regular_engine()
        .cloned()
        .ok_or_else(|| {
            CompileError::new(
                ErrorCode::Unsupported,
                Stage::L3,
                "vocabulary-bound byte artifacts require a regular schema executable",
            )
        })?;
    let bound = trie_for_mode_core(handle, mode).map_err(core_error_to_compile_error)?;
    let mode = adapt_packed_mode(&engine, handle, mode, bound.as_ref());
    let ir_hash = cached.ir_hash();
    let provenance = Provenance {
        profile: maskforge_core::error::ProvenanceProfile::CachedArtifact,
        source_input_hash: HashState::Hash(ir_hash),
        ir_hash: HashState::Hash(ir_hash),
        cache_valid: true,
    };
    match mode {
        CoreBindMode::Naive => CompiledArtifact::new_from_handle_shared(handle, engine, provenance),
        CoreBindMode::TrieJointByte
        | CoreBindMode::TrieJointByteLazy
        | CoreBindMode::TrieJointBytePacked => {
            let bound = bound.as_ref().ok_or_else(|| {
                CompileError::new(
                    ErrorCode::ArtifactMismatch,
                    Stage::L4Bind,
                    "bound trie absent",
                )
            })?;
            CompiledArtifact::new_from_bound_trie_shared(handle, engine, provenance, mode, bound)
        }
        _ => Err(CompileError::new(
            ErrorCode::Unsupported,
            Stage::L4Bind,
            "bind mode not supported through the FFI",
        )),
    }
}

/// Compiles using shared executable and vocabulary caches.
fn bind_json_schema(
    schema: &str,
    assume_closed: bool,
    handle: &VocabularyHandle,
    mode: CoreBindMode,
    cache: &ArtifactCache,
) -> Result<Arc<CompiledArtifact>, CoreError> {
    let executable = global_executable_cache()
        .get_or_compile_json(schema, options(assume_closed))
        .map_err(CoreError::Compile)?;
    cache
        .get_or_build_executable(
            executable.semantic_identity(),
            handle.fingerprint(),
            handle.mask_vocab_size(),
            mode,
            || bind_cached_executable(&executable, handle, mode),
        )
        .map_err(CoreError::Compile)
}

/// Same contract as [`bind_json_schema`], keyed on the exact MFIR wire bytes instead of schema
/// text - a repeat call skips decode and revalidation entirely on a cache hit.
fn bind_mfir_wire(
    wire: &[u8],
    handle: &VocabularyHandle,
    mode: CoreBindMode,
    cache: &ArtifactCache,
) -> Result<Arc<CompiledArtifact>, CoreError> {
    let executable = global_executable_cache()
        .get_or_compile_wire(wire)
        .map_err(CoreError::Compile)?;
    cache
        .get_or_build_executable(
            executable.semantic_identity(),
            handle.fingerprint(),
            handle.mask_vocab_size(),
            mode,
            || bind_cached_executable(&executable, handle, mode),
        )
        .map_err(CoreError::Compile)
}

#[cfg(test)]
mod bind_input_cached_tests {
    use super::*;

    fn test_handle() -> VocabularyHandle {
        VocabularyHandle::from_packed_with_logits_vocab_size(
            b"ab",
            &[0, 1, 2],
            &[0, 1],
            &[0, 1, 2],
            2,
            None,
        )
        .expect("handle")
    }

    #[test]
    fn every_requested_bind_mode_goes_through_the_cache_and_hits_on_repeat() {
        let handle = test_handle();
        let cache = ArtifactCache::new();
        let schema = r#"{"type":"boolean"}"#;

        for mode in [
            CoreBindMode::Naive,
            CoreBindMode::TrieJointByteLazy,
            CoreBindMode::TrieJointBytePacked,
        ] {
            let first = bind_json_schema(schema, false, &handle, mode, &cache).expect("first bind");
            assert_eq!(
                first.provenance().profile,
                maskforge_core::error::ProvenanceProfile::CachedArtifact,
                "{mode:?} must go through the cache"
            );
            let second =
                bind_json_schema(schema, false, &handle, mode, &cache).expect("second bind");
            assert!(
                Arc::ptr_eq(&first, &second),
                "{mode:?}: a repeat identical request must be a cache hit, not a rebuild"
            );
        }
    }

    #[test]
    fn explicit_lazy_and_naive_requests_get_distinct_cache_entries_from_packed() {
        let handle = test_handle();
        let cache = ArtifactCache::new();
        let schema = r#"{"type":"boolean"}"#;

        let naive = bind_json_schema(schema, false, &handle, CoreBindMode::Naive, &cache)
            .expect("naive bind");
        let lazy = bind_json_schema(
            schema,
            false,
            &handle,
            CoreBindMode::TrieJointByteLazy,
            &cache,
        )
        .expect("lazy bind");
        let packed = bind_json_schema(
            schema,
            false,
            &handle,
            CoreBindMode::TrieJointBytePacked,
            &cache,
        )
        .expect("packed bind");

        assert!(!Arc::ptr_eq(&naive, &lazy));
        assert!(!Arc::ptr_eq(&lazy, &packed));
        assert_eq!(cache.len(), 3, "each requested mode is its own cache entry");
    }
}

/// Process-wide bounded trie cache shared by equal vocabulary handles.
fn global_trie_cache() -> &'static TrieCache {
    static CACHE: OnceLock<TrieCache> = OnceLock::new();
    CACHE.get_or_init(TrieCache::new)
}

/// Maps `GateConfigError` to its own machine-matchable code (`InvalidGateConfig` /
/// `GateAlreadyInitialized`), not the catch-all `Unsupported` every other structural failure uses.
fn gate_config_err(e: maskforge_core::index::GateConfigError) -> PyErr {
    let code = match e {
        maskforge_core::index::GateConfigError::InvalidValue => "InvalidGateConfig",
        maskforge_core::index::GateConfigError::AlreadyInitialized => "GateAlreadyInitialized",
    };
    structured_err(
        code.to_string(),
        "L4Bind".to_string(),
        None,
        None,
        None,
        e.message().to_string(),
        None,
    )
}

/// Configures packed-build concurrency and memory before first use.
#[pyfunction]
pub fn configure_vocab_build_budget(max_concurrent: usize, byte_budget: usize) -> PyResult<()> {
    maskforge_core::index::configure_vocab_build_gate(max_concurrent, byte_budget)
        .map_err(gate_config_err)
}

/// Sets the shared lazy-serving memory gate's `(max_concurrent, byte_budget)` before the first
/// `byte_trie_lazy` query. Same contract as `configure_vocab_build_budget`.
#[pyfunction]
pub fn configure_vocab_serving_budget(max_concurrent: usize, byte_budget: usize) -> PyResult<()> {
    maskforge_core::index::configure_vocab_serving_gate(max_concurrent, byte_budget)
        .map_err(gate_config_err)
}

/// `(active_bytes, peak_bytes, peak_concurrent, rejected)` for the shared packed-build memory gate.
#[pyfunction]
pub fn vocab_build_budget_stats() -> (usize, usize, usize, u64) {
    maskforge_core::index::vocab_build_gate_stats()
}

/// `(active_bytes, peak_bytes, peak_concurrent, rejected)` for the shared lazy-serving memory gate.
#[pyfunction]
pub fn vocab_serving_budget_stats() -> (usize, usize, usize, u64) {
    maskforge_core::index::vocab_serving_gate_stats()
}

/// Sets the shared automaton (regex/DFA) memory gate's `(max_concurrent, byte_budget)`.
#[pyfunction]
pub fn configure_automaton_build_budget(max_concurrent: usize, byte_budget: usize) -> PyResult<()> {
    maskforge_core::index::configure_automaton_build_gate(max_concurrent, byte_budget)
        .map_err(gate_config_err)
}

/// `(active_bytes, peak_bytes, peak_concurrent, rejected)` for the shared automaton memory gate.
#[pyfunction]
pub fn automaton_build_budget_stats() -> (usize, usize, usize, u64) {
    maskforge_core::index::automaton_build_gate_stats()
}

/// `bind` - the one choke point every FFI bind path uses.
fn bind_via_configured_cache(handle: &VocabularyHandle) -> Result<BoundByteTrie, CompileError> {
    global_trie_cache().bind(handle)
}

/// Returns shared trie cache retention and lifetime traffic counters.
#[pyfunction]
pub fn trie_cache_stats() -> (usize, usize, u64, u64, u64, u64) {
    let c = global_trie_cache();
    let (hits, misses, evictions, oversized_bypasses) = c.stats();
    (
        c.len(),
        c.bytes(),
        hits,
        misses,
        evictions,
        oversized_bypasses,
    )
}

/// Immutable vocabulary handle with a deterministic fingerprint and bounded cache.
#[pyclass(frozen, module = "maskforge._native")]
pub struct PyVocabulary {
    handle: VocabularyHandle,
    artifact_cache: Arc<ArtifactCache>,
}

#[pymethods]
impl PyVocabulary {
    /// Builds a validated handle after admitting its peak allocation.
    #[new]
    #[pyo3(signature = (eos_token_id, tokens, logits_vocab_size=None))]
    fn new(
        py: Python<'_>,
        eos_token_id: u32,
        tokens: &Bound<'_, PyAny>,
        logits_vocab_size: Option<usize>,
    ) -> PyResult<Self> {
        let (tb, bo, ti, io, permit) = admit_and_pack_tokens(py, tokens)?;
        let handle = py
            .detach(move || -> Result<VocabularyHandle, CoreError> {
                let handle = VocabularyHandle::from_packed_admitted_with_logits_vocab_size(
                    &tb,
                    &bo,
                    &ti,
                    &io,
                    eos_token_id,
                    logits_vocab_size,
                    &permit,
                )?;
                drop(permit);
                Ok(handle)
            })
            .map_err(|e| core_err(&e))?;
        Ok(Self {
            handle,
            artifact_cache: Arc::new(ArtifactCache::new()),
        })
    }

    /// Builds from four packed little-endian CSR buffers while the GIL is released.
    #[staticmethod]
    #[pyo3(signature = (
        token_bytes, byte_offsets, token_ids, id_offsets, eos_token_id, logits_vocab_size=None
    ))]
    fn from_packed_buffers(
        py: Python<'_>,
        token_bytes: &Bound<'_, PyAny>,
        byte_offsets: &Bound<'_, PyAny>,
        token_ids: &Bound<'_, PyAny>,
        id_offsets: &Bound<'_, PyAny>,
        eos_token_id: u32,
        logits_vocab_size: Option<usize>,
    ) -> PyResult<Self> {
        let ((token_bytes, byte_offsets, token_ids, id_offsets), permit) =
            bounded_buffers_to_vecs(py, [token_bytes, byte_offsets, token_ids, id_offsets])?;
        let handle = py
            .detach(move || {
                let result = build_handle_from_packed_core(
                    &token_bytes,
                    &byte_offsets,
                    &token_ids,
                    &id_offsets,
                    eos_token_id,
                    logits_vocab_size,
                    &permit,
                );
                drop(permit);
                result
            })
            .map_err(|e| core_err(&e))?;
        Ok(Self {
            handle,
            artifact_cache: Arc::new(ArtifactCache::new()),
        })
    }

    /// Builds from dense id-ordered token bytes; `None` entries are skipped.
    #[staticmethod]
    #[pyo3(signature = (decoded_tokens, eos_token_id, logits_vocab_size=None))]
    fn from_id_ordered_tokens(
        py: Python<'_>,
        decoded_tokens: &Bound<'_, PyAny>,
        eos_token_id: u32,
        logits_vocab_size: Option<usize>,
    ) -> PyResult<Self> {
        let (token_bytes, byte_offsets, present, permit) =
            admit_and_pack_dense_tokens(py, decoded_tokens)?;
        let handle = py
            .detach(move || -> Result<VocabularyHandle, CoreError> {
                let handle =
                    VocabularyHandle::from_dense_id_ordered_admitted_with_logits_vocab_size(
                        &token_bytes,
                        &byte_offsets,
                        &present,
                        eos_token_id,
                        logits_vocab_size,
                        &permit,
                    )?;
                drop(permit);
                Ok(handle)
            })
            .map_err(|e| core_err(&e))?;
        Ok(Self {
            handle,
            artifact_cache: Arc::new(ArtifactCache::new()),
        })
    }

    /// Builds from serialized fast-tokenizer JSON (no network); reuses `from_pretrained`'s decode pipeline.
    #[cfg(feature = "tokenizer-processing")]
    #[staticmethod]
    #[pyo3(signature = (tokenizer_json, eos_token_id, logits_vocab_size=None))]
    fn from_tokenizer_json(
        py: Python<'_>,
        tokenizer_json: &str,
        eos_token_id: u32,
        logits_vocab_size: Option<usize>,
    ) -> PyResult<Self> {
        let json = tokenizer_json.to_string();
        let handle = py.detach(move || -> Result<VocabularyHandle, PyErr> {
            let vocab = maskforge_core::build_vocabulary_from_tokenizer_json(&json, eos_token_id)
                .map_err(|e| vocab_err(&e))?;
            VocabularyHandle::new_with_logits_vocab_size(Arc::new(vocab), logits_vocab_size)
                .map_err(|e| compile_err(&e))
        })?;
        Ok(Self {
            handle,
            artifact_cache: Arc::new(ArtifactCache::new()),
        })
    }

    /// The EOS token id (a side value, never a mask bit).
    fn eos_token_id(&self) -> u32 {
        self.handle.eos_token_id()
    }

    /// The number of stored ordinary token ids plus one for EOS (NOT the mask width).
    fn len(&self) -> usize {
        self.handle.len()
    }

    /// The mask width this handle serves: `logits_vocab_size` if given at construction, else
    /// inferred as `max(largest token id, EOS id) + 1`.
    fn mask_vocab_size(&self) -> usize {
        self.handle.mask_vocab_size()
    }

    /// True when no ordinary tokens are present (EOS is not counted).
    fn is_empty(&self) -> bool {
        self.handle.is_empty()
    }

    /// The deterministic 32-byte content fingerprint, equal for the same tokens in any order.
    fn fingerprint(&self) -> Vec<u8> {
        self.handle.fingerprint().as_bytes().to_vec()
    }

    /// Prewarms the shared trie for this vocabulary and bind mode.
    #[pyo3(signature = (bind_mode = "byte_trie"))]
    fn prepare(&self, py: Python<'_>, bind_mode: &str) -> PyResult<()> {
        let mode = parse_bind_mode(bind_mode)?;
        let handle = self.handle.clone();
        py.detach(move || trie_for_mode_core(&handle, mode).map(|_| ()))
            .map_err(|e| core_err(&e))
    }

    /// Returns this vocabulary's cache retention and traffic counters.
    fn artifact_cache_stats(&self) -> (usize, usize, u64, u64, u64, u64) {
        let c = &self.artifact_cache;
        let (hits, misses, evictions, oversized_bypasses) = c.stats();
        (
            c.len(),
            c.bytes(),
            hits,
            misses,
            evictions,
            oversized_bypasses,
        )
    }

    /// Drops this vocabulary's retained artifacts, releasing their bytes.
    fn clear_artifact_cache(&self) {
        self.artifact_cache.clear();
    }

    #[cfg(feature = "bench-internals")]
    fn clear_artifact_cache_for_bench(&self) {
        self.artifact_cache.clear();
    }
}

/// A mutable, per-sequence matcher for schemas whose IR contains an `OpenObject`, which the
/// byte-DFA compiler cannot fold into one anchored automaton; see `maskforge_core::structured`.
#[pyclass(module = "maskforge._native")]
pub struct PyStructuredMatcher {
    program: Arc<StructuredProgram>,
    matcher: StructuredMatcher,
    vocab: VocabularyHandle,
    trie: Option<BoundByteTrie>,
}

#[pymethods]
impl PyStructuredMatcher {
    /// Starts a matcher over `ir` at the empty prefix, sharing `vocabulary`'s canonical records - no
    /// per-mask copy of the vocabulary, only a cheap `Arc`-backed handle clone.
    #[new]
    fn new(py: Python<'_>, ir: &PySchemaIR, vocabulary: &PyVocabulary) -> PyResult<Self> {
        let schema = ir.inner.clone();
        let handle = vocabulary.handle.clone();
        let bind_handle = handle.clone();
        let (program, matcher, trie) = py
            .detach(move || {
                let executable = global_executable_cache()
                    .get_or_compile_ir(schema.clone())
                    .map_err(StructuredMatcherError::from)?;
                let program = match executable.executable().structured_program() {
                    Some(program) => program.clone(),
                    None => {
                        StructuredProgram::compile(schema).map_err(StructuredMatcherError::from)?
                    }
                };
                let matcher = StructuredMatcher::new(program.clone())?;
                let trie = program
                    .uses_incremental_backend()
                    .then(|| bind_via_configured_cache(&bind_handle))
                    .transpose()
                    .map_err(StructuredMatcherError::from)?;
                Ok::<_, StructuredMatcherError>((program, matcher, trie))
            })
            .map_err(|error| structured_matcher_err(&error))?;
        Ok(Self {
            program,
            matcher,
            vocab: handle,
            trie,
        })
    }

    #[cfg(feature = "bench-internals")]
    #[staticmethod]
    fn reference_for_bench(
        py: Python<'_>,
        ir: &PySchemaIR,
        vocabulary: &PyVocabulary,
    ) -> PyResult<Self> {
        let program = StructuredProgram::reference_for_bench(ir.inner.clone());
        let matcher = py
            .detach({
                let program = program.clone();
                move || StructuredMatcher::new(program)
            })
            .map_err(|error| structured_matcher_err(&error))?;
        Ok(Self {
            program,
            matcher,
            vocab: vocabulary.handle.clone(),
            trie: None,
        })
    }

    /// Returns to the empty prefix (a fresh matcher over the same IR).
    fn reset(&mut self) -> PyResult<()> {
        self.matcher = StructuredMatcher::new(self.program.clone())
            .map_err(|error| structured_matcher_err(&error))?;
        Ok(())
    }

    /// Stopping now yields a complete, valid instance.
    fn is_finished(&self) -> bool {
        self.matcher.is_accepting()
    }

    /// The position has reached a permanent dead end: neither accepting now nor extendable.
    fn is_dead(&self) -> bool {
        self.matcher.is_dead()
    }

    /// EOS is legal exactly where the position accepts.
    fn eos_legal(&self) -> bool {
        self.matcher.eos_legal()
    }

    /// The mask width this matcher's vocabulary serves.
    fn mask_vocab_size(&self) -> usize {
        self.vocab.mask_vocab_size()
    }

    /// Runtime backend selected for this matcher.
    fn backend_name(&self) -> &'static str {
        self.matcher.backend_name()
    }

    #[cfg(feature = "bench-internals")]
    fn fallback_reason(&self) -> Option<&'static str> {
        self.program.fallback_reason()
    }

    fn retained_plan_bytes(&self) -> Option<usize> {
        self.program.retained_plan_bytes()
    }

    /// Bytes currently retained by the mutable sequence state.
    fn retained_session_bytes(&self) -> usize {
        self.matcher.retained_session_bytes()
    }

    /// Peak charged bytes observed by the mutable sequence state.
    fn peak_session_bytes(&self) -> usize {
        self.matcher.peak_session_bytes()
    }

    /// Validator cursors currently owned by the structured matcher tree.
    #[cfg(feature = "bench-internals")]
    fn active_validator_count(&self) -> usize {
        self.matcher.active_validator_count_for_bench()
    }

    /// `(frame depth, wrapper-frame depth)` for the committed incremental state.
    #[cfg(feature = "bench-internals")]
    fn frame_profile(&self) -> (usize, usize) {
        self.matcher.frame_profile_for_bench()
    }

    #[cfg(feature = "bench-internals")]
    fn session_retention_breakdown(&self) -> (usize, usize) {
        self.matcher.retention_breakdown_for_bench()
    }

    /// Work performed by the most recent trie-mask call.
    #[cfg(feature = "bench-internals")]
    fn mask_work_metrics(&self) -> (u64, u64, u64, u64, u64, u64, u64, u64) {
        let metrics = self.matcher.mask_work_metrics_for_bench();
        (
            metrics.trie_nodes,
            metrics.trie_edges,
            metrics.pushed_bytes,
            metrics.checkpoints,
            metrics.rollbacks,
            metrics.dynamic_resolutions,
            metrics.dynamic_scope_steps,
            metrics.local_dfa_transitions,
        )
    }

    /// Proof-cache activity from the most recent trie-mask call.
    #[cfg(feature = "bench-internals")]
    fn slice_proof_metrics(&self) -> (u64, u64, usize) {
        let metrics = self.matcher.slice_proof_metrics_for_bench();
        (metrics.hits, metrics.misses, metrics.entries)
    }

    /// `(vocabulary slice bytes, local proof-cache bytes)` retained by adaptive masking.
    #[cfg(feature = "bench-internals")]
    fn adaptive_cache_bytes(&self) -> (usize, usize) {
        (
            self.vocab.token_slice_retained_bytes(),
            self.matcher.slice_proof_cache_bytes_for_bench(),
        )
    }

    /// Fills `out` (a caller-owned little-endian `bytearray`, sized `4 * ceil(mask_vocab_size / 32)`)
    /// with the allowed-token mask at the current position.
    fn compute_mask_into(&mut self, out: &Bound<'_, PyByteArray>) -> PyResult<()> {
        // SAFETY: GIL held throughout and the writer is pure Rust; Python cannot resize `out`.
        let dst = unsafe { out.as_bytes_mut() };
        self.matcher
            .write_mask_le_bytes_into(&self.vocab, self.trie.as_ref(), dst)
            .map_err(|error| structured_matcher_err(&error))
    }

    #[cfg(feature = "bench-internals")]
    fn compute_full_trie_mask_into(&mut self, out: &Bound<'_, PyByteArray>) -> PyResult<()> {
        // SAFETY: GIL held throughout and the writer is pure Rust; Python cannot resize out.
        let dst = unsafe { out.as_bytes_mut() };
        let trie = self
            .trie
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing byte trie"))?;
        self.matcher
            .write_full_trie_mask_for_bench(&self.vocab, trie, dst)
            .map_err(|error| structured_matcher_err(&error))
    }

    #[cfg(feature = "bench-internals")]
    fn compute_record_mask_into(&mut self, out: &Bound<'_, PyByteArray>) -> PyResult<()> {
        // SAFETY: GIL held throughout and the writer is pure Rust; Python cannot resize out.
        let dst = unsafe { out.as_bytes_mut() };
        self.matcher
            .write_record_mask_le_bytes_into(
                self.vocab.mask_vocab_size(),
                self.vocab.iter_records(),
                dst,
            )
            .map_err(|error| structured_matcher_err(&error))
    }

    /// Appends `token`'s bytes if that keeps the position accepting or extendable; returns whether
    /// it was applied. An unknown token id is always rejected, never a panic.
    fn commit_token(&mut self, token: u32) -> PyResult<bool> {
        match self.vocab.token_bytes(token) {
            Some(bytes) => self
                .matcher
                .advance(bytes)
                .map_err(|error| structured_matcher_err(&error)),
            None => Ok(false),
        }
    }
}

/// A frozen typed IR handle exposing read-only structure (version, content address, diagnostics,
/// MFIR bytes).
#[pyclass(frozen, module = "maskforge._native")]
pub struct PySchemaIR {
    inner: Arc<SchemaIR>,
}

#[pymethods]
impl PySchemaIR {
    /// The IR schema version.
    fn ir_version(&self) -> u16 {
        self.inner.ir_version()
    }

    /// The number of arena nodes.
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Whether every occurrence is supported (no diagnostics), so the IR can be compiled.
    fn is_supported(&self) -> bool {
        self.inner.diagnostics().len() == 0
    }

    /// Whether this IR needs `PyStructuredMatcher` instead of `compile_ir`/`compile_json_schema*`.
    fn requires_structured_backend(&self) -> bool {
        self.inner.requires_structured_backend()
    }

    /// Returns isolated structured-program compile time and retained plan bytes.
    #[cfg(feature = "bench-internals")]
    fn structured_plan_compile_for_bench(&self) -> PyResult<(u64, usize)> {
        let (program, elapsed) = StructuredProgram::compile_timed_for_bench(self.inner.clone())
            .map_err(|error| compile_err(&error))?;
        let retained = program.retained_plan_bytes().ok_or_else(|| {
            PyErr::new::<MaskforgeError, _>((
                "Unsupported",
                "benchmark schema selected the reference backend",
            ))
        })?;
        Ok((elapsed, retained))
    }

    /// Returns total plan-build time, SCC-analysis time, and retained plan bytes.
    #[cfg(feature = "bench-internals")]
    fn structured_plan_profile_for_bench(&self) -> PyResult<(u64, u64, usize)> {
        let (program, total_ns, scc_ns) =
            StructuredProgram::compile_profiled_for_bench(self.inner.clone())
                .map_err(|error| compile_err(&error))?;
        let retained = program.retained_plan_bytes().ok_or_else(|| {
            PyErr::new::<MaskforgeError, _>(("Unsupported", "incremental plan is absent"))
        })?;
        Ok((total_ns, scc_ns, retained))
    }

    /// Returns named regular compile summary counters without an ambiguous timing tuple.
    #[cfg(feature = "bench-internals")]
    fn regular_compile_profile_for_bench(&self) -> PyResult<(u64, u64, u64, u64, u64, u64, u64)> {
        let (_engine, profile) = maskforge_core::compile::compile_ir_profiled(&self.inner)
            .map_err(|error| compile_err(&error))?;
        Ok((
            profile
                .durations_ns
                .get(maskforge_core::compile::CompilePhase::RouteAnalysis),
            profile
                .durations_ns
                .get(maskforge_core::compile::CompilePhase::DfaMaterialization),
            profile.counters.dfa_states,
            profile.counters.byte_classes,
            profile.counters.graph_edges,
            profile.counters.transition_cells,
            profile.retained.automaton,
        ))
    }

    /// Per-node shared route decisions and bounded regular cost estimates.
    #[cfg(feature = "bench-internals")]
    fn route_profile_for_bench(
        &self,
    ) -> Vec<(u32, String, String, String, String, String, String)> {
        self.inner
            .route_table()
            .routes()
            .iter()
            .enumerate()
            .map(|(index, route)| {
                (
                    u32::try_from(index).unwrap_or(u32::MAX),
                    format!("{:?}", route.kind),
                    format!("{:?}", route.reason),
                    format!("{:?}", route.regular.states),
                    format!("{:?}", route.regular.transition_cells),
                    format!("{:?}", route.regular.retained_bytes),
                    format!("{:?}", route.regular.peak_bytes),
                )
            })
            .collect()
    }

    /// Root execution kind and stable primary reason.
    #[cfg(feature = "bench-internals")]
    fn root_route_for_bench(&self) -> (String, String) {
        self.inner.route_table().get(self.inner.root()).map_or_else(
            || {
                (
                    "Structured".to_string(),
                    "StructuredOnlySemantics".to_string(),
                )
            },
            |route| (format!("{:?}", route.kind), format!("{:?}", route.reason)),
        )
    }

    /// Returns sorted reachable IR node-kind names for routing census builds.
    #[cfg(feature = "bench-internals")]
    fn reachable_node_kinds_for_bench(&self) -> PyResult<Vec<&'static str>> {
        StructuredProgram::reachable_node_kinds_for_bench(&self.inner)
            .map_err(|error| compile_err(&error))
    }

    /// The blake3 content address (32 bytes), stable across serialization.
    fn canonical_hash(&self) -> Vec<u8> {
        self.inner.canonical_hash().to_vec()
    }

    /// The serialized MFIR envelope: this is the validated wire data, not the typed object.
    fn to_wire(&self) -> Vec<u8> {
        self.inner.to_wire()
    }

    /// One `(machine_code, json_pointer)` pair per rejected occurrence.
    fn diagnostics(&self) -> Vec<(String, String)> {
        self.inner
            .diagnostics()
            .map(|d| {
                let pointer = self.inner.str_at(d.json_pointer).unwrap_or("").to_string();
                (format!("{:?}", d.reason), pointer)
            })
            .collect()
    }
}

/// Builds a frozen typed IR from JSON Schema text. A malformed schema raises `MaskforgeError`;
/// an unsupported schema returns an IR whose `diagnostics()` are non-empty (it will not compile).
#[pyfunction]
#[pyo3(signature = (schema, assume_closed = false))]
pub fn schema_to_ir(py: Python<'_>, schema: &str, assume_closed: bool) -> PyResult<PySchemaIR> {
    let schema = schema.to_owned();
    let ir = py
        .detach(move || core_schema_to_ir(&schema, options(assume_closed)))
        .map_err(|e| compile_err(&e))?;
    Ok(PySchemaIR {
        inner: Arc::new(ir),
    })
}

/// Builds one frozen IR from a root schema and already-supplied in-memory schema resources.
#[pyfunction]
#[pyo3(signature = (
    schema, retrieval_uri = None, resources = None, assume_closed = false,
    max_document_bytes = None, max_total_input_bytes = None
))]
pub fn schema_to_ir_with_resources(
    py: Python<'_>,
    schema: &str,
    retrieval_uri: Option<&str>,
    resources: Option<std::collections::HashMap<String, String>>,
    assume_closed: bool,
    max_document_bytes: Option<usize>,
    max_total_input_bytes: Option<usize>,
) -> PyResult<PySchemaIR> {
    let schema = schema.to_owned();
    let retrieval_uri = retrieval_uri.map(str::to_owned);
    let resources = resources.unwrap_or_default().into_iter().collect();
    let ir = py
        .detach(move || {
            schema_set_to_ir_owned(
                schema,
                retrieval_uri,
                resources,
                assume_closed,
                max_document_bytes,
                max_total_input_bytes,
            )
        })
        .map_err(|e| compile_err(&e))?;
    Ok(PySchemaIR {
        inner: Arc::new(ir),
    })
}

#[cfg(feature = "bench-internals")]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    schema, retrieval_uri = None, resources = None, assume_closed = false,
    format_assertion = false, max_document_bytes = None, max_total_input_bytes = None
))]
pub fn schema_to_ir_with_resources_profile_for_bench(
    py: Python<'_>,
    schema: &str,
    retrieval_uri: Option<&str>,
    resources: Option<std::collections::HashMap<String, String>>,
    assume_closed: bool,
    format_assertion: bool,
    max_document_bytes: Option<usize>,
    max_total_input_bytes: Option<usize>,
) -> PyResult<(PySchemaIR, Vec<(&'static str, u64)>)> {
    let schema = schema.to_owned();
    let retrieval_uri = retrieval_uri.map(str::to_owned);
    let resources = resources.unwrap_or_default().into_iter().collect();
    let (ir, timings) = py
        .detach(move || {
            schema_set_to_ir_profiled_owned(
                schema,
                retrieval_uri,
                resources,
                assume_closed,
                format_assertion,
                max_document_bytes,
                max_total_input_bytes,
            )
        })
        .map_err(|error| compile_err(&error))?;
    Ok((
        PySchemaIR {
            inner: Arc::new(ir),
        },
        timings,
    ))
}

/// Compatibility spelling for an absolute-URI resolver map with no root retrieval URI.
#[pyfunction]
#[pyo3(signature = (schema, resolver, assume_closed = false))]
pub fn schema_to_ir_with_external_refs(
    py: Python<'_>,
    schema: &str,
    resolver: std::collections::HashMap<String, String>,
    assume_closed: bool,
) -> PyResult<PySchemaIR> {
    let schema = schema.to_owned();
    let resources = resolver.into_iter().collect();
    let ir = py
        .detach(move || schema_set_to_ir_owned(schema, None, resources, assume_closed, None, None))
        .map_err(|e| compile_err(&e))?;
    Ok(PySchemaIR {
        inner: Arc::new(ir),
    })
}

/// Builds a frozen typed IR from a bare regex string (the migration shim).
#[pyfunction]
pub fn regex_ir(regex: &str) -> PyResult<PySchemaIR> {
    let ir = regex_to_ir(regex, CompileOptions::default()).map_err(|e| compile_err(&e))?;
    Ok(PySchemaIR {
        inner: Arc::new(ir),
    })
}

/// Decodes validated MFIR bytes back into a frozen typed IR. Fails closed on any tampering.
#[pyfunction]
pub fn ir_from_wire(bytes: Vec<u8>) -> PyResult<PySchemaIR> {
    let ir = SchemaIR::from_wire(&bytes).map_err(|e| compile_err(&e))?;
    Ok(PySchemaIR {
        inner: Arc::new(ir),
    })
}

/// Compiles a typed IR into a `PyIndex`. Byte-DFA only: prefer `ConstraintSession.from_json_schema`.
#[pyfunction]
#[pyo3(signature = (
    ir_wire, eos_token_id, tokens, bind_mode = "byte_trie", logits_vocab_size = None
))]
pub fn compile_ir(
    py: Python<'_>,
    ir_wire: Vec<u8>,
    eos_token_id: u32,
    tokens: &Bound<'_, PyAny>,
    bind_mode: &str,
    logits_vocab_size: Option<usize>,
) -> PyResult<PyIndex> {
    let mode = parse_bind_mode(bind_mode)?;
    let (tb, bo, ti, io, permit) = admit_and_pack_tokens(py, tokens)?;
    let inner = py
        .detach(move || {
            let ir = SchemaIR::from_wire(&ir_wire)?;
            let handle = VocabularyHandle::from_packed_admitted_with_logits_vocab_size(
                &tb,
                &bo,
                &ti,
                &io,
                eos_token_id,
                logits_vocab_size,
                &permit,
            )?;
            drop(permit);
            let bound = trie_for_mode_core(&handle, mode)?;
            bind_artifact_with_handle(&ir, &handle, mode, bound.as_ref())
        })
        .map_err(|e| core_err(&e))?;
    Ok(PyIndex { inner })
}

/// Compiles JSON Schema text into a `PyIndex`. Byte-DFA only: prefer `ConstraintSession.from_json_schema`.
#[pyfunction]
#[pyo3(signature = (
    schema, eos_token_id, tokens, assume_closed = false, bind_mode = "byte_trie",
    logits_vocab_size = None
))]
pub fn compile_json_schema(
    py: Python<'_>,
    schema: &str,
    eos_token_id: u32,
    tokens: &Bound<'_, PyAny>,
    assume_closed: bool,
    bind_mode: &str,
    logits_vocab_size: Option<usize>,
) -> PyResult<PyIndex> {
    let mode = parse_bind_mode(bind_mode)?;
    let schema = schema.to_owned();
    let (tb, bo, ti, io, permit) = admit_and_pack_tokens(py, tokens)?;
    let inner = py
        .detach(move || {
            let ir = core_schema_to_ir(&schema, options(assume_closed))?;
            let handle = VocabularyHandle::from_packed_admitted_with_logits_vocab_size(
                &tb,
                &bo,
                &ti,
                &io,
                eos_token_id,
                logits_vocab_size,
                &permit,
            )?;
            drop(permit);
            let bound = trie_for_mode_core(&handle, mode)?;
            bind_artifact_with_handle(&ir, &handle, mode, bound.as_ref())
        })
        .map_err(|e| core_err(&e))?;
    Ok(PyIndex { inner })
}

/// Compiles with an existing vocabulary. Byte-DFA only: prefer `ConstraintSession.from_json_schema`.
#[pyfunction]
#[pyo3(signature = (vocabulary, schema, assume_closed = false, bind_mode = "byte_trie"))]
pub fn compile_json_schema_with_vocabulary(
    py: Python<'_>,
    vocabulary: &PyVocabulary,
    schema: &str,
    assume_closed: bool,
    bind_mode: &str,
) -> PyResult<PyIndex> {
    let mode = parse_bind_mode(bind_mode)?;
    let handle = vocabulary.handle.clone();
    let cache = Arc::clone(&vocabulary.artifact_cache);
    let schema = schema.to_owned();
    let inner = py
        .detach(move || bind_json_schema(&schema, assume_closed, &handle, mode, &cache))
        .map_err(|e| core_err(&e))?;
    Ok(PyIndex { inner })
}

/// Process-wide executable-cache stats: (hits, misses, evictions, oversized_bypasses, entries, bytes).
#[pyfunction]
pub fn executable_cache_stats() -> (u64, u64, u64, u64, usize, usize) {
    let stats = global_executable_cache().stats();
    (
        stats.hits,
        stats.misses,
        stats.evictions,
        stats.oversized_bypasses,
        stats.entries,
        stats.bytes,
    )
}

/// Drops every cached executable from the process-wide cache, releasing their retained bytes.
#[pyfunction]
pub fn clear_executable_cache() {
    global_executable_cache().clear();
}

/// Times one vocabulary-neutral executable lookup or build without vocabulary binding.
#[cfg(feature = "bench-internals")]
#[pyfunction]
#[pyo3(signature = (schema, assume_closed = false))]
pub fn compile_schema_executable_for_bench(
    py: Python<'_>,
    schema: &str,
    assume_closed: bool,
) -> PyResult<(String, u64)> {
    let schema = schema.to_owned();
    py.detach(move || {
        let started = std::time::Instant::now();
        let executable = global_executable_cache()
            .get_or_compile_json(&schema, options(assume_closed))
            .map_err(|error| compile_err(&error))?;
        let kind = match executable.executable().as_ref() {
            maskforge_core::runtime::ExecutableSchema::Regular(_) => "regular",
            maskforge_core::runtime::ExecutableSchema::Structured(_) => "structured",
            maskforge_core::runtime::ExecutableSchema::Hybrid(_) => "hybrid",
        };
        let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        Ok((kind.to_string(), elapsed))
    })
}

/// Compiles validated MFIR with an existing vocabulary and both cache tiers.
#[pyfunction]
#[pyo3(signature = (vocabulary, ir_wire, bind_mode = "byte_trie"))]
pub fn compile_ir_with_vocabulary(
    py: Python<'_>,
    vocabulary: &PyVocabulary,
    ir_wire: Vec<u8>,
    bind_mode: &str,
) -> PyResult<PyIndex> {
    let mode = parse_bind_mode(bind_mode)?;
    let handle = vocabulary.handle.clone();
    let cache = Arc::clone(&vocabulary.artifact_cache);
    let inner = py
        .detach(move || bind_mfir_wire(&ir_wire, &handle, mode, &cache))
        .map_err(|e| core_err(&e))?;
    Ok(PyIndex { inner })
}

/// Builds a named correctness-corpus index for differential tests.
#[cfg(feature = "test-utils")]
#[pyfunction]
#[pyo3(signature = (name, eos_token_id, tokens, bind_mode = "byte_trie"))]
pub fn corpus_index(
    name: &str,
    eos_token_id: u32,
    tokens: Vec<(Vec<u8>, Vec<u32>)>,
    bind_mode: &str,
) -> PyResult<PyIndex> {
    let mode = parse_bind_mode(bind_mode)?;
    let engine = corpus_cases()
        .map_err(|e| compile_err(&e))?
        .into_iter()
        .find(|c| c.name == name)
        .ok_or_else(|| {
            structured_err(
                "Malformed".to_string(),
                "L1".to_string(),
                None,
                None,
                None,
                format!("no corpus case named {name}"),
                None,
            )
        })?
        .engine;
    let vocab = Arc::new(build_vocab_core(eos_token_id, tokens).map_err(|e| core_err(&e))?);
    let inner = build_artifact_core(
        engine,
        Some(vocab),
        Provenance::reference(HashState::Hash([1; 32])),
        mode,
        None,
        None,
    )
    .map(Arc::new)
    .map_err(|e| core_err(&e))?;
    Ok(PyIndex { inner })
}

/// The JSON Schema (draft 2020-12) equivalent to a named corpus case, for the jsonschema bridge.
#[cfg(feature = "test-utils")]
#[pyfunction]
pub fn corpus_json_schema(name: &str) -> PyResult<String> {
    corpus_cases()
        .map_err(|e| compile_err(&e))?
        .into_iter()
        .find(|c| c.name == name)
        .map(|c| c.json_schema.to_string())
        .ok_or_else(|| {
            structured_err(
                "Malformed".to_string(),
                "L1".to_string(),
                None,
                None,
                None,
                format!("no corpus case named {name}"),
                None,
            )
        })
}

/// The names of every correctness-corpus case, in corpus order.
#[cfg(feature = "test-utils")]
#[pyfunction]
pub fn corpus_names() -> PyResult<Vec<String>> {
    Ok(corpus_cases()
        .map_err(|e| compile_err(&e))?
        .into_iter()
        .map(|c| c.name.to_string())
        .collect())
}

/// The instance strings for a named corpus case (minified JSON text, one per sample).
#[cfg(feature = "test-utils")]
#[pyfunction]
pub fn corpus_samples(name: &str) -> PyResult<Vec<String>> {
    corpus_cases()
        .map_err(|e| compile_err(&e))?
        .into_iter()
        .find(|c| c.name == name)
        .map(|c| c.samples.iter().map(|s| s.instance.to_string()).collect())
        .ok_or_else(|| {
            structured_err(
                "Malformed".to_string(),
                "L1".to_string(),
                None,
                None,
                None,
                format!("no corpus case named {name}"),
                None,
            )
        })
}

/// The compile-bind-session facade: `PyCompiler` -> `PySchemaProgram` -> `PyBoundSchema` ->
/// `PySession`. Wraps `maskforge_core::api` directly; no routing logic lives here.
#[pyclass(module = "maskforge._native", name = "Compiler")]
pub struct PyCompiler {
    inner: Compiler,
}

#[pymethods]
impl PyCompiler {
    #[new]
    #[pyo3(signature = (executable_cache_bytes=None, trie_cache_bytes=None, bind_policy=None))]
    fn new(
        executable_cache_bytes: Option<usize>,
        trie_cache_bytes: Option<usize>,
        bind_policy: Option<&str>,
    ) -> PyResult<Self> {
        let mut options = CompilerOptions::default();
        if let Some(bytes) = executable_cache_bytes {
            options = options.with_executable_cache_bytes(bytes);
        }
        if let Some(bytes) = trie_cache_bytes {
            options = options.with_trie_cache_bytes(bytes);
        }
        if let Some(policy) = bind_policy {
            options = options.with_bind_policy(match policy {
                "adaptive" => BindPolicy::Adaptive,
                "eager" => BindPolicy::Eager,
                "lazy" => BindPolicy::Lazy,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "bind_policy must be 'adaptive', 'eager' or 'lazy', got {other:?}"
                    )))
                }
            });
        }
        Ok(Self {
            inner: Compiler::new(options),
        })
    }

    /// A compiler that caches nothing; every compile is a fresh build.
    #[staticmethod]
    fn without_cache() -> Self {
        Self {
            inner: Compiler::without_cache(),
        }
    }

    fn compile_json_schema(&self, py: Python<'_>, schema: String) -> PyResult<PySchemaProgram> {
        py.detach(|| self.inner.compile_json_schema(&schema))
            .map(|inner| PySchemaProgram { inner })
            .map_err(|e| compile_err(&e))
    }

    #[pyo3(signature = (
        schema, retrieval_uri=None, resources=None, max_document_bytes=None,
        max_total_input_bytes=None
    ))]
    fn compile_json_schema_with_resources(
        &self,
        py: Python<'_>,
        schema: String,
        retrieval_uri: Option<String>,
        resources: Option<std::collections::HashMap<String, String>>,
        max_document_bytes: Option<usize>,
        max_total_input_bytes: Option<usize>,
    ) -> PyResult<PySchemaProgram> {
        let resources: Vec<(String, String)> = resources.unwrap_or_default().into_iter().collect();
        let limits = resource_limits_with_overrides(max_document_bytes, max_total_input_bytes);
        py.detach(|| {
            self.inner.compile_json_schema_with_resources(
                &schema,
                retrieval_uri.as_deref(),
                &resources,
                CompileOptions::default(),
                limits,
            )
        })
        .map(|inner| PySchemaProgram { inner })
        .map_err(|e| compile_err(&e))
    }

    /// `(hits, misses, evictions, oversized_bypasses, entries, bytes)`.
    fn cache_stats(&self) -> (u64, u64, u64, u64, usize, usize) {
        let s = self.inner.cache_stats();
        (
            s.hits,
            s.misses,
            s.evictions,
            s.oversized_bypasses,
            s.entries,
            s.bytes,
        )
    }

    fn clear_caches(&self) {
        self.inner.clear_caches();
    }
}

/// A compiled schema, vocabulary-independent and reusable across every vocabulary that needs it.
#[pyclass(
    frozen,
    skip_from_py_object,
    module = "maskforge._native",
    name = "SchemaProgram"
)]
#[derive(Clone)]
pub struct PySchemaProgram {
    inner: SchemaProgram,
}

#[pymethods]
impl PySchemaProgram {
    fn bind(&self, py: Python<'_>, vocabulary: &PyCompiledVocabulary) -> PyResult<PyBoundSchema> {
        let vocab = vocabulary.inner.clone();
        py.detach(|| self.inner.bind(&vocab))
            .map(|inner| PyBoundSchema { inner })
            .map_err(|e| compile_err(&e))
    }
}

/// A prepared, reusable vocabulary; build once and reuse across binds.
#[pyclass(
    frozen,
    skip_from_py_object,
    module = "maskforge._native",
    name = "CompiledVocabulary"
)]
#[derive(Clone)]
pub struct PyCompiledVocabulary {
    inner: CompiledVocabulary,
}

#[pymethods]
impl PyCompiledVocabulary {
    /// Wraps an existing `PyVocabulary`'s already-canonicalized handle; no rebuild.
    #[staticmethod]
    fn from_vocabulary(vocabulary: &PyVocabulary) -> Self {
        Self {
            inner: CompiledVocabulary::from_handle(vocabulary.handle.clone()),
        }
    }

    fn mask_vocab_size(&self) -> usize {
        self.inner.mask_vocab_size()
    }

    fn eos_token_id(&self) -> u32 {
        self.inner.eos_token_id()
    }

    fn token_bytes<'py>(&self, py: Python<'py>, token_id: u32) -> Option<Bound<'py, PyBytes>> {
        self.inner
            .token_bytes(token_id)
            .map(|bytes| PyBytes::new(py, bytes))
    }
}

/// A schema bound to one vocabulary; start as many independent sessions from it as needed.
#[pyclass(
    frozen,
    skip_from_py_object,
    module = "maskforge._native",
    name = "BoundSchema"
)]
#[derive(Clone)]
pub struct PyBoundSchema {
    inner: BoundSchema,
}

#[pymethods]
impl PyBoundSchema {
    fn start_session(&self) -> PyResult<PySession> {
        self.inner
            .start_session()
            .map(|inner| PySession {
                inner,
                mask_scratch: Vec::new(),
            })
            .map_err(|e| session_err(&e))
    }

    fn mask_word_count(&self) -> usize {
        self.inner.mask_word_count()
    }

    fn mask_vocab_size(&self) -> usize {
        self.inner.mask_vocab_size()
    }

    fn eos_token_id(&self) -> u32 {
        self.inner.eos_token_id().get()
    }
}

/// One generation sequence's mutable state. Never shared between concurrent sequences.
#[pyclass(module = "maskforge._native", name = "Session")]
pub struct PySession {
    inner: Session,
    // Reused across calls so a steady-state write_mask allocates nothing after the first call.
    mask_scratch: Vec<u8>,
}

#[pymethods]
impl PySession {
    /// Fills `out` (a `bytearray` of `4 * mask_word_count()` bytes) with the packed mask; no per-call heap allocation after warm-up.
    fn write_mask(&mut self, py: Python<'_>, out: &Bound<'_, PyByteArray>) -> PyResult<()> {
        let expected_bytes = self.inner.mask_word_count().saturating_mul(4);
        if out.len() != expected_bytes {
            return Err(session_err(&SessionError::MaskByteBufferLength {
                expected: expected_bytes,
                actual: out.len(),
            }));
        }
        if self.mask_scratch.len() != expected_bytes {
            self.mask_scratch.resize(expected_bytes, 0);
        }
        let inner = &mut self.inner;
        let scratch = &mut self.mask_scratch;
        py.detach(|| inner.write_mask_le_bytes(scratch))
            .map_err(|e| session_err(&e))?;
        // Another thread may have resized `out` while the GIL was released, so the length
        // checked before `detach` no longer proves anything.
        if out.len() != expected_bytes {
            return Err(session_err(&SessionError::MaskByteBufferLength {
                expected: expected_bytes,
                actual: out.len(),
            }));
        }
        // SAFETY: the GIL is held again and the length was revalidated above, so no concurrent
        // Python access can resize `out` between that check and this write.
        unsafe { out.as_bytes_mut() }.copy_from_slice(&self.mask_scratch);
        Ok(())
    }

    fn advance(&mut self, py: Python<'_>, token_id: u32) -> PyResult<()> {
        py.detach(|| self.inner.advance(TokenId(token_id)))
            .map_err(|e| session_err(&e))
    }

    fn reset(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.reset())
            .map_err(|e| session_err(&e))
    }

    fn is_accepting(&self) -> bool {
        self.inner.is_accepting()
    }

    fn is_dead(&self) -> bool {
        self.inner.is_dead()
    }

    fn is_stopped(&self) -> bool {
        self.inner.is_stopped()
    }

    fn mask_word_count(&self) -> usize {
        self.inner.mask_word_count()
    }

    fn mask_vocab_size(&self) -> usize {
        self.inner.mask_vocab_size()
    }

    fn eos_token_id(&self) -> u32 {
        self.inner.eos_token_id().get()
    }
}
