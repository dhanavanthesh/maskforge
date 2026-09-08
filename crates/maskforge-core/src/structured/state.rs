//! Mutable per-session incremental JSON parser and validator.
//! Each byte is transactional and rolls back on rejection or error.

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::mem::size_of;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::ir::{AnchorId, ResourceId, UnevaluatedKind};
use crate::primitives::{NodeId, StateId};

use super::canonical::{BuilderMark, CanonicalBuilder, CanonicalInsert, CanonicalSet};
use super::lexer::{DecodeStep, JsonStringDecoder};
use super::limits::{
    grow_bounded, plan_growth, GrowthRequest, SessionMemory, StructuredLimits,
    StructuredRuntimeError,
};
use super::plan::{
    AdditionalPlan, CombinatorKind, ContainsMatch, FiniteKeyTrie, NodePlan, NumberPlan, PropertyId,
    StructuredPlan, BITSET_WORD_BITS,
};
use super::validator::{scalar_is_complete, step_scalar, ScalarPhase, ScalarStep};

use crate::error::LimitKind;

const WS: [u8; 4] = [0x20, 0x09, 0x0a, 0x0d];
static NEXT_STRUCTURED_STATE_ID: AtomicUsize = AtomicUsize::new(1);

/// A compact handle into the session's `KeyArena`, replacing an owned string copy or an
/// `Rc<str>` (not `Send`, unusable once sessions run on worker threads under a released GIL).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct KeyId(u32);

/// Session-scoped arena for dynamic property names.
/// Keys are stored once and reclaimed when their objects commit.
#[derive(Clone, PartialEq, Debug, Default)]
struct KeyArena {
    bytes: String,
    spans: Vec<(u32, u32)>,
}

impl KeyArena {
    /// Checked lookup: a corrupt or stale `KeyId` (never expected - every live id is reclaimed
    /// exactly when its owning object closes) returns `None`, never panics or reads OOB.
    fn get(&self, id: KeyId) -> Option<&str> {
        let (start, len) = *self.spans.get(id.0 as usize)?;
        let end = u32::checked_add(start, len)?;
        self.bytes.get(start as usize..end as usize)
    }

    fn mark(&self) -> (u32, u32) {
        (self.bytes.len() as u32, self.spans.len() as u32)
    }
}

/// One hash bucket's contents: inline for the overwhelmingly common single-key case, so a new
/// dynamic key costs zero heap allocations unless it genuinely collides with another.
#[derive(Clone, PartialEq, Debug)]
enum DynamicSlot {
    One(KeyId),
    Many(Vec<KeyId>),
}

impl DynamicSlot {
    fn contains(&self, id_matches: impl Fn(KeyId) -> bool) -> bool {
        match self {
            Self::One(id) => id_matches(*id),
            Self::Many(ids) => ids.iter().any(|&id| id_matches(id)),
        }
    }
}

/// Which bucket mutation a new dynamic key needs, decided before any reservation or mutation.
#[derive(Clone, Copy)]
enum SlotTransition {
    ToOne,
    ToMany2 { existing: KeyId },
    ToManyPush { old_len: usize },
}

fn classify_slot_transition(existing: Option<&DynamicSlot>) -> SlotTransition {
    match existing {
        None => SlotTransition::ToOne,
        Some(DynamicSlot::One(id)) => SlotTransition::ToMany2 { existing: *id },
        Some(DynamicSlot::Many(ids)) => SlotTransition::ToManyPush { old_len: ids.len() },
    }
}

/// Duplicate check + checked span arithmetic for interning one key, shared by the
/// schema-constrained object path and the `AnyJson` object path. `Ok(None)` means a duplicate.
fn classify_key_intern(
    key_arena: &KeyArena,
    seen_dynamic: &FxHashMap<u64, DynamicSlot>,
    key_text: &str,
    hash: u64,
    limits: &StructuredLimits,
) -> Result<Option<(KeyId, u32, u32, SlotTransition)>, StructuredRuntimeError> {
    let duplicate = seen_dynamic
        .get(&hash)
        .is_some_and(|slot| slot.contains(|id| key_arena.get(id) == Some(key_text)));
    if duplicate {
        return Ok(None);
    }
    let cap = limits.max_session_bytes;
    let start = u32::try_from(key_arena.bytes.len())
        .map_err(|_| StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, cap))?;
    let len = u32::try_from(key_text.len()).map_err(|_| {
        StructuredRuntimeError::new(LimitKind::KeyBytes, usize::MAX, limits.max_key_bytes)
    })?;
    let span_idx = u32::try_from(key_arena.spans.len()).map_err(|_| {
        StructuredRuntimeError::new(
            LimitKind::PropertyCount,
            usize::MAX,
            limits.max_properties as usize,
        )
    })?;
    let transition = classify_slot_transition(seen_dynamic.get(&hash));
    Ok(Some((KeyId(span_idx), start, len, transition)))
}

/// Reserves the arena and dynamic-key-bucket capacity growth for one intern, charging each
/// reservation's real delta immediately - never several allocations before any charge.
fn reserve_key_intern_growth(
    key_arena: &mut KeyArena,
    seen_dynamic: &mut FxHashMap<u64, DynamicSlot>,
    hash: u64,
    key_len: usize,
    transition: SlotTransition,
    session_cap: usize,
    mem: &mut SessionMemory,
) -> Result<Option<Vec<KeyId>>, StructuredRuntimeError> {
    let cap = session_cap;
    let kind = LimitKind::SessionBytes;
    let bytes = &mut key_arena.bytes;
    let req = GrowthRequest {
        len: bytes.len(),
        capacity: bytes.capacity(),
        additional: key_len,
        item_size: 1,
    };
    grow_bounded(mem, cap, kind, req, |n| {
        bytes
            .try_reserve_exact(n)
            .map(|_| bytes.capacity())
            .map_err(|_| ())
    })?;
    let spans = &mut key_arena.spans;
    let req = GrowthRequest {
        len: spans.len(),
        capacity: spans.capacity(),
        additional: 1,
        item_size: size_of::<(u32, u32)>(),
    };
    grow_bounded(mem, cap, kind, req, |n| {
        spans
            .try_reserve_exact(n)
            .map(|_| spans.capacity())
            .map_err(|_| ())
    })?;
    let mut prebuilt_many = None;
    match transition {
        SlotTransition::ToOne => {
            let req = GrowthRequest {
                len: seen_dynamic.len(),
                capacity: seen_dynamic.capacity(),
                additional: 1,
                item_size: size_of::<u64>() + size_of::<DynamicSlot>() + 8,
            };
            grow_bounded(mem, cap, kind, req, |n| {
                seen_dynamic
                    .try_reserve(n)
                    .map(|_| seen_dynamic.capacity())
                    .map_err(|_| ())
            })?;
        }
        SlotTransition::ToMany2 { .. } => {
            let mut v: Vec<KeyId> = Vec::new();
            let req = GrowthRequest {
                len: 0,
                capacity: 0,
                additional: 2,
                item_size: size_of::<KeyId>(),
            };
            grow_bounded(mem, cap, kind, req, |n| {
                v.try_reserve_exact(n).map(|_| v.capacity()).map_err(|_| ())
            })?;
            prebuilt_many = Some(v);
        }
        SlotTransition::ToManyPush { .. } => {
            let Some(DynamicSlot::Many(ids)) = seen_dynamic.get_mut(&hash) else {
                unreachable!("classified as ToManyPush")
            };
            let req = GrowthRequest {
                len: ids.len(),
                capacity: ids.capacity(),
                additional: 1,
                item_size: size_of::<KeyId>(),
            };
            grow_bounded(mem, cap, kind, req, |n| {
                ids.try_reserve_exact(n)
                    .map(|_| ids.capacity())
                    .map_err(|_| ())
            })?;
        }
    }
    Ok(prebuilt_many)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ObjectPhase {
    Start,
    BeforeFirstKeyOrClose,
    InKey,
    AfterKeyBeforeColon,
    AfterColonBeforeValue,
    AfterValueBeforeCommaOrClose,
    AfterCommaBeforeKey,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct KeyOwner {
    state_identity: usize,
    frame_depth: u32,
    key_generation: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum KeyLeasePhase {
    Free,
    Prepared,
    TransferredToCommit,
}

/// Bounded decoded-key capacity lease for one structured cursor.
/// Wrapper cursors own separate slots and never share mutable key scratch.
#[derive(Debug)]
struct KeyCapacityLease {
    buffer: String,
    owner: Option<KeyOwner>,
    generation: u64,
    phase: KeyLeasePhase,
}

impl Default for KeyCapacityLease {
    fn default() -> Self {
        Self {
            buffer: String::new(),
            owner: None,
            generation: 0,
            phase: KeyLeasePhase::Free,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
enum PresenceBits {
    None,
    Inline(u64),
    Heap(Box<[u64]>),
}

#[derive(Clone, Default, PartialEq, Debug)]
enum EvaluatedSet {
    #[default]
    None,
    Inline(u64),
    Heap(Box<[u64]>),
    All,
}

impl EvaluatedSet {
    fn contains(&self, ordinal: u32) -> bool {
        let index = ordinal as usize;
        match self {
            Self::None => false,
            Self::Inline(word) => index < 64 && word & (1u64 << index) != 0,
            Self::Heap(words) => words
                .get(index / 64)
                .is_some_and(|word| word & (1u64 << (index % 64)) != 0),
            Self::All => true,
        }
    }

    /// One 64-bit word of the set, zero-extended past whatever storage this variant owns.
    /// `All` returns every bit set rather than allocating a dense bitset to represent it.
    fn word(&self, index: usize) -> u64 {
        match self {
            Self::None => 0,
            Self::Inline(word) if index == 0 => *word,
            Self::Inline(_) => 0,
            Self::Heap(words) => words.get(index).copied().unwrap_or(0),
            Self::All => u64::MAX,
        }
    }

    fn retained_bytes(&self) -> usize {
        match self {
            Self::Heap(words) => words.len().saturating_mul(size_of::<u64>()),
            Self::None | Self::Inline(_) | Self::All => 0,
        }
    }

    fn reset(&mut self) {
        match self {
            Self::None => {}
            Self::Inline(word) => *word = 0,
            Self::Heap(words) => words.fill(0),
            Self::All => *self = Self::None,
        }
    }
}

#[derive(Clone, Default, PartialEq, Debug)]
enum RootAnnotations {
    #[default]
    None,
    Object(EvaluatedSet),
    Array(EvaluatedSet),
}

impl RootAnnotations {
    fn retained_bytes(&self) -> usize {
        match self {
            Self::Object(set) | Self::Array(set) => set.retained_bytes(),
            Self::None => 0,
        }
    }

    fn shape_and_words(&self) -> Option<(bool, usize, bool)> {
        let (is_object, set) = match self {
            Self::Object(set) => (true, set),
            Self::Array(set) => (false, set),
            Self::None => return None,
        };
        let (words, all) = match set {
            EvaluatedSet::None => (0, false),
            EvaluatedSet::Inline(_) => (1, false),
            EvaluatedSet::Heap(words) => (words.len(), false),
            EvaluatedSet::All => (0, true),
        };
        Some((is_object, words, all))
    }

    fn word(&self, index: usize) -> u64 {
        match self {
            Self::Object(set) | Self::Array(set) => set.word(index),
            Self::None => 0,
        }
    }
}

impl PresenceBits {
    fn new(name_count: usize, limit: usize) -> Result<Self, StructuredRuntimeError> {
        match name_count {
            0 => Ok(Self::None),
            1..=64 => Ok(Self::Inline(0)),
            _ => {
                let words = name_count.div_ceil(64);
                let mut storage = Vec::new();
                storage.try_reserve_exact(words).map_err(|_| {
                    StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, limit)
                })?;
                storage.resize(words, 0);
                Ok(Self::Heap(storage.into_boxed_slice()))
            }
        }
    }

    fn contains(&self, index: usize) -> bool {
        match self {
            Self::None => false,
            Self::Inline(word) => index < 64 && (word >> index) & 1 == 1,
            Self::Heap(words) => words
                .get(index / 64)
                .is_some_and(|word| (word >> (index % 64)) & 1 == 1),
        }
    }

    fn set(&mut self, index: usize) {
        match self {
            Self::None => {}
            Self::Inline(word) if index < 64 => *word |= 1 << index,
            Self::Heap(words) if index / 64 < words.len() => {
                words[index / 64] |= 1 << (index % 64);
            }
            Self::Inline(_) | Self::Heap(_) => {}
        }
    }

    fn clear(&mut self, index: usize) {
        match self {
            Self::None => {}
            Self::Inline(word) if index < 64 => *word &= !(1 << index),
            Self::Heap(words) if index / 64 < words.len() => {
                words[index / 64] &= !(1 << (index % 64));
            }
            Self::Inline(_) | Self::Heap(_) => {}
        }
    }

    fn reset(&mut self) {
        match self {
            Self::None => {}
            Self::Inline(word) => *word = 0,
            Self::Heap(words) => words.fill(0),
        }
    }

    fn allocation_charge(&self) -> usize {
        match self {
            Self::Heap(words) => words.len().saturating_mul(size_of::<u64>()),
            Self::None | Self::Inline(_) => 0,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
struct ObjectFrame {
    node: NodeId,
    phase: ObjectPhase,
    key: String,
    /// A recycled, emptied key buffer from a prior committed property, reused by `start_key`
    /// instead of allocating a fresh `String` for every property.
    spare_key: String,
    /// Incremented whenever this frame begins a new key, preventing a lease prepared for an old
    /// key from being observed after a quote/comma boundary reuses the same frame depth.
    key_generation: u64,
    key_decoder: JsonStringDecoder,
    property_name: Option<CursorId>,
    resolved_value: Option<ValueObligations>,
    missing_required: Vec<u64>,
    seen_known: Vec<u64>,
    /// Dynamic keys seen so far, by hash bucket (a bucket holds more than one `KeyId` only on a
    /// genuine hash collision): O(1) average membership with no owned string per entry.
    seen_dynamic: FxHashMap<u64, DynamicSlot>,
    /// `key_arena`'s fill level when this frame was pushed; the floor `commit_checkpoint` may
    /// reclaim down to once this frame closes, never invalidating a still-open ancestor's keys.
    arena_start: (u32, u32),
    property_count: u32,
    dependent: Option<Box<DependentFrame>>,
    dependent_required_seen: PresenceBits,
    evaluated: EvaluatedSet,
}

#[derive(Clone, Copy)]
struct DependentCursorMark {
    dependency: u32,
    cursor: CursorId,
    mark: CursorMark,
}

#[derive(Clone)]
struct DependentFrame {
    cursors: Vec<CursorId>,
    active_words: Vec<u64>,
    owned_words: Vec<u64>,
    triggered_words: Vec<u64>,
    marks: Vec<DependentCursorMark>,
    allocation_charge: usize,
}

impl PartialEq for DependentFrame {
    fn eq(&self, other: &Self) -> bool {
        self.cursors == other.cursors
            && self.active_words == other.active_words
            && self.owned_words == other.owned_words
            && self.triggered_words == other.triggered_words
            && self.allocation_charge == other.allocation_charge
    }
}

impl std::fmt::Debug for DependentFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DependentFrame")
            .field("cursors", &self.cursors)
            .field("active_words", &self.active_words)
            .field("owned_words", &self.owned_words)
            .field("triggered_words", &self.triggered_words)
            .field("allocation_charge", &self.allocation_charge)
            .finish()
    }
}

/// Bounded upper bound on how many distinct schemas one property value may need to satisfy
/// inline (a known name overlapping a pattern, or several overlapping patterns) before spilling.
const MAX_INLINE_OBLIGATIONS: usize = 4;

/// Every schema one property value must satisfy simultaneously. Inline for the common 0-4
/// case; a fresh key never allocates unless it needs a fifth distinct schema.
#[derive(Clone, PartialEq, Debug, Default)]
struct SchemaObligationSet {
    inline: [NodeId; MAX_INLINE_OBLIGATIONS],
    inline_len: u8,
    spill: Vec<NodeId>,
}

impl SchemaObligationSet {
    fn len(&self) -> usize {
        self.inline_len as usize + self.spill.len()
    }

    fn iter(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.inline[..self.inline_len as usize]
            .iter()
            .copied()
            .chain(self.spill.iter().copied())
    }

    fn contains(&self, id: NodeId) -> bool {
        self.iter().any(|x| x == id)
    }

    /// Test-only insert with no session accounting. A live session always goes through
    /// `StructuredState::insert_obligation`, which reserves and charges the spill growth.
    #[cfg(test)]
    fn insert_uncounted(&mut self, id: NodeId) -> bool {
        if self.contains(id) {
            return false;
        }
        if (self.inline_len as usize) < MAX_INLINE_OBLIGATIONS {
            self.inline[self.inline_len as usize] = id;
            self.inline_len += 1;
        } else {
            self.spill.push(id);
        }
        true
    }
}

/// What one property value must satisfy: either `additionalProperties: true` (matched by
/// `AnyJson`, no schema at all), or a concrete set of schemas - never both at once.
#[derive(Clone, PartialEq, Debug)]
enum ValueObligations {
    AnyJson,
    Schemas(SchemaObligationSet),
}

impl Default for ValueObligations {
    fn default() -> Self {
        ValueObligations::Schemas(SchemaObligationSet::default())
    }
}

/// One property value's lockstep validation for two or more schemas.
#[derive(Clone)]
struct ObligationFrame {
    cursors: Vec<CursorId>,
    marks: Vec<CursorMark>,
    allocation_charge: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct BranchMark {
    branch_index: u32,
    cursor: CursorId,
    mark: CursorMark,
}

#[derive(Clone)]
struct CombinatorFrame {
    node: NodeId,
    kind: CombinatorKind,
    cursors: Vec<CursorId>,
    active_words: Vec<u64>,
    owned_words: Vec<u64>,
    marks: Vec<BranchMark>,
    allocation_charge: usize,
}

impl std::fmt::Debug for CombinatorFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CombinatorFrame")
            .field("node", &self.node)
            .field("kind", &self.kind)
            .field("cursors", &self.cursors)
            .field("active_words", &self.active_words)
            .field("owned_words", &self.owned_words)
            .finish()
    }
}

impl PartialEq for CombinatorFrame {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
            && self.kind == other.kind
            && self.cursors == other.cursors
            && self.active_words == other.active_words
            && self.owned_words == other.owned_words
            && self.marks == other.marks
    }
}

impl std::fmt::Debug for ObligationFrame {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ObligationFrame")
            .field("cursors", &self.cursors)
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StringPhase {
    BeforeQuote,
    Body,
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct StringFrame {
    node: NodeId,
    phase: StringPhase,
    decoder: JsonStringDecoder,
    scalars: u32,
    pattern_state: Option<StateId>,
}

/// Whether a safe-token slice can be admitted without walking it.
/// Uncertain cases return `Unknown` and use the exact walker.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SliceCertificate {
    AllNoBoundary,
    Unknown,
}

#[derive(Clone, Copy)]
pub(crate) enum SliceProof {
    All(NodeId, StateId),
    CandidateLive(NodeId, StateId),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SlicePreparation {
    Ready,
    Ineligible,
}

pub(crate) struct ProofContext {
    visited: [usize; 64],
    depth: usize,
    fuel: usize,
    max_bytes: usize,
    projected_memory: Option<(usize, usize)>,
    pub(crate) body_prefix: bool,
    // Optional branch bounds cannot exclude tokens accepted by another branch.
    required: bool,
    pub(crate) remaining_scalars: Option<usize>,
    #[cfg(test)]
    fail_key_lease_at: Option<usize>,
    #[cfg(test)]
    key_lease_acquisitions: usize,
}

impl ProofContext {
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            visited: [0; 64],
            depth: 0,
            fuel: 64,
            max_bytes,
            projected_memory: None,
            body_prefix: false,
            required: true,
            remaining_scalars: None,
            #[cfg(test)]
            fail_key_lease_at: None,
            #[cfg(test)]
            key_lease_acquisitions: 0,
        }
    }

    pub(crate) fn with_fuel(max_bytes: usize, fuel: usize) -> Self {
        Self {
            fuel,
            ..Self::new(max_bytes)
        }
    }

    pub(crate) fn for_preparation(
        max_bytes: usize,
        live: usize,
        limit: usize,
        fuel: usize,
    ) -> Self {
        Self {
            projected_memory: Some((live, limit)),
            ..Self::with_fuel(max_bytes, fuel)
        }
    }

    fn admit_growth(&mut self, request: GrowthRequest) -> bool {
        let Some((live, limit)) = self.projected_memory else {
            return request
                .len
                .checked_add(request.additional)
                .is_some_and(|n| n <= request.capacity);
        };
        let Ok(plan) = plan_growth(request, live, limit) else {
            return false;
        };
        let Some(projected) = request
            .len
            .checked_add(plan.reserve_additional)
            .and_then(|n| {
                n.saturating_sub(request.capacity)
                    .checked_mul(request.item_size)
            })
            .and_then(|n| live.checked_add(n))
        else {
            return false;
        };
        self.projected_memory = Some((projected, limit));
        true
    }

    #[cfg(test)]
    fn fail_key_lease_at(&mut self, acquisition: usize) {
        self.fail_key_lease_at = Some(acquisition);
    }

    fn before_key_lease_acquisition(&mut self, limit: usize) -> Result<(), StructuredRuntimeError> {
        #[cfg(test)]
        {
            let acquisition = self.key_lease_acquisitions;
            self.key_lease_acquisitions = acquisition.checked_add(1).ok_or_else(|| {
                StructuredRuntimeError::new(LimitKind::AllocationBytes, usize::MAX, limit)
            })?;
            if self.fail_key_lease_at == Some(acquisition) {
                return Err(StructuredRuntimeError::new(
                    LimitKind::AllocationBytes,
                    1,
                    limit,
                ));
            }
        }
        let _ = limit;
        Ok(())
    }
}

pub(crate) struct LocalSliceResidual<'a> {
    pub(crate) node: NodeId,
    pub(crate) engine: Option<&'a crate::automaton::RefEngine>,
    pub(crate) pattern_state: Option<StateId>,
    pub(crate) remaining_document_bytes: usize,
    /// Decoded scalars still allowed here, when a `maxLength` bounds them. A byte budget cannot
    /// express this: one four-byte token can spend a single scalar.
    pub(crate) remaining_scalars: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct StringEnumFrame {
    node: NodeId,
    phase: StringPhase,
    decoder: JsonStringDecoder,
    lo: u32,
    hi: u32,
    decoded_bytes: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct NumberAccumulator {
    negative: bool,
    fraction: bool,
    exponent: bool,
    exponent_negative: bool,
    exponent_value: i64,
    fraction_digits: u32,
    core_digits: u32,
    leading: [u8; 20],
    leading_len: u8,
    pending_zeros: u32,
    core_mod: u64,
    bytes: u32,
    started: bool,
    finished: bool,
}

impl NumberAccumulator {
    const fn new() -> Self {
        Self {
            negative: false,
            fraction: false,
            exponent: false,
            exponent_negative: false,
            exponent_value: 0,
            fraction_digits: 0,
            core_digits: 0,
            leading: [0; 20],
            leading_len: 0,
            pending_zeros: 0,
            core_mod: 0,
            bytes: 0,
            started: false,
            finished: false,
        }
    }

    fn push(&mut self, byte: u8, modulus: u64) {
        if WS.contains(&byte) {
            if self.started {
                self.finished = true;
            }
            return;
        }
        self.started = true;
        self.bytes = self.bytes.saturating_add(1);
        if self.finished {
            return;
        }
        match byte {
            b'-' if !self.exponent && self.core_digits == 0 && !self.fraction => {
                self.negative = true;
            }
            b'-' if self.exponent => self.exponent_negative = true,
            b'+' if self.exponent => {}
            b'.' => self.fraction = true,
            b'e' | b'E' => self.exponent = true,
            digit @ b'0'..=b'9' if self.exponent => {
                self.exponent_value = self
                    .exponent_value
                    .saturating_mul(10)
                    .saturating_add(i64::from(digit - b'0'))
                    .min(1_000_000_000);
            }
            digit @ b'0'..=b'9' => {
                if self.fraction {
                    self.fraction_digits = self.fraction_digits.saturating_add(1);
                }
                let digit = digit - b'0';
                if digit == 0 {
                    if self.core_digits > 0 {
                        self.pending_zeros = self.pending_zeros.saturating_add(1);
                    }
                    return;
                }
                self.flush_pending_zeros(modulus);
                self.pending_zeros = 0;
                self.push_core_digit(digit, modulus);
            }
            _ => {}
        }
    }

    fn flush_pending_zeros(&mut self, modulus: u64) {
        let available = self
            .leading
            .len()
            .saturating_sub(usize::from(self.leading_len));
        let copied = available.min(self.pending_zeros as usize);
        self.leading_len = self.leading_len.saturating_add(copied as u8);
        self.core_digits = self.core_digits.saturating_add(self.pending_zeros);
        if modulus > 1 {
            let factor = modular_power_of_ten(u64::from(self.pending_zeros), modulus);
            self.core_mod =
                ((u128::from(self.core_mod) * u128::from(factor)) % u128::from(modulus)) as u64;
        }
    }

    fn push_core_digit(&mut self, digit: u8, modulus: u64) {
        if usize::from(self.leading_len) < self.leading.len() {
            self.leading[usize::from(self.leading_len)] = digit;
            self.leading_len += 1;
        }
        self.core_digits = self.core_digits.saturating_add(1);
        if modulus > 1 {
            self.core_mod =
                ((u128::from(self.core_mod) * 10 + u128::from(digit)) % u128::from(modulus)) as u64;
        }
    }

    fn signed_exponent(&self) -> i64 {
        if self.exponent_negative {
            -self.exponent_value
        } else {
            self.exponent_value
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct NumberFrame {
    node: NodeId,
    state: StateId,
    accumulator: NumberAccumulator,
}

impl PartialEq for ObligationFrame {
    fn eq(&self, other: &Self) -> bool {
        self.cursors == other.cursors && self.marks == other.marks
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ArrayPhase {
    Start,
    BeforeFirstItemOrClose,
    AfterValueBeforeCommaOrClose,
    AfterCommaBeforeValue,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ContainsCandidate {
    None,
    Always,
    Dead,
    Cursor(CursorId),
}

/// Tracks one in-progress array item for `uniqueItems` and `contains`.
#[cfg_attr(test, derive(Clone))]
#[derive(PartialEq, Debug)]
struct ItemTracking {
    builder: Option<CanonicalBuilder>,
    contains: ContainsCandidate,
}

/// Schema-bounded array state too large to unroll into a byte DFA.
/// Its index selects the next prefix or tail item schema.
#[cfg_attr(test, derive(Clone))]
#[derive(PartialEq, Debug)]
struct ArrayFrame {
    node: NodeId,
    phase: ArrayPhase,
    index: u32,
    canonical: Option<Vec<CanonicalSet>>,
    contains_count: u32,
    item: Option<ItemTracking>,
    evaluated: EvaluatedSet,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ContainsCountClass {
    BelowMinimum,
    WithinAllowedRange,
    AtMaximum,
    AboveMaximum,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RemainingItemCapacity {
    Bounded(u32),
    Unbounded,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum UnevaluatedItemState {
    AlreadyAnnotated,
    RequiresContainsAnnotation,
    Irrelevant,
    Unknown,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ContainsCandidateState {
    None,
    Always,
    Dead,
    Regular(NodeId, StateId),
    Structured(CursorId),
}

/// Abstract state for a lexical slice inside an active array item.
/// It tracks whether `contains` or item annotations can change future validity.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ArrayItemProductState {
    item_parser: SliceCertificate,
    candidate: ContainsCandidateState,
    count: ContainsCountClass,
    remaining: RemainingItemCapacity,
    unevaluated: UnevaluatedItemState,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ContainsSlicePolicy {
    Unrestricted,
    MustStayLive(NodeId, StateId),
    MustKeepStructuredCandidate(CursorId),
    Unknown,
}

struct PreparedFrame {
    frame: Frame,
    charge: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NegationInnerState {
    Active,
    PrunedOwned,
    Released,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct NegationFrame {
    node: NodeId,
    syntax: CursorId,
    inner: CursorId,
    inner_state: NegationInnerState,
}

#[cfg_attr(test, derive(Clone))]
#[derive(PartialEq, Debug)]
struct UnevaluatedFrame {
    node: NodeId,
    kind: UnevaluatedKind,
    unevaluated: NodeId,
    scope: CursorId,
    tracker: CursorId,
    current_candidate: Option<UnevaluatedCandidate>,
    candidates: EvaluatedSet,
    location_count: u32,
    instance_kind: Option<UnevaluatedKind>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct UnevaluatedCandidate {
    ordinal: u32,
    cursor: CursorId,
    dead: bool,
}

impl ArrayFrame {
    fn canonical_ref(&self) -> Option<&CanonicalSet> {
        self.canonical.as_deref().and_then(|sets| sets.first())
    }

    fn canonical_mut(&mut self) -> Option<&mut CanonicalSet> {
        self.canonical
            .as_deref_mut()
            .and_then(|sets| sets.first_mut())
    }
}

#[cfg_attr(test, derive(Clone))]
#[derive(PartialEq, Debug)]
enum Frame {
    Regular { node: NodeId, state: StateId },
    Number(NumberFrame),
    String(StringFrame),
    StringEnum(StringEnumFrame),
    Object(ObjectFrame),
    Obligation(ObligationFrame),
    Combinator(CombinatorFrame),
    Negation(NegationFrame),
    Unevaluated(UnevaluatedFrame),
    Array(ArrayFrame),
    Any(AnyFrame),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AnyArrayPhase {
    BeforeFirstOrClose,
    AfterValue,
    AfterComma,
}

/// Object state for `additionalProperties: true`.
/// Dynamic keys use the shared arena and duplicate-key tracking.
#[derive(Clone, PartialEq, Debug)]
struct AnyObjectState {
    phase: ObjectPhase,
    key: String,
    /// Changes whenever a new logical key starts, invalidating stale speculative leases.
    key_generation: u64,
    /// A recycled, emptied key buffer from a prior committed property (see `ObjectFrame::spare_key`).
    spare_key: String,
    key_decoder: JsonStringDecoder,
    seen_dynamic: FxHashMap<u64, DynamicSlot>,
    /// `key_arena`'s fill level when this object was allocated; see `ObjectFrame::arena_start`.
    arena_start: (u32, u32),
    properties: u32,
}

/// A compact handle into `StructuredState::any_object_arena`, replacing a per-object `Box`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct AnyObjectId(u32);

/// One unconstrained JSON value matched without recursion.
/// Arrays and objects use child frames; objects use compact arena handles.
#[derive(Clone, PartialEq, Debug)]
enum AnyFrame {
    Scalar {
        phase: ScalarPhase,
        bytes: u32,
    },
    StringBody {
        decoder: JsonStringDecoder,
        bytes: u32,
    },
    Array {
        phase: AnyArrayPhase,
        items: u32,
    },
    Object(AnyObjectId),
}

#[derive(Clone, Copy)]
enum AnnotationTarget {
    Object(usize),
    Array(usize),
    RootObject,
    RootArray,
    UnevaluatedCandidates(usize),
}

/// One reversible mutation. `rollback` pops these in reverse, restoring exactly what changed -
/// no entry ever holds more than the single field (or one owned frame) it touched.
enum Undo {
    RollbackUnevaluatedCursor {
        cursor: CursorId,
        mark: CursorMark,
    },
    StartUnevaluatedCandidate {
        depth: usize,
        cursor: CursorId,
    },
    FinishUnevaluatedCandidate {
        depth: usize,
        ordinal: u32,
        cursor: CursorId,
        dead: bool,
        old_count: u32,
    },
    MarkUnevaluatedCandidateDead {
        depth: usize,
    },
    SetUnevaluatedInstanceKind {
        depth: usize,
    },
    SetRootAnnotations {
        old: RootAnnotations,
        restore_target: AnnotationTarget,
    },
    ReplaceRootAnnotations {
        old: RootAnnotations,
    },
    EvaluatedWord {
        target: AnnotationTarget,
        word: usize,
        old: u64,
    },
    ReplaceEvaluatedSet {
        target: AnnotationTarget,
        old: EvaluatedSet,
        new_charge: usize,
    },
    SetRegularState {
        depth: usize,
        old: StateId,
    },
    SetNumberFrame {
        depth: usize,
        old: NumberFrame,
    },
    RollbackObligationCursor {
        cursor: CursorId,
        mark: CursorMark,
    },
    RollbackCombinatorCursor {
        cursor: CursorId,
        mark: CursorMark,
    },
    ReactivateCombinatorBranch {
        depth: usize,
        branch: u32,
    },
    RollbackNegationCursor {
        cursor: CursorId,
        mark: CursorMark,
    },
    ReactivateNegationInner {
        depth: usize,
    },
    RollbackDependentCursor {
        cursor: CursorId,
        mark: CursorMark,
    },
    ReactivateDependentSchema {
        depth: usize,
        dependency: u32,
    },
    ClearDependentTrigger {
        depth: usize,
        dependency: u32,
    },
    ClearDependentRequiredPresence {
        depth: usize,
        name: u32,
    },
    SetStringFrame {
        depth: usize,
        old: StringFrame,
    },
    SetStringEnumFrame {
        depth: usize,
        old: StringEnumFrame,
    },
    SetObjectPhase {
        depth: usize,
        old: ObjectPhase,
    },
    SetArrayPhase {
        depth: usize,
        old: ArrayPhase,
    },
    RestoreArrayIndex {
        depth: usize,
        old: u32,
    },
    StartArrayItem {
        depth: usize,
    },
    RestoreItemContainsCandidate {
        depth: usize,
        old: ContainsCandidate,
    },
    RollbackContainsCursor {
        cursor: CursorId,
        mark: CursorMark,
    },
    RollbackCanonicalBuild {
        depth: usize,
        mark: BuilderMark,
    },
    FinishArrayItem {
        depth: usize,
        old: ItemTracking,
    },
    RollbackCanonicalInsert {
        depth: usize,
        fingerprint: u64,
        old_bucket_len: usize,
        bucket_created: bool,
        bucket_capacity_bytes: usize,
    },
    RestoreContainsCount {
        depth: usize,
        old: u32,
    },
    ReplaceAnyFrame {
        depth: usize,
        old: AnyFrame,
    },
    SetAnyScalar {
        depth: usize,
        old_phase: ScalarPhase,
        old_bytes: u32,
    },
    SetAnyString {
        depth: usize,
        old_decoder: JsonStringDecoder,
        old_bytes: u32,
    },
    SetAnyArrayPhase {
        depth: usize,
        old: AnyArrayPhase,
    },
    RestoreAnyArrayItems {
        depth: usize,
        old: u32,
    },
    SetAnyObjectPhase {
        depth: usize,
        old: ObjectPhase,
    },
    TruncateAnyKey {
        depth: usize,
        old_len: usize,
        lease_generation: Option<u64>,
    },
    ResetAnyKey {
        depth: usize,
        old_key: String,
        old_generation: u64,
        old_decoder: JsonStringDecoder,
    },
    RestoreAnyObjectDecoder {
        depth: usize,
        old: JsonStringDecoder,
    },
    AnyRemoveSeenDynamicBucket {
        depth: usize,
        hash: u64,
    },
    AnyRestoreSeenDynamicOne {
        depth: usize,
        hash: u64,
        id: KeyId,
    },
    AnyTruncateSeenDynamicMany {
        depth: usize,
        hash: u64,
        old_len: usize,
    },
    RestoreAnyObjectProperties {
        depth: usize,
        old: u32,
    },
    TruncateKey {
        depth: usize,
        old_len: usize,
        lease_generation: Option<u64>,
    },
    ResetKey {
        depth: usize,
        old_key: String,
        old_generation: u64,
        old_decoder: JsonStringDecoder,
    },
    RestoreDecoder {
        depth: usize,
        old: JsonStringDecoder,
    },
    StartPropertyName {
        depth: usize,
        cursor: CursorId,
    },
    RollbackPropertyName {
        cursor: CursorId,
        mark: CursorMark,
    },
    FinishPropertyName {
        depth: usize,
        cursor: CursorId,
    },
    RestoreResolvedValue {
        depth: usize,
        old: Option<ValueObligations>,
    },
    RestorePropertyCount {
        depth: usize,
        old: u32,
    },
    RestoreRequiredWord {
        depth: usize,
        word: usize,
        old: u64,
    },
    RemoveSeenKnown {
        depth: usize,
        id: PropertyId,
    },
    RemoveSeenDynamicBucket {
        depth: usize,
        hash: u64,
    },
    RestoreSeenDynamicOne {
        depth: usize,
        hash: u64,
        id: KeyId,
    },
    TruncateSeenDynamicMany {
        depth: usize,
        hash: u64,
        old_len: usize,
    },
    InternKey {
        old_arena_len: u32,
        old_span_count: u32,
    },
    PopFrame,
    PushFrame(Frame, ScopeLease),
    SetRootClosed {
        old: bool,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Checkpoint {
    undo_len: usize,
    document_bytes: usize,
    accepting: bool,
    dead: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct CursorId {
    index: u32,
    generation: u32,
}

const NO_SCOPE: u32 = u32::MAX;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ScopeId(u32);

struct ScopeSlot {
    resource: AtomicU32,
    parent: AtomicU32,
    depth: AtomicU32,
    refs: AtomicUsize,
    next_free: AtomicU32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct DynamicResolutionKey {
    scope_index: u32,
    scope_generation: u64,
    anchor: AnchorId,
    initial_target: NodeId,
}

struct DynamicMemoSlot {
    version: AtomicU64,
    scope_index: AtomicU32,
    scope_generation: AtomicU64,
    initial_target: AtomicU32,
    target: AtomicU32,
}

struct ScopeArena {
    slots: Box<[ScopeSlot]>,
    generations: Box<[AtomicU64]>,
    dynamic_memo: Box<[DynamicMemoSlot]>,
    free_head: AtomicU32,
    #[cfg(feature = "bench-internals")]
    dynamic_resolutions: AtomicU64,
    #[cfg(feature = "bench-internals")]
    dynamic_scope_steps: AtomicU64,
}

struct ScopeLease {
    arena: Arc<ScopeArena>,
    id: ScopeId,
}

impl ScopeArena {
    #[cfg(test)]
    fn new(capacity: usize) -> Result<(Arc<Self>, usize), StructuredRuntimeError> {
        Self::new_with_anchor_capacity(capacity, 0)
    }

    fn new_with_anchor_capacity(
        capacity: usize,
        anchor_capacity: usize,
    ) -> Result<(Arc<Self>, usize), StructuredRuntimeError> {
        let capacity = capacity.max(1);
        u32::try_from(capacity).map_err(|_| {
            StructuredRuntimeError::new(LimitKind::ActiveValidators, capacity, u32::MAX as usize)
        })?;
        let mut slots = Vec::new();
        slots.try_reserve_exact(capacity).map_err(|_| {
            StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, usize::MAX)
        })?;
        for index in 0..capacity {
            let next = index
                .checked_add(1)
                .filter(|next| *next < capacity)
                .and_then(|next| u32::try_from(next).ok())
                .unwrap_or(NO_SCOPE);
            slots.push(ScopeSlot {
                resource: AtomicU32::new(0),
                parent: AtomicU32::new(NO_SCOPE),
                depth: AtomicU32::new(0),
                refs: AtomicUsize::new(0),
                next_free: AtomicU32::new(next),
            });
        }
        let generations = allocate_generations(capacity)?;
        let dynamic_memo = allocate_dynamic_memo(anchor_capacity)?;
        let bytes = slots
            .capacity()
            .checked_mul(size_of::<ScopeSlot>())
            .and_then(|value| {
                value.checked_add(generations.len().checked_mul(size_of::<AtomicU64>())?)
            })
            .and_then(|value| {
                value.checked_add(
                    dynamic_memo
                        .len()
                        .checked_mul(size_of::<DynamicMemoSlot>())?,
                )
            })
            .and_then(|value| value.checked_add(size_of::<ScopeArena>()))
            .ok_or_else(|| {
                StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, usize::MAX)
            })?;
        Ok((
            Arc::new(Self {
                slots: slots.into_boxed_slice(),
                generations,
                dynamic_memo,
                free_head: AtomicU32::new(0),
                #[cfg(feature = "bench-internals")]
                dynamic_resolutions: AtomicU64::new(0),
                #[cfg(feature = "bench-internals")]
                dynamic_scope_steps: AtomicU64::new(0),
            }),
            bytes,
        ))
    }

    fn root(self: &Arc<Self>, resource: ResourceId) -> Result<ScopeLease, StructuredRuntimeError> {
        self.allocate(None, resource, 1)
    }

    fn enter(
        self: &Arc<Self>,
        parent: &ScopeLease,
        resource: ResourceId,
        max_depth: u32,
    ) -> Result<ScopeLease, StructuredRuntimeError> {
        if self.resource(parent.id)? == resource {
            return Ok(parent.clone());
        }
        let depth = self.depth(parent.id)?.checked_add(1).ok_or_else(|| {
            StructuredRuntimeError::new(
                LimitKind::DynamicScopeDepth,
                usize::MAX,
                max_depth as usize,
            )
        })?;
        if depth > max_depth {
            return Err(StructuredRuntimeError::new(
                LimitKind::DynamicScopeDepth,
                depth as usize,
                max_depth as usize,
            ));
        }
        self.retain(parent.id)?;
        match self.allocate(Some(parent.id), resource, depth) {
            Ok(lease) => Ok(lease),
            Err(error) => {
                self.release(parent.id);
                Err(error)
            }
        }
    }

    fn allocate(
        self: &Arc<Self>,
        parent: Option<ScopeId>,
        resource: ResourceId,
        depth: u32,
    ) -> Result<ScopeLease, StructuredRuntimeError> {
        let mut head = self.free_head.load(Ordering::Acquire);
        loop {
            if head == NO_SCOPE {
                return Err(StructuredRuntimeError::new(
                    LimitKind::ActiveValidators,
                    self.slots.len().saturating_add(1),
                    self.slots.len(),
                ));
            }
            let slot = self.slot(ScopeId(head))?;
            let next = slot.next_free.load(Ordering::Relaxed);
            match self.free_head.compare_exchange_weak(
                head,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let Some(generation) = self.generations[head as usize]
                        .load(Ordering::Relaxed)
                        .checked_add(1)
                    else {
                        head = self.free_head.load(Ordering::Acquire);
                        continue;
                    };
                    self.generations[head as usize].store(generation, Ordering::Relaxed);
                    slot.resource.store(resource.get(), Ordering::Relaxed);
                    slot.parent
                        .store(parent.map_or(NO_SCOPE, |id| id.0), Ordering::Relaxed);
                    slot.depth.store(depth, Ordering::Relaxed);
                    slot.refs.store(1, Ordering::Release);
                    return Ok(ScopeLease {
                        arena: self.clone(),
                        id: ScopeId(head),
                    });
                }
                Err(observed) => head = observed,
            }
        }
    }

    fn retain(&self, id: ScopeId) -> Result<(), StructuredRuntimeError> {
        let slot = self.slot(id)?;
        let mut current = slot.refs.load(Ordering::Relaxed);
        loop {
            let next = current
                .checked_add(1)
                .filter(|_| current != 0)
                .ok_or_else(|| self.error())?;
            match slot.refs.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => current = observed,
            }
        }
    }

    fn release(&self, mut id: ScopeId) {
        loop {
            let Ok(slot) = self.slot(id) else {
                return;
            };
            if slot.refs.fetch_sub(1, Ordering::AcqRel) != 1 {
                return;
            }
            let parent = slot.parent.load(Ordering::Relaxed);
            let mut head = self.free_head.load(Ordering::Acquire);
            loop {
                slot.next_free.store(head, Ordering::Relaxed);
                match self.free_head.compare_exchange_weak(
                    head,
                    id.0,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => break,
                    Err(observed) => head = observed,
                }
            }
            if parent == NO_SCOPE {
                return;
            }
            id = ScopeId(parent);
        }
    }

    fn resource(&self, id: ScopeId) -> Result<ResourceId, StructuredRuntimeError> {
        Ok(ResourceId(self.slot(id)?.resource.load(Ordering::Relaxed)))
    }

    fn parent(&self, id: ScopeId) -> Result<Option<ScopeId>, StructuredRuntimeError> {
        let parent = self.slot(id)?.parent.load(Ordering::Relaxed);
        Ok((parent != NO_SCOPE).then_some(ScopeId(parent)))
    }

    fn depth(&self, id: ScopeId) -> Result<u32, StructuredRuntimeError> {
        Ok(self.slot(id)?.depth.load(Ordering::Relaxed))
    }

    fn generation(&self, id: ScopeId) -> Result<u64, StructuredRuntimeError> {
        self.generations
            .get(id.0 as usize)
            .map(|generation| generation.load(Ordering::Relaxed))
            .ok_or_else(|| self.error())
    }

    fn slot(&self, id: ScopeId) -> Result<&ScopeSlot, StructuredRuntimeError> {
        self.slots.get(id.0 as usize).ok_or_else(|| self.error())
    }

    fn error(&self) -> StructuredRuntimeError {
        StructuredRuntimeError::new(LimitKind::ActiveValidators, usize::MAX, self.slots.len())
    }

    fn dynamic_memo_get(&self, key: DynamicResolutionKey) -> Option<NodeId> {
        let slot = self.dynamic_memo.get(key.anchor.get() as usize)?;
        let before = slot.version.load(Ordering::Acquire);
        if before & 1 != 0 {
            return None;
        }
        let matches = slot.scope_index.load(Ordering::Relaxed) == key.scope_index
            && slot.scope_generation.load(Ordering::Relaxed) == key.scope_generation
            && slot.initial_target.load(Ordering::Relaxed) == key.initial_target.get();
        let target = slot.target.load(Ordering::Relaxed);
        let after = slot.version.load(Ordering::Acquire);
        (matches && before == after && after & 1 == 0).then_some(NodeId(target))
    }

    fn dynamic_memo_store(&self, key: DynamicResolutionKey, target: NodeId) {
        let Some(slot) = self.dynamic_memo.get(key.anchor.get() as usize) else {
            return;
        };
        let mut version = slot.version.load(Ordering::Acquire);
        loop {
            if version & 1 != 0 || version > u64::MAX - 2 {
                return;
            }
            match slot.version.compare_exchange_weak(
                version,
                version + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(observed) => version = observed,
            }
        }
        slot.scope_index.store(key.scope_index, Ordering::Relaxed);
        slot.scope_generation
            .store(key.scope_generation, Ordering::Relaxed);
        slot.initial_target
            .store(key.initial_target.get(), Ordering::Relaxed);
        slot.target.store(target.get(), Ordering::Relaxed);
        slot.version.store(version + 2, Ordering::Release);
    }
}

fn allocate_dynamic_memo(
    capacity: usize,
) -> Result<Box<[DynamicMemoSlot]>, StructuredRuntimeError> {
    let mut entries = Vec::new();
    entries.try_reserve_exact(capacity).map_err(|_| {
        StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, usize::MAX)
    })?;
    for _ in 0..capacity {
        entries.push(DynamicMemoSlot {
            version: AtomicU64::new(0),
            scope_index: AtomicU32::new(NO_SCOPE),
            scope_generation: AtomicU64::new(0),
            initial_target: AtomicU32::new(u32::MAX),
            target: AtomicU32::new(u32::MAX),
        });
    }
    Ok(entries.into_boxed_slice())
}

fn allocate_generations(capacity: usize) -> Result<Box<[AtomicU64]>, StructuredRuntimeError> {
    let mut generations = Vec::new();
    generations.try_reserve_exact(capacity).map_err(|_| {
        StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, usize::MAX)
    })?;
    for _ in 0..capacity {
        generations.push(AtomicU64::new(0));
    }
    Ok(generations.into_boxed_slice())
}

impl Clone for ScopeLease {
    fn clone(&self) -> Self {
        self.arena
            .retain(self.id)
            .expect("a live scope lease has a nonzero validated reference count");
        Self {
            arena: self.arena.clone(),
            id: self.id,
        }
    }
}

impl Drop for ScopeLease {
    fn drop(&mut self) {
        self.arena.release(self.id);
    }
}

fn scopes_are_equivalent(left: &ScopeLease, right: &ScopeLease) -> bool {
    if !Arc::ptr_eq(&left.arena, &right.arena) {
        return false;
    }
    let mut left_id = Some(left.id);
    let mut right_id = Some(right.id);
    loop {
        match (left_id, right_id) {
            (None, None) => return true,
            (Some(left_id_value), Some(right_id_value)) => {
                let Ok(left_resource) = left.arena.resource(left_id_value) else {
                    return false;
                };
                let Ok(right_resource) = right.arena.resource(right_id_value) else {
                    return false;
                };
                if left_resource != right_resource {
                    return false;
                }
                let Ok(next_left) = left.arena.parent(left_id_value) else {
                    return false;
                };
                let Ok(next_right) = right.arena.parent(right_id_value) else {
                    return false;
                };
                left_id = next_left;
                right_id = next_right;
            }
            _ => return false,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct StructuredCursorId(u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CursorSlot {
    Regular { node: NodeId, state: StateId },
    Structured(StructuredCursorId),
    Vacant,
}

/// How many `free_slots`/`free_structured` entries a commit will push, tallied by real slot kind.
#[derive(Default)]
struct ReleaseCounts {
    free_slots: usize,
    free_structured: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CursorStateMark {
    Regular(StateId),
    Structured {
        cursor: StructuredCursorId,
        mark: Checkpoint,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct CursorMark {
    generation: u32,
    epoch: u32,
    state: CursorStateMark,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CursorStep {
    Alive,
    Complete,
    Dead,
}

struct ValidatorLedger {
    live: AtomicUsize,
    limit: usize,
}

impl ValidatorLedger {
    fn ensure_available(&self, additional: usize) -> Result<(), StructuredRuntimeError> {
        let current = self.live.load(Ordering::Relaxed);
        let next = current.checked_add(additional).ok_or_else(|| {
            StructuredRuntimeError::new(LimitKind::ActiveValidators, usize::MAX, self.limit)
        })?;
        if next > self.limit {
            return Err(StructuredRuntimeError::new(
                LimitKind::ActiveValidators,
                next,
                self.limit,
            ));
        }
        Ok(())
    }

    fn try_acquire(&self) -> Result<(), StructuredRuntimeError> {
        self.try_acquire_many(1)
    }

    fn try_acquire_many(&self, additional: usize) -> Result<(), StructuredRuntimeError> {
        let mut current = self.live.load(Ordering::Relaxed);
        loop {
            let next = current.checked_add(additional).ok_or_else(|| {
                StructuredRuntimeError::new(LimitKind::ActiveValidators, usize::MAX, self.limit)
            })?;
            if next > self.limit {
                return Err(StructuredRuntimeError::new(
                    LimitKind::ActiveValidators,
                    next,
                    self.limit,
                ));
            }
            match self.live.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => current = observed,
            }
        }
    }

    fn try_release(&self) -> bool {
        self.try_release_many(1)
    }

    fn try_release_many(&self, released: usize) -> bool {
        let mut current = self.live.load(Ordering::Relaxed);
        loop {
            let Some(next) = current.checked_sub(released) else {
                return false;
            };
            match self.live.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }
}

struct ValidatorReservation {
    ledger: Arc<ValidatorLedger>,
    armed: bool,
}

impl ValidatorReservation {
    fn new(ledger: Arc<ValidatorLedger>) -> Self {
        Self {
            ledger,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ValidatorReservation {
    fn drop(&mut self) {
        if self.armed {
            let released = self.ledger.try_release();
            debug_assert!(released, "validator reservation ledger underflow");
        }
    }
}

fn initial_frame_floor(plan: &StructuredPlan, node: NodeId) -> Option<usize> {
    let owned = match plan.node(node) {
        NodePlan::Object(object) => object
            .required_template
            .len()
            .checked_mul(size_of::<u64>())?
            .checked_mul(2)?,
        NodePlan::Array(array) if array.unique_items => size_of::<CanonicalSet>(),
        NodePlan::Regular(_)
        | NodePlan::Number { .. }
        | NodePlan::String(_)
        | NodePlan::StringEnum(_)
        | NodePlan::Array(_)
        | NodePlan::Combinator(_)
        | NodePlan::Negation { .. }
        | NodePlan::Unevaluated(_)
        | NodePlan::Ref { .. }
        | NodePlan::DynamicRef { .. }
        | NodePlan::Unsupported => 0,
    };
    size_of::<Frame>().checked_add(owned)
}

enum PlanHandle<'a> {
    Borrowed(&'a StructuredPlan),
    Owned(Arc<StructuredPlan>),
}

impl Clone for PlanHandle<'_> {
    fn clone(&self) -> Self {
        match self {
            Self::Borrowed(plan) => Self::Borrowed(plan),
            Self::Owned(plan) => Self::Owned(plan.clone()),
        }
    }
}

impl PlanHandle<'_> {
    fn as_ref(&self) -> &StructuredPlan {
        match self {
            Self::Borrowed(plan) => plan,
            Self::Owned(plan) => plan,
        }
    }
}

struct CursorMachine<'a> {
    plan: PlanHandle<'a>,
    validator_ledger: Arc<ValidatorLedger>,
    slots: Vec<CursorSlot>,
    generations: Vec<u32>,
    epochs: Vec<u32>,
    structured: Vec<Option<StructuredState<'a>>>,
    // A wide aggregate preserves exact deltas even when the public usize total saturates.
    structured_retained: u128,
    free_slots: Vec<u32>,
    free_structured: Vec<u32>,
    ledger_active: bool,
}

#[derive(Clone, Copy)]
struct CursorPoolMark {
    slots_len: usize,
    structured_len: usize,
    free_slots_len: usize,
    free_structured_len: usize,
    reused_slot: Option<(usize, u32, u32)>,
}

impl<'a> CursorMachine<'a> {
    #[cfg(test)]
    fn new(plan: &'a StructuredPlan) -> Self {
        let ledger = Arc::new(ValidatorLedger {
            live: AtomicUsize::new(0),
            limit: plan.limits.max_active_validators,
        });
        Self::from_handle(PlanHandle::Borrowed(plan), ledger)
    }

    fn from_handle(plan: PlanHandle<'a>, validator_ledger: Arc<ValidatorLedger>) -> Self {
        Self {
            plan,
            validator_ledger,
            slots: Vec::new(),
            generations: Vec::new(),
            epochs: Vec::new(),
            structured: Vec::new(),
            structured_retained: 0,
            free_slots: Vec::new(),
            free_structured: Vec::new(),
            ledger_active: true,
        }
    }

    #[cfg(test)]
    fn start(&mut self, node: NodeId) -> Result<CursorId, StructuredRuntimeError> {
        let (arena, _) = ScopeArena::new(scope_arena_capacity(self.plan.as_ref())?)?;
        let resource = self
            .plan
            .as_ref()
            .node_resource(node)
            .ok_or_else(|| self.error())?;
        self.start_in_context(node, 0, arena.root(resource)?)
    }

    #[cfg(test)]
    fn start_in_context(
        &mut self,
        node: NodeId,
        depth_offset: usize,
        scope: ScopeLease,
    ) -> Result<CursorId, StructuredRuntimeError> {
        let (node, scope) = resolve_runtime_node(self.plan.as_ref(), node, scope)?;
        self.start_resolved_in_context(node, depth_offset, scope)
    }

    fn start_resolved_in_context(
        &mut self,
        node: NodeId,
        depth_offset: usize,
        scope: ScopeLease,
    ) -> Result<CursorId, StructuredRuntimeError> {
        self.reserve_slot()?;
        let slot = match self.plan.as_ref().node(node) {
            NodePlan::Regular(engine) => {
                self.validator_ledger.try_acquire()?;
                CursorSlot::Regular {
                    node,
                    state: engine.start(),
                }
            }
            NodePlan::Number { .. }
            | NodePlan::Object(_)
            | NodePlan::Array(_)
            | NodePlan::String(_)
            | NodePlan::StringEnum(_)
            | NodePlan::Combinator(_)
            | NodePlan::Negation { .. }
            | NodePlan::Unevaluated(_) => {
                if let Some(cursor) =
                    self.take_reusable_structured(Some(node), depth_offset, &scope)?
                {
                    CursorSlot::Structured(cursor)
                } else {
                    self.validator_ledger.try_acquire()?;
                    let mut reservation = ValidatorReservation::new(self.validator_ledger.clone());
                    let cursor = self.alloc_structured(node, false, depth_offset, scope)?;
                    reservation.disarm();
                    CursorSlot::Structured(cursor)
                }
            }
            NodePlan::Ref { .. } | NodePlan::DynamicRef { .. } | NodePlan::Unsupported => {
                return Err(self.error())
            }
        };
        let id = self.insert_slot(slot);
        Ok(id)
    }

    #[cfg(test)]
    fn start_growth_floor(&self, node: NodeId) -> Option<usize> {
        let (arena, _) = ScopeArena::new(scope_arena_capacity(self.plan.as_ref()).ok()?).ok()?;
        let resource = self.plan.as_ref().node_resource(node)?;
        let (node, _) =
            resolve_runtime_node(self.plan.as_ref(), node, arena.root(resource).ok()?).ok()?;
        self.start_growth_floor_resolved(node)
    }

    fn start_growth_floor_resolved(&self, node: NodeId) -> Option<usize> {
        fn vec_floor(len: usize, capacity: usize, item: usize) -> Option<usize> {
            len.checked_add(1)?
                .saturating_sub(capacity)
                .checked_mul(item)
        }

        let mut growth = 0usize;
        if self.free_slots.is_empty() {
            growth = growth.checked_add(vec_floor(
                self.slots.len(),
                self.slots.capacity(),
                size_of::<CursorSlot>(),
            )?)?;
            growth = growth.checked_add(vec_floor(
                self.generations.len(),
                self.generations.capacity(),
                size_of::<u32>(),
            )?)?;
            growth = growth.checked_add(vec_floor(
                self.epochs.len(),
                self.epochs.capacity(),
                size_of::<u32>(),
            )?)?;
            growth = growth.checked_add(
                self.slots
                    .len()
                    .checked_add(1)?
                    .saturating_sub(self.free_slots.capacity())
                    .checked_mul(size_of::<u32>())?,
            )?;
        }
        let structured = matches!(
            self.plan.as_ref().node(node),
            NodePlan::Object(_)
                | NodePlan::Array(_)
                | NodePlan::String(_)
                | NodePlan::StringEnum(_)
                | NodePlan::Combinator(_)
                | NodePlan::Negation { .. }
                | NodePlan::Unevaluated(_)
        );
        if structured {
            if self.free_structured.is_empty() {
                growth = growth.checked_add(vec_floor(
                    self.structured.len(),
                    self.structured.capacity(),
                    size_of::<Option<StructuredState<'a>>>(),
                )?)?;
                growth = growth.checked_add(
                    self.structured
                        .len()
                        .checked_add(1)?
                        .saturating_sub(self.free_structured.capacity())
                        .checked_mul(size_of::<u32>())?,
                )?;
            }
            growth = growth.checked_add(initial_frame_floor(self.plan.as_ref(), node)?)?;
        }
        Some(growth)
    }

    fn start_any_growth_floor(&self) -> Option<usize> {
        fn vec_floor(len: usize, capacity: usize, item: usize) -> Option<usize> {
            len.checked_add(1)?
                .saturating_sub(capacity)
                .checked_mul(item)
        }

        let mut growth = 0usize;
        if self.free_slots.is_empty() {
            growth = growth.checked_add(vec_floor(
                self.slots.len(),
                self.slots.capacity(),
                size_of::<CursorSlot>(),
            )?)?;
            growth = growth.checked_add(vec_floor(
                self.generations.len(),
                self.generations.capacity(),
                size_of::<u32>(),
            )?)?;
            growth = growth.checked_add(vec_floor(
                self.epochs.len(),
                self.epochs.capacity(),
                size_of::<u32>(),
            )?)?;
            growth = growth.checked_add(
                self.slots
                    .len()
                    .checked_add(1)?
                    .saturating_sub(self.free_slots.capacity())
                    .checked_mul(size_of::<u32>())?,
            )?;
        }
        if self.free_structured.is_empty() {
            growth = growth.checked_add(vec_floor(
                self.structured.len(),
                self.structured.capacity(),
                size_of::<Option<StructuredState<'a>>>(),
            )?)?;
            growth = growth.checked_add(
                self.structured
                    .len()
                    .checked_add(1)?
                    .saturating_sub(self.free_structured.capacity())
                    .checked_mul(size_of::<u32>())?,
            )?;
        }
        growth = growth.checked_add(size_of::<Frame>())?;
        Some(growth)
    }

    #[cfg(test)]
    fn start_any(&mut self) -> Result<CursorId, StructuredRuntimeError> {
        let (arena, _) = ScopeArena::new(scope_arena_capacity(self.plan.as_ref())?)?;
        let resource = self
            .plan
            .as_ref()
            .node_resource(self.plan.as_ref().root)
            .ok_or_else(|| self.error())?;
        self.start_any_in_context(0, arena.root(resource)?)
    }

    fn start_any_in_context(
        &mut self,
        depth_offset: usize,
        scope: ScopeLease,
    ) -> Result<CursorId, StructuredRuntimeError> {
        self.reserve_slot()?;
        let cursor =
            if let Some(cursor) = self.take_reusable_structured(None, depth_offset, &scope)? {
                cursor
            } else {
                self.validator_ledger.try_acquire()?;
                let mut reservation = ValidatorReservation::new(self.validator_ledger.clone());
                let cursor =
                    self.alloc_structured(self.plan.as_ref().root, true, depth_offset, scope)?;
                reservation.disarm();
                cursor
            };
        let id = self.insert_slot(CursorSlot::Structured(cursor));
        Ok(id)
    }

    fn checkpoint(&self, id: CursorId) -> Result<CursorMark, StructuredRuntimeError> {
        let index = self.slot_index(id)?;
        let state = match &self.slots[index] {
            CursorSlot::Regular { state, .. } => CursorStateMark::Regular(*state),
            CursorSlot::Structured(cursor) => CursorStateMark::Structured {
                cursor: *cursor,
                mark: self.structured_state(*cursor)?.checkpoint(),
            },
            CursorSlot::Vacant => return Err(self.error()),
        };
        Ok(CursorMark {
            generation: id.generation,
            epoch: self.epochs[index],
            state,
        })
    }

    fn try_push_byte(
        &mut self,
        id: CursorId,
        byte: u8,
    ) -> Result<CursorStep, StructuredRuntimeError> {
        match *self.slot(id)? {
            CursorSlot::Regular { node, state } => {
                let (next, accepting) = {
                    let NodePlan::Regular(engine) = self.plan.as_ref().node(node) else {
                        return Err(self.error());
                    };
                    let Some(next) = engine.consume_token(state, &[byte]) else {
                        return Ok(CursorStep::Dead);
                    };
                    if engine.is_dead(next) {
                        return Ok(CursorStep::Dead);
                    }
                    (next, engine.is_accepting(next))
                };
                *self.slot_mut(id)? = CursorSlot::Regular { node, state: next };
                Ok(if accepting {
                    CursorStep::Complete
                } else {
                    CursorStep::Alive
                })
            }
            CursorSlot::Structured(cursor) => {
                let pushed = self.structured_state_mut(cursor)?.try_push_byte(byte);
                if !pushed? {
                    return Ok(CursorStep::Dead);
                }
                Ok(if self.structured_state(cursor)?.is_accepting() {
                    CursorStep::Complete
                } else {
                    CursorStep::Alive
                })
            }
            CursorSlot::Vacant => Err(self.error()),
        }
    }

    fn rollback(&mut self, id: CursorId, mark: CursorMark) -> Result<(), StructuredRuntimeError> {
        let index = self.slot_index(id)?;
        if mark.generation != id.generation || mark.epoch != self.epochs[index] {
            return Err(self.error());
        }
        match (self.slots[index], mark.state) {
            (CursorSlot::Regular { node, .. }, CursorStateMark::Regular(state)) => {
                *self.slot_mut(id)? = CursorSlot::Regular { node, state };
                Ok(())
            }
            (
                CursorSlot::Structured(cursor),
                CursorStateMark::Structured {
                    cursor: marked,
                    mark,
                },
            ) if cursor == marked => {
                self.structured_state_mut(cursor)?.rollback(mark);
                Ok(())
            }
            _ => Err(self.error()),
        }
    }

    #[cfg(test)]
    fn commit(&mut self, id: CursorId) -> Result<(), StructuredRuntimeError> {
        let index = self.slot_index(id)?;
        self.epochs[index]
            .checked_add(1)
            .ok_or_else(|| self.error())?;
        match self.slots[index] {
            CursorSlot::Regular { .. } => {}
            CursorSlot::Structured(cursor) => {
                self.structured_state(cursor)?.validate_commit_tree()?;
                self.structured_state_mut(cursor)?.apply_commit_tree();
            }
            CursorSlot::Vacant => return Err(self.error()),
        }
        self.epochs[index] = self.epochs[index]
            .checked_add(1)
            .expect("commit validation checked cursor epoch");
        Ok(())
    }

    fn is_accepting(&self, id: CursorId) -> bool {
        match self.slot(id) {
            Ok(CursorSlot::Regular { node, state }) => match self.plan.as_ref().node(*node) {
                NodePlan::Regular(engine) => engine.is_accepting(*state),
                NodePlan::Object(_)
                | NodePlan::Number { .. }
                | NodePlan::Array(_)
                | NodePlan::String(_)
                | NodePlan::StringEnum(_)
                | NodePlan::Combinator(_)
                | NodePlan::Negation { .. }
                | NodePlan::Unevaluated(_)
                | NodePlan::Ref { .. }
                | NodePlan::DynamicRef { .. }
                | NodePlan::Unsupported => false,
            },
            Ok(CursorSlot::Structured(cursor)) => self
                .structured_state(*cursor)
                .is_ok_and(|state| state.is_accepting()),
            Ok(CursorSlot::Vacant) | Err(_) => false,
        }
    }

    fn accepted_annotation_summary(
        &self,
        id: CursorId,
    ) -> Result<Option<(bool, usize, bool)>, StructuredRuntimeError> {
        if !self.is_accepting(id) {
            return Err(self.error());
        }
        match self.slot(id)? {
            CursorSlot::Regular { .. } => Ok(None),
            CursorSlot::Structured(cursor) => self
                .structured_state(*cursor)?
                .accepted_annotation_summary(),
            CursorSlot::Vacant => Err(self.error()),
        }
    }

    fn accepted_annotation_word(
        &self,
        id: CursorId,
        word: usize,
    ) -> Result<u64, StructuredRuntimeError> {
        if !self.is_accepting(id) {
            return Err(self.error());
        }
        match self.slot(id)? {
            CursorSlot::Regular { .. } => Ok(0),
            CursorSlot::Structured(cursor) => self
                .structured_state(*cursor)?
                .accepted_annotation_word(word),
            CursorSlot::Vacant => Err(self.error()),
        }
    }

    /// Returns this cursor's accepted annotation when its shape matches `kind`.
    fn accepted_annotation_word_for_kind(
        &self,
        id: CursorId,
        kind: UnevaluatedKind,
        word: usize,
    ) -> Result<u64, StructuredRuntimeError> {
        if !self.is_accepting(id) {
            return Err(self.error());
        }
        match self.slot(id)? {
            CursorSlot::Regular { .. } => Ok(0),
            CursorSlot::Structured(cursor) => self
                .structured_state(*cursor)?
                .accepted_annotation_word_for_kind(kind, word),
            CursorSlot::Vacant => Err(self.error()),
        }
    }

    fn accepted_annotation_contains(
        &self,
        id: CursorId,
        kind: UnevaluatedKind,
        ordinal: u32,
    ) -> Result<bool, StructuredRuntimeError> {
        if !self.is_accepting(id) {
            return Err(self.error());
        }
        match self.slot(id)? {
            CursorSlot::Regular { .. } => Ok(false),
            CursorSlot::Structured(cursor) => self
                .structured_state(*cursor)?
                .accepted_annotation_contains(kind, ordinal),
            CursorSlot::Vacant => Err(self.error()),
        }
    }

    fn any_candidate_start(
        &self,
        id: CursorId,
        kind: UnevaluatedKind,
        byte: u8,
    ) -> Result<Option<u32>, StructuredRuntimeError> {
        if WS.contains(&byte) {
            return Ok(None);
        }
        let CursorSlot::Structured(cursor) = *self.slot(id)? else {
            return Err(self.error());
        };
        self.structured_state(cursor)?
            .any_candidate_start(kind, byte)
    }

    fn any_candidate_open(&self, id: CursorId) -> Result<bool, StructuredRuntimeError> {
        let CursorSlot::Structured(cursor) = *self.slot(id)? else {
            return Err(self.error());
        };
        let state = self.structured_state(cursor)?;
        Ok(!state.root_closed && state.frames.len() > 1)
    }

    fn structured_root_closed(&self, id: CursorId) -> Result<bool, StructuredRuntimeError> {
        let CursorSlot::Structured(cursor) = *self.slot(id)? else {
            return Ok(false);
        };
        Ok(self.structured_state(cursor)?.root_closed)
    }

    fn any_root_before_value(&self, id: CursorId) -> Result<bool, StructuredRuntimeError> {
        let CursorSlot::Structured(cursor) = *self.slot(id)? else {
            return Ok(false);
        };
        let state = self.structured_state(cursor)?;
        Ok(matches!(
            state.frames.as_slice(),
            [Frame::Any(AnyFrame::Scalar {
                phase: ScalarPhase::BeforeValue,
                ..
            })]
        ))
    }

    fn any_root_key_state(
        &self,
        id: CursorId,
    ) -> Result<Option<(&str, bool)>, StructuredRuntimeError> {
        let CursorSlot::Structured(cursor) = *self.slot(id)? else {
            return Ok(None);
        };
        Ok(self.structured_state(cursor)?.any_root_key_state())
    }

    fn any_root_key_viable(
        &self,
        id: CursorId,
        trie: &FiniteKeyTrie,
    ) -> Result<Option<bool>, StructuredRuntimeError> {
        let CursorSlot::Structured(cursor) = *self.slot(id)? else {
            return Ok(None);
        };
        Ok(self.structured_state(cursor)?.any_root_key_viable(trie))
    }

    fn next_array_item_annotation_possible(
        &self,
        id: CursorId,
    ) -> Result<Option<bool>, StructuredRuntimeError> {
        let CursorSlot::Structured(cursor) = *self.slot(id)? else {
            return Ok(None);
        };
        let state = self.structured_state(cursor)?;
        Ok(state.next_root_array_item_annotation_possible())
    }

    fn any_root_next_array_item(
        &self,
        id: CursorId,
    ) -> Result<Option<u32>, StructuredRuntimeError> {
        let CursorSlot::Structured(cursor) = *self.slot(id)? else {
            return Ok(None);
        };
        Ok(self.structured_state(cursor)?.any_root_next_array_item())
    }

    fn can_extend_value(&self, id: CursorId) -> bool {
        match self.slot(id) {
            Ok(CursorSlot::Regular { node, state }) => {
                let NodePlan::Regular(engine) = self.plan.as_ref().node(*node) else {
                    return false;
                };
                let live = engine.live_classes(*state);
                engine
                    .class_table
                    .members
                    .iter()
                    .enumerate()
                    .any(|(class, bytes)| {
                        live.contains(class) && bytes.iter().any(|byte| !WS.contains(byte))
                    })
            }
            Ok(CursorSlot::Structured(cursor)) => self
                .structured_state(*cursor)
                .is_ok_and(StructuredState::can_extend_value),
            Ok(CursorSlot::Vacant) | Err(_) => false,
        }
    }

    fn node(&self, id: CursorId) -> Option<NodeId> {
        match self.slot(id).ok()? {
            CursorSlot::Regular { node, .. } => Some(*node),
            CursorSlot::Structured(cursor) => self
                .structured_state(*cursor)
                .ok()
                .and_then(|state| state.initial_node),
            CursorSlot::Vacant => None,
        }
    }

    fn own_retained_bytes(&self) -> usize {
        [
            self.slots
                .capacity()
                .saturating_mul(size_of::<CursorSlot>()),
            self.generations.capacity().saturating_mul(size_of::<u32>()),
            self.epochs.capacity().saturating_mul(size_of::<u32>()),
            self.structured
                .capacity()
                .saturating_mul(size_of::<Option<StructuredState<'a>>>()),
            self.free_slots.capacity().saturating_mul(size_of::<u32>()),
            self.free_structured
                .capacity()
                .saturating_mul(size_of::<u32>()),
        ]
        .into_iter()
        .fold(0usize, usize::saturating_add)
    }

    fn refresh_structured_retained(&mut self) {
        self.structured_retained = self
            .structured
            .iter()
            .filter_map(Option::as_ref)
            .map(|state| state.retained_bytes() as u128)
            .sum();
    }

    fn retained_bytes(&self) -> usize {
        let total = self
            .own_retained_bytes()
            .saturating_add(usize::try_from(self.structured_retained).unwrap_or(usize::MAX));
        #[cfg(debug_assertions)]
        debug_assert_eq!(total, self.recomputed_retained_bytes());
        total
    }

    #[cfg(any(test, debug_assertions))]
    fn recomputed_retained_bytes(&self) -> usize {
        self.structured.iter().filter_map(Option::as_ref).fold(
            self.own_retained_bytes(),
            |total, state| {
                total
                    .saturating_add(state.session_memory.live())
                    .saturating_add(state.cursors.recomputed_retained_bytes())
            },
        )
    }

    fn set_cursor_budget(
        &mut self,
        id: CursorId,
        total_budget: usize,
    ) -> Result<(), StructuredRuntimeError> {
        let slot = *self.slot(id)?;
        let cursor = match slot {
            CursorSlot::Structured(cursor) => cursor,
            CursorSlot::Regular { .. } => return Ok(()),
            CursorSlot::Vacant => return Err(self.error()),
        };
        let current = self.structured_state(cursor)?.retained_bytes();
        let others = self
            .retained_bytes()
            .checked_sub(current)
            .ok_or_else(|| self.error())?;
        let budget = total_budget.checked_sub(others).ok_or_else(|| {
            StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, total_budget)
        })?;
        self.structured_state_mut(cursor)?.limits.max_session_bytes = budget;
        Ok(())
    }

    fn apply_commit_tree(&mut self) {
        for state in self.structured.iter_mut().filter_map(Option::as_mut) {
            state.apply_commit_tree();
        }
        self.refresh_structured_retained();
        for epoch in &mut self.epochs {
            *epoch += 1;
        }
    }

    fn discard_prepared_key_leases_tree(&mut self) {
        for state in self.structured.iter_mut().filter_map(Option::as_mut) {
            state.discard_prepared_key_leases_tree();
        }
        self.refresh_structured_retained();
    }

    #[cfg(test)]
    fn key_lease_tree_debug(&self) -> (usize, usize) {
        self.structured.iter().filter_map(Option::as_ref).fold(
            (0, 0),
            |(count, capacity), state| {
                let (nested_count, nested_capacity) = state.key_lease_tree_debug();
                (
                    count.saturating_add(nested_count),
                    capacity.saturating_add(nested_capacity),
                )
            },
        )
    }

    fn validate_commit_tree(&self) -> Result<(), StructuredRuntimeError> {
        for state in self.structured.iter().filter_map(Option::as_ref) {
            state.validate_commit_tree()?;
        }
        self.epochs
            .iter()
            .all(|epoch| epoch.checked_add(1).is_some())
            .then_some(())
            .ok_or_else(|| self.error())
    }

    fn validate_cursor(&self, id: CursorId) -> Result<(), StructuredRuntimeError> {
        self.slot(id).map(|_| ())
    }

    fn count_cursor_release(
        &self,
        id: CursorId,
        counts: &mut ReleaseCounts,
    ) -> Result<(), StructuredRuntimeError> {
        let index = self.slot_index(id)?;
        match self.slots[index] {
            CursorSlot::Regular { .. } => counts.free_slots += 1,
            CursorSlot::Structured(_) => {
                counts.free_slots += 1;
                counts.free_structured += 1;
            }
            CursorSlot::Vacant => return Err(self.error()),
        }
        Ok(())
    }

    fn release(&mut self, id: CursorId) -> Result<(), StructuredRuntimeError> {
        let index = self.slot_index(id)?;
        let cursor = match self.slots[index] {
            CursorSlot::Structured(cursor) => Some(cursor),
            CursorSlot::Regular { .. } => None,
            CursorSlot::Vacant => return Err(self.error()),
        };
        let structured_index = cursor
            .map(|cursor| self.structured_index(cursor))
            .transpose()?;
        if let (Some(cursor), Some(structured_index)) = (cursor, structured_index) {
            let reusable = self.structured[structured_index]
                .as_mut()
                .is_some_and(StructuredState::reset_for_reuse);
            self.refresh_structured_retained();
            if reusable {
                let released = self.structured[structured_index]
                    .as_ref()
                    .ok_or_else(|| self.error())?
                    .cached_validator_count()?;
                if !self.validator_ledger.try_release_many(released) {
                    return Err(self.error());
                }
                let error = self.error();
                let Some(state) = self.structured[structured_index].as_mut() else {
                    return Err(error);
                };
                state.cursors.set_ledger_active(false)?;
            } else {
                let nested = self.structured[structured_index].take();
                self.refresh_structured_retained();
                drop(nested);
                if !self.validator_ledger.try_release() {
                    return Err(self.error());
                }
            }
            self.free_structured.push(cursor.0);
        } else if !self.validator_ledger.try_release() {
            return Err(self.error());
        }
        self.slots[index] = CursorSlot::Vacant;
        self.free_slots.push(id.index);
        Ok(())
    }

    fn release_discard(&mut self, id: CursorId) -> Result<(), StructuredRuntimeError> {
        let index = self.slot_index(id)?;
        let cursor = match self.slots[index] {
            CursorSlot::Structured(cursor) => Some(cursor),
            CursorSlot::Regular { .. } => None,
            CursorSlot::Vacant => return Err(self.error()),
        };
        if let Some(cursor) = cursor {
            let structured_index = self.structured_index(cursor)?;
            let nested = self.structured[structured_index].take();
            self.refresh_structured_retained();
            drop(nested);
            self.free_structured.push(cursor.0);
        }
        if !self.validator_ledger.try_release() {
            return Err(self.error());
        }
        self.slots[index] = CursorSlot::Vacant;
        self.free_slots.push(id.index);
        Ok(())
    }

    fn pool_mark(&self) -> Result<CursorPoolMark, StructuredRuntimeError> {
        let reused_slot = self
            .free_slots
            .last()
            .map(|raw| {
                let index = usize::try_from(*raw).map_err(|_| self.error())?;
                Ok((index, self.generations[index], self.epochs[index]))
            })
            .transpose()?;
        Ok(CursorPoolMark {
            slots_len: self.slots.len(),
            structured_len: self.structured.len(),
            free_slots_len: self.free_slots.len(),
            free_structured_len: self.free_structured.len(),
            reused_slot,
        })
    }

    fn rollback_unpublished(
        &mut self,
        id: CursorId,
        mark: CursorPoolMark,
    ) -> Result<(), StructuredRuntimeError> {
        self.release_discard(id)?;
        self.slots.truncate(mark.slots_len);
        self.generations.truncate(mark.slots_len);
        self.epochs.truncate(mark.slots_len);
        self.structured.truncate(mark.structured_len);
        self.refresh_structured_retained();
        self.free_slots.truncate(mark.free_slots_len);
        self.free_structured.truncate(mark.free_structured_len);
        if let Some((index, generation, epoch)) = mark.reused_slot {
            self.generations[index] = generation;
            self.epochs[index] = epoch;
        }
        Ok(())
    }

    fn reserve_slot(&mut self) -> Result<(), StructuredRuntimeError> {
        if self.free_slots.is_empty() {
            u32::try_from(self.slots.len()).map_err(|_| self.error())?;
            self.slots.try_reserve_exact(1).map_err(|_| self.error())?;
            self.generations
                .try_reserve_exact(1)
                .map_err(|_| self.error())?;
            self.epochs.try_reserve_exact(1).map_err(|_| self.error())?;
            let free_needed = self
                .slots
                .len()
                .checked_add(1)
                .and_then(|target| target.checked_sub(self.free_slots.len()))
                .ok_or_else(|| self.error())?;
            self.free_slots
                .try_reserve_exact(free_needed)
                .map_err(|_| self.error())?;
        } else {
            let raw = *self.free_slots.last().expect("free slot exists");
            let index = usize::try_from(raw).map_err(|_| self.error())?;
            self.generations[index]
                .checked_add(1)
                .ok_or_else(|| self.error())?;
        }
        Ok(())
    }

    fn insert_slot(&mut self, slot: CursorSlot) -> CursorId {
        if let Some(raw) = self.free_slots.pop() {
            let index = usize::try_from(raw).expect("cursor index came from usize");
            self.generations[index] = self.generations[index]
                .checked_add(1)
                .expect("cursor generation checked before insertion");
            self.epochs[index] = 0;
            self.slots[index] = slot;
            return CursorId {
                index: raw,
                generation: self.generations[index],
            };
        }
        let raw = u32::try_from(self.slots.len()).expect("cursor count checked before insertion");
        self.slots.push(slot);
        self.generations.push(0);
        self.epochs.push(0);
        CursorId {
            index: raw,
            generation: 0,
        }
    }

    fn validator_count(&self) -> Result<usize, StructuredRuntimeError> {
        let mut count = 0usize;
        for slot in &self.slots {
            match slot {
                CursorSlot::Regular { .. } => {
                    count = count.checked_add(1).ok_or_else(|| self.error())?;
                }
                CursorSlot::Structured(cursor) => {
                    let nested = self.structured_state(*cursor)?;
                    let nested_count = nested.cursors.validator_count()?;
                    count = count
                        .checked_add(1)
                        .and_then(|value| value.checked_add(nested_count))
                        .ok_or_else(|| self.error())?;
                }
                CursorSlot::Vacant => {}
            }
        }
        Ok(count)
    }

    fn active_validator_count(&self) -> Result<usize, StructuredRuntimeError> {
        if self.ledger_active {
            self.validator_count()
        } else {
            Ok(0)
        }
    }

    fn set_ledger_active(&mut self, active: bool) -> Result<(), StructuredRuntimeError> {
        self.ledger_active = active;
        for index in 0..self.slots.len() {
            let CursorSlot::Structured(cursor) = self.slots[index] else {
                continue;
            };
            self.structured_state_mut(cursor)?
                .cursors
                .set_ledger_active(active)?;
        }
        Ok(())
    }

    fn take_reusable_structured(
        &mut self,
        node: Option<NodeId>,
        depth_offset: usize,
        scope: &ScopeLease,
    ) -> Result<Option<StructuredCursorId>, StructuredRuntimeError> {
        let Some(position) = self.free_structured.iter().rposition(|raw| {
            usize::try_from(*raw)
                .ok()
                .and_then(|index| self.structured.get(index))
                .and_then(Option::as_ref)
                .is_some_and(|state| state.can_reuse(node, depth_offset, scope))
        }) else {
            return Ok(None);
        };
        let raw = self.free_structured[position];
        let index = usize::try_from(raw).map_err(|_| self.error())?;
        let nested = self.structured[index]
            .as_ref()
            .ok_or_else(|| self.error())?
            .cursors
            .active_validator_count()?;
        debug_assert_eq!(nested, 0, "cached validators are ledger-inactive");
        let active = self.structured[index]
            .as_ref()
            .ok_or_else(|| self.error())?
            .cached_validator_count()?;
        self.validator_ledger.try_acquire_many(active)?;
        let missing = self.error();
        let Some(state) = self.structured[index].as_mut() else {
            return Err(missing);
        };
        if let Err(error) = state.cursors.set_ledger_active(true) {
            let released = self.validator_ledger.try_release_many(active);
            debug_assert!(released, "reused validator reservation is balanced");
            return Err(error);
        }
        self.free_structured.swap_remove(position);
        Ok(Some(StructuredCursorId(raw)))
    }

    fn alloc_structured(
        &mut self,
        node: NodeId,
        any: bool,
        depth_offset: usize,
        scope: ScopeLease,
    ) -> Result<StructuredCursorId, StructuredRuntimeError> {
        let state = if any {
            StructuredState::new_any(
                self.plan.clone(),
                depth_offset,
                self.validator_ledger.clone(),
                scope,
            )?
        } else {
            StructuredState::new_at_with_scope(
                self.plan.clone(),
                node,
                depth_offset,
                self.validator_ledger.clone(),
                scope,
            )?
        };
        if let Some(raw) = self.free_structured.pop() {
            let index = usize::try_from(raw).map_err(|_| self.error())?;
            self.structured[index] = Some(state);
            self.refresh_structured_retained();
            return Ok(StructuredCursorId(raw));
        }
        self.structured
            .try_reserve_exact(1)
            .map_err(|_| self.error())?;
        let free_needed = self
            .structured
            .len()
            .checked_add(1)
            .and_then(|target| target.checked_sub(self.free_structured.len()))
            .ok_or_else(|| self.error())?;
        self.free_structured
            .try_reserve_exact(free_needed)
            .map_err(|_| self.error())?;
        let raw = u32::try_from(self.structured.len()).map_err(|_| self.error())?;
        self.structured.push(Some(state));
        self.refresh_structured_retained();
        Ok(StructuredCursorId(raw))
    }

    fn slot_index(&self, id: CursorId) -> Result<usize, StructuredRuntimeError> {
        let index = usize::try_from(id.index).map_err(|_| self.error())?;
        (index < self.slots.len() && self.generations[index] == id.generation)
            .then_some(index)
            .ok_or_else(|| self.error())
    }

    fn structured_index(&self, id: StructuredCursorId) -> Result<usize, StructuredRuntimeError> {
        let index = usize::try_from(id.0).map_err(|_| self.error())?;
        (index < self.structured.len())
            .then_some(index)
            .ok_or_else(|| self.error())
    }

    fn slot(&self, id: CursorId) -> Result<&CursorSlot, StructuredRuntimeError> {
        Ok(&self.slots[self.slot_index(id)?])
    }

    fn slot_mut(&mut self, id: CursorId) -> Result<&mut CursorSlot, StructuredRuntimeError> {
        let index = self.slot_index(id)?;
        Ok(&mut self.slots[index])
    }

    fn structured_state(
        &self,
        id: StructuredCursorId,
    ) -> Result<&StructuredState<'a>, StructuredRuntimeError> {
        self.structured[self.structured_index(id)?]
            .as_ref()
            .ok_or_else(|| self.error())
    }

    fn structured_state_mut(
        &mut self,
        id: StructuredCursorId,
    ) -> Result<RetainedStateMut<'_, 'a>, StructuredRuntimeError> {
        let index = self.structured_index(id)?;
        let state = self.structured[index].as_mut().ok_or_else(|| {
            StructuredRuntimeError::new(LimitKind::ActiveValidators, usize::MAX, 0)
        })?;
        let before = state.retained_bytes() as u128;
        Ok(RetainedStateMut {
            state,
            total: &mut self.structured_retained,
            before,
        })
    }

    fn error(&self) -> StructuredRuntimeError {
        StructuredRuntimeError::new(
            LimitKind::ActiveValidators,
            usize::MAX,
            self.plan.as_ref().limits.max_active_validators,
        )
    }
}

struct RetainedStateMut<'s, 'a> {
    state: &'s mut StructuredState<'a>,
    total: &'s mut u128,
    before: u128,
}

impl<'a> std::ops::Deref for RetainedStateMut<'_, 'a> {
    type Target = StructuredState<'a>;

    fn deref(&self) -> &Self::Target {
        self.state
    }
}

impl std::ops::DerefMut for RetainedStateMut<'_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.state
    }
}

impl Drop for RetainedStateMut<'_, '_> {
    fn drop(&mut self) {
        let after = self.state.retained_bytes() as u128;
        debug_assert!(*self.total >= self.before);
        *self.total = self
            .total
            .checked_sub(self.before)
            .and_then(|total| total.checked_add(after))
            .unwrap_or(u128::MAX);
    }
}

impl Drop for CursorMachine<'_> {
    fn drop(&mut self) {
        if !self.ledger_active {
            return;
        }
        let occupied = self
            .slots
            .iter()
            .filter(|slot| !matches!(slot, CursorSlot::Vacant))
            .count();
        for _ in 0..occupied {
            let released = self.validator_ledger.try_release();
            debug_assert!(released, "cursor validator ledger underflow");
        }
    }
}

pub(crate) struct StructuredState<'a> {
    state_identity: usize,
    plan: PlanHandle<'a>,
    frames: Vec<Frame>,
    frame_scopes: Vec<ScopeLease>,
    undo: Vec<Undo>,
    accepting: bool,
    dead: bool,
    root_closed: bool,
    root_annotations: RootAnnotations,
    document_bytes: usize,
    session_memory: SessionMemory,
    key_arena: KeyArena,
    key_lease: KeyCapacityLease,
    /// Per-session-instance randomized `SipHash` builder: an attacker choosing dynamic property
    /// names cannot predict or force hash collisions across different sessions.
    key_hasher: RandomState,
    /// Backing storage for every `AnyFrame::Object`; a closed slot is recycled via `any_object_free`.
    any_object_arena: Vec<AnyObjectState>,
    any_object_free: Vec<u32>,
    object_frame_free: Vec<ObjectFrame>,
    combinator_frame_free: Vec<CombinatorFrame>,
    cursors: CursorMachine<'a>,
    _validator_ledger: Arc<ValidatorLedger>,
    validator_ledger_charge: usize,
    scope_arena_charge: usize,
    limits: StructuredLimits,
    depth_offset: usize,
    byte_observed: bool,
    initial_node: Option<NodeId>,
    initial_checkpoint: Checkpoint,
    resettable_to_initial: bool,
}

/// Resolved object-key metadata and schemas required for its value.
struct Resolution {
    known_id: Option<PropertyId>,
    obligations: ValueObligations,
    annotates: bool,
}

enum ObligationStep {
    Rejected,
    Advanced,
    Popped,
}

enum CombinatorStep {
    Rejected,
    Advanced,
    Popped,
}

enum NegationStep {
    Rejected,
    Advanced,
    Popped,
}

enum UnevaluatedStep {
    Rejected,
    Advanced,
    Popped { replay: bool },
}

enum AnyStep {
    Rejected,
    Advanced,
    Popped,
}

impl<'a> StructuredState<'a> {
    fn can_reuse(&self, node: Option<NodeId>, depth_offset: usize, scope: &ScopeLease) -> bool {
        self.resettable_to_initial
            && self.initial_node == node
            && self.depth_offset == depth_offset
            && self
                .frame_scopes
                .first()
                .is_some_and(|cached| scopes_are_equivalent(cached, scope))
    }

    fn cached_validator_count(&self) -> Result<usize, StructuredRuntimeError> {
        self.cursors
            .validator_count()?
            .checked_add(1)
            .ok_or_else(|| self.cursors.error())
    }

    fn reset_for_reuse(&mut self) -> bool {
        if !self.resettable_to_initial {
            return false;
        }
        self.rollback(self.initial_checkpoint);
        self.discard_prepared_key_lease();
        self.undo.len() == self.initial_checkpoint.undo_len
            && self.document_bytes == self.initial_checkpoint.document_bytes
            && !self.byte_observed
    }

    fn accepted_annotation_summary(
        &self,
    ) -> Result<Option<(bool, usize, bool)>, StructuredRuntimeError> {
        if !self.accepting {
            return Err(self.cursors.error());
        }
        if let Some(summary) = self.root_annotations.shape_and_words() {
            return Ok(Some(summary));
        }
        let [Frame::Combinator(frame)] = self.frames.as_slice() else {
            return Ok(None);
        };
        let mut summary: Option<(bool, usize, bool)> = None;
        for (index, &cursor) in frame.cursors.iter().enumerate() {
            if !bit_is_set(&frame.active_words, index) || !self.cursors.is_accepting(cursor) {
                continue;
            }
            let Some(branch) = self.cursors.accepted_annotation_summary(cursor)? else {
                continue;
            };
            match summary {
                None => summary = Some(branch),
                Some((shape, words, all)) if shape == branch.0 => {
                    summary = Some((shape, words.max(branch.1), all || branch.2));
                }
                Some(_) => return Ok(None),
            }
        }
        Ok(summary)
    }

    fn accepted_annotation_word(&self, word: usize) -> Result<u64, StructuredRuntimeError> {
        if !self.accepting {
            return Err(self.cursors.error());
        }
        if self.root_annotations.shape_and_words().is_some() {
            return Ok(self.root_annotations.word(word));
        }
        let [Frame::Combinator(frame)] = self.frames.as_slice() else {
            return Ok(0);
        };
        let mut merged = 0u64;
        for (index, &cursor) in frame.cursors.iter().enumerate() {
            if bit_is_set(&frame.active_words, index) && self.cursors.is_accepting(cursor) {
                merged |= self.cursors.accepted_annotation_word(cursor, word)?;
            }
        }
        Ok(merged)
    }

    /// Returns the annotation word only when its shape matches `kind`.
    fn accepted_annotation_word_for_kind(
        &self,
        kind: UnevaluatedKind,
        word: usize,
    ) -> Result<u64, StructuredRuntimeError> {
        let Some((is_object, _, _)) = self.accepted_annotation_summary()? else {
            return Ok(0);
        };
        let matches = match kind {
            UnevaluatedKind::Properties => is_object,
            UnevaluatedKind::Items => !is_object,
        };
        if !matches {
            return Ok(0);
        }
        self.accepted_annotation_word(word)
    }

    fn accepted_annotation_contains(
        &self,
        kind: UnevaluatedKind,
        ordinal: u32,
    ) -> Result<bool, StructuredRuntimeError> {
        if !self.accepting {
            return Err(self.cursors.error());
        }
        match (kind, &self.root_annotations) {
            (UnevaluatedKind::Properties, RootAnnotations::Object(set))
            | (UnevaluatedKind::Items, RootAnnotations::Array(set)) => {
                return Ok(set.contains(ordinal));
            }
            _ => {}
        }
        let [Frame::Combinator(frame)] = self.frames.as_slice() else {
            return Ok(false);
        };
        for (index, &cursor) in frame.cursors.iter().enumerate() {
            if bit_is_set(&frame.active_words, index)
                && self.cursors.is_accepting(cursor)
                && self
                    .cursors
                    .accepted_annotation_contains(cursor, kind, ordinal)?
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn any_candidate_start(
        &self,
        kind: UnevaluatedKind,
        byte: u8,
    ) -> Result<Option<u32>, StructuredRuntimeError> {
        let Some(root) = self.frames.first() else {
            return Ok(None);
        };
        match (kind, root) {
            (UnevaluatedKind::Items, Frame::Any(AnyFrame::Array { phase, items }))
                if matches!(
                    phase,
                    AnyArrayPhase::BeforeFirstOrClose | AnyArrayPhase::AfterComma
                ) && byte != b']' =>
            {
                Ok(Some(*items))
            }
            (UnevaluatedKind::Properties, Frame::Any(AnyFrame::Object(id))) => {
                let object = self
                    .any_object_arena
                    .get(usize::try_from(id.0).map_err(|_| self.cursors.error())?)
                    .ok_or_else(|| self.cursors.error())?;
                if object.phase == ObjectPhase::AfterColonBeforeValue {
                    object
                        .properties
                        .checked_sub(1)
                        .map(Some)
                        .ok_or_else(|| self.cursors.error())
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    fn any_root_next_array_item(&self) -> Option<u32> {
        let [Frame::Any(AnyFrame::Array { phase, items })] = self.frames.as_slice() else {
            return None;
        };
        (*phase == AnyArrayPhase::AfterValue).then_some(*items)
    }

    fn any_root_key_state(&self) -> Option<(&str, bool)> {
        let [Frame::Any(AnyFrame::Object(id))] = self.frames.as_slice() else {
            return None;
        };
        let object = self.any_object_arena.get(id.0 as usize)?;
        if !object.key_decoder.at_boundary() {
            return None;
        }
        let complete = match object.phase {
            ObjectPhase::InKey => false,
            ObjectPhase::AfterKeyBeforeColon => true,
            _ => return None,
        };
        Some((self.candidate_key(0), complete))
    }

    fn any_root_key_viable(&self, trie: &FiniteKeyTrie) -> Option<bool> {
        let [Frame::Any(AnyFrame::Object(id))] = self.frames.as_slice() else {
            return None;
        };
        let object = self.any_object_arena.get(id.0 as usize)?;
        // After a comma, prove an unseen key can still begin at the empty prefix.
        let (key, complete) = if object.phase == ObjectPhase::AfterCommaBeforeKey {
            ("", false)
        } else if object.phase == ObjectPhase::InKey && !object.key_decoder.at_boundary() {
            let key = self.candidate_key(0);
            return object
                .key_decoder
                .any_pending_scalar_satisfies_with_supplementary(
                    4096,
                    |scalar| {
                        let mut encoded = [0u8; 4];
                        let suffix: &str = scalar.encode_utf8(&mut encoded);
                        let seen_pattern_completions = object
                            .seen_dynamic
                            .values()
                            .flat_map(|slot| match slot {
                                DynamicSlot::One(id) => std::slice::from_ref(id),
                                DynamicSlot::Many(ids) => ids.as_slice(),
                            })
                            .filter(|&&id| {
                                self.key_arena.get(id).is_some_and(|name| {
                                    name.strip_prefix(key)
                                        .is_some_and(|rest| rest.starts_with(suffix))
                                        && trie.pattern_accepts(name)
                                })
                            })
                            .count();
                        trie.permits_after_scalar(key, scalar, seen_pattern_completions, |name| {
                            let hash = self.key_hasher.hash_one(name.as_bytes());
                            object.seen_dynamic.get(&hash).is_some_and(|slot| {
                                slot.contains(|id| self.key_arena.get(id) == Some(name))
                            })
                        })
                    },
                    || {
                        let seen_pattern_completions = object
                            .seen_dynamic
                            .values()
                            .flat_map(|slot| match slot {
                                DynamicSlot::One(id) => std::slice::from_ref(id),
                                DynamicSlot::Many(ids) => ids.as_slice(),
                            })
                            .filter(|&&id| {
                                self.key_arena.get(id).is_some_and(|name| {
                                    name.starts_with(key) && trie.pattern_accepts(name)
                                })
                            })
                            .count();
                        trie.permits_after_any_scalar(key, seen_pattern_completions, |name| {
                            let hash = self.key_hasher.hash_one(name.as_bytes());
                            object.seen_dynamic.get(&hash).is_some_and(|slot| {
                                slot.contains(|id| self.key_arena.get(id) == Some(name))
                            })
                        })
                    },
                    |start, end| {
                        trie.permits_after_supplementary_range(key, start, end, |name| {
                            let hash = self.key_hasher.hash_one(name.as_bytes());
                            object.seen_dynamic.get(&hash).is_some_and(|slot| {
                                slot.contains(|id| self.key_arena.get(id) == Some(name))
                            })
                        })
                    },
                );
        } else {
            self.any_root_key_state()?
        };
        let seen_pattern_completions = object
            .seen_dynamic
            .values()
            .flat_map(|slot| match slot {
                DynamicSlot::One(id) => std::slice::from_ref(id),
                DynamicSlot::Many(ids) => ids.as_slice(),
            })
            .filter(|&&id| {
                self.key_arena
                    .get(id)
                    .is_some_and(|name| name.starts_with(key) && trie.pattern_accepts(name))
            })
            .count();
        let viable = trie.permits(key, complete, seen_pattern_completions, |name| {
            let hash = self.key_hasher.hash_one(name.as_bytes());
            object
                .seen_dynamic
                .get(&hash)
                .is_some_and(|slot| slot.contains(|id| self.key_arena.get(id) == Some(name)))
        });
        Some(viable)
    }

    fn mark_all_evaluated(
        &mut self,
        target: AnnotationTarget,
    ) -> Result<(), StructuredRuntimeError> {
        if matches!(self.evaluated_ref(target), EvaluatedSet::All) {
            return Ok(());
        }
        self.reserve_undo(1)?;
        let old = std::mem::replace(self.evaluated_mut(target), EvaluatedSet::All);
        self.push_undo(Undo::ReplaceEvaluatedSet {
            target,
            old,
            new_charge: 0,
        })
    }

    fn merge_cursor_annotations(
        &mut self,
        target: AnnotationTarget,
        cursor: CursorId,
    ) -> Result<(), StructuredRuntimeError> {
        let Some((is_object, words, all)) = self.cursors.accepted_annotation_summary(cursor)?
        else {
            return Ok(());
        };
        if is_object
            != matches!(
                target,
                AnnotationTarget::Object(_) | AnnotationTarget::RootObject
            )
        {
            return Ok(());
        }
        if all {
            return self.mark_all_evaluated(target);
        }
        for word_index in 0..words {
            let mut word = self.cursors.accepted_annotation_word(cursor, word_index)?;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                let ordinal = word_index
                    .checked_mul(64)
                    .and_then(|base| base.checked_add(bit))
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| self.cursors.error())?;
                self.insert_evaluated(target, ordinal)?;
                word &= word - 1;
            }
        }
        Ok(())
    }

    fn publish_root_annotations(
        &mut self,
        target: AnnotationTarget,
    ) -> Result<(), StructuredRuntimeError> {
        self.reserve_undo(1)?;
        let annotations = match target {
            AnnotationTarget::Object(_) => {
                RootAnnotations::Object(std::mem::take(self.evaluated_mut(target)))
            }
            AnnotationTarget::Array(_) => {
                RootAnnotations::Array(std::mem::take(self.evaluated_mut(target)))
            }
            AnnotationTarget::RootObject
            | AnnotationTarget::RootArray
            | AnnotationTarget::UnevaluatedCandidates(_) => {
                unreachable!("frame publication requires a frame target")
            }
        };
        let old = std::mem::replace(&mut self.root_annotations, annotations);
        self.push_undo(Undo::SetRootAnnotations {
            old,
            restore_target: target,
        })
    }

    fn evaluated_ref(&self, target: AnnotationTarget) -> &EvaluatedSet {
        match target {
            AnnotationTarget::Object(depth) => &self.object_ref(depth).evaluated,
            AnnotationTarget::Array(depth) => &self.array_ref(depth).evaluated,
            AnnotationTarget::RootObject => match &self.root_annotations {
                RootAnnotations::Object(set) => set,
                _ => unreachable!("root object annotation target has object storage"),
            },
            AnnotationTarget::RootArray => match &self.root_annotations {
                RootAnnotations::Array(set) => set,
                _ => unreachable!("root array annotation target has array storage"),
            },
            AnnotationTarget::UnevaluatedCandidates(depth) => match &self.frames[depth] {
                Frame::Unevaluated(frame) => &frame.candidates,
                _ => unreachable!("candidate annotation target has unevaluated storage"),
            },
        }
    }

    fn evaluated_mut(&mut self, target: AnnotationTarget) -> &mut EvaluatedSet {
        match target {
            AnnotationTarget::Object(depth) => &mut self.object_mut(depth).evaluated,
            AnnotationTarget::Array(depth) => &mut self.array_mut(depth).evaluated,
            AnnotationTarget::RootObject => match &mut self.root_annotations {
                RootAnnotations::Object(set) => set,
                _ => unreachable!("root object annotation target has object storage"),
            },
            AnnotationTarget::RootArray => match &mut self.root_annotations {
                RootAnnotations::Array(set) => set,
                _ => unreachable!("root array annotation target has array storage"),
            },
            AnnotationTarget::UnevaluatedCandidates(depth) => match &mut self.frames[depth] {
                Frame::Unevaluated(frame) => &mut frame.candidates,
                _ => unreachable!("candidate annotation target has unevaluated storage"),
            },
        }
    }

    fn ensure_root_annotation_target(
        &mut self,
        is_object: bool,
    ) -> Result<AnnotationTarget, StructuredRuntimeError> {
        let target = if is_object {
            AnnotationTarget::RootObject
        } else {
            AnnotationTarget::RootArray
        };
        if matches!(
            (&self.root_annotations, is_object),
            (RootAnnotations::Object(_), true) | (RootAnnotations::Array(_), false)
        ) {
            return Ok(target);
        }
        if !matches!(self.root_annotations, RootAnnotations::None) {
            return Err(self.cursors.error());
        }
        self.reserve_undo(1)?;
        let replacement = if is_object {
            RootAnnotations::Object(EvaluatedSet::None)
        } else {
            RootAnnotations::Array(EvaluatedSet::None)
        };
        let old = std::mem::replace(&mut self.root_annotations, replacement);
        self.push_undo(Undo::ReplaceRootAnnotations { old })?;
        Ok(target)
    }

    fn insert_evaluated(
        &mut self,
        target: AnnotationTarget,
        ordinal: u32,
    ) -> Result<(), StructuredRuntimeError> {
        if self.evaluated_ref(target).contains(ordinal) {
            return Ok(());
        }
        let index = usize::try_from(ordinal).map_err(|_| {
            StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                usize::MAX,
                self.limits.max_session_bytes,
            )
        })?;
        let word_index = index / 64;
        let mask = 1u64 << (index % 64);
        match self.evaluated_ref(target) {
            EvaluatedSet::Inline(old) if word_index == 0 => {
                let old = *old;
                self.reserve_undo(1)?;
                *self.evaluated_mut(target) = EvaluatedSet::Inline(old | mask);
                self.push_undo(Undo::EvaluatedWord {
                    target,
                    word: 0,
                    old,
                })
            }
            EvaluatedSet::Heap(words) if word_index < words.len() => {
                let old = words[word_index];
                self.reserve_undo(1)?;
                let EvaluatedSet::Heap(words) = self.evaluated_mut(target) else {
                    unreachable!("evaluated representation changed")
                };
                words[word_index] = old | mask;
                self.push_undo(Undo::EvaluatedWord {
                    target,
                    word: word_index,
                    old,
                })
            }
            EvaluatedSet::All => Ok(()),
            EvaluatedSet::None if word_index == 0 => {
                self.reserve_undo(1)?;
                let old = std::mem::replace(self.evaluated_mut(target), EvaluatedSet::Inline(mask));
                self.push_undo(Undo::ReplaceEvaluatedSet {
                    target,
                    old,
                    new_charge: 0,
                })
            }
            EvaluatedSet::None | EvaluatedSet::Inline(_) | EvaluatedSet::Heap(_) => {
                let required_words = word_index.checked_add(1).ok_or_else(|| {
                    StructuredRuntimeError::new(
                        LimitKind::SessionBytes,
                        usize::MAX,
                        self.limits.max_session_bytes,
                    )
                })?;
                let words = required_words.checked_next_power_of_two().ok_or_else(|| {
                    StructuredRuntimeError::new(
                        LimitKind::SessionBytes,
                        usize::MAX,
                        self.limits.max_session_bytes,
                    )
                })?;
                let bytes = words.checked_mul(size_of::<u64>()).ok_or_else(|| {
                    StructuredRuntimeError::new(
                        LimitKind::SessionBytes,
                        usize::MAX,
                        self.limits.max_session_bytes,
                    )
                })?;
                if !self.session_would_fit(bytes) {
                    return Err(StructuredRuntimeError::new(
                        LimitKind::SessionBytes,
                        self.session_memory.live().saturating_add(bytes),
                        self.limits.max_session_bytes,
                    ));
                }
                self.reserve_undo(1)?;
                let mut storage = Vec::new();
                storage.try_reserve_exact(words).map_err(|_| {
                    StructuredRuntimeError::new(
                        LimitKind::SessionBytes,
                        usize::MAX,
                        self.limits.max_session_bytes,
                    )
                })?;
                storage.resize(words, 0u64);
                match self.evaluated_ref(target) {
                    EvaluatedSet::Inline(word) => storage[0] = *word,
                    EvaluatedSet::Heap(old) => storage[..old.len()].copy_from_slice(old),
                    EvaluatedSet::None => {}
                    EvaluatedSet::All => unreachable!("handled above"),
                }
                storage[word_index] |= mask;
                self.charge_session_bytes_after_precheck(bytes)?;
                let old = std::mem::replace(
                    self.evaluated_mut(target),
                    EvaluatedSet::Heap(storage.into_boxed_slice()),
                );
                self.push_undo(Undo::ReplaceEvaluatedSet {
                    target,
                    old,
                    new_charge: bytes,
                })
            }
        }
    }

    fn release_owned_cursor(&mut self, cursor: CursorId) {
        self.cursors
            .release(cursor)
            .expect("owned cursor was validated before installation");
    }

    fn discard_item(&mut self, item: ItemTracking) {
        if let ContainsCandidate::Cursor(cursor) = item.contains {
            self.release_owned_cursor(cursor);
        }
    }

    fn discard_new_frame(&mut self, frame: Frame) {
        match frame {
            Frame::Obligation(obligation) => {
                for cursor in obligation.cursors {
                    self.release_owned_cursor(cursor);
                }
                self.session_memory.release(obligation.allocation_charge);
            }
            Frame::Combinator(mut combinator) => {
                for (index, cursor) in combinator.cursors.drain(..).enumerate() {
                    if bit_is_set(&combinator.owned_words, index) {
                        self.release_owned_cursor(cursor);
                    }
                }
                combinator.active_words.clear();
                combinator.owned_words.clear();
                combinator.marks.clear();
                if self.combinator_frame_free.len() < self.combinator_frame_free.capacity() {
                    self.combinator_frame_free.push(combinator);
                } else {
                    self.session_memory.release(combinator.allocation_charge);
                }
            }
            Frame::Negation(frame) => {
                self.release_owned_cursor(frame.syntax);
                if frame.inner_state != NegationInnerState::Released {
                    self.release_owned_cursor(frame.inner);
                }
            }
            Frame::Unevaluated(frame) => {
                self.release_owned_cursor(frame.scope);
                self.release_owned_cursor(frame.tracker);
                if let Some(candidate) = frame.current_candidate {
                    self.release_owned_cursor(candidate.cursor);
                }
                self.session_memory
                    .release(frame.candidates.retained_bytes());
            }
            Frame::Array(mut array) => {
                if let Some(item) = array.item.take() {
                    self.discard_item(item);
                }
                self.session_memory.release(array_owned_charge(&array));
            }
            Frame::Object(mut object) => {
                if let Some(cursor) = object.property_name.take() {
                    self.release_owned_cursor(cursor);
                }
                let had_dependent = object.dependent.is_some();
                if let Some(dependent) = object.dependent.take() {
                    for (index, cursor) in dependent.cursors.into_iter().enumerate() {
                        if bit_is_set(&dependent.owned_words, index) {
                            self.release_owned_cursor(cursor);
                        }
                    }
                    self.session_memory.release(dependent.allocation_charge);
                }
                let reusable = object.resolved_value.is_none()
                    && !had_dependent
                    && !matches!(object.dependent_required_seen, PresenceBits::Heap(_))
                    && object.seen_dynamic.is_empty()
                    && matches!(
                        self.plan.as_ref().node(object.node),
                        NodePlan::Object(plan)
                            if plan.dependent_schemas.is_empty()
                    );
                if reusable && self.object_frame_free.len() < self.object_frame_free.capacity() {
                    self.object_frame_free.push(object);
                } else {
                    self.session_memory
                        .release(object_actual_owned_charge(&object));
                }
            }
            Frame::Regular { .. }
            | Frame::Number(_)
            | Frame::String(_)
            | Frame::StringEnum(_)
            | Frame::Any(_) => {}
        }
    }

    pub(crate) fn try_new(
        plan: Arc<StructuredPlan>,
    ) -> Result<StructuredState<'static>, StructuredRuntimeError> {
        let root = plan.root;
        StructuredState::new_at(PlanHandle::Owned(plan), root)
    }

    #[cfg(test)]
    pub(crate) fn new(plan: &'a StructuredPlan) -> Self {
        Self::new_at(PlanHandle::Borrowed(plan), plan.root).unwrap_or_else(|_| {
            let ledger = Arc::new(ValidatorLedger {
                live: AtomicUsize::new(0),
                limit: plan.limits.max_active_validators,
            });
            let mut state = Self::empty(PlanHandle::Borrowed(plan), ledger);
            state.dead = true;
            state
        })
    }

    fn new_at(plan: PlanHandle<'a>, node: NodeId) -> Result<Self, StructuredRuntimeError> {
        debug_assert!(
            plan.as_ref().max_concurrent_key_cursors <= plan.as_ref().limits.max_active_validators
        );
        let ledger = Arc::new(ValidatorLedger {
            live: AtomicUsize::new(0),
            limit: plan.as_ref().limits.max_active_validators,
        });
        let (scope_arena, scope_arena_charge) = ScopeArena::new_with_anchor_capacity(
            scope_arena_capacity(plan.as_ref())?,
            plan.as_ref().dynamic_anchor_capacity(),
        )?;
        let resource = plan.as_ref().node_resource(node).ok_or_else(|| {
            StructuredRuntimeError::new(LimitKind::ActiveValidators, usize::MAX, 0)
        })?;
        let scope = scope_arena.root(resource)?;
        let mut state = Self::new_at_with_scope(plan, node, 0, ledger, scope)?;
        let allocation = size_of::<ValidatorLedger>()
            .checked_add(2usize.saturating_mul(size_of::<usize>()))
            .ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::SessionBytes,
                    usize::MAX,
                    state.limits.max_session_bytes,
                )
            })?;
        state.session_memory.charge(
            allocation,
            state.limits.max_session_bytes,
            LimitKind::SessionBytes,
        )?;
        state.validator_ledger_charge = allocation;
        state.session_memory.charge(
            scope_arena_charge,
            state.limits.max_session_bytes,
            LimitKind::SessionBytes,
        )?;
        state.scope_arena_charge = scope_arena_charge;
        Ok(state)
    }

    #[cfg(test)]
    fn new_at_with_context(
        plan: PlanHandle<'a>,
        node: NodeId,
        depth_offset: usize,
        validator_ledger: Arc<ValidatorLedger>,
    ) -> Result<Self, StructuredRuntimeError> {
        let (arena, _) = ScopeArena::new(scope_arena_capacity(plan.as_ref())?)?;
        let resource = plan.as_ref().node_resource(node).ok_or_else(|| {
            StructuredRuntimeError::new(LimitKind::ActiveValidators, usize::MAX, 0)
        })?;
        let scope = arena.root(resource)?;
        Self::new_at_with_scope(plan, node, depth_offset, validator_ledger, scope)
    }

    fn new_at_with_scope(
        plan: PlanHandle<'a>,
        node: NodeId,
        depth_offset: usize,
        validator_ledger: Arc<ValidatorLedger>,
        scope: ScopeLease,
    ) -> Result<Self, StructuredRuntimeError> {
        let (node, scope) = resolve_runtime_node(plan.as_ref(), node, scope)?;
        let mut state = Self::empty(plan.clone(), validator_ledger);
        state.depth_offset = depth_offset;
        let observed = depth_offset.checked_add(1).unwrap_or(usize::MAX);
        if observed > state.limits.max_depth as usize {
            return Err(StructuredRuntimeError::new(
                LimitKind::RecursionDepth,
                observed,
                state.limits.max_depth as usize,
            ));
        }
        state.frames.try_reserve_exact(1).map_err(|_| {
            StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                usize::MAX,
                state.limits.max_session_bytes,
            )
        })?;
        state.frame_scopes.try_reserve_exact(1).map_err(|_| {
            StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                usize::MAX,
                state.limits.max_session_bytes,
            )
        })?;
        state.frame_scopes.push(scope);
        let prepared = match state.prepare_resolved_child_frame(node, (0, 0)) {
            Ok(Some(prepared)) => prepared,
            Ok(None) => {
                let _ = state.frame_scopes.pop();
                state.dead = true;
                state.charge_initial_frame()?;
                state.recompute_accepting();
                return Ok(state);
            }
            Err(error) => {
                let _ = state.frame_scopes.pop();
                return Err(error);
            }
        };
        state.frames.push(prepared.frame);
        state.charge_initial_frame()?;
        state.recompute_accepting();
        state.initial_node = Some(node);
        state.initial_checkpoint = state.checkpoint();
        Ok(state)
    }

    fn new_any(
        plan: PlanHandle<'a>,
        depth_offset: usize,
        validator_ledger: Arc<ValidatorLedger>,
        scope: ScopeLease,
    ) -> Result<Self, StructuredRuntimeError> {
        let mut state = Self::empty(plan, validator_ledger);
        state.depth_offset = depth_offset;
        let observed = depth_offset.checked_add(1).unwrap_or(usize::MAX);
        if observed > state.limits.max_depth as usize {
            return Err(StructuredRuntimeError::new(
                LimitKind::RecursionDepth,
                observed,
                state.limits.max_depth as usize,
            ));
        }
        state.frames.try_reserve_exact(1).map_err(|_| {
            StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                usize::MAX,
                state.limits.max_session_bytes,
            )
        })?;
        state.frame_scopes.try_reserve_exact(1).map_err(|_| {
            StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                usize::MAX,
                state.limits.max_session_bytes,
            )
        })?;
        state.frames.push(Frame::Any(AnyFrame::Scalar {
            phase: ScalarPhase::BeforeValue,
            bytes: 0,
        }));
        state.frame_scopes.push(scope);
        state.charge_initial_frame()?;
        state.recompute_accepting();
        state.initial_checkpoint = state.checkpoint();
        Ok(state)
    }

    fn empty(plan: PlanHandle<'a>, validator_ledger: Arc<ValidatorLedger>) -> Self {
        let limits = plan.as_ref().limits;
        let state_identity = NEXT_STRUCTURED_STATE_ID.fetch_add(1, Ordering::Relaxed);
        debug_assert_ne!(
            state_identity,
            usize::MAX,
            "structured state identity exhausted"
        );
        Self {
            state_identity,
            plan: plan.clone(),
            frames: Vec::new(),
            frame_scopes: Vec::new(),
            undo: Vec::new(),
            accepting: false,
            dead: false,
            root_closed: false,
            root_annotations: RootAnnotations::None,
            document_bytes: 0,
            session_memory: SessionMemory::default(),
            key_arena: KeyArena::default(),
            key_lease: KeyCapacityLease::default(),
            key_hasher: RandomState::new(),
            any_object_arena: Vec::new(),
            any_object_free: Vec::new(),
            object_frame_free: Vec::new(),
            combinator_frame_free: Vec::new(),
            cursors: CursorMachine::from_handle(plan, validator_ledger.clone()),
            _validator_ledger: validator_ledger,
            validator_ledger_charge: 0,
            scope_arena_charge: 0,
            limits,
            depth_offset: 0,
            byte_observed: false,
            initial_node: None,
            initial_checkpoint: Checkpoint {
                undo_len: 0,
                document_bytes: 0,
                accepting: false,
                dead: false,
            },
            resettable_to_initial: true,
        }
    }

    fn charge_initial_frame(&mut self) -> Result<(), StructuredRuntimeError> {
        let frame_bytes = self.frames.capacity().checked_mul(size_of::<Frame>());
        let owned_bytes = self.frames.first().and_then(|frame| match frame {
            Frame::Object(object) => object
                .missing_required
                .capacity()
                .checked_mul(size_of::<u64>())?
                .checked_add(object.seen_known.capacity().checked_mul(size_of::<u64>())?),
            Frame::Array(array) => Some(array_owned_charge(array)),
            Frame::Regular { .. }
            | Frame::Number(_)
            | Frame::String(_)
            | Frame::StringEnum(_)
            | Frame::Obligation(_)
            | Frame::Combinator(_)
            | Frame::Negation(_)
            | Frame::Unevaluated(_)
            | Frame::Any(_) => Some(0),
        });
        let total = frame_bytes
            .and_then(|bytes| {
                bytes.checked_add(
                    self.frame_scopes
                        .capacity()
                        .checked_mul(size_of::<ScopeLease>())?,
                )
            })
            .and_then(|bytes| bytes.checked_add(owned_bytes?))
            .ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::SessionBytes,
                    usize::MAX,
                    self.limits.max_session_bytes,
                )
            })?;
        self.session_memory.charge(
            total,
            self.limits.max_session_bytes,
            LimitKind::SessionBytes,
        )
    }

    pub(crate) fn is_accepting(&self) -> bool {
        self.accepting
    }

    pub(crate) fn is_dead(&self) -> bool {
        self.dead
    }

    /// Conservatively checks whether the next byte is impossible in the visible frame.
    /// Unknown cases continue through the exact walker.
    pub(crate) fn next_byte_is_lexically_impossible(&self, byte: u8) -> bool {
        if self.dead {
            return true;
        }
        if self.root_closed {
            return !WS.contains(&byte);
        }
        for (depth, frame) in self.frames.iter().enumerate().rev() {
            let impossible = match frame {
                Frame::Obligation(frame)
                    if frame
                        .cursors
                        .iter()
                        .all(|&cursor| self.cursors.is_accepting(cursor)) =>
                {
                    continue;
                }
                Frame::Regular { node, state } => {
                    let engine = self.engine_for(*node);
                    let dead = engine
                        .consume_token(*state, &[byte])
                        .is_none_or(|next| engine.is_dead(next));
                    if !dead {
                        return false;
                    }
                    if depth != 0 && engine.is_accepting(*state) {
                        continue;
                    }
                    true
                }
                Frame::Object(frame) => match frame.phase {
                    ObjectPhase::AfterKeyBeforeColon => !WS.contains(&byte) && byte != b':',
                    ObjectPhase::AfterValueBeforeCommaOrClose => {
                        !WS.contains(&byte) && !matches!(byte, b',' | b'}')
                    }
                    _ => false,
                },
                Frame::Array(frame) => match frame.phase {
                    ArrayPhase::AfterValueBeforeCommaOrClose => {
                        !WS.contains(&byte) && !matches!(byte, b',' | b']')
                    }
                    _ => false,
                },
                Frame::Any(AnyFrame::Object(id)) => self
                    .any_object_arena
                    .get(id.0 as usize)
                    .is_some_and(|frame| match frame.phase {
                        ObjectPhase::AfterKeyBeforeColon => !WS.contains(&byte) && byte != b':',
                        ObjectPhase::AfterValueBeforeCommaOrClose => {
                            !WS.contains(&byte) && !matches!(byte, b',' | b'}')
                        }
                        _ => false,
                    }),
                Frame::Any(AnyFrame::Array {
                    phase: AnyArrayPhase::AfterValue,
                    ..
                }) => !WS.contains(&byte) && !matches!(byte, b',' | b']'),
                _ => false,
            };
            return impossible;
        }
        false
    }

    pub(crate) fn engine_for_node(&self, node: NodeId) -> Option<&crate::automaton::RefEngine> {
        match self.plan.as_ref().node(node) {
            NodePlan::Regular(engine) => Some(engine),
            NodePlan::String(plan) => plan.pattern.as_deref(),
            _ => None,
        }
    }

    /// Certifies whether safe tokens stay within lexical and semantic boundaries.
    /// Uncertain cases use the exact walker; unresolved automata are added to `proofs`.
    pub(crate) fn certify_slice(
        &self,
        proofs: &mut Vec<SliceProof>,
        context: &mut ProofContext,
    ) -> SliceCertificate {
        let identity = self as *const Self as usize;
        if self.dead
            || self.root_closed
            || context.fuel == 0
            || context.depth == context.visited.len()
            || context.visited[..context.depth].contains(&identity)
        {
            return SliceCertificate::Unknown;
        }
        if !self.slice_resources_ready(context) {
            return SliceCertificate::Unknown;
        }
        context.fuel -= 1;
        context.visited[context.depth] = identity;
        context.depth += 1;
        let result = self.certify_frames(proofs, context);
        let result = if result == SliceCertificate::AllNoBoundary {
            self.certify_dependency_observers(proofs, context)
        } else {
            result
        };
        let result = if result == SliceCertificate::AllNoBoundary {
            self.certify_unevaluated_observers(proofs, context)
        } else {
            result
        };
        let result = if result == SliceCertificate::AllNoBoundary {
            self.certify_array_item_observers(proofs, context)
        } else {
            result
        };
        context.depth -= 1;
        result
    }

    fn certify_dependency_observers(
        &self,
        proofs: &mut Vec<SliceProof>,
        context: &mut ProofContext,
    ) -> SliceCertificate {
        for frame in &self.frames {
            let Frame::Object(object) = frame else {
                continue;
            };
            let Some(dependent) = &object.dependent else {
                continue;
            };
            for (index, &cursor) in dependent.cursors.iter().enumerate() {
                if !bit_is_set(&dependent.active_words, index) {
                    continue;
                }
                let required = context.required;
                context.required &= bit_is_set(&dependent.triggered_words, index);
                let result = self.certify_cursors(&[cursor], proofs, context);
                context.required = required;
                if result == SliceCertificate::Unknown {
                    return result;
                }
            }
        }
        SliceCertificate::AllNoBoundary
    }

    fn certify_unevaluated_observers(
        &self,
        proofs: &mut Vec<SliceProof>,
        context: &mut ProofContext,
    ) -> SliceCertificate {
        for frame in self.frames.iter().take(self.frames.len().saturating_sub(1)) {
            let Frame::Unevaluated(frame) = frame else {
                continue;
            };
            if self.certify_cursors(&[frame.scope, frame.tracker], proofs, context)
                == SliceCertificate::Unknown
            {
                return SliceCertificate::Unknown;
            }
            let Some(candidate) = frame.current_candidate.filter(|candidate| !candidate.dead)
            else {
                continue;
            };
            let required = context.required;
            context.required = false;
            let result = self.certify_cursors(&[candidate.cursor], proofs, context);
            context.required = required;
            if result == SliceCertificate::Unknown {
                return result;
            }
        }
        SliceCertificate::AllNoBoundary
    }

    fn certify_array_item_observers(
        &self,
        proofs: &mut Vec<SliceProof>,
        context: &mut ProofContext,
    ) -> SliceCertificate {
        for (depth, frame) in self.frames.iter().enumerate() {
            let Frame::Array(array) = frame else {
                continue;
            };
            let Some(item) = &array.item else {
                continue;
            };
            if item.builder.is_some() {
                return SliceCertificate::Unknown;
            }
            match self.contains_slice_policy(depth) {
                ContainsSlicePolicy::Unrestricted => {}
                ContainsSlicePolicy::Unknown => return SliceCertificate::Unknown,
                ContainsSlicePolicy::MustStayLive(node, state) => {
                    if context.fuel == 0 {
                        return SliceCertificate::Unknown;
                    }
                    context.fuel -= 1;
                    if proofs.try_reserve(1).is_err() {
                        return SliceCertificate::Unknown;
                    }
                    proofs.push(SliceProof::CandidateLive(node, state));
                }
                ContainsSlicePolicy::MustKeepStructuredCandidate(cursor) => {
                    let proof_start = proofs.len();
                    let required = context.required;
                    context.required = true;
                    let result = self.certify_cursors(&[cursor], proofs, context);
                    context.required = required;
                    if result == SliceCertificate::Unknown
                        || proofs.len() != proof_start.saturating_add(1)
                    {
                        proofs.truncate(proof_start);
                        return SliceCertificate::Unknown;
                    }
                    let SliceProof::All(node, state) = proofs[proof_start] else {
                        proofs.truncate(proof_start);
                        return SliceCertificate::Unknown;
                    };
                    proofs[proof_start] = SliceProof::CandidateLive(node, state);
                }
            }
        }
        SliceCertificate::AllNoBoundary
    }

    fn contains_slice_policy(&self, depth: usize) -> ContainsSlicePolicy {
        let array = self.array_ref(depth);
        let plan = super::plan::array_plan_of(self.plan.as_ref(), array.node);
        let Some(contains) = &plan.contains else {
            return ContainsSlicePolicy::Unrestricted;
        };
        let Some(item) = &array.item else {
            // Fresh-item proofs are deliberately distinct and remain exact until the array-frame
            // certificate in the later safe-interior-array milestone can prove their setup.
            return ContainsSlicePolicy::Unknown;
        };

        let count = if contains
            .max
            .is_some_and(|maximum| array.contains_count > maximum)
        {
            ContainsCountClass::AboveMaximum
        } else if contains.max == Some(array.contains_count) {
            ContainsCountClass::AtMaximum
        } else if array.contains_count < contains.min {
            ContainsCountClass::BelowMinimum
        } else {
            ContainsCountClass::WithinAllowedRange
        };
        let remaining = plan
            .max_items
            .map_or(RemainingItemCapacity::Unbounded, |maximum| {
                RemainingItemCapacity::Bounded(maximum.saturating_sub(array.index))
            });
        let ordinal = array.index.saturating_sub(1) as usize;
        let directly_annotated = ordinal < plan.prefix.len() || plan.tail_annotates;
        let rejecting_unevaluated = self.frames.iter().any(|frame| {
            let Frame::Unevaluated(frame) = frame else {
                return false;
            };
            frame.kind == UnevaluatedKind::Items
                && matches!(self.plan.as_ref().node(frame.node), NodePlan::Unevaluated(plan)
                    if plan.rejects_unevaluated)
        });
        let unevaluated = if directly_annotated {
            UnevaluatedItemState::AlreadyAnnotated
        } else if rejecting_unevaluated
            && self.plan.as_ref().annotations_required(array.node)
            && self
                .frames
                .iter()
                .filter(|frame| matches!(frame, Frame::Array(_)))
                .count()
                == 1
        {
            // With one active array there is an unambiguous matching annotation producer.
            UnevaluatedItemState::RequiresContainsAnnotation
        } else if rejecting_unevaluated {
            UnevaluatedItemState::Unknown
        } else {
            UnevaluatedItemState::Irrelevant
        };
        let candidate = match item.contains {
            ContainsCandidate::None => ContainsCandidateState::None,
            ContainsCandidate::Always => ContainsCandidateState::Always,
            ContainsCandidate::Dead => ContainsCandidateState::Dead,
            ContainsCandidate::Cursor(cursor) => match self.cursors.slot(cursor) {
                Ok(CursorSlot::Regular { node, state }) => {
                    ContainsCandidateState::Regular(*node, *state)
                }
                Ok(CursorSlot::Structured(_)) => ContainsCandidateState::Structured(cursor),
                Ok(CursorSlot::Vacant) | Err(_) => return ContainsSlicePolicy::Unknown,
            },
        };
        let product = ArrayItemProductState {
            item_parser: SliceCertificate::AllNoBoundary,
            candidate,
            count,
            remaining,
            unevaluated,
        };
        self.classify_array_item_product(product, contains.min, array.contains_count)
    }

    fn classify_array_item_product(
        &self,
        product: ArrayItemProductState,
        minimum: u32,
        matched: u32,
    ) -> ContainsSlicePolicy {
        if product.item_parser != SliceCertificate::AllNoBoundary
            || product.unevaluated == UnevaluatedItemState::Unknown
            || matches!(
                product.count,
                ContainsCountClass::AboveMaximum | ContainsCountClass::AtMaximum
            ) && !matches!(product.candidate, ContainsCandidateState::Dead)
        {
            return ContainsSlicePolicy::Unknown;
        }
        let current_required_by_minimum = match (product.count, product.remaining) {
            (ContainsCountClass::BelowMinimum, RemainingItemCapacity::Bounded(remaining)) => {
                matched.saturating_add(remaining) < minimum
            }
            _ => false,
        };
        let must_match = current_required_by_minimum
            || product.unevaluated == UnevaluatedItemState::RequiresContainsAnnotation;
        if !must_match {
            return ContainsSlicePolicy::Unrestricted;
        }
        match product.candidate {
            ContainsCandidateState::Always => ContainsSlicePolicy::Unrestricted,
            ContainsCandidateState::Regular(node, state) => {
                ContainsSlicePolicy::MustStayLive(node, state)
            }
            ContainsCandidateState::Structured(cursor) => {
                ContainsSlicePolicy::MustKeepStructuredCandidate(cursor)
            }
            ContainsCandidateState::None | ContainsCandidateState::Dead => {
                ContainsSlicePolicy::Unknown
            }
        }
    }

    fn slice_resources_ready(&self, context: &mut ProofContext) -> bool {
        let max_bytes = context.max_bytes;
        if max_bytes
            > self
                .limits
                .max_document_bytes
                .saturating_sub(self.document_bytes)
            || self.retained_bytes() > self.limits.max_session_bytes
        {
            return false;
        }
        for (depth, frame) in self.frames.iter().enumerate() {
            if !self.array_slice_observers_ready(depth) {
                return false;
            }
            if matches!(frame, Frame::Object(object) if object.dependent.as_ref().is_some_and(|dependent|
                dependent.marks.capacity() < active_branch_count(&dependent.active_words)))
            {
                return false;
            }
        }
        let Some(entries) = self.slice_undo_entries(max_bytes) else {
            return false;
        };
        if entries > self.limits.max_undo_entries
            || !context.admit_growth(GrowthRequest {
                len: self.undo.len(),
                capacity: self.undo.capacity(),
                additional: entries - self.undo.len(),
                item_size: size_of::<Undo>(),
            })
        {
            return false;
        }
        let key = match self.frames.last() {
            Some(Frame::Object(_)) => {
                self.key_lease_growth_request(self.frames.len() - 1, max_bytes)
            }
            Some(Frame::Any(AnyFrame::Object(id))) => {
                debug_assert!(self.any_object_arena.get(id.0 as usize).is_some());
                self.key_lease_growth_request(self.frames.len() - 1, max_bytes)
            }
            Some(Frame::Any(AnyFrame::StringBody { bytes, .. })) => {
                return max_bytes <= self.limits.max_string_bytes.saturating_sub(*bytes as usize)
            }
            _ => None,
        };
        key.is_none_or(|request| {
            max_bytes
                <= self
                    .limits
                    .max_key_bytes
                    .saturating_sub(self.frames.last().map_or(0, |frame| {
                        match frame {
                            Frame::Object(object) => object.key.len(),
                            Frame::Any(AnyFrame::Object(id)) => self
                                .any_object_arena
                                .get(id.0 as usize)
                                .map_or(0, |object| object.key.len()),
                            _ => 0,
                        }
                    }))
                && context.admit_growth(request)
        })
    }

    fn slice_undo_entries(&self, max_bytes: usize) -> Option<usize> {
        let (per_byte, headroom) = match self.frames.last() {
            Some(Frame::Obligation(frame)) => (frame.cursors.len(), 0),
            Some(Frame::Combinator(frame)) => {
                let active = active_branch_count(&frame.active_words);
                (active, frame.cursors.len().saturating_sub(active))
            }
            Some(Frame::Unevaluated(frame)) => (
                2 + usize::from(
                    frame
                        .current_candidate
                        .is_some_and(|candidate| !candidate.dead),
                ),
                0,
            ),
            Some(Frame::Negation(frame)) => {
                let active = usize::from(frame.inner_state == NegationInnerState::Active);
                (1 + active, 1 - active)
            }
            Some(Frame::Object(frame)) => (2 + usize::from(frame.property_name.is_some()), 0),
            Some(Frame::Any(AnyFrame::Object(_))) => (2, 0),
            _ => (1, 0),
        };
        let observers =
            self.frames
                .iter()
                .enumerate()
                .try_fold(0usize, |total, (depth, frame)| {
                    let active = match frame {
                        Frame::Object(object) => object
                            .dependent
                            .as_ref()
                            .map_or(0, |dependent| active_branch_count(&dependent.active_words)),
                        Frame::Unevaluated(frame) if depth + 1 < self.frames.len() => {
                            2 + usize::from(
                                frame
                                    .current_candidate
                                    .is_some_and(|candidate| !candidate.dead),
                            )
                        }
                        Frame::Array(array) => array.item.as_ref().map_or(0, |item| {
                            usize::from(matches!(item.contains, ContainsCandidate::Cursor(_)))
                        }),
                        _ => 0,
                    };
                    total.checked_add(active)
                })?;
        let per_byte = per_byte.checked_add(observers)?;
        max_bytes
            .checked_mul(per_byte)
            .and_then(|n| n.checked_add(headroom))
            .and_then(|n| self.undo.len().checked_add(n))
    }

    fn slice_cursor_at(&self, index: usize) -> Option<CursorId> {
        let mut remaining = index;
        let mut local_index = 0;
        while let Some(cursor) = self.local_slice_cursor_at(local_index) {
            if remaining == 0 {
                return Some(cursor);
            }
            remaining -= 1;
            local_index += 1;
        }
        for frame in &self.frames {
            let Frame::Object(object) = frame else {
                continue;
            };
            let Some(dependent) = &object.dependent else {
                continue;
            };
            for (index, &cursor) in dependent.cursors.iter().enumerate() {
                if bit_is_set(&dependent.active_words, index) {
                    if remaining == 0 {
                        return Some(cursor);
                    }
                    remaining -= 1;
                }
            }
        }
        for frame in self.frames.iter().take(self.frames.len().saturating_sub(1)) {
            let Frame::Unevaluated(frame) = frame else {
                continue;
            };
            for cursor in [
                Some(frame.scope),
                Some(frame.tracker),
                frame
                    .current_candidate
                    .filter(|candidate| !candidate.dead)
                    .map(|candidate| candidate.cursor),
            ]
            .into_iter()
            .flatten()
            {
                if remaining == 0 {
                    return Some(cursor);
                }
                remaining -= 1;
            }
        }
        for frame in &self.frames {
            let Frame::Array(array) = frame else {
                continue;
            };
            let Some(ItemTracking {
                contains: ContainsCandidate::Cursor(cursor),
                ..
            }) = &array.item
            else {
                continue;
            };
            if remaining == 0 {
                return Some(*cursor);
            }
            remaining -= 1;
        }
        None
    }

    fn local_slice_cursor_at(&self, index: usize) -> Option<CursorId> {
        match self.frames.last()? {
            Frame::Obligation(frame) => frame.cursors.get(index).copied(),
            Frame::Combinator(frame) => frame
                .cursors
                .iter()
                .enumerate()
                .filter(|(i, _)| bit_is_set(&frame.active_words, *i))
                .nth(index)
                .map(|(_, cursor)| *cursor),
            Frame::Unevaluated(frame) => match index {
                0 => Some(frame.scope),
                1 => Some(frame.tracker),
                2 => frame
                    .current_candidate
                    .filter(|candidate| !candidate.dead)
                    .map(|candidate| candidate.cursor),
                _ => None,
            },
            Frame::Negation(frame) => match index {
                0 => Some(frame.syntax),
                1 if frame.inner_state == NegationInnerState::Active => Some(frame.inner),
                _ => None,
            },
            Frame::Object(frame) if index == 0 => frame.property_name,
            _ => None,
        }
    }

    pub(crate) fn prepare_certified_slice(
        &mut self,
        context: &mut ProofContext,
    ) -> Result<(), StructuredRuntimeError> {
        let identity = self as *const Self as usize;
        if context.fuel == 0
            || context.depth == context.visited.len()
            || context.visited[..context.depth].contains(&identity)
        {
            return Err(self.cursors.error());
        }
        context.fuel -= 1;
        context.visited[context.depth] = identity;
        context.depth += 1;
        let result = self.prepare_certified_slice_inner(context);
        context.depth -= 1;
        if result.is_err() {
            self.discard_prepared_key_leases_tree();
        }
        result
    }

    fn prepare_certified_slice_inner(
        &mut self,
        context: &mut ProofContext,
    ) -> Result<(), StructuredRuntimeError> {
        let entries = self.slice_undo_entries(context.max_bytes).ok_or_else(|| {
            StructuredRuntimeError::new(
                LimitKind::UndoEntries,
                usize::MAX,
                self.limits.max_undo_entries,
            )
        })?;
        self.reserve_undo(entries - self.undo.len())?;
        let object_depth = match self.frames.last() {
            Some(Frame::Object(_) | Frame::Any(AnyFrame::Object(_))) => Some(self.frames.len() - 1),
            _ => None,
        };
        if let Some(depth) = object_depth {
            context.before_key_lease_acquisition(self.limits.max_session_bytes)?;
            self.prepare_key_lease(depth, context.max_bytes)?;
        }
        for index in 0..64 {
            let Some(cursor) = self.slice_cursor_at(index) else {
                break;
            };
            let budget = self
                .limits
                .max_session_bytes
                .checked_sub(self.session_memory.live())
                .ok_or_else(|| {
                    StructuredRuntimeError::new(
                        LimitKind::SessionBytes,
                        usize::MAX,
                        self.limits.max_session_bytes,
                    )
                })?;
            self.cursors.set_cursor_budget(cursor, budget)?;
            if let CursorSlot::Structured(id) = *self.cursors.slot(cursor)? {
                self.cursors
                    .structured_state_mut(id)?
                    .prepare_certified_slice(context)?;
            }
        }
        Ok(())
    }

    fn certify_frames(
        &self,
        proofs: &mut Vec<SliceProof>,
        fuel: &mut ProofContext,
    ) -> SliceCertificate {
        match self.frames.last() {
            // These three already carry their own lexical position, so the existing residual is
            // the certificate; it only has to hand over the automaton that needs proving.
            Some(Frame::Regular { .. }) | Some(Frame::String(_)) | Some(Frame::Object(_)) => {
                if fuel.body_prefix
                    && matches!(self.frames.last(), Some(Frame::String(frame))
                    if matches!(self.plan.as_ref().node(frame.node), NodePlan::String(plan)
                        if plan.pattern.is_some()))
                {
                    return SliceCertificate::Unknown;
                }
                let Some(residual) = self.local_slice_residual() else {
                    return SliceCertificate::Unknown;
                };
                if let Some(remaining) = residual.remaining_scalars {
                    if fuel.body_prefix || !fuel.required {
                        return SliceCertificate::Unknown;
                    }
                    fuel.remaining_scalars = Some(
                        fuel.remaining_scalars
                            .map_or(remaining, |old| old.min(remaining)),
                    );
                }
                if let (Some(_), Some(state)) = (residual.engine, residual.pattern_state) {
                    if proofs.try_reserve(1).is_err() {
                        return SliceCertificate::Unknown;
                    }
                    proofs.push(SliceProof::All(residual.node, state));
                }
                SliceCertificate::AllNoBoundary
            }
            // Every cursor is stepped by the byte, so every one of them must certify.
            Some(Frame::Obligation(frame)) => self.certify_cursors(&frame.cursors, proofs, fuel),
            // The tracker and scope both receive the bytes, as does the live candidate.
            Some(Frame::Unevaluated(frame)) => {
                let finite_keys_active = matches!(
                    self.plan.as_ref().node(frame.node),
                    NodePlan::Unevaluated(plan) if plan.finite_evaluated_keys.is_some()
                );
                if finite_keys_active {
                    match self.cursors.any_root_key_state(frame.tracker) {
                        Ok(Some(_)) | Err(_) => return SliceCertificate::Unknown,
                        Ok(None) => {}
                    }
                }
                let mut cursors = [frame.scope, frame.tracker];
                if self.certify_cursors(&cursors, proofs, fuel) == SliceCertificate::Unknown {
                    return SliceCertificate::Unknown;
                }
                match frame.current_candidate.filter(|candidate| !candidate.dead) {
                    Some(candidate) => {
                        cursors[0] = candidate.cursor;
                        let required = fuel.required;
                        fuel.required = false;
                        let result = self.certify_cursors(&cursors[..1], proofs, fuel);
                        fuel.required = required;
                        result
                    }
                    None => SliceCertificate::AllNoBoundary,
                }
            }
            // Every active branch must certify so no branch set changes or completes.
            Some(Frame::Combinator(frame)) => {
                let required = fuel.required;
                // Inactive branches cannot revive while the surviving string remains open.
                fuel.required &= frame.kind == CombinatorKind::All
                    || active_branch_count(&frame.active_words) == 1;
                let mut any_active = false;
                for (index, &cursor) in frame.cursors.iter().enumerate() {
                    if bit_is_set(&frame.active_words, index) {
                        any_active = true;
                        if self.certify_cursors(&[cursor], proofs, fuel)
                            == SliceCertificate::Unknown
                        {
                            fuel.required = required;
                            return SliceCertificate::Unknown;
                        }
                    }
                }
                fuel.required = required;
                if !any_active {
                    return SliceCertificate::Unknown;
                }
                SliceCertificate::AllNoBoundary
            }
            // Both cursors receive the byte. Negation is decided when the value closes, and a
            // certified slice closes nothing, so its verdict cannot move here.
            Some(Frame::Negation(frame)) => {
                if self.certify_cursors(&[frame.syntax], proofs, fuel) == SliceCertificate::Unknown
                {
                    return SliceCertificate::Unknown;
                }
                if frame.inner_state != NegationInnerState::Active {
                    return SliceCertificate::AllNoBoundary;
                }
                let required = fuel.required;
                fuel.required = false;
                let result = self.certify_cursors(&[frame.inner], proofs, fuel);
                fuel.required = required;
                result
            }
            // Unconstrained JSON certifies only inside string or key content.
            Some(Frame::Any(frame)) => match frame {
                AnyFrame::StringBody { decoder, .. } if decoder.at_boundary() => {
                    SliceCertificate::AllNoBoundary
                }
                AnyFrame::Object(id) => match self.any_object_arena.get(id.0 as usize) {
                    Some(object)
                        if object.phase == ObjectPhase::InKey
                            && object.key_decoder.at_boundary() =>
                    {
                        SliceCertificate::AllNoBoundary
                    }
                    _ => SliceCertificate::Unknown,
                },
                _ => SliceCertificate::Unknown,
            },
            // Array frames handle punctuation or open children; interior strings live in the child.
            Some(Frame::Array(_)) => SliceCertificate::Unknown,
            _ => SliceCertificate::Unknown,
        }
    }

    fn certify_cursors(
        &self,
        cursors: &[CursorId],
        proofs: &mut Vec<SliceProof>,
        fuel: &mut ProofContext,
    ) -> SliceCertificate {
        for &cursor in cursors {
            if fuel.fuel == 0 {
                return SliceCertificate::Unknown;
            }
            fuel.fuel -= 1;
            let certificate = match self.cursors.slot(cursor) {
                Ok(CursorSlot::Regular { node, state }) => {
                    if proofs.try_reserve(1).is_err() {
                        return SliceCertificate::Unknown;
                    }
                    proofs.push(SliceProof::All(*node, *state));
                    SliceCertificate::AllNoBoundary
                }
                Ok(CursorSlot::Structured(id)) => match self.cursors.structured_state(*id) {
                    Ok(state) => state.certify_slice(proofs, fuel),
                    Err(_) => SliceCertificate::Unknown,
                },
                Ok(CursorSlot::Vacant) | Err(_) => SliceCertificate::Unknown,
            };
            if certificate == SliceCertificate::Unknown {
                return SliceCertificate::Unknown;
            }
        }
        SliceCertificate::AllNoBoundary
    }

    pub(crate) fn local_slice_residual(&self) -> Option<LocalSliceResidual<'_>> {
        if self.dead || self.root_closed {
            return None;
        }
        let remaining_document_bytes = self
            .limits
            .max_document_bytes
            .saturating_sub(self.document_bytes);
        // Safe string-body tokens cannot close strings, escape, or reach ancestor frames.
        match self.frames.last()? {
            Frame::Regular { node, state } => Some(LocalSliceResidual {
                node: *node,
                engine: Some(self.engine_for(*node)),
                pattern_state: Some(*state),
                remaining_document_bytes,
                remaining_scalars: None,
            }),
            Frame::String(frame) => {
                let NodePlan::String(plan) = self.plan.as_ref().node(frame.node) else {
                    return None;
                };
                if frame.phase != StringPhase::Body || !frame.decoder.at_boundary() {
                    return None;
                }
                // maxLength counts decoded scalars, which the scalar-bounded slice tables express
                // exactly; a byte budget cannot, since one four-byte token spends one scalar.
                let remaining_scalars = match plan.max_scalars {
                    Some(max) => Some((max as usize).checked_sub(frame.scalars as usize)?),
                    None => None,
                };
                // `maxLength` may invalidate a live DFA transition, so use the exact walker.
                if plan.pattern.is_some() && remaining_scalars.is_some() {
                    return None;
                }
                Some(LocalSliceResidual {
                    node: frame.node,
                    engine: plan.pattern.as_deref(),
                    pattern_state: frame.pattern_state,
                    remaining_document_bytes,
                    remaining_scalars,
                })
            }
            // Closed-object key prefixes require exact walking until unseen-key viability is proven.
            Frame::Object(frame) => {
                if frame.phase != ObjectPhase::InKey || !frame.key_decoder.at_boundary() {
                    return None;
                }
                let depth = self.frames.len().checked_sub(1)?;
                if self.current_object_key_viable(depth).is_some() {
                    return None;
                }
                // Key and document budgets both bound safe-token admission.
                let remaining = remaining_document_bytes
                    .min(self.limits.max_key_bytes.saturating_sub(frame.key.len()));
                match frame.property_name {
                    // A propertyNames cursor may itself be a nested structured state; only its
                    // regular form carries the engine the slice proof needs.
                    Some(cursor) => {
                        let CursorSlot::Regular { node, state } = self.cursors.slot(cursor).ok()?
                        else {
                            return None;
                        };
                        Some(LocalSliceResidual {
                            node: *node,
                            engine: Some(self.engine_for(*node)),
                            pattern_state: Some(*state),
                            remaining_document_bytes: remaining,
                            remaining_scalars: None,
                        })
                    }
                    // An unrestricted key: no automaton constrains its bytes, so every safe token
                    // is already accepted and the slice only avoids re-proving that per token.
                    None => Some(LocalSliceResidual {
                        node: frame.node,
                        engine: None,
                        pattern_state: None,
                        remaining_document_bytes: remaining,
                        remaining_scalars: None,
                    }),
                }
            }
            _ => None,
        }
    }

    pub(crate) fn prepare_local_slice(
        &mut self,
        max_token_bytes: usize,
    ) -> Result<SlicePreparation, StructuredRuntimeError> {
        if self.local_slice_residual().is_none()
            || max_token_bytes
                > self
                    .limits
                    .max_document_bytes
                    .saturating_sub(self.document_bytes)
        {
            return Ok(SlicePreparation::Ineligible);
        }
        if self.frames.iter().enumerate().any(|(depth, frame)| {
            self.array_item_tracking_active(depth)
                || matches!(frame, Frame::Object(object) if object.dependent.is_some())
        }) {
            return Ok(SlicePreparation::Ineligible);
        }
        // Preflight both buffers so a predictable budget miss leaves their capacities unchanged.
        let key_growth = match self.frames.last() {
            Some(Frame::Object(frame)) if frame.phase == ObjectPhase::InKey => {
                self.key_lease_growth_request(self.frames.len() - 1, max_token_bytes)
            }
            _ => None,
        };
        let entries_per_byte = match self.frames.last() {
            Some(Frame::Object(frame)) => 2 + usize::from(frame.property_name.is_some()),
            _ => 1,
        };
        let undo_entries = max_token_bytes
            .checked_mul(entries_per_byte)
            .ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::UndoEntries,
                    usize::MAX,
                    self.limits.max_undo_entries,
                )
            })?;
        let next_undo = self.undo.len().checked_add(undo_entries).ok_or_else(|| {
            StructuredRuntimeError::new(
                LimitKind::UndoEntries,
                usize::MAX,
                self.limits.max_undo_entries,
            )
        })?;
        if next_undo > self.limits.max_undo_entries {
            return Ok(SlicePreparation::Ineligible);
        }
        let cap = self.local_session_budget()?;
        let undo_request = GrowthRequest {
            len: self.undo.len(),
            capacity: self.undo.capacity(),
            additional: undo_entries,
            item_size: size_of::<Undo>(),
        };
        let Ok(undo_plan) = plan_growth(undo_request, self.session_memory.live(), cap) else {
            return Ok(SlicePreparation::Ineligible);
        };
        let projected_live = undo_request
            .len
            .checked_add(undo_plan.reserve_additional)
            .and_then(|n| {
                n.saturating_sub(undo_request.capacity)
                    .checked_mul(undo_request.item_size)
            })
            .and_then(|n| self.session_memory.live().checked_add(n))
            .ok_or_else(|| StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, cap))?;
        if let Some(req) = key_growth {
            req.len.checked_add(req.additional).ok_or_else(|| {
                StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, cap)
            })?;
            if plan_growth(req, projected_live, cap).is_err() {
                return Ok(SlicePreparation::Ineligible);
            }
        }
        self.reserve_undo(undo_entries)?;
        if key_growth.is_some() {
            self.prepare_key_lease(self.frames.len() - 1, max_token_bytes)?;
        }
        Ok(SlicePreparation::Ready)
    }

    fn can_extend_value(&self) -> bool {
        if self.root_closed || self.dead {
            return false;
        }
        match &self.frames[..] {
            [Frame::Regular { node, state }] => {
                let engine = self.engine_for(*node);
                let live = engine.live_classes(*state);
                engine
                    .class_table
                    .members
                    .iter()
                    .enumerate()
                    .any(|(class, bytes)| {
                        live.contains(class) && bytes.iter().any(|byte| !WS.contains(byte))
                    })
            }
            [Frame::Combinator(frame)] => {
                frame.cursors.iter().enumerate().any(|(index, cursor)| {
                    bit_is_set(&frame.active_words, index) && self.cursors.can_extend_value(*cursor)
                })
            }
            [Frame::Negation(frame)] => self.cursors.can_extend_value(frame.syntax),
            [Frame::StringEnum(frame)] => {
                let NodePlan::StringEnum(plan) = self.plan.as_ref().node(frame.node) else {
                    return false;
                };
                frame.phase == StringPhase::BeforeQuote
                    || plan.contains_exact(frame.lo, frame.hi, frame.decoded_bytes)
                    || plan.can_continue(frame.lo, frame.hi, frame.decoded_bytes)
            }
            [Frame::Any(AnyFrame::Scalar { phase, .. })] => *phase != ScalarPhase::LiteralDone,
            [] => false,
            _ => true,
        }
    }

    pub(crate) fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            undo_len: self.undo.len(),
            document_bytes: self.document_bytes,
            accepting: self.accepting,
            dead: self.dead,
        }
    }

    pub(crate) fn rollback(&mut self, mark: Checkpoint) {
        while self.undo.len() > mark.undo_len {
            match self.undo.pop().expect("checked non-empty above") {
                Undo::RollbackUnevaluatedCursor { cursor, mark } => {
                    self.cursors
                        .rollback(cursor, mark)
                        .expect("unevaluated cursor mark belongs to active checkpoint");
                }
                Undo::StartUnevaluatedCandidate { depth, cursor } => {
                    let current = match &mut self.frames[depth] {
                        Frame::Unevaluated(frame) => frame.current_candidate.take(),
                        _ => unreachable!("expected an unevaluated frame"),
                    };
                    debug_assert_eq!(current.map(|candidate| candidate.cursor), Some(cursor));
                    self.cursors
                        .release(cursor)
                        .expect("installed candidate cursor remains owned until rollback");
                }
                Undo::FinishUnevaluatedCandidate {
                    depth,
                    ordinal,
                    cursor,
                    dead,
                    old_count,
                } => match &mut self.frames[depth] {
                    Frame::Unevaluated(frame) => {
                        frame.current_candidate = Some(UnevaluatedCandidate {
                            ordinal,
                            cursor,
                            dead,
                        });
                        frame.location_count = old_count;
                    }
                    _ => unreachable!("expected an unevaluated frame"),
                },
                Undo::MarkUnevaluatedCandidateDead { depth } => match &mut self.frames[depth] {
                    Frame::Unevaluated(frame) => {
                        frame
                            .current_candidate
                            .as_mut()
                            .expect("candidate exists when its dead flag changes")
                            .dead = false;
                    }
                    _ => unreachable!("expected an unevaluated frame"),
                },
                Undo::SetUnevaluatedInstanceKind { depth } => match &mut self.frames[depth] {
                    Frame::Unevaluated(frame) => frame.instance_kind = None,
                    _ => unreachable!("expected an unevaluated frame"),
                },
                Undo::SetRootAnnotations {
                    old,
                    restore_target,
                } => {
                    let current = std::mem::replace(&mut self.root_annotations, old);
                    let restored = match (restore_target, current) {
                        (AnnotationTarget::Object(_), RootAnnotations::Object(set))
                        | (AnnotationTarget::Array(_), RootAnnotations::Array(set)) => set,
                        _ => unreachable!("root annotation location type changed"),
                    };
                    *self.evaluated_mut(restore_target) = restored;
                }
                Undo::ReplaceRootAnnotations { old } => {
                    self.root_annotations = old;
                }
                Undo::EvaluatedWord { target, word, old } => match self.evaluated_mut(target) {
                    EvaluatedSet::Inline(value) if word == 0 => *value = old,
                    EvaluatedSet::Heap(words) => words[word] = old,
                    EvaluatedSet::None | EvaluatedSet::Inline(_) | EvaluatedSet::All => {
                        unreachable!("evaluated word undo targets the original representation")
                    }
                },
                Undo::ReplaceEvaluatedSet {
                    target,
                    old,
                    new_charge,
                } => {
                    *self.evaluated_mut(target) = old;
                    self.session_memory.release(new_charge);
                }
                Undo::SetRegularState { depth, old } => {
                    if let Frame::Regular { state, .. } = &mut self.frames[depth] {
                        *state = old;
                    }
                }
                Undo::SetNumberFrame { depth, old } => {
                    self.frames[depth] = Frame::Number(old);
                }
                Undo::RollbackObligationCursor { cursor, mark } => {
                    self.cursors
                        .rollback(cursor, mark)
                        .expect("cursor mark belongs to active checkpoint");
                }
                Undo::RollbackCombinatorCursor { cursor, mark } => {
                    self.cursors
                        .rollback(cursor, mark)
                        .expect("combinator cursor mark belongs to active checkpoint");
                }
                Undo::ReactivateCombinatorBranch { depth, branch } => {
                    if let Frame::Combinator(frame) = &mut self.frames[depth] {
                        set_bit(&mut frame.active_words, branch as usize);
                    }
                }
                Undo::RollbackNegationCursor { cursor, mark } => {
                    self.cursors
                        .rollback(cursor, mark)
                        .expect("negation cursor mark belongs to active checkpoint");
                }
                Undo::ReactivateNegationInner { depth } => {
                    if let Frame::Negation(frame) = &mut self.frames[depth] {
                        frame.inner_state = NegationInnerState::Active;
                    }
                }
                Undo::RollbackDependentCursor { cursor, mark } => {
                    self.cursors
                        .rollback(cursor, mark)
                        .expect("dependent cursor mark belongs to active checkpoint");
                }
                Undo::ReactivateDependentSchema { depth, dependency } => {
                    if let Some(frame) = self.object_mut(depth).dependent.as_mut() {
                        set_bit(&mut frame.active_words, dependency as usize);
                    }
                }
                Undo::ClearDependentTrigger { depth, dependency } => {
                    if let Some(frame) = self.object_mut(depth).dependent.as_mut() {
                        clear_bit(&mut frame.triggered_words, dependency as usize);
                    }
                }
                Undo::ClearDependentRequiredPresence { depth, name } => {
                    self.object_mut(depth)
                        .dependent_required_seen
                        .clear(name as usize);
                }
                Undo::SetStringFrame { depth, old } => self.frames[depth] = Frame::String(old),
                Undo::SetStringEnumFrame { depth, old } => {
                    self.frames[depth] = Frame::StringEnum(old)
                }
                Undo::SetObjectPhase { depth, old } => self.object_mut(depth).phase = old,
                Undo::SetArrayPhase { depth, old } => self.array_mut(depth).phase = old,
                Undo::RestoreArrayIndex { depth, old } => self.array_mut(depth).index = old,
                Undo::StartArrayItem { depth } => {
                    if let Some(item) = self.array_mut(depth).item.take() {
                        self.discard_item(item);
                    }
                }
                Undo::RestoreItemContainsCandidate { depth, old } => {
                    if let Some(item) = self.array_mut(depth).item.as_mut() {
                        item.contains = old;
                    } else if let ContainsCandidate::Cursor(cursor) = old {
                        self.release_owned_cursor(cursor);
                    }
                }
                Undo::RollbackContainsCursor { cursor, mark } => {
                    self.cursors
                        .rollback(cursor, mark)
                        .expect("cursor mark belongs to active checkpoint");
                }
                Undo::RollbackCanonicalBuild { depth, mark } => {
                    let Frame::Array(af) = &mut self.frames[depth] else {
                        unreachable!("expected an array frame")
                    };
                    let ArrayFrame {
                        item, canonical, ..
                    } = af;
                    let builder = item
                        .as_mut()
                        .expect("set when this undo was pushed")
                        .builder
                        .as_mut()
                        .expect("unique_items implies a builder");
                    canonical
                        .as_deref_mut()
                        .and_then(|sets| sets.first_mut())
                        .expect("unique_items implies a canonical set")
                        .rollback_item_byte(builder, mark);
                }
                Undo::FinishArrayItem { depth, old } => self.array_mut(depth).item = Some(old),
                Undo::RollbackCanonicalInsert {
                    depth,
                    fingerprint,
                    old_bucket_len,
                    bucket_created,
                    bucket_capacity_bytes,
                } => {
                    self.array_mut(depth)
                        .canonical_mut()
                        .expect("unique_items implies a canonical set")
                        .rollback_insert(
                            fingerprint,
                            old_bucket_len,
                            bucket_created,
                            bucket_capacity_bytes,
                        );
                }
                Undo::RestoreContainsCount { depth, old } => {
                    self.array_mut(depth).contains_count = old
                }
                Undo::ReplaceAnyFrame { depth, old } => {
                    // A rolled-back object frame must return its arena slot to the free list, or the
                    // arena grows on every speculative retry; the push is pre-reserved, never allocates.
                    if let Frame::Any(AnyFrame::Object(id)) = &self.frames[depth] {
                        self.any_object_free.push(id.0);
                    }
                    self.frames[depth] = Frame::Any(old);
                }
                Undo::SetAnyScalar {
                    depth,
                    old_phase,
                    old_bytes,
                } => {
                    if let AnyFrame::Scalar { phase, bytes } = self.any_mut(depth) {
                        *phase = old_phase;
                        *bytes = old_bytes;
                    }
                }
                Undo::SetAnyString {
                    depth,
                    old_decoder,
                    old_bytes,
                } => {
                    if let AnyFrame::StringBody { decoder, bytes } = self.any_mut(depth) {
                        *decoder = old_decoder;
                        *bytes = old_bytes;
                    }
                }
                Undo::SetAnyArrayPhase { depth, old } => {
                    if let AnyFrame::Array { phase, .. } = self.any_mut(depth) {
                        *phase = old;
                    }
                }
                Undo::RestoreAnyArrayItems { depth, old } => {
                    if let AnyFrame::Array { items, .. } = self.any_mut(depth) {
                        *items = old;
                    }
                }
                Undo::SetAnyObjectPhase { depth, old } => self.any_object_mut(depth).phase = old,
                Undo::TruncateAnyKey {
                    depth,
                    old_len,
                    lease_generation,
                } => {
                    if let Some(generation) = lease_generation {
                        let owner = self
                            .key_owner(depth)
                            .expect("any-key lease undo owner remains representable");
                        debug_assert_eq!(
                            self.key_lease.owner,
                            Some(owner),
                            "wrong key lease owner"
                        );
                        debug_assert_eq!(
                            self.key_lease.generation, generation,
                            "stale key lease generation"
                        );
                        debug_assert_eq!(self.key_lease.phase, KeyLeasePhase::Prepared);
                        self.key_lease.buffer.truncate(old_len);
                    } else {
                        self.any_object_mut(depth).key.truncate(old_len);
                    }
                }
                Undo::ResetAnyKey {
                    depth,
                    old_key,
                    old_generation,
                    old_decoder,
                } => {
                    // The current key buffer is dropped when the prior key is restored; release its
                    // charge so speculative retries over any-object keys stay accounting-exact.
                    let dropped = {
                        let obj = self.any_object_mut(depth);
                        obj.key_decoder = old_decoder;
                        obj.key_generation = old_generation;
                        std::mem::replace(&mut obj.key, old_key)
                    };
                    self.session_memory.release(dropped.capacity());
                }
                Undo::RestoreAnyObjectDecoder { depth, old } => {
                    self.any_object_mut(depth).key_decoder = old;
                }
                Undo::AnyRemoveSeenDynamicBucket { depth, hash } => {
                    self.any_object_mut(depth).seen_dynamic.remove(&hash);
                }
                Undo::AnyRestoreSeenDynamicOne { depth, hash, id } => {
                    self.any_object_mut(depth)
                        .seen_dynamic
                        .insert(hash, DynamicSlot::One(id));
                }
                Undo::AnyTruncateSeenDynamicMany {
                    depth,
                    hash,
                    old_len,
                } => {
                    if let Some(DynamicSlot::Many(ids)) =
                        self.any_object_mut(depth).seen_dynamic.get_mut(&hash)
                    {
                        ids.truncate(old_len);
                    }
                }
                Undo::RestoreAnyObjectProperties { depth, old } => {
                    self.any_object_mut(depth).properties = old;
                }
                Undo::TruncateKey {
                    depth,
                    old_len,
                    lease_generation,
                } => {
                    if let Some(generation) = lease_generation {
                        let owner = self
                            .key_owner(depth)
                            .expect("key lease undo owner remains representable");
                        debug_assert_eq!(
                            self.key_lease.owner,
                            Some(owner),
                            "wrong key lease owner"
                        );
                        debug_assert_eq!(
                            self.key_lease.generation, generation,
                            "stale key lease generation"
                        );
                        debug_assert_eq!(self.key_lease.phase, KeyLeasePhase::Prepared);
                        self.key_lease.buffer.truncate(old_len);
                    } else {
                        self.object_mut(depth).key.truncate(old_len);
                    }
                }
                Undo::ResetKey {
                    depth,
                    old_key,
                    old_generation,
                    old_decoder,
                } => {
                    // The current key buffer is recycled into `spare_key`; the spare it displaces is
                    // truly dropped, so release its charge here to keep the ledger exact on rollback.
                    let dropped = {
                        let obj = self.object_mut(depth);
                        let recycled = std::mem::replace(&mut obj.key, old_key);
                        obj.key_generation = old_generation;
                        obj.key_decoder = old_decoder;
                        std::mem::replace(&mut obj.spare_key, recycled)
                    };
                    self.session_memory.release(dropped.capacity());
                }
                Undo::RestoreDecoder { depth, old } => self.object_mut(depth).key_decoder = old,
                Undo::StartPropertyName { depth, cursor } => {
                    self.object_mut(depth).property_name = None;
                    self.release_owned_cursor(cursor);
                }
                Undo::RollbackPropertyName { cursor, mark } => {
                    self.cursors
                        .rollback(cursor, mark)
                        .expect("property-name mark belongs to active checkpoint");
                }
                Undo::FinishPropertyName { depth, cursor } => {
                    self.object_mut(depth).property_name = Some(cursor);
                }
                Undo::RestoreResolvedValue { depth, old } => {
                    let dropped =
                        std::mem::replace(&mut self.object_mut(depth).resolved_value, old);
                    if let Some(value) = dropped {
                        self.session_memory.release(obligation_owned_bytes(&value));
                    }
                }
                Undo::RestorePropertyCount { depth, old } => {
                    self.object_mut(depth).property_count = old;
                }
                Undo::RestoreRequiredWord { depth, word, old } => {
                    self.object_mut(depth).missing_required[word] = old;
                }
                Undo::RemoveSeenKnown { depth, id } => {
                    clear_bit(&mut self.object_mut(depth).seen_known, id.0 as usize)
                }
                Undo::RemoveSeenDynamicBucket { depth, hash } => {
                    self.object_mut(depth).seen_dynamic.remove(&hash);
                }
                Undo::RestoreSeenDynamicOne { depth, hash, id } => {
                    // Reverting a one-to-many split drops the spill vector, so release its charge
                    // here (its capacity is truly freed) to keep the ledger exact on rollback.
                    let old = self
                        .object_mut(depth)
                        .seen_dynamic
                        .insert(hash, DynamicSlot::One(id));
                    if let Some(DynamicSlot::Many(spill)) = old {
                        self.session_memory
                            .release(spill.capacity().saturating_mul(size_of::<KeyId>()));
                    }
                }
                Undo::TruncateSeenDynamicMany {
                    depth,
                    hash,
                    old_len,
                } => {
                    if let Some(DynamicSlot::Many(ids)) =
                        self.object_mut(depth).seen_dynamic.get_mut(&hash)
                    {
                        ids.truncate(old_len);
                    }
                }
                Undo::InternKey {
                    old_arena_len,
                    old_span_count,
                } => {
                    self.key_arena.bytes.truncate(old_arena_len as usize);
                    self.key_arena.spans.truncate(old_span_count as usize);
                }
                Undo::PopFrame => {
                    if let Some(frame) = self.frames.pop() {
                        let _ = self.frame_scopes.pop();
                        self.discard_new_frame(frame);
                    }
                }
                Undo::PushFrame(frame, scope) => {
                    self.frames.push(frame);
                    self.frame_scopes.push(scope);
                }
                Undo::SetRootClosed { old } => self.root_closed = old,
            }
        }
        self.document_bytes = mark.document_bytes;
        self.accepting = mark.accepting;
        self.dead = mark.dead;
    }

    /// Commits undo history and reclaims key storage for closed objects.
    /// Stack order ensures no live `KeyId` is invalidated.
    pub(crate) fn commit_checkpoint(&mut self) -> Result<(), StructuredRuntimeError> {
        self.validate_commit_tree()?;
        self.apply_commit_tree();
        Ok(())
    }

    fn validate_commit_tree(&self) -> Result<(), StructuredRuntimeError> {
        self.cursors.validate_commit_tree()?;
        let mut counts = ReleaseCounts::default();
        let mut returned_any_objects = 0usize;
        for frame in &self.frames {
            match frame {
                Frame::Object(object) => {
                    if let Some(cursor) = object.property_name {
                        self.cursors.validate_cursor(cursor)?;
                    }
                    if let Some(dependent) = &object.dependent {
                        for (index, &cursor) in dependent.cursors.iter().enumerate() {
                            if bit_is_set(&dependent.owned_words, index) {
                                self.cursors.validate_cursor(cursor)?;
                                if !bit_is_set(&dependent.active_words, index) {
                                    self.cursors.count_cursor_release(cursor, &mut counts)?;
                                }
                            }
                        }
                    }
                }
                Frame::Obligation(obligation) => {
                    for &cursor in &obligation.cursors {
                        self.cursors.validate_cursor(cursor)?;
                    }
                }
                Frame::Combinator(combinator) => {
                    for (index, &cursor) in combinator.cursors.iter().enumerate() {
                        if bit_is_set(&combinator.owned_words, index) {
                            self.cursors.validate_cursor(cursor)?;
                            if !bit_is_set(&combinator.active_words, index) {
                                self.cursors.count_cursor_release(cursor, &mut counts)?;
                            }
                        }
                    }
                }
                Frame::Negation(frame) => {
                    self.cursors.validate_cursor(frame.syntax)?;
                    if frame.inner_state != NegationInnerState::Released {
                        self.cursors.validate_cursor(frame.inner)?;
                        if frame.inner_state == NegationInnerState::PrunedOwned {
                            self.cursors
                                .count_cursor_release(frame.inner, &mut counts)?;
                        }
                    }
                }
                Frame::Unevaluated(frame) => {
                    self.cursors.validate_cursor(frame.scope)?;
                    self.cursors.validate_cursor(frame.tracker)?;
                    if let Some(candidate) = frame.current_candidate {
                        self.cursors.validate_cursor(candidate.cursor)?;
                    }
                }
                Frame::Array(array) => {
                    if let Some(ItemTracking {
                        contains: ContainsCandidate::Cursor(cursor),
                        ..
                    }) = &array.item
                    {
                        self.cursors.validate_cursor(*cursor)?;
                    }
                }
                Frame::Any(AnyFrame::Object(id)) => self.validate_any_object_id(*id)?,
                Frame::Regular { .. }
                | Frame::Number(_)
                | Frame::String(_)
                | Frame::StringEnum(_)
                | Frame::Any(AnyFrame::Scalar { .. })
                | Frame::Any(AnyFrame::Array { .. })
                | Frame::Any(AnyFrame::StringBody { .. }) => {}
            }
        }
        for (undo_index, undo) in self.undo.iter().enumerate() {
            match undo {
                Undo::PushFrame(frame, _) => {
                    self.count_frame_release(frame, &mut counts)?;
                    if let Frame::Any(AnyFrame::Object(id)) = frame {
                        self.validate_any_object_id(*id)?;
                        for earlier in &self.undo[..undo_index] {
                            if matches!(earlier, Undo::PushFrame(Frame::Any(AnyFrame::Object(old)), _) if old == id)
                            {
                                return Err(self.cursors.error());
                            }
                        }
                        returned_any_objects = returned_any_objects
                            .checked_add(1)
                            .ok_or_else(|| self.cursors.error())?;
                    }
                }
                Undo::FinishPropertyName { cursor, .. } => {
                    self.cursors.count_cursor_release(*cursor, &mut counts)?;
                }
                Undo::RollbackObligationCursor { cursor, .. }
                | Undo::RollbackContainsCursor { cursor, .. }
                | Undo::RollbackPropertyName { cursor, .. }
                | Undo::RollbackNegationCursor { cursor, .. }
                | Undo::RollbackDependentCursor { cursor, .. }
                | Undo::StartPropertyName { cursor, .. } => {
                    self.cursors.validate_cursor(*cursor)?;
                }
                Undo::RollbackUnevaluatedCursor { cursor, .. }
                | Undo::StartUnevaluatedCandidate { cursor, .. } => {
                    self.cursors.validate_cursor(*cursor)?;
                }
                Undo::FinishUnevaluatedCandidate { cursor, .. } => {
                    self.cursors.count_cursor_release(*cursor, &mut counts)?;
                }
                Undo::RestoreItemContainsCandidate {
                    old: ContainsCandidate::Cursor(cursor),
                    ..
                }
                | Undo::FinishArrayItem {
                    old:
                        ItemTracking {
                            contains: ContainsCandidate::Cursor(cursor),
                            ..
                        },
                    ..
                } => {
                    self.cursors.count_cursor_release(*cursor, &mut counts)?;
                }
                _ => {}
            }
        }
        if self
            .cursors
            .free_slots
            .len()
            .saturating_add(counts.free_slots)
            > self.cursors.free_slots.capacity()
            || self
                .cursors
                .free_structured
                .len()
                .saturating_add(counts.free_structured)
                > self.cursors.free_structured.capacity()
        {
            return Err(self.cursors.error());
        }
        if self
            .any_object_free
            .len()
            .checked_add(returned_any_objects)
            .is_none_or(|needed| needed > self.any_object_free.capacity())
        {
            return Err(self.cursors.error());
        }
        Ok(())
    }

    fn validate_any_object_id(&self, id: AnyObjectId) -> Result<(), StructuredRuntimeError> {
        let index = usize::try_from(id.0).map_err(|_| self.cursors.error())?;
        self.any_object_arena
            .get(index)
            .map(|_| ())
            .ok_or_else(|| self.cursors.error())
    }

    /// Validates and tallies every cursor a closed frame will reclaim, by real slot kind.
    fn count_frame_release(
        &self,
        frame: &Frame,
        counts: &mut ReleaseCounts,
    ) -> Result<(), StructuredRuntimeError> {
        match frame {
            Frame::Object(object) => {
                if let Some(cursor) = object.property_name {
                    self.cursors.count_cursor_release(cursor, counts)?;
                }
                if let Some(dependent) = &object.dependent {
                    for (index, &cursor) in dependent.cursors.iter().enumerate() {
                        if bit_is_set(&dependent.owned_words, index) {
                            self.cursors.count_cursor_release(cursor, counts)?;
                        }
                    }
                }
            }
            Frame::Obligation(obligation) => {
                for &cursor in &obligation.cursors {
                    self.cursors.count_cursor_release(cursor, counts)?;
                }
            }
            Frame::Combinator(combinator) => {
                for (index, &cursor) in combinator.cursors.iter().enumerate() {
                    if bit_is_set(&combinator.owned_words, index) {
                        self.cursors.count_cursor_release(cursor, counts)?;
                    }
                }
            }
            Frame::Negation(frame) => {
                self.cursors.count_cursor_release(frame.syntax, counts)?;
                if frame.inner_state != NegationInnerState::Released {
                    self.cursors.count_cursor_release(frame.inner, counts)?;
                }
            }
            Frame::Unevaluated(frame) => {
                self.cursors.count_cursor_release(frame.scope, counts)?;
                self.cursors.count_cursor_release(frame.tracker, counts)?;
                if let Some(candidate) = frame.current_candidate {
                    self.cursors
                        .count_cursor_release(candidate.cursor, counts)?;
                }
            }
            Frame::Array(array) => {
                if let Some(ItemTracking {
                    contains: ContainsCandidate::Cursor(cursor),
                    ..
                }) = &array.item
                {
                    self.cursors.count_cursor_release(*cursor, counts)?;
                }
            }
            Frame::Regular { .. }
            | Frame::Number(_)
            | Frame::String(_)
            | Frame::StringEnum(_)
            | Frame::Any(_) => {}
        }
        Ok(())
    }

    fn apply_commit(&mut self) {
        if self.document_bytes != self.initial_checkpoint.document_bytes {
            self.resettable_to_initial = false;
        }
        let mut reclaim_to: Option<(u32, u32)> = None;
        while let Some(u) = self.undo.pop() {
            match u {
                Undo::SetRootAnnotations { old, .. } => {
                    self.session_memory.release(old.retained_bytes());
                }
                Undo::ReplaceRootAnnotations { old } => {
                    self.session_memory.release(old.retained_bytes());
                }
                Undo::ReplaceEvaluatedSet { old, .. } => {
                    self.session_memory.release(old.retained_bytes());
                }
                Undo::ResetKey { depth, old_key, .. } => {
                    if let Some(Frame::Object(obj)) = self.frames.get_mut(depth) {
                        if old_key.capacity() > obj.spare_key.capacity() {
                            self.session_memory.release(obj.spare_key.capacity());
                            obj.spare_key = old_key;
                        } else {
                            self.session_memory.release(old_key.capacity());
                        }
                    } else {
                        self.session_memory.release(old_key.capacity());
                    }
                }
                Undo::ResetAnyKey { depth, old_key, .. } => {
                    let object = self
                        .frames
                        .get(depth)
                        .and_then(|frame| match frame {
                            Frame::Any(AnyFrame::Object(id)) => usize::try_from(id.0).ok(),
                            _ => None,
                        })
                        .and_then(|index| self.any_object_arena.get_mut(index));
                    let released = if let Some(obj) = object {
                        if old_key.capacity() > obj.spare_key.capacity() {
                            std::mem::replace(&mut obj.spare_key, old_key).capacity()
                        } else {
                            old_key.capacity()
                        }
                    } else {
                        old_key.capacity()
                    };
                    self.session_memory.release(released);
                }
                Undo::PushFrame(Frame::Object(mut obj), _) => {
                    if let Some(cursor) = obj.property_name.take() {
                        self.release_owned_cursor(cursor);
                    }
                    if let Some(dependent) = obj.dependent.take() {
                        for (index, cursor) in dependent.cursors.into_iter().enumerate() {
                            if bit_is_set(&dependent.owned_words, index) {
                                self.release_owned_cursor(cursor);
                            }
                        }
                        self.session_memory.release(dependent.allocation_charge);
                    }
                    reclaim_to = Some(reclaim_to.map_or(obj.arena_start, |r| {
                        (r.0.min(obj.arena_start.0), r.1.min(obj.arena_start.1))
                    }));
                    self.session_memory
                        .release(object_actual_owned_charge(&obj));
                }
                Undo::PushFrame(Frame::Any(AnyFrame::Object(id)), _) => {
                    if let Ok(index) = usize::try_from(id.0) {
                        if let Some(object) = self.any_object_arena.get(index) {
                            let arena_start = object.arena_start;
                            reclaim_to = Some(reclaim_to.map_or(arena_start, |r| {
                                (r.0.min(arena_start.0), r.1.min(arena_start.1))
                            }));
                            self.any_object_free.push(id.0);
                        }
                    }
                }
                Undo::PushFrame(Frame::Obligation(obligation), _) => {
                    self.discard_new_frame(Frame::Obligation(obligation));
                }
                Undo::PushFrame(Frame::Combinator(combinator), _) => {
                    self.discard_new_frame(Frame::Combinator(combinator));
                }
                Undo::PushFrame(Frame::Negation(frame), _) => {
                    self.discard_new_frame(Frame::Negation(frame));
                }
                Undo::PushFrame(Frame::Unevaluated(frame), _) => {
                    self.discard_new_frame(Frame::Unevaluated(frame));
                }
                Undo::PushFrame(Frame::Array(mut array), _) => {
                    if let Some(item) = array.item.take() {
                        self.discard_item(item);
                    }
                    self.session_memory.release(array_owned_charge(&array));
                }
                Undo::FinishPropertyName { cursor, .. } => {
                    self.release_owned_cursor(cursor);
                }
                Undo::RestoreItemContainsCandidate {
                    old: ContainsCandidate::Cursor(cursor),
                    ..
                } => {
                    self.release_owned_cursor(cursor);
                }
                Undo::FinishArrayItem { old, .. } => {
                    self.discard_item(old);
                }
                Undo::FinishUnevaluatedCandidate { cursor, .. } => {
                    self.release_owned_cursor(cursor);
                }
                Undo::RestoreResolvedValue {
                    old: Some(value), ..
                } => {
                    self.session_memory.release(obligation_owned_bytes(&value));
                }
                _ => {}
            }
        }
        if let Some((bytes_mark, spans_mark)) = reclaim_to {
            self.key_arena.bytes.truncate(bytes_mark as usize);
            self.key_arena.spans.truncate(spans_mark as usize);
        }
        // No outer `BuilderMark` survives the drain above, so every active item builder's own
        // undo history is unreachable now too - drop it before it grows for the whole document.
        for frame in &mut self.frames {
            if let Frame::Array(af) = frame {
                if let Some(item) = af.item.as_mut() {
                    if let Some(builder) = item.builder.as_mut() {
                        builder.commit_history();
                    }
                }
            }
        }
        for depth in 0..self.frames.len() {
            let branches = match &self.frames[depth] {
                Frame::Combinator(frame) => frame.cursors.len(),
                _ => continue,
            };
            for index in 0..branches {
                let cursor = match &self.frames[depth] {
                    Frame::Combinator(frame)
                        if bit_is_set(&frame.owned_words, index)
                            && !bit_is_set(&frame.active_words, index) =>
                    {
                        Some(frame.cursors[index])
                    }
                    _ => None,
                };
                if let Some(cursor) = cursor {
                    self.release_owned_cursor(cursor);
                    if let Frame::Combinator(frame) = &mut self.frames[depth] {
                        clear_bit(&mut frame.owned_words, index);
                    }
                }
            }
        }
        for depth in 0..self.frames.len() {
            let dependencies = match &self.frames[depth] {
                Frame::Object(object) => object
                    .dependent
                    .as_ref()
                    .map_or(0, |dependent| dependent.cursors.len()),
                _ => continue,
            };
            for index in 0..dependencies {
                let cursor = match &self.frames[depth] {
                    Frame::Object(object) => object.dependent.as_ref().and_then(|dependent| {
                        (bit_is_set(&dependent.owned_words, index)
                            && !bit_is_set(&dependent.active_words, index))
                        .then_some(dependent.cursors[index])
                    }),
                    _ => None,
                };
                if let Some(cursor) = cursor {
                    self.release_owned_cursor(cursor);
                    if let Some(dependent) = self.object_mut(depth).dependent.as_mut() {
                        clear_bit(&mut dependent.owned_words, index);
                    }
                }
            }
        }
        for depth in 0..self.frames.len() {
            let cursor = match &self.frames[depth] {
                Frame::Negation(frame) if frame.inner_state == NegationInnerState::PrunedOwned => {
                    Some(frame.inner)
                }
                _ => None,
            };
            if let Some(cursor) = cursor {
                self.release_owned_cursor(cursor);
                if let Frame::Negation(frame) = &mut self.frames[depth] {
                    frame.inner_state = NegationInnerState::Released;
                }
            }
        }
    }

    fn apply_commit_tree(&mut self) {
        self.cursors.apply_commit_tree();
        self.transfer_or_release_key_lease();
        self.apply_commit();
    }

    pub(crate) fn discard_prepared_key_leases_tree(&mut self) {
        self.discard_prepared_key_lease();
        self.cursors.discard_prepared_key_leases_tree();
    }

    /// One transaction: `Ok(true)` iff `byte` legally extends the position to a still-viable
    /// state; every other outcome leaves `self` byte-for-byte as it was before this call.
    pub(crate) fn try_push_byte(&mut self, byte: u8) -> Result<bool, StructuredRuntimeError> {
        self.try_push_byte_inner(byte, true)
    }

    /// A speculative mask walk never observes root acceptance before its rollback.
    pub(crate) fn try_push_byte_for_mask(
        &mut self,
        byte: u8,
    ) -> Result<bool, StructuredRuntimeError> {
        self.try_push_byte_inner(byte, false)
    }

    fn try_push_byte_inner(
        &mut self,
        byte: u8,
        update_accepting: bool,
    ) -> Result<bool, StructuredRuntimeError> {
        let mark = self.checkpoint();
        let next_doc_bytes = self.document_bytes.checked_add(1).ok_or_else(|| {
            StructuredRuntimeError::new(
                LimitKind::DocumentBytes,
                usize::MAX,
                self.limits.max_document_bytes,
            )
        })?;
        if next_doc_bytes > self.limits.max_document_bytes {
            return Err(StructuredRuntimeError::new(
                LimitKind::DocumentBytes,
                next_doc_bytes,
                self.limits.max_document_bytes,
            ));
        }
        self.document_bytes = next_doc_bytes;
        self.byte_observed = false;
        let routed = self.route_byte(byte);
        self.byte_observed = false;
        match routed {
            Ok(true) if !self.dead && self.array_futures_satisfiable() => {
                if update_accepting {
                    self.recompute_accepting();
                }
                Ok(true)
            }
            Ok(_) => {
                self.rollback(mark);
                Ok(false)
            }
            Err(e) => {
                self.rollback(mark);
                Err(e)
            }
        }
    }

    /// Rejects bounded arrays that cannot reach `minContains` with remaining slots.
    fn array_futures_satisfiable(&self) -> bool {
        self.frames.iter().all(|frame| {
            let Frame::Array(array) = frame else {
                return true;
            };
            let plan = super::plan::array_plan_of(self.plan.as_ref(), array.node);
            let Some(contains) = &plan.contains else {
                return true;
            };
            if contains
                .max
                .is_some_and(|maximum| array.contains_count > maximum)
            {
                return false;
            }
            let current_can_match = array.item.as_ref().is_some_and(|item| match item.contains {
                ContainsCandidate::Always => true,
                ContainsCandidate::Cursor(cursor) => {
                    self.cursors.is_accepting(cursor) || self.cursors.can_extend_value(cursor)
                }
                ContainsCandidate::None | ContainsCandidate::Dead => false,
            });
            if contains.max == Some(array.contains_count)
                && matches!(
                    array.item.as_ref().map(|item| item.contains),
                    Some(ContainsCandidate::Always)
                )
            {
                return false;
            }
            let RemainingItemCapacity::Bounded(remaining) = plan
                .max_items
                .map_or(RemainingItemCapacity::Unbounded, |maximum| {
                    RemainingItemCapacity::Bounded(maximum.saturating_sub(array.index))
                })
            else {
                return true;
            };
            array
                .contains_count
                .saturating_add(remaining)
                .saturating_add(u32::from(current_can_match))
                >= contains.min
        })
    }

    /// Complete iff every frame closed or the sole frame left is a top-level `Regular` value
    /// that is itself accepting. A dead state (e.g. an unsupported root) is never accepting.
    fn recompute_accepting(&mut self) {
        self.accepting = !self.dead
            && (self.root_closed
                || match &self.frames[..] {
                    [] => true,
                    [Frame::Regular { node, state }] => self.engine_for(*node).is_accepting(*state),
                    [Frame::Number(frame)] => {
                        let NodePlan::Number { engine, constraint } =
                            self.plan.as_ref().node(frame.node)
                        else {
                            unreachable!("number frame references number plan")
                        };
                        engine.is_accepting(frame.state)
                            && number_accepts(frame.accumulator, *constraint)
                    }
                    [Frame::Combinator(frame)] => combinator_frame_accepts(frame, &self.cursors),
                    [Frame::Negation(frame)] => {
                        self.cursors.is_accepting(frame.syntax)
                            && (frame.inner_state != NegationInnerState::Active
                                || !self.cursors.is_accepting(frame.inner))
                    }
                    [Frame::Unevaluated(frame)] => {
                        frame.current_candidate.is_none()
                            && self.cursors.is_accepting(frame.tracker)
                            && self.cursors.is_accepting(frame.scope)
                            && (frame.instance_kind != Some(frame.kind)
                                || (0..frame.location_count).all(|ordinal| {
                                    self.cursors
                                        .accepted_annotation_contains(
                                            frame.scope,
                                            frame.kind,
                                            ordinal,
                                        )
                                        .unwrap_or(false)
                                        || frame.candidates.contains(ordinal)
                                }))
                    }
                    [Frame::Any(AnyFrame::Scalar { phase, .. })] => scalar_is_complete(*phase),
                    _ => false,
                });
    }

    fn engine_for(&self, node: NodeId) -> &crate::automaton::RefEngine {
        match self.plan.as_ref().node(node) {
            NodePlan::Regular(engine) => engine,
            _ => unreachable!("Frame::Regular must only reference a NodePlan::Regular node"),
        }
    }

    fn object_ref(&self, depth: usize) -> &ObjectFrame {
        match &self.frames[depth] {
            Frame::Object(o) => o,
            Frame::Regular { .. }
            | Frame::Number(_)
            | Frame::String(_)
            | Frame::StringEnum(_)
            | Frame::Obligation(_)
            | Frame::Combinator(_)
            | Frame::Negation(_)
            | Frame::Unevaluated(_)
            | Frame::Array(_)
            | Frame::Any(_) => {
                unreachable!("expected an object frame")
            }
        }
    }

    fn object_mut(&mut self, depth: usize) -> &mut ObjectFrame {
        match &mut self.frames[depth] {
            Frame::Object(o) => o,
            Frame::Regular { .. }
            | Frame::Number(_)
            | Frame::String(_)
            | Frame::StringEnum(_)
            | Frame::Obligation(_)
            | Frame::Combinator(_)
            | Frame::Negation(_)
            | Frame::Unevaluated(_)
            | Frame::Array(_)
            | Frame::Any(_) => {
                unreachable!("expected an object frame")
            }
        }
    }

    fn lease_error(&self) -> StructuredRuntimeError {
        StructuredRuntimeError::new(
            LimitKind::SessionBytes,
            usize::MAX,
            self.limits.max_session_bytes,
        )
    }

    fn key_owner(&self, depth: usize) -> Result<KeyOwner, StructuredRuntimeError> {
        let key_generation = match self.frames.get(depth) {
            Some(Frame::Object(object)) => object.key_generation,
            Some(Frame::Any(AnyFrame::Object(id))) => self
                .any_object_arena
                .get(id.0 as usize)
                .map(|object| object.key_generation)
                .ok_or_else(|| self.lease_error())?,
            _ => return Err(self.lease_error()),
        };
        Ok(KeyOwner {
            state_identity: self.state_identity,
            frame_depth: u32::try_from(depth).map_err(|_| self.lease_error())?,
            key_generation,
        })
    }

    fn committed_key(&self, depth: usize) -> Option<&str> {
        match self.frames.get(depth)? {
            Frame::Object(object) => Some(&object.key),
            Frame::Any(AnyFrame::Object(id)) => self
                .any_object_arena
                .get(id.0 as usize)
                .map(|object| object.key.as_str()),
            _ => None,
        }
    }

    fn committed_key_capacity(&self, depth: usize) -> Option<usize> {
        match self.frames.get(depth)? {
            Frame::Object(object) => Some(object.key.capacity()),
            Frame::Any(AnyFrame::Object(id)) => self
                .any_object_arena
                .get(id.0 as usize)
                .map(|object| object.key.capacity()),
            _ => None,
        }
    }

    fn prepared_key_lease(&self, depth: usize) -> Option<&KeyCapacityLease> {
        let owner = self.key_owner(depth).ok()?;
        (self.key_lease.phase == KeyLeasePhase::Prepared && self.key_lease.owner == Some(owner))
            .then_some(&self.key_lease)
    }

    fn key_lease_growth_request(&self, depth: usize, max_bytes: usize) -> Option<GrowthRequest> {
        let required = self.committed_key(depth)?.len().checked_add(max_bytes)?;
        Some(GrowthRequest {
            len: self.key_lease.buffer.len(),
            capacity: self.key_lease.buffer.capacity(),
            additional: required.saturating_sub(self.key_lease.buffer.len()),
            item_size: 1,
        })
    }

    fn prepare_key_lease(
        &mut self,
        depth: usize,
        max_bytes: usize,
    ) -> Result<(), StructuredRuntimeError> {
        let owner = self.key_owner(depth)?;
        let required = self
            .committed_key(depth)
            .ok_or_else(|| self.lease_error())?
            .len()
            .checked_add(max_bytes)
            .ok_or_else(|| self.lease_error())?;
        if self.key_lease.phase == KeyLeasePhase::TransferredToCommit
            && self.key_lease.owner == Some(owner)
            && self
                .committed_key_capacity(depth)
                .is_some_and(|capacity| capacity >= required)
        {
            return Ok(());
        }
        if self.key_lease.owner.is_some_and(|old| old != owner) {
            self.release_key_lease();
        }
        let generation = self
            .key_lease
            .generation
            .checked_add(1)
            .ok_or_else(|| self.lease_error())?;
        let request = self
            .key_lease_growth_request(depth, max_bytes)
            .ok_or_else(|| self.lease_error())?;
        let cap = self.local_session_budget()?;
        let result = grow_bounded(
            &mut self.session_memory,
            cap,
            LimitKind::SessionBytes,
            request,
            |n| {
                self.key_lease
                    .buffer
                    .try_reserve_exact(n)
                    .map(|_| self.key_lease.buffer.capacity())
                    .map_err(|_| ())
            },
        );
        if let Err(error) = result {
            self.release_key_lease();
            return Err(error);
        }
        self.key_lease.buffer.clear();
        let StructuredState {
            frames,
            any_object_arena,
            key_lease,
            ..
        } = &mut *self;
        match &frames[depth] {
            Frame::Object(object) => key_lease.buffer.push_str(&object.key),
            Frame::Any(AnyFrame::Object(id)) => {
                key_lease
                    .buffer
                    .push_str(&any_object_arena[id.0 as usize].key);
            }
            _ => unreachable!("lease owner checked object frame"),
        }
        self.key_lease.owner = Some(owner);
        self.key_lease.generation = generation;
        self.key_lease.phase = KeyLeasePhase::Prepared;
        Ok(())
    }

    fn release_key_lease(&mut self) {
        let capacity = self.key_lease.buffer.capacity();
        self.key_lease.buffer = String::new();
        self.key_lease.owner = None;
        self.key_lease.phase = KeyLeasePhase::Free;
        self.session_memory.release(capacity);
    }

    fn discard_prepared_key_lease(&mut self) {
        if self.key_lease.phase == KeyLeasePhase::Prepared {
            self.release_key_lease();
        }
    }

    /// Makes the capacity proven by the last mask the committed key capacity without allocating.
    /// If the owner closed or a new key incarnation replaced it, the stale lease is released.
    fn transfer_or_release_key_lease(&mut self) {
        if self.key_lease.phase == KeyLeasePhase::Free {
            return;
        }
        let Some(owner) = self.key_lease.owner else {
            debug_assert!(false, "prepared key lease has no owner");
            self.release_key_lease();
            return;
        };
        let depth = owner.frame_depth as usize;
        let matches = self.state_identity == owner.state_identity
            && self.key_owner(depth).is_ok_and(|current| current == owner);
        if !matches {
            self.release_key_lease();
            return;
        }
        let (key, phase) = match &mut self.frames[depth] {
            Frame::Object(object) => (&mut object.key, object.phase),
            Frame::Any(AnyFrame::Object(id)) => {
                let object = &mut self.any_object_arena[id.0 as usize];
                (&mut object.key, object.phase)
            }
            _ => unreachable!("owner validation checked object frame"),
        };
        if self.key_lease.phase == KeyLeasePhase::Prepared {
            std::mem::swap(key, &mut self.key_lease.buffer);
            self.key_lease.phase = KeyLeasePhase::TransferredToCommit;
        }

        // Release speculative key capacity once `:` resolves the property.
        if !matches!(phase, ObjectPhase::InKey | ObjectPhase::AfterKeyBeforeColon) {
            let committed_capacity = std::mem::take(key).capacity();
            let displaced_capacity = std::mem::take(&mut self.key_lease.buffer).capacity();
            self.key_lease.owner = None;
            self.key_lease.phase = KeyLeasePhase::Free;
            self.session_memory
                .release(committed_capacity.saturating_add(displaced_capacity));
        }
    }

    /// Keeps every quote-closure consumer on one decoded-key view.
    fn candidate_key(&self, depth: usize) -> &str {
        self.prepared_key_lease(depth).map_or_else(
            || {
                self.committed_key(depth)
                    .expect("candidate key belongs to an object frame")
            },
            |lease| lease.buffer.as_str(),
        )
    }

    fn candidate_key_len(&self, depth: usize) -> usize {
        self.candidate_key(depth).len()
    }

    fn array_ref(&self, depth: usize) -> &ArrayFrame {
        match &self.frames[depth] {
            Frame::Array(a) => a,
            Frame::Regular { .. }
            | Frame::Number(_)
            | Frame::String(_)
            | Frame::StringEnum(_)
            | Frame::Obligation(_)
            | Frame::Combinator(_)
            | Frame::Negation(_)
            | Frame::Unevaluated(_)
            | Frame::Object(_)
            | Frame::Any(_) => {
                unreachable!("expected an array frame")
            }
        }
    }

    fn array_mut(&mut self, depth: usize) -> &mut ArrayFrame {
        match &mut self.frames[depth] {
            Frame::Array(a) => a,
            Frame::Regular { .. }
            | Frame::Number(_)
            | Frame::String(_)
            | Frame::StringEnum(_)
            | Frame::Obligation(_)
            | Frame::Combinator(_)
            | Frame::Negation(_)
            | Frame::Unevaluated(_)
            | Frame::Object(_)
            | Frame::Any(_) => {
                unreachable!("expected an array frame")
            }
        }
    }

    fn any_mut(&mut self, depth: usize) -> &mut AnyFrame {
        match &mut self.frames[depth] {
            Frame::Any(a) => a,
            Frame::Regular { .. }
            | Frame::Number(_)
            | Frame::String(_)
            | Frame::StringEnum(_)
            | Frame::Obligation(_)
            | Frame::Combinator(_)
            | Frame::Negation(_)
            | Frame::Unevaluated(_)
            | Frame::Object(_)
            | Frame::Array(_) => {
                unreachable!("expected an any-value frame")
            }
        }
    }

    fn any_object_ref(&self, depth: usize) -> &AnyObjectState {
        match &self.frames[depth] {
            Frame::Any(AnyFrame::Object(id)) => &self.any_object_arena[id.0 as usize],
            _ => unreachable!("expected an any-object frame"),
        }
    }

    fn any_object_mut(&mut self, depth: usize) -> &mut AnyObjectState {
        let id = match &self.frames[depth] {
            Frame::Any(AnyFrame::Object(id)) => *id,
            _ => unreachable!("expected an any-object frame"),
        };
        &mut self.any_object_arena[id.0 as usize]
    }

    /// Allocates a fresh `AnyObjectState`, reusing a `commit_checkpoint`-released slot when one
    /// exists so a run of small objects costs at most one real arena growth.
    fn alloc_any_object(&mut self) -> Result<AnyObjectId, StructuredRuntimeError> {
        let arena_start = self.key_arena.mark();
        if let Some(idx) = self.any_object_free.pop() {
            let next_generation = self.any_object_arena[idx as usize]
                .key_generation
                .checked_add(1)
                .ok_or_else(|| self.lease_error())?;
            let obj = &mut self.any_object_arena[idx as usize];
            obj.phase = ObjectPhase::BeforeFirstKeyOrClose;
            obj.key.clear();
            obj.key_generation = next_generation;
            obj.key_decoder = JsonStringDecoder::new();
            obj.seen_dynamic.clear();
            obj.arena_start = arena_start;
            obj.properties = 0;
            return Ok(AnyObjectId(idx));
        }
        let cap = self.local_session_budget()?;
        let idx = u32::try_from(self.any_object_arena.len())
            .map_err(|_| StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, cap))?;
        let StructuredState {
            any_object_arena,
            any_object_free,
            session_memory,
            ..
        } = self;
        let req = GrowthRequest {
            len: any_object_arena.len(),
            capacity: any_object_arena.capacity(),
            additional: 1,
            item_size: size_of::<AnyObjectState>(),
        };
        grow_bounded(session_memory, cap, LimitKind::SessionBytes, req, |n| {
            any_object_arena
                .try_reserve_exact(n)
                .map(|_| any_object_arena.capacity())
                .map_err(|_| ())
        })?;
        // Match free-list capacity to the arena so the infallible commit can return slots without allocating.
        if any_object_free.capacity() < any_object_arena.capacity() {
            let additional = any_object_arena.capacity() - any_object_free.len();
            let req = GrowthRequest {
                len: any_object_free.len(),
                capacity: any_object_free.capacity(),
                additional,
                item_size: size_of::<u32>(),
            };
            grow_bounded(session_memory, cap, LimitKind::SessionBytes, req, |n| {
                any_object_free
                    .try_reserve_exact(n)
                    .map(|_| any_object_free.capacity())
                    .map_err(|_| ())
            })?;
        }
        self.any_object_arena.push(AnyObjectState {
            phase: ObjectPhase::BeforeFirstKeyOrClose,
            key: String::new(),
            key_generation: 0,
            spare_key: String::new(),
            key_decoder: JsonStringDecoder::new(),
            seen_dynamic: FxHashMap::default(),
            arena_start,
            properties: 0,
        });
        Ok(AnyObjectId(idx))
    }

    /// Reserves undo entries and charges only actual capacity growth.
    /// Called before mutation so `push_undo` cannot fail partway.
    fn reserve_undo(&mut self, additional: usize) -> Result<(), StructuredRuntimeError> {
        let next = self.undo.len().checked_add(additional).ok_or_else(|| {
            StructuredRuntimeError::new(
                LimitKind::UndoEntries,
                usize::MAX,
                self.limits.max_undo_entries,
            )
        })?;
        if next > self.limits.max_undo_entries {
            return Err(StructuredRuntimeError::new(
                LimitKind::UndoEntries,
                next,
                self.limits.max_undo_entries,
            ));
        }
        let cap = self.local_session_budget()?;
        let StructuredState {
            undo,
            session_memory,
            ..
        } = self;
        let req = GrowthRequest {
            len: undo.len(),
            capacity: undo.capacity(),
            additional,
            item_size: size_of::<Undo>(),
        };
        grow_bounded(session_memory, cap, LimitKind::SessionBytes, req, |n| {
            undo.try_reserve_exact(n)
                .map(|_| undo.capacity())
                .map_err(|_| ())
        })
    }

    /// Records one undo entry. Capacity must already be reserved by a prior `reserve_undo` call.
    #[cfg_attr(debug_assertions, track_caller)]
    fn push_undo(&mut self, undo: Undo) -> Result<(), StructuredRuntimeError> {
        debug_assert!(
            self.undo.len() < self.undo.capacity(),
            "push_undo would allocate without accounting at {}",
            std::panic::Location::caller()
        );
        self.undo.push(undo);
        Ok(())
    }

    /// Rejects a growth request BEFORE any real reservation when even its minimum size alone
    /// would exceed the session budget.
    fn session_would_fit(&self, additional: usize) -> bool {
        self.local_session_budget()
            .is_ok_and(|cap| self.session_memory.would_fit(additional, cap))
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.session_memory
            .live()
            .saturating_add(self.cursors.retained_bytes())
    }

    pub(crate) fn validate_retained_budget(&self) -> Result<(), StructuredRuntimeError> {
        let retained = self.retained_bytes();
        if retained > self.limits.max_session_bytes {
            return Err(StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                retained,
                self.limits.max_session_bytes,
            ));
        }
        Ok(())
    }

    #[cfg(all(test, feature = "bench-internals"))]
    pub(crate) fn key_lease_debug(&self) -> (u8, usize, usize) {
        (
            match self.key_lease.phase {
                KeyLeasePhase::Free => 0,
                KeyLeasePhase::Prepared => 1,
                KeyLeasePhase::TransferredToCommit => 2,
            },
            self.key_lease.buffer.capacity(),
            self.recomputed_retained_bytes(),
        )
    }

    #[cfg(test)]
    pub(crate) fn key_lease_tree_debug(&self) -> (usize, usize) {
        let own = usize::from(self.key_lease.phase != KeyLeasePhase::Free);
        let (nested, capacity) = self.cursors.key_lease_tree_debug();
        (
            own.saturating_add(nested),
            self.key_lease.buffer.capacity().saturating_add(capacity),
        )
    }

    #[cfg(test)]
    pub(crate) fn set_unique_canonical_limit_for_test(&mut self, limit: usize) {
        self.limits.max_unique_canonical_bytes = limit;
    }

    #[cfg(test)]
    pub(crate) fn set_session_limit_for_test(&mut self, limit: usize) {
        self.limits.max_session_bytes = limit;
    }

    #[cfg(all(test, feature = "bench-internals"))]
    pub(crate) fn set_document_limit_for_test(&mut self, limit: usize) {
        self.limits.max_document_bytes = limit;
    }

    #[cfg(all(test, feature = "bench-internals"))]
    pub(crate) fn set_key_limit_for_test(&mut self, limit: usize) {
        self.limits.max_key_bytes = limit;
    }

    #[cfg(all(test, feature = "bench-internals"))]
    pub(crate) fn set_string_limit_for_test(&mut self, limit: usize) {
        self.limits.max_string_bytes = limit;
    }

    #[cfg(all(test, feature = "bench-internals"))]
    pub(crate) fn set_undo_limit_for_test(&mut self, limit: usize) {
        self.limits.max_undo_entries = limit;
    }

    pub(crate) fn peak_session_bytes(&self) -> usize {
        self.session_memory.peak().max(self.retained_bytes())
    }

    /// Bounds traversal breadth across active and structured cursors.
    /// The separate visited stack remains the recursion and cycle guard.
    pub(crate) fn certificate_fuel(&self) -> Option<usize> {
        self._validator_ledger
            .live
            .load(Ordering::Relaxed)
            .checked_mul(2)?
            .checked_add(1)
    }

    #[cfg(feature = "bench-internals")]
    pub(crate) fn active_validator_count_for_bench(&self) -> usize {
        self._validator_ledger.live.load(Ordering::Relaxed)
    }

    #[cfg(feature = "bench-internals")]
    pub(crate) fn frame_profile_for_bench(&self) -> (usize, usize) {
        let wrappers = self
            .frames
            .iter()
            .filter(|frame| {
                matches!(
                    frame,
                    Frame::Obligation(_)
                        | Frame::Combinator(_)
                        | Frame::Negation(_)
                        | Frame::Unevaluated(_)
                )
            })
            .count();
        (self.frames.len(), wrappers)
    }

    #[cfg(feature = "bench-internals")]
    pub(crate) fn retention_breakdown_for_bench(&self) -> (usize, usize) {
        (
            self.retained_bytes(),
            self.undo.capacity().saturating_mul(size_of::<Undo>()),
        )
    }

    #[cfg(feature = "bench-internals")]
    pub(crate) fn reset_dynamic_work_metrics(&self) {
        if let Ok(scope) = self.current_scope() {
            scope.arena.dynamic_resolutions.store(0, Ordering::Relaxed);
            scope.arena.dynamic_scope_steps.store(0, Ordering::Relaxed);
        }
    }

    #[cfg(feature = "bench-internals")]
    pub(crate) fn dynamic_work_metrics(&self) -> (u64, u64) {
        self.current_scope().map_or((0, 0), |scope| {
            (
                scope.arena.dynamic_resolutions.load(Ordering::Relaxed),
                scope.arena.dynamic_scope_steps.load(Ordering::Relaxed),
            )
        })
    }

    fn local_session_budget(&self) -> Result<usize, StructuredRuntimeError> {
        self.limits
            .max_session_bytes
            .checked_sub(self.cursors.retained_bytes())
            .ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::SessionBytes,
                    usize::MAX,
                    self.limits.max_session_bytes,
                )
            })
    }

    /// Charges a delta already validated by `session_would_fit`; an overshoot here means the
    /// allocator granted more than the checked request, and the session goes terminal.
    fn charge_session_bytes_after_precheck(
        &mut self,
        delta: usize,
    ) -> Result<(), StructuredRuntimeError> {
        let cap = self.local_session_budget()?;
        self.session_memory
            .charge_after_precheck(delta, cap, LimitKind::SessionBytes)
    }

    fn current_scope(&self) -> Result<&ScopeLease, StructuredRuntimeError> {
        self.frame_scopes.last().ok_or_else(|| self.cursors.error())
    }

    fn start_cursor(&mut self, node: NodeId) -> Result<CursorId, StructuredRuntimeError> {
        let depth_offset = self
            .depth_offset
            .checked_add(self.frames.len())
            .ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::RecursionDepth,
                    usize::MAX,
                    self.limits.max_depth as usize,
                )
            })?;
        self.start_cursor_at_depth(node, depth_offset)
    }

    fn start_cursor_at_depth(
        &mut self,
        node: NodeId,
        depth_offset: usize,
    ) -> Result<CursorId, StructuredRuntimeError> {
        let (node, scope) =
            resolve_runtime_node(self.plan.as_ref(), node, self.current_scope()?.clone())?;
        let floor = self
            .cursors
            .start_growth_floor_resolved(node)
            .ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::SessionBytes,
                    usize::MAX,
                    self.limits.max_session_bytes,
                )
            })?;
        let observed = self
            .retained_bytes()
            .checked_add(floor)
            .unwrap_or(usize::MAX);
        if observed > self.limits.max_session_bytes {
            return Err(StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                observed,
                self.limits.max_session_bytes,
            ));
        }
        let pool_mark = self.cursors.pool_mark()?;
        let cursor = self
            .cursors
            .start_resolved_in_context(node, depth_offset, scope)?;
        if self.retained_bytes() > self.limits.max_session_bytes {
            self.cursors.rollback_unpublished(cursor, pool_mark)?;
            return Err(StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                self.retained_bytes(),
                self.limits.max_session_bytes,
            ));
        }
        Ok(cursor)
    }

    fn start_any_cursor_at_depth(
        &mut self,
        depth_offset: usize,
    ) -> Result<CursorId, StructuredRuntimeError> {
        let scope = self.current_scope()?.clone();
        let floor = self.cursors.start_any_growth_floor().ok_or_else(|| {
            StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                usize::MAX,
                self.limits.max_session_bytes,
            )
        })?;
        let observed = self
            .retained_bytes()
            .checked_add(floor)
            .unwrap_or(usize::MAX);
        if observed > self.limits.max_session_bytes {
            return Err(StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                observed,
                self.limits.max_session_bytes,
            ));
        }
        let pool_mark = self.cursors.pool_mark()?;
        let cursor = self.cursors.start_any_in_context(depth_offset, scope)?;
        if self.retained_bytes() > self.limits.max_session_bytes {
            self.cursors.rollback_unpublished(cursor, pool_mark)?;
            return Err(StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                self.retained_bytes(),
                self.limits.max_session_bytes,
            ));
        }
        Ok(cursor)
    }

    fn push_cursor_byte(
        &mut self,
        cursor: CursorId,
        byte: u8,
    ) -> Result<CursorStep, StructuredRuntimeError> {
        let budget = self
            .limits
            .max_session_bytes
            .checked_sub(self.session_memory.live())
            .ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::SessionBytes,
                    usize::MAX,
                    self.limits.max_session_bytes,
                )
            })?;
        self.cursors.set_cursor_budget(cursor, budget)?;
        let mark = self.cursors.checkpoint(cursor)?;
        let step = self.cursors.try_push_byte(cursor, byte)?;
        if self.retained_bytes() > self.limits.max_session_bytes {
            self.cursors.rollback(cursor, mark)?;
            return Err(StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                self.retained_bytes(),
                self.limits.max_session_bytes,
            ));
        }
        Ok(step)
    }

    fn set_object_phase(
        &mut self,
        depth: usize,
        new: ObjectPhase,
    ) -> Result<(), StructuredRuntimeError> {
        self.reserve_undo(1)?;
        let old = std::mem::replace(&mut self.object_mut(depth).phase, new);
        self.push_undo(Undo::SetObjectPhase { depth, old })
    }

    fn start_key(&mut self, depth: usize) -> Result<bool, StructuredRuntimeError> {
        let property_names =
            super::plan::object_plan_of(self.plan.as_ref(), self.object_ref(depth).node)
                .property_names;
        let property_name = if let Some(node) = property_names {
            let cursor = self.start_cursor(node)?;
            let step = match self.push_cursor_byte(cursor, b'"') {
                Ok(step) => step,
                Err(error) => {
                    self.cursors.release(cursor)?;
                    return Err(error);
                }
            };
            match step {
                CursorStep::Dead => {
                    self.cursors.release(cursor)?;
                    return Ok(false);
                }
                CursorStep::Alive | CursorStep::Complete => Some(cursor),
            }
        } else {
            None
        };
        if let Err(error) = self.reserve_undo(1 + usize::from(property_name.is_some())) {
            if let Some(cursor) = property_name {
                self.cursors.release(cursor)?;
            }
            return Err(error);
        }
        let next_generation = self
            .object_ref(depth)
            .key_generation
            .checked_add(1)
            .ok_or_else(|| self.lease_error())?;
        let obj = self.object_mut(depth);
        let mut new_key = std::mem::take(&mut obj.spare_key);
        new_key.clear();
        let old_key = std::mem::replace(&mut obj.key, new_key);
        let old_generation = std::mem::replace(&mut obj.key_generation, next_generation);
        let old_decoder = std::mem::replace(&mut obj.key_decoder, JsonStringDecoder::new());
        self.push_undo(Undo::ResetKey {
            depth,
            old_key,
            old_generation,
            old_decoder,
        })?;
        if let Some(cursor) = property_name {
            self.object_mut(depth).property_name = Some(cursor);
            self.push_undo(Undo::StartPropertyName { depth, cursor })?;
        }
        self.set_object_phase(depth, ObjectPhase::InKey)?;
        Ok(true)
    }

    /// Pushes a nested value frame, charging its depth and memory against the session budget.
    /// `Ok(false)` means the target node is `Unsupported` (a plan shape gap, not a resource cap).
    fn push_child(&mut self, node: NodeId) -> Result<bool, StructuredRuntimeError> {
        let (node, scope) =
            resolve_runtime_node(self.plan.as_ref(), node, self.current_scope()?.clone())?;
        let next_depth = self
            .depth_offset
            .checked_add(self.frames.len())
            .and_then(|depth| depth.checked_add(1))
            .ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::RecursionDepth,
                    usize::MAX,
                    self.limits.max_depth as usize,
                )
            })?;
        if next_depth > self.limits.max_depth as usize {
            return Err(StructuredRuntimeError::new(
                LimitKind::RecursionDepth,
                next_depth,
                self.limits.max_depth as usize,
            ));
        }
        let cap = self.local_session_budget()?;
        self.reserve_undo(1)?;
        let StructuredState {
            frames,
            session_memory,
            ..
        } = self;
        let req = GrowthRequest {
            len: frames.len(),
            capacity: frames.capacity(),
            additional: 1,
            item_size: size_of::<Frame>(),
        };
        grow_bounded(session_memory, cap, LimitKind::SessionBytes, req, |n| {
            frames
                .try_reserve_exact(n)
                .map(|_| frames.capacity())
                .map_err(|_| ())
        })?;
        let req = GrowthRequest {
            len: self.frame_scopes.len(),
            capacity: self.frame_scopes.capacity(),
            additional: 1,
            item_size: size_of::<ScopeLease>(),
        };
        grow_bounded(
            &mut self.session_memory,
            cap,
            LimitKind::SessionBytes,
            req,
            |n| {
                self.frame_scopes
                    .try_reserve_exact(n)
                    .map(|_| self.frame_scopes.capacity())
                    .map_err(|_| ())
            },
        )?;
        let arena_start = self.key_arena.mark();
        self.reserve_object_frame_free(node)?;
        self.reserve_combinator_frame_free(node)?;
        self.frame_scopes.push(scope);
        let prepared = match self.prepare_resolved_child_frame(node, arena_start) {
            Ok(Some(prepared)) => prepared,
            Ok(None) => {
                let _ = self.frame_scopes.pop();
                return Ok(false);
            }
            Err(error) => {
                let _ = self.frame_scopes.pop();
                return Err(error);
            }
        };
        if !self.session_would_fit(prepared.charge) {
            let _ = self.frame_scopes.pop();
            return Err(StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                usize::MAX,
                cap,
            ));
        }
        if let Err(error) = self.charge_session_bytes_after_precheck(prepared.charge) {
            let _ = self.frame_scopes.pop();
            return Err(error);
        }
        self.frames.push(prepared.frame);
        self.push_undo(Undo::PopFrame)?;
        Ok(true)
    }

    fn reserve_object_frame_free(&mut self, node: NodeId) -> Result<(), StructuredRuntimeError> {
        let NodePlan::Object(plan) = self.plan.as_ref().node(node) else {
            return Ok(());
        };
        if !plan.dependent_schemas.is_empty()
            || plan
                .dependent_required
                .as_ref()
                .is_some_and(|presence| presence.name_count > 64)
            || self
                .object_frame_free
                .iter()
                .any(|frame| frame.node == node)
        {
            return Ok(());
        }
        let cap = self.local_session_budget()?;
        let req = GrowthRequest {
            len: self.object_frame_free.len(),
            capacity: self.object_frame_free.capacity(),
            additional: 1,
            item_size: size_of::<ObjectFrame>(),
        };
        grow_bounded(
            &mut self.session_memory,
            cap,
            LimitKind::SessionBytes,
            req,
            |n| {
                self.object_frame_free
                    .try_reserve_exact(n)
                    .map(|_| self.object_frame_free.capacity())
                    .map_err(|_| ())
            },
        )
    }

    fn reserve_combinator_frame_free(
        &mut self,
        node: NodeId,
    ) -> Result<(), StructuredRuntimeError> {
        if !matches!(self.plan.as_ref().node(node), NodePlan::Combinator(_))
            || self
                .combinator_frame_free
                .iter()
                .any(|frame| frame.node == node)
        {
            return Ok(());
        }
        let cap = self.local_session_budget()?;
        let req = GrowthRequest {
            len: self.combinator_frame_free.len(),
            capacity: self.combinator_frame_free.capacity(),
            additional: 1,
            item_size: size_of::<CombinatorFrame>(),
        };
        grow_bounded(
            &mut self.session_memory,
            cap,
            LimitKind::SessionBytes,
            req,
            |n| {
                self.combinator_frame_free
                    .try_reserve_exact(n)
                    .map(|_| self.combinator_frame_free.capacity())
                    .map_err(|_| ())
            },
        )
    }

    fn prepare_resolved_child_frame(
        &mut self,
        node: NodeId,
        arena_start: (u32, u32),
    ) -> Result<Option<PreparedFrame>, StructuredRuntimeError> {
        if matches!(self.plan.as_ref().node(node), NodePlan::Combinator(_)) {
            return self.prepare_combinator_frame(node).map(Some);
        }
        if matches!(self.plan.as_ref().node(node), NodePlan::Negation { .. }) {
            return self.prepare_negation_frame(node).map(Some);
        }
        if matches!(self.plan.as_ref().node(node), NodePlan::Unevaluated(_)) {
            return self.prepare_unevaluated_frame(node).map(Some);
        }
        if matches!(self.plan.as_ref().node(node), NodePlan::Object(plan) if !plan.dependent_schemas.is_empty())
        {
            return self
                .prepare_dependent_object_frame(node, arena_start)
                .map(Some);
        }
        let Some(index) = self
            .object_frame_free
            .iter()
            .position(|frame| frame.node == node)
        else {
            return prepare_frame(self.plan.as_ref(), node, arena_start);
        };
        let mut object = self.object_frame_free.swap_remove(index);
        let NodePlan::Object(plan) = self.plan.as_ref().node(node) else {
            return Err(reference_resolution_error(self.plan.as_ref()));
        };
        object.phase = ObjectPhase::Start;
        object.key.clear();
        object.spare_key.clear();
        object.key_generation = 0;
        object.key_decoder = JsonStringDecoder::new();
        object.property_name = None;
        object.resolved_value = None;
        object
            .missing_required
            .copy_from_slice(&plan.required_template);
        object.seen_known.fill(0);
        object.dependent_required_seen.reset();
        object.arena_start = arena_start;
        object.property_count = 0;
        object.evaluated.reset();
        Ok(Some(PreparedFrame {
            frame: Frame::Object(object),
            charge: 0,
        }))
    }

    fn prepare_combinator_frame(
        &mut self,
        node: NodeId,
    ) -> Result<PreparedFrame, StructuredRuntimeError> {
        let (kind, branch_count) = match self.plan.as_ref().node(node) {
            NodePlan::Combinator(plan) => (plan.kind, plan.branches.len()),
            _ => return Err(reference_resolution_error(self.plan.as_ref())),
        };
        self._validator_ledger.ensure_available(branch_count)?;
        let word_count = branch_count.div_ceil(BITSET_WORD_BITS);
        let limit = self.limits.max_session_bytes;
        let pooled = self
            .combinator_frame_free
            .iter()
            .position(|frame| frame.node == node)
            .map(|index| self.combinator_frame_free.swap_remove(index));
        let (mut cursors, mut active_words, mut owned_words, mut marks, allocation_charge, reused) =
            if let Some(frame) = pooled {
                (
                    frame.cursors,
                    frame.active_words,
                    frame.owned_words,
                    frame.marks,
                    frame.allocation_charge,
                    true,
                )
            } else {
                let mut cursors = Vec::new();
                cursors.try_reserve_exact(branch_count).map_err(|_| {
                    StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, limit)
                })?;
                let mut active_words = Vec::new();
                active_words.try_reserve_exact(word_count).map_err(|_| {
                    StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, limit)
                })?;
                let mut owned_words = Vec::new();
                owned_words.try_reserve_exact(word_count).map_err(|_| {
                    StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, limit)
                })?;
                let mut marks = Vec::new();
                marks.try_reserve_exact(branch_count).map_err(|_| {
                    StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, limit)
                })?;
                let charge = cursors
                    .capacity()
                    .checked_mul(size_of::<CursorId>())
                    .and_then(|bytes| {
                        bytes.checked_add(active_words.capacity().checked_mul(size_of::<u64>())?)
                    })
                    .and_then(|bytes| {
                        bytes.checked_add(owned_words.capacity().checked_mul(size_of::<u64>())?)
                    })
                    .and_then(|bytes| {
                        bytes.checked_add(marks.capacity().checked_mul(size_of::<BranchMark>())?)
                    })
                    .ok_or_else(|| {
                        StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, limit)
                    })?;
                (cursors, active_words, owned_words, marks, charge, false)
            };
        cursors.clear();
        active_words.clear();
        owned_words.clear();
        marks.clear();
        active_words.resize(word_count, u64::MAX);
        owned_words.resize(word_count, u64::MAX);
        if branch_count % BITSET_WORD_BITS != 0 {
            let valid = branch_count % BITSET_WORD_BITS;
            let mask = (1u64 << valid) - 1;
            if let Some(last) = active_words.last_mut() {
                *last = mask;
            }
            if let Some(last) = owned_words.last_mut() {
                *last = mask;
            }
        }
        let branch_depth = self
            .depth_offset
            .checked_add(self.frames.len())
            .and_then(|depth| depth.checked_add(1))
            .ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::RecursionDepth,
                    usize::MAX,
                    self.limits.max_depth as usize,
                )
            })?;
        if !reused && !self.session_would_fit(allocation_charge) {
            return Err(StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                self.retained_bytes().saturating_add(allocation_charge),
                limit,
            ));
        }
        if !reused {
            self.charge_session_bytes_after_precheck(allocation_charge)?;
        }
        for index in 0..branch_count {
            let branch = match self.plan.as_ref().node(node) {
                NodePlan::Combinator(plan) => plan.branches[index],
                _ => return Err(reference_resolution_error(self.plan.as_ref())),
            };
            match self.start_cursor_at_depth(branch, branch_depth) {
                Ok(cursor) => cursors.push(cursor),
                Err(error) => {
                    for cursor in cursors.drain(..).rev() {
                        self.release_owned_cursor(cursor);
                    }
                    if reused {
                        active_words.clear();
                        owned_words.clear();
                        marks.clear();
                        self.combinator_frame_free.push(CombinatorFrame {
                            node,
                            kind,
                            cursors,
                            active_words,
                            owned_words,
                            marks,
                            allocation_charge,
                        });
                    } else {
                        self.session_memory.release(allocation_charge);
                    }
                    return Err(error);
                }
            }
        }
        Ok(PreparedFrame {
            frame: Frame::Combinator(CombinatorFrame {
                node,
                kind,
                cursors,
                active_words,
                owned_words,
                marks,
                allocation_charge,
            }),
            charge: 0,
        })
    }

    fn prepare_dependent_object_frame(
        &mut self,
        node: NodeId,
        arena_start: (u32, u32),
    ) -> Result<PreparedFrame, StructuredRuntimeError> {
        let count = match self.plan.as_ref().node(node) {
            NodePlan::Object(plan) => plan.dependent_schemas.len(),
            _ => return Err(reference_resolution_error(self.plan.as_ref())),
        };
        self._validator_ledger.ensure_available(count)?;
        let word_count = count.div_ceil(BITSET_WORD_BITS);
        let limit = self.limits.max_session_bytes;
        let mut cursors = Vec::new();
        cursors
            .try_reserve_exact(count)
            .map_err(|_| StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, limit))?;
        let mut active_words = Vec::new();
        active_words
            .try_reserve_exact(word_count)
            .map_err(|_| StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, limit))?;
        active_words.resize(word_count, u64::MAX);
        let mut owned_words = Vec::new();
        owned_words
            .try_reserve_exact(word_count)
            .map_err(|_| StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, limit))?;
        owned_words.resize(word_count, u64::MAX);
        let mut triggered_words = Vec::new();
        triggered_words
            .try_reserve_exact(word_count)
            .map_err(|_| StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, limit))?;
        triggered_words.resize(word_count, 0);
        let mut marks = Vec::new();
        marks
            .try_reserve_exact(count)
            .map_err(|_| StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, limit))?;
        if count % BITSET_WORD_BITS != 0 {
            let mask = (1u64 << (count % BITSET_WORD_BITS)) - 1;
            *active_words
                .last_mut()
                .ok_or_else(|| self.cursors.error())? = mask;
            *owned_words.last_mut().ok_or_else(|| self.cursors.error())? = mask;
        }
        let allocation_charge = cursors
            .capacity()
            .checked_mul(size_of::<CursorId>())
            .and_then(|bytes| bytes.checked_add(size_of::<DependentFrame>()))
            .and_then(|bytes| bytes.checked_add(active_words.capacity() * size_of::<u64>()))
            .and_then(|bytes| bytes.checked_add(owned_words.capacity() * size_of::<u64>()))
            .and_then(|bytes| bytes.checked_add(triggered_words.capacity() * size_of::<u64>()))
            .and_then(|bytes| {
                bytes.checked_add(marks.capacity() * size_of::<DependentCursorMark>())
            })
            .ok_or_else(|| {
                StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, limit)
            })?;
        let Some(mut prepared) = prepare_frame(self.plan.as_ref(), node, arena_start)? else {
            return Err(reference_resolution_error(self.plan.as_ref()));
        };
        let construction_charge =
            allocation_charge
                .checked_add(prepared.charge)
                .ok_or_else(|| {
                    StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, limit)
                })?;
        if !self.session_would_fit(construction_charge) {
            return Err(StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                self.retained_bytes().saturating_add(construction_charge),
                limit,
            ));
        }
        let child_depth = self
            .depth_offset
            .checked_add(self.frames.len())
            .and_then(|depth| depth.checked_add(1))
            .ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::RecursionDepth,
                    usize::MAX,
                    self.limits.max_depth as usize,
                )
            })?;
        if child_depth > self.limits.max_depth as usize {
            return Err(StructuredRuntimeError::new(
                LimitKind::RecursionDepth,
                child_depth,
                self.limits.max_depth as usize,
            ));
        }
        self.charge_session_bytes_after_precheck(construction_charge)?;
        for index in 0..count {
            let schema = match self.plan.as_ref().node(node) {
                NodePlan::Object(plan) => plan.dependent_schemas[index].schema,
                _ => return Err(reference_resolution_error(self.plan.as_ref())),
            };
            match self.start_cursor_at_depth(schema, child_depth) {
                Ok(cursor) => cursors.push(cursor),
                Err(error) => {
                    for cursor in cursors.drain(..).rev() {
                        self.release_owned_cursor(cursor);
                    }
                    self.session_memory.release(construction_charge);
                    return Err(error);
                }
            }
        }
        let Frame::Object(object) = &mut prepared.frame else {
            return Err(reference_resolution_error(self.plan.as_ref()));
        };
        object.dependent = Some(Box::new(DependentFrame {
            cursors,
            active_words,
            owned_words,
            triggered_words,
            marks,
            allocation_charge,
        }));
        self.session_memory.release(prepared.charge);
        Ok(prepared)
    }

    fn prepare_negation_frame(
        &mut self,
        node: NodeId,
    ) -> Result<PreparedFrame, StructuredRuntimeError> {
        let inner = match self.plan.as_ref().node(node) {
            NodePlan::Negation { inner } => *inner,
            _ => return Err(reference_resolution_error(self.plan.as_ref())),
        };
        self._validator_ledger.ensure_available(2)?;
        let child_depth = self
            .depth_offset
            .checked_add(self.frames.len())
            .and_then(|depth| depth.checked_add(1))
            .ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::RecursionDepth,
                    usize::MAX,
                    self.limits.max_depth as usize,
                )
            })?;
        if child_depth > self.limits.max_depth as usize {
            return Err(StructuredRuntimeError::new(
                LimitKind::RecursionDepth,
                child_depth,
                self.limits.max_depth as usize,
            ));
        }
        let pool_mark = self.cursors.pool_mark()?;
        let syntax = self.start_any_cursor_at_depth(child_depth)?;
        let inner_cursor = match self.start_cursor_at_depth(inner, child_depth) {
            Ok(cursor) => cursor,
            Err(error) => {
                self.cursors.rollback_unpublished(syntax, pool_mark)?;
                return Err(error);
            }
        };
        Ok(PreparedFrame {
            frame: Frame::Negation(NegationFrame {
                node,
                syntax,
                inner: inner_cursor,
                inner_state: NegationInnerState::Active,
            }),
            charge: 0,
        })
    }

    fn prepare_unevaluated_frame(
        &mut self,
        node: NodeId,
    ) -> Result<PreparedFrame, StructuredRuntimeError> {
        let (kind, scope, unevaluated) = match self.plan.as_ref().node(node) {
            NodePlan::Unevaluated(plan) => (plan.kind, plan.scope, plan.unevaluated),
            _ => return Err(reference_resolution_error(self.plan.as_ref())),
        };
        self._validator_ledger.ensure_available(2)?;
        let child_depth = self
            .depth_offset
            .checked_add(self.frames.len())
            .and_then(|depth| depth.checked_add(1))
            .ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::RecursionDepth,
                    usize::MAX,
                    self.limits.max_depth as usize,
                )
            })?;
        if child_depth > self.limits.max_depth as usize {
            return Err(StructuredRuntimeError::new(
                LimitKind::RecursionDepth,
                child_depth,
                self.limits.max_depth as usize,
            ));
        }
        let pool_mark = self.cursors.pool_mark()?;
        let scope_cursor = self.start_cursor_at_depth(scope, child_depth)?;
        let tracker = match self.start_any_cursor_at_depth(child_depth) {
            Ok(cursor) => cursor,
            Err(error) => {
                self.cursors.rollback_unpublished(scope_cursor, pool_mark)?;
                return Err(error);
            }
        };
        Ok(PreparedFrame {
            frame: Frame::Unevaluated(UnevaluatedFrame {
                node,
                kind,
                unevaluated,
                scope: scope_cursor,
                tracker,
                current_candidate: None,
                candidates: EvaluatedSet::None,
                location_count: 0,
                instance_kind: None,
            }),
            charge: 0,
        })
    }

    fn push_obligation_child(
        &mut self,
        obligations: &SchemaObligationSet,
    ) -> Result<bool, StructuredRuntimeError> {
        let next_depth = self
            .depth_offset
            .checked_add(self.frames.len())
            .and_then(|depth| depth.checked_add(1))
            .ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::RecursionDepth,
                    usize::MAX,
                    self.limits.max_depth as usize,
                )
            })?;
        if next_depth > self.limits.max_depth as usize {
            return Err(StructuredRuntimeError::new(
                LimitKind::RecursionDepth,
                next_depth,
                self.limits.max_depth as usize,
            ));
        }
        let n = obligations.len();
        self._validator_ledger.ensure_available(n)?;
        let cap = self.local_session_budget()?;
        let item_bytes = n
            .checked_mul(size_of::<CursorId>() + size_of::<CursorMark>())
            .ok_or_else(|| StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, cap))?;
        let frame_floor = if self.frames.len() == self.frames.capacity() {
            size_of::<Frame>()
        } else {
            0
        };
        let scope_floor = if self.frame_scopes.len() == self.frame_scopes.capacity() {
            size_of::<ScopeLease>()
        } else {
            0
        };
        if !self.session_would_fit(frame_floor + scope_floor + item_bytes) {
            return Err(StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                usize::MAX,
                cap,
            ));
        }
        let frames_cap_before = self.frames.capacity();
        self.frames
            .try_reserve_exact(1)
            .map_err(|_| StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, cap))?;
        let scopes_cap_before = self.frame_scopes.capacity();
        self.frame_scopes
            .try_reserve_exact(1)
            .map_err(|_| StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, cap))?;
        let mut cursors = Vec::new();
        cursors
            .try_reserve_exact(n)
            .map_err(|_| StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, cap))?;
        let mut marks = Vec::new();
        marks
            .try_reserve_exact(n)
            .map_err(|_| StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, cap))?;
        let frame_charge = (self.frames.capacity() - frames_cap_before) * size_of::<Frame>();
        let scope_charge =
            (self.frame_scopes.capacity() - scopes_cap_before) * size_of::<ScopeLease>();
        let allocation_charge =
            cursors.capacity() * size_of::<CursorId>() + marks.capacity() * size_of::<CursorMark>();
        let grew = frame_charge + scope_charge + allocation_charge;
        self.charge_session_bytes_after_precheck(grew)?;
        if let Err(error) = self.reserve_undo(1) {
            self.session_memory.release(allocation_charge);
            return Err(error);
        }
        for node in obligations.iter() {
            match self.start_cursor(node) {
                Ok(cursor) => cursors.push(cursor),
                Err(error) => {
                    for cursor in cursors {
                        self.release_owned_cursor(cursor);
                    }
                    self.session_memory.release(allocation_charge);
                    return Err(error);
                }
            }
        }
        let scope = self.current_scope()?.clone();
        self.frames.push(Frame::Obligation(ObligationFrame {
            cursors,
            marks,
            allocation_charge,
        }));
        self.frame_scopes.push(scope);
        self.push_undo(Undo::PopFrame)?;
        Ok(true)
    }

    fn route_byte(&mut self, byte: u8) -> Result<bool, StructuredRuntimeError> {
        if self.root_closed {
            return Ok(WS.contains(&byte));
        }
        loop {
            if self.frames.is_empty() {
                return Ok(false);
            }
            let depth = self.frames.len() - 1;
            match &self.frames[depth] {
                Frame::Regular { node, state } => {
                    let (node, state) = (*node, *state);
                    let engine = self.engine_for(node);
                    if let Some(next) = engine.consume_token(state, &[byte]) {
                        if !engine.is_dead(next) {
                            let next_accepting = engine.is_accepting(next);
                            let only_trailing_whitespace = next_accepting
                                && (0u8..=u8::MAX).all(|candidate| {
                                    WS.contains(&candidate)
                                        || !engine.live_classes(next).contains(usize::from(
                                            engine.class_of_byte(candidate).get(),
                                        ))
                                });
                            self.reserve_undo(1)?;
                            self.frames[depth] = Frame::Regular { node, state: next };
                            self.push_undo(Undo::SetRegularState { depth, old: state })?;
                            self.record_item_byte(depth, byte)?;
                            // Finalize a complete accepted item before trailing whitespace masks duplicates.
                            if next_accepting
                                && (self.tracked_canonical_item_is_closed(depth)
                                    || only_trailing_whitespace)
                            {
                                if !self.finish_tracked_item_if_direct_child(depth)? {
                                    return Ok(false);
                                }
                                self.reserve_undo(1)?;
                                let popped = self.frames.pop().expect("checked above");
                                let scope = self.frame_scopes.pop().expect("frame scope exists");
                                self.push_undo(Undo::PushFrame(popped, scope))?;
                                if depth == 0 {
                                    self.reserve_undo(1)?;
                                    self.root_closed = true;
                                    self.push_undo(Undo::SetRootClosed { old: false })?;
                                }
                            }
                            return Ok(true);
                        }
                    }
                    if !engine.is_accepting(state) {
                        return Ok(false);
                    }
                    if !self.finish_tracked_item_if_direct_child(depth)? {
                        return Ok(false);
                    }
                    self.reserve_undo(1)?;
                    let popped = self.frames.pop().expect("checked above");
                    let scope = self.frame_scopes.pop().expect("frame scope exists");
                    self.push_undo(Undo::PushFrame(popped, scope))?;
                    if depth == 0 {
                        self.reserve_undo(1)?;
                        self.root_closed = true;
                        self.push_undo(Undo::SetRootClosed { old: false })?;
                        return Ok(WS.contains(&byte));
                    }
                }
                Frame::Number(current) => {
                    let current = *current;
                    let NodePlan::Number { engine, constraint } =
                        self.plan.as_ref().node(current.node)
                    else {
                        unreachable!("number frame references number plan")
                    };
                    if let Some(next_state) = engine.consume_token(current.state, &[byte]) {
                        if !engine.is_dead(next_state) {
                            if !WS.contains(&byte) {
                                let next_bytes =
                                    current.accumulator.bytes.checked_add(1).ok_or_else(|| {
                                        StructuredRuntimeError::new(
                                            LimitKind::NumberBytes,
                                            usize::MAX,
                                            self.limits.max_number_bytes,
                                        )
                                    })?;
                                if next_bytes as usize > self.limits.max_number_bytes {
                                    return Err(StructuredRuntimeError::new(
                                        LimitKind::NumberBytes,
                                        next_bytes as usize,
                                        self.limits.max_number_bytes,
                                    ));
                                }
                            }
                            let mut next = current;
                            next.state = next_state;
                            next.accumulator.push(
                                byte,
                                constraint
                                    .multiple_of
                                    .map_or(1, |(coefficient, _)| coefficient),
                            );
                            if !number_prefix_sign_viable(next.accumulator, *constraint) {
                                return Ok(false);
                            }
                            self.reserve_undo(1)?;
                            self.frames[depth] = Frame::Number(next);
                            self.push_undo(Undo::SetNumberFrame {
                                depth,
                                old: current,
                            })?;
                            self.record_item_byte(depth, byte)?;
                            return Ok(true);
                        }
                    }
                    if !engine.is_accepting(current.state)
                        || !number_accepts(current.accumulator, *constraint)
                    {
                        return Ok(false);
                    }
                    if !self.finish_tracked_item_if_direct_child(depth)? {
                        return Ok(false);
                    }
                    self.reserve_undo(1)?;
                    let popped = self.frames.pop().expect("checked above");
                    let scope = self.frame_scopes.pop().expect("frame scope exists");
                    self.push_undo(Undo::PushFrame(popped, scope))?;
                    if depth == 0 {
                        self.reserve_undo(1)?;
                        self.root_closed = true;
                        self.push_undo(Undo::SetRootClosed { old: false })?;
                        return Ok(WS.contains(&byte));
                    }
                }
                Frame::String(_) => {
                    let before = self.frames.len();
                    let ok = self.route_string_byte(depth, byte)?;
                    if ok && self.frames.len() == before {
                        self.record_item_byte(depth, byte)?;
                    }
                    return Ok(ok);
                }
                Frame::StringEnum(_) => {
                    let before = self.frames.len();
                    let ok = self.route_string_enum_byte(depth, byte)?;
                    if ok && self.frames.len() == before {
                        self.record_item_byte(depth, byte)?;
                    }
                    return Ok(ok);
                }
                Frame::Object(_) => {
                    let before = self.frames.len();
                    let ok = self.route_object_byte(depth, byte)?;
                    if ok && self.frames.len() == before {
                        self.record_item_byte(depth, byte)?;
                    }
                    return Ok(ok);
                }
                Frame::Array(_) => {
                    let before = self.frames.len();
                    let ok = self.route_array_byte(depth, byte)?;
                    if ok && self.frames.len() == before {
                        self.record_item_byte(depth, byte)?;
                    }
                    return Ok(ok);
                }
                Frame::Obligation(_) => match self.route_obligation_byte(depth, byte)? {
                    ObligationStep::Rejected => return Ok(false),
                    ObligationStep::Advanced => {
                        self.record_item_byte(depth, byte)?;
                        return Ok(true);
                    }
                    ObligationStep::Popped if self.root_closed => {
                        return Ok(WS.contains(&byte));
                    }
                    ObligationStep::Popped => continue,
                },
                Frame::Combinator(_) => match self.route_combinator_byte(depth, byte)? {
                    CombinatorStep::Rejected => return Ok(false),
                    CombinatorStep::Advanced => {
                        self.record_item_byte(depth, byte)?;
                        return Ok(true);
                    }
                    CombinatorStep::Popped if self.root_closed => {
                        return Ok(WS.contains(&byte));
                    }
                    CombinatorStep::Popped => continue,
                },
                Frame::Negation(_) => match self.route_negation_byte(depth, byte)? {
                    NegationStep::Rejected => return Ok(false),
                    NegationStep::Advanced => {
                        self.record_item_byte(depth, byte)?;
                        return Ok(true);
                    }
                    NegationStep::Popped if self.root_closed => {
                        return Ok(WS.contains(&byte));
                    }
                    NegationStep::Popped => continue,
                },
                Frame::Unevaluated(_) => match self.route_unevaluated_byte(depth, byte)? {
                    UnevaluatedStep::Rejected => return Ok(false),
                    UnevaluatedStep::Advanced => {
                        self.record_item_byte(depth, byte)?;
                        return Ok(true);
                    }
                    UnevaluatedStep::Popped { replay: false } => return Ok(true),
                    UnevaluatedStep::Popped { replay: true } if self.root_closed => {
                        return Ok(WS.contains(&byte));
                    }
                    UnevaluatedStep::Popped { replay: true } => continue,
                },
                Frame::Any(_) => {
                    let before = self.frames.len();
                    match self.route_any_byte(depth, byte)? {
                        AnyStep::Rejected => return Ok(false),
                        AnyStep::Advanced => {
                            // No tracked array can live inside an `AnyJson` subtree, so a byte
                            // that pushed a deeper frame still records correctly at this depth.
                            if self.frames.len() >= before {
                                self.record_item_byte(depth, byte)?;
                            }
                            return Ok(true);
                        }
                        AnyStep::Popped if self.root_closed => return Ok(WS.contains(&byte)),
                        AnyStep::Popped => continue,
                    }
                }
            }
        }
    }

    fn route_string_byte(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<bool, StructuredRuntimeError> {
        let Frame::String(current) = self.frames[depth] else {
            unreachable!("expected string frame")
        };
        if current.phase == StringPhase::BeforeQuote {
            if WS.contains(&byte) {
                return Ok(true);
            }
            if byte != b'"' {
                return Ok(false);
            }
            self.reserve_undo(1)?;
            let mut next = current;
            next.phase = StringPhase::Body;
            self.frames[depth] = Frame::String(next);
            self.push_undo(Undo::SetStringFrame {
                depth,
                old: current,
            })?;
            return Ok(true);
        }
        let NodePlan::String(plan) = self.plan.as_ref().node(current.node) else {
            unreachable!("string frame references string plan")
        };
        if byte == b'"' && current.decoder.at_boundary() {
            if current.scalars < plan.min_scalars
                || plan
                    .max_scalars
                    .is_some_and(|maximum| current.scalars > maximum)
                || plan.pattern.as_ref().is_some_and(|engine| {
                    !current
                        .pattern_state
                        .is_some_and(|state| engine.is_accepting(state))
                })
            {
                return Ok(false);
            }
            return self.close_string_value(depth, byte);
        }
        let mut next = current;
        match next.decoder.push(byte) {
            DecodeStep::Invalid => return Ok(false),
            DecodeStep::Continue => {
                if plan
                    .max_scalars
                    .is_some_and(|maximum| current.scalars >= maximum)
                {
                    return Ok(false);
                }
                if let (Some(engine), Some(state)) = (&plan.pattern, next.pattern_state) {
                    if next
                        .decoder
                        .any_pending_scalar_satisfies_with_supplementary(
                            4096,
                            |scalar| {
                                let mut encoded = [0u8; 4];
                                engine
                                    .consume_token(
                                        state,
                                        scalar.encode_utf8(&mut encoded).as_bytes(),
                                    )
                                    .is_some_and(|next| !engine.is_dead(next))
                            },
                            || engine.can_consume_any_scalar(state),
                            |start, end| engine.can_consume_supplementary_range(state, start, end),
                        )
                        .is_some_and(|viable| !viable)
                    {
                        return Ok(false);
                    }
                }
            }
            DecodeStep::Scalar(scalar) => {
                next.scalars = next.scalars.checked_add(1).ok_or_else(|| {
                    StructuredRuntimeError::new(
                        LimitKind::StringBytes,
                        usize::MAX,
                        self.limits.max_string_bytes,
                    )
                })?;
                if plan
                    .max_scalars
                    .is_some_and(|maximum| next.scalars > maximum)
                {
                    return Ok(false);
                }
                if let (Some(engine), Some(state)) = (&plan.pattern, next.pattern_state) {
                    let mut encoded = [0u8; 4];
                    let bytes = scalar.encode_utf8(&mut encoded).as_bytes();
                    let Some(pattern_state) = engine.consume_token(state, bytes) else {
                        return Ok(false);
                    };
                    if engine.is_dead(pattern_state) {
                        return Ok(false);
                    }
                    next.pattern_state = Some(pattern_state);
                    if plan
                        .max_scalars
                        .is_some_and(|maximum| next.scalars == maximum)
                        && !engine.is_accepting(pattern_state)
                    {
                        return Ok(false);
                    }
                }
            }
        }
        self.reserve_undo(1)?;
        self.frames[depth] = Frame::String(next);
        self.push_undo(Undo::SetStringFrame {
            depth,
            old: current,
        })?;
        Ok(true)
    }

    fn route_string_enum_byte(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<bool, StructuredRuntimeError> {
        let Frame::StringEnum(current) = self.frames[depth] else {
            unreachable!("expected string enum frame")
        };
        let NodePlan::StringEnum(plan) = self.plan.as_ref().node(current.node) else {
            unreachable!("string enum frame references string enum plan")
        };
        if current.phase == StringPhase::BeforeQuote {
            if WS.contains(&byte) {
                return Ok(true);
            }
            if byte != b'"'
                || !self.string_enum_has_unseen_candidate(depth, plan, current.lo, current.hi)
            {
                return Ok(false);
            }
            let mut next = current;
            next.phase = StringPhase::Body;
            return self.replace_string_enum_frame(depth, current, next);
        }
        if byte == b'"' && current.decoder.at_boundary() {
            if !plan.contains_exact(current.lo, current.hi, current.decoded_bytes) {
                return Ok(false);
            }
            if !self.string_enum_has_unseen_candidate(depth, plan, current.lo, current.hi) {
                return Ok(false);
            }
            return self.close_string_value(depth, byte);
        }
        let mut next = current;
        match next.decoder.push(byte) {
            DecodeStep::Invalid => return Ok(false),
            DecodeStep::Continue => {
                // A pending scalar is impossible when every narrowed enum member already ends.
                if !plan.can_continue(next.lo, next.hi, next.decoded_bytes) {
                    return Ok(false);
                }
                if next
                    .decoder
                    .any_pending_scalar_satisfies_with_supplementary(
                        4096,
                        |scalar| {
                            self.string_enum_scalar_has_unseen_candidate(
                                depth,
                                plan,
                                next.lo,
                                next.hi,
                                next.decoded_bytes,
                                scalar,
                            )
                        },
                        || {
                            Some(self.string_enum_any_scalar_has_unseen_candidate(
                                depth,
                                plan,
                                next.lo,
                                next.hi,
                                next.decoded_bytes,
                            ))
                        },
                        |start, end| {
                            self.string_enum_supplementary_has_unseen_candidate(
                                depth,
                                plan,
                                next.lo,
                                next.hi,
                                next.decoded_bytes,
                                start..=end,
                            )
                        },
                    )
                    .is_some_and(|viable| !viable)
                {
                    return Ok(false);
                }
            }
            DecodeStep::Scalar(scalar) => {
                let mut encoded = [0u8; 4];
                for &decoded in scalar.encode_utf8(&mut encoded).as_bytes() {
                    let Some((lo, hi)) = plan.narrow(next.lo, next.hi, next.decoded_bytes, decoded)
                    else {
                        return Ok(false);
                    };
                    next.lo = lo;
                    next.hi = hi;
                    next.decoded_bytes = next.decoded_bytes.checked_add(1).ok_or_else(|| {
                        StructuredRuntimeError::new(
                            LimitKind::StringBytes,
                            usize::MAX,
                            self.limits.max_string_bytes,
                        )
                    })?;
                    if usize::try_from(next.decoded_bytes)
                        .ok()
                        .is_none_or(|length| length > self.limits.max_string_bytes)
                    {
                        return Err(StructuredRuntimeError::new(
                            LimitKind::StringBytes,
                            usize::try_from(next.decoded_bytes).unwrap_or(usize::MAX),
                            self.limits.max_string_bytes,
                        ));
                    }
                }
                if !self.string_enum_has_unseen_candidate(depth, plan, next.lo, next.hi) {
                    return Ok(false);
                }
            }
        }
        self.replace_string_enum_frame(depth, current, next)
    }

    /// A finite enum prefix under `uniqueItems` must retain at least one decoded member not
    /// already committed in the direct parent array. JSON escapes narrow the same decoded range.
    fn string_enum_has_unseen_candidate(
        &self,
        depth: usize,
        plan: &super::plan::StringEnumPlan,
        lo: u32,
        hi: u32,
    ) -> bool {
        if depth == 0 {
            return true;
        }
        let Some(Frame::Array(array)) = self.frames.get(depth - 1) else {
            return true;
        };
        let seen = array.canonical_ref();
        let (Ok(lo), Ok(hi)) = (usize::try_from(lo), usize::try_from(hi)) else {
            return false;
        };
        (lo..hi).any(|index| {
            plan.member(index).is_some_and(|candidate| {
                seen.is_none_or(|set| !set.contains_string(candidate))
                    && self.string_enum_member_respects_contains_cap(array, candidate)
            })
        })
    }

    fn string_enum_member_respects_contains_cap(
        &self,
        array: &ArrayFrame,
        candidate: &[u8],
    ) -> bool {
        let plan = super::plan::array_plan_of(self.plan.as_ref(), array.node);
        let Some(contains) = &plan.contains else {
            return true;
        };
        if contains.max != Some(array.contains_count) {
            return true;
        }
        match contains.matches {
            ContainsMatch::Never => true,
            ContainsMatch::Always => false,
            ContainsMatch::Schema(node) => self
                .contains_schema_matches_decoded_string(node, candidate)
                .is_none_or(|matches| !matches),
        }
    }

    fn contains_schema_matches_decoded_string(
        &self,
        mut node: NodeId,
        candidate: &[u8],
    ) -> Option<bool> {
        for _ in 0..=self.plan.as_ref().nodes.len() {
            match self.plan.as_ref().node(node) {
                NodePlan::Ref { target } => node = *target,
                NodePlan::StringEnum(members) => {
                    return Some((0..members.member_count()).any(|index| {
                        members
                            .member(index)
                            .is_some_and(|member| member == candidate)
                    }));
                }
                NodePlan::Regular(engine) => {
                    let encoded = encode_decoded_json_string(candidate)?;
                    return Some(engine.accepts(&encoded));
                }
                _ => return None,
            }
        }
        None
    }

    fn string_enum_scalar_has_unseen_candidate(
        &self,
        depth: usize,
        plan: &super::plan::StringEnumPlan,
        mut lo: u32,
        mut hi: u32,
        mut position: u32,
        scalar: char,
    ) -> bool {
        let mut encoded = [0u8; 4];
        for &decoded in scalar.encode_utf8(&mut encoded).as_bytes() {
            let Some(range) = plan.narrow(lo, hi, position, decoded) else {
                return false;
            };
            (lo, hi) = range;
            let Some(next) = position.checked_add(1) else {
                return false;
            };
            position = next;
            if usize::try_from(position)
                .ok()
                .is_none_or(|length| length > self.limits.max_string_bytes)
            {
                return false;
            }
        }
        self.string_enum_has_unseen_candidate(depth, plan, lo, hi)
    }

    fn string_enum_any_scalar_has_unseen_candidate(
        &self,
        depth: usize,
        plan: &super::plan::StringEnumPlan,
        lo: u32,
        hi: u32,
        position: u32,
    ) -> bool {
        let (Ok(lo), Ok(hi), Ok(position)) = (
            usize::try_from(lo),
            usize::try_from(hi),
            usize::try_from(position),
        ) else {
            return false;
        };
        let parent = depth
            .checked_sub(1)
            .and_then(|parent| self.frames.get(parent));
        let array = match parent {
            Some(Frame::Array(array)) => Some(array),
            _ => None,
        };
        (lo..hi).any(|index| {
            plan.member(index).is_some_and(|candidate| {
                candidate
                    .get(position..)
                    .is_some_and(|tail| !tail.is_empty())
                    && array.is_none_or(|array| {
                        array
                            .canonical_ref()
                            .is_none_or(|set| !set.contains_string(candidate))
                            && self.string_enum_member_respects_contains_cap(array, candidate)
                    })
            })
        })
    }

    fn string_enum_supplementary_has_unseen_candidate(
        &self,
        depth: usize,
        plan: &super::plan::StringEnumPlan,
        lo: u32,
        hi: u32,
        position: u32,
        scalar_range: std::ops::RangeInclusive<u32>,
    ) -> Option<bool> {
        let (lo, hi, position) = (
            usize::try_from(lo).ok()?,
            usize::try_from(hi).ok()?,
            usize::try_from(position).ok()?,
        );
        if lo >= hi || hi > plan.member_count() {
            return Some(false);
        }
        let seen = depth
            .checked_sub(1)
            .and_then(|parent| self.frames.get(parent))
            .and_then(|frame| match frame {
                Frame::Array(array) => array.canonical_ref(),
                _ => None,
            });
        for index in lo..hi {
            let member = plan.member(index)?;
            let tail = member.get(position..)?;
            let scalar = std::str::from_utf8(tail).ok()?.chars().next()?;
            let scalar = u32::from(scalar);
            if scalar_range.contains(&scalar) && seen.is_none_or(|set| !set.contains_string(member))
            {
                return Some(true);
            }
        }
        Some(false)
    }

    fn replace_string_enum_frame(
        &mut self,
        depth: usize,
        old: StringEnumFrame,
        new: StringEnumFrame,
    ) -> Result<bool, StructuredRuntimeError> {
        self.reserve_undo(1)?;
        self.frames[depth] = Frame::StringEnum(new);
        self.push_undo(Undo::SetStringEnumFrame { depth, old })?;
        Ok(true)
    }

    fn close_string_value(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<bool, StructuredRuntimeError> {
        self.record_item_byte(depth, byte)?;
        if !self.finish_tracked_item_if_direct_child(depth)? {
            return Ok(false);
        }
        self.reserve_undo(1)?;
        let popped = self.frames.pop().expect("string frame exists");
        let scope = self.frame_scopes.pop().expect("frame scope exists");
        self.push_undo(Undo::PushFrame(popped, scope))?;
        if depth == 0 {
            self.reserve_undo(1)?;
            self.root_closed = true;
            self.push_undo(Undo::SetRootClosed { old: false })?;
        }
        Ok(true)
    }

    /// Feeds a byte to ancestor arrays' active `uniqueItems` trackers.
    /// The consuming array's punctuation is excluded.
    fn record_item_byte(
        &mut self,
        consumer_depth: usize,
        byte: u8,
    ) -> Result<(), StructuredRuntimeError> {
        if self.byte_observed {
            return Ok(());
        }
        self.byte_observed = true;
        for depth in 0..consumer_depth {
            if self.array_item_tracking_active(depth) {
                self.feed_array_item_tracker(depth, byte)?;
            }
        }
        for depth in 0..=consumer_depth {
            if matches!(self.frames.get(depth), Some(Frame::Object(object)) if object.dependent.is_some())
            {
                self.feed_object_dependencies(depth, byte)?;
            }
        }
        Ok(())
    }

    fn feed_object_dependencies(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<(), StructuredRuntimeError> {
        let count = self
            .object_ref(depth)
            .dependent
            .as_ref()
            .map_or(0, |frame| active_branch_count(&frame.active_words));
        self.reserve_undo(count)?;
        if let Some(frame) = self.object_mut(depth).dependent.as_mut() {
            frame.marks.clear();
        }
        let dependencies = self
            .object_ref(depth)
            .dependent
            .as_ref()
            .map_or(0, |frame| frame.cursors.len());
        for index in 0..dependencies {
            let cursor = match self.object_ref(depth).dependent.as_ref() {
                Some(frame) if bit_is_set(&frame.active_words, index) => frame.cursors[index],
                Some(_) => continue,
                None => return Err(self.cursors.error()),
            };
            let mark = self.cursors.checkpoint(cursor)?;
            let dependency = u32::try_from(index).map_err(|_| self.cursors.error())?;
            if let Some(frame) = self.object_mut(depth).dependent.as_mut() {
                frame.marks.push(DependentCursorMark {
                    dependency,
                    cursor,
                    mark,
                });
            }
            let attempt = self
                .object_ref(depth)
                .dependent
                .as_ref()
                .and_then(|frame| frame.marks.last())
                .copied()
                .ok_or_else(|| self.cursors.error())?;
            match self.push_cursor_byte(attempt.cursor, byte)? {
                CursorStep::Alive | CursorStep::Complete => {
                    self.push_undo(Undo::RollbackDependentCursor {
                        cursor: attempt.cursor,
                        mark: attempt.mark,
                    })?;
                }
                CursorStep::Dead => {
                    self.cursors.rollback(attempt.cursor, attempt.mark)?;
                    if let Some(frame) = self.object_mut(depth).dependent.as_mut() {
                        clear_bit(&mut frame.active_words, index);
                    }
                    self.push_undo(Undo::ReactivateDependentSchema {
                        depth,
                        dependency: attempt.dependency,
                    })?;
                }
            }
        }
        let rejected = self
            .object_ref(depth)
            .dependent
            .as_ref()
            .is_some_and(|frame| {
                frame
                    .triggered_words
                    .iter()
                    .zip(&frame.active_words)
                    .any(|(triggered, active)| triggered & !active != 0)
            });
        if rejected {
            self.dead = true;
        }
        Ok(())
    }

    /// Closes the tracked item at `depth - 1` when `depth` is its direct child frame closing -
    /// `Ok(false)` iff `uniqueItems` rejects it as a duplicate.
    fn finish_tracked_item_if_direct_child(
        &mut self,
        depth: usize,
    ) -> Result<bool, StructuredRuntimeError> {
        if depth > 0 && self.array_item_tracking_active(depth - 1) {
            return self.finish_array_item_tracker(depth - 1);
        }
        Ok(true)
    }

    /// Whether a parent array's canonical builder has observed a complete value.
    fn tracked_canonical_item_is_closed(&self, depth: usize) -> bool {
        if depth == 0 {
            return false;
        }
        let Some(Frame::Array(array)) = self.frames.get(depth - 1) else {
            return false;
        };
        let Some(item) = &array.item else {
            return false;
        };
        let Some(builder) = &item.builder else {
            return false;
        };
        array
            .canonical_ref()
            .is_some_and(|set| set.item_is_done(builder))
    }

    /// Advances all validators in an obligation frame in lockstep.
    /// Closes the frame only when all validators accept and cannot extend.
    fn route_obligation_byte(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<ObligationStep, StructuredRuntimeError> {
        let n = match &self.frames[depth] {
            Frame::Obligation(ob) => ob.cursors.len(),
            _ => unreachable!("expected an obligation frame"),
        };
        self.reserve_undo(n)?;
        let all_accepting = match &self.frames[depth] {
            Frame::Obligation(ob) => ob
                .cursors
                .iter()
                .all(|&cursor| self.cursors.is_accepting(cursor)),
            _ => unreachable!("expected an obligation frame"),
        };
        if let Frame::Obligation(ob) = &mut self.frames[depth] {
            ob.marks.clear();
        }
        for i in 0..n {
            let cursor = match &self.frames[depth] {
                Frame::Obligation(ob) => ob.cursors[i],
                _ => unreachable!("expected an obligation frame"),
            };
            let mark = self.cursors.checkpoint(cursor)?;
            if let Frame::Obligation(ob) = &mut self.frames[depth] {
                ob.marks.push(mark);
            }
            match self.push_cursor_byte(cursor, byte) {
                Ok(CursorStep::Alive | CursorStep::Complete) => {}
                Ok(CursorStep::Dead) => {
                    self.rollback_obligation_attempts(depth, i + 1)?;
                    if all_accepting {
                        return self.pop_obligation(depth);
                    }
                    return Ok(ObligationStep::Rejected);
                }
                Err(error) => {
                    self.rollback_obligation_attempts(depth, i)?;
                    return Err(error);
                }
            }
        }
        for i in 0..n {
            let (cursor, mark) = match &self.frames[depth] {
                Frame::Obligation(ob) => (ob.cursors[i], ob.marks[i]),
                _ => unreachable!("expected an obligation frame"),
            };
            self.push_undo(Undo::RollbackObligationCursor { cursor, mark })?;
        }
        if let Frame::Obligation(obligation) = &mut self.frames[depth] {
            obligation.marks.clear();
        }
        Ok(ObligationStep::Advanced)
    }

    fn rollback_obligation_attempts(
        &mut self,
        depth: usize,
        count: usize,
    ) -> Result<(), StructuredRuntimeError> {
        for i in (0..count).rev() {
            let (cursor, mark) = match &self.frames[depth] {
                Frame::Obligation(ob) => (ob.cursors[i], ob.marks[i]),
                _ => unreachable!("expected an obligation frame"),
            };
            self.cursors.rollback(cursor, mark)?;
        }
        if let Frame::Obligation(obligation) = &mut self.frames[depth] {
            obligation.marks.clear();
        }
        Ok(())
    }

    fn pop_obligation(&mut self, depth: usize) -> Result<ObligationStep, StructuredRuntimeError> {
        if !self.finish_tracked_item_if_direct_child(depth)? {
            return Ok(ObligationStep::Rejected);
        }
        self.reserve_undo(1)?;
        let popped = self.frames.pop().expect("checked above");
        let scope = self.frame_scopes.pop().expect("frame scope exists");
        self.push_undo(Undo::PushFrame(popped, scope))?;
        if depth == 0 {
            self.reserve_undo(1)?;
            self.root_closed = true;
            self.push_undo(Undo::SetRootClosed { old: false })?;
        }
        Ok(ObligationStep::Popped)
    }

    fn route_combinator_byte(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<CombinatorStep, StructuredRuntimeError> {
        let (kind, branch_count, accepting_before) = match &self.frames[depth] {
            Frame::Combinator(frame) => (
                frame.kind,
                frame.cursors.len(),
                accepting_branch_count(frame, &self.cursors),
            ),
            _ => unreachable!("expected a combinator frame"),
        };
        self.reserve_undo(branch_count)?;
        if let Frame::Combinator(frame) = &mut self.frames[depth] {
            frame.marks.clear();
        }
        for index in 0..branch_count {
            let cursor = match &self.frames[depth] {
                Frame::Combinator(frame) if bit_is_set(&frame.active_words, index) => {
                    frame.cursors[index]
                }
                Frame::Combinator(_) => continue,
                _ => unreachable!("expected a combinator frame"),
            };
            let mark = self.cursors.checkpoint(cursor)?;
            let branch_index = u32::try_from(index).map_err(|_| self.cursors.error())?;
            if let Frame::Combinator(frame) = &mut self.frames[depth] {
                frame.marks.push(BranchMark {
                    branch_index,
                    cursor,
                    mark,
                });
            }
            match self.push_cursor_byte(cursor, byte) {
                Ok(CursorStep::Alive | CursorStep::Complete) => {}
                Ok(CursorStep::Dead) => {
                    self.cursors.rollback(cursor, mark)?;
                    if let Frame::Combinator(frame) = &mut self.frames[depth] {
                        clear_bit(&mut frame.active_words, index);
                    }
                }
                Err(error) => {
                    self.restore_combinator_attempts(depth)?;
                    return Err(error);
                }
            }
        }
        let survivors = match &self.frames[depth] {
            Frame::Combinator(frame) => active_branch_count(&frame.active_words),
            _ => unreachable!("expected a combinator frame"),
        };
        let boundary = match kind {
            CombinatorKind::All => survivors != branch_count,
            CombinatorKind::Any | CombinatorKind::ExactlyOne => survivors == 0,
        };
        if boundary {
            self.restore_combinator_attempts(depth)?;
            if combinator_accepts(kind, accepting_before, branch_count) {
                return self.pop_combinator(depth);
            }
            return Ok(CombinatorStep::Rejected);
        }
        let accepting_now = match &self.frames[depth] {
            Frame::Combinator(frame) => accepting_branch_count(frame, &self.cursors),
            _ => unreachable!("expected a combinator frame"),
        };
        let can_extend = match &self.frames[depth] {
            Frame::Combinator(frame) => frame.cursors.iter().enumerate().any(|(index, cursor)| {
                if !bit_is_set(&frame.active_words, index)
                    || !self.cursors.can_extend_value(*cursor)
                {
                    return false;
                }
                if kind != CombinatorKind::ExactlyOne {
                    return true;
                }
                let node = self.cursors.node(*cursor);
                frame
                    .cursors
                    .iter()
                    .enumerate()
                    .filter(|(other, candidate)| {
                        bit_is_set(&frame.active_words, *other)
                            && self.cursors.node(**candidate) == node
                    })
                    .count()
                    == 1
            }),
            _ => unreachable!("expected a combinator frame"),
        };
        if !combinator_accepts(kind, accepting_now, branch_count) && !can_extend {
            self.restore_combinator_attempts(depth)?;
            return Ok(CombinatorStep::Rejected);
        }
        let marks_len = match &self.frames[depth] {
            Frame::Combinator(frame) => frame.marks.len(),
            _ => unreachable!("expected a combinator frame"),
        };
        for index in 0..marks_len {
            let branch_mark = match &self.frames[depth] {
                Frame::Combinator(frame) => frame.marks[index],
                _ => unreachable!("expected a combinator frame"),
            };
            let active = match &self.frames[depth] {
                Frame::Combinator(frame) => {
                    bit_is_set(&frame.active_words, branch_mark.branch_index as usize)
                }
                _ => unreachable!("expected a combinator frame"),
            };
            if active {
                self.push_undo(Undo::RollbackCombinatorCursor {
                    cursor: branch_mark.cursor,
                    mark: branch_mark.mark,
                })?;
            } else {
                self.push_undo(Undo::ReactivateCombinatorBranch {
                    depth,
                    branch: branch_mark.branch_index,
                })?;
            }
        }
        if let Frame::Combinator(frame) = &mut self.frames[depth] {
            frame.marks.clear();
        }
        Ok(CombinatorStep::Advanced)
    }

    fn route_negation_byte(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<NegationStep, StructuredRuntimeError> {
        let frame = match self.frames[depth] {
            Frame::Negation(frame) => frame,
            _ => unreachable!("expected a negation frame"),
        };
        let syntax_accepting_before = self.cursors.is_accepting(frame.syntax);
        let inner_accepting_before = (frame.inner_state == NegationInnerState::Active)
            .then(|| self.cursors.is_accepting(frame.inner));
        let syntax_mark = self.cursors.checkpoint(frame.syntax)?;
        let inner_mark = if frame.inner_state == NegationInnerState::Active {
            Some(self.cursors.checkpoint(frame.inner)?)
        } else {
            None
        };
        self.reserve_undo(if inner_mark.is_some() { 2 } else { 1 })?;
        match self.push_cursor_byte(frame.syntax, byte)? {
            CursorStep::Dead => {
                self.cursors.rollback(frame.syntax, syntax_mark)?;
                if !syntax_accepting_before {
                    return Ok(NegationStep::Rejected);
                }
                let inner_rejected = frame.inner_state != NegationInnerState::Active
                    || inner_accepting_before == Some(false);
                if !inner_rejected {
                    return Ok(NegationStep::Rejected);
                }
                self.pop_negation(depth)
            }
            CursorStep::Alive | CursorStep::Complete => {
                let Some(mark) = inner_mark else {
                    self.push_undo(Undo::RollbackNegationCursor {
                        cursor: frame.syntax,
                        mark: syntax_mark,
                    })?;
                    return Ok(NegationStep::Advanced);
                };
                match self.push_cursor_byte(frame.inner, byte) {
                    Ok(CursorStep::Alive | CursorStep::Complete) => {
                        let accepts = self.cursors.is_accepting(frame.syntax)
                            && !self.cursors.is_accepting(frame.inner);
                        if !accepts && !self.cursors.can_extend_value(frame.syntax) {
                            self.cursors.rollback(frame.inner, mark)?;
                            self.cursors.rollback(frame.syntax, syntax_mark)?;
                            return Ok(NegationStep::Rejected);
                        }
                        self.push_undo(Undo::RollbackNegationCursor {
                            cursor: frame.syntax,
                            mark: syntax_mark,
                        })?;
                        self.push_undo(Undo::RollbackNegationCursor {
                            cursor: frame.inner,
                            mark,
                        })?;
                    }
                    Ok(CursorStep::Dead) => {
                        if let Frame::Negation(current) = &mut self.frames[depth] {
                            current.inner_state = NegationInnerState::PrunedOwned;
                        }
                        self.push_undo(Undo::RollbackNegationCursor {
                            cursor: frame.syntax,
                            mark: syntax_mark,
                        })?;
                        self.push_undo(Undo::ReactivateNegationInner { depth })?;
                    }
                    Err(error) => {
                        self.cursors.rollback(frame.syntax, syntax_mark)?;
                        self.cursors.rollback(frame.inner, mark)?;
                        return Err(error);
                    }
                }
                Ok(NegationStep::Advanced)
            }
        }
    }

    fn pop_negation(&mut self, depth: usize) -> Result<NegationStep, StructuredRuntimeError> {
        if !self.finish_tracked_item_if_direct_child(depth)? {
            return Ok(NegationStep::Rejected);
        }
        if !self.finish_tracked_item_if_direct_child(depth)? {
            return Ok(NegationStep::Rejected);
        }
        self.reserve_undo(1)?;
        let popped = self.frames.pop().expect("negation frame exists");
        let scope = self.frame_scopes.pop().expect("frame scope exists");
        self.push_undo(Undo::PushFrame(popped, scope))?;
        if depth == 0 {
            self.reserve_undo(1)?;
            self.root_closed = true;
            self.push_undo(Undo::SetRootClosed { old: false })?;
        }
        Ok(NegationStep::Popped)
    }

    fn restore_combinator_attempts(&mut self, depth: usize) -> Result<(), StructuredRuntimeError> {
        let marks_len = match &self.frames[depth] {
            Frame::Combinator(frame) => frame.marks.len(),
            _ => unreachable!("expected a combinator frame"),
        };
        for index in (0..marks_len).rev() {
            let branch_mark = match &self.frames[depth] {
                Frame::Combinator(frame) => frame.marks[index],
                _ => unreachable!("expected a combinator frame"),
            };
            let active = match &self.frames[depth] {
                Frame::Combinator(frame) => {
                    bit_is_set(&frame.active_words, branch_mark.branch_index as usize)
                }
                _ => unreachable!("expected a combinator frame"),
            };
            if active {
                self.cursors
                    .rollback(branch_mark.cursor, branch_mark.mark)?;
            } else if let Frame::Combinator(frame) = &mut self.frames[depth] {
                set_bit(&mut frame.active_words, branch_mark.branch_index as usize);
            }
        }
        if let Frame::Combinator(frame) = &mut self.frames[depth] {
            frame.marks.clear();
        }
        Ok(())
    }

    fn pop_combinator(&mut self, depth: usize) -> Result<CombinatorStep, StructuredRuntimeError> {
        if !self.finish_tracked_item_if_direct_child(depth)? {
            return Ok(CombinatorStep::Rejected);
        }
        if depth == 0 {
            let branches = match &self.frames[depth] {
                Frame::Combinator(frame) => frame.cursors.len(),
                _ => unreachable!("expected a combinator frame"),
            };
            for index in 0..branches {
                let cursor = match &self.frames[depth] {
                    Frame::Combinator(frame)
                        if bit_is_set(&frame.active_words, index)
                            && self.cursors.is_accepting(frame.cursors[index]) =>
                    {
                        frame.cursors[index]
                    }
                    Frame::Combinator(_) => continue,
                    _ => unreachable!("expected a combinator frame"),
                };
                if let Some((is_object, _, _)) = self.cursors.accepted_annotation_summary(cursor)? {
                    let target = self.ensure_root_annotation_target(is_object)?;
                    self.merge_cursor_annotations(target, cursor)?;
                }
            }
        }
        self.reserve_undo(1)?;
        let popped = self.frames.pop().expect("checked above");
        let scope = self.frame_scopes.pop().expect("frame scope exists");
        self.push_undo(Undo::PushFrame(popped, scope))?;
        if depth == 0 {
            self.reserve_undo(1)?;
            self.root_closed = true;
            self.push_undo(Undo::SetRootClosed { old: false })?;
        }
        Ok(CombinatorStep::Popped)
    }

    fn start_unevaluated_candidate(
        &mut self,
        depth: usize,
        ordinal: u32,
    ) -> Result<(), StructuredRuntimeError> {
        let schema = match &self.frames[depth] {
            Frame::Unevaluated(frame) if frame.current_candidate.is_none() => frame.unevaluated,
            Frame::Unevaluated(_) => return Err(self.cursors.error()),
            _ => unreachable!("expected an unevaluated frame"),
        };
        let pool_mark = self.cursors.pool_mark()?;
        let cursor = self.start_cursor(schema)?;
        if let Err(error) = self.reserve_undo(1) {
            self.cursors.rollback_unpublished(cursor, pool_mark)?;
            return Err(error);
        }
        let Frame::Unevaluated(frame) = &mut self.frames[depth] else {
            unreachable!("expected an unevaluated frame")
        };
        frame.current_candidate = Some(UnevaluatedCandidate {
            ordinal,
            cursor,
            dead: false,
        });
        self.push_undo(Undo::StartUnevaluatedCandidate { depth, cursor })
    }

    fn finish_unevaluated_candidate(
        &mut self,
        depth: usize,
        accepts: bool,
    ) -> Result<(), StructuredRuntimeError> {
        let candidate = match &self.frames[depth] {
            Frame::Unevaluated(frame) => frame
                .current_candidate
                .ok_or_else(|| self.cursors.error())?,
            _ => unreachable!("expected an unevaluated frame"),
        };
        if accepts {
            self.insert_evaluated(
                AnnotationTarget::UnevaluatedCandidates(depth),
                candidate.ordinal,
            )?;
        }
        let old_count = match &self.frames[depth] {
            Frame::Unevaluated(frame) => frame.location_count,
            _ => unreachable!("expected an unevaluated frame"),
        };
        let (limit_kind, limit) = match &self.frames[depth] {
            Frame::Unevaluated(frame) if frame.kind == UnevaluatedKind::Properties => (
                LimitKind::PropertyCount,
                self.limits.max_properties as usize,
            ),
            Frame::Unevaluated(_) => (LimitKind::ArrayLength, self.limits.max_items as usize),
            _ => unreachable!("expected an unevaluated frame"),
        };
        let new_count = candidate
            .ordinal
            .checked_add(1)
            .ok_or_else(|| StructuredRuntimeError::new(limit_kind, usize::MAX, limit))?;
        if new_count as usize > limit {
            return Err(StructuredRuntimeError::new(
                limit_kind,
                new_count as usize,
                limit,
            ));
        }
        self.reserve_undo(1)?;
        let Frame::Unevaluated(frame) = &mut self.frames[depth] else {
            unreachable!("expected an unevaluated frame")
        };
        frame.current_candidate = None;
        frame.location_count = new_count;
        self.push_undo(Undo::FinishUnevaluatedCandidate {
            depth,
            ordinal: candidate.ordinal,
            cursor: candidate.cursor,
            dead: candidate.dead,
            old_count,
        })
    }

    fn finish_unevaluated_frame(
        &mut self,
        depth: usize,
    ) -> Result<UnevaluatedStep, StructuredRuntimeError> {
        let (kind, scope, count, matching_instance) = match &self.frames[depth] {
            Frame::Unevaluated(frame) => (
                frame.kind,
                frame.scope,
                frame.location_count,
                frame.instance_kind == Some(frame.kind),
            ),
            _ => unreachable!("expected an unevaluated frame"),
        };
        if !self.cursors.is_accepting(scope) {
            return Ok(UnevaluatedStep::Rejected);
        }
        if matching_instance {
            let word_count = (count as usize).div_ceil(64);
            for word_index in 0..word_count {
                let scope_word = self
                    .cursors
                    .accepted_annotation_word_for_kind(scope, kind, word_index)?;
                let candidate_word = match &self.frames[depth] {
                    Frame::Unevaluated(frame) => frame.candidates.word(word_index),
                    _ => unreachable!("expected an unevaluated frame"),
                };
                let covered = scope_word | candidate_word;
                let remaining = (count as usize) - word_index * 64;
                let required = if remaining >= 64 {
                    u64::MAX
                } else {
                    (1u64 << remaining) - 1
                };
                if covered & required != required {
                    return Ok(UnevaluatedStep::Rejected);
                }
            }
        }
        if depth == 0 {
            if matching_instance {
                let target =
                    self.ensure_root_annotation_target(kind == UnevaluatedKind::Properties)?;
                self.mark_all_evaluated(target)?;
            } else if let Some((is_object, _, _)) =
                self.cursors.accepted_annotation_summary(scope)?
            {
                let target = self.ensure_root_annotation_target(is_object)?;
                self.merge_cursor_annotations(target, scope)?;
            }
        }
        if !self.finish_tracked_item_if_direct_child(depth)? {
            return Ok(UnevaluatedStep::Rejected);
        }
        self.reserve_undo(1)?;
        let popped = self.frames.pop().expect("unevaluated frame exists");
        let scope = self.frame_scopes.pop().expect("frame scope exists");
        self.push_undo(Undo::PushFrame(popped, scope))?;
        if depth == 0 {
            self.reserve_undo(1)?;
            self.root_closed = true;
            self.push_undo(Undo::SetRootClosed { old: false })?;
        }
        Ok(UnevaluatedStep::Popped { replay: false })
    }

    fn route_unevaluated_byte(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<UnevaluatedStep, StructuredRuntimeError> {
        let (kind, scope, tracker, current) = match &self.frames[depth] {
            Frame::Unevaluated(frame) => (
                frame.kind,
                frame.scope,
                frame.tracker,
                frame.current_candidate,
            ),
            _ => unreachable!("expected an unevaluated frame"),
        };
        let start = self.cursors.any_candidate_start(tracker, kind, byte)?;
        let root_before_value = self.cursors.any_root_before_value(tracker)?;
        if root_before_value && !WS.contains(&byte) {
            let instance_kind = match byte {
                b'{' => Some(UnevaluatedKind::Properties),
                b'[' => Some(UnevaluatedKind::Items),
                _ => None,
            };
            if instance_kind.is_some()
                && matches!(&self.frames[depth], Frame::Unevaluated(frame) if frame.instance_kind.is_none())
            {
                self.reserve_undo(1)?;
                let Frame::Unevaluated(frame) = &mut self.frames[depth] else {
                    unreachable!("expected an unevaluated frame")
                };
                frame.instance_kind = instance_kind;
                self.push_undo(Undo::SetUnevaluatedInstanceKind { depth })?;
            }
        }
        if let Some(ordinal) = start {
            self.start_unevaluated_candidate(depth, ordinal)?;
        } else if current.is_none() {
            debug_assert!(
                matches!(&self.frames[depth], Frame::Unevaluated(frame) if frame.current_candidate.is_none())
            );
        }

        let scope_accepting_before = self.cursors.is_accepting(scope);
        if kind == UnevaluatedKind::Items && byte == b',' {
            let rejects_unevaluated = match self.plan.as_ref().node(match &self.frames[depth] {
                Frame::Unevaluated(frame) => frame.node,
                _ => return Err(self.cursors.error()),
            }) {
                NodePlan::Unevaluated(plan) => plan.rejects_unevaluated,
                _ => return Err(self.cursors.error()),
            };
            let next_item = self.cursors.any_root_next_array_item(tracker)?;
            let annotation_possible = self.cursors.next_array_item_annotation_possible(scope)?;
            if rejects_unevaluated
                && next_item.is_some()
                && annotation_possible.is_some_and(|possible| !possible)
            {
                return Ok(UnevaluatedStep::Rejected);
            }
        }
        let scope_mark = self.cursors.checkpoint(scope)?;
        let scope_step = self.push_cursor_byte(scope, byte)?;
        let scope_replay = match scope_step {
            CursorStep::Dead if scope_accepting_before => true,
            CursorStep::Dead => return Ok(UnevaluatedStep::Rejected),
            CursorStep::Alive | CursorStep::Complete => {
                self.reserve_undo(1)?;
                self.push_undo(Undo::RollbackUnevaluatedCursor {
                    cursor: scope,
                    mark: scope_mark,
                })?;
                false
            }
        };

        let tracker_accepting_before = self.cursors.is_accepting(tracker);
        let tracker_mark = self.cursors.checkpoint(tracker)?;
        let tracker_step = self.push_cursor_byte(tracker, byte)?;
        match tracker_step {
            CursorStep::Dead if tracker_accepting_before && scope_replay => {
                return self.finish_unevaluated_frame(depth).map(|step| match step {
                    UnevaluatedStep::Popped { .. } => UnevaluatedStep::Popped { replay: true },
                    other => other,
                });
            }
            CursorStep::Dead => return Ok(UnevaluatedStep::Rejected),
            CursorStep::Alive | CursorStep::Complete if scope_replay => {
                return Ok(UnevaluatedStep::Rejected);
            }
            CursorStep::Alive | CursorStep::Complete => {
                self.reserve_undo(1)?;
                self.push_undo(Undo::RollbackUnevaluatedCursor {
                    cursor: tracker,
                    mark: tracker_mark,
                })?;
            }
        }

        let finite_key_viable = {
            let node = match &self.frames[depth] {
                Frame::Unevaluated(frame) => frame.node,
                _ => return Err(self.cursors.error()),
            };
            let plan = match self.plan.as_ref().node(node) {
                NodePlan::Unevaluated(plan) => plan,
                _ => return Err(self.cursors.error()),
            };
            match (&plan.finite_evaluated_keys, kind) {
                (Some(trie), UnevaluatedKind::Properties) => self
                    .cursors
                    .any_root_key_viable(tracker, trie)?
                    .unwrap_or(true),
                _ => true,
            }
        };
        if !finite_key_viable {
            return Ok(UnevaluatedStep::Rejected);
        }

        let candidate = match &self.frames[depth] {
            Frame::Unevaluated(frame) => frame.current_candidate,
            _ => unreachable!("expected an unevaluated frame"),
        };
        let mut candidate_accepting_before = false;
        let mut candidate_step = None;
        if let Some(candidate) = candidate.filter(|candidate| !candidate.dead) {
            candidate_accepting_before = self.cursors.is_accepting(candidate.cursor);
            let mark = self.cursors.checkpoint(candidate.cursor)?;
            let step = self.push_cursor_byte(candidate.cursor, byte)?;
            if step != CursorStep::Dead {
                self.reserve_undo(1)?;
                self.push_undo(Undo::RollbackUnevaluatedCursor {
                    cursor: candidate.cursor,
                    mark,
                })?;
            }
            candidate_step = Some(step);
        }
        let open = self.cursors.any_candidate_open(tracker)?;
        if let Some(candidate) = candidate {
            if !open {
                let accepts = match candidate_step {
                    Some(CursorStep::Dead) => candidate_accepting_before,
                    Some(CursorStep::Alive | CursorStep::Complete) => {
                        self.cursors.is_accepting(candidate.cursor)
                    }
                    None => false,
                };
                self.finish_unevaluated_candidate(depth, accepts)?;
            } else if candidate_step == Some(CursorStep::Dead) && !candidate.dead {
                self.reserve_undo(1)?;
                let Frame::Unevaluated(frame) = &mut self.frames[depth] else {
                    unreachable!("expected an unevaluated frame")
                };
                frame
                    .current_candidate
                    .as_mut()
                    .expect("candidate remains open")
                    .dead = true;
                self.push_undo(Undo::MarkUnevaluatedCandidateDead { depth })?;
            }
        }
        if self.cursors.structured_root_closed(tracker)? {
            return self.finish_unevaluated_frame(depth);
        }
        Ok(UnevaluatedStep::Advanced)
    }

    fn route_object_byte(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<bool, StructuredRuntimeError> {
        match self.object_ref(depth).phase {
            ObjectPhase::Start => {
                if WS.contains(&byte) {
                    return Ok(true);
                }
                if byte != b'{' {
                    return Ok(false);
                }
                self.set_object_phase(depth, ObjectPhase::BeforeFirstKeyOrClose)?;
                Ok(true)
            }
            ObjectPhase::BeforeFirstKeyOrClose => {
                if WS.contains(&byte) {
                    return Ok(true);
                }
                if byte == b'"' {
                    if self
                        .object_key_prefix_viable(depth, "", None)
                        .is_some_and(|viable| !viable)
                    {
                        return Ok(false);
                    }
                    return self.start_key(depth);
                }
                if byte == b'}' {
                    return self.close_object(depth, byte);
                }
                Ok(false)
            }
            ObjectPhase::InKey => self.push_key_byte(depth, byte),
            ObjectPhase::AfterKeyBeforeColon => {
                if WS.contains(&byte) {
                    return Ok(true);
                }
                if byte != b':' {
                    return Ok(false);
                }
                self.resolve_at_colon(depth)
            }
            ObjectPhase::AfterColonBeforeValue => {
                if WS.contains(&byte) {
                    return Ok(true);
                }
                let Some(target) = self.object_mut(depth).resolved_value.take() else {
                    return Err(self.cursors.error());
                };
                let pushed = match &target {
                    ValueObligations::AnyJson => self.push_any_child(),
                    ValueObligations::Schemas(s) if s.len() == 1 => {
                        let Some(node) = s.iter().next() else {
                            self.object_mut(depth).resolved_value = Some(target);
                            return Err(self.cursors.error());
                        };
                        self.push_child(node)
                    }
                    ValueObligations::Schemas(s) => self.push_obligation_child(s),
                };
                let pushed = match pushed {
                    Ok(true) => true,
                    Ok(false) => {
                        self.object_mut(depth).resolved_value = Some(target);
                        return Ok(false);
                    }
                    Err(error) => {
                        self.object_mut(depth).resolved_value = Some(target);
                        return Err(error);
                    }
                };
                debug_assert!(pushed);
                // Reserve all parent undo records after read-only depth checks.
                if let Err(error) = self.reserve_undo(3) {
                    self.object_mut(depth).resolved_value = Some(target);
                    return Err(error);
                }
                self.push_undo(Undo::RestoreResolvedValue {
                    depth,
                    old: Some(target),
                })?;
                self.set_object_phase(depth, ObjectPhase::AfterValueBeforeCommaOrClose)?;
                let old_count = self.object_ref(depth).property_count;
                let new_count = old_count.checked_add(1).ok_or_else(|| {
                    StructuredRuntimeError::new(
                        LimitKind::PropertyCount,
                        usize::MAX,
                        self.limits.max_properties as usize,
                    )
                })?;
                if new_count as usize > self.limits.max_properties as usize {
                    return Err(StructuredRuntimeError::new(
                        LimitKind::PropertyCount,
                        new_count as usize,
                        self.limits.max_properties as usize,
                    ));
                }
                self.reserve_undo(1)?;
                self.object_mut(depth).property_count = new_count;
                self.push_undo(Undo::RestorePropertyCount {
                    depth,
                    old: old_count,
                })?;
                self.route_byte(byte)
            }
            ObjectPhase::AfterValueBeforeCommaOrClose => {
                if WS.contains(&byte) {
                    return Ok(true);
                }
                if byte == b',' {
                    let object = self.object_ref(depth);
                    let plan = super::plan::object_plan_of(self.plan.as_ref(), object.node);
                    if plan
                        .max_properties
                        .is_some_and(|maximum| object.property_count >= maximum)
                        || self
                            .object_key_prefix_viable(depth, "", None)
                            .is_some_and(|viable| !viable)
                    {
                        return Ok(false);
                    }
                    self.set_object_phase(depth, ObjectPhase::AfterCommaBeforeKey)?;
                    return Ok(true);
                }
                if byte == b'}' {
                    return self.close_object(depth, byte);
                }
                Ok(false)
            }
            ObjectPhase::AfterCommaBeforeKey => {
                if WS.contains(&byte) {
                    return Ok(true);
                }
                if byte != b'"' {
                    return Ok(false);
                }
                if self
                    .object_key_prefix_viable(depth, "", None)
                    .is_some_and(|viable| !viable)
                {
                    return Ok(false);
                }
                self.start_key(depth)
            }
        }
    }

    fn push_key_byte(&mut self, depth: usize, byte: u8) -> Result<bool, StructuredRuntimeError> {
        let old_decoder = self.object_ref(depth).key_decoder;
        if byte == b'"' && old_decoder.at_boundary() {
            if let Some(cursor) = self.object_ref(depth).property_name {
                self.reserve_undo(2)?;
                let mark = self.cursors.checkpoint(cursor)?;
                if self.push_cursor_byte(cursor, byte)? == CursorStep::Dead {
                    return Ok(false);
                }
                self.push_undo(Undo::RollbackPropertyName { cursor, mark })?;
                if !self.cursors.is_accepting(cursor) {
                    return Ok(false);
                }
                self.object_mut(depth).property_name = None;
                self.push_undo(Undo::FinishPropertyName { depth, cursor })?;
            }
            if self.key_is_duplicate(depth)? {
                return Ok(false);
            }
            if self
                .object_complete_key_viable(depth)
                .is_some_and(|viable| !viable)
            {
                return Ok(false);
            }
            self.record_dependent_required_name(depth)?;
            if !self.trigger_dependent_schema(depth)? {
                return Ok(false);
            }
            self.set_object_phase(depth, ObjectPhase::AfterKeyBeforeColon)?;
            return Ok(true);
        }
        if let Some(cursor) = self.object_ref(depth).property_name {
            self.reserve_undo(1)?;
            let mark = self.cursors.checkpoint(cursor)?;
            if self.push_cursor_byte(cursor, byte)? == CursorStep::Dead {
                return Ok(false);
            }
            self.push_undo(Undo::RollbackPropertyName { cursor, mark })?;
        }
        let mut decoder = old_decoder;
        let step = decoder.push(byte);
        if matches!(step, DecodeStep::Invalid) {
            return Ok(false);
        }
        if let DecodeStep::Scalar(c) = step {
            let old_len = self.candidate_key_len(depth);
            let added = c.len_utf8();
            let next_len = old_len.checked_add(added).ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::KeyBytes,
                    usize::MAX,
                    self.limits.max_key_bytes,
                )
            })?;
            if next_len > self.limits.max_key_bytes {
                return Err(StructuredRuntimeError::new(
                    LimitKind::KeyBytes,
                    next_len,
                    self.limits.max_key_bytes,
                ));
            }
            self.reserve_undo(1)?;
            let cap = self.local_session_budget()?;
            let owner = self.key_owner(depth)?;
            let lease_generation = (self.key_lease.phase == KeyLeasePhase::Prepared
                && self.key_lease.owner == Some(owner))
            .then_some(self.key_lease.generation);
            let StructuredState {
                frames,
                key_lease,
                session_memory,
                ..
            } = &mut *self;
            let key = if lease_generation.is_some() {
                &mut key_lease.buffer
            } else {
                let Frame::Object(obj) = &mut frames[depth] else {
                    unreachable!("expected an object frame")
                };
                &mut obj.key
            };
            let req = GrowthRequest {
                len: key.len(),
                capacity: key.capacity(),
                additional: added,
                item_size: 1,
            };
            grow_bounded(session_memory, cap, LimitKind::SessionBytes, req, |n| {
                key.try_reserve_exact(n)
                    .map(|_| key.capacity())
                    .map_err(|_| ())
            })?;
            key.push(c);
            self.push_undo(Undo::TruncateKey {
                depth,
                old_len,
                lease_generation,
            })?;
        }
        self.reserve_undo(1)?;
        self.object_mut(depth).key_decoder = decoder;
        self.push_undo(Undo::RestoreDecoder {
            depth,
            old: old_decoder,
        })?;
        if self
            .current_object_key_viable(depth)
            .is_some_and(|viable| !viable)
        {
            return Ok(false);
        }
        Ok(true)
    }

    fn current_object_key_viable(&self, depth: usize) -> Option<bool> {
        let object = self.object_ref(depth);
        let prefix = self.candidate_key(depth);
        if object.key_decoder.at_boundary() {
            self.object_key_prefix_viable(depth, prefix, None)
        } else {
            object
                .key_decoder
                .any_pending_scalar_satisfies_with_supplementary(
                    4096,
                    |scalar| {
                        self.object_key_prefix_viable(depth, prefix, Some(scalar))
                            .unwrap_or(true)
                    },
                    || self.object_key_any_scalar_viable(depth, prefix),
                    |start, end| self.object_key_supplementary_viable(depth, prefix, start, end),
                )
        }
    }

    fn object_key_supplementary_viable(
        &self,
        depth: usize,
        prefix: &str,
        start: u32,
        end: u32,
    ) -> Option<bool> {
        let object = self.object_ref(depth);
        let plan = super::plan::object_plan_of(self.plan.as_ref(), object.node);
        if !matches!(plan.additional, AdditionalPlan::Forbid) {
            return None;
        }
        let exact_viable = plan.known_by_name.iter().any(|(name, id)| {
            let index = id.0 as usize;
            plan.known
                .get(index)
                .is_some_and(|property| property.maybe_satisfiable)
                && !bit_is_set(&object.seen_known, index)
                && name.strip_prefix(prefix).is_some_and(|rest| {
                    rest.chars()
                        .next()
                        .is_some_and(|scalar| (start..=end).contains(&u32::from(scalar)))
                })
                && !plan.patterns.iter().any(|pattern| {
                    !pattern.maybe_satisfiable && pattern.engine.accepts(name.as_bytes())
                })
        });
        if exact_viable {
            return Some(true);
        }
        let Some(pattern) = plan.viable_pattern.as_deref() else {
            return Some(false);
        };
        let Some(state) = pattern.consume_token(pattern.start(), prefix.as_bytes()) else {
            return Some(false);
        };
        pattern.can_consume_supplementary_range(state, start, end)
    }

    fn object_key_any_scalar_viable(&self, depth: usize, prefix: &str) -> Option<bool> {
        let object = self.object_ref(depth);
        let plan = super::plan::object_plan_of(self.plan.as_ref(), object.node);
        if !matches!(plan.additional, AdditionalPlan::Forbid) {
            return None;
        }
        if plan.known_by_name.iter().any(|(name, id)| {
            let index = id.0 as usize;
            plan.known
                .get(index)
                .is_some_and(|property| property.maybe_satisfiable)
                && !bit_is_set(&object.seen_known, index)
                && name
                    .strip_prefix(prefix)
                    .is_some_and(|rest| !rest.is_empty())
                && !plan.patterns.iter().any(|pattern| {
                    !pattern.maybe_satisfiable && pattern.engine.accepts(name.as_bytes())
                })
        }) {
            return Some(true);
        }
        let Some(pattern) = plan.viable_pattern.as_deref() else {
            return Some(false);
        };
        let Some(state) = pattern.consume_token(pattern.start(), prefix.as_bytes()) else {
            return Some(false);
        };
        let unavailable = object
            .seen_dynamic
            .values()
            .flat_map(|slot| match slot {
                DynamicSlot::One(id) => std::slice::from_ref(id),
                DynamicSlot::Many(ids) => ids.as_slice(),
            })
            .filter(|&&id| {
                self.key_arena.get(id).is_some_and(|name| {
                    name.starts_with(prefix) && pattern.accepts(name.as_bytes())
                })
            })
            .count();
        match pattern.can_consume_any_scalar(state) {
            Some(true) if unavailable == 0 => Some(true),
            Some(true) | None => None,
            Some(false) => Some(false),
        }
    }

    fn object_complete_key_viable(&self, depth: usize) -> Option<bool> {
        let object = self.object_ref(depth);
        let plan = super::plan::object_plan_of(self.plan.as_ref(), object.node);
        if !matches!(plan.additional, AdditionalPlan::Forbid) {
            return None;
        }
        let key = self.candidate_key(depth);
        let known = plan
            .known_by_name
            .get(key)
            .and_then(|id| plan.known.get(id.0 as usize));
        let matching_patterns = plan
            .patterns
            .iter()
            .filter(|entry| entry.engine.accepts(key.as_bytes()));
        let mut pattern_matches = false;
        let mut impossible_pattern = false;
        for pattern in matching_patterns {
            pattern_matches = true;
            impossible_pattern |= !pattern.maybe_satisfiable;
        }
        if known.is_some_and(|property| !property.maybe_satisfiable) || impossible_pattern {
            return Some(false);
        }
        Some(known.is_some() || pattern_matches)
    }

    /// Optionally proves a prefix can still reach an unseen closed-object key.
    fn object_key_prefix_viable(
        &self,
        depth: usize,
        prefix: &str,
        scalar: Option<char>,
    ) -> Option<bool> {
        let object = self.object_ref(depth);
        let plan = super::plan::object_plan_of(self.plan.as_ref(), object.node);
        if !matches!(plan.additional, AdditionalPlan::Forbid) {
            return None;
        }
        let mut encoded = [0u8; 4];
        let suffix = scalar.map(|value| value.encode_utf8(&mut encoded) as &str);
        let extends_prefix = |name: &str| {
            name.strip_prefix(prefix)
                .is_some_and(|rest| suffix.is_none_or(|suffix| rest.starts_with(suffix)))
        };
        let exact_viable = plan.known_by_name.iter().any(|(name, id)| {
            let index = id.0 as usize;
            plan.known
                .get(index)
                .is_some_and(|property| property.maybe_satisfiable)
                && extends_prefix(name)
                && !bit_is_set(&object.seen_known, index)
                && !plan.patterns.iter().any(|pattern| {
                    !pattern.maybe_satisfiable && pattern.engine.accepts(name.as_bytes())
                })
        });
        if exact_viable {
            return Some(true);
        }
        let Some(pattern) = plan.viable_pattern.as_deref() else {
            return Some(false);
        };
        let Some(state) = pattern
            .consume_token(pattern.start(), prefix.as_bytes())
            .and_then(|state| {
                suffix.map_or(Some(state), |suffix| {
                    pattern.consume_token(state, suffix.as_bytes())
                })
            })
        else {
            return Some(false);
        };
        let unavailable_known = plan
            .known_by_name
            .iter()
            .filter(|(name, id)| {
                (bit_is_set(&object.seen_known, id.0 as usize)
                    || plan
                        .known
                        .get(id.0 as usize)
                        .is_some_and(|property| !property.maybe_satisfiable))
                    && extends_prefix(name)
                    && pattern.accepts(name.as_bytes())
            })
            .count();
        let seen_dynamic = object
            .seen_dynamic
            .values()
            .flat_map(|slot| match slot {
                DynamicSlot::One(id) => std::slice::from_ref(id),
                DynamicSlot::Many(ids) => ids.as_slice(),
            })
            .filter(|&&id| {
                self.key_arena
                    .get(id)
                    .is_some_and(|name| extends_prefix(name) && pattern.accepts(name.as_bytes()))
            })
            .count();
        let unavailable = unavailable_known.saturating_add(seen_dynamic) as u64;
        Some(pattern.residual_cardinality(state) > unavailable)
    }

    fn key_is_duplicate(&self, depth: usize) -> Result<bool, StructuredRuntimeError> {
        let object = self.object_ref(depth);
        let plan = super::plan::object_plan_of(self.plan.as_ref(), object.node);
        let key = self.candidate_key(depth);
        if let Some(id) = plan.known_by_name.get(key) {
            return Ok(bit_is_set(&object.seen_known, id.0 as usize));
        }
        let hash = self.key_hasher.hash_one(key.as_bytes());
        Ok(classify_key_intern(
            &self.key_arena,
            &object.seen_dynamic,
            key,
            hash,
            &self.limits,
        )?
        .is_none())
    }

    fn trigger_dependent_schema(&mut self, depth: usize) -> Result<bool, StructuredRuntimeError> {
        let dependency = {
            let object = self.object_ref(depth);
            let key = self.candidate_key(depth);
            super::plan::object_plan_of(self.plan.as_ref(), object.node)
                .dependent_by_name
                .get(key)
                .copied()
        };
        let Some(dependency) = dependency else {
            return Ok(true);
        };
        let index = usize::try_from(dependency).map_err(|_| self.cursors.error())?;
        let (len, already_triggered, active) = self
            .object_ref(depth)
            .dependent
            .as_ref()
            .map(|frame| {
                (
                    frame.cursors.len(),
                    index < frame.cursors.len() && bit_is_set(&frame.triggered_words, index),
                    index < frame.cursors.len() && bit_is_set(&frame.active_words, index),
                )
            })
            .ok_or_else(|| self.cursors.error())?;
        if index >= len {
            return Err(self.cursors.error());
        }
        if !already_triggered {
            self.reserve_undo(1)?;
            if let Some(frame) = self.object_mut(depth).dependent.as_mut() {
                set_bit(&mut frame.triggered_words, index);
            }
            self.push_undo(Undo::ClearDependentTrigger { depth, dependency })?;
        }
        Ok(active)
    }

    fn record_dependent_required_name(
        &mut self,
        depth: usize,
    ) -> Result<(), StructuredRuntimeError> {
        let name = {
            let object = self.object_ref(depth);
            let key = self.candidate_key(depth);
            super::plan::object_plan_of(self.plan.as_ref(), object.node)
                .dependent_required
                .as_ref()
                .and_then(|plan| plan.name_by_text.get(key))
                .copied()
        };
        let Some(name) = name else {
            return Ok(());
        };
        let index = usize::try_from(name).map_err(|_| self.cursors.error())?;
        if self
            .object_ref(depth)
            .dependent_required_seen
            .contains(index)
        {
            return Err(self.cursors.error());
        }
        self.reserve_undo(1)?;
        self.object_mut(depth).dependent_required_seen.set(index);
        self.push_undo(Undo::ClearDependentRequiredPresence { depth, name })
    }

    /// Adds `id` to `set` if not already present, charging real spill growth and enforcing
    /// `max_active_validators`. Inline slots (0-4) are free: no reservation, no charge.
    fn insert_obligation_uncounted(
        set: &mut SchemaObligationSet,
        id: NodeId,
        max_active_validators: usize,
        max_session_bytes: usize,
    ) -> Result<(), StructuredRuntimeError> {
        if set.contains(id) {
            return Ok(());
        }
        if (set.inline_len as usize) < MAX_INLINE_OBLIGATIONS {
            set.inline[set.inline_len as usize] = id;
            set.inline_len += 1;
            return Ok(());
        }
        let next_len = set.len().checked_add(1).ok_or_else(|| {
            StructuredRuntimeError::new(
                LimitKind::ActiveValidators,
                usize::MAX,
                max_active_validators,
            )
        })?;
        if next_len > max_active_validators {
            return Err(StructuredRuntimeError::new(
                LimitKind::ActiveValidators,
                next_len,
                max_active_validators,
            ));
        }
        let spill = &mut set.spill;
        spill.try_reserve_exact(1).map_err(|_| {
            StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, max_session_bytes)
        })?;
        set.spill.push(id);
        Ok(())
    }

    #[cfg(test)]
    fn insert_obligation(
        &mut self,
        set: &mut SchemaObligationSet,
        id: NodeId,
    ) -> Result<(), StructuredRuntimeError> {
        if set.spill.len() == set.spill.capacity() && !self.session_would_fit(size_of::<NodeId>()) {
            return Err(StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                self.session_memory
                    .live()
                    .saturating_add(size_of::<NodeId>()),
                self.limits.max_session_bytes,
            ));
        }
        let before = set.spill.capacity();
        Self::insert_obligation_uncounted(
            set,
            id,
            self.limits.max_active_validators,
            self.limits.max_session_bytes,
        )?;
        let growth = set
            .spill
            .capacity()
            .saturating_sub(before)
            .saturating_mul(size_of::<NodeId>());
        if !self.session_would_fit(growth) {
            return Err(StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                self.session_memory.live().saturating_add(growth),
                self.limits.max_session_bytes,
            ));
        }
        self.charge_session_bytes_after_precheck(growth)
    }

    /// Gathers exact, pattern, or additional-property schemas for a key.
    /// Pattern schemas are tested independently.
    fn resolve_property(
        &mut self,
        depth: usize,
    ) -> Result<Option<Resolution>, StructuredRuntimeError> {
        let node = self.object_ref(depth).node;
        let op = super::plan::object_plan_of(self.plan.as_ref(), node);
        let known_id = {
            let key = self.candidate_key(depth);
            op.known_by_name.get(key).copied()
        };
        let mut schemas = SchemaObligationSet::default();
        let max_active_validators = self.limits.max_active_validators;
        let max_session_bytes = self.limits.max_session_bytes;
        let mut any_matched = false;
        if let Some(id) = known_id {
            let value = op.known[id.0 as usize].value;
            Self::insert_obligation_uncounted(
                &mut schemas,
                value,
                max_active_validators,
                max_session_bytes,
            )?;
            any_matched = true;
        }
        let pattern_work = op
            .patterns
            .len()
            .checked_mul(self.candidate_key_len(depth))
            .ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::MaskWork,
                    usize::MAX,
                    usize::try_from(self.limits.max_mask_work).unwrap_or(usize::MAX),
                )
            })?;
        let max_pattern_work = usize::try_from(self.limits.max_mask_work).unwrap_or(usize::MAX);
        if pattern_work > max_pattern_work {
            return Err(StructuredRuntimeError::new(
                LimitKind::MaskWork,
                pattern_work,
                max_pattern_work,
            ));
        }
        for pattern in &op.patterns[..] {
            let matches = {
                let key = self.candidate_key(depth);
                super::pattern::property_pattern_matches(&pattern.engine, key.as_bytes())
            };
            if matches {
                Self::insert_obligation_uncounted(
                    &mut schemas,
                    pattern.value,
                    max_active_validators,
                    max_session_bytes,
                )?;
                any_matched = true;
            }
        }
        let (obligations, annotates) = if any_matched {
            (ValueObligations::Schemas(schemas), true)
        } else {
            match op.additional {
                AdditionalPlan::Schema(target) => {
                    Self::insert_obligation_uncounted(
                        &mut schemas,
                        target,
                        max_active_validators,
                        max_session_bytes,
                    )?;
                    (ValueObligations::Schemas(schemas), true)
                }
                AdditionalPlan::Open => (ValueObligations::AnyJson, false),
                AdditionalPlan::AllowAny => (ValueObligations::AnyJson, true),
                AdditionalPlan::Forbid => return Ok(None),
            }
        };
        Ok(Some(Resolution {
            known_id,
            obligations,
            annotates,
        }))
    }

    fn resolve_at_colon(&mut self, depth: usize) -> Result<bool, StructuredRuntimeError> {
        let Some(resolution) = self.resolve_property(depth)? else {
            return Ok(false);
        };
        if resolution.annotates
            && self
                .plan
                .as_ref()
                .annotations_required(self.object_ref(depth).node)
        {
            let ordinal = self.object_ref(depth).property_count;
            self.insert_evaluated(AnnotationTarget::Object(depth), ordinal)?;
        }
        let obligations = if let Some(id) = resolution.known_id {
            let idx = id.0 as usize;
            if bit_is_set(&self.object_ref(depth).seen_known, idx) {
                return Ok(false);
            }
            self.reserve_undo(1)?;
            set_bit(&mut self.object_mut(depth).seen_known, idx);
            self.push_undo(Undo::RemoveSeenKnown { depth, id })?;
            let word = idx / BITSET_WORD_BITS;
            let old = self.object_ref(depth).missing_required[word];
            let new = old & !(1u64 << (idx % BITSET_WORD_BITS));
            if new != old {
                self.reserve_undo(1)?;
                self.object_mut(depth).missing_required[word] = new;
                self.push_undo(Undo::RestoreRequiredWord { depth, word, old })?;
            }
            resolution.obligations
        } else {
            match self.intern_dynamic_key(depth, resolution.obligations)? {
                Some(obligations) => obligations,
                None => return Ok(false),
            }
        };
        self.reserve_undo(1)?;
        let owned_bytes = obligation_owned_bytes(&obligations);
        if !self.session_would_fit(owned_bytes) {
            let observed = self
                .retained_bytes()
                .checked_add(owned_bytes)
                .unwrap_or(usize::MAX);
            return Err(StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                observed,
                self.limits.max_session_bytes,
            ));
        }
        self.charge_session_bytes_after_precheck(owned_bytes)
            .map_err(|_| {
                let observed = self
                    .retained_bytes()
                    .checked_add(owned_bytes)
                    .unwrap_or(usize::MAX);
                StructuredRuntimeError::new(
                    LimitKind::SessionBytes,
                    observed,
                    self.limits.max_session_bytes,
                )
            })?;
        let old = self.object_mut(depth).resolved_value.replace(obligations);
        self.push_undo(Undo::RestoreResolvedValue { depth, old })?;
        self.set_object_phase(depth, ObjectPhase::AfterColonBeforeValue)?;
        Ok(true)
    }

    /// Interns the current key into `key_arena`. `Ok(None)` means a genuine duplicate.
    fn intern_dynamic_key(
        &mut self,
        depth: usize,
        obligations: ValueObligations,
    ) -> Result<Option<ValueObligations>, StructuredRuntimeError> {
        let limits = self.limits;
        let hash = self
            .key_hasher
            .hash_one(self.candidate_key(depth).as_bytes());
        let candidate = {
            let obj = self.object_ref(depth);
            classify_key_intern(
                &self.key_arena,
                &obj.seen_dynamic,
                self.candidate_key(depth),
                hash,
                &limits,
            )?
        };
        let Some((id, start, len, transition)) = candidate else {
            return Ok(None);
        };
        let key_len = self.candidate_key_len(depth);

        self.reserve_undo(2)?;
        let session_cap = self.local_session_budget()?;
        let StructuredState {
            key_arena,
            frames,
            session_memory,
            ..
        } = &mut *self;
        let Frame::Object(obj) = &mut frames[depth] else {
            unreachable!("expected an object frame")
        };
        let prebuilt_many = reserve_key_intern_growth(
            key_arena,
            &mut obj.seen_dynamic,
            hash,
            key_len,
            transition,
            session_cap,
            session_memory,
        )?;

        let old_arena_len = self.key_arena.bytes.len() as u32;
        let old_span_count = self.key_arena.spans.len() as u32;
        let owner = self.key_owner(depth)?;
        let StructuredState {
            key_arena,
            frames,
            key_lease,
            ..
        } = &mut *self;
        let Frame::Object(obj) = &mut frames[depth] else {
            unreachable!("expected an object frame")
        };
        let key = if key_lease.phase == KeyLeasePhase::Prepared && key_lease.owner == Some(owner) {
            key_lease.buffer.as_str()
        } else {
            obj.key.as_str()
        };
        key_arena.bytes.push_str(key);
        key_arena.spans.push((start, len));
        self.push_undo(Undo::InternKey {
            old_arena_len,
            old_span_count,
        })?;

        match transition {
            SlotTransition::ToOne => {
                self.object_mut(depth)
                    .seen_dynamic
                    .insert(hash, DynamicSlot::One(id));
                self.push_undo(Undo::RemoveSeenDynamicBucket { depth, hash })?;
            }
            SlotTransition::ToMany2 { existing } => {
                let mut v = prebuilt_many.expect("reserved above");
                v.push(existing);
                v.push(id);
                self.object_mut(depth)
                    .seen_dynamic
                    .insert(hash, DynamicSlot::Many(v));
                self.push_undo(Undo::RestoreSeenDynamicOne {
                    depth,
                    hash,
                    id: existing,
                })?;
            }
            SlotTransition::ToManyPush { old_len } => {
                let obj = self.object_mut(depth);
                let Some(DynamicSlot::Many(ids)) = obj.seen_dynamic.get_mut(&hash) else {
                    unreachable!("classified as ToManyPush")
                };
                ids.push(id);
                self.push_undo(Undo::TruncateSeenDynamicMany {
                    depth,
                    hash,
                    old_len,
                })?;
            }
        }
        Ok(Some(obligations))
    }

    /// The `AnyJson` object path's equivalent of `intern_dynamic_key`, sharing the same
    /// classify/reserve helpers but its own accessor and `Any*` undo variants.
    fn intern_any_dynamic_key(&mut self, depth: usize) -> Result<bool, StructuredRuntimeError> {
        let limits = self.limits;
        let hash = self
            .key_hasher
            .hash_one(self.candidate_key(depth).as_bytes());
        let obj = self.any_object_ref(depth);
        let Some((id, start, len, transition)) = classify_key_intern(
            &self.key_arena,
            &obj.seen_dynamic,
            self.candidate_key(depth),
            hash,
            &limits,
        )?
        else {
            return Ok(false);
        };
        let key_len = self.candidate_key_len(depth);

        self.reserve_undo(2)?;
        let session_cap = self.local_session_budget()?;
        let obj_id = match &self.frames[depth] {
            Frame::Any(AnyFrame::Object(id)) => *id,
            _ => unreachable!("expected an any-object frame"),
        };
        let StructuredState {
            key_arena,
            any_object_arena,
            session_memory,
            ..
        } = &mut *self;
        let obj = &mut any_object_arena[obj_id.0 as usize];
        let prebuilt_many = reserve_key_intern_growth(
            key_arena,
            &mut obj.seen_dynamic,
            hash,
            key_len,
            transition,
            session_cap,
            session_memory,
        )?;

        let old_arena_len = self.key_arena.bytes.len() as u32;
        let old_span_count = self.key_arena.spans.len() as u32;
        let owner = self.key_owner(depth)?;
        let StructuredState {
            key_arena,
            any_object_arena,
            key_lease,
            ..
        } = &mut *self;
        let obj = &mut any_object_arena[obj_id.0 as usize];
        let key = if key_lease.phase == KeyLeasePhase::Prepared && key_lease.owner == Some(owner) {
            key_lease.buffer.as_str()
        } else {
            obj.key.as_str()
        };
        key_arena.bytes.push_str(key);
        key_arena.spans.push((start, len));
        self.push_undo(Undo::InternKey {
            old_arena_len,
            old_span_count,
        })?;

        match transition {
            SlotTransition::ToOne => {
                self.any_object_mut(depth)
                    .seen_dynamic
                    .insert(hash, DynamicSlot::One(id));
                self.push_undo(Undo::AnyRemoveSeenDynamicBucket { depth, hash })?;
            }
            SlotTransition::ToMany2 { existing } => {
                let mut v = prebuilt_many.expect("reserved above");
                v.push(existing);
                v.push(id);
                self.any_object_mut(depth)
                    .seen_dynamic
                    .insert(hash, DynamicSlot::Many(v));
                self.push_undo(Undo::AnyRestoreSeenDynamicOne {
                    depth,
                    hash,
                    id: existing,
                })?;
            }
            SlotTransition::ToManyPush { old_len } => {
                let obj = self.any_object_mut(depth);
                let Some(DynamicSlot::Many(ids)) = obj.seen_dynamic.get_mut(&hash) else {
                    unreachable!("classified as ToManyPush")
                };
                ids.push(id);
                self.push_undo(Undo::AnyTruncateSeenDynamicMany {
                    depth,
                    hash,
                    old_len,
                })?;
            }
        }
        Ok(true)
    }

    fn set_array_phase(
        &mut self,
        depth: usize,
        new: ArrayPhase,
    ) -> Result<(), StructuredRuntimeError> {
        self.reserve_undo(1)?;
        let old = std::mem::replace(&mut self.array_mut(depth).phase, new);
        self.push_undo(Undo::SetArrayPhase { depth, old })
    }

    /// What the item at `index` must satisfy, or `None` for a closed tuple past its prefix.
    fn array_item_schema(&self, depth: usize, index: u32) -> Option<super::plan::TailPlan> {
        let ap = super::plan::array_plan_of(self.plan.as_ref(), self.array_ref(depth).node);
        match ap.prefix.get(index as usize) {
            Some(&id) => Some(super::plan::TailPlan::Schema(id)),
            None => match ap.tail {
                super::plan::TailPlan::Closed => None,
                other => Some(other),
            },
        }
    }

    fn route_array_byte(&mut self, depth: usize, byte: u8) -> Result<bool, StructuredRuntimeError> {
        match self.array_ref(depth).phase {
            ArrayPhase::Start => {
                let plan =
                    super::plan::array_plan_of(self.plan.as_ref(), self.array_ref(depth).node);
                if plan
                    .finite_unique_capacity
                    .is_some_and(|capacity| plan.min_items > capacity)
                {
                    return Ok(false);
                }
                if WS.contains(&byte) {
                    return Ok(true);
                }
                if byte != b'[' {
                    return Ok(false);
                }
                self.set_array_phase(depth, ArrayPhase::BeforeFirstItemOrClose)?;
                Ok(true)
            }
            ArrayPhase::BeforeFirstItemOrClose => {
                if WS.contains(&byte) {
                    return Ok(true);
                }
                if byte == b']' {
                    return self.close_array(depth, byte);
                }
                self.push_array_item(depth, byte)
            }
            ArrayPhase::AfterValueBeforeCommaOrClose => {
                if WS.contains(&byte) {
                    return Ok(true);
                }
                if byte == b',' {
                    if !self.next_array_item_viable(depth) {
                        return Ok(false);
                    }
                    self.set_array_phase(depth, ArrayPhase::AfterCommaBeforeValue)?;
                    return Ok(true);
                }
                if byte == b']' {
                    return self.close_array(depth, byte);
                }
                Ok(false)
            }
            ArrayPhase::AfterCommaBeforeValue => {
                if WS.contains(&byte) {
                    return Ok(true);
                }
                self.push_array_item(depth, byte)
            }
        }
    }

    fn next_array_item_viable(&self, depth: usize) -> bool {
        let array = self.array_ref(depth);
        let Some(schema) = self.array_item_schema(depth, array.index) else {
            return true;
        };
        let super::plan::TailPlan::Schema(node) = schema else {
            return true;
        };
        let NodePlan::StringEnum(members) = self.plan.as_ref().node(node) else {
            return true;
        };
        let Some(seen) = array.canonical_ref() else {
            return true;
        };
        (0..members.member_count()).any(|index| {
            members
                .member(index)
                .is_some_and(|member| !seen.contains_string(member))
        })
    }

    fn next_root_array_item_annotation_possible(&self) -> Option<bool> {
        let Some(Frame::Array(array)) = self.frames.first() else {
            return None;
        };
        if array.phase != ArrayPhase::AfterValueBeforeCommaOrClose {
            return None;
        }
        let plan = super::plan::array_plan_of(self.plan.as_ref(), array.node);
        let ordinal = array.index as usize;
        if ordinal < plan.prefix.len() || plan.tail_annotates {
            return Some(true);
        }
        let Some(contains) = &plan.contains else {
            return Some(false);
        };
        if contains.max == Some(array.contains_count) {
            return Some(false);
        }
        Some(!matches!(contains.matches, ContainsMatch::Never))
    }

    fn push_array_item(&mut self, depth: usize, byte: u8) -> Result<bool, StructuredRuntimeError> {
        let index = self.array_ref(depth).index;
        let ap = super::plan::array_plan_of(self.plan.as_ref(), self.array_ref(depth).node);
        if ap.max_items.is_some_and(|max| index >= max) {
            return Ok(false);
        }
        let Some(schema) = self.array_item_schema(depth, index) else {
            return Ok(false);
        };
        let annotates = (index as usize) < ap.prefix.len() || ap.tail_annotates;
        if annotates
            && self
                .plan
                .as_ref()
                .annotations_required(self.array_ref(depth).node)
        {
            self.insert_evaluated(AnnotationTarget::Array(depth), index)?;
        }
        let next_index = index.checked_add(1).ok_or_else(|| {
            StructuredRuntimeError::new(
                LimitKind::ArrayLength,
                usize::MAX,
                self.limits.max_items as usize,
            )
        })?;
        self.reserve_undo(1)?;
        self.array_mut(depth).index = next_index;
        self.push_undo(Undo::RestoreArrayIndex { depth, old: index })?;
        self.set_array_phase(depth, ArrayPhase::AfterValueBeforeCommaOrClose)?;
        self.start_item_tracking(depth)?;
        let pushed = match schema {
            super::plan::TailPlan::Schema(id) => self.push_child(id)?,
            super::plan::TailPlan::AllowAny => self.push_any_child()?,
            super::plan::TailPlan::Closed => unreachable!("array_item_schema excludes Closed"),
        };
        if !pushed {
            return Ok(false);
        }
        self.route_byte(byte)
    }

    fn close_array(&mut self, depth: usize, byte: u8) -> Result<bool, StructuredRuntimeError> {
        let ap = super::plan::array_plan_of(self.plan.as_ref(), self.array_ref(depth).node);
        let annotations_required = self
            .plan
            .as_ref()
            .annotations_required(self.array_ref(depth).node);
        let index = self.array_ref(depth).index;
        if index < ap.min_items {
            return Ok(false);
        }
        if let Some(cp) = &ap.contains {
            let count = self.array_ref(depth).contains_count;
            if count < cp.min || cp.max.is_some_and(|max| count > max) {
                return Ok(false);
            }
        }
        self.record_item_byte(depth, byte)?;
        if !self.finish_tracked_item_if_direct_child(depth)? {
            return Ok(false);
        }
        if depth == 0 && annotations_required {
            self.publish_root_annotations(AnnotationTarget::Array(depth))?;
        }
        self.reserve_undo(1)?;
        let popped = self.frames.pop().expect("checked by caller");
        let scope = self.frame_scopes.pop().expect("frame scope exists");
        self.push_undo(Undo::PushFrame(popped, scope))?;
        if depth == 0 {
            self.reserve_undo(1)?;
            self.root_closed = true;
            self.push_undo(Undo::SetRootClosed { old: false })?;
        }
        Ok(true)
    }

    /// Whether the item currently open at `depth` (an array frame) is tracked for
    /// `uniqueItems`/`contains` - checked before feeding it a byte.
    fn array_item_tracking_active(&self, depth: usize) -> bool {
        matches!(&self.frames[depth], Frame::Array(af) if af.item.is_some())
    }

    fn array_slice_observers_ready(&self, depth: usize) -> bool {
        let Some(Frame::Array(array)) = self.frames.get(depth) else {
            return true;
        };
        array
            .item
            .as_ref()
            .is_none_or(|item| item.builder.is_none())
    }

    /// Starts tracking a new item only when the array needs `uniqueItems` or any `contains` -
    /// most items in most arrays skip this entirely.
    fn start_item_tracking(&mut self, depth: usize) -> Result<(), StructuredRuntimeError> {
        let ap = super::plan::array_plan_of(self.plan.as_ref(), self.array_ref(depth).node);
        let contains_match = ap.contains.as_ref().map(|c| c.matches);
        if !ap.unique_items && ap.contains.is_none() {
            return Ok(());
        }
        let budget = self.limits.max_unique_canonical_bytes;
        let Frame::Array(af) = &self.frames[depth] else {
            unreachable!("expected an array frame")
        };
        let builder = ap
            .unique_items
            .then(|| {
                af.canonical_ref()
                    .expect("unique_items implies a canonical set")
                    .start_item(budget)
            })
            .transpose()?;
        let contains = match contains_match {
            None => ContainsCandidate::None,
            Some(super::plan::ContainsMatch::Never) => ContainsCandidate::Dead,
            Some(super::plan::ContainsMatch::Always) => ContainsCandidate::Always,
            Some(super::plan::ContainsMatch::Schema(id)) => match self.start_cursor(id) {
                Ok(cursor) => ContainsCandidate::Cursor(cursor),
                Err(error) => return Err(error),
            },
        };
        if let Err(error) = self.reserve_undo(1) {
            if let ContainsCandidate::Cursor(cursor) = contains {
                self.release_owned_cursor(cursor);
            }
            return Err(error);
        }
        self.array_mut(depth).item = Some(ItemTracking { builder, contains });
        self.push_undo(Undo::StartArrayItem { depth })
    }

    /// Advances the item tracker at `depth` by one byte: the parallel `contains` candidate
    /// engine (if any), then the `uniqueItems` canonical builder.
    fn feed_array_item_tracker(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<(), StructuredRuntimeError> {
        self.advance_item_contains_cursor(depth, byte)?;
        let budget = self.limits.max_unique_canonical_bytes;
        self.reserve_undo(1)?;
        let Frame::Array(af) = &mut self.frames[depth] else {
            unreachable!("expected an array frame")
        };
        let ArrayFrame {
            item, canonical, ..
        } = af;
        let Some(builder) = item.as_mut().expect("checked by caller").builder.as_mut() else {
            return Ok(());
        };
        let set = canonical
            .as_deref_mut()
            .and_then(|sets| sets.first_mut())
            .expect("unique_items implies a canonical set");
        let mark = set.feed_item(builder, budget, byte)?;
        self.push_undo(Undo::RollbackCanonicalBuild { depth, mark })
    }

    /// Advances the parallel `contains` candidate engine; dying here only means this item stops
    /// being a contains candidate, never a rejection of the item's own schema.
    fn advance_item_contains_cursor(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<(), StructuredRuntimeError> {
        let candidate = self
            .array_ref(depth)
            .item
            .as_ref()
            .expect("checked by caller")
            .contains;
        let ContainsCandidate::Cursor(cursor) = candidate else {
            return Ok(());
        };
        self.reserve_undo(1)?;
        let mark = self.cursors.checkpoint(cursor)?;
        match self.push_cursor_byte(cursor, byte)? {
            CursorStep::Dead => {
                self.array_mut(depth)
                    .item
                    .as_mut()
                    .expect("checked by caller")
                    .contains = ContainsCandidate::Dead;
                self.push_undo(Undo::RestoreItemContainsCandidate {
                    depth,
                    old: candidate,
                })
            }
            CursorStep::Alive | CursorStep::Complete => {
                self.push_undo(Undo::RollbackContainsCursor { cursor, mark })
            }
        }
    }

    /// Finalizes the item at `depth`: `uniqueItems` rejects a duplicate here (`Ok(false)`,
    /// leaving state unchanged), otherwise `contains_count` advances and tracking clears.
    fn finish_array_item_tracker(&mut self, depth: usize) -> Result<bool, StructuredRuntimeError> {
        let ap = super::plan::array_plan_of(self.plan.as_ref(), self.array_ref(depth).node);
        let item = self
            .array_ref(depth)
            .item
            .as_ref()
            .expect("checked by caller");
        let contains_matched = match ap.contains.as_ref().map(|c| &c.matches) {
            None => false,
            Some(super::plan::ContainsMatch::Never) => false,
            Some(super::plan::ContainsMatch::Always) => true,
            Some(super::plan::ContainsMatch::Schema(_)) => match item.contains {
                ContainsCandidate::Cursor(cursor) => self.cursors.is_accepting(cursor),
                ContainsCandidate::None | ContainsCandidate::Always | ContainsCandidate::Dead => {
                    false
                }
            },
        };
        let max_contains = ap.contains.as_ref().and_then(|c| c.max);
        if ap.unique_items && !self.finish_canonical_item(depth)? {
            return Ok(false);
        }
        if contains_matched
            && self
                .plan
                .as_ref()
                .annotations_required(self.array_ref(depth).node)
        {
            let ordinal = self
                .array_ref(depth)
                .index
                .checked_sub(1)
                .ok_or_else(|| self.cursors.error())?;
            self.insert_evaluated(AnnotationTarget::Array(depth), ordinal)?;
        }
        self.reserve_undo(1)?;
        let old = self
            .array_mut(depth)
            .item
            .take()
            .expect("checked by caller");
        self.push_undo(Undo::FinishArrayItem { depth, old })?;
        if contains_matched {
            let old_count = self.array_ref(depth).contains_count;
            let new_count = old_count.checked_add(1).ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::ArrayLength,
                    usize::MAX,
                    self.limits.max_items as usize,
                )
            })?;
            if max_contains.is_some_and(|max| new_count > max) {
                return Ok(false);
            }
            self.reserve_undo(1)?;
            self.array_mut(depth).contains_count = new_count;
            self.push_undo(Undo::RestoreContainsCount {
                depth,
                old: old_count,
            })?;
        }
        Ok(true)
    }

    /// Closes an item builder and inserts its canonical value into the array set.
    /// `Ok(false)` indicates a duplicate.
    fn finish_canonical_item(&mut self, depth: usize) -> Result<bool, StructuredRuntimeError> {
        let budget = self.limits.max_unique_canonical_bytes;
        self.reserve_undo(2)?;
        let Frame::Array(af) = &mut self.frames[depth] else {
            unreachable!("expected an array frame")
        };
        let ArrayFrame {
            item, canonical, ..
        } = af;
        let builder = item
            .as_mut()
            .expect("checked by caller")
            .builder
            .as_mut()
            .expect("unique_items implies a builder");
        let set = canonical
            .as_deref_mut()
            .and_then(|sets| sets.first_mut())
            .expect("unique_items implies a canonical set");
        let builder_mark = set.item_mark(builder);
        match set.finish_item(builder, budget)? {
            CanonicalInsert::Duplicate => Ok(false),
            CanonicalInsert::Inserted {
                fingerprint,
                old_bucket_len,
                bucket_created,
                bucket_capacity_bytes,
            } => {
                self.push_undo(Undo::RollbackCanonicalBuild {
                    depth,
                    mark: builder_mark,
                })?;
                self.push_undo(Undo::RollbackCanonicalInsert {
                    depth,
                    fingerprint,
                    old_bucket_len,
                    bucket_created,
                    bucket_capacity_bytes,
                })?;
                Ok(true)
            }
        }
    }

    /// Pushes one fresh `AnyJson` value frame: `additionalProperties: true`, `items: true`, or
    /// any other position with no schema at all.
    fn push_any_child(&mut self) -> Result<bool, StructuredRuntimeError> {
        let next_depth = self.frames.len().checked_add(1).ok_or_else(|| {
            StructuredRuntimeError::new(
                LimitKind::RecursionDepth,
                usize::MAX,
                self.limits.max_depth as usize,
            )
        })?;
        if next_depth > self.limits.max_depth as usize {
            return Err(StructuredRuntimeError::new(
                LimitKind::RecursionDepth,
                next_depth,
                self.limits.max_depth as usize,
            ));
        }
        let cap = self.local_session_budget()?;
        let StructuredState {
            frames,
            session_memory,
            ..
        } = self;
        let req = GrowthRequest {
            len: frames.len(),
            capacity: frames.capacity(),
            additional: 1,
            item_size: size_of::<Frame>(),
        };
        grow_bounded(session_memory, cap, LimitKind::SessionBytes, req, |n| {
            frames
                .try_reserve_exact(n)
                .map(|_| frames.capacity())
                .map_err(|_| ())
        })?;
        let req = GrowthRequest {
            len: self.frame_scopes.len(),
            capacity: self.frame_scopes.capacity(),
            additional: 1,
            item_size: size_of::<ScopeLease>(),
        };
        grow_bounded(
            &mut self.session_memory,
            cap,
            LimitKind::SessionBytes,
            req,
            |n| {
                self.frame_scopes
                    .try_reserve_exact(n)
                    .map(|_| self.frame_scopes.capacity())
                    .map_err(|_| ())
            },
        )?;
        self.reserve_undo(1)?;
        let scope = self.current_scope()?.clone();
        self.frames.push(Frame::Any(AnyFrame::Scalar {
            phase: ScalarPhase::BeforeValue,
            bytes: 0,
        }));
        self.frame_scopes.push(scope);
        self.push_undo(Undo::PopFrame)?;
        Ok(true)
    }

    fn route_any_byte(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<AnyStep, StructuredRuntimeError> {
        enum Kind {
            Scalar,
            String,
            Array,
            Object,
        }
        let kind = match &self.frames[depth] {
            Frame::Any(AnyFrame::Scalar { .. }) => Kind::Scalar,
            Frame::Any(AnyFrame::StringBody { .. }) => Kind::String,
            Frame::Any(AnyFrame::Array { .. }) => Kind::Array,
            Frame::Any(AnyFrame::Object(_)) => Kind::Object,
            _ => unreachable!("expected an any-value frame"),
        };
        match kind {
            Kind::Scalar => self.route_any_scalar_byte(depth, byte),
            Kind::String => self.route_any_string_byte(depth, byte),
            Kind::Array => self.route_any_array_byte(depth, byte),
            Kind::Object => self.route_any_object_byte(depth, byte),
        }
    }

    fn route_any_scalar_byte(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<AnyStep, StructuredRuntimeError> {
        let (phase, bytes) = match &self.frames[depth] {
            Frame::Any(AnyFrame::Scalar { phase, bytes }) => (*phase, *bytes),
            _ => unreachable!("expected a scalar any-frame"),
        };
        if phase == ScalarPhase::BeforeValue && WS.contains(&byte) {
            return Ok(AnyStep::Advanced);
        }
        match step_scalar(phase, byte) {
            ScalarStep::Continue(next) => {
                let next_bytes = bytes.checked_add(1).ok_or_else(|| {
                    StructuredRuntimeError::new(
                        LimitKind::NumberBytes,
                        usize::MAX,
                        self.limits.max_number_bytes,
                    )
                })?;
                if matches!(next, ScalarPhase::Number(_))
                    && next_bytes as usize > self.limits.max_number_bytes
                {
                    return Err(StructuredRuntimeError::new(
                        LimitKind::NumberBytes,
                        next_bytes as usize,
                        self.limits.max_number_bytes,
                    ));
                }
                self.reserve_undo(1)?;
                *self.any_mut(depth) = AnyFrame::Scalar {
                    phase: next,
                    bytes: next_bytes,
                };
                self.push_undo(Undo::SetAnyScalar {
                    depth,
                    old_phase: phase,
                    old_bytes: bytes,
                })?;
                Ok(AnyStep::Advanced)
            }
            ScalarStep::StartString => {
                self.reserve_undo(1)?;
                let old = std::mem::replace(
                    self.any_mut(depth),
                    AnyFrame::StringBody {
                        decoder: JsonStringDecoder::new(),
                        bytes: 0,
                    },
                );
                self.push_undo(Undo::ReplaceAnyFrame { depth, old })?;
                Ok(AnyStep::Advanced)
            }
            ScalarStep::StartArray => {
                self.reserve_undo(1)?;
                let old = std::mem::replace(
                    self.any_mut(depth),
                    AnyFrame::Array {
                        phase: AnyArrayPhase::BeforeFirstOrClose,
                        items: 0,
                    },
                );
                self.push_undo(Undo::ReplaceAnyFrame { depth, old })?;
                Ok(AnyStep::Advanced)
            }
            ScalarStep::StartObject => {
                let id = self.alloc_any_object()?;
                self.reserve_undo(1)?;
                let old = std::mem::replace(self.any_mut(depth), AnyFrame::Object(id));
                self.push_undo(Undo::ReplaceAnyFrame { depth, old })?;
                Ok(AnyStep::Advanced)
            }
            // `Done` from `step_scalar` always means complete (a literal only reaches it on the
            // full word match; a number only on a non-continuing byte in an already-valid state).
            ScalarStep::Done => {
                if !self.pop_any_frame(depth, None)? {
                    return Ok(AnyStep::Rejected);
                }
                Ok(AnyStep::Popped)
            }
            ScalarStep::Invalid => Ok(AnyStep::Rejected),
        }
    }

    /// Pops a completed `AnyJson` frame. `close_byte` is `Some` iff the byte is this frame's own
    /// closing delimiter (fed to any tracker first); `Ok(false)` iff a duplicate rejects it.
    fn pop_any_frame(
        &mut self,
        depth: usize,
        close_byte: Option<u8>,
    ) -> Result<bool, StructuredRuntimeError> {
        if let Some(byte) = close_byte {
            self.record_item_byte(depth, byte)?;
        }
        if !self.finish_tracked_item_if_direct_child(depth)? {
            return Ok(false);
        }
        self.reserve_undo(1)?;
        let popped = self.frames.pop().expect("checked by caller");
        let scope = self.frame_scopes.pop().expect("frame scope exists");
        self.push_undo(Undo::PushFrame(popped, scope))?;
        if depth == 0 {
            self.reserve_undo(1)?;
            self.root_closed = true;
            self.push_undo(Undo::SetRootClosed { old: false })?;
        }
        Ok(true)
    }

    fn route_any_string_byte(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<AnyStep, StructuredRuntimeError> {
        let (decoder, bytes) = match &self.frames[depth] {
            Frame::Any(AnyFrame::StringBody { decoder, bytes }) => (*decoder, *bytes),
            _ => unreachable!("expected a string any-frame"),
        };
        if byte == b'"' && decoder.at_boundary() {
            if !self.pop_any_frame(depth, Some(byte))? {
                return Ok(AnyStep::Rejected);
            }
            return Ok(AnyStep::Advanced);
        }
        let mut next_decoder = decoder;
        if matches!(next_decoder.push(byte), DecodeStep::Invalid) {
            return Ok(AnyStep::Rejected);
        }
        let next_bytes = bytes.checked_add(1).ok_or_else(|| {
            StructuredRuntimeError::new(
                LimitKind::StringBytes,
                usize::MAX,
                self.limits.max_string_bytes,
            )
        })?;
        if next_bytes as usize > self.limits.max_string_bytes {
            return Err(StructuredRuntimeError::new(
                LimitKind::StringBytes,
                next_bytes as usize,
                self.limits.max_string_bytes,
            ));
        }
        self.reserve_undo(1)?;
        *self.any_mut(depth) = AnyFrame::StringBody {
            decoder: next_decoder,
            bytes: next_bytes,
        };
        self.push_undo(Undo::SetAnyString {
            depth,
            old_decoder: decoder,
            old_bytes: bytes,
        })?;
        Ok(AnyStep::Advanced)
    }

    fn route_any_array_byte(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<AnyStep, StructuredRuntimeError> {
        let (phase, items) = match &self.frames[depth] {
            Frame::Any(AnyFrame::Array { phase, items }) => (*phase, *items),
            _ => unreachable!("expected an array any-frame"),
        };
        match phase {
            AnyArrayPhase::BeforeFirstOrClose => {
                if WS.contains(&byte) {
                    return Ok(AnyStep::Advanced);
                }
                if byte == b']' {
                    if !self.pop_any_frame(depth, Some(byte))? {
                        return Ok(AnyStep::Rejected);
                    }
                    return Ok(AnyStep::Advanced);
                }
                self.start_any_item(depth, phase, items, AnyArrayPhase::AfterValue, byte)
            }
            AnyArrayPhase::AfterValue => {
                if WS.contains(&byte) {
                    return Ok(AnyStep::Advanced);
                }
                if byte == b',' {
                    self.reserve_undo(1)?;
                    if let AnyFrame::Array { phase, .. } = self.any_mut(depth) {
                        *phase = AnyArrayPhase::AfterComma;
                    }
                    self.push_undo(Undo::SetAnyArrayPhase {
                        depth,
                        old: AnyArrayPhase::AfterValue,
                    })?;
                    return Ok(AnyStep::Advanced);
                }
                if byte == b']' {
                    if !self.pop_any_frame(depth, Some(byte))? {
                        return Ok(AnyStep::Rejected);
                    }
                    return Ok(AnyStep::Advanced);
                }
                Ok(AnyStep::Rejected)
            }
            AnyArrayPhase::AfterComma => {
                if WS.contains(&byte) {
                    return Ok(AnyStep::Advanced);
                }
                self.start_any_item(depth, phase, items, AnyArrayPhase::AfterValue, byte)
            }
        }
    }

    fn start_any_item(
        &mut self,
        depth: usize,
        old_phase: AnyArrayPhase,
        items: u32,
        next_phase: AnyArrayPhase,
        byte: u8,
    ) -> Result<AnyStep, StructuredRuntimeError> {
        if self.limits.max_items > 0 && items >= self.limits.max_items {
            return Err(StructuredRuntimeError::new(
                LimitKind::ArrayLength,
                items as usize,
                self.limits.max_items as usize,
            ));
        }
        let next_items = items.checked_add(1).ok_or_else(|| {
            StructuredRuntimeError::new(
                LimitKind::ArrayLength,
                usize::MAX,
                self.limits.max_items as usize,
            )
        })?;
        self.reserve_undo(2)?;
        if let AnyFrame::Array { phase, items } = self.any_mut(depth) {
            *phase = next_phase;
            *items = next_items;
        }
        self.push_undo(Undo::SetAnyArrayPhase {
            depth,
            old: old_phase,
        })?;
        self.push_undo(Undo::RestoreAnyArrayItems { depth, old: items })?;
        if !self.push_any_child()? {
            return Ok(AnyStep::Rejected);
        }
        match self.route_any_byte(self.frames.len() - 1, byte)? {
            AnyStep::Rejected => Ok(AnyStep::Rejected),
            _ => Ok(AnyStep::Advanced),
        }
    }

    fn set_any_object_phase(
        &mut self,
        depth: usize,
        new: ObjectPhase,
    ) -> Result<(), StructuredRuntimeError> {
        self.reserve_undo(1)?;
        let old = std::mem::replace(&mut self.any_object_mut(depth).phase, new);
        self.push_undo(Undo::SetAnyObjectPhase { depth, old })
    }

    fn start_any_key(&mut self, depth: usize) -> Result<(), StructuredRuntimeError> {
        let next_generation = self
            .any_object_ref(depth)
            .key_generation
            .checked_add(1)
            .ok_or_else(|| self.lease_error())?;
        self.reserve_undo(1)?;
        let obj = self.any_object_mut(depth);
        let mut new_key = std::mem::take(&mut obj.spare_key);
        new_key.clear();
        let old_key = std::mem::replace(&mut obj.key, new_key);
        let old_generation = std::mem::replace(&mut obj.key_generation, next_generation);
        let old_decoder = std::mem::replace(&mut obj.key_decoder, JsonStringDecoder::new());
        self.push_undo(Undo::ResetAnyKey {
            depth,
            old_key,
            old_generation,
            old_decoder,
        })?;
        self.set_any_object_phase(depth, ObjectPhase::InKey)
    }

    fn push_any_key_byte(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<AnyStep, StructuredRuntimeError> {
        let old_decoder = self.any_object_ref(depth).key_decoder;
        if byte == b'"' && old_decoder.at_boundary() {
            self.set_any_object_phase(depth, ObjectPhase::AfterKeyBeforeColon)?;
            return Ok(AnyStep::Advanced);
        }
        let mut decoder = old_decoder;
        let step = decoder.push(byte);
        if matches!(step, DecodeStep::Invalid) {
            return Ok(AnyStep::Rejected);
        }
        if let DecodeStep::Scalar(c) = step {
            let old_len = self.candidate_key_len(depth);
            let added = c.len_utf8();
            let next_len = old_len.checked_add(added).ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::KeyBytes,
                    usize::MAX,
                    self.limits.max_key_bytes,
                )
            })?;
            if next_len > self.limits.max_key_bytes {
                return Err(StructuredRuntimeError::new(
                    LimitKind::KeyBytes,
                    next_len,
                    self.limits.max_key_bytes,
                ));
            }
            self.reserve_undo(1)?;
            let cap = self.local_session_budget()?;
            let owner = self.key_owner(depth)?;
            let lease_generation = (self.key_lease.phase == KeyLeasePhase::Prepared
                && self.key_lease.owner == Some(owner))
            .then_some(self.key_lease.generation);
            let StructuredState {
                frames,
                any_object_arena,
                key_lease,
                session_memory,
                ..
            } = &mut *self;
            let key = if lease_generation.is_some() {
                &mut key_lease.buffer
            } else {
                let Frame::Any(AnyFrame::Object(id)) = &frames[depth] else {
                    unreachable!("expected an any-object frame")
                };
                &mut any_object_arena[id.0 as usize].key
            };
            let req = GrowthRequest {
                len: key.len(),
                capacity: key.capacity(),
                additional: added,
                item_size: 1,
            };
            grow_bounded(session_memory, cap, LimitKind::SessionBytes, req, |n| {
                key.try_reserve_exact(n)
                    .map(|_| key.capacity())
                    .map_err(|_| ())
            })?;
            key.push(c);
            self.push_undo(Undo::TruncateAnyKey {
                depth,
                old_len,
                lease_generation,
            })?;
        }
        self.reserve_undo(1)?;
        self.any_object_mut(depth).key_decoder = decoder;
        self.push_undo(Undo::RestoreAnyObjectDecoder {
            depth,
            old: old_decoder,
        })?;
        Ok(AnyStep::Advanced)
    }

    /// Rejects a duplicate decoded key; a fresh one interns into the shared `key_arena`.
    fn resolve_any_colon(&mut self, depth: usize) -> Result<AnyStep, StructuredRuntimeError> {
        if !self.intern_any_dynamic_key(depth)? {
            return Ok(AnyStep::Rejected);
        }
        let old_count = self.any_object_ref(depth).properties;
        let new_count = old_count.checked_add(1).ok_or_else(|| {
            StructuredRuntimeError::new(
                LimitKind::PropertyCount,
                usize::MAX,
                self.limits.max_properties as usize,
            )
        })?;
        if new_count as usize > self.limits.max_properties as usize {
            return Err(StructuredRuntimeError::new(
                LimitKind::PropertyCount,
                new_count as usize,
                self.limits.max_properties as usize,
            ));
        }
        self.reserve_undo(1)?;
        self.any_object_mut(depth).properties = new_count;
        self.push_undo(Undo::RestoreAnyObjectProperties {
            depth,
            old: old_count,
        })?;
        self.set_any_object_phase(depth, ObjectPhase::AfterColonBeforeValue)?;
        Ok(AnyStep::Advanced)
    }

    fn route_any_object_byte(
        &mut self,
        depth: usize,
        byte: u8,
    ) -> Result<AnyStep, StructuredRuntimeError> {
        let phase = self.any_object_ref(depth).phase;
        match phase {
            ObjectPhase::Start => unreachable!("an AnyJson object starts past the '{{' already"),
            ObjectPhase::BeforeFirstKeyOrClose => {
                if WS.contains(&byte) {
                    return Ok(AnyStep::Advanced);
                }
                if byte == b'"' {
                    self.start_any_key(depth)?;
                    return Ok(AnyStep::Advanced);
                }
                if byte == b'}' {
                    if !self.pop_any_frame(depth, Some(byte))? {
                        return Ok(AnyStep::Rejected);
                    }
                    return Ok(AnyStep::Advanced);
                }
                Ok(AnyStep::Rejected)
            }
            ObjectPhase::InKey => self.push_any_key_byte(depth, byte),
            ObjectPhase::AfterKeyBeforeColon => {
                if WS.contains(&byte) {
                    return Ok(AnyStep::Advanced);
                }
                if byte != b':' {
                    return Ok(AnyStep::Rejected);
                }
                self.resolve_any_colon(depth)
            }
            ObjectPhase::AfterColonBeforeValue => {
                if WS.contains(&byte) {
                    return Ok(AnyStep::Advanced);
                }
                self.set_any_object_phase(depth, ObjectPhase::AfterValueBeforeCommaOrClose)?;
                if !self.push_any_child()? {
                    return Ok(AnyStep::Rejected);
                }
                match self.route_any_byte(self.frames.len() - 1, byte)? {
                    AnyStep::Rejected => Ok(AnyStep::Rejected),
                    _ => Ok(AnyStep::Advanced),
                }
            }
            ObjectPhase::AfterValueBeforeCommaOrClose => {
                if WS.contains(&byte) {
                    return Ok(AnyStep::Advanced);
                }
                if byte == b',' {
                    self.set_any_object_phase(depth, ObjectPhase::AfterCommaBeforeKey)?;
                    return Ok(AnyStep::Advanced);
                }
                if byte == b'}' {
                    if !self.pop_any_frame(depth, Some(byte))? {
                        return Ok(AnyStep::Rejected);
                    }
                    return Ok(AnyStep::Advanced);
                }
                Ok(AnyStep::Rejected)
            }
            ObjectPhase::AfterCommaBeforeKey => {
                if WS.contains(&byte) {
                    return Ok(AnyStep::Advanced);
                }
                if byte != b'"' {
                    return Ok(AnyStep::Rejected);
                }
                self.start_any_key(depth)?;
                Ok(AnyStep::Advanced)
            }
        }
    }

    fn close_object(&mut self, depth: usize, byte: u8) -> Result<bool, StructuredRuntimeError> {
        self.record_item_byte(depth, byte)?;
        if self.dead {
            return Ok(false);
        }
        let op = super::plan::object_plan_of(self.plan.as_ref(), self.object_ref(depth).node);
        let count = self.object_ref(depth).property_count;
        if !self
            .object_ref(depth)
            .missing_required
            .iter()
            .all(|&w| w == 0)
            || op.min_properties.is_some_and(|lo| count < lo)
            || op.max_properties.is_some_and(|hi| count > hi)
        {
            return Ok(false);
        }
        if self
            .object_ref(depth)
            .dependent
            .as_ref()
            .is_some_and(|frame| {
                frame.cursors.iter().enumerate().any(|(index, &cursor)| {
                    bit_is_set(&frame.triggered_words, index)
                        && (!bit_is_set(&frame.active_words, index)
                            || !self.cursors.is_accepting(cursor))
                })
            })
        {
            return Ok(false);
        }
        if !self.dependent_required_satisfied(depth)? {
            return Ok(false);
        }
        let annotations_required = self
            .plan
            .as_ref()
            .annotations_required(self.object_ref(depth).node);
        let dependencies = if annotations_required {
            self.object_ref(depth)
                .dependent
                .as_ref()
                .map_or(0, |frame| frame.cursors.len())
        } else {
            0
        };
        for index in 0..dependencies {
            let cursor = self.object_ref(depth).dependent.as_ref().and_then(|frame| {
                (bit_is_set(&frame.triggered_words, index)
                    && bit_is_set(&frame.active_words, index))
                .then_some(frame.cursors[index])
            });
            if let Some(cursor) = cursor {
                self.merge_cursor_annotations(AnnotationTarget::Object(depth), cursor)?;
            }
        }
        if depth == 0 && annotations_required {
            self.publish_root_annotations(AnnotationTarget::Object(depth))?;
        }
        if !self.finish_tracked_item_if_direct_child(depth)? {
            return Ok(false);
        }
        self.reserve_undo(1)?;
        let popped = self.frames.pop().expect("checked by caller");
        let scope = self.frame_scopes.pop().expect("frame scope exists");
        self.push_undo(Undo::PushFrame(popped, scope))?;
        if depth == 0 {
            self.reserve_undo(1)?;
            self.root_closed = true;
            self.push_undo(Undo::SetRootClosed { old: false })?;
        }
        Ok(true)
    }

    fn dependent_required_satisfied(&self, depth: usize) -> Result<bool, StructuredRuntimeError> {
        let object = self.object_ref(depth);
        let object_plan = super::plan::object_plan_of(self.plan.as_ref(), object.node);
        let Some(plan) = &object_plan.dependent_required else {
            return Ok(true);
        };
        for rule in &plan.rules {
            let trigger = usize::try_from(rule.trigger).map_err(|_| self.cursors.error())?;
            if !object.dependent_required_seen.contains(trigger) {
                continue;
            }
            let start = usize::try_from(rule.required_start).map_err(|_| self.cursors.error())?;
            let len = usize::try_from(rule.required_len).map_err(|_| self.cursors.error())?;
            let end = start.checked_add(len).ok_or_else(|| self.cursors.error())?;
            let required = plan
                .required_ids
                .get(start..end)
                .ok_or_else(|| self.cursors.error())?;
            if required.iter().any(|id| {
                usize::try_from(*id)
                    .ok()
                    .is_none_or(|index| !object.dependent_required_seen.contains(index))
            }) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn number_accepts(value: NumberAccumulator, plan: NumberPlan) -> bool {
    if plan.integer_only && !number_is_integer(value) {
        return false;
    }
    if let Some((coefficient, scale)) = plan.multiple_of {
        if !number_is_multiple(value, coefficient, scale) {
            return false;
        }
    }
    if let Some((coefficient, scale, exclusive)) = plan.minimum {
        let ordering = compare_number_to_bound(value, coefficient, scale);
        if ordering == std::cmp::Ordering::Less
            || exclusive && ordering == std::cmp::Ordering::Equal
        {
            return false;
        }
    }
    if let Some((coefficient, scale, exclusive)) = plan.maximum {
        let ordering = compare_number_to_bound(value, coefficient, scale);
        if ordering == std::cmp::Ordering::Greater
            || exclusive && ordering == std::cmp::Ordering::Equal
        {
            return false;
        }
    }
    true
}

fn number_prefix_sign_viable(value: NumberAccumulator, plan: NumberPlan) -> bool {
    if value.negative {
        return plan.minimum.is_none_or(|(coefficient, _, exclusive)| {
            coefficient < 0 || coefficient == 0 && !exclusive
        });
    }
    plan.maximum
        .is_none_or(|(coefficient, _, _)| coefficient >= 0)
}

fn number_is_integer(value: NumberAccumulator) -> bool {
    value.core_digits == 0
        || value
            .signed_exponent()
            .saturating_sub(i64::from(value.fraction_digits))
            .saturating_add(i64::from(value.pending_zeros))
            >= 0
}

fn number_is_multiple(value: NumberAccumulator, modulus: u64, scale: u32) -> bool {
    if value.core_digits == 0 {
        return true;
    }
    let shift = value
        .signed_exponent()
        .saturating_sub(i64::from(value.fraction_digits))
        .saturating_add(i64::from(value.pending_zeros))
        .saturating_add(i64::from(scale));
    if shift < 0 {
        return false;
    }
    if modulus == 1 {
        return true;
    }
    let exponent = match u64::try_from(shift) {
        Ok(exponent) => exponent,
        Err(_) => return false,
    };
    let factor = modular_power_of_ten(exponent, modulus);
    (u128::from(value.core_mod) * u128::from(factor)) % u128::from(modulus) == 0
}

fn modular_power_of_ten(mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1 % modulus;
    let mut base = 10 % modulus;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = ((u128::from(result) * u128::from(base)) % u128::from(modulus)) as u64;
        }
        base = ((u128::from(base) * u128::from(base)) % u128::from(modulus)) as u64;
        exponent >>= 1;
    }
    result
}

fn compare_number_to_bound(
    value: NumberAccumulator,
    bound: i128,
    scale: u32,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if value.core_digits == 0 {
        return 0i128.cmp(&bound);
    }
    let value_negative = value.negative;
    let bound_negative = bound < 0;
    if value_negative != bound_negative {
        return if value_negative {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let magnitude = compare_number_magnitude_to_bound(value, bound.unsigned_abs(), scale);
    if value_negative {
        magnitude.reverse()
    } else {
        magnitude
    }
}

fn compare_number_magnitude_to_bound(
    value: NumberAccumulator,
    mut bound: u128,
    mut scale: u32,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if bound == 0 {
        return Ordering::Greater;
    }
    while scale > 0 && bound % 10 == 0 {
        bound /= 10;
        scale -= 1;
    }
    let (bound_digits, bound_len) = u128_decimal_digits(bound);
    let value_shift = value
        .signed_exponent()
        .saturating_sub(i64::from(value.fraction_digits))
        .saturating_add(i64::from(value.pending_zeros));
    let value_scientific = i64::from(value.core_digits)
        .saturating_add(value_shift)
        .saturating_sub(1);
    let bound_scientific = i64::try_from(bound_len)
        .map_or(i64::MAX, |length| length)
        .saturating_sub(i64::from(scale))
        .saturating_sub(1);
    match value_scientific.cmp(&bound_scientific) {
        Ordering::Equal => {}
        ordering => return ordering,
    }
    let compare_len = usize::from(value.leading_len).max(bound_len);
    for index in 0..compare_len {
        let left = value.leading.get(index).copied().unwrap_or(0);
        let right = bound_digits.get(index).copied().unwrap_or(0);
        match left.cmp(&right) {
            Ordering::Equal => {}
            ordering => return ordering,
        }
    }
    if usize::try_from(value.core_digits).is_ok_and(|digits| digits > compare_len) {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn u128_decimal_digits(mut value: u128) -> ([u8; 39], usize) {
    let mut reverse = [0u8; 39];
    let mut length = 0usize;
    while value > 0 {
        reverse[length] = (value % 10) as u8;
        value /= 10;
        length += 1;
    }
    let mut digits = [0u8; 39];
    for index in 0..length {
        digits[index] = reverse[length - index - 1];
    }
    (digits, length)
}

fn prepare_frame(
    plan: &StructuredPlan,
    node: NodeId,
    arena_start: (u32, u32),
) -> Result<Option<PreparedFrame>, StructuredRuntimeError> {
    match plan.node(node) {
        NodePlan::Regular(engine) => Ok(Some(PreparedFrame {
            frame: Frame::Regular {
                node,
                state: engine.start(),
            },
            charge: 0,
        })),
        NodePlan::Number { engine, .. } => Ok(Some(PreparedFrame {
            frame: Frame::Number(NumberFrame {
                node,
                state: engine.start(),
                accumulator: NumberAccumulator::new(),
            }),
            charge: 0,
        })),
        NodePlan::String(plan) => Ok(Some(PreparedFrame {
            frame: Frame::String(StringFrame {
                node,
                phase: StringPhase::BeforeQuote,
                decoder: JsonStringDecoder::new(),
                scalars: 0,
                pattern_state: plan.pattern.as_ref().map(|engine| engine.start()),
            }),
            charge: 0,
        })),
        NodePlan::StringEnum(enum_plan) => Ok(Some(PreparedFrame {
            frame: Frame::StringEnum(StringEnumFrame {
                node,
                phase: StringPhase::BeforeQuote,
                decoder: JsonStringDecoder::new(),
                lo: 0,
                hi: u32::try_from(enum_plan.member_count())
                    .map_err(|_| reference_resolution_error(plan))?,
                decoded_bytes: 0,
            }),
            charge: 0,
        })),
        NodePlan::Object(op) => {
            let mut missing_required = Vec::new();
            missing_required
                .try_reserve_exact(op.required_template.len())
                .map_err(|_| {
                    StructuredRuntimeError::new(
                        LimitKind::SessionBytes,
                        usize::MAX,
                        plan.limits.max_session_bytes,
                    )
                })?;
            missing_required.extend_from_slice(&op.required_template);
            let mut seen_known = Vec::new();
            seen_known
                .try_reserve_exact(op.required_template.len())
                .map_err(|_| {
                    StructuredRuntimeError::new(
                        LimitKind::SessionBytes,
                        usize::MAX,
                        plan.limits.max_session_bytes,
                    )
                })?;
            seen_known.resize(op.required_template.len(), 0);
            let dependent_required_seen = PresenceBits::new(
                op.dependent_required
                    .as_ref()
                    .map_or(0, |presence| presence.name_count as usize),
                plan.limits.max_session_bytes,
            )?;
            // Charge the real reserved capacity, matching the capacity-based release at destruction.
            let charge = missing_required
                .capacity()
                .checked_add(seen_known.capacity())
                .and_then(|words| words.checked_mul(size_of::<u64>()))
                .and_then(|bytes| bytes.checked_add(dependent_required_seen.allocation_charge()))
                .ok_or_else(|| {
                    StructuredRuntimeError::new(
                        LimitKind::SessionBytes,
                        usize::MAX,
                        plan.limits.max_session_bytes,
                    )
                })?;
            Ok(Some(PreparedFrame {
                frame: Frame::Object(ObjectFrame {
                    node,
                    phase: ObjectPhase::Start,
                    key: String::new(),
                    spare_key: String::new(),
                    key_generation: 0,
                    key_decoder: JsonStringDecoder::new(),
                    property_name: None,
                    resolved_value: None,
                    missing_required,
                    seen_known,
                    seen_dynamic: FxHashMap::default(),
                    arena_start,
                    property_count: 0,
                    dependent: None,
                    dependent_required_seen,
                    evaluated: EvaluatedSet::None,
                }),
                charge,
            }))
        }
        NodePlan::Array(ap) => {
            let limit = plan.limits.max_session_bytes;
            let canonical = ap
                .unique_items
                .then(|| try_box_canonical_set(limit))
                .transpose()?;
            let charge = canonical
                .as_ref()
                .map_or(0, |sets| sets.capacity() * size_of::<CanonicalSet>());
            Ok(Some(PreparedFrame {
                frame: Frame::Array(ArrayFrame {
                    node,
                    phase: ArrayPhase::Start,
                    index: 0,
                    canonical,
                    contains_count: 0,
                    item: None,
                    evaluated: EvaluatedSet::None,
                }),
                charge,
            }))
        }
        NodePlan::Combinator(_) | NodePlan::Negation { .. } => {
            Err(reference_resolution_error(plan))
        }
        NodePlan::Ref { .. } | NodePlan::DynamicRef { .. } => Err(reference_resolution_error(plan)),
        NodePlan::Unevaluated(_) | NodePlan::Unsupported => Ok(None),
    }
}

fn resolve_runtime_node(
    plan: &StructuredPlan,
    start: NodeId,
    mut scope: ScopeLease,
) -> Result<(NodeId, ScopeLease), StructuredRuntimeError> {
    let mut node = start;
    let mut hops = 0usize;
    loop {
        let index = usize::try_from(node.get()).map_err(|_| reference_resolution_error(plan))?;
        let entry = plan
            .nodes
            .get(index)
            .ok_or_else(|| reference_resolution_error(plan))?;
        let resource = plan
            .node_resource(node)
            .ok_or_else(|| reference_resolution_error(plan))?;
        if plan.dynamic_scope_enabled {
            scope = scope
                .arena
                .enter(&scope, resource, plan.limits.max_dynamic_scope_depth)?;
        }
        let target = match entry {
            NodePlan::Ref { target } => *target,
            NodePlan::DynamicRef {
                initial_target,
                anchor,
            } => resolve_dynamic_target(plan, *initial_target, *anchor, &scope)?,
            _ => return Ok((node, scope)),
        };
        node = target;
        hops = hops
            .checked_add(1)
            .ok_or_else(|| reference_resolution_error(plan))?;
        if hops > plan.nodes.len() {
            return Err(reference_resolution_error(plan));
        }
    }
}

fn resolve_dynamic_target(
    plan: &StructuredPlan,
    initial_target: NodeId,
    anchor: AnchorId,
    scope: &ScopeLease,
) -> Result<NodeId, StructuredRuntimeError> {
    #[cfg(feature = "bench-internals")]
    scope
        .arena
        .dynamic_resolutions
        .fetch_add(1, Ordering::Relaxed);
    let key = DynamicResolutionKey {
        scope_index: scope.id.0,
        scope_generation: scope.arena.generation(scope.id)?,
        anchor,
        initial_target,
    };
    if let Some(target) = scope.arena.dynamic_memo_get(key) {
        return Ok(target);
    }
    let mut selected = initial_target;
    let mut cursor = Some(scope.id);
    let mut work = 0usize;
    while let Some(id) = cursor {
        #[cfg(feature = "bench-internals")]
        scope
            .arena
            .dynamic_scope_steps
            .fetch_add(1, Ordering::Relaxed);
        let resource = scope.arena.resource(id)?;
        if let Some(target) = plan.dynamic_target(anchor, resource) {
            selected = target;
        }
        cursor = scope.arena.parent(id)?;
        work = work
            .checked_add(1)
            .ok_or_else(|| reference_resolution_error(plan))?;
        if work > plan.limits.max_dynamic_scope_depth as usize {
            return Err(StructuredRuntimeError::new(
                LimitKind::DynamicScopeDepth,
                work,
                plan.limits.max_dynamic_scope_depth as usize,
            ));
        }
    }
    scope.arena.dynamic_memo_store(key, selected);
    Ok(selected)
}

fn reference_resolution_error(plan: &StructuredPlan) -> StructuredRuntimeError {
    StructuredRuntimeError::new(
        LimitKind::ActiveValidators,
        usize::MAX,
        plan.limits.max_active_validators,
    )
}

fn scope_arena_capacity(plan: &StructuredPlan) -> Result<usize, StructuredRuntimeError> {
    if !plan.dynamic_scope_enabled {
        return Ok(1);
    }
    plan.limits
        .max_active_validators
        .checked_add(1)
        .ok_or_else(|| {
            StructuredRuntimeError::new(
                LimitKind::ActiveValidators,
                usize::MAX,
                plan.limits.max_active_validators,
            )
        })
}

fn try_box_canonical_set(limit: usize) -> Result<Vec<CanonicalSet>, StructuredRuntimeError> {
    let mut sets = Vec::new();
    sets.try_reserve_exact(1)
        .map_err(|_| StructuredRuntimeError::new(LimitKind::SessionBytes, usize::MAX, limit))?;
    sets.push(CanonicalSet::new());
    Ok(sets)
}

fn array_owned_charge(array: &ArrayFrame) -> usize {
    array
        .canonical
        .as_ref()
        .map_or(0, |sets| {
            sets.capacity().saturating_mul(size_of::<CanonicalSet>())
        })
        .saturating_add(array.evaluated.retained_bytes())
}

fn obligation_owned_bytes(value: &ValueObligations) -> usize {
    match value {
        ValueObligations::AnyJson => 0,
        ValueObligations::Schemas(set) => set
            .spill
            .capacity()
            .checked_mul(size_of::<NodeId>())
            .unwrap_or(usize::MAX),
    }
}

fn active_branch_count(words: &[u64]) -> usize {
    words.iter().map(|word| word.count_ones() as usize).sum()
}

fn accepting_branch_count(frame: &CombinatorFrame, cursors: &CursorMachine<'_>) -> usize {
    frame
        .cursors
        .iter()
        .enumerate()
        .filter(|(index, cursor)| {
            bit_is_set(&frame.active_words, *index) && cursors.is_accepting(**cursor)
        })
        .count()
}

fn combinator_accepts(kind: CombinatorKind, accepting: usize, required: usize) -> bool {
    match kind {
        CombinatorKind::All => accepting == required,
        CombinatorKind::Any => accepting >= 1,
        CombinatorKind::ExactlyOne => accepting == 1,
    }
}

fn combinator_frame_accepts(frame: &CombinatorFrame, cursors: &CursorMachine<'_>) -> bool {
    combinator_accepts(
        frame.kind,
        accepting_branch_count(frame, cursors),
        frame.cursors.len(),
    )
}

fn object_actual_owned_charge(object: &ObjectFrame) -> usize {
    let bitsets = object
        .missing_required
        .capacity()
        .checked_add(object.seen_known.capacity())
        .and_then(|words| words.checked_mul(size_of::<u64>()))
        .unwrap_or(usize::MAX);
    let strings = object
        .key
        .capacity()
        .checked_add(object.spare_key.capacity())
        .unwrap_or(usize::MAX);
    let obligations = object
        .resolved_value
        .as_ref()
        .map_or(0, obligation_owned_bytes);
    let dependent = object
        .dependent
        .as_ref()
        .map_or(0, |frame| frame.allocation_charge);
    let table = object
        .seen_dynamic
        .capacity()
        .checked_mul(size_of::<u64>() + size_of::<DynamicSlot>() + 8)
        .unwrap_or(usize::MAX);
    let collisions = object
        .seen_dynamic
        .values()
        .filter_map(|slot| match slot {
            DynamicSlot::One(_) => Some(0),
            DynamicSlot::Many(ids) => ids.capacity().checked_mul(size_of::<KeyId>()),
        })
        .try_fold(0usize, usize::checked_add)
        .unwrap_or(usize::MAX);
    bitsets
        .checked_add(object.dependent_required_seen.allocation_charge())
        .and_then(|total| total.checked_add(strings))
        .and_then(|total| total.checked_add(table))
        .and_then(|total| total.checked_add(collisions))
        .and_then(|total| total.checked_add(obligations))
        .and_then(|total| total.checked_add(dependent))
        .and_then(|total| total.checked_add(object.evaluated.retained_bytes()))
        .unwrap_or(usize::MAX)
}

#[cfg(test)]
fn dynamic_table_owned_charge(table: &FxHashMap<u64, DynamicSlot>) -> usize {
    let buckets = table
        .capacity()
        .saturating_mul(size_of::<u64>() + size_of::<DynamicSlot>() + 8);
    table.values().fold(buckets, |total, slot| {
        total.saturating_add(match slot {
            DynamicSlot::One(_) => 0,
            DynamicSlot::Many(ids) => ids.capacity().saturating_mul(size_of::<KeyId>()),
        })
    })
}

#[cfg(test)]
fn any_object_owned_charge(object: &AnyObjectState) -> usize {
    object
        .key
        .capacity()
        .saturating_add(object.spare_key.capacity())
        .saturating_add(dynamic_table_owned_charge(&object.seen_dynamic))
}

#[cfg(test)]
impl StructuredState<'_> {
    fn empty_for_test(
        plan: &StructuredPlan,
        validator_ledger: Arc<ValidatorLedger>,
    ) -> Result<StructuredState<'_>, StructuredRuntimeError> {
        let mut state = StructuredState::empty(PlanHandle::Borrowed(plan), validator_ledger);
        let (arena, arena_charge) = ScopeArena::new(scope_arena_capacity(plan)?)?;
        let resource = plan
            .node_resource(plan.root)
            .ok_or_else(|| reference_resolution_error(plan))?;
        state.frame_scopes.try_reserve_exact(1).map_err(|_| {
            StructuredRuntimeError::new(
                LimitKind::SessionBytes,
                usize::MAX,
                state.limits.max_session_bytes,
            )
        })?;
        let scope_charge = state
            .frame_scopes
            .capacity()
            .checked_mul(size_of::<ScopeLease>())
            .and_then(|bytes| bytes.checked_add(arena_charge))
            .ok_or_else(|| {
                StructuredRuntimeError::new(
                    LimitKind::SessionBytes,
                    usize::MAX,
                    state.limits.max_session_bytes,
                )
            })?;
        state.session_memory.charge(
            scope_charge,
            state.limits.max_session_bytes,
            LimitKind::SessionBytes,
        )?;
        state.scope_arena_charge = arena_charge;
        state.frame_scopes.push(arena.root(resource)?);
        Ok(state)
    }

    pub(crate) fn recomputed_session_bytes(&self) -> usize {
        let mut total = self
            .validator_ledger_charge
            .saturating_add(self.scope_arena_charge)
            .saturating_add(self.root_annotations.retained_bytes())
            .saturating_add(self.frames.capacity().saturating_mul(size_of::<Frame>()))
            .saturating_add(
                self.frame_scopes
                    .capacity()
                    .saturating_mul(size_of::<ScopeLease>()),
            )
            .saturating_add(self.undo.capacity().saturating_mul(size_of::<Undo>()))
            .saturating_add(self.key_arena.bytes.capacity())
            .saturating_add(self.key_lease.buffer.capacity())
            .saturating_add(
                self.key_arena
                    .spans
                    .capacity()
                    .saturating_mul(size_of::<(u32, u32)>()),
            )
            .saturating_add(
                self.any_object_arena
                    .capacity()
                    .saturating_mul(size_of::<AnyObjectState>()),
            )
            .saturating_add(
                self.any_object_free
                    .capacity()
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(
                self.object_frame_free
                    .capacity()
                    .saturating_mul(size_of::<ObjectFrame>()),
            )
            .saturating_add(
                self.combinator_frame_free
                    .capacity()
                    .saturating_mul(size_of::<CombinatorFrame>()),
            );
        total = self.any_object_arena.iter().fold(total, |sum, object| {
            sum.saturating_add(any_object_owned_charge(object))
        });
        total = self.frames.iter().fold(total, |sum, frame| {
            sum.saturating_add(frame_owned_charge(frame))
        });
        total = self.object_frame_free.iter().fold(total, |sum, object| {
            sum.saturating_add(object_actual_owned_charge(object))
        });
        total = self.combinator_frame_free.iter().fold(total, |sum, frame| {
            sum.saturating_add(frame.allocation_charge)
        });
        self.undo.iter().fold(total, |sum, undo| {
            sum.saturating_add(undo_owned_charge(undo))
        })
    }

    #[cfg(test)]
    pub(crate) fn recomputed_retained_bytes(&self) -> usize {
        self.recomputed_session_bytes()
            .saturating_add(self.cursors.recomputed_retained_bytes())
    }

    #[cfg(all(test, feature = "bench-internals"))]
    pub(crate) fn accounting_breakdown_for_test(&self) -> String {
        format!(
            "local_ledger={} cursor_ledger={} frames={}x{} frame_scopes={}x{} undo={}x{} key_lease={} any_arena={}x{} any_free={}x{} object_free={}x{} combinator_free={}x{} local_actual={} cursor_actual={}",
            self.session_memory.live(),
            self.cursors.retained_bytes(),
            self.frames.capacity(),
            size_of::<Frame>(),
            self.frame_scopes.capacity(),
            size_of::<ScopeLease>(),
            self.undo.capacity(),
            size_of::<Undo>(),
            self.key_lease.buffer.capacity(),
            self.any_object_arena.capacity(),
            size_of::<AnyObjectState>(),
            self.any_object_free.capacity(),
            size_of::<u32>(),
            self.object_frame_free.capacity(),
            size_of::<ObjectFrame>(),
            self.combinator_frame_free.capacity(),
            size_of::<CombinatorFrame>(),
            self.recomputed_session_bytes(),
            self.cursors.recomputed_retained_bytes(),
        )
    }
}

#[cfg(test)]
fn frame_owned_charge(frame: &Frame) -> usize {
    match frame {
        Frame::Object(object) => object_actual_owned_charge(object),
        Frame::Obligation(obligation) => obligation.allocation_charge,
        Frame::Combinator(combinator) => combinator.allocation_charge,
        Frame::Array(array) => array_owned_charge(array),
        Frame::Unevaluated(frame) => frame.candidates.retained_bytes(),
        Frame::Regular { .. }
        | Frame::Number(_)
        | Frame::String(_)
        | Frame::StringEnum(_)
        | Frame::Negation(_)
        | Frame::Any(_) => 0,
    }
}

#[cfg(test)]
fn undo_owned_charge(undo: &Undo) -> usize {
    match undo {
        Undo::ResetKey { old_key, .. } | Undo::ResetAnyKey { old_key, .. } => old_key.capacity(),
        Undo::RestoreResolvedValue {
            old: Some(value), ..
        } => obligation_owned_bytes(value),
        Undo::PushFrame(frame, _) => frame_owned_charge(frame),
        Undo::SetRootAnnotations { old, .. } | Undo::ReplaceRootAnnotations { old } => {
            old.retained_bytes()
        }
        Undo::ReplaceEvaluatedSet { old, .. } => old.retained_bytes(),
        _ => 0,
    }
}

fn encode_decoded_json_string(candidate: &[u8]) -> Option<Vec<u8>> {
    let text = std::str::from_utf8(candidate).ok()?;
    let capacity = candidate.len().checked_mul(6)?.checked_add(2)?;
    let mut encoded = Vec::new();
    encoded.try_reserve_exact(capacity).ok()?;
    encoded.push(b'"');
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for scalar in text.chars() {
        match scalar {
            '"' => encoded.extend_from_slice(br#"\""#),
            '\\' => encoded.extend_from_slice(br"\\"),
            '\u{08}' => encoded.extend_from_slice(br"\b"),
            '\u{0c}' => encoded.extend_from_slice(br"\f"),
            '\n' => encoded.extend_from_slice(br"\n"),
            '\r' => encoded.extend_from_slice(br"\r"),
            '\t' => encoded.extend_from_slice(br"\t"),
            '\u{00}'..='\u{1f}' => {
                let value = scalar as u8;
                encoded.extend_from_slice(br"\u00");
                encoded.push(HEX[usize::from(value >> 4)]);
                encoded.push(HEX[usize::from(value & 0x0f)]);
            }
            _ => {
                let mut bytes = [0u8; 4];
                encoded.extend_from_slice(scalar.encode_utf8(&mut bytes).as_bytes());
            }
        }
    }
    encoded.push(b'"');
    Some(encoded)
}

fn bit_is_set(bits: &[u64], idx: usize) -> bool {
    (bits[idx / BITSET_WORD_BITS] >> (idx % BITSET_WORD_BITS)) & 1 == 1
}

fn set_bit(bits: &mut [u64], idx: usize) {
    bits[idx / BITSET_WORD_BITS] |= 1u64 << (idx % BITSET_WORD_BITS);
}

fn clear_bit(bits: &mut [u64], idx: usize) {
    bits[idx / BITSET_WORD_BITS] &= !(1u64 << (idx % BITSET_WORD_BITS));
}

#[cfg(test)]
mod tests;
