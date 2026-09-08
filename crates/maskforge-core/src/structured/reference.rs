//! Independent recursive-descent oracle for tests and benchmark-only comparisons.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::lexer::{DecodeStep, JsonStringDecoder};
use super::limits::{resource_limit, StructuredLimits, StructuredRuntimeError};

use crate::automaton::{
    build_from_regex, build_from_unicode_regex, Combinator, ProductAutomaton, RefEngine,
};
use crate::compile::compile_node;
use crate::error::{CompileError, LimitKind, Stage};
use crate::ir::{
    AdditionalPolicy, AnchorId, Charset, ContainsConstraint, ContainsPolicy, ItemsPolicy, Node,
    ResourceId, SchemaIR, UnevaluatedKind,
};
use crate::mask::Bitmask;
use crate::primitives::{NodeId, StateId, TokenId};

const MAX_DEPTH: u32 = 256;

/// Whether the schema at `bytes` root accepts `bytes` as one complete, anchored JSON value.
pub fn accepts(ir: &SchemaIR, bytes: &[u8]) -> bool {
    if ir.diagnostics().next().is_some() {
        return false;
    }
    accepts_with_cache(ir, bytes, &mut MatchCache::new(ir))
}

/// `accepts`, reusing a caller-owned `cache` so its compiled "regular" engines amortize across
/// many calls against the same `ir` (the hot path a full-vocabulary mask scan needs).
fn accepts_with_cache(ir: &SchemaIR, bytes: &[u8], cache: &mut MatchCache<'_>) -> bool {
    matches_complete(ir, ir.root(), bytes, cache, 0)
}

/// A position inside a `SchemaIR`'s structured backend, mirroring `runtime::Matcher` for schemas
/// that need the recursive-descent path instead of the byte-DFA (an `OpenObject` is reachable).
pub(crate) struct ReferenceMatcher {
    ir: Arc<SchemaIR>,
    buffer: Vec<u8>,
    limits: StructuredLimits,
}

impl ReferenceMatcher {
    /// Starts a matcher at the empty prefix.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn new(ir: Arc<SchemaIR>) -> Self {
        Self::new_with_limits(ir, StructuredLimits::default())
    }

    pub(crate) fn new_with_limits(ir: Arc<SchemaIR>, limits: StructuredLimits) -> Self {
        Self {
            ir,
            buffer: Vec::new(),
            limits,
        }
    }

    #[cfg(feature = "bench-internals")]
    pub(crate) fn retained_bytes(&self) -> usize {
        self.buffer.capacity()
    }

    /// Stopping here yields a complete, valid instance.
    #[must_use]
    pub fn is_accepting(&self) -> bool {
        accepts(&self.ir, &self.buffer)
    }

    /// EOS is legal exactly where the position accepts.
    #[must_use]
    pub fn eos_legal(&self) -> bool {
        self.is_accepting()
    }

    /// The position has reached a permanent dead end: neither accepting now nor extendable.
    #[must_use]
    pub fn is_dead(&self) -> bool {
        !self.is_accepting() && !can_continue(&self.ir, &self.buffer)
    }

    /// Appends `token_bytes` if doing so keeps the position accepting or extendable; returns
    /// whether it was applied. The position never changes on a rejected token.
    #[cfg(test)]
    pub(crate) fn advance(&mut self, token_bytes: &[u8]) -> bool {
        self.try_advance(token_bytes).unwrap_or(false)
    }

    pub(crate) fn try_advance(
        &mut self,
        token_bytes: &[u8],
    ) -> Result<bool, StructuredRuntimeError> {
        if token_bytes.is_empty() {
            return Ok(false);
        }
        let original_len = self.buffer.len();
        let required = original_len.checked_add(token_bytes.len()).ok_or_else(|| {
            StructuredRuntimeError::new(
                crate::error::LimitKind::DocumentBytes,
                usize::MAX,
                self.limits.max_document_bytes,
            )
        })?;
        if required > self.limits.max_document_bytes {
            return Err(StructuredRuntimeError::new(
                crate::error::LimitKind::DocumentBytes,
                required,
                self.limits.max_document_bytes,
            ));
        }
        self.buffer.try_reserve(token_bytes.len()).map_err(|_| {
            StructuredRuntimeError::new(
                crate::error::LimitKind::DocumentBytes,
                required,
                self.limits.max_document_bytes,
            )
        })?;
        self.buffer.extend_from_slice(token_bytes);
        if accepts(&self.ir, &self.buffer) || can_continue(&self.ir, &self.buffer) {
            Ok(true)
        } else {
            self.buffer.truncate(original_len);
            Ok(false)
        }
    }

    /// The allowed ordinary-token mask at the current position, walked one candidate at a time
    /// (the same record shape as `RefEngine::allowed_tokens_from_records`).
    pub(crate) fn allowed_mask_from_records<'a>(
        &self,
        mask_vocab_size: usize,
        records: impl Iterator<Item = (&'a [u8], &'a [u32])>,
    ) -> Result<Bitmask, CompileError> {
        let mut out = Bitmask::zeros(mask_vocab_size);
        if self.ir.diagnostics().next().is_some() {
            return Ok(out);
        }
        // Reused across every record: compiled engines are schema-derived, not input-derived.
        let mut cache = MatchCache::new(&self.ir);
        let mut buf = Vec::new();
        let mut work = 0u64;
        for (bytes, ids) in records {
            let delta = u64::try_from(bytes.len())
                .ok()
                .and_then(|value| value.checked_add(u64::try_from(ids.len()).ok()?))
                .ok_or_else(|| {
                    resource_limit(
                        LimitKind::MaskWork,
                        usize::MAX,
                        usize::try_from(self.limits.max_mask_work).unwrap_or(usize::MAX),
                        Stage::L4Bind,
                    )
                })?;
            work = work.checked_add(delta).ok_or_else(|| {
                resource_limit(
                    LimitKind::MaskWork,
                    usize::MAX,
                    usize::try_from(self.limits.max_mask_work).unwrap_or(usize::MAX),
                    Stage::L4Bind,
                )
            })?;
            if work > self.limits.max_mask_work {
                return Err(resource_limit(
                    LimitKind::MaskWork,
                    usize::try_from(work).unwrap_or(usize::MAX),
                    usize::try_from(self.limits.max_mask_work).unwrap_or(usize::MAX),
                    Stage::L4Bind,
                ));
            }
            let required = self.buffer.len().checked_add(bytes.len()).ok_or_else(|| {
                resource_limit(
                    LimitKind::DocumentBytes,
                    usize::MAX,
                    self.limits.max_document_bytes,
                    Stage::L4Bind,
                )
            })?;
            if required > self.limits.max_document_bytes {
                return Err(resource_limit(
                    LimitKind::DocumentBytes,
                    required,
                    self.limits.max_document_bytes,
                    Stage::L4Bind,
                ));
            }
            buf.clear();
            buf.try_reserve(required).map_err(|_| {
                resource_limit(
                    LimitKind::DocumentBytes,
                    required,
                    self.limits.max_document_bytes,
                    Stage::L4Bind,
                )
            })?;
            buf.extend_from_slice(&self.buffer);
            buf.extend_from_slice(bytes);
            if accepts_with_cache(&self.ir, &buf, &mut cache)
                || can_continue_with_cache(&self.ir, &buf, &mut cache)
            {
                for &id in ids {
                    out.set(TokenId(id))?;
                }
            }
        }
        Ok(out)
    }

    #[cfg(feature = "bench-internals")]
    pub(crate) fn write_mask_le_bytes_into<'a>(
        &self,
        mask_vocab_size: usize,
        records: impl Iterator<Item = (&'a [u8], &'a [u32])>,
        out: &mut [u8],
    ) -> Result<(), CompileError> {
        if self.ir.diagnostics().next().is_some() {
            return Ok(());
        }
        let mut cache = MatchCache::new(&self.ir);
        let mut buf = Vec::new();
        let mut work = 0u64;
        for (bytes, ids) in records {
            let delta = u64::try_from(bytes.len())
                .ok()
                .and_then(|value| value.checked_add(u64::try_from(ids.len()).ok()?))
                .ok_or_else(|| {
                    resource_limit(
                        LimitKind::MaskWork,
                        usize::MAX,
                        usize::try_from(self.limits.max_mask_work).unwrap_or(usize::MAX),
                        Stage::L4Bind,
                    )
                })?;
            work = work.checked_add(delta).ok_or_else(|| {
                resource_limit(
                    LimitKind::MaskWork,
                    usize::MAX,
                    usize::try_from(self.limits.max_mask_work).unwrap_or(usize::MAX),
                    Stage::L4Bind,
                )
            })?;
            if work > self.limits.max_mask_work {
                return Err(resource_limit(
                    LimitKind::MaskWork,
                    usize::try_from(work).unwrap_or(usize::MAX),
                    usize::try_from(self.limits.max_mask_work).unwrap_or(usize::MAX),
                    Stage::L4Bind,
                ));
            }
            let required = self.buffer.len().checked_add(bytes.len()).ok_or_else(|| {
                resource_limit(
                    LimitKind::DocumentBytes,
                    usize::MAX,
                    self.limits.max_document_bytes,
                    Stage::L4Bind,
                )
            })?;
            if required > self.limits.max_document_bytes {
                return Err(resource_limit(
                    LimitKind::DocumentBytes,
                    required,
                    self.limits.max_document_bytes,
                    Stage::L4Bind,
                ));
            }
            buf.clear();
            buf.try_reserve(required).map_err(|_| {
                resource_limit(
                    LimitKind::DocumentBytes,
                    required,
                    self.limits.max_document_bytes,
                    Stage::L4Bind,
                )
            })?;
            buf.extend_from_slice(&self.buffer);
            buf.extend_from_slice(bytes);
            if accepts_with_cache(&self.ir, &buf, &mut cache)
                || can_continue_with_cache(&self.ir, &buf, &mut cache)
            {
                for &id in ids {
                    set_mask_bit_le(out, mask_vocab_size, id)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(feature = "bench-internals")]
fn set_mask_bit_le(out: &mut [u8], width: usize, id: u32) -> Result<(), CompileError> {
    let id = usize::try_from(id).map_err(|_| mask_output_error())?;
    if id >= width {
        return Err(mask_output_error());
    }
    let byte = id.checked_div(8).ok_or_else(mask_output_error)?;
    let bit = u32::try_from(id % 8).map_err(|_| mask_output_error())?;
    out[byte] |= 1u8 << bit;
    Ok(())
}

#[cfg(feature = "bench-internals")]
fn mask_output_error() -> CompileError {
    CompileError::new(
        crate::error::ErrorCode::ArtifactOutOfBounds,
        Stage::L4Bind,
        "structured mask token id is outside the vocabulary width",
    )
}

/// Compiled/derived state reused across many match attempts against one `SchemaIR`.
struct MatchCache<'a> {
    ir: &'a SchemaIR,
    engines: FxHashMap<NodeId, Option<RefEngine>>,
    unsafe_memo: FxHashMap<NodeId, bool>,
    patterns: FxHashMap<String, RefEngine>,
    pattern_value_engines: FxHashMap<NodeId, Option<RefEngine>>,
    dynamic_scope: Vec<ResourceId>,
}

impl<'a> MatchCache<'a> {
    fn new(ir: &'a SchemaIR) -> Self {
        Self {
            ir,
            engines: FxHashMap::default(),
            unsafe_memo: FxHashMap::default(),
            patterns: FxHashMap::default(),
            pattern_value_engines: FxHashMap::default(),
            dynamic_scope: Vec::new(),
        }
    }

    fn enter_resource(&mut self, node: NodeId) -> bool {
        let Some(resource) = self.ir.node_resource(node) else {
            return false;
        };
        if self.dynamic_scope.last() == Some(&resource) {
            return false;
        }
        self.dynamic_scope.push(resource);
        true
    }

    fn leave_resource(&mut self, pushed: bool) {
        if pushed {
            self.dynamic_scope.pop();
        }
    }

    fn resolve_dynamic(&self, initial_target: NodeId, anchor: AnchorId) -> NodeId {
        let Some(initial_anchor) = self.ir.anchors().get(anchor.get() as usize) else {
            return initial_target;
        };
        let mut selected = initial_target;
        for resource_id in self.dynamic_scope.iter().rev() {
            let Some(resource) = self.ir.resources().get(resource_id.get() as usize) else {
                continue;
            };
            let start = resource.dynamic_anchors.off as usize;
            let Some(end) = start.checked_add(resource.dynamic_anchors.len as usize) else {
                continue;
            };
            let Some(anchors) = self.ir.anchors().get(start..end) else {
                continue;
            };
            if let Some(binding) = anchors
                .iter()
                .find(|candidate| candidate.name == initial_anchor.name)
            {
                selected = binding.target;
            }
        }
        selected
    }

    /// Returns a byte DFA for `id`, or `None` when the subtree requires structured handling.
    /// Structural checks avoid treating capacity limits as validation failures.
    fn regular_engine(&mut self, id: NodeId) -> Option<&RefEngine> {
        if let std::collections::hash_map::Entry::Vacant(e) = self.engines.entry(id) {
            let unsafe_shuffle =
                subtree_needs_structured_backend(self.ir, id, &mut self.unsafe_memo);
            e.insert(if unsafe_shuffle {
                None
            } else {
                compile_node(self.ir, id).ok()
            });
        }
        self.engines.get(&id).and_then(Option::as_ref)
    }

    /// Returns a DFA for decoded `StringPattern` scalar bytes.
    /// It bypasses document-level regular-engine routing.
    fn pattern_value_engine(&mut self, id: NodeId) -> Option<&RefEngine> {
        if let std::collections::hash_map::Entry::Vacant(e) = self.pattern_value_engines.entry(id) {
            // This engine walks decoded scalar bytes, not complete JSON string values.
            let engine = match self.ir.node(id) {
                Some(Node::StringPattern {
                    regex,
                    charset: Charset::Utf8Any,
                    ..
                }) => self
                    .ir
                    .str_at(*regex)
                    .and_then(|re| build_from_unicode_regex(&format!("(?:{re})")).ok()),
                _ => None,
            };
            e.insert(engine);
        }
        self.pattern_value_engines.get(&id).and_then(Option::as_ref)
    }

    fn pattern_matches(&mut self, regex_src: &str, key: &str) -> bool {
        let (anchor_start, mid, anchor_end) = split_anchors(regex_src);
        if !self.patterns.contains_key(mid) {
            let Ok(engine) = build_from_regex(mid) else {
                return false;
            };
            self.patterns.insert(mid.to_string(), engine);
        }
        let engine = self.patterns.get(mid).expect("just inserted above");
        regex_matches_anywhere(engine, key.as_bytes(), anchor_start, anchor_end)
    }
}

/// Whether the shared route table forbids compiling this subtree as one regular engine.
fn subtree_needs_structured_backend(
    ir: &SchemaIR,
    id: NodeId,
    memo: &mut FxHashMap<NodeId, bool>,
) -> bool {
    if let Some(&v) = memo.get(&id) {
        return v;
    }
    let result = ir
        .route_table()
        .get(id)
        .is_none_or(|route| route.kind != crate::routing::ExecutionKind::Regular);
    memo.insert(id, result);
    result
}

/// Strips a leading `^`/trailing `$` (ECMA search anchors); the middle is what a byte-DFA can run.
pub(crate) fn split_anchors(src: &str) -> (bool, &str, bool) {
    let mut s = src;
    let start = s.starts_with('^');
    if start {
        s = &s[1..];
    }
    let end = s.ends_with('$') && !s.ends_with("\\$");
    if end {
        s = &s[..s.len() - 1];
    }
    (start, s, end)
}

/// Unanchored (or partially anchored) regex search over `bytes`, walking `engine` from every
/// eligible start position since the byte-DFA itself is always fully anchored.
pub(crate) fn regex_matches_anywhere(
    engine: &RefEngine,
    bytes: &[u8],
    anchor_start: bool,
    anchor_end: bool,
) -> bool {
    let starts: Vec<usize> = if anchor_start {
        vec![0]
    } else {
        (0..=bytes.len()).collect()
    };
    for start in starts {
        let mut state = engine.start();
        let mut pos = start;
        loop {
            if engine.is_accepting(state) && (!anchor_end || pos == bytes.len()) {
                return true;
            }
            if pos >= bytes.len() {
                break;
            }
            let Some(next) = engine.consume_token(state, &bytes[pos..pos + 1]) else {
                break;
            };
            if engine.is_dead(next) {
                break;
            }
            state = next;
            pos += 1;
        }
    }
    false
}

/// `bytes` must be exactly one complete value matching `id`, with nothing left over.
fn matches_complete(
    ir: &SchemaIR,
    id: NodeId,
    bytes: &[u8],
    cache: &mut MatchCache<'_>,
    depth: u32,
) -> bool {
    if let Some(engine) = cache.regular_engine(id) {
        return engine.accepts(bytes);
    }
    let mut cur = Cursor::new(bytes);
    cur.skip_ws();
    let ok = match_value(ir, id, cache, &mut cur, depth);
    cur.skip_ws();
    ok && cur.at_end()
}

/// Consumes exactly one value matching `id` starting at `cur`'s position, advancing `cur` past it.
fn match_value(
    ir: &SchemaIR,
    id: NodeId,
    cache: &mut MatchCache<'_>,
    cur: &mut Cursor<'_>,
    depth: u32,
) -> bool {
    if depth > MAX_DEPTH {
        return false;
    }
    let pushed = cache.enter_resource(id);
    let result = match_value_in_scope(ir, id, cache, cur, depth);
    cache.leave_resource(pushed);
    result
}

fn match_value_in_scope(
    ir: &SchemaIR,
    id: NodeId,
    cache: &mut MatchCache<'_>,
    cur: &mut Cursor<'_>,
    depth: u32,
) -> bool {
    if let Some(engine) = cache.regular_engine(id) {
        let Some((start, end)) = span_of_any_value(cur, depth) else {
            return false;
        };
        return engine.accepts(&cur.s[start..end]);
    }
    let Some(node) = ir.node(id) else {
        return false;
    };
    match node {
        Node::Array {
            items,
            min_items,
            max_items,
            unique_items,
            contains,
            ..
        } => match_array(
            ir,
            *items,
            *min_items,
            *max_items,
            *unique_items,
            *contains,
            cache,
            cur,
            depth,
        ),
        Node::Tuple {
            prefix,
            tail,
            min_items,
            max_items,
            unique_items,
            contains,
            ..
        } => match_tuple(
            ir,
            *prefix,
            *tail,
            *min_items,
            *max_items,
            *unique_items,
            *contains,
            cache,
            cur,
            depth,
        ),
        Node::Object {
            fields,
            required,
            dependent,
            dependent_required,
            ..
        } => {
            let known = ir.props_at(*fields).unwrap_or(&[]);
            let dependent = ir.props_at(*dependent).unwrap_or(&[]);
            let dependent_required = ir.dependent_required_at(*dependent_required).unwrap_or(&[]);
            match_object(
                ir,
                known,
                *required,
                &[],
                AdditionalPolicy::Forbid,
                None,
                None,
                None,
                dependent,
                dependent_required,
                cache,
                cur,
                depth,
            )
        }
        Node::OpenObject {
            known,
            known_required,
            patterns,
            additional,
            property_names,
            min_properties,
            max_properties,
            dependent,
            dependent_required,
        } => {
            let known = ir.props_at(*known).unwrap_or(&[]);
            let patterns = ir.props_at(*patterns).unwrap_or(&[]);
            let dependent = ir.props_at(*dependent).unwrap_or(&[]);
            let dependent_required = ir.dependent_required_at(*dependent_required).unwrap_or(&[]);
            match_object(
                ir,
                known,
                *known_required,
                patterns,
                *additional,
                *property_names,
                *min_properties,
                *max_properties,
                dependent,
                dependent_required,
                cache,
                cur,
                depth,
            )
        }
        Node::Union { branches } => {
            let Some((start, end)) = span_of_any_value(cur, depth) else {
                return false;
            };
            let slice = &cur.s[start..end];
            ir.refs_at(*branches)
                .unwrap_or(&[])
                .iter()
                .any(|&b| matches_complete(ir, b, slice, cache, depth + 1))
        }
        Node::Intersection { branches } => {
            let Some((start, end)) = span_of_any_value(cur, depth) else {
                return false;
            };
            let slice = &cur.s[start..end];
            ir.refs_at(*branches)
                .unwrap_or(&[])
                .iter()
                .all(|&b| matches_complete(ir, b, slice, cache, depth + 1))
        }
        Node::ExactlyOne { branches } => {
            let Some((start, end)) = span_of_any_value(cur, depth) else {
                return false;
            };
            let slice = &cur.s[start..end];
            ir.refs_at(*branches)
                .unwrap_or(&[])
                .iter()
                .filter(|&&b| matches_complete(ir, b, slice, cache, depth + 1))
                .count()
                == 1
        }
        Node::Not { inner } => {
            let Some((start, end)) = span_of_any_value(cur, depth) else {
                return false;
            };
            !matches_complete(ir, *inner, &cur.s[start..end], cache, depth + 1)
        }
        Node::Ref { def } => match ir.def_target(*def) {
            Some(target) => match_value(ir, target, cache, cur, depth + 1),
            None => false,
        },
        Node::DynamicRef {
            initial_target,
            anchor,
        } => {
            let target = cache.resolve_dynamic(*initial_target, *anchor);
            match_value(ir, target, cache, cur, depth + 1)
        }
        Node::Unevaluated {
            kind,
            scope,
            unevaluated,
        } => {
            let Some((start, end)) = span_of_any_value(cur, depth) else {
                return false;
            };
            let slice = &cur.s[start..end];
            match kind {
                UnevaluatedKind::Properties => {
                    unevaluated_ok(ir, *scope, *unevaluated, slice, cache, depth + 1)
                }
                UnevaluatedKind::Items => {
                    unevaluated_items_ok(ir, *scope, *unevaluated, slice, cache, depth + 1)
                }
            }
        }
        Node::StringPattern {
            min_len,
            max_len,
            charset: Charset::Utf8CountedCodepoints,
            ..
        } => match cur.parse_string_raw_counted() {
            Some((_, n)) => {
                min_len.is_none_or(|lo| n >= u64::from(lo))
                    && max_len.is_none_or(|hi| n <= u64::from(hi))
            }
            None => false,
        },
        Node::Enum { values } if ir.is_large_string_enum(node) => match cur.parse_string_raw() {
            Some(raw) => decode_json_string(&raw[1..raw.len() - 1]).is_some_and(|s| {
                enum_string_members(ir, *values)
                    .binary_search(&s.as_str())
                    .is_ok()
            }),
            None => false,
        },
        Node::StringPattern {
            charset: Charset::Utf8Any,
            ..
        } if ir.is_user_pattern(node) => match cur.parse_string_raw() {
            Some(raw) => {
                let Some(decoded) = decode_json_string(&raw[1..raw.len() - 1]) else {
                    return false;
                };
                let Some(engine) = cache.pattern_value_engine(id) else {
                    return false;
                };
                let state = walk_engine_over_bytes(engine, decoded.as_bytes());
                engine.is_accepting(state)
            }
            None => false,
        },
        Node::Number {
            integer_only,
            minimum,
            maximum,
            multiple_of,
        } => {
            let start = cur.i;
            cur.parse_number_raw().is_some()
                && oracle_number_accepts(
                    &cur.s[start..cur.i],
                    *integer_only,
                    *minimum,
                    *maximum,
                    *multiple_of,
                )
        }
        Node::Null
        | Node::Boolean
        | Node::Never
        | Node::StringConst { .. }
        | Node::StringPattern { .. }
        | Node::Integer { .. }
        | Node::LexicalNumber { .. }
        | Node::Enum { .. }
        | Node::Unsupported { .. } => false,
    }
}

/// The enum's string members in sorted order (the arena stores enum literals strictly ascending).
fn enum_string_members(ir: &SchemaIR, values: crate::ir::LitSlice) -> Vec<&str> {
    ir.lits_at(values)
        .unwrap_or(&[])
        .iter()
        .filter_map(|l| match l {
            crate::ir::ScalarLit::Str(s) => Some(s.as_str()),
            _ => None,
        })
        .collect()
}

/// Whether some sorted member starts with `prefix` (a contiguous range, found in O(log N)).
fn enum_has_prefix(members: &[&str], prefix: &str) -> bool {
    let pp = members.partition_point(|m| *m < prefix);
    members.get(pp).is_some_and(|m| m.starts_with(prefix))
}

/// Decodes the longest cleanly-decodable prefix of an open JSON string body, stopping before any
/// incomplete trailing escape or UTF-8 byte; a shorter result only loosens the mask, never wrong.
fn decode_confirmed_prefix(body: &[u8]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < body.len() {
        if body[i] == b'\\' {
            match body.get(i + 1) {
                None => break,
                Some(b'u') => {
                    let Some(cp) = parse_hex4(body, i + 2) else {
                        break;
                    };
                    if (0xD800..=0xDBFF).contains(&cp) {
                        let Some(low) = parse_hex4(body, i + 8) else {
                            break;
                        };
                        let c = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                        let Some(ch) = char::from_u32(c) else { break };
                        out.push(ch);
                        i += 12;
                    } else {
                        let Some(ch) = char::from_u32(cp) else { break };
                        out.push(ch);
                        i += 6;
                    }
                }
                Some(&e) => {
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        _ => break,
                    };
                    out.push(ch);
                    i += 2;
                }
            }
        } else {
            let len = match body[i] {
                0x00..=0x7f => 1,
                0xc0..=0xdf => 2,
                0xe0..=0xef => 3,
                0xf0..=0xf7 => 4,
                _ => break,
            };
            let Some(chunk) = body.get(i..i + len) else {
                break;
            };
            let Ok(s) = std::str::from_utf8(chunk) else {
                break;
            };
            out.push_str(s);
            i += len;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn match_array(
    ir: &SchemaIR,
    items: ItemsPolicy,
    min_items: u32,
    max_items: Option<u32>,
    unique_items: bool,
    contains: Option<ContainsConstraint>,
    cache: &mut MatchCache<'_>,
    cur: &mut Cursor<'_>,
    depth: u32,
) -> bool {
    if !cur.eat(b'[') {
        return false;
    }
    cur.skip_ws();
    let mut count = 0u32;
    let mut contains_count = 0u32;
    let mut seen: rustc_hash::FxHashSet<Vec<u8>> = rustc_hash::FxHashSet::default();
    if cur.peek() != Some(b']') {
        loop {
            let start = cur.i;
            let item_ok = match items {
                ItemsPolicy::Schema(id) => match_value(ir, id, cache, cur, depth + 1),
                ItemsPolicy::AllowAny => span_of_any_value(cur, depth + 1).is_some(),
            };
            if !item_ok {
                return false;
            }
            let item_bytes = &cur.s[start..cur.i];
            if unique_items {
                match canonical_value_for_uniqueness(item_bytes) {
                    Some(canon) if !seen.contains(&canon) => {
                        seen.insert(canon);
                    }
                    _ => return false,
                }
            }
            if item_matches_contains(ir, contains, item_bytes, cache, depth + 1) {
                contains_count += 1;
            }
            count += 1;
            cur.skip_ws();
            match cur.peek() {
                Some(b',') => {
                    cur.advance();
                    cur.skip_ws();
                }
                Some(b']') => break,
                _ => return false,
            }
        }
    }
    if !cur.eat(b']') {
        return false;
    }
    count >= min_items
        && max_items.is_none_or(|hi| count <= hi)
        && contains_count_satisfies(contains, contains_count)
}

/// Whether one already-consumed item span counts toward a `contains` constraint.
fn item_matches_contains(
    ir: &SchemaIR,
    contains: Option<ContainsConstraint>,
    item_bytes: &[u8],
    cache: &mut MatchCache<'_>,
    depth: u32,
) -> bool {
    match contains.map(|c| c.policy) {
        None | Some(ContainsPolicy::Never) => false,
        Some(ContainsPolicy::Always) => true,
        Some(ContainsPolicy::Schema(id)) => matches_complete(ir, id, item_bytes, cache, depth),
    }
}

/// Whether a closed array/tuple's `contains_count` satisfies its `contains` bounds, or is
/// trivially satisfied when there is no `contains` constraint at all.
fn contains_count_satisfies(contains: Option<ContainsConstraint>, contains_count: u32) -> bool {
    match contains {
        None => true,
        Some(c) => contains_count >= c.min && c.max.is_none_or(|hi| contains_count <= hi),
    }
}

#[allow(clippy::too_many_arguments)]
fn match_tuple(
    ir: &SchemaIR,
    prefix: crate::ir::RefSlice,
    tail: Option<NodeId>,
    min_items: u32,
    max_items: Option<u32>,
    unique_items: bool,
    contains: Option<ContainsConstraint>,
    cache: &mut MatchCache<'_>,
    cur: &mut Cursor<'_>,
    depth: u32,
) -> bool {
    if !cur.eat(b'[') {
        return false;
    }
    cur.skip_ws();
    let prefix_ids = ir.refs_at(prefix).unwrap_or(&[]);
    let mut count = 0u32;
    let mut contains_count = 0u32;
    let mut seen: rustc_hash::FxHashSet<Vec<u8>> = rustc_hash::FxHashSet::default();
    if cur.peek() != Some(b']') {
        loop {
            let schema = match prefix_ids.get(count as usize) {
                Some(&id) => id,
                None => match tail {
                    Some(id) => id,
                    None => return false,
                },
            };
            let start = cur.i;
            if !match_value(ir, schema, cache, cur, depth + 1) {
                return false;
            }
            let item_bytes = &cur.s[start..cur.i];
            if unique_items {
                match canonical_value_for_uniqueness(item_bytes) {
                    Some(canon) if !seen.contains(&canon) => {
                        seen.insert(canon);
                    }
                    _ => return false,
                }
            }
            if item_matches_contains(ir, contains, item_bytes, cache, depth + 1) {
                contains_count += 1;
            }
            count += 1;
            cur.skip_ws();
            match cur.peek() {
                Some(b',') => {
                    cur.advance();
                    cur.skip_ws();
                }
                Some(b']') => break,
                _ => return false,
            }
        }
    }
    if !cur.eat(b']') {
        return false;
    }
    count >= min_items
        && max_items.is_none_or(|hi| count <= hi)
        && contains_count_satisfies(contains, contains_count)
}

#[allow(clippy::too_many_arguments)]
fn match_object(
    ir: &SchemaIR,
    known: &[(crate::ir::StrRef, NodeId)],
    known_required: crate::ir::BitSlice,
    patterns: &[(crate::ir::StrRef, NodeId)],
    additional: AdditionalPolicy,
    property_names: Option<NodeId>,
    min_properties: Option<u32>,
    max_properties: Option<u32>,
    dependent: &[(crate::ir::StrRef, NodeId)],
    dependent_required: &[crate::ir::DependentRequiredPair],
    cache: &mut MatchCache<'_>,
    cur: &mut Cursor<'_>,
    depth: u32,
) -> bool {
    let obj_start = cur.i;
    if !cur.eat(b'{') {
        return false;
    }
    cur.skip_ws();
    let known: Vec<(&str, NodeId)> = known
        .iter()
        .filter_map(|(name, id)| ir.str_at(*name).map(|s| (s, *id)))
        .collect();
    let patterns: Vec<(&str, NodeId)> = patterns
        .iter()
        .filter_map(|(re, id)| ir.str_at(*re).map(|s| (s, *id)))
        .collect();
    let mut seen_known = vec![false; known.len()];
    let mut seen_keys: rustc_hash::FxHashSet<String> = rustc_hash::FxHashSet::default();
    let mut count = 0u32;
    if cur.peek() != Some(b'}') {
        loop {
            let Some(key) = cur.parse_and_decode_string() else {
                return false;
            };
            if !seen_keys.insert(key.clone()) {
                return false;
            }
            cur.skip_ws();
            if !cur.eat(b':') {
                return false;
            }
            cur.skip_ws();
            if let Some(pn) = property_names {
                let key_json = json_quote(&key);
                if !matches_complete(ir, pn, key_json.as_bytes(), cache, depth + 1) {
                    return false;
                }
            }
            let mut applicable: Vec<NodeId> = Vec::new();
            if let Some(pos) = known.iter().position(|(name, _)| *name == key) {
                seen_known[pos] = true;
                applicable.push(known[pos].1);
            }
            for (regex_src, schema_id) in &patterns {
                if cache.pattern_matches(regex_src, &key) {
                    applicable.push(*schema_id);
                }
            }
            if applicable.is_empty() {
                match additional {
                    AdditionalPolicy::Forbid => return false,
                    AdditionalPolicy::AllowAny | AdditionalPolicy::Open => {
                        if span_of_any_value(cur, depth + 1).is_none() {
                            return false;
                        }
                    }
                    AdditionalPolicy::Schema(id) => {
                        let Some((start, end)) = span_of_any_value(cur, depth + 1) else {
                            return false;
                        };
                        if !matches_complete(ir, id, &cur.s[start..end], cache, depth + 1) {
                            return false;
                        }
                    }
                }
            } else {
                let Some((start, end)) = span_of_any_value(cur, depth + 1) else {
                    return false;
                };
                let slice = &cur.s[start..end];
                if !applicable
                    .iter()
                    .all(|&id| matches_complete(ir, id, slice, cache, depth + 1))
                {
                    return false;
                }
            }
            count += 1;
            cur.skip_ws();
            match cur.peek() {
                Some(b',') => {
                    cur.advance();
                    cur.skip_ws();
                }
                Some(b'}') => break,
                _ => return false,
            }
        }
    }
    if !cur.eat(b'}') {
        return false;
    }
    let obj_end = cur.i;
    for (i, seen) in seen_known.iter().enumerate() {
        if !seen && ir.is_required(known_required, i as u32) {
            return false;
        }
    }
    if !(count >= min_properties.unwrap_or(0) && max_properties.is_none_or(|hi| count <= hi)) {
        return false;
    }
    dependent.iter().all(|(name, schema)| {
        ir.str_at(*name).is_none_or(|key| {
            !seen_keys.contains(key)
                || matches_complete(ir, *schema, &cur.s[obj_start..obj_end], cache, depth + 1)
        })
    }) && dependent_required_satisfied(ir, dependent_required, &seen_keys)
}

fn dependent_required_satisfied(
    ir: &SchemaIR,
    pairs: &[crate::ir::DependentRequiredPair],
    seen: &rustc_hash::FxHashSet<String>,
) -> bool {
    pairs.iter().all(|pair| {
        let Some(trigger) = ir.str_at(pair.trigger) else {
            return false;
        };
        let Some(required) = ir.str_at(pair.required) else {
            return false;
        };
        !seen.contains(trigger) || seen.contains(required)
    })
}

type ObjectMember = (String, (usize, usize));

/// `slice` matches `scope`, and every object property `scope` did not annotate matches `unevaluated`.
fn unevaluated_ok(
    ir: &SchemaIR,
    scope: NodeId,
    unevaluated: NodeId,
    slice: &[u8],
    cache: &mut MatchCache<'_>,
    depth: u32,
) -> bool {
    if depth > MAX_DEPTH || !matches_complete(ir, scope, slice, cache, depth) {
        return false;
    }
    let Some(members) = parse_object_members(slice) else {
        return true;
    };
    let mut evaluated: rustc_hash::FxHashSet<usize> = rustc_hash::FxHashSet::default();
    collect_evaluated_props(ir, scope, slice, &members, cache, &mut evaluated, depth);
    for (i, (_, span)) in members.iter().enumerate() {
        if !evaluated.contains(&i)
            && !matches_complete(ir, unevaluated, &slice[span.0..span.1], cache, depth)
        {
            return false;
        }
    }
    true
}

/// Parses `slice` as one JSON object, returning each top-level key and its value span, or `None`
/// when the value is not an object (`unevaluatedProperties` then does not apply).
fn parse_object_members(slice: &[u8]) -> Option<Vec<ObjectMember>> {
    let mut cur = Cursor::new(slice);
    cur.skip_ws();
    if !cur.eat(b'{') {
        return None;
    }
    cur.skip_ws();
    let mut out = Vec::new();
    if cur.peek() != Some(b'}') {
        loop {
            let key = cur.parse_and_decode_string()?;
            cur.skip_ws();
            if !cur.eat(b':') {
                return None;
            }
            cur.skip_ws();
            let span = span_of_any_value(&mut cur, 0)?;
            out.push((key, span));
            cur.skip_ws();
            match cur.peek() {
                Some(b',') => {
                    cur.advance();
                    cur.skip_ws();
                }
                Some(b'}') => break,
                _ => return None,
            }
        }
    }
    Some(out)
}

/// Marks every `members` index the in-place applicator tree at `node_id` annotates, descending only
/// through applicators sharing this instance location (allOf/anyOf/oneOf/$ref/dependentSchemas).
fn collect_evaluated_props(
    ir: &SchemaIR,
    node_id: NodeId,
    slice: &[u8],
    members: &[ObjectMember],
    cache: &mut MatchCache<'_>,
    out: &mut rustc_hash::FxHashSet<usize>,
    depth: u32,
) {
    if depth > MAX_DEPTH {
        return;
    }
    let Some(node) = ir.node(node_id) else {
        return;
    };
    match node {
        Node::Object {
            fields, dependent, ..
        } => {
            let known = ir.props_at(*fields).unwrap_or(&[]);
            mark_object_members(
                ir,
                known,
                &[],
                AdditionalPolicy::Forbid,
                members,
                cache,
                out,
            );
            descend_dependent(ir, *dependent, slice, members, cache, out, depth);
        }
        Node::OpenObject {
            known,
            patterns,
            additional,
            dependent,
            ..
        } => {
            let known = ir.props_at(*known).unwrap_or(&[]);
            let patterns = ir.props_at(*patterns).unwrap_or(&[]);
            mark_object_members(ir, known, patterns, *additional, members, cache, out);
            descend_dependent(ir, *dependent, slice, members, cache, out, depth);
        }
        Node::Intersection { branches } => {
            for b in ir.refs_at(*branches).unwrap_or(&[]).iter().copied() {
                collect_evaluated_props(ir, b, slice, members, cache, out, depth + 1);
            }
        }
        Node::Union { branches } | Node::ExactlyOne { branches } => {
            for b in ir.refs_at(*branches).unwrap_or(&[]).iter().copied() {
                if matches_complete(ir, b, slice, cache, depth + 1) {
                    collect_evaluated_props(ir, b, slice, members, cache, out, depth + 1);
                }
            }
        }
        Node::Ref { def } => {
            if let Some(target) = ir.def_target(*def) {
                collect_evaluated_props(ir, target, slice, members, cache, out, depth + 1);
            }
        }
        Node::Unevaluated {
            kind,
            scope,
            unevaluated,
        } => {
            let (kind, scope, unevaluated) = (*kind, *scope, *unevaluated);
            collect_evaluated_props(ir, scope, slice, members, cache, out, depth + 1);
            if kind == UnevaluatedKind::Properties {
                for (i, (_, span)) in members.iter().enumerate() {
                    if !out.contains(&i)
                        && matches_complete(
                            ir,
                            unevaluated,
                            &slice[span.0..span.1],
                            cache,
                            depth + 1,
                        )
                    {
                        out.insert(i);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Marks members matching a known name or pattern, and every unmatched member when `additional`
/// annotates (`additionalProperties` as a schema or `true`, never `Open`/`Forbid`).
fn mark_object_members(
    ir: &SchemaIR,
    known: &[(crate::ir::StrRef, NodeId)],
    patterns: &[(crate::ir::StrRef, NodeId)],
    additional: AdditionalPolicy,
    members: &[ObjectMember],
    cache: &mut MatchCache<'_>,
    out: &mut rustc_hash::FxHashSet<usize>,
) {
    for (i, (key, _)) in members.iter().enumerate() {
        if out.contains(&i) {
            continue;
        }
        let matched = known
            .iter()
            .any(|(n, _)| ir.str_at(*n) == Some(key.as_str()))
            || patterns.iter().any(|(re, _)| {
                ir.str_at(*re)
                    .is_some_and(|r| cache.pattern_matches(r, key))
            });
        if matched
            || matches!(
                additional,
                AdditionalPolicy::AllowAny | AdditionalPolicy::Schema(_)
            )
        {
            out.insert(i);
        }
    }
}

/// Descends `dependentSchemas` whose key is present: an in-place applicator on the same instance.
fn descend_dependent(
    ir: &SchemaIR,
    dependent: crate::ir::PropSlice,
    slice: &[u8],
    members: &[ObjectMember],
    cache: &mut MatchCache<'_>,
    out: &mut rustc_hash::FxHashSet<usize>,
    depth: u32,
) {
    for &(name, schema) in ir.props_at(dependent).unwrap_or(&[]) {
        if let Some(k) = ir.str_at(name) {
            if members.iter().any(|(key, _)| key == k) {
                collect_evaluated_props(ir, schema, slice, members, cache, out, depth + 1);
            }
        }
    }
}

/// `slice` matches `scope`, and every array item `scope` did not annotate matches `unevaluated`.
fn unevaluated_items_ok(
    ir: &SchemaIR,
    scope: NodeId,
    unevaluated: NodeId,
    slice: &[u8],
    cache: &mut MatchCache<'_>,
    depth: u32,
) -> bool {
    if depth > MAX_DEPTH || !matches_complete(ir, scope, slice, cache, depth) {
        return false;
    }
    let Some(members) = parse_array_members(slice) else {
        return true;
    };
    let mut evaluated: rustc_hash::FxHashSet<usize> = rustc_hash::FxHashSet::default();
    collect_evaluated_items(ir, scope, slice, &members, cache, &mut evaluated, depth);
    for (i, span) in members.iter().enumerate() {
        if !evaluated.contains(&i)
            && !matches_complete(ir, unevaluated, &slice[span.0..span.1], cache, depth)
        {
            return false;
        }
    }
    true
}

/// Parses `slice` as one JSON array, returning each element's span, or `None` when the value is not
/// an array (`unevaluatedItems` then does not apply).
fn parse_array_members(slice: &[u8]) -> Option<Vec<(usize, usize)>> {
    let mut cur = Cursor::new(slice);
    cur.skip_ws();
    if !cur.eat(b'[') {
        return None;
    }
    cur.skip_ws();
    let mut out = Vec::new();
    if cur.peek() != Some(b']') {
        loop {
            let span = span_of_any_value(&mut cur, 0)?;
            out.push(span);
            cur.skip_ws();
            match cur.peek() {
                Some(b',') => {
                    cur.advance();
                    cur.skip_ws();
                }
                Some(b']') => break,
                _ => return None,
            }
        }
    }
    Some(out)
}

/// Marks every `members` index the in-place applicator tree at `node_id` annotates: `prefixItems`
/// positions, an `items`/tail schema textually present, and every index `contains` matches.
fn collect_evaluated_items(
    ir: &SchemaIR,
    node_id: NodeId,
    slice: &[u8],
    members: &[(usize, usize)],
    cache: &mut MatchCache<'_>,
    out: &mut rustc_hash::FxHashSet<usize>,
    depth: u32,
) {
    if depth > MAX_DEPTH {
        return;
    }
    let Some(node) = ir.node(node_id) else {
        return;
    };
    match node {
        Node::Array {
            contains,
            items_annotates,
            ..
        } => {
            if *items_annotates {
                out.extend(0..members.len());
            }
            mark_contains(ir, contains, slice, members, cache, out);
        }
        Node::Tuple {
            prefix,
            tail_annotates,
            contains,
            ..
        } => {
            let n = ir.refs_at(*prefix).map_or(0, <[NodeId]>::len);
            out.extend(0..n.min(members.len()));
            if *tail_annotates {
                out.extend(n..members.len());
            }
            mark_contains(ir, contains, slice, members, cache, out);
        }
        Node::Intersection { branches } => {
            for b in ir.refs_at(*branches).unwrap_or(&[]).iter().copied() {
                collect_evaluated_items(ir, b, slice, members, cache, out, depth + 1);
            }
        }
        Node::Union { branches } | Node::ExactlyOne { branches } => {
            for b in ir.refs_at(*branches).unwrap_or(&[]).iter().copied() {
                if matches_complete(ir, b, slice, cache, depth + 1) {
                    collect_evaluated_items(ir, b, slice, members, cache, out, depth + 1);
                }
            }
        }
        Node::Ref { def } => {
            if let Some(target) = ir.def_target(*def) {
                collect_evaluated_items(ir, target, slice, members, cache, out, depth + 1);
            }
        }
        Node::Unevaluated {
            kind,
            scope,
            unevaluated,
        } => {
            let (kind, scope, unevaluated) = (*kind, *scope, *unevaluated);
            collect_evaluated_items(ir, scope, slice, members, cache, out, depth + 1);
            if kind == UnevaluatedKind::Items {
                for (i, span) in members.iter().enumerate() {
                    if !out.contains(&i)
                        && matches_complete(
                            ir,
                            unevaluated,
                            &slice[span.0..span.1],
                            cache,
                            depth + 1,
                        )
                    {
                        out.insert(i);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Marks every index whose item matches a `contains` constraint (`Always` marks all, `Schema`
/// marks each match, `Never` marks none).
fn mark_contains(
    ir: &SchemaIR,
    contains: &Option<crate::ir::ContainsConstraint>,
    slice: &[u8],
    members: &[(usize, usize)],
    cache: &mut MatchCache<'_>,
    out: &mut rustc_hash::FxHashSet<usize>,
) {
    let Some(c) = contains else {
        return;
    };
    match c.policy {
        crate::ir::ContainsPolicy::Never => {}
        crate::ir::ContainsPolicy::Always => out.extend(0..members.len()),
        crate::ir::ContainsPolicy::Schema(id) => {
            for (i, span) in members.iter().enumerate() {
                if !out.contains(&i) && matches_complete(ir, id, &slice[span.0..span.1], cache, 0) {
                    out.insert(i);
                }
            }
        }
    }
}

/// JSON-quotes `s` (escaping `"`, `\`, and control bytes) for a synthetic key-as-value check.
fn json_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Canonical JSON value used for structural `uniqueItems` equality.
/// Numbers compare mathematically and object keys are order-independent.
fn canonical_value_for_uniqueness(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut cur = Cursor::new(bytes);
    let out = canonicalize_value(&mut cur, 0)?;
    cur.skip_ws();
    cur.at_end().then_some(out)
}

fn canonicalize_value(cur: &mut Cursor<'_>, depth: u32) -> Option<Vec<u8>> {
    if depth > MAX_DEPTH {
        return None;
    }
    match cur.peek()? {
        b'n' => {
            cur.expect_literal_prefix(b"null").ok()?;
            Some(b"null".to_vec())
        }
        b't' => {
            cur.expect_literal_prefix(b"true").ok()?;
            Some(b"true".to_vec())
        }
        b'f' => {
            cur.expect_literal_prefix(b"false").ok()?;
            Some(b"false".to_vec())
        }
        b'"' => Some(json_quote(&cur.expect_and_decode_string().ok()?).into_bytes()),
        b'-' | b'0'..=b'9' => {
            let start = cur.i;
            scan_number_complete(cur)?;
            Some(canonicalize_number(&cur.s[start..cur.i]))
        }
        b'[' => {
            cur.advance();
            cur.skip_ws();
            let mut items = Vec::new();
            if cur.peek() != Some(b']') {
                loop {
                    items.push(canonicalize_value(cur, depth + 1)?);
                    cur.skip_ws();
                    match cur.peek()? {
                        b',' => {
                            cur.advance();
                            cur.skip_ws();
                        }
                        b']' => break,
                        _ => return None,
                    }
                }
            }
            cur.advance();
            let mut out = vec![b'['];
            out.extend(items.join(&b','));
            out.push(b']');
            Some(out)
        }
        b'{' => {
            cur.advance();
            cur.skip_ws();
            let mut pairs: Vec<(String, Vec<u8>)> = Vec::new();
            if cur.peek() != Some(b'}') {
                loop {
                    let key = cur.expect_and_decode_string().ok()?;
                    cur.skip_ws();
                    if cur.peek()? != b':' {
                        return None;
                    }
                    cur.advance();
                    cur.skip_ws();
                    pairs.push((key, canonicalize_value(cur, depth + 1)?));
                    cur.skip_ws();
                    match cur.peek()? {
                        b',' => {
                            cur.advance();
                            cur.skip_ws();
                        }
                        b'}' => break,
                        _ => return None,
                    }
                }
            }
            cur.advance();
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            let mut out = vec![b'{'];
            for (i, (k, v)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                out.extend(json_quote(k).into_bytes());
                out.push(b':');
                out.extend(v);
            }
            out.push(b'}');
            Some(out)
        }
        _ => None,
    }
}

/// Consumes the complete number span already isolated by the caller.
fn scan_number_complete(cur: &mut Cursor<'_>) -> Option<()> {
    if cur.peek() == Some(b'-') {
        cur.advance();
    }
    match cur.peek()? {
        b'0' => cur.advance(),
        b'1'..=b'9' => {
            cur.advance();
            while matches!(cur.peek(), Some(b'0'..=b'9')) {
                cur.advance();
            }
        }
        _ => return None,
    }
    if cur.peek() == Some(b'.') {
        cur.advance();
        if !matches!(cur.peek(), Some(b'0'..=b'9')) {
            return None;
        }
        while matches!(cur.peek(), Some(b'0'..=b'9')) {
            cur.advance();
        }
    }
    if matches!(cur.peek(), Some(b'e' | b'E')) {
        cur.advance();
        if matches!(cur.peek(), Some(b'+' | b'-')) {
            cur.advance();
        }
        if !matches!(cur.peek(), Some(b'0'..=b'9')) {
            return None;
        }
        while matches!(cur.peek(), Some(b'0'..=b'9')) {
            cur.advance();
        }
    }
    Some(())
}

/// Normalizes a JSON number to `(sign)digits e exponent` form.
/// Equivalent spellings, including `0` and `-0`, produce the same bytes.
fn canonicalize_number(raw: &[u8]) -> Vec<u8> {
    let s = std::str::from_utf8(raw).unwrap_or("0");
    let negative = s.starts_with('-');
    let s = s.strip_prefix('-').unwrap_or(s);
    let (mantissa, exp_part) = match s.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i64>().unwrap_or(0)),
        None => (s, 0),
    };
    let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut digits: Vec<u8> = int_part.bytes().chain(frac_part.bytes()).collect();
    let mut exponent = exp_part - i64::try_from(frac_part.len()).unwrap_or(0);
    while digits.len() > 1 && digits.last() == Some(&b'0') {
        digits.pop();
        exponent += 1;
    }
    let lead = digits
        .iter()
        .take_while(|&&b| b == b'0')
        .count()
        .min(digits.len() - 1);
    digits.drain(0..lead);
    if digits == b"0" {
        return b"0".to_vec();
    }
    let mut out = Vec::new();
    if negative {
        out.push(b'-');
    }
    out.extend(&digits);
    out.push(b'e');
    out.extend(exponent.to_string().into_bytes());
    out
}

struct OracleDecimal {
    negative: bool,
    digits: Vec<u8>,
    exponent: i64,
}

fn parse_oracle_decimal(raw: &[u8]) -> Option<OracleDecimal> {
    let text = std::str::from_utf8(raw).ok()?;
    let (negative, unsigned) = text
        .strip_prefix('-')
        .map_or((false, text), |rest| (true, rest));
    let (mantissa, exponent_text) = unsigned
        .split_once(['e', 'E'])
        .map_or((unsigned, "0"), |parts| parts);
    let exponent = exponent_text.parse::<i64>().unwrap_or_else(|_| {
        if exponent_text.starts_with('-') {
            i64::MIN
        } else {
            i64::MAX
        }
    });
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let total = integer.len().checked_add(fraction.len())?;
    let mut all_digits = Vec::new();
    all_digits.try_reserve_exact(total).ok()?;
    all_digits.extend_from_slice(integer.as_bytes());
    all_digits.extend_from_slice(fraction.as_bytes());
    let Some(first) = all_digits.iter().position(|digit| *digit != b'0') else {
        return Some(OracleDecimal {
            negative: false,
            digits: vec![b'0'],
            exponent: 0,
        });
    };
    let end = all_digits
        .iter()
        .rposition(|digit| *digit != b'0')?
        .checked_add(1)?;
    let trailing = all_digits.len().checked_sub(end)?;
    let fraction_len = i64::try_from(fraction.len()).unwrap_or(i64::MAX);
    let trailing = i64::try_from(trailing).unwrap_or(i64::MAX);
    let exponent = exponent
        .saturating_sub(fraction_len)
        .saturating_add(trailing);
    let mut digits = Vec::new();
    digits.try_reserve_exact(end.checked_sub(first)?).ok()?;
    digits.extend_from_slice(&all_digits[first..end]);
    Some(OracleDecimal {
        negative,
        digits,
        exponent,
    })
}

fn oracle_number_accepts(
    raw: &[u8],
    integer_only: bool,
    minimum: Option<(i128, u32, bool)>,
    maximum: Option<(i128, u32, bool)>,
    multiple_of: Option<(u64, u32)>,
) -> bool {
    let Some(value) = parse_oracle_decimal(raw) else {
        return false;
    };
    if integer_only && value.digits != b"0" && value.exponent < 0 {
        return false;
    }
    if multiple_of
        .is_some_and(|(coefficient, scale)| !oracle_decimal_multiple(&value, coefficient, scale))
    {
        return false;
    }
    if let Some((coefficient, scale, exclusive)) = minimum {
        let ordering = oracle_decimal_bound_cmp(&value, coefficient, scale);
        if ordering == std::cmp::Ordering::Less
            || exclusive && ordering == std::cmp::Ordering::Equal
        {
            return false;
        }
    }
    if let Some((coefficient, scale, exclusive)) = maximum {
        let ordering = oracle_decimal_bound_cmp(&value, coefficient, scale);
        if ordering == std::cmp::Ordering::Greater
            || exclusive && ordering == std::cmp::Ordering::Equal
        {
            return false;
        }
    }
    true
}

fn oracle_number_prefix_sign_viable(
    raw: &[u8],
    minimum: Option<(i128, u32, bool)>,
    maximum: Option<(i128, u32, bool)>,
) -> bool {
    if raw.first() == Some(&b'-') {
        return minimum.is_none_or(|(coefficient, _, exclusive)| {
            coefficient < 0 || coefficient == 0 && !exclusive
        });
    }
    maximum.is_none_or(|(coefficient, _, _)| coefficient >= 0)
}

fn oracle_decimal_multiple(value: &OracleDecimal, modulus: u64, scale: u32) -> bool {
    if value.digits == b"0" {
        return true;
    }
    if modulus == 0 {
        return false;
    }
    let shift = value.exponent.saturating_add(i64::from(scale));
    if shift < 0 {
        return false;
    }
    let mut remainder = 0u64;
    for digit in &value.digits {
        remainder =
            ((u128::from(remainder) * 10 + u128::from(*digit - b'0')) % u128::from(modulus)) as u64;
    }
    let mut factor = 1 % modulus;
    let mut base = 10 % modulus;
    let mut power = u64::try_from(shift).unwrap_or(u64::MAX);
    while power != 0 {
        if power & 1 != 0 {
            factor = ((u128::from(factor) * u128::from(base)) % u128::from(modulus)) as u64;
        }
        base = ((u128::from(base) * u128::from(base)) % u128::from(modulus)) as u64;
        power >>= 1;
    }
    (u128::from(remainder) * u128::from(factor)) % u128::from(modulus) == 0
}

fn oracle_decimal_bound_cmp(
    value: &OracleDecimal,
    coefficient: i128,
    scale: u32,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if value.digits == b"0" {
        return 0i128.cmp(&coefficient);
    }
    let bound_negative = coefficient < 0;
    if value.negative != bound_negative {
        return if value.negative {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let bound = coefficient.unsigned_abs().to_string();
    let value_scientific = i64::try_from(value.digits.len())
        .unwrap_or(i64::MAX)
        .saturating_add(value.exponent);
    let bound_scientific = i64::try_from(bound.len())
        .unwrap_or(i64::MAX)
        .saturating_sub(i64::from(scale));
    let magnitude = match value_scientific.cmp(&bound_scientific) {
        Ordering::Equal => {
            let length = value.digits.len().max(bound.len());
            (0..length)
                .find_map(|index| {
                    let left = value.digits.get(index).copied().unwrap_or(b'0');
                    let right = bound.as_bytes().get(index).copied().unwrap_or(b'0');
                    (left != right).then(|| left.cmp(&right))
                })
                .unwrap_or(Ordering::Equal)
        }
        ordering => ordering,
    };
    if value.negative {
        magnitude.reverse()
    } else {
        magnitude
    }
}

/// Skips over one complete, arbitrary JSON value (used for `additionalProperties: true` and for
/// isolating a value's byte span before checking it against several applicable schemas).
fn span_of_any_value(cur: &mut Cursor<'_>, depth: u32) -> Option<(usize, usize)> {
    if depth > MAX_DEPTH {
        return None;
    }
    let start = cur.i;
    match cur.peek()? {
        b'n' => {
            if !cur.eat_literal(b"null") {
                return None;
            }
        }
        b't' => {
            if !cur.eat_literal(b"true") {
                return None;
            }
        }
        b'f' => {
            if !cur.eat_literal(b"false") {
                return None;
            }
        }
        b'"' => {
            cur.parse_string_raw()?;
        }
        b'-' | b'0'..=b'9' => cur.parse_number_raw()?,
        b'[' => {
            cur.advance();
            cur.skip_ws();
            if cur.peek() != Some(b']') {
                loop {
                    span_of_any_value(cur, depth + 1)?;
                    cur.skip_ws();
                    match cur.peek() {
                        Some(b',') => {
                            cur.advance();
                            cur.skip_ws();
                        }
                        Some(b']') => break,
                        _ => return None,
                    }
                }
            }
            if !cur.eat(b']') {
                return None;
            }
        }
        b'{' => {
            cur.advance();
            cur.skip_ws();
            if cur.peek() != Some(b'}') {
                loop {
                    cur.parse_string_raw()?;
                    cur.skip_ws();
                    if !cur.eat(b':') {
                        return None;
                    }
                    cur.skip_ws();
                    span_of_any_value(cur, depth + 1)?;
                    cur.skip_ws();
                    match cur.peek() {
                        Some(b',') => {
                            cur.advance();
                            cur.skip_ws();
                        }
                        Some(b'}') => break,
                        _ => return None,
                    }
                }
            }
            if !cur.eat(b'}') {
                return None;
            }
        }
        _ => return None,
    }
    Some((start, cur.i))
}

/// A byte cursor over one JSON document; every method advances `i` only on success.
#[derive(Copy, Clone)]
struct Cursor<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn new(s: &'a [u8]) -> Self {
        Self { s, i: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn advance(&mut self) {
        self.i += 1;
    }

    fn eat(&mut self, b: u8) -> bool {
        if self.peek() == Some(b) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn eat_literal(&mut self, lit: &[u8]) -> bool {
        if self.s[self.i..].starts_with(lit) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn at_end(&self) -> bool {
        self.i >= self.s.len()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.advance();
        }
    }

    /// Complete-document counterpart of `expect_string_raw` - same `JsonStringDecoder` validation,
    /// `None` (not a prefix distinction) on any structural or UTF-8/escape failure.
    fn parse_string_raw(&mut self) -> Option<&'a [u8]> {
        self.parse_string_raw_counted().map(|(raw, _)| raw)
    }

    /// `parse_string_raw`, also returning the decoded scalar count - see `expect_string_raw_counted`.
    fn parse_string_raw_counted(&mut self) -> Option<(&'a [u8], u64)> {
        let start = self.i;
        if !self.eat(b'"') {
            return None;
        }
        let mut dec = JsonStringDecoder::new();
        loop {
            let b = self.peek()?;
            if b == b'"' && dec.at_boundary() {
                self.advance();
                break;
            }
            self.advance();
            if dec.push(b) == DecodeStep::Invalid {
                return None;
            }
        }
        Some((&self.s[start..self.i], dec.decoded_codepoints()))
    }

    fn parse_and_decode_string(&mut self) -> Option<String> {
        let raw = self.parse_string_raw()?;
        decode_json_string(&raw[1..raw.len() - 1])
    }

    fn parse_number_raw(&mut self) -> Option<()> {
        let start = self.i;
        self.eat(b'-');
        match self.peek()? {
            b'0' => self.advance(),
            b'1'..=b'9' => {
                self.advance();
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.advance();
                }
            }
            _ => return None,
        }
        if self.peek() == Some(b'.') {
            self.advance();
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return None;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.advance();
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.advance();
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.advance();
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return None;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.advance();
            }
        }
        (self.i > start).then_some(())
    }
}

/// Decodes a JSON string's inner bytes (no surrounding quotes), including `\uXXXX` surrogate pairs.
pub(crate) fn decode_json_string(inner: &[u8]) -> Option<String> {
    let mut out = String::with_capacity(inner.len());
    let mut i = 0;
    while i < inner.len() {
        if inner[i] != b'\\' {
            let start = i;
            while i < inner.len() && inner[i] != b'\\' {
                i += 1;
            }
            out.push_str(std::str::from_utf8(&inner[start..i]).ok()?);
            continue;
        }
        i += 1;
        match *inner.get(i)? {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{8}'),
            b'f' => out.push('\u{c}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                i += 1;
                let cp = parse_hex4(inner, i)?;
                i += 4;
                if (0xD800..=0xDBFF).contains(&cp) {
                    if inner.get(i) != Some(&b'\\') || inner.get(i + 1) != Some(&b'u') {
                        return None;
                    }
                    let low = parse_hex4(inner, i + 2)?;
                    if !(0xDC00..=0xDFFF).contains(&low) {
                        return None;
                    }
                    i += 6;
                    let c = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    out.push(char::from_u32(c)?);
                    continue;
                } else if (0xDC00..=0xDFFF).contains(&cp) {
                    return None;
                } else {
                    out.push(char::from_u32(cp)?);
                }
            }
            _ => return None,
        }
        i += 1;
    }
    Some(out)
}

fn parse_hex4(bytes: &[u8], at: usize) -> Option<u32> {
    let s = std::str::from_utf8(bytes.get(at..at + 4)?).ok()?;
    u32::from_str_radix(s, 16).ok()
}

/// Why a prefix-aware parse step stopped: `Eof` means more bytes could still complete this exact
/// position (the prefix is not yet proven invalid); `Mismatch` means no continuation can fix it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
enum Stop {
    Eof,
    Mismatch,
}

type PResult<T> = Result<T, Stop>;

/// Whether `bytes`, treated as a prefix that more bytes may still be appended to, could still lead
/// to some eventually-accepted document (mirrors `RefEngine::can_continue` for the byte-DFA path).
pub fn can_continue(ir: &SchemaIR, bytes: &[u8]) -> bool {
    if ir.diagnostics().next().is_some() {
        return false;
    }
    can_continue_with_cache(ir, bytes, &mut MatchCache::new(ir))
}

/// `can_continue`, reusing a caller-owned `cache` - see `accepts_with_cache`.
fn can_continue_with_cache(ir: &SchemaIR, bytes: &[u8], cache: &mut MatchCache<'_>) -> bool {
    let mut cur = Cursor::new(bytes);
    cur.skip_ws();
    if matches!(cur.peek(), Some(b'-' | b'0'..=b'9')) {
        if let Some(verdict) = finite_integer_prefix_has_witness(ir, bytes, cache) {
            return verdict;
        }
    }
    match verdict_value(ir, ir.root(), cache, &mut cur, 0) {
        Ok(()) => {
            cur.skip_ws();
            cur.at_end()
        }
        Err(Stop::Eof) => true,
        Err(Stop::Mismatch) => false,
    }
}

fn finite_evaluated_property_names(ir: &SchemaIR, scope: NodeId) -> Option<Vec<&str>> {
    let mut visited = vec![false; ir.node_count()];
    let mut work = vec![scope];
    let mut names = Vec::new();
    while let Some(id) = work.pop() {
        let index = usize::try_from(id.get()).ok()?;
        let seen = visited.get_mut(index)?;
        if *seen {
            continue;
        }
        *seen = true;
        match ir.node(id)? {
            Node::Object {
                fields, dependent, ..
            } => {
                names.extend(
                    ir.props_at(*fields)?
                        .iter()
                        .map(|(name, _)| ir.str_at(*name))
                        .collect::<Option<Vec<_>>>()?,
                );
                work.extend(ir.props_at(*dependent)?.iter().map(|(_, schema)| *schema));
            }
            Node::OpenObject {
                known,
                patterns,
                additional,
                dependent,
                ..
            } => {
                if !ir.props_at(*patterns)?.is_empty()
                    || matches!(
                        additional,
                        AdditionalPolicy::AllowAny | AdditionalPolicy::Schema(_)
                    )
                {
                    return None;
                }
                names.extend(
                    ir.props_at(*known)?
                        .iter()
                        .map(|(name, _)| ir.str_at(*name))
                        .collect::<Option<Vec<_>>>()?,
                );
                work.extend(ir.props_at(*dependent)?.iter().map(|(_, schema)| *schema));
            }
            Node::Intersection { branches }
            | Node::Union { branches }
            | Node::ExactlyOne { branches } => work.extend_from_slice(ir.refs_at(*branches)?),
            Node::Not { .. } => {}
            Node::Ref { .. } | Node::DynamicRef { .. } | Node::Unsupported { .. } => return None,
            Node::Unevaluated {
                kind,
                scope,
                unevaluated,
            } => {
                if *kind == UnevaluatedKind::Properties
                    && !matches!(ir.node(*unevaluated), Some(Node::Never))
                {
                    return None;
                }
                work.push(*scope);
            }
            Node::Null
            | Node::Boolean
            | Node::Never
            | Node::StringConst { .. }
            | Node::StringPattern { .. }
            | Node::Integer { .. }
            | Node::Number { .. }
            | Node::LexicalNumber { .. }
            | Node::Enum { .. }
            | Node::Array { .. }
            | Node::Tuple { .. } => {}
        }
    }
    names.sort_unstable();
    names.dedup();
    Some(names)
}

fn decode_open_string_state(body: &[u8]) -> Option<(String, JsonStringDecoder)> {
    let mut decoder = JsonStringDecoder::new();
    let mut decoded = String::new();
    for &byte in body {
        match decoder.push(byte) {
            DecodeStep::Scalar(value) => decoded.push(value),
            DecodeStep::Continue => {}
            DecodeStep::Invalid => return None,
        }
    }
    Some((decoded, decoder))
}

fn finite_object_key_prefix_viable(bytes: &[u8], names: &[&str], depth: u32) -> bool {
    let mut cur = Cursor::new(bytes);
    cur.skip_ws();
    if cur.expect(b'{').is_err() {
        return true;
    }
    let mut seen = rustc_hash::FxHashSet::default();
    let mut member_required = false;
    loop {
        cur.skip_ws();
        if cur.at_end() {
            return !member_required || names.iter().any(|name| !seen.contains(*name));
        }
        if cur.peek() == Some(b'}') {
            return true;
        }
        let start = cur.i;
        let key = match cur.expect_string_raw() {
            Ok(raw) => {
                let Some(key) = decode_json_string(&raw[1..raw.len() - 1]) else {
                    return true;
                };
                if !names.contains(&key.as_str()) || !seen.insert(key.clone()) {
                    return false;
                }
                key
            }
            Err(Stop::Eof) => {
                let Some(body) = cur.s.get(start + 1..cur.i) else {
                    return true;
                };
                let Some((prefix, decoder)) = decode_open_string_state(body) else {
                    return true;
                };
                let viable_prefix = |candidate: &str| {
                    names
                        .iter()
                        .any(|name| name.starts_with(candidate) && !seen.contains(*name))
                };
                if decoder.at_boundary() {
                    return viable_prefix(&prefix);
                }
                return decoder
                    .any_pending_scalar_satisfies_with_supplementary(
                        4096,
                        |scalar| {
                            let mut candidate = prefix.clone();
                            candidate.push(scalar);
                            viable_prefix(&candidate)
                        },
                        || {
                            Some(names.iter().any(|name| {
                                !seen.contains(*name)
                                    && name
                                        .strip_prefix(&prefix)
                                        .is_some_and(|suffix| suffix.chars().next().is_some())
                            }))
                        },
                        |lo, hi| {
                            Some(names.iter().any(|name| {
                                !seen.contains(*name)
                                    && name.strip_prefix(&prefix).is_some_and(|suffix| {
                                        suffix.chars().next().is_some_and(|scalar| {
                                            (lo..=hi).contains(&u32::from(scalar))
                                        })
                                    })
                            }))
                        },
                    )
                    .unwrap_or(true);
            }
            Err(Stop::Mismatch) => return true,
        };
        debug_assert!(seen.contains(&key));
        cur.skip_ws();
        if cur.expect(b':').is_err() {
            return true;
        }
        cur.skip_ws();
        if verdict_span_any_value(&mut cur, depth + 1).is_err() {
            return true;
        }
        cur.skip_ws();
        match cur.peek() {
            Some(b',') => {
                cur.advance();
                member_required = true;
            }
            Some(b'}') | None => return true,
            Some(_) => return true,
        }
    }
}

#[derive(Clone, Copy)]
struct IntegerRange {
    minimum: Option<i128>,
    maximum: Option<i128>,
    integer_only: bool,
}

fn finite_integer_prefix_has_witness(
    ir: &SchemaIR,
    bytes: &[u8],
    cache: &mut MatchCache<'_>,
) -> Option<bool> {
    let raw = bytes
        .iter()
        .position(|byte| !matches!(byte, b' ' | b'\t' | b'\n' | b'\r'))
        .map_or(&[][..], |start| &bytes[start..]);
    if raw.len() > 8 {
        return None;
    }
    let range = integer_range(ir, ir.root(), 0)?;
    if !range.integer_only {
        return None;
    }
    let minimum = range.minimum?;
    let maximum = range.maximum?;
    let width = maximum.checked_sub(minimum)?.checked_add(1)?;
    if !(0..=256).contains(&width) {
        return None;
    }
    Some((minimum..=maximum).any(|value| {
        integer_has_number_prefix(value, raw)
            && matches_complete(ir, ir.root(), value.to_string().as_bytes(), cache, 0)
    }))
}

fn integer_range(ir: &SchemaIR, id: NodeId, depth: u32) -> Option<IntegerRange> {
    if depth > MAX_DEPTH {
        return None;
    }
    match ir.node(id)? {
        Node::Integer {
            minimum, maximum, ..
        } => Some(IntegerRange {
            minimum: minimum.map(i128::from),
            maximum: maximum.map(i128::from),
            integer_only: true,
        }),
        Node::Number {
            integer_only,
            minimum,
            maximum,
            ..
        } => Some(IntegerRange {
            minimum: minimum.and_then(integer_minimum),
            maximum: maximum.and_then(integer_maximum),
            integer_only: *integer_only,
        }),
        Node::Intersection { branches } => {
            let mut out = IntegerRange {
                minimum: None,
                maximum: None,
                integer_only: false,
            };
            for &branch in ir.refs_at(*branches)? {
                let next = integer_range(ir, branch, depth + 1)?;
                out.minimum = match (out.minimum, next.minimum) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (a, b) => a.or(b),
                };
                out.maximum = match (out.maximum, next.maximum) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
                out.integer_only |= next.integer_only;
            }
            Some(out)
        }
        Node::Union { branches } | Node::ExactlyOne { branches } => {
            let mut ids = ir.refs_at(*branches)?.iter().copied();
            let first = integer_range(ir, ids.next()?, depth + 1)?;
            ids.try_fold(first, |out, branch| {
                let next = integer_range(ir, branch, depth + 1)?;
                Some(IntegerRange {
                    minimum: Some(out.minimum?.min(next.minimum?)),
                    maximum: Some(out.maximum?.max(next.maximum?)),
                    integer_only: out.integer_only && next.integer_only,
                })
            })
        }
        Node::Ref { def } => integer_range(ir, ir.def_target(*def)?, depth + 1),
        _ => None,
    }
}

fn integer_minimum((coefficient, scale, exclusive): (i128, u32, bool)) -> Option<i128> {
    let divisor = 10i128.checked_pow(scale)?;
    let quotient = coefficient.div_euclid(divisor);
    let remainder = coefficient.rem_euclid(divisor);
    quotient.checked_add(i128::from(remainder != 0 || exclusive))
}

fn integer_maximum((coefficient, scale, exclusive): (i128, u32, bool)) -> Option<i128> {
    let divisor = 10i128.checked_pow(scale)?;
    let quotient = coefficient.div_euclid(divisor);
    quotient.checked_sub(i128::from(
        exclusive && coefficient.rem_euclid(divisor) == 0,
    ))
}

fn integer_has_number_prefix(value: i128, prefix: &[u8]) -> bool {
    let canonical = value.to_string();
    if canonical.as_bytes().starts_with(prefix) {
        return true;
    }
    let (sign, digits) = canonical
        .strip_prefix('-')
        .map_or(("", canonical.as_str()), |digits| ("-", digits));
    let limit = prefix.len().saturating_add(2).min(512);
    for trailing in 0..=limit {
        let mut coefficient = String::with_capacity(digits.len().saturating_add(trailing));
        coefficient.push_str(digits);
        coefficient.extend(std::iter::repeat_n('0', trailing));
        for point in 1..=coefficient.len() {
            let fraction = coefficient.len().saturating_sub(point);
            let exponent = i128::try_from(fraction)
                .ok()
                .and_then(|fraction| fraction.checked_sub(i128::try_from(trailing).ok()?));
            let Some(exponent) = exponent else {
                continue;
            };
            if number_spelling_has_prefix(sign, &coefficient, point, exponent, prefix, limit) {
                return true;
            }
        }
        for leading in 0..=limit {
            let exponent = leading
                .checked_add(coefficient.len())
                .and_then(|value| value.checked_sub(trailing))
                .and_then(|value| i128::try_from(value).ok());
            let Some(exponent) = exponent else {
                continue;
            };
            let fraction = format!("{}{}", "0".repeat(leading), coefficient);
            if number_spelling_has_prefix(sign, &format!("0{fraction}"), 1, exponent, prefix, limit)
            {
                return true;
            }
        }
    }
    false
}

fn number_spelling_has_prefix(
    sign: &str,
    coefficient: &str,
    point: usize,
    exponent: i128,
    prefix: &[u8],
    zero_limit: usize,
) -> bool {
    let (integer, fraction) = coefficient.split_at(point);
    let significands = if fraction.is_empty() {
        vec![format!("{sign}{integer}"), format!("{sign}{integer}.0")]
    } else {
        vec![format!("{sign}{integer}.{fraction}")]
    };
    for significand in significands {
        if exponent == 0 && significand.as_bytes().starts_with(prefix) {
            return true;
        }
        let magnitude = exponent.unsigned_abs().to_string();
        for zeros in 0..=zero_limit {
            let padded = format!("{}{}", "0".repeat(zeros), magnitude);
            for marker in ['e', 'E'] {
                let signed = if exponent < 0 {
                    format!("{significand}{marker}-{padded}")
                } else {
                    format!("{significand}{marker}+{padded}")
                };
                if signed.as_bytes().starts_with(prefix) {
                    return true;
                }
                if exponent >= 0 {
                    let plain = format!("{significand}{marker}{padded}");
                    if plain.as_bytes().starts_with(prefix) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

impl<'a> Cursor<'a> {
    fn peek_or_eof(&self) -> PResult<u8> {
        self.peek().ok_or(Stop::Eof)
    }

    fn expect(&mut self, b: u8) -> PResult<()> {
        match self.peek() {
            Some(x) if x == b => {
                self.advance();
                Ok(())
            }
            Some(_) => Err(Stop::Mismatch),
            None => Err(Stop::Eof),
        }
    }

    /// Matches as much of `lit` as `self` has remaining bytes for; `Eof` iff every available byte
    /// agrees with `lit` but fewer than `lit.len()` bytes remain.
    fn expect_literal_prefix(&mut self, lit: &[u8]) -> PResult<()> {
        let avail = self.s.len() - self.i;
        let check_len = avail.min(lit.len());
        if self.s[self.i..self.i + check_len] != lit[..check_len] {
            return Err(Stop::Mismatch);
        }
        if avail < lit.len() {
            return Err(Stop::Eof);
        }
        self.i += lit.len();
        Ok(())
    }

    /// Consumes one quoted JSON string, returning `Eof` until it closes.
    /// A `JsonStringDecoder` validates UTF-8 and escape sequences.
    fn expect_string_raw(&mut self) -> PResult<&'a [u8]> {
        self.expect_string_raw_counted().map(|(raw, _)| raw)
    }

    /// `expect_string_raw`, also returning the decoded scalar count from the same decoder pass -
    /// a `minLength`/`maxLength` caller needs no second full re-decode to count codepoints.
    fn expect_string_raw_counted(&mut self) -> PResult<(&'a [u8], u64)> {
        let start = self.i;
        self.expect(b'"')?;
        let mut dec = JsonStringDecoder::new();
        loop {
            let b = self.peek_or_eof()?;
            if b == b'"' && dec.at_boundary() {
                self.advance();
                break;
            }
            self.advance();
            if dec.push(b) == DecodeStep::Invalid {
                return Err(Stop::Mismatch);
            }
        }
        Ok((&self.s[start..self.i], dec.decoded_codepoints()))
    }

    fn expect_and_decode_string(&mut self) -> PResult<String> {
        let raw = self.expect_string_raw()?;
        decode_json_string(&raw[1..raw.len() - 1]).ok_or(Stop::Mismatch)
    }

    /// Consumes a JSON number from the current position.
    /// A legal number at end-of-buffer remains ambiguous and returns `Eof`.
    fn expect_number(&mut self) -> PResult<()> {
        if self.peek() == Some(b'-') {
            self.advance();
        }
        match self.peek_or_eof()? {
            b'0' => self.advance(),
            b'1'..=b'9' => {
                self.advance();
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.advance();
                }
            }
            _ => return Err(Stop::Mismatch),
        }
        if self.at_end() {
            return Err(Stop::Eof);
        }
        if self.peek() == Some(b'.') {
            self.advance();
            match self.peek_or_eof()? {
                b'0'..=b'9' => {
                    self.advance();
                    while matches!(self.peek(), Some(b'0'..=b'9')) {
                        self.advance();
                    }
                }
                _ => return Err(Stop::Mismatch),
            }
            if self.at_end() {
                return Err(Stop::Eof);
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.advance();
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.advance();
            }
            match self.peek_or_eof()? {
                b'0'..=b'9' => {
                    self.advance();
                    while matches!(self.peek(), Some(b'0'..=b'9')) {
                        self.advance();
                    }
                }
                _ => return Err(Stop::Mismatch),
            }
            if self.at_end() {
                return Err(Stop::Eof);
            }
        }
        Ok(())
    }
}

/// Walks `engine` from `cur` using maximal munch.
/// A live end-of-buffer is ambiguous, while a dead prefix is a mismatch.
fn walk_regular(engine: &RefEngine, cur: &mut Cursor<'_>) -> PResult<()> {
    let mut state = engine.start();
    loop {
        let Some(b) = cur.peek() else {
            return Err(Stop::Eof);
        };
        let next = engine
            .consume_token(state, &[b])
            .unwrap_or_else(|| engine.dead());
        if engine.is_dead(next) {
            return if engine.is_accepting(state) {
                Ok(())
            } else {
                Err(Stop::Mismatch)
            };
        }
        state = next;
        cur.advance();
    }
}

/// Walks several engines that must accept the same value.
/// A value completes only when every engine ends in an accepting state.
fn walk_regular_all(engines: &[&RefEngine], cur: &mut Cursor<'_>) -> PResult<()> {
    let mut states: Vec<StateId> = engines.iter().map(|e| e.start()).collect();
    loop {
        let Some(b) = cur.peek() else {
            return Err(Stop::Eof);
        };
        let mut next_states = Vec::with_capacity(states.len());
        for (engine, &state) in engines.iter().zip(&states) {
            let next = engine
                .consume_token(state, &[b])
                .unwrap_or_else(|| engine.dead());
            if engine.is_dead(next) && !engine.is_accepting(state) {
                return Err(Stop::Mismatch);
            }
            next_states.push(next);
        }
        if engines.iter().zip(&next_states).all(|(e, &s)| e.is_dead(s)) {
            return Ok(());
        }
        if engines.iter().zip(&next_states).any(|(e, &s)| e.is_dead(s)) {
            return Err(Stop::Mismatch);
        }
        states = next_states;
        cur.advance();
    }
}

/// Finds a definitively closed JSON-value span in a prefix.
/// Closed spans can use ordinary complete-value validation.
fn verdict_span_any_value(cur: &mut Cursor<'_>, depth: u32) -> PResult<(usize, usize)> {
    if depth > MAX_DEPTH {
        return Err(Stop::Mismatch);
    }
    let start = cur.i;
    match cur.peek_or_eof()? {
        b'n' => cur.expect_literal_prefix(b"null")?,
        b't' => cur.expect_literal_prefix(b"true")?,
        b'f' => cur.expect_literal_prefix(b"false")?,
        b'"' => {
            cur.expect_string_raw()?;
        }
        b'-' | b'0'..=b'9' => cur.expect_number()?,
        b'[' => {
            cur.advance();
            cur.skip_ws();
            if cur.peek() != Some(b']') {
                loop {
                    verdict_span_any_value(cur, depth + 1)?;
                    cur.skip_ws();
                    match cur.peek_or_eof()? {
                        b',' => {
                            cur.advance();
                            cur.skip_ws();
                        }
                        b']' => break,
                        _ => return Err(Stop::Mismatch),
                    }
                }
            }
            cur.expect(b']')?;
        }
        b'{' => {
            cur.advance();
            cur.skip_ws();
            if cur.peek() != Some(b'}') {
                loop {
                    cur.expect_string_raw()?;
                    cur.skip_ws();
                    cur.expect(b':')?;
                    cur.skip_ws();
                    verdict_span_any_value(cur, depth + 1)?;
                    cur.skip_ws();
                    match cur.peek_or_eof()? {
                        b',' => {
                            cur.advance();
                            cur.skip_ws();
                        }
                        b'}' => break,
                        _ => return Err(Stop::Mismatch),
                    }
                }
            }
            cur.expect(b'}')?;
        }
        _ => return Err(Stop::Mismatch),
    }
    Ok((start, cur.i))
}

/// Prefix-aware analogue of `match_value`.
fn verdict_value(
    ir: &SchemaIR,
    id: NodeId,
    cache: &mut MatchCache<'_>,
    cur: &mut Cursor<'_>,
    depth: u32,
) -> PResult<()> {
    if depth > MAX_DEPTH {
        return Err(Stop::Mismatch);
    }
    let pushed = cache.enter_resource(id);
    let result = verdict_value_in_scope(ir, id, cache, cur, depth);
    cache.leave_resource(pushed);
    result
}

fn verdict_value_in_scope(
    ir: &SchemaIR,
    id: NodeId,
    cache: &mut MatchCache<'_>,
    cur: &mut Cursor<'_>,
    depth: u32,
) -> PResult<()> {
    if let Some(engine) = cache.regular_engine(id) {
        return walk_regular(engine, cur);
    }
    let Some(node) = ir.node(id) else {
        return Err(Stop::Mismatch);
    };
    match node {
        Node::Array {
            items,
            min_items,
            max_items,
            unique_items,
            contains,
            ..
        } => verdict_array(
            ir,
            *items,
            *min_items,
            *max_items,
            *unique_items,
            *contains,
            cache,
            cur,
            depth,
        ),
        Node::Tuple {
            prefix,
            tail,
            min_items,
            max_items,
            unique_items,
            contains,
            ..
        } => verdict_tuple(
            ir,
            *prefix,
            *tail,
            *min_items,
            *max_items,
            *unique_items,
            *contains,
            cache,
            cur,
            depth,
        ),
        Node::Object {
            fields,
            required,
            dependent,
            dependent_required,
            ..
        } => {
            let known = ir.props_at(*fields).unwrap_or(&[]);
            let dependent = ir.props_at(*dependent).unwrap_or(&[]);
            let dependent_required = ir.dependent_required_at(*dependent_required).unwrap_or(&[]);
            verdict_object(
                ir,
                known,
                *required,
                &[],
                AdditionalPolicy::Forbid,
                None,
                None,
                None,
                dependent,
                dependent_required,
                cache,
                cur,
                depth,
            )
        }
        Node::OpenObject {
            known,
            known_required,
            patterns,
            additional,
            property_names,
            min_properties,
            max_properties,
            dependent,
            dependent_required,
        } => {
            let known = ir.props_at(*known).unwrap_or(&[]);
            let patterns = ir.props_at(*patterns).unwrap_or(&[]);
            let dependent = ir.props_at(*dependent).unwrap_or(&[]);
            let dependent_required = ir.dependent_required_at(*dependent_required).unwrap_or(&[]);
            verdict_object(
                ir,
                known,
                *known_required,
                patterns,
                *additional,
                *property_names,
                *min_properties,
                *max_properties,
                dependent,
                dependent_required,
                cache,
                cur,
                depth,
            )
        }
        Node::Union { branches } => {
            let ids = ir.refs_at(*branches).unwrap_or(&[]);
            let mut saw_eof = false;
            for &branch in ids {
                let mut branch_cur = *cur;
                match verdict_value(ir, branch, cache, &mut branch_cur, depth + 1) {
                    Ok(()) => {
                        *cur = branch_cur;
                        return Ok(());
                    }
                    Err(Stop::Eof) => saw_eof = true,
                    Err(Stop::Mismatch) => {}
                }
            }
            Err(if saw_eof { Stop::Eof } else { Stop::Mismatch })
        }
        Node::Intersection { branches } => {
            let ids = ir.refs_at(*branches).unwrap_or(&[]);
            let mut completed = None;
            let mut saw_eof = false;
            for &branch in ids {
                let mut branch_cur = *cur;
                match verdict_value(ir, branch, cache, &mut branch_cur, depth + 1) {
                    Ok(()) => completed = Some(branch_cur),
                    Err(Stop::Eof) => saw_eof = true,
                    Err(Stop::Mismatch) => return Err(Stop::Mismatch),
                }
            }
            if saw_eof {
                Err(Stop::Eof)
            } else if let Some(done) = completed {
                *cur = done;
                Ok(())
            } else {
                Err(Stop::Mismatch)
            }
        }
        Node::ExactlyOne { branches } => {
            let ids = ir.refs_at(*branches).unwrap_or(&[]);
            let mut completed = None;
            let mut matches = 0usize;
            let mut saw_eof = false;
            let mut viable = Vec::new();
            for &branch in ids {
                let mut branch_cur = *cur;
                match verdict_value(ir, branch, cache, &mut branch_cur, depth + 1) {
                    Ok(()) => {
                        completed = Some(branch_cur);
                        matches += 1;
                        viable.push(branch);
                    }
                    Err(Stop::Eof) => {
                        saw_eof = true;
                        viable.push(branch);
                    }
                    Err(Stop::Mismatch) => {}
                }
            }
            let unique_viable = viable
                .iter()
                .any(|branch| viable.iter().filter(|other| *other == branch).count() == 1);
            if saw_eof && unique_viable {
                Err(Stop::Eof)
            } else if matches == 1 {
                if let Some(done) = completed {
                    *cur = done;
                }
                Ok(())
            } else {
                Err(Stop::Mismatch)
            }
        }
        Node::Not { inner } => {
            let (start, end) = verdict_span_any_value(cur, depth)?;
            let slice = &cur.s[start..end];
            if matches_complete(ir, *inner, slice, cache, depth + 1) {
                Err(Stop::Mismatch)
            } else {
                Ok(())
            }
        }
        Node::Ref { def } => match ir.def_target(*def) {
            Some(target) => verdict_value(ir, target, cache, cur, depth + 1),
            None => Err(Stop::Mismatch),
        },
        Node::DynamicRef {
            initial_target,
            anchor,
        } => {
            let target = cache.resolve_dynamic(*initial_target, *anchor);
            verdict_value(ir, target, cache, cur, depth + 1)
        }
        Node::Unevaluated {
            kind,
            scope,
            unevaluated,
        } => {
            let initial = *cur;
            let (start, end) = match verdict_span_any_value(cur, depth) {
                Ok(span) => span,
                Err(Stop::Eof) => {
                    if *kind == UnevaluatedKind::Properties
                        && matches!(ir.node(*unevaluated), Some(Node::Never))
                        && finite_evaluated_property_names(ir, *scope).is_some_and(|names| {
                            !finite_object_key_prefix_viable(&cur.s[initial.i..], &names, depth + 1)
                        })
                    {
                        return Err(Stop::Mismatch);
                    }
                    let mut scope_cur = initial;
                    return match verdict_value(ir, *scope, cache, &mut scope_cur, depth + 1) {
                        Err(Stop::Mismatch) => Err(Stop::Mismatch),
                        Ok(()) | Err(Stop::Eof) => Err(Stop::Eof),
                    };
                }
                Err(Stop::Mismatch) => return Err(Stop::Mismatch),
            };
            let slice = &cur.s[start..end];
            let ok = match kind {
                UnevaluatedKind::Properties => {
                    unevaluated_ok(ir, *scope, *unevaluated, slice, cache, depth + 1)
                }
                UnevaluatedKind::Items => {
                    unevaluated_items_ok(ir, *scope, *unevaluated, slice, cache, depth + 1)
                }
            };
            if ok {
                Ok(())
            } else {
                Err(Stop::Mismatch)
            }
        }
        Node::StringPattern {
            min_len,
            max_len,
            charset: Charset::Utf8CountedCodepoints,
            ..
        } => {
            let start = cur.i;
            match cur.expect_string_raw_counted() {
                Ok((_, n))
                    if min_len.is_none_or(|lo| n >= u64::from(lo))
                        && max_len.is_none_or(|hi| n <= u64::from(hi)) =>
                {
                    Ok(())
                }
                Ok(_) => Err(Stop::Mismatch),
                Err(Stop::Eof) => {
                    let body = cur.s.get(start + 1..cur.i).unwrap_or(&[]);
                    if max_len.is_some_and(|hi| confirmed_codepoints(body) > u64::from(hi)) {
                        Err(Stop::Mismatch)
                    } else {
                        Err(Stop::Eof)
                    }
                }
                Err(e) => Err(e),
            }
        }
        Node::Enum { values } if ir.is_large_string_enum(node) => {
            let members = enum_string_members(ir, *values);
            let start = cur.i;
            match cur.expect_string_raw() {
                Ok(raw) => {
                    let s = decode_json_string(&raw[1..raw.len() - 1]).ok_or(Stop::Mismatch)?;
                    if members.binary_search(&s.as_str()).is_ok() {
                        Ok(())
                    } else {
                        Err(Stop::Mismatch)
                    }
                }
                Err(Stop::Eof) => {
                    let body = cur.s.get(start + 1..cur.i).unwrap_or(&[]);
                    if enum_has_prefix(&members, &decode_confirmed_prefix(body)) {
                        Err(Stop::Eof)
                    } else {
                        Err(Stop::Mismatch)
                    }
                }
                Err(e) => Err(e),
            }
        }
        Node::StringPattern {
            charset: Charset::Utf8Any,
            ..
        } if ir.is_user_pattern(node) => {
            let start = cur.i;
            match cur.expect_string_raw() {
                Ok(raw) => {
                    let decoded =
                        decode_json_string(&raw[1..raw.len() - 1]).ok_or(Stop::Mismatch)?;
                    let Some(engine) = cache.pattern_value_engine(id) else {
                        return Err(Stop::Mismatch);
                    };
                    let state = walk_engine_over_bytes(engine, decoded.as_bytes());
                    if engine.is_accepting(state) {
                        Ok(())
                    } else {
                        Err(Stop::Mismatch)
                    }
                }
                Err(Stop::Eof) => {
                    let body = cur.s.get(start + 1..cur.i).unwrap_or(&[]);
                    let confirmed = decode_confirmed_prefix(body);
                    let Some(engine) = cache.pattern_value_engine(id) else {
                        return Err(Stop::Mismatch);
                    };
                    let state = walk_engine_over_bytes(engine, confirmed.as_bytes());
                    if engine.is_dead(state) {
                        Err(Stop::Mismatch)
                    } else {
                        Err(Stop::Eof)
                    }
                }
                Err(e) => Err(e),
            }
        }
        Node::Number {
            integer_only,
            minimum,
            maximum,
            multiple_of,
        } => {
            let start = cur.i;
            match cur.expect_number() {
                Ok(()) => oracle_number_accepts(
                    &cur.s[start..cur.i],
                    *integer_only,
                    *minimum,
                    *maximum,
                    *multiple_of,
                )
                .then_some(())
                .ok_or(Stop::Mismatch),
                Err(Stop::Eof) => {
                    if oracle_number_prefix_sign_viable(&cur.s[start..cur.i], *minimum, *maximum) {
                        Err(Stop::Eof)
                    } else {
                        Err(Stop::Mismatch)
                    }
                }
                Err(error) => Err(error),
            }
        }
        Node::Null
        | Node::Boolean
        | Node::Never
        | Node::StringConst { .. }
        | Node::StringPattern { .. }
        | Node::Integer { .. }
        | Node::LexicalNumber { .. }
        | Node::Enum { .. }
        | Node::Unsupported { .. } => Err(Stop::Mismatch),
    }
}

/// Walks `engine` over `bytes` one at a time, stopping early once dead - the shared primitive
/// behind both the complete-value and confirmed-prefix arms of the decoded-pattern match.
fn walk_engine_over_bytes(engine: &RefEngine, bytes: &[u8]) -> StateId {
    let mut state = engine.start();
    for &b in bytes {
        state = engine
            .consume_token(state, &[b])
            .unwrap_or_else(|| engine.dead());
        if engine.is_dead(state) {
            break;
        }
    }
    state
}

/// Complete codepoints in an open JSON string body, stopping at any incomplete trailing escape or
/// UTF-8 sequence; an undercount is safe here (it only gates early over-max rejection).
fn confirmed_codepoints(body: &[u8]) -> u64 {
    let mut i = 0;
    let mut n = 0u64;
    while i < body.len() {
        let b = body[i];
        let step = match b {
            b'\\' => match body.get(i + 1) {
                None => break,
                Some(b'u') => {
                    if parse_hex4(body, i + 2).is_some_and(|h| (0xD800..=0xDBFF).contains(&h)) {
                        12
                    } else if i + 6 <= body.len() {
                        6
                    } else {
                        break;
                    }
                }
                Some(_) => 2,
            },
            0x00..=0x7f => 1,
            0x80..=0xbf => 1,
            0xc0..=0xdf => 2,
            0xe0..=0xef => 3,
            _ => 4,
        };
        if i + step > body.len() {
            break;
        }
        i += step;
        n += 1;
    }
    n
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn verdict_array(
    ir: &SchemaIR,
    items: ItemsPolicy,
    min_items: u32,
    max_items: Option<u32>,
    unique_items: bool,
    contains: Option<ContainsConstraint>,
    cache: &mut MatchCache<'_>,
    cur: &mut Cursor<'_>,
    depth: u32,
) -> PResult<()> {
    cur.expect(b'[')?;
    cur.skip_ws();
    let mut count = 0u32;
    let mut contains_count = 0u32;
    let mut seen: rustc_hash::FxHashSet<Vec<u8>> = rustc_hash::FxHashSet::default();
    if cur.peek() != Some(b']') {
        loop {
            let start = cur.i;
            match items {
                ItemsPolicy::Schema(id) => verdict_value(ir, id, cache, cur, depth + 1)?,
                ItemsPolicy::AllowAny => {
                    verdict_span_any_value(cur, depth + 1)?;
                }
            }
            let item_bytes = &cur.s[start..cur.i];
            if unique_items {
                match canonical_value_for_uniqueness(item_bytes) {
                    Some(canon) if !seen.contains(&canon) => {
                        seen.insert(canon);
                    }
                    _ => return Err(Stop::Mismatch),
                }
            }
            if item_matches_contains(ir, contains, item_bytes, cache, depth + 1) {
                contains_count += 1;
            }
            count += 1;
            if max_items.is_some_and(|hi| count > hi) {
                return Err(Stop::Mismatch);
            }
            cur.skip_ws();
            match cur.peek_or_eof()? {
                b',' => {
                    if max_items.is_some_and(|hi| count >= hi) {
                        return Err(Stop::Mismatch);
                    }
                    cur.advance();
                    cur.skip_ws();
                }
                b']' => break,
                _ => return Err(Stop::Mismatch),
            }
        }
    }
    cur.expect(b']')?;
    if count >= min_items
        && max_items.is_none_or(|hi| count <= hi)
        && contains_count_satisfies(contains, contains_count)
    {
        Ok(())
    } else {
        Err(Stop::Mismatch)
    }
}

#[allow(clippy::too_many_arguments)]
fn verdict_tuple(
    ir: &SchemaIR,
    prefix: crate::ir::RefSlice,
    tail: Option<NodeId>,
    min_items: u32,
    max_items: Option<u32>,
    unique_items: bool,
    contains: Option<ContainsConstraint>,
    cache: &mut MatchCache<'_>,
    cur: &mut Cursor<'_>,
    depth: u32,
) -> PResult<()> {
    cur.expect(b'[')?;
    cur.skip_ws();
    let prefix_ids = ir.refs_at(prefix).unwrap_or(&[]);
    let mut count = 0u32;
    let mut contains_count = 0u32;
    let mut seen: rustc_hash::FxHashSet<Vec<u8>> = rustc_hash::FxHashSet::default();
    if cur.peek() != Some(b']') {
        loop {
            let schema = match prefix_ids.get(count as usize) {
                Some(&id) => id,
                None => tail.ok_or(Stop::Mismatch)?,
            };
            let start = cur.i;
            verdict_value(ir, schema, cache, cur, depth + 1)?;
            let item_bytes = &cur.s[start..cur.i];
            if unique_items {
                match canonical_value_for_uniqueness(item_bytes) {
                    Some(canon) if !seen.contains(&canon) => {
                        seen.insert(canon);
                    }
                    _ => return Err(Stop::Mismatch),
                }
            }
            if item_matches_contains(ir, contains, item_bytes, cache, depth + 1) {
                contains_count += 1;
            }
            count += 1;
            if max_items.is_some_and(|hi| count > hi) {
                return Err(Stop::Mismatch);
            }
            cur.skip_ws();
            match cur.peek_or_eof()? {
                b',' => {
                    if max_items.is_some_and(|hi| count >= hi) {
                        return Err(Stop::Mismatch);
                    }
                    cur.advance();
                    cur.skip_ws();
                }
                b']' => break,
                _ => return Err(Stop::Mismatch),
            }
        }
    }
    cur.expect(b']')?;
    if count >= min_items
        && max_items.is_none_or(|hi| count <= hi)
        && contains_count_satisfies(contains, contains_count)
    {
        Ok(())
    } else {
        Err(Stop::Mismatch)
    }
}

#[allow(clippy::too_many_arguments)]
fn verdict_object(
    ir: &SchemaIR,
    known: &[(crate::ir::StrRef, NodeId)],
    known_required: crate::ir::BitSlice,
    patterns: &[(crate::ir::StrRef, NodeId)],
    additional: AdditionalPolicy,
    property_names: Option<NodeId>,
    min_properties: Option<u32>,
    max_properties: Option<u32>,
    dependent: &[(crate::ir::StrRef, NodeId)],
    dependent_required: &[crate::ir::DependentRequiredPair],
    cache: &mut MatchCache<'_>,
    cur: &mut Cursor<'_>,
    depth: u32,
) -> PResult<()> {
    let obj_start = cur.i;
    cur.expect(b'{')?;
    cur.skip_ws();
    let known: Vec<(&str, NodeId)> = known
        .iter()
        .filter_map(|(name, id)| ir.str_at(*name).map(|s| (s, *id)))
        .collect();
    let patterns: Vec<(&str, NodeId)> = patterns
        .iter()
        .filter_map(|(re, id)| ir.str_at(*re).map(|s| (s, *id)))
        .collect();
    let mut seen_known = vec![false; known.len()];
    let mut seen_keys: rustc_hash::FxHashSet<String> = rustc_hash::FxHashSet::default();
    let mut count = 0u32;
    if cur.peek() != Some(b'}') {
        loop {
            let key_start = cur.i;
            let key = match cur.expect_and_decode_string() {
                Ok(key) => key,
                Err(Stop::Eof) => {
                    let Some(body) = cur.s.get(key_start + 1..cur.i) else {
                        return Err(Stop::Eof);
                    };
                    let Some((prefix, decoder)) = decode_open_string_state(body) else {
                        return Err(Stop::Eof);
                    };
                    let viable = if decoder.at_boundary() {
                        reference_object_key_prefix_viable(
                            ir, &known, &patterns, additional, &seen_keys, &prefix, cache,
                        )
                    } else {
                        decoder.any_pending_scalar_satisfies_with_supplementary(
                            4096,
                            |scalar| {
                                let mut composed = prefix.clone();
                                composed.push(scalar);
                                reference_object_key_prefix_viable(
                                    ir, &known, &patterns, additional, &seen_keys, &composed, cache,
                                )
                                .unwrap_or(true)
                            },
                            || {
                                reference_object_key_any_scalar_viable(
                                    ir, &known, &patterns, additional, &seen_keys, &prefix,
                                )
                            },
                            |start, end| {
                                reference_object_key_supplementary_viable(
                                    ir, &known, &patterns, additional, &seen_keys, &prefix, start,
                                    end,
                                )
                            },
                        )
                    };
                    return match viable {
                        Some(true) | None => Err(Stop::Eof),
                        Some(false) => Err(Stop::Mismatch),
                    };
                }
                Err(Stop::Mismatch) => return Err(Stop::Mismatch),
            };
            if !seen_keys.insert(key.clone()) {
                return Err(Stop::Mismatch);
            }
            let known_key = known.iter().any(|(name, _)| *name == key);
            let pattern_key = patterns
                .iter()
                .any(|(source, _)| cache.pattern_matches(source, &key));
            if !known_key && !pattern_key && matches!(additional, AdditionalPolicy::Forbid) {
                return Err(Stop::Mismatch);
            }
            if let Some(pn) = property_names {
                let key_json = json_quote(&key);
                if !matches_complete(ir, pn, key_json.as_bytes(), cache, depth + 1) {
                    return Err(Stop::Mismatch);
                }
            }
            cur.skip_ws();
            cur.expect(b':')?;
            cur.skip_ws();
            let mut applicable: Vec<NodeId> = Vec::new();
            if let Some(pos) = known.iter().position(|(name, _)| *name == key) {
                seen_known[pos] = true;
                applicable.push(known[pos].1);
            }
            for (regex_src, schema_id) in &patterns {
                if cache.pattern_matches(regex_src, &key) {
                    applicable.push(*schema_id);
                }
            }
            if applicable.is_empty() {
                match additional {
                    AdditionalPolicy::Forbid => return Err(Stop::Mismatch),
                    AdditionalPolicy::AllowAny | AdditionalPolicy::Open => {
                        verdict_span_any_value(cur, depth + 1)?;
                    }
                    AdditionalPolicy::Schema(id) => {
                        verdict_value(ir, id, cache, cur, depth + 1)?;
                    }
                }
            } else if let [only] = applicable[..] {
                verdict_value(ir, only, cache, cur, depth + 1)?;
            } else {
                for &id in &applicable {
                    cache.regular_engine(id);
                }
                let engines: Option<Vec<&RefEngine>> = applicable
                    .iter()
                    .map(|id| cache.engines.get(id).and_then(Option::as_ref))
                    .collect();
                match engines {
                    Some(engines) => walk_regular_all(&engines, cur)?,
                    None => {
                        let (start, end) = verdict_span_any_value(cur, depth + 1)?;
                        let slice = &cur.s[start..end];
                        if !applicable
                            .iter()
                            .all(|&id| matches_complete(ir, id, slice, cache, depth + 1))
                        {
                            return Err(Stop::Mismatch);
                        }
                    }
                }
            }
            count += 1;
            cur.skip_ws();
            match cur.peek_or_eof()? {
                b',' => {
                    if max_properties.is_some_and(|maximum| count >= maximum)
                        || reference_object_key_prefix_viable(
                            ir, &known, &patterns, additional, &seen_keys, "", cache,
                        ) == Some(false)
                    {
                        return Err(Stop::Mismatch);
                    }
                    cur.advance();
                    cur.skip_ws();
                }
                b'}' => break,
                _ => return Err(Stop::Mismatch),
            }
        }
    }
    cur.expect(b'}')?;
    let obj_end = cur.i;
    for (i, seen) in seen_known.iter().enumerate() {
        if !seen && ir.is_required(known_required, i as u32) {
            return Err(Stop::Mismatch);
        }
    }
    if !(count >= min_properties.unwrap_or(0) && max_properties.is_none_or(|hi| count <= hi)) {
        return Err(Stop::Mismatch);
    }
    let satisfied = dependent.iter().all(|(name, schema)| {
        ir.str_at(*name).is_none_or(|key| {
            !seen_keys.contains(key)
                || matches_complete(ir, *schema, &cur.s[obj_start..obj_end], cache, depth + 1)
        })
    });
    if satisfied && dependent_required_satisfied(ir, dependent_required, &seen_keys) {
        Ok(())
    } else {
        Err(Stop::Mismatch)
    }
}

fn reference_object_key_prefix_viable(
    ir: &SchemaIR,
    known: &[(&str, NodeId)],
    patterns: &[(&str, NodeId)],
    additional: AdditionalPolicy,
    seen: &rustc_hash::FxHashSet<String>,
    prefix: &str,
    cache: &mut MatchCache<'_>,
) -> Option<bool> {
    if !matches!(additional, AdditionalPolicy::Forbid) {
        return None;
    }
    if known.iter().any(|&(name, value)| {
        !matches!(ir.node(value), Some(Node::Never))
            && name.starts_with(prefix)
            && !seen.contains(name)
            && !patterns.iter().any(|(source, pattern_value)| {
                matches!(ir.node(*pattern_value), Some(Node::Never))
                    && cache.pattern_matches(source, name)
            })
    }) {
        return Some(true);
    }
    let Some(engine) = reference_viable_pattern_engine(ir, patterns)? else {
        return Some(false);
    };
    let Some(state) = engine.consume_token(engine.start(), prefix.as_bytes()) else {
        return Some(false);
    };
    let unavailable = seen
        .iter()
        .filter(|name| name.starts_with(prefix) && engine.accepts(name.as_bytes()))
        .count() as u64;
    Some(engine.residual_cardinality(state) > unavailable)
}

fn reference_viable_pattern_engine(
    ir: &SchemaIR,
    patterns: &[(&str, NodeId)],
) -> Option<Option<RefEngine>> {
    let mut live = Vec::new();
    let mut false_patterns = Vec::new();
    live.try_reserve_exact(patterns.len()).ok()?;
    false_patterns.try_reserve_exact(patterns.len()).ok()?;
    for &(source, value) in patterns {
        let engine = super::pattern::build_property_search_engine(source).ok()?;
        if matches!(ir.node(value), Some(Node::Never)) {
            false_patterns.push(engine);
        } else {
            live.push(engine);
        }
    }
    if live.is_empty() {
        return Some(None);
    }
    let live_refs: Vec<_> = live.iter().collect();
    let allowed = ProductAutomaton::build_reachable(&live_refs, Combinator::Any)
        .ok()?
        .into_engine()
        .ok()?;
    if false_patterns.is_empty() {
        return Some(Some(allowed));
    }
    let false_refs: Vec<_> = false_patterns.iter().collect();
    let forbidden = ProductAutomaton::build_reachable(&false_refs, Combinator::Any)
        .ok()?
        .into_engine()
        .ok()?;
    ProductAutomaton::build_reachable(&[&allowed, &forbidden], Combinator::Difference)
        .ok()?
        .into_engine()
        .ok()
        .map(Some)
}

#[allow(clippy::too_many_arguments)]
fn reference_object_key_supplementary_viable(
    ir: &SchemaIR,
    known: &[(&str, NodeId)],
    patterns: &[(&str, NodeId)],
    additional: AdditionalPolicy,
    seen: &rustc_hash::FxHashSet<String>,
    prefix: &str,
    start: u32,
    end: u32,
) -> Option<bool> {
    if !matches!(additional, AdditionalPolicy::Forbid) {
        return None;
    }
    if known.iter().any(|&(name, value)| {
        !matches!(ir.node(value), Some(Node::Never))
            && !seen.contains(name)
            && name.strip_prefix(prefix).is_some_and(|rest| {
                rest.chars()
                    .next()
                    .is_some_and(|scalar| (start..=end).contains(&u32::from(scalar)))
            })
            && !patterns.iter().any(|(source, pattern_value)| {
                matches!(ir.node(*pattern_value), Some(Node::Never))
                    && super::pattern::build_property_search_engine(source)
                        .ok()
                        .is_some_and(|engine| engine.accepts(name.as_bytes()))
            })
    }) {
        return Some(true);
    }
    let Some(engine) = reference_viable_pattern_engine(ir, patterns)? else {
        return Some(false);
    };
    let Some(state) = engine.consume_token(engine.start(), prefix.as_bytes()) else {
        return Some(false);
    };
    engine.can_consume_supplementary_range(state, start, end)
}

#[allow(clippy::too_many_arguments)]
fn reference_object_key_any_scalar_viable(
    ir: &SchemaIR,
    known: &[(&str, NodeId)],
    patterns: &[(&str, NodeId)],
    additional: AdditionalPolicy,
    seen: &rustc_hash::FxHashSet<String>,
    prefix: &str,
) -> Option<bool> {
    if !matches!(additional, AdditionalPolicy::Forbid) {
        return None;
    }
    if known.iter().any(|&(name, value)| {
        !matches!(ir.node(value), Some(Node::Never))
            && !seen.contains(name)
            && name
                .strip_prefix(prefix)
                .is_some_and(|rest| !rest.is_empty())
            && !patterns.iter().any(|(source, pattern_value)| {
                matches!(ir.node(*pattern_value), Some(Node::Never))
                    && super::pattern::build_property_search_engine(source)
                        .ok()
                        .is_some_and(|engine| engine.accepts(name.as_bytes()))
            })
    }) {
        return Some(true);
    }
    let Some(engine) = reference_viable_pattern_engine(ir, patterns)? else {
        return Some(false);
    };
    let Some(state) = engine.consume_token(engine.start(), prefix.as_bytes()) else {
        return Some(false);
    };
    let unavailable = seen
        .iter()
        .filter(|name| name.starts_with(prefix) && engine.accepts(name.as_bytes()))
        .count();
    match engine.can_consume_any_scalar(state) {
        Some(true) if unavailable == 0 => Some(true),
        Some(true) | None => None,
        Some(false) => Some(false),
    }
}

#[cfg(test)]
type StructuredMatcher = ReferenceMatcher;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_pure_reference_cycle_rejects_every_instance_and_prefix() {
        let ir = crate::frontend::schema_to_ir(
            r##"{"$ref":"#"}"##,
            crate::ir::CompileOptions::default(),
        )
        .unwrap();
        for bytes in [b"".as_slice(), b"null", b"{}", b"[", b" "] {
            assert!(!accepts(&ir, bytes));
            assert!(!can_continue(&ir, bytes));
        }
    }

    #[test]
    fn mutual_pure_reference_cycle_rejects_every_instance_and_prefix() {
        let ir = crate::frontend::schema_to_ir(
            r##"{
                "$defs": {
                    "a": {"$ref":"#/$defs/b"},
                    "b": {"$ref":"#/$defs/a"}
                },
                "$ref":"#/$defs/a"
            }"##,
            crate::ir::CompileOptions::default(),
        )
        .unwrap();
        for bytes in [b"".as_slice(), b"null", b"{}", b"[", b" "] {
            assert!(!accepts(&ir, bytes));
            assert!(!can_continue(&ir, bytes));
        }
    }
    use crate::ir::{Builder, Charset, CompileOptions, ObjectClosure};
    use crate::schema_to_ir;

    fn opts() -> CompileOptions {
        CompileOptions::default()
    }

    fn from_schema(schema: &str) -> SchemaIR {
        schema_to_ir(schema, opts()).unwrap()
    }

    #[test]
    fn native_codepoint_counting_matches_ground_truth_on_hard_strings() {
        let bodies = [
            "",
            "abc",
            "a\\nb",
            "a\\\\b",
            "a\\\"b",
            "\\u00e9",
            "\\ud83d\\ude00",
            "\\ud83d\\ude00\\ud83d\\ude00",
            "\\u4e2d\\u6587",
            "héllo",
            "日本語テキスト",
            "😀😀😀",
        ];
        for &min in &[None, Some(0u32), Some(1), Some(2), Some(300)] {
            for &max in &[None, Some(0u32), Some(1), Some(2), Some(257), Some(1024)] {
                if let (Some(lo), Some(hi)) = (min, max) {
                    if hi < lo {
                        continue;
                    }
                }
                let mut parts = vec![r#""type":"string""#.to_string()];
                if let Some(v) = min {
                    parts.push(format!(r#""minLength":{v}"#));
                }
                if let Some(v) = max {
                    parts.push(format!(r#""maxLength":{v}"#));
                }
                let ir = from_schema(&format!("{{{}}}", parts.join(",")));
                assert!(ir.diagnostics().count() == 0);
                for body in &bodies {
                    let text = format!("\"{body}\"");
                    let true_len = serde_json::from_str::<String>(&text)
                        .unwrap()
                        .chars()
                        .count() as u64;
                    let expect = min.is_none_or(|lo| true_len >= u64::from(lo))
                        && max.is_none_or(|hi| true_len <= u64::from(hi));
                    assert_eq!(
                        accepts(&ir, text.as_bytes()),
                        expect,
                        "body={body:?} min={min:?} max={max:?} true_len={true_len}"
                    );
                }
            }
        }
    }

    #[test]
    fn native_codepoint_bound_routes_to_structured_and_masks_correctly() {
        let ir = from_schema(r#"{"type":"string","maxLength":300}"#);
        assert!(ir.requires_structured_backend());
        assert!(accepts(&ir, format!("\"{}\"", "x".repeat(300)).as_bytes()));
        assert!(!accepts(&ir, format!("\"{}\"", "x".repeat(301)).as_bytes()));
        let ir = std::sync::Arc::new(ir);
        let mut m = StructuredMatcher::new(ir);
        assert!(m.advance(b"\""));
        for _ in 0..300 {
            assert!(m.advance(b"x"));
        }
        assert!(!m.advance(b"x"), "the 301st content char must be rejected");
        assert!(m.advance(b"\""));
        assert!(m.is_accepting());
    }

    /// An open object forces the structured backend; a sibling `pattern` field's own compiled
    /// regex must still gate the mid-string mask, not the generic "any plain byte is legal" path.
    #[test]
    fn pattern_constrained_field_mask_rejects_a_char_outside_the_pattern_mid_string() {
        let ir = std::sync::Arc::new(from_schema(
            r#"{"type":"object","properties":{"code":{"type":"string","pattern":"^[a-z]+$"}}}"#,
        ));
        let mut m = StructuredMatcher::new(ir);
        for tok in [b"{".as_slice(), b"\"", b"code", b"\"", b":", b"\"", b"ab"] {
            assert!(m.advance(tok), "prefix {:?}", std::str::from_utf8(tok));
        }
        let toks: Vec<(Vec<u8>, Vec<u32>)> = (b'a'..=b'z')
            .chain(b'0'..=b'9')
            .map(|c| (vec![c], vec![u32::from(c)]))
            .collect();
        let mask = m
            .allowed_mask_from_records(u32::from(b'z') as usize + 1, records(&toks).into_iter())
            .unwrap();
        for c in b'0'..=b'9' {
            assert!(
                !mask.get(TokenId(u32::from(c))),
                "digit {} violates ^[a-z]+$ but was allowed",
                c as char
            );
        }
        for c in b'a'..=b'z' {
            assert!(
                mask.get(TokenId(u32::from(c))),
                "letter {} should stay allowed",
                c as char
            );
        }
    }

    /// A single-digit-splitting tokenizer must not trip the bare-number check mid-string here either.
    #[test]
    fn native_codepoint_bound_mask_does_not_apply_leading_zero_rule_inside_the_string() {
        let ir = std::sync::Arc::new(from_schema(r#"{"type":"string","maxLength":300}"#));
        let mut m = StructuredMatcher::new(ir);
        for tok in [b"\"".as_slice(), b"1"] {
            assert!(m.advance(tok));
        }
        let toks: Vec<(Vec<u8>, Vec<u32>)> = (0..10)
            .map(|d| (d.to_string().into_bytes(), vec![d]))
            .collect();
        let mask = m
            .allowed_mask_from_records(10, records(&toks).into_iter())
            .unwrap();
        for d in 0..10u32 {
            assert!(
                mask.get(TokenId(d)),
                "digit {d} is plain string content, not a number"
            );
        }
    }

    #[test]
    fn large_string_enum_uses_the_direct_trie_and_matches_exactly() {
        let members: Vec<String> = (0..500).map(|i| format!("value_{i:04}")).collect();
        let vals = members
            .iter()
            .map(|m| format!("\"{m}\""))
            .collect::<Vec<_>>()
            .join(",");
        let ir = from_schema(&format!("{{\"enum\":[{vals}]}}"));
        assert!(!ir.requires_structured_backend());
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(accepts(&ir, b"\"value_0000\""));
        assert!(accepts(&ir, b"\"value_0499\""));
        assert!(!accepts(&ir, b"\"value_0500\""));
        assert!(!accepts(&ir, b"\"nope\""));
        assert!(!accepts(&ir, b"\"value_000\""));
    }

    #[test]
    fn large_string_enum_masks_toward_valid_members_only() {
        let members: Vec<String> = (0..300).map(|i| format!("cat_{i:03}")).collect();
        let vals = members
            .iter()
            .map(|m| format!("\"{m}\""))
            .collect::<Vec<_>>()
            .join(",");
        let ir = std::sync::Arc::new(from_schema(&format!("{{\"enum\":[{vals}]}}")));
        let mut m = StructuredMatcher::new(ir);
        for tok in [b"\"".as_slice(), b"cat_", b"0", b"4", b"2"] {
            assert!(m.advance(tok), "prefix {:?}", std::str::from_utf8(tok));
        }
        assert!(!m.advance(b"9"), "cat_0429 is not a member");
        assert!(m.advance(b"\""));
        assert!(m.is_accepting());
    }

    /// A single-digit-splitting tokenizer (e.g. Qwen2.5) must not trip the bare-number check.
    #[test]
    fn large_string_enum_mask_does_not_apply_leading_zero_rule_inside_the_string() {
        let members: Vec<String> = (0..300).map(|i| format!("cat_{i:03}")).collect();
        let vals = members
            .iter()
            .map(|m| format!("\"{m}\""))
            .collect::<Vec<_>>()
            .join(",");
        let ir = std::sync::Arc::new(from_schema(&format!("{{\"enum\":[{vals}]}}")));
        let mut m = StructuredMatcher::new(ir);
        for tok in [b"\"".as_slice(), b"cat_", b"0"] {
            assert!(m.advance(tok));
        }
        let toks: Vec<(Vec<u8>, Vec<u32>)> = (0..10)
            .map(|d| (d.to_string().into_bytes(), vec![d]))
            .collect();
        let mask = m
            .allowed_mask_from_records(10, records(&toks).into_iter())
            .unwrap();
        assert!(
            mask.get(TokenId(4)),
            "'4' must continue toward cat_040..cat_049, not be rejected as a leading zero"
        );
    }

    #[test]
    fn ten_thousand_member_enum_compiles_without_crash() {
        let members: Vec<String> = (0..10_000).map(|i| format!("m{i:05}")).collect();
        let vals = members
            .iter()
            .map(|m| format!("\"{m}\""))
            .collect::<Vec<_>>()
            .join(",");
        let ir = from_schema(&format!("{{\"enum\":[{vals}]}}"));
        assert!(!ir.requires_structured_backend());
        assert!(accepts(&ir, b"\"m00000\""));
        assert!(accepts(&ir, b"\"m09999\""));
        assert!(!accepts(&ir, b"\"m10000\""));
    }

    #[test]
    fn small_enum_stays_on_fast_dfa_path() {
        let ir = from_schema(r#"{"enum":["a","b","c"]}"#);
        assert!(!ir.requires_structured_backend());
    }

    /// `oneOf`/`allOf` of two `multipleOf` branches blows the product+elimination DFA size cap;
    /// routing to the structured backend avoids the compile ever attempting it.
    #[test]
    fn one_of_multiple_of_routes_to_structured_and_matches() {
        let ir = from_schema(
            r#"{"oneOf":[{"type":"integer","multipleOf":3},{"type":"integer","multipleOf":5}]}"#,
        );
        assert!(ir.requires_structured_backend());
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(accepts(&ir, b"3"), "3 is a multiple of 3 only");
        assert!(accepts(&ir, b"5"), "5 is a multiple of 5 only");
        assert!(!accepts(&ir, b"15"), "15 satisfies both: exactly-one fails");
        assert!(!accepts(&ir, b"7"), "7 satisfies neither");
    }

    #[test]
    fn all_of_multiple_of_routes_to_structured_and_matches() {
        let ir = from_schema(
            r#"{"allOf":[{"type":"integer","multipleOf":3},{"type":"integer","multipleOf":5}]}"#,
        );
        assert!(ir.requires_structured_backend());
        assert!(accepts(&ir, b"15"), "15 is a multiple of both 3 and 5");
        assert!(!accepts(&ir, b"3"), "3 is not a multiple of 5");
    }

    /// Nesting `allOf` inside `oneOf` is caught by the arena-wide scan without extra code: the
    /// inner `allOf` node is itself visited and flagged on its own multipleOf branches.
    #[test]
    fn one_of_of_all_of_multiple_of_routes_to_structured_and_matches() {
        let ir = from_schema(
            r#"{"oneOf":[
                {"allOf":[{"type":"integer","multipleOf":3},{"type":"integer","minimum":0,"maximum":30}]},
                {"allOf":[{"type":"integer","multipleOf":5},{"type":"integer","minimum":0,"maximum":30}]}
            ]}"#,
        );
        assert!(ir.requires_structured_backend());
        assert!(accepts(&ir, b"3"), "3 satisfies the first branch only");
        assert!(accepts(&ir, b"25"), "25 satisfies the second branch only");
        assert!(
            !accepts(&ir, b"15"),
            "15 satisfies both branches: exactly-one fails"
        );
    }

    /// The number-tail fast path assumes any digit continuation is presumptively fine; that
    /// breaks once a bound makes further digits provably unable to fix an exactly-one failure.
    #[test]
    fn one_of_of_all_of_multiple_of_mask_agrees_with_advance_on_a_bounded_doomed_prefix() {
        let ir = std::sync::Arc::new(from_schema(
            r#"{"oneOf":[
                {"allOf":[{"type":"integer","multipleOf":3},{"type":"integer","minimum":0,"maximum":30}]},
                {"allOf":[{"type":"integer","multipleOf":5},{"type":"integer","minimum":0,"maximum":30}]}
            ]}"#,
        ));
        let mut m = StructuredMatcher::new(ir.clone());
        assert!(m.advance(b"1"));
        let mut direct = StructuredMatcher::new(ir.clone());
        assert!(direct.advance(b"1"));
        assert!(!direct.advance(b"5"));
        let toks: Vec<(Vec<u8>, Vec<u32>)> = (0..10)
            .map(|d| (d.to_string().into_bytes(), vec![d]))
            .collect();
        let isolated = m
            .allowed_mask_from_records(10, [(b"5".as_slice(), [5u32].as_slice())].into_iter())
            .unwrap();
        assert!(!isolated.get(TokenId(5)));
        for prior in 0..5 {
            let mut probe = StructuredMatcher::new(ir.clone());
            assert!(probe.advance(b"1"));
            let prior_bytes = prior.to_string().into_bytes();
            let probe_mask = probe
                .allowed_mask_from_records(
                    10,
                    [
                        (prior_bytes.as_slice(), [prior].as_slice()),
                        (b"5".as_slice(), [5u32].as_slice()),
                    ]
                    .into_iter(),
                )
                .unwrap();
            assert!(!probe_mask.get(TokenId(5)), "prior={prior}");
        }
        let mask = m
            .allowed_mask_from_records(10, records(&toks).into_iter())
            .unwrap();
        assert!(
            !mask.get(TokenId(5)),
            "150-159 all exceed the [0,30] bound: the mask must agree with advance"
        );
        assert!(!m.advance(b"5"), "commit must match the mask verdict");
    }

    #[test]
    fn pattern_properties_multiple_of_overlap_routes_to_structured_and_matches() {
        let ir = from_schema(
            r#"{"type":"object","patternProperties":{
                "^a":{"type":"integer","multipleOf":3},
                "b$":{"type":"integer","multipleOf":5}
            }}"#,
        );
        assert!(ir.requires_structured_backend());
        assert!(
            accepts(&ir, br#"{"ab":15}"#),
            "15 is a multiple of both 3 and 5"
        );
        assert!(!accepts(&ir, br#"{"ab":3}"#), "3 is not a multiple of 5");
    }

    #[test]
    fn any_of_multiple_of_stays_on_fast_dfa_path() {
        let ir = from_schema(
            r#"{"anyOf":[{"type":"integer","multipleOf":3},{"type":"integer","multipleOf":5}]}"#,
        );
        assert!(!ir.requires_structured_backend());
        assert!(
            accepts(&ir, b"15"),
            "15 satisfies both, anyOf only needs one"
        );
        assert!(!accepts(&ir, b"7"), "7 satisfies neither");
    }

    /// A non-scalar (object) `oneOf` branch is just as likely to blow a product-combination cap
    /// as a `multipleOf` branch; the routing predicate must catch both, not just the first found.
    #[test]
    fn one_of_object_shaped_branches_routes_to_structured_and_matches() {
        let ir = from_schema(
            r#"{"oneOf":[
                {"type":"object","additionalProperties":false,"required":["kind","name","value"],
                 "properties":{"kind":{"const":"a"},"name":{"type":"string"},"value":{"type":"integer"}}},
                {"type":"object","additionalProperties":false,"required":["kind","label","amount"],
                 "properties":{"kind":{"const":"b"},"label":{"type":"string"},"amount":{"type":"number"}}}
            ]}"#,
        );
        assert!(ir.requires_structured_backend());
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(accepts(&ir, br#"{"kind":"a","name":"x","value":1}"#));
        assert!(accepts(&ir, br#"{"kind":"b","label":"y","amount":2.5}"#));
        assert!(
            !accepts(
                &ir,
                br#"{"kind":"a","name":"x","value":1,"label":"y","amount":2.5}"#
            ),
            "satisfying both branches at once fails exactly-one"
        );
        assert!(!accepts(&ir, br#"{"kind":"c","name":"x","value":1}"#));
    }

    /// A fixed-size array branch past the safe combinator threshold blows the same product cap
    /// as a multi-field object; routing must catch it too.
    #[test]
    fn one_of_large_array_branches_routes_to_structured_and_matches() {
        let ir = from_schema(
            r#"{"oneOf":[
                {"type":"array","items":{"type":"string"},"minItems":10,"maxItems":10},
                {"type":"array","items":{"type":"integer"},"minItems":10,"maxItems":10}
            ]}"#,
        );
        assert!(ir.requires_structured_backend());
        let strs = format!("[{}]", [r#""x""#; 10].join(","));
        let ints = format!("[{}]", ["1"; 10].join(","));
        let short = format!("[{}]", ["1"; 9].join(","));
        assert!(accepts(&ir, strs.as_bytes()));
        assert!(accepts(&ir, ints.as_bytes()));
        assert!(
            !accepts(&ir, short.as_bytes()),
            "9 items is short of both branches"
        );
    }

    #[test]
    fn small_array_branches_stay_on_fast_dfa_path() {
        let ir = from_schema(
            r#"{"oneOf":[
                {"type":"array","items":{"type":"string"},"minItems":5,"maxItems":5},
                {"type":"array","items":{"type":"integer"},"minItems":5,"maxItems":5}
            ]}"#,
        );
        assert!(!ir.requires_structured_backend());
    }

    #[test]
    fn unbounded_array_branches_route_to_structured() {
        let ir = from_schema(
            r#"{"oneOf":[
                {"type":"array","items":{"type":"string"}},
                {"type":"array","items":{"type":"integer"}}
            ]}"#,
        );
        assert!(ir.requires_structured_backend());
    }

    #[test]
    fn blog_four_field_object_with_enum_routes_to_structured_and_matches() {
        let ir = from_schema(
            r#"{"type":"object","required":["name","brand","category","price"],"properties":{"name":{"type":"string"},"brand":{"type":"string"},"category":{"enum":["Headphones","Earbuds","Speakers"]},"price":{"type":"number"}}}"#,
        );
        assert!(ir.requires_structured_backend());
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(accepts(
            &ir,
            br#"{"name":"X","brand":"Y","category":"Earbuds","price":9.99}"#
        ));
        assert!(!accepts(
            &ir,
            br#"{"name":"X","brand":"Y","category":"NotACategory","price":9.99}"#
        ));
    }

    /// A small enum stays on the fast DFA path in isolation, but nested inside an open-object
    /// schema its own mid-string mask must still reject a byte no member starts with.
    #[test]
    fn blog_four_field_object_mask_rejects_a_byte_no_category_member_starts_with() {
        let ir = std::sync::Arc::new(from_schema(
            r#"{"type":"object","required":["name","brand","category","price"],"properties":{"name":{"type":"string"},"brand":{"type":"string"},"category":{"enum":["Headphones","Earbuds","Speakers"]},"price":{"type":"number"}}}"#,
        ));
        let mut m = StructuredMatcher::new(ir);
        for tok in [
            b"{".as_slice(),
            b"\"name\"",
            b":",
            b"\"X\"",
            b",",
            b"\"brand\"",
            b":",
            b"\"Y\"",
            b",",
            b"\"category\"",
            b":",
            b"\"",
        ] {
            assert!(m.advance(tok), "prefix {:?}", std::str::from_utf8(tok));
        }
        let toks: Vec<(Vec<u8>, Vec<u32>)> = vec![(b"N".to_vec(), vec![0])];
        let mask = m
            .allowed_mask_from_records(1, records(&toks).into_iter())
            .unwrap();
        assert!(
            !mask.get(TokenId(0)),
            "no category member starts with N, but the mid-string shortcut wrongly allowed it"
        );
    }

    /// A large enum elsewhere in the schema disables the mid-string shortcut schema-wide; an
    /// unrelated plain string field must still allow a digit after a lone trailing digit.
    #[test]
    fn plain_string_field_digit_mask_unaffected_by_a_sibling_large_enum() {
        let members: Vec<String> = (0..300).map(|i| format!("Category{i}")).collect();
        let vals = members
            .iter()
            .map(|m| format!("\"{m}\""))
            .collect::<Vec<_>>()
            .join(",");
        let ir = std::sync::Arc::new(from_schema(&format!(
            r#"{{"type":"object","properties":{{"name":{{"type":"string"}},"category":{{"enum":[{vals}]}}}}}}"#
        )));
        let mut m = StructuredMatcher::new(ir);
        for tok in [
            b"{".as_slice(),
            b"\"",
            b"name",
            b"\"",
            b":",
            b"\"",
            b"cat_",
            b"0",
        ] {
            assert!(m.advance(tok), "prefix {:?}", std::str::from_utf8(tok));
        }
        let toks: Vec<(Vec<u8>, Vec<u32>)> = (0..10)
            .map(|d| (d.to_string().into_bytes(), vec![d]))
            .collect();
        let mask = m
            .allowed_mask_from_records(10, records(&toks).into_iter())
            .unwrap();
        for d in 0..10u32 {
            assert!(
                mask.get(TokenId(d)),
                "digit {d} is plain \"name\" content, not a number"
            );
        }
    }

    /// A length-bounded (not just plain) `name` field beside a large enum, with an explicit
    /// open profile: the same digit-after-digit content must still mask correctly.
    #[test]
    fn bounded_string_field_digit_mask_unaffected_by_a_sibling_large_enum() {
        let members: Vec<String> = (0..200).map(|i| format!("Category{i}")).collect();
        let vals = members
            .iter()
            .map(|m| format!("\"{m}\""))
            .collect::<Vec<_>>()
            .join(",");
        let ir = std::sync::Arc::new(from_schema(&format!(
            r#"{{"type":"object","required":["name","category"],"additionalProperties":true,
                "properties":{{"name":{{"type":"string","minLength":1,"maxLength":50}},
                "category":{{"enum":[{vals}]}}}}}}"#
        )));
        let mut m = StructuredMatcher::new(ir);
        for tok in [b"{".as_slice(), b"\"name\"", b":", b"\"cat_0"] {
            assert!(m.advance(tok), "prefix {:?}", std::str::from_utf8(tok));
        }
        assert!(
            m.advance(b"4"),
            "digit after digit in a bounded string is not a bare number"
        );
    }

    #[test]
    fn closed_one_string_one_int_stays_on_fast_dfa_path() {
        let ir = from_schema(
            r#"{"type":"object","additionalProperties":false,"properties":{"name":{"type":"string"},"id":{"type":"integer"}}}"#,
        );
        assert!(!ir.requires_structured_backend());
    }

    #[test]
    fn closed_object_with_three_unbounded_strings_routes_to_structured() {
        let ir = from_schema(
            r#"{"type":"object","additionalProperties":false,"properties":{"a":{"type":"string"},"b":{"type":"string"},"c":{"type":"string"}}}"#,
        );
        assert!(ir.requires_structured_backend());
    }

    #[test]
    fn large_maxitems_array_routes_to_structured_and_counts_natively() {
        let ir = from_schema(r#"{"type":"array","items":{"type":"integer"},"maxItems":1000}"#);
        assert!(ir.requires_structured_backend());
        assert_eq!(ir.diagnostics().count(), 0);
        let ok = format!("[{}]", vec!["1"; 1000].join(","));
        let over = format!("[{}]", vec!["1"; 1001].join(","));
        assert!(accepts(&ir, ok.as_bytes()));
        assert!(!accepts(&ir, over.as_bytes()));
        assert!(accepts(&ir, b"[]"));
    }

    #[test]
    fn small_maxitems_array_stays_on_fast_dfa_path() {
        let ir = from_schema(r#"{"type":"array","items":{"type":"integer"},"maxItems":64}"#);
        assert!(!ir.requires_structured_backend());
    }

    #[test]
    fn large_maxitems_array_masks_reject_the_overflow_element() {
        let ir = std::sync::Arc::new(from_schema(
            r#"{"type":"array","items":{"type":"boolean"},"maxItems":100}"#,
        ));
        let mut m = StructuredMatcher::new(ir);
        assert!(m.advance(b"["));
        for i in 0..100 {
            if i > 0 {
                assert!(m.advance(b","));
            }
            assert!(m.advance(b"true"));
        }
        assert!(
            !m.advance(b","),
            "a comma opening the 101st element must be rejected"
        );
        assert!(m.advance(b"]"));
        assert!(m.is_accepting());
    }

    #[test]
    fn native_codepoint_min_bound_rejects_too_short() {
        let ir = from_schema(r#"{"type":"string","minLength":300}"#);
        assert!(ir.requires_structured_backend());
        assert!(!accepts(&ir, format!("\"{}\"", "x".repeat(299)).as_bytes()));
        assert!(accepts(&ir, format!("\"{}\"", "x".repeat(300)).as_bytes()));
    }

    #[test]
    fn unevaluated_properties_false_with_plain_properties() {
        let ir = from_schema(
            r#"{"type":"object","properties":{"a":{"type":"boolean"}},"unevaluatedProperties":false}"#,
        );
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(accepts(&ir, br#"{"a":true}"#));
        assert!(!accepts(&ir, br#"{"a":true,"b":1}"#));
    }

    #[test]
    fn unevaluated_properties_sees_allof_and_ref_annotations() {
        let ir = from_schema(
            r##"{"allOf":[{"$ref":"#/$defs/base"},{"properties":{"b":{"type":"integer"}}}],"unevaluatedProperties":false,"$defs":{"base":{"properties":{"a":{"type":"boolean"}}}}}"##,
        );
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(accepts(&ir, br#"{"a":true,"b":1}"#));
        assert!(!accepts(&ir, br#"{"a":true,"b":1,"c":null}"#));
    }

    #[test]
    fn unevaluated_properties_disjunctive_anyof_takes_the_matching_branch() {
        let ir = from_schema(
            r#"{"anyOf":[{"properties":{"a":{"type":"boolean"}}},{"properties":{"b":{"type":"boolean"}}}],"unevaluatedProperties":false}"#,
        );
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(accepts(&ir, br#"{"a":true}"#));
        assert!(accepts(&ir, br#"{"b":true}"#));
        assert!(!accepts(&ir, br#"{"c":true}"#));
    }

    #[test]
    fn unevaluated_properties_if_then_else_annotations() {
        let ir = from_schema(
            r#"{"if":{"properties":{"kind":{"const":"a"}},"required":["kind"]},"then":{"properties":{"x":{"type":"integer"}}},"else":{"properties":{"y":{"type":"integer"}}},"unevaluatedProperties":false}"#,
        );
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(accepts(&ir, br#"{"kind":"a","x":1}"#));
        assert!(!accepts(&ir, br#"{"kind":"a","y":1}"#));
        assert!(accepts(&ir, br#"{"y":1}"#));
        assert!(!accepts(&ir, br#"{"x":1}"#));
    }

    #[test]
    fn unevaluated_properties_schema_governs_stray_keys() {
        let ir = from_schema(
            r#"{"properties":{"a":{"type":"boolean"}},"unevaluatedProperties":{"type":"string","minLength":2}}"#,
        );
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(accepts(&ir, br#"{"a":true,"extra":"ok"}"#));
        assert!(!accepts(&ir, br#"{"a":true,"extra":"x"}"#));
        assert!(!accepts(&ir, br#"{"a":true,"extra":1}"#));
    }

    #[test]
    fn unevaluated_properties_scope_isolation_does_not_leak_into_nested_property() {
        let ir = from_schema(
            r#"{"properties":{"inner":{"properties":{"a":{"type":"boolean"}},"additionalProperties":true}},"unevaluatedProperties":false}"#,
        );
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(accepts(&ir, br#"{"inner":{"a":true,"anything":123}}"#));
        assert!(!accepts(&ir, br#"{"inner":{"a":true},"stray":1}"#));
    }

    #[test]
    fn unevaluated_properties_matches_the_real_corpus_allof_ref_shape() {
        let ir = from_schema(
            r##"{"$defs":{"extractor":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]},"plugin":{"allOf":[{"$ref":"#/$defs/extractor"},{"properties":{"config":{"type":"object","additionalProperties":true}}}],"unevaluatedProperties":false}},"$ref":"#/$defs/plugin"}"##,
        );
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(accepts(
            &ir,
            br#"{"name":"tap-csv","config":{"path":"/x"}}"#
        ));
        assert!(!accepts(&ir, br#"{"name":"tap-csv","unexpected":1}"#));
    }

    #[test]
    fn recursive_array_of_arrays_matches_any_nesting_depth() {
        let ir = from_schema(
            r##"{"type":"array","items":{"$ref":"#/$defs/node"},"$defs":{"node":{"type":"array","items":{"$ref":"#/$defs/node"}}}}"##,
        );
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(accepts(&ir, b"[]"));
        assert!(accepts(&ir, b"[[]]"));
        assert!(accepts(&ir, b"[[],[[]]]"));
        assert!(accepts(&ir, b"[[[[[]]]]]"));
        assert!(!accepts(&ir, b"[1]"));
        assert!(!accepts(&ir, b"[[true]]"));
        assert!(!accepts(&ir, b"{}"));
    }

    #[test]
    fn recursive_json_value_via_whole_document_self_ref() {
        let ir = from_schema(
            r##"{"type":"object","properties":{"children":{"type":"array","items":{"$ref":"#"}}},"required":["children"],"additionalProperties":false}"##,
        );
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(accepts(&ir, br#"{"children":[]}"#));
        assert!(accepts(&ir, br#"{"children":[{"children":[]}]}"#));
        assert!(accepts(
            &ir,
            br#"{"children":[{"children":[]},{"children":[{"children":[]}]}]}"#
        ));
        assert!(!accepts(&ir, br#"{"children":[1]}"#));
        assert!(!accepts(&ir, br#"{}"#));
        assert!(!accepts(&ir, br#"{"children":[{"other":[]}]}"#));
    }

    #[test]
    fn mutually_recursive_defs_match() {
        let ir = from_schema(
            r##"{"$ref":"#/$defs/a","$defs":{"a":{"type":"array","items":{"$ref":"#/$defs/b"}},"b":{"type":"array","items":{"$ref":"#/$defs/a"}}}}"##,
        );
        assert_eq!(ir.diagnostics().count(), 0);
        assert!(accepts(&ir, b"[]"));
        assert!(accepts(&ir, b"[[]]"));
        assert!(accepts(&ir, b"[[[]]]"));
        assert!(!accepts(&ir, b"[1]"));
    }

    #[test]
    fn recursive_matcher_advances_token_by_token() {
        let ir = std::sync::Arc::new(from_schema(
            r##"{"type":"array","items":{"$ref":"#/$defs/node"},"$defs":{"node":{"type":"array","items":{"$ref":"#/$defs/node"}}}}"##,
        ));
        let mut m = StructuredMatcher::new(ir);
        for tok in [b"[".as_slice(), b"[", b"]", b",", b"[", b"]", b"]"] {
            assert!(m.advance(tok), "token {:?}", std::str::from_utf8(tok));
        }
        assert!(m.is_accepting());
    }

    #[test]
    fn additional_properties_true_allows_arbitrary_extra_keys() {
        let mut b = Builder::new(opts());
        let a_val = b.boolean().unwrap();
        let n = b
            .open_object(
                vec![("a".into(), a_val)],
                &[true],
                Vec::new(),
                AdditionalPolicy::AllowAny,
                None,
                None,
                None,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        assert!(accepts(&ir, br#"{"a":true}"#));
        assert!(accepts(&ir, br#"{"a":true,"extra":[1,2,{"x":null}]}"#));
        assert!(
            accepts(&ir, br#"{"extra":1,"a":false}"#),
            "order independent"
        );
        assert!(
            !accepts(&ir, br#"{"extra":1}"#),
            "required known field missing"
        );
        assert!(!accepts(&ir, br#"{"a":true,"a":false}"#), "duplicate key");
    }

    #[test]
    fn additional_properties_schema_enforces_type_on_extra_keys() {
        let mut b = Builder::new(opts());
        let a_val = b.boolean().unwrap();
        let extra_schema = b.integer(None, None).unwrap();
        let n = b
            .open_object(
                vec![("a".into(), a_val)],
                &[true],
                Vec::new(),
                AdditionalPolicy::Schema(extra_schema),
                None,
                None,
                None,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        assert!(accepts(&ir, br#"{"a":true,"x":1,"y":-7}"#));
        assert!(!accepts(&ir, br#"{"a":true,"x":"not an int"}"#));
        assert!(!accepts(&ir, br#"{"a":true,"x":true}"#));
    }

    // "a" is known:boolean; additionalProperties:integer must never govern "a"'s value, even
    // though "a":1 would satisfy the additional schema on its own.
    #[test]
    fn known_property_takes_precedence_over_additional_schema() {
        let mut b = Builder::new(opts());
        let a_val = b.boolean().unwrap();
        let extra_schema = b.integer(None, None).unwrap();
        let n = b
            .open_object(
                vec![("a".into(), a_val)],
                &[true],
                Vec::new(),
                AdditionalPolicy::Schema(extra_schema),
                None,
                None,
                None,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        assert!(
            !accepts(&ir, br#"{"a":1}"#),
            "a=1 must fail: a is boolean, not additional-int"
        );
        assert!(accepts(&ir, br#"{"a":true}"#));
    }

    #[test]
    fn pattern_properties_matches_and_enforces_value_schema() {
        let mut b = Builder::new(opts());
        let pat_val = b.integer(None, None).unwrap();
        let n = b
            .open_object(
                Vec::new(),
                &[],
                vec![("^S_".into(), pat_val)],
                AdditionalPolicy::Forbid,
                None,
                None,
                None,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        assert!(accepts(&ir, br#"{"S_x":1,"S_y":-2}"#));
        assert!(
            !accepts(&ir, br#"{"S_x":"nope"}"#),
            "wrong value type under the pattern"
        );
        assert!(
            !accepts(&ir, br#"{"other":1}"#),
            "key does not match the pattern, forbidden"
        );
    }

    #[test]
    fn pattern_properties_without_anchors_matches_anywhere() {
        let mut b = Builder::new(opts());
        let pat_val = b.boolean().unwrap();
        let n = b
            .open_object(
                Vec::new(),
                &[],
                vec![("mid".into(), pat_val)],
                AdditionalPolicy::Forbid,
                None,
                None,
                None,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        assert!(
            accepts(&ir, br#"{"amidb":true}"#),
            "unanchored pattern matches anywhere"
        );
        assert!(!accepts(&ir, br#"{"nomatch":true}"#));
    }

    // Two overlapping patterns both match "ab"; the value must satisfy both schemas.
    #[test]
    fn multiple_matching_patterns_require_intersection() {
        let mut b = Builder::new(opts());
        let small = b.integer(Some(0), Some(9)).unwrap();
        let even_ish = b.integer(Some(0), Some(4)).unwrap();
        let n = b
            .open_object(
                Vec::new(),
                &[],
                vec![("^a".into(), small), ("b$".into(), even_ish)],
                AdditionalPolicy::Forbid,
                None,
                None,
                None,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        assert!(
            accepts(&ir, br#"{"ab":3}"#),
            "3 satisfies both [0,9] and [0,4]"
        );
        assert!(
            !accepts(&ir, br#"{"ab":7}"#),
            "7 satisfies [0,9] but not [0,4]"
        );
    }

    /// Two overlapping `patternProperties` schemas on one key: the mask must reject a digit
    /// outside their intersection immediately, not defer to the number's closing delimiter.
    #[test]
    fn overlapping_pattern_properties_mask_rejects_outside_the_intersection_immediately() {
        let mut b = Builder::new(opts());
        let small = b.integer(Some(0), Some(9)).unwrap();
        let even_ish = b.integer(Some(0), Some(4)).unwrap();
        let n = b
            .open_object(
                Vec::new(),
                &[],
                vec![("^a".into(), small), ("b$".into(), even_ish)],
                AdditionalPolicy::Forbid,
                None,
                None,
                None,
            )
            .unwrap();
        let ir = std::sync::Arc::new(b.finish(n).unwrap());
        let mut m = StructuredMatcher::new(ir);
        for tok in [b"{".as_slice(), b"\"ab\"", b":"] {
            assert!(m.advance(tok));
        }
        let toks: Vec<(Vec<u8>, Vec<u32>)> = (b'0'..=b'9')
            .map(|d| (vec![d], vec![u32::from(d)]))
            .collect();
        let mask = m
            .allowed_mask_from_records(u32::from(b'9') as usize + 1, records(&toks).into_iter())
            .unwrap();
        for d in 0..=4u32 {
            assert!(
                mask.get(TokenId(u32::from(b'0') + d)),
                "{d} is in both ranges"
            );
        }
        for d in 5..=9u32 {
            assert!(
                !mask.get(TokenId(u32::from(b'0') + d)),
                "{d} satisfies [0,9] but not [0,4]: must be rejected now, not at the comma/brace"
            );
        }
    }

    #[test]
    fn property_names_schema_constrains_every_key() {
        let mut b = Builder::new(opts());
        let names = b
            .string_pattern("[a-z]+", None, None, crate::ir::Charset::Utf8Any)
            .unwrap();
        let n = b
            .open_object(
                Vec::new(),
                &[],
                Vec::new(),
                AdditionalPolicy::AllowAny,
                Some(names),
                None,
                None,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        assert!(accepts(&ir, br#"{"abc":1}"#));
        assert!(
            !accepts(&ir, br#"{"ABC":1}"#),
            "uppercase key violates propertyNames"
        );
    }

    #[test]
    fn min_max_properties_enforced() {
        let mut b = Builder::new(opts());
        let n = b
            .open_object(
                Vec::new(),
                &[],
                Vec::new(),
                AdditionalPolicy::AllowAny,
                None,
                Some(1),
                Some(2),
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        assert!(!accepts(&ir, br#"{}"#), "below minProperties");
        assert!(accepts(&ir, br#"{"a":1}"#));
        assert!(accepts(&ir, br#"{"a":1,"b":2}"#));
        assert!(
            !accepts(&ir, br#"{"a":1,"b":2,"c":3}"#),
            "above maxProperties"
        );
    }

    #[test]
    fn permutations_of_a_five_field_open_object_all_accept() {
        let names = ["a", "b", "c", "d", "e"];
        let mut b = Builder::new(opts());
        let known: Vec<(String, NodeId)> = names
            .iter()
            .map(|n| (n.to_string(), b.boolean().unwrap()))
            .collect();
        let n = b
            .open_object(
                known,
                &[true; 5],
                Vec::new(),
                AdditionalPolicy::Forbid,
                None,
                None,
                None,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();

        fn permutations(n: usize) -> Vec<Vec<usize>> {
            if n == 0 {
                return vec![Vec::new()];
            }
            let mut out = Vec::new();
            for p in permutations(n - 1) {
                for pos in 0..=p.len() {
                    let mut np = p.clone();
                    np.insert(pos, n - 1);
                    out.push(np);
                }
            }
            out
        }
        for perm in permutations(5) {
            let body = perm
                .iter()
                .map(|&i| format!(r#""{}":true"#, names[i]))
                .collect::<Vec<_>>()
                .join(",");
            let doc = format!("{{{body}}}");
            assert!(
                accepts(&ir, doc.as_bytes()),
                "permutation {perm:?} must accept: {doc}"
            );
        }
        assert!(
            !accepts(&ir, br#"{"a":true,"b":true,"c":true,"d":true}"#),
            "missing e"
        );
    }

    #[test]
    fn nested_open_object_inside_a_known_property() {
        let mut b = Builder::new(opts());
        let inner_val = b.integer(None, None).unwrap();
        let inner = b
            .open_object(
                Vec::new(),
                &[],
                Vec::new(),
                AdditionalPolicy::Schema(inner_val),
                None,
                None,
                None,
            )
            .unwrap();
        let outer = b
            .open_object(
                vec![("inner".into(), inner)],
                &[true],
                Vec::new(),
                AdditionalPolicy::Forbid,
                None,
                None,
                None,
            )
            .unwrap();
        let ir = b.finish(outer).unwrap();
        assert!(accepts(&ir, br#"{"inner":{"x":1,"y":2}}"#));
        assert!(!accepts(&ir, br#"{"inner":{"x":"bad"}}"#));
        assert!(
            !accepts(&ir, br#"{"inner":{},"extra":1}"#),
            "outer forbids additional"
        );
    }

    #[test]
    fn open_object_inside_array_items() {
        let mut b = Builder::new(opts());
        let val = b.boolean().unwrap();
        let obj = b
            .open_object(
                vec![("ok".into(), val)],
                &[true],
                Vec::new(),
                AdditionalPolicy::AllowAny,
                None,
                None,
                None,
            )
            .unwrap();
        let arr = b.array(obj, 0, None).unwrap();
        let ir = b.finish(arr).unwrap();
        assert!(accepts(&ir, br#"[{"ok":true},{"ok":false,"x":1}]"#));
        assert!(!accepts(&ir, br#"[{"ok":true},{"missing":1}]"#));
    }

    #[test]
    fn whole_document_whitespace_variants_are_accepted() {
        let mut b = Builder::new(opts());
        let val = b.boolean().unwrap();
        let n = b
            .open_object(
                vec![("a".into(), val)],
                &[true],
                Vec::new(),
                AdditionalPolicy::AllowAny,
                None,
                None,
                None,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        assert!(accepts(&ir, br#"{ "a" : true , "x" : [ 1 , 2 ] }"#));
        assert!(accepts(&ir, b" { \"a\" : true } "));
    }

    #[test]
    fn pathologically_deep_nesting_rejects_without_overflow() {
        let mut b = Builder::new(opts());
        let val = b.number().unwrap();
        let n = b
            .open_object(
                Vec::new(),
                &[],
                Vec::new(),
                AdditionalPolicy::Schema(val),
                None,
                None,
                None,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        let deep_array = "[".repeat(10_000) + &"]".repeat(10_000);
        let doc = format!(r#"{{"x":{deep_array}}}"#);
        assert!(
            !accepts(&ir, doc.as_bytes()),
            "must reject, not overflow the stack"
        );
    }

    #[test]
    fn closed_object_becomes_irregular_and_structural_when_a_nested_value_is_open() {
        let mut b = Builder::new(opts());
        let open_val = b.number().unwrap();
        let inner_open = b
            .open_object(
                Vec::new(),
                &[],
                Vec::new(),
                AdditionalPolicy::Schema(open_val),
                None,
                None,
                None,
            )
            .unwrap();
        let outer = b
            .object(
                vec![("meta".into(), inner_open)],
                &[true],
                ObjectClosure::Forbidden,
            )
            .unwrap();
        let ir = b.finish(outer).unwrap();
        assert!(accepts(&ir, br#"{"meta":{"a":1,"b":2}}"#));
        assert!(!accepts(&ir, br#"{"meta":{"a":"bad"}}"#));
    }

    #[test]
    fn a_fully_regular_schema_still_matches_via_the_fast_path() {
        let mut b = Builder::new(opts());
        let a = b.boolean().unwrap();
        let n = b
            .object(vec![("a".into(), a)], &[true], ObjectClosure::Forbidden)
            .unwrap();
        let ir = b.finish(n).unwrap();
        assert!(accepts(&ir, br#"{"a":true}"#));
        assert!(accepts(&ir, br#"{ "a" : false }"#));
        assert!(!accepts(&ir, br#"{"a":true,"b":1}"#));
    }

    fn open_bool_field(name: &str) -> crate::ir::SchemaIR {
        let mut b = Builder::new(opts());
        let val = b.boolean().unwrap();
        let n = b
            .open_object(
                vec![(name.to_string(), val)],
                &[true],
                Vec::new(),
                AdditionalPolicy::AllowAny,
                None,
                None,
                None,
            )
            .unwrap();
        b.finish(n).unwrap()
    }

    #[test]
    fn can_continue_true_for_every_true_prefix_of_a_valid_document() {
        let ir = open_bool_field("a");
        let full = br#"{"a":true,"extra":123}"#;
        for end in 1..full.len() {
            assert!(
                can_continue(&ir, &full[..end]),
                "prefix {:?} of a valid document must not be dead",
                std::str::from_utf8(&full[..end]).unwrap()
            );
        }
        assert!(accepts(&ir, full));
    }

    #[test]
    fn can_continue_false_for_a_definitely_wrong_structural_byte() {
        let ir = open_bool_field("a");
        assert!(!can_continue(&ir, b"x"), "does not even start a JSON value");
        assert!(
            !can_continue(&ir, br#"{"a":tru9"#),
            "'true' cannot become 'tru9'"
        );
        assert!(
            !can_continue(&ir, br#"{"b":true}"#),
            "required 'a' can never appear once object is done"
        );
    }

    #[test]
    fn can_continue_true_mid_string_including_partial_escapes() {
        let ir = open_bool_field("a");
        assert!(can_continue(&ir, br#"{"a"#));
        assert!(
            can_continue(&ir, b"{\"a\\"),
            "lone backslash: escape kind not seen yet"
        );
        assert!(can_continue(&ir, b"{\"a\\u12"), "partial \\u escape");
        assert!(
            !can_continue(&ir, b"{\"a\\z"),
            "'z' is not a legal escape character"
        );
        assert!(!can_continue(&ir, b"{\"a\\u12g4"), "'g' is not a hex digit");
    }

    #[test]
    fn can_continue_ambiguous_integer_tail_is_not_dead() {
        let mut b = Builder::new(opts());
        let val = b.integer(None, None).unwrap();
        let n = b
            .open_object(
                Vec::new(),
                &[],
                Vec::new(),
                AdditionalPolicy::Schema(val),
                None,
                None,
                None,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        assert!(
            can_continue(&ir, br#"{"x":7"#),
            "7 could become 70, 71, etc."
        );
        assert!(accepts(&ir, br#"{"x":7}"#));
        assert!(
            !can_continue(&ir, br#"{"x":7."#),
            "an integer schema never accepts a decimal point, no matter what follows"
        );
    }

    #[test]
    fn can_continue_ambiguous_number_dot_tail_is_not_dead() {
        let mut b = Builder::new(opts());
        let val = b.number().unwrap();
        let n = b
            .open_object(
                Vec::new(),
                &[],
                Vec::new(),
                AdditionalPolicy::Schema(val),
                None,
                None,
                None,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        assert!(
            can_continue(&ir, br#"{"x":7."#),
            "a trailing dot with no digit yet could still become 7.5"
        );
        assert!(
            !can_continue(&ir, br#"{"x":7.e"#),
            "a dot directly followed by 'e' is never valid, no matter what comes after"
        );
        assert!(accepts(&ir, br#"{"x":7.5}"#));
    }

    #[test]
    fn can_continue_false_once_trailing_garbage_follows_a_complete_document() {
        let ir = open_bool_field("a");
        assert!(accepts(&ir, br#"{"a":true}"#));
        assert!(
            !can_continue(&ir, br#"{"a":true}x"#),
            "extra content after a complete document can never be fixed by more bytes"
        );
        assert!(
            can_continue(&ir, br#"{"a":true} "#),
            "trailing whitespace alone is still fine"
        );
    }

    #[test]
    fn can_continue_required_field_enforced_only_at_the_closing_brace() {
        let mut b = Builder::new(opts());
        let a_val = b.boolean().unwrap();
        let b_val = b.boolean().unwrap();
        let n = b
            .open_object(
                vec![("a".into(), a_val), ("b".into(), b_val)],
                &[true, true],
                Vec::new(),
                AdditionalPolicy::Forbid,
                None,
                None,
                None,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        assert!(
            can_continue(&ir, br#"{"a":true"#),
            "b hasn't been ruled out yet"
        );
        assert!(
            !can_continue(&ir, br#"{"a":true}"#),
            "closed with b missing: dead"
        );
        assert!(accepts(&ir, br#"{"a":true,"b":false}"#));
    }

    #[test]
    fn can_continue_duplicate_key_is_dead_as_soon_as_the_second_key_closes() {
        let ir = open_bool_field("a");
        assert!(can_continue(&ir, br#"{"a":true,"a"#));
        assert!(
            !can_continue(&ir, br#"{"a":true,"a""#),
            "the second \"a\" key is a duplicate the instant it closes"
        );
    }

    #[test]
    fn can_continue_nested_open_object_prefix() {
        let mut b = Builder::new(opts());
        let inner_val = b.integer(None, None).unwrap();
        let inner = b
            .open_object(
                Vec::new(),
                &[],
                Vec::new(),
                AdditionalPolicy::Schema(inner_val),
                None,
                None,
                None,
            )
            .unwrap();
        let outer = b
            .open_object(
                vec![("inner".into(), inner)],
                &[true],
                Vec::new(),
                AdditionalPolicy::Forbid,
                None,
                None,
                None,
            )
            .unwrap();
        let ir = b.finish(outer).unwrap();
        assert!(can_continue(&ir, br#"{"inner":{"x":1,"y":"#));
        assert!(
            !can_continue(&ir, br#"{"inner":{"x":"bad"#),
            "a string can never satisfy the inner integer schema, caught as soon as it starts"
        );
    }

    #[test]
    fn mask_generation_style_filters_candidate_continuations() {
        let ir = open_bool_field("a");
        let prefix = br#"{"a":true,"#.to_vec();
        let candidates: &[&[u8]] = &[br#""b":1}"#, br#"}"#, b"9", br#""a":1}"#];
        let results: Vec<bool> = candidates
            .iter()
            .map(|c| {
                let mut buf = prefix.clone();
                buf.extend_from_slice(c);
                can_continue(&ir, &buf) || accepts(&ir, &buf)
            })
            .collect();
        assert!(
            results[0],
            "\"b\":1}} is a legal continuation (AllowAny extra key)"
        );
        assert!(
            !results[1],
            "a bare }} right after a trailing comma is invalid JSON"
        );
        assert!(
            !results[2],
            "a bare digit is not a legal object continuation here"
        );
        assert!(!results[3], "\"a\" is already present: duplicate key");
    }

    fn tokens() -> Vec<(Vec<u8>, Vec<u32>)> {
        vec![
            (b"{".to_vec(), vec![0]),
            (b"}".to_vec(), vec![1]),
            (b"\"".to_vec(), vec![2]),
            (b"a".to_vec(), vec![3]),
            (b":".to_vec(), vec![4]),
            (b"true".to_vec(), vec![5]),
            (b"false".to_vec(), vec![6]),
            (b",".to_vec(), vec![7]),
            (b"x".to_vec(), vec![8]),
            (b"1".to_vec(), vec![9]),
        ]
    }

    fn records(toks: &[(Vec<u8>, Vec<u32>)]) -> Vec<(&[u8], &[u32])> {
        toks.iter()
            .map(|(b, ids)| (b.as_slice(), ids.as_slice()))
            .collect()
    }

    /// Asserts `mask[id] == (accepts(prefix+bytes) || can_continue(prefix+bytes))` for every candidate.
    fn assert_mask_matches_advance(ir: SchemaIR, prefix: &[u8], toks: &[(Vec<u8>, Vec<u32>)]) {
        let vocab_size = toks
            .iter()
            .flat_map(|(_, ids)| ids.iter())
            .max()
            .map_or(0, |&m| m + 1) as usize;
        let ir = Arc::new(ir);
        let mut m = StructuredMatcher::new(ir.clone());
        assert!(
            m.advance(prefix),
            "test setup: prefix itself must be reachable"
        );
        let mask = m
            .allowed_mask_from_records(vocab_size, records(toks).into_iter())
            .unwrap();
        for (bytes, ids) in toks {
            let mut full = prefix.to_vec();
            full.extend_from_slice(bytes);
            let expected = accepts(&ir, &full) || can_continue(&ir, &full);
            for &id in ids {
                assert_eq!(
                    mask.get(TokenId(id)),
                    expected,
                    "prefix={:?} bytes={:?} expected={expected}",
                    String::from_utf8_lossy(prefix),
                    String::from_utf8_lossy(bytes),
                );
            }
        }
    }

    #[test]
    fn mask_rejects_a_key_candidate_that_cannot_complete_the_only_valid_property_name() {
        let ir = from_schema(
            r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}"#,
        );
        let toks = vec![(b"x".to_vec(), vec![0]), (b"m".to_vec(), vec![1])];
        assert_mask_matches_advance(ir, br#"{"na"#, &toks);
    }

    #[test]
    fn mask_rejects_a_digit_that_would_exceed_a_bounded_maximum() {
        let ir = from_schema(r#"{"type":"integer","maximum":100}"#);
        let toks = vec![(b"1".to_vec(), vec![0]), (b"0".to_vec(), vec![1])];
        assert_mask_matches_advance(ir, b"10", &toks);
    }

    #[test]
    fn mask_respects_minimum_and_exclusive_bounds() {
        let ir = from_schema(r#"{"type":"integer","minimum":10,"exclusiveMaximum":15}"#);
        let toks = vec![
            (b"0".to_vec(), vec![0]),
            (b"4".to_vec(), vec![1]),
            (b"5".to_vec(), vec![2]),
            (b"9".to_vec(), vec![3]),
        ];
        assert_mask_matches_advance(ir, b"1", &toks);
    }

    #[test]
    fn mask_respects_multiple_of() {
        let ir = from_schema(r#"{"type":"integer","multipleOf":5}"#);
        let toks = vec![
            (b"0".to_vec(), vec![0]),
            (b"5".to_vec(), vec![1]),
            (b"3".to_vec(), vec![2]),
        ];
        assert_mask_matches_advance(ir, b"1", &toks);
    }

    #[test]
    fn mask_rejects_invalid_utf8_continuation_inside_a_string() {
        let ir = from_schema(r#"{"type":"string"}"#);
        let toks = vec![
            (vec![b'a'], vec![0]),
            (vec![0xff], vec![1]),
            (vec![0xc2, 0x28], vec![2]),
        ];
        assert_mask_matches_advance(ir, b"\"", &toks);
    }

    #[test]
    fn mask_matches_advance_for_escaped_property_names() {
        let ir = from_schema(
            r#"{"type":"object","properties":{"a\"b":{"type":"boolean"}},"required":["a\"b"],"additionalProperties":false}"#,
        );
        let toks = vec![(br#"\""#.to_vec(), vec![0]), (b"x".to_vec(), vec![1])];
        assert_mask_matches_advance(ir, br#"{"a"#, &toks);
    }

    #[test]
    fn mask_matches_advance_for_pattern_properties_and_property_names() {
        let toks = vec![
            (b"x".to_vec(), vec![0]),
            (b"y".to_vec(), vec![1]),
            (b"1".to_vec(), vec![2]),
        ];
        let schema = r#"{"type":"object","patternProperties":{"^x[0-9]$":{"type":"boolean"}},"propertyNames":{"pattern":"^x[0-9]$"},"additionalProperties":false}"#;
        assert_mask_matches_advance(from_schema(schema), br#"{"#, &toks);
        assert_mask_matches_advance(from_schema(schema), br#"{"x"#, &toks);
    }

    #[test]
    fn structured_matcher_generates_a_complete_valid_document_token_by_token() {
        let ir = Arc::new(open_bool_field("a"));
        let mut m = StructuredMatcher::new(ir);
        assert!(!m.is_accepting());
        for tok in [b"{".as_slice(), b"\"", b"a", b"\"", b":", b"true", b"}"] {
            let mask = m
                .allowed_mask_from_records(16, records(&tokens()).into_iter())
                .unwrap();
            let bytes_ok = mask.count_ones() > 0;
            assert!(
                bytes_ok,
                "some token must be allowed before committing {tok:?}"
            );
            assert!(m.advance(tok), "token {tok:?} must be accepted");
        }
        assert!(m.is_accepting());
        assert!(m.eos_legal());
    }

    #[test]
    fn structured_matcher_mask_excludes_tokens_that_would_kill_the_parse() {
        let ir = Arc::new(open_bool_field("a"));
        let mut m = StructuredMatcher::new(ir);
        for tok in [b"{".as_slice(), b"\"", b"a", b"\"", b":"] {
            assert!(m.advance(tok));
        }
        let toks = tokens();
        let mask = m
            .allowed_mask_from_records(16, records(&toks).into_iter())
            .unwrap();
        assert!(mask.get(TokenId(5)), "true is allowed for a boolean field");
        assert!(mask.get(TokenId(6)), "false is allowed for a boolean field");
        assert!(
            !mask.get(TokenId(1)),
            "'}}' would leave \"a\" without a value"
        );
        assert!(
            !mask.get(TokenId(9)),
            "a digit can never become true/false: rejected immediately, not deferred to close"
        );
        assert!(!m.advance(b"1"), "commit must match the mask verdict");
    }

    #[test]
    fn structured_matcher_rejects_a_token_that_would_duplicate_a_key() {
        let ir = Arc::new(open_bool_field("a"));
        let mut m = StructuredMatcher::new(ir);
        for tok in [
            b"{".as_slice(),
            b"\"",
            b"a",
            b"\"",
            b":",
            b"true",
            b",",
            b"\"",
            b"a",
        ] {
            assert!(m.advance(tok), "token {tok:?} must be accepted");
        }
        assert!(
            !m.advance(b"\""),
            "closing the second \"a\" key duplicates the first"
        );
    }

    #[test]
    fn structured_matcher_advance_never_moves_the_position_on_rejection() {
        let ir = Arc::new(open_bool_field("a"));
        let mut m = StructuredMatcher::new(ir);
        assert!(m.advance(b"{"));
        assert!(!m.advance(b"}"), "empty object is missing required 'a'");
        assert!(
            m.advance(b"\""),
            "rejection must not have moved the position"
        );
    }

    #[test]
    fn mid_number_fast_path_never_accepts_a_leading_zero_and_speeds_multi_digit_runs() {
        let mut b = Builder::new(opts());
        let val = b.number().unwrap();
        let n = b
            .open_object(
                vec![("n".to_string(), val)],
                &[true],
                Vec::new(),
                AdditionalPolicy::AllowAny,
                None,
                None,
                None,
            )
            .unwrap();
        let ir = Arc::new(b.finish(n).unwrap());
        let toks: Vec<(Vec<u8>, Vec<u32>)> = vec![
            (b"{\"n\":".to_vec(), vec![0]),
            (b"1".to_vec(), vec![1]),
            (b"0".to_vec(), vec![2]),
            (b"05".to_vec(), vec![3]),
            (b"23".to_vec(), vec![4]),
        ];
        let mut m = StructuredMatcher::new(ir.clone());
        assert!(m.advance(b"{\"n\":"));
        assert!(m.advance(b"1"), "1 starts a normal multi-digit integer");
        let mask = m
            .allowed_mask_from_records(8, records(&toks).into_iter())
            .unwrap();
        assert!(
            mask.get(TokenId(4)),
            "\"23\" legally extends \"1\" to \"123\""
        );
        assert!(m.advance(b"23"), "\"123\" is a normal multi-digit integer");
        assert!(m.advance(b"}"), "closing now yields a valid document");

        let mut m2 = StructuredMatcher::new(ir);
        assert!(m2.advance(b"{\"n\":"));
        assert!(
            m2.advance(b"0"),
            "a lone 0 is itself a complete integer part"
        );
        let mask2 = m2
            .allowed_mask_from_records(8, records(&toks).into_iter())
            .unwrap();
        assert!(
            !mask2.get(TokenId(1)),
            "\"1\" directly after \"0\" would spell the invalid literal \"01\""
        );
        assert!(
            !mask2.get(TokenId(3)),
            "\"05\" is internally a leading-zero violation regardless of what precedes it"
        );
        assert!(!m2.advance(b"1"), "commit must match the mask verdict");
    }

    fn open_string_field_max_len(name: &str, max_len: u32) -> crate::ir::SchemaIR {
        let mut b = Builder::new(opts());
        // One ASCII byte repeated up to `max_len` times: an IR-level ASCII length bound.
        let val = b
            .string_pattern(
                r"[\x20-\x21\x23-\x5b\x5d-\x7e]",
                None,
                Some(max_len),
                Charset::AsciiPrintableNoQuoteBackslash,
            )
            .unwrap();
        let n = b
            .open_object(
                vec![(name.to_string(), val)],
                &[true],
                Vec::new(),
                AdditionalPolicy::AllowAny,
                None,
                None,
                None,
            )
            .unwrap();
        b.finish(n).unwrap()
    }

    /// A regex/length-restricted field disables the mid-string fast path schema-wide, so the real
    /// max-length check applies to every candidate, closed or not, not just ones that close.
    #[test]
    fn mid_string_slow_path_enforces_max_length_even_before_the_string_closes() {
        let ir = Arc::new(open_string_field_max_len("s", 3));
        let toks: Vec<(Vec<u8>, Vec<u32>)> = vec![
            (b"{\"s\":\"ab".to_vec(), vec![0]),
            (b"c\"".to_vec(), vec![1]),  // closes at length 3: legal
            (b"cd\"".to_vec(), vec![2]), // closes at length 4: exceeds max_len 3
            (b"cd".to_vec(), vec![3]),   // 4th content char: already past max_len, dead on arrival
            (b"}".to_vec(), vec![4]),
        ];
        let mut m = StructuredMatcher::new(ir);
        assert!(m.advance(b"{\"s\":\"ab"));
        let mask = m
            .allowed_mask_from_records(8, records(&toks).into_iter())
            .unwrap();
        assert!(mask.get(TokenId(1)), "closing at max_len 3 is legal");
        assert!(
            !mask.get(TokenId(2)),
            "closing past max_len 3 must still be rejected"
        );
        assert!(
            !mask.get(TokenId(3)),
            "a 4th content char is rejected immediately, not deferred to the close quote"
        );
        assert!(!m.advance(b"cd"), "commit must match the mask verdict");
    }

    #[test]
    fn mid_string_fast_path_excludes_a_raw_control_byte() {
        let ir = Arc::new(open_string_field_max_len("s", 100));
        let toks: Vec<(Vec<u8>, Vec<u32>)> = vec![
            (b"{\"s\":\"ab".to_vec(), vec![0]),
            (b"c\x01d".to_vec(), vec![1]),
            (b"cd".to_vec(), vec![2]),
        ];
        let mut m = StructuredMatcher::new(ir);
        assert!(m.advance(b"{\"s\":\"ab"));
        let mask = m
            .allowed_mask_from_records(8, records(&toks).into_iter())
            .unwrap();
        assert!(
            !mask.get(TokenId(1)),
            "a raw control byte is never legal inside a string"
        );
        assert!(mask.get(TokenId(2)), "plain content stays allowed");
        assert!(!m.advance(b"c\x01d"));
    }

    // Regular-engine capacity limits require structured interpretation, not rejection.
    #[test]
    fn a_closed_object_too_large_for_one_dfa_still_hand_interprets_correctly() {
        let schema = r##"{
            "type": "object",
            "properties": {
                "s": {
                    "type": "object",
                    "patternProperties": {
                        "^[0-9]+$": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": false,
                                "properties": {
                                    "a": {"type": "integer"},
                                    "b": {"type": "integer"},
                                    "c": {"type": "number"},
                                    "d": {"type": "integer"}
                                },
                                "required": ["a", "b", "c", "d"]
                            }
                        }
                    }
                }
            },
            "additionalProperties": false,
            "required": ["s"]
        }"##;
        let ir = schema_to_ir(schema, CompileOptions::default()).unwrap();
        assert_eq!(ir.diagnostics().len(), 0);
        let doc = br#"{"s":{"1":[{"a":1,"b":2,"c":-25,"d":4},{"a":5,"b":6,"c":-25,"d":8}]}}"#;
        assert!(
            accepts(&ir, doc),
            "a genuinely valid document must be accepted"
        );
        assert!(can_continue(&ir, &doc[..doc.len() - 1]));
        assert!(
            !accepts(&ir, br#"{"s":{"1":[{"a":1,"b":2,"c":"oops","d":4}]}}"#),
            "a real type mismatch must still be rejected"
        );
    }

    // Closed objects can fall back to structured interpretation at the DFA capacity limit.
    #[test]
    fn a_root_object_too_wide_for_one_shuffle_is_still_matched_field_by_field() {
        let props: Vec<String> = (0..6)
            .map(|i| format!(r#""f{i}":{{"type":"integer"}}"#))
            .collect();
        let required: Vec<String> = (0..6).map(|i| format!(r#""f{i}""#)).collect();
        let schema = format!(
            r#"{{"type":"object","properties":{{{}}},"additionalProperties":false,"required":[{}]}}"#,
            props.join(","),
            required.join(",")
        );
        let ir = schema_to_ir(&schema, CompileOptions::default()).unwrap();
        assert_eq!(ir.diagnostics().len(), 0);
        assert!(
            ir.requires_structured_backend(),
            "no OpenObject here, but 6 fields is past the safe shuffle pre-check"
        );
        let doc = br#"{"f0":0,"f1":1,"f2":2,"f3":3,"f4":4,"f5":5}"#;
        assert!(accepts(&ir, doc));
        assert!(can_continue(&ir, &doc[..doc.len() - 1]));
        assert!(!accepts(&ir, br#"{"f0":0,"f1":1,"f2":2,"f3":3,"f4":4}"#));
        assert!(!accepts(
            &ir,
            br#"{"f0":0,"f1":1,"f2":2,"f3":3,"f4":4,"f5":5,"f6":6}"#
        ));
    }

    // Every case here is transcribed from the official JSON Schema Test Suite's uniqueItems.json,
    // the authoritative source for this equality relation, not this dialect's own convention.
    #[test]
    fn canonical_value_matches_the_official_test_suites_number_equality() {
        for (a, b) in [
            (&b"1"[..], &b"1"[..]),
            (b"1.0", b"1"),
            (b"1.00", b"1.0"),
            (b"10e-1", b"1"),
            (b"0", b"-0"),
            (b"1e2", b"100"),
        ] {
            assert_eq!(
                canonical_value_for_uniqueness(a),
                canonical_value_for_uniqueness(b),
                "{a:?} vs {b:?}"
            );
        }
        // 0/false and 1/true are never equal despite looking numerically similar.
        assert_ne!(
            canonical_value_for_uniqueness(b"0"),
            canonical_value_for_uniqueness(b"false")
        );
        assert_ne!(
            canonical_value_for_uniqueness(b"1"),
            canonical_value_for_uniqueness(b"true")
        );
    }

    #[test]
    fn canonical_value_ignores_object_key_order_but_not_array_order() {
        assert_eq!(
            canonical_value_for_uniqueness(br#"{"a":1,"b":2}"#),
            canonical_value_for_uniqueness(br#"{"b":2,"a":1}"#),
        );
        assert_ne!(
            canonical_value_for_uniqueness(br#"{"a":1,"b":2}"#),
            canonical_value_for_uniqueness(br#"{"a":2,"b":1}"#),
        );
        assert_ne!(
            canonical_value_for_uniqueness(b"[1,2]"),
            canonical_value_for_uniqueness(b"[2,1]"),
            "array element order is significant, unlike object keys"
        );
    }

    /// Schema-agnostic top-level distinctness check, exactly mirroring what `match_array` does per
    /// item - used to bulk-verify against the official suite without building a schema per case.
    fn array_items_are_unique(json_array: &[u8]) -> bool {
        let mut cur = Cursor::new(json_array);
        assert_eq!(cur.peek(), Some(b'['));
        cur.advance();
        cur.skip_ws();
        let mut seen: rustc_hash::FxHashSet<Vec<u8>> = rustc_hash::FxHashSet::default();
        if cur.peek() != Some(b']') {
            loop {
                let canon = canonicalize_value(&mut cur, 0).expect("valid fixture item");
                if !seen.insert(canon) {
                    return false;
                }
                cur.skip_ws();
                match cur.peek() {
                    Some(b',') => {
                        cur.advance();
                        cur.skip_ws();
                    }
                    Some(b']') => break,
                    other => panic!("malformed fixture array: {other:?}"),
                }
            }
        }
        true
    }

    // Every array/expected-verdict pair here is the `{"uniqueItems": true}` test group transcribed
    // verbatim from the official JSON Schema Test Suite's uniqueItems.json.
    #[test]
    fn matches_the_official_test_suite_uniqueitems_group_exactly() {
        let cases: &[(bool, &[u8])] = &[
            (true, b"[1,2]"),
            (false, b"[1,1]"),
            (false, b"[1,2,1]"),
            (false, br#"[1.0,1.0,1]"#),
            (true, b"[0,false]"),
            (true, b"[1,true]"),
            (true, br#"["foo","bar","baz"]"#),
            (false, br#"["foo","bar","foo"]"#),
            (true, br#"[{"foo":"bar"},{"foo":"baz"}]"#),
            (false, br#"[{"foo":"bar"},{"foo":"bar"}]"#),
            (
                false,
                br#"[{"foo":"bar","bar":"foo"},{"bar":"foo","foo":"bar"}]"#,
            ),
            (
                true,
                br#"[{"foo":{"bar":{"baz":true}}},{"foo":{"bar":{"baz":false}}}]"#,
            ),
            (
                false,
                br#"[{"foo":{"bar":{"baz":true}}},{"foo":{"bar":{"baz":true}}}]"#,
            ),
            (true, br#"[["foo"],["bar"]]"#),
            (false, br#"[["foo"],["foo"]]"#),
            (false, br#"[["foo"],["bar"],["foo"]]"#),
            (true, b"[[1],[true]]"),
            (true, b"[[0],[false]]"),
            (true, br#"[[[1],"foo"],[[true],"foo"]]"#),
            (true, br#"[[[0],"foo"],[[false],"foo"]]"#),
            (true, br#"[{},[1],true,null,1,"{}"]"#),
            (false, br#"[{},[1],true,null,{},1]"#),
            (true, br#"[{"a":1,"b":2},{"a":2,"b":1}]"#),
            (false, br#"[{"a":1,"b":2},{"b":2,"a":1}]"#),
            (true, br#"[{"a":false},{"a":0}]"#),
            (true, br#"[{"a":true},{"a":1}]"#),
        ];
        for (expected, data) in cases {
            assert_eq!(
                array_items_are_unique(data),
                *expected,
                "{}",
                std::str::from_utf8(data).unwrap()
            );
        }
    }

    fn unique_int_array() -> crate::ir::SchemaIR {
        let mut b = Builder::new(opts());
        let item = b.integer(None, None).unwrap();
        let n = b.array_unique(item, 0, None, true).unwrap();
        b.finish(n).unwrap()
    }

    #[test]
    fn unique_items_end_to_end_rejects_numerically_equal_duplicates() {
        let ir = unique_int_array();
        assert!(accepts(&ir, b"[1,2,3]"));
        assert!(accepts(&ir, b"[]"));
        assert!(!accepts(&ir, b"[1,1]"));
        assert!(!accepts(&ir, b"[1,2,1]"));
        assert!(can_continue(&ir, b"[1,2,"));
        assert!(
            can_continue(&ir, b"[1,1"),
            "the second 1's span hasn't closed yet - it could still become 10, 12, ..."
        );
        assert!(
            !can_continue(&ir, b"[1,1,"),
            "closing the second 1 now reveals the duplicate"
        );
    }

    fn unique_object_array() -> crate::ir::SchemaIR {
        let mut b = Builder::new(opts());
        let a = b.integer(None, None).unwrap();
        let bb = b.integer(None, None).unwrap();
        let obj = b
            .object(
                vec![("a".into(), a), ("b".into(), bb)],
                &[true, true],
                ObjectClosure::Forbidden,
            )
            .unwrap();
        let n = b.array_unique(obj, 0, None, true).unwrap();
        b.finish(n).unwrap()
    }

    #[test]
    fn unique_items_end_to_end_treats_reordered_object_keys_as_duplicates() {
        let ir = unique_object_array();
        assert!(accepts(&ir, br#"[{"a":1,"b":2},{"a":2,"b":1}]"#));
        assert!(
            !accepts(&ir, br#"[{"a":1,"b":2},{"b":2,"a":1}]"#),
            "same value under a different key order: still a duplicate"
        );
    }

    // `uniqueItems` applies to tuple arrays as well as homogeneous arrays.
    fn unique_bool_pair_tuple() -> crate::ir::SchemaIR {
        let mut b = Builder::new(opts());
        let a = b.boolean().unwrap();
        let bb = b.boolean().unwrap();
        let n = b.tuple_unique(vec![a, bb], None, 0, None, true).unwrap();
        b.finish(n).unwrap()
    }

    #[test]
    fn unique_items_applies_to_prefix_items_tuples_too() {
        let ir = unique_bool_pair_tuple();
        assert!(accepts(&ir, b"[false,true]"));
        assert!(accepts(&ir, b"[true,false]"));
        assert!(
            !accepts(&ir, b"[false,false]"),
            "both positions are false: not unique"
        );
        assert!(
            !accepts(&ir, b"[true,true]"),
            "both positions are true: not unique"
        );
    }

    fn not_of(inner: NodeId, mut b: Builder) -> crate::ir::SchemaIR {
        let n = b.not(inner).unwrap();
        b.finish(n).unwrap()
    }

    #[test]
    fn not_defers_judgment_until_the_value_span_closes() {
        let mut b = Builder::new(opts());
        let int_ty = b.integer(None, None).unwrap();
        let ir = not_of(int_ty, b);
        assert!(
            can_continue(&ir, b"5"),
            "5 could still become 5.5, which is not an integer"
        );
        assert!(
            !accepts(&ir, b"5"),
            "5 alone is an integer: not must reject"
        );
        assert!(
            accepts(&ir, b"5.5"),
            "5.5 is not an integer: not must accept"
        );
    }

    #[test]
    fn not_of_a_nested_object_defers_until_the_object_closes() {
        let mut b = Builder::new(opts());
        let string_ty = b
            .string_pattern("[a-z]+", None, None, Charset::Utf8Any)
            .unwrap();
        let inner = b
            .object(
                vec![("foo".into(), string_ty)],
                &[true],
                ObjectClosure::Forbidden,
            )
            .unwrap();
        let ir = not_of(inner, b);
        assert!(
            can_continue(&ir, br#"{"foo":"#),
            "the object is still open, so not's verdict cannot be decided yet"
        );
        assert!(!accepts(&ir, br#"{"foo":"bar"}"#));
        assert!(accepts(&ir, br#"{"foo":1}"#));
    }

    fn array_with_contains(
        items: ItemsPolicy,
        contains: ContainsConstraint,
    ) -> crate::ir::SchemaIR {
        let mut b = Builder::new(opts());
        let n = b
            .array_full(items, 0, None, false, Some(contains), true)
            .unwrap();
        b.finish(n).unwrap()
    }

    #[test]
    fn contains_always_requires_at_least_one_element_by_default() {
        let ir = array_with_contains(
            ItemsPolicy::AllowAny,
            ContainsConstraint {
                policy: ContainsPolicy::Always,
                min: 1,
                max: None,
            },
        );
        assert!(!accepts(&ir, b"[]"), "empty array has zero matches");
        assert!(accepts(&ir, b"[1]"));
        assert!(accepts(&ir, b"[1,2,3]"));
    }

    #[test]
    fn contains_never_is_satisfiable_only_when_min_contains_is_zero() {
        let never_min_zero = array_with_contains(
            ItemsPolicy::AllowAny,
            ContainsConstraint {
                policy: ContainsPolicy::Never,
                min: 0,
                max: None,
            },
        );
        assert!(accepts(&never_min_zero, b"[]"));
        assert!(
            accepts(&never_min_zero, b"[1,2,3]"),
            "min=0 is trivially satisfied no matter what the array holds"
        );

        let never_min_one = array_with_contains(
            ItemsPolicy::AllowAny,
            ContainsConstraint {
                policy: ContainsPolicy::Never,
                min: 1,
                max: None,
            },
        );
        assert!(!accepts(&never_min_one, b"[]"));
        assert!(
            !accepts(&never_min_one, b"[1,2,3]"),
            "contains:false can never reach min=1, regardless of array contents"
        );
    }

    #[test]
    fn contains_schema_counts_only_matching_elements_and_min_max_are_inclusive() {
        let mut b = Builder::new(opts());
        let five = b.enum_values(vec![crate::ir::ScalarLit::Int(5)]).unwrap();
        let contains_n = b
            .array_full(
                ItemsPolicy::AllowAny,
                0,
                None,
                false,
                Some(ContainsConstraint {
                    policy: ContainsPolicy::Schema(five),
                    min: 2,
                    max: Some(3),
                }),
                true,
            )
            .unwrap();
        let ir = b.finish(contains_n).unwrap();
        assert!(!accepts(&ir, b"[5]"), "one match: below min=2");
        assert!(accepts(&ir, b"[5,5]"), "two matches: exactly min=2");
        assert!(accepts(&ir, b"[5,5,5]"), "three matches: exactly max=3");
        assert!(!accepts(&ir, b"[5,5,5,5]"), "four matches: above max=3");
        assert!(
            accepts(&ir, b"[1,5,2,5,3]"),
            "non-matching elements do not count toward the total"
        );
    }

    #[test]
    fn contains_and_items_apply_independently_to_the_same_array() {
        // Mirrors the official suite's "items + contains" group: `items` constrains every element,
        // `contains` separately requires at least one element also matching its own schema.
        let mut b = Builder::new(opts());
        let boolean_items = b.boolean().unwrap();
        let bool_true = b
            .enum_values(vec![crate::ir::ScalarLit::Bool(true)])
            .unwrap();
        let n = b
            .array_full(
                ItemsPolicy::Schema(boolean_items),
                0,
                None,
                false,
                Some(ContainsConstraint {
                    policy: ContainsPolicy::Schema(bool_true),
                    min: 1,
                    max: None,
                }),
                true,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        assert!(
            accepts(&ir, b"[false,true]"),
            "every element is boolean and one is true: both constraints hold"
        );
        assert!(
            !accepts(&ir, b"[false,false]"),
            "every element is boolean but none is true: contains fails"
        );
        assert!(
            !accepts(&ir, b"[1,true]"),
            "one element is not boolean: items fails even though contains would hold"
        );
    }

    #[test]
    fn contains_applies_to_prefix_items_tuples_too() {
        let mut b = Builder::new(opts());
        let a = b.boolean().unwrap();
        let bb = b.boolean().unwrap();
        let bool_true = b
            .enum_values(vec![crate::ir::ScalarLit::Bool(true)])
            .unwrap();
        let n = b
            .tuple_full(
                vec![a, bb],
                None,
                0,
                None,
                false,
                Some(ContainsConstraint {
                    policy: ContainsPolicy::Schema(bool_true),
                    min: 1,
                    max: None,
                }),
                true,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        assert!(accepts(&ir, b"[false,true]"));
        assert!(!accepts(&ir, b"[false,false]"), "no position is true");
    }

    // Transcribed from contains.json/minContains.json/maxContains.json with an explicit
    // "type":"array" added (typeless dispatch is a pre-existing, unrelated scope limit).
    #[test]
    #[allow(clippy::type_complexity)]
    fn matches_the_official_test_suite_contains_groups_exactly() {
        let cases: &[(&str, &[(&[u8], bool)])] = &[
            (
                r#"{"type":"array","contains":{"type":"integer","minimum":5}}"#,
                &[
                    (b"[3,4,5]" as &[u8], true),
                    (b"[3,4,6]", true),
                    (b"[3,4,5,6]", true),
                    (b"[2,3,4]", false),
                    (b"[]", false),
                ],
            ),
            (
                r#"{"type":"array","contains":{"const":5}}"#,
                &[
                    (b"[3,4,5]", true),
                    (b"[3,4,5,5]", true),
                    (b"[1,2,3,4]", false),
                ],
            ),
            (
                r#"{"type":"array","contains":true}"#,
                &[(b"[\"foo\"]", true), (b"[]", false)],
            ),
            (
                r#"{"type":"array","contains":false}"#,
                &[(b"[\"foo\"]", false), (b"[]", false)],
            ),
            (
                r#"{"type":"array","items":{"type":"integer","multipleOf":2},"contains":{"type":"integer","multipleOf":3}}"#,
                &[
                    (b"[2,4,8]", false),
                    (b"[3,6,9]", false),
                    (b"[6,12]", true),
                    (b"[1,5]", false),
                ],
            ),
            (
                r#"{"type":"array","contains":{"type":"null"}}"#,
                &[(b"[null]", true)],
            ),
            (
                r#"{"type":"array","minContains":1}"#,
                &[(b"[1]", true), (b"[]", true)],
            ),
            (
                r#"{"type":"array","contains":{"const":1},"minContains":1}"#,
                &[
                    (b"[]", false),
                    (b"[2]", false),
                    (b"[1]", true),
                    (b"[1,2]", true),
                    (b"[1,1]", true),
                ],
            ),
            (
                r#"{"type":"array","contains":{"const":1},"minContains":2}"#,
                &[
                    (b"[]", false),
                    (b"[1]", false),
                    (b"[1,2]", false),
                    (b"[1,1]", true),
                    (b"[1,1,1]", true),
                    (b"[1,2,1]", true),
                ],
            ),
            (
                r#"{"type":"array","contains":{"const":1},"minContains":2.0}"#,
                &[(b"[1]", false), (b"[1,1]", true)],
            ),
            (
                r#"{"type":"array","contains":{"const":1},"maxContains":2,"minContains":2}"#,
                &[
                    (b"[]", false),
                    (b"[1]", false),
                    (b"[1,1,1]", false),
                    (b"[1,1]", true),
                ],
            ),
            // "maxContains < minContains" is omitted: like maxItems < minItems, this dialect
            // rejects it as Malformed at compile time instead of an always-false schema.
            (
                r#"{"type":"array","contains":{"const":1},"minContains":0}"#,
                &[(b"[]", true), (b"[2]", true)],
            ),
            (
                r#"{"type":"array","contains":{"const":1},"minContains":0,"maxContains":1}"#,
                &[(b"[]", true), (b"[1]", true), (b"[1,1]", false)],
            ),
            (
                r#"{"type":"array","maxContains":1}"#,
                &[(b"[1]", true), (b"[1,2]", true)],
            ),
            (
                r#"{"type":"array","contains":{"const":1},"maxContains":1}"#,
                &[
                    (b"[]", false),
                    (b"[1]", true),
                    (b"[1,1]", false),
                    (b"[1,2]", true),
                    (b"[1,2,1]", false),
                ],
            ),
            (
                r#"{"type":"array","contains":{"const":1},"maxContains":1.0}"#,
                &[(b"[1]", true), (b"[1,1]", false)],
            ),
            (
                r#"{"type":"array","contains":{"const":1},"minContains":1,"maxContains":3}"#,
                &[(b"[]", false), (b"[1,1]", true), (b"[1,1,1,1]", false)],
            ),
            (
                r#"{"type":"array","contains":{"const":1},"minContains":0,"maxContains":0}"#,
                &[(b"[]", true), (b"[1]", false)],
            ),
        ];
        let impossible = schema_to_ir(
            r#"{"type":"array","contains":{"const":1},"maxContains":1,"minContains":3}"#,
            opts(),
        )
        .unwrap();
        assert!(!crate::structured::accepts(&impossible, b"[]"));
        assert!(!crate::structured::accepts(&impossible, b"[1]"));
        for (schema, group) in cases {
            let ir = schema_to_ir(schema, opts()).unwrap();
            assert_eq!(
                ir.diagnostics().next(),
                None,
                "schema should be fully supported: {schema}"
            );
            for (data, expected) in *group {
                assert_eq!(
                    accepts(&ir, data),
                    *expected,
                    "schema={schema} data={:?}",
                    std::str::from_utf8(data).unwrap()
                );
            }
        }
    }

    fn root_with_bar_dependent() -> crate::ir::SchemaIR {
        let mut b = Builder::new(opts());
        let foo_int = b.integer(None, None).unwrap();
        let bar_int = b.integer(None, None).unwrap();
        let dependent_schema = b
            .object_full(
                vec![("foo".into(), foo_int), ("bar".into(), bar_int)],
                &[false, false],
                ObjectClosure::Forbidden,
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
        let root = b
            .open_object_full(
                Vec::new(),
                &[],
                Vec::new(),
                AdditionalPolicy::AllowAny,
                None,
                None,
                None,
                vec![("bar".to_string(), dependent_schema)],
                Vec::new(),
            )
            .unwrap();
        b.finish(root).unwrap()
    }

    #[test]
    fn dependent_schema_applies_only_when_its_key_is_present() {
        let ir = root_with_bar_dependent();
        assert!(
            accepts(&ir, br#"{"foo":"quux"}"#),
            "bar is absent, so the dependent schema never triggers"
        );
        assert!(accepts(&ir, br#"{"foo":1,"bar":2}"#));
        assert!(
            !accepts(&ir, br#"{"foo":"quux","bar":2}"#),
            "bar present forces foo to satisfy the dependent schema's own foo:integer"
        );
        assert!(!accepts(&ir, br#"{"bar":"quux"}"#));
    }

    #[test]
    fn dependent_schema_is_checked_against_the_whole_instance_not_the_keyed_value() {
        // The dependent schema for "bar" constrains "foo" too - it is not scoped to bar's own
        // value, it is an additional whole-object schema that only activates when bar is present.
        let ir = root_with_bar_dependent();
        assert!(accepts(&ir, br#"{"bar":1,"foo":1}"#));
        assert!(
            !accepts(&ir, br#"{"bar":1,"foo":1,"extra":true}"#),
            "the dependent schema forbids additional keys once it is active"
        );
    }

    #[test]
    fn dependent_schema_on_a_closed_object_still_routes_through_the_structured_backend() {
        let mut b = Builder::new(opts());
        let any_int = b.integer(None, None).unwrap();
        let dependent_schema = b
            .object_full(
                vec![("a".into(), any_int)],
                &[false],
                ObjectClosure::Forbidden,
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
        let root = b
            .object_full(
                vec![("a".into(), any_int)],
                &[false],
                ObjectClosure::Forbidden,
                vec![("a".to_string(), dependent_schema)],
                Vec::new(),
            )
            .unwrap();
        let ir = b.finish(root).unwrap();
        assert!(
            ir.requires_structured_backend(),
            "a non-empty dependentSchemas run always needs the structured backend"
        );
        assert!(accepts(&ir, br#"{"a":1}"#));
        assert!(
            accepts(&ir, b"{}"),
            "a is optional and absent, so the dependency never triggers"
        );
    }

    // Transcribed from dependentSchemas.json with explicit types added; the "boolean subschemas"
    // and "incompatible with root" (`{}`) groups are omitted - unsupported in any schema position.
    #[test]
    #[allow(clippy::type_complexity)]
    fn matches_the_official_test_suite_dependent_schemas_groups_exactly() {
        let single_dependency = r#"{
            "type":"object","additionalProperties":true,
            "dependentSchemas":{"bar":{"type":"object","additionalProperties":true,
                "properties":{"foo":{"type":"integer"},"bar":{"type":"integer"}}}}
        }"#;
        let escaped_characters = br#"{
            "type":"object","additionalProperties":true,
            "dependentSchemas":{
                "foo\tbar":{"type":"object","additionalProperties":true,"minProperties":4},
                "foo'bar":{"type":"object","additionalProperties":true,
                    "properties":{"foo\"bar":{"type":"integer"}},"required":["foo\"bar"]}
            }
        }"#;
        let cases: &[(&str, &[(&[u8], bool)])] = &[
            (
                single_dependency,
                &[
                    (br#"{"foo":1,"bar":2}"# as &[u8], true),
                    (br#"{"foo":"quux"}"#, true),
                    (br#"{"foo":"quux","bar":2}"#, false),
                    (br#"{"foo":2,"bar":"quux"}"#, false),
                    (br#"{"foo":"quux","bar":"quux"}"#, false),
                ],
            ),
            (
                std::str::from_utf8(escaped_characters).unwrap(),
                &[
                    (br#"{"foo\tbar":1,"a":2,"b":3,"c":4}"# as &[u8], true),
                    (br#"{"foo'bar":{"foo\"bar":1}}"#, false),
                    (br#"{"foo\tbar":1,"a":2}"#, false),
                    (br#"{"foo'bar":1}"#, false),
                ],
            ),
        ];
        for (schema, group) in cases {
            let ir = schema_to_ir(schema, opts()).unwrap();
            assert_eq!(
                ir.diagnostics().next(),
                None,
                "schema should be fully supported: {schema}"
            );
            for (data, expected) in *group {
                assert_eq!(
                    accepts(&ir, data),
                    *expected,
                    "schema={schema} data={:?}",
                    std::str::from_utf8(data).unwrap()
                );
            }
        }
    }

    // A large array is O(n) in `contains`'s element count, not exponential: each element gets one
    // schema check and one counter increment, never a product over prior elements.
    #[test]
    fn contains_stays_linear_on_a_fifty_thousand_element_array() {
        let mut b = Builder::new(opts());
        let five = b.enum_values(vec![crate::ir::ScalarLit::Int(5)]).unwrap();
        let n = b
            .array_full(
                ItemsPolicy::AllowAny,
                0,
                None,
                false,
                Some(ContainsConstraint {
                    policy: ContainsPolicy::Schema(five),
                    min: 1,
                    max: None,
                }),
                true,
            )
            .unwrap();
        let ir = b.finish(n).unwrap();
        let mut data = String::from("[");
        for i in 0..50_000u32 {
            if i > 0 {
                data.push(',');
            }
            data.push_str(&i.to_string());
        }
        data.push_str(",5]");
        let start = std::time::Instant::now();
        assert!(accepts(&ir, data.as_bytes()));
        assert!(
            start.elapsed() < std::time::Duration::from_secs(2),
            "contains counting over 50k elements took {:?}, expected sub-linear-cost behavior",
            start.elapsed()
        );
    }

    // dependentSchemas re-validates the whole object once per triggered key, not once per property
    // seen; a wide object with one dependent key stays proportional to the object, not quadratic.
    #[test]
    fn dependent_schema_stays_linear_on_a_ten_thousand_field_object() {
        let mut b = Builder::new(opts());
        let any_int = b.integer(None, None).unwrap();
        let dependent_schema = b
            .open_object_full(
                Vec::new(),
                &[],
                Vec::new(),
                AdditionalPolicy::AllowAny,
                None,
                None,
                None,
                Vec::new(),
                Vec::new(),
            )
            .unwrap();
        let root = b
            .open_object_full(
                vec![("trigger".into(), any_int)],
                &[false],
                Vec::new(),
                AdditionalPolicy::AllowAny,
                None,
                None,
                None,
                vec![("trigger".to_string(), dependent_schema)],
                Vec::new(),
            )
            .unwrap();
        let ir = b.finish(root).unwrap();
        let mut data = String::from("{\"trigger\":1");
        for i in 0..10_000u32 {
            data.push_str(&format!(",\"k{i}\":{i}"));
        }
        data.push('}');
        let start = std::time::Instant::now();
        assert!(accepts(&ir, data.as_bytes()));
        assert!(
            start.elapsed() < std::time::Duration::from_secs(2),
            "dependentSchemas over a 10k-field object took {:?}",
            start.elapsed()
        );
    }

    /// A byte the mask marks allowed must always be a byte `advance` actually accepts; walks a
    /// sample of corpus schemas reaching the structured backend, one printable byte at a time.
    #[test]
    fn mask_agrees_with_advance_across_the_structured_corpus() {
        const MAX_SCHEMAS: usize = 400;
        const MAX_STEPS: usize = 12;
        let root = format!(
            "{}/../../integration/jsonschemabench/data",
            env!("CARGO_MANIFEST_DIR")
        );
        let toks: Vec<(Vec<u8>, Vec<u32>)> = (0x20u32..=0x7e)
            .map(|b| (vec![b as u8], vec![b - 0x20]))
            .collect();
        let mut checked = 0usize;
        let mut divergences: Vec<String> = Vec::new();
        let mut stack = vec![std::path::PathBuf::from(&root)];
        'schemas: while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if checked >= MAX_SCHEMAS {
                    break 'schemas;
                }
                if path.extension().is_none_or(|e| e != "json") {
                    continue;
                }
                let Ok(text) = std::fs::read_to_string(&path) else {
                    continue;
                };
                let Ok(ir) = crate::frontend::schema_to_ir(&text, CompileOptions::default()) else {
                    continue;
                };
                if ir.diagnostics().next().is_some() || !ir.requires_structured_backend() {
                    continue;
                }
                checked += 1;
                let ir = std::sync::Arc::new(ir);
                let mut m = StructuredMatcher::new(ir.clone());
                let mut prefix: Vec<u8> = Vec::new();
                for _ in 0..MAX_STEPS {
                    let Ok(mask) = m.allowed_mask_from_records(95, records(&toks).into_iter())
                    else {
                        break;
                    };
                    let mut next_byte = None;
                    for b in 0x20u32..=0x7e {
                        let mask_says = mask.get(TokenId(b - 0x20));
                        let mut trial = prefix.clone();
                        trial.push(b as u8);
                        let would_advance = accepts(&ir, &trial) || can_continue(&ir, &trial);
                        if mask_says != would_advance {
                            divergences.push(format!(
                                "{}: prefix={:?} byte={:?} mask_says={mask_says} advance_says={would_advance}",
                                path.display(),
                                String::from_utf8_lossy(&prefix),
                                b as u8 as char,
                            ));
                        }
                        if mask_says && next_byte.is_none() {
                            next_byte = Some(b as u8);
                        }
                    }
                    match next_byte {
                        Some(b) => {
                            prefix.push(b);
                            assert!(m.advance(&[b]), "mask said {b} allowed but advance refused");
                        }
                        None => break,
                    }
                }
            }
        }
        assert!(
            checked > 50,
            "expected a real sample, only checked {checked}"
        );
        assert!(
            divergences.is_empty(),
            "{} mask/advance divergences:\n{}",
            divergences.len(),
            divergences.join("\n")
        );
    }

    #[test]
    fn properties_pattern_properties_and_additional_schema_route_independently() {
        // A known name wins over a pattern match; a pattern-matched key uses its own schema; any
        // remaining key falls to the additional schema - all three resolved independently per key.
        let ir = from_schema(
            r#"{"type":"object","properties":{"id":{"type":"integer"}},
                "patternProperties":{"^x_":{"type":"boolean"}},
                "additionalProperties":{"type":"string"}}"#,
        );
        assert!(accepts(&ir, br#"{"id":1,"x_flag":true,"other":"ok"}"#));
        assert!(!accepts(&ir, br#"{"id":"not an int"}"#));
        assert!(!accepts(&ir, br#"{"x_flag":"not a bool"}"#));
        assert!(!accepts(&ir, br#"{"other":123}"#));
    }

    #[test]
    fn contains_and_unique_items_combine_on_an_array_of_nested_objects() {
        let ir = from_schema(
            r#"{"type":"array","items":{"type":"object"},"uniqueItems":true,"contains":{"properties":{"k":{"const":1}}}}"#,
        );
        assert!(accepts(&ir, br#"[{"k":1},{"a":1,"b":2}]"#));
        assert!(!accepts(&ir, br#"[{"k":1},{"a":1,"b":2},{"b":2,"a":1}]"#));
        // Neither item is required to carry "k"; `properties` only constrains it when present, so
        // both vacuously satisfy the `contains` schema - this is `contains` semantics, not a miss.
        assert!(accepts(&ir, br#"[{"a":1},{"b":2}]"#));
        assert!(!accepts(&ir, br#"[{"k":2},{"k":3}]"#));
    }

    #[test]
    fn prefix_items_with_an_open_any_json_tail_accepts_arbitrary_trailing_values() {
        let ir = from_schema(r#"{"type":"array","prefixItems":[{"type":"integer"}],"items":true}"#);
        assert!(ir.requires_structured_backend());
        assert!(accepts(&ir, br#"[1,"anything",[1,2],{"x":true},null]"#));
        assert!(!accepts(&ir, br#"["not an integer",1]"#));
    }
}
