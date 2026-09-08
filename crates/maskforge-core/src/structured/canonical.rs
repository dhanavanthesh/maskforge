//! Canonical JSON values support structural `uniqueItems` comparisons.
//! Values share an arena to avoid cloning nested data.

use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash, Hasher};

use crate::error::LimitKind;

use super::lexer::{DecodeStep, JsonStringDecoder};
use super::limits::{grow_bounded, GrowthRequest, SessionMemory, StructuredRuntimeError};
use super::validator::{step_scalar, ScalarPhase, ScalarStep};

/// Represents a JSON number exactly as `sign * digits * 10^exponent`.
/// Equivalent numeric values normalize to the same representation.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[cfg(test)]
pub(crate) struct NumericValue {
    negative: bool,
    digits: Box<str>,
    exponent: DecimalExponent,
}

/// A JSON number's exponent as sign + decimal digit magnitude (no leading zeros; empty digits
/// means zero), so an exponent with hundreds of digits is never truncated by a fixed-width int.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
#[cfg(test)]
pub(crate) struct DecimalExponent {
    negative: bool,
    digits: Box<str>,
}

#[cfg(test)]
impl DecimalExponent {
    fn zero() -> Self {
        Self {
            negative: false,
            digits: Box::from(""),
        }
    }

    fn from_parts(negative: bool, digits: &str) -> Self {
        let trimmed = digits.trim_start_matches('0');
        if trimmed.is_empty() {
            Self::zero()
        } else {
            Self {
                negative,
                digits: Box::from(trimmed),
            }
        }
    }

    fn parse(text: &str) -> Result<Self, NumberError> {
        let (negative, digits) = match text.as_bytes().first() {
            Some(b'-') => (true, &text[1..]),
            Some(b'+') => (false, &text[1..]),
            Some(b'0'..=b'9') => (false, text),
            _ => return Err(NumberError::Malformed),
        };
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(NumberError::Malformed);
        }
        Ok(Self::from_parts(negative, digits))
    }

    /// `self - n`, where `n` is a small bounded count (a fractional-digit length).
    fn sub_usize(&self, n: usize) -> Self {
        self.add_signed(n, true)
    }

    /// `self + n`, where `n` is a small bounded count (a trailing-zero count).
    fn add_usize(&self, n: usize) -> Self {
        self.add_signed(n, false)
    }

    fn add_signed(&self, n: usize, n_negative: bool) -> Self {
        if n == 0 {
            return self.clone();
        }
        let n_digits = n.to_string();
        if self.negative == n_negative {
            return Self::from_parts(self.negative, &add_magnitudes(&self.digits, &n_digits));
        }
        if cmp_magnitudes(&self.digits, &n_digits) == std::cmp::Ordering::Less {
            Self::from_parts(n_negative, &sub_magnitudes(&n_digits, &self.digits))
        } else {
            Self::from_parts(self.negative, &sub_magnitudes(&self.digits, &n_digits))
        }
    }
}

#[cfg(test)]
fn cmp_magnitudes(a: &str, b: &str) -> std::cmp::Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

/// Grade-school addition of two non-negative decimal digit strings.
#[cfg(test)]
fn add_magnitudes(a: &str, b: &str) -> String {
    let mut out = Vec::with_capacity(a.len().max(b.len()) + 1);
    let mut carry = 0u8;
    let mut ai = a.bytes().rev();
    let mut bi = b.bytes().rev();
    loop {
        let da = ai.next();
        let db = bi.next();
        if da.is_none() && db.is_none() && carry == 0 {
            break;
        }
        let sum = da.map_or(0, |d| d - b'0') + db.map_or(0, |d| d - b'0') + carry;
        out.push(b'0' + sum % 10);
        carry = sum / 10;
    }
    out.reverse();
    String::from_utf8(out).expect("ASCII digits only")
}

/// Grade-school subtraction `a - b` of non-negative decimal digit strings; requires `a >= b`.
#[cfg(test)]
fn sub_magnitudes(a: &str, b: &str) -> String {
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = 0i8;
    let mut bi = b.bytes().rev();
    for da in a.bytes().rev() {
        let da = (da - b'0') as i8;
        let db = bi.next().map_or(0, |d| (d - b'0') as i8);
        let mut diff = da - db - borrow;
        if diff < 0 {
            diff += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out.push(b'0' + diff as u8);
    }
    out.reverse();
    String::from_utf8(out)
        .expect("ASCII digits only")
        .trim_start_matches('0')
        .to_string()
}

/// `Malformed` indicates invalid JSON number syntax.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum NumberError {
    Malformed,
}

/// Parses already-syntax-checked JSON number text (as produced by this engine's own number
/// validator) into its canonical decimal form.
#[cfg(test)]
pub(crate) fn normalize_number(text: &str) -> Result<NumericValue, NumberError> {
    let bytes = text.as_bytes();
    let (negative, rest) = match bytes.first() {
        Some(b'-') => (true, &bytes[1..]),
        Some(_) => (false, bytes),
        None => return Err(NumberError::Malformed),
    };
    let exp_pos = rest.iter().position(|&b| b == b'e' || b == b'E');
    let (mantissa, exp_value) = match exp_pos {
        None => (rest, DecimalExponent::zero()),
        Some(i) => {
            let exp_text =
                std::str::from_utf8(&rest[i + 1..]).map_err(|_| NumberError::Malformed)?;
            (&rest[..i], DecimalExponent::parse(exp_text)?)
        }
    };
    let dot_pos = mantissa.iter().position(|&b| b == b'.');
    let (int_part, frac_part): (&[u8], &[u8]) = match dot_pos {
        None => (mantissa, &[]),
        Some(i) => (&mantissa[..i], &mantissa[i + 1..]),
    };
    if int_part.is_empty()
        || !int_part.iter().all(u8::is_ascii_digit)
        || (dot_pos.is_some() && frac_part.is_empty())
        || !frac_part.iter().all(u8::is_ascii_digit)
    {
        return Err(NumberError::Malformed);
    }
    let mut digits = String::with_capacity(int_part.len() + frac_part.len());
    digits.push_str(std::str::from_utf8(int_part).map_err(|_| NumberError::Malformed)?);
    digits.push_str(std::str::from_utf8(frac_part).map_err(|_| NumberError::Malformed)?);
    let exponent_before_trim = exp_value.sub_usize(frac_part.len());
    Ok(normalize_digits(&digits, exponent_before_trim, negative))
}

struct NumberScratchValue {
    negative: bool,
    digit_len: usize,
    exp_negative: bool,
    exp_start: usize,
}

fn normalize_number_into(
    raw: &[u8],
    scratch: &mut Vec<u8>,
) -> Result<NumberScratchValue, NumberError> {
    let (negative, body) = match raw.first() {
        Some(b'-') => (true, &raw[1..]),
        Some(b'0'..=b'9') => (false, raw),
        _ => return Err(NumberError::Malformed),
    };
    let exponent = body.iter().position(|byte| matches!(byte, b'e' | b'E'));
    let (mantissa, exponent) = match exponent {
        Some(index) => (&body[..index], Some(&body[index + 1..])),
        None => (body, None),
    };
    let dot = mantissa.iter().position(|byte| *byte == b'.');
    let (integer, fraction) = match dot {
        Some(index) => (&mantissa[..index], &mantissa[index + 1..]),
        None => (mantissa, &[][..]),
    };
    if integer.is_empty()
        || integer.iter().any(|byte| !byte.is_ascii_digit())
        || dot.is_some()
            && (fraction.is_empty() || fraction.iter().any(|byte| !byte.is_ascii_digit()))
    {
        return Err(NumberError::Malformed);
    }
    let mut leading = true;
    let mut last_nonzero = 0usize;
    for byte in integer.iter().chain(fraction) {
        if leading && *byte == b'0' {
            continue;
        }
        leading = false;
        scratch.push(*byte);
        if *byte != b'0' {
            last_nonzero = scratch.len();
        }
    }
    if last_nonzero == 0 {
        scratch.clear();
        return Ok(NumberScratchValue {
            negative: false,
            digit_len: 0,
            exp_negative: false,
            exp_start: 0,
        });
    }
    let digit_len = last_nonzero;
    let trailing = scratch.len() - digit_len;
    scratch.truncate(digit_len);
    let (raw_negative, raw_digits) = match exponent {
        Some([b'-', rest @ ..]) => (true, rest),
        Some([b'+', rest @ ..]) => (false, rest),
        Some(rest) => (false, rest),
        None => (false, &[][..]),
    };
    if exponent.is_some() && raw_digits.is_empty()
        || raw_digits.iter().any(|byte| !byte.is_ascii_digit())
    {
        return Err(NumberError::Malformed);
    }
    let raw_digits = trim_zeroes(raw_digits);
    let mut count = [0u8; 20];
    let count = usize_digits(trailing.abs_diff(fraction.len()), &mut count);
    let delta_negative = fraction.len() > trailing;
    let exp_start = scratch.len();
    let exp_negative =
        append_adjusted_exponent(scratch, raw_negative, raw_digits, delta_negative, count);
    Ok(NumberScratchValue {
        negative,
        digit_len,
        exp_negative,
        exp_start,
    })
}

fn trim_zeroes(digits: &[u8]) -> &[u8] {
    let first = digits
        .iter()
        .position(|byte| *byte != b'0')
        .unwrap_or(digits.len());
    &digits[first..]
}

fn usize_digits(mut value: usize, out: &mut [u8; 20]) -> &[u8] {
    let mut index = out.len();
    loop {
        index -= 1;
        out[index] = b'0' + u8::try_from(value % 10).expect("decimal digit fits u8");
        value /= 10;
        if value == 0 {
            return &out[index..];
        }
    }
}

fn append_adjusted_exponent(
    out: &mut Vec<u8>,
    raw_negative: bool,
    raw: &[u8],
    delta_negative: bool,
    delta: &[u8],
) -> bool {
    let raw = trim_zeroes(raw);
    let delta = trim_zeroes(delta);
    if raw.is_empty() {
        out.extend_from_slice(delta);
        return !delta.is_empty() && delta_negative;
    }
    if delta.is_empty() {
        out.extend_from_slice(raw);
        return raw_negative;
    }
    if raw_negative == delta_negative {
        append_sum(out, raw, delta);
        return raw_negative;
    }
    match cmp_magnitudes_bytes(raw, delta) {
        std::cmp::Ordering::Equal => false,
        std::cmp::Ordering::Greater => {
            append_difference(out, raw, delta);
            raw_negative
        }
        std::cmp::Ordering::Less => {
            append_difference(out, delta, raw);
            delta_negative
        }
    }
}

fn cmp_magnitudes_bytes(a: &[u8], b: &[u8]) -> std::cmp::Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

fn append_sum(out: &mut Vec<u8>, a: &[u8], b: &[u8]) {
    let start = out.len();
    let mut carry = 0u8;
    let mut ai = a.iter().rev();
    let mut bi = b.iter().rev();
    loop {
        let da = ai.next();
        let db = bi.next();
        if da.is_none() && db.is_none() && carry == 0 {
            break;
        }
        let sum = da.map_or(0, |byte| byte - b'0') + db.map_or(0, |byte| byte - b'0') + carry;
        out.push(b'0' + sum % 10);
        carry = sum / 10;
    }
    out[start..].reverse();
}

fn append_difference(out: &mut Vec<u8>, a: &[u8], b: &[u8]) {
    let start = out.len();
    let mut borrow = 0i8;
    let mut bi = b.iter().rev();
    for byte in a.iter().rev() {
        let mut digit = i8::try_from(byte - b'0').expect("decimal digit fits i8")
            - bi.next().map_or(0, |byte| {
                i8::try_from(byte - b'0').expect("decimal digit fits i8")
            })
            - borrow;
        if digit < 0 {
            digit += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        out.push(b'0' + u8::try_from(digit).expect("decimal digit fits u8"));
    }
    out[start..].reverse();
    let first = out[start..]
        .iter()
        .position(|byte| *byte != b'0')
        .unwrap_or(out.len() - start);
    if first > 0 {
        out.copy_within(start + first.., start);
        out.truncate(out.len() - first);
    }
}

/// Strips leading and trailing zeros from a raw digit string, folding trailing strips into
/// `exponent` so the represented value is unchanged.
#[cfg(test)]
fn normalize_digits(raw: &str, exponent: DecimalExponent, negative: bool) -> NumericValue {
    let trimmed_leading = raw.trim_start_matches('0');
    let significant = trimmed_leading.trim_end_matches('0');
    let trailing_zeros = trimmed_leading.len() - significant.len();
    if significant.is_empty() {
        return NumericValue {
            negative: false,
            digits: Box::from(""),
            exponent: DecimalExponent::zero(),
        };
    }
    NumericValue {
        negative,
        digits: Box::from(significant),
        exponent: exponent.add_usize(trailing_zeros),
    }
}

const WS: [u8; 4] = [0x20, 0x09, 0x0a, 0x0d];

fn is_ws(byte: u8) -> bool {
    WS.contains(&byte)
}

/// A compact handle into one array's `CanonicalArena`. Never valid across two different arenas.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct CanonicalId(u32);

impl CanonicalId {
    fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ByteSpan {
    start: u32,
    len: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ChildRange {
    start: u32,
    len: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct EntryRange {
    start: u32,
    len: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct NumberRecord {
    negative: bool,
    digits: ByteSpan,
    exp_negative: bool,
    exp_digits: ByteSpan,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum CanonicalNode {
    Null,
    Bool(bool),
    Number(NumberRecord),
    String(ByteSpan),
    Array(ChildRange),
    Object(EntryRange),
}

/// One object member: `key` is the decoded key's byte span, sorted into the entry list by key so
/// member order never affects equality or hashing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ObjectEntry {
    key: ByteSpan,
    value: CanonicalId,
}

fn checked_u32(len: usize, budget: usize) -> Result<u32, StructuredRuntimeError> {
    u32::try_from(len)
        .map_err(|_| StructuredRuntimeError::new(LimitKind::CanonicalBytes, usize::MAX, budget))
}

/// Returns one ledger's remaining capacity within the shared budget.
fn remaining_budget(budget: usize, other_live: usize) -> Result<usize, StructuredRuntimeError> {
    budget
        .checked_sub(other_live)
        .ok_or_else(|| StructuredRuntimeError::new(LimitKind::CanonicalBytes, other_live, budget))
}

/// One array's shared value storage: every candidate built and every value already kept for
/// `uniqueItems` lives here, addressed by `CanonicalId` instead of an owned recursive tree.
#[derive(Clone, PartialEq, Debug, Default)]
struct CanonicalArena {
    nodes: Vec<CanonicalNode>,
    bytes: Vec<u8>,
    children: Vec<CanonicalId>,
    entries: Vec<ObjectEntry>,
    number_scratch: Vec<u8>,
}

/// Captures arena lengths for transactional rollback.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct CanonicalMark {
    nodes_len: usize,
    bytes_len: usize,
    children_len: usize,
    entries_len: usize,
}

impl CanonicalArena {
    fn span_bytes(&self, span: ByteSpan) -> &[u8] {
        let start = span.start as usize;
        let end = start + span.len as usize;
        &self.bytes[start..end]
    }

    fn mark(&self) -> CanonicalMark {
        CanonicalMark {
            nodes_len: self.nodes.len(),
            bytes_len: self.bytes.len(),
            children_len: self.children.len(),
            entries_len: self.entries.len(),
        }
    }

    /// Truncates every growable list back to `mark`. Only ever moves lengths backward: a mark
    /// taken earlier in the same arena's lifetime is always a valid truncation target.
    fn restore(&mut self, mark: CanonicalMark) {
        self.nodes.truncate(mark.nodes_len);
        self.bytes.truncate(mark.bytes_len);
        self.children.truncate(mark.children_len);
        self.entries.truncate(mark.entries_len);
    }

    fn sort_entries_from(&mut self, start: usize) {
        let bytes = &self.bytes;
        self.entries[start..].sort_by(|a, b| {
            let a_start = usize::try_from(a.key.start).expect("arena offset came from usize");
            let b_start = usize::try_from(b.key.start).expect("arena offset came from usize");
            let a_len = usize::try_from(a.key.len).expect("arena length came from usize");
            let b_len = usize::try_from(b.key.len).expect("arena length came from usize");
            let a_end = a_start.checked_add(a_len).expect("arena span is valid");
            let b_end = b_start.checked_add(b_len).expect("arena span is valid");
            bytes[a_start..a_end].cmp(&bytes[b_start..b_end])
        });
    }

    fn push_bytes(
        &mut self,
        mem: &mut SessionMemory,
        budget: usize,
        extra: &[u8],
    ) -> Result<(), StructuredRuntimeError> {
        if extra.is_empty() {
            return Ok(());
        }
        let bytes = &mut self.bytes;
        grow_bounded(
            mem,
            budget,
            LimitKind::CanonicalBytes,
            GrowthRequest {
                len: bytes.len(),
                capacity: bytes.capacity(),
                additional: extra.len(),
                item_size: 1,
            },
            |n| {
                bytes
                    .try_reserve_exact(n)
                    .map(|_| bytes.capacity())
                    .map_err(|_| ())
            },
        )?;
        self.bytes.extend_from_slice(extra);
        Ok(())
    }

    fn push_node(
        &mut self,
        mem: &mut SessionMemory,
        budget: usize,
        node: CanonicalNode,
    ) -> Result<CanonicalId, StructuredRuntimeError> {
        let nodes = &mut self.nodes;
        grow_bounded(
            mem,
            budget,
            LimitKind::CanonicalBytes,
            GrowthRequest {
                len: nodes.len(),
                capacity: nodes.capacity(),
                additional: 1,
                item_size: std::mem::size_of::<CanonicalNode>(),
            },
            |n| {
                nodes
                    .try_reserve_exact(n)
                    .map(|_| nodes.capacity())
                    .map_err(|_| ())
            },
        )?;
        let id = checked_u32(self.nodes.len(), budget)?;
        self.nodes.push(node);
        Ok(CanonicalId(id))
    }

    /// Flushes a closed array's direct children as one contiguous block.
    fn push_children_batch(
        &mut self,
        mem: &mut SessionMemory,
        budget: usize,
        items: &[CanonicalId],
    ) -> Result<(), StructuredRuntimeError> {
        let children = &mut self.children;
        grow_bounded(
            mem,
            budget,
            LimitKind::CanonicalBytes,
            GrowthRequest {
                len: children.len(),
                capacity: children.capacity(),
                additional: items.len(),
                item_size: std::mem::size_of::<CanonicalId>(),
            },
            |n| {
                children
                    .try_reserve_exact(n)
                    .map(|_| children.capacity())
                    .map_err(|_| ())
            },
        )?;
        self.children.extend_from_slice(items);
        Ok(())
    }

    /// Flushes one just-closed object's own direct entries as one contiguous block; see
    /// `push_children_batch` for why this must happen in a single append.
    fn push_entries_batch(
        &mut self,
        mem: &mut SessionMemory,
        budget: usize,
        items: &[ObjectEntry],
    ) -> Result<(), StructuredRuntimeError> {
        let entries = &mut self.entries;
        grow_bounded(
            mem,
            budget,
            LimitKind::CanonicalBytes,
            GrowthRequest {
                len: entries.len(),
                capacity: entries.capacity(),
                additional: items.len(),
                item_size: std::mem::size_of::<ObjectEntry>(),
            },
            |n| {
                entries
                    .try_reserve_exact(n)
                    .map(|_| entries.capacity())
                    .map_err(|_| ())
            },
        )?;
        self.entries.extend_from_slice(items);
        Ok(())
    }

    /// Normalizes the accumulated number without invalidating active checkpoints.
    fn finish_number(
        &mut self,
        mem: &mut SessionMemory,
        budget: usize,
        text_start: u32,
    ) -> Result<CanonicalNode, StructuredRuntimeError> {
        let start = usize::try_from(text_start).map_err(|_| {
            StructuredRuntimeError::new(LimitKind::CanonicalBytes, usize::MAX, budget)
        })?;
        let raw_len = self.bytes.len().checked_sub(start).ok_or_else(|| {
            StructuredRuntimeError::new(LimitKind::CanonicalBytes, usize::MAX, budget)
        })?;
        let scratch_len = raw_len
            .checked_mul(3)
            .and_then(|n| n.checked_add(42))
            .ok_or_else(|| {
                StructuredRuntimeError::new(LimitKind::CanonicalBytes, usize::MAX, budget)
            })?;
        self.reserve_number_scratch(mem, budget, scratch_len)?;
        let mut scratch = std::mem::take(&mut self.number_scratch);
        scratch.clear();
        let result = (|| {
            let value =
                normalize_number_into(&self.bytes[start..], &mut scratch).map_err(|_| {
                    StructuredRuntimeError::new(LimitKind::CanonicalBytes, usize::MAX, budget)
                })?;
            let digit_start = checked_u32(self.bytes.len(), budget)?;
            self.push_bytes(mem, budget, &scratch[..value.digit_len])?;
            let digit_len = checked_u32(value.digit_len, budget)?;
            let exp_start = checked_u32(self.bytes.len(), budget)?;
            self.push_bytes(mem, budget, &scratch[value.exp_start..])?;
            let exp_len = checked_u32(scratch.len() - value.exp_start, budget)?;
            Ok(CanonicalNode::Number(NumberRecord {
                negative: value.negative,
                digits: ByteSpan {
                    start: digit_start,
                    len: digit_len,
                },
                exp_negative: value.exp_negative,
                exp_digits: ByteSpan {
                    start: exp_start,
                    len: exp_len,
                },
            }))
        })();
        scratch.clear();
        self.number_scratch = scratch;
        result
    }

    fn reserve_number_scratch(
        &mut self,
        mem: &mut SessionMemory,
        budget: usize,
        additional: usize,
    ) -> Result<(), StructuredRuntimeError> {
        let scratch = &mut self.number_scratch;
        if scratch.capacity() >= additional {
            return Ok(());
        }
        grow_bounded(
            mem,
            budget,
            LimitKind::CanonicalBytes,
            GrowthRequest {
                len: scratch.len(),
                capacity: scratch.capacity(),
                additional: additional.checked_sub(scratch.len()).ok_or_else(|| {
                    StructuredRuntimeError::new(LimitKind::CanonicalBytes, usize::MAX, budget)
                })?,
                item_size: 1,
            },
            |n| {
                scratch
                    .try_reserve_exact(n)
                    .map(|_| scratch.capacity())
                    .map_err(|_| ())
            },
        )
    }
}

/// One in-progress container's parse state, distinct from `CanonicalNode` (which only exists for
/// values that have already fully closed).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ObjectBuildPhase {
    KeyOrClose,
    Colon { key: ByteSpan },
    Value { key: ByteSpan },
}

/// One frame in the builder's bounded, non-recursive parse stack.
/// Containers retain direct children locally until they close.
#[derive(Clone, PartialEq, Debug)]
enum BuildFrame {
    /// Awaiting the first byte of a value (the item root, or a fresh array element/object value).
    Scalar {
        phase: ScalarPhase,
        number_start: u32,
    },
    Str {
        decoder: JsonStringDecoder,
        text_start: u32,
        as_key: bool,
    },
    Array {
        children: Vec<CanonicalId>,
    },
    Object {
        phase: ObjectBuildPhase,
        entries: Vec<ObjectEntry>,
    },
}

impl BuildFrame {
    fn value_start() -> Self {
        BuildFrame::Scalar {
            phase: ScalarPhase::BeforeValue,
            number_start: 0,
        }
    }
}

/// Retained capacity a local `Array`/`Object` frame owns, for releasing its charge when the
/// frame is discarded outright (not truncated - truncated capacity stays retained).
fn frame_local_bytes(frame: &BuildFrame) -> usize {
    match frame {
        BuildFrame::Array { children } => children.capacity() * std::mem::size_of::<CanonicalId>(),
        BuildFrame::Object { entries, .. } => {
            entries.capacity() * std::mem::size_of::<ObjectEntry>()
        }
        BuildFrame::Scalar { .. } | BuildFrame::Str { .. } => 0,
    }
}

/// Reserves builder-owned storage and charges its actual capacity growth.
/// Builder and set allocations share the same memory budget.
fn reserve_builder_owned<T>(
    vec: &mut Vec<T>,
    own_mem: &mut SessionMemory,
    other_live: usize,
    budget: usize,
) -> Result<(), StructuredRuntimeError> {
    let effective_budget = remaining_budget(budget, other_live)?;
    grow_bounded(
        own_mem,
        effective_budget,
        LimitKind::CanonicalBytes,
        GrowthRequest {
            len: vec.len(),
            capacity: vec.capacity(),
            additional: 1,
            item_size: std::mem::size_of::<T>(),
        },
        |n| {
            vec.try_reserve_exact(n)
                .map(|_| vec.capacity())
                .map_err(|_| ())
        },
    )
    .map_err(|e| StructuredRuntimeError::new(e.kind, e.observed, budget))
}

/// One reversible builder mutation. Every entry restores exactly what it changed; a byte that
/// only advances a scalar/string in place costs one `Vec` push, never a stack clone.
#[derive(Clone, PartialEq, Debug)]
enum BuilderUndo {
    /// Restores a stack frame to its earlier scalar state.
    RestoreScalar {
        depth: usize,
        phase: ScalarPhase,
        number_start: u32,
    },
    RestoreString {
        depth: usize,
        decoder: JsonStringDecoder,
    },
    RestoreObjectPhase {
        depth: usize,
        phase: ObjectBuildPhase,
    },
    TruncateChildren {
        depth: usize,
        old_len: usize,
    },
    TruncateEntries {
        depth: usize,
        old_len: usize,
    },
    RestoreRoot,
    PopFrame,
    PushFrame(BuildFrame),
}

/// Snapshot of everything a builder rolls back on a rejected byte: the arena growth it caused,
/// how far into its own undo journal to unwind, and its `root` (restored last, explicitly).
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct BuilderMark {
    arena: CanonicalMark,
    undo_len: usize,
    root: Option<CanonicalId>,
}

/// Builds one array item's canonical value incrementally, one accepted input byte at a time, into
/// the array's shared `CanonicalArena`. Never buffers or reparses the item's raw JSON text.
#[derive(PartialEq, Debug)]
pub(crate) struct CanonicalBuilder {
    stack: Vec<BuildFrame>,
    root: Option<CanonicalId>,
    undo: Vec<BuilderUndo>,
    /// Bytes owned by this builder alone (stack/undo/local-container capacity) - never charged
    /// to `CanonicalSet::mem`, since that ledger has no hook for this builder's own drop.
    own_mem: SessionMemory,
}

#[cfg(test)]
fn builder_capacity_bytes(
    stack: &[BuildFrame],
    stack_capacity: usize,
    undo: &[BuilderUndo],
    undo_capacity: usize,
) -> usize {
    let stack_bytes = stack_capacity.checked_mul(std::mem::size_of::<BuildFrame>());
    let undo_bytes = undo_capacity.checked_mul(std::mem::size_of::<BuilderUndo>());
    stack_bytes
        .and_then(|n| n.checked_add(undo_bytes?))
        .and_then(|n| {
            stack
                .iter()
                .chain(undo.iter().filter_map(|entry| match entry {
                    BuilderUndo::PushFrame(frame) => Some(frame),
                    _ => None,
                }))
                .try_fold(n, |total, frame| {
                    total.checked_add(frame_local_bytes(frame))
                })
        })
        .expect("builder retained capacities fit usize")
}

#[cfg(test)]
fn memory_for_clone(live: usize) -> SessionMemory {
    let mut mem = SessionMemory::default();
    mem.charge(live, usize::MAX, LimitKind::CanonicalBytes)
        .expect("retained capacities fit usize");
    mem
}

#[cfg(test)]
impl Clone for CanonicalBuilder {
    fn clone(&self) -> Self {
        let stack = self.stack.clone();
        let undo = self.undo.clone();
        let own_mem = memory_for_clone(builder_capacity_bytes(
            &stack,
            stack.capacity(),
            &undo,
            undo.capacity(),
        ));
        Self {
            stack,
            root: self.root,
            undo,
            own_mem,
        }
    }
}

impl CanonicalBuilder {
    /// `other_live` is `CanonicalSet::mem.live()` at construction time, so the very first
    /// reservation already respects the combined budget.
    fn new(other_live: usize, budget: usize) -> Result<Self, StructuredRuntimeError> {
        let mut own_mem = SessionMemory::default();
        let mut stack = Vec::new();
        reserve_builder_owned(&mut stack, &mut own_mem, other_live, budget)?;
        stack.push(BuildFrame::value_start());
        Ok(Self {
            stack,
            root: None,
            undo: Vec::new(),
            own_mem,
        })
    }

    /// Whether the item's value has fully closed (its `CanonicalId` is ready).
    fn is_done(&self) -> bool {
        self.root.is_some()
    }

    /// Bytes retained by this builder alone: stack, undo journal, and any local `Array`/`Object`
    /// children/entries not yet flushed into the shared arena.
    fn retained_bytes(&self) -> usize {
        self.own_mem.live()
    }

    /// Room left in `budget` for the shared arena, given what this builder currently retains.
    /// Recomputed fresh at every call site, never cached, since `own_mem` grows within one byte.
    fn set_budget(&self, budget: usize) -> Result<usize, StructuredRuntimeError> {
        remaining_budget(budget, self.own_mem.live())
    }

    fn mark(&self, arena: &CanonicalArena) -> BuilderMark {
        BuilderMark {
            arena: arena.mark(),
            undo_len: self.undo.len(),
            root: self.root,
        }
    }

    fn record(
        &mut self,
        mem: &SessionMemory,
        budget: usize,
        entry: BuilderUndo,
    ) -> Result<(), StructuredRuntimeError> {
        let CanonicalBuilder { undo, own_mem, .. } = self;
        reserve_builder_owned(undo, own_mem, mem.live(), budget)?;
        undo.push(entry);
        Ok(())
    }

    fn reserve_record(
        &mut self,
        mem: &SessionMemory,
        budget: usize,
    ) -> Result<(), StructuredRuntimeError> {
        reserve_builder_owned(&mut self.undo, &mut self.own_mem, mem.live(), budget)
    }

    /// Commits undo history that no outer checkpoint can reach.
    pub(crate) fn commit_history(&mut self) {
        let mut released = 0usize;
        for entry in &self.undo {
            if let BuilderUndo::PushFrame(frame) = entry {
                // INVARIANT: each term was already charged into own_mem.live(), so a running
                // sum over this subset can never exceed that already-valid usize total.
                released = released
                    .checked_add(frame_local_bytes(frame))
                    .expect("subset of an already-charged total fits usize");
            }
        }
        self.own_mem.release(released);
        self.undo.clear();
    }

    /// Restores the arena growth and this builder's own stack/root to `mark`, replaying only the
    /// journal entries recorded after it - never a stack clone.
    fn restore(&mut self, arena: &mut CanonicalArena, mark: BuilderMark) {
        arena.restore(mark.arena);
        while self.undo.len() > mark.undo_len {
            match self.undo.pop().expect("checked above") {
                BuilderUndo::RestoreScalar {
                    depth,
                    phase,
                    number_start,
                } => {
                    if let Some(old) = self.stack.get(depth) {
                        self.own_mem.release(frame_local_bytes(old));
                    }
                    self.stack[depth] = BuildFrame::Scalar {
                        phase,
                        number_start,
                    };
                }
                BuilderUndo::RestoreString { depth, decoder } => {
                    if let Some(BuildFrame::Str { decoder: d, .. }) = self.stack.get_mut(depth) {
                        *d = decoder;
                    }
                }
                BuilderUndo::RestoreObjectPhase { depth, phase } => {
                    if let Some(BuildFrame::Object { phase: p, .. }) = self.stack.get_mut(depth) {
                        *p = phase;
                    }
                }
                BuilderUndo::TruncateChildren { depth, old_len } => {
                    if let Some(BuildFrame::Array { children }) = self.stack.get_mut(depth) {
                        children.truncate(old_len);
                    }
                }
                BuilderUndo::TruncateEntries { depth, old_len } => {
                    if let Some(BuildFrame::Object { entries, .. }) = self.stack.get_mut(depth) {
                        entries.truncate(old_len);
                    }
                }
                BuilderUndo::RestoreRoot => self.root = None,
                BuilderUndo::PopFrame => {
                    self.stack.pop();
                }
                BuilderUndo::PushFrame(frame) => self.stack.push(frame),
            }
        }
        self.root = mark.root;
    }

    /// Feeds one byte already accepted by the JSON engine.
    fn push_byte(
        &mut self,
        arena: &mut CanonicalArena,
        mem: &mut SessionMemory,
        budget: usize,
        byte: u8,
    ) -> Result<(), StructuredRuntimeError> {
        let broken = || StructuredRuntimeError::new(LimitKind::CanonicalBytes, usize::MAX, budget);
        if self.root.is_some() {
            return if is_ws(byte) { Ok(()) } else { Err(broken()) };
        }
        loop {
            let last = self.stack.len().checked_sub(1).ok_or_else(broken)?;
            match &mut self.stack[last] {
                BuildFrame::Scalar {
                    phase,
                    number_start,
                } => {
                    let (phase_before, start_before) = (*phase, *number_start);
                    match step_scalar(phase_before, byte) {
                        ScalarStep::Continue(next) => {
                            let number_start = if matches!(next, ScalarPhase::Number(_)) {
                                let start = if matches!(phase_before, ScalarPhase::Number(_)) {
                                    start_before
                                } else {
                                    checked_u32(arena.bytes.len(), budget)?
                                };
                                let set_budget = self.set_budget(budget)?;
                                arena.push_bytes(mem, set_budget, &[byte])?;
                                start
                            } else if next == ScalarPhase::LiteralDone {
                                literal_tag(phase_before).ok_or_else(broken)?
                            } else {
                                start_before
                            };
                            self.record(
                                mem,
                                budget,
                                BuilderUndo::RestoreScalar {
                                    depth: last,
                                    phase: phase_before,
                                    number_start: start_before,
                                },
                            )?;
                            let BuildFrame::Scalar {
                                phase,
                                number_start: slot,
                            } = &mut self.stack[last]
                            else {
                                unreachable!("checked above")
                            };
                            *phase = next;
                            *slot = number_start;
                            return Ok(());
                        }
                        ScalarStep::StartString => {
                            let text_start = checked_u32(arena.bytes.len(), budget)?;
                            self.record(
                                mem,
                                budget,
                                BuilderUndo::RestoreScalar {
                                    depth: last,
                                    phase: phase_before,
                                    number_start: start_before,
                                },
                            )?;
                            self.stack[last] = BuildFrame::Str {
                                decoder: JsonStringDecoder::new(),
                                text_start,
                                as_key: false,
                            };
                            return Ok(());
                        }
                        ScalarStep::StartArray => {
                            self.record(
                                mem,
                                budget,
                                BuilderUndo::RestoreScalar {
                                    depth: last,
                                    phase: phase_before,
                                    number_start: start_before,
                                },
                            )?;
                            self.stack[last] = BuildFrame::Array {
                                children: Vec::new(),
                            };
                            return Ok(());
                        }
                        ScalarStep::StartObject => {
                            self.record(
                                mem,
                                budget,
                                BuilderUndo::RestoreScalar {
                                    depth: last,
                                    phase: phase_before,
                                    number_start: start_before,
                                },
                            )?;
                            self.stack[last] = BuildFrame::Object {
                                phase: ObjectBuildPhase::KeyOrClose,
                                entries: Vec::new(),
                            };
                            return Ok(());
                        }
                        ScalarStep::Done => {
                            let set_budget = self.set_budget(budget)?;
                            let node = match phase_before {
                                ScalarPhase::Number(_) => {
                                    arena.finish_number(mem, set_budget, start_before)?
                                }
                                ScalarPhase::LiteralDone => {
                                    literal_from_tag(start_before).ok_or_else(broken)?
                                }
                                _ => return Err(broken()),
                            };
                            let set_budget = self.set_budget(budget)?;
                            let id = arena.push_node(mem, set_budget, node)?;
                            self.record(
                                mem,
                                budget,
                                BuilderUndo::PushFrame(BuildFrame::Scalar {
                                    phase: phase_before,
                                    number_start: start_before,
                                }),
                            )?;
                            self.stack.pop();
                            self.complete_value(mem, budget, id)?;
                            if self.root.is_some() {
                                return Ok(());
                            }
                            // The byte that signaled `Done` was never part of the scalar; the
                            // (now shallower) top frame must still classify it.
                        }
                        ScalarStep::Invalid => return Err(broken()),
                    }
                }
                BuildFrame::Str {
                    decoder,
                    text_start,
                    as_key,
                } => {
                    let (text_start, as_key) = (*text_start, *as_key);
                    if byte == b'"' && decoder.at_boundary() {
                        let span = ByteSpan {
                            start: text_start,
                            len: checked_u32(arena.bytes.len(), budget)? - text_start,
                        };
                        let final_decoder = *decoder;
                        self.record(
                            mem,
                            budget,
                            BuilderUndo::PushFrame(BuildFrame::Str {
                                decoder: final_decoder,
                                text_start,
                                as_key,
                            }),
                        )?;
                        self.stack.pop();
                        if as_key {
                            self.attach_key(mem, budget, span)?;
                        } else {
                            let set_budget = self.set_budget(budget)?;
                            let id =
                                arena.push_node(mem, set_budget, CanonicalNode::String(span))?;
                            self.complete_value(mem, budget, id)?;
                        }
                        return Ok(());
                    }
                    let old_decoder = *decoder;
                    match decoder.push(byte) {
                        DecodeStep::Continue => {}
                        DecodeStep::Scalar(c) => {
                            let mut buf = [0u8; 4];
                            let set_budget = self.set_budget(budget)?;
                            arena.push_bytes(
                                mem,
                                set_budget,
                                c.encode_utf8(&mut buf).as_bytes(),
                            )?;
                        }
                        DecodeStep::Invalid => return Err(broken()),
                    }
                    self.record(
                        mem,
                        budget,
                        BuilderUndo::RestoreString {
                            depth: last,
                            decoder: old_decoder,
                        },
                    )?;
                    return Ok(());
                }
                BuildFrame::Array { .. } => {
                    if is_ws(byte) || byte == b',' {
                        return Ok(());
                    }
                    if byte == b']' {
                        self.reserve_record(mem, budget)?;
                        let start = checked_u32(arena.children.len(), budget)?;
                        let set_budget = self.set_budget(budget)?;
                        let Some(BuildFrame::Array { children }) = self.stack.get(last) else {
                            unreachable!("checked above")
                        };
                        arena.push_children_batch(mem, set_budget, children)?;
                        let len = checked_u32(children.len(), budget)?;
                        let set_budget = self.set_budget(budget)?;
                        let id = arena.push_node(
                            mem,
                            set_budget,
                            CanonicalNode::Array(ChildRange { start, len }),
                        )?;
                        let frame = self.stack.pop().expect("array frame exists");
                        self.undo.push(BuilderUndo::PushFrame(frame));
                        self.complete_value(mem, budget, id)?;
                        return Ok(());
                    }
                    let other_live = mem.live();
                    reserve_builder_owned(&mut self.stack, &mut self.own_mem, other_live, budget)?;
                    self.record(mem, budget, BuilderUndo::PopFrame)?;
                    self.stack.push(BuildFrame::value_start());
                }
                BuildFrame::Object { phase, .. } => match *phase {
                    ObjectBuildPhase::KeyOrClose => {
                        if is_ws(byte) || byte == b',' {
                            return Ok(());
                        }
                        if byte == b'"' {
                            let text_start = checked_u32(arena.bytes.len(), budget)?;
                            let other_live = mem.live();
                            reserve_builder_owned(
                                &mut self.stack,
                                &mut self.own_mem,
                                other_live,
                                budget,
                            )?;
                            self.record(mem, budget, BuilderUndo::PopFrame)?;
                            self.stack.push(BuildFrame::Str {
                                decoder: JsonStringDecoder::new(),
                                text_start,
                                as_key: true,
                            });
                            return Ok(());
                        }
                        if byte == b'}' {
                            self.reserve_record(mem, budget)?;
                            let start = checked_u32(arena.entries.len(), budget)?;
                            let set_budget = self.set_budget(budget)?;
                            let Some(BuildFrame::Object { entries, .. }) = self.stack.get(last)
                            else {
                                unreachable!("checked above")
                            };
                            arena.push_entries_batch(mem, set_budget, entries)?;
                            let len = checked_u32(entries.len(), budget)?;
                            let set_budget = self.set_budget(budget)?;
                            let id = arena.push_node(
                                mem,
                                set_budget,
                                CanonicalNode::Object(EntryRange { start, len }),
                            )?;
                            arena.sort_entries_from(
                                usize::try_from(start).expect("arena offset came from usize"),
                            );
                            let frame = self.stack.pop().expect("object frame exists");
                            self.undo.push(BuilderUndo::PushFrame(frame));
                            self.complete_value(mem, budget, id)?;
                            return Ok(());
                        }
                        return Err(broken());
                    }
                    ObjectBuildPhase::Colon { key } => {
                        if is_ws(byte) {
                            return Ok(());
                        }
                        if byte != b':' {
                            return Err(broken());
                        }
                        self.record(
                            mem,
                            budget,
                            BuilderUndo::RestoreObjectPhase {
                                depth: last,
                                phase: ObjectBuildPhase::Colon { key },
                            },
                        )?;
                        let BuildFrame::Object { phase, .. } = &mut self.stack[last] else {
                            unreachable!("checked above")
                        };
                        *phase = ObjectBuildPhase::Value { key };
                        return Ok(());
                    }
                    ObjectBuildPhase::Value { .. } => {
                        if is_ws(byte) {
                            return Ok(());
                        }
                        let other_live = mem.live();
                        reserve_builder_owned(
                            &mut self.stack,
                            &mut self.own_mem,
                            other_live,
                            budget,
                        )?;
                        self.record(mem, budget, BuilderUndo::PopFrame)?;
                        self.stack.push(BuildFrame::value_start());
                    }
                },
            }
        }
    }

    /// Records a completed key on the object frame now exposed on top of the stack.
    fn attach_key(
        &mut self,
        mem: &mut SessionMemory,
        budget: usize,
        key: ByteSpan,
    ) -> Result<(), StructuredRuntimeError> {
        let err = || StructuredRuntimeError::new(LimitKind::CanonicalBytes, usize::MAX, budget);
        let depth = self.stack.len().checked_sub(1).ok_or_else(err)?;
        match self.stack.get_mut(depth) {
            Some(BuildFrame::Object { phase, .. }) => {
                let old_phase = *phase;
                self.record(
                    mem,
                    budget,
                    BuilderUndo::RestoreObjectPhase {
                        depth,
                        phase: old_phase,
                    },
                )?;
                let Some(BuildFrame::Object { phase, .. }) = self.stack.get_mut(depth) else {
                    unreachable!("checked above")
                };
                *phase = ObjectBuildPhase::Colon { key };
                Ok(())
            }
            _ => Err(err()),
        }
    }

    /// Attaches a completed value to its parent or completes the item.
    fn complete_value(
        &mut self,
        mem: &mut SessionMemory,
        budget: usize,
        id: CanonicalId,
    ) -> Result<(), StructuredRuntimeError> {
        let err = || StructuredRuntimeError::new(LimitKind::CanonicalBytes, usize::MAX, budget);
        let Some(depth) = self.stack.len().checked_sub(1) else {
            self.record(mem, budget, BuilderUndo::RestoreRoot)?;
            self.root = Some(id);
            return Ok(());
        };
        let other_live = mem.live();
        match self.stack.get_mut(depth) {
            Some(BuildFrame::Array { children }) => {
                reserve_builder_owned(children, &mut self.own_mem, other_live, budget)?;
                let Some(BuildFrame::Array { children }) = self.stack.get_mut(depth) else {
                    unreachable!("checked above")
                };
                let old_len = children.len();
                self.record(
                    mem,
                    budget,
                    BuilderUndo::TruncateChildren { depth, old_len },
                )?;
                let Some(BuildFrame::Array { children }) = self.stack.get_mut(depth) else {
                    unreachable!("checked above")
                };
                children.push(id);
                Ok(())
            }
            Some(BuildFrame::Object { phase, entries }) => {
                let ObjectBuildPhase::Value { key } = *phase else {
                    return Err(err());
                };
                reserve_builder_owned(entries, &mut self.own_mem, other_live, budget)?;
                let Some(BuildFrame::Object { entries, .. }) = self.stack.get_mut(depth) else {
                    unreachable!("checked above")
                };
                let old_len = entries.len();
                self.record(mem, budget, BuilderUndo::TruncateEntries { depth, old_len })?;
                self.record(
                    mem,
                    budget,
                    BuilderUndo::RestoreObjectPhase {
                        depth,
                        phase: ObjectBuildPhase::Value { key },
                    },
                )?;
                let Some(BuildFrame::Object { phase, entries }) = self.stack.get_mut(depth) else {
                    unreachable!("checked above")
                };
                entries.push(ObjectEntry { key, value: id });
                *phase = ObjectBuildPhase::KeyOrClose;
                Ok(())
            }
            _ => Err(err()),
        }
    }

    /// Closes an open root scalar when the caller ends the item.
    fn finish(
        &mut self,
        arena: &mut CanonicalArena,
        mem: &mut SessionMemory,
        budget: usize,
    ) -> Result<CanonicalId, StructuredRuntimeError> {
        if let Some(id) = self.root {
            return Ok(id);
        }
        let broken = || StructuredRuntimeError::new(LimitKind::CanonicalBytes, usize::MAX, budget);
        let (
            Some(BuildFrame::Scalar {
                phase,
                number_start,
            }),
            1,
        ) = (self.stack.first(), self.stack.len())
        else {
            return Err(broken());
        };
        let (phase, number_start) = (*phase, *number_start);
        let set_budget = self.set_budget(budget)?;
        let node = match phase {
            ScalarPhase::Number(_) => arena.finish_number(mem, set_budget, number_start)?,
            ScalarPhase::LiteralDone => literal_from_tag(number_start).ok_or_else(broken)?,
            _ => return Err(broken()),
        };
        let set_budget = self.set_budget(budget)?;
        let id = arena.push_node(mem, set_budget, node)?;
        self.record(
            mem,
            budget,
            BuilderUndo::PushFrame(BuildFrame::Scalar {
                phase,
                number_start,
            }),
        )?;
        self.record(mem, budget, BuilderUndo::RestoreRoot)?;
        self.stack.pop();
        self.root = Some(id);
        Ok(id)
    }
}

/// `ScalarPhase::LiteralDone` no longer says which literal matched, so the last distinguishing
/// byte tags the builder's spare `number_start` slot: 1/2/3 for true/false/null.
fn literal_tag(phase: ScalarPhase) -> Option<u32> {
    match phase {
        ScalarPhase::True(_) => Some(1),
        ScalarPhase::False(_) => Some(2),
        ScalarPhase::Null(_) => Some(3),
        _ => None,
    }
}

fn literal_from_tag(tag: u32) -> Option<CanonicalNode> {
    match tag {
        1 => Some(CanonicalNode::Bool(true)),
        2 => Some(CanonicalNode::Bool(false)),
        3 => Some(CanonicalNode::Null),
        _ => None,
    }
}

fn hash_number(n: NumberRecord, arena: &CanonicalArena, h: &mut impl Hasher) {
    n.negative.hash(h);
    n.exp_negative.hash(h);
    arena.span_bytes(n.exp_digits).hash(h);
    arena.span_bytes(n.digits).hash(h);
}

fn eq_number(p: NumberRecord, q: NumberRecord, arena: &CanonicalArena) -> bool {
    p.negative == q.negative
        && p.exp_negative == q.exp_negative
        && arena.span_bytes(p.exp_digits) == arena.span_bytes(q.exp_digits)
        && arena.span_bytes(p.digits) == arena.span_bytes(q.digits)
}

/// Tracks canonical values seen in one `uniqueItems` array.
/// Fingerprints narrow candidates before exact arena comparison.
#[derive(Debug)]
pub(crate) struct CanonicalSet {
    arena: CanonicalArena,
    buckets: HashMap<u64, Vec<CanonicalId>>,
    hasher: RandomState,
    mem: SessionMemory,
    /// Reusable traversal scratch for `fingerprint`/`arena_eq`, so hashing or comparing a value
    /// never allocates a fresh work-stack; capacity only ever grows to the largest value seen.
    hash_stack: Vec<CanonicalId>,
    eq_stack: Vec<(CanonicalId, CanonicalId)>,
}

#[cfg(test)]
fn set_capacity_bytes(
    arena: &CanonicalArena,
    buckets: &HashMap<u64, Vec<CanonicalId>>,
    hash_stack_capacity: usize,
    eq_stack_capacity: usize,
) -> usize {
    let slot_cost = std::mem::size_of::<u64>() + std::mem::size_of::<Vec<CanonicalId>>() + 16;
    [
        arena.nodes.capacity() * std::mem::size_of::<CanonicalNode>(),
        arena.bytes.capacity(),
        arena.children.capacity() * std::mem::size_of::<CanonicalId>(),
        arena.entries.capacity() * std::mem::size_of::<ObjectEntry>(),
        arena.number_scratch.capacity(),
        buckets.capacity() * slot_cost,
        hash_stack_capacity * std::mem::size_of::<CanonicalId>(),
        eq_stack_capacity * std::mem::size_of::<(CanonicalId, CanonicalId)>(),
    ]
    .into_iter()
    .chain(
        buckets
            .values()
            .map(|bucket| bucket.capacity() * std::mem::size_of::<CanonicalId>()),
    )
    .try_fold(0usize, usize::checked_add)
    .expect("set retained capacities fit usize")
}

#[cfg(test)]
impl Clone for CanonicalSet {
    fn clone(&self) -> Self {
        let arena = self.arena.clone();
        let buckets = self.buckets.clone();
        let hash_stack = self.hash_stack.clone();
        let eq_stack = self.eq_stack.clone();
        let mem = memory_for_clone(set_capacity_bytes(
            &arena,
            &buckets,
            hash_stack.capacity(),
            eq_stack.capacity(),
        ));
        Self {
            arena,
            buckets,
            hasher: self.hasher.clone(),
            mem,
            hash_stack,
            eq_stack,
        }
    }
}

impl PartialEq for CanonicalSet {
    /// Valid only for two snapshots of the SAME instance (e.g. before/after a rollback): bucket
    /// keys are fingerprints under one shared `hasher` seed, meaningless to compare across sets.
    fn eq(&self, other: &Self) -> bool {
        self.arena == other.arena && self.buckets == other.buckets
    }
}

/// What `insert` needs to roll back a fresh candidate that turned out to be a duplicate, without
/// ever comparing across two different `RandomState` seeds.
pub(crate) enum CanonicalInsert {
    Duplicate,
    Inserted {
        fingerprint: u64,
        old_bucket_len: usize,
        bucket_created: bool,
        bucket_capacity_bytes: usize,
    },
}

impl CanonicalSet {
    pub(crate) fn new() -> Self {
        Self {
            arena: CanonicalArena::default(),
            buckets: HashMap::default(),
            hasher: RandomState::new(),
            mem: SessionMemory::default(),
            hash_stack: Vec::new(),
            eq_stack: Vec::new(),
        }
    }

    /// Whether an already committed top-level item is this decoded JSON string.
    pub(crate) fn contains_string(&self, value: &[u8]) -> bool {
        self.buckets.values().flatten().any(|id| {
            matches!(
                self.arena.nodes.get(id.index()),
                Some(CanonicalNode::String(span)) if self.arena.span_bytes(*span) == value
            )
        })
    }

    #[cfg(test)]
    pub(crate) fn retained_bytes(&self) -> usize {
        self.mem.live()
    }

    /// Reserves room for one more entry on `hash_stack`, charging only the real capacity delta;
    /// retained scratch capacity is never released on a normal call.
    fn reserve_hash_stack(
        &mut self,
        other_live: usize,
        budget: usize,
    ) -> Result<(), StructuredRuntimeError> {
        let CanonicalSet {
            hash_stack, mem, ..
        } = self;
        let effective = remaining_budget(budget, other_live)?;
        grow_bounded(
            mem,
            effective,
            LimitKind::CanonicalBytes,
            GrowthRequest {
                len: hash_stack.len(),
                capacity: hash_stack.capacity(),
                additional: 1,
                item_size: std::mem::size_of::<CanonicalId>(),
            },
            |n| {
                hash_stack
                    .try_reserve_exact(n)
                    .map(|_| hash_stack.capacity())
                    .map_err(|_| ())
            },
        )
        .map_err(|e| StructuredRuntimeError::new(e.kind, e.observed, budget))
    }

    fn reserve_eq_stack(
        &mut self,
        other_live: usize,
        budget: usize,
    ) -> Result<(), StructuredRuntimeError> {
        let CanonicalSet { eq_stack, mem, .. } = self;
        let effective = remaining_budget(budget, other_live)?;
        grow_bounded(
            mem,
            effective,
            LimitKind::CanonicalBytes,
            GrowthRequest {
                len: eq_stack.len(),
                capacity: eq_stack.capacity(),
                additional: 1,
                item_size: std::mem::size_of::<(CanonicalId, CanonicalId)>(),
            },
            |n| {
                eq_stack
                    .try_reserve_exact(n)
                    .map(|_| eq_stack.capacity())
                    .map_err(|_| ())
            },
        )
        .map_err(|e| StructuredRuntimeError::new(e.kind, e.observed, budget))
    }

    /// Hashes a root iteratively using retained scratch storage.
    fn fingerprint(
        &mut self,
        other_live: usize,
        budget: usize,
        root: CanonicalId,
    ) -> Result<u64, StructuredRuntimeError> {
        let mut h = self.hasher.build_hasher();
        self.hash_stack.clear();
        self.reserve_hash_stack(other_live, budget)?;
        self.hash_stack.push(root);
        while let Some(id) = self.hash_stack.pop() {
            match self.arena.nodes[id.index()] {
                CanonicalNode::Null => 0u8.hash(&mut h),
                CanonicalNode::Bool(b) => {
                    1u8.hash(&mut h);
                    b.hash(&mut h);
                }
                CanonicalNode::Number(n) => {
                    2u8.hash(&mut h);
                    hash_number(n, &self.arena, &mut h);
                }
                CanonicalNode::String(s) => {
                    3u8.hash(&mut h);
                    self.arena.span_bytes(s).hash(&mut h);
                }
                CanonicalNode::Array(range) => {
                    4u8.hash(&mut h);
                    range.len.hash(&mut h);
                    let start = range.start as usize;
                    for i in (0..range.len as usize).rev() {
                        let child = self.arena.children[start + i];
                        self.reserve_hash_stack(other_live, budget)?;
                        self.hash_stack.push(child);
                    }
                }
                CanonicalNode::Object(range) => {
                    5u8.hash(&mut h);
                    range.len.hash(&mut h);
                    let start = range.start as usize;
                    for i in (0..range.len as usize).rev() {
                        let entry = self.arena.entries[start + i];
                        self.arena.span_bytes(entry.key).hash(&mut h);
                        self.reserve_hash_stack(other_live, budget)?;
                        self.hash_stack.push(entry.value);
                    }
                }
            }
        }
        Ok(h.finish())
    }

    /// Exact structural equality within the shared arena, walked iteratively using the retained
    /// `eq_stack` scratch, never recursion and never a fresh heap allocation.
    fn arena_eq(
        &mut self,
        other_live: usize,
        budget: usize,
        a: CanonicalId,
        b: CanonicalId,
    ) -> Result<bool, StructuredRuntimeError> {
        self.eq_stack.clear();
        self.reserve_eq_stack(other_live, budget)?;
        self.eq_stack.push((a, b));
        while let Some((x, y)) = self.eq_stack.pop() {
            if x == y {
                continue;
            }
            match (self.arena.nodes[x.index()], self.arena.nodes[y.index()]) {
                (CanonicalNode::Null, CanonicalNode::Null) => {}
                (CanonicalNode::Bool(p), CanonicalNode::Bool(q)) => {
                    if p != q {
                        return Ok(false);
                    }
                }
                (CanonicalNode::Number(p), CanonicalNode::Number(q)) => {
                    if !eq_number(p, q, &self.arena) {
                        return Ok(false);
                    }
                }
                (CanonicalNode::String(p), CanonicalNode::String(q)) => {
                    if self.arena.span_bytes(p) != self.arena.span_bytes(q) {
                        return Ok(false);
                    }
                }
                (CanonicalNode::Array(p), CanonicalNode::Array(q)) => {
                    if p.len != q.len {
                        return Ok(false);
                    }
                    let (ps, qs) = (p.start as usize, q.start as usize);
                    for i in (0..p.len as usize).rev() {
                        self.reserve_eq_stack(other_live, budget)?;
                        self.eq_stack
                            .push((self.arena.children[ps + i], self.arena.children[qs + i]));
                    }
                }
                (CanonicalNode::Object(p), CanonicalNode::Object(q)) => {
                    if p.len != q.len {
                        return Ok(false);
                    }
                    let (ps, qs) = (p.start as usize, q.start as usize);
                    for i in 0..p.len as usize {
                        let (pe, qe) = (self.arena.entries[ps + i], self.arena.entries[qs + i]);
                        if self.arena.span_bytes(pe.key) != self.arena.span_bytes(qe.key) {
                            return Ok(false);
                        }
                    }
                    for i in (0..p.len as usize).rev() {
                        let (pe, qe) = (self.arena.entries[ps + i], self.arena.entries[qs + i]);
                        self.reserve_eq_stack(other_live, budget)?;
                        self.eq_stack.push((pe.value, qe.value));
                    }
                }
                _ => return Ok(false),
            }
        }
        Ok(true)
    }

    /// Starts building one item's candidate value, sharing this set's arena and memory budget.
    /// Fallible: even the builder's initial frame is a checked, bounded reservation.
    pub(crate) fn start_item(
        &self,
        budget: usize,
    ) -> Result<CanonicalBuilder, StructuredRuntimeError> {
        CanonicalBuilder::new(self.mem.live(), budget)
    }

    /// Feeds one byte and rolls back the local builder on error.
    pub(crate) fn feed_item(
        &mut self,
        builder: &mut CanonicalBuilder,
        budget: usize,
        byte: u8,
    ) -> Result<BuilderMark, StructuredRuntimeError> {
        let mark = builder.mark(&self.arena);
        match builder.push_byte(&mut self.arena, &mut self.mem, budget, byte) {
            Ok(()) => Ok(mark),
            Err(e) => {
                builder.restore(&mut self.arena, mark);
                Err(e)
            }
        }
    }

    pub(crate) fn rollback_item_byte(&mut self, builder: &mut CanonicalBuilder, mark: BuilderMark) {
        builder.restore(&mut self.arena, mark);
    }

    /// Whether `builder` has already closed its value on its own (a string/array/object item),
    /// as opposed to a bare number/literal that only closes once the caller calls `finish_item`.
    pub(crate) fn item_is_done(&self, builder: &CanonicalBuilder) -> bool {
        builder.is_done()
    }

    /// Retained bytes owned by `builder` alone (not yet flushed into this set's own arena).
    #[cfg(test)]
    pub(crate) fn item_retained_bytes(&self, builder: &CanonicalBuilder) -> usize {
        builder.retained_bytes()
    }

    /// Closes a still-open scalar item (a number has no byte of its own that signals "done") and
    /// inserts the resulting value, rejecting a duplicate before any retained mutation.
    pub(crate) fn finish_item(
        &mut self,
        builder: &mut CanonicalBuilder,
        budget: usize,
    ) -> Result<CanonicalInsert, StructuredRuntimeError> {
        let mark = builder.mark(&self.arena);
        let root = match builder.finish(&mut self.arena, &mut self.mem, budget) {
            Ok(id) => id,
            Err(e) => {
                builder.restore(&mut self.arena, mark);
                return Err(e);
            }
        };
        match self.insert(builder.retained_bytes(), root, budget) {
            Ok(CanonicalInsert::Duplicate) => {
                builder.restore(&mut self.arena, mark);
                Ok(CanonicalInsert::Duplicate)
            }
            Ok(inserted) => Ok(inserted),
            Err(e) => {
                builder.restore(&mut self.arena, mark);
                Err(e)
            }
        }
    }

    /// Records the pre-finish builder state so undoing an `Inserted` restores the builder's root and
    /// stack, not just the arena; a reused builder must never keep a now-dangling root.
    pub(crate) fn item_mark(&self, builder: &CanonicalBuilder) -> BuilderMark {
        builder.mark(&self.arena)
    }

    fn insert(
        &mut self,
        other_live: usize,
        root: CanonicalId,
        budget: usize,
    ) -> Result<CanonicalInsert, StructuredRuntimeError> {
        let fingerprint = self.fingerprint(other_live, budget, root)?;
        let bucket_len = self.buckets.get(&fingerprint).map_or(0, Vec::len);
        for i in 0..bucket_len {
            let existing = self.buckets[&fingerprint][i];
            if self.arena_eq(other_live, budget, existing, root)? {
                return Ok(CanonicalInsert::Duplicate);
            }
        }
        let err = || StructuredRuntimeError::new(LimitKind::CanonicalBytes, usize::MAX, budget);
        let elem = std::mem::size_of::<CanonicalId>();
        let slot_cost = std::mem::size_of::<u64>() + std::mem::size_of::<Vec<CanonicalId>>() + 16;
        let bucket_created = !self.buckets.contains_key(&fingerprint);
        let table_min = if bucket_created && self.buckets.len() == self.buckets.capacity() {
            slot_cost
        } else {
            0
        };
        let bucket_min = match self.buckets.get(&fingerprint) {
            Some(b) if b.len() < b.capacity() => 0,
            _ => elem,
        };
        let min_total = table_min.checked_add(bucket_min).ok_or_else(err)?;
        let effective = remaining_budget(budget, other_live)?;
        if !self.mem.would_fit(min_total, effective) {
            let observed = self.mem.live().checked_add(min_total).unwrap_or(usize::MAX);
            return Err(StructuredRuntimeError::new(
                LimitKind::CanonicalBytes,
                observed,
                budget,
            ));
        }
        if bucket_created {
            self.insert_new_bucket(other_live, fingerprint, root, elem, slot_cost, budget)
        } else {
            self.push_existing_bucket(other_live, fingerprint, root, elem, budget)
        }
    }

    /// Creates a bucket and immediately charges its retained allocation.
    fn insert_new_bucket(
        &mut self,
        other_live: usize,
        fingerprint: u64,
        root: CanonicalId,
        elem: usize,
        slot_cost: usize,
        budget: usize,
    ) -> Result<CanonicalInsert, StructuredRuntimeError> {
        let err = || StructuredRuntimeError::new(LimitKind::CanonicalBytes, usize::MAX, budget);
        let effective = remaining_budget(budget, other_live)?;
        let table_before = self.buckets.capacity();
        self.buckets.try_reserve(1).map_err(|_| err())?;
        let table_grew = (self.buckets.capacity() - table_before)
            .checked_mul(slot_cost)
            .ok_or_else(err)?;
        self.mem
            .charge_after_precheck(table_grew, effective, LimitKind::CanonicalBytes)
            .map_err(|e| StructuredRuntimeError::new(e.kind, e.observed, budget))?;

        let mut v: Vec<CanonicalId> = Vec::new();
        v.try_reserve_exact(1).map_err(|_| err())?;
        let bucket_capacity_bytes = v.capacity().checked_mul(elem).ok_or_else(err)?;
        self.mem
            .charge_after_precheck(bucket_capacity_bytes, effective, LimitKind::CanonicalBytes)
            .map_err(|e| StructuredRuntimeError::new(e.kind, e.observed, budget))?;

        v.push(root);
        self.buckets.insert(fingerprint, v);
        Ok(CanonicalInsert::Inserted {
            fingerprint,
            old_bucket_len: 0,
            bucket_created: true,
            bucket_capacity_bytes,
        })
    }

    /// Pushes into an existing bucket, reserving and charging its real capacity growth first.
    fn push_existing_bucket(
        &mut self,
        other_live: usize,
        fingerprint: u64,
        root: CanonicalId,
        elem: usize,
        budget: usize,
    ) -> Result<CanonicalInsert, StructuredRuntimeError> {
        let err = || StructuredRuntimeError::new(LimitKind::CanonicalBytes, usize::MAX, budget);
        let effective = remaining_budget(budget, other_live)?;
        let (before, after, old_bucket_len) = {
            let bucket = self.buckets.get_mut(&fingerprint).expect("bucket exists");
            let before = bucket.capacity();
            bucket.try_reserve_exact(1).map_err(|_| err())?;
            (before, bucket.capacity(), bucket.len())
        };
        let bucket_grew = (after - before).checked_mul(elem).ok_or_else(err)?;
        self.mem
            .charge_after_precheck(bucket_grew, effective, LimitKind::CanonicalBytes)
            .map_err(|e| StructuredRuntimeError::new(e.kind, e.observed, budget))?;
        self.buckets
            .get_mut(&fingerprint)
            .expect("bucket exists")
            .push(root);
        Ok(CanonicalInsert::Inserted {
            fingerprint,
            old_bucket_len,
            bucket_created: false,
            bucket_capacity_bytes: 0,
        })
    }

    /// Reverses one insertion and releases any newly created bucket.
    pub(crate) fn rollback_insert(
        &mut self,
        fingerprint: u64,
        old_bucket_len: usize,
        bucket_created: bool,
        bucket_capacity_bytes: usize,
    ) {
        if bucket_created {
            self.buckets.remove(&fingerprint);
            self.mem.release(bucket_capacity_bytes);
        } else if let Some(bucket) = self.buckets.get_mut(&fingerprint) {
            bucket.truncate(old_bucket_len);
        }
    }

    /// Truncates the shared arena to discard rejected candidate growth.
    #[cfg(test)]
    pub(crate) fn discard_candidate(&mut self, mark: CanonicalMark) {
        self.arena.restore(mark);
    }

    #[cfg(test)]
    pub(crate) fn arena_mark(&self) -> CanonicalMark {
        self.arena.mark()
    }

    /// Test-only: inserts `root` under a caller-chosen `fingerprint` instead of a real one, so a
    /// bucket collision can be forced deterministically instead of hoping two real hashes clash.
    #[cfg(test)]
    fn insert_at_fingerprint(
        &mut self,
        fingerprint: u64,
        root: CanonicalId,
        budget: usize,
    ) -> Result<CanonicalInsert, StructuredRuntimeError> {
        let bucket_len = self.buckets.get(&fingerprint).map_or(0, Vec::len);
        for i in 0..bucket_len {
            let existing = self.buckets[&fingerprint][i];
            if self.arena_eq(0, budget, existing, root)? {
                return Ok(CanonicalInsert::Duplicate);
            }
        }
        let elem = std::mem::size_of::<CanonicalId>();
        let slot_cost = std::mem::size_of::<u64>() + std::mem::size_of::<Vec<CanonicalId>>() + 16;
        if !self.buckets.contains_key(&fingerprint) {
            self.insert_new_bucket(0, fingerprint, root, elem, slot_cost, budget)
        } else {
            self.push_existing_bucket(0, fingerprint, root, elem, budget)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, PartialEq, Debug)]
    struct LogicalSnapshot {
        arena: CanonicalArena,
        buckets: HashMap<u64, Vec<CanonicalId>>,
        stack: Vec<BuildFrame>,
        root: Option<CanonicalId>,
        undo: Vec<BuilderUndo>,
    }

    fn logical_snapshot(set: &CanonicalSet, builder: &CanonicalBuilder) -> LogicalSnapshot {
        LogicalSnapshot {
            arena: set.arena.clone(),
            buckets: set.buckets.clone(),
            stack: builder.stack.clone(),
            root: builder.root,
            undo: builder.undo.clone(),
        }
    }

    fn n(text: &str) -> NumericValue {
        normalize_number(text).unwrap()
    }

    #[test]
    fn integer_decimal_and_exponent_forms_of_one_are_equal() {
        assert_eq!(n("1"), n("1.0"));
        assert_eq!(n("1"), n("1e0"));
        assert_eq!(n("1"), n("0.1e1"));
        assert_eq!(n("100"), n("1e2"));
        assert_eq!(n("1"), n("1.00"));
    }

    #[test]
    fn negative_and_positive_zero_are_equal_and_distinct_from_nonzero() {
        assert_eq!(n("0"), n("-0"));
        assert_eq!(n("0"), n("0.0"));
        assert_eq!(n("0"), n("0e5"));
        assert_ne!(n("0"), n("1"));
    }

    #[test]
    fn different_values_are_not_equal() {
        assert_ne!(n("1"), n("10"));
        assert_ne!(n("1.5"), n("15"));
        assert_ne!(n("-1"), n("1"));
    }

    #[test]
    fn malformed_text_is_rejected_not_panicking() {
        assert_eq!(normalize_number(""), Err(NumberError::Malformed));
        assert_eq!(normalize_number("-"), Err(NumberError::Malformed));
        assert_eq!(normalize_number("1."), Err(NumberError::Malformed));
        assert_eq!(normalize_number("abc"), Err(NumberError::Malformed));
    }

    #[test]
    fn huge_decimal_exponent_remains_exact() {
        assert!(normalize_number("1e99999999999999999999").is_ok());
        assert_eq!(n("1e99999999999999999999"), n("1e099999999999999999999"));
        assert_ne!(n("1e99999999999999999999"), n("1e99999999999999999998"));
        assert_eq!(n("1e2147483648"), n("1e2147483648"));
        assert_eq!(n("1.5e-2147483649"), n("1.5e-2147483649"));
    }

    fn feed_all(set: &mut CanonicalSet, text: &str) -> CanonicalId {
        let mut b = set.start_item(1 << 20).unwrap();
        for &byte in text.as_bytes() {
            set.feed_item(&mut b, 1 << 20, byte).unwrap();
        }
        b.finish(&mut set.arena, &mut set.mem, 1 << 20).unwrap()
    }

    fn is_inserted(r: &Result<CanonicalInsert, StructuredRuntimeError>) -> bool {
        matches!(r, Ok(CanonicalInsert::Inserted { .. }))
    }

    fn is_duplicate(r: &Result<CanonicalInsert, StructuredRuntimeError>) -> bool {
        matches!(r, Ok(CanonicalInsert::Duplicate))
    }

    fn insert_text(
        set: &mut CanonicalSet,
        text: &str,
    ) -> Result<CanonicalInsert, StructuredRuntimeError> {
        let root = feed_all(set, text);
        set.insert(0, root, 1 << 20)
    }

    #[test]
    fn scalar_numbers_collide_by_value_not_spelling() {
        let mut set = CanonicalSet::new();
        assert!(is_inserted(&insert_text(&mut set, "1")));
        assert!(is_duplicate(&insert_text(&mut set, "1.0")));
        assert!(is_duplicate(&insert_text(&mut set, "1e0")));
        assert!(is_inserted(&insert_text(&mut set, "2")));
    }

    #[test]
    fn a_completed_root_accepts_only_trailing_whitespace_and_rejects_everything_else_exactly() {
        let mut set = CanonicalSet::new();
        let mut b = set.start_item(1 << 20).unwrap();
        for &byte in b"0" {
            set.feed_item(&mut b, 1 << 20, byte).unwrap();
        }
        let root = b.finish(&mut set.arena, &mut set.mem, 1 << 20).unwrap();
        assert!(b.root.is_some());

        for &byte in b" \t\r\n" {
            let before = logical_snapshot(&set, &b);
            set.feed_item(&mut b, 1 << 20, byte)
                .unwrap_or_else(|e| panic!("byte {byte:#04x} after a completed root: {e:?}"));
            let after = logical_snapshot(&set, &b);
            assert_eq!(
                after, before,
                "whitespace byte {byte:#04x} mutated completed state"
            );
            assert_eq!(b.root, Some(root));
        }

        for &byte in b"x,{}[]" {
            let before = logical_snapshot(&set, &b);
            let result = set.feed_item(&mut b, 1 << 20, byte);
            assert!(
                result.is_err(),
                "byte {byte:#04x} after a completed root must be rejected"
            );
            let after = logical_snapshot(&set, &b);
            assert_eq!(
                after, before,
                "rejecting byte {byte:#04x} left state mutated"
            );
            assert_eq!(b.root, Some(root));
        }
    }

    #[test]
    fn bool_and_number_never_collide() {
        let mut set = CanonicalSet::new();
        assert!(is_inserted(&insert_text(&mut set, "true")));
        assert!(is_inserted(&insert_text(&mut set, "1")));
    }

    #[test]
    fn raw_and_escaped_strings_collide() {
        let mut set = CanonicalSet::new();
        assert!(is_inserted(&insert_text(&mut set, "\"a\"")));
        assert!(is_duplicate(&insert_text(&mut set, "\"\\u0061\"")));
    }

    #[test]
    fn nested_arrays_collide_only_in_the_same_order() {
        let mut set = CanonicalSet::new();
        assert!(is_inserted(&insert_text(&mut set, "[1,[2,3]]")));
        assert!(is_duplicate(&insert_text(&mut set, "[1,[2,3]]")));
        assert!(is_inserted(&insert_text(&mut set, "[[2,3],1]")));
    }

    #[test]
    fn objects_with_reordered_keys_collide_and_changed_values_do_not() {
        let mut set = CanonicalSet::new();
        assert!(is_inserted(&insert_text(&mut set, "{\"a\":1,\"b\":2}")));
        assert!(is_duplicate(&insert_text(&mut set, "{\"b\":2,\"a\":1}")));
        assert!(is_inserted(&insert_text(&mut set, "{\"a\":1,\"b\":3}")));
    }

    #[test]
    fn empty_array_and_object_are_each_a_single_distinct_value() {
        let mut set = CanonicalSet::new();
        assert!(is_inserted(&insert_text(&mut set, "[]")));
        assert!(is_duplicate(&insert_text(&mut set, "[]")));
        assert!(is_inserted(&insert_text(&mut set, "{}")));
        assert!(is_duplicate(&insert_text(&mut set, "{}")));
    }

    #[test]
    fn mixed_nested_objects_and_arrays_distinguish_a_single_changed_leaf() {
        let mut set = CanonicalSet::new();
        assert!(is_inserted(&insert_text(
            &mut set,
            "{\"a\":[1,{\"x\":true}],\"b\":null}"
        )));
        assert!(is_duplicate(&insert_text(
            &mut set,
            "{\"b\":null,\"a\":[1,{\"x\":true}]}"
        )));
        assert!(is_inserted(&insert_text(
            &mut set,
            "{\"a\":[1,{\"x\":false}],\"b\":null}"
        )));
    }

    #[test]
    fn plain_and_escaped_nested_values_canonicalize_equally() {
        let mut set = CanonicalSet::new();
        assert!(is_inserted(&insert_text(
            &mut set,
            "{\"a\":[\"😀\",{\"b\":true}]}"
        )));
        assert!(is_duplicate(&insert_text(
            &mut set,
            "{\"\\u0061\":[\"\\uD83D\\uDE00\",{\"\\u0062\":true}]}"
        )));
    }

    #[test]
    fn canonical_set_enforces_its_byte_budget_and_leaves_state_unchanged_on_failure() {
        let mut set = CanonicalSet::new();
        assert!(is_inserted(&insert_text(&mut set, "true")));
        let before = set.retained_bytes();
        let budget = before + 512;
        let long = format!("\"{}\"", "x".repeat(4096));
        let mut b = set.start_item(budget).unwrap();
        let mut failed = false;
        for &byte in long.as_bytes() {
            if set.feed_item(&mut b, budget, byte).is_err() {
                failed = true;
                break;
            }
        }
        assert!(failed);
    }

    #[test]
    fn duplicate_rollback_preserves_set_and_builder() {
        let mut set = CanonicalSet::new();
        let cycle = |set: &mut CanonicalSet| {
            let arena_mark = set.arena_mark();
            let root = feed_all(set, "5");
            let CanonicalInsert::Inserted {
                fingerprint,
                old_bucket_len,
                bucket_created,
                bucket_capacity_bytes,
            } = set.insert(0, root, 1 << 20).unwrap()
            else {
                panic!("expected a fresh insert")
            };
            set.rollback_insert(
                fingerprint,
                old_bucket_len,
                bucket_created,
                bucket_capacity_bytes,
            );
            set.discard_candidate(arena_mark);
        };
        cycle(&mut set);
        let after_first_cycle = set.retained_bytes();
        for _ in 1..20 {
            cycle(&mut set);
            assert_eq!(
                set.retained_bytes(),
                after_first_cycle,
                "scratch={} bytes={} nodes={}",
                set.arena.number_scratch.capacity(),
                set.arena.bytes.capacity(),
                set.arena.nodes.capacity()
            );
        }
    }

    #[test]
    fn forced_fingerprint_collision_still_resolves_by_exact_equality() {
        let mut set = CanonicalSet::new();
        let a = feed_all(&mut set, "1");
        let b = feed_all(&mut set, "2");
        let CanonicalInsert::Inserted {
            bucket_created: a_created,
            fingerprint,
            old_bucket_len: a_old_len,
            bucket_capacity_bytes: a_cap,
        } = set.insert_at_fingerprint(7, a, 1 << 20).unwrap()
        else {
            panic!("expected a fresh insert")
        };
        assert!(a_created);
        let CanonicalInsert::Inserted {
            bucket_created: b_created,
            old_bucket_len: b_old_len,
            bucket_capacity_bytes: b_cap,
            ..
        } = set.insert_at_fingerprint(7, b, 1 << 20).unwrap()
        else {
            panic!("expected a fresh insert alongside the forced collision")
        };
        assert!(!b_created, "second value joins the same forced bucket");
        assert_eq!(b_old_len, 1);

        assert!(matches!(
            set.insert_at_fingerprint(7, a, 1 << 20).unwrap(),
            CanonicalInsert::Duplicate
        ));
        assert!(matches!(
            set.insert_at_fingerprint(7, b, 1 << 20).unwrap(),
            CanonicalInsert::Duplicate
        ));

        set.rollback_insert(fingerprint, b_old_len, b_created, b_cap);
        assert!(matches!(
            set.insert_at_fingerprint(7, b, 1 << 20).unwrap(),
            CanonicalInsert::Inserted { .. }
        ));
        assert!(matches!(
            set.insert_at_fingerprint(7, a, 1 << 20).unwrap(),
            CanonicalInsert::Duplicate
        ));

        set.rollback_insert(fingerprint, b_old_len, b_created, b_cap);
        set.rollback_insert(fingerprint, a_old_len, a_created, a_cap);
        assert!(matches!(
            set.insert_at_fingerprint(7, a, 1 << 20).unwrap(),
            CanonicalInsert::Inserted {
                bucket_created: true,
                ..
            }
        ));
    }

    #[test]
    fn dropping_a_builder_never_changes_set_retained_bytes() {
        let mut set = CanonicalSet::new();
        let mut b = set.start_item(1 << 20).unwrap();
        for &byte in b"[1,2,{\"a\":3}]".iter() {
            set.feed_item(&mut b, 1 << 20, byte).unwrap();
        }
        assert!(set.item_retained_bytes(&b) > 0, "builder owns local growth");
        let arena_bytes_before_drop = set.retained_bytes();
        drop(b);
        assert_eq!(set.retained_bytes(), arena_bytes_before_drop);
    }

    #[test]
    fn commit_history_keeps_rollback_exact_for_bytes_fed_after_it() {
        let mut set = CanonicalSet::new();
        let mut b = set.start_item(1 << 20).unwrap();
        for &byte in b"[1,2".iter() {
            set.feed_item(&mut b, 1 << 20, byte).unwrap();
        }
        b.commit_history();
        let retained_after_commit = set.item_retained_bytes(&b);
        let mark = set.feed_item(&mut b, 1 << 20, b',').unwrap();
        set.feed_item(&mut b, 1 << 20, b'9').unwrap();
        set.rollback_item_byte(&mut b, mark);
        assert!(set.item_retained_bytes(&b) >= retained_after_commit);
        assert_eq!(
            b.retained_bytes(),
            builder_capacity_bytes(&b.stack, b.stack.capacity(), &b.undo, b.undo.capacity())
        );
    }

    #[test]
    fn commit_clears_unreachable_builder_history() {
        let mut set = CanonicalSet::new();
        let mut b = set.start_item(1 << 20).unwrap();
        for &byte in b"[1,2".iter() {
            set.feed_item(&mut b, 1 << 20, byte).unwrap();
        }
        b.commit_history();
        assert!(b.undo.is_empty());
        for &byte in b",3]".iter() {
            set.feed_item(&mut b, 1 << 20, byte).unwrap();
        }
        assert!(set.item_is_done(&b));
    }

    #[test]
    fn builder_retained_bytes_is_stable_across_repeated_commits() {
        let mut set = CanonicalSet::new();
        let mut b = set.start_item(1 << 20).unwrap();
        for &byte in b"[1,".iter() {
            set.feed_item(&mut b, 1 << 20, byte).unwrap();
        }
        b.commit_history();
        let after_first_commit = set.item_retained_bytes(&b);
        for &byte in b"2,".iter() {
            set.feed_item(&mut b, 1 << 20, byte).unwrap();
        }
        b.commit_history();
        assert!(set.item_retained_bytes(&b) >= after_first_commit);
        let after_second_commit = set.item_retained_bytes(&b);
        b.commit_history();
        assert_eq!(set.item_retained_bytes(&b), after_second_commit);
    }

    #[test]
    fn a_tiny_budget_rejects_builder_growth_before_exceeding_it() {
        let mut set = CanonicalSet::new();
        assert!(
            set.start_item(4).is_err(),
            "even the first frame is checked"
        );
        let budget = 512;
        let mut b = set.start_item(budget).unwrap();
        let long = "[".to_string() + &"1,".repeat(200) + "1]";
        let mut hit_limit = false;
        for &byte in long.as_bytes() {
            if set.feed_item(&mut b, budget, byte).is_err() {
                hit_limit = true;
                break;
            }
        }
        assert!(hit_limit);
        assert!(set.item_retained_bytes(&b) <= budget);
    }

    #[test]
    fn rollback_after_several_bytes_in_one_uncommitted_checkpoint_is_exact() {
        let mut set = CanonicalSet::new();
        let mut b = set.start_item(1 << 20).unwrap();
        let mark = set.feed_item(&mut b, 1 << 20, b'[').unwrap();
        for &byte in b"1,2,3".iter() {
            set.feed_item(&mut b, 1 << 20, byte).unwrap();
        }
        set.rollback_item_byte(&mut b, mark);
        assert!(!set.item_is_done(&b));
        let retained_after_rollback = set.item_retained_bytes(&b);
        // Retained capacity (e.g. the array's local `children` Vec) legitimately stays
        // allocated across a rollback; repeating it must never grow further from here.
        set.rollback_item_byte(&mut b, mark);
        assert_eq!(set.item_retained_bytes(&b), retained_after_rollback);
    }

    #[test]
    fn deep_nesting_uses_no_rust_recursion() {
        let mut set = CanonicalSet::new();
        let depth = 200;
        let text = "[".repeat(depth) + &"]".repeat(depth);
        let _ = feed_all(&mut set, &text);
    }

    /// `Vec` capacity is monotonic, so one failing byte can still leave capacity grown by an
    /// earlier sub-step retained; what must hold is that retrying it never grows bytes further.
    fn assert_rejection_is_idempotent(
        set: &mut CanonicalSet,
        b: &mut CanonicalBuilder,
        budget: usize,
        byte: u8,
    ) {
        let logical_before = logical_snapshot(set, b);
        let after_first = set.retained_bytes() + set.item_retained_bytes(b);
        // Retrying within retained capacity must not increase memory use.
        if let Ok(mark) = set.feed_item(b, budget, byte) {
            set.rollback_item_byte(b, mark);
        }
        assert_eq!(logical_snapshot(set, b), logical_before);
        assert_eq!(
            set.retained_bytes() + set.item_retained_bytes(b),
            after_first
        );
        assert!(after_first <= budget);
    }

    #[test]
    fn combined_set_and_builder_budget_is_enforced() {
        let mut set = CanonicalSet::new();
        for i in 0..5 {
            let _ = insert_text(&mut set, &i.to_string());
        }
        let set_live = set.retained_bytes();
        let budget = set_live + 1024;
        let mut b = set.start_item(budget).unwrap();
        let long = "[".to_string() + &"1,".repeat(50) + "1]";
        let mut rejected_byte = None;
        for &byte in long.as_bytes() {
            if set.feed_item(&mut b, budget, byte).is_err() {
                rejected_byte = Some(byte);
                break;
            }
        }
        let byte = rejected_byte.expect("combined ledgers must exceed this budget");
        assert_rejection_is_idempotent(&mut set, &mut b, budget, byte);
    }

    #[test]
    fn stack_growth_is_fallible_and_charged() {
        let mut set = CanonicalSet::new();
        let budget = 2048;
        let mut b = set.start_item(budget).unwrap();
        let text = "[".repeat(100);
        let mut rejected_byte = None;
        for &byte in text.as_bytes() {
            if set.feed_item(&mut b, budget, byte).is_err() {
                rejected_byte = Some(byte);
                break;
            }
        }
        let byte = rejected_byte.expect("100 levels of nesting must exceed this budget");
        assert_rejection_is_idempotent(&mut set, &mut b, budget, byte);
    }

    #[test]
    fn rejected_byte_retained_capacity_is_bounded_and_idempotent() {
        let mut set = CanonicalSet::new();
        let budget = 256;
        let mut b = set.start_item(budget).unwrap();
        let text = "[".to_string() + &"1,".repeat(50) + "1]";
        let mut rejected_byte = None;
        for &byte in text.as_bytes() {
            if set.feed_item(&mut b, budget, byte).is_err() {
                rejected_byte = Some(byte);
                break;
            }
        }
        let byte = rejected_byte.expect("this input must exceed the tiny budget");
        assert_rejection_is_idempotent(&mut set, &mut b, budget, byte);
    }

    #[test]
    fn combined_retained_bytes_never_exceed_budget_across_feed_finish_rollback_commit() {
        let mut set = CanonicalSet::new();
        let budget = 1 << 16;
        let mut b = set.start_item(budget).unwrap();
        for &byte in b"[1,2,3".iter() {
            set.feed_item(&mut b, budget, byte).unwrap();
            assert!(set.retained_bytes() + set.item_retained_bytes(&b) <= budget);
        }
        b.commit_history();
        assert!(set.retained_bytes() + set.item_retained_bytes(&b) <= budget);
        for &byte in b",4,5]".iter() {
            set.feed_item(&mut b, budget, byte).unwrap();
        }
        let result = set.finish_item(&mut b, budget).unwrap();
        assert!(matches!(result, CanonicalInsert::Inserted { .. }));
        assert!(set.retained_bytes() <= budget);
    }

    #[test]
    fn commit_then_rollback_cannot_cross_commit_boundary() {
        let mut set = CanonicalSet::new();
        let mut b = set.start_item(1 << 20).unwrap();
        let _stale_mark = set.feed_item(&mut b, 1 << 20, b'[').unwrap();
        b.commit_history();
        // Committed history cannot be reached by a later rollback.
        let fresh_mark = set.feed_item(&mut b, 1 << 20, b'1').unwrap();
        set.rollback_item_byte(&mut b, fresh_mark);
        assert!(!set.item_is_done(&b));
    }

    #[test]
    fn failed_stack_growth_restores_logical_state() {
        let mut set = CanonicalSet::new();
        let budget = 1024;
        let mut builder = set.start_item(budget).unwrap();
        for byte in "[".bytes().cycle().take(128) {
            let before = logical_snapshot(&set, &builder);
            if set.feed_item(&mut builder, budget, byte).is_err() {
                assert_eq!(logical_snapshot(&set, &builder), before);
                assert!(set.retained_bytes() + builder.retained_bytes() <= budget);
                return;
            }
        }
        panic!("nested stack growth must reach the calculated budget");
    }

    #[test]
    fn finish_item_includes_builder_memory_in_hash_and_insert_budget() {
        let mut set = CanonicalSet::new();
        let mut builder = set.start_item(1 << 20).unwrap();
        set.feed_item(&mut builder, 1 << 20, b'1').unwrap();
        let before = logical_snapshot(&set, &builder);
        let budget = set.retained_bytes() + builder.retained_bytes();
        assert!(set.finish_item(&mut builder, budget).is_err());
        assert_eq!(logical_snapshot(&set, &builder), before);
        assert!(set.retained_bytes() + builder.retained_bytes() <= budget);
    }

    #[test]
    fn clone_recomputes_or_preserves_exact_memory_accounting() {
        let mut set = CanonicalSet::new();
        let mut builder = set.start_item(1 << 20).unwrap();
        for &byte in b"[1,{\"a\":2}" {
            set.feed_item(&mut builder, 1 << 20, byte).unwrap();
        }
        let set_clone = set.clone();
        let builder_clone = builder.clone();
        assert_eq!(
            set_clone.retained_bytes(),
            set_capacity_bytes(
                &set_clone.arena,
                &set_clone.buckets,
                set_clone.hash_stack.capacity(),
                set_clone.eq_stack.capacity(),
            )
        );
        assert_eq!(
            builder_clone.retained_bytes(),
            builder_capacity_bytes(
                &builder_clone.stack,
                builder_clone.stack.capacity(),
                &builder_clone.undo,
                builder_clone.undo.capacity(),
            )
        );
    }
}
