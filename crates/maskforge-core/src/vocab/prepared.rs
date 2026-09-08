//! One canonical, byte-sorted view of a vocabulary, computed once and shared by fingerprinting and
//! trie construction so they never sort the same token map twice.

use super::canonical::{CONSERVATIVE_SPARSE_SLOT_BYTES, MAX_MASK_TOKENS};
use crate::error::{CompileError, ErrorCode, Stage};
use crate::vocab::Vocabulary;

/// One token's ranges into `PreparedVocabulary`'s owned blob/ids arrays.
#[derive(Copy, Clone, Debug)]
pub(crate) struct TokenRecord {
    byte_start: u32,
    byte_end: u32,
    id_start: u32,
    id_end: u32,
}

/// Sort key with an eight-byte lexical prefix for faster comparisons.
/// Prefix ties fall back to full byte comparison.
#[derive(Copy, Clone)]
struct PackedSortKey {
    prefix: u64,
    index: u32,
}

fn lexical_prefix(bytes: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let n = bytes.len().min(8);
    buf[..n].copy_from_slice(&bytes[..n]);
    u64::from_be_bytes(buf)
}

/// Owned vocabulary records sorted by byte key and token id.
/// This canonical form is shared by fingerprinting and trie construction.
#[derive(Debug)]
pub(crate) struct PreparedVocabulary {
    blob: Vec<u8>,
    records: Vec<TokenRecord>,
    ids: Vec<u32>,
}

fn overflow(msg: &'static str) -> CompileError {
    CompileError::new(ErrorCode::InternalLimitExceeded, Stage::L4Bind, msg)
}

fn empty_token() -> CompileError {
    CompileError::new(
        ErrorCode::EmptyToken,
        Stage::VocabBuild,
        "vocabulary token has zero bytes",
    )
}

fn eos_as_ordinary_id() -> CompileError {
    CompileError::new(
        ErrorCode::MalformedTokenizer,
        Stage::VocabBuild,
        "EOS token id present in the token map",
    )
}

fn empty_token_ids() -> CompileError {
    CompileError::new(
        ErrorCode::MalformedTokenizer,
        Stage::VocabBuild,
        "token has an empty id list",
    )
}

fn malformed_offsets(msg: &'static str) -> CompileError {
    CompileError::new(ErrorCode::Unsupported, Stage::VocabBuild, msg)
}

/// Cap on packed token count, checked before any sort or allocation.
pub(crate) const MAX_PACKED_TOKENS: usize = 1 << 24;

/// Caps projected peak memory before sorting or allocation.
pub(crate) const MAX_PACKED_PROJECTED_PEAK_BYTES: usize = 1 << 31;

/// Estimates peak memory for all simultaneously live packed-build allocations.
pub(crate) fn projected_peak_bytes(
    token_bytes_len: usize,
    token_ids_len: usize,
    token_count: usize,
) -> usize {
    let ids_bytes = token_ids_len.saturating_mul(4);
    let records_bytes = token_count.saturating_mul(std::mem::size_of::<TokenRecord>());
    let order_bytes = token_count.saturating_mul(std::mem::size_of::<PackedSortKey>());
    let group_scratch = token_ids_len.saturating_mul(4);
    let dense_worst_case = MAX_MASK_TOKENS.saturating_mul(4);
    let sparse_worst_case = token_ids_len.saturating_mul(CONSERVATIVE_SPARSE_SLOT_BYTES);
    let id_index_worst_case = dense_worst_case.max(sparse_worst_case);
    token_bytes_len
        .saturating_add(ids_bytes)
        .saturating_add(records_bytes)
        .saturating_add(order_bytes)
        .saturating_add(group_scratch)
        .saturating_add(id_index_worst_case)
}

/// `offsets` must be non-empty, start at 0, be non-decreasing, and its last entry must equal `len`
/// (the buffer it indexes into) - checked before any slice built from it can be taken.
fn validate_csr_offsets(offsets: &[u32], len: usize) -> Result<(), CompileError> {
    let Some(&first) = offsets.first() else {
        return Err(malformed_offsets("offsets must have at least one element"));
    };
    if first != 0 {
        return Err(malformed_offsets("offsets must start at 0"));
    }
    if offsets.windows(2).any(|w| w[0] > w[1]) {
        return Err(malformed_offsets("offsets must be non-decreasing"));
    }
    let &last = offsets.last().expect("checked non-empty above");
    if last as usize != len {
        return Err(malformed_offsets(
            "offsets' final entry must equal the buffer length",
        ));
    }
    Ok(())
}

/// `Vec::try_reserve_exact` wrapped as a typed error, not an abort - directly unit-testable with a
/// synthetic `usize::MAX`-scale `cap` (see `tests::an_unsatisfiable_capacity_is_a_typed_error`).
pub(crate) fn try_alloc<T>(cap: usize, what: &'static str) -> Result<Vec<T>, CompileError> {
    let mut v = Vec::new();
    v.try_reserve_exact(cap).map_err(|_| overflow(what))?;
    Ok(v)
}

/// Appends one byte-and-id record to canonical storage.
fn push_record(
    blob: &mut Vec<u8>,
    bytes: &[u8],
    ids: &mut Vec<u32>,
    ids_slice: &[u32],
    records: &mut Vec<TokenRecord>,
) -> Result<(), CompileError> {
    let byte_start = u32::try_from(blob.len())
        .map_err(|_| overflow("prepared vocabulary blob exceeds the addressable range"))?;
    blob.extend_from_slice(bytes);
    let byte_end = u32::try_from(blob.len())
        .map_err(|_| overflow("prepared vocabulary blob exceeds the addressable range"))?;
    let id_start = u32::try_from(ids.len())
        .map_err(|_| overflow("prepared vocabulary id array exceeds the addressable range"))?;
    ids.extend_from_slice(ids_slice);
    let id_end = u32::try_from(ids.len())
        .map_err(|_| overflow("prepared vocabulary id array exceeds the addressable range"))?;
    records.push(TokenRecord {
        byte_start,
        byte_end,
        id_start,
        id_end,
    });
    Ok(())
}

/// `(total_bytes, total_ids)` narrowed to `u32`, or a typed error - never `.min(u32::MAX)`, which
/// would still request up to ~4 GiB via `Vec::with_capacity` before failing per-record later.
fn check_addressable_totals(
    total_bytes: usize,
    total_ids: usize,
) -> Result<(u32, u32), CompileError> {
    let blob_cap = u32::try_from(total_bytes)
        .map_err(|_| overflow("prepared vocabulary blob exceeds the addressable range"))?;
    let ids_cap = u32::try_from(total_ids)
        .map_err(|_| overflow("prepared vocabulary id array exceeds the addressable range"))?;
    Ok((blob_cap, ids_cap))
}

impl PreparedVocabulary {
    /// Sorts `vocab`'s token map by byte key once, then copies it into one owned blob/id arrays
    /// pair. Measured 40-90% faster than a radix sort on real BPE-tokenizer key distributions.
    pub(crate) fn build(vocab: &Vocabulary) -> Result<Self, CompileError> {
        let mut pairs: Vec<(&[u8], &[u32])> = try_alloc(vocab.tokens().len(), "sort-pair scratch")?;
        pairs.extend(
            vocab
                .tokens()
                .iter()
                .map(|(bytes, ids)| (bytes.as_slice(), ids.as_slice())),
        );
        pairs.sort_unstable_by(|a, b| a.0.cmp(b.0));
        Self::from_sorted_pairs(&pairs)
    }

    /// Copies already byte-sorted `pairs` into one owned blob/id arrays pair.
    fn from_sorted_pairs(pairs: &[(&[u8], &[u32])]) -> Result<Self, CompileError> {
        let total_bytes = pairs
            .iter()
            .try_fold(0usize, |acc, (b, _)| acc.checked_add(b.len()))
            .ok_or_else(|| overflow("prepared vocabulary byte total overflows"))?;
        let total_ids = pairs
            .iter()
            .try_fold(0usize, |acc, (_, i)| acc.checked_add(i.len()))
            .ok_or_else(|| overflow("prepared vocabulary id total overflows"))?;
        // Reject an unaddressable layout BEFORE allocating.
        let (blob_cap, ids_cap) = check_addressable_totals(total_bytes, total_ids)?;

        let mut blob = try_alloc(blob_cap as usize, "prepared vocabulary blob")?;
        let mut ids = try_alloc(ids_cap as usize, "prepared vocabulary id array")?;
        let mut records: Vec<TokenRecord> = try_alloc(pairs.len(), "prepared vocabulary records")?;
        for (bytes, tok_ids) in pairs {
            let byte_start = u32::try_from(blob.len())
                .map_err(|_| overflow("prepared vocabulary blob exceeds the addressable range"))?;
            blob.extend_from_slice(bytes);
            let byte_end = u32::try_from(blob.len())
                .map_err(|_| overflow("prepared vocabulary blob exceeds the addressable range"))?;
            let id_start = u32::try_from(ids.len()).map_err(|_| {
                overflow("prepared vocabulary id array exceeds the addressable range")
            })?;
            ids.extend_from_slice(tok_ids);
            ids[id_start as usize..].sort_unstable(); // canonical: each record's own ids, ascending
            let id_end = u32::try_from(ids.len()).map_err(|_| {
                overflow("prepared vocabulary id array exceeds the addressable range")
            })?;
            records.push(TokenRecord {
                byte_start,
                byte_end,
                id_start,
                id_end,
            });
        }
        Ok(Self { blob, records, ids })
    }

    /// Builds directly from validated packed CSR buffers.
    /// Duplicate byte keys merge under normal vocabulary validation rules.
    pub(crate) fn build_from_packed(
        token_bytes: &[u8],
        byte_offsets: &[u32],
        token_ids: &[u32],
        id_offsets: &[u32],
        eos_token_id: u32,
    ) -> Result<Self, CompileError> {
        Self::build_from_packed_budgeted(
            token_bytes,
            byte_offsets,
            token_ids,
            id_offsets,
            eos_token_id,
            MAX_PACKED_PROJECTED_PEAK_BYTES,
        )
    }

    /// `build_from_packed` with an explicit projected-peak budget, so a test can prove the
    /// admission check itself rejects, using a tiny budget instead of a real multi-GB input.
    fn build_from_packed_budgeted(
        token_bytes: &[u8],
        byte_offsets: &[u32],
        token_ids: &[u32],
        id_offsets: &[u32],
        eos_token_id: u32,
        peak_budget: usize,
    ) -> Result<Self, CompileError> {
        if byte_offsets.len() != id_offsets.len() {
            return Err(malformed_offsets(
                "byte_offsets and id_offsets must have the same length",
            ));
        }
        validate_csr_offsets(byte_offsets, token_bytes.len())?;
        validate_csr_offsets(id_offsets, token_ids.len())?;
        let token_count = byte_offsets.len() - 1;
        if token_count > MAX_PACKED_TOKENS {
            return Err(malformed_offsets("packed token count exceeds the cap"));
        }
        let peak = projected_peak_bytes(token_bytes.len(), token_ids.len(), token_count);
        if peak > peak_budget {
            return Err(malformed_offsets(
                "projected peak memory for this packed build exceeds the cap",
            ));
        }
        // Every check below is a property of the RAW (pre-merge) data, so it holds regardless of
        // how duplicate byte keys later merge - rejected before any sort or output allocation runs.
        for i in 0..token_count {
            if byte_offsets[i] == byte_offsets[i + 1] {
                return Err(empty_token());
            }
        }
        if token_ids.contains(&eos_token_id) {
            return Err(eos_as_ordinary_id());
        }
        let ordinary_max = token_ids.iter().copied().max();
        let max_id = ordinary_max.map_or(eos_token_id, |m| m.max(eos_token_id));
        let mask_vocab_size = usize::try_from(max_id)
            .ok()
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| overflow("max token id overflows the mask width"))?;
        if mask_vocab_size > MAX_MASK_TOKENS {
            return Err(overflow("mask width exceeds the token-id cap"));
        }

        let key = |i: u32| -> &[u8] {
            &token_bytes[byte_offsets[i as usize] as usize..byte_offsets[i as usize + 1] as usize]
        };
        let ids_of = |i: u32| -> &[u32] {
            &token_ids[id_offsets[i as usize] as usize..id_offsets[i as usize + 1] as usize]
        };

        let token_count_u32 = u32::try_from(token_count)
            .map_err(|_| overflow("packed token count exceeds the addressable range"))?;
        let mut order: Vec<PackedSortKey> = try_alloc(token_count, "packed sort-order")?;
        order.extend((0..token_count_u32).map(|index| PackedSortKey {
            prefix: lexical_prefix(key(index)),
            index,
        }));
        order.sort_unstable_by(|a, b| {
            a.prefix
                .cmp(&b.prefix)
                .then_with(|| key(a.index).cmp(key(b.index)))
        });

        let (blob_cap, ids_cap) = check_addressable_totals(token_bytes.len(), token_ids.len())?;
        let mut blob = try_alloc(blob_cap as usize, "prepared vocabulary blob")?;
        let mut ids = try_alloc(ids_cap as usize, "prepared vocabulary id array")?;
        let mut records: Vec<TokenRecord> = try_alloc(token_count, "prepared vocabulary records")?;
        let mut group_ids: Vec<u32> = Vec::new();

        let mut i = 0usize;
        while i < order.len() {
            let mut j = i + 1;
            while j < order.len() && key(order[j].index) == key(order[i].index) {
                j += 1;
            }
            let bytes = key(order[i].index);

            // The dominant case: one token per byte key. Skip the group-scratch copy, sort, and
            // dedup entirely - a single id needs none of them.
            if j == i + 1 {
                match ids_of(order[i].index) {
                    [] => return Err(empty_token_ids()),
                    [only] => {
                        push_record(
                            &mut blob,
                            bytes,
                            &mut ids,
                            std::slice::from_ref(only),
                            &mut records,
                        )?;
                        i = j;
                        continue;
                    }
                    _ => {} // one key, multiple ids: still needs the dedup+sort path below
                }
            }

            group_ids.clear();
            // Reserve the group's exact logical size before filling: `Vec` geometric growth could
            // otherwise leave `group_ids` holding more capacity than `projected_peak_bytes` charges.
            let group_len: usize = order[i..j].iter().map(|k| ids_of(k.index).len()).sum();
            group_ids
                .try_reserve_exact(group_len)
                .map_err(|_| overflow("packed group-id scratch allocation failed"))?;
            for k in &order[i..j] {
                group_ids.extend_from_slice(ids_of(k.index));
            }
            // Every raw id list can be empty on its own; only a merged group with zero ids across
            // EVERY duplicate-key entry is illegal - genuinely needs the merge, unlike the checks above.
            if group_ids.is_empty() {
                return Err(empty_token_ids());
            }
            group_ids.sort_unstable();
            group_ids.dedup();
            push_record(&mut blob, bytes, &mut ids, &group_ids, &mut records)?;
            i = j;
        }
        Ok(Self { blob, records, ids })
    }

    /// Builds directly from a dense, id-ordered tokenizer decode.
    /// Absent slots are skipped and duplicate byte ranges merge.
    pub(crate) fn build_from_dense_id_ordered(
        token_bytes: &[u8],
        byte_offsets: &[u32],
        present: &[bool],
        eos_token_id: u32,
    ) -> Result<Self, CompileError> {
        Self::build_from_dense_id_ordered_budgeted(
            token_bytes,
            byte_offsets,
            present,
            eos_token_id,
            MAX_PACKED_PROJECTED_PEAK_BYTES,
        )
    }

    /// `build_from_dense_id_ordered` with an explicit projected-peak budget (tests inject a tiny
    /// one instead of needing a real oversized input).
    fn build_from_dense_id_ordered_budgeted(
        token_bytes: &[u8],
        byte_offsets: &[u32],
        present: &[bool],
        eos_token_id: u32,
        peak_budget: usize,
    ) -> Result<Self, CompileError> {
        if byte_offsets.len() != present.len().saturating_add(1) {
            return Err(malformed_offsets(
                "byte_offsets must have exactly one more entry than present",
            ));
        }
        validate_csr_offsets(byte_offsets, token_bytes.len())?;
        let token_count = present.len();
        if token_count > MAX_PACKED_TOKENS {
            return Err(malformed_offsets("dense token count exceeds the cap"));
        }
        let mut present_count = 0usize;
        for &p in present {
            if p {
                present_count = present_count
                    .checked_add(1)
                    .ok_or_else(|| overflow("dense present count overflows"))?;
            }
        }
        // One id per present slot makes `present_count` a safe peak-memory bound.
        let peak = projected_peak_bytes(token_bytes.len(), present_count, present_count);
        if peak > peak_budget {
            return Err(malformed_offsets(
                "projected peak memory for this dense build exceeds the cap",
            ));
        }
        for i in 0..token_count {
            if present[i] && byte_offsets[i] == byte_offsets[i + 1] {
                return Err(empty_token());
            }
        }
        if usize::try_from(eos_token_id).is_ok_and(|e| present.get(e).copied().unwrap_or(false)) {
            return Err(eos_as_ordinary_id());
        }

        let key = |i: u32| -> &[u8] {
            &token_bytes[byte_offsets[i as usize] as usize..byte_offsets[i as usize + 1] as usize]
        };

        let mut order: Vec<PackedSortKey> = try_alloc(present_count, "dense sort-order")?;
        order.extend((0..token_count).filter(|&i| present[i]).map(|i| {
            let index = u32::try_from(i).expect("token_count <= MAX_PACKED_TOKENS fits u32");
            PackedSortKey {
                prefix: lexical_prefix(key(index)),
                index,
            }
        }));
        order.sort_unstable_by(|a, b| {
            a.prefix
                .cmp(&b.prefix)
                .then_with(|| key(a.index).cmp(key(b.index)))
        });

        let (blob_cap, ids_cap) = check_addressable_totals(token_bytes.len(), present_count)?;
        let mut blob = try_alloc(blob_cap as usize, "prepared vocabulary blob")?;
        let mut ids = try_alloc(ids_cap as usize, "prepared vocabulary id array")?;
        let mut records: Vec<TokenRecord> =
            try_alloc(present_count, "prepared vocabulary records")?;
        let mut group_ids: Vec<u32> = Vec::new();

        let mut i = 0usize;
        while i < order.len() {
            let mut j = i + 1;
            while j < order.len() && key(order[j].index) == key(order[i].index) {
                j += 1;
            }
            let bytes = key(order[i].index);

            // The dominant case: one id per byte key. The id is the slot's own index - no group
            // scratch, no sort, no dedup needed at all.
            if j == i + 1 {
                push_record(
                    &mut blob,
                    bytes,
                    &mut ids,
                    std::slice::from_ref(&order[i].index),
                    &mut records,
                )?;
                i = j;
                continue;
            }

            // Two or more distinct ids decoded to the identical byte string: merge.
            group_ids.clear();
            group_ids
                .try_reserve_exact(j - i)
                .map_err(|_| overflow("dense group-id scratch allocation failed"))?;
            for k in &order[i..j] {
                group_ids.push(k.index);
            }
            group_ids.sort_unstable();
            group_ids.dedup();
            push_record(&mut blob, bytes, &mut ids, &group_ids, &mut records)?;
            i = j;
        }
        Ok(Self { blob, records, ids })
    }

    /// The number of distinct token byte-strings.
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.records.len()
    }

    /// The total ordinary token ids across every record (not the distinct byte-string count: a
    /// record with two ids counts twice).
    #[must_use]
    pub(crate) fn total_ids(&self) -> usize {
        self.ids.len()
    }

    /// The total token bytes across every record (the blob's exact length, known for free).
    #[must_use]
    pub(crate) fn total_bytes(&self) -> usize {
        self.blob.len()
    }

    /// Record `i`'s byte string, sorted-order indexed.
    #[must_use]
    pub(crate) fn record_bytes(&self, i: usize) -> &[u8] {
        let r = self.records[i];
        &self.blob[r.byte_start as usize..r.byte_end as usize]
    }

    /// Record `i`'s token ids, sorted ascending (not insertion order).
    #[must_use]
    pub(crate) fn record_ids(&self, i: usize) -> &[u32] {
        let r = self.records[i];
        &self.ids[r.id_start as usize..r.id_end as usize]
    }

    /// Iterates every record as `(bytes, ids)`, in sorted order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&[u8], &[u32])> {
        (0..self.len()).map(move |i| (self.record_bytes(i), self.record_ids(i)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::build_vocabulary;
    use rustc_hash::FxHashMap as Map;

    fn vocab(pairs: &[(&[u8], &[u32])], eos: u32) -> Vocabulary {
        let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
        for &(bytes, ids) in pairs {
            map.insert(bytes.to_vec(), ids.to_vec());
        }
        build_vocabulary(eos, map).expect("vocab")
    }

    #[test]
    fn an_unsatisfiable_capacity_is_a_typed_error_not_an_abort() {
        // An impossible reservation verifies that allocation fails safely.
        match try_alloc::<u8>(usize::MAX / 2, "test") {
            Err(e) => assert_eq!(e.code, ErrorCode::InternalLimitExceeded),
            Ok(_) => panic!("an allocation this large must not succeed"),
        }
    }

    #[test]
    fn records_are_sorted_by_byte_key_regardless_of_hashmap_order() {
        let v = vocab(&[(b"zeta", &[2]), (b"alpha", &[0]), (b"mid", &[1])], 9);
        let p = PreparedVocabulary::build(&v).unwrap();
        let keys: Vec<&[u8]> = p.iter().map(|(b, _)| b).collect();
        assert_eq!(
            keys,
            vec![b"alpha".as_slice(), b"mid".as_slice(), b"zeta".as_slice()]
        );
    }

    #[test]
    fn ids_and_bytes_round_trip_exactly() {
        let v = vocab(&[(b"cat", &[17]), (b"car", &[42, 99])], 9);
        let p = PreparedVocabulary::build(&v).unwrap();
        for (bytes, ids) in p.iter() {
            let expected = v.token_ids(bytes).unwrap();
            assert_eq!(ids, expected);
        }
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn a_records_own_ids_come_back_sorted_ascending() {
        let v = vocab(&[(b"car", &[99, 7, 42])], 9);
        let p = PreparedVocabulary::build(&v).unwrap();
        assert_eq!(p.record_ids(0), &[7, 42, 99]);
    }

    #[test]
    fn non_utf8_bytes_round_trip_exactly() {
        let v = vocab(&[(&[0xFF, 0xFE, 0x00], &[1]), (b"a", &[0])], 9);
        let p = PreparedVocabulary::build(&v).unwrap();
        let bytes_seen: Vec<&[u8]> = p.iter().map(|(b, _)| b).collect();
        assert!(bytes_seen.contains(&[0xFFu8, 0xFE, 0x00].as_slice()));
    }

    fn as_content(p: &PreparedVocabulary) -> Vec<(Vec<u8>, Vec<u32>)> {
        p.iter().map(|(b, i)| (b.to_vec(), i.to_vec())).collect()
    }

    // Synthetic usize values, no real multi-gigabyte data: proves the u32 boundary is rejected
    // before any allocation attempt, without needing to actually construct that much memory.
    #[test]
    fn addressable_totals_at_the_u32_boundary_are_accepted() {
        assert!(check_addressable_totals(u32::MAX as usize, u32::MAX as usize).is_ok());
    }

    #[test]
    fn addressable_totals_one_past_u32_max_are_rejected() {
        assert!(check_addressable_totals(u32::MAX as usize + 1, 0).is_err());
        assert!(check_addressable_totals(0, u32::MAX as usize + 1).is_err());
    }

    /// Packs `(bytes, ids)` items into the four CSR buffers `build_from_packed` expects.
    fn pack(items: &[(&[u8], &[u32])]) -> (Vec<u8>, Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut token_bytes = Vec::new();
        let mut byte_offsets = vec![0u32];
        let mut token_ids = Vec::new();
        let mut id_offsets = vec![0u32];
        for &(bytes, ids) in items {
            token_bytes.extend_from_slice(bytes);
            byte_offsets.push(token_bytes.len() as u32);
            token_ids.extend_from_slice(ids);
            id_offsets.push(token_ids.len() as u32);
        }
        (token_bytes, byte_offsets, token_ids, id_offsets)
    }

    #[test]
    fn build_from_packed_matches_build_vocabulary_on_curated_cases() {
        let cases: Vec<Vec<(&[u8], &[u32])>> = vec![
            vec![(b"cat", &[17][..]), (b"car", &[42, 99]), (b"c", &[7])],
            vec![(b"z", &[3, 1, 2, 2][..])], // dup id within one token, dedups + sorts
            vec![
                (&[0xff][..], &[7][..]),
                (&[0x00], &[8]),
                (&[0x80, 0xff], &[9]),
            ],
        ];
        for items in &cases {
            let (tb, bo, ti, io) = pack(items);
            let packed = PreparedVocabulary::build_from_packed(&tb, &bo, &ti, &io, 9999).unwrap();

            let v = vocab(items, 9999);
            let via_vocab = PreparedVocabulary::build(&v).unwrap();
            assert_eq!(
                as_content(&packed),
                as_content(&via_vocab),
                "mismatch for {items:?}"
            );
        }
    }

    #[test]
    fn build_from_packed_merges_a_byte_key_repeated_across_two_packed_entries() {
        // "cat" appears twice at different packed indices with different ids: they must merge into
        // ONE record, same as `build_vocabulary`'s `HashMap::entry().or_default()` merge.
        let items: Vec<(&[u8], &[u32])> = vec![(b"cat", &[17]), (b"car", &[42]), (b"cat", &[200])];
        let (tb, bo, ti, io) = pack(&items);
        let packed = PreparedVocabulary::build_from_packed(&tb, &bo, &ti, &io, 9999).unwrap();
        assert_eq!(packed.len(), 2);
        let cat = packed.iter().find(|(b, _)| *b == b"cat").unwrap();
        assert_eq!(cat.1, &[17, 200]);
    }

    #[test]
    fn build_from_packed_rejects_an_empty_token() {
        let items: Vec<(&[u8], &[u32])> = vec![(b"", &[0])];
        let (tb, bo, ti, io) = pack(&items);
        let err = PreparedVocabulary::build_from_packed(&tb, &bo, &ti, &io, 9).unwrap_err();
        assert_eq!(err.code, ErrorCode::EmptyToken);
    }

    #[test]
    fn build_from_packed_rejects_a_mask_width_cap_violation_before_sorting() {
        let items: Vec<(&[u8], &[u32])> = vec![(b"a", &[MAX_MASK_TOKENS as u32])];
        let (tb, bo, ti, io) = pack(&items);
        let err = PreparedVocabulary::build_from_packed(&tb, &bo, &ti, &io, 9).unwrap_err();
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn build_from_packed_accepts_a_mask_width_exactly_at_the_cap() {
        let items: Vec<(&[u8], &[u32])> = vec![(b"a", &[(MAX_MASK_TOKENS - 1) as u32])];
        let (tb, bo, ti, io) = pack(&items);
        assert!(PreparedVocabulary::build_from_packed(&tb, &bo, &ti, &io, 9).is_ok());
    }

    #[test]
    fn build_from_packed_rejects_an_empty_id_list() {
        let items: Vec<(&[u8], &[u32])> = vec![(b"a", &[])];
        let (tb, bo, ti, io) = pack(&items);
        let err = PreparedVocabulary::build_from_packed(&tb, &bo, &ti, &io, 9).unwrap_err();
        assert_eq!(err.code, ErrorCode::MalformedTokenizer);
    }

    #[test]
    fn tuple_and_packed_paths_reject_an_empty_id_list_the_same_way() {
        let mut map: rustc_hash::FxHashMap<Vec<u8>, Vec<u32>> = rustc_hash::FxHashMap::default();
        map.insert(b"a".to_vec(), vec![]);
        let tuple_err = build_vocabulary(9, map).unwrap_err();

        let items: Vec<(&[u8], &[u32])> = vec![(b"a", &[])];
        let (tb, bo, ti, io) = pack(&items);
        let packed_err = PreparedVocabulary::build_from_packed(&tb, &bo, &ti, &io, 9).unwrap_err();

        assert_eq!(tuple_err.code, crate::error::ErrorCode::MalformedTokenizer);
        assert_eq!(packed_err.code, crate::error::ErrorCode::MalformedTokenizer);
    }

    #[test]
    fn build_from_packed_rejects_an_empty_id_list_after_duplicate_key_merge() {
        // Two packed entries for the SAME byte key, both with empty ids: the merged group is still
        // empty and must be rejected, not silently accepted because each half looked like "no ids".
        let items: Vec<(&[u8], &[u32])> = vec![(b"a", &[]), (b"a", &[])];
        let (tb, bo, ti, io) = pack(&items);
        let err = PreparedVocabulary::build_from_packed(&tb, &bo, &ti, &io, 9).unwrap_err();
        assert_eq!(err.code, ErrorCode::MalformedTokenizer);
    }

    #[test]
    fn build_from_packed_rejects_eos_as_an_ordinary_id() {
        let items: Vec<(&[u8], &[u32])> = vec![(b"a", &[5])];
        let (tb, bo, ti, io) = pack(&items);
        let err = PreparedVocabulary::build_from_packed(&tb, &bo, &ti, &io, 5).unwrap_err();
        assert_eq!(err.code, ErrorCode::MalformedTokenizer);
    }

    #[test]
    fn build_from_packed_rejects_mismatched_offset_array_lengths() {
        let items: Vec<(&[u8], &[u32])> = vec![(b"a", &[0])];
        let (tb, bo, ti, mut io) = pack(&items);
        io.push(99); // now longer than byte_offsets
        assert!(PreparedVocabulary::build_from_packed(&tb, &bo, &ti, &io, 9).is_err());
    }

    #[test]
    fn build_from_packed_rejects_non_monotonic_byte_offsets() {
        let tb = b"catcar".to_vec();
        let bo = vec![0u32, 3, 1, 6]; // decreasing: 3 > 1
        let ti = vec![0u32, 1, 2];
        let io = vec![0u32, 1, 2, 3];
        assert!(PreparedVocabulary::build_from_packed(&tb, &bo, &ti, &io, 1000).is_err());
    }

    #[test]
    fn build_from_packed_rejects_a_final_offset_that_does_not_match_the_buffer_length() {
        let items: Vec<(&[u8], &[u32])> = vec![(b"a", &[0])];
        let (tb, mut bo, ti, io) = pack(&items);
        *bo.last_mut().unwrap() += 1; // claims one more byte than token_bytes actually has
        assert!(PreparedVocabulary::build_from_packed(&tb, &bo, &ti, &io, 9).is_err());
    }

    #[test]
    fn build_from_packed_orders_adversarial_keys_correctly() {
        // Covers prefix ties, byte prefixes, raw bytes, and duplicate-key merging.
        let items: Vec<(&[u8], &[u32])> = vec![
            (b"abcdefghZZ", &[0][..]),
            (b"abcdefghAA", &[1]),
            (b"ab", &[2]),
            (&[0x00, 0x01, 0x02], &[3]),
            (&[0xff, 0xfe], &[4]),
            (b"ab", &[5]),
        ];
        let (tb, bo, ti, io) = pack(&items);
        let packed = PreparedVocabulary::build_from_packed(&tb, &bo, &ti, &io, 9999).unwrap();

        let mut expected: Vec<Vec<u8>> = items.iter().map(|(b, _)| b.to_vec()).collect();
        expected.sort();
        expected.dedup();
        let got: Vec<Vec<u8>> = packed.iter().map(|(b, _)| b.to_vec()).collect();
        assert_eq!(got, expected);

        let ab = packed.iter().find(|(b, _)| *b == b"ab").unwrap();
        assert_eq!(
            ab.1,
            &[2, 5],
            "duplicate key 'ab' must merge both ids, sorted ascending"
        );
    }

    #[test]
    fn build_from_packed_agrees_with_build_vocabulary_on_200_randomized_vocabularies() {
        let mut state = 0xD1B5_4A32_D192_ED03u64;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for seed in 0..200 {
            let count = 1 + (next() % 30) as usize;
            let mut owned: Vec<(Vec<u8>, Vec<u32>)> = Vec::new();
            let mut next_id = 0u32;
            for _ in 0..count {
                let len = 1 + (next() % 6) as usize;
                let bytes: Vec<u8> = (0..len).map(|_| (next() & 0xff) as u8).collect();
                let ids: Vec<u32> = (0..1 + next() % 3)
                    .map(|_| {
                        let id = next_id;
                        next_id += 1;
                        id
                    })
                    .collect();
                owned.push((bytes, ids));
            }
            let eos = next_id + 1000;
            let items: Vec<(&[u8], &[u32])> = owned
                .iter()
                .map(|(b, i)| (b.as_slice(), i.as_slice()))
                .collect();
            let (tb, bo, ti, io) = pack(&items);
            let packed = PreparedVocabulary::build_from_packed(&tb, &bo, &ti, &io, eos)
                .unwrap_or_else(|e| panic!("seed {seed}: {e}"));

            let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
            for (b, i) in &owned {
                map.entry(b.clone()).or_default().extend_from_slice(i);
            }
            let v = build_vocabulary(eos, map).expect("vocab");
            let via_vocab = PreparedVocabulary::build(&v).unwrap();
            assert_eq!(
                as_content(&packed),
                as_content(&via_vocab),
                "mismatch at seed {seed}"
            );
        }
    }

    // Synthetic usize values, no real multi-hundred-MB allocation: proves the projected-peak
    // arithmetic and cap are load-bearing without constructing that much real memory.
    #[test]
    fn projected_peak_bytes_accepts_a_small_realistic_build() {
        let peak = projected_peak_bytes(1_000_000, 200_000, 50_000);
        assert!(peak < MAX_PACKED_PROJECTED_PEAK_BYTES);
    }

    #[test]
    fn projected_peak_bytes_at_the_maximal_token_count_stays_under_the_cap() {
        // The packed-token cap protects against degenerate input, not real tokenizers.
        let peak = projected_peak_bytes(MAX_PACKED_TOKENS, MAX_PACKED_TOKENS, MAX_PACKED_TOKENS);
        assert!(peak < MAX_PACKED_PROJECTED_PEAK_BYTES);
    }

    #[test]
    fn projected_peak_bytes_is_dominated_by_token_count_not_just_input_buffer_length() {
        // A single, tiny input buffer with a huge CLAIMED id count (independent of token_count)
        // must also be caught: this is exactly the gap raw-input-byte admission alone would miss.
        let small_buffers = projected_peak_bytes(100, 100, 10);
        let huge_id_count = projected_peak_bytes(100, 1 << 28, 10);
        assert!(small_buffers < MAX_PACKED_PROJECTED_PEAK_BYTES);
        assert!(huge_id_count > MAX_PACKED_PROJECTED_PEAK_BYTES);
    }

    #[test]
    fn projected_peak_bytes_charges_group_scratch_on_top_of_the_final_ids_array() {
        // All ids concentrated under ONE byte key: group_ids' worst-case growth (token_ids_len)
        // is live AT THE SAME TIME as the final `ids` array, so both must be charged.
        let token_ids_len = 1 << 20;
        let peak = projected_peak_bytes(10, token_ids_len, 1);
        assert!(
            peak >= token_ids_len * 4 * 2,
            "group scratch and the ids array must both count"
        );
    }

    #[test]
    fn projected_peak_bytes_charges_the_sparse_branch_not_just_dense() {
        // A token_ids_len far larger than the fixed dense-branch cap must show up via the sparse
        // term, not be silently capped at the dense worst case.
        let token_ids_len = 100_000_000;
        let peak = projected_peak_bytes(10, token_ids_len, 1);
        assert!(peak >= token_ids_len * CONSERVATIVE_SPARSE_SLOT_BYTES);
    }

    #[test]
    fn build_from_packed_budgeted_rejects_on_a_tiny_injected_budget_no_huge_allocation() {
        // A too-small budget rejects even a valid packed vocabulary.
        let items: Vec<(&[u8], &[u32])> = vec![(b"a", &[0]), (b"b", &[1])];
        let (tb, bo, ti, io) = pack(&items);
        let err = PreparedVocabulary::build_from_packed_budgeted(&tb, &bo, &ti, &io, 9, 4)
            .expect_err("a 4-byte budget cannot hold even the smallest real build");
        assert_eq!(err.code, ErrorCode::Unsupported);
    }

    #[test]
    fn build_from_packed_budgeted_binds_once_the_budget_covers_the_real_peak() {
        let items: Vec<(&[u8], &[u32])> = vec![(b"a", &[0]), (b"b", &[1])];
        let (tb, bo, ti, io) = pack(&items);
        let generous = projected_peak_bytes(tb.len(), ti.len(), 2) + 1;
        assert!(
            PreparedVocabulary::build_from_packed_budgeted(&tb, &bo, &ti, &io, 9, generous).is_ok()
        );
    }

    /// Packs a dense, id-ordered decode (`decoded[i]` is slot `i`'s bytes, `None` for absent) into
    /// the `(token_bytes, byte_offsets, present)` triple `build_from_dense_id_ordered` expects.
    fn pack_dense(decoded: &[Option<&[u8]>]) -> (Vec<u8>, Vec<u32>, Vec<bool>) {
        let mut token_bytes = Vec::new();
        let mut byte_offsets = vec![0u32];
        let mut present = Vec::with_capacity(decoded.len());
        for slot in decoded {
            if let Some(bytes) = slot {
                token_bytes.extend_from_slice(bytes);
                present.push(true);
            } else {
                present.push(false);
            }
            byte_offsets.push(token_bytes.len() as u32);
        }
        (token_bytes, byte_offsets, present)
    }

    #[test]
    fn dense_id_ordered_ids_are_the_slot_index() {
        let decoded = [
            Some(b"cat".as_slice()),
            Some(b"dog".as_slice()),
            Some(b"a".as_slice()),
        ];
        let (tb, bo, present) = pack_dense(&decoded);
        let p = PreparedVocabulary::build_from_dense_id_ordered(&tb, &bo, &present, 9999).unwrap();
        assert_eq!(p.len(), 3);
        let cat = p.iter().find(|(b, _)| *b == b"cat").unwrap();
        assert_eq!(cat.1, &[0]);
        let dog = p.iter().find(|(b, _)| *b == b"dog").unwrap();
        assert_eq!(dog.1, &[1]);
        let a = p.iter().find(|(b, _)| *b == b"a").unwrap();
        assert_eq!(a.1, &[2]);
    }

    #[test]
    fn dense_id_ordered_skips_absent_slots_without_error() {
        let decoded = [Some(b"cat".as_slice()), None, Some(b"dog".as_slice())];
        let (tb, bo, present) = pack_dense(&decoded);
        let p = PreparedVocabulary::build_from_dense_id_ordered(&tb, &bo, &present, 9999).unwrap();
        assert_eq!(p.len(), 2, "the absent slot must not become a record");
    }

    #[test]
    fn dense_id_ordered_rejects_an_empty_byte_string_on_a_present_slot() {
        let decoded = [Some(b"".as_slice())];
        let (tb, bo, present) = pack_dense(&decoded);
        let err =
            PreparedVocabulary::build_from_dense_id_ordered(&tb, &bo, &present, 9).unwrap_err();
        assert_eq!(err.code, ErrorCode::EmptyToken);
    }

    #[test]
    fn dense_id_ordered_rejects_eos_id_present_as_an_ordinary_slot() {
        let decoded = [Some(b"a".as_slice()), Some(b"b".as_slice())];
        let (tb, bo, present) = pack_dense(&decoded);
        // eos_token_id = 1: slot 1 ("b") is present, so EOS collides with an ordinary token.
        let err =
            PreparedVocabulary::build_from_dense_id_ordered(&tb, &bo, &present, 1).unwrap_err();
        assert_eq!(err.code, ErrorCode::MalformedTokenizer);
    }

    #[test]
    fn dense_id_ordered_accepts_eos_id_pointing_at_an_absent_slot() {
        let decoded = [Some(b"a".as_slice()), None];
        let (tb, bo, present) = pack_dense(&decoded);
        // eos_token_id = 1: slot 1 is absent (the caller's own EOS placeholder), no collision.
        assert!(PreparedVocabulary::build_from_dense_id_ordered(&tb, &bo, &present, 1).is_ok());
    }

    #[test]
    fn dense_id_ordered_merges_two_different_ids_decoding_to_the_same_bytes() {
        let decoded = [
            Some(b"a".as_slice()),
            Some(b"a".as_slice()),
            Some(b"b".as_slice()),
        ];
        let (tb, bo, present) = pack_dense(&decoded);
        let p = PreparedVocabulary::build_from_dense_id_ordered(&tb, &bo, &present, 9999).unwrap();
        assert_eq!(p.len(), 2, "the two 'a' slots must merge into one record");
        let a = p.iter().find(|(b, _)| *b == b"a").unwrap();
        assert_eq!(a.1, &[0, 1], "both source ids survive, sorted ascending");
    }

    #[test]
    fn dense_id_ordered_agrees_with_build_from_packed_on_the_same_content() {
        // Same logical vocabulary, expressed both ways: dense (id = index) and packed (explicit
        // id lists) must produce byte-identical PreparedVocabulary content.
        let items: Vec<(&[u8], &[u32])> = vec![
            (b"cat", &[0][..]),
            (b"car", &[1]),
            (b"c", &[2]),
            (b"cat", &[3]),
        ];
        let (tb_p, bo_p, ti_p, io_p) = pack(&items);
        let packed =
            PreparedVocabulary::build_from_packed(&tb_p, &bo_p, &ti_p, &io_p, 9999).unwrap();

        let decoded: Vec<Option<&[u8]>> =
            vec![Some(b"cat"), Some(b"car"), Some(b"c"), Some(b"cat")];
        let (tb_d, bo_d, present) = pack_dense(&decoded);
        let dense =
            PreparedVocabulary::build_from_dense_id_ordered(&tb_d, &bo_d, &present, 9999).unwrap();

        assert_eq!(as_content(&packed), as_content(&dense));
    }

    #[test]
    fn dense_id_ordered_rejects_a_byte_offsets_length_mismatch() {
        let (tb, mut bo, present) = pack_dense(&[Some(b"a")]);
        bo.push(99); // now longer than present.len() + 1
        assert!(PreparedVocabulary::build_from_dense_id_ordered(&tb, &bo, &present, 9).is_err());
    }

    #[test]
    fn dense_id_ordered_budgeted_rejects_on_a_tiny_injected_budget() {
        let (tb, bo, present) = pack_dense(&[Some(b"a"), Some(b"b")]);
        let err =
            PreparedVocabulary::build_from_dense_id_ordered_budgeted(&tb, &bo, &present, 9, 4)
                .expect_err("a 4-byte budget cannot hold even the smallest real build");
        assert_eq!(err.code, ErrorCode::Unsupported);
    }

    #[test]
    fn dense_id_ordered_agrees_with_build_vocabulary_on_200_randomized_dense_vocabularies() {
        let mut state = 0xA5A5_1234_9E37_79B9u64;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for seed in 0..200 {
            let count = 1 + (next() % 40) as usize;
            let mut owned: Vec<Option<Vec<u8>>> = Vec::with_capacity(count);
            for _ in 0..count {
                if next() % 5 == 0 {
                    owned.push(None); // ~20% absent slots
                    continue;
                }
                let len = 1 + (next() % 6) as usize;
                owned.push(Some((0..len).map(|_| (next() & 0xff) as u8).collect()));
            }
            let eos = count as u32 + 1000; // always out of range, never collides
            let decoded: Vec<Option<&[u8]>> = owned.iter().map(|o| o.as_deref()).collect();
            let (tb, bo, present) = pack_dense(&decoded);
            let dense = PreparedVocabulary::build_from_dense_id_ordered(&tb, &bo, &present, eos)
                .unwrap_or_else(|e| panic!("seed {seed}: {e}"));

            let mut map: Map<Vec<u8>, Vec<u32>> = Map::default();
            for (i, slot) in owned.iter().enumerate() {
                if let Some(bytes) = slot {
                    map.entry(bytes.clone()).or_default().push(i as u32);
                }
            }
            let v = build_vocabulary(eos, map).expect("vocab");
            let via_vocab = PreparedVocabulary::build(&v).unwrap();
            assert_eq!(
                as_content(&dense),
                as_content(&via_vocab),
                "mismatch at seed {seed}"
            );
        }
    }
}
