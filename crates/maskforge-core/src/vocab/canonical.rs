//! Canonical vocabulary storage shared by fingerprinting, lookup, and trie construction.

use rustc_hash::FxHashMap;

use super::prepared::{try_alloc, PreparedVocabulary};
use super::Vocabulary;
use crate::error::{CompileError, ErrorCode, Stage};
use crate::index::VocabFingerprint;

/// Token id to record index. Dense when ids are roughly contiguous (real tokenizers), sparse
/// otherwise - a dense array over a huge id range would waste more than it saves.
#[derive(Debug)]
enum IdIndex {
    Dense(Box<[u32]>),
    Sparse(FxHashMap<u32, u32>),
}

const ABSENT: u32 = u32::MAX;

/// Conservative per-entry byte cost of an `FxHashMap<u32, u32>` slot: the 8-byte payload plus
/// hashbrown's control byte, load-factor slack, and transient resize overlap.
pub(crate) const CONSERVATIVE_SPARSE_SLOT_BYTES: usize = 24;

impl IdIndex {
    /// `ordinary_max` excludes EOS - EOS is never looked up through this index, so it must not
    /// size a dense table it will never occupy a slot in.
    fn build(
        prepared: &PreparedVocabulary,
        ordinary_max: Option<u32>,
    ) -> Result<Self, CompileError> {
        let Some(max_id) = ordinary_max else {
            return Ok(Self::Dense(Box::new([])));
        };
        let total_ids = prepared.total_ids();
        let dense_bytes = (max_id as usize).saturating_add(1).saturating_mul(4);
        let sparse_bytes = total_ids.saturating_mul(CONSERVATIVE_SPARSE_SLOT_BYTES);
        if dense_bytes <= sparse_bytes {
            let mut table: Vec<u32> = try_alloc(max_id as usize + 1, "id-index dense table")?;
            table.resize(max_id as usize + 1, ABSENT);
            for (i, (_, ids)) in prepared.iter().enumerate() {
                let ri = u32::try_from(i).map_err(|_| overflow("record index exceeds u32"))?;
                for &id in ids {
                    // A distinct byte string already claimed this id: ambiguous, reject it.
                    if table[id as usize] != ABSENT && table[id as usize] != ri {
                        return Err(ambiguous_id());
                    }
                    table[id as usize] = ri;
                }
            }
            Ok(Self::Dense(table.into_boxed_slice()))
        } else {
            let mut map = FxHashMap::default();
            map.try_reserve(total_ids)
                .map_err(|_| overflow("id-index sparse map allocation failed"))?;
            for (i, (_, ids)) in prepared.iter().enumerate() {
                let ri = u32::try_from(i).map_err(|_| overflow("record index exceeds u32"))?;
                for &id in ids {
                    if let Some(&existing) = map.get(&id) {
                        if existing != ri {
                            return Err(ambiguous_id());
                        }
                    }
                    map.insert(id, ri);
                }
            }
            Ok(Self::Sparse(map))
        }
    }

    fn record_of(&self, id: u32) -> Option<usize> {
        match self {
            Self::Dense(table) => match table.get(id as usize) {
                Some(&ABSENT) | None => None,
                Some(&ri) => Some(ri as usize),
            },
            Self::Sparse(map) => map.get(&id).map(|&ri| ri as usize),
        }
    }
}

fn overflow(msg: &'static str) -> CompileError {
    CompileError::new(ErrorCode::InternalLimitExceeded, Stage::L4Bind, msg)
}

fn ambiguous_id() -> CompileError {
    CompileError::new(
        ErrorCode::MalformedTokenizer,
        Stage::ArtifactValidate,
        "one token id maps to two different byte sequences",
    )
}

fn logits_vocab_size_too_small() -> CompileError {
    CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L4Bind,
        "explicit logits vocab size is smaller than the largest token/EOS id",
    )
}

/// Mask-width cap: `max_token_id + 1` above this is rejected before any mask is allocated. A mask
/// is one bit per id, so this bounds a single row at `1 << 21` bits = 256 KiB.
pub(crate) const MAX_MASK_TOKENS: usize = 1 << 21;

/// The vocabulary's one canonical representation: byte-sorted records (shared with fingerprinting
/// and trie construction), a token-id reverse index, EOS, mask width, and content fingerprint.
#[derive(Debug)]
pub(crate) struct CanonicalVocabulary {
    prepared: PreparedVocabulary,
    id_index: IdIndex,
    eos_token_id: u32,
    mask_vocab_size: usize,
    fingerprint: VocabFingerprint,
}

impl CanonicalVocabulary {
    /// Builds canonical storage from a validated vocabulary with inferred mask width.
    pub(crate) fn from_vocabulary(vocab: &Vocabulary) -> Result<Self, CompileError> {
        Self::from_vocabulary_with_logits_vocab_size(vocab, None)
    }

    /// Builds canonical storage with an explicit logits mask width.
    /// Widths smaller than observed token or EOS ids are rejected.
    pub(crate) fn from_vocabulary_with_logits_vocab_size(
        vocab: &Vocabulary,
        logits_vocab_size: Option<usize>,
    ) -> Result<Self, CompileError> {
        let prepared = PreparedVocabulary::build(vocab)?;
        Self::from_prepared(prepared, vocab.eos_token_id(), logits_vocab_size)
    }

    /// Builds canonical storage directly from packed CSR buffers.
    /// Duplicate byte keys merge without intermediate token maps.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_packed_with_logits_vocab_size(
        token_bytes: &[u8],
        byte_offsets: &[u32],
        token_ids: &[u32],
        id_offsets: &[u32],
        eos_token_id: u32,
        logits_vocab_size: Option<usize>,
    ) -> Result<Self, CompileError> {
        let prepared = PreparedVocabulary::build_from_packed(
            token_bytes,
            byte_offsets,
            token_ids,
            id_offsets,
            eos_token_id,
        )?;
        Self::from_prepared(prepared, eos_token_id, logits_vocab_size)
    }

    /// Builds canonical storage from dense, id-ordered tokenizer bytes.
    pub(crate) fn from_dense_id_ordered_with_logits_vocab_size(
        token_bytes: &[u8],
        byte_offsets: &[u32],
        present: &[bool],
        eos_token_id: u32,
        logits_vocab_size: Option<usize>,
    ) -> Result<Self, CompileError> {
        let prepared = PreparedVocabulary::build_from_dense_id_ordered(
            token_bytes,
            byte_offsets,
            present,
            eos_token_id,
        )?;
        Self::from_prepared(prepared, eos_token_id, logits_vocab_size)
    }

    fn from_prepared(
        prepared: PreparedVocabulary,
        eos_token_id: u32,
        logits_vocab_size: Option<usize>,
    ) -> Result<Self, CompileError> {
        let (fingerprint, ordinary_max) =
            VocabFingerprint::of_prepared_with_ordinary_max(&prepared, eos_token_id);
        let max_id = ordinary_max.map_or(eos_token_id, |m| m.max(eos_token_id));
        let inferred_width = usize::try_from(max_id)
            .ok()
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| overflow("max token id overflows the mask width"))?;
        let mask_vocab_size = match logits_vocab_size {
            Some(width) if width < inferred_width => return Err(logits_vocab_size_too_small()),
            Some(width) => width,
            None => inferred_width,
        };
        if mask_vocab_size > MAX_MASK_TOKENS {
            return Err(overflow("mask width exceeds the token-id cap"));
        }
        let id_index = IdIndex::build(&prepared, ordinary_max)?;
        Ok(Self {
            prepared,
            id_index,
            eos_token_id,
            mask_vocab_size,
            fingerprint,
        })
    }

    #[must_use]
    pub(crate) fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    #[must_use]
    pub(crate) fn mask_vocab_size(&self) -> usize {
        self.mask_vocab_size
    }

    /// Packed `u32` words in one mask row: `ceil(mask_vocab_size / 32)`.
    #[must_use]
    pub(crate) fn mask_width_words(&self) -> usize {
        self.mask_vocab_size.div_ceil(32)
    }

    #[must_use]
    pub(crate) fn fingerprint(&self) -> VocabFingerprint {
        self.fingerprint
    }

    /// The number of stored ordinary token ids plus one for EOS (matches `Vocabulary::len()`'s
    /// public contract - NOT the distinct byte-string count and NOT the mask width).
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.prepared.total_ids() + 1
    }

    /// True when the token map has no records - matches `Vocabulary::is_empty()`.
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.prepared.len() == 0
    }

    /// `id`'s byte string, or `None` if `id` names no ordinary token (including EOS, which is a
    /// side value never present as an ordinary id).
    #[must_use]
    pub(crate) fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.id_index
            .record_of(id)
            .map(|ri| self.prepared.record_bytes(ri))
    }

    /// Every record as `(bytes, ids)`, byte-sorted - the same order fingerprinting and trie
    /// construction read, and what the Naive oracle walks instead of a raw `HashMap`.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&[u8], &[u32])> {
        self.prepared.iter()
    }

    /// The shared canonical sorted form, for a trie builder to consume without re-sorting.
    pub(crate) fn prepared(&self) -> &PreparedVocabulary {
        &self.prepared
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::build_vocabulary;

    fn vocab(pairs: &[(&[u8], &[u32])], eos: u32) -> Vocabulary {
        let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
        for &(bytes, ids) in pairs {
            map.insert(bytes.to_vec(), ids.to_vec());
        }
        build_vocabulary(eos, map).expect("vocab")
    }

    #[test]
    fn token_bytes_round_trips_every_id() {
        let v = vocab(&[(b"cat", &[17]), (b"car", &[42, 99]), (b"c", &[7])], 9999);
        let c = CanonicalVocabulary::from_vocabulary(&v).unwrap();
        assert_eq!(c.token_bytes(17), Some(b"cat".as_slice()));
        assert_eq!(c.token_bytes(42), Some(b"car".as_slice()));
        assert_eq!(c.token_bytes(99), Some(b"car".as_slice()));
        assert_eq!(c.token_bytes(7), Some(b"c".as_slice()));
        assert_eq!(c.token_bytes(9999), None); // EOS is not an ordinary id
        assert_eq!(c.token_bytes(123), None); // never assigned
    }

    #[test]
    fn mask_vocab_size_covers_the_max_id_and_eos() {
        let v = vocab(&[(b"a", &[3])], 10);
        let c = CanonicalVocabulary::from_vocabulary(&v).unwrap();
        assert_eq!(c.mask_vocab_size(), 11); // max(3, 10) + 1
    }

    #[test]
    fn fingerprint_matches_vocab_fingerprint_of() {
        let v = vocab(&[(b"a", &[0]), (b"b", &[1])], 9);
        let c = CanonicalVocabulary::from_vocabulary(&v).unwrap();
        assert_eq!(
            c.fingerprint().as_bytes(),
            VocabFingerprint::of(&v).unwrap().as_bytes()
        );
    }

    #[test]
    fn sparse_id_space_still_resolves_correctly() {
        // A huge gap between ids forces the sparse index path, not the dense one.
        let v = vocab(&[(b"a", &[0]), (b"b", &[1_000_000])], 2_000_000);
        let c = CanonicalVocabulary::from_vocabulary(&v).unwrap();
        assert_eq!(c.token_bytes(0), Some(b"a".as_slice()));
        assert_eq!(c.token_bytes(1_000_000), Some(b"b".as_slice()));
        assert_eq!(c.token_bytes(500_000), None);
        assert_eq!(c.len(), 3); // 2 ordinary ids + 1 for EOS
    }

    #[test]
    fn one_id_under_two_byte_sequences_is_rejected_dense_and_sparse() {
        let dense = vocab(&[(b"a", &[0]), (b"b", &[0])], 9);
        assert!(CanonicalVocabulary::from_vocabulary(&dense).is_err());

        // A wide id gap (via an unrelated third token) forces the sparse index path.
        let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
        map.insert(b"a".to_vec(), vec![0]);
        map.insert(b"b".to_vec(), vec![0]); // same id as "a" under a different byte string
        map.insert(b"c".to_vec(), vec![500_000]);
        let sparse_conflict = build_vocabulary(600_000, map).unwrap();
        assert!(CanonicalVocabulary::from_vocabulary(&sparse_conflict).is_err());
    }

    #[test]
    fn dense_and_sparse_paths_agree_on_200_randomized_vocabularies() {
        fn next_rand(state: &mut u64) -> u64 {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        for seed in 0..200u64 {
            let mut st = seed.wrapping_add(1);
            let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
            let mut next_id = 0u32;
            let sparse = seed % 2 == 0;
            for _ in 0..(4 + next_rand(&mut st) % 30) {
                let len = 1 + (next_rand(&mut st) % 6) as usize;
                let bytes: Vec<u8> = (0..len)
                    .map(|_| (next_rand(&mut st) & 0xff) as u8)
                    .collect();
                let ids = map.entry(bytes).or_default();
                let step = if sparse {
                    1 + next_rand(&mut st) % 5000
                } else {
                    1
                };
                for _ in 0..1 + next_rand(&mut st) % 3 {
                    ids.push(next_id);
                    next_id += u32::try_from(step).unwrap();
                }
            }
            let eos = next_id + 1000;
            let v = build_vocabulary(eos, map).expect("vocab");
            let c = CanonicalVocabulary::from_vocabulary(&v).unwrap();
            for (bytes, ids) in v.tokens() {
                for &id in ids {
                    assert_eq!(c.token_bytes(id), Some(bytes.as_slice()), "seed {seed}");
                }
            }
        }
    }

    #[test]
    fn empty_vocabulary_is_empty_and_len_counts_only_eos() {
        let v = vocab(&[], 9);
        let c = CanonicalVocabulary::from_vocabulary(&v).unwrap();
        assert!(c.is_empty());
        assert_eq!(c.len(), 1); // 0 ordinary ids + 1 for EOS
    }

    #[test]
    fn a_byte_key_with_an_empty_id_list_is_rejected() {
        // An empty id list represents nothing decodable; the choke-point rejects it before a
        // CanonicalVocabulary (or a persisted trie derived from it) ever has to reason about it.
        let mut map: FxHashMap<Vec<u8>, Vec<u32>> = FxHashMap::default();
        map.insert(b"a".to_vec(), vec![]);
        assert_eq!(
            build_vocabulary(9, map).unwrap_err().code,
            crate::error::ErrorCode::MalformedTokenizer
        );
    }

    #[test]
    fn sparse_eos_does_not_force_a_huge_dense_table_for_small_ordinary_ids() {
        // Sparse EOS must not force a dense index sized to the EOS id.
        let v = vocab(&[(b"a", &[0]), (b"b", &[1])], 1_000_000);
        let c = CanonicalVocabulary::from_vocabulary(&v).unwrap();
        assert_eq!(c.token_bytes(0), Some(b"a".as_slice()));
        assert_eq!(c.token_bytes(1), Some(b"b".as_slice()));
        assert_eq!(c.token_bytes(1_000_000), None); // EOS is never an ordinary id
        assert_eq!(c.mask_vocab_size(), 1_000_001); // mask width still covers EOS
    }

    #[test]
    fn mask_vocab_size_at_the_cap_is_accepted_one_above_is_rejected() {
        let at_cap = vocab(&[(b"a", &[(MAX_MASK_TOKENS - 1) as u32])], 0);
        assert_eq!(
            CanonicalVocabulary::from_vocabulary(&at_cap)
                .unwrap()
                .mask_vocab_size(),
            MAX_MASK_TOKENS
        );
        let over_cap = vocab(&[(b"a", &[MAX_MASK_TOKENS as u32])], 0);
        assert!(CanonicalVocabulary::from_vocabulary(&over_cap).is_err());
    }

    #[test]
    fn explicit_logits_vocab_size_wider_than_inferred_is_used_directly() {
        let v = vocab(&[(b"a", &[5])], 9); // inferred width would be 10 (max(5, 9) + 1)
        let c = CanonicalVocabulary::from_vocabulary_with_logits_vocab_size(&v, Some(50000))
            .expect("wider explicit width accepted");
        assert_eq!(c.mask_vocab_size(), 50000);
    }

    #[test]
    fn explicit_logits_vocab_size_smaller_than_inferred_is_rejected() {
        let v = vocab(&[(b"a", &[500])], 9); // inferred width is 501
        let err = CanonicalVocabulary::from_vocabulary_with_logits_vocab_size(&v, Some(100))
            .expect_err("a width narrower than an actual id must be rejected");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }

    #[test]
    fn explicit_logits_vocab_size_equal_to_inferred_is_accepted() {
        let v = vocab(&[(b"a", &[5])], 9); // inferred width is 10
        let c = CanonicalVocabulary::from_vocabulary_with_logits_vocab_size(&v, Some(10))
            .expect("width exactly matching the inferred one is accepted");
        assert_eq!(c.mask_vocab_size(), 10);
    }

    #[test]
    fn explicit_logits_vocab_size_does_not_change_the_fingerprint() {
        let v = vocab(&[(b"a", &[5])], 9);
        let narrow = CanonicalVocabulary::from_vocabulary(&v).unwrap();
        let wide =
            CanonicalVocabulary::from_vocabulary_with_logits_vocab_size(&v, Some(50000)).unwrap();
        assert_eq!(narrow.fingerprint(), wide.fingerprint());
        assert_ne!(narrow.mask_vocab_size(), wide.mask_vocab_size());
    }

    #[test]
    fn explicit_logits_vocab_size_above_the_cap_is_rejected() {
        let v = vocab(&[(b"a", &[5])], 9);
        let err = CanonicalVocabulary::from_vocabulary_with_logits_vocab_size(
            &v,
            Some(MAX_MASK_TOKENS + 1),
        )
        .expect_err("an explicit width past the cap must still be rejected");
        assert_eq!(err.code, ErrorCode::InternalLimitExceeded);
    }
}
