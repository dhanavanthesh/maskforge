//! Partitions vocabulary tokens for bounded string matching.

use std::mem::size_of;
use std::sync::Arc;

use super::VocabTrie;
use crate::error::{CompileError, ErrorCode, Stage};
use crate::structured::lexer::{DecodeStep, JsonStringDecoder};
use crate::vocab::CanonicalVocabulary;

const MAX_CATALOG_BYTES: usize = 64 << 20;

#[derive(Debug)]
pub(crate) struct TokenSlice {
    token_mask: Box<[u64]>,
    filtered_trie: VocabTrie,
    scalar_count: Option<u16>,
    max_byte_len: usize,
}

pub(crate) const MAX_BOUNDED_LEN: usize = 32;

#[derive(Debug)]
pub(crate) struct TokenSliceCatalog {
    safe: TokenSlice,
    uncertain_trie: VocabTrie,
    body_extra: TokenSlice,
    body_uncertain_trie: VocabTrie,
    body_boundary_trie: VocabTrie,
    body_boundary_max_bytes: usize,
    body_invalid_trie: VocabTrie,
    body_invalid_max_bytes: usize,
    scalar_overflow_trie: VocabTrie,
    retained_bytes: usize,
    byte_len_le: Box<[Box<[u64]>]>,
    scalar_len_le: Box<[Box<[u64]>]>,
    max_scalar_len: usize,
}

impl TokenSliceCatalog {
    pub(crate) fn try_build(
        vocabulary: &CanonicalVocabulary,
    ) -> Result<Option<Arc<Self>>, CompileError> {
        Self::try_build_with_budget(vocabulary, MAX_CATALOG_BYTES)
    }

    fn try_build_with_budget(
        vocabulary: &CanonicalVocabulary,
        byte_budget: usize,
    ) -> Result<Option<Arc<Self>>, CompileError> {
        let estimated = estimated_build_bytes(vocabulary)?;
        if estimated > byte_budget {
            return Ok(None);
        }
        let _permit = crate::mem_gate::acquire_vocab_build_bytes(estimated)?;
        let fingerprint = vocabulary.fingerprint();
        let safe_trie = VocabTrie::build_byte_filtered_from_prepared(
            vocabulary.prepared(),
            fingerprint,
            |bytes| safe_scalar_count(bytes).is_some(),
        )?;
        let uncertain_trie = VocabTrie::build_byte_filtered_from_prepared(
            vocabulary.prepared(),
            fingerprint,
            |bytes| safe_scalar_count(bytes).is_none(),
        )?;
        let token_mask = safe_token_mask(vocabulary)?;
        let body_extra_trie = VocabTrie::build_byte_filtered_from_prepared(
            vocabulary.prepared(),
            fingerprint,
            body_extra,
        )?;
        let body_uncertain_trie = VocabTrie::build_byte_filtered_from_prepared(
            vocabulary.prepared(),
            fingerprint,
            |bytes| !body_prefix(bytes),
        )?;
        let body_extra_mask = token_mask_where(vocabulary, body_extra)?;
        let body_boundary_trie = VocabTrie::build_byte_filtered_from_prepared(
            vocabulary.prepared(),
            fingerprint,
            |bytes| body_class(bytes) == BodyClass::Boundary,
        )?;
        let body_boundary_max_bytes = vocabulary
            .iter()
            .filter_map(|(bytes, _)| {
                (body_class(bytes) == BodyClass::Boundary).then_some(bytes.len())
            })
            .max()
            .unwrap_or(0);
        let body_invalid_trie = VocabTrie::build_byte_filtered_from_prepared(
            vocabulary.prepared(),
            fingerprint,
            |bytes| body_class(bytes) == BodyClass::Invalid,
        )?;
        let body_invalid_max_bytes = vocabulary
            .iter()
            .filter_map(|(bytes, _)| {
                (body_class(bytes) == BodyClass::Invalid).then_some(bytes.len())
            })
            .max()
            .unwrap_or(0);
        let body_max_bytes = vocabulary
            .iter()
            .filter_map(|(bytes, _)| body_extra(bytes).then_some(bytes.len()))
            .max()
            .unwrap_or(0);
        let scalar_overflow_trie = VocabTrie::build_byte_filtered_from_prepared(
            vocabulary.prepared(),
            fingerprint,
            |bytes| safe_scalar_count(bytes).is_some_and(|n| usize::from(n) > MAX_BOUNDED_LEN),
        )?;
        let max_byte_len = vocabulary
            .iter()
            .filter_map(|(bytes, _)| safe_scalar_count(bytes).map(|_| bytes.len()))
            .max()
            .unwrap_or(0);
        let max_scalar_len = vocabulary
            .iter()
            .filter_map(|(bytes, _)| safe_scalar_count(bytes).map(usize::from))
            .max()
            .unwrap_or(0);
        let byte_len_le = cumulative_masks(vocabulary, |bytes| Some(bytes.len()))?;
        let scalar_len_le = cumulative_masks(vocabulary, |bytes| {
            safe_scalar_count(bytes).map(usize::from)
        })?;
        let table_bytes = byte_len_le
            .iter()
            .chain(scalar_len_le.iter())
            .try_fold(0usize, |acc, mask| {
                acc.checked_add(mask.len().checked_mul(size_of::<u64>())?)
            })
            .and_then(|bytes| bytes.checked_add(2 * MAX_BOUNDED_LEN * size_of::<Box<[u64]>>()))
            .ok_or_else(slice_error)?;
        let retained_bytes = retained_bytes(&safe_trie, &uncertain_trie, &token_mask)?
            .checked_add(table_bytes)
            .and_then(|bytes| bytes.checked_add(scalar_overflow_trie.heap_bytes()))
            .and_then(|bytes| bytes.checked_add(body_extra_trie.heap_bytes()))
            .and_then(|bytes| bytes.checked_add(body_uncertain_trie.heap_bytes()))
            .and_then(|bytes| bytes.checked_add(body_boundary_trie.heap_bytes()))
            .and_then(|bytes| bytes.checked_add(body_invalid_trie.heap_bytes()))
            .and_then(|bytes| {
                bytes.checked_add(body_extra_mask.len().checked_mul(size_of::<u64>())?)
            })
            .ok_or_else(slice_error)?;
        if retained_bytes > byte_budget {
            return Ok(None);
        }
        Ok(Some(Arc::new(Self {
            safe: TokenSlice {
                token_mask,
                filtered_trie: safe_trie,
                scalar_count: None,
                max_byte_len,
            },
            uncertain_trie,
            body_extra: TokenSlice {
                token_mask: body_extra_mask,
                filtered_trie: body_extra_trie,
                scalar_count: None,
                max_byte_len: body_max_bytes,
            },
            body_uncertain_trie,
            body_boundary_trie,
            body_boundary_max_bytes,
            body_invalid_trie,
            body_invalid_max_bytes,
            scalar_overflow_trie,
            retained_bytes,
            byte_len_le,
            scalar_len_le,
            max_scalar_len,
        })))
    }

    pub(crate) fn bounded_mask(
        &self,
        max_bytes: usize,
        max_scalars: Option<usize>,
    ) -> Option<BoundedMask<'_>> {
        let bytes = if max_bytes >= self.safe.max_byte_len {
            self.safe.token_mask()
        } else if max_bytes == 0 {
            &[]
        } else {
            &**self.byte_len_le.get(max_bytes - 1)?
        };
        let scalars = match max_scalars {
            None => None,
            Some(limit) if limit >= self.max_scalar_len => None,
            Some(0) => Some(&[][..]),
            Some(limit) => Some(&**self.scalar_len_le.get(limit.min(MAX_BOUNDED_LEN) - 1)?),
        };
        let overflow = max_scalars.is_some_and(|n| n > MAX_BOUNDED_LEN && n < self.max_scalar_len);
        Some(BoundedMask {
            bytes,
            scalars,
            overflow,
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BoundedMask<'a> {
    bytes: &'a [u64],
    scalars: Option<&'a [u64]>,
    overflow: bool,
}

impl BoundedMask<'_> {
    pub(crate) fn needs_overflow_walk(&self) -> bool {
        self.overflow
    }
    pub(crate) fn word(&self, index: usize) -> u64 {
        let word = self.bytes.get(index).copied().unwrap_or(0);
        match self.scalars {
            Some(scalars) => word & scalars.get(index).copied().unwrap_or(0),
            None => word,
        }
    }

    pub(crate) fn words(&self) -> usize {
        self.bytes.len()
    }
}

fn cumulative_masks(
    vocabulary: &CanonicalVocabulary,
    measure: impl Fn(&[u8]) -> Option<usize>,
) -> Result<Box<[Box<[u64]>]>, CompileError> {
    let words = vocabulary.mask_vocab_size().div_ceil(u64::BITS as usize);
    let mut exact = Vec::new();
    exact
        .try_reserve_exact(MAX_BOUNDED_LEN)
        .map_err(|_| slice_error())?;
    for _ in 0..MAX_BOUNDED_LEN {
        let mut mask = Vec::new();
        mask.try_reserve_exact(words).map_err(|_| slice_error())?;
        mask.resize(words, 0u64);
        exact.push(mask.into_boxed_slice());
    }
    for (bytes, ids) in vocabulary.iter() {
        if safe_scalar_count(bytes).is_none() {
            continue;
        }
        let Some(size) = measure(bytes) else { continue };
        if size == 0 || size > MAX_BOUNDED_LEN {
            continue;
        }
        for &id in ids {
            let index = usize::try_from(id).map_err(|_| slice_error())?;
            let word = exact[size - 1]
                .get_mut(index / u64::BITS as usize)
                .ok_or_else(slice_error)?;
            *word |= 1u64 << (index % u64::BITS as usize);
        }
    }
    for level in 1..exact.len() {
        let (previous, current) = exact.split_at_mut(level);
        for (bits, prefix) in current[0].iter_mut().zip(previous[level - 1].iter()) {
            *bits |= *prefix;
        }
    }
    Ok(exact.into_boxed_slice())
}

impl TokenSliceCatalog {
    pub(crate) fn safe(&self) -> &TokenSlice {
        &self.safe
    }

    pub(crate) fn uncertain_trie(&self) -> &VocabTrie {
        &self.uncertain_trie
    }

    pub(crate) fn body_extra(&self) -> &TokenSlice {
        &self.body_extra
    }

    pub(crate) fn body_uncertain_trie(&self) -> &VocabTrie {
        &self.body_uncertain_trie
    }

    pub(crate) fn body_boundary_trie(&self) -> &VocabTrie {
        &self.body_boundary_trie
    }

    pub(crate) fn body_boundary_max_bytes(&self) -> usize {
        self.body_boundary_max_bytes
    }

    pub(crate) fn body_invalid_trie(&self) -> &VocabTrie {
        &self.body_invalid_trie
    }

    pub(crate) fn body_invalid_max_bytes(&self) -> usize {
        self.body_invalid_max_bytes
    }

    pub(crate) fn scalar_overflow_trie(&self) -> &VocabTrie {
        &self.scalar_overflow_trie
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
}

impl TokenSlice {
    pub(crate) fn token_mask(&self) -> &[u64] {
        &self.token_mask
    }

    pub(crate) fn filtered_trie(&self) -> &VocabTrie {
        &self.filtered_trie
    }

    pub(crate) fn scalar_count(&self) -> Option<u16> {
        self.scalar_count
    }

    pub(crate) fn max_byte_len(&self) -> usize {
        self.max_byte_len
    }
}

fn safe_scalar_count(bytes: &[u8]) -> Option<u16> {
    if bytes.is_empty()
        || bytes
            .iter()
            .any(|byte| *byte < 0x20 || matches!(*byte, b'"' | b'\\'))
    {
        return None;
    }
    let text = std::str::from_utf8(bytes).ok()?;
    u16::try_from(text.chars().count()).ok()
}

fn safe_token_mask(vocabulary: &CanonicalVocabulary) -> Result<Box<[u64]>, CompileError> {
    token_mask_where(vocabulary, |bytes| safe_scalar_count(bytes).is_some())
}

fn body_prefix(bytes: &[u8]) -> bool {
    body_class(bytes) == BodyClass::Prefix
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BodyClass {
    Prefix,
    Boundary,
    Invalid,
}

fn body_class(bytes: &[u8]) -> BodyClass {
    let mut decoder = JsonStringDecoder::new();
    if bytes.is_empty() {
        return BodyClass::Invalid;
    }
    for &byte in bytes {
        if byte == b'"' && decoder.at_boundary() {
            return BodyClass::Boundary;
        }
        if decoder.push(byte) == DecodeStep::Invalid {
            return BodyClass::Invalid;
        }
    }
    BodyClass::Prefix
}

fn body_extra(bytes: &[u8]) -> bool {
    safe_scalar_count(bytes).is_none() && body_prefix(bytes)
}

fn token_mask_where(
    vocabulary: &CanonicalVocabulary,
    include: impl Fn(&[u8]) -> bool,
) -> Result<Box<[u64]>, CompileError> {
    let words = vocabulary.mask_vocab_size().div_ceil(u64::BITS as usize);
    let mut mask = Vec::new();
    mask.try_reserve_exact(words).map_err(|_| slice_error())?;
    mask.resize(words, 0u64);
    for (bytes, ids) in vocabulary.iter() {
        if !include(bytes) {
            continue;
        }
        for &id in ids {
            let index = usize::try_from(id).map_err(|_| slice_error())?;
            let word = mask
                .get_mut(index / u64::BITS as usize)
                .ok_or_else(slice_error)?;
            *word |= 1u64 << (index % u64::BITS as usize);
        }
    }
    Ok(mask.into_boxed_slice())
}

fn estimated_build_bytes(vocabulary: &CanonicalVocabulary) -> Result<usize, CompileError> {
    let body = vocabulary
        .iter()
        .filter(|(bytes, _)| safe_scalar_count(bytes).is_none())
        .try_fold(0usize, |total, (bytes, ids)| {
            total
                .checked_add(bytes.len().checked_mul(48)?)?
                .checked_add(ids.len().checked_mul(16)?)
        })
        .and_then(|bytes| {
            bytes.checked_add(vocabulary.mask_vocab_size().div_ceil(64).checked_mul(8)?)
        })
        .ok_or_else(slice_error)?;
    let overflow = vocabulary
        .iter()
        .filter(|(bytes, _)| {
            safe_scalar_count(bytes).is_some_and(|n| usize::from(n) > MAX_BOUNDED_LEN)
        })
        .try_fold(0usize, |total, (bytes, ids)| {
            total
                .checked_add(bytes.len().checked_mul(24)?)?
                .checked_add(ids.len().checked_mul(8)?)
        })
        .ok_or_else(slice_error)?;
    vocabulary
        .prepared()
        .total_bytes()
        .checked_mul(24)
        .and_then(|bytes| bytes.checked_add(vocabulary.prepared().total_ids().checked_mul(8)?))
        .and_then(|bytes| {
            bytes.checked_add(vocabulary.mask_vocab_size().div_ceil(64).checked_mul(8)?)
        })
        .and_then(|bytes| bytes.checked_add(7 * 16))
        .and_then(|bytes| bytes.checked_add(overflow))
        .and_then(|bytes| bytes.checked_add(body))
        .and_then(|bytes| {
            let table = vocabulary
                .mask_vocab_size()
                .div_ceil(64)
                .checked_mul(size_of::<u64>())?
                .checked_add(size_of::<Box<[u64]>>())?
                .checked_mul(2 * MAX_BOUNDED_LEN)?;
            bytes
                .checked_add(table)?
                .checked_add(size_of::<TokenSliceCatalog>())
        })
        .ok_or_else(slice_error)
}

fn retained_bytes(
    safe: &VocabTrie,
    uncertain: &VocabTrie,
    mask: &[u64],
) -> Result<usize, CompileError> {
    safe.heap_bytes()
        .checked_add(uncertain.heap_bytes())
        .and_then(|bytes| bytes.checked_add(mask.len().checked_mul(size_of::<u64>())?))
        .and_then(|bytes| bytes.checked_add(size_of::<TokenSliceCatalog>()))
        .ok_or_else(slice_error)
}

fn slice_error() -> CompileError {
    CompileError::new(
        ErrorCode::InternalLimitExceeded,
        Stage::L4Bind,
        "token slice catalogue allocation or size limit exceeded",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::build_vocabulary;
    use rustc_hash::FxHashMap;

    #[test]
    fn body_prefix_partition_respects_the_first_lexical_boundary() {
        for bytes in [
            b"a".as_slice(),
            br#"\u0061"#,
            br#"\""#,
            br#"\uD83D\uDE00"#,
            b"\xf0\x9f",
            b"\\",
            br#"\uD83D"#,
        ] {
            assert!(body_prefix(bytes), "{bytes:?}");
        }
        for bytes in [
            b"".as_slice(),
            b"\"\n",
            b"a\" ",
            b"a\n",
            b"\xf0\x9f\"",
            b"\xff",
            br#"\uDE00"#,
            br#"\uD83Dx"#,
        ] {
            assert!(!body_prefix(bytes), "{bytes:?}");
        }
    }

    #[test]
    fn safe_tokens_require_complete_boundary_free_utf8() {
        assert_eq!(safe_scalar_count(b"a"), Some(1));
        assert_eq!(safe_scalar_count("aλ😀".as_bytes()), Some(3));
        for bytes in [b"".as_slice(), b"\"", b"\\", b"\x1f", b"\xc3", b"\xff"] {
            assert_eq!(safe_scalar_count(bytes), None, "{bytes:?}");
        }
    }

    #[test]
    fn catalogue_mask_excludes_structural_partial_sparse_and_eos_ids() {
        let mut tokens = FxHashMap::default();
        tokens.insert(b"a".to_vec(), vec![0, 7]);
        tokens.insert(b"\"".to_vec(), vec![1]);
        tokens.insert(b"\\".to_vec(), vec![2]);
        tokens.insert(vec![0xf0, 0x9f], vec![3]);
        tokens.insert(b"\x00".to_vec(), vec![4]);
        tokens.insert(b"z".to_vec(), vec![100]);
        let vocabulary = build_vocabulary(127, tokens).expect("vocabulary");
        let canonical = CanonicalVocabulary::from_vocabulary(&vocabulary).expect("canonical");
        let catalog = TokenSliceCatalog::try_build(&canonical)
            .expect("catalogue build")
            .expect("catalogue admitted");
        let mask = catalog.safe().token_mask();
        for id in [0usize, 7, 100] {
            assert_ne!(mask[id / 64] & (1u64 << (id % 64)), 0, "id {id}");
        }
        for id in [1usize, 2, 3, 4, 127] {
            assert_eq!(mask[id / 64] & (1u64 << (id % 64)), 0, "id {id}");
        }
        assert!(catalog.retained_bytes() <= MAX_CATALOG_BYTES);
        assert!(TokenSliceCatalog::try_build_with_budget(&canonical, 0)
            .expect("zero-budget fallback")
            .is_none());
    }

    #[test]
    fn bounded_masks_match_independent_byte_and_scalar_counts() {
        let pieces = [
            "a",
            "ab",
            "\u{e9}",
            "\u{20ac}",
            "\u{1f600}",
            "e\u{301}",
            "abcdefghijklmnopqrstuvwxyz0123456789",
        ];
        let tokens = pieces
            .iter()
            .enumerate()
            .map(|(id, text)| (text.as_bytes().to_vec(), vec![id as u32]))
            .collect();
        let vocabulary = build_vocabulary(127, tokens).unwrap();
        let canonical = CanonicalVocabulary::from_vocabulary(&vocabulary).unwrap();
        let catalog = TokenSliceCatalog::try_build(&canonical).unwrap().unwrap();
        for bytes in [0, 1, 2, 3, 4, 8, 32, 36, usize::MAX] {
            for scalars in [
                None,
                Some(0),
                Some(1),
                Some(2),
                Some(32),
                Some(36),
                Some(100),
            ] {
                let mask = catalog.bounded_mask(bytes, scalars).expect("exact bound");
                for (id, piece) in pieces.iter().enumerate() {
                    assert_eq!(
                        mask.word(id / 64) & (1 << (id % 64)) != 0,
                        piece.len() <= bytes && scalars.is_none_or(|n| piece.chars().count() <= n),
                        "bytes={bytes}, scalars={scalars:?}, piece={piece:?}"
                    );
                }
            }
        }
        assert!(
            catalog.bounded_mask(33, None).is_none(),
            "unrepresented byte bound"
        );
        assert!(catalog
            .bounded_mask(100, Some(33))
            .unwrap()
            .needs_overflow_walk());
        assert!(estimated_build_bytes(&canonical).unwrap() >= catalog.retained_bytes());
    }
}
