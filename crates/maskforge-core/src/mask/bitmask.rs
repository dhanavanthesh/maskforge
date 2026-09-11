// Derived from outlines-core; see PROVENANCE.md.

//! Stores allowed-token masks as packed `u32` words.

use crate::error::{CompileError, ErrorCode, Stage};
use crate::primitives::TokenId;

const WORD_BITS: usize = 32;

/// A dense allowed-token bitmask over `vocab_size` tokens, packed into little-endian `u32` words.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Bitmask {
    words: Vec<u32>,
    vocab_size: usize,
}

impl Bitmask {
    /// Allocates an all-zero mask sized for `vocab_size` tokens (`ceil(vocab_size/32)` words).
    #[must_use]
    pub fn zeros(vocab_size: usize) -> Self {
        Self {
            words: vec![0; vocab_size.div_ceil(WORD_BITS)],
            vocab_size,
        }
    }

    /// Builds a mask from packed words, clearing unused high bits.
    #[must_use]
    pub fn from_words(words: &[u32], vocab_size: usize) -> Self {
        let need = vocab_size.div_ceil(WORD_BITS);
        if need == 0 {
            return Self {
                words: Vec::new(),
                vocab_size,
            };
        }
        let mut buf = vec![0u32; need];
        let take = words.len().min(need);
        buf[..take].copy_from_slice(&words[..take]);
        let remainder = vocab_size % WORD_BITS;
        if remainder != 0 {
            buf[need - 1] &= (1u32 << remainder) - 1;
        }
        Self {
            words: buf,
            vocab_size,
        }
    }

    /// Sets the bit for `id`. Fails (never truncates) when `id` is outside `0..vocab_size`.
    pub fn set(&mut self, id: TokenId) -> Result<(), CompileError> {
        let idx = id.get() as usize;
        if idx >= self.vocab_size {
            return Err(CompileError::new(
                ErrorCode::ArtifactOutOfBounds,
                Stage::ArtifactValidate,
                "token id is outside the mask vocab size",
            )
            .with_observed(idx.to_string()));
        }
        self.words[idx / WORD_BITS] |= 1u32 << (idx % WORD_BITS);
        Ok(())
    }

    /// Returns whether the bit for `id` is set (`false` for out-of-range ids).
    #[must_use]
    pub fn get(&self, id: TokenId) -> bool {
        let idx = id.get() as usize;
        if idx >= self.vocab_size {
            return false;
        }
        self.words[idx / WORD_BITS] & (1u32 << (idx % WORD_BITS)) != 0
    }

    /// The packed words (the wire format): `ceil(vocab_size/32)` little-endian `u32`s.
    #[must_use]
    pub fn as_words(&self) -> &[u32] {
        &self.words
    }

    /// The number of tokens this mask covers.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns the number of set token bits.
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|w| usize::try_from(w.count_ones()).expect("popcount of a u32 fits usize"))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_count_is_ceil_div_32() {
        assert_eq!(Bitmask::zeros(0).as_words().len(), 0);
        assert_eq!(Bitmask::zeros(1).as_words().len(), 1);
        assert_eq!(Bitmask::zeros(32).as_words().len(), 1);
        assert_eq!(Bitmask::zeros(33).as_words().len(), 2);
    }

    #[test]
    fn set_get_and_unused_high_bits_zero() {
        let mut m = Bitmask::zeros(40);
        m.set(TokenId(0)).unwrap();
        m.set(TokenId(39)).unwrap();
        assert!(m.get(TokenId(0)));
        assert!(m.get(TokenId(39)));
        assert!(!m.get(TokenId(1)));
        assert_eq!(m.count_ones(), 2);
        assert_eq!(m.as_words()[1] & !0x0000_00FF, 0);
    }

    #[test]
    fn oversized_id_fails_not_truncates() {
        let mut m = Bitmask::zeros(10);
        assert_eq!(
            m.set(TokenId(10)).unwrap_err().code,
            ErrorCode::ArtifactOutOfBounds
        );
        assert!(!m.get(TokenId(10)));
    }

    #[test]
    fn word_boundary_ids_31_32_63() {
        let mut m = Bitmask::zeros(64);
        for id in [31u32, 32, 63] {
            m.set(TokenId(id)).unwrap();
        }
        assert!(m.get(TokenId(31)) && m.get(TokenId(32)) && m.get(TokenId(63)));
        assert!(!m.get(TokenId(30)) && !m.get(TokenId(33)));
        assert_eq!(m.as_words().len(), 2);
        assert_eq!(m.count_ones(), 3);
    }

    #[test]
    fn empty_vocab_has_zero_words() {
        let m = Bitmask::zeros(0);
        assert_eq!(m.as_words().len(), 0);
        assert_eq!(m.count_ones(), 0);
        assert!(!m.get(TokenId(0)));
    }

    #[test]
    fn non_multiple_of_32_vocab_sizes_63_and_65() {
        assert_eq!(Bitmask::zeros(63).as_words().len(), 2);
        assert_eq!(Bitmask::zeros(65).as_words().len(), 3);
        let mut m = Bitmask::zeros(65);
        m.set(TokenId(64)).unwrap();
        assert!(m.get(TokenId(64)));
        assert!(!m.get(TokenId(63)));
        assert_eq!(
            m.as_words()[2] & !0b1,
            0,
            "id 64 is bit 0 of the third word"
        );
    }

    #[test]
    fn token_id_65_is_out_of_range_for_a_65_vocab() {
        let mut m = Bitmask::zeros(65);
        assert_eq!(
            m.set(TokenId(65)).unwrap_err().code,
            ErrorCode::ArtifactOutOfBounds
        );
    }

    #[test]
    fn repeated_writes_to_the_same_bit_are_idempotent() {
        let mut m = Bitmask::zeros(40);
        m.set(TokenId(5)).unwrap();
        m.set(TokenId(5)).unwrap();
        m.set(TokenId(5)).unwrap();
        assert!(m.get(TokenId(5)));
        assert_eq!(m.count_ones(), 1);
    }

    #[test]
    fn far_oversized_token_id_fails_not_truncates() {
        let mut m = Bitmask::zeros(10);
        assert_eq!(
            m.set(TokenId(u32::MAX)).unwrap_err().code,
            ErrorCode::ArtifactOutOfBounds
        );
        assert!(!m.get(TokenId(u32::MAX)));
    }

    #[test]
    fn from_words_masks_dirty_high_bits_beyond_vocab_size() {
        assert_eq!(Bitmask::from_words(&[u32::MAX], 1).as_words(), &[1]);
        let m31 = Bitmask::from_words(&[u32::MAX], 31);
        assert_eq!(m31.as_words(), &[0x7FFF_FFFF]);
        assert_eq!(m31.count_ones(), 31);
        let m32 = Bitmask::from_words(&[u32::MAX], 32);
        assert_eq!(m32.as_words(), &[u32::MAX]);
        assert_eq!(m32.count_ones(), 32);
        let m33 = Bitmask::from_words(&[u32::MAX, u32::MAX], 33);
        assert_eq!(m33.as_words(), &[u32::MAX, 1]);
        assert_eq!(m33.count_ones(), 33);
    }

    #[test]
    fn from_words_at_vocab_size_zero_does_not_panic_and_has_no_words() {
        let m = Bitmask::from_words(&[u32::MAX], 0);
        assert_eq!(m.as_words(), &[] as &[u32]);
        assert_eq!(m.count_ones(), 0);
    }

    #[test]
    fn from_words_count_ones_never_counts_out_of_range_bits() {
        for size in [1usize, 31, 32, 33, 65] {
            let words = vec![u32::MAX; size.div_ceil(32)];
            let m = Bitmask::from_words(&words, size);
            assert_eq!(m.count_ones(), size, "size {size}");
        }
    }
}
