//! The single vocabulary choke-point: the one validated constructor every consumer goes through.

use rustc_hash::FxHashMap as HashMap;

use super::error::Error;
use super::vocabulary::Vocabulary;
use super::{Token, TokenId};
use crate::error::VocabError;

/// Token bytes mapped to the ids that decode to them.
pub type TokenMap = HashMap<Token, Vec<TokenId>>;

/// The token map [`build_vocabulary_from_str_keys`] accepts, keyed by `String`.
pub type StringTokenMap = HashMap<String, Vec<TokenId>>;

/// Validated `Vocabulary` constructor: rejects empty-byte tokens, an empty id list, and an EOS id
/// present in the token map. Duplicate byte sequences merge; raw non-UTF-8 bytes are valid.
pub fn build_vocabulary(eos_token_id: TokenId, tokens: TokenMap) -> Result<Vocabulary, VocabError> {
    Vocabulary::from_raw_bytes_map(eos_token_id, tokens).map_err(|e| map_error(e, eos_token_id))
}

/// Validated `Vocabulary` constructor over `String` keys. Same rules as `build_vocabulary`.
pub fn build_vocabulary_from_str_keys(
    eos_token_id: TokenId,
    tokens: StringTokenMap,
) -> Result<Vocabulary, VocabError> {
    Vocabulary::from_raw_string_map(eos_token_id, tokens).map_err(|e| map_error(e, eos_token_id))
}

/// Validated `Vocabulary` constructor from serialized fast-tokenizer JSON; same choke-point as `build_vocabulary`.
#[cfg(feature = "tokenizer-processing")]
pub fn build_vocabulary_from_tokenizer_json(
    json: &str,
    eos_token_id: TokenId,
) -> Result<Vocabulary, VocabError> {
    Vocabulary::from_tokenizer_json(json, eos_token_id).map_err(|e| map_error(e, eos_token_id))
}

/// Converts vocabulary construction errors to the public `VocabError` contract.
#[allow(unreachable_patterns)]
fn map_error(e: Error, eos_token_id: TokenId) -> VocabError {
    match e {
        Error::EmptyToken => VocabError::empty_token(),
        Error::EmptyTokenIds => VocabError::malformed_tokenizer("token has an empty id list"),
        Error::EOSTokenDisallowed => VocabError::malformed_tokenizer_for(
            crate::primitives::TokenId(eos_token_id),
            "EOS token id present in the token map",
        ),
        #[cfg(feature = "tokenizer-processing")]
        Error::TokenizerJsonTooLarge { .. } => {
            VocabError::limit_exceeded("tokenizer JSON exceeds the accepted byte limit")
        }
        #[cfg(feature = "tokenizer-processing")]
        Error::TokenizersError(_) => {
            VocabError::malformed_tokenizer("the tokenizer library rejected the tokenizer data")
        }
        #[cfg(feature = "tokenizer-processing")]
        Error::UnsupportedTokenizer { .. } => {
            VocabError::malformed_tokenizer("tokenizer model or decoder is unsupported")
        }
        #[cfg(feature = "tokenizer-processing")]
        Error::UnsupportedByTokenProcessor => {
            VocabError::malformed_tokenizer("tokenizer is unsupported by the token processor")
        }
        #[cfg(feature = "tokenizer-processing")]
        Error::DecoderUnpackingFailed => {
            VocabError::malformed_tokenizer("tokenizer decoder could not be unpacked")
        }
        #[cfg(feature = "tokenizer-processing")]
        Error::ByteProcessorFailed => {
            VocabError::malformed_tokenizer("byte-level token processing failed")
        }
        #[cfg(feature = "tokenizer-processing")]
        Error::ByteFallbackProcessorFailed => {
            VocabError::malformed_tokenizer("byte-fallback token processing failed")
        }
        _ => VocabError::malformed_tokenizer("vocabulary construction rejected the token map"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorCode;

    fn map(entries: &[(&[u8], &[TokenId])]) -> HashMap<Token, Vec<TokenId>> {
        entries
            .iter()
            .map(|(bytes, ids)| (bytes.to_vec(), ids.to_vec()))
            .collect()
    }

    #[test]
    fn empty_byte_token_is_rejected() {
        let v = build_vocabulary(0, map(&[(b"", &[1])]));
        assert_eq!(v.unwrap_err().code, ErrorCode::EmptyToken);
    }

    #[test]
    fn str_keyed_choke_point_rejects_empty_and_eos() {
        let mut sm: HashMap<String, Vec<TokenId>> = HashMap::default();
        sm.insert(String::new(), vec![1]);
        assert_eq!(
            build_vocabulary_from_str_keys(0, sm).unwrap_err().code,
            ErrorCode::EmptyToken
        );
        let mut ok: HashMap<String, Vec<TokenId>> = HashMap::default();
        ok.insert("a".to_string(), vec![7]);
        let v = build_vocabulary_from_str_keys(0, ok).expect("valid");
        assert_eq!(v.token_ids("a"), Some(&[7][..]));
    }

    #[test]
    fn str_and_bytes_keyed_construction_agree() {
        let mut sm: HashMap<String, Vec<TokenId>> = HashMap::default();
        sm.insert("1".to_string(), vec![1]);
        sm.insert("a".to_string(), vec![2]);
        let by_str = build_vocabulary_from_str_keys(3, sm).expect("valid");
        let by_bytes = build_vocabulary(3, map(&[(b"1", &[1]), (b"a", &[2])])).expect("valid");
        assert_eq!(by_str.token_ids("1"), by_bytes.token_ids(b"1"));
        assert_eq!(by_str.token_ids("a"), by_bytes.token_ids(b"a"));
        assert_eq!(by_str.eos_token_id(), by_bytes.eos_token_id());
        assert_eq!(by_str.len(), by_bytes.len());
    }

    #[test]
    fn duplicate_bytes_different_ids_preserved() {
        let v = build_vocabulary(0, map(&[(b"ab", &[1, 2])])).expect("valid");
        let ids = v.token_ids(b"ab").expect("present");
        assert!(ids.contains(&1) && ids.contains(&2));
    }

    #[test]
    fn raw_non_utf8_payload_is_accepted() {
        let v = build_vocabulary(0, map(&[(&[0xFF], &[7])])).expect("valid");
        assert_eq!(v.token_ids([0xFFu8]), Some(&[7][..]));
    }

    #[test]
    fn eos_in_token_map_is_malformed_metadata() {
        let e = build_vocabulary(3, map(&[(b"x", &[3])])).unwrap_err();
        assert_eq!(e.code, ErrorCode::MalformedTokenizer);
        assert_eq!(e.token, Some(crate::primitives::TokenId(3)));
    }

    #[test]
    fn no_public_construction_path_admits_an_empty_token() {
        use super::super::vocabulary::Vocabulary;
        assert_eq!(
            build_vocabulary(0, map(&[(b"", &[1])])).unwrap_err().code,
            ErrorCode::EmptyToken
        );
        let mut v = Vocabulary::new(0);
        assert!(v.try_insert(Vec::<u8>::new(), 1).is_err());
        let mut m: HashMap<Token, Vec<TokenId>> = HashMap::default();
        m.insert(Vec::new(), vec![1]);
        assert!(Vocabulary::from_raw_bytes_map(0u32, m).is_err());
        let mut sm: HashMap<String, Vec<TokenId>> = HashMap::default();
        sm.insert(String::new(), vec![1]);
        assert!(Vocabulary::from_raw_string_map(0u32, sm).is_err());
    }
}
