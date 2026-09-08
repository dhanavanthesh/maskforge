//! Creates `Vocabulary` manually or from pretrained large language model.

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
#[cfg(feature = "tokenizer-processing")]
use tokenizers::normalizers::Sequence;
#[cfg(feature = "tokenizer-processing")]
use tokenizers::{NormalizerWrapper, Tokenizer};

use super::error::{Error, Result};
#[cfg(feature = "huggingface-hub")]
use super::locator::{HFLocator, Locator};
#[cfg(feature = "tokenizer-processing")]
use super::processor::TokenProcessor;
#[cfg(feature = "huggingface-hub")]
use super::FromPretrainedParameters;
use super::{Token, TokenId};

/// Largest tokenizer JSON accepted before parsing.
#[cfg(feature = "tokenizer-processing")]
pub const MAX_TOKENIZER_JSON_BYTES: usize = 64 * 1024 * 1024;

/// Vocabulary for a large language model.
#[cfg_attr(
    feature = "huggingface-hub",
    doc = r##"
### Create a vocabulary from a pretrained model.
```no_run
use maskforge_core::Vocabulary;

let vocabulary = Vocabulary::from_pretrained("openai-community/gpt2", None);
```

### Create a vocabulary from a pretrained model with some additional parameters.
```no_run
use maskforge_core::{FromPretrainedParameters, Vocabulary};

let params = FromPretrainedParameters {
    revision: "607a30d783dfa663caf39e06633721c8d4cfcd7e".to_string(),
    ..Default::default()
};
let vocabulary = Vocabulary::from_pretrained("openai-community/gpt2", Some(params));
```

### Create an empty vocabulary and manually insert some tokens.
```
use maskforge_core::Vocabulary;

let eos_token_id = 1;
let mut vocabulary = Vocabulary::new(eos_token_id);

vocabulary.try_insert("token", 0).expect("New token inserted");
assert_eq!(vocabulary.token_ids("token"), Some(&[0][..]));
assert_eq!(vocabulary.tokens().len(), 1);
assert_eq!(vocabulary.eos_token_id(), eos_token_id);

vocabulary.remove("token");
assert_eq!(vocabulary.token_ids("token"), None);
```
"##
)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Vocabulary {
    eos_token_id: TokenId,
    tokens: HashMap<Token, Vec<TokenId>>,
}

impl Vocabulary {
    /// Creates an empty vocabulary.
    pub fn new(eos_token_id: TokenId) -> Self {
        Self {
            eos_token_id,
            tokens: HashMap::default(),
        }
    }

    /// Creates the vocabulary of pre-trained model from Hugging Face Hub.
    #[cfg(feature = "huggingface-hub")]
    pub fn from_pretrained(
        model: &str,
        parameters: Option<FromPretrainedParameters>,
    ) -> Result<Self> {
        Self::from_pretrained_with_locator::<HFLocator>(model, parameters)
    }

    #[doc(hidden)]
    #[inline(always)]
    #[cfg(feature = "huggingface-hub")]
    fn from_pretrained_with_locator<L: Locator>(
        model: &str,
        parameters: Option<FromPretrainedParameters>,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_pretrained(model, parameters.clone())?;

        // Locate eos_token_id in defined locations.
        let eos_token_id = match L::locate_eos_token_id(model, &tokenizer, &parameters) {
            Ok(id) => id,
            Err(tried) => {
                let reason = if tried.is_empty() {
                    "EOS token id".to_string()
                } else {
                    format!("EOS token id not found; tried: {}", tried.join("; "))
                };
                return Err(Error::UnsupportedTokenizer {
                    model: model.to_string(),
                    reason,
                });
            }
        };

        Self::from_tokenizer(tokenizer, eos_token_id, model)
    }

    /// Builds from fast-tokenizer JSON with a caller-resolved EOS id.
    /// Oversized input is rejected before parsing.
    #[cfg(feature = "tokenizer-processing")]
    pub fn from_tokenizer_json(json: &str, eos_token_id: TokenId) -> Result<Self> {
        if json.len() > MAX_TOKENIZER_JSON_BYTES {
            return Err(Error::TokenizerJsonTooLarge {
                actual: json.len(),
                limit: MAX_TOKENIZER_JSON_BYTES,
            });
        }
        let tokenizer = Tokenizer::from_bytes(json.as_bytes())?;
        Self::from_tokenizer(tokenizer, eos_token_id, "<inline tokenizer JSON>")
    }

    /// The one token-processing pipeline both tokenizer constructors share.
    #[cfg(feature = "tokenizer-processing")]
    fn from_tokenizer(
        mut tokenizer: Tokenizer,
        eos_token_id: TokenId,
        source: &str,
    ) -> Result<Self> {
        Self::filter_prepend_normalizers(&mut tokenizer);

        // Start building the vocabulary from eos_token_id and added tokens.
        let mut vocabulary = Vocabulary::new(eos_token_id);
        for (id, added_token) in tokenizer.get_added_tokens_decoder().iter() {
            if !added_token.special && id != &eos_token_id {
                vocabulary.try_insert(added_token.content.clone(), *id)?
            }
        }

        // Process each vocabulary token according to the tokenizer's level.
        let Ok(processor) = TokenProcessor::new(&tokenizer) else {
            return Err(Error::UnsupportedTokenizer {
                model: source.to_string(),
                reason: "Token processor".to_string(),
            });
        };
        for (token, token_id) in tokenizer.get_vocab(false) {
            if token_id != eos_token_id {
                let processed_token = processor.process(&token)?;
                vocabulary.try_insert(processed_token, token_id)?;
            }
        }

        Ok(vocabulary)
    }

    /// Returns all tokens with their token ids in vocabulary.
    pub fn tokens(&self) -> &HashMap<Token, Vec<TokenId>> {
        &self.tokens
    }

    /// Returns all token ids per provided token if available in the vocabulary.
    pub fn token_ids(&self, token: impl AsRef<[u8]>) -> Option<&[TokenId]> {
        self.tokens.get(token.as_ref()).map(Vec::as_slice)
    }

    /// Gets the identifier of the special end of the sentence token.
    pub fn eos_token_id(&self) -> TokenId {
        self.eos_token_id
    }

    /// Inserts a token with the given id. Idempotent: repeating `(token, id)` is a no-op.
    /// A different id under the same bytes is still kept.
    pub fn try_insert(&mut self, token: impl Into<Token>, id: TokenId) -> Result<(), Error> {
        if id == self.eos_token_id {
            return Err(Error::EOSTokenDisallowed);
        }
        let token = token.into();
        if token.is_empty() {
            return Err(Error::EmptyToken);
        }
        let ids = self.tokens.entry(token).or_default();
        if !ids.contains(&id) {
            ids.push(id);
        }
        Ok(())
    }

    /// Removes a given token from the vocabulary.
    pub fn remove(&mut self, token: impl Into<Token>) {
        let token = token.into();
        self.tokens.remove(&token);
    }

    /// Returns ordinary token ids plus EOS, not the sparse mask width.
    /// Overflowing counts saturate instead of wrapping.
    pub fn len(&self) -> usize {
        self.tokens
            .values()
            .try_fold(0usize, |acc, ids| acc.checked_add(ids.len()))
            .and_then(|n| n.checked_add(1))
            .unwrap_or(usize::MAX)
    }

    /// True when no ordinary tokens are present (EOS is not counted).
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Filters out `Prepend` kind of tokenizer's normalizers.
    #[cfg(feature = "tokenizer-processing")]
    fn filter_prepend_normalizers(tokenizer: &mut Tokenizer) {
        // Remove prepend normalizers so their marker characters do not enter token bytes.
        if let Some(normalizer) = tokenizer.get_normalizer() {
            match normalizer {
                NormalizerWrapper::Sequence(normalization_sequence) => {
                    let new_sequence = Sequence::new(
                        normalization_sequence
                            .as_ref()
                            .iter()
                            .filter_map(|normalizer| match normalizer {
                                NormalizerWrapper::Prepend(_) => None,
                                _ => Some(normalizer.clone()),
                            })
                            .collect(),
                    );
                    tokenizer.with_normalizer(new_sequence.into());
                }
                NormalizerWrapper::Prepend(_) => {
                    tokenizer.with_normalizer(None::<NormalizerWrapper>);
                }
                _ => {}
            }
        }
    }
}

/// Removes duplicate ids in place, keeping first-occurrence order (matches `try_insert`). Small
/// lists (the norm) dedup without a heap allocation; only large lists build a `HashSet`.
fn dedup_ids(ids: &mut Vec<TokenId>) {
    if ids.len() <= 1 {
        return;
    }
    // A short id list (the norm even for merged tokens) deduplicates in place, preserving
    // first-occurrence order, with no allocation - `contains` over <=8 kept ids is cheaper than a set.
    if ids.len() <= 8 {
        let mut write = 1;
        for read in 1..ids.len() {
            if !ids[..write].contains(&ids[read]) {
                ids[write] = ids[read];
                write += 1;
            }
        }
        ids.truncate(write);
        return;
    }
    let mut seen = HashSet::default();
    ids.retain(|id| seen.insert(*id));
}

impl std::fmt::Display for Vocabulary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Vocabulary object with eos_token_id={:?} and the following tokens to token_ids:",
            self.eos_token_id
        )?;
        for (token, token_ids) in self.tokens.iter() {
            writeln!(
                f,
                "{:?} -> {:?}",
                token
                    .iter()
                    .map(|b| format!("0x{:02X}", b))
                    .collect::<Vec<_>>(),
                token_ids
            )?;
        }
        Ok(())
    }
}

impl Vocabulary {
    /// Raw map constructor, not part of the public API; `build::build_vocabulary` validates first.
    /// Duplicate ids within one token are collapsed (see `try_insert`).
    pub(crate) fn from_raw_bytes_map(
        eos_token_id: TokenId,
        mut tokens: HashMap<Token, Vec<TokenId>>,
    ) -> Result<Self, Error> {
        if tokens.keys().any(|t| t.is_empty()) {
            return Err(Error::EmptyToken);
        }
        if tokens.values().any(Vec::is_empty) {
            return Err(Error::EmptyTokenIds);
        }
        if tokens.iter().any(|(_, ids)| ids.contains(&eos_token_id)) {
            return Err(Error::EOSTokenDisallowed);
        }
        for ids in tokens.values_mut() {
            dedup_ids(ids);
        }
        Ok(Vocabulary {
            eos_token_id,
            tokens,
        })
    }

    /// Raw map constructor over `String` keys. Same visibility rule as `from_raw_bytes_map`.
    pub(crate) fn from_raw_string_map(
        eos_token_id: TokenId,
        tokens: HashMap<String, Vec<TokenId>>,
    ) -> Result<Self, Error> {
        Ok(Vocabulary {
            eos_token_id,
            tokens: tokens
                .into_iter()
                .map(|(k, mut v)| {
                    if k.is_empty() {
                        Err(Error::EmptyToken)
                    } else if v.is_empty() {
                        Err(Error::EmptyTokenIds)
                    } else if v.contains(&eos_token_id) {
                        Err(Error::EOSTokenDisallowed)
                    } else {
                        dedup_ids(&mut v);
                        Ok((k.into_bytes(), v))
                    }
                })
                .collect::<Result<HashMap<Token, Vec<TokenId>>, _>>()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_identical_insertion_is_idempotent() {
        let mut v = Vocabulary::new(0);
        let before_len = v.len();
        v.try_insert("a", 5).unwrap();
        let after_first = v.len();
        v.try_insert("a", 5).unwrap();
        v.try_insert("a", 5).unwrap();
        assert_eq!(
            v.len(),
            after_first,
            "repeated identical insert must not grow len()"
        );
        assert_eq!(v.token_ids("a"), Some(&[5][..]));
        assert_eq!(after_first, before_len + 1);
    }

    #[test]
    fn dedup_ids_small_and_large_paths_agree_across_the_boundary() {
        // A reference dedup (order-preserving, always the HashSet path) the fast paths must match on
        // every length, so the len<=1 / len<=8 shortcuts never drift from the general case (M053).
        fn reference(ids: &[TokenId]) -> Vec<TokenId> {
            let mut seen = HashSet::default();
            ids.iter().copied().filter(|id| seen.insert(*id)).collect()
        }
        // Deterministic pseudo-random ids with heavy duplication, spanning the <=1, <=8, and >8 arms.
        let mut state = 0x1234_5678u32;
        for len in 0..40usize {
            let raw: Vec<TokenId> = (0..len)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    state % 6 // small id space forces frequent duplicates
                })
                .collect();
            let mut got = raw.clone();
            dedup_ids(&mut got);
            assert_eq!(got, reference(&raw), "len={len} raw={raw:?}");
        }
    }

    #[test]
    fn same_bytes_two_different_ids_are_both_kept() {
        let mut v = Vocabulary::new(0);
        v.try_insert("a", 5).unwrap();
        v.try_insert("a", 6).unwrap();
        let ids = v.token_ids("a").unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&5) && ids.contains(&6));
    }

    #[test]
    fn token_id_u32_max_is_accepted_as_an_ordinary_id() {
        let mut v = Vocabulary::new(0);
        v.try_insert("a", u32::MAX).unwrap();
        assert_eq!(v.token_ids("a"), Some(&[u32::MAX][..]));
    }

    #[test]
    fn sparse_ids_do_not_inflate_len_beyond_stored_entries() {
        let mut v = Vocabulary::new(0);
        v.try_insert("low", 1).unwrap();
        v.try_insert("high", 1000).unwrap();
        assert_eq!(v.len(), 3); // 2 tokens + 1 for EOS, not 1001
    }

    #[test]
    fn from_raw_bytes_map_dedups_duplicate_ids_within_one_token() {
        let mut m: HashMap<Token, Vec<TokenId>> = HashMap::default();
        m.insert(b"a".to_vec(), vec![5, 5, 6, 5]);
        let v = Vocabulary::from_raw_bytes_map(0, m).unwrap();
        let ids = v.token_ids("a").unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&5) && ids.contains(&6));
    }

    #[test]
    fn from_raw_string_map_dedups_duplicate_ids_within_one_token() {
        let mut m: HashMap<String, Vec<TokenId>> = HashMap::default();
        m.insert("a".to_string(), vec![5, 5, 6, 5]);
        let v = Vocabulary::from_raw_string_map(0, m).unwrap();
        let ids = v.token_ids("a").unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&5) && ids.contains(&6));
    }

    #[test]
    fn duplicate_removal_preserves_first_occurrence_order_not_sorted() {
        // Deliberately out of numeric order: 9, 3, 7. A sort would reorder to 3, 7, 9.
        let mut v = Vocabulary::new(0);
        for id in [9, 3, 7, 9, 3] {
            v.try_insert("a", id).unwrap();
        }
        assert_eq!(v.token_ids("a"), Some(&[9, 3, 7][..]));

        let mut m: HashMap<Token, Vec<TokenId>> = HashMap::default();
        m.insert(b"a".to_vec(), vec![9, 3, 7, 9, 3]);
        let raw = Vocabulary::from_raw_bytes_map(0, m).unwrap();
        assert_eq!(raw.token_ids("a"), Some(&[9, 3, 7][..]));
    }

    #[test]
    fn raw_bytes_and_string_map_construction_agree_on_id_order() {
        let ids = vec![9, 3, 7, 9, 3];
        let mut bytes_map: HashMap<Token, Vec<TokenId>> = HashMap::default();
        bytes_map.insert(b"a".to_vec(), ids.clone());
        let by_bytes = Vocabulary::from_raw_bytes_map(0, bytes_map).unwrap();

        let mut str_map: HashMap<String, Vec<TokenId>> = HashMap::default();
        str_map.insert("a".to_string(), ids);
        let by_str = Vocabulary::from_raw_string_map(0, str_map).unwrap();

        assert_eq!(by_bytes.token_ids("a"), by_str.token_ids("a"));
        assert_eq!(by_bytes.token_ids("a"), Some(&[9, 3, 7][..]));
    }

    #[test]
    fn basic_interface() {
        let eos_token_id = 3;
        let mut vocabulary = Vocabulary::new(eos_token_id);

        match vocabulary.try_insert("eos-token", eos_token_id) {
            Err(Error::EOSTokenDisallowed) => {}
            _ => unreachable!(),
        }

        // New empty vocabulary.
        assert_eq!(vocabulary.eos_token_id, eos_token_id);
        assert!(vocabulary.tokens.is_empty());

        for (token, id) in [("zero", 0), ("one", 1), ("two", 2)] {
            vocabulary.try_insert(token, id).expect("Insert failed");
            assert_eq!(vocabulary.token_ids(token), Some(&[id][..]));
        }
        assert_eq!(vocabulary.tokens.len(), 3);
        assert_eq!(vocabulary.tokens().len(), 3);

        // Confirm different types.
        vocabulary.try_insert(b"four", 4).expect("Insert failed");
        assert_eq!(vocabulary.token_ids("four"), Some(&[4][..]));

        vocabulary
            .try_insert(b"five".to_vec(), 5)
            .expect("Insert failed");
        assert_eq!(vocabulary.token_ids("five"), Some(&[5][..]));

        vocabulary
            .try_insert("six".to_string(), 6)
            .expect("Insert failed");
        assert_eq!(vocabulary.token_ids("six"), Some(&[6][..]));

        vocabulary.remove(b"four");
        assert_eq!(vocabulary.token_ids("four"), None);

        vocabulary.remove(b"five".to_vec());
        assert_eq!(vocabulary.token_ids("five"), None);

        vocabulary.remove("six".to_string());
        assert_eq!(vocabulary.token_ids("six"), None);
    }

    #[test]
    fn new_empty_vocabulary_from_hashmap() {
        let map: HashMap<Token, Vec<TokenId>> = HashMap::default();
        let vocabulary = Vocabulary::from_raw_bytes_map(1_u32, map).expect("Vocabulary failed");
        assert_eq!(vocabulary.eos_token_id, 1);
        assert!(vocabulary.tokens.is_empty());
    }

    #[cfg(feature = "huggingface-hub")]
    #[test]
    fn supported_pretrained_models() {
        // Support is expected for these:
        for model in [
            // GPT 2
            "openai-community/gpt2",
            // Llama 2
            "hf-internal-testing/Llama-2-7B-GPTQ",
            // Llama 3
            // OpenCoder: shares llama tokenizers
            "hf-internal-testing/llama-3-8b-internal",
            // Qwen
            "Qwen/Qwen2-7B-Instruct",
            // Salamandra
            "BSC-LT/salamandra-2b",
        ] {
            let vocabulary = Vocabulary::from_pretrained(model, None);
            match vocabulary {
                Ok(v) => {
                    assert_eq!(v.eos_token_id, v.eos_token_id());
                    assert!(!v.tokens.is_empty());
                }
                Err(_) => unreachable!(),
            }
        }
    }

    #[cfg(feature = "huggingface-hub")]
    #[test]
    fn pretrained_from_gpt2() {
        let model = "openai-community/gpt2";
        let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");
        let vocabulary = Vocabulary::from_pretrained(model, None).expect("Vocabulary failed");

        let v_eos = vocabulary.eos_token_id;
        assert_eq!(v_eos, vocabulary.eos_token_id());
        assert_eq!(v_eos, 50256);
        assert_eq!(
            tokenizer.id_to_token(v_eos).expect("Token not found"),
            "<|endoftext|>"
        );

        let token = "Ġal";
        let btoken = token.as_bytes().to_vec();
        assert!(vocabulary.token_ids(&btoken).is_none());
        assert!(tokenizer.token_to_id(token).is_some());

        for (v_token, t_token_expected) in [("abc", "abc"), (" O", "ĠO")] {
            let v_ids = vocabulary.token_ids(v_token.as_bytes());
            assert!(v_ids.is_some());
            for v_id in v_ids.unwrap() {
                let t_token = tokenizer
                    .id_to_token(*v_id)
                    .expect("Token id not found in tokenizer");
                assert_eq!(&t_token, t_token_expected);
            }
        }
    }

    #[cfg(feature = "huggingface-hub")]
    #[test]
    fn from_tokenizer_json_matches_from_pretrained_exactly() {
        let model = "openai-community/gpt2";
        let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");
        let json = tokenizer.to_string(false).expect("serialize tokenizer");
        let via_pretrained = Vocabulary::from_pretrained(model, None).expect("from_pretrained");
        let via_json = Vocabulary::from_tokenizer_json(&json, via_pretrained.eos_token_id())
            .expect("from_tokenizer_json");
        assert_eq!(via_json, via_pretrained);
    }

    #[cfg(feature = "huggingface-hub")]
    #[test]
    fn pretrained_from_llama() {
        use rustc_hash::FxHashSet as HashSet;

        let model = "hf-internal-testing/llama-tokenizer";
        let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");
        let vocabulary = Vocabulary::from_pretrained(model, None).expect("Vocabulary failed");

        let v_eos = vocabulary.eos_token_id;
        assert_eq!(v_eos, vocabulary.eos_token_id());
        assert_eq!(v_eos, 2);
        assert_eq!(
            tokenizer.id_to_token(v_eos).expect("Token not found"),
            "</s>"
        );

        let tests: &[(Vec<u8>, &[&str])] = &[
            ("abc".as_bytes().to_vec(), &["abc"]),
            (" al".as_bytes().to_vec(), &["▁al"]),
            (" O".as_bytes().to_vec(), &["▁O"]),
            ("   ".as_bytes().to_vec(), &["▁▁▁"]),
            (" ".as_bytes().to_vec(), &["▁", "<0x20>"]),
            ("a".as_bytes().to_vec(), &["a", "<0x61>"]),
            (vec![0xFF], &["<0xFF>"]),
            (vec![0x20], &["▁", "<0x20>"]),
        ];
        for (v_token, t_tokens_expected) in tests {
            let v_ids = vocabulary.token_ids(v_token);
            assert!(v_ids.is_some());

            let t_tokens = v_ids
                .unwrap()
                .iter()
                .map(|v_id| {
                    tokenizer
                        .id_to_token(*v_id)
                        .expect("Token id not found in tokenizer")
                })
                .collect::<HashSet<String>>();
            let expected = HashSet::from_iter(t_tokens_expected.iter().map(|s| s.to_string()));
            assert_eq!(t_tokens, expected)
        }
    }

    #[cfg(feature = "huggingface-hub")]
    #[test]
    fn token_processor_error() {
        let model = "hf-internal-testing/tiny-random-XLMRobertaXLForCausalLM";
        let vocabulary = Vocabulary::from_pretrained(model, None);

        match vocabulary {
            Err(Error::UnsupportedTokenizer { model, reason }) => {
                assert_eq!(model, model.to_string());
                assert_eq!(&reason, "Token processor");
            }
            _ => unreachable!(),
        }
    }

    #[cfg(feature = "huggingface-hub")]
    #[test]
    fn tokenizer_error() {
        let model = "hf-internal-testing/some-non-existent-model";
        let vocabulary = Vocabulary::from_pretrained(model, None);

        match vocabulary {
            Err(Error::TokenizersError(e)) => assert!(!e.to_string().is_empty()),
            _ => unreachable!(),
        }
    }

    #[cfg(feature = "huggingface-hub")]
    struct NoneLocator;
    #[cfg(feature = "huggingface-hub")]
    impl Locator for NoneLocator {
        fn locate_eos_token_id(
            _model: &str,
            _tokenizer: &Tokenizer,
            _parameters: &Option<FromPretrainedParameters>,
        ) -> Result<TokenId, Vec<String>> {
            Err(Vec::new())
        }
    }

    #[cfg(feature = "huggingface-hub")]
    #[test]
    fn unable_to_locate_eos_token_id_error() {
        let model = "hf-internal-testing/tiny-random-XLMRobertaXLForCausalLM";
        let vocabulary = Vocabulary::from_pretrained_with_locator::<NoneLocator>(model, None);

        match vocabulary {
            Err(Error::UnsupportedTokenizer { model, reason }) => {
                assert_eq!(model, model.to_string());
                assert_eq!(&reason, "EOS token id");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    #[cfg(feature = "huggingface-hub")]
    fn prepend_normalizers_filtered_out() {
        use tokenizers::normalizers::{Prepend, Sequence};

        let prepend = Prepend::new("_".to_string());
        let prepend_normalizer = NormalizerWrapper::Prepend(prepend);
        let sequence = Sequence::new(vec![prepend_normalizer.clone()]);
        let sequence_normalizer = NormalizerWrapper::Sequence(sequence);

        let model = "hf-internal-testing/llama-tokenizer";
        let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");

        for normalizer in [prepend_normalizer, sequence_normalizer] {
            let mut normalized_t = tokenizer.clone();
            normalized_t.with_normalizer(Some(normalizer));
            Vocabulary::filter_prepend_normalizers(&mut normalized_t);
            if let Some(n) = normalized_t.get_normalizer() {
                match n {
                    NormalizerWrapper::Sequence(seq) => {
                        for n in seq.as_ref() {
                            if let NormalizerWrapper::Prepend(_) = n {
                                unreachable!()
                            }
                        }
                    }
                    NormalizerWrapper::Prepend(_) => unreachable!(),
                    _ => {}
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "huggingface-hub")]
    fn other_normalizers_being_kept() {
        use tokenizers::normalizers::BertNormalizer;

        let model = "hf-internal-testing/llama-tokenizer";
        let normalizer = NormalizerWrapper::BertNormalizer(BertNormalizer::default());
        let mut tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");
        tokenizer.with_normalizer(Some(normalizer));

        Vocabulary::filter_prepend_normalizers(&mut tokenizer);

        assert!(tokenizer.get_normalizer().is_some());
    }
}
