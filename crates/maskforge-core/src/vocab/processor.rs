// Modified from outlines-core; see PROVENANCE.md.

//! Post-processing operations for the tokens before they being inserted into
//! `Vocabulary`, strategies depend on the tokenizer's level.

use std::collections::HashMap;

use once_cell::sync::Lazy;
use serde::Deserialize;
use tokenizers::normalizers::Replace;
use tokenizers::{DecoderWrapper, Tokenizer};

use super::error::{Error, Result};

/// Maps GPT-2 byte-encoded Unicode characters back to raw bytes.
static CHAR_MAP: Lazy<HashMap<char, u8>> = Lazy::new(|| {
    let mut char_map = HashMap::with_capacity(256);
    let mut key = 0x100u32;
    for byte in 0..=255u8 {
        let char = byte as char;
        if matches!(
            char, '!'..='~' | '\u{00A1}'..='\u{00AC}' | '\u{00AE}'..='\u{00FF}',
        ) {
            char_map.insert(char, byte);
        } else if let Some(ch) = char::from_u32(key) {
            char_map.insert(ch, byte);
            key += 1;
        }
    }
    char_map
});

/// Recognizes different tokenizer's levels.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TokenProcessorLevel {
    /// Matches byte level tokenizer (e.g., gpt2).
    Byte,
    /// Matches byte fallback tokenizer (e.g., llama), which have `<0x__>` tokens for
    /// all `__` >= `0x80` to represent incomplete UTF-8 sequences.
    ByteFallback(Mods),
}

/// Modifications to be applied by `TokenProcessor`of `ByteFallback` level.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Mods {
    spacechar: String,
}

impl Mods {
    /// Default modification for `ByteFallback` level. Skips `replace`'s allocation when no marker
    /// is present (the common case).
    fn apply_bytes(&self, token: &str) -> Vec<u8> {
        if token.contains(self.spacechar.as_str()) {
            token.replace(&self.spacechar, " ").into_bytes()
        } else {
            token.as_bytes().to_vec()
        }
    }
}

/// Deserialization shape for the replacement pattern in an HF `ReplaceDecoder`.
#[derive(Debug, Deserialize)]
struct ReplaceDecoder {
    content: String,
    pattern: ReplacePattern,
}

impl ReplaceDecoder {
    fn space_replacement(&self) -> Option<String> {
        if self.content != " " {
            return None;
        }
        match &self.pattern {
            ReplacePattern::String(pattern) => {
                let mut chars = pattern.chars();
                let char = chars.next();
                if let Some(replacement) = char {
                    if chars.next().is_none() {
                        return Some(replacement.to_string());
                    }
                }
                None
            }
        }
    }
}

#[derive(Debug, Deserialize)]
enum ReplacePattern {
    String(String),
}

/// Token processor to adjust tokens according to the tokenizer's level.
#[derive(Debug)]
pub(crate) struct TokenProcessor {
    level: TokenProcessorLevel,
}

impl TokenProcessor {
    /// Create new `TokenProcessor` with the level defined based on tokenizer's decoders.
    pub(crate) fn new(tokenizer: &Tokenizer) -> Result<Self> {
        match tokenizer.get_decoder() {
            None => Err(Error::UnsupportedByTokenProcessor),
            Some(decoder) => match decoder {
                DecoderWrapper::ByteLevel(_) => Ok(Self {
                    level: TokenProcessorLevel::Byte,
                }),
                DecoderWrapper::Sequence(decoding_sequence) => {
                    let mut is_byte_fallback = false;
                    let mut spacechar = ' '.to_string();

                    for decoder in decoding_sequence.get_decoders() {
                        match decoder {
                            DecoderWrapper::ByteFallback(_) => {
                                is_byte_fallback = true;
                            }
                            DecoderWrapper::Replace(replace) => {
                                // `Replace` decoder would replace a pattern in the output with something else,
                                // which we need to know.
                                let decoder = Self::unpack_decoder(replace)?;
                                if let Some(replacement) = decoder.space_replacement() {
                                    spacechar = replacement;
                                }
                            }
                            _ => {}
                        }
                    }

                    if is_byte_fallback {
                        Ok(Self {
                            level: TokenProcessorLevel::ByteFallback(Mods { spacechar }),
                        })
                    } else {
                        Err(Error::UnsupportedByTokenProcessor)
                    }
                }
                _ => Err(Error::UnsupportedByTokenProcessor),
            },
        }
    }

    /// Operates on each token based on the level of `TokenProcessor`.
    pub(crate) fn process(&self, token: &str) -> Result<Vec<u8>> {
        match &self.level {
            TokenProcessorLevel::Byte => token
                .chars()
                .map(|char| {
                    CHAR_MAP
                        .get(&char)
                        .copied()
                        .ok_or(Error::ByteProcessorFailed)
                })
                .collect(),
            TokenProcessorLevel::ByteFallback(mods) => {
                // If the token is of form `<0x__>`:
                if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                    // Get to a single byte specified in the __ part and parse it in base 16 to a byte.
                    match u8::from_str_radix(&token[3..5], 16) {
                        Ok(byte) => Ok([byte].to_vec()),
                        Err(_) => Err(Error::ByteFallbackProcessorFailed),
                    }
                } else {
                    Ok(mods.apply_bytes(token))
                }
            }
        }
    }

    /// Since all fields of HF's `Replace` are private with no getters, it needs to be unpacked
    /// into local `ReplaceDecoder` structure.
    #[cfg(not(tarpaulin_include))]
    fn unpack_decoder(decoder: &Replace) -> Result<ReplaceDecoder> {
        match serde_json::to_value(decoder) {
            Err(_) => Err(Error::DecoderUnpackingFailed),
            Ok(value) => match serde_json::from_value(value) {
                Ok(d) => Ok(d),
                Err(_) => Err(Error::DecoderUnpackingFailed),
            },
        }
    }
}

// These tests fetch a real tokenizer via Tokenizer::from_pretrained, which needs huggingface-hub.
#[cfg(all(test, feature = "huggingface-hub"))]
mod tests {
    use super::*;

    #[test]
    fn byte_level_processor() {
        let model = "openai-community/gpt2";
        let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");
        let processor = TokenProcessor::new(&tokenizer).expect("Processor failed");

        assert_eq!(processor.level, TokenProcessorLevel::Byte);

        for (ch, byte) in [
            ('Ā', 0x00),
            ('ā', 0x01),
            ('Ă', 0x02),
            ('Ğ', 0x1E),
            ('ğ', 0x1F),
            ('Ġ', 0x20),
            ('!', 0x21),
            ('"', 0x22),
            ('#', 0x23),
            ('|', 0x7C),
            ('}', 0x7D),
            ('~', 0x7E),
            ('ġ', 0x7F),
            ('Ģ', 0x80),
            ('ģ', 0x81),
            ('ŀ', 0x9E),
            ('Ł', 0x9F),
            ('ł', 0xA0),
            ('¡', 0xA1),
            ('¢', 0xA2),
            ('£', 0xA3),
            ('ª', 0xAA),
            ('«', 0xAB),
            ('¬', 0xAC),
            ('Ń', 0xAD),
            ('®', 0xAE),
            ('¯', 0xAF),
            ('°', 0xB0),
            ('ý', 0xFD),
            ('þ', 0xFE),
            ('ÿ', 0xFF),
        ] {
            let processed = processor.process(&ch.to_string()).expect("Not processed");
            assert_eq!(processed, [byte]);
        }
    }

    #[test]
    fn byte_fallback_level_processor() {
        let model = "hf-internal-testing/llama-tokenizer";
        let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");
        let processor = TokenProcessor::new(&tokenizer).expect("Processor failed");
        let spacechar = '▁'.to_string();
        let mods = Mods {
            spacechar: spacechar.clone(),
        };

        assert_eq!(processor.level, TokenProcessorLevel::ByteFallback(mods));

        for (input, expected) in [
            ("abc", vec![0x61, 0x62, 0x63]),
            ("<0x61>", vec![0x61]),
            ("<0x61>a", vec![0x3C, 0x30, 0x78, 0x36, 0x31, 0x3E, 0x61]),
            (&spacechar, vec![0x20]),
            (
                &format!("{}{}abc", spacechar, spacechar),
                vec![0x20, 0x20, 0x61, 0x62, 0x63],
            ),
            (
                &format!("{}{}{}", spacechar, spacechar, spacechar),
                vec![0x20, 0x20, 0x20],
            ),
        ] {
            let processed = processor.process(input).expect("Not processed");
            assert_eq!(processed, expected);
        }
    }

    #[test]
    fn unsupported_tokenizer_error() {
        let model = "hf-internal-testing/tiny-random-XLMRobertaXLForCausalLM";
        let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");

        let result = TokenProcessor::new(&tokenizer);
        match result {
            Err(Error::UnsupportedByTokenProcessor) => {}
            _ => unreachable!(),
        }
    }

    #[test]
    fn byte_processor_error() {
        let model = "openai-community/gpt2";
        let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");
        let processor = TokenProcessor::new(&tokenizer).expect("Processor failed");

        for token in ["𝒜𝒷𝒸𝒟𝓔", "🦄🌈🌍🔥🎉", "京东购物"] {
            let result = processor.process(token);
            match result {
                Err(Error::ByteProcessorFailed) => {}
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn byte_fallback_processor_error() {
        let model = "hf-internal-testing/llama-tokenizer";
        let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");
        let processor = TokenProcessor::new(&tokenizer).expect("Processor failed");

        let result = processor.process("<0x6y>");
        match result {
            Err(Error::ByteFallbackProcessorFailed) => {}
            _ => unreachable!(),
        }
    }

    #[test]
    fn only_get_spacechar_replacement() {
        let one_char = "_".to_string();
        let pattern = ReplacePattern::String(one_char);
        let not_spacechar = "-".to_string();
        let decoder = ReplaceDecoder {
            content: not_spacechar,
            pattern,
        };
        assert!(decoder.space_replacement().is_none());
    }

    #[test]
    fn only_one_pattern_char_for_spacechar_replacement() {
        let two_chars = "_*".to_string();
        let pattern = ReplacePattern::String(two_chars);
        let spacechar = " ".to_string();
        let decoder = ReplaceDecoder {
            content: spacechar,
            pattern,
        };
        assert!(decoder.space_replacement().is_none());
    }

    #[test]
    fn tokenizer_without_decoders_is_unsupported() {
        use tokenizers::models::bpe::BPE;

        let tokenizer = Tokenizer::new(BPE::default());
        let result = TokenProcessor::new(&tokenizer);
        match result {
            Err(Error::UnsupportedByTokenProcessor) => {}
            _ => unreachable!(),
        }
    }

    #[test]
    fn tokenizer_without_supported_decoders_in_sequence_is_unsupported() {
        use tokenizers::decoders::sequence::Sequence;
        use tokenizers::decoders::wordpiece::WordPiece;
        use tokenizers::models::bpe::BPE;

        let mut tokenizer = Tokenizer::new(BPE::default());
        let decoder = WordPiece::default();
        let sequence = Sequence::new(vec![DecoderWrapper::WordPiece(decoder)]);
        let decoder_sequence = DecoderWrapper::Sequence(sequence);
        tokenizer.with_decoder(Some(decoder_sequence));

        let result = TokenProcessor::new(&tokenizer);
        match result {
            Err(Error::UnsupportedByTokenProcessor) => {}
            _ => unreachable!(),
        }
    }
}
