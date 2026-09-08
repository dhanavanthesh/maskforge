//! KEEP-local vocabulary error, a trimmed port of the upstream `Error` variants.
//! The machine-contract error is `crate::error::VocabError`; `build.rs` translates into it.

use thiserror::Error;

pub type Result<T, E = Error> = std::result::Result<T, E>;

// `TokenizersError` mirrors the upstream variant name the KEEP modules and tests refer to;
// keep it rather than renaming to satisfy `clippy::enum_variant_names`.
#[allow(clippy::enum_variant_names)]
#[derive(Error, Debug)]
pub enum Error {
    #[error("EOS token should not be inserted into Vocabulary")]
    EOSTokenDisallowed,
    // MaskForge hardening over upstream: no construction path may admit an empty-byte token.
    #[error("empty-byte token is not allowed")]
    EmptyToken,
    // MaskForge hardening over upstream: a token with zero ids represents nothing decodable.
    #[error("token has an empty id list")]
    EmptyTokenIds,
    #[cfg(feature = "tokenizer-processing")]
    #[error(transparent)]
    TokenizersError(#[from] tokenizers::Error),
    #[cfg(feature = "tokenizer-processing")]
    #[error("Unsupported tokenizer for {model}: {reason}")]
    UnsupportedTokenizer { model: String, reason: String },
    #[cfg(feature = "tokenizer-processing")]
    #[error("Tokenizer is not supported by token processor")]
    UnsupportedByTokenProcessor,
    #[cfg(feature = "tokenizer-processing")]
    #[error("Decoder unpacking failed for token processor")]
    DecoderUnpackingFailed,
    #[cfg(feature = "tokenizer-processing")]
    #[error("Token processing failed for byte level processor")]
    ByteProcessorFailed,
    #[cfg(feature = "tokenizer-processing")]
    #[error("Token processing failed for byte fallback level processor")]
    ByteFallbackProcessorFailed,
    // Caller-supplied tokenizer JSON is untrusted input; bound it before the parser allocates.
    #[cfg(feature = "tokenizer-processing")]
    #[error("tokenizer JSON is {actual} bytes, over the {limit}-byte limit")]
    TokenizerJsonTooLarge { actual: usize, limit: usize },
}
