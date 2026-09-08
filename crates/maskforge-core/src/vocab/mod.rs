//! Vocabulary subtree. `build.rs` is the validated choke-point.
//! `vocabulary.rs`/`processor.rs`/`locator.rs` originate from outlines-core (PROVENANCE.md).

/// Token content (raw bytes; non-UTF-8 payloads are valid).
pub type Token = Vec<u8>;

/// Token identifier (a bare `u32`); converted to the checked `TokenId` newtype at the mask boundary.
pub type TokenId = u32;

#[cfg(feature = "huggingface-hub")]
pub use tokenizers::FromPretrainedParameters;

mod build;
mod canonical;
mod error;
#[cfg(feature = "huggingface-hub")]
mod locator;
mod prepared;
#[cfg(feature = "tokenizer-processing")]
mod processor;
mod vocabulary;

#[cfg(feature = "tokenizer-processing")]
pub use build::build_vocabulary_from_tokenizer_json;
pub use build::{build_vocabulary, build_vocabulary_from_str_keys, StringTokenMap, TokenMap};
pub(crate) use canonical::CanonicalVocabulary;
pub(crate) use prepared::{projected_peak_bytes, try_alloc, PreparedVocabulary};
pub use vocabulary::Vocabulary;
