//! Parsing known locations for `eos_token_id` information.

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use serde::{Deserialize, Serialize};
use tokenizers::{FromPretrainedParameters, Tokenizer};

use super::TokenId;

/// Mapping of characters to bytes for GPT-2 like tokenizers.
/// List of common eos token locations appearing on hugging face hub, ordered by priority.
const COMMON_LOCATIONS: &[EosTokenLocation] = &[
    // Prefer `generation_config.json` when it declares `eos_token_id`.
    EosTokenLocation {
        file: "generation_config.json",
        location: EosTokenField::Id,
    },
    // Fall back to `tokenizer_config.json` when it declares an EOS token value.
    EosTokenLocation {
        file: "tokenizer_config.json",
        location: EosTokenField::Value,
    },
    // Support object-valued EOS token metadata in `tokenizer_config.json`.
    EosTokenLocation {
        file: "tokenizer_config.json",
        location: EosTokenField::Object,
    },
];

/// `Id` kind of `EosTokenField`, when `eos_token_id` provided as an id.
#[derive(Debug, Serialize, Deserialize)]
struct Id {
    eos_token_id: u64,
}

/// `Value` kind of `EosTokenField`, when `eos_token` provided as a text, so that its id
/// will be fetched from the tokenizer.
#[derive(Debug, Serialize, Deserialize)]
struct Value {
    eos_token: String,
}

/// `Object` kind of `EosTokenField`, when `eos_token` provided as a `Content`.
#[derive(Debug, Serialize, Deserialize)]
struct Object {
    eos_token: Content,
}

/// `eos_token` provided in a `Content`.
#[derive(Debug, Serialize, Deserialize)]
struct Content {
    content: String,
}

/// Specifies in which part in config's json to check for eos token id.
enum EosTokenField {
    Id,
    Value,
    Object,
}

/// Defines location of the end of sentence token id in the config file.
struct EosTokenLocation {
    file: &'static str,
    location: EosTokenField,
}

/// Locates eos token id. One pass: `Ok` on the first hit, `Err` with every attempted location's
/// reason if all miss (never re-reads a location it already tried).
pub(crate) trait Locator {
    fn locate_eos_token_id(
        model: &str,
        tokenizer: &Tokenizer,
        parameters: &Option<FromPretrainedParameters>,
    ) -> Result<TokenId, Vec<String>>;
}

/// Locates eos token id by searching in defined common locations in hugging face.
pub(crate) struct HFLocator;

impl Locator for HFLocator {
    fn locate_eos_token_id(
        model: &str,
        tokenizer: &Tokenizer,
        parameters: &Option<FromPretrainedParameters>,
    ) -> Result<TokenId, Vec<String>> {
        let mut reasons = Vec::with_capacity(COMMON_LOCATIONS.len());
        for location in COMMON_LOCATIONS {
            match location.lookup(model, tokenizer, parameters) {
                Ok(id) => return Ok(id),
                Err(reason) => reasons.push(reason),
            }
        }
        Err(reasons)
    }
}

impl EosTokenLocation {
    /// Finds eos token within this location. `Err` carries the specific reason, not a bare miss.
    fn lookup(
        &self,
        model: &str,
        tokenizer: &Tokenizer,
        parameters: &Option<FromPretrainedParameters>,
    ) -> Result<TokenId, String> {
        let file = self.file;
        let file_path = Self::download_config(model, file, parameters)
            .map_err(|e| format!("{file}: could not download from the hub: {e}"))?;
        let reader = std::fs::File::open(&file_path)
            .map_err(|e| format!("{file}: could not open the downloaded file: {e}"))?;

        match self.location {
            EosTokenField::Id => {
                let config: Id = serde_json::from_reader(reader)
                    .map_err(|e| format!("{file}: eos_token_id field missing or malformed: {e}"))?;
                u32::try_from(config.eos_token_id)
                    .map_err(|_| format!("{file}: eos_token_id does not fit a u32"))
            }
            EosTokenField::Value => {
                let config: Value = serde_json::from_reader(reader)
                    .map_err(|e| format!("{file}: eos_token field missing or malformed: {e}"))?;
                tokenizer.token_to_id(&config.eos_token).ok_or_else(|| {
                    format!(
                        "{file}: eos_token {:?} not found in the tokenizer vocabulary",
                        config.eos_token
                    )
                })
            }
            EosTokenField::Object => {
                let config: Object = serde_json::from_reader(reader).map_err(|e| {
                    format!("{file}: eos_token.content field missing or malformed: {e}")
                })?;
                tokenizer
                    .token_to_id(&config.eos_token.content)
                    .ok_or_else(|| {
                        format!(
                            "{file}: eos_token.content {:?} not found in the tokenizer vocabulary",
                            config.eos_token.content
                        )
                    })
            }
        }
    }

    /// Downloads related config file from Hugging Face Hub.
    fn download_config(
        project: &str,
        file: &str,
        parameters: &Option<FromPretrainedParameters>,
    ) -> tokenizers::Result<std::path::PathBuf> {
        // Adapted from
        // https://github.com/huggingface/tokenizers/blob/9b77c054ef4297c7057fa8db875368c7c02f1bfc/tokenizers/src/utils/from_pretrained.rs#L26

        let params = parameters.clone().unwrap_or_default();

        // Validate the model identifier before constructing the Hub request.
        Self::validate(project)?;
        Self::validate(&params.revision)?;

        let repo = Repo::with_revision(project.to_string(), RepoType::Model, params.revision);
        let api = ApiBuilder::new()
            .with_token(params.token)
            .build()?
            .repo(repo);

        Ok(api.get(file)?)
    }

    fn validate(input: &str) -> tokenizers::Result<()> {
        let valid_chars = ['-', '_', '.', '/'];

        if !input
            .chars()
            .all(|c: char| c.is_alphanumeric() || valid_chars.contains(&c))
        {
            return Err(format!(
                "Input {input} contains invalid characters, expected only alphanumeric or {}",
                valid_chars
                    .iter()
                    .map(|x| format!("'{}'", x))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_locations() {
        for (model, expected_token_id, expected_token) in &[
            ("openai-community/gpt2", 50256, "<|endoftext|>"),
            ("microsoft/phi-2", 50256, "<|endoftext|>"),
            ("hf-internal-testing/llama-tokenizer", 2, "</s>"),
        ] {
            let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");
            let located = HFLocator::locate_eos_token_id(model, &tokenizer, &None)
                .expect("Token id is not located");

            assert_eq!(located, *expected_token_id);
            assert_eq!(
                tokenizer.id_to_token(located).expect("Token is not found"),
                expected_token.to_string()
            );
        }
    }

    #[test]
    fn bad_location() {
        let bad_location = EosTokenLocation {
            file: "tokenizer_config.json",
            location: EosTokenField::Id,
        };
        let model = "microsoft/phi-2";
        let tokenizer = Tokenizer::from_pretrained(model, None).expect("Tokenizer failed");

        let reason = bad_location
            .lookup(model, &tokenizer, &None)
            .expect_err("field genuinely absent from this file");
        assert!(reason.contains("tokenizer_config.json"), "{reason}");

        let bad_file = EosTokenLocation {
            file: "generation_config.json",
            location: EosTokenField::Value,
        };
        let reason = bad_file
            .lookup(model, &tokenizer, &None)
            .expect_err("field genuinely absent from this file");
        assert!(reason.contains("generation_config.json"), "{reason}");
    }

    #[test]
    fn miss_reports_exactly_one_reason_per_location_no_repeated_lookups() {
        // A repo that does not exist: every location's download fails once each, giving exactly
        // one reason per COMMON_LOCATIONS member, not a doubled count from a separate diagnostics pass.
        let model = "hf-internal-testing/this-repo-does-not-exist-xyz123";
        let tokenizer = Tokenizer::new(tokenizers::models::bpe::BPE::default());
        let reasons = HFLocator::locate_eos_token_id(model, &tokenizer, &None)
            .expect_err("a nonexistent repo cannot resolve any location");
        assert_eq!(reasons.len(), COMMON_LOCATIONS.len());
    }

    #[test]
    fn validate_config_input() {
        let input = "bad_model_name*";
        assert!(EosTokenLocation::validate(input).is_err());
    }
}
