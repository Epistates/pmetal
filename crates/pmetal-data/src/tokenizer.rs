//! Tokenizer integration with config-aware special token resolution.
//!
//! Loads `special_tokens_map.json` and `tokenizer_config.json` alongside
//! `tokenizer.json` to resolve special token IDs authoritatively, falling
//! back to heuristic name-based lookup when config files are absent.

use pmetal_core::Result;
use std::path::Path;

/// Resolved special token IDs from model config files.
#[derive(Debug, Default)]
struct SpecialTokenIds {
    bos: Option<u32>,
    eos: Option<u32>,
    pad: Option<u32>,
    unk: Option<u32>,
}

/// Extract a token string from a JSON value that is either a plain string
/// or an object with a `"content"` key (HuggingFace `AddedToken` format).
fn extract_token_string(value: &serde_json::Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("content").and_then(|v| v.as_str()))
}

/// Wrapper around the tokenizers library.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    special: SpecialTokenIds,
}

impl Tokenizer {
    /// Load a tokenizer from a model directory.
    ///
    /// Reads `tokenizer.json` for the core tokenizer, then parses
    /// `special_tokens_map.json` and `tokenizer_config.json` (if present)
    /// to resolve bos/eos/pad/unk token IDs from authoritative config rather
    /// than relying solely on heuristic name matching.
    pub fn from_model_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let dir = model_dir.as_ref();
        let tokenizer_path = dir.join("tokenizer.json");
        let inner = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))?;

        let special = Self::load_special_tokens(&inner, dir);
        Ok(Self { inner, special })
    }

    /// Load a tokenizer from a local file (no config-aware resolution).
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))?;

        // If the file lives inside a model directory, try to load configs
        let special = path
            .parent()
            .map(|dir| Self::load_special_tokens(&inner, dir))
            .unwrap_or_default();

        Ok(Self { inner, special })
    }

    /// Load a tokenizer from bytes (no config-aware resolution).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))?;
        Ok(Self {
            inner,
            special: SpecialTokenIds::default(),
        })
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode text with special tokens.
    pub fn encode_with_special_tokens(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))
    }

    /// Decode token IDs to text without skipping special tokens.
    pub fn decode_with_special_tokens(&self, ids: &[u32]) -> Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get the underlying tokenizer.
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }

    /// Get pad token ID if available.
    ///
    /// Resolution order:
    /// 1. Config files (`special_tokens_map.json` / `tokenizer_config.json`)
    /// 2. Heuristic: common pad token names in vocabulary
    /// 3. Fallback to EOS token
    pub fn pad_token_id(&self) -> Option<u32> {
        self.special.pad.or_else(|| {
            self.inner
                .token_to_id("<pad>")
                .or_else(|| self.inner.token_to_id("[PAD]"))
                .or_else(|| self.inner.token_to_id("<|pad|>"))
                .or_else(|| self.inner.token_to_id("<|finetune_right_pad_id|>"))
                .or_else(|| self.eos_token_id())
        })
    }

    /// Get EOS token ID if available.
    ///
    /// Resolution order:
    /// 1. Config files (`special_tokens_map.json` / `tokenizer_config.json`)
    /// 2. Heuristic: common EOS token names in vocabulary
    pub fn eos_token_id(&self) -> Option<u32> {
        self.special.eos.or_else(|| {
            self.inner
                .token_to_id("</s>")
                .or_else(|| self.inner.token_to_id("<|endoftext|>"))
                .or_else(|| self.inner.token_to_id("<|end_of_text|>"))
                .or_else(|| self.inner.token_to_id("<eos>"))
        })
    }

    /// Get the EOS token string if available.
    ///
    /// Resolves the EOS token ID and converts it back to the token's surface
    /// form using the vocabulary, preserving the exact string representation
    /// (e.g. `<|endoftext|>` for Qwen3, `</s>` for Llama 2/Mistral).
    pub fn eos_token_str(&self) -> Option<String> {
        self.eos_token_id()
            .and_then(|id| self.inner.id_to_token(id))
    }

    /// Get BOS token ID if available.
    ///
    /// Resolution order:
    /// 1. Config files (`special_tokens_map.json` / `tokenizer_config.json`)
    /// 2. Heuristic: common BOS token names in vocabulary
    pub fn bos_token_id(&self) -> Option<u32> {
        self.special.bos.or_else(|| {
            self.inner
                .token_to_id("<s>")
                .or_else(|| self.inner.token_to_id("<|begin_of_text|>"))
                .or_else(|| self.inner.token_to_id("<bos>"))
        })
    }

    /// Get UNK token ID if available.
    pub fn unk_token_id(&self) -> Option<u32> {
        self.special.unk.or_else(|| {
            self.inner
                .token_to_id("<unk>")
                .or_else(|| self.inner.token_to_id("[UNK]"))
        })
    }

    /// Resolve a token string to its ID via the tokenizer vocabulary.
    fn resolve_token(inner: &tokenizers::Tokenizer, token_str: &str) -> Option<u32> {
        inner.token_to_id(token_str)
    }

    /// Load special token mappings from model config files.
    ///
    /// Reads `special_tokens_map.json` first (authoritative), then
    /// fills in any missing values from `tokenizer_config.json`.
    fn load_special_tokens(inner: &tokenizers::Tokenizer, model_dir: &Path) -> SpecialTokenIds {
        let mut ids = SpecialTokenIds::default();

        // 1. Try special_tokens_map.json (most authoritative for token strings)
        let stm_path = model_dir.join("special_tokens_map.json");
        if let Ok(content) = std::fs::read_to_string(&stm_path) {
            if let Ok(map) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(obj) = map.as_object() {
                    for (key, value) in obj {
                        let token_str = match extract_token_string(value) {
                            Some(s) => s,
                            None => continue,
                        };
                        let token_id = match Self::resolve_token(inner, token_str) {
                            Some(id) => id,
                            None => continue,
                        };
                        match key.as_str() {
                            "bos_token" => ids.bos = Some(token_id),
                            "eos_token" => ids.eos = Some(token_id),
                            "pad_token" => ids.pad = Some(token_id),
                            "unk_token" => ids.unk = Some(token_id),
                            _ => {}
                        }
                    }
                    tracing::debug!(
                        "Loaded special tokens from special_tokens_map.json: \
                         bos={:?} eos={:?} pad={:?} unk={:?}",
                        ids.bos,
                        ids.eos,
                        ids.pad,
                        ids.unk
                    );
                }
            }
        }

        // 2. Fill gaps from tokenizer_config.json
        let tc_path = model_dir.join("tokenizer_config.json");
        if let Ok(content) = std::fs::read_to_string(&tc_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                let keys = ["bos_token", "eos_token", "pad_token", "unk_token"];
                for key in &keys {
                    let slot = match *key {
                        "bos_token" => &mut ids.bos,
                        "eos_token" => &mut ids.eos,
                        "pad_token" => &mut ids.pad,
                        "unk_token" => &mut ids.unk,
                        _ => unreachable!(),
                    };
                    if slot.is_some() {
                        continue;
                    }
                    if let Some(val) = config.get(*key) {
                        if let Some(token_str) = extract_token_string(val) {
                            if let Some(id) = Self::resolve_token(inner, token_str) {
                                *slot = Some(id);
                            }
                        }
                    }
                }
                tracing::debug!(
                    "After tokenizer_config.json: bos={:?} eos={:?} pad={:?} unk={:?}",
                    ids.bos,
                    ids.eos,
                    ids.pad,
                    ids.unk
                );
            }
        }

        ids
    }
}
