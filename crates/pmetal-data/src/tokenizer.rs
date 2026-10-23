//! Tokenizer integration with config-aware special token resolution.
//!
//! Loads `special_tokens_map.json` and `tokenizer_config.json` alongside
//! `tokenizer.json` to resolve special token IDs authoritatively, falling
//! back to heuristic name-based lookup when config files are absent.

use pmetal_core::{PMetalError, Result};
use pmetal_gguf::GgufContent;
use std::path::{Path, PathBuf};

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

/// Resolve a `.gguf` weight file from a model path.
///
/// Accepts a direct `.gguf` file, or a directory containing `model.gguf` or a
/// single `.gguf`. Returns `None` when no unambiguous GGUF is present (callers
/// then fall back to the HuggingFace directory loader).
fn resolve_gguf_path(path: &Path) -> Option<PathBuf> {
    if path.is_file() {
        return path
            .extension()
            .and_then(|e| e.to_str())
            .filter(|e| e.eq_ignore_ascii_case("gguf"))
            .map(|_| path.to_path_buf());
    }
    if !path.is_dir() {
        return None;
    }
    let mut gguf_files: Vec<PathBuf> = std::fs::read_dir(path)
        .ok()?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
        })
        .collect();
    gguf_files.sort();

    if let Some(model_gguf) = gguf_files
        .iter()
        .find(|p| p.file_name().and_then(|n| n.to_str()) == Some("model.gguf"))
    {
        return Some(model_gguf.clone());
    }
    match gguf_files.len() {
        1 => Some(gguf_files.remove(0)),
        _ => None,
    }
}

/// A tokenized text: the ids, and for each token the span of the input it
/// covers.
///
/// `offsets` are **byte** offsets into the text that was encoded, so
/// `&text[start..end]` is the piece a token stands for. A token the
/// tokenizer adds on its own (a BOS from the post-processor) has the empty
/// span `(0, 0)`. Byte-level tokenizers can split one multi-byte character
/// across tokens; each of those tokens then covers the whole character.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedText {
    /// Token ids.
    pub ids: Vec<u32>,
    /// Token strings, as they appear in the vocabulary (`Ġthe`, `▁the`).
    pub tokens: Vec<String>,
    /// `(start, end)` byte span of the input each token covers.
    pub offsets: Vec<(usize, usize)>,
    /// Whether each token is a special token (a BOS the post-processor
    /// added, or a `<|im_start|>` found in the text).
    pub special_tokens_mask: Vec<bool>,
}

impl TokenizedText {
    fn from_encoding(encoding: &tokenizers::Encoding) -> Self {
        Self {
            ids: encoding.get_ids().to_vec(),
            tokens: encoding.get_tokens().to_vec(),
            offsets: encoding.get_offsets().to_vec(),
            special_tokens_mask: encoding
                .get_special_tokens_mask()
                .iter()
                .map(|&m| m != 0)
                .collect(),
        }
    }

    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The offsets as character (not byte) positions of `text`, for callers
    /// that index text by character. `text` must be what was encoded.
    pub fn char_offsets(&self, text: &str) -> Vec<(usize, usize)> {
        // Byte index -> character index, for every character boundary.
        let mut char_at = vec![0usize; text.len() + 1];
        let mut chars = 0;
        for (byte, c) in text.char_indices() {
            for slot in &mut char_at[byte..byte + c.len_utf8()] {
                *slot = chars;
            }
            chars += 1;
        }
        char_at[text.len()] = chars;
        self.offsets
            .iter()
            .map(|&(start, end)| (char_at[start.min(text.len())], char_at[end.min(text.len())]))
            .collect()
    }
}

/// The chat template a GGUF embeds, with the strings of its BOS and EOS
/// tokens.
pub(crate) struct GgufChatTemplate {
    /// `tokenizer.chat_template`.
    pub template: String,
    /// The BOS token's string.
    pub bos_token: Option<String>,
    /// The EOS token's string.
    pub eos_token: Option<String>,
}

/// The chat template embedded in the GGUF at `path` (a `.gguf` file, or a
/// directory holding one), if it has any.
pub(crate) fn gguf_chat_template(path: &Path) -> Option<GgufChatTemplate> {
    let content = GgufContent::from_file(resolve_gguf_path(path)?).ok()?;
    let data = content.tokenizer_data()?;
    let template = data.chat_template.filter(|t| !t.trim().is_empty())?;
    let token = |id: Option<u32>| id.and_then(|i| data.tokens.get(i as usize).cloned());
    Some(GgufChatTemplate {
        template,
        bos_token: token(data.bos_id),
        eos_token: token(data.eos_id),
    })
}

/// The ids a GGUF at `path` (a `.gguf` file, or a directory holding one)
/// says end a generation: end of text, end of turn, end of message.
pub(crate) fn gguf_stop_token_ids(path: &Path) -> Vec<u32> {
    let Some(gguf) = resolve_gguf_path(path) else {
        return Vec::new();
    };
    let Some(data) = GgufContent::from_file(gguf)
        .ok()
        .and_then(|content| content.tokenizer_data())
    else {
        return Vec::new();
    };
    [data.eos_id, data.eot_id, data.eom_id]
        .into_iter()
        .flatten()
        .collect()
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
        Ok(Self { inner, special }.with_the_models_bos(dir))
    }

    /// Load a tokenizer from a model path (file or directory), resolving GGUF
    /// metadata when no `tokenizer.json` is available.
    ///
    /// Resolution order:
    /// 1. Directory with `tokenizer.json` → canonical HuggingFace loader.
    /// 2. A `.gguf` file, or a directory whose weights are a single `.gguf` →
    ///    reconstruct the tokenizer from `tokenizer.ggml.*` metadata.
    /// 3. Otherwise fall back to the directory loader (clear error if nothing).
    pub fn from_model_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        if path.is_dir() && path.join("tokenizer.json").exists() {
            return Self::from_model_dir(path);
        }
        if let Some(gguf_path) = resolve_gguf_path(path) {
            let content = GgufContent::from_file(&gguf_path)
                .map_err(|e| PMetalError::Tokenizer(format!("GGUF read: {e}")))?;
            return Self::from_gguf(&content);
        }
        Self::from_model_dir(path)
    }

    /// Reconstruct a tokenizer from already-parsed GGUF content.
    ///
    /// Special-token ids (bos/eos/pad/unk) come straight from
    /// `tokenizer.ggml.*_token_id` metadata rather than name heuristics.
    pub fn from_gguf(content: &GgufContent) -> Result<Self> {
        let data = content
            .tokenizer_data()
            .ok_or_else(|| PMetalError::Tokenizer("GGUF file has no tokenizer metadata".into()))?;
        let inner = crate::gguf_tokenizer::build_tokenizer(&data)?;
        let special = SpecialTokenIds {
            bos: data.bos_id,
            eos: data.eos_id,
            pad: data.pad_id,
            unk: data.unk_id,
        };
        Ok(Self { inner, special })
    }

    /// Load a tokenizer from a local file (no config-aware resolution).
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))?;

        // If the file lives inside a model directory, try to load configs
        let dir = path.parent();
        let special = dir
            .map(|dir| Self::load_special_tokens(&inner, dir))
            .unwrap_or_default();

        let tokenizer = Self { inner, special };
        Ok(match dir {
            Some(dir) => tokenizer.with_the_models_bos(dir),
            None => tokenizer,
        })
    }

    /// Make this tokenizer start an encoding that asks for special tokens
    /// with the BOS its model expects.
    ///
    /// Most tokenizers say so themselves: the post-processor of Llama 3 or
    /// Gemma 3 puts the BOS in front. Gemma 4's does not; its chat template
    /// writes the BOS, and `tokenizer(text)` of a plain text then starts
    /// without one, which is a sequence the model never trained on (a
    /// perplexity of 4596 against 819 with the BOS). So when the
    /// post-processor adds none, the model's files decide whether it expects
    /// one: `add_bos_token` in `tokenizer_config.json` if it says, else
    /// whether the chat template opens a conversation with `bos_token`.
    /// A model that expects none (Qwen3) keeps encoding to none.
    ///
    /// With the BOS in the post-processor, every path that asks for special
    /// tokens ([`encode_with_special_tokens`](Self::encode_with_special_tokens)
    /// and the batch and offset encoders) adds it, once: text that opens with
    /// the BOS already is encoded without the post-processor.
    fn with_the_models_bos(mut self, model_dir: &Path) -> Self {
        let Some(bos_id) = self.bos_token_id() else {
            return self;
        };
        let Some(bos) = self.inner.id_to_token(bos_id) else {
            return self;
        };
        if self.post_processor_adds(bos_id) {
            return self;
        }
        let expects = crate::chat_templates::read_tokenizer_config(model_dir)
            .get("add_bos_token")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or_else(|| {
                let eos = self
                    .eos_token_id()
                    .and_then(|id| self.inner.id_to_token(id));
                crate::chat_templates::template_opens_with_bos(model_dir, &bos, eos.as_deref())
            });
        if !expects {
            return self;
        }
        let processor =
            match crate::gguf_tokenizer::special_tokens_processor(Some((bos, bos_id)), None) {
                Ok(processor) => processor,
                Err(e) => {
                    tracing::warn!("not adding the BOS the model expects: {e}");
                    return self;
                }
            };
        // Whatever the post-processor does already (trimming offsets, an
        // EOS) stays, and the BOS goes in front of its result.
        let combined: tokenizers::processors::PostProcessorWrapper =
            match self.inner.get_post_processor() {
                Some(existing) => tokenizers::processors::sequence::Sequence::new(vec![
                    existing.clone(),
                    processor.into(),
                ])
                .into(),
                None => processor.into(),
            };
        self.inner.with_post_processor(Some(combined));
        self
    }

    /// Whether encoding with special tokens puts the BOS `bos_id` at the
    /// front of a text.
    fn post_processor_adds(&self, bos_id: u32) -> bool {
        self.inner
            .encode_fast("a", true)
            .is_ok_and(|encoding| encoding.get_ids().first() == Some(&bos_id))
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

    /// Encode text to token IDs, adding none of the tokenizer's own special
    /// tokens.
    ///
    /// Special tokens that appear *in* `text` (`<|im_start|>`, `<bos>`, …) are
    /// still recognised as single tokens. This is the call for text a chat
    /// template rendered: the template already wrote the BOS its model wants,
    /// and `transformers` tokenizes a rendered chat with
    /// `add_special_tokens=False` for that reason.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode_fast(text, false)
            .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode text the way `tokenizer(text)` does: the tokenizer's
    /// post-processor adds its own BOS / EOS around it.
    ///
    /// Use it for plain text (a pre-training document, a raw prompt). Text a
    /// chat template rendered already carries its BOS and goes through
    /// [`encode`](Self::encode); text that opens with the BOS token anyway is
    /// not given a second one, as a model trained on one BOS never saw two.
    pub fn encode_with_special_tokens(&self, text: &str) -> Result<Vec<u32>> {
        let add = !self.opens_with_bos(text);
        let encoding = self
            .inner
            .encode_fast(text, add)
            .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode text and keep the span of `text` each token covers.
    ///
    /// `add_special_tokens` is as in [`encode`](Self::encode) (`false`) and
    /// [`encode_with_special_tokens`](Self::encode_with_special_tokens)
    /// (`true`). Cross-tokenizer work (distillation aligning a teacher's
    /// tokens with a student's) and loss masks over a character span read the
    /// offsets.
    pub fn encode_with_offsets(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<TokenizedText> {
        let add = add_special_tokens && !self.opens_with_bos(text);
        let encoding = self
            .inner
            .encode(text, add)
            .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))?;
        Ok(TokenizedText::from_encoding(&encoding))
    }

    /// Encode many texts at once, in parallel, with the offsets of each.
    ///
    /// This is the call for tokenizing a dataset: the batch is split across
    /// the tokenizer's thread pool, and the results come back in input order.
    pub fn encode_batch_with_offsets<S: AsRef<str> + Sync>(
        &self,
        texts: &[S],
        add_special_tokens: bool,
    ) -> Result<Vec<TokenizedText>> {
        let inputs: Vec<tokenizers::EncodeInput<'_>> = texts
            .iter()
            .map(|t| tokenizers::EncodeInput::Single(t.as_ref().into()))
            .collect();
        let mut encodings = self
            .inner
            .encode_batch(inputs, add_special_tokens)
            .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))?;
        if add_special_tokens {
            for (encoding, text) in encodings.iter_mut().zip(texts) {
                if self.opens_with_bos(text.as_ref()) {
                    *encoding = self
                        .inner
                        .encode(text.as_ref(), false)
                        .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))?;
                }
            }
        }
        Ok(encodings.iter().map(TokenizedText::from_encoding).collect())
    }

    /// Encode many texts at once, in parallel; the ids of
    /// [`encode_batch_with_offsets`](Self::encode_batch_with_offsets).
    pub fn encode_batch<S: AsRef<str> + Sync>(
        &self,
        texts: &[S],
        add_special_tokens: bool,
    ) -> Result<Vec<Vec<u32>>> {
        let inputs: Vec<tokenizers::EncodeInput<'_>> = texts
            .iter()
            .map(|t| tokenizers::EncodeInput::Single(t.as_ref().into()))
            .collect();
        let mut encodings = self
            .inner
            .encode_batch_fast(inputs, add_special_tokens)
            .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))?;
        if add_special_tokens {
            for (encoding, text) in encodings.iter_mut().zip(texts) {
                if self.opens_with_bos(text.as_ref()) {
                    *encoding = self
                        .inner
                        .encode_fast(text.as_ref(), false)
                        .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))?;
                }
            }
        }
        Ok(encodings.iter().map(|e| e.get_ids().to_vec()).collect())
    }

    /// Whether `text` starts with the BOS token's own string, which a
    /// post-processor would put a second one in front of.
    fn opens_with_bos(&self, text: &str) -> bool {
        self.bos_token_id()
            .and_then(|id| self.inner.id_to_token(id))
            .is_some_and(|bos| text.starts_with(&bos))
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
            let explicit = self
                .inner
                .token_to_id("<pad>")
                .or_else(|| self.inner.token_to_id("[PAD]"))
                .or_else(|| self.inner.token_to_id("<|pad|>"))
                .or_else(|| self.inner.token_to_id("<|finetune_right_pad_id|>"));
            if explicit.is_some() {
                return explicit;
            }
            let eos = self.eos_token_id();
            if eos.is_some() {
                tracing::warn!("pad_token_id not configured, falling back to eos_token_id");
            }
            eos
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

/// Small in-memory tokenizers for tests, here and in crates that take a
/// [`Tokenizer`].
pub mod testing {
    use super::Tokenizer;

    /// A word-level tokenizer over `words` (text splits on whitespace; the
    /// first word stands for unknown ones), with `markers` as added tokens,
    /// `(content, special)`, numbered after the words. Decoding joins tokens
    /// with spaces.
    pub fn word_level(words: &[&str], markers: &[(&str, bool)]) -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        for (i, w) in words.iter().enumerate() {
            vocab.insert((*w).to_string(), (i as u32).into());
        }
        let added: Vec<serde_json::Value> = markers
            .iter()
            .enumerate()
            .map(|(i, (content, special))| {
                let id = (words.len() + i) as u32;
                vocab.insert((*content).to_string(), id.into());
                serde_json::json!({
                    "id": id, "content": content, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": special
                })
            })
            .collect();
        let json = serde_json::json!({
            "version": "1.0",
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": words[0]}
        });
        Tokenizer::from_bytes(json.to_string().as_bytes()).expect("a word-level tokenizer")
    }

    /// GPT-2's byte-to-character table: the printable stand-in of each byte.
    fn byte_chars() -> [char; 256] {
        let printable: Vec<u32> = (0x21..=0x7e)
            .chain(0xa1..=0xac)
            .chain(0xae..=0xff)
            .collect();
        let mut table = ['\0'; 256];
        let mut next = 0;
        for byte in 0..256u32 {
            let c = if printable.contains(&byte) {
                byte
            } else {
                next += 1;
                255 + next
            };
            table[byte as usize] = char::from_u32(c).expect("a scalar value");
        }
        table
    }

    /// A byte-level BPE tokenizer (the GPT-2 / Llama 3 / Qwen layout) whose
    /// vocabulary is the 256 byte characters, then one token per entry of
    /// `merges` (`"a b"`, in the byte-character alphabet), then `specials`
    /// as added tokens. With `bos`, the post-processor puts that special
    /// token (which must be among `specials`) in front of an encoding that
    /// asks for special tokens.
    pub fn byte_level_bpe(merges: &[&str], specials: &[&str], bos: Option<&str>) -> Tokenizer {
        let table = byte_chars();
        let mut vocab = serde_json::Map::new();
        for (byte, c) in table.iter().enumerate() {
            vocab.insert(c.to_string(), (byte as u32).into());
        }
        let mut next = 256u32;
        for merge in merges {
            let merged: String = merge.split(' ').collect();
            vocab.insert(merged, next.into());
            next += 1;
        }
        let mut added = Vec::new();
        let mut bos_id = None;
        for special in specials {
            if Some(*special) == bos {
                bos_id = Some(next);
            }
            added.push(serde_json::json!({
                "id": next, "content": special, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true
            }));
            vocab.insert((*special).to_string(), next.into());
            next += 1;
        }
        let post_processor = match (bos, bos_id) {
            (Some(bos), Some(id)) => serde_json::json!({
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": bos, "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}}
                ],
                "pair": [
                    {"SpecialToken": {"id": bos, "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"Sequence": {"id": "B", "type_id": 1}}
                ],
                "special_tokens": {bos: {"id": bos, "ids": [id], "tokens": [bos]}}
            }),
            _ => serde_json::Value::Null,
        };
        let json = serde_json::json!({
            "version": "1.0",
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": {
                "type": "ByteLevel", "add_prefix_space": false,
                "trim_offsets": true, "use_regex": true
            },
            "post_processor": post_processor,
            "decoder": {
                "type": "ByteLevel", "add_prefix_space": true,
                "trim_offsets": true, "use_regex": true
            },
            "model": {
                "type": "BPE", "vocab": vocab, "merges": merges,
                "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false,
                "ignore_merges": false
            }
        });
        Tokenizer::from_bytes(json.to_string().as_bytes()).expect("a byte-level BPE tokenizer")
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{byte_level_bpe, word_level};

    fn bpe() -> super::Tokenizer {
        // "h" "i" -> "hi"; "Ġ" "t" -> "Ġt"
        byte_level_bpe(&["h i", "\u{120} t"], &["<s>", "<|im_start|>"], Some("<s>"))
    }

    #[test]
    fn offsets_are_byte_spans_of_the_input() {
        let tk = bpe();
        let text = "hi \u{e9}\u{1F600}";
        let enc = tk.encode_with_offsets(text, false).unwrap();
        assert_eq!(enc.ids.len(), enc.offsets.len());
        assert_eq!(enc.ids, tk.encode(text).unwrap());
        // Every token's span is a valid slice of the input, the spans never
        // go backwards, and together they cover the whole text.
        let mut end = 0;
        for &(s, e) in &enc.offsets {
            assert!(
                text.is_char_boundary(s) && text.is_char_boundary(e),
                "{s}..{e}"
            );
            assert!(s >= end.min(s) && e >= s);
            end = end.max(e);
        }
        assert_eq!(end, text.len());
        assert_eq!(&text[enc.offsets[0].0..enc.offsets[0].1], "hi");
        // "hi" is one merged token; the e-acute is two bytes, two tokens, and
        // both cover the whole character.
        let last = enc.offsets.len() - 1;
        assert_eq!(&text[enc.offsets[last].0..enc.offsets[last].1], "\u{1F600}");
    }

    #[test]
    fn char_offsets_index_by_character() {
        let tk = bpe();
        let text = "h\u{e9}\u{1F600}i";
        let enc = tk.encode_with_offsets(text, false).unwrap();
        let chars: Vec<char> = text.chars().collect();
        for (&(bs, be), (cs, ce)) in enc.offsets.iter().zip(enc.char_offsets(text)) {
            let by_bytes: String = text[bs..be].to_string();
            let by_chars: String = chars[cs..ce].iter().collect();
            assert_eq!(by_bytes, by_chars);
        }
    }

    #[test]
    fn rendered_chat_text_is_not_given_a_second_bos() {
        let tk = bpe();
        let bos = tk.inner().token_to_id("<s>").unwrap();
        // A chat template that writes `{{ bos_token }}` itself.
        let rendered = "<s><|im_start|>hi";
        let ids = tk.encode(rendered).unwrap();
        assert_eq!(ids.iter().filter(|&&i| i == bos).count(), 1);
        // The post-processor path adds one in front of plain text, as
        // `tokenizer(text)` does, and none in front of text that has one.
        let ids = tk.encode_with_special_tokens("hi").unwrap();
        assert_eq!(ids.iter().filter(|&&i| i == bos).count(), 1);
        assert_eq!(ids[0], bos);
        assert_eq!(
            tk.encode_with_special_tokens(rendered).unwrap(),
            tk.encode(rendered).unwrap()
        );
        // Offsets: the added BOS covers nothing, the one in the text covers its text.
        let enc = tk.encode_with_offsets("hi", true).unwrap();
        assert_eq!(enc.offsets[0], (0, 0));
        assert!(enc.special_tokens_mask[0]);
        let enc = tk.encode_with_offsets(rendered, true).unwrap();
        assert_eq!(enc.offsets[0], (0, 3));
        assert_eq!(enc.ids, tk.encode(rendered).unwrap());
        // ...and the same holds a batch at a time.
        let texts = ["hi", rendered];
        let batch = tk.encode_batch(&texts, true).unwrap();
        assert_eq!(batch[0], tk.encode_with_special_tokens("hi").unwrap());
        assert_eq!(batch[1], tk.encode(rendered).unwrap());
        let batch = tk.encode_batch_with_offsets(&texts, true).unwrap();
        assert_eq!(batch[1].ids, tk.encode(rendered).unwrap());
    }

    #[test]
    fn batches_equal_one_at_a_time_in_order() {
        let tk = bpe();
        let texts: Vec<String> = (0..64).map(|i| format!("hi t{i} \u{e9}")).collect();
        let batch = tk.encode_batch(&texts, false).unwrap();
        let with_offsets = tk.encode_batch_with_offsets(&texts, true).unwrap();
        for (i, text) in texts.iter().enumerate() {
            assert_eq!(batch[i], tk.encode(text).unwrap());
            assert_eq!(with_offsets[i], tk.encode_with_offsets(text, true).unwrap());
        }
    }

    /// A model directory: the tokenizer, `tokenizer_config.json`, and a chat
    /// template, as the checkpoint ships them.
    fn model_dir(
        tk: &super::Tokenizer,
        config: serde_json::Value,
        chat_template: Option<&str>,
    ) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        tk.inner()
            .save(dir.path().join("tokenizer.json"), false)
            .unwrap();
        std::fs::write(dir.path().join("tokenizer_config.json"), config.to_string()).unwrap();
        if let Some(template) = chat_template {
            std::fs::write(dir.path().join("chat_template.jinja"), template).unwrap();
        }
        dir
    }

    /// Gemma 4's layout: a `<bos>` token, no BOS in the post-processor, and a
    /// chat template that writes the BOS itself behind its macros.
    const GEMMA4_TEMPLATE: &str = "{%- macro fmt(x) -%}{{ x }}{%- endmacro -%}\n\
        {{- bos_token -}}\n\
        {%- for m in messages -%}<|im_start|>{{ m['role'] }}\\n{{ fmt(m['content']) }}{%- endfor -%}\n\
        {%- if add_generation_prompt -%}<|im_start|>model\\n{%- endif -%}";

    fn count(ids: &[u32], id: u32) -> usize {
        ids.iter().filter(|&&i| i == id).count()
    }

    #[test]
    fn a_model_whose_template_writes_the_bos_gets_one_from_plain_text() {
        // Gemma 4: no BOS in the post-processor; the chat template has it.
        let none = byte_level_bpe(&["h i"], &["<bos>", "<|im_start|>"], None);
        let dir = model_dir(
            &none,
            serde_json::json!({"bos_token": "<bos>"}),
            Some(GEMMA4_TEMPLATE),
        );
        assert_eq!(
            none.encode_with_special_tokens("hi").unwrap(),
            none.encode("hi").unwrap(),
            "the tokenizer alone adds nothing"
        );
        let tk = super::Tokenizer::from_model_dir(dir.path()).unwrap();
        let bos = tk.inner().token_to_id("<bos>").unwrap();

        let ids = tk.encode_with_special_tokens("hi").unwrap();
        assert_eq!(ids[0], bos);
        assert_eq!(count(&ids, bos), 1);
        assert_eq!(ids[1..], none.encode("hi").unwrap()[..]);

        // Every path that asks for special tokens gets the one BOS...
        let enc = tk.encode_with_offsets("hi", true).unwrap();
        assert_eq!((enc.ids[0], enc.offsets[0]), (bos, (0, 0)));
        let batch = tk.encode_batch(&["hi", "hi hi"], true).unwrap();
        assert!(
            batch
                .iter()
                .all(|ids| ids[0] == bos && count(ids, bos) == 1)
        );
        let batch = tk.encode_batch_with_offsets(&["hi"], true).unwrap();
        assert_eq!(batch[0].ids[0], bos);
        // ...and text a chat template rendered, or that opens with the BOS,
        // is never given a second.
        let rendered = "<bos><|im_start|>hi";
        assert_eq!(
            count(&tk.encode_with_special_tokens(rendered).unwrap(), bos),
            1
        );
        assert_eq!(count(&tk.encode(rendered).unwrap(), bos), 1);
        assert_eq!(
            count(&tk.encode_batch(&[rendered], true).unwrap()[0], bos),
            1
        );
        // Text without special tokens asked for stays bare.
        assert_eq!(tk.encode("hi").unwrap(), none.encode("hi").unwrap());
    }

    #[test]
    fn a_tokenizer_that_adds_its_own_bos_is_not_given_another() {
        // Llama 3: the post-processor adds `<s>`, and so does the template.
        let llama = bpe();
        let template = "{{- bos_token -}}{%- for m in messages -%}{{ m['content'] }}{%- endfor -%}";
        let dir = model_dir(
            &llama,
            serde_json::json!({"bos_token": "<s>", "add_bos_token": true}),
            Some(template),
        );
        let tk = super::Tokenizer::from_model_dir(dir.path()).unwrap();
        let bos = tk.inner().token_to_id("<s>").unwrap();
        let ids = tk.encode_with_special_tokens("hi").unwrap();
        assert_eq!(ids[0], bos);
        assert_eq!(count(&ids, bos), 1);
        let enc = tk.encode_with_offsets("hi", true).unwrap();
        assert_eq!(count(&enc.ids, bos), 1);
        let batch = tk.encode_batch(&["hi"], true).unwrap();
        assert_eq!(count(&batch[0], bos), 1);
    }

    #[test]
    fn a_model_that_starts_without_a_bos_gets_none() {
        // Qwen3: ChatML, a `<|endoftext|>` that is no BOS, no `bos_token`.
        let none = byte_level_bpe(&["h i"], &["<|endoftext|>", "<|im_start|>"], None);
        let template = "{%- for m in messages -%}<|im_start|>{{ m['role'] }}\\n{{ m['content'] }}{%- endfor -%}";
        let dir = model_dir(
            &none,
            serde_json::json!({"eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>"}),
            Some(template),
        );
        let tk = super::Tokenizer::from_model_dir(dir.path()).unwrap();
        assert_eq!(
            tk.encode_with_special_tokens("hi").unwrap(),
            none.encode("hi").unwrap()
        );
        assert_eq!(
            tk.encode_batch(&["hi"], true).unwrap()[0],
            none.encode("hi").unwrap()
        );
        // A token that is called `<s>` and is no BOS of the config is not one
        // either: a template that does not write it does not expect it.
        let with_s = byte_level_bpe(&["h i"], &["<s>", "<|im_start|>"], None);
        let dir = model_dir(&with_s, serde_json::json!({}), Some(template));
        let tk = super::Tokenizer::from_model_dir(dir.path()).unwrap();
        assert_eq!(
            tk.encode_with_special_tokens("hi").unwrap(),
            with_s.encode("hi").unwrap()
        );
    }

    #[test]
    fn the_configs_add_bos_token_decides_when_there_is_no_template() {
        let none = byte_level_bpe(&["h i"], &["<bos>"], None);
        let yes = model_dir(
            &none,
            serde_json::json!({"bos_token": "<bos>", "add_bos_token": true}),
            None,
        );
        let tk = super::Tokenizer::from_model_dir(yes.path()).unwrap();
        let bos = tk.inner().token_to_id("<bos>").unwrap();
        assert_eq!(tk.encode_with_special_tokens("hi").unwrap()[0], bos);
        // ...and an explicit `false` beats a template that writes one.
        let no = model_dir(
            &none,
            serde_json::json!({"bos_token": "<bos>", "add_bos_token": false}),
            Some(GEMMA4_TEMPLATE),
        );
        let tk = super::Tokenizer::from_model_dir(no.path()).unwrap();
        assert_eq!(count(&tk.encode_with_special_tokens("hi").unwrap(), bos), 0);
    }

    #[test]
    fn word_level_offsets_follow_whitespace() {
        let tk = word_level(&["<unk>", "a", "bb"], &[]);
        let enc = tk.encode_with_offsets("a  bb", false).unwrap();
        assert_eq!(enc.offsets, vec![(0, 1), (3, 5)]);
    }
}
