//! Dependency-free extraction of tokenizer data from GGUF metadata.
//!
//! GGUF checkpoints embed their tokenizer as `tokenizer.ggml.*` metadata rather
//! than a HuggingFace `tokenizer.json`. This module reads that metadata into a
//! neutral [`GgufTokenizerData`] struct — no `tokenizers` dependency, so the
//! low-level GGUF crate stays lean. Higher layers (e.g. `pmetal-data`) turn this
//! into a runnable `tokenizers::Tokenizer`.

use crate::{GgufContent, MetadataValue, keys};

/// llama.cpp token-type codes (`enum llama_token_type`). Stored per token in
/// `tokenizer.ggml.token_type`.
pub mod token_type {
    /// Ordinary vocabulary token.
    pub const NORMAL: i32 = 1;
    /// Unknown token (`<unk>`).
    pub const UNKNOWN: i32 = 2;
    /// Control / special token (`<s>`, `</s>`, chat markers) — must not be split.
    pub const CONTROL: i32 = 3;
    /// User-defined added token — must not be split.
    pub const USER_DEFINED: i32 = 4;
    /// Unused slot.
    pub const UNUSED: i32 = 5;
    /// Raw byte token (`<0xNN>`).
    pub const BYTE: i32 = 6;
}

/// Tokenizer data extracted from GGUF metadata.
///
/// Field presence depends on the tokenizer model: BPE checkpoints (`gpt2`)
/// carry `merges` and usually no `scores`; SentencePiece checkpoints (`llama`)
/// carry `scores` and no `merges`.
#[derive(Debug, Clone, Default)]
pub struct GgufTokenizerData {
    /// llama.cpp tokenizer model: `"llama"` (SPM), `"gpt2"` (BPE), `"bert"`, …
    pub model: String,
    /// Pre-tokenizer identifier (`tokenizer.ggml.pre`), e.g. `"llama-bpe"`,
    /// `"qwen2"`, `"default"`. Absent on older exports.
    pub pre: Option<String>,
    /// Vocabulary, indexed by token id.
    pub tokens: Vec<String>,
    /// Per-token unigram scores (SPM). Empty when absent.
    pub scores: Vec<f32>,
    /// Per-token type codes (see [`token_type`]). Empty when absent.
    pub token_types: Vec<i32>,
    /// BPE merge rules, each `"left right"`. Empty when absent.
    pub merges: Vec<String>,
    /// Beginning-of-sequence token id.
    pub bos_id: Option<u32>,
    /// End-of-sequence token id.
    pub eos_id: Option<u32>,
    /// Unknown token id.
    pub unk_id: Option<u32>,
    /// Padding token id.
    pub pad_id: Option<u32>,
    /// Whether the model prepends BOS by default.
    pub add_bos: Option<bool>,
    /// Whether the model appends EOS by default.
    pub add_eos: Option<bool>,
    /// Embedded chat template (`tokenizer.chat_template`), if any.
    pub chat_template: Option<String>,
}

impl GgufTokenizerData {
    /// True when the vocabulary is populated enough to build a tokenizer.
    pub fn is_usable(&self) -> bool {
        !self.tokens.is_empty()
    }

    /// Ids flagged as CONTROL or USER_DEFINED — special tokens that must be
    /// preserved verbatim (never split) during (de)tokenization.
    pub fn special_token_ids(&self) -> Vec<u32> {
        self.token_types
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t == token_type::CONTROL || t == token_type::USER_DEFINED)
            .map(|(i, _)| i as u32)
            .collect()
    }
}

fn get_u32(content: &GgufContent, key: &str) -> Option<u32> {
    match content.get_metadata(key)? {
        MetadataValue::Uint32(v) => Some(*v),
        MetadataValue::Int32(v) => u32::try_from(*v).ok(),
        MetadataValue::Uint64(v) => u32::try_from(*v).ok(),
        MetadataValue::Int64(v) => u32::try_from(*v).ok(),
        MetadataValue::Uint16(v) => Some(*v as u32),
        _ => None,
    }
}

fn get_bool(content: &GgufContent, key: &str) -> Option<bool> {
    match content.get_metadata(key)? {
        MetadataValue::Bool(v) => Some(*v),
        _ => None,
    }
}

fn get_string(content: &GgufContent, key: &str) -> Option<String> {
    match content.get_metadata(key)? {
        MetadataValue::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn string_array(content: &GgufContent, key: &str) -> Vec<String> {
    match content.get_metadata(key) {
        Some(MetadataValue::Array(arr)) => arr
            .iter()
            .filter_map(|v| match v {
                MetadataValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn f32_array(content: &GgufContent, key: &str) -> Vec<f32> {
    match content.get_metadata(key) {
        Some(MetadataValue::Array(arr)) => arr
            .iter()
            .filter_map(|v| match v {
                MetadataValue::Float32(f) => Some(*f),
                MetadataValue::Float64(f) => Some(*f as f32),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn i32_array(content: &GgufContent, key: &str) -> Vec<i32> {
    match content.get_metadata(key) {
        Some(MetadataValue::Array(arr)) => arr
            .iter()
            .filter_map(|v| match v {
                MetadataValue::Int32(i) => Some(*i),
                MetadataValue::Uint32(i) => i32::try_from(*i).ok(),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

impl GgufContent {
    /// Extract tokenizer data from GGUF metadata.
    ///
    /// Returns `None` when the file carries no token vocabulary
    /// (`tokenizer.ggml.tokens`), i.e. there is nothing to reconstruct.
    pub fn tokenizer_data(&self) -> Option<GgufTokenizerData> {
        let tokens = string_array(self, keys::TOKENIZER_TOKENS);
        if tokens.is_empty() {
            return None;
        }
        Some(GgufTokenizerData {
            model: get_string(self, keys::TOKENIZER_MODEL).unwrap_or_default(),
            pre: get_string(self, keys::TOKENIZER_PRE),
            tokens,
            scores: f32_array(self, keys::TOKENIZER_SCORES),
            token_types: i32_array(self, keys::TOKENIZER_TOKEN_TYPE),
            merges: string_array(self, keys::TOKENIZER_MERGES),
            bos_id: get_u32(self, keys::TOKENIZER_BOS_TOKEN_ID),
            eos_id: get_u32(self, keys::TOKENIZER_EOS_TOKEN_ID),
            unk_id: get_u32(self, keys::TOKENIZER_UNK_TOKEN_ID),
            pad_id: get_u32(self, keys::TOKENIZER_PAD_TOKEN_ID),
            add_bos: get_bool(self, keys::TOKENIZER_ADD_BOS),
            add_eos: get_bool(self, keys::TOKENIZER_ADD_EOS),
            chat_template: get_string(self, keys::TOKENIZER_CHAT_TEMPLATE),
        })
    }
}
