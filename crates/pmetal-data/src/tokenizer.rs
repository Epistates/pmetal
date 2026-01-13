//! Tokenizer integration.

use pmetal_core::Result;
use std::path::Path;

/// Wrapper around the tokenizers library.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Load a tokenizer from a local file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Load a tokenizer from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|e| pmetal_core::PMetalError::Tokenizer(e.to_string()))?;
        Ok(Self { inner })
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
    /// Tries common pad token names, falls back to EOS token.
    pub fn pad_token_id(&self) -> Option<u32> {
        // Try common pad token names
        self.inner
            .token_to_id("<pad>")
            .or_else(|| self.inner.token_to_id("[PAD]"))
            .or_else(|| self.inner.token_to_id("<|pad|>"))
            .or_else(|| self.inner.token_to_id("<|finetune_right_pad_id|>"))
            // Fallback to EOS token
            .or_else(|| self.inner.token_to_id("</s>"))
            .or_else(|| self.inner.token_to_id("<|endoftext|>"))
            .or_else(|| self.inner.token_to_id("<|end_of_text|>"))
    }

    /// Get EOS token ID if available.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.inner
            .token_to_id("</s>")
            .or_else(|| self.inner.token_to_id("<|endoftext|>"))
            .or_else(|| self.inner.token_to_id("<|end_of_text|>"))
            .or_else(|| self.inner.token_to_id("<eos>"))
    }

    /// Get BOS token ID if available.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.inner
            .token_to_id("<s>")
            .or_else(|| self.inner.token_to_id("<|begin_of_text|>"))
            .or_else(|| self.inner.token_to_id("<bos>"))
    }
}
