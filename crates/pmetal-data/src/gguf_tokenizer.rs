//! Reconstruct a runnable `tokenizers::Tokenizer` from GGUF metadata.
//!
//! GGUF-only checkpoints (no sibling `tokenizer.json`) embed their tokenizer as
//! `tokenizer.ggml.*` metadata. [`pmetal_gguf`] reads that into a neutral
//! [`GgufTokenizerData`]; this module turns it into a HuggingFace
//! `tokenizers::Tokenizer`, mirroring how llama.cpp's two dominant tokenizer
//! models map onto the `tokenizers` crate:
//!
//! - **BPE** (`gpt2`-family: Qwen2/Qwen3, Phi, GPT-OSS): vocab + `merges` →
//!   [`tokenizers::models::bpe::BPE`] with a byte-level pre-tokenizer/decoder.
//! - **SentencePiece / SPM** (`llama`: Llama, Mistral): vocab + per-token
//!   `scores` → [`tokenizers::models::unigram::Unigram`] with a Metaspace
//!   pre-tokenizer/decoder and byte fallback.
//!
//! Control and user-defined tokens (chat markers, `<s>`, `</s>`, …) are
//! registered as special tokens so they round-trip verbatim.
//!
//! Byte-exact parity against llama.cpp on real checkpoints is validated in
//! internal testing; the committed tests below assert the reconstruction
//! mechanics (round-trip encode/decode + special-token handling) on synthetic
//! vocabularies.

use ahash::AHashMap;
use pmetal_core::{PMetalError, Result};
use pmetal_gguf::tokenizer::GgufTokenizerData;
use tokenizers::{
    AddedToken, Tokenizer as HfTokenizer,
    decoders::{
        byte_fallback::ByteFallback, fuse::Fuse, sequence::Sequence as DecoderSequence,
        strip::Strip,
    },
    models::{bpe::BpeBuilder, unigram::Unigram},
    normalizers::{Prepend, Replace, Sequence as NormalizerSequence},
    pre_tokenizers::{
        byte_level::ByteLevel,
        metaspace::{Metaspace, PrependScheme},
    },
};

/// SentencePiece whitespace marker (U+2581, "▁").
const SPM_SPACE: char = '\u{2581}';

/// Build a `tokenizers::Tokenizer` from GGUF tokenizer metadata.
pub fn build_tokenizer(data: &GgufTokenizerData) -> Result<HfTokenizer> {
    if !data.is_usable() {
        return Err(PMetalError::Tokenizer(
            "GGUF metadata has no tokenizer vocabulary (tokenizer.ggml.tokens)".into(),
        ));
    }

    // Presence of merge rules is the reliable BPE discriminator; the `model`
    // string ("gpt2" / "llama") is a secondary hint for older exports.
    let is_bpe = !data.merges.is_empty() || data.model == "gpt2";
    let mut tokenizer = if is_bpe {
        build_bpe(data)?
    } else {
        build_unigram(data)?
    };

    register_special_tokens(&mut tokenizer, data)?;
    Ok(tokenizer)
}

fn token_string(data: &GgufTokenizerData, id: Option<u32>) -> Option<String> {
    id.and_then(|i| data.tokens.get(i as usize).cloned())
}

fn build_bpe(data: &GgufTokenizerData) -> Result<HfTokenizer> {
    let vocab: AHashMap<String, u32> = data
        .tokens
        .iter()
        .enumerate()
        .map(|(id, tok)| (tok.clone(), id as u32))
        .collect();

    // GGUF stores each merge as a single "left right" string.
    let merges: Vec<(String, String)> = data
        .merges
        .iter()
        .filter_map(|m| {
            m.split_once(' ')
                .map(|(a, b)| (a.to_string(), b.to_string()))
        })
        .collect();

    let mut builder = BpeBuilder::new().vocab_and_merges(vocab, merges);
    if let Some(unk) = token_string(data, data.unk_id) {
        builder = builder.unk_token(unk);
    }
    let bpe = builder
        .build()
        .map_err(|e| PMetalError::Tokenizer(format!("GGUF BPE build: {e}")))?;

    let mut tokenizer = HfTokenizer::new(bpe);
    // gpt2-style byte-level: no added prefix space, byte<->unicode remapping on
    // both encode and decode.
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
    tokenizer.with_decoder(Some(ByteLevel::new(false, true, true)));
    Ok(tokenizer)
}

fn build_unigram(data: &GgufTokenizerData) -> Result<HfTokenizer> {
    if data.scores.len() != data.tokens.len() {
        return Err(PMetalError::Tokenizer(format!(
            "GGUF SPM tokenizer has {} tokens but {} scores",
            data.tokens.len(),
            data.scores.len()
        )));
    }

    let vocab: Vec<(String, f64)> = data
        .tokens
        .iter()
        .zip(&data.scores)
        .map(|(tok, score)| (tok.clone(), *score as f64))
        .collect();

    let unigram = Unigram::from(vocab, data.unk_id.map(|i| i as usize), true)
        .map_err(|e| PMetalError::Tokenizer(format!("GGUF Unigram build: {e}")))?;

    let mut tokenizer = HfTokenizer::new(unigram);

    // SentencePiece normalization: prepend a leading whitespace marker and map
    // spaces to it, so the Unigram model sees the SPM surface form.
    let normalizer = NormalizerSequence::new(vec![
        Prepend::new(SPM_SPACE.to_string()).into(),
        Replace::new(" ", SPM_SPACE.to_string())
            .map_err(|e| PMetalError::Tokenizer(format!("GGUF SPM normalizer: {e}")))?
            .into(),
    ]);
    tokenizer
        .with_normalizer(Some(normalizer))
        .map_err(|e| PMetalError::Tokenizer(format!("GGUF SPM normalizer: {e}")))?;
    tokenizer.with_pre_tokenizer(Some(Metaspace::new(SPM_SPACE, PrependScheme::Never, false)));

    // Decode: reverse byte fallback (<0xNN> -> raw bytes), fuse the pieces, map
    // ▁ back to spaces, then strip the single leading space SPM prepends.
    let decoder = DecoderSequence::new(vec![
        Replace::new(SPM_SPACE.to_string(), " ")
            .map_err(|e| PMetalError::Tokenizer(format!("GGUF SPM decoder: {e}")))?
            .into(),
        ByteFallback::new().into(),
        Fuse::new().into(),
        Strip::new(' ', 1, 0).into(),
    ]);
    tokenizer.with_decoder(Some(decoder));
    Ok(tokenizer)
}

/// Register CONTROL / USER_DEFINED tokens as special so they are never split
/// and survive an encode→decode round-trip verbatim.
fn register_special_tokens(tokenizer: &mut HfTokenizer, data: &GgufTokenizerData) -> Result<()> {
    let special: Vec<AddedToken> = data
        .special_token_ids()
        .into_iter()
        .filter_map(|id| data.tokens.get(id as usize))
        .map(|tok| AddedToken::from(tok.clone(), true))
        .collect();
    if !special.is_empty() {
        tokenizer
            .add_special_tokens(special)
            .map_err(|e| PMetalError::Tokenizer(format!("GGUF special tokens: {e}")))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_gguf::tokenizer::token_type;

    #[test]
    fn bpe_round_trip_with_merges() {
        // Byte-level BPE: printable ASCII maps to itself, so "a"+"b" merge to
        // "ab". A control token must be preserved verbatim.
        let data = GgufTokenizerData {
            model: "gpt2".into(),
            tokens: vec!["a".into(), "b".into(), "ab".into(), "<s>".into()],
            merges: vec!["a b".into()],
            token_types: vec![
                token_type::NORMAL,
                token_type::NORMAL,
                token_type::NORMAL,
                token_type::CONTROL,
            ],
            bos_id: Some(3),
            ..Default::default()
        };
        let tk = build_tokenizer(&data).expect("build bpe");

        let enc = tk.encode("ab", false).expect("encode");
        assert_eq!(enc.get_ids(), &[2], "‘a’+‘b’ should merge to ‘ab’ (id 2)");
        assert_eq!(tk.decode(&[2], false).expect("decode"), "ab");

        // Control token stays a single unit and survives the round-trip.
        let enc = tk.encode("<s>ab", true).expect("encode special");
        assert_eq!(enc.get_ids(), &[3, 2]);
    }

    #[test]
    fn spm_round_trip_with_scores() {
        // Unigram/SPM: the normalizer prepends ‘▁’ and maps spaces to it; the
        // whole-word piece "▁hi" outscores the character segmentation.
        let data = GgufTokenizerData {
            model: "llama".into(),
            tokens: vec![
                "<unk>".into(),
                "\u{2581}hi".into(),
                "\u{2581}".into(),
                "h".into(),
                "i".into(),
            ],
            scores: vec![0.0, 0.0, -5.0, -5.0, -5.0],
            token_types: vec![token_type::UNKNOWN, 1, 1, 1, 1],
            unk_id: Some(0),
            ..Default::default()
        };
        let tk = build_tokenizer(&data).expect("build unigram");

        let enc = tk.encode("hi", false).expect("encode");
        assert_eq!(enc.get_ids(), &[1], "‘hi’ should be the single piece ‘▁hi’");
        assert_eq!(tk.decode(&[1], false).expect("decode"), "hi");
    }

    #[test]
    fn empty_vocab_is_rejected() {
        let data = GgufTokenizerData::default();
        assert!(build_tokenizer(&data).is_err());
    }
}
