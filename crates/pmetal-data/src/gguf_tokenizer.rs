//! Reconstruct a runnable `tokenizers::Tokenizer` from GGUF metadata.
//!
//! GGUF-only checkpoints (no sibling `tokenizer.json`) embed their tokenizer as
//! `tokenizer.ggml.*` metadata. [`pmetal_gguf`] reads that into a neutral
//! [`GgufTokenizerData`]; this module turns it into a HuggingFace
//! `tokenizers::Tokenizer` that splits, merges and numbers text the way the
//! format defines. `tokenizer.ggml.model` picks the algorithm:
//!
//! - **`gpt2`** (Llama 3, Qwen, GLM, gpt-oss, DeepSeek, …) and **`gemma4`**:
//!   byte-pair encoding with the vocabulary and the ordered `merges`.
//!   `tokenizer.ggml.pre` names how text is cut into pieces first (a list of
//!   patterns); a name this does not know is an error
//!   rather than a guess, since tokenizing with the wrong cuts changes the ids
//!   of ordinary text. Pieces are mapped to the byte-level alphabet, except
//!   for Gemma 4, which works on raw characters with a `▁` for the space and
//!   `<0xNN>` tokens for bytes the vocabulary lacks.
//! - **`llama`** (Llama 2, Mistral, Gemma 2/3, …): SentencePiece's BPE. The
//!   format stores no merges for it, only a score per token, and merging picks
//!   the adjacent pair whose joined token has the highest score; that is a
//!   merge list sorted by descending score, which is what is built here. The
//!   text is prefixed with a `▁` (`tokenizer.ggml.add_space_prefix`), spaces
//!   become `▁`, and bytes the vocabulary lacks fall back to `<0xNN>`.
//!
//! Other models (WordPiece, the Unigram of T5, RWKV, …) are refused. The
//! control and user-defined tokens (chat markers, `<s>`, `<think>`, …) are
//! registered as added tokens so they are never split; control tokens are
//! special (skipped when decoding), user-defined ones are not. The tokenizer
//! adds a BOS (and EOS) the way `tokenizer.ggml.add_bos_token` and
//! `add_eos_token` say.
//!
//! The tests build tokenizers from small vocabularies and check them against
//! the ids a reference implementation of the format produced.

mod pre_table;

use pmetal_core::{PMetalError, Result};
use pmetal_gguf::tokenizer::{GgufTokenizerData, token_type};
use std::collections::HashMap;
use tokenizers::SplitDelimiterBehavior;
use tokenizers::{
    AddedToken, Tokenizer as HfTokenizer,
    decoders::{
        byte_fallback::ByteFallback, fuse::Fuse, sequence::Sequence as DecoderSequence,
        strip::Strip,
    },
    models::bpe::{BpeBuilder, Merges, Vocab},
    normalizers::{NFC, Prepend, Replace, Sequence as NormalizerSequence},
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence as PreTokenizerSequence,
        split::{Split, SplitPattern},
    },
    processors::template::TemplateProcessing,
};

/// SentencePiece whitespace marker (U+2581, "▁").
const SPM_SPACE: &str = "\u{2581}";

/// How a `tokenizer.ggml.model` value tokenizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Algorithm {
    /// Byte-pair encoding over the stored merges.
    Bpe,
    /// SentencePiece BPE, merging by token score.
    Spm,
}

/// Build a `tokenizers::Tokenizer` from GGUF tokenizer metadata.
pub fn build_tokenizer(data: &GgufTokenizerData) -> Result<HfTokenizer> {
    if !data.is_usable() {
        return Err(PMetalError::Tokenizer(
            "GGUF metadata has no tokenizer vocabulary (tokenizer.ggml.tokens)".into(),
        ));
    }
    for (what, len) in [
        ("token types", data.token_types.len()),
        ("scores", data.scores.len()),
    ] {
        if len != 0 && len != data.tokens.len() {
            return Err(PMetalError::Tokenizer(format!(
                "GGUF tokenizer has {} tokens but {len} {what}",
                data.tokens.len()
            )));
        }
    }

    let algorithm = match data.model.as_str() {
        "gpt2" | "gemma4" => Algorithm::Bpe,
        "llama" => Algorithm::Spm,
        // Older exports leave the model out; merges say which it is.
        "" if !data.merges.is_empty() => Algorithm::Bpe,
        "" if !data.scores.is_empty() => Algorithm::Spm,
        other => {
            return Err(PMetalError::Tokenizer(format!(
                "unsupported GGUF tokenizer model {other:?}: only byte-pair (`gpt2`, `gemma4`) \
                 and SentencePiece (`llama`) tokenizers can be reconstructed; use the model's \
                 tokenizer.json"
            )));
        }
    };

    let (mut tokenizer, add_bos_default) = match algorithm {
        Algorithm::Bpe => build_bpe(data)?,
        Algorithm::Spm => (build_spm(data)?, true),
    };
    register_added_tokens(&mut tokenizer, data)?;
    add_bos_and_eos(&mut tokenizer, data, add_bos_default)?;
    Ok(tokenizer)
}

/// The vocabulary text of every token, as `token_to_id` keys: a token with no
/// text is named by its id, and when two share text the later wins.
fn vocabulary(data: &GgufTokenizerData) -> Vocab {
    data.tokens
        .iter()
        .enumerate()
        .map(|(id, text)| {
            let text = if text.is_empty() {
                format!("[EMPTY_{id}]")
            } else {
                text.clone()
            };
            (text, id as u32)
        })
        .collect()
}

fn token_string(data: &GgufTokenizerData, id: Option<u32>) -> Option<String> {
    id.and_then(|i| data.tokens.get(i as usize).cloned())
}

/// The type code of token `id` (`NORMAL` when the file lists none).
fn type_of(data: &GgufTokenizerData, id: usize) -> i32 {
    data.token_types
        .get(id)
        .copied()
        .unwrap_or(token_type::NORMAL)
}

fn config_error(what: &str, e: impl std::fmt::Display) -> PMetalError {
    PMetalError::Tokenizer(format!("GGUF tokenizer {what}: {e}"))
}

/// Drop the merges whose parts or result are not tokens: the vocabulary
/// cannot express them, and the builder would refuse the whole list.
fn usable_merges(merges: Vec<(String, String)>, vocab: &Vocab) -> Merges {
    let before = merges.len();
    let kept: Merges = merges
        .into_iter()
        .filter(|(left, right)| {
            vocab.contains_key(left)
                && vocab.contains_key(right)
                && vocab.contains_key(&format!("{left}{right}"))
        })
        .collect();
    if kept.len() != before {
        tracing::warn!(
            dropped = before - kept.len(),
            "GGUF tokenizer: merges naming tokens outside the vocabulary were ignored"
        );
    }
    kept
}

// ---------------------------------------------------------------------------
// Byte-pair encoding
// ---------------------------------------------------------------------------

/// The merge list as pairs. A merge is stored as one `"left right"` string;
/// the left part may itself begin with a space, so the split is at the first
/// space after the first character.
fn parse_merges(merges: &[String]) -> Merges {
    merges
        .iter()
        .filter_map(|merge| {
            let first_len = merge.chars().next()?.len_utf8();
            let space = first_len + merge[first_len..].find(' ')?;
            Some((merge[..space].to_string(), merge[space + 1..].to_string()))
        })
        .collect()
}

fn build_bpe(data: &GgufTokenizerData) -> Result<(HfTokenizer, bool)> {
    let pre = if data.model == "gemma4" {
        "gemma4"
    } else {
        data.pre.as_deref().unwrap_or("")
    };
    if data.pre.is_none() && data.model != "gemma4" {
        tracing::warn!(
            "GGUF tokenizer has no `tokenizer.ggml.pre`: using the default pre-tokenizer, which \
             tokenizes some text differently from the model's own"
        );
    }
    let spec = pre_table::lookup(pre).ok_or_else(|| {
        PMetalError::Tokenizer(format!(
            "unsupported GGUF pre-tokenizer {pre:?} (`tokenizer.ggml.pre`): its patterns are \
             not known here, so text would be split differently from the model's own; use \
             the model's tokenizer.json"
        ))
    })?;

    let vocab = vocabulary(data);
    let merges = usable_merges(parse_merges(&data.merges), &vocab);
    let mut builder = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .ignore_merges(spec.ignore_merges);
    if !spec.byte_level {
        builder = builder.byte_fallback(true).fuse_unk(true);
    }
    if let Some(unk) = token_string(data, data.unk_id) {
        builder = builder.unk_token(unk);
    }
    let bpe = builder.build().map_err(|e| config_error("BPE", e))?;
    let mut tokenizer = HfTokenizer::new(bpe);

    // Qwen's tokenizer normalizes to NFC; the metadata does not say so.
    if matches!(pre, "qwen2" | "qwen35") {
        tokenizer
            .with_normalizer(Some(NFC))
            .map_err(|e| config_error("normalizer", e))?;
    }

    let mut splits = Vec::with_capacity(spec.patterns.len() + 1);
    for pattern in spec.patterns {
        splits.push(
            Split::new(
                SplitPattern::Regex((*pattern).to_string()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map_err(|e| config_error("pre-tokenizer pattern", e))?
            .into(),
        );
    }
    if spec.byte_level {
        // Every byte of a piece becomes its stand-in character; the pieces
        // were cut by the patterns above, not by the byte-level pass.
        splits.push(ByteLevel::new(false, false, false).into());
        tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    } else {
        // Raw characters, the space written `▁`.
        tokenizer
            .with_normalizer(Some(
                Replace::new(" ", SPM_SPACE).map_err(|e| config_error("normalizer", e))?,
            ))
            .map_err(|e| config_error("normalizer", e))?;
        tokenizer.with_decoder(Some(DecoderSequence::new(vec![
            Replace::new(SPM_SPACE, " ")
                .map_err(|e| config_error("decoder", e))?
                .into(),
            ByteFallback::new().into(),
            Fuse::new().into(),
        ])));
    }
    tokenizer.with_pre_tokenizer(Some(PreTokenizerSequence::new(splits)));
    Ok((tokenizer, spec.add_bos || pre == "gemma4"))
}

// ---------------------------------------------------------------------------
// SentencePiece BPE
// ---------------------------------------------------------------------------

/// The merges a score list stands for. Of two adjacent pieces, the one whose
/// joined token scores higher merges first, so every way of cutting a token
/// into two tokens is a merge, ranked by the token's score (and, among equal
/// scores, the longer pieces first). Only ordinary and user-defined tokens
/// are merge results (Gemma's runs of `▁` are user-defined): control, byte
/// and unused tokens are never built from smaller ones.
fn merges_from_scores(data: &GgufTokenizerData, vocab: &Vocab) -> Merges {
    struct Candidate<'a> {
        score: f32,
        left: &'a str,
        right: &'a str,
        left_id: u32,
        right_id: u32,
    }
    let mut candidates: Vec<Candidate<'_>> = Vec::new();
    for (id, token) in data.tokens.iter().enumerate() {
        if !matches!(
            type_of(data, id),
            token_type::NORMAL | token_type::USER_DEFINED
        ) || token.is_empty()
        {
            continue;
        }
        // The vocabulary keeps one id per text; a duplicate defers to it.
        if vocab.get(token.as_str()) != Some(&(id as u32)) {
            continue;
        }
        let score = data.scores.get(id).copied().unwrap_or(0.0);
        let mut cuts: Vec<Candidate<'_>> = token
            .char_indices()
            .skip(1)
            .filter_map(|(at, _)| {
                let (left, right) = token.split_at(at);
                Some(Candidate {
                    score,
                    left,
                    right,
                    left_id: *vocab.get(left)?,
                    right_id: *vocab.get(right)?,
                })
            })
            .collect();
        cuts.sort_by_key(|c| (c.left_id, c.right_id));
        candidates.extend(cuts);
    }
    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| b.left.chars().count().cmp(&a.left.chars().count()))
            .then_with(|| b.right.chars().count().cmp(&a.right.chars().count()))
    });
    candidates
        .into_iter()
        .map(|c| (c.left.to_string(), c.right.to_string()))
        .collect()
}

fn build_spm(data: &GgufTokenizerData) -> Result<HfTokenizer> {
    if data.scores.len() != data.tokens.len() {
        return Err(PMetalError::Tokenizer(format!(
            "GGUF SentencePiece tokenizer has {} tokens but {} scores",
            data.tokens.len(),
            data.scores.len()
        )));
    }
    let vocab = vocabulary(data);
    let merges = merges_from_scores(data, &vocab);
    let mut builder = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .byte_fallback(true)
        .fuse_unk(true);
    if let Some(unk) = token_string(data, data.unk_id) {
        builder = builder.unk_token(unk);
    }
    let bpe = builder.build().map_err(|e| config_error("BPE", e))?;
    let mut tokenizer = HfTokenizer::new(bpe);

    // The text is prefixed with `▁` (after every special token as well as at
    // the start) and spaces become `▁`; no cutting into pieces happens, the
    // whole text merges at once.
    let add_space_prefix = data.add_space_prefix.unwrap_or(true);
    let replace = Replace::new(" ", SPM_SPACE).map_err(|e| config_error("normalizer", e))?;
    let normalizer = if add_space_prefix {
        NormalizerSequence::new(vec![
            Prepend::new(SPM_SPACE.to_string()).into(),
            replace.into(),
        ])
    } else {
        NormalizerSequence::new(vec![replace.into()])
    };
    tokenizer
        .with_normalizer(Some(normalizer))
        .map_err(|e| config_error("normalizer", e))?;

    // Decode: map `▁` back to spaces, reverse byte fallback, fuse the pieces,
    // and drop the one leading space the prefix made.
    let mut decoders = vec![
        Replace::new(SPM_SPACE, " ")
            .map_err(|e| config_error("decoder", e))?
            .into(),
        ByteFallback::new().into(),
        Fuse::new().into(),
    ];
    if add_space_prefix {
        decoders.push(Strip::new(' ', 1, 0).into());
    }
    tokenizer.with_decoder(Some(DecoderSequence::new(decoders)));
    Ok(tokenizer)
}

// ---------------------------------------------------------------------------
// Added tokens and the BOS / EOS template
// ---------------------------------------------------------------------------

/// Register the control, user-defined and unknown tokens as added tokens, so
/// they are matched in the text before it is cut and survive a round trip.
/// Control and unknown tokens are special (skipped when decoding);
/// user-defined ones (`<think>`, `<tool_call>`, …) are ordinary text.
fn register_added_tokens(tokenizer: &mut HfTokenizer, data: &GgufTokenizerData) -> Result<()> {
    let mut special = Vec::new();
    let mut ordinary = Vec::new();
    for (id, text) in data.tokens.iter().enumerate() {
        let kind = type_of(data, id);
        if text.is_empty() {
            continue;
        }
        match kind {
            token_type::CONTROL | token_type::UNKNOWN => {
                special.push(AddedToken::from(text.clone(), true).normalized(false));
            }
            token_type::USER_DEFINED => {
                ordinary.push(AddedToken::from(text.clone(), false).normalized(false));
            }
            _ => {}
        }
    }
    tokenizer
        .add_special_tokens(special)
        .map_err(|e| config_error("special tokens", e))?;
    tokenizer
        .add_tokens(ordinary)
        .map_err(|e| config_error("added tokens", e))?;
    Ok(())
}

/// Put the BOS (and EOS) around an encoding that asks for special tokens, as
/// `tokenizer.ggml.add_bos_token` / `add_eos_token` say. Without the keys a
/// SentencePiece tokenizer adds a BOS and a byte-pair one does what its
/// `pre` name does (Llama 3 and Mistral's Tekken add one, most do not).
fn add_bos_and_eos(
    tokenizer: &mut HfTokenizer,
    data: &GgufTokenizerData,
    add_bos_default: bool,
) -> Result<()> {
    let bos = token_string(data, data.bos_id).zip(data.bos_id);
    let eos = token_string(data, data.eos_id).zip(data.eos_id);
    let bos = bos.filter(|_| data.add_bos.unwrap_or(add_bos_default));
    let eos = eos.filter(|_| data.add_eos.unwrap_or(false));
    if bos.is_none() && eos.is_none() {
        return Ok(());
    }

    // A template piece is a token's text, or `$A` / `$B` for the sequences.
    let special = |(token, _): &(String, u32)| token.clone();
    let mut single = Vec::new();
    single.extend(bos.iter().map(special));
    single.push("$A".to_string());
    single.extend(eos.iter().map(special));
    let mut pair = single.clone();
    pair.extend(bos.iter().map(special));
    pair.push("$B:1".to_string());
    pair.extend(eos.iter().map(special));

    let tokens: HashMap<String, u32> = bos.iter().chain(&eos).cloned().collect();
    let processor = TemplateProcessing::builder()
        .try_single(single)
        .map_err(|e| config_error("template", e))?
        .try_pair(pair)
        .map_err(|e| config_error("template", e))?
        .special_tokens(tokens.into_iter().collect::<Vec<_>>())
        .build()
        .map_err(|e| config_error("template", e))?;
    tokenizer.with_post_processor(Some(processor));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str = include_str!("../tests/fixtures/gguf_tokenizer_reference.json");

    fn fixture() -> Value {
        serde_json::from_str(FIXTURE).expect("the fixture parses")
    }

    fn strings(value: &Value) -> Vec<String> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    }

    fn ids(value: &Value) -> Vec<u32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect()
    }

    fn types(value: &Value) -> Vec<i32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect()
    }

    fn scores(value: &Value) -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_f64().unwrap() as f32)
            .collect()
    }

    fn encode(tokenizer: &HfTokenizer, text: &str, add_special_tokens: bool) -> Vec<u32> {
        tokenizer
            .encode_fast(text, add_special_tokens)
            .unwrap()
            .get_ids()
            .to_vec()
    }

    fn spm_data(spm: &Value) -> GgufTokenizerData {
        GgufTokenizerData {
            model: "llama".into(),
            tokens: strings(&spm["tokens"]),
            scores: scores(&spm["scores"]),
            token_types: types(&spm["token_types"]),
            bos_id: Some(1),
            eos_id: Some(2),
            unk_id: Some(0),
            add_space_prefix: Some(spm["add_space_prefix"].as_bool().unwrap()),
            ..Default::default()
        }
    }

    fn tiny_bpe() -> GgufTokenizerData {
        let bpe = &fixture()["bpe"];
        GgufTokenizerData {
            model: "gpt2".into(),
            pre: Some("llama-bpe".into()),
            tokens: strings(&bpe["tokens"]),
            token_types: types(&bpe["token_types"]),
            merges: strings(&bpe["merges"]),
            ..Default::default()
        }
    }

    /// Every `tokenizer.ggml.pre` name cuts text the way the format's own
    /// tokenizer does: the ids of ~80 texts (numbers, contractions, newlines,
    /// CJK, emoji, combining marks, special tokens) agree for each of 21 names
    /// over one small byte-level vocabulary.
    #[test]
    fn byte_pair_tokenizers_match_the_reference_for_every_pre_tokenizer() {
        let fixture = fixture();
        let cases = strings(&fixture["cases"]);
        let mut checked = 0;
        for (pre, expected) in fixture["bpe"]["ids"].as_object().unwrap() {
            let data = GgufTokenizerData {
                pre: Some(pre.clone()),
                ..tiny_bpe()
            };
            let tokenizer = build_tokenizer(&data).unwrap_or_else(|e| panic!("{pre}: {e}"));
            for (text, want) in cases.iter().zip(expected.as_array().unwrap()) {
                // Qwen's tokenizer composes combining marks first, which the
                // format's own tokenizer does not (see `qwen_text_is_normalized_to_nfc`).
                let combining = text.chars().any(|c| ('\u{300}'..='\u{36f}').contains(&c));
                if combining && matches!(pre.as_str(), "qwen2" | "qwen35") {
                    continue;
                }
                // These two split cased runs with the o200k pattern (checked
                // against gpt-oss's own tokenizer below); the reference's
                // regex engine cuts "camelCase" differently from that tokenizer.
                let camel = text
                    .chars()
                    .zip(text.chars().skip(1))
                    .any(|(a, b)| a.is_lowercase() && b.is_uppercase());
                if camel && matches!(pre.as_str(), "tiny_aya" | "youtu") {
                    continue;
                }
                assert_eq!(
                    encode(&tokenizer, text, false),
                    ids(want),
                    "{pre}: {text:?}"
                );
                checked += 1;
            }
        }
        assert!(checked >= 20 * 70, "{checked} comparisons");
    }

    /// SentencePiece BPE: merging by score, `▁` for spaces, the prefix on and
    /// off, byte fallback, user-defined and control tokens, and the BOS the
    /// metadata asks for.
    #[test]
    fn sentencepiece_tokenizers_match_the_reference() {
        let fixture = fixture();
        let cases = strings(&fixture["cases"]);
        for (name, spm) in fixture["spm"].as_object().unwrap() {
            let tokenizer = build_tokenizer(&spm_data(spm)).unwrap();
            for (text, want) in cases.iter().zip(spm["ids"].as_array().unwrap()) {
                assert_eq!(
                    encode(&tokenizer, text, false),
                    ids(want),
                    "{name}: {text:?}"
                );
            }
            // With specials on, a BOS goes in front (SentencePiece's default).
            for (text, want) in cases.iter().zip(spm["ids_with_bos"].as_array().unwrap()) {
                assert_eq!(
                    encode(&tokenizer, text, true),
                    ids(want),
                    "{name} +bos: {text:?}"
                );
            }
        }
    }

    #[test]
    fn every_pre_tokenizer_pattern_compiles() {
        for name in pre_table::NAMES {
            let data = GgufTokenizerData {
                pre: Some((*name).to_string()),
                ..tiny_bpe()
            };
            build_tokenizer(&data).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }

    #[test]
    fn an_unknown_pre_tokenizer_or_model_is_refused_not_guessed() {
        let data = GgufTokenizerData {
            pre: Some("a-new-model".into()),
            ..tiny_bpe()
        };
        let e = build_tokenizer(&data).unwrap_err().to_string();
        assert!(
            e.contains("a-new-model") && e.contains("tokenizer.json"),
            "{e}"
        );

        let data = GgufTokenizerData {
            model: "bert".into(),
            ..tiny_bpe()
        };
        let e = build_tokenizer(&data).unwrap_err().to_string();
        assert!(e.contains("bert"), "{e}");
        assert!(build_tokenizer(&GgufTokenizerData::default()).is_err());
    }

    /// Control tokens are special; user-defined ones (`<think>`) are text
    /// that happens to be one token, so decoding keeps them.
    #[test]
    fn control_tokens_are_special_and_user_defined_ones_are_not() {
        let data = tiny_bpe();
        let tokenizer = build_tokenizer(&data).unwrap();
        let text = "<|im_start|>hi<think>x</think><|im_end|>";
        let ids = encode(&tokenizer, text, false);
        let id_of = |t: &str| data.tokens.iter().position(|x| x == t).unwrap() as u32;
        assert!(ids.contains(&id_of("<|im_start|>")) && ids.contains(&id_of("<think>")));
        assert_eq!(tokenizer.decode(&ids, true).unwrap(), "hi<think>x</think>");
        assert_eq!(tokenizer.decode(&ids, false).unwrap(), text);
    }

    /// `add_bos_token` / `add_eos_token` decide what the post-processor adds;
    /// a family whose name says it adds a BOS (Llama 3) does so unless the
    /// metadata says not to.
    #[test]
    fn bos_and_eos_follow_the_metadata() {
        let base = tiny_bpe();
        let bos = base
            .tokens
            .iter()
            .position(|t| t == "<|endoftext|>")
            .unwrap() as u32;
        let eos = base.tokens.iter().position(|t| t == "<|im_end|>").unwrap() as u32;
        let with = |pre: &str, add_bos, add_eos| {
            let data = GgufTokenizerData {
                pre: Some(pre.into()),
                bos_id: Some(bos),
                eos_id: Some(eos),
                add_bos,
                add_eos,
                ..tiny_bpe()
            };
            encode(&build_tokenizer(&data).unwrap(), "hi", true)
        };
        let plain = with("qwen2", None, None);
        assert!(!plain.contains(&bos) && !plain.contains(&eos));
        let llama3 = with("llama-bpe", None, None);
        assert_eq!(llama3[0], bos);
        assert_eq!(llama3[1..], plain[..]);
        assert_eq!(with("llama-bpe", Some(false), None), plain);
        let both = with("qwen2", Some(true), Some(true));
        assert_eq!((both[0], *both.last().unwrap()), (bos, eos));
    }

    /// Qwen's tokenizer normalizes to NFC, so a decomposed `é` is the same
    /// ids as the composed one.
    #[test]
    fn qwen_text_is_normalized_to_nfc() {
        let data = GgufTokenizerData {
            pre: Some("qwen35".into()),
            ..tiny_bpe()
        };
        let tokenizer = build_tokenizer(&data).unwrap();
        assert_eq!(
            encode(&tokenizer, "cafe\u{301}", false),
            encode(&tokenizer, "caf\u{e9}", false)
        );
        let llama = build_tokenizer(&tiny_bpe()).unwrap();
        assert_ne!(
            encode(&llama, "cafe\u{301}", false),
            encode(&llama, "caf\u{e9}", false)
        );
    }

    /// A merge naming a token the vocabulary lacks is skipped; it used to
    /// make the whole tokenizer fail to build.
    #[test]
    fn merges_outside_the_vocabulary_are_ignored() {
        let mut data = tiny_bpe();
        data.merges.insert(0, "nope zilch".into());
        let tokenizer = build_tokenizer(&data).unwrap();
        let clean = build_tokenizer(&tiny_bpe()).unwrap();
        assert_eq!(
            encode(&tokenizer, "the quick brown fox", false),
            encode(&clean, "the quick brown fox", false)
        );
    }

    #[test]
    fn mismatched_arrays_are_refused() {
        let mut data = tiny_bpe();
        data.token_types.pop();
        assert!(build_tokenizer(&data).is_err());
    }

    /// Real vocabularies, when a directory of them is given:
    /// `PMETAL_GGUF_TOKENIZER_DIR` holds `<name>.gguf` next to `<name>.json`,
    /// a list of `{"text", "ids"}` the model's own tokenizer produced (ids
    /// without BOS). Every text must tokenize identically; a vocabulary with
    /// no reference must at least round-trip the fixture's texts.
    #[test]
    fn real_vocabularies_match_when_given() {
        let Some(dir) = std::env::var_os("PMETAL_GGUF_TOKENIZER_DIR") else {
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let mut checked = 0;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if path.extension().and_then(|e| e.to_str()) != Some("gguf") || name.starts_with("._") {
                continue;
            }
            let content = pmetal_gguf::GgufContent::from_file(&path).unwrap();
            let data = content.tokenizer_data().unwrap();
            let tokenizer = build_tokenizer(&data).unwrap_or_else(|e| panic!("{name}: {e}"));
            let expected = path.with_extension("json");
            if !expected.exists() {
                for text in strings(&fixture()["cases"]) {
                    let got = encode(&tokenizer, &text, false);
                    let back = tokenizer.decode(&got, false).unwrap();
                    assert_eq!(back.trim_start(), text.trim_start(), "{name}: {text:?}");
                }
                continue;
            }
            let reference: Vec<Value> =
                serde_json::from_str(&std::fs::read_to_string(expected).unwrap()).unwrap();
            for case in reference {
                let text = case["text"].as_str().unwrap();
                assert_eq!(
                    encode(&tokenizer, text, false),
                    ids(&case["ids"]),
                    "{name}: {text:?}"
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "no reference in {}", dir.display());
    }
}
