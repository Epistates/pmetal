//! What the export writes for each model family, pinned.
//!
//! Each family's fixture is the shape of that family's real files (the ids
//! and the config keys that decide the metadata), kept small. The values these
//! tests pin are the ones that were wrong, or are easy to get wrong, and that
//! no engine reports: it tokenizes with a wrong beginning-of-text id and
//! generates, a little worse, from there.

use super::sentencepiece::testing::encode_model;
use super::{ExportArch, add_hf_config_metadata, add_tokenizer_metadata};
use pmetal_gguf::{GgufBuilder, GgufContent, MetadataValue};
use serde_json::{Value, json};
use std::io::Cursor;

/// A `tokenizer.json`: the model's own vocabulary (ids from 0), the tokens the
/// tokenizer adds on top (ids after it), and the parts that decide the family.
fn tokenizer_json(
    vocab: &[&str],
    merges: &[&str],
    added: &[(&str, bool)],
    parts: Value,
    byte_fallback: bool,
) -> String {
    let mut model_vocab = serde_json::Map::new();
    for (id, token) in vocab.iter().enumerate() {
        model_vocab.insert((*token).to_string(), json!(id));
    }
    let added_tokens: Vec<Value> = added
        .iter()
        .enumerate()
        .map(|(i, (content, special))| {
            json!({
                "id": vocab.len() + i, "content": content, "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false, "special": special,
            })
        })
        .collect();
    let mut tokenizer = json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null,
            "continuing_subword_prefix": null, "end_of_word_suffix": null,
            "fuse_unk": false, "byte_fallback": byte_fallback, "ignore_merges": false,
            "vocab": model_vocab, "merges": merges,
        },
    });
    for (key, value) in parts.as_object().unwrap() {
        tokenizer[key] = value.clone();
    }
    tokenizer.to_string()
}

fn byte_level() -> Value {
    json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true})
}

/// The metadata an export of the files in `files` writes.
fn export(config: Value, files: &[(&str, Vec<u8>)]) -> (ExportArch, GgufContent) {
    let dir = tempfile::tempdir().unwrap();
    for (name, bytes) in files {
        std::fs::write(dir.path().join(name), bytes).unwrap();
    }
    let arch = ExportArch::detect(Some(&config)).unwrap();
    let mut builder = GgufBuilder::with_model(arch.name, "fixture");
    add_hf_config_metadata(
        &mut builder,
        &arch,
        arch.model_config(Some(&config)).as_ref(),
        crate::cli::quantize::QuantizeMethod::Bf16,
    );
    add_tokenizer_metadata(&mut builder, dir.path(), Some(&config)).unwrap();
    let bytes = builder.build_to_bytes().unwrap();
    (arch, GgufContent::read(&mut Cursor::new(bytes)).unwrap())
}

fn id(content: &GgufContent, key: &str) -> Option<u64> {
    content.get_u64(key)
}

fn array_of_i32(content: &GgufContent, key: &str) -> Vec<i32> {
    match content.get_metadata(key) {
        Some(MetadataValue::Array(values)) => values
            .iter()
            .map(|v| match v {
                MetadataValue::Int32(v) => *v,
                other => panic!("{key}: {other:?}"),
            })
            .collect(),
        other => panic!("{key} is {other:?}"),
    }
}

fn array_of_strings(content: &GgufContent, key: &str) -> Vec<String> {
    match content.get_metadata(key) {
        Some(MetadataValue::Array(values)) => values
            .iter()
            .map(|v| match v {
                MetadataValue::String(s) => s.clone(),
                other => panic!("{key}: {other:?}"),
            })
            .collect(),
        other => panic!("{key} is {other:?}"),
    }
}

const NORMAL: i32 = 1;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const BYTE: i32 = 6;

// ── Qwen3 ─────────────────────────────────────────────────────────────────

/// Qwen3-0.6B: a byte-level BPE vocabulary that has `<s>`, `</s>` and `<unk>`
/// as ordinary tokens (it does), a ChatML control set, no beginning-of-text
/// token in `tokenizer_config.json`, and `add_bos_token` false.
fn qwen3() -> (ExportArch, GgufContent) {
    let tokenizer = tokenizer_json(
        &["<s>", "</s>", "<unk>", "a", "b", "ab"],
        &["a b"],
        &[
            ("<|endoftext|>", true),
            ("<|im_start|>", true),
            ("<|im_end|>", true),
            ("<tool_call>", false),
            ("<think>", false),
        ],
        json!({"pre_tokenizer": byte_level(), "decoder": byte_level()}),
        false,
    );
    let tokenizer_config = json!({
        "add_bos_token": false, "bos_token": null, "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>", "unk_token": null,
        "chat_template": "{{ messages }}",
    });
    let config = json!({
        "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
        "hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072,
        "max_position_embeddings": 40960, "vocab_size": 14, "rope_theta": 1000000,
        "rms_norm_eps": 1e-6, "tie_word_embeddings": true,
        "bos_token_id": 6, "eos_token_id": 8,
    });
    export(
        config,
        &[
            ("tokenizer.json", tokenizer.into_bytes()),
            (
                "tokenizer_config.json",
                tokenizer_config.to_string().into_bytes(),
            ),
        ],
    )
}

#[test]
fn qwen3_special_ids_come_from_the_configs_not_from_ordinary_tokens() {
    let (arch, content) = qwen3();
    assert_eq!(arch.name, "qwen3");
    // `<|endoftext|>` is id 6, `<|im_end|>` id 8. The vocabulary also holds
    // `<s>` (0), `</s>` (1) and `<unk>` (2) as plain text tokens: none of them
    // is a beginning of text, an end of text, or an unknown token.
    assert_eq!(id(&content, "tokenizer.ggml.eos_token_id"), Some(8));
    assert_eq!(id(&content, "tokenizer.ggml.padding_token_id"), Some(6));
    // The tokenizer config names no beginning-of-text token; the model config
    // states one.
    assert_eq!(id(&content, "tokenizer.ggml.bos_token_id"), Some(6));
    assert_eq!(id(&content, "tokenizer.ggml.unknown_token_id"), None);
    assert_eq!(
        content.get_bool("tokenizer.ggml.add_bos_token"),
        Some(false)
    );
}

#[test]
fn qwen3_tokenizer_family_and_token_types() {
    let (_, content) = qwen3();
    assert_eq!(content.get_str("tokenizer.ggml.model"), Some("gpt2"));
    assert_eq!(content.get_str("tokenizer.ggml.pre"), Some("qwen2"));
    let tokens = array_of_strings(&content, "tokenizer.ggml.tokens");
    let types = array_of_i32(&content, "tokenizer.ggml.token_type");
    let type_of = |token: &str| types[tokens.iter().position(|t| t == token).unwrap()];
    // The ChatML markers are control tokens; the tool and reasoning tags are
    // text the model writes, so user-defined (a control token is hidden from
    // the output).
    assert_eq!(type_of("<|im_start|>"), CONTROL);
    assert_eq!(type_of("<|im_end|>"), CONTROL);
    assert_eq!(type_of("<tool_call>"), USER_DEFINED);
    assert_eq!(type_of("<think>"), USER_DEFINED);
    assert_eq!(type_of("ab"), 1);
    // The embedding is wider than the tokenizer: the rest is padding.
    assert_eq!(tokens.len(), 14);
    assert_eq!(tokens[13], "[PAD13]");
    assert_eq!(types[13], 5);
    assert!(content.get_metadata("tokenizer.ggml.merges").is_some());
}

#[test]
fn qwen3_architecture_keys() {
    let (_, content) = qwen3();
    assert_eq!(content.get_u64("qwen3.attention.key_length"), Some(128));
    assert_eq!(content.get_u64("qwen3.attention.value_length"), Some(128));
    assert_eq!(content.get_u64("qwen3.attention.head_count_kv"), Some(8));
    assert_eq!(content.get_f64("qwen3.rope.freq_base"), Some(1_000_000.0));
    // The private key an earlier export wrote beside the standard ones.
    assert!(content.get_metadata("qwen3.attention.head_dim").is_none());
    assert_eq!(
        content.get_str("tokenizer.chat_template"),
        Some("{{ messages }}")
    );
}

// ── Llama 3 ───────────────────────────────────────────────────────────────

/// Llama-3.2-1B-Instruct: its template puts `<|begin_of_text|>` in front of
/// every sequence, and `tokenizer_config.json` does not say so.
fn llama3() -> (ExportArch, GgufContent) {
    let tokenizer = tokenizer_json(
        &["a", "b", "ab"],
        &["a b"],
        &[
            ("<|begin_of_text|>", true),
            ("<|end_of_text|>", true),
            ("<|eot_id|>", true),
            ("<|finetune_right_pad_id|>", true),
        ],
        json!({
            "pre_tokenizer": byte_level(),
            "decoder": byte_level(),
            "post_processor": {"type": "Sequence", "processors": [
                {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true},
                {"type": "TemplateProcessing",
                 "single": [{"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
                            {"Sequence": {"id": "A", "type_id": 0}}],
                 "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
                 "special_tokens": {"<|begin_of_text|>": {"id": "<|begin_of_text|>",
                                    "ids": [3], "tokens": ["<|begin_of_text|>"]}}},
            ]},
        }),
        false,
    );
    let tokenizer_config = json!({
        "bos_token": "<|begin_of_text|>", "eos_token": "<|eot_id|>",
        "pad_token": "<|finetune_right_pad_id|>",
    });
    let config = json!({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": 2048, "num_hidden_layers": 16, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 64, "intermediate_size": 8192,
        "max_position_embeddings": 131072, "vocab_size": 7, "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5, "tie_word_embeddings": true,
        "bos_token_id": 3, "eos_token_id": [3, 5],
        "rope_scaling": {"factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                         "original_max_position_embeddings": 8192, "rope_type": "llama3"},
    });
    export(
        config,
        &[
            ("tokenizer.json", tokenizer.into_bytes()),
            (
                "tokenizer_config.json",
                tokenizer_config.to_string().into_bytes(),
            ),
        ],
    )
}

#[test]
fn llama3_special_ids_and_the_template_that_adds_the_beginning_of_text() {
    let (arch, content) = llama3();
    assert_eq!(arch.name, "llama");
    assert_eq!(id(&content, "tokenizer.ggml.bos_token_id"), Some(3));
    // `eos_token` in the tokenizer config wins over the list in `config.json`.
    assert_eq!(id(&content, "tokenizer.ggml.eos_token_id"), Some(5));
    assert_eq!(id(&content, "tokenizer.ggml.padding_token_id"), Some(6));
    // The tokenizer config has no `add_bos_token`: the template says it.
    assert_eq!(content.get_bool("tokenizer.ggml.add_bos_token"), Some(true));
    assert_eq!(
        content.get_bool("tokenizer.ggml.add_eos_token"),
        Some(false)
    );
    assert_eq!(content.get_str("tokenizer.ggml.model"), Some("gpt2"));
    assert_eq!(content.get_str("tokenizer.ggml.pre"), Some("llama-bpe"));
}

#[test]
fn llama3_rope_scaling_is_a_tensor_not_metadata() {
    let (arch, content) = llama3();
    // Stating it in the metadata as well would apply it twice.
    for key in [
        "llama.rope.scaling.type",
        "llama.rope.scaling.factor",
        "llama.rope.scaling.original_context_length",
    ] {
        assert!(content.get_metadata(key).is_none(), "{key}");
    }
    let extra = arch.extra_tensors();
    assert_eq!(extra.len(), 1);
    assert_eq!(extra[0].0, "rope_freqs.weight");
    assert_eq!(extra[0].1, [32]);
    assert_eq!(content.get_f64("llama.rope.freq_base"), Some(500_000.0));
    assert_eq!(content.get_u64("llama.rope.dimension_count"), Some(64));
}

/// The check text is hashed as `AutoTokenizer.encode` encodes it, with the
/// template's special tokens: without the beginning of text, the hash of a
/// Llama 3 tokenizer is another family's.
#[test]
fn the_pre_tokenizer_hash_is_taken_with_the_special_tokens_added() {
    let dir = tempfile::tempdir().unwrap();
    let tokenizer = tokenizer_json(
        &["a", "b", "ab"],
        &["a b"],
        &[("<|begin_of_text|>", true)],
        json!({
            "pre_tokenizer": byte_level(),
            "post_processor": {"type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
                           {"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [{"Sequence": {"id": "A", "type_id": 0}}],
                "special_tokens": {"<|begin_of_text|>": {"id": "<|begin_of_text|>",
                                   "ids": [2], "tokens": ["<|begin_of_text|>"]}}},
        }),
        false,
    );
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, tokenizer).unwrap();
    let tokenizer = tokenizers::Tokenizer::from_file(&path).unwrap();
    let hashed = super::tokenizer::pre_tokenizer_hash(&tokenizer).unwrap();
    assert_ne!(
        hashed,
        super::tokenizer::hash_token_ids(
            tokenizer
                .encode(super::tokenizer::LLAMA_PRETOKENIZER_CHECK, false)
                .unwrap()
                .get_ids()
        )
    );
}

// ── Gemma 3 ───────────────────────────────────────────────────────────────

/// Gemma-3-1B-it: SentencePiece pieces, scored, with byte pieces, behind a
/// BPE `tokenizer.json` that is not byte-level; one id more than the model's
/// embedding; a run of spaces among the added tokens.
fn gemma3() -> (ExportArch, GgufContent) {
    let tokenizer = tokenizer_json(
        &[
            "<pad>",
            "<eos>",
            "<bos>",
            "<unk>",
            "<0x0A>",
            "\u{2581}the",
            "\u{2581}\u{2581}",
        ],
        &[],
        &[
            ("<start_of_turn>", true),
            ("\u{2581}\u{2581}\u{2581}", false),
            ("[multimodal]", false),
            ("<image_soft_token>", true),
        ],
        json!({
            "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "\u{2581}"},
            "decoder": {"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
                {"type": "ByteFallback"}, {"type": "Fuse"}]},
        }),
        true,
    );
    let model = encode_model(
        &[
            ("<pad>", 0.0, Some(3)),
            ("<eos>", 0.0, Some(3)),
            ("<bos>", 0.0, Some(3)),
            ("<unk>", 0.0, Some(2)),
            ("<0x0A>", 0.0, Some(6)),
            ("\u{2581}the", -1.5, None),
            ("\u{2581}\u{2581}", -2.5, None),
        ],
        false,
    );
    let tokenizer_config = json!({
        "add_bos_token": true, "add_eos_token": false,
        "bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>",
    });
    let config = json!({
        "architectures": ["Gemma3ForCausalLM"], "model_type": "gemma3_text",
        "hidden_size": 1152, "num_hidden_layers": 26, "num_attention_heads": 4,
        "num_key_value_heads": 1, "head_dim": 256, "intermediate_size": 6912,
        "max_position_embeddings": 32768, "vocab_size": 10, "rope_theta": 1000000.0,
        "rope_local_base_freq": 10000.0, "sliding_window": 512, "rms_norm_eps": 1e-6,
    });
    export(
        config,
        &[
            ("tokenizer.json", tokenizer.into_bytes()),
            ("tokenizer.model", model),
            (
                "tokenizer_config.json",
                tokenizer_config.to_string().into_bytes(),
            ),
            ("chat_template.jinja", b"{{ bos_token }}".to_vec()),
        ],
    )
}

#[test]
fn gemma3_is_a_sentencepiece_tokenizer_with_scores() {
    let (arch, content) = gemma3();
    assert_eq!(arch.name, "gemma3");
    assert_eq!(content.get_str("tokenizer.ggml.model"), Some("llama"));
    assert_eq!(content.get_str("tokenizer.ggml.pre"), Some("default"));
    // The pieces are scored, and there is no merge list: the engine merges by
    // score.
    assert!(content.get_metadata("tokenizer.ggml.merges").is_none());
    match content.get_metadata("tokenizer.ggml.scores") {
        Some(MetadataValue::Array(scores)) => {
            assert_eq!(scores.len(), 10);
            assert_eq!(scores[5], MetadataValue::Float32(-1.5));
        }
        other => panic!("{other:?}"),
    }
    // The model's own `add_dummy_prefix` is off.
    assert_eq!(
        content.get_bool("tokenizer.ggml.add_space_prefix"),
        Some(false)
    );
    // The chat template moved out of `tokenizer_config.json` is still found.
    assert_eq!(
        content.get_str("tokenizer.chat_template"),
        Some("{{ bos_token }}")
    );
}

#[test]
fn gemma3_special_ids_types_and_the_extra_token_the_embedding_lacks() {
    let (_, content) = gemma3();
    assert_eq!(id(&content, "tokenizer.ggml.bos_token_id"), Some(2));
    assert_eq!(id(&content, "tokenizer.ggml.eos_token_id"), Some(1));
    assert_eq!(id(&content, "tokenizer.ggml.padding_token_id"), Some(0));
    assert_eq!(id(&content, "tokenizer.ggml.unknown_token_id"), Some(3));
    assert_eq!(content.get_bool("tokenizer.ggml.add_bos_token"), Some(true));
    assert_eq!(
        content.get_bool("tokenizer.ggml.add_eos_token"),
        Some(false)
    );

    let tokens = array_of_strings(&content, "tokenizer.ggml.tokens");
    let types = array_of_i32(&content, "tokenizer.ggml.token_type");
    // The vocabulary is as wide as the embedding (10), not as the tokenizer (11):
    // an id past the embedding names nothing the model has.
    assert_eq!(tokens.len(), 10);
    assert_eq!(types[4], BYTE);
    assert_eq!(types[1], CONTROL);
    // An added token that is not special is text. The space marker in it is a
    // space again, as the reference writes it.
    assert_eq!(tokens[8], "   ");
    assert_eq!(types[8], USER_DEFINED);
    assert_eq!(tokens[9], "[multimodal]");
    assert_eq!(types[9], USER_DEFINED);
    assert_eq!(tokens[7], "<start_of_turn>");
    assert_eq!(types[7], CONTROL);
    // An ordinary piece keeps the marker.
    assert_eq!(tokens[5], "\u{2581}the");
}

#[test]
fn gemma3_architecture_keys() {
    let (_, content) = gemma3();
    assert_eq!(content.get_u64("gemma3.attention.key_length"), Some(256));
    assert_eq!(
        content.get_u64("gemma3.attention.sliding_window"),
        Some(512)
    );
    assert_eq!(content.get_f64("gemma3.rope.freq_base"), Some(1_000_000.0));
    assert_eq!(content.get_f64("gemma3.rope.freq_base_swa"), Some(10_000.0));
}

// ── Gemma 4 ────────────────────────────────────────────────────────────────

/// The language model of a Gemma 4 12B-shaped checkpoint (48 layers, five
/// sliding-window layers to one full-attention layer, full-attention heads of
/// 512 with a single key/value head that is also the value) in the multimodal
/// wrapper it ships in.
fn gemma4() -> (ExportArch, GgufContent) {
    let layer_types: Vec<&str> = (0..48)
        .map(|l| {
            if l % 6 == 5 {
                "full_attention"
            } else {
                "sliding_attention"
            }
        })
        .collect();
    let config = json!({
        "architectures": ["Gemma4UnifiedForConditionalGeneration"],
        "model_type": "gemma4_unified", "tie_word_embeddings": true,
        "text_config": {
            "model_type": "gemma4_unified_text", "hidden_size": 3840,
            "intermediate_size": 15360, "num_hidden_layers": 48,
            "num_attention_heads": 16, "num_key_value_heads": 8,
            "num_global_key_value_heads": 1, "attention_k_eq_v": true,
            "head_dim": 256, "global_head_dim": 512, "vocab_size": 262144,
            "max_position_embeddings": 131072, "rms_norm_eps": 1e-6,
            "sliding_window": 1024, "final_logit_softcapping": 30.0,
            "hidden_size_per_layer_input": 0, "layer_types": layer_types,
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                   "rope_type": "proportional"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            },
        },
    });
    export(config, &[])
}

/// What the reference loader reads of a Gemma 4 file, per kind of layer:
/// the two head widths, the key/value heads of every layer, which layers
/// slide, both rope bases, and the (zero) per-layer input width it requires.
#[test]
fn gemma4_architecture_keys() {
    let (arch, content) = gemma4();
    assert_eq!(arch.name, "gemma4");
    assert_eq!(content.architecture(), Some("gemma4"));
    for (key, value) in [
        ("gemma4.embedding_length", 3840),
        ("gemma4.block_count", 48),
        ("gemma4.feed_forward_length", 15360),
        ("gemma4.attention.head_count", 16),
        ("gemma4.attention.key_length", 512),
        ("gemma4.attention.value_length", 512),
        ("gemma4.attention.key_length_swa", 256),
        ("gemma4.attention.value_length_swa", 256),
        ("gemma4.attention.sliding_window", 1024),
        ("gemma4.rope.dimension_count", 512),
        ("gemma4.rope.dimension_count_swa", 256),
        ("gemma4.embedding_length_per_layer_input", 0),
    ] {
        assert_eq!(content.get_u64(key), Some(value), "{key}");
    }
    assert_eq!(content.get_f64("gemma4.rope.freq_base"), Some(1_000_000.0));
    assert_eq!(content.get_f64("gemma4.rope.freq_base_swa"), Some(10_000.0));
    assert_eq!(
        content.get_f64("gemma4.final_logit_softcapping"),
        Some(30.0)
    );
    let heads = match content.get_metadata("gemma4.attention.head_count_kv") {
        Some(MetadataValue::Array(values)) => values
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect::<Vec<_>>(),
        other => panic!("head_count_kv is {other:?}"),
    };
    assert_eq!(heads.len(), 48);
    assert_eq!(&heads[..7], &[8, 8, 8, 8, 8, 1, 8]);
    assert_eq!(heads[47], 1);
    let pattern = match content.get_metadata("gemma4.attention.sliding_window_pattern") {
        Some(MetadataValue::Array(values)) => values
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect::<Vec<_>>(),
        other => panic!("sliding_window_pattern is {other:?}"),
    };
    assert_eq!(pattern.iter().filter(|s| **s).count(), 40);
    assert!(!pattern[5] && pattern[6]);
    // The model's norm multiplies by its weight: nothing is added to a norm.
    assert!(!arch.extra_tensors().is_empty());
}

/// Gemma 4 tokenizes raw characters (no space prefix) and its engine adds the
/// beginning of text, which neither its tokenizer config nor its post-processor
/// says.
#[test]
fn gemma4_tokenizer_adds_a_beginning_of_text_and_no_space_prefix() {
    let tokenizer = tokenizer_json(
        &["<pad>", "<eos>", "<bos>", "a", "b", "ab"],
        &["a b"],
        &[("<start_of_turn>", true)],
        json!({}),
        true,
    );
    let tokenizer_config =
        json!({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"});
    let config = json!({
        "architectures": ["Gemma4UnifiedForConditionalGeneration"], "model_type": "gemma4_unified",
        "text_config": {"hidden_size": 8, "vocab_size": 7, "num_attention_heads": 2},
    });
    let (_, content) = export(
        config,
        &[
            ("tokenizer.json", tokenizer.into_bytes()),
            (
                "tokenizer_config.json",
                tokenizer_config.to_string().into_bytes(),
            ),
        ],
    );
    assert_eq!(content.get_str("tokenizer.ggml.model"), Some("gemma4"));
    assert_eq!(content.get_bool("tokenizer.ggml.add_bos_token"), Some(true));
    assert_eq!(
        content.get_bool("tokenizer.ggml.add_space_prefix"),
        Some(false)
    );
    assert_eq!(content.get_u64("tokenizer.ggml.bos_token_id"), Some(2));
}

/// A token is a control token of the file when the tokenizer flags it special
/// (the 24 added tokens of Gemma 4's `tokenizer.json`: its pad, eos, bos, unk
/// and mask, the turn, channel, tool and thinking markers, the media markers)
/// and a normal one otherwise.
///
/// `</s>` (id 212 in the real vocabulary) is the second kind: an HTML tag among
/// `</sub>` and `</sup>`, in the model's own vocabulary and no added token. An
/// engine that treats the name `</s>` as an end-of-sequence marker warns that
/// it is "control-looking" and not control-type, then drops it from its stop
/// tokens as the normal token it is. Marking it control would stop that
/// warning by hiding the text `</s>` from a generation, a change from the
/// tokenizer that nothing asks for.
#[test]
fn gemma4_control_tokens_are_the_ones_the_tokenizer_flags_special() {
    let vocab = ["a", "b", "</s>", "</sub>"];
    let tokenizer = tokenizer_json(
        &vocab,
        &[],
        &[
            ("<pad>", true),
            ("<eos>", true),
            ("<bos>", true),
            ("<unk>", true),
            ("<mask>", true),
            ("<|turn>", true),
            ("<turn|>", true),
            ("<|channel>", true),
            ("<|image|>", true),
            // Not special, and not spelled like a control token.
            ("<br>", false),
        ],
        json!({"pre_tokenizer": byte_level(), "decoder": byte_level()}),
        false,
    );
    let tokenizer_config = json!({
        "bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>",
    });
    let config = json!({
        "architectures": ["Gemma4UnifiedForConditionalGeneration"], "model_type": "gemma4_unified",
        "text_config": {"hidden_size": 8, "vocab_size": 13, "num_attention_heads": 2},
    });
    let (_, content) = export(
        config,
        &[
            ("tokenizer.json", tokenizer.into_bytes()),
            (
                "tokenizer_config.json",
                tokenizer_config.to_string().into_bytes(),
            ),
        ],
    );
    let tokens = array_of_strings(&content, "tokenizer.ggml.tokens");
    let types = array_of_i32(&content, "tokenizer.ggml.token_type");
    let type_of = |token: &str| {
        let at = tokens.iter().position(|t| t == token).unwrap();
        types[at]
    };
    // The added tokens flagged special are control tokens.
    for token in [
        "<pad>",
        "<eos>",
        "<bos>",
        "<unk>",
        "<mask>",
        "<|turn>",
        "<turn|>",
        "<|channel>",
        "<|image|>",
    ] {
        assert_eq!(type_of(token), CONTROL, "{token}");
    }
    // `</s>` and its neighbours are text; so is an added token that is neither
    // flagged nor spelled like a control token.
    for token in ["</s>", "</sub>", "a"] {
        assert_eq!(type_of(token), NORMAL, "{token}");
    }
    assert_eq!(type_of("<br>"), USER_DEFINED);
}

// ── a Mistral-family model with no beginning-of-text in its config ─────────

#[test]
fn a_vocabulary_with_s_tokens_is_not_given_a_beginning_of_text() {
    // The failure this guards: a byte-level vocabulary has `<s>`, `</s>` and
    // `<unk>` as ordinary tokens, and an export that looks them up by name
    // when no config names one writes them as the model's special ids.
    let tokenizer = tokenizer_json(
        &["<s>", "</s>", "<unk>", "a"],
        &[],
        &[("<|endoftext|>", true)],
        json!({"pre_tokenizer": byte_level(), "decoder": byte_level()}),
        false,
    );
    let config =
        json!({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "vocab_size": 5});
    let (_, content) = export(config, &[("tokenizer.json", tokenizer.into_bytes())]);
    for key in [
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.padding_token_id",
    ] {
        assert!(content.get_metadata(key).is_none(), "{key}");
    }
}
