//! Datasets and the other tiny checkpoints the trainer commands read, beside
//! the tiny model ([`crate::tiny_model`]). The outputs are gibberish, which
//! does not matter: these tests are about how a command ends.

use std::path::Path;

use serde_json::json;

use crate::tiny_model::{
    HEAD_DIM, HEADS, HIDDEN, INTERMEDIATE, KV_HEADS, LAYERS, Rng, Tensors, ones, random_tensor,
    vocab, write_safetensors,
};

/// A tiny DFlash draft for the tiny model, in `dir`.
pub fn write_tiny_draft(dir: &Path, seed: u64) {
    std::fs::create_dir_all(dir).unwrap();
    let taps = [0usize, LAYERS - 1];
    let config = json!({
        "architectures": ["DFlashDraftModel"],
        "model_type": "dflash_qwen3",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 2,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-6,
        "vocab_size": vocab().len(),
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
        "block_size": 4,
        "dflash_config": {"target_layer_ids": taps, "mask_token_id": 0},
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let mut rng = Rng(seed);
    let mut tensors = Tensors::new();
    tensors.insert(
        "fc.weight".into(),
        random_tensor(&mut rng, &[HIDDEN, taps.len() * HIDDEN]),
    );
    tensors.insert("hidden_norm.weight".into(), ones(&[HIDDEN]));
    tensors.insert("norm.weight".into(), ones(&[HIDDEN]));
    for i in 0..2 {
        let p = format!("layers.{i}");
        for (name, shape) in [
            ("self_attn.q_proj", [HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.k_proj", [KV_HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.v_proj", [KV_HEADS * HEAD_DIM, HIDDEN]),
            ("self_attn.o_proj", [HIDDEN, HEADS * HEAD_DIM]),
            ("mlp.gate_proj", [INTERMEDIATE, HIDDEN]),
            ("mlp.up_proj", [INTERMEDIATE, HIDDEN]),
            ("mlp.down_proj", [HIDDEN, INTERMEDIATE]),
        ] {
            tensors.insert(
                format!("{p}.{name}.weight"),
                random_tensor(&mut rng, &shape),
            );
        }
        for (name, size) in [
            ("self_attn.q_norm", HEAD_DIM),
            ("self_attn.k_norm", HEAD_DIM),
            ("input_layernorm", HIDDEN),
            ("post_attention_layernorm", HIDDEN),
        ] {
            tensors.insert(format!("{p}.{name}.weight"), ones(&[size]));
        }
    }
    write_safetensors(&dir.join("model.safetensors"), &tensors);
}

const WORDS: [&str; 10] = [
    "the", "cat", "sat", "on", "a", "mat", "and", "the", "dog", "ran",
];

/// A sentence of `words` words, chosen by `i`.
fn sentence(i: usize, words: usize) -> String {
    let mut rng = Rng(i as u64 * 7919 + 1);
    let picked: Vec<&str> = (0..words)
        .map(|_| WORDS[(rng.next_u64() % WORDS.len() as u64) as usize])
        .collect();
    picked.join(" ")
}

fn write_jsonl(path: &Path, rows: impl IntoIterator<Item = serde_json::Value>) {
    let text: String = rows.into_iter().map(|row| format!("{row}\n")).collect();
    std::fs::write(path, text).unwrap();
}

/// Plain text rows, for `train`, `distill` and `tokenize`.
pub fn write_text_dataset(path: &Path) {
    write_jsonl(
        path,
        (0..16).map(|i| json!({"text": format!("{}.", sentence(i, 12))})),
    );
}

/// Prompts with an answer, for `grpo` and `rlkd`.
pub fn write_prompt_dataset(path: &Path) {
    write_jsonl(
        path,
        (0..8).map(|i| json!({"prompt": sentence(i, 4), "answer": sentence(i + 100, 2)})),
    );
}

/// Chosen and rejected completions, for `preference`.
pub fn write_preference_dataset(path: &Path) {
    write_jsonl(
        path,
        (0..8).map(|i| {
            json!({"prompt": sentence(i, 4),
                   "chosen": sentence(i + 100, 3),
                   "rejected": sentence(i + 200, 3)})
        }),
    );
}

/// Pairs of texts the tiny encoder's tokenizer knows, for `embed-train`.
pub fn write_pair_dataset(path: &Path) {
    const WORDS: [&str; 8] = ["one", "two", "three", "four", "red", "blue", "green", "cat"];
    write_jsonl(
        path,
        (0..8).map(|i| {
            let text = format!("{} {}", WORDS[i], WORDS[(i + 3) % WORDS.len()]);
            json!({"text_a": text, "text_b": text})
        }),
    );
}

const BERT_CONFIG: &str = r#"{
    "model_type": "bert", "vocab_size": 16, "hidden_size": 32,
    "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 64,
    "max_position_embeddings": 32, "type_vocab_size": 2, "layer_norm_eps": 1e-12,
    "hidden_act": "gelu", "pad_token_id": 0
}"#;

const BERT_TOKENIZER: &str = r###"{
    "version": "1.0",
    "added_tokens": [
        {"id": 0, "content": "[PAD]", "special": true, "single_word": false,
         "lstrip": false, "rstrip": false, "normalized": false},
        {"id": 1, "content": "[UNK]", "special": true, "single_word": false,
         "lstrip": false, "rstrip": false, "normalized": false},
        {"id": 2, "content": "[CLS]", "special": true, "single_word": false,
         "lstrip": false, "rstrip": false, "normalized": false},
        {"id": 3, "content": "[SEP]", "special": true, "single_word": false,
         "lstrip": false, "rstrip": false, "normalized": false}
    ],
    "normalizer": {"type": "Lowercase"},
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
        ],
        "pair": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 1}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 1}}
        ],
        "special_tokens": {
            "[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
            "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]}
        }
    },
    "model": {
        "type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
        "max_input_chars_per_word": 100,
        "vocab": {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3,
                  "one": 4, "two": 5, "three": 6, "four": 7,
                  "red": 8, "blue": 9, "green": 10, "cat": 11, "dog": 12, "bird": 13}
    }
}"###;

/// A tiny, randomly initialised BERT encoder in `dir`, written by the library
/// itself so its tensors carry the names the loader reads.
pub fn write_tiny_encoder(dir: &Path) {
    use pmetal::models::architectures::encoder::EncoderModel;
    use pmetal::runtime::compat::{Array, random};

    std::fs::create_dir_all(dir).unwrap();
    random::seed(5);
    let model = EncoderModel::from_config_str(BERT_CONFIG).unwrap();
    let tensors = pmetal::models::loader::encoder_checkpoint_tensors(&model).unwrap();
    let entries: Vec<(&str, &Array)> = tensors.iter().map(|(n, a)| (n.as_str(), a)).collect();
    Array::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &entries).unwrap();
    std::fs::write(dir.join("config.json"), BERT_CONFIG).unwrap();
    std::fs::write(dir.join("tokenizer.json"), BERT_TOKENIZER).unwrap();
}
