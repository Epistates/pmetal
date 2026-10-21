//! Tiny checkpoints and datasets for running the real `pmetal` binary: a
//! complete, random Qwen3 (config, weights, tokenizer) the commands can load,
//! and the small files each trainer command reads. The outputs are gibberish,
//! which does not matter: these tests are about how a command ends.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::json;

/// One character per token, then the two special tokens.
const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz .,\n";
const UNK: &str = "<unk>";
const EOS: &str = "<eos>";

const HIDDEN: usize = 32;
const LAYERS: usize = 2;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const INTERMEDIATE: usize = 64;

/// The vocabulary: `<unk>`, `<eos>`, then the characters of [`ALPHABET`].
fn vocab() -> BTreeMap<String, u32> {
    let mut vocab = BTreeMap::new();
    vocab.insert(UNK.to_string(), 0);
    vocab.insert(EOS.to_string(), 1);
    for (i, c) in ALPHABET.chars().enumerate() {
        vocab.insert(c.to_string(), i as u32 + 2);
    }
    vocab
}

/// SplitMix64: the same weights for the same seed, on every machine.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

type Tensors = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

fn random_tensor(rng: &mut Rng, shape: &[usize]) -> (Vec<usize>, Vec<f32>) {
    let count: usize = shape.iter().product();
    (
        shape.to_vec(),
        (0..count).map(|_| rng.unit() * 0.08).collect(),
    )
}

fn ones(shape: &[usize]) -> (Vec<usize>, Vec<f32>) {
    (shape.to_vec(), vec![1.0; shape.iter().product()])
}

/// A safetensors file of `f32` tensors.
fn write_safetensors(path: &Path, tensors: &Tensors) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, (shape, values)) in tensors {
        let bytes = values.len() * 4;
        header.insert(
            name.clone(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + bytes]}),
        );
        offset += bytes;
    }
    let mut blob = serde_json::to_vec(&header).unwrap();
    while blob.len() % 8 != 0 {
        blob.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + blob.len() + offset);
    out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
    out.extend_from_slice(&blob);
    for (_, values) in tensors.values() {
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, out).unwrap();
}

/// A tiny tokenizer: every character is a token.
fn write_tokenizer(dir: &Path) {
    let token = |id: u32, content: &str| {
        json!({"id": id, "content": content, "single_word": false, "lstrip": false,
               "rstrip": false, "normalized": false, "special": true})
    };
    let tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [token(0, UNK), token(1, EOS)],
        "normalizer": null,
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "."},
                          "behavior": "Isolated", "invert": false},
        "post_processor": null,
        "decoder": {"type": "Fuse"},
        "model": {"type": "WordLevel", "vocab": vocab(), "unk_token": UNK},
    });
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        json!({"eos_token": EOS, "unk_token": UNK, "bos_token": null, "pad_token": EOS})
            .to_string(),
    )
    .unwrap();
}

/// A complete, tiny, randomly initialised Qwen3 checkpoint in `dir`.
pub fn write_tiny_model(dir: &Path, seed: u64) {
    std::fs::create_dir_all(dir).unwrap();
    let vocab_size = vocab().len();
    let config = json!({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": vocab_size,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 256,
        "tie_word_embeddings": true,
        "hidden_act": "silu",
        "attention_bias": false,
        "eos_token_id": 1,
        "bos_token_id": 1,
        "torch_dtype": "float32",
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        json!({"do_sample": true, "temperature": 0.8, "top_k": 8, "top_p": 0.95,
               "eos_token_id": 1})
        .to_string(),
    )
    .unwrap();

    let mut rng = Rng(seed);
    let mut tensors = Tensors::new();
    tensors.insert(
        "model.embed_tokens.weight".into(),
        random_tensor(&mut rng, &[vocab_size, HIDDEN]),
    );
    tensors.insert("model.norm.weight".into(), ones(&[HIDDEN]));
    for i in 0..LAYERS {
        let p = format!("model.layers.{i}");
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
    write_tokenizer(dir);
}

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
