//! A complete, tiny, randomly initialised Qwen3 checkpoint (config, weights,
//! tokenizer) for running the real `pmetal` binary, and the pieces it is made
//! of, for the fixtures built alongside it. Its outputs are gibberish, which
//! does not matter to the tests that use it.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::json;

/// One character per token, then the two special tokens.
const ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz .,\n";
const UNK: &str = "<unk>";
const EOS: &str = "<eos>";

pub const HIDDEN: usize = 32;
pub const LAYERS: usize = 2;
pub const HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 8;
pub const INTERMEDIATE: usize = 64;

/// The vocabulary: `<unk>`, `<eos>`, then the characters of [`ALPHABET`].
pub fn vocab() -> BTreeMap<String, u32> {
    let mut vocab = BTreeMap::new();
    vocab.insert(UNK.to_string(), 0);
    vocab.insert(EOS.to_string(), 1);
    for (i, c) in ALPHABET.chars().enumerate() {
        vocab.insert(c.to_string(), i as u32 + 2);
    }
    vocab
}

/// SplitMix64: the same weights for the same seed, on every machine.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

pub type Tensors = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

pub fn random_tensor(rng: &mut Rng, shape: &[usize]) -> (Vec<usize>, Vec<f32>) {
    let count: usize = shape.iter().product();
    (
        shape.to_vec(),
        (0..count).map(|_| rng.unit() * 0.08).collect(),
    )
}

pub fn ones(shape: &[usize]) -> (Vec<usize>, Vec<f32>) {
    (shape.to_vec(), vec![1.0; shape.iter().product()])
}

/// A safetensors file of `f32` tensors.
pub fn write_safetensors(path: &Path, tensors: &Tensors) {
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
