//! Gemma GGUFs load for inference.
//!
//! A tiny Gemma checkpoint is written twice: as the Hugging Face tensors the
//! safetensors path reads (`w`, applied by the model as `1 + w`), and as the
//! GGUF the converter writes (norm weights `w + 1`, the GGUF tensor names, the
//! window, rope bases and soft caps as metadata). Both are loaded through
//! `DynamicModel` and must compute the same logits. The fixtures are built
//! here, not by the exporter, so the loader is held to the format and not to
//! the writer.
//!
//! The sequence is longer than the window and the model has more layers than
//! one period of the local/global pattern, so a wrong window, pattern or rope
//! base reaches the logits.

use pmetal_gguf::GgufBuilder;
use pmetal_models::DynamicModel;
use pmetal_models::loader::load_gemma_weights;
use pmetal_models::weight_format::WeightLoader;
use pmetal_runtime::compat::{Array, Dtype};
use std::collections::HashMap;

const HIDDEN: usize = 32;
const FFN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const VOCAB: usize = 16;
const WINDOW: u32 = 4;
const SEQUENCE: usize = 12;

#[derive(Clone, Copy, PartialEq)]
enum Generation {
    Gemma1,
    Gemma2,
    Gemma3,
}

impl Generation {
    fn architecture(self) -> &'static str {
        match self {
            Self::Gemma1 => "gemma",
            Self::Gemma2 => "gemma2",
            Self::Gemma3 => "gemma3",
        }
    }

    fn layers(self) -> usize {
        match self {
            // Gemma 3 is global one layer in six.
            Self::Gemma3 => 6,
            _ => 4,
        }
    }
}

/// Deterministic, roughly normal values in `[-scale, scale]`.
struct Rng(u64);

impl Rng {
    fn values(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let mut t = 0.0f32;
                for _ in 0..4 {
                    self.0 ^= self.0 << 13;
                    self.0 ^= self.0 >> 7;
                    self.0 ^= self.0 << 17;
                    t += (self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                }
                t * scale
            })
            .collect()
    }
}

/// The checkpoint's tensors in Hugging Face names: `(name, shape, values)`.
fn checkpoint(generation: Generation) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut tensors = Vec::new();
    let mut add = |name: String, shape: &[usize], scale: f32| {
        let n: usize = shape.iter().product();
        tensors.push((name, shape.to_vec(), rng.values(n, scale)));
    };
    add("model.embed_tokens.weight".into(), &[VOCAB, HIDDEN], 1.0);
    // Zero-centred, as a Gemma norm is trained: `1 + w` is the scale.
    add("model.norm.weight".into(), &[HIDDEN], 0.5);
    for layer in 0..generation.layers() {
        let p = format!("model.layers.{layer}");
        add(format!("{p}.input_layernorm.weight"), &[HIDDEN], 0.5);
        add(format!("{p}.post_attention_layernorm.weight"), &[HIDDEN], 0.5);
        if generation != Generation::Gemma1 {
            add(format!("{p}.pre_feedforward_layernorm.weight"), &[HIDDEN], 0.5);
            add(format!("{p}.post_feedforward_layernorm.weight"), &[HIDDEN], 0.5);
        }
        if generation == Generation::Gemma3 {
            add(format!("{p}.self_attn.q_norm.weight"), &[HEAD_DIM], 0.5);
            add(format!("{p}.self_attn.k_norm.weight"), &[HEAD_DIM], 0.5);
        }
        add(format!("{p}.self_attn.q_proj.weight"), &[HEADS * HEAD_DIM, HIDDEN], 0.4);
        add(format!("{p}.self_attn.k_proj.weight"), &[KV_HEADS * HEAD_DIM, HIDDEN], 0.4);
        add(format!("{p}.self_attn.v_proj.weight"), &[KV_HEADS * HEAD_DIM, HIDDEN], 0.4);
        add(format!("{p}.self_attn.o_proj.weight"), &[HIDDEN, HEADS * HEAD_DIM], 0.4);
        add(format!("{p}.mlp.gate_proj.weight"), &[FFN, HIDDEN], 0.4);
        add(format!("{p}.mlp.up_proj.weight"), &[FFN, HIDDEN], 0.4);
        add(format!("{p}.mlp.down_proj.weight"), &[HIDDEN, FFN], 0.4);
    }
    tensors
}

/// The `config.json` of the checkpoint, as Hugging Face publishes it.
fn config_json(generation: Generation) -> String {
    let mut config = serde_json::json!({
        "model_type": generation.architecture(),
        "vocab_size": VOCAB, "hidden_size": HIDDEN, "intermediate_size": FFN,
        "num_hidden_layers": generation.layers(),
        "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM, "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6, "hidden_act": "gelu_pytorch_tanh",
    });
    match generation {
        Generation::Gemma1 => config["rope_theta"] = 10000.0.into(),
        Generation::Gemma2 => {
            config["rope_theta"] = 10000.0.into();
            config["sliding_window"] = WINDOW.into();
            config["attn_logit_softcapping"] = 50.0.into();
            config["final_logit_softcapping"] = 30.0.into();
            config["query_pre_attn_scalar"] = HEAD_DIM.into();
        }
        Generation::Gemma3 => {
            config["rope_theta"] = 1_000_000.0.into();
            config["rope_local_base_freq"] = 10000.0.into();
            config["sliding_window"] = WINDOW.into();
            config["query_pre_attn_scalar"] = HEAD_DIM.into();
            config["rope_scaling"] = serde_json::json!({"rope_type": "linear", "factor": 8.0});
        }
    }
    config.to_string()
}

/// The GGUF the converter writes for the checkpoint.
fn gguf(generation: Generation, mutate: impl FnOnce(&mut GgufBuilder)) -> Vec<u8> {
    let arch = generation.architecture();
    let mut b = GgufBuilder::with_model(arch, "tiny-gemma");
    let k = |suffix: &str| format!("{arch}.{suffix}");
    b.add_u32(k("embedding_length"), HIDDEN as u32);
    b.add_u32(k("block_count"), generation.layers() as u32);
    b.add_u32(k("attention.head_count"), HEADS as u32);
    b.add_u32(k("attention.head_count_kv"), KV_HEADS as u32);
    b.add_u32(k("feed_forward_length"), FFN as u32);
    b.add_u32(k("context_length"), 128);
    b.add_u32(k("attention.key_length"), HEAD_DIM as u32);
    b.add_u32(k("attention.value_length"), HEAD_DIM as u32);
    b.add_f32(k("attention.layer_norm_rms_epsilon"), 1e-6);
    match generation {
        Generation::Gemma1 => {
            b.add_f32(k("rope.freq_base"), 10000.0);
        }
        Generation::Gemma2 => {
            b.add_f32(k("rope.freq_base"), 10000.0);
            b.add_u32(k("attention.sliding_window"), WINDOW);
            b.add_f32(k("attn_logit_softcapping"), 50.0);
            b.add_f32(k("final_logit_softcapping"), 30.0);
        }
        Generation::Gemma3 => {
            b.add_f32(k("rope.freq_base"), 1_000_000.0);
            b.add_f32(k("rope.freq_base_swa"), 10000.0);
            b.add_u32(k("attention.sliding_window"), WINDOW);
            b.add_string(k("rope.scaling.type"), "linear");
            b.add_f32(k("rope.scaling.factor"), 8.0);
        }
    }
    let tokens: Vec<String> = (0..VOCAB).map(|i| format!("<t{i}>")).collect();
    b.add_string_array("tokenizer.ggml.tokens", tokens);
    for (name, shape, mut values) in checkpoint(generation) {
        let gguf_name = if generation == Generation::Gemma1 {
            WeightLoader::hf_to_gguf_name(&name)
        } else {
            WeightLoader::hf_to_gguf_name_gemma(&name)
        };
        // The converter adds the 1 the model adds back when it runs.
        if name.ends_with("norm.weight") {
            values.iter_mut().for_each(|v| *v += 1.0);
        }
        b.add_f32_tensor(gguf_name, shape.iter().map(|&d| d as u64).collect(), values);
    }
    mutate(&mut b);
    b.build_to_bytes().expect("build the gguf")
}

/// The model the safetensors path builds: the config, then the checkpoint.
fn reference(generation: Generation) -> DynamicModel {
    let mut model = DynamicModel::from_config(&config_json(generation)).expect("config");
    let DynamicModel::Gemma(gemma) = &mut model else {
        panic!("a Gemma config builds a Gemma");
    };
    let weights: HashMap<String, Array> = checkpoint(generation)
        .into_iter()
        .map(|(name, shape, values)| {
            let shape: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
            (name, Array::from_slice(&values, &shape))
        })
        .collect();
    load_gemma_weights(gemma, &weights).expect("load the checkpoint");
    model
}

fn load_gguf(bytes: Vec<u8>) -> Result<DynamicModel, String> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, bytes).expect("write gguf");
    DynamicModel::load(&path).map_err(|e| e.to_string())
}

fn logits(model: &mut DynamicModel) -> Vec<f32> {
    let ids: Vec<i32> = (0..SEQUENCE as i32).map(|i| (i * 5 + 3) % VOCAB as i32).collect();
    let ids = Array::from_slice(&ids, &[1, SEQUENCE as i32]);
    let out = model.forward(&ids, None).expect("forward");
    let mut out = out.as_dtype(Dtype::Float32.as_i32());
    let n = out.size();
    let values = out.to_f32_vec(n).expect("read logits");
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
    values
}

fn assert_same_logits(generation: Generation) {
    let mut want = reference(generation);
    let mut got = load_gguf(gguf(generation, |_| {})).expect("the gguf loads");
    assert_eq!(got.architecture(), want.architecture());
    let (want, got) = (logits(&mut want), logits(&mut got));
    assert!(want.iter().all(|v| v.is_finite()));
    let spread = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(spread > 1e-2, "the logits are degenerate: {spread}");
    let worst = want
        .iter()
        .zip(&got)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst <= 2e-5 * spread.max(1.0),
        "the GGUF's logits differ from the checkpoint's by {worst:e} (magnitude {spread})"
    );
}

#[test]
fn a_gemma_3_gguf_computes_what_its_checkpoint_does() {
    assert_same_logits(Generation::Gemma3);
}

#[test]
fn a_gemma_2_gguf_computes_what_its_checkpoint_does() {
    assert_same_logits(Generation::Gemma2);
}

#[test]
fn a_gemma_1_gguf_computes_what_its_checkpoint_does() {
    assert_same_logits(Generation::Gemma1);
}

/// The metadata is the model: the window, the two rope bases and the soft caps
/// come from the file, with nothing assumed in their place.
#[test]
fn the_gguf_metadata_reaches_the_model() {
    let DynamicModel::Gemma(model) = load_gguf(gguf(Generation::Gemma3, |_| {})).unwrap() else {
        panic!("gemma");
    };
    let config = model.config();
    assert!(config.is_gemma3 && !config.is_gemma2);
    assert_eq!(config.sliding_window, Some(WINDOW as i32));
    assert_eq!(config.rope_theta, 1_000_000.0);
    assert_eq!(config.rope_local_base_freq, Some(10000.0));
    assert_eq!(config.query_pre_attn_scalar, Some(HEAD_DIM as i32));
    assert_eq!(config.num_hidden_layers, 6);

    let DynamicModel::Gemma(model) = load_gguf(gguf(Generation::Gemma2, |_| {})).unwrap() else {
        panic!("gemma");
    };
    let config = model.config();
    assert!(config.is_gemma2 && !config.is_gemma3);
    assert_eq!(config.attn_logit_softcapping, Some(50.0));
    assert_eq!(config.final_logit_softcapping, Some(30.0));
}

/// A head the file carries of its own is not Gemma's tied one; an embedding
/// standing in for it would answer wrongly without a word.
#[test]
fn a_gemma_file_with_a_head_of_its_own_is_refused() {
    let bytes = gguf(Generation::Gemma3, |b| {
        b.add_f32_tensor(
            "output.weight",
            vec![VOCAB as u64, HIDDEN as u64],
            vec![0.1; VOCAB * HIDDEN],
        );
    });
    let error = load_gguf(bytes).expect_err("refused");
    assert!(error.contains("output.weight"), "{error}");
}

/// A tensor no Gemma weight reads is named, not dropped.
#[test]
fn a_tensor_the_model_has_no_weight_for_is_an_error() {
    let bytes = gguf(Generation::Gemma3, |b| {
        b.add_f32_tensor("blk.0.mystery.weight", vec![4], vec![0.0; 4]);
    });
    let error = load_gguf(bytes).expect_err("refused");
    assert!(error.contains("mystery"), "{error}");
}

/// Each of these is a number the file states and a model could assume instead.
/// Changed in the file alone, each has to move the logits: if one did not, it
/// would not be read.
#[test]
fn each_metadata_value_the_file_states_moves_the_logits() {
    let cases: [(Generation, &str, fn(&mut GgufBuilder)); 5] = [
        (Generation::Gemma3, "window", |b| {
            b.add_u32("gemma3.attention.sliding_window", 2);
        }),
        (Generation::Gemma3, "local rope base", |b| {
            b.add_f32("gemma3.rope.freq_base_swa", 50.0);
        }),
        (Generation::Gemma3, "global rope scaling", |b| {
            b.add_f32("gemma3.rope.scaling.factor", 1.0);
        }),
        (Generation::Gemma2, "attention soft cap", |b| {
            b.add_f32("gemma2.attn_logit_softcapping", 0.5);
        }),
        (Generation::Gemma2, "final soft cap", |b| {
            b.add_f32("gemma2.final_logit_softcapping", 0.5);
        }),
    ];
    for (generation, what, mutate) in cases {
        let mut want = reference(generation);
        let mut got = load_gguf(gguf(generation, mutate)).expect("the gguf loads");
        let (want, got) = (logits(&mut want), logits(&mut got));
        let moved = want
            .iter()
            .zip(&got)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(moved > 1e-3, "the {what} did not reach the logits ({moved:e})");
    }
}
