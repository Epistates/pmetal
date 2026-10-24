//! The dense Gemma 4 text model loads from a GGUF.
//!
//! The checkpoint is a tiny Gemma 4 with the features that make it what it is:
//! sliding-window and full-attention layers with different head widths, the
//! full-attention layers taking their values from their keys, and a quarter of
//! their heads rotated. It is written as the Hugging Face tensors the
//! safetensors path reads and as the GGUF a converter writes (per-layer arrays
//! in the metadata, no `attn_v` on the full-attention layers, the unrotated
//! pairs of `rope_freqs` at `1e30`, norms stored as they are), and both must
//! compute the same logits. The fixture is built here, not by the exporter.

use pmetal_gguf::{GgufBuilder, MetadataValue};
use pmetal_models::DynamicModel;
use pmetal_models::architectures::gemma4::load_gemma4_weights;
use pmetal_models::weight_format::WeightLoader;
use pmetal_runtime::compat::{Array, Dtype, ModuleParametersExt};
use std::collections::HashMap;

const HIDDEN: usize = 32;
const FFN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const GLOBAL_KV_HEADS: usize = 1;
const HEAD_DIM: usize = 8;
const GLOBAL_HEAD_DIM: usize = 16;
const VOCAB: usize = 16;
const WINDOW: u32 = 4;
const SEQUENCE: usize = 12;
const SLIDING: [bool; 6] = [true, true, false, true, true, false];
/// A quarter of a full-attention head rotates: 2 of its 8 pairs.
const ROTATED_PAIRS: usize = 2;

fn config_json() -> String {
    serde_json::json!({
        "model_type": "gemma4_text", "vocab_size": VOCAB, "hidden_size": HIDDEN,
        "intermediate_size": FFN, "num_hidden_layers": SLIDING.len(),
        "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM, "global_head_dim": GLOBAL_HEAD_DIM,
        "num_global_key_value_heads": GLOBAL_KV_HEADS, "attention_k_eq_v": true,
        "max_position_embeddings": 128, "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true, "sliding_window": WINDOW,
        "final_logit_softcapping": 30.0,
        "layer_types": SLIDING.iter()
            .map(|s| if *s { "sliding_attention" } else { "full_attention" })
            .collect::<Vec<_>>(),
        "rope_parameters": {
            "full_attention": {"rope_type": "proportional", "rope_theta": 1_000_000.0,
                               "partial_rotary_factor": 0.25},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
        "hidden_activation": "gelu_pytorch_tanh",
    })
    .to_string()
}

/// Deterministic, roughly normal values in `[-scale, scale]`.
fn values(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let mut t = 0.0f32;
            for _ in 0..4 {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                t += (*seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            }
            t * scale
        })
        .collect()
}

/// A checkpoint's tensors: `(name, shape, values)`.
type Checkpoint = Vec<(String, Vec<usize>, Vec<f32>)>;

/// The reference model, loaded from checkpoint tensors the model itself names:
/// `(name, shape, values)` for every parameter, so none is left out.
fn reference() -> (DynamicModel, Checkpoint) {
    let mut model = DynamicModel::from_config(&config_json()).expect("config");
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut tensors = Vec::new();
    let mut keys: Vec<(String, Vec<usize>)> = model
        .flatten_params()
        .into_iter()
        .map(|(key, array)| {
            (
                key.to_string(),
                array.shape().iter().map(|&d| d as usize).collect(),
            )
        })
        .collect();
    keys.sort();
    for (key, shape) in keys {
        let n: usize = shape.iter().product();
        let scale = if key.ends_with("layer_scalar") {
            0.2
        } else {
            0.5
        };
        let mut data = values(&mut seed, n, scale);
        if key.ends_with("layer_scalar") {
            data.iter_mut().for_each(|v| *v += 1.0);
        }
        tensors.push((key, shape, data));
    }
    let weights: HashMap<String, Array> = tensors
        .iter()
        .map(|(name, shape, data)| {
            let shape: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
            (name.clone(), Array::from_slice(data, &shape))
        })
        .collect();
    let DynamicModel::Gemma4(gemma4) = &mut model else {
        panic!("a Gemma 4 config builds a Gemma 4");
    };
    load_gemma4_weights(gemma4, &weights).expect("load the checkpoint");
    (model, tensors)
}

fn array(values: Vec<MetadataValue>) -> MetadataValue {
    MetadataValue::Array(values)
}

/// The GGUF a converter writes for the checkpoint `tensors`.
fn gguf(
    tensors: &[(String, Vec<usize>, Vec<f32>)],
    change: impl FnOnce(&mut GgufBuilder),
) -> Vec<u8> {
    gguf_rotating(tensors, ROTATED_PAIRS, change)
}

/// [`gguf`] with a chosen number of rotated pairs in the full-attention heads.
fn gguf_rotating(
    tensors: &[(String, Vec<usize>, Vec<f32>)],
    rotated_pairs: usize,
    change: impl FnOnce(&mut GgufBuilder),
) -> Vec<u8> {
    let mut b = GgufBuilder::with_model("gemma4", "tiny-gemma4");
    b.add_u32("gemma4.embedding_length", HIDDEN as u32);
    b.add_u32("gemma4.block_count", SLIDING.len() as u32);
    b.add_u32("gemma4.feed_forward_length", FFN as u32);
    b.add_u32("gemma4.attention.head_count", HEADS as u32);
    b.add_metadata(
        "gemma4.attention.head_count_kv",
        array(
            SLIDING
                .iter()
                .map(|s| MetadataValue::Uint32(if *s { KV_HEADS } else { GLOBAL_KV_HEADS } as u32))
                .collect(),
        ),
    );
    b.add_u32("gemma4.attention.key_length", GLOBAL_HEAD_DIM as u32);
    b.add_u32("gemma4.attention.value_length", GLOBAL_HEAD_DIM as u32);
    b.add_u32("gemma4.attention.key_length_swa", HEAD_DIM as u32);
    b.add_u32("gemma4.attention.value_length_swa", HEAD_DIM as u32);
    b.add_u32("gemma4.attention.sliding_window", WINDOW);
    b.add_metadata(
        "gemma4.attention.sliding_window_pattern",
        array(SLIDING.iter().map(|s| MetadataValue::Bool(*s)).collect()),
    );
    b.add_f32("gemma4.attention.layer_norm_rms_epsilon", 1e-6);
    b.add_u32("gemma4.context_length", 128);
    b.add_u32("gemma4.embedding_length_per_layer_input", 0);
    b.add_f32("gemma4.rope.freq_base", 1_000_000.0);
    b.add_f32("gemma4.rope.freq_base_swa", 10000.0);
    b.add_u32("gemma4.rope.dimension_count", GLOBAL_HEAD_DIM as u32);
    b.add_u32("gemma4.rope.dimension_count_swa", HEAD_DIM as u32);
    b.add_f32("gemma4.final_logit_softcapping", 30.0);
    let tokens: Vec<String> = (0..VOCAB).map(|i| format!("<t{i}>")).collect();
    b.add_string_array("tokenizer.ggml.tokens", tokens);
    for (name, shape, data) in tensors {
        b.add_f32_tensor(
            WeightLoader::hf_to_gguf_name_gemma(name),
            shape.iter().map(|&d| d as u64).collect(),
            data.clone(),
        );
    }
    let mut factors = vec![1e30f32; GLOBAL_HEAD_DIM / 2];
    factors[..rotated_pairs].fill(1.0);
    b.add_f32_tensor("rope_freqs.weight", vec![factors.len() as u64], factors);
    change(&mut b);
    b.build_to_bytes().expect("build the gguf")
}

fn load(bytes: Vec<u8>) -> Result<DynamicModel, String> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, bytes).expect("write gguf");
    DynamicModel::load(&path).map_err(|e| e.to_string())
}

fn logits(model: &mut DynamicModel) -> Vec<f32> {
    let ids: Vec<i32> = (0..SEQUENCE as i32)
        .map(|i| (i * 5 + 3) % VOCAB as i32)
        .collect();
    let ids = Array::from_slice(&ids, &[1, SEQUENCE as i32]);
    let out = model.forward(&ids, None).expect("forward");
    let mut out = out.as_dtype(Dtype::Float32.as_i32());
    let n = out.size();
    let values = out.to_f32_vec(n).expect("read logits");
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
    values
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
}

#[test]
fn a_gemma_4_gguf_computes_what_its_checkpoint_does() {
    let (mut want, tensors) = reference();
    let mut got = load(gguf(&tensors, |_| {})).expect("the gguf loads");
    let (want, got) = (logits(&mut want), logits(&mut got));
    assert!(want.iter().all(|v| v.is_finite()));
    let spread = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(spread > 1e-2, "the logits are degenerate: {spread}");
    let error = worst(&want, &got);
    assert!(
        error <= 2e-5 * spread.max(1.0),
        "the GGUF's logits differ from the checkpoint's by {error:e} (magnitude {spread})"
    );
}

/// What the file states reaches the model: which layers slide, what the
/// full-attention layers rotate, the window.
#[test]
fn the_gguf_decides_the_layers_the_rope_and_the_window() {
    let (_, tensors) = reference();
    let config = |change: fn(&mut GgufBuilder)| {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.gguf");
        std::fs::write(&path, gguf(&tensors, change)).unwrap();
        pmetal_models::weight_format::gemma4_config_from_path(&path)
    };
    let c = config(|_| {}).unwrap();
    assert!(c.attention_k_eq_v);
    assert_eq!((c.head_dim, c.global_head_dim), (8, Some(16)));
    assert_eq!(
        (c.num_key_value_heads, c.num_global_key_value_heads),
        (2, Some(1))
    );
    assert_eq!(c.sliding_window, 4);
    assert_eq!(c.final_logit_softcapping, Some(30.0));
    assert!(c.tie_word_embeddings);
    assert_eq!(c.layer_types[2], "full_attention");
    assert_eq!(c.layer_types[3], "sliding_attention");
    let rope = c.rope_parameters.as_ref().unwrap();
    assert_eq!(rope.full_attention.partial_rotary_factor, 0.25);
    assert_eq!(rope.full_attention.rope_theta, Some(1_000_000.0));
    assert_eq!(rope.sliding_attention.rope_theta, Some(10000.0));
}

/// Each of these is a number the file states and a model could assume instead.
#[test]
fn each_value_the_file_states_moves_the_logits() {
    let (mut want, tensors) = reference();
    let want = logits(&mut want);
    type Change = fn(&mut GgufBuilder);
    let cases: [(&str, Change); 3] = [
        ("window", |b| {
            b.add_u32("gemma4.attention.sliding_window", 2);
        }),
        ("local rope base", |b| {
            b.add_f32("gemma4.rope.freq_base_swa", 50.0);
        }),
        ("soft cap", |b| {
            b.add_f32("gemma4.final_logit_softcapping", 0.5);
        }),
    ];
    for (what, change) in cases {
        let mut got = load(gguf(&tensors, change)).expect("the gguf loads");
        let moved = worst(&want, &logits(&mut got));
        assert!(
            moved > 1e-3,
            "the {what} did not reach the logits ({moved:e})"
        );
    }
    // The rotated share of the full-attention heads is the tensor's.
    let mut got = load(gguf_rotating(&tensors, 4, |_| {})).expect("the gguf loads");
    let moved = worst(&want, &logits(&mut got));
    assert!(
        moved > 1e-3,
        "the rotated share did not reach the logits ({moved:e})"
    );
}

/// The Gemma 4 models this does not run are refused by name.
#[test]
fn what_the_loader_does_not_run_is_refused_by_name() {
    let (_, tensors) = reference();
    let error = load(gguf(&tensors, |b| {
        b.add_u32("gemma4.embedding_length_per_layer_input", 16);
    }))
    .expect_err("per-layer inputs are refused");
    assert!(error.contains("per-layer input"), "{error}");

    let error = load(gguf(&tensors, |b| {
        b.add_f32_tensor("blk.0.ffn_gate_inp.weight", vec![4, 4], vec![0.0; 16]);
    }))
    .expect_err("experts are refused");
    assert!(error.contains("mixture-of-experts"), "{error}");

    let error = load(gguf(&tensors, |b| {
        b.add_f32_tensor(
            "output.weight",
            vec![VOCAB as u64, HIDDEN as u64],
            vec![0.1; VOCAB * HIDDEN],
        );
    }))
    .expect_err("a head of its own is refused");
    assert!(error.contains("output.weight"), "{error}");
}
