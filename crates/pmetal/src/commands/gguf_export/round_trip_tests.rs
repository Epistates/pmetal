//! A checkpoint exported to GGUF and loaded again computes the checkpoint's
//! logits.
//!
//! This is the writer and the loader held against each other on a whole model,
//! the one thing neither's own tests can see: a tensor the writer names, lays
//! out or offsets in a way the loader does not undo is not an error in either,
//! only a model that answers differently.

use crate::commands::quantize::{ImportanceSource, run_quantization};
use pmetal_models::DynamicModel;
use pmetal_runtime::compat::{Array, Dtype};
use std::path::Path;

const HIDDEN: usize = 32;
const FFN: usize = 64;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const VOCAB: usize = 32;
const LAYERS: usize = 6;
const SEQUENCE: usize = 12;

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

/// Write a tiny Gemma 3 checkpoint (bf16, one file) and its config.
fn write_gemma3(dir: &Path) {
    use safetensors::tensor::{Dtype as StDtype, TensorView, serialize};
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut add = |name: String, shape: &[usize], scale: f32| {
        let n: usize = shape.iter().product();
        let bytes = rng
            .values(n, scale)
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        tensors.push((name, shape.to_vec(), bytes));
    };
    add("model.embed_tokens.weight".into(), &[VOCAB, HIDDEN], 1.0);
    add("model.norm.weight".into(), &[HIDDEN], 0.5);
    for layer in 0..LAYERS {
        let p = format!("model.layers.{layer}");
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            add(format!("{p}.{norm}.weight"), &[HIDDEN], 0.5);
        }
        add(format!("{p}.self_attn.q_norm.weight"), &[HEAD_DIM], 0.5);
        add(format!("{p}.self_attn.k_norm.weight"), &[HEAD_DIM], 0.5);
        add(
            format!("{p}.self_attn.q_proj.weight"),
            &[HEADS * HEAD_DIM, HIDDEN],
            0.4,
        );
        add(
            format!("{p}.self_attn.k_proj.weight"),
            &[KV_HEADS * HEAD_DIM, HIDDEN],
            0.4,
        );
        add(
            format!("{p}.self_attn.v_proj.weight"),
            &[KV_HEADS * HEAD_DIM, HIDDEN],
            0.4,
        );
        add(
            format!("{p}.self_attn.o_proj.weight"),
            &[HIDDEN, HEADS * HEAD_DIM],
            0.4,
        );
        add(format!("{p}.mlp.gate_proj.weight"), &[FFN, HIDDEN], 0.4);
        add(format!("{p}.mlp.up_proj.weight"), &[FFN, HIDDEN], 0.4);
        add(format!("{p}.mlp.down_proj.weight"), &[HIDDEN, FFN], 0.4);
    }
    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, shape, data)| {
            (
                name.clone(),
                TensorView::new(StDtype::BF16, shape.clone(), data).unwrap(),
            )
        })
        .collect();
    std::fs::write(
        dir.join("model.safetensors"),
        serialize(views, None).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "architectures": ["Gemma3ForCausalLM"], "model_type": "gemma3_text",
            "vocab_size": VOCAB, "hidden_size": HIDDEN, "intermediate_size": FFN,
            "num_hidden_layers": LAYERS, "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS, "head_dim": HEAD_DIM,
            "max_position_embeddings": 128, "rms_norm_eps": 1e-6,
            "hidden_activation": "gelu_pytorch_tanh", "tie_word_embeddings": true,
            "rope_theta": 1_000_000.0, "rope_local_base_freq": 10000.0,
            "sliding_window": 4, "query_pre_attn_scalar": HEAD_DIM,
            "rope_scaling": {"rope_type": "linear", "factor": 8.0},
        })
        .to_string(),
    )
    .unwrap();
}

fn logits(model: &mut DynamicModel) -> Vec<f32> {
    let ids: Vec<i32> = (0..SEQUENCE as i32)
        .map(|i| (i * 5 + 3) % VOCAB as i32)
        .collect();
    let ids = Array::from_slice(&ids, &[1, SEQUENCE as i32]);
    let out = model.forward(&ids, None).expect("forward");
    let mut out = out.as_dtype(Dtype::Float32.as_i32());
    let n = out.size();
    out.to_f32_vec(n).expect("read the logits")
}

async fn export(dir: &Path, method: crate::cli::quantize::QuantizeMethod) -> DynamicModel {
    let out = dir.join(format!("{}.gguf", method.as_str()));
    run_quantization(
        dir.to_str().unwrap(),
        out.to_str().unwrap(),
        ImportanceSource::None,
        method,
        false,
        None,
        0.01,
    )
    .await
    .expect("export");
    DynamicModel::load(&out).expect("the exported file loads")
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
}

/// Gemma 3, exported at bf16 (the weights as they were) and at q8_0 (their
/// quantization), loaded back and run: the logits are the checkpoint's, within
/// what the type loses.
#[tokio::test]
async fn a_gemma_3_checkpoint_survives_its_own_export() {
    let dir = tempfile::tempdir().unwrap();
    write_gemma3(dir.path());
    let mut reference = DynamicModel::load(dir.path()).expect("the checkpoint loads");
    let want = logits(&mut reference);
    let spread = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(spread > 1e-2 && want.iter().all(|v| v.is_finite()));

    let mut bf16 = export(dir.path(), crate::cli::quantize::QuantizeMethod::Bf16).await;
    let bf16 = logits(&mut bf16);
    let mut q8 = export(dir.path(), crate::cli::quantize::QuantizeMethod::Q8_0).await;
    let q8 = logits(&mut q8);

    // bf16 keeps every weight; what differs is the arithmetic the two models
    // run it in (the checkpoint's own half precision against f32).
    let bf16_error = worst(&want, &bf16) / spread;
    // q8_0 rounds every matrix to 8 bits.
    let q8_error = worst(&want, &q8) / spread;
    println!("bf16: {bf16_error:e}, q8_0: {q8_error:e} (of {spread})");
    assert!(bf16_error < 1.5e-2, "bf16 export: {bf16_error:e}");
    assert!(q8_error < 3.5e-2, "q8_0 export: {q8_error:e}");
}
