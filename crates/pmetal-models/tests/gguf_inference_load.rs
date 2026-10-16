//! End-to-end GGUF inference loading (Phase B).
//!
//! Builds a tiny synthetic Llama GGUF with `GgufBuilder`, loads it through the
//! generic inference entry point `DynamicModel::load`, and runs a forward pass.
//! This exercises the full path: format detection → GGUF metadata → arch config
//! → dequantized weight assignment → logits — with no external download.

use pmetal_gguf::GgufBuilder;
use pmetal_models::{DynamicModel, ModelArchitecture};
use pmetal_runtime::compat::{Array, Dtype};

// Tiny dimensions: keep the fixture sub-kilobyte while still exercising GQA
// (kv_heads < heads), multi-layer stacking, and a non-tied LM head.
const HIDDEN: u64 = 32;
const LAYERS: u64 = 2;
const HEADS: u64 = 4;
const KV_HEADS: u64 = 2;
const HEAD_DIM: u64 = HIDDEN / HEADS; // 8
const INTERMEDIATE: u64 = 64;
const VOCAB: u64 = 16;

fn ramp(rows: u64, cols: u64, scale: f32) -> Vec<f32> {
    let n = (rows * cols) as usize;
    // Small deterministic non-constant values so the forward produces finite,
    // varied logits rather than a degenerate constant.
    (0..n)
        .map(|i| ((i as f32 % 7.0) - 3.0) * scale / (n as f32).sqrt())
        .collect()
}

fn build_tiny_llama_gguf() -> Vec<u8> {
    build_llama_gguf(
        Some(ramp(VOCAB, HIDDEN, 1.0)),
        ramp(HEADS * HEAD_DIM, HIDDEN, 0.5),
    )
}

/// A tiny Llama GGUF with a chosen `output.weight` (`None` omits it, which is
/// how a GGUF says its head is tied to the embedding) and the `attn_q.weight`
/// rows exactly as written, i.e. already in the GGUF converter's permuted order.
fn build_llama_gguf(head: Option<Vec<f32>>, q_as_stored: Vec<f32>) -> Vec<u8> {
    let mut b = GgufBuilder::with_model("llama", "tiny-llama-test");

    b.add_u32("llama.embedding_length", HIDDEN as u32);
    b.add_u32("llama.block_count", LAYERS as u32);
    b.add_u32("llama.attention.head_count", HEADS as u32);
    b.add_u32("llama.attention.head_count_kv", KV_HEADS as u32);
    b.add_u32("llama.feed_forward_length", INTERMEDIATE as u32);
    b.add_u32("llama.attention.head_dim", HEAD_DIM as u32);
    b.add_u32("llama.context_length", 128);
    b.add_f32("llama.attention.layer_norm_rms_epsilon", 1e-5);
    b.add_f32("llama.rope.freq_base", 10000.0);

    // Vocab via the tokenizer token array (matches real GGUFs).
    let tokens: Vec<String> = (0..VOCAB).map(|i| format!("<t{i}>")).collect();
    b.add_string_array("tokenizer.ggml.tokens", tokens);

    let q_dim = HEADS * HEAD_DIM; // 32
    let kv_dim = KV_HEADS * HEAD_DIM; // 16

    b.add_f32_tensor(
        "token_embd.weight",
        vec![VOCAB, HIDDEN],
        ramp(VOCAB, HIDDEN, 1.0),
    );
    b.add_f32_tensor(
        "output_norm.weight",
        vec![HIDDEN],
        vec![1.0; HIDDEN as usize],
    );
    if let Some(head) = head {
        b.add_f32_tensor("output.weight", vec![VOCAB, HIDDEN], head);
    }

    for layer in 0..LAYERS {
        b.add_f32_tensor(
            format!("blk.{layer}.attn_norm.weight"),
            vec![HIDDEN],
            vec![1.0; HIDDEN as usize],
        );
        b.add_f32_tensor(
            format!("blk.{layer}.attn_q.weight"),
            vec![q_dim, HIDDEN],
            q_as_stored.clone(),
        );
        b.add_f32_tensor(
            format!("blk.{layer}.attn_k.weight"),
            vec![kv_dim, HIDDEN],
            ramp(kv_dim, HIDDEN, 0.5),
        );
        b.add_f32_tensor(
            format!("blk.{layer}.attn_v.weight"),
            vec![kv_dim, HIDDEN],
            ramp(kv_dim, HIDDEN, 0.5),
        );
        b.add_f32_tensor(
            format!("blk.{layer}.attn_output.weight"),
            vec![HIDDEN, q_dim],
            ramp(HIDDEN, q_dim, 0.5),
        );
        b.add_f32_tensor(
            format!("blk.{layer}.ffn_norm.weight"),
            vec![HIDDEN],
            vec![1.0; HIDDEN as usize],
        );
        b.add_f32_tensor(
            format!("blk.{layer}.ffn_gate.weight"),
            vec![INTERMEDIATE, HIDDEN],
            ramp(INTERMEDIATE, HIDDEN, 0.5),
        );
        b.add_f32_tensor(
            format!("blk.{layer}.ffn_up.weight"),
            vec![INTERMEDIATE, HIDDEN],
            ramp(INTERMEDIATE, HIDDEN, 0.5),
        );
        b.add_f32_tensor(
            format!("blk.{layer}.ffn_down.weight"),
            vec![HIDDEN, INTERMEDIATE],
            ramp(HIDDEN, INTERMEDIATE, 0.5),
        );
    }

    b.build_to_bytes().expect("build tiny llama gguf")
}

#[test]
fn loads_tiny_llama_gguf_and_runs_forward() {
    let dir = tempfile::tempdir().expect("tempdir");
    let gguf_path = dir.path().join("model.gguf");
    std::fs::write(&gguf_path, build_tiny_llama_gguf()).expect("write gguf");

    // Generic entry point must detect GGUF format and route to load_gguf.
    let mut model = DynamicModel::load(&gguf_path).expect("load gguf");
    assert_eq!(model.architecture(), ModelArchitecture::Llama);
    assert_eq!(model.vocab_size(), VOCAB as i32);
    assert_eq!(model.hidden_size(), HIDDEN as i32);

    // Forward a short prompt; logits must be finite with the expected shape.
    let input_ids = Array::from_slice(&[1i32, 2, 3, 4], &[1, 4]).as_dtype(Dtype::Int32.as_i32());
    let logits = model.forward(&input_ids, None).expect("forward");
    let dims = logits.shape().to_vec();
    assert_eq!(dims[dims.len() - 1], VOCAB as i32, "last dim = vocab");

    let logits_f32 = logits.as_dtype(Dtype::Float32.as_i32());
    logits_f32.eval();
    let flat: Vec<f32> = logits_f32.as_slice::<f32>().to_vec();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "logits contain non-finite values"
    );
}

#[test]
fn loads_tiny_llama_gguf_via_directory() {
    // A directory whose only weights are GGUF should also resolve.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("model.gguf"), build_tiny_llama_gguf()).expect("write gguf");

    let model = DynamicModel::load(dir.path()).expect("load gguf dir");
    assert_eq!(model.architecture(), ModelArchitecture::Llama);
}

fn load_gguf_bytes(bytes: Vec<u8>) -> DynamicModel {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("model.gguf");
    std::fs::write(&path, bytes).expect("write gguf");
    DynamicModel::load(&path).expect("load gguf")
}

fn param(model: &DynamicModel, name: &str) -> Vec<f32> {
    use pmetal_runtime::compat::ModuleParametersExt;
    let array = model
        .flatten_params()
        .into_iter()
        .find(|(key, _)| **key == *name)
        .map(|(_, array)| array)
        .unwrap_or_else(|| panic!("the model has no parameter {name}"));
    let n = array.size();
    let mut array = array.as_dtype(Dtype::Float32.as_i32());
    array.to_f32_vec(n).expect("read parameter")
}

/// ⚠️ A GGUF with an `output.weight` has an untied head. The Llama config
/// defaults to tied, so the head was dropped as an unmatched tensor and the
/// embedding matrix answered in its place.
#[test]
fn an_untied_gguf_head_is_the_models_head() {
    let head: Vec<f32> = (0..VOCAB * HIDDEN).map(|i| (i % 5) as f32 - 2.0).collect();
    let embedding = ramp(VOCAB, HIDDEN, 1.0);
    assert_ne!(
        head, embedding,
        "the test needs a head that differs from the embedding"
    );

    let model = load_gguf_bytes(build_llama_gguf(
        Some(head.clone()),
        ramp(HEADS * HEAD_DIM, HIDDEN, 0.5),
    ));

    assert_eq!(param(&model, "lm_head.weight"), head);
    assert_eq!(param(&model, "model.embed_tokens.weight"), embedding);
}

/// A GGUF with no `output.weight` is tied and loads without a head.
#[test]
fn a_gguf_without_an_output_tensor_is_tied() {
    use pmetal_runtime::compat::ModuleParametersExt;
    let model = load_gguf_bytes(build_llama_gguf(None, ramp(HEADS * HEAD_DIM, HIDDEN, 0.5)));
    assert!(
        model
            .flatten_params()
            .keys()
            .all(|k| !k.starts_with("lm_head")),
        "a tied model has no head of its own"
    );
}

/// ⚠️ the GGUF converter stores a `llama` model's q and k rows permuted for its
/// interleaved RoPE (`LlamaModel.permute` in its converter): head row
/// `a·d/2 + b` is written at row `2b + a`. Written out here by hand, not by the
/// loader's reshape, so the two cannot agree by construction. The model has to
/// hold the published (Hugging Face order) rows.
#[test]
fn llama_gguf_query_rows_are_loaded_in_hugging_face_order() {
    // Every row distinct, in the order Hugging Face publishes them.
    let published: Vec<Vec<f32>> = (0..HEADS * HEAD_DIM)
        .map(|r| (0..HIDDEN).map(|c| (r * 100 + c) as f32 * 0.01).collect())
        .collect();
    let (heads, d) = (HEADS as usize, HEAD_DIM as usize);
    let mut stored = vec![Vec::new(); published.len()];
    for h in 0..heads {
        for a in 0..2 {
            for b in 0..d / 2 {
                stored[h * d + 2 * b + a] = published[h * d + a * (d / 2) + b].clone();
            }
        }
    }

    let model = load_gguf_bytes(build_llama_gguf(
        Some(ramp(VOCAB, HIDDEN, 1.0)),
        stored.into_iter().flatten().collect(),
    ));

    let want: Vec<f32> = published.into_iter().flatten().collect();
    assert_eq!(
        param(&model, "model.layers.0.self_attn.q_proj.weight"),
        want
    );
}
