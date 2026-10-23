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
    build_llama_gguf_with(head, q_as_stored, |_| {})
}

/// [`build_llama_gguf`] with a hook on the builder, for the metadata and
/// tensors a particular file adds (`rope_freqs`, `rope.scaling.*`).
fn build_llama_gguf_with(
    head: Option<Vec<f32>>,
    q_as_stored: Vec<f32>,
    extra: impl FnOnce(&mut GgufBuilder),
) -> Vec<u8> {
    let mut b = GgufBuilder::with_model("llama", "tiny-llama-test");
    extra(&mut b);

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

/// Llama 3.x GGUFs carry their banded RoPE scaling as a `rope_freqs` tensor of
/// per-frequency divisors, which the loader used to set aside with a warning:
/// positions past the original context then rotated at the unscaled
/// frequencies. The loaded model's frequencies are the ones `rope_type:
/// "llama3"` computes, the factors are written here from the converter's
/// recipe rather than the loader's, and the logits move with them.
#[test]
fn a_gguf_rope_freqs_tensor_is_the_models_rope_scaling() {
    use pmetal_runtime::rope::Rotary;

    let (factor, low, high, original) = (8.0f64, 1.0f64, 4.0f64, 128.0f64);
    let plain = Rotary::plain(HEAD_DIM as i32, HEAD_DIM as i32, 10_000.0);
    let factors: Vec<f32> = plain
        .inverse_frequencies(0)
        .iter()
        .map(|&f| {
            let wavelength = 2.0 * std::f64::consts::PI / f as f64;
            (if wavelength < original / high {
                1.0
            } else if wavelength > original / low {
                factor
            } else {
                let smooth = (original / wavelength - low) / (high - low);
                1.0 / ((1.0 - smooth) / factor + smooth)
            }) as f32
        })
        .collect();
    assert!(
        factors.iter().any(|f| *f > 1.0) && factors.contains(&1.0),
        "the test needs both bands: {factors:?}"
    );

    let scaled_bytes = build_llama_gguf_with(
        Some(ramp(VOCAB, HIDDEN, 1.0)),
        ramp(HEADS * HEAD_DIM, HIDDEN, 0.5),
        |b| {
            b.add_f32_tensor("rope_freqs.weight", vec![HEAD_DIM / 2], factors.clone());
        },
    );
    let mut scaled = load_gguf_bytes(scaled_bytes);
    let mut unscaled = load_gguf_bytes(build_tiny_llama_gguf());

    let DynamicModel::Llama(model) = &scaled else {
        panic!("a Llama GGUF loads as Llama");
    };
    let got = model.model.layers[0]
        .self_attn
        .rotary
        .inverse_frequencies(0);
    let want = Rotary::from_config(
        HEAD_DIM as i32,
        pmetal_runtime::rope::RopeConfig::from_json(&serde_json::json!({
            "rope_theta": 10_000.0,
            "rope_scaling": {"rope_type": "llama3", "factor": factor,
                "low_freq_factor": low, "high_freq_factor": high,
                "original_max_position_embeddings": original},
        })),
        10_000.0,
        1.0,
    )
    .expect("llama3 parses")
    .inverse_frequencies(0);
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-6 * w.abs() + 1e-9,
            "frequency {i}: {g} against {w}"
        );
    }

    // The scaling reaches the logits.
    let ids = Array::from_slice(&[1i32, 2, 3, 4, 5, 6], &[1, 6]);
    let a = scaled.forward(&ids, None).expect("scaled forward");
    let b = unscaled.forward(&ids, None).expect("unscaled forward");
    let (mut a, mut b) = (
        a.as_dtype(Dtype::Float32.as_i32()),
        b.as_dtype(Dtype::Float32.as_i32()),
    );
    let n = a.size();
    let (a, b) = (a.to_f32_vec(n).unwrap(), b.to_f32_vec(n).unwrap());
    assert!(a.iter().all(|v| v.is_finite()));
    let moved = a
        .iter()
        .zip(&b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(
        moved > 1e-4,
        "rope_freqs moved the logits by only {moved:e}"
    );
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
}

/// `rope.scaling.*` metadata is a `rope_scaling` dict; a type the loader does
/// not know is refused, and so is a file with metadata and a tensor both.
#[test]
fn gguf_rope_scaling_metadata_is_read_and_unknown_types_are_refused() {
    use pmetal_runtime::rope::RopeScaling;

    let linear = load_gguf_bytes(build_llama_gguf_with(
        Some(ramp(VOCAB, HIDDEN, 1.0)),
        ramp(HEADS * HEAD_DIM, HIDDEN, 0.5),
        |b| {
            b.add_string("llama.rope.scaling.type", "linear");
            b.add_f32("llama.rope.scaling.factor", 4.0);
        },
    ));
    let DynamicModel::Llama(model) = &linear else {
        panic!("llama");
    };
    assert_eq!(
        model.model.layers[0].self_attn.rotary.rotary().scaling,
        RopeScaling::Linear { factor: 4.0 }
    );

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("model.gguf");
    std::fs::write(
        &path,
        build_llama_gguf_with(
            Some(ramp(VOCAB, HIDDEN, 1.0)),
            ramp(HEADS * HEAD_DIM, HIDDEN, 0.5),
            |b| {
                b.add_string("llama.rope.scaling.type", "longrope");
            },
        ),
    )
    .unwrap();
    let err = DynamicModel::load(&path).expect_err("refused");
    assert!(err.to_string().contains("longrope"), "{err}");

    std::fs::write(
        &path,
        build_llama_gguf_with(
            Some(ramp(VOCAB, HIDDEN, 1.0)),
            ramp(HEADS * HEAD_DIM, HIDDEN, 0.5),
            |b| {
                b.add_string("llama.rope.scaling.type", "linear");
                b.add_f32("llama.rope.scaling.factor", 2.0);
                b.add_f32_tensor("rope_freqs.weight", vec![HEAD_DIM / 2], vec![1.0; 4]);
            },
        ),
    )
    .unwrap();
    let err = DynamicModel::load(&path).expect_err("refused");
    assert!(err.to_string().contains("rope_freqs"), "{err}");
}

/// Phi-3 GGUFs fuse the attention projections (`attn_qkv`, rows q, k, v) and
/// the SwiGLU gate and up (`ffn_up`, rows gate, up), keep LongRoPE's factors in
/// `rope_factors_short` / `rope_factors_long` tensors, and give `phi3.*`
/// metadata where the head is `hidden / heads` and `rope.dimension_count` only
/// the rotated part of it. Written out from a model's own tensors in that
/// layout, the file loads to the model the safetensors checkpoint does, past the
/// original context where the long table applies.
#[test]
fn a_phi3_gguf_loads_to_the_model_its_safetensors_checkpoint_does() {
    use pmetal_runtime::compat::{ModuleParametersExt, random};

    const PHI_HIDDEN: u64 = 32;
    const PHI_HEADS: u64 = 4;
    const PHI_KV: u64 = 2;
    const PHI_HEAD_DIM: u64 = PHI_HIDDEN / PHI_HEADS; // 8
    const PHI_ROT: u64 = 4; // partial rotary factor 0.5
    const PHI_FF: u64 = 64;
    const PHI_LAYERS: u64 = 2;
    const ORIGINAL: u32 = 8;
    let short = [1.0f32, 1.5];
    let long = [2.0f32, 4.0];

    let config = serde_json::json!({
        "model_type": "phi3", "vocab_size": VOCAB, "hidden_size": PHI_HIDDEN,
        "intermediate_size": PHI_FF, "num_hidden_layers": PHI_LAYERS,
        "num_attention_heads": PHI_HEADS, "num_key_value_heads": PHI_KV,
        "max_position_embeddings": 64, "original_max_position_embeddings": ORIGINAL,
        "rope_theta": 10000.0, "partial_rotary_factor": 0.5, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": false,
        "rope_scaling": {"rope_type": "longrope", "short_factor": short, "long_factor": long},
    });

    // A model with random weights, saved as the Hugging Face checkpoint.
    random::seed(21);
    let model = DynamicModel::from_config(&config.to_string()).expect("build phi3");
    let params: Vec<(String, Array)> = model
        .flatten_params()
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    let entries: Vec<(&str, &Array)> = params.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(
        dir.path().join("model.safetensors").to_str().unwrap(),
        &entries,
    )
    .expect("save the checkpoint");
    let mut from_safetensors = DynamicModel::load(dir.path()).expect("the checkpoint loads");

    // The same tensors in the GGUF's layout.
    let host = |name: &str| -> Vec<f32> {
        let (_, array) = params
            .iter()
            .find(|(k, _)| k == name)
            .unwrap_or_else(|| panic!("no parameter {name}"));
        let mut array = array.as_dtype(Dtype::Float32.as_i32());
        let n = array.size();
        array.to_f32_vec(n).expect("read")
    };
    let rows = |name: &str, rows: u64| (vec![rows, PHI_HIDDEN], host(name));
    let mut b = GgufBuilder::with_model("phi3", "tiny-phi3-test");
    b.add_u32("phi3.embedding_length", PHI_HIDDEN as u32);
    b.add_u32("phi3.block_count", PHI_LAYERS as u32);
    b.add_u32("phi3.attention.head_count", PHI_HEADS as u32);
    b.add_u32("phi3.attention.head_count_kv", PHI_KV as u32);
    b.add_u32("phi3.feed_forward_length", PHI_FF as u32);
    b.add_u32("phi3.rope.dimension_count", PHI_ROT as u32);
    b.add_u32("phi3.context_length", 64);
    b.add_u32("phi3.rope.scaling.original_context_length", ORIGINAL);
    b.add_f32("phi3.attention.layer_norm_rms_epsilon", 1e-5);
    b.add_f32("phi3.rope.freq_base", 10000.0);
    b.add_string_array(
        "tokenizer.ggml.tokens",
        (0..VOCAB).map(|i| format!("<t{i}>")).collect(),
    );
    b.add_f32_tensor(
        "token_embd.weight",
        vec![VOCAB, PHI_HIDDEN],
        host("model.embed_tokens.weight"),
    );
    b.add_f32_tensor(
        "output_norm.weight",
        vec![PHI_HIDDEN],
        host("model.norm.weight"),
    );
    b.add_f32_tensor(
        "output.weight",
        vec![VOCAB, PHI_HIDDEN],
        host("lm_head.weight"),
    );
    b.add_f32_tensor("rope_factors_short.weight", vec![2], short.to_vec());
    b.add_f32_tensor("rope_factors_long.weight", vec![2], long.to_vec());
    let qkv_rows = (PHI_HEADS + 2 * PHI_KV) * PHI_HEAD_DIM;
    for l in 0..PHI_LAYERS {
        let p = format!("model.layers.{l}");
        let mut qkv = host(&format!("{p}.self_attn.q_proj.weight"));
        qkv.extend(host(&format!("{p}.self_attn.k_proj.weight")));
        qkv.extend(host(&format!("{p}.self_attn.v_proj.weight")));
        b.add_f32_tensor(
            format!("blk.{l}.attn_qkv.weight"),
            vec![qkv_rows, PHI_HIDDEN],
            qkv,
        );
        let (shape, data) = rows(&format!("{p}.self_attn.o_proj.weight"), PHI_HIDDEN);
        b.add_f32_tensor(format!("blk.{l}.attn_output.weight"), shape, data);
        b.add_f32_tensor(
            format!("blk.{l}.attn_norm.weight"),
            vec![PHI_HIDDEN],
            host(&format!("{p}.input_layernorm.weight")),
        );
        b.add_f32_tensor(
            format!("blk.{l}.ffn_norm.weight"),
            vec![PHI_HIDDEN],
            host(&format!("{p}.post_attention_layernorm.weight")),
        );
        // Gate then up, fused.
        b.add_f32_tensor(
            format!("blk.{l}.ffn_up.weight"),
            vec![2 * PHI_FF, PHI_HIDDEN],
            host(&format!("{p}.mlp.gate_up_proj.weight")),
        );
        b.add_f32_tensor(
            format!("blk.{l}.ffn_down.weight"),
            vec![PHI_HIDDEN, PHI_FF],
            host(&format!("{p}.mlp.down_proj.weight")),
        );
    }
    let mut from_gguf = load_gguf_bytes(b.build_to_bytes().expect("build phi3 gguf"));
    assert_eq!(from_gguf.architecture(), ModelArchitecture::Phi);

    // 12 tokens against an original context of 8: the long table applies.
    let ids = Array::from_slice(
        &(0..12)
            .map(|i| (i * 5 + 1) % VOCAB as i32)
            .collect::<Vec<_>>(),
        &[1, 12],
    );
    let a = from_safetensors
        .forward(&ids, None)
        .expect("safetensors forward");
    let b = from_gguf.forward(&ids, None).expect("gguf forward");
    let n = a.size();
    let (mut a, mut b) = (
        a.as_dtype(Dtype::Float32.as_i32()),
        b.as_dtype(Dtype::Float32.as_i32()),
    );
    let (a, b) = (a.to_f32_vec(n).unwrap(), b.to_f32_vec(n).unwrap());
    assert!(a.iter().all(|v| v.is_finite()));
    let worst = a
        .iter()
        .zip(&b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
    let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
    assert!(
        worst / scale < 1e-5,
        "the GGUF differs from its checkpoint by {worst:e} of {scale:e}"
    );
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
}

/// ⚠️ A file that does not say how large its vocabulary is cannot be loaded:
/// the size used to default to 32000, which sized the embedding for a model it
/// did not belong to.
#[test]
fn a_gguf_with_no_vocabulary_is_refused_not_defaulted() {
    let mut b = GgufBuilder::with_model("llama", "no-vocab");
    b.add_u32("llama.embedding_length", HIDDEN as u32)
        .add_u32("llama.block_count", LAYERS as u32)
        .add_u32("llama.attention.head_count", HEADS as u32)
        .add_u32("llama.feed_forward_length", INTERMEDIATE as u32);
    let bytes = b.build_to_bytes().expect("build");
    let content = pmetal_gguf::GgufContent::read(&mut std::io::Cursor::new(bytes)).expect("read");
    let err = pmetal_models::GgufModelConfig::from_gguf(&content).expect_err("no vocabulary");
    assert!(err.to_string().contains("vocab_size"), "{err}");
}
