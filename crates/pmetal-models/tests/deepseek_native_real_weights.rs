//! The released `deepseek-ai/DeepSeek-V2-Lite` on the native engine against
//! transformers, depth by depth.
//!
//! The reference is a safetensors file with `input_ids` (16 tokens) and, for
//! each depth N, the logits of transformers' `DeepseekV2ForCausalLM`
//! truncated to its first N layers in bf16 (`logits_bf16_N`) and in fp32
//! (`logits_fp32_N`), as f32. The engine runs the same bf16 checkpoint (64
//! routed experts and 2 shared, softmax routing over 6 of them, a direct Q
//! projection, YaRN with `mscale` 0.707).
//!
//! bf16 moves a deep MoE's logits by itself, because a router's near-tie flips
//! an expert, so the engine is not held to bf16 transformers' logits but asked
//! to be about as close to fp32 as transformers' own bf16 run is, at every
//! depth: the distance to fp32 is the format's noise floor. The model is 31 GB
//! and not committed, so this is `#[ignore]`d and gated on the variables (run
//! it alone; it needs the GPU's memory):
//!
//! ```bash
//! PMETAL_DEEPSEEK_V2_LITE_DIR=<snapshot dir> PMETAL_DEEPSEEK_V2_LITE_REFERENCE=<reference.safetensors> \
//!     cargo test -p pmetal-models --test deepseek_native_real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use pmetal_cache::test_utils::{argmax_last_axis, load_shard, ref_tensor};
use pmetal_runtime::compat::Array;
use pmetal_runtime::compat::ops::slice_axis;
use pmetal_runtime::deepseek_native::{NativeCache, forward_step, load_config, load_model};
use serial_test::serial;

fn drain_bridge(op: &str) {
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
}

/// Per-position cosine of `[1, T, vocab]` logits.
fn cosines(got: &Array, want: &Array) -> Vec<f32> {
    let got = got.as_dtype(pmetal_runtime::compat::Dtype::Float32.as_i32());
    let dot = got.multiply(want).sum_axis(-1, false);
    let norms = got
        .square()
        .sum_axis(-1, false)
        .sqrt()
        .multiply(&want.square().sum_axis(-1, false).sqrt());
    let cos = dot.divide(&norms).squeeze(0);
    cos.eval();
    cos.as_slice::<f32>().to_vec()
}

fn worst(cosines: &[f32]) -> f32 {
    cosines.iter().copied().fold(f32::INFINITY, f32::min)
}

/// MLX's 4-bit conversion of DeepSeek-V2-Lite-Chat
/// (`mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx`: the experts stacked as
/// `switch_mlp`, `kv_b_proj`, the shared experts and the embeddings packed,
/// the router gate dense) against the MLX implementation that wrote it,
/// on the same packed weights: the logits over 16 tokens, and the greedy
/// reply to a chat prompt token for token.
///
/// The reference file holds `input_ids`, `logits`, `prompt_ids` and
/// `generated_ids`, from the model's own MLX forward and greedy decode.
///
/// ```bash
/// PMETAL_DEEPSEEK_MLX_DIR=<snapshot dir> PMETAL_DEEPSEEK_MLX_REFERENCE=<reference.safetensors> \
///     cargo test -p pmetal-models --test deepseek_native_real_weights -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs the 9 GB MLX 4-bit DeepSeek-V2-Lite-Chat checkpoint and its MLX reference"]
#[serial]
fn native_deepseek_runs_the_mlx_4bit_conversion_as_mlx_does() {
    let (Some(dir), Some(file)) = (
        std::env::var_os("PMETAL_DEEPSEEK_MLX_DIR"),
        std::env::var_os("PMETAL_DEEPSEEK_MLX_REFERENCE"),
    ) else {
        eprintln!("PMETAL_DEEPSEEK_MLX_DIR / _REFERENCE not set; nothing to check");
        return;
    };
    let dir = PathBuf::from(dir);
    let reference = load_shard(&PathBuf::from(file));
    let ints = |key: &str| -> Vec<i32> {
        let a = ref_tensor(&reference, key).as_dtype(pmetal_runtime::compat::Dtype::Int32.as_i32());
        a.eval();
        a.as_slice::<i32>().to_vec()
    };
    let config = load_config(&dir).expect("config parses");
    let weights = load_model(&dir, &config).expect("weights load");
    drain_bridge("load");
    assert!(weights.experts_packed(), "the experts stay packed");

    // The logits, as one forward.
    let mut cache = NativeCache::new_empty(&weights);
    let logits = forward_step(&weights, ref_tensor(&reference, "input_ids"), &mut cache);
    drain_bridge("forward");
    let cos = cosines(&logits, ref_tensor(&reference, "logits"));
    println!("per-position cosine to MLX: {cos:.6?}");
    assert!(
        worst(&cos) > 0.999,
        "logits drifted from MLX's on the same weights: worst cosine {}",
        worst(&cos)
    );

    // Greedy decode of the chat prompt, through the cache.
    let prompt = ints("prompt_ids");
    let want = ints("generated_ids");
    let ids = |tokens: &[i32]| Array::from_i32_slice_shaped(tokens, &[1, tokens.len() as i32]);
    let mut cache = NativeCache::new_empty(&weights);
    let mut next = {
        let logits = forward_step(&weights, &ids(&prompt), &mut cache);
        argmax_last_axis(&slice_axis(
            &logits,
            1,
            prompt.len() as i32 - 1,
            prompt.len() as i32,
        ))[0]
    };
    let mut got = vec![next];
    while got.len() < want.len() {
        let logits = forward_step(&weights, &ids(&[next]), &mut cache);
        next = argmax_last_axis(&logits)[0];
        got.push(next);
    }
    drain_bridge("decode");
    println!("native greedy tokens: {got:?}\nMLX greedy tokens:    {want:?}");
    assert_eq!(got, want, "the greedy reply differs from MLX's");
}

#[test]
#[ignore = "needs the 31 GB DeepSeek-V2-Lite checkpoint and its transformers reference"]
#[serial]
fn native_deepseek_v2_lite_is_as_close_to_fp32_as_transformers_bf16() {
    let (Some(dir), Some(file)) = (
        std::env::var_os("PMETAL_DEEPSEEK_V2_LITE_DIR"),
        std::env::var_os("PMETAL_DEEPSEEK_V2_LITE_REFERENCE"),
    ) else {
        eprintln!("PMETAL_DEEPSEEK_V2_LITE_DIR / _REFERENCE not set; nothing to check");
        return;
    };
    let dir = PathBuf::from(dir);
    let reference = load_shard(&PathBuf::from(file));
    let ids = ref_tensor(&reference, "input_ids").clone();
    let mut depths: Vec<i32> = reference
        .keys()
        .filter_map(|k| k.strip_prefix("logits_fp32_")?.parse().ok())
        .collect();
    depths.sort_unstable();
    assert!(!depths.is_empty(), "the reference holds no logits_fp32_N");

    for n in depths {
        let fp32 = ref_tensor(&reference, &format!("logits_fp32_{n}")).clone();
        let bf16 = ref_tensor(&reference, &format!("logits_bf16_{n}")).clone();
        let mut config = load_config(&dir).expect("config parses");
        config.num_hidden_layers = n;
        let weights = load_model(&dir, &config).expect("weights load");
        drain_bridge("load");
        let mut cache = NativeCache::new_empty(&weights);
        let logits = forward_step(&weights, &ids, &mut cache);
        drain_bridge("forward");

        let native = worst(&cosines(&logits, &fp32));
        let floor = worst(&cosines(&bf16, &fp32));
        println!(
            "{n:>2} layers: engine vs fp32 worst cosine {native:.6}, \
             transformers bf16 vs fp32 {floor:.6}"
        );
        // The engine's distance from fp32 against bf16's own, with room for
        // a single position's near-tie landing on the other side.
        assert!(
            1.0 - native <= 3.0 * (1.0 - floor) + 2e-5,
            "{n} layers: the engine is {:.2e} from fp32 against transformers bf16's {:.2e}",
            1.0 - native,
            1.0 - floor
        );
    }
}
