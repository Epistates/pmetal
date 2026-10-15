//! The model a Qwen3.5-family checkpoint trains as is the model it is served
//! as: the `DynamicModel` engine (what training, adapters and the speculative
//! decoders run on) and the native engine (`pmetal infer`) give the same
//! logits for the same weights.
//!
//! Gated and ignored: it loads multi-GB checkpoints, so run it alone, with
//! `PMETAL_QWEN_CHECKPOINTS` a path list (`:`-separated) of
//! released checkpoint directories. NVIDIA ModelOpt NVFP4 releases exercise the
//! packed dense and routed-expert weights on both engines.
//!
//! ```bash
//! PMETAL_QWEN_CHECKPOINTS=/models/Qwen3.5-0.8B:/models/Qwen3.6-35B-A3B-NVFP4 \
//!     cargo test -p pmetal-models --test qwen3_next_train_serve -- --ignored --nocapture
//! ```
//!
//! Each checkpoint runs a 16-token prompt through the engine's cached prefill,
//! then decodes greedy tokens one at a time, the dynamic engine's choices fed
//! to both. The checkpoint's activations are bf16, which the two engines round
//! at different points (the native engine fuses a few ops the layer modules
//! run one by one), so the logits agree to bf16 noise, not bit for bit: the
//! gate is cosine similarity (0.99) and the greedy token. A wrong packed layout, a
//! gate and up swapped, a conv state carried wrongly or an expert's tensor
//! scale dropped lands far outside it.

use pmetal_cache::test_utils::{cosine_similarity_flat, max_abs_diff, max_abs_value};
use pmetal_models::DynamicModel;
use pmetal_runtime::compat::{Array, Dtype, ops::slice_axis};
use pmetal_runtime::decode::prompt_tokens_to_input;
use pmetal_runtime::qwen3_native::{NativeCache, forward_step_last, load_config, load_model};

const PROMPT: [u32; 16] = [
    785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576, 6722,
];
const DECODE_STEPS: usize = 6;
/// bf16 engines of the same weights that round in different places. A dense
/// 0.8B checkpoint agrees to 0.9999; a 256-expert MoE to 0.997 or better, since
/// a router near a tie picks another expert on one engine's rounding.
const MIN_COSINE: f32 = 0.99;

fn drain(op: &str) {
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| panic!("{op}: runtime error: {e}"));
}

fn last_row(logits: &Array) -> Array {
    let t = logits.dim(1);
    slice_axis(logits, 1, t - 1, t)
        .reshape(&[1, 1, -1])
        .as_dtype(Dtype::Float32.as_i32())
}

fn argmax(row: &Array) -> u32 {
    pmetal_cache::test_utils::argmax_last_axis(row)[0] as u32
}

/// Logits of the last prompt position and of each decode step on the
/// `DynamicModel` engine, with the greedy tokens it chose.
fn dynamic_run(dir: &std::path::Path) -> (Vec<Array>, Vec<u32>) {
    let started = std::time::Instant::now();
    let mut model = DynamicModel::load(dir).expect("dynamic engine loads");
    drain("dynamic load");
    println!("dynamic engine loaded in {:.1?}", started.elapsed());
    let mut kv = model.create_cache(PROMPT.len() + DECODE_STEPS + 1);
    let mut mamba = model
        .create_mamba_cache()
        .expect("hybrid model has a GDN cache");

    let mut rows = Vec::new();
    let mut tokens = Vec::new();
    let mut input = prompt_tokens_to_input(&PROMPT);
    for _ in 0..=DECODE_STEPS {
        let logits = model
            .forward_with_hybrid_cache(&input, None, Some(&mut kv), Some(&mut mamba))
            .expect("dynamic forward");
        drain("dynamic forward");
        let row = last_row(&logits);
        row.eval();
        println!("dynamic step done at {:.1?}", started.elapsed());
        let next = argmax(&row);
        rows.push(row);
        tokens.push(next);
        input = prompt_tokens_to_input(&[next]);
    }
    (rows, tokens)
}

/// The same tokens through the native engine: the prompt, then `tokens`.
fn native_run(dir: &std::path::Path, tokens: &[u32]) -> Vec<Array> {
    let started = std::time::Instant::now();
    let config = load_config(dir).expect("native config parses");
    let weights = load_model(dir, &config).expect("native engine loads");
    drain("native load");
    println!("native engine loaded in {:.1?}", started.elapsed());
    let mut cache = NativeCache::new_empty(&weights);

    let mut rows = Vec::new();
    let mut input = prompt_tokens_to_input(&PROMPT);
    for step in 0..=DECODE_STEPS {
        let logits = forward_step_last(&weights, &input, &mut cache);
        drain("native forward");
        let row = logits
            .reshape(&[1, 1, -1])
            .as_dtype(Dtype::Float32.as_i32());
        row.eval();
        rows.push(row);
        if let Some(&next) = tokens.get(step) {
            input = prompt_tokens_to_input(&[next]);
        }
    }
    rows
}

#[test]
#[ignore = "needs PMETAL_QWEN_CHECKPOINTS (released checkpoints); run it alone"]
fn the_dynamic_engine_matches_the_native_engine() {
    let list = std::env::var("PMETAL_QWEN_CHECKPOINTS").expect("set PMETAL_QWEN_CHECKPOINTS");
    for dir in list.split(':').filter(|d| !d.is_empty()) {
        let dir = std::path::Path::new(dir);
        println!("\n== {} ==", dir.display());

        let (dynamic, tokens) = dynamic_run(dir);
        pmetal_runtime::inline_array::clear_cache();
        let native = native_run(dir, &tokens);
        pmetal_runtime::inline_array::clear_cache();

        let mut failures = Vec::new();
        for (step, (d, n)) in dynamic.iter().zip(&native).enumerate() {
            let cosine = cosine_similarity_flat(d, n);
            let diff = max_abs_diff(d, n);
            let scale = max_abs_value(d);
            let same = argmax(d) == argmax(n);
            println!(
                "step {step}: cosine {cosine:.6}  max |diff| {diff:.3e} (logit scale {scale:.2})  greedy {} vs {}",
                argmax(d),
                argmax(n)
            );
            if !(cosine >= MIN_COSINE && same && cosine.is_finite()) {
                failures.push(step);
            }
        }
        assert!(
            failures.is_empty(),
            "{}: steps {failures:?} differ between the engines",
            dir.display()
        );
    }
}
