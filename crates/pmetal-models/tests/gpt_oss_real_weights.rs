//! The released `openai/gpt-oss-20b` (MXFP4 experts) on both engines against
//! transformers' greedy decode of a harmony-formatted prompt.
//!
//! The reference comes from `.strategy/parity/dump_gpt_oss_20b_reference.py`:
//! transformers loads the release, dequantizes its MXFP4 experts to bf16 and
//! decodes greedily on the CPU, recording the logits each token was picked
//! from. Here both engines keep the experts packed and must pick the same
//! tokens, on a cached decode from the prompt and on one uncached forward over
//! prompt and reply.
//!
//! bf16 moves the logits by itself, so the dumper's `--floor-of` also reruns
//! transformers in fp32: given that (`PMETAL_GPT_OSS_REFERENCE_FP32`), each
//! engine has to be about as close to fp32 as transformers in bf16 is.
//!
//! The checkpoint is 13 GB and not committed, so this is `#[ignore]`d and
//! gated on the variables (run it through the GPU queue):
//!
//! ```bash
//! PMETAL_GPT_OSS_DIR=<snapshot dir> PMETAL_GPT_OSS_REFERENCE=<reference.safetensors> \
//! PMETAL_GPT_OSS_REFERENCE_FP32=<reference_fp32.safetensors> \
//!     gpuq cargo test -p pmetal-models --test gpt_oss_real_weights -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use pmetal_bridge::compat::{Array, ops::slice_axis};
use pmetal_mlx::test_utils::{argmax_last_axis, load_shard, ref_tensor};
use serial_test::serial;

struct Reference {
    dir: PathBuf,
    prompt: Vec<i32>,
    generated: Vec<i32>,
    /// transformers in bf16, the format both engines run.
    logits: Array,
    /// The same rows from transformers in fp32 (`--floor-of`), when given.
    fp32: Option<Array>,
}

fn reference() -> Option<Reference> {
    let dir = std::env::var_os("PMETAL_GPT_OSS_DIR")?;
    let file = std::env::var_os("PMETAL_GPT_OSS_REFERENCE")?;
    let shard = load_shard(&PathBuf::from(file));
    let ints = |key: &str| {
        let a = ref_tensor(&shard, key).as_dtype(pmetal_bridge::compat::Dtype::Int32.as_i32());
        a.eval();
        a.as_slice::<i32>().to_vec()
    };
    let fp32 = std::env::var_os("PMETAL_GPT_OSS_REFERENCE_FP32")
        .map(|f| ref_tensor(&load_shard(&PathBuf::from(f)), "logits").clone());
    Some(Reference {
        dir: PathBuf::from(dir),
        prompt: ints("prompt_ids"),
        generated: ints("generated_ids"),
        logits: ref_tensor(&shard, "logits").clone(),
        fp32,
    })
}

fn ids(tokens: &[i32]) -> Array {
    Array::from_i32_slice_shaped(tokens, &[1, tokens.len() as i32])
}

fn drain_bridge(op: &str) {
    pmetal_bridge::check_last_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
}

/// Max |diff| and the smallest per-row cosine of `[G, vocab]` logits.
fn compare(got: &Array, want: &Array) -> (f32, f32) {
    let got = got.as_dtype(pmetal_bridge::compat::Dtype::Float32.as_i32());
    let diff = got.subtract(want).abs().max(None).item_f32();
    let dot = got.multiply(want).sum_axis(-1, false);
    let norms = got
        .square()
        .sum_axis(-1, false)
        .sqrt()
        .multiply(&want.square().sum_axis(-1, false).sqrt());
    let cos = dot.divide(&norms).min(None).item_f32();
    (diff, cos)
}

/// The rows of a `[1, T, vocab]` forward the generated tokens were picked
/// from: positions `P - 1 .. P + G - 1`.
fn picked_rows(logits: &Array, r: &Reference) -> Array {
    let p = r.prompt.len() as i32;
    let g = r.generated.len() as i32;
    slice_axis(logits, 1, p - 1, p + g - 1).squeeze(0)
}

fn report(engine: &str, how: &str, got: &Array, r: &Reference) {
    let (diff, cos) = compare(got, &r.logits);
    let tokens = argmax_last_axis(got);
    let agree = tokens
        .iter()
        .zip(&r.generated)
        .filter(|(a, b)| a == b)
        .count();
    println!(
        "{engine} {how}: {agree}/{} greedy tokens as transformers bf16, max |logit diff| \
         {diff:.3}, min cosine {cos:.6}",
        r.generated.len()
    );
    assert_eq!(
        tokens, r.generated,
        "{engine} {how}: greedy tokens differ from transformers"
    );
    // Against fp32, the engine in bf16 has to be about as close as
    // transformers in bf16 is: that is all the format allows.
    if let Some(fp32) = &r.fp32 {
        let (floor_diff, floor_cos) = compare(&r.logits, fp32);
        let (diff32, cos32) = compare(got, fp32);
        println!(
            "{engine} {how}: vs transformers fp32 max |diff| {diff32:.3}, min cosine \
             {cos32:.6} (transformers bf16: {floor_diff:.3}, {floor_cos:.6})"
        );
        assert!(
            1.0 - cos32 <= 3.0 * (1.0 - floor_cos),
            "{engine} {how}: {:.2e} from fp32 against bf16's own {:.2e}",
            1.0 - cos32,
            1.0 - floor_cos
        );
    }
}

#[test]
#[serial]
#[ignore = "requires PMETAL_GPT_OSS_DIR, PMETAL_GPT_OSS_REFERENCE and the 13 GB release"]
fn native_gpt_oss_20b_decodes_as_transformers() {
    use pmetal_bridge::gpt_oss_native::{NativeCache, forward_step, load_config, load_model};

    let Some(r) = reference() else {
        eprintln!("PMETAL_GPT_OSS_DIR / PMETAL_GPT_OSS_REFERENCE not set; skipping");
        return;
    };
    let config = load_config(&r.dir).expect("config");
    let weights = load_model(&r.dir, &config).expect("weights");
    drain_bridge("native load");
    assert!(weights.experts_packed(), "the MXFP4 experts stay packed");

    let all: Vec<i32> = r.prompt.iter().chain(&r.generated).copied().collect();
    let mut cache = NativeCache::new_empty(&weights);
    let logits = forward_step(&weights, &ids(&all[..all.len() - 1]), &mut cache);
    drain_bridge("native forward");
    report("native", "uncached forward", &picked_rows(&logits, &r), &r);

    // Cached greedy decode from the prompt, feeding the engine's own picks.
    let mut cache = NativeCache::new_empty(&weights);
    let prefill = forward_step(&weights, &ids(&r.prompt), &mut cache);
    let p = r.prompt.len() as i32;
    let mut rows = vec![slice_axis(&prefill, 1, p - 1, p).squeeze(0)];
    for _ in 1..r.generated.len() {
        let last = argmax_last_axis(rows.last().unwrap());
        let step = forward_step(&weights, &ids(&last), &mut cache);
        drain_bridge("native decode step");
        rows.push(step.squeeze(0));
    }
    let decoded = pmetal_bridge::compat::ops::concatenate_axis(&rows.iter().collect::<Vec<_>>(), 0);
    report("native", "cached greedy decode", &decoded, &r);
}

#[test]
#[serial]
#[ignore = "requires PMETAL_GPT_OSS_DIR, PMETAL_GPT_OSS_REFERENCE and the 13 GB release"]
fn dynamic_gpt_oss_20b_decodes_as_transformers() {
    use pmetal_models::DynamicModel;

    let Some(r) = reference() else {
        eprintln!("PMETAL_GPT_OSS_DIR / PMETAL_GPT_OSS_REFERENCE not set; skipping");
        return;
    };
    let mut model = DynamicModel::load(&r.dir).expect("checkpoint loads");
    drain_bridge("dynamic load");

    let all: Vec<i32> = r.prompt.iter().chain(&r.generated).copied().collect();
    let logits = model
        .forward(&ids(&all[..all.len() - 1]), None)
        .expect("forward");
    drain_bridge("dynamic forward");
    report("dynamic", "uncached forward", &picked_rows(&logits, &r), &r);

    let mut cache = model.create_cache(all.len() + 1);
    let prefill = model
        .forward_with_cache(&ids(&r.prompt), None, Some(&mut cache))
        .expect("prefill");
    let p = r.prompt.len() as i32;
    let mut rows = vec![slice_axis(&prefill, 1, p - 1, p).squeeze(0)];
    for _ in 1..r.generated.len() {
        let last = argmax_last_axis(rows.last().unwrap());
        let step = model
            .forward_with_cache(&ids(&last), None, Some(&mut cache))
            .expect("decode step");
        drain_bridge("dynamic decode step");
        rows.push(step.squeeze(0));
    }
    let decoded = pmetal_bridge::compat::ops::concatenate_axis(&rows.iter().collect::<Vec<_>>(), 0);
    report("dynamic", "cached greedy decode", &decoded, &r);
}
