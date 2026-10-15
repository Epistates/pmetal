//! Decode speed and logits of a real checkpoint on the `DynamicModel` path,
//! the one `pmetal infer` takes with an adapter, FP8 or an experts dir.
//!
//! Gated and ignored: it loads a checkpoint and measures speed, so run it
//! alone, with nothing else on the GPU.
//!
//! ```bash
//! PMETAL_DECODE_MODEL=/path/to/checkpoint \
//! PMETAL_DECODE_LOGITS=/path/to/logits.bin \
//!     cargo test -p pmetal-models --test decode_throughput -- --ignored --nocapture
//! ```
//!
//! It prints greedy decode tok/s over 128 tokens through
//! `generate_cached_async_streaming`, and writes the f32 logits of the
//! prompt's last position and of eight cached decode steps to
//! `PMETAL_DECODE_LOGITS` (when set), so two builds can be compared.

use pmetal_models::dispatcher::DynamicModel;
use pmetal_models::generation::{GenerationConfig, generate_cached_async_streaming};
use pmetal_runtime::compat::Array;

const PROMPT: [u32; 16] = [
    785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576, 6722,
];

fn last_row_f32(logits: &Array) -> Vec<f32> {
    let shape = logits.shape();
    let (seq, vocab) = (shape[1], shape[2]);
    let row = logits
        .slice(&[0, seq - 1, 0], &[1, seq, vocab])
        .as_dtype(pmetal_runtime::compat::Dtype::Float32.as_i32());
    row.as_slice::<f32>().to_vec()
}

#[test]
#[ignore = "needs a real checkpoint; set PMETAL_DECODE_MODEL and run it alone (it measures speed)"]
fn decode_throughput_and_logits() {
    let dir = std::env::var("PMETAL_DECODE_MODEL").expect("set PMETAL_DECODE_MODEL");
    let mut model = DynamicModel::load(&dir).expect("checkpoint loads");

    // Logits: prefill, then eight greedy cached steps.
    let mut cache = model.create_cache(PROMPT.len() + 16);
    let mut mamba = model.create_mamba_cache();
    let ids: Vec<i32> = PROMPT.iter().map(|&t| t as i32).collect();
    let mut logits = model
        .forward_with_hybrid_cache(
            &Array::from_i32_slice_shaped(&ids, &[1, ids.len() as i32]),
            None,
            Some(&mut cache),
            mamba.as_mut(),
        )
        .expect("prefill");
    let mut rows = Vec::new();
    for _ in 0..9 {
        let row = last_row_f32(&logits);
        let next = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as i32;
        rows.push(row);
        logits = model
            .forward_with_hybrid_cache(
                &Array::from_i32_slice_shaped(&[next], &[1, 1]),
                None,
                Some(&mut cache),
                mamba.as_mut(),
            )
            .expect("decode step");
    }
    if let Ok(path) = std::env::var("PMETAL_DECODE_LOGITS") {
        let bytes: Vec<u8> = rows
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(&path, bytes).expect("write logits");
        println!("wrote {} logit rows to {path}", rows.len());
    }

    // Throughput: three runs of 128 greedy tokens, best of three.
    let mut best = 0f64;
    for _ in 0..3 {
        let mut cache = model.create_cache(PROMPT.len() + 160);
        let mut mamba = model.create_mamba_cache();
        let config = GenerationConfig::greedy(128);
        let out = generate_cached_async_streaming(
            |input, cache| {
                model.forward_with_hybrid_cache(input, None, Some(cache), mamba.as_mut())
            },
            &PROMPT,
            config,
            &mut cache,
            |_| true,
        )
        .expect("generate");
        let tps = out.decode_metrics.expect("decode metrics").tok_per_sec;
        best = best.max(tps);
    }
    println!("decode: {best:.1} tok/s (best of 3, 128 greedy tokens)");
    pmetal_runtime::check_unobserved_error().expect("no bridge op failed");
}
