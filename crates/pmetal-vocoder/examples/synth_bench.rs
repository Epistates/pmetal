//! Time mel-to-waveform synthesis.
//!
//! ```text
//! cargo run --release -p pmetal-vocoder --example synth_bench -- [v2|base] [frames] [f32|bf16|f16] [ops]
//! ```
//!
//! `ops` skips the fused activation kernels, for comparison.
//!
//! Weights are random (the cost does not depend on their values); the report
//! gives milliseconds per call and the real-time factor, the seconds of audio
//! produced per second of compute.

use pmetal_runtime::compat::{Array, Dtype};
use pmetal_vocoder::{BigVGAN, BigVGANConfig};
use std::time::Instant;

fn main() -> pmetal_vocoder::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let preset = args.first().map_or("v2", String::as_str);
    let frames: i32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(93);
    let dtype = match args.get(2).map_or("f32", String::as_str) {
        "bf16" => Dtype::Bfloat16,
        "f16" => Dtype::Float16,
        _ => Dtype::Float32,
    };
    let config = match preset {
        "base" => BigVGANConfig::base_24khz_100band(),
        _ => BigVGANConfig::v2_24khz_100band(),
    };

    let mut model = BigVGAN::new(config)?;
    model.to_dtype(dtype);
    if args.get(3).is_some_and(|s| s == "ops") {
        // Plain ops, no fused kernels: the baseline to compare against.
        model.fuse_weight_norm();
    } else {
        model.prepare_for_inference();
    }
    let mel =
        Array::random_normal(&[1, model.config.num_mels, frames], Dtype::Float32).as_dtype(dtype);
    let seconds = f64::from(frames * model.config.hop_size) / f64::from(model.config.sampling_rate);

    let run = || -> pmetal_vocoder::Result<()> {
        let audio = model.forward(&mel)?;
        audio.eval();
        Ok(())
    };
    for _ in 0..2 {
        run()?; // warm-up: kernel compilation, allocator
    }
    let runs = 5;
    let start = Instant::now();
    for _ in 0..runs {
        run()?;
    }
    let per_call = start.elapsed().as_secs_f64() / f64::from(runs);
    pmetal_runtime::check_unobserved_error().expect("a runtime op failed");
    println!(
        "{preset} {dtype:?} {frames} frames ({seconds:.2} s of audio): {:.1} ms/call, {:.1}x real time",
        per_call * 1e3,
        seconds / per_call
    );
    Ok(())
}
