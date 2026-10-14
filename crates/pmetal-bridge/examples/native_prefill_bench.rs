//! Prefill speed and peak memory of a native engine: the prompt through one
//! forward pass computing every position's logits, against the engine's own
//! prefill (passes of `PREFILL_CHUNK` tokens, the last position's logits only).
//! The two alternate, so drift in the machine's load shows in both.
//!
//! Usage: cargo run -p pmetal-bridge --release --example native_prefill_bench -- \
//!          --arch qwen3|gemma4 --model /path/to/checkpoint --prompt-tokens 4096 [--rounds 3]

use std::path::Path;
use std::time::Instant;

use pmetal_bridge::decode::{SamplingParams, prefill_first_token};
use pmetal_bridge::inline_array::{clear_cache, get_peak_memory, reset_peak_memory};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let arch = flag("--arch").unwrap_or_else(|| "qwen3".into());
    let model = flag("--model").expect("--model <dir>");
    let tokens: usize = flag("--prompt-tokens").map_or(4096, |v| v.parse().unwrap());
    let rounds: usize = flag("--rounds").map_or(3, |v| v.parse().unwrap());
    let model = Path::new(&model);

    // A prompt of in-vocabulary ids; the content does not change the work.
    let prompt: Vec<u32> = (0..tokens)
        .map(|i| 1000 + (i * 7919 % 20000) as u32)
        .collect();

    let mut measure: Box<dyn FnMut(bool) -> (f64, usize, u32)> = match arch.as_str() {
        "qwen3" => {
            use pmetal_bridge::qwen3_native::{
                NativeCache, forward_step, load_config, load_model, prefill_first_token as engine,
            };
            let config = load_config(model).expect("config");
            let weights = load_model(model, &config).expect("weights");
            Box::new(move |chunked| {
                let mut cache = NativeCache::new_empty(&weights);
                time(|| {
                    if chunked {
                        engine(&weights, &mut cache, &prompt, SamplingParams::new(0.0)).0
                    } else {
                        prefill_first_token(
                            &weights,
                            &mut cache,
                            &prompt,
                            SamplingParams::new(0.0),
                            forward_step,
                        )
                        .0
                    }
                })
            })
        }
        "gemma4" => {
            use pmetal_bridge::gemma4_native::{
                build_cache, forward_step, load_config, load_model, prefill_first_token as engine,
            };
            let config = load_config(model).expect("config");
            let weights = load_model(model, &config).expect("weights");
            Box::new(move |chunked| {
                let mut cache = build_cache(&weights, &config);
                time(|| {
                    if chunked {
                        engine(&weights, &mut cache, &prompt, SamplingParams::new(0.0)).0
                    } else {
                        prefill_first_token(
                            &weights,
                            &mut cache,
                            &prompt,
                            SamplingParams::new(0.0),
                            forward_step,
                        )
                        .0
                    }
                })
            })
        }
        other => panic!("--arch is qwen3 or gemma4, not {other}"),
    };

    // One untimed pass of each warms the compiled graphs.
    measure(false);
    measure(true);
    for round in 0..rounds {
        for chunked in [false, true] {
            let (secs, peak, token) = measure(chunked);
            println!(
                "round {round} {:>26}: {:8.1} ms, {:8.0} tok/s, peak {:7.0} MB (first token {token})",
                if chunked {
                    "passes, last logits only"
                } else {
                    "whole prompt, all logits"
                },
                secs * 1e3,
                tokens as f64 / secs,
                peak as f64 / 1e6,
            );
        }
    }
}

/// Seconds and peak bytes of one prefill, and its first token.
fn time(prefill: impl FnOnce() -> u32) -> (f64, usize, u32) {
    clear_cache();
    reset_peak_memory();
    let start = Instant::now();
    let token = prefill();
    let secs = start.elapsed().as_secs_f64();
    pmetal_bridge::check_unobserved_error().expect("no bridge op failed");
    (secs, get_peak_memory(), token)
}
