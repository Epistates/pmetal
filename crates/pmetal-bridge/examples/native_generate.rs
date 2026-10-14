//! Greedy generation on a native engine from a synthetic prompt: prints the
//! generated token ids and the decode speed, so two builds can be compared
//! token for token on a real checkpoint.
//!
//! Usage: cargo run -p pmetal-bridge --release --example native_generate -- \
//!          --arch qwen3|gemma4 --model /path/to/checkpoint \
//!          [--prompt-tokens 1024] [--decode 256]

use std::path::Path;

use pmetal_bridge::decode::SamplingParams;

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
    let prompt_tokens: usize = flag("--prompt-tokens").map_or(1024, |v| v.parse().unwrap());
    let decode: usize = flag("--decode").map_or(256, |v| v.parse().unwrap());
    let model = Path::new(&model);

    // A prompt of in-vocabulary ids; its content does not change the work.
    let prompt: Vec<u32> = (0..prompt_tokens)
        .map(|i| 1000 + (i * 7919 % 20000) as u32)
        .collect();
    let params = SamplingParams::new(0.0);

    let started = std::time::Instant::now();
    let (first, tail, metrics) = match arch.as_str() {
        "qwen3" => {
            use pmetal_bridge::qwen3_native::{
                NativeCache, generate, load_config, load_model, prefill_first_token,
            };
            let config = load_config(model).expect("config");
            let weights = load_model(model, &config).expect("weights");
            let mut cache = NativeCache::new_empty(&weights);
            let (first, sampler) = prefill_first_token(&weights, &mut cache, &prompt, params);
            let (tail, metrics) = generate(
                &weights,
                &mut cache,
                first,
                decode.saturating_sub(1),
                sampler,
                |_| true,
            );
            (first, tail, metrics)
        }
        "gemma4" => {
            use pmetal_bridge::gemma4_native::{
                build_cache, generate, load_config, load_model, prefill_first_token,
            };
            let config = load_config(model).expect("config");
            let weights = load_model(model, &config).expect("weights");
            let mut cache = build_cache(&weights, &config);
            let (first, sampler) = prefill_first_token(&weights, &mut cache, &prompt, params);
            let (tail, metrics) = generate(
                &weights,
                &config,
                &mut cache,
                first,
                decode.saturating_sub(1),
                sampler,
                &mut |_| true,
            );
            (first, tail, metrics)
        }
        other => panic!("--arch is qwen3 or gemma4, not {other}"),
    };
    pmetal_bridge::check_unobserved_error().expect("no bridge op failed");

    let mut tokens = vec![first];
    tokens.extend(tail);
    println!("tokens: {tokens:?}");
    if let Some(metrics) = metrics {
        println!("decode: {:.1} tok/s", metrics.tok_per_sec);
    }
    println!("total: {:.1} s", started.elapsed().as_secs_f64());
}
