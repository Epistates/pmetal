//! Benchmark ANE inference using the easy API.
//!
//! ```sh
//! cargo run -p pmetal --example inference_ane --features easy,ane --
//!     --model Qwen/Qwen3-0.6B --prompt "Explain quantum physics."
//! ```

use pmetal_core::Device;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();
    let model = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("unsloth/Qwen3-0.6B");
    let prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("Once upon a time,");

    println!("========================================");
    println!("  PMetal ANE Inference Benchmark");
    println!("========================================");
    println!("Model:  {}", model);
    println!("Device: ANE (Neural Engine)");
    println!(
        "----------------------------------------
"
    );

    let result = pmetal::easy::infer(model)
        .device(Device::Ane)
        .temperature(0.0) // Greedy for benchmarking
        .max_tokens(128)
        .generate(prompt)
        .await?;

    println!(
        "Response:
{}
",
        result.text
    );
    println!("---");
    println!(
        "Throughput: {:.1} tok/s ({} tokens)",
        result.tokens_per_sec, result.tokens_generated
    );
    println!(
        "========================================
"
    );

    Ok(())
}
