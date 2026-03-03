//! Run inference using the easy API.
//!
//! ```sh
//! cargo run -p pmetal --example inference_easy --features easy -- \
//!     --model Qwen/Qwen3-0.6B --prompt "What is 2+2?"
//! ```

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
        .unwrap_or("Qwen/Qwen3-0.6B");
    let prompt = args
        .iter()
        .position(|a| a == "--prompt")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("What is the capital of France?");
    let lora = args
        .iter()
        .position(|a| a == "--lora")
        .and_then(|i| args.get(i + 1));

    println!("Running inference with {model}...\n");

    let mut builder = pmetal::easy::infer(model).temperature(0.7).max_tokens(256);

    if let Some(lora_path) = lora {
        builder = builder.lora(lora_path);
    }

    let result = builder.generate(prompt).await?;

    println!("{}\n", result.text);
    println!("---");
    println!(
        "Generated {} tokens ({:.1} tok/s)",
        result.tokens_generated, result.tokens_per_sec
    );

    Ok(())
}
