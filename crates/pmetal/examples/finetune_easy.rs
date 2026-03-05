//! Fine-tune a model using the easy API.
//!
//! ```sh
//! cargo run -p pmetal --example finetune_easy --features easy -- \
//!     --model Qwen/Qwen3-0.6B --dataset data.jsonl
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
    let dataset = args
        .iter()
        .position(|a| a == "--dataset")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("data.jsonl");

    println!("Fine-tuning {model} on {dataset}...\n");

    let result = pmetal::easy::finetune(model, dataset)
        .lora(16, 32.0)
        .epochs(3)
        .learning_rate(2e-4)
        .batch_size(1)
        .output("./output")
        .run()
        .await?;

    println!("\nTraining complete!");
    println!("  Final loss:   {:.4}", result.final_loss);
    println!("  Total steps:  {}", result.total_steps);
    println!("  Total tokens: {}", result.total_tokens);
    println!("  LoRA weights: {}", result.lora_weights_path.display());

    Ok(())
}
