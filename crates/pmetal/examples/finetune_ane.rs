//! Benchmark ANE fine-tuning using the training loop.
//!
//! ```sh
//! cargo run -p pmetal --example finetune_ane --features ane -- \
//!     --dim 768 --seq 256 --steps 5
//! ```

use pmetal_metal::ane::dynamic_trainer::DynamicAneTrainerConfig;
use pmetal_trainer::ane_training::{AneTrainingLoop, AneTrainingLoopConfig};
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();
    let dim = args
        .iter()
        .position(|a| a == "--dim")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(768);
    let seq = args
        .iter()
        .position(|a| a == "--seq")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let steps = args
        .iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    println!("========================================");
    println!("  PMetal ANE Fine-tuning Benchmark");
    println!("========================================");
    println!("Model Dim: {}", dim);
    println!("Seq Len:   {}", seq);
    println!("Steps:     {}", steps);
    println!("Device:    ANE (Neural Engine)");
    println!("----------------------------------------\n");

    let n_layers = 1; // Keep it small for fast benchmark
    let h_dim = dim * 4;
    let v_size = 32000;

    let trainer_config = DynamicAneTrainerConfig {
        dim,
        hidden_dim: h_dim,
        n_heads: 12,
        n_layers,
        vocab_size: v_size,
        seq_len: seq,
        learning_rate: 1e-4,
        ..Default::default()
    };

    let loop_config = AneTrainingLoopConfig {
        trainer: trainer_config,
        num_batches: steps,
        max_steps: steps,
        log_every: 1,
        save_every: None,
        output_dir: PathBuf::from("./pmetal-ane-test"),
    };

    let mut training_loop = AneTrainingLoop::new(loop_config);

    // Calculate exact weight count
    let total_weights = v_size * dim // embed
        + n_layers * (dim + dim * dim * 4 + dim + h_dim * dim * 3) // per-layer
        + dim; // final rms

    println!("Allocating {} weights...", total_weights);
    let weights = vec![0.01f32; total_weights];
    training_loop.load_weights_flat(&weights);

    // Create synthetic data for benchmarking
    let mut data = Vec::new();
    for _ in 0..steps {
        let mut batch = Vec::new();
        for _ in 0..4 {
            batch.push((vec![1u16; seq], vec![2u16; seq]));
        }
        data.push(batch);
    }

    println!("Compiling kernels...");
    training_loop.trainer_mut().compile_kernels()?;

    println!("Starting training...");
    let start = std::time::Instant::now();
    let state = training_loop.train(&data)?;
    let elapsed = start.elapsed();

    println!("\n----------------------------------------");
    println!("  Training Complete");
    println!("----------------------------------------");
    println!("Final Loss:   {:.4}", state.loss);
    println!("Total Tokens: {}", state.tokens_processed);
    println!("Avg Speed:    {:.1} tok/s", state.tokens_per_sec());
    println!("Wall Time:    {:.2}s", elapsed.as_secs_f32());
    println!("========================================\n");

    Ok(())
}
