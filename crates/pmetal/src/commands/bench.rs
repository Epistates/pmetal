use pmetal_core::{DatasetConfig, LoraConfig, ModelConfig, TrainingConfig};
use pmetal_lora::LlamaLoraForCausalLM;
use pmetal_models::architectures::llama::LlamaConfig;
use pmetal_trainer::orchestrator::FullTrainingConfig;

/// Run benchmark.
pub(crate) async fn run_benchmark(
    model: &str,
    batch_size: usize,
    seq_len: usize,
) -> anyhow::Result<()> {
    tracing::info!(
        model = %model,
        batch_size = batch_size,
        seq_len = seq_len,
        "Running benchmark"
    );

    println!("Benchmark Configuration:");
    println!("  Model:      {}", model);
    println!("  Batch Size: {}", batch_size);
    println!("  Seq Length: {}", seq_len);
    println!("\nBenchmarking in progress...");

    // Create dummy config
    let llama_config = LlamaConfig {
        vocab_size: 32000,
        hidden_size: 2048,
        intermediate_size: 5632,
        num_hidden_layers: 22,
        num_attention_heads: 32,
        num_key_value_heads: Some(4),
        max_position_embeddings: 2048,
        ..Default::default()
    };

    let lora_config = LoraConfig {
        r: 16,
        alpha: 32.0,
        ..Default::default()
    };

    let mut model_inst = LlamaLoraForCausalLM::new(llama_config, lora_config)?;

    // Create dummy data
    let input_ids = mlx_rs::Array::zeros::<i32>(&[batch_size as i32, seq_len as i32])?;

    // Warmup
    println!("Warming up...");
    for _ in 0..3 {
        let output = model_inst.forward(&input_ids, None)?;
        output.eval()?;
    }

    // Benchmark
    let iterations = 10;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        let output = model_inst.forward(&input_ids, None)?;
        output.eval()?;
    }

    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_millis() as f64 / iterations as f64;
    let tokens_per_sec = (batch_size * seq_len) as f64 / (avg_ms / 1000.0);

    println!("\nResults:");
    println!("  Avg Time:       {:.2} ms/iteration", avg_ms);
    println!("  Throughput:     {:.0} tokens/sec", tokens_per_sec);

    let stats = pmetal_mlx::memory::get_memory_stats();
    println!("  Memory Used:    {:.2} GB", stats.used_gb());
    println!("  Peak Memory:    {:.2} GB", stats.peak_gb());

    Ok(())
}

/// Generate a sample configuration file.
pub(crate) fn generate_sample_config(output: &str) -> anyhow::Result<()> {
    let config = FullTrainingConfig {
        model: ModelConfig {
            model_id: "meta-llama/Llama-3.2-1B".to_string(),
            max_seq_len: 2048,
            ..Default::default()
        },
        lora: LoraConfig {
            r: 16,
            alpha: 32.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            ..Default::default()
        },
        training: TrainingConfig {
            learning_rate: 2e-4,
            batch_size: 4,
            num_epochs: 3,
            warmup_steps: 100,
            max_seq_len: 2048,
            output_dir: "./output".to_string(),
            logging_steps: 10,
            save_steps: Some(500),
            ..Default::default()
        },
        dataset: DatasetConfig {
            dataset_id: "./data/train.jsonl".to_string(),
            shuffle: true,
            ..Default::default()
        },
    };

    let yaml = serde_yaml::to_string(&config)?;
    std::fs::write(output, yaml)?;

    println!("Sample configuration written to: {}", output);
    println!("\nYou can edit this file and run training with:");
    println!("  pmetal train --config {}", output);

    Ok(())
}

/// Benchmark FFI overhead to compare Rust mlx-rs vs Python mlx performance.
///
/// Python baseline: ~7420 argmax ops/sec (~0.135ms per op) on Qwen3 vocab size.
pub(crate) fn run_ffi_benchmark() -> anyhow::Result<()> {
    use mlx_rs::{
        Array,
        ops::indexing::{argmax, argmax_axis},
        transforms::eval,
    };
    use std::time::Instant;

    println!("FFI Overhead Benchmark");
    println!("======================\n");

    // Warmup - ensure Metal is ready
    println!("Warming up Metal...");
    let warmup = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
    let _ = argmax(&warmup, None)?;
    eval([&warmup])?;

    // Create test array similar to Qwen3 logits
    let vocab_size = 151936; // Qwen3 vocab size
    println!("Creating logits array with vocab_size = {}", vocab_size);
    let logits_data: Vec<f32> = (0..vocab_size).map(|i| (i as f32) * 0.001).collect();
    let logits = Array::from_slice(&logits_data, &[1, vocab_size as i32]);
    eval([&logits])?;

    // Benchmark 1: argmax operations
    let n_iters = 1000;
    println!("\n--- Test 1: argmax({} iterations) ---", n_iters);

    let start = Instant::now();
    for _ in 0..n_iters {
        let result = argmax_axis(&logits, -1, None)?;
        eval([&result])?;
    }
    let elapsed = start.elapsed();

    let per_op_us = elapsed.as_micros() as f64 / n_iters as f64;
    let per_op_ms = per_op_us / 1000.0;
    let ops_per_sec = 1_000_000.0 / per_op_us;

    println!("Total time: {:?}", elapsed);
    println!("Per operation: {:.3} ms ({:.1} us)", per_op_ms, per_op_us);
    println!("Operations per second: {:.0}", ops_per_sec);
    println!("Python baseline: ~7420 ops/sec (~0.135 ms/op)");
    println!("Ratio to Python: {:.2}x", ops_per_sec / 7420.0);

    // Benchmark 2: argmax + reshape (generation loop pattern)
    println!("\n--- Test 2: argmax + reshape (generation pattern) ---");

    let token = Array::from_slice(&[42i32], &[1]);
    eval([&token])?;

    let start = Instant::now();
    for _ in 0..n_iters {
        // Reshape token [1] -> [1, 1] (like our generation loop)
        let input = token.reshape(&[1, 1])?;
        // Simulate logits extraction
        let result = argmax_axis(&logits, -1, None)?;
        eval([&input, &result])?;
    }
    let elapsed = start.elapsed();

    let per_op_us = elapsed.as_micros() as f64 / n_iters as f64;
    let per_op_ms = per_op_us / 1000.0;

    println!("Total time: {:?}", elapsed);
    println!("Per operation: {:.3} ms ({:.1} us)", per_op_ms, per_op_us);
    println!("Equivalent tok/s: {:.0}", 1_000_000.0 / per_op_us);

    Ok(())
}

/// Benchmark generation loop timing with a real model.
///
/// This profiles each step of the generation loop to compare with mlx_lm's timing.
pub(crate) async fn run_gen_benchmark(model_id: &str) -> anyhow::Result<()> {
    use mlx_rs::{
        Array,
        ops::indexing::{IndexOp, argmax},
        transforms::{async_eval, eval},
    };
    use pmetal_models::DynamicModel;
    use std::path::PathBuf;
    use std::time::Instant;

    println!("=== Generation Loop Benchmark ===");
    println!("Model: {}\n", model_id);

    // Download model if needed
    let model_path = if model_id.contains('/') && !PathBuf::from(model_id).exists() {
        pmetal_hub::download_model(model_id, None, None).await?
    } else {
        PathBuf::from(model_id)
    };

    // Load model
    println!("Loading model...");
    let mut model = DynamicModel::load(&model_path)?;
    println!("Model loaded.\n");

    // Create KV cache
    let mut cache = model.create_cache(256);

    // Warmup
    println!("Warming up...");
    let token = Array::from_slice(&[42i32], &[1, 1]);
    let _ = model.forward_with_cache(&token, None, Some(&mut cache))?;
    eval([&token])?;
    cache = model.create_cache(256);
    println!("Warmup complete.\n");

    // Profile generation loop
    let n_tokens = 50;
    println!("Profiling {} token generation...\n", n_tokens);

    // Initial forward pass
    let logits = model.forward_with_cache(&token, None, Some(&mut cache))?;
    let mut current_token = argmax(&logits.index((.., -1, ..)), None)?;
    async_eval([&current_token])?;

    let mut times = std::collections::HashMap::new();
    times.insert("reshape", Vec::new());
    times.insert("forward", Vec::new());
    times.insert("extract_logits", Vec::new());
    times.insert("argmax", Vec::new());
    times.insert("async_eval", Vec::new());
    times.insert("item", Vec::new());
    times.insert("total", Vec::new());

    for _ in 0..n_tokens {
        let total_start = Instant::now();

        // Reshape token to [1, 1]
        let t0 = Instant::now();
        let next_input = current_token.reshape(&[1, 1])?;
        times.entry("reshape").or_default().push(t0.elapsed());

        // Forward pass
        let t0 = Instant::now();
        let next_logits = model.forward_with_cache(&next_input, None, Some(&mut cache))?;
        times.entry("forward").or_default().push(t0.elapsed());

        // Extract last logits
        let t0 = Instant::now();
        let last_logits = next_logits.index((.., -1, ..));
        times
            .entry("extract_logits")
            .or_default()
            .push(t0.elapsed());

        // Argmax
        let t0 = Instant::now();
        let next_token = argmax(&last_logits, None)?;
        times.entry("argmax").or_default().push(t0.elapsed());

        // Async eval for next
        let t0 = Instant::now();
        async_eval([&next_token])?;
        times.entry("async_eval").or_default().push(t0.elapsed());

        // Extract current token (sync point)
        let t0 = Instant::now();
        let _ = current_token.item::<u32>();
        times.entry("item").or_default().push(t0.elapsed());

        times
            .entry("total")
            .or_default()
            .push(total_start.elapsed());

        current_token = next_token;
    }

    // Print results
    println!("=== Generation Loop Timing ===");
    for (name, durations) in &times {
        let avg_us: f64 =
            durations.iter().map(|d| d.as_micros() as f64).sum::<f64>() / durations.len() as f64;
        let avg_ms = avg_us / 1000.0;
        println!("{:15}: {:7.3}ms", name, avg_ms);
    }

    let total_avg: f64 = times["total"]
        .iter()
        .map(|d| d.as_micros() as f64)
        .sum::<f64>()
        / times["total"].len() as f64;
    println!("\nEffective tok/s: {:.0}", 1_000_000.0 / total_avg);

    println!("\n=== Comparison ===");
    println!("Python mlx_lm reference:");
    println!("  build_graph: 0.570ms");
    println!("  async_eval:  3.111ms");
    println!("  item:        0.004ms");
    println!("  total:       3.686ms (271 tok/s)");

    Ok(())
}
