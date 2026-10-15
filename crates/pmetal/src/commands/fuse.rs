use std::path::{Path, PathBuf};

/// Fuse LoRA weights into base model weights.
///
/// Folds the adapter into the base model with the streaming merge of
/// `pmetal-merge` (memory-mapped, f64 arithmetic, atomic output), which
/// reads the scale (`alpha / r`, `alpha / sqrt(r)` under rsLoRA) and DoRA
/// magnitudes from the adapter itself and keeps the base's tensor layout and
/// sidecar files. `alpha_override` and `rank_override` replace the adapter's
/// own values; either one alone is combined with the other from
/// `adapter_config.json`.
pub(crate) async fn run_fuse(
    model_path: &str,
    lora_path: &str,
    output_path: &str,
    alpha_override: Option<f32>,
    rank_override: Option<usize>,
) -> anyhow::Result<()> {
    println!("  PMetal LoRA Fuse");
    println!("========================================");

    let scaling = if alpha_override.is_some() || rank_override.is_some() {
        let config = read_adapter_config(lora_path);
        let rank = rank_override
            .map(|r| r as f64)
            .or_else(|| config.as_ref().and_then(|c| c["r"].as_f64()))
            .ok_or_else(|| anyhow::anyhow!("--alpha needs --rank: no r in adapter_config.json"))?;
        let alpha = alpha_override
            .map(f64::from)
            .or_else(|| {
                config
                    .as_ref()
                    .and_then(|c| c["lora_alpha"].as_f64().or_else(|| c["alpha"].as_f64()))
            })
            .unwrap_or(rank);
        println!("LoRA rank: {rank}, alpha: {alpha}, scale: {:.3}", alpha / rank);
        Some(alpha / rank)
    } else {
        None
    };

    fuse_streaming(model_path, lora_path, output_path, false, scaling).await
}

/// `adapter_config.json` next to the adapter, if it parses.
fn read_adapter_config(lora_path: &str) -> Option<serde_json::Value> {
    let dir = if Path::new(lora_path).is_dir() {
        PathBuf::from(lora_path)
    } else {
        Path::new(lora_path)
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf()
    };
    std::fs::read_to_string(dir.join("adapter_config.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

/// Fuse LoRA weights using the f64-accurate streaming merge path.
///
/// Reads `adapter_config.json` from the adapter directory, resolves the base
/// model (single-file or sharded), and writes the merged safetensors to
/// `output_path`.  Non-weight files (tokenizer, config.json, etc.) are copied
/// verbatim.
pub(crate) async fn run_fuse_accurate(
    model_path: &str,
    lora_path: &str,
    output_path: &str,
    low_memory: bool,
) -> anyhow::Result<()> {
    println!("  PMetal LoRA Fuse (f64-accurate path)");
    println!("========================================");
    fuse_streaming(model_path, lora_path, output_path, low_memory, None).await
}

async fn fuse_streaming(
    model_path: &str,
    lora_path: &str,
    output_path: &str,
    low_memory: bool,
    scaling: Option<f64>,
) -> anyhow::Result<()> {
    use pmetal_merge::{AccurateMergeConfig, streaming_lora_merge};

    // Resolve model path (could be HF ID or local path)
    tracing::info!("Resolving model: {}", model_path);
    let model_dir: PathBuf = pmetal_hub::resolve_model_path(model_path, None, None).await?;

    // The adapter path must be a directory containing adapter_config.json,
    // or one of its weight files.
    let adapter_dir: PathBuf = if Path::new(lora_path).is_dir() {
        PathBuf::from(lora_path)
    } else {
        Path::new(lora_path)
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf()
    };

    println!("Base model:   {}", model_dir.display());
    println!("LoRA adapter: {}", adapter_dir.display());
    println!("Output:       {output_path}");
    if low_memory {
        println!("Mode:         low-memory (tiled, 512 rows/tile)");
    }
    println!();

    let mut config = AccurateMergeConfig::new(&model_dir, &adapter_dir, PathBuf::from(output_path));
    if low_memory {
        config = config.with_low_memory(512);
    }
    if let Some(scaling) = scaling {
        config = config.with_scaling(scaling);
    }

    print!("Merging... ");
    let stats = streaming_lora_merge(&config).map_err(|e| anyhow::anyhow!("LoRA fuse failed: {e}"))?;

    println!("done");
    println!();
    println!("Tensors merged:  {}", stats.tensors_merged);
    println!("Tensors copied:  {}", stats.tensors_copied);
    println!(
        "Bytes written:   {:.2} GB",
        stats.bytes_written as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("Elapsed:         {:.1}ms", stats.elapsed_ms);
    println!();
    println!("========================================");
    println!("Fused model saved to: {output_path}");
    println!();
    println!("Next steps:");
    println!("  Inference:  pmetal infer -m {output_path} -p \"Your prompt\"");
    println!("  Quantize:   pmetal quantize -m {output_path} -o {output_path}.gguf");

    Ok(())
}
