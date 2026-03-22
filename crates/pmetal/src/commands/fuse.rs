use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Fuse LoRA weights into base model weights.
///
/// Loads LoRA adapter weights and merges them into the base model's corresponding
/// layer, copies all other files (config, tokenizer, etc.), and saves the result.
pub(crate) async fn run_fuse(
    model_path: &str,
    lora_path: &str,
    output_path: &str,
    alpha_override: Option<f32>,
    rank_override: Option<usize>,
) -> anyhow::Result<()> {
    println!("  PMetal LoRA Fuse");
    println!("========================================");

    // Resolve model path (could be HF ID or local path)
    let model_dir: PathBuf = if model_path.contains('/') && !PathBuf::from(model_path).exists() {
        tracing::info!("Resolving HuggingFace model: {}", model_path);
        pmetal_hub::download_model(model_path, None, None).await?
    } else {
        PathBuf::from(model_path)
    };
    println!("Base model:   {}", model_dir.display());

    // Resolve LoRA adapter path
    let lora_file = if Path::new(lora_path).is_dir() {
        let f = Path::new(lora_path).join("lora_weights.safetensors");
        if !f.exists() {
            anyhow::bail!("No lora_weights.safetensors found in {}", lora_path);
        }
        f
    } else {
        PathBuf::from(lora_path)
    };
    println!("LoRA adapter: {}", lora_file.display());
    println!("Output:       {}", output_path);

    // Load base model weights
    print!("\nLoading base model weights... ");
    let mut base_weights = pmetal_models::loader::load_weights(&model_dir)?;
    println!("OK ({} tensors)", base_weights.len());

    // Load LoRA adapter weights
    print!("Loading LoRA adapter weights... ");
    let lora_weights =
        mlx_rs::Array::load_safetensors(&lora_file).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("OK ({} tensors)", lora_weights.len());

    // Read rank and alpha from adapter_config.json, with CLI overrides taking precedence
    let lora_dir = if std::path::Path::new(lora_path).is_dir() {
        std::path::PathBuf::from(lora_path)
    } else {
        std::path::Path::new(lora_path)
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf()
    };
    let adapter_config = std::fs::read_to_string(lora_dir.join("adapter_config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());

    let rank = rank_override.unwrap_or_else(|| {
        // Prefer adapter_config.json, then fall back to shape detection
        if let Some(ref cfg) = adapter_config {
            if let Some(r) = cfg["r"].as_u64() {
                return r as usize;
            }
        }
        // Fallback: detect from attention lora_a shapes (skip rank-0 MLP weights)
        lora_weights
            .iter()
            .filter(|(k, _)| k.contains("self_attn") && k.contains("lora_a"))
            .map(|(_, v)| *v.shape().iter().min().unwrap_or(&16) as usize)
            .next()
            .unwrap_or(16)
    });

    let alpha = alpha_override.unwrap_or_else(|| {
        if let Some(ref cfg) = adapter_config {
            if let Some(a) = cfg["alpha"].as_f64() {
                tracing::info!("Loaded r={}, alpha={} from adapter_config.json", rank, a);
                return a as f32;
            }
        }
        rank as f32
    });
    let scale = alpha / rank as f32;
    println!("LoRA rank: {rank}, alpha: {alpha}, scale: {scale:.3}");

    // Apply LoRA: W_fused = W_base + scale * (B @ A)
    print!("Fusing weights... ");
    let mut fused_count = 0usize;

    // Group LoRA weights by layer: find matching (lora_a, lora_b) pairs
    let mut lora_a_map: HashMap<String, &mlx_rs::Array> = HashMap::new();
    let mut lora_b_map: HashMap<String, &mlx_rs::Array> = HashMap::new();

    for (name, array) in &lora_weights {
        if let Some(base_name) = name.strip_suffix(".lora_a") {
            lora_a_map.insert(base_name.to_string(), array);
        } else if let Some(base_name) = name.strip_suffix(".lora_b") {
            lora_b_map.insert(base_name.to_string(), array);
        }
    }

    for (layer_name, lora_a) in &lora_a_map {
        let Some(lora_b) = lora_b_map.get(layer_name) else {
            tracing::warn!("Missing lora_b for {layer_name}, skipping");
            continue;
        };

        // Map LoRA layer name to base weight name
        // LoRA names: "layers.0.self_attn.q_proj.lora_a" → base: "model.layers.0.self_attn.q_proj.weight"
        let base_key = if layer_name.starts_with("model.") {
            format!("{layer_name}.weight")
        } else {
            format!("model.{layer_name}.weight")
        };

        let Some(base_weight) = base_weights.get(&base_key) else {
            tracing::warn!("Base weight {base_key} not found, skipping");
            continue;
        };

        // Compute delta = scale * (B @ A) and add to base weight
        // lora_a: [r, in_features], lora_b: [out_features, r]
        // delta: [out_features, in_features]
        let base_dtype = base_weight.dtype();
        let delta = mlx_rs::ops::matmul(lora_b, lora_a)?;
        let scaled_delta = mlx_rs::ops::multiply(&delta, mlx_rs::array!(scale))?;
        let fused = mlx_rs::ops::add(base_weight, &scaled_delta)?;
        // Cast back to base dtype (LoRA is f32, base is typically bf16/f16)
        let fused = if fused.dtype() != base_dtype {
            fused.as_dtype(base_dtype)?
        } else {
            fused
        };

        base_weights.insert(base_key, fused);
        fused_count += 1;
    }
    println!("OK ({fused_count} layers fused)");

    // Create output directory
    let output_dir = Path::new(output_path);
    std::fs::create_dir_all(output_dir)?;

    // Copy non-weight files from base model (config.json, tokenizer, etc.)
    print!("Copying model files... ");
    let mut copied = 0usize;
    for entry in std::fs::read_dir(&model_dir)?.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Skip weight files (we'll save our own) and symlinks
        if name_str.ends_with(".safetensors")
            || name_str == "model.safetensors.index.json"
            || name_str.starts_with(".")
        {
            continue;
        }
        let dest = output_dir.join(&name);
        if entry.path().is_file() {
            std::fs::copy(entry.path(), &dest)?;
            copied += 1;
        }
    }
    println!("OK ({copied} files)");

    // Save fused weights as a single safetensors file
    print!("Saving fused model... ");
    let output_file = output_dir.join("model.safetensors");
    let metadata = std::collections::HashMap::from([("format".to_string(), "mlx".to_string())]);
    mlx_rs::Array::save_safetensors(&base_weights, Some(&metadata), &output_file)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Generate model.safetensors.index.json (required by LM Studio and other tools)
    let mut weight_map = serde_json::Map::new();
    let mut total_size: u64 = 0;
    for (key, arr) in &base_weights {
        weight_map.insert(
            key.clone(),
            serde_json::Value::String("model.safetensors".to_string()),
        );
        total_size += arr.nbytes() as u64;
    }
    let index = serde_json::json!({
        "metadata": { "total_size": total_size },
        "weight_map": weight_map,
    });
    std::fs::write(
        output_dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index)?,
    )?;

    let size = std::fs::metadata(&output_file)
        .map(|m| m.len())
        .unwrap_or(0);
    let gb = size as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("OK ({:.2} GB)", gb);

    println!("\n========================================");
    println!("Fused model saved to: {output_path}");
    println!("\nNext steps:");
    println!(
        "  Inference:  pmetal infer -m {} -p \"Your prompt\"",
        output_path
    );
    println!(
        "  Quantize:   pmetal quantize -m {} -o {}.gguf",
        output_path, output_path
    );
    println!(
        "  Ollama:     pmetal ollama create -n my-model -b {}",
        output_path
    );

    Ok(())
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
    use pmetal_merge::{AccurateMergeConfig, streaming_lora_merge};

    println!("  PMetal LoRA Fuse (f64-accurate path)");
    println!("========================================");

    // Resolve model path (could be HF ID or local path)
    let model_dir: PathBuf = if model_path.contains('/') && !PathBuf::from(model_path).exists() {
        tracing::info!("Resolving HuggingFace model: {}", model_path);
        pmetal_hub::download_model(model_path, None, None).await?
    } else {
        PathBuf::from(model_path)
    };

    // The adapter path must be a directory containing adapter_config.json.
    let adapter_dir: PathBuf = if std::path::Path::new(lora_path).is_dir() {
        PathBuf::from(lora_path)
    } else {
        // If given a file, use the parent directory.
        std::path::Path::new(lora_path)
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_path_buf()
    };

    println!("Base model:   {}", model_dir.display());
    println!("LoRA adapter: {}", adapter_dir.display());
    println!("Output:       {output_path}");
    if low_memory {
        println!("Mode:         low-memory (tiled, 512 rows/tile)");
    } else {
        println!("Mode:         standard (full-matrix f64)");
    }
    println!();

    let mut config = AccurateMergeConfig::new(&model_dir, &adapter_dir, PathBuf::from(output_path));
    if low_memory {
        config = config.with_low_memory(512);
    }

    print!("Merging... ");
    let stats = streaming_lora_merge(&config)
        .map_err(|e| anyhow::anyhow!("f64-accurate LoRA merge failed: {e}"))?;

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
