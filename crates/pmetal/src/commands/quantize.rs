use std::path::{Path, PathBuf};

/// Run model quantization.
pub(crate) async fn run_quantization(
    model_path: &str,
    output_path: &str,
    imatrix_path: Option<&str>,
    method: crate::QuantizeMethod,
    kl_calibrate: bool,
    target_bpw: Option<f32>,
    kl_threshold: f64,
) -> anyhow::Result<()> {
    use pmetal_gguf::{
        GgufBuilder,
        dynamic::{
            CalibrationMap, DynamicQuantizationConfig, DynamicQuantizer, KlCalibrationConfig,
        },
        imatrix::IMatrix,
        quantize::quantize,
    };

    println!("========================================");
    println!("  PMetal GGUF Quantization");
    println!("========================================");
    println!("Model:    {}", model_path);
    println!("Output:   {}", output_path);
    println!("Method:   {}", method.as_str());
    if let Some(imp) = imatrix_path {
        println!("IMatrix:  {}", imp);
    }
    if kl_calibrate {
        println!("KL Calib: enabled (threshold={:.4})", kl_threshold);
        if let Some(bpw) = target_bpw {
            println!("Target BPW: {:.2}", bpw);
        }
    }
    println!("========================================\n");

    // Resolve HuggingFace model ID to local path
    let resolved_model_path: PathBuf =
        if model_path.contains('/') && !PathBuf::from(model_path).exists() {
            tracing::info!("Resolving HuggingFace model: {}", model_path);
            pmetal_hub::download_model(model_path, None, None).await?
        } else {
            PathBuf::from(model_path)
        };

    // 1. Load IMatrix if provided
    let imatrix = if let Some(path) = imatrix_path {
        tracing::info!("Loading IMatrix from {}", path);
        Some(IMatrix::load(Path::new(path))?)
    } else {
        None
    };

    // 2. Initialize quantizer
    let quantizer = if let Some(base_type) = method.to_ggml_type() {
        let config = DynamicQuantizationConfig {
            base_type,
            high_precision_type: base_type,
            fallback_type: base_type,
            ..Default::default()
        };
        DynamicQuantizer::new(config, None)
    } else {
        let config = DynamicQuantizationConfig::default();
        DynamicQuantizer::new(config, imatrix)
    };

    // 3. Load Model Weights
    tracing::info!("Scanning model weights from {:?}...", resolved_model_path);
    let weights = pmetal_models::loader::load_weights(&resolved_model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?;
    tracing::info!("Loaded {} tensors", weights.len());

    // 4. Detect Architecture
    let config_path = resolved_model_path.join("config.json");
    let mut architecture = "llama".to_string();

    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(archs) = json.get("architectures").and_then(|v| v.as_array()) {
                    if let Some(arch_str) = archs.first().and_then(|v| v.as_str()) {
                        architecture = match arch_str {
                            "LlamaForCausalLM" => "llama".to_string(),
                            "MistralForCausalLM" => "mistral".to_string(),
                            "Qwen2ForCausalLM" => "qwen2".to_string(),
                            "GemmaForCausalLM" | "Gemma2ForCausalLM" => "gemma".to_string(),
                            "PhiForCausalLM" | "Phi3ForCausalLM" => "phi".to_string(),
                            _ => {
                                tracing::warn!(
                                    "Unknown architecture '{}', defaulting to 'llama'",
                                    arch_str
                                );
                                "llama".to_string()
                            }
                        };
                        tracing::info!(
                            "Detected architecture: {} (from {})",
                            architecture,
                            arch_str
                        );
                    }
                }
            }
        }
    } else {
        tracing::warn!("config.json not found, defaulting architecture to 'llama'");
    }

    // 5. KL-divergence calibration pass (optional)
    let mut float_cache: std::collections::HashMap<String, (Vec<f32>, Vec<i32>)> =
        std::collections::HashMap::new();
    let calibration_map: CalibrationMap;

    if kl_calibrate {
        println!(
            "Running KL calibration pass over {} tensors...",
            weights.len()
        );

        let mut tensor_data: Vec<(String, Vec<f32>, Vec<i32>)> = Vec::new();
        let mut sorted_names: Vec<_> = weights.keys().cloned().collect();
        sorted_names.sort();

        for name in &sorted_names {
            let tensor = weights
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Tensor {} missing after key listing", name))?;

            tensor
                .eval()
                .map_err(|e| anyhow::anyhow!("MLX eval error for {}: {}", name, e))?;

            let data_f32: Vec<f32> = match tensor.dtype() {
                pmetal_mlx::Dtype::Float32 => tensor.as_slice::<f32>().to_vec(),
                pmetal_mlx::Dtype::Float16 | pmetal_mlx::Dtype::Bfloat16 => {
                    let t_f32 = tensor.as_dtype(pmetal_mlx::Dtype::Float32).map_err(|e| {
                        anyhow::anyhow!("Dtype conversion error for {}: {}", name, e)
                    })?;
                    t_f32
                        .eval()
                        .map_err(|e| anyhow::anyhow!("MLX eval error for {}: {}", name, e))?;
                    t_f32.as_slice::<f32>().to_vec()
                }
                _ => continue,
            };

            let shape_i32: Vec<i32> = tensor.shape().iter().map(|&d| d as i32).collect();
            float_cache.insert(name.clone(), (data_f32.clone(), shape_i32.clone()));
            tensor_data.push((name.clone(), data_f32, shape_i32));
        }

        let kl_config = KlCalibrationConfig {
            kl_threshold,
            target_bpw,
            ..Default::default()
        };

        calibration_map = quantizer.calibrate_all(&tensor_data, &kl_config);

        let tensor_sizes: Vec<(String, usize)> = tensor_data
            .iter()
            .map(|(n, d, _)| (n.clone(), d.len()))
            .collect();
        let summary = quantizer.summarize_calibration(&calibration_map, &tensor_sizes);
        println!(
            "Calibration complete: {} tensors, avg KL={:.6}, worst={} ({:.6}), est. BPW={:.2}",
            summary.total_tensors,
            summary.avg_kl_score,
            summary.worst_tensor,
            summary.max_kl_score,
            summary.estimated_bpw,
        );
        let mut type_vec: Vec<_> = summary.type_counts.iter().collect();
        type_vec.sort_by_key(|(t, _)| format!("{:?}", t));
        for (dtype, count) in type_vec {
            println!("  {:?}: {} tensors", dtype, count);
        }
        println!();
    } else {
        calibration_map = CalibrationMap::new();
    }

    // 6. Initialize GGUF Builder
    let mut builder = GgufBuilder::with_model(&architecture, "quantized-model");

    // 7. Quantize and Write
    tracing::info!("Starting quantization...");

    let mut keys: Vec<_> = weights.keys().collect();
    keys.sort();

    for name in keys {
        let shape_u64: Vec<u64>;
        let data_f32: Vec<f32>;

        if let Some((cached_data, cached_shape)) = float_cache.remove(name) {
            shape_u64 = cached_shape.iter().map(|&d| d as u64).collect();
            data_f32 = cached_data;
        } else {
            let tensor = weights
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Tensor {} not found in loaded weights", name))?;

            let shape = tensor.shape();
            shape_u64 = shape.iter().map(|&d| d as u64).collect();

            tensor
                .eval()
                .map_err(|e| anyhow::anyhow!("MLX eval error: {}", e))?;

            data_f32 = match tensor.dtype() {
                pmetal_mlx::Dtype::Float32 => tensor.as_slice::<f32>().to_vec(),
                pmetal_mlx::Dtype::Float16 | pmetal_mlx::Dtype::Bfloat16 => {
                    let t_f32 = tensor
                        .as_dtype(pmetal_mlx::Dtype::Float32)
                        .map_err(|e| anyhow::anyhow!("Dtype conversion error: {}", e))?;
                    t_f32
                        .eval()
                        .map_err(|e| anyhow::anyhow!("MLX eval error: {}", e))?;
                    t_f32.as_slice::<f32>().to_vec()
                }
                _ => {
                    tracing::warn!("Skipping non-float tensor: {}", name);
                    continue;
                }
            };
        }

        let target_type = if calibration_map.is_empty() {
            quantizer.get_tensor_type(name, &shape_u64)
        } else {
            quantizer.get_tensor_type_calibrated(name, &shape_u64, &calibration_map)
        };

        tracing::info!("Quantizing {} to {:?}", name, target_type);
        let quantized_data = quantize(&data_f32, target_type)
            .map_err(|e| anyhow::anyhow!("Quantization error for {}: {:?}", name, e))?;

        builder.add_raw_tensor(name, shape_u64, target_type, quantized_data);
    }

    // 8. Write GGUF output
    let validated_output = crate::validate_output_path(output_path, "quantization output")?;
    let mut file = std::fs::File::create(&validated_output)?;
    builder.write(&mut file)?;

    println!("Quantization complete!");
    Ok(())
}
