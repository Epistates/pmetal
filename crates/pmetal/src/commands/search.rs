use indicatif::ProgressBar;

/// Run HuggingFace Hub search with device fit analysis.
pub(crate) async fn run_search(
    query: &str,
    limit: usize,
    download: bool,
    detailed: bool,
    json_output: bool,
) -> anyhow::Result<()> {
    use pmetal_hub::{DeviceSpec, FitLevel, ModelSpec, estimate_fit, search_models};

    // Get device info for fit estimation
    let device_spec = match pmetal_metal::context::MetalContext::global() {
        Ok(ctx) => {
            let props = ctx.properties();
            Some(DeviceSpec {
                memory_gb: props.recommended_working_set_size as f64 / (1024.0 * 1024.0 * 1024.0),
                bandwidth_gbps: props.memory_bandwidth_gbps,
                unified_memory: props.has_unified_memory,
            })
        }
        Err(_) => None,
    };

    let bar = ProgressBar::new_spinner();
    bar.set_message(format!("Searching HuggingFace for '{query}'..."));
    bar.enable_steady_tick(std::time::Duration::from_millis(100));

    let results = search_models(query, limit, None).await?;
    bar.finish_and_clear();

    if results.is_empty() {
        if json_output {
            println!("[]");
        } else {
            println!("No models found for '{query}'.");
        }
        return Ok(());
    }

    // JSON output mode: emit structured data and return early
    if json_output {
        let json_results: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                let fit_info =
                    if let (Some(dev), Some(params_b)) = (&device_spec, r.estimated_params_b) {
                        let quant = pmetal_hub::fit::detect_quantization_from_id(&r.model_id);
                        let model_spec = pmetal_hub::ModelSpec {
                            params_b,
                            quantization: quant,
                            context_length: 4096,
                            num_kv_heads: None,
                            head_dim: None,
                            num_layers: None,
                            is_moe: r.tags.iter().any(|t| t == "moe"),
                            num_experts: None,
                            active_experts: None,
                            kv_cache_bits: None,
                        };
                        let fit = estimate_fit(&model_spec, dev);
                        serde_json::json!({
                            "level": fit.fit_level.label(),
                            "total_gb": fit.total_required_gb,
                            "weights_gb": fit.weights_gb,
                        })
                    } else {
                        serde_json::Value::Null
                    };

                serde_json::json!({
                    "model_id": r.model_id,
                    "downloads": r.downloads,
                    "likes": r.likes,
                    "tags": r.tags,
                    "estimated_params_b": r.estimated_params_b,
                    "fit": fit_info,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&json_results)?);
        return Ok(());
    }

    // Text output
    println!("\nSearch results for '{query}':");
    println!("{}", "=".repeat(60));

    let mut first_fit: Option<String> = None;

    for result in &results {
        let size_str = result
            .estimated_params_b
            .map(|b| {
                if b >= 1.0 {
                    format!("{:.1}B", b)
                } else {
                    format!("{:.0}M", b * 1000.0)
                }
            })
            .unwrap_or_else(|| "?".to_string());

        // Fit indicator
        let fit_str = if let (Some(dev), Some(params_b)) = (&device_spec, result.estimated_params_b)
        {
            let quant = pmetal_hub::fit::detect_quantization_from_id(&result.model_id);
            let model_spec = ModelSpec {
                params_b,
                quantization: quant,
                context_length: 4096,
                num_kv_heads: None,
                head_dim: None,
                num_layers: None,
                is_moe: result.tags.iter().any(|t| t == "moe"),
                num_experts: None,
                active_experts: None,
                kv_cache_bits: None,
            };
            let fit = estimate_fit(&model_spec, dev);
            if first_fit.is_none() && matches!(fit.fit_level, FitLevel::Fits | FitLevel::Tight) {
                first_fit = Some(result.model_id.clone());
            }
            match fit.fit_level {
                FitLevel::Fits => " [fits]".to_string(),
                FitLevel::Tight => " [tight]".to_string(),
                FitLevel::TooLarge => " [too large]".to_string(),
            }
        } else {
            String::new()
        };

        println!(
            "{} ({}, {}d, {}l{})",
            result.model_id, size_str, result.downloads, result.likes, fit_str
        );

        if detailed {
            if let (Some(dev), Some(params_b)) = (&device_spec, result.estimated_params_b) {
                let quant = pmetal_hub::fit::detect_quantization_from_id(&result.model_id);
                let model_spec = ModelSpec {
                    params_b,
                    quantization: quant,
                    context_length: 4096,
                    num_kv_heads: None,
                    head_dim: None,
                    num_layers: None,
                    is_moe: result.tags.iter().any(|t| t == "moe"),
                    num_experts: None,
                    active_experts: None,
                    kv_cache_bits: None,
                };
                let fit = estimate_fit(&model_spec, dev);
                println!(
                    "  Weights: {:.1}GB  KV: {:.1}GB  Overhead: {:.1}GB  Training: {:.1}GB ({})  Batch: {}",
                    fit.weights_gb,
                    fit.kv_cache_gb,
                    fit.overhead_gb,
                    fit.training_memory_gb,
                    fit.training_fit.label(),
                    fit.recommended_batch_size,
                );
                for note in &fit.notes {
                    println!("  {note}");
                }
            }
        }
    }

    // Auto-download first fitting model
    if download {
        if let Some(model_id) = first_fit {
            println!("\nDownloading {model_id}...");
            let path = pmetal_hub::download_model(&model_id, None, None).await?;
            println!("Downloaded to: {}", path.display());
        } else {
            println!("\nNo models fit on this device. Try a smaller model or quantized variant.");
        }
    }

    Ok(())
}

/// Truncate a string to max_len, appending ".." if truncated.
#[allow(dead_code)] // Used by TUI search display
pub(crate) fn truncate_str(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len.saturating_sub(2)).collect();
        format!("{truncated}..")
    }
}
