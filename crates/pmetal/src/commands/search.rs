use indicatif::ProgressBar;

/// Context length the search results are sized for.
const FIT_CONTEXT: u32 = 4096;

/// How search results are ordered.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SortArg {
    /// Most downloaded in the last 30 days
    Downloads,
    /// Most liked
    Likes,
    /// Trending
    Trending,
    /// Most recently updated
    Updated,
}

impl From<SortArg> for pmetal_hub::search::SearchSort {
    fn from(arg: SortArg) -> Self {
        match arg {
            SortArg::Downloads => Self::Downloads,
            SortArg::Likes => Self::Likes,
            SortArg::Trending => Self::Trending,
            SortArg::Updated => Self::Updated,
        }
    }
}

/// What `pmetal search` was asked for.
pub(crate) struct SearchRequest {
    pub(crate) query: String,
    pub(crate) limit: usize,
    pub(crate) download: bool,
    pub(crate) detailed: bool,
    pub(crate) json: bool,
    pub(crate) tags: Vec<String>,
    pub(crate) author: Option<String>,
    pub(crate) pipeline_tag: String,
    pub(crate) sort: SortArg,
}

/// Run HuggingFace Hub search with device fit analysis.
pub(crate) async fn run_search(request: SearchRequest) -> anyhow::Result<()> {
    use pmetal_hub::search::{SearchOptions, search_models_with};
    use pmetal_hub::{DeviceSpec, FitLevel};

    let SearchRequest {
        query,
        limit,
        download,
        detailed,
        json: json_output,
        tags,
        author,
        pipeline_tag,
        sort,
    } = request;
    let query = query.as_str();

    // The device's recommended working set, asked of the device: not a share
    // of installed RAM.
    let device_spec = match pmetal_metal::context::MetalContext::global() {
        Ok(ctx) => {
            let props = ctx.properties();
            Some(DeviceSpec::from_device(
                props.recommended_working_set_size,
                props.memory_bandwidth_gbps,
                props.has_unified_memory,
            ))
        }
        Err(_) => None,
    };

    let bar = ProgressBar::new_spinner();
    bar.set_message(format!("Searching HuggingFace for '{query}'..."));
    bar.enable_steady_tick(std::time::Duration::from_millis(100));

    let mut options = SearchOptions::new(query);
    options.limit = limit;
    options.tags = tags;
    options.author = author;
    options.pipeline_tag = (pipeline_tag != "any").then_some(pipeline_tag);
    options.sort = sort.into();
    let results = search_models_with(&options).await?;
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
                let fit_info = device_spec
                    .as_ref()
                    .and_then(|dev| r.fit(dev, FIT_CONTEXT))
                    .map(|fit| {
                        serde_json::json!({
                            "level": fit.fit_level.label(),
                            "total_gb": fit.total_required_gb,
                            "weights_gb": fit.weights_gb,
                            "weights_measured": fit.weights_measured,
                            "context_length": fit.context_length,
                            "estimated_tps": fit.estimated_tps,
                        })
                    })
                    .unwrap_or(serde_json::Value::Null);

                serde_json::json!({
                    "model_id": r.model_id,
                    "downloads": r.downloads,
                    "likes": r.likes,
                    "tags": r.tags,
                    "license": r.license,
                    "gated": r.gated,
                    "estimated_params_b": r.estimated_params_b,
                    "safetensors_bytes": r.safetensors_bytes,
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
            .map(pmetal_hub::fit::format_params)
            .unwrap_or_else(|| "?".to_string());

        // Fit indicator
        let fit = device_spec
            .as_ref()
            .and_then(|dev| result.fit(dev, FIT_CONTEXT));
        let fit_str = match &fit {
            Some(fit) => {
                if first_fit.is_none() && matches!(fit.fit_level, FitLevel::Fits | FitLevel::Tight)
                {
                    first_fit = Some(result.model_id.clone());
                }
                match fit.fit_level {
                    FitLevel::Fits => " [fits]".to_string(),
                    FitLevel::Tight => " [tight]".to_string(),
                    FitLevel::TooLarge => " [too large]".to_string(),
                }
            }
            None => String::new(),
        };
        let gated = if result.gated { " [gated]" } else { "" };

        println!(
            "{} ({}, {}d, {}l{}{})",
            result.model_id, size_str, result.downloads, result.likes, fit_str, gated
        );

        if detailed {
            if let Some(fit) = &fit {
                println!(
                    "  Weights: {:.1}GB{}  KV: {:.1}GB ({} tokens)  Overhead: {:.1}GB  Training: {:.1}GB ({})  Batch: {}",
                    fit.weights_gb,
                    if fit.weights_measured { "" } else { " (est.)" },
                    fit.kv_cache_gb,
                    fit.context_length,
                    fit.overhead_gb,
                    fit.training_memory_gb,
                    fit.training_fit.label(),
                    fit.recommended_batch_size,
                );
                println!(
                    "  Decode: ~{:.0} tok/s (estimated from memory bandwidth)",
                    fit.estimated_tps
                );
                for note in &fit.notes {
                    println!("  {note}");
                }
            }
            if let Some(license) = &result.license {
                println!("  License: {license}");
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
