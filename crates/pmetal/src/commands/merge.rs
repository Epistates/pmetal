/// Merge two models using the pmetal-merge crate.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_merge_command(
    model_a: &str,
    model_b: &str,
    output: &str,
    method: &str,
    base: Option<&str>,
    t: f32,
    weight_a: f32,
    weight_b: f32,
    density: f32,
    dtype: &str,
) -> anyhow::Result<()> {
    use pmetal_merge::{
        MergeConfig, MergeMethodConfig, MergeParameters, ModelConfig as MergeModelConfig,
        ParameterSetting, TokenizerConfig,
    };

    println!("PMetal Model Merge");
    println!("==================");
    println!("Model A: {model_a}");
    println!("Model B: {model_b}");
    println!("Method:  {method}");
    println!("Output:  {output}");
    println!();

    // Resolve HuggingFace model IDs to local paths
    println!("Resolving model A...");
    let path_a = pmetal_hub::resolve_model_path(model_a, None, None).await?;

    println!("Resolving model B...");
    let path_b = pmetal_hub::resolve_model_path(model_b, None, None).await?;

    let base_path = if let Some(base_id) = base {
        println!("Resolving base model...");
        Some(
            pmetal_hub::resolve_model_path(base_id, None, None)
                .await?
                .to_string_lossy()
                .to_string(),
        )
    } else {
        None
    };

    let merge_method = match method.to_lowercase().as_str() {
        "linear" => MergeMethodConfig::Linear,
        "slerp" => MergeMethodConfig::Slerp,
        "task_arithmetic" | "task-arithmetic" => MergeMethodConfig::TaskArithmetic,
        "ties" => MergeMethodConfig::Ties,
        "dare_ties" | "dare-ties" => MergeMethodConfig::DareTies,
        "dare_linear" | "dare-linear" => MergeMethodConfig::DareLinear,
        "della" => MergeMethodConfig::Della,
        "della_linear" | "della-linear" => MergeMethodConfig::DellaLinear,
        "breadcrumbs" => MergeMethodConfig::Breadcrumbs,
        "model_stock" | "model-stock" => MergeMethodConfig::ModelStock,
        "consensus" => MergeMethodConfig::Consensus,
        "multislerp" | "multi_slerp" | "multi-slerp" => MergeMethodConfig::MultiSlerp,
        "nearswap" => MergeMethodConfig::Nearswap,
        "passthrough" => MergeMethodConfig::Passthrough,
        other => anyhow::bail!(
            "Unknown merge method '{}'. Valid options: linear, slerp, task_arithmetic, ties, \
             dare_ties, dare_linear, della, della_linear, breadcrumbs, model_stock, consensus, multislerp, nearswap, passthrough",
            other
        ),
    };

    let config = MergeConfig {
        merge_method,
        models: vec![
            MergeModelConfig {
                model: path_a.to_string_lossy().to_string(),
                parameters: MergeParameters {
                    weight: Some(ParameterSetting::Scalar(weight_a)),
                    density: Some(ParameterSetting::Scalar(density)),
                    t: Some(ParameterSetting::Scalar(t)),
                    ..Default::default()
                },
            },
            MergeModelConfig {
                model: path_b.to_string_lossy().to_string(),
                parameters: MergeParameters {
                    weight: Some(ParameterSetting::Scalar(weight_b)),
                    density: Some(ParameterSetting::Scalar(density)),
                    t: Some(ParameterSetting::Scalar(1.0 - t)),
                    ..Default::default()
                },
            },
        ],
        base_model: base_path,
        output_path: Some(std::path::PathBuf::from(output)),
        dtype: dtype.to_string(),
        parameters: MergeParameters::default(),
        tokenizer: Some(TokenizerConfig {
            source: "first".to_string(),
        }),
        slices: None,
        allow_mixed_dtype: false,
        sanity: pmetal_merge::SanityLevel::default(),
        dry_run: false,
        ..Default::default()
    };

    println!("Running merge...");
    let result_path =
        pmetal_merge::run_merge(&config).map_err(|e| anyhow::anyhow!("Merge failed: {}", e))?;

    println!("\nMerge complete!");
    println!("Output: {}", result_path.display());
    println!("\nNext steps:");
    println!(
        "  pmetal infer -m {} -p \"Your prompt\"",
        result_path.display()
    );
    Ok(())
}
