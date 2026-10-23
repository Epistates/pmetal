/// The merge method a name stands for: one of `MergeSpec`'s `method` options
/// (the one list every surface offers), spelled with underscores or hyphens.
fn parse_method(name: &str) -> anyhow::Result<pmetal_merge::MergeMethodConfig> {
    use pmetal_merge::MergeMethodConfig;
    let name = name.trim().to_lowercase().replace('-', "_");
    match name.as_str() {
        // The library's name for it has an inner underscore.
        "multislerp" | "multi_slerp" => Ok(MergeMethodConfig::MultiSlerp),
        _ => serde_json::from_value(serde_json::Value::String(name.clone())).map_err(|_| {
            anyhow::anyhow!(
                "Unknown merge method '{name}'. Valid options: {}",
                merge_method_names().join(", ")
            )
        }),
    }
}

/// The method names `MergeSpec` offers.
fn merge_method_names() -> Vec<&'static str> {
    use pmetal_core::{FieldKind, JobFields};
    pmetal_core::jobs::MergeSpec::field_descriptors()
        .iter()
        .find_map(|d| match d.kind {
            FieldKind::Enum { options } if d.name == "method" => Some(options.to_vec()),
            _ => None,
        })
        .unwrap_or_default()
}

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
        MergeConfig, MergeParameters, ModelConfig as MergeModelConfig, ParameterSetting,
        TokenizerConfig,
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

    let merge_method = parse_method(method)?;

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every method the spec offers is one the command runs, and the base
    /// model rule the spec checks is the library's own.
    #[test]
    fn every_offered_method_is_a_method_the_command_runs() {
        let names = merge_method_names();
        assert_eq!(names.len(), 16);
        for name in &names {
            let config = parse_method(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            let library_needs_base =
                pmetal_merge::create_merge_method(&config).requires_base_model();
            assert_eq!(
                library_needs_base,
                !pmetal_core::jobs::merge::METHODS_WITHOUT_BASE.contains(name),
                "{name}"
            );
        }
    }

    #[test]
    fn a_method_is_spelled_with_underscores_or_hyphens_and_any_case() {
        for (typed, wanted) in [
            ("dare-ties", "dare_ties"),
            ("Task_Arithmetic", "task_arithmetic"),
            ("multi-slerp", "multislerp"),
            ("ram-plus", "ram_plus"),
        ] {
            let got = parse_method(typed).unwrap();
            assert_eq!(
                format!("{got:?}"),
                format!("{:?}", parse_method(wanted).unwrap()),
                "{typed}"
            );
        }
        let err = parse_method("fisher").unwrap_err().to_string();
        assert!(
            err.contains("Unknown merge method 'fisher'") && err.contains("ram_plus"),
            "{err}"
        );
    }
}
