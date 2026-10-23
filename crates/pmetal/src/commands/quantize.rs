use crate::commands::gguf_export::{
    ExportArch, Recipe, add_hf_config_metadata, add_tokenizer_metadata, model_name,
};
use crate::commands::quantize_source::TensorMeta;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ExpertProjection {
    Gate,
    Up,
    Down,
}

impl ExpertProjection {
    fn gguf_suffix(self) -> &'static str {
        match self {
            Self::Gate => "ffn_gate_exps.weight",
            Self::Up => "ffn_up_exps.weight",
            Self::Down => "ffn_down_exps.weight",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ExpertGroupKey {
    layer: usize,
    projection: ExpertProjection,
}

#[derive(Debug, Clone)]
struct ExpertTensor {
    expert: usize,
    name: String,
}

fn parse_expert_tensor_name(name: &str) -> Option<(ExpertGroupKey, usize)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer_str, suffix) = rest.split_once('.')?;
    let layer = layer_str.parse::<usize>().ok()?;

    let patterns = [
        (
            "mlp.experts.",
            [
                ("gate_proj.weight", ExpertProjection::Gate),
                ("up_proj.weight", ExpertProjection::Up),
                ("down_proj.weight", ExpertProjection::Down),
            ],
        ),
        (
            "block_sparse_moe.experts.",
            [
                ("w1.weight", ExpertProjection::Gate),
                ("w3.weight", ExpertProjection::Up),
                ("w2.weight", ExpertProjection::Down),
            ],
        ),
    ];

    for (prefix, projections) in patterns {
        let Some(expert_rest) = suffix.strip_prefix(prefix) else {
            continue;
        };
        let (expert_str, projection_suffix) = expert_rest.split_once('.')?;
        let expert = expert_str.parse::<usize>().ok()?;
        for (candidate, projection) in projections {
            if projection_suffix == candidate {
                return Some((ExpertGroupKey { layer, projection }, expert));
            }
        }
    }

    None
}

fn collect_expert_groups<'a>(
    names: impl Iterator<Item = &'a str>,
) -> HashMap<ExpertGroupKey, Vec<ExpertTensor>> {
    let mut groups: HashMap<ExpertGroupKey, Vec<ExpertTensor>> = HashMap::new();
    for name in names {
        if let Some((key, expert)) = parse_expert_tensor_name(name) {
            groups.entry(key).or_default().push(ExpertTensor {
                expert,
                name: name.to_string(),
            });
        }
    }
    groups.retain(|_, tensors| tensors.len() > 1);
    groups
}

/// The experts of a group in expert order, and the shape of their stack
/// `[experts, out, in]`; the experts must be the run `0..n` and agree in shape.
fn expert_stack_layout(
    group: &[ExpertTensor],
    shapes: &HashMap<&str, &Vec<usize>>,
) -> anyhow::Result<(Vec<ExpertTensor>, Vec<u64>)> {
    let mut sorted = group.to_vec();
    sorted.sort_by_key(|tensor| tensor.expert);
    for (expected, tensor) in sorted.iter().enumerate() {
        if tensor.expert != expected {
            anyhow::bail!(
                "Expert tensors for {} are not contiguous: expected expert {}, found {}",
                tensor.name,
                expected,
                tensor.expert
            );
        }
    }
    let shape_of = |tensor: &ExpertTensor| {
        shapes
            .get(tensor.name.as_str())
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Expert tensor {} is not a float tensor", tensor.name))
    };
    let base_shape = shape_of(&sorted[0])?;
    if base_shape.len() != 2 {
        anyhow::bail!(
            "Expert tensor {} must be 2D for GGUF expert stacking, got {:?}",
            sorted[0].name,
            base_shape
        );
    }
    for tensor in &sorted {
        let shape = shape_of(tensor)?;
        if shape != base_shape {
            anyhow::bail!(
                "Expert tensor {} shape {:?} does not match {:?}",
                tensor.name,
                shape,
                base_shape
            );
        }
    }
    let mut shape = Vec::with_capacity(base_shape.len() + 1);
    shape.push(sorted.len() as u64);
    shape.extend(base_shape.iter().map(|&dim| dim as u64));
    Ok((sorted, shape))
}

/// Run `pmetal quantize`: GGUF (the default) or MLX safetensors, from the base
/// model, or from the base model with a LoRA adapter fused in first.
pub(crate) async fn run_quantize(spec: pmetal_core::jobs::QuantizeSpec) -> anyhow::Result<()> {
    use clap::ValueEnum;

    if spec.format == "mlx" {
        // The MLX path takes no adapter: fuse it first with `pmetal fuse`.
        anyhow::ensure!(
            spec.lora.is_none(),
            "--lora is not supported with --format mlx; fuse the adapter first with `pmetal fuse`, \
             then quantize the fused model"
        );
        anyhow::ensure!(
            spec.imatrix.is_none() && spec.calibration_dataset.is_none(),
            "--imatrix and --calibration-dataset weight the GGUF quantizers and have no effect \
             on --format mlx output"
        );
        return run_quantization_mlx(
            &spec.model,
            &spec.output,
            spec.bits,
            spec.group_size,
            spec.target_bpw,
        )
        .await;
    }

    let method = crate::cli::quantize::QuantizeMethod::from_str(&spec.method, true)
        .map_err(|e| anyhow::anyhow!("--method: {e}"))?;
    let importance = if let Some(dataset) = spec.calibration_dataset.as_deref() {
        ImportanceSource::Measure {
            dataset,
            ctx_size: spec.calibration_ctx_size,
            chunks: spec.calibration_chunks,
        }
    } else if let Some(path) = spec.imatrix.as_deref() {
        ImportanceSource::File(path)
    } else {
        ImportanceSource::None
    };
    let Some(lora) = spec.lora.as_deref() else {
        return run_quantization(
            &spec.model,
            &spec.output,
            importance,
            method,
            spec.kl_calibrate,
            spec.target_bpw,
            spec.kl_threshold,
        )
        .await;
    };

    // Fuse the adapter into a scratch copy beside the output, quantize that,
    // and remove it however the run ends: it is as large as the model.
    #[cfg(not(feature = "lora"))]
    {
        let _ = lora;
        anyhow::bail!("fusing a LoRA adapter while quantizing needs the `lora` feature");
    }
    #[cfg(feature = "lora")]
    {
        let output = crate::validate_output_path(&spec.output, "quantization output")?;
        let scratch_parent = output
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let scratch = tempfile::Builder::new()
            .prefix(".pmetal-fused-")
            .tempdir_in(&scratch_parent)
            .map_err(|e| {
                anyhow::anyhow!(
                    "could not make a scratch directory in '{}': {e}",
                    scratch_parent.display()
                )
            })?;
        let fused = scratch.path().join("model");
        crate::commands::fuse::run_fuse(pmetal_core::jobs::FuseSpec {
            model: spec.model.clone(),
            lora: lora.to_string(),
            output: fused.to_string_lossy().into_owned(),
            ..Default::default()
        })
        .await?;
        run_quantization(
            &fused.to_string_lossy(),
            &spec.output,
            importance,
            method,
            spec.kl_calibrate,
            spec.target_bpw,
            spec.kl_threshold,
        )
        .await
    }
}

/// Where the quantization's per-channel importance comes from.
#[derive(Clone, Copy)]
pub(crate) enum ImportanceSource<'a> {
    /// None: every weight counts the same.
    None,
    /// An importance matrix file, measured earlier (`pmetal imatrix`).
    File(&'a str),
    /// Measured now, on this calibration data, from the model being quantized.
    Measure {
        dataset: &'a str,
        ctx_size: usize,
        /// 0: all the data makes.
        chunks: usize,
    },
}

/// Run model quantization.
pub(crate) async fn run_quantization(
    model_path: &str,
    output_path: &str,
    importance: ImportanceSource<'_>,
    method: crate::cli::quantize::QuantizeMethod,
    kl_calibrate: bool,
    target_bpw: Option<f32>,
    kl_threshold: f64,
) -> anyhow::Result<()> {
    use pmetal_gguf::{
        GgufBuilder, TensorInfo,
        dynamic::{
            CalibrationMap, DynamicQuantizationConfig, DynamicQuantizer, KlCalibrationConfig,
        },
        imatrix::IMatrix,
        quantize::{fit_to_row_length, quantize_tensor},
    };

    println!("========================================");
    println!("  PMetal GGUF Quantization");
    println!("========================================");
    println!("Model:    {}", model_path);
    println!("Output:   {}", output_path);
    println!("Method:   {}", method.as_str());
    match importance {
        ImportanceSource::None => {}
        ImportanceSource::File(path) => println!("IMatrix:  {path}"),
        ImportanceSource::Measure { dataset, .. } => {
            println!("IMatrix:  measured on {dataset}")
        }
    }
    if kl_calibrate {
        println!("KL Calib: enabled (threshold={:.4})", kl_threshold);
        if let Some(bpw) = target_bpw {
            println!("Target BPW: {:.2}", bpw);
        }
    }
    println!("========================================\n");

    // Resolve HuggingFace model ID to local path
    tracing::info!("Resolving model: {}", model_path);
    let resolved_model_path: PathBuf =
        pmetal_hub::resolve_model_path(model_path, None, None).await?;

    // 0. The architecture the file will name. Checked first: a model that
    // cannot be exported should not be measured, or read, before it is refused.
    let config_json: Option<serde_json::Value> =
        std::fs::read_to_string(resolved_model_path.join("config.json"))
            .ok()
            .map(|text| serde_json::from_str(&text))
            .transpose()
            .map_err(|e| anyhow::anyhow!("config.json is not valid JSON: {e}"))?;
    let arch = ExportArch::detect(config_json.as_ref())?;
    tracing::info!("GGUF architecture: {}", arch.name);

    // 1. The importance matrix: read, or measured on the model being quantized
    // (which is loaded for that and dropped before any weight is quantized).
    let imatrix = match importance {
        ImportanceSource::None => None,
        ImportanceSource::File(path) => {
            tracing::info!("Loading IMatrix from {}", path);
            Some(IMatrix::load(Path::new(path))?)
        }
        ImportanceSource::Measure {
            dataset,
            ctx_size,
            chunks,
        } => {
            println!("Measuring the importance matrix...");
            let run = crate::commands::imatrix::measure_imatrix(
                &resolved_model_path,
                model_path,
                dataset,
                ctx_size,
                (chunks > 0).then_some(chunks),
                pmetal_core::jobs::ImatrixSpec::default().batch_size,
            )
            .await?;
            println!(
                "Measured {} matrices over {} chunks ({} tokens)\n",
                run.imatrix.len(),
                run.chunks,
                run.tokens
            );
            Some(run.imatrix)
        }
    };

    // 2. Initialize quantizer
    let has_imatrix = imatrix.is_some();
    let quantizer = if let Some(base_type) = method.to_ggml_type() {
        let config = DynamicQuantizationConfig {
            base_type,
            high_precision_type: base_type,
            fallback_type: base_type,
            ..Default::default()
        };
        // The imatrix still weights the quantization error of a fixed-type
        // method; every tier is the same type, so it does not change the type.
        DynamicQuantizer::new(config, imatrix)
    } else {
        let config = DynamicQuantizationConfig::default();
        DynamicQuantizer::new(config, imatrix)
    };

    // 3. Open the weights. They are read one tensor at a time, here and below:
    // the model is never held in memory (only one tensor, its f32 copy and its
    // quantized bytes are, and the experts of one layer's projection).
    let validated_output = crate::validate_output_path(output_path, "quantization output")?;
    let mut source = crate::commands::quantize_source::open(&resolved_model_path)?;
    let metas: Vec<TensorMeta> = source.tensors().to_vec();
    tracing::info!("Found {} float tensors", metas.len());

    // 5. Plan: which tensors the file holds and in what order, under the names
    // it gives them (the names the imatrix and the calibration are keyed by).
    // The routed experts of a layer's projection are one stacked tensor, which
    // is how they are scored and weighted below as well as how they are
    // written.
    tracing::info!("Planning the quantization...");
    let entries = plan_entries(&metas, &arch)?;
    // A tensor's values as the file keeps them: in the layout the architecture
    // stores them in, which is not always the one the checkpoint publishes.
    let read_entry = |source: &mut dyn crate::commands::quantize_source::TensorSource,
                      from: &Origin,
                      shape: &[u64]|
     -> anyhow::Result<Vec<f32>> {
        match from {
            Origin::Tensor(name) => {
                let mut values = source.read_f32(name)?;
                arch.lay_out(name, shape, &mut values);
                Ok(values)
            }
            Origin::Experts(experts) => {
                let mut stacked = Vec::new();
                for expert in experts {
                    stacked.append(&mut source.read_f32(&expert.name)?);
                }
                Ok(stacked)
            }
            Origin::Computed(values) => Ok(values.clone()),
        }
    };

    // 6. KL-divergence calibration pass (optional): each tensor is read, scored
    // and let go of; only the choices are kept.
    let calibration_map: CalibrationMap = if kl_calibrate {
        println!(
            "Running KL calibration pass over {} tensors...",
            entries.len()
        );
        let listing: Vec<(String, Vec<i32>)> = entries
            .iter()
            .filter(|e| !matches!(e.from, Origin::Computed(_)))
            .map(|e| {
                (
                    e.gguf_name.clone(),
                    e.shape.iter().map(|&d| d as i32).collect(),
                )
            })
            .collect();
        let by_name: HashMap<&str, (&Origin, &[u64])> = entries
            .iter()
            .map(|e| (e.gguf_name.as_str(), (&e.from, e.shape.as_slice())))
            .collect();
        let kl_config = KlCalibrationConfig {
            kl_threshold,
            target_bpw,
            ..Default::default()
        };
        let map = quantizer.calibrate_streaming(
            &listing,
            |gguf_name| {
                let (from, shape) = by_name[gguf_name];
                read_entry(&mut *source, from, shape)
            },
            &kl_config,
        )?;

        let tensor_sizes: Vec<(String, usize)> = entries
            .iter()
            .map(|e| {
                (
                    e.gguf_name.clone(),
                    e.shape.iter().product::<u64>() as usize,
                )
            })
            .collect();
        let summary = quantizer.summarize_calibration(&map, &tensor_sizes);
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
        map
    } else {
        CalibrationMap::new()
    };

    // 7. Metadata
    let mut builder = GgufBuilder::with_model(
        arch.name,
        model_name(model_path, &resolved_model_path).as_str(),
    );
    // The model's own configuration: the language model's, in a multimodal
    // checkpoint. The tokenizer is found from the checkpoint's.
    let model_config = arch.model_config(config_json.as_ref());
    add_hf_config_metadata(&mut builder, &arch, model_config.as_ref(), method);
    add_tokenizer_metadata(&mut builder, &resolved_model_path, config_json.as_ref())?;

    // Each tensor's type. The header of a GGUF file comes first and names every
    // tensor's type and size, which depend on the shapes alone.
    let recipe = Recipe::new(
        method,
        &arch,
        model_config.as_ref(),
        has_imatrix,
        entries.iter().map(|e| e.gguf_name.as_str()),
    );
    let choose_type = |gguf_name: &str, shape_u64: &[u64]| {
        let proposed = if calibration_map.is_empty() {
            quantizer.get_tensor_type(gguf_name, shape_u64)
        } else {
            quantizer.get_tensor_type_calibrated(gguf_name, shape_u64, &calibration_map)
        };
        let chosen = recipe.tensor_type(gguf_name, shape_u64, proposed);
        // A block never straddles a row: fall back to a type whose block
        // divides the row, as the reference does.
        let n_per_row = shape_u64.last().copied().unwrap_or(1) as usize;
        fit_to_row_length(chosen, n_per_row)
    };
    let plan: Vec<PlannedTensor> = entries
        .into_iter()
        .map(|e| PlannedTensor {
            dtype: choose_type(&e.gguf_name, &e.shape),
            gguf_name: e.gguf_name,
            shape: e.shape,
            from: e.from,
        })
        .collect();

    // 8. Quantize and write, one tensor at a time. The file is built beside the
    // destination and moved into place when complete.
    let infos: Vec<TensorInfo> = plan
        .iter()
        .map(|t| TensorInfo::new(t.gguf_name.clone(), t.shape.clone(), t.dtype))
        .collect();
    let mut writer = builder
        .file_writer(&validated_output, infos)
        .map_err(|e| anyhow::anyhow!("could not start '{}': {e}", validated_output.display()))?;
    let total = plan.len();
    for (done, planned) in plan.iter().enumerate() {
        let mut data_f32 = read_entry(&mut *source, &planned.from, &planned.shape)?;
        let n_per_row = planned.shape.last().copied().unwrap_or(1) as usize;
        if planned.dtype == pmetal_gguf::GgmlType::F16 {
            clamp_to_f16(&mut data_f32, &planned.gguf_name);
        }
        match &planned.from {
            Origin::Computed(_) => tracing::info!(
                "[{}/{total}] Writing {} to {:?}",
                done + 1,
                planned.gguf_name,
                planned.dtype
            ),
            Origin::Tensor(name) => tracing::info!(
                "[{}/{total}] Quantizing {} ({}) to {:?}",
                done + 1,
                name,
                planned.gguf_name,
                planned.dtype
            ),
            Origin::Experts(experts) => tracing::info!(
                "[{}/{total}] Quantizing stacked expert tensor {} ({} experts) to {:?}",
                done + 1,
                planned.gguf_name,
                experts.len(),
                planned.dtype
            ),
        }
        let importance = quantizer.importance_for(&planned.gguf_name);
        let quantized = quantize_tensor(&data_f32, planned.dtype, n_per_row, importance.as_deref())
            .map_err(|e| {
                anyhow::anyhow!("Quantization error for {}: {:?}", planned.gguf_name, e)
            })?;
        drop(data_f32);
        match &planned.from {
            Origin::Tensor(name) => source.done_with(name),
            Origin::Experts(experts) => {
                for expert in experts {
                    source.done_with(&expert.name);
                }
            }
            Origin::Computed(_) => {}
        }
        writer
            .write_tensor(&planned.gguf_name, &quantized)
            .map_err(|e| anyhow::anyhow!("could not write {}: {e}", planned.gguf_name))?;
    }
    writer
        .finish()
        .map_err(|e| anyhow::anyhow!("could not finish '{}': {e}", validated_output.display()))?;

    println!("Quantization complete!");
    Ok(())
}

/// The tensors of the output file, in file order, under the names the file
/// gives them. The per-expert tensors of a layer's projection are one stacked
/// `[experts, out, in]` tensor, listed first in layer order; everything else
/// follows in the order of `metas`.
fn plan_entries(metas: &[TensorMeta], arch: &ExportArch) -> anyhow::Result<Vec<PlanEntry>> {
    let shapes: HashMap<&str, &Vec<usize>> = metas
        .iter()
        .map(|meta| (meta.name.as_str(), &meta.shape))
        .collect();
    let expert_groups = collect_expert_groups(metas.iter().map(|meta| meta.name.as_str()));
    let expert_source_keys: HashSet<&str> = expert_groups
        .values()
        .flat_map(|group| group.iter().map(|tensor| tensor.name.as_str()))
        .collect();

    let mut entries: Vec<PlanEntry> = Vec::new();
    let mut group_keys: Vec<ExpertGroupKey> = expert_groups.keys().copied().collect();
    group_keys.sort_by_key(|key| (key.layer, key.projection.gguf_suffix()));
    for key in group_keys {
        let (experts, shape) = expert_stack_layout(&expert_groups[&key], &shapes)?;
        entries.push(PlanEntry {
            gguf_name: format!("blk.{}.{}", key.layer, key.projection.gguf_suffix()),
            shape,
            from: Origin::Experts(experts),
        });
    }
    for meta in metas {
        if expert_source_keys.contains(meta.name.as_str()) {
            continue;
        }
        // A tensor the file does not hold (a tied head) is left out.
        let Some(gguf_name) = arch.tensor_name(&meta.name)? else {
            continue;
        };
        entries.push(PlanEntry {
            gguf_name,
            shape: meta.shape.iter().map(|&d| d as u64).collect(),
            from: Origin::Tensor(meta.name.clone()),
        });
    }
    for (gguf_name, shape, values) in arch.extra_tensors() {
        entries.push(PlanEntry {
            gguf_name,
            shape,
            from: Origin::Computed(values),
        });
    }
    Ok(entries)
}

/// A tensor of the output file before its type is chosen.
struct PlanEntry {
    gguf_name: String,
    /// Row-major, as the builder takes it (the writer reverses).
    shape: Vec<u64>,
    from: Origin,
}

/// One tensor of the output file, in file order.
struct PlannedTensor {
    gguf_name: String,
    /// Row-major, as the builder takes it (the writer reverses).
    shape: Vec<u64>,
    dtype: pmetal_gguf::GgmlType,
    from: Origin,
}

/// Where a planned tensor's values come from.
enum Origin {
    /// One tensor of the model, by its name there.
    Tensor(String),
    /// The experts of a layer's projection, stacked, in expert order.
    Experts(Vec<ExpertTensor>),
    /// Values the model has no tensor for, worked out from its config.
    Computed(Vec<f32>),
}

/// The largest finite value an `f16` holds.
const F16_MAX: f32 = 65504.0;

/// Bring the values of a tensor that is stored as `f16` into its range.
///
/// A value beyond it would be written as infinity, and one infinite weight
/// turns every output that reads it to NaN. Saturating changes the weight, but
/// by far less than that does; it is reported so that the choice of `f16` for
/// a model with such weights is not a silent one.
fn clamp_to_f16(values: &mut [f32], name: &str) {
    let mut clamped = 0usize;
    for value in values.iter_mut() {
        if value.abs() > F16_MAX && value.is_finite() {
            *value = value.clamp(-F16_MAX, F16_MAX);
            clamped += 1;
        }
    }
    if clamped > 0 {
        tracing::warn!(
            "{name}: {clamped} values are beyond the range of f16 and were saturated; \
             bf16 holds this model's weights without loss"
        );
    }
}

// ── MLX safetensors path ──────────────────────────────────────────────────────

/// Run MLX-format safetensors quantization with per-tensor quality-based bit allocation.
///
/// `output_path` is treated as a directory.  The directory is created if it
/// does not exist.  Inside it the function writes:
/// - `model.safetensors`   — quantized weights in MLX affine format
/// - `config.json`         — source config + injected `quantization_config`
/// - `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
///   `merges.txt`, `vocab.json`, `tokenizer.model` (copied if present)
pub(crate) async fn run_quantization_mlx(
    model_path: &str,
    output_path: &str,
    default_bits: i32,
    group_size: i32,
    target_bpw: Option<f32>,
) -> anyhow::Result<()> {
    use pmetal_runtime::mlx_quant;

    println!("========================================");
    println!("  PMetal MLX Safetensors Quantization");
    println!("========================================");
    println!("Model:      {}", model_path);
    println!("Output:     {}", output_path);
    println!("Bits:       {}", default_bits);
    println!("Group size: {}", group_size);
    if let Some(bpw) = target_bpw {
        println!("Target BPW: {:.2}", bpw);
    }
    println!("========================================\n");

    // 1. Resolve HuggingFace model ID to local path.
    println!("Resolving model: {}", model_path);
    let resolved_model_path: std::path::PathBuf =
        pmetal_hub::resolve_model_path(model_path, None, None).await?;

    // 2. Load all weights as InlineArray (stays on GPU, avoids f32 copy).
    println!("Loading weights...");
    let weights = pmetal_models::loader::load_weights(&resolved_model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?;
    println!("Loaded {} tensors", weights.len());

    // 3. Run the full pipeline: evaluate quality → allocate bits → quantize → save.
    let output_dir = std::path::PathBuf::from(output_path);
    let effective_bpw = target_bpw.unwrap_or(default_bits as f32);
    let source_config = resolved_model_path.join("config.json");

    println!(
        "Evaluating tensor quality and allocating bits (target BPW={:.2})...",
        effective_bpw
    );

    let assignments = mlx_quant::quantize_model(
        &weights,
        &source_config,
        &output_dir,
        effective_bpw,
        group_size,
        mlx_quant::DEFAULT_BITS_CANDIDATES,
        &[], // no extra critical tensor patterns
    )
    .map_err(|e| anyhow::anyhow!("Quantization failed: {}", e))?;

    // 4. Print allocation summary.
    let mut counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    let mut total_params: usize = 0;
    let mut total_weighted_bits: f64 = 0.0;
    for a in &assignments {
        *counts.entry(a.bits).or_insert(0) += 1;
        total_params += a.param_count;
        let bits = if a.bits == 0 { 16 } else { a.bits };
        total_weighted_bits += a.param_count as f64 * bits as f64;
    }
    let final_bpw = if total_params > 0 {
        total_weighted_bits / total_params as f64
    } else {
        0.0
    };

    println!("\nBit allocation summary:");
    let mut bit_keys: Vec<_> = counts.keys().collect();
    bit_keys.sort();
    for &bits in &bit_keys {
        let count = counts[bits];
        if *bits == 0 {
            println!("  bf16 (passthrough): {} tensors", count);
        } else {
            println!("  Q{}: {} tensors", *bits, count);
        }
    }
    println!("Effective BPW: {:.3}", final_bpw);
    println!("Total tensors: {}", assignments.len());

    // 5. Copy tokenizer files from source to output.
    let tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
        "tokenizer.model",
    ];
    for fname in &tokenizer_files {
        let src = resolved_model_path.join(fname);
        if src.exists() {
            let dst = output_dir.join(fname);
            std::fs::copy(&src, &dst).ok();
        }
    }

    println!("\nMLX quantization complete!");
    println!("Output: {}", output_dir.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_gguf::GgmlType;

    #[test]
    fn expert_tensor_names_are_grouped_by_layer_and_projection() {
        let (key, expert) =
            parse_expert_tensor_name("model.layers.12.mlp.experts.3.down_proj.weight").unwrap();
        assert_eq!(key.layer, 12);
        assert_eq!(key.projection, ExpertProjection::Down);
        assert_eq!(expert, 3);

        let (key, expert) =
            parse_expert_tensor_name("model.layers.2.block_sparse_moe.experts.7.w1.weight")
                .unwrap();
        assert_eq!(key.layer, 2);
        assert_eq!(key.projection, ExpertProjection::Gate);
        assert_eq!(expert, 7);
    }

    // ── streaming quantization, end to end on a tiny model ───────────────────

    /// A float in `[-1, 1]` that depends on where it is: tensors whose values
    /// differ, so a tensor written in the wrong slot or the wrong row shows.
    fn pattern(seed: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 37 + seed * 11) % 101) as f32 - 50.0) / 50.0)
            .collect()
    }

    /// `bf16` rounding of `values`, as stored in the checkpoint.
    fn as_stored(values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_f32())
            .collect()
    }

    /// A checkpoint of bf16 tensors in two shards, with a config.
    fn write_tiny_model(dir: &Path, experts: bool) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        use safetensors::tensor::{Dtype as StDtype, TensorView, serialize};
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        let mut add = |name: &str, shape: &[usize]| {
            let n: usize = shape.iter().product();
            let seed = tensors.len();
            tensors.push((
                name.to_string(),
                shape.to_vec(),
                as_stored(&pattern(seed, n)),
            ));
        };
        add("model.embed_tokens.weight", &[32, 64]);
        add("model.norm.weight", &[64]);
        add("lm_head.weight", &[32, 64]);
        add("model.layers.0.input_layernorm.weight", &[64]);
        add("model.layers.0.self_attn.q_proj.weight", &[64, 64]);
        add("model.layers.0.self_attn.k_proj.weight", &[32, 64]);
        add("model.layers.0.self_attn.v_proj.weight", &[32, 64]);
        add("model.layers.0.self_attn.o_proj.weight", &[64, 64]);
        if experts {
            for e in 0..3 {
                for (proj, shape) in [
                    ("gate_proj", [16, 64]),
                    ("up_proj", [16, 64]),
                    ("down_proj", [64, 16]),
                ] {
                    add(
                        &format!("model.layers.0.mlp.experts.{e}.{proj}.weight"),
                        &shape,
                    );
                }
            }
        } else {
            add("model.layers.0.mlp.gate_proj.weight", &[128, 64]);
            add("model.layers.0.mlp.up_proj.weight", &[128, 64]);
            add("model.layers.0.mlp.down_proj.weight", &[64, 128]);
        }

        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["Qwen2ForCausalLM"],"hidden_size":64,"num_hidden_layers":1,
               "num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":128,
               "vocab_size":32,"max_position_embeddings":128}"#,
        )
        .unwrap();
        // Two shards with an index, the layout of a real download.
        let (first, second) = tensors.split_at(tensors.len() / 2);
        let mut weight_map = std::collections::BTreeMap::new();
        for (file, part) in [
            ("model-00001-of-00002.safetensors", first),
            ("model-00002-of-00002.safetensors", second),
        ] {
            let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = part
                .iter()
                .map(|(name, shape, values)| {
                    let data = values
                        .iter()
                        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                        .collect();
                    (name.clone(), shape.clone(), data)
                })
                .collect();
            let views: Vec<(String, TensorView<'_>)> = bytes
                .iter()
                .map(|(name, shape, data)| {
                    weight_map.insert(name.clone(), (*file).to_string());
                    (
                        name.clone(),
                        TensorView::new(StDtype::BF16, shape.clone(), data).unwrap(),
                    )
                })
                .collect();
            std::fs::write(dir.join(file), serialize(views, None).unwrap()).unwrap();
        }
        pmetal_core::safetensors_index::ShardIndex::new(weight_map, 0)
            .write(&dir.join(pmetal_core::safetensors_index::INDEX_FILE))
            .unwrap();
        tensors
    }

    fn dequantized(model: &pmetal_gguf::GgufModel, name: &str) -> (Vec<f32>, Vec<u64>, GgmlType) {
        let info = model
            .tensor_info(name)
            .unwrap_or_else(|| panic!("no tensor {name}"));
        let shape: Vec<i32> = info.dimensions.iter().map(|&d| d as i32).collect();
        let values =
            pmetal_gguf::dequant::dequantize(model.tensor_data(name).unwrap(), info.dtype, &shape)
                .unwrap();
        (values, info.dimensions.clone(), info.dtype)
    }

    fn worst_difference(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    async fn quantize_dir(
        dir: &Path,
        out: &Path,
        method: crate::cli::quantize::QuantizeMethod,
        kl: bool,
        bpw: Option<f32>,
    ) -> anyhow::Result<()> {
        run_quantization(
            dir.to_str().unwrap(),
            out.to_str().unwrap(),
            ImportanceSource::None,
            method,
            kl,
            bpw,
            0.01,
        )
        .await
    }

    #[tokio::test]
    async fn a_model_is_quantized_tensor_by_tensor_into_a_valid_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let tensors = write_tiny_model(dir.path(), false);
        let out = dir.path().join("out").join("model.gguf");
        quantize_dir(
            dir.path(),
            &out,
            crate::cli::quantize::QuantizeMethod::Q8_0,
            false,
            None,
        )
        .await
        .unwrap();

        // Only the finished file is there: no temporary left beside it.
        let entries: Vec<_> = std::fs::read_dir(out.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, ["model.gguf"]);

        let model = pmetal_gguf::GgufModel::open(&out).unwrap();
        assert_eq!(model.num_tensors(), tensors.len());
        for (hf_name, shape, values) in &tensors {
            let gguf_name = pmetal_models::weight_format::WeightLoader::hf_to_gguf_name(hf_name);
            let (got, dims, dtype) = dequantized(&model, &gguf_name);
            let want_dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
            assert_eq!(dims, want_dims, "{gguf_name}");
            let worst = worst_difference(&got, values);
            if shape.len() == 2 {
                assert_eq!(dtype, GgmlType::Q8_0, "{gguf_name}");
                assert!(worst < 0.01, "{gguf_name}: q8_0 differs by {worst}");
            } else {
                // A norm weight stays dense, in f32.
                assert_eq!(dtype, GgmlType::F32, "{gguf_name}");
                assert!(worst < 1e-3, "{gguf_name}: differs by {worst}");
            }
        }
    }

    /// The experts of a layer's projection are one stacked tensor
    /// `[experts, out, in]` in expert order, however the checkpoint lists them.
    #[tokio::test]
    async fn per_expert_tensors_are_stacked_in_expert_order() {
        let dir = tempfile::tempdir().unwrap();
        let tensors = write_tiny_model(dir.path(), true);
        let out = dir.path().join("moe.gguf");
        quantize_dir(
            dir.path(),
            &out,
            crate::cli::quantize::QuantizeMethod::Q8_0,
            false,
            None,
        )
        .await
        .unwrap();
        let model = pmetal_gguf::GgufModel::open(&out).unwrap();
        let (stack, dims, _) = dequantized(&model, "blk.0.ffn_up_exps.weight");
        assert_eq!(dims, [3, 16, 64]);
        let mut want = Vec::new();
        for e in 0..3 {
            let name = format!("model.layers.0.mlp.experts.{e}.up_proj.weight");
            want.extend(&tensors.iter().find(|(n, ..)| *n == name).unwrap().2);
        }
        let worst = worst_difference(&stack, &want);
        assert!(worst < 0.01, "the stacked experts differ by {worst}");
        // No per-expert tensor is left beside the stack.
        assert!(model.tensor_names().all(|n| !n.contains("experts.0")));
    }

    #[tokio::test]
    async fn kl_calibration_streams_too_and_picks_types_within_the_budget() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_model(dir.path(), false);
        let out = dir.path().join("kl.gguf");
        quantize_dir(
            dir.path(),
            &out,
            crate::cli::quantize::QuantizeMethod::Dynamic,
            true,
            Some(5.0),
        )
        .await
        .unwrap();
        let model = pmetal_gguf::GgufModel::open(&out).unwrap();
        assert!(model.num_tensors() > 0);
        let (_, _, dtype) = dequantized(&model, "blk.0.ffn_up.weight");
        assert_ne!(dtype, GgmlType::F32);
    }

    /// The calibration and the importance matrix are keyed by the names the
    /// file gives its tensors, so the routed experts of a projection are listed
    /// as the one stacked tensor they are written as, never one by one.
    #[test]
    fn the_plan_lists_the_stacked_experts_once_under_the_name_the_file_gives_them() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_model(dir.path(), true);
        let source = crate::commands::quantize_source::open(dir.path()).unwrap();
        let config =
            serde_json::from_str(&std::fs::read_to_string(dir.path().join("config.json")).unwrap())
                .unwrap();
        let arch = ExportArch::detect(Some(&config)).unwrap();
        let entries = plan_entries(source.tensors(), &arch).unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.gguf_name.as_str()).collect();
        for proj in ["gate", "up", "down"] {
            let name = format!("blk.0.ffn_{proj}_exps.weight");
            assert_eq!(names.iter().filter(|n| **n == name).count(), 1, "{name}");
        }
        assert!(names.iter().all(|n| !n.contains("experts.")), "{names:?}");
        let up = entries
            .iter()
            .find(|e| e.gguf_name == "blk.0.ffn_up_exps.weight")
            .unwrap();
        assert_eq!(up.shape, [3, 16, 64]);
        assert!(matches!(&up.from, Origin::Experts(experts) if experts.len() == 3));
    }

    /// An importance matrix on disk, from `(name, per-token means, token counts)`.
    fn write_imatrix(path: &Path, entries: &[(&str, Vec<f32>, Vec<f32>)]) {
        let mut imatrix = pmetal_gguf::imatrix::IMatrix::new();
        for (name, mean, counts) in entries {
            imatrix.insert_measured(name.to_string(), mean.clone(), counts.clone());
        }
        imatrix.save_gguf(path).unwrap();
    }

    /// `sum_j importance_j * (want - got)^2` over the rows of one slice.
    fn weighted_error(want: &[f32], got: &[f32], importance: &[f32]) -> f64 {
        let n = importance.len();
        want.iter()
            .zip(got)
            .enumerate()
            .map(|(i, (w, g))| f64::from(importance[i % n]) * f64::from(w - g).powi(2))
            .sum()
    }

    /// Quantizing with the importance matrix of a model gives a smaller error
    /// where the activations are large than quantizing without it.
    #[tokio::test]
    async fn an_importance_matrix_lowers_the_weighted_error_of_the_quantized_model() {
        use crate::cli::quantize::QuantizeMethod;
        let dir = tempfile::tempdir().unwrap();
        let tensors = write_tiny_model(dir.path(), false);
        // Four loud channels of every 64-wide input.
        let importance: Vec<f32> = (0..64).map(|j| if j < 4 { 400.0 } else { 1.0 }).collect();
        let names = [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
        ];
        let entries: Vec<(&str, Vec<f32>, Vec<f32>)> = names
            .iter()
            .map(|n| (*n, importance.clone(), vec![1024.0]))
            .collect();
        let matrix = dir.path().join("imatrix.gguf");
        write_imatrix(&matrix, &entries);

        let plain = dir.path().join("plain.gguf");
        let weighted = dir.path().join("weighted.gguf");
        quantize_dir(dir.path(), &plain, QuantizeMethod::Iq4Nl, false, None)
            .await
            .unwrap();
        run_quantization(
            dir.path().to_str().unwrap(),
            weighted.to_str().unwrap(),
            ImportanceSource::File(matrix.to_str().unwrap()),
            QuantizeMethod::Iq4Nl,
            false,
            None,
            0.01,
        )
        .await
        .unwrap();

        let (plain, weighted) = (
            pmetal_gguf::GgufModel::open(&plain).unwrap(),
            pmetal_gguf::GgufModel::open(&weighted).unwrap(),
        );
        let (mut without, mut with) = (0.0, 0.0);
        for gguf_name in names {
            let original = tensors
                .iter()
                .find(|(hf, ..)| {
                    pmetal_models::weight_format::WeightLoader::hf_to_gguf_name(hf) == gguf_name
                })
                .map(|(_, _, values)| values)
                .unwrap();
            let (a, _, dtype) = dequantized(&plain, gguf_name);
            let (b, ..) = dequantized(&weighted, gguf_name);
            assert_eq!(dtype, GgmlType::Iq4Nl, "{gguf_name}");
            let (ea, eb) = (
                weighted_error(original, &a, &importance),
                weighted_error(original, &b, &importance),
            );
            assert!(eb < ea, "{gguf_name}: weighted error {eb} against {ea}");
            without += ea;
            with += eb;
        }
        assert!(with < 0.9 * without, "{with} against {without}");
    }

    /// The matrix of a stacked-experts tensor has one run of channels per expert,
    /// found under the stacked name: experts quantized with it are weighted by
    /// their own channels, not left unweighted for want of a key that matches.
    #[tokio::test]
    async fn stacked_experts_are_weighted_by_their_own_importance() {
        use crate::cli::quantize::QuantizeMethod;
        let dir = tempfile::tempdir().unwrap();
        let tensors = write_tiny_model(dir.path(), true);
        // Expert e is loud in channels [8e, 8e + 4).
        let importance: Vec<Vec<f32>> = (0..3)
            .map(|e| {
                (0..64)
                    .map(|j| {
                        if (8 * e..8 * e + 4).contains(&j) {
                            400.0
                        } else {
                            1.0
                        }
                    })
                    .collect()
            })
            .collect();
        let stacked: Vec<f32> = importance.iter().flatten().copied().collect();
        let matrix = dir.path().join("imatrix.gguf");
        write_imatrix(
            &matrix,
            &[(
                "blk.0.ffn_up_exps.weight",
                stacked,
                vec![300.0, 500.0, 200.0],
            )],
        );

        let plain = dir.path().join("plain.gguf");
        let weighted = dir.path().join("weighted.gguf");
        quantize_dir(dir.path(), &plain, QuantizeMethod::Iq4Nl, false, None)
            .await
            .unwrap();
        run_quantization(
            dir.path().to_str().unwrap(),
            weighted.to_str().unwrap(),
            ImportanceSource::File(matrix.to_str().unwrap()),
            QuantizeMethod::Iq4Nl,
            false,
            None,
            0.01,
        )
        .await
        .unwrap();

        let (a, dims, _) = dequantized(
            &pmetal_gguf::GgufModel::open(&plain).unwrap(),
            "blk.0.ffn_up_exps.weight",
        );
        let (b, ..) = dequantized(
            &pmetal_gguf::GgufModel::open(&weighted).unwrap(),
            "blk.0.ffn_up_exps.weight",
        );
        assert_eq!(dims, [3, 16, 64]);
        let per_expert = 16 * 64;
        for (e, importance) in importance.iter().enumerate() {
            let name = format!("model.layers.0.mlp.experts.{e}.up_proj.weight");
            let original = &tensors.iter().find(|(n, ..)| *n == name).unwrap().2;
            let range = e * per_expert..(e + 1) * per_expert;
            let (ea, eb) = (
                weighted_error(original, &a[range.clone()], importance),
                weighted_error(original, &b[range], importance),
            );
            assert!(eb < ea, "expert {e}: weighted error {eb} against {ea}");
        }
    }

    /// A matrix that does not describe the tensor it is keyed to is an error
    /// naming the tensor, not a quantization with the wrong weights.
    #[tokio::test]
    async fn an_importance_matrix_of_the_wrong_width_is_an_error_naming_the_tensor() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_model(dir.path(), false);
        let matrix = dir.path().join("imatrix.gguf");
        write_imatrix(
            &matrix,
            &[("blk.0.ffn_up.weight", vec![1.0; 48], vec![10.0])],
        );
        let result = run_quantization(
            dir.path().to_str().unwrap(),
            dir.path().join("out.gguf").to_str().unwrap(),
            ImportanceSource::File(matrix.to_str().unwrap()),
            crate::cli::quantize::QuantizeMethod::Iq4Nl,
            false,
            None,
            0.01,
        )
        .await;
        let error = result.unwrap_err().to_string();
        assert!(error.contains("blk.0.ffn_up.weight"), "{error}");
        assert!(!dir.path().join("out.gguf").exists());
    }

    /// An importance matrix weights the GGUF quantizers; MLX output has none to
    /// weight, and a flag that does nothing is refused.
    #[tokio::test]
    async fn mlx_output_refuses_an_importance_matrix() {
        for (imatrix, calibration) in [(Some("im.gguf"), None), (None, Some("data.txt"))] {
            let spec = pmetal_core::jobs::QuantizeSpec {
                model: "m".into(),
                output: "o".into(),
                format: "mlx".into(),
                imatrix: imatrix.map(Into::into),
                calibration_dataset: calibration.map(Into::into),
                ..Default::default()
            };
            let error = run_quantize(spec).await.unwrap_err().to_string();
            assert!(error.contains("--format mlx"), "{error}");
        }
    }

    /// What a `llama` file holds that the checkpoint does not publish as it is:
    /// the query and key rows reordered, the head left out when it is the
    /// embedding, the norms in f32 whatever the type of the file, and Llama 3's
    /// RoPE scaling as a tensor.
    #[tokio::test]
    async fn a_llama_export_is_laid_out_the_way_the_engines_read_it() {
        let dir = tempfile::tempdir().unwrap();
        let tensors = write_tiny_model(dir.path(), false);
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "num_hidden_layers": 1,
            "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 128,
            "vocab_size": 32, "max_position_embeddings": 128, "rope_theta": 500000.0,
            "tie_word_embeddings": true,
            "rope_scaling": {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                             "original_max_position_embeddings": 64, "rope_type": "llama3"},
        });
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        let out = dir.path().join("llama.gguf");
        quantize_dir(
            dir.path(),
            &out,
            crate::cli::quantize::QuantizeMethod::Bf16,
            false,
            None,
        )
        .await
        .unwrap();
        let model = pmetal_gguf::GgufModel::open(&out).unwrap();

        // The head is the embedding: it is not written.
        assert!(model.tensor_info("output.weight").is_none());
        // The norms are f32 although the file is bf16; the matrices are bf16.
        let (_, _, norm) = dequantized(&model, "blk.0.attn_norm.weight");
        assert_eq!(norm, GgmlType::F32);
        let (_, _, output_norm) = dequantized(&model, "output_norm.weight");
        assert_eq!(output_norm, GgmlType::F32);
        let (_, _, up) = dequantized(&model, "blk.0.ffn_up.weight");
        assert_eq!(up, GgmlType::Bf16);

        // Query and key rows are reordered, head by head; the rest is not.
        let arch = ExportArch::detect(Some(&config)).unwrap();
        for (hf_name, gguf_name) in [
            (
                "model.layers.0.self_attn.q_proj.weight",
                "blk.0.attn_q.weight",
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                "blk.0.attn_k.weight",
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                "blk.0.attn_v.weight",
            ),
        ] {
            let (_, shape, values) = tensors.iter().find(|(n, ..)| n == hf_name).unwrap();
            let shape: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
            let mut want = values.clone();
            arch.lay_out(hf_name, &shape, &mut want);
            let (got, ..) = dequantized(&model, gguf_name);
            assert_eq!(got, want, "{gguf_name}");
            assert_eq!(got == *values, hf_name.contains("v_proj"), "{gguf_name}");
        }

        // The scaling is a tensor of 64 / 4 / 2 = 8 divisors, and not metadata.
        let (factors, dims, dtype) = dequantized(&model, "rope_freqs.weight");
        assert_eq!((dims, dtype), (vec![8], GgmlType::F32));
        assert_eq!(factors.len(), 8);
        assert!(factors.iter().all(|&f| (1.0..=8.0).contains(&f)));
        assert!(model.get_metadata("llama.rope.scaling.factor").is_none());
    }

    /// An architecture whose tensors have no known names is refused before
    /// anything is read, not exported under names no engine looks for.
    #[tokio::test]
    async fn an_architecture_with_no_gguf_layout_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_model(dir.path(), false);
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"architectures":["PhiForCausalLM"]}"#,
        )
        .unwrap();
        let out = dir.path().join("phi.gguf");
        let error = quantize_dir(
            dir.path(),
            &out,
            crate::cli::quantize::QuantizeMethod::Bf16,
            false,
            None,
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(error.contains("PhiForCausalLM"), "{error}");
        assert!(!out.exists());
    }

    /// A shard cut short is an error, and the destination is not touched.
    #[tokio::test]
    async fn a_damaged_shard_fails_without_leaving_an_output() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_model(dir.path(), false);
        let shard = dir.path().join("model-00002-of-00002.safetensors");
        let bytes = std::fs::read(&shard).unwrap();
        std::fs::write(&shard, &bytes[..bytes.len() - 64]).unwrap();
        let out = dir.path().join("bad.gguf");
        let result = quantize_dir(
            dir.path(),
            &out,
            crate::cli::quantize::QuantizeMethod::Q8_0,
            false,
            None,
        )
        .await;
        assert!(result.is_err());
        assert!(!out.exists());
    }
}
