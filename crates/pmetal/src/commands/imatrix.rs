//! `pmetal imatrix`: measure an importance matrix on calibration data.
//!
//! The model runs over chunks of the calibration text and the activations at
//! each weight matrix are summed per input channel
//! ([`pmetal_models::calibration`]). The result is written in the GGUF imatrix
//! format, or the legacy `.dat` one for a `.dat` output path, and is what
//! `pmetal quantize --imatrix` consumes. `pmetal quantize
//! --calibration-dataset` does the same measurement in the same run.

use std::path::{Path, PathBuf};

use indicatif::{ProgressBar, ProgressStyle};
use pmetal_data::Tokenizer;
use pmetal_data::calibration::{CalibrationOptions, CalibrationSet, load_calibration_chunks};
use pmetal_gguf::imatrix::IMatrix;
use pmetal_models::DynamicModel;
use pmetal_models::calibration::{ImatrixRun, collect_imatrix};

/// Run `pmetal imatrix`.
pub(crate) async fn run_imatrix(spec: pmetal_core::jobs::ImatrixSpec) -> anyhow::Result<()> {
    println!("========================================");
    println!("  PMetal Importance Matrix");
    println!("========================================");
    println!("Model:    {}", spec.model);
    println!("Data:     {}", spec.dataset);
    println!("Output:   {}", spec.output);
    println!("Context:  {} tokens per chunk", spec.ctx_size);
    println!("========================================\n");

    let output = crate::validate_output_path(&spec.output, "importance matrix output")?;
    let model_path = pmetal_hub::resolve_model_path(&spec.model, None, None).await?;
    let run = measure_imatrix(
        &model_path,
        &spec.model,
        &spec.dataset,
        spec.ctx_size,
        (spec.chunks > 0).then_some(spec.chunks),
        spec.batch_size,
    )
    .await?;

    write_imatrix(&run.imatrix, &output)?;
    println!(
        "Wrote {} matrices measured over {} chunks ({} tokens) to {}",
        run.imatrix.len(),
        run.chunks,
        run.tokens,
        output.display()
    );
    Ok(())
}

/// Write `imatrix` as a GGUF file, or as the legacy format for a `.dat` path.
pub(crate) fn write_imatrix(imatrix: &IMatrix, path: &Path) -> anyhow::Result<()> {
    let legacy = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("dat"));
    if legacy {
        imatrix.save(path)?;
    } else {
        imatrix.save_gguf(path)?;
    }
    Ok(())
}

/// Where a calibration dataset is read from: a local file or directory, or a
/// Hub dataset id downloaded to the cache.
async fn resolve_calibration_path(dataset: &str) -> anyhow::Result<PathBuf> {
    let local = Path::new(dataset);
    if local.exists() {
        return Ok(pmetal_data::TrainingDataset::resolve_dataset_path_pub(
            local,
        )?);
    }
    #[cfg(feature = "trainer")]
    {
        pmetal_trainer::resolve_dataset_path(dataset).await
    }
    #[cfg(not(feature = "trainer"))]
    {
        anyhow::bail!(
            "calibration data '{dataset}' is not a local path, and downloading a dataset needs a \
             build with the `trainer` feature"
        )
    }
}

/// Load the calibration data, run `model_path` over it and measure the
/// importance matrix.
///
/// `model_name` is what the model is called (its Hub id or path), for finding
/// its chat template. The model is loaded for the measurement and dropped
/// when it ends.
pub(crate) async fn measure_imatrix(
    model_path: &Path,
    model_name: &str,
    dataset: &str,
    ctx_size: usize,
    max_chunks: Option<usize>,
    batch_size: usize,
) -> anyhow::Result<ImatrixRun> {
    let tokenizer = Tokenizer::from_model_dir(model_path)?;
    let template = pmetal_data::chat_templates::detect_chat_template(model_path, model_name);
    let data_path = resolve_calibration_path(dataset).await?;
    let CalibrationSet {
        chunks,
        documents,
        total_tokens,
    } = load_calibration_chunks(
        &data_path,
        &tokenizer,
        &CalibrationOptions {
            chunk_size: ctx_size,
            max_chunks,
            template: Some(&template),
        },
    )?;
    println!(
        "Calibration data: {documents} documents, {total_tokens} tokens, {} chunks of up to {ctx_size}",
        chunks.len()
    );
    if chunks.len() < 8 {
        println!(
            "  (few chunks: an importance matrix from so little text describes that text more \
             than the model; a few hundred chunks is typical)"
        );
    }

    println!("Loading model...");
    let mut model = DynamicModel::load(model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load the model: {e}"))?;

    let bar = ProgressBar::new(chunks.len() as u64);
    bar.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {bar:40} {pos}/{len} chunks ({eta})")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("=> "),
    );
    let mut run = collect_imatrix(&mut model, &chunks, batch_size, |done, _| {
        bar.set_position(done as u64)
    })
    .map_err(|e| anyhow::anyhow!("Measuring the activations failed: {e}"))?;
    bar.finish_and_clear();

    run.imatrix.dataset_name = Path::new(dataset)
        .file_name()
        .and_then(|n| n.to_str())
        .map(str::to_string)
        .or_else(|| Some(dataset.to_string()));
    if run.rejected > 0 {
        eprintln!(
            "warning: {} activations did not fit the matrix they named and were not counted",
            run.rejected
        );
    }
    if !run.unmeasured.is_empty() {
        let shown: Vec<&str> = run.unmeasured.iter().take(6).map(String::as_str).collect();
        eprintln!(
            "warning: {} weight matrices were not measured and will be quantized without \
             importance: {}{}",
            run.unmeasured.len(),
            shown.join(", "),
            if run.unmeasured.len() > shown.len() {
                ", ..."
            } else {
                ""
            }
        );
    }
    Ok(run)
}
