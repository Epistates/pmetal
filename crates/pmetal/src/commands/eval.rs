use std::path::Path;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use pmetal_core::jobs::EvalSpec;
use pmetal_data::{DatasetFormat, Tokenizer, TrainingDataset};
use pmetal_lora::{DynamicLoraModel, TrainableModel};
use pmetal_models::DynamicModel;
use pmetal_runtime::compat::Array;
use pmetal_trainer::eval::{self, LoadedModel, Metric, Task, TaskData, TaskOptions, TaskReport};
use serde_json::{Value, json};

/// The model being scored: the checkpoint alone, or with an adapter attached.
enum EvalModel {
    Base(DynamicModel),
    Adapted(DynamicLoraModel),
}

impl EvalModel {
    /// Load `model_path`, with the adapter at `lora_path` attached if there is one.
    fn load(model_path: &Path, lora_path: Option<&str>) -> anyhow::Result<Self> {
        // ⚠️ A base model is not a rank-0 adapter: `AdaptedModel::attach` rejects
        // rank 0, which left eval unable to score any model without `--lora`.
        match lora_path {
            None => {
                Ok(Self::Base(DynamicModel::load(model_path).map_err(|e| {
                    anyhow::anyhow!("Failed to load model: {}", e)
                })?))
            }
            Some(lp) => {
                let lp = Path::new(lp);
                let lora_config = pmetal::inference_runner::lora_config_for_adapter(lp)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let mut model = DynamicLoraModel::from_pretrained(model_path, lora_config)?;
                model
                    .load_lora_weights(lp)
                    .map_err(|e| anyhow::anyhow!("Failed to load LoRA weights: {}", e))?;
                Ok(Self::Adapted(model))
            }
        }
    }

    fn forward(&mut self, input_ids: &Array) -> anyhow::Result<Array> {
        let logits = match self {
            Self::Base(model) => model.forward(input_ids, None).map_err(|e| e.to_string()),
            Self::Adapted(model) => model.forward(input_ids, None).map_err(|e| e.to_string()),
        };
        logits.map_err(|e| anyhow::anyhow!("Forward pass failed: {e}"))
    }

    /// The model as task evaluation runs it.
    fn into_loaded(self) -> LoadedModel {
        match self {
            Self::Base(model) => LoadedModel::base(model),
            Self::Adapted(model) => LoadedModel::adapted(model),
        }
    }
}

/// Evaluate a model: perplexity on a dataset, or a benchmark task.
pub(crate) async fn run_eval(spec: EvalSpec) -> anyhow::Result<()> {
    match spec.task.clone() {
        Some(task) => {
            let task: Task = task.parse().map_err(anyhow::Error::msg)?;
            run_task(spec, task).await
        }
        None => run_perplexity(spec).await,
    }
}

/// Write `results` to `path`, creating its directory.
fn write_results(path: &str, results: &Value) -> anyhow::Result<()> {
    let path = Path::new(path);
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(results)? + "\n")?;
    Ok(())
}

fn metric_line(name: &str, metric: &Metric) -> String {
    match metric.stderr {
        Some(stderr) => format!("{name:<34} {:>7.4} ± {stderr:.4}", metric.value),
        None => format!("{name:<34} {:>7.4}", metric.value),
    }
}

/// The file the samples of a run are written to: the results file's name with
/// `.samples.jsonl` in place of its extension.
fn samples_path(output: &str) -> std::path::PathBuf {
    Path::new(output).with_extension("samples.jsonl")
}

/// Score a benchmark task.
async fn run_task(spec: EvalSpec, task: Task) -> anyhow::Result<()> {
    let EvalSpec {
        model: ref model_id,
        ref dataset,
        ref lora,
        max_seq_len,
        num_samples,
        seed,
        batch_size,
        max_gen_tokens,
        ref output,
        log_samples,
        json: json_output,
        fewshot,
        ..
    } = spec;
    let options = TaskOptions {
        fewshot,
        limit: num_samples,
        seed,
        max_gen_tokens,
        log_samples,
    };
    let shots = fewshot.unwrap_or_else(|| task.default_fewshot());
    let max_len = if max_seq_len == 0 { 8192 } else { max_seq_len };

    if !json_output {
        println!("PMetal Eval");
        println!("===========");
        println!("Model:   {model_id}");
        println!("Task:    {task} ({shots}-shot)");
        if !dataset.is_empty() {
            println!("Data:    {dataset}");
        }
        if num_samples > 0 {
            println!("Limit:   {num_samples}");
        }
        if let Some(lp) = lora {
            println!("LoRA:    {lp}");
        }
        println!();
    }

    let model_path = pmetal_hub::resolve_model_path(model_id, None, None).await?;
    let tokenizer = Tokenizer::from_model_dir(&model_path)?;
    let chat_template = pmetal_data::chat_templates::detect_chat_template(&model_path, model_id);

    // The task's data: the user's file, or the standard dataset's files.
    let data = if dataset.is_empty() {
        eval::download_task_data(task, shots).await?
    } else {
        let path = pmetal_trainer::resolve_dataset_path(dataset).await?;
        TaskData::from_files(&path, None)?
    };

    let stop_tokens = pmetal_data::inference_config::collect_all_stop_tokens(
        &model_path,
        &tokenizer,
        Some(chat_template.template_type),
    );
    let mut model = EvalModel::load(&model_path, lora.as_deref())?
        .into_loaded()
        .with_batch_size(batch_size)
        .with_max_len(max_len)
        .with_stop_tokens(stop_tokens);

    let started = Instant::now();
    let bar = (!json_output).then(|| {
        let b = ProgressBar::new(0);
        b.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40} {pos}/{len} ({eta})")
                .expect("a valid progress template")
                .progress_chars("=> "),
        );
        b
    });
    let report = eval::evaluate(
        &mut model,
        &tokenizer,
        task,
        &data,
        &options,
        &mut |done, total| {
            if let Some(bar) = &bar {
                bar.set_length(total as u64);
                bar.set_position(done as u64);
            }
        },
    )?;
    if let Some(bar) = bar {
        bar.finish_and_clear();
    }
    let elapsed = started.elapsed().as_secs_f64();

    let results = json!({
        "model": model_id,
        "lora": lora,
        "task": report.task,
        "config": {
            "fewshot": report.fewshot,
            "limit": num_samples,
            "seed": seed,
            "batch_size": batch_size,
            "max_gen_tokens": max_gen_tokens,
            "max_seq_len": max_len,
            "dataset": (!dataset.is_empty()).then_some(dataset),
        },
        "results": report,
        "elapsed_seconds": elapsed,
        "pmetal_version": env!("CARGO_PKG_VERSION"),
    });
    if let Some(output) = output {
        write_results(output, &results)?;
        if log_samples {
            let lines: String = report
                .samples_log
                .iter()
                .map(|sample| format!("{sample}\n"))
                .collect();
            std::fs::write(samples_path(output), lines)?;
        }
    }

    if json_output {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        print_report(&report, elapsed, output.as_deref());
    }
    Ok(())
}

fn print_report(report: &TaskReport, elapsed: f64, output: Option<&str>) {
    println!("Results: {} ({}-shot)", report.task, report.fewshot);
    println!("=======");
    println!("Samples:  {}", report.samples);
    if report.skipped > 0 {
        println!(
            "Skipped:  {} (an instruction this build cannot check)",
            report.skipped
        );
    }
    for (name, metric) in &report.metrics {
        println!("{}", metric_line(name, metric));
    }
    if report.groups.len() > 1 {
        println!();
        for (subject, group) in &report.groups {
            let line = group
                .metrics
                .iter()
                .map(|(name, m)| format!("{name} {:.3}", m.value))
                .collect::<Vec<_>>()
                .join("  ");
            println!("  {subject:<40} n={:<4} {line}", group.samples);
        }
    }
    println!();
    println!("Time:     {elapsed:.1}s");
    if let Some(output) = output {
        println!("Wrote:    {output}");
    }
}

/// Evaluate model perplexity on a dataset.
async fn run_perplexity(spec: EvalSpec) -> anyhow::Result<()> {
    use pmetal_runtime::compat::indexing::take_along_axis;

    let EvalSpec {
        model: ref model_id,
        dataset: ref dataset_path,
        ref lora,
        max_seq_len,
        num_samples,
        json: json_output,
        ref output,
        ..
    } = spec;
    let max_seq_len = if max_seq_len == 0 { 1024 } else { max_seq_len };
    let lora_path = lora.as_deref();

    // Resolve model
    let model_path = pmetal_hub::resolve_model_path(model_id, None, None).await?;

    if !json_output {
        println!("PMetal Eval");
        println!("===========");
        println!("Model:   {}", model_id);
        println!("Dataset: {}", dataset_path);
        println!("MaxLen:  {}", max_seq_len);
        if let Some(lp) = lora_path {
            println!("LoRA:    {}", lp);
        }
        println!();
    }

    // Load tokenizer: from the directory's `tokenizer.json`, or from the
    // metadata of a GGUF file.
    let tokenizer = Tokenizer::from_model_path(&model_path)?;

    // Load dataset
    let chat_template = pmetal_data::chat_templates::detect_chat_template(&model_path, model_id);
    let dataset_file = pmetal_trainer::resolve_dataset_path(dataset_path).await?;
    let dataset = TrainingDataset::from_jsonl_tokenized(
        &dataset_file,
        &tokenizer,
        DatasetFormat::Auto,
        max_seq_len,
        Some(&chat_template),
        None,
    )?;

    let mut model = EvalModel::load(&model_path, lora_path)?;

    // Evaluate perplexity
    let samples = dataset.samples();
    let eval_samples = if num_samples == 0 || num_samples > samples.len() {
        samples.len()
    } else {
        num_samples
    };

    if !json_output {
        println!("Evaluating {} samples...", eval_samples);
    }

    let mut total_nll: f64 = 0.0;
    let mut total_tokens: usize = 0;
    let bar = if !json_output {
        let b = ProgressBar::new(eval_samples as u64);
        b.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40} {pos}/{len} ({eta})")?
                .progress_chars("=> "),
        );
        Some(b)
    } else {
        None
    };

    for sample in samples.iter().take(eval_samples) {
        let tokens: Vec<i32> = sample.input_ids.iter().map(|&t| t as i32).collect();
        if tokens.len() < 2 {
            continue;
        }
        let n = tokens.len();
        let input_array =
            pmetal_runtime::compat::Array::from_slice(&tokens[..n - 1], &[1, (n - 1) as i32]);

        let logits = model.forward(&input_array)?;

        // logits: [1, seq-1, vocab]  → [seq-1, vocab]
        let logits = logits.squeeze_axes(&[0i32]);
        let log_probs = pmetal_runtime::compat::nn::log_softmax(&logits, -1);

        // Gather log-probs for the true tokens using take_along_axis
        // log_probs: [seq-1, vocab], indices: [seq-1, 1] → gathered: [seq-1, 1]
        let target_ids: Vec<i32> = sample.input_ids[1..].iter().map(|&t| t as i32).collect();
        let target_arr =
            pmetal_runtime::compat::Array::from_slice(&target_ids, &[(n - 1) as i32, 1]);
        let gathered = take_along_axis(&log_probs, &target_arr, 1);

        let gathered = gathered.as_dtype(pmetal_runtime::compat::Dtype::Float32.as_i32());
        gathered.eval();
        let nll: f32 = gathered.as_slice::<f32>().iter().map(|&v| -v).sum();

        total_nll += nll as f64;
        total_tokens += n - 1;

        if let Some(ref b) = bar {
            b.inc(1);
        }
    }

    if let Some(b) = bar {
        b.finish_and_clear();
    }

    if total_tokens == 0 {
        anyhow::bail!("No tokens to evaluate — dataset may be empty or all samples too short");
    }

    let avg_nll = total_nll / total_tokens as f64;
    let perplexity = avg_nll.exp();

    let results = json!({
        "model": model_id,
        "dataset": dataset_path,
        "num_samples": eval_samples,
        "total_tokens": total_tokens,
        "avg_nll": avg_nll,
        "perplexity": perplexity,
    });
    if let Some(output) = output {
        write_results(output, &results)?;
    }
    if json_output {
        println!("{}", serde_json::to_string_pretty(&results)?);
    } else {
        println!("Results");
        println!("=======");
        println!("Samples evaluated: {}", eval_samples);
        println!("Total tokens:      {}", total_tokens);
        println!("Average NLL:       {:.4}", avg_nll);
        println!("Perplexity:        {:.2}", perplexity);
    }

    Ok(())
}
