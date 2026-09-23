use indicatif::{ProgressBar, ProgressStyle};
use pmetal_bridge::compat::Array;
use pmetal_data::{DatasetFormat, Tokenizer, TrainingDataset};
use pmetal_lora::{DynamicLoraModel, TrainableModel};
use pmetal_models::DynamicModel;

/// The model being scored: the checkpoint alone, or with an adapter attached.
enum EvalModel {
    Base(DynamicModel),
    Adapted(DynamicLoraModel),
}

impl EvalModel {
    fn forward(&mut self, input_ids: &Array) -> anyhow::Result<Array> {
        let logits = match self {
            Self::Base(model) => model.forward(input_ids, None).map_err(|e| e.to_string()),
            Self::Adapted(model) => model.forward(input_ids, None).map_err(|e| e.to_string()),
        };
        logits.map_err(|e| anyhow::anyhow!("Forward pass failed: {e}"))
    }
}

/// Evaluate model perplexity on a dataset.
pub(crate) async fn run_eval(
    model_id: &str,
    dataset_path: &str,
    lora_path: Option<&str>,
    max_seq_len: usize,
    num_samples: usize,
    json_output: bool,
) -> anyhow::Result<()> {
    use pmetal_bridge::compat::indexing::take_along_axis;

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

    // Load tokenizer
    let tokenizer = Tokenizer::from_model_dir(&model_path)?;

    // Load dataset
    let chat_template = pmetal_data::chat_templates::detect_chat_template(&model_path, model_id);
    let dataset = TrainingDataset::from_jsonl_tokenized(
        dataset_path,
        &tokenizer,
        DatasetFormat::Auto,
        max_seq_len,
        Some(&chat_template),
        None,
    )?;

    // ⚠️ A base model is not a rank-0 adapter: `AdaptedModel::attach` rejects
    // rank 0, which left eval unable to score any model without `--lora`.
    let mut model = match lora_path {
        None => EvalModel::Base(
            DynamicModel::load(&model_path)
                .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?,
        ),
        Some(lp) => {
            let lp = std::path::Path::new(lp);
            let lora_config = pmetal::inference_runner::lora_config_for_adapter(lp);
            let lora_file = if lp.is_dir() {
                lp.join("lora_weights.safetensors")
            } else {
                lp.to_path_buf()
            };
            let mut model = DynamicLoraModel::from_pretrained(&model_path, lora_config)?;
            model
                .load_lora_weights(&lora_file)
                .map_err(|e| anyhow::anyhow!("Failed to load LoRA weights: {}", e))?;
            EvalModel::Adapted(model)
        }
    };

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
            pmetal_bridge::compat::Array::from_slice(&tokens[..n - 1], &[1, (n - 1) as i32]);

        let logits = model.forward(&input_array)?;

        // logits: [1, seq-1, vocab]  → [seq-1, vocab]
        let logits = logits.squeeze_axes(&[0i32]);
        let log_probs = pmetal_bridge::compat::nn::log_softmax(&logits, -1);

        // Gather log-probs for the true tokens using take_along_axis
        // log_probs: [seq-1, vocab], indices: [seq-1, 1] → gathered: [seq-1, 1]
        let target_ids: Vec<i32> = sample.input_ids[1..].iter().map(|&t| t as i32).collect();
        let target_arr =
            pmetal_bridge::compat::Array::from_slice(&target_ids, &[(n - 1) as i32, 1]);
        let gathered = take_along_axis(&log_probs, &target_arr, 1);

        let gathered = gathered.as_dtype(pmetal_bridge::compat::Dtype::Float32.as_i32());
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

    if json_output {
        let obj = serde_json::json!({
            "model": model_id,
            "dataset": dataset_path,
            "num_samples": eval_samples,
            "total_tokens": total_tokens,
            "avg_nll": avg_nll,
            "perplexity": perplexity,
        });
        println!("{}", serde_json::to_string_pretty(&obj)?);
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
