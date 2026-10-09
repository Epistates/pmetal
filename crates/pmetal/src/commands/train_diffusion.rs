//! `pmetal train-diffusion` — block-diffusion LoRA fine-tuning for DiffusionGemma.
//!
//! DiffusionGemma has no causal `forward(ids) -> logits`, so it does not fit the
//! training loop `pmetal train` runs. This command drives the dedicated
//! [`DiffusionGemmaTrainer`] instead, over the same adapted model: it loads the
//! model through the inference dispatcher, packs its base for QLoRA when asked
//! and attaches LoRA adapters exactly as `pmetal train` does, collates each
//! `(prompt, response)` example into an encoder context + fixed-length decoder
//! canvas, and runs the denoising objective. Only the LoRA adapters are trained.

use std::path::Path;

use anyhow::{Context, Result, bail};

use pmetal_bridge::compat::Array;
use pmetal_core::LoraConfig;
use pmetal_data::Tokenizer;
use pmetal_lora::{AdaptedModel, QLoraConfig, QLoraScheme};
use pmetal_models::DynamicModel;
use pmetal_trainer::orchestrator::save_adapter_config;
use pmetal_trainer::{
    DiffusionGemmaTrainConfig, DiffusionGemmaTrainer, quantize_diffusion_gemma_base,
};

use crate::QuantizationMethod;
use crate::cli::train_diffusion::TrainDiffusionArgs;

pub(crate) async fn run_train_diffusion(args: TrainDiffusionArgs) -> Result<()> {
    // 1. Resolve + load the model through the inference dispatcher.
    let model_path = pmetal_hub::resolve_model_path(&args.model, None, None)
        .await
        .map_err(|e| anyhow::anyhow!("resolve model {}: {e}", args.model))?;
    let mut base =
        DynamicModel::load(&model_path).map_err(|e| anyhow::anyhow!("load model: {e}"))?;
    let Some(dg) = base.as_diffusion_gemma_mut() else {
        bail!(
            "train-diffusion requires a DiffusionGemma model, got {:?}",
            base.architecture()
        );
    };
    let canvas_len = dg.canvas_length as usize;
    let vocab = dg.vocab_size;

    // 2. Tokenizer + pad token for canvas padding.
    let tokenizer = Tokenizer::from_model_dir(&model_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
    let pad_id = tokenizer
        .pad_token_id()
        .or_else(|| tokenizer.eos_token_id())
        .unwrap_or(0);
    let bos_id = tokenizer.bos_token_id();

    // 3. Optionally pack the frozen base (QLoRA), as `pmetal train` does,
    //    then attach LoRA adapters to the encoder and decoder projections.
    //    Packing first seeds a DoRA magnitude from the weight the layer
    //    computes with.
    let scheme = match args.quantization {
        QuantizationMethod::None => None,
        QuantizationMethod::Nf4 => Some(QLoraScheme::Nf4),
        QuantizationMethod::Fp4 => Some(QLoraScheme::Fp4),
        QuantizationMethod::Int8 => Some(QLoraScheme::Int8),
    };
    if let Some(scheme) = scheme {
        let qlora = QLoraConfig {
            scheme,
            group_size: args.quant_block_size as i32,
            double_quant: args.double_quant,
        };
        let packed = quantize_diffusion_gemma_base(&mut base, &qlora)
            .map_err(|e| anyhow::anyhow!("QLoRA base quantization: {e}"))?;
        tracing::info!(
            "QLoRA: {} projections packed as {scheme}, {:.1} MB -> {:.1} MB, and the experts",
            packed.packed,
            packed.dense_bytes as f64 / 1e6,
            packed.packed_bytes as f64 / 1e6,
        );
    }
    let lora_cfg = LoraConfig {
        r: args.lora_r,
        alpha: args.lora_alpha,
        target_modules: args.lora_targets.clone(),
        ..Default::default()
    };
    // Seeded here so the adapters' initialisation is reproducible too.
    pmetal_bridge::compat::random::seed(args.seed);
    let mut model = AdaptedModel::attach(base, lora_cfg.clone()).map_err(|e| {
        anyhow::anyhow!("attach LoRA (--lora-targets {:?}): {e}", args.lora_targets)
    })?;
    let n_adapter_params = model.lora_parameters().len();

    // 4. Load the (prompt, response) dataset.
    let examples = load_prompt_response_jsonl(&args.dataset)
        .with_context(|| format!("load dataset {}", args.dataset))?;
    anyhow::ensure!(!examples.is_empty(), "dataset {} is empty", args.dataset);

    // 5. Trainer.
    let train_cfg = DiffusionGemmaTrainConfig {
        learning_rate: args.learning_rate,
        weight_decay: args.weight_decay,
        max_grad_norm: args.max_grad_norm,
        corrupted_only: args.corrupted_only,
        seed: args.seed,
        ..Default::default()
    };
    let mut trainer = DiffusionGemmaTrainer::new(train_cfg, vocab);

    tracing::info!(
        "train-diffusion: {} adapter params, {} examples, canvas_len={}, {} steps",
        n_adapter_params,
        examples.len(),
        canvas_len,
        args.steps
    );

    // 6. Training loop — cycle the dataset to reach the requested step count.
    let output_dir = Path::new(&args.output);
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("create output dir {}", args.output))?;
    let mut running_loss = 0.0f64;
    for step in 0..args.steps {
        let (prompt, response) = &examples[step % examples.len()];
        let ctx_ids = tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("tokenize prompt: {e}"))?;
        let resp_ids = tokenizer
            .encode(response)
            .map_err(|e| anyhow::anyhow!("tokenize response: {e}"))?;

        let context = context_array(&ctx_ids, args.max_context_len, bos_id);
        let canvas = canvas_array(&resp_ids, canvas_len, pad_id);

        let stats = trainer
            .train_step(&mut model, &context, &canvas)
            .map_err(|e| anyhow::anyhow!("train step {step}: {e}"))?;
        running_loss = if step == 0 {
            stats.loss as f64
        } else {
            0.98 * running_loss + 0.02 * stats.loss as f64
        };

        if args.log_every > 0 && step % args.log_every == 0 {
            tracing::info!(
                "step {}: loss={:.4} (ema={:.4}){}",
                stats.step,
                stats.loss,
                running_loss,
                stats
                    .grad_norm
                    .map(|n| format!(", grad_norm={n:.2}"))
                    .unwrap_or_default()
            );
        }

        if args.checkpoint_every > 0 && step > 0 && step % args.checkpoint_every == 0 {
            let ckpt = output_dir.join("lora_weights.safetensors");
            save_adapters(&model, &lora_cfg, &ckpt, &args.model)
                .with_context(|| "checkpoint save")?;
            tracing::info!("checkpoint at step {step} → {}", ckpt.display());
        }
    }

    // 7. Save the final adapters.
    let final_path = output_dir.join("lora_weights.safetensors");
    save_adapters(&model, &lora_cfg, &final_path, &args.model).with_context(|| "save adapters")?;
    println!(
        "DiffusionGemma LoRA adapters saved to {} (final ema loss={:.4})",
        final_path.display(),
        running_loss
    );
    Ok(())
}

/// Write the adapters and the `adapter_config.json` beside them that
/// `infer --lora` and `fuse` read their rank, alpha and targets from, as
/// `pmetal train` does.
fn save_adapters(
    model: &AdaptedModel,
    lora: &LoraConfig,
    path: &Path,
    base_model: &str,
) -> Result<()> {
    model
        .save_lora_weights(path)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    save_adapter_config(
        path,
        lora.r,
        lora.alpha,
        &lora.target_modules,
        lora.use_rslora,
        Some(base_model),
    )
}

/// Build the encoder context `[1, ctx]` from a prompt's token ids, truncated to
/// the last `max_len` tokens. Guarantees at least one token (falls back to BOS,
/// else the first available id, else 0) so the encoder always has input.
fn context_array(ids: &[u32], max_len: usize, bos_id: Option<u32>) -> Array {
    let max_len = max_len.max(1);
    let start = ids.len().saturating_sub(max_len);
    let mut kept: Vec<i32> = ids[start..].iter().map(|&x| x as i32).collect();
    if kept.is_empty() {
        kept.push(bos_id.unwrap_or(0) as i32);
    }
    let len = kept.len() as i32;
    Array::from_slice(&kept, &[1, len])
}

/// Build the fixed-length decoder canvas `[1, canvas_len]` from a response's
/// token ids: truncate to `canvas_len`, pad the tail with `pad_id`.
fn canvas_array(ids: &[u32], canvas_len: usize, pad_id: u32) -> Array {
    let canvas_len = canvas_len.max(1);
    let mut v: Vec<i32> = ids.iter().take(canvas_len).map(|&x| x as i32).collect();
    v.resize(canvas_len, pad_id as i32);
    Array::from_slice(&v, &[1, canvas_len as i32])
}

/// Parse a JSONL dataset into `(prompt, response)` pairs. Accepts the field
/// aliases `prompt`/`context`/`input` and `response`/`target`/`output`. Lines
/// missing either field are skipped.
fn load_prompt_response_jsonl(path: &str) -> Result<Vec<(String, String)>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("{path}:{}: invalid JSON", i + 1))?;
        let field = |names: &[&str]| -> Option<String> {
            names
                .iter()
                .find_map(|n| value.get(*n).and_then(|v| v.as_str()))
                .map(str::to_string)
        };
        let prompt = field(&["prompt", "context", "input"]);
        let response = field(&["response", "target", "output"]);
        if let (Some(p), Some(r)) = (prompt, response) {
            out.push((p, r));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canvas_truncates_and_pads() {
        // Longer than canvas → truncated to first `canvas_len`.
        let mut long = canvas_array(&[1, 2, 3, 4, 5], 3, 0);
        assert_eq!(long.shape(), &[1, 3]);
        assert_eq!(long.to_f32_vec(3).unwrap(), vec![1.0, 2.0, 3.0]);

        // Shorter than canvas → tail padded with pad_id.
        let mut short = canvas_array(&[7, 8], 4, 99);
        assert_eq!(short.shape(), &[1, 4]);
        assert_eq!(short.to_f32_vec(4).unwrap(), vec![7.0, 8.0, 99.0, 99.0]);
    }

    #[test]
    fn context_keeps_last_tokens_and_never_empty() {
        // Truncation keeps the most recent tokens.
        let mut ctx = context_array(&[10, 11, 12, 13], 2, Some(1));
        assert_eq!(ctx.shape(), &[1, 2]);
        assert_eq!(ctx.to_f32_vec(2).unwrap(), vec![12.0, 13.0]);

        // Empty prompt falls back to BOS.
        let mut empty = context_array(&[], 8, Some(2));
        assert_eq!(empty.shape(), &[1, 1]);
        assert_eq!(empty.to_f32_vec(1).unwrap(), vec![2.0]);

        // Empty prompt, no BOS → 0.
        let mut empty0 = context_array(&[], 8, None);
        assert_eq!(empty0.to_f32_vec(1).unwrap(), vec![0.0]);
    }

    #[test]
    fn jsonl_parses_aliases_and_skips_incomplete() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.jsonl");
        std::fs::write(
            &path,
            concat!(
                "{\"prompt\": \"a\", \"response\": \"b\"}\n",
                "\n",
                "{\"context\": \"c\", \"target\": \"d\"}\n",
                "{\"input\": \"e\", \"output\": \"f\"}\n",
                "{\"prompt\": \"only prompt\"}\n"
            ),
        )
        .unwrap();
        let rows = load_prompt_response_jsonl(path.to_str().unwrap()).unwrap();
        assert_eq!(
            rows,
            vec![
                ("a".to_string(), "b".to_string()),
                ("c".to_string(), "d".to_string()),
                ("e".to_string(), "f".to_string()),
            ]
        );
    }
}
