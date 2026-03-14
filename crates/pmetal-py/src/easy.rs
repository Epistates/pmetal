//! Top-level easy API functions for Python.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::runtime_err;

/// Fine-tune a model with sensible defaults.
///
/// Args:
///     model_id: HuggingFace model ID or local path
///     dataset_path: Path to JSONL training dataset
///     lora_r: LoRA rank (default 16)
///     lora_alpha: LoRA alpha (default 32.0)
///     epochs: Number of training epochs (default 3)
///     learning_rate: Learning rate (default 2e-4)
///     batch_size: Batch size (default 4)
///     max_seq_len: Maximum sequence length (default 2048)
///     output: Output directory (default "./output")
///
/// Returns:
///     dict with keys: final_loss, total_steps, total_tokens, output_dir, lora_weights_path
#[pyfunction]
#[pyo3(signature = (
    model_id,
    dataset_path,
    lora_r=16,
    lora_alpha=32.0,
    epochs=3,
    learning_rate=2e-4,
    batch_size=4,
    max_seq_len=2048,
    output="./output",
))]
#[allow(clippy::too_many_arguments)]
pub fn finetune<'py>(
    py: Python<'py>,
    model_id: &str,
    dataset_path: &str,
    lora_r: usize,
    lora_alpha: f32,
    epochs: usize,
    learning_rate: f64,
    batch_size: usize,
    max_seq_len: usize,
    output: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let model_id = model_id.to_string();
    let dataset_path = dataset_path.to_string();
    let output = output.to_string();

    let result = py
        .detach(move || {
            crate::hub::shared_runtime().block_on(async {
                let result = pmetal::easy::finetune(&model_id, &dataset_path)
                    .lora(lora_r, lora_alpha)
                    .epochs(epochs)
                    .learning_rate(learning_rate)
                    .batch_size(batch_size)
                    .max_seq_len(max_seq_len)
                    .output(&output)
                    .run()
                    .await
                    .map_err(|e| e.to_string())?;

                Ok::<_, String>((
                    result.final_loss,
                    result.total_steps,
                    result.total_tokens,
                    result.output_dir.to_string_lossy().to_string(),
                    result.lora_weights_path.to_string_lossy().to_string(),
                ))
            })
        })
        .map_err(runtime_err)?;

    let dict = PyDict::new(py);
    dict.set_item("final_loss", result.0)?;
    dict.set_item("total_steps", result.1)?;
    dict.set_item("total_tokens", result.2)?;
    dict.set_item("output_dir", result.3)?;
    dict.set_item("lora_weights_path", result.4)?;
    Ok(dict)
}

/// Run inference with a model.
///
/// Args:
///     model_id: HuggingFace model ID or local path
///     prompt: Text prompt
///     lora: Optional path to LoRA weights
///     max_tokens: Maximum tokens to generate (default 256)
///     temperature: Sampling temperature (default 0.7)
///     seed: Random seed for reproducibility
///
/// Returns:
///     Generated text string
#[pyfunction]
#[pyo3(signature = (model_id, prompt, lora=None, max_tokens=256, temperature=0.7, seed=None))]
pub fn infer(
    py: Python<'_>,
    model_id: &str,
    prompt: &str,
    lora: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    seed: Option<u64>,
) -> PyResult<String> {
    let model_id = model_id.to_string();
    let prompt = prompt.to_string();
    let lora = lora.map(String::from);

    py.detach(move || {
        crate::hub::shared_runtime().block_on(async {
            let mut builder = pmetal::easy::infer(&model_id)
                .temperature(temperature)
                .max_tokens(max_tokens);

            if let Some(lora_path) = lora {
                builder = builder.lora(lora_path);
            }

            if let Some(s) = seed {
                builder = builder.seed(s);
            }

            let result = builder.generate(&prompt).await.map_err(|e| e.to_string())?;

            Ok::<_, String>(result.text)
        })
    })
    .map_err(runtime_err)
}
