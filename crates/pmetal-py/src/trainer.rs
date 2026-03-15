//! Python wrapper for the training loop.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use tracing::info_span;

use crate::callbacks::{
    PyLoggingCallback, PyMetricsJsonCallback, PyProgressCallback, PythonCallbackBridge,
};
use crate::config::{PyLoraConfig, PyTrainingConfig};
use crate::error::pmetal_to_pyerr;
use crate::hub::is_hf_model_id;

/// Map any Display error to `PMetalError::Training`.
fn training_err(e: impl std::fmt::Display) -> pmetal_core::PMetalError {
    pmetal_core::PMetalError::Training(e.to_string())
}

/// Map any Display error to `PMetalError::ModelLoad`.
fn model_err(e: impl std::fmt::Display) -> pmetal_core::PMetalError {
    pmetal_core::PMetalError::ModelLoad(e.to_string())
}

fn build_training_callbacks(
    py: Python<'_>,
    py_callbacks: &[Py<PyAny>],
) -> PyResult<Vec<Box<dyn pmetal_core::TrainingCallback>>> {
    let mut rust_callbacks: Vec<Box<dyn pmetal_core::TrainingCallback>> = Vec::new();
    let mut bridged_callbacks = Vec::new();

    for callback in py_callbacks {
        let bound = callback.bind(py);

        if let Ok(progress) = bound.extract::<PyRef<'_, PyProgressCallback>>() {
            rust_callbacks.push(Box::new(pmetal_trainer::ProgressCallback::new(
                progress.total_steps,
            )));
            continue;
        }

        if let Ok(logging) = bound.extract::<PyRef<'_, PyLoggingCallback>>() {
            rust_callbacks.push(Box::new(pmetal_trainer::LoggingCallback::new(
                logging.log_every,
            )));
            continue;
        }

        if let Ok(metrics) = bound.extract::<PyRef<'_, PyMetricsJsonCallback>>() {
            let callback = pmetal_trainer::MetricsJsonCallback::new(&metrics.path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            rust_callbacks.push(Box::new(callback));
            continue;
        }

        bridged_callbacks.push(callback.clone_ref(py));
    }

    if !bridged_callbacks.is_empty() {
        rust_callbacks.push(Box::new(PythonCallbackBridge {
            py_callbacks: bridged_callbacks,
        }));
    }

    Ok(rust_callbacks)
}

#[pyclass(name = "Trainer")]
pub struct PyTrainer {
    model_id: String,
    lora_config: pmetal_core::LoraConfig,
    training_config: pmetal_core::TrainingConfig,
    dataset_path: String,
    eval_dataset_path: Option<String>,
    flash_attention: bool,
    sequence_packing: bool,
    gradient_checkpointing: bool,
    metal_fused_optimizer: bool,
    embedding_lr: Option<f32>,
    py_callbacks: Vec<Py<PyAny>>,
}

#[pymethods]
impl PyTrainer {
    /// Create a new trainer.
    ///
    /// Args:
    ///     model_id: HuggingFace model ID or local path
    ///     lora_config: LoRA configuration
    ///     training_config: Training hyperparameters
    ///     dataset_path: Path to JSONL training dataset
    ///     eval_dataset_path: Optional path to evaluation dataset
    #[new]
    #[pyo3(signature = (model_id, lora_config, training_config, dataset_path, eval_dataset_path=None))]
    fn new(
        model_id: &str,
        lora_config: &PyLoraConfig,
        training_config: &PyTrainingConfig,
        dataset_path: &str,
        eval_dataset_path: Option<&str>,
    ) -> Self {
        Self {
            model_id: model_id.to_string(),
            lora_config: lora_config.0.clone(),
            training_config: training_config.0.clone(),
            dataset_path: dataset_path.to_string(),
            eval_dataset_path: eval_dataset_path.map(String::from),
            flash_attention: true,
            sequence_packing: true,
            gradient_checkpointing: false,
            metal_fused_optimizer: false,
            embedding_lr: None,
            py_callbacks: Vec::new(),
        }
    }

    /// Add a Python callback object.
    ///
    /// Built-in PMetal callback classes are mapped to their native Rust
    /// implementations. Arbitrary Python objects are bridged through the
    /// training callback interface if they implement callback methods.
    fn add_callback(&mut self, callback: Py<PyAny>) {
        self.py_callbacks.push(callback);
    }

    /// Enable or disable sequence packing.
    fn set_sequence_packing(&mut self, enabled: bool) {
        self.sequence_packing = enabled;
    }

    /// Enable or disable gradient checkpointing.
    fn set_gradient_checkpointing(&mut self, enabled: bool) {
        self.gradient_checkpointing = enabled;
    }

    /// Enable or disable Metal fused optimizer.
    fn set_metal_fused_optimizer(&mut self, enabled: bool) {
        self.metal_fused_optimizer = enabled;
    }

    /// Set separate embedding learning rate.
    fn set_embedding_lr(&mut self, lr: f32) {
        self.embedding_lr = Some(lr);
    }

    /// Run the training loop.
    ///
    /// Returns:
    ///     dict with keys: final_loss, total_steps, total_tokens, output_dir, lora_weights_path
    fn train<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let rust_callbacks = build_training_callbacks(py, &self.py_callbacks)?;

        // Capture all settings before releasing the GIL
        let model_id = self.model_id.clone();
        let lora_config = self.lora_config.clone();
        let training_config = self.training_config.clone();
        let dataset_path = self.dataset_path.clone();
        let eval_dataset_path = self.eval_dataset_path.clone();
        let flash_attention = self.flash_attention;
        let sequence_packing = self.sequence_packing;
        let gradient_checkpointing = self.gradient_checkpointing;
        let metal_fused_optimizer = self.metal_fused_optimizer;
        let embedding_lr = self.embedding_lr;
        let mut rust_callbacks = rust_callbacks;

        // Use PMetalError as the intermediate error type to preserve exception fidelity
        let result = py
            .detach(move || {
                crate::hub::shared_runtime().block_on(async {
                    // Resolve model path
                    let model_path = {
                        let _span = info_span!("model_resolve", model_id = %model_id).entered();
                        if is_hf_model_id(&model_id) {
                            let path = pmetal_hub::download_model(&model_id, None, None).await?;
                            let _ =
                                pmetal_hub::download_file(&model_id, "tokenizer.json", None, None)
                                    .await;
                            let _ = pmetal_hub::download_file(
                                &model_id,
                                "tokenizer_config.json",
                                None,
                                None,
                            )
                            .await;
                            path
                        } else {
                            PathBuf::from(&model_id)
                        }
                    };

                    // Load tokenizer (with config-aware special token resolution)
                    let tokenizer = {
                        let _span =
                            info_span!("load_tokenizer", path = %model_path.display()).entered();
                        pmetal_data::Tokenizer::from_model_dir(&model_path)?
                    };

                    // Detect chat template
                    let chat_template =
                        pmetal_data::chat_templates::detect_chat_template(&model_path, &model_id);

                    // Load dataset
                    let (train_dataset, eval_dataset) = {
                        let _span = info_span!(
                            "load_dataset",
                            path = %dataset_path,
                            has_eval = eval_dataset_path.is_some(),
                        )
                        .entered();

                        let train = pmetal_data::TrainingDataset::from_jsonl_tokenized(
                            &dataset_path,
                            &tokenizer,
                            pmetal_data::DatasetFormat::Auto,
                            training_config.max_seq_len,
                            Some(&chat_template),
                        )?;

                        let eval = if let Some(ref eval_path) = eval_dataset_path {
                            Some(pmetal_data::TrainingDataset::from_jsonl_tokenized(
                                eval_path,
                                &tokenizer,
                                pmetal_data::DatasetFormat::Auto,
                                training_config.max_seq_len,
                                Some(&chat_template),
                            )?)
                        } else {
                            None
                        };

                        (train, eval)
                    };

                    // Load model
                    let model = {
                        let _span =
                            info_span!("load_model", path = %model_path.display()).entered();
                        pmetal_lora::DynamicLoraModel::from_pretrained(&model_path, lora_config)
                            .map_err(model_err)?
                    };

                    // Set up checkpoint manager
                    let output_dir = PathBuf::from(&training_config.output_dir);
                    std::fs::create_dir_all(&output_dir)?;
                    let checkpoint_dir = output_dir.join("checkpoints");
                    let checkpoint_manager =
                        pmetal_trainer::CheckpointManager::new(&checkpoint_dir)
                            .map_err(training_err)?
                            .with_max_checkpoints(3);

                    // Build training loop config
                    let dataloader_config = pmetal_data::DataLoaderConfig {
                        batch_size: training_config.batch_size,
                        max_seq_len: training_config.max_seq_len,
                        shuffle: true,
                        seed: training_config.seed,
                        pad_token_id: tokenizer.pad_token_id().unwrap_or(0),
                        drop_last: false,
                    };

                    let loop_config = pmetal_trainer::TrainingLoopConfig {
                        training: training_config.clone(),
                        dataloader: dataloader_config,
                        use_metal_flash_attention: flash_attention,
                        log_every: 10,
                        checkpoint_every: training_config.save_steps.unwrap_or(500),
                        eval_every: if eval_dataset.is_some() { 100 } else { 0 },
                        use_jit_compilation: false,
                        use_sequence_packing: sequence_packing,
                        gradient_checkpointing,
                        gradient_checkpointing_layers: 4,
                        embedding_lr,
                        eager_evaluation: false,
                        use_metal_fused_optimizer: metal_fused_optimizer,
                    };

                    let mut training_loop = pmetal_trainer::TrainingLoop::new(loop_config);
                    for callback in rust_callbacks.drain(..) {
                        training_loop.add_callback(callback);
                    }

                    // Run training
                    let model = {
                        let _span = info_span!(
                            "training_loop",
                            model = %model_path.display(),
                            sequence_packing,
                        )
                        .entered();

                        if sequence_packing {
                            training_loop
                                .run_packed(
                                    model,
                                    train_dataset,
                                    eval_dataset,
                                    Some(&checkpoint_manager),
                                )
                                .map_err(training_err)?
                        } else {
                            let mut model = model;
                            training_loop
                                .run(
                                    &mut model,
                                    train_dataset,
                                    eval_dataset,
                                    Some(&checkpoint_manager),
                                )
                                .map_err(training_err)?;
                            model
                        }
                    };

                    // Save weights
                    let weights_path = {
                        let _span = info_span!("save_weights", output_dir = %output_dir.display())
                            .entered();
                        use pmetal_lora::TrainableModel;
                        let path = output_dir.join("lora_weights.safetensors");
                        model.save_lora_weights(&path).map_err(training_err)?;
                        path
                    };

                    Ok::<_, pmetal_core::PMetalError>((
                        training_loop.current_loss(),
                        training_loop.current_step(),
                        training_loop.total_tokens(),
                        output_dir.to_string_lossy().to_string(),
                        weights_path.to_string_lossy().to_string(),
                    ))
                })
            })
            .map_err(pmetal_to_pyerr)?;

        // Build Python dict result
        let dict = PyDict::new(py);
        dict.set_item("final_loss", result.0)?;
        dict.set_item("total_steps", result.1)?;
        dict.set_item("total_tokens", result.2)?;
        dict.set_item("output_dir", result.3)?;
        dict.set_item("lora_weights_path", result.4)?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "Trainer(model='{}', dataset='{}', lora_r={}, epochs={})",
            self.model_id, self.dataset_path, self.lora_config.r, self.training_config.num_epochs,
        )
    }
}
