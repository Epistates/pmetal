//! Python wrapper for the training loop.

use std::path::PathBuf;

use pyo3::prelude::*;

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
    /// Note: Callbacks are not yet wired into the training loop.
    /// This method stores the callback for future use.
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
    fn train(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        // Warn if callbacks are registered but not yet functional
        if !self.py_callbacks.is_empty() {
            let warnings = py.import("warnings")?;
            warnings.call_method1(
                "warn",
                ("Callbacks are registered but not yet connected to the training loop. They will not fire.",),
            )?;
        }

        // Capture all settings before entering allow_threads
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

        // Use PMetalError as the intermediate error type to preserve exception fidelity
        let result = py
            .allow_threads(move || {
                crate::hub::shared_runtime().block_on(async {
                    // Resolve model path
                    let model_path = if is_hf_model_id(&model_id) {
                        let path = pmetal_hub::download_model(&model_id, None, None).await?;
                        let _ = pmetal_hub::download_file(&model_id, "tokenizer.json", None, None)
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
                    };

                    // Load tokenizer
                    let tokenizer_path = model_path.join("tokenizer.json");
                    let tokenizer = pmetal_data::Tokenizer::from_file(&tokenizer_path)?;

                    // Detect chat template
                    let chat_template =
                        pmetal_data::chat_templates::detect_chat_template(&model_path, &model_id);

                    // Load dataset
                    let train_dataset = pmetal_data::TrainingDataset::from_jsonl_tokenized(
                        &dataset_path,
                        &tokenizer,
                        pmetal_data::DatasetFormat::Auto,
                        training_config.max_seq_len,
                        Some(&chat_template),
                    )?;

                    let eval_dataset = if let Some(ref eval_path) = eval_dataset_path {
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

                    // Load model
                    let model =
                        pmetal_lora::DynamicLoraModel::from_pretrained(&model_path, lora_config)
                            .map_err(model_err)?;

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

                    // Run training
                    let model = if sequence_packing {
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
                    };

                    // Save weights
                    use pmetal_lora::TrainableModel;
                    let weights_path = output_dir.join("lora_weights.safetensors");
                    model
                        .save_lora_weights(&weights_path)
                        .map_err(training_err)?;

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
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("final_loss", result.0)?;
        dict.set_item("total_steps", result.1)?;
        dict.set_item("total_tokens", result.2)?;
        dict.set_item("output_dir", result.3)?;
        dict.set_item("lora_weights_path", result.4)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "Trainer(model='{}', dataset='{}', lora_r={}, epochs={})",
            self.model_id, self.dataset_path, self.lora_config.r, self.training_config.num_epochs,
        )
    }
}
