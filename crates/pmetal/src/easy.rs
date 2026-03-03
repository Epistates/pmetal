//! High-level builder API for fine-tuning and inference.
//!
//! This module provides ergonomic one-liner APIs that wrap the CLI's orchestration
//! logic into composable builder structs.
//!
//! # Fine-tuning
//!
//! ```rust,no_run
//! # async fn example() -> pmetal_core::Result<()> {
//! let result = pmetal::easy::finetune("Qwen/Qwen3-0.6B", "data.jsonl")
//!     .lora(16, 32.0)
//!     .epochs(3)
//!     .learning_rate(2e-4)
//!     .output("./output")
//!     .run()
//!     .await?;
//! println!("Final loss: {:.4}", result.final_loss);
//! # Ok(())
//! # }
//! ```
//!
//! # Inference
//!
//! ```rust,no_run
//! # async fn example() -> pmetal_core::Result<()> {
//! let result = pmetal::easy::infer("Qwen/Qwen3-0.6B")
//!     .temperature(0.7)
//!     .generate("What is 2+2?")
//!     .await?;
//! println!("{}", result.text);
//! # Ok(())
//! # }
//! ```

use std::path::{Path, PathBuf};

use pmetal_core::{LoraConfig, PMetalError, Result, TrainingConfig};
use pmetal_data::{DataLoaderConfig, DatasetFormat, Tokenizer, TrainingDataset};
use pmetal_lora::{DynamicLoraModel, TrainableModel};
use pmetal_models::{DynamicModel, GenerationConfig, generate_cached_async};
use pmetal_trainer::{CheckpointManager, TrainingLoop, TrainingLoopConfig};

/// Map any Display error to `PMetalError::Training`.
fn training_err(e: impl std::fmt::Display) -> PMetalError {
    PMetalError::Training(e.to_string())
}

/// Map any Display error to `PMetalError::Mlx`.
fn mlx_err(e: impl std::fmt::Display) -> PMetalError {
    PMetalError::Mlx(e.to_string())
}

/// Map any Display error to `PMetalError::ModelLoad`.
fn model_err(e: impl std::fmt::Display) -> PMetalError {
    PMetalError::ModelLoad(e.to_string())
}

/// Check if a string looks like a HuggingFace model ID (e.g., "org/model")
/// rather than a local filesystem path.
fn is_hf_model_id(s: &str) -> bool {
    !s.starts_with('/') && !s.starts_with('.') && s.contains('/')
}

/// Result of a fine-tuning run.
#[derive(Debug, Clone)]
pub struct FinetuneResult {
    /// Final training loss.
    pub final_loss: f64,
    /// Total training steps completed.
    pub total_steps: usize,
    /// Total tokens processed.
    pub total_tokens: usize,
    /// Output directory path.
    pub output_dir: PathBuf,
    /// Path to saved LoRA weights.
    pub lora_weights_path: PathBuf,
}

/// Result of an inference run.
#[derive(Debug, Clone)]
pub struct InferResult {
    /// Generated text.
    pub text: String,
    /// Number of tokens generated.
    pub tokens_generated: usize,
    /// Generation throughput.
    pub tokens_per_sec: f64,
}

/// Create a fine-tuning builder for the given model and dataset.
///
/// # Arguments
/// * `model` - HuggingFace model ID (e.g., `"Qwen/Qwen3-0.6B"`) or local path
/// * `dataset` - Path to JSONL training dataset
pub fn finetune(model: impl Into<String>, dataset: impl Into<String>) -> FinetuneBuilder {
    FinetuneBuilder::new(model.into(), dataset.into())
}

/// Create an inference builder for the given model.
///
/// # Arguments
/// * `model` - HuggingFace model ID (e.g., `"Qwen/Qwen3-0.6B"`) or local path
pub fn infer(model: impl Into<String>) -> InferBuilder {
    InferBuilder::new(model.into())
}

/// Builder for fine-tuning a model with LoRA.
///
/// Wraps the full training pipeline: model download, tokenizer loading,
/// dataset preparation, LoRA initialization, training loop, and weight saving.
pub struct FinetuneBuilder {
    model_id: String,
    dataset_path: String,
    eval_dataset_path: Option<String>,
    lora_r: usize,
    lora_alpha: f32,
    learning_rate: f64,
    batch_size: usize,
    num_epochs: usize,
    max_seq_len: usize,
    output_dir: String,
    flash_attention: bool,
    sequence_packing: bool,
    gradient_checkpointing: bool,
    gradient_checkpointing_layers: usize,
    metal_fused_optimizer: bool,
    embedding_lr: Option<f32>,
}

impl FinetuneBuilder {
    fn new(model_id: String, dataset_path: String) -> Self {
        Self {
            model_id,
            dataset_path,
            eval_dataset_path: None,
            lora_r: 16,
            lora_alpha: 32.0,
            learning_rate: 2e-4,
            batch_size: 4,
            num_epochs: 3,
            max_seq_len: 2048,
            output_dir: "./output".to_string(),
            flash_attention: true,
            sequence_packing: true,
            gradient_checkpointing: false,
            gradient_checkpointing_layers: 4,
            metal_fused_optimizer: false,
            embedding_lr: None,
        }
    }

    /// Set LoRA rank and alpha.
    pub fn lora(mut self, r: usize, alpha: f32) -> Self {
        self.lora_r = r;
        self.lora_alpha = alpha;
        self
    }

    /// Set number of training epochs.
    pub fn epochs(mut self, n: usize) -> Self {
        self.num_epochs = n;
        self
    }

    /// Set learning rate.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    /// Set maximum sequence length.
    pub fn max_seq_len(mut self, n: usize) -> Self {
        self.max_seq_len = n;
        self
    }

    /// Set output directory for checkpoints and final weights.
    pub fn output(mut self, path: impl Into<String>) -> Self {
        self.output_dir = path.into();
        self
    }

    /// Set evaluation dataset path.
    pub fn eval_dataset(mut self, path: impl Into<String>) -> Self {
        self.eval_dataset_path = Some(path.into());
        self
    }

    /// Enable or disable flash attention.
    pub fn flash_attention(mut self, enabled: bool) -> Self {
        self.flash_attention = enabled;
        self
    }

    /// Enable or disable sequence packing.
    pub fn sequence_packing(mut self, enabled: bool) -> Self {
        self.sequence_packing = enabled;
        self
    }

    /// Enable or disable gradient checkpointing.
    pub fn gradient_checkpointing(mut self, enabled: bool) -> Self {
        self.gradient_checkpointing = enabled;
        self
    }

    /// Enable or disable Metal fused optimizer.
    pub fn metal_fused_optimizer(mut self, enabled: bool) -> Self {
        self.metal_fused_optimizer = enabled;
        self
    }

    /// Set separate learning rate for embedding layers.
    pub fn embedding_lr(mut self, lr: f32) -> Self {
        self.embedding_lr = Some(lr);
        self
    }

    /// Run the fine-tuning pipeline.
    pub async fn run(self) -> Result<FinetuneResult> {
        // Resolve model path (download if HuggingFace ID)
        let model_path = resolve_model_path(&self.model_id).await?;

        // Load tokenizer (reads special_tokens_map.json and tokenizer_config.json too)
        let tokenizer = Tokenizer::from_model_dir(&model_path).map_err(|e| {
            PMetalError::ModelLoad(format!(
                "Failed to load tokenizer from {model_path:?}: {e}"
            ))
        })?;

        // Detect chat template
        let chat_template =
            pmetal_data::chat_templates::detect_chat_template(&model_path, &self.model_id);

        // Load and tokenize training dataset
        let train_dataset = TrainingDataset::from_jsonl_tokenized(
            &self.dataset_path,
            &tokenizer,
            DatasetFormat::Auto,
            self.max_seq_len,
            Some(&chat_template),
        )?;

        // Load evaluation dataset if provided
        let eval_dataset = if let Some(ref eval_path) = self.eval_dataset_path {
            Some(TrainingDataset::from_jsonl_tokenized(
                eval_path,
                &tokenizer,
                DatasetFormat::Auto,
                self.max_seq_len,
                Some(&chat_template),
            )?)
        } else {
            None
        };

        // Create LoRA config
        let lora_config = LoraConfig {
            r: self.lora_r,
            alpha: self.lora_alpha,
            ..Default::default()
        };

        // Initialize model with LoRA adapters
        let model =
            DynamicLoraModel::from_pretrained(&model_path, lora_config).map_err(model_err)?;

        // Set up checkpoint manager
        let output_dir = PathBuf::from(&self.output_dir);
        let checkpoint_dir = output_dir.join("checkpoints");
        let checkpoint_manager = CheckpointManager::new(&checkpoint_dir)
            .map_err(training_err)?
            .with_max_checkpoints(3);

        // Build training config
        let training_config = TrainingConfig {
            learning_rate: self.learning_rate,
            batch_size: self.batch_size,
            num_epochs: self.num_epochs,
            max_seq_len: self.max_seq_len,
            output_dir: self.output_dir.clone(),
            ..Default::default()
        };

        let dataloader_config = DataLoaderConfig {
            batch_size: self.batch_size,
            max_seq_len: self.max_seq_len,
            shuffle: true,
            seed: 42,
            pad_token_id: tokenizer.pad_token_id().unwrap_or(0),
            drop_last: false,
        };

        let training_loop_config = TrainingLoopConfig {
            training: training_config,
            dataloader: dataloader_config,
            use_metal_flash_attention: self.flash_attention,
            log_every: 10,
            checkpoint_every: 500,
            eval_every: if eval_dataset.is_some() { 100 } else { 0 },
            use_jit_compilation: false,
            use_sequence_packing: self.sequence_packing,
            gradient_checkpointing: self.gradient_checkpointing,
            gradient_checkpointing_layers: self.gradient_checkpointing_layers,
            embedding_lr: self.embedding_lr,
            eager_evaluation: false,
            use_metal_fused_optimizer: self.metal_fused_optimizer,
        };

        let mut training_loop = TrainingLoop::new(training_loop_config);

        // Run training (sequence packing by default)
        let model = if self.sequence_packing {
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

        // Save final LoRA weights
        let lora_weights_path = output_dir.join("lora_weights.safetensors");
        model
            .save_lora_weights(&lora_weights_path)
            .map_err(training_err)?;

        Ok(FinetuneResult {
            final_loss: training_loop.current_loss(),
            total_steps: training_loop.current_step(),
            total_tokens: training_loop.total_tokens(),
            output_dir,
            lora_weights_path,
        })
    }
}

/// Builder for running inference on a model.
///
/// Wraps the full inference pipeline: model download, tokenizer loading,
/// chat template detection, KV-cached generation, and decoding.
pub struct InferBuilder {
    model_id: String,
    lora_path: Option<String>,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    max_tokens: usize,
    repetition_penalty: f32,
    seed: Option<u64>,
    fp8: bool,
    #[cfg(feature = "ane")]
    device: Option<pmetal_core::Device>,
}

impl InferBuilder {
    fn new(model_id: String) -> Self {
        Self {
            model_id,
            lora_path: None,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            min_p: 0.05,
            max_tokens: 256,
            repetition_penalty: 1.0,
            seed: None,
            fp8: false,
            #[cfg(feature = "ane")]
            device: None,
        }
    }

    /// Set path to LoRA adapter weights.
    pub fn lora(mut self, path: impl Into<String>) -> Self {
        self.lora_path = Some(path.into());
        self
    }

    /// Set sampling temperature.
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Set top-k sampling.
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set top-p (nucleus) sampling.
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Set min-p sampling threshold.
    pub fn min_p(mut self, p: f32) -> Self {
        self.min_p = p;
        self
    }

    /// Set max tokens to generate.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    /// Enable FP8 quantization for inference.
    pub fn fp8(mut self, enabled: bool) -> Self {
        self.fp8 = enabled;
        self
    }

    /// Set random seed for reproducible generation.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Set the compute device (ANE inference requires `ane` feature).
    #[cfg(feature = "ane")]
    pub fn device(mut self, device: pmetal_core::Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Generate text from the given prompt.
    pub async fn generate(self, prompt: &str) -> Result<InferResult> {
        // Resolve model path
        let model_path = resolve_model_path(&self.model_id).await?;

        // Load tokenizer (reads special_tokens_map.json and tokenizer_config.json too)
        let tokenizer = Tokenizer::from_model_dir(&model_path).map_err(|e| {
            PMetalError::ModelLoad(format!(
                "Failed to load tokenizer from {model_path:?}: {e}"
            ))
        })?;

        // Branch: ANE inference
        #[cfg(feature = "ane")]
        if matches!(self.device, Some(pmetal_core::Device::Ane)) {
            return self.generate_ane(&model_path, &tokenizer, prompt);
        }

        // Branch: LoRA inference vs standard inference
        if let Some(lora_path) = &self.lora_path {
            return self.generate_with_lora(&model_path, lora_path, &tokenizer, prompt);
        }

        // Load model with auto-detected architecture
        let mut model = DynamicModel::load(&model_path).map_err(mlx_err)?;

        // Apply FP8 quantization if requested
        if self.fp8 {
            model.quantize_fp8().map_err(mlx_err)?;
        }

        // Tokenize
        let input_ids = tokenizer.encode(prompt)?;

        // Configure generation
        let gen_config = self.build_gen_config(&tokenizer);

        // Create KV cache
        let max_seq_len = input_ids.len() + self.max_tokens + 64;
        let mut cache = model.create_cache(max_seq_len);

        // Generate
        let start = std::time::Instant::now();
        let output = generate_cached_async(
            |input, cache| model.forward_with_hybrid_cache(input, None, Some(cache), None),
            &input_ids,
            gen_config,
            &mut cache,
        )
        .map_err(mlx_err)?;
        let elapsed = start.elapsed();

        // Decode generated tokens
        let prompt_len = input_ids.len();
        let generated_tokens = &output.token_ids[prompt_len..];
        let text = tokenizer.decode(generated_tokens)?;

        Ok(InferResult {
            text,
            tokens_generated: output.num_generated,
            tokens_per_sec: output.num_generated as f64 / elapsed.as_secs_f64(),
        })
    }

    /// Inference with LoRA adapter.
    fn generate_with_lora(
        &self,
        model_path: &Path,
        lora_path: &str,
        tokenizer: &Tokenizer,
        prompt: &str,
    ) -> Result<InferResult> {
        let lora_config = LoraConfig::default();
        let mut model =
            DynamicLoraModel::from_pretrained(model_path, lora_config).map_err(model_err)?;
        model.load_lora_weights(lora_path).map_err(model_err)?;

        let input_ids = tokenizer.encode(prompt)?;
        let gen_config = self.build_gen_config(tokenizer);

        let max_seq_len = input_ids.len() + self.max_tokens + 64;
        let mut cache = model.create_cache(max_seq_len).ok_or_else(|| {
            PMetalError::NotImplemented("Model does not support KV cache".to_string())
        })?;

        let start = std::time::Instant::now();
        let output = generate_cached_async(
            |input, cache| {
                model
                    .forward_with_cache(input, None, Some(cache))
                    .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))
            },
            &input_ids,
            gen_config,
            &mut cache,
        )
        .map_err(mlx_err)?;
        let elapsed = start.elapsed();

        let prompt_len = input_ids.len();
        let generated_tokens = &output.token_ids[prompt_len..];
        let text = tokenizer.decode(generated_tokens)?;

        Ok(InferResult {
            text,
            tokens_generated: output.num_generated,
            tokens_per_sec: output.num_generated as f64 / elapsed.as_secs_f64(),
        })
    }

    /// Inference via ANE with SafeTensors/flat weight loading, LoRA fusion, KV cache.
    #[cfg(feature = "ane")]
    fn generate_ane(
        &self,
        model_path: &Path,
        tokenizer: &Tokenizer,
        prompt: &str,
    ) -> Result<InferResult> {
        use pmetal_metal::ane::inference::{AneInferenceConfig, AneInferenceEngine};

        // Read model config
        let config_path = model_path.join("config.json");
        let config_text = std::fs::read_to_string(&config_path).map_err(|e| {
            PMetalError::ModelLoad(format!(
                "Failed to read config.json at {config_path:?}: {e}"
            ))
        })?;

        fn extract_usize(json: &str, key: &str) -> std::result::Result<usize, PMetalError> {
            let needle = format!("\"{key}\"");
            let pos = json
                .find(&needle)
                .ok_or_else(|| PMetalError::ModelLoad(format!("config.json missing '{key}'")))?;
            let after_key = &json[pos + needle.len()..];
            let num_start = after_key
                .find(|c: char| c.is_ascii_digit())
                .ok_or_else(|| {
                    PMetalError::ModelLoad(format!("config.json: no value for '{key}'"))
                })?;
            let num_str = &after_key[num_start..];
            let num_end = num_str
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(num_str.len());
            num_str[..num_end].parse::<usize>().map_err(|_| {
                PMetalError::ModelLoad(format!("config.json: invalid value for '{key}'"))
            })
        }

        fn extract_usize_optional(json: &str, key: &str) -> Option<usize> {
            extract_usize(json, key).ok()
        }

        let dim = extract_usize(&config_text, "hidden_size")?;
        let hidden_dim = extract_usize(&config_text, "intermediate_size")?;
        let n_heads = extract_usize(&config_text, "num_attention_heads")?;
        let n_layers = extract_usize(&config_text, "num_hidden_layers")?;
        let vocab_size = extract_usize(&config_text, "vocab_size")?;
        // GQA: read n_kv_heads, default to n_heads
        let n_kv_heads =
            extract_usize_optional(&config_text, "num_key_value_heads").unwrap_or(n_heads);

        // Parse rope_theta and rms_norm_eps from config.json
        fn extract_float(json: &str, key: &str) -> Option<f32> {
            let needle = format!("\"{key}\"");
            let pos = json.find(&needle)?;
            let after_key = &json[pos + needle.len()..];
            // Skip to colon, then whitespace
            let after_colon = after_key.find(':').map(|i| &after_key[i + 1..])?;
            let trimmed = after_colon.trim_start();
            // Extract the numeric value (may contain digits, '.', 'e', 'E', '+', '-')
            let end = trimmed
                .find(|c: char| {
                    !c.is_ascii_digit() && c != '.' && c != 'e' && c != 'E' && c != '+' && c != '-'
                })
                .unwrap_or(trimmed.len());
            trimmed[..end].parse::<f64>().ok().map(|v| v as f32)
        }
        let rope_theta = extract_float(&config_text, "rope_theta").unwrap_or(1_000_000.0);
        let rms_norm_eps = extract_float(&config_text, "rms_norm_eps").unwrap_or(1e-6);
        let head_dim = extract_usize_optional(&config_text, "head_dim");

        let ane_config = AneInferenceConfig {
            dim,
            hidden_dim,
            n_heads,
            n_kv_heads,
            n_layers,
            vocab_size,
            max_seq_len: 256,
            temperature: self.temperature,
            top_k: self.top_k,
            max_tokens: self.max_tokens,
            eos_token_id: tokenizer.eos_token_id(),
            rope_theta,
            rms_norm_eps,
            head_dim,
            ..Default::default()
        };

        let mut engine = AneInferenceEngine::new(ane_config)
            .map_err(|e| PMetalError::ModelLoad(format!("ANE engine init failed: {e}")))?;

        // Try SafeTensors first, fall back to model.bin
        let safetensors_single = model_path.join("model.safetensors");
        let safetensors_multi = model_path.join("model-00001-of-00002.safetensors");
        let safetensors_index = model_path.join("model.safetensors.index.json");
        let weights_bin = model_path.join("model.bin");

        if safetensors_single.exists() {
            engine
                .load_weights_safetensors(&safetensors_single)
                .map_err(|e| PMetalError::ModelLoad(format!("SafeTensors load failed: {e}")))?;
        } else if safetensors_index.exists() || safetensors_multi.exists() {
            engine
                .load_weights_safetensors(model_path)
                .map_err(|e| PMetalError::ModelLoad(format!("SafeTensors load failed: {e}")))?;
        } else if weights_bin.exists() {
            let weight_data = std::fs::read(&weights_bin)
                .map_err(|e| PMetalError::ModelLoad(format!("Failed to read model.bin: {e}")))?;
            if weight_data.len() % 4 != 0 {
                return Err(PMetalError::ModelLoad(
                    "model.bin size must be a multiple of 4 bytes".into(),
                ));
            }
            #[allow(unsafe_code)]
            let (prefix, weights, suffix) = unsafe { weight_data.align_to::<f32>() };
            if !prefix.is_empty() || !suffix.is_empty() {
                return Err(PMetalError::ModelLoad(
                    "model.bin data is not properly aligned for f32".into(),
                ));
            }
            engine.load_weights_flat(weights);
        } else {
            return Err(PMetalError::ModelLoad(format!(
                "No weight files found in {:?}. Expected model.safetensors or model.bin.",
                model_path
            )));
        }

        // Check for LoRA adapter and merge if present
        let adapter_config = model_path.join("adapter_config.json");
        let adapter_weights = model_path.join("adapter_model.safetensors");
        if adapter_config.exists() && adapter_weights.exists() {
            engine
                .load_lora_adapter(model_path)
                .map_err(|e| PMetalError::ModelLoad(format!("LoRA merge failed: {e}")))?;
        }

        // Also check user-specified LoRA path
        if let Some(ref lora_path) = self.lora_path {
            let lora_dir = Path::new(lora_path);
            engine
                .load_lora_adapter(lora_dir)
                .map_err(|e| PMetalError::ModelLoad(format!("LoRA merge failed: {e}")))?;
        }

        // Compile kernels (after LoRA merge)
        engine
            .compile_kernels()
            .map_err(|e| PMetalError::Training(format!("ANE compile failed: {e}")))?;

        // Tokenize
        let input_ids_u32 = tokenizer.encode(prompt)?;

        // Generate with KV cache
        let start = std::time::Instant::now();
        let output_ids = engine
            .generate_cached(&input_ids_u32)
            .map_err(|e| PMetalError::Training(format!("ANE generate failed: {e}")))?;
        let elapsed = start.elapsed();

        // Decode generated tokens
        let prompt_len = input_ids_u32.len();
        let generated: Vec<u32> = output_ids[prompt_len..].to_vec();
        let text = tokenizer.decode(&generated)?;

        Ok(InferResult {
            text,
            tokens_generated: generated.len(),
            tokens_per_sec: generated.len() as f64 / elapsed.as_secs_f64(),
        })
    }

    /// Build a `GenerationConfig` from builder settings.
    fn build_gen_config(&self, tokenizer: &Tokenizer) -> GenerationConfig {
        let mut config = if self.temperature < 1e-6 {
            GenerationConfig::greedy(self.max_tokens)
        } else {
            GenerationConfig::sampling(self.max_tokens, self.temperature)
                .with_top_k(self.top_k)
                .with_top_p(self.top_p)
                .with_min_p(self.min_p)
                .with_repetition_penalty(self.repetition_penalty)
        };

        // Use actual EOS token, don't hardcode a fallback
        if let Some(eos) = tokenizer.eos_token_id() {
            config = config.with_stop_tokens(vec![eos]);
        }

        if let Some(seed) = self.seed {
            config = config.with_seed(seed);
        }

        config
    }
}

/// Resolve a model ID to a local path, downloading from HuggingFace Hub if necessary.
async fn resolve_model_path(model_id: &str) -> Result<PathBuf> {
    if is_hf_model_id(model_id) {
        // HuggingFace model ID — download_model fetches all repo files
        let path = pmetal_hub::download_model(model_id, None, None).await?;
        Ok(path)
    } else {
        Ok(PathBuf::from(model_id))
    }
}
