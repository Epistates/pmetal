//! PMetal CLI - LLM fine-tuning for Apple Silicon.

use std::path::{Component, Path, PathBuf};

use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use pmetal_core::{DatasetConfig, LoraConfig, ModelConfig, TrainingConfig};
use pmetal_data::{DataLoaderConfig, DatasetFormat, Tokenizer, TrainingDataset};
use pmetal_lora::{
    DynamicLoraModel, LlamaLoraForCausalLM, LlamaQloraForCausalLM, QLoraConfig, TrainableModel,
};
use pmetal_mlx::quantization::QuantScheme;
use pmetal_models::architectures::llama::LlamaConfig;
use pmetal_models::ollama::{templates as ollama_templates, ModelfileBuilder};
use pmetal_models::WeightFormat;
use pmetal_trainer::{CheckpointManager, MetricsJsonCallback, TrainingLoop, TrainingLoopConfig};
use serde::{Deserialize, Serialize};

/// Combined configuration for training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullTrainingConfig {
    /// Model configuration.
    #[serde(default)]
    pub model: ModelConfig,

    /// LoRA configuration.
    #[serde(default)]
    pub lora: LoraConfig,

    /// Training hyperparameters.
    #[serde(default)]
    pub training: TrainingConfig,

    /// Dataset configuration.
    #[serde(default)]
    pub dataset: DatasetConfig,
}

impl Default for FullTrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            lora: LoraConfig::default(),
            training: TrainingConfig::default(),
            dataset: DatasetConfig::default(),
        }
    }
}

/// Quantization method for QLoRA.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum QuantizationMethod {
    /// No quantization (standard LoRA)
    #[default]
    None,
    /// NF4 (Normal Float 4-bit) - recommended, optimal for normally distributed weights
    Nf4,
    /// FP4 (Float Point 4-bit)
    Fp4,
    /// INT8 (8-bit integer)
    Int8,
}

#[derive(Parser)]
#[command(name = "pmetal")]
#[command(author, version, about = "LLM fine-tuning optimized for Apple Silicon", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fine-tune a model using LoRA/QLoRA
    Train {
        /// Path to training configuration file (YAML)
        #[arg(short, long)]
        config: Option<String>,

        /// Model ID (HuggingFace or local path)
        #[arg(short, long)]
        model: Option<String>,

        /// Dataset path (JSONL file)
        #[arg(short, long)]
        dataset: Option<String>,

        /// Evaluation dataset path (optional JSONL file)
        #[arg(long)]
        eval_dataset: Option<String>,

        /// Output directory
        #[arg(short, long, default_value = "./output")]
        output: String,

        /// LoRA rank
        #[arg(long, default_value = "16")]
        lora_r: usize,

        /// Learning rate
        #[arg(long, default_value = "2e-4")]
        learning_rate: f64,

        /// Batch size
        #[arg(long, default_value = "4")]
        batch_size: usize,

        /// Number of epochs
        #[arg(long, default_value = "3")]
        epochs: usize,

        /// Maximum sequence length
        #[arg(long, default_value = "2048")]
        max_seq_len: usize,

        /// Gradient accumulation steps
        #[arg(long, default_value = "1")]
        gradient_accumulation_steps: usize,

        /// Use Metal FlashAttention for efficient O(n) memory forward pass
        #[arg(long)]
        use_metal_flash_attention: bool,

        /// Maximum gradient norm for clipping (0 to disable)
        #[arg(long, default_value = "1.0")]
        max_grad_norm: f64,

        /// Resume from checkpoint
        #[arg(long)]
        resume: bool,

        /// Quantization method for QLoRA (none, nf4, fp4, int8)
        #[arg(long, value_enum, default_value = "none")]
        quantization: QuantizationMethod,

        /// Block size for quantization (default: 64)
        #[arg(long, default_value = "64")]
        quant_block_size: usize,

        /// Enable double quantization for absmax values
        #[arg(long)]
        double_quant: bool,

        /// Use fused training step (combines forward/backward/optimizer in single call)
        /// Provides better throughput but requires gradient_accumulation_steps=1
        #[arg(long)]
        fused: bool,

        /// Use Metal fused optimizer for maximum throughput.
        /// Uses custom Metal kernels that process all parameters in a single dispatch.
        /// Expected ~40% throughput improvement. Requires Apple Silicon.
        #[arg(long)]
        use_metal_fused_optimizer: bool,

        /// Enable sequence packing for 2-5x throughput on variable-length datasets.
        /// Packs multiple shorter sequences into single batches with block-diagonal
        /// attention masks to prevent cross-sequence attention.
        #[arg(long)]
        use_sequence_packing: bool,

        /// Enable JIT compilation for kernel fusion (up to 50% throughput improvement).
        /// Compiles training step into optimized Metal kernels after optimizer warmup.
        #[arg(long)]
        use_jit_compilation: bool,

        /// Enable gradient checkpointing to reduce memory usage.
        /// Trades compute for memory (~30% slower) but allows ~2x larger batch sizes.
        #[arg(long)]
        gradient_checkpointing: bool,

        /// Number of layers per checkpoint block (default: 4).
        /// Lower = more memory savings but slower. Only used with --gradient-checkpointing.
        #[arg(long, default_value = "4")]
        gradient_checkpointing_layers: usize,

        /// Path to log training metrics as JSONL (Wandb-compatible).
        /// Metrics can be imported to Wandb: `wandb sync path/to/metrics.jsonl`
        #[arg(long)]
        log_metrics: Option<String>,

        /// Separate learning rate for embedding layers.
        /// Unsloth recommends 5e-5 for embeddings vs 2e-4 for LoRA params.
        /// Improves training stability for large vocabulary models.
        #[arg(long)]
        embedding_lr: Option<f32>,
    },

    /// Run inference with a model
    Infer {
        /// Model ID or path
        #[arg(short, long)]
        model: String,

        /// LoRA adapter path (optional)
        #[arg(long)]
        lora: Option<String>,

        /// Input prompt
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long, default_value = "256")]
        max_tokens: usize,

        /// Temperature for sampling (0 = greedy). Defaults to model's generation_config.json
        #[arg(long)]
        temperature: Option<f32>,

        /// Top-k sampling (0 = disabled). Defaults to model's generation_config.json
        #[arg(long)]
        top_k: Option<usize>,

        /// Top-p nucleus sampling (0.0-1.0). Defaults to model's generation_config.json
        #[arg(long)]
        top_p: Option<f32>,

        /// Min-p dynamic sampling (0.0 = disabled). Defaults to model's generation_config.json
        #[arg(long)]
        min_p: Option<f32>,

        /// Repetition penalty applied to prompt + output (1.0 = disabled, 1.0-1.2 typical)
        #[arg(long)]
        repetition_penalty: Option<f32>,

        /// Frequency penalty proportional to token count (0.0 = disabled, 0.0-2.0 typical)
        #[arg(long)]
        frequency_penalty: Option<f32>,

        /// Presence penalty for any appeared token (0.0 = disabled, Qwen3 recommends 0-2)
        #[arg(long)]
        presence_penalty: Option<f32>,

        /// Random seed for reproducible generation
        #[arg(long)]
        seed: Option<u64>,

        /// Apply chat template (auto-detected from tokenizer)
        #[arg(long)]
        chat: bool,

        /// System message for chat mode
        #[arg(long)]
        system: Option<String>,

        /// Disable thinking mode for models that support it (e.g., Qwen3)
        /// By default, the model decides when to use thinking based on query complexity
        #[arg(long)]
        no_thinking: bool,

        /// Use fused Metal sampling kernel for better battery performance
        /// (bypasses mlx-rs sampling, uses single GPU kernel launch)
        #[arg(long)]
        metal_sampler: bool,

        /// Use JIT-compiled sampling for better performance
        /// (matches mlx_lm's @mx.compile approach for kernel fusion)
        #[arg(long)]
        compiled: bool,

        /// Use dedicated GPU stream for generation (like mlx_lm's generation_stream)
        /// (may improve pipelining and reduce scheduling overhead)
        #[arg(long)]
        stream: bool,

        /// Use minimal async generation (for performance debugging)
        #[arg(long)]
        minimal: bool,

        /// Show thinking content in output (if model generates it)
        #[arg(long)]
        show_thinking: bool,
    },

    /// Download a model from HuggingFace
    Download {
        /// Model ID
        model: String,

        /// Specific revision
        #[arg(long)]
        revision: Option<String>,
    },

    /// Show memory usage and available capacity
    Memory,

    /// Benchmark FFI overhead (for performance analysis)
    BenchFfi,

    /// Benchmark generation loop timing (detailed profiling)
    BenchGen {
        /// Model to benchmark
        #[arg(short, long, default_value = "unsloth/Qwen3-0.6B")]
        model: String,
    },

    /// Benchmark training performance
    Bench {
        /// Model to benchmark
        #[arg(short, long, default_value = "meta-llama/Llama-3.2-1B")]
        model: String,

        /// Batch size
        #[arg(short, long, default_value = "1")]
        batch_size: usize,

        /// Sequence length
        #[arg(short, long, default_value = "512")]
        seq_len: usize,
    },

    /// Generate a sample configuration file
    Init {
        /// Output path for the config file
        #[arg(short, long, default_value = "config.yaml")]
        output: String,
    },

    /// Export trained model for Ollama
    Ollama {
        #[command(subcommand)]
        action: OllamaAction,
    },

    /// Quantize a model to GGUF format (supports Dynamic 2.0)
    Quantize {
        /// Source model path (Safetensors/HF)
        #[arg(short, long)]
        model: String,

        /// Output GGUF file path
        #[arg(short, long)]
        output: String,

        /// Path to Importance Matrix (imatrix.dat) for dynamic quantization
        #[arg(long)]
        imatrix: Option<String>,

        /// Quantization method (e.g., q4_k_m, q8_0) or "dynamic"
        #[arg(long, default_value = "dynamic")]
        method: String,
    },
}

/// Ollama subcommands for model export and registration.
#[derive(Subcommand)]
enum OllamaAction {
    /// Generate a Modelfile for a trained model
    Modelfile {
        /// Base model (GGUF path or Ollama model name)
        #[arg(short, long)]
        base: String,

        /// LoRA adapter path (optional)
        #[arg(long)]
        lora: Option<String>,

        /// Output Modelfile path
        #[arg(short, long, default_value = "Modelfile")]
        output: String,

        /// System prompt
        #[arg(long)]
        system: Option<String>,

        /// Temperature (0.0-2.0)
        #[arg(long)]
        temperature: Option<f32>,

        /// Context window size
        #[arg(long)]
        num_ctx: Option<i32>,

        /// Top-k sampling
        #[arg(long)]
        top_k: Option<i32>,

        /// Top-p nucleus sampling
        #[arg(long)]
        top_p: Option<f32>,

        /// Model template (auto-detected from architecture if not specified)
        #[arg(long, value_enum)]
        template: Option<OllamaTemplate>,

        /// License text for the model
        #[arg(long)]
        license: Option<String>,
    },

    /// Create and register a model with Ollama
    Create {
        /// Model name for Ollama (e.g., "my-finetuned-model")
        #[arg(short, long)]
        name: String,

        /// Base model (GGUF path or Ollama model name)
        #[arg(short, long)]
        base: String,

        /// LoRA adapter path (optional)
        #[arg(long)]
        lora: Option<String>,

        /// System prompt
        #[arg(long)]
        system: Option<String>,

        /// Temperature (0.0-2.0)
        #[arg(long)]
        temperature: Option<f32>,

        /// Context window size
        #[arg(long)]
        num_ctx: Option<i32>,

        /// Model template (auto-detected from architecture if not specified)
        #[arg(long, value_enum)]
        template: Option<OllamaTemplate>,
    },

    /// List available templates
    Templates,
}

/// Ollama template presets.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OllamaTemplate {
    /// Llama 3 chat format
    Llama3,
    /// Qwen3/ChatML format
    Qwen3,
    /// Gemma instruct format
    Gemma,
    /// Mistral instruct format
    Mistral,
    /// Phi-3 instruct format
    Phi3,
    /// DeepSeek chat format
    DeepSeek,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            config,
            model,
            dataset,
            eval_dataset,
            output,
            lora_r,
            learning_rate,
            batch_size,
            epochs,
            max_seq_len,
            gradient_accumulation_steps,
            use_metal_flash_attention,
            max_grad_norm,
            resume,
            quantization,
            quant_block_size,
            double_quant,
            fused,
            use_metal_fused_optimizer,
            use_sequence_packing,
            use_jit_compilation,
            gradient_checkpointing,
            gradient_checkpointing_layers,
            log_metrics,
            embedding_lr,
        } => {
            run_training(
                config,
                model,
                dataset,
                eval_dataset,
                output,
                lora_r,
                learning_rate,
                batch_size,
                epochs,
                max_seq_len,
                gradient_accumulation_steps,
                use_metal_flash_attention,
                max_grad_norm,
                resume,
                quantization,
                quant_block_size,
                double_quant,
                fused,
                use_metal_fused_optimizer,
                use_sequence_packing,
                use_jit_compilation,
                gradient_checkpointing,
                gradient_checkpointing_layers,
                log_metrics,
                embedding_lr,
            )
            .await?;
        }

        Commands::Infer {
            model,
            lora,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            seed,
            chat,
            system,
            no_thinking,
            metal_sampler,
            compiled,
            stream,
            minimal,
            show_thinking,
        } => {
            run_inference(
                &model,
                lora.as_deref(),
                &prompt,
                max_tokens,
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                frequency_penalty,
                presence_penalty,
                seed,
                chat,
                system.as_deref(),
                no_thinking,
                metal_sampler,
                compiled,
                stream,
                minimal,
                show_thinking,
            )
            .await?;
        }

        Commands::Download { model, revision } => {
            tracing::info!(model = %model, "Downloading model");
            let path = pmetal_hub::download_model(&model, revision.as_deref(), None).await?;
            println!("Model downloaded to: {}", path.display());
        }

        Commands::Memory => {
            let stats = pmetal_mlx::memory::get_memory_stats();
            println!("Memory Statistics:");
            println!("  Total:     {:.2} GB", stats.total_gb());
            println!("  Used:      {:.2} GB", stats.used_gb());
            println!("  Available: {:.2} GB", stats.available_gb());
            println!("  Peak:      {:.2} GB", stats.peak_gb());
        }

        Commands::Bench {
            model,
            batch_size,
            seq_len,
        } => {
            run_benchmark(&model, batch_size, seq_len).await?;
        }

        Commands::Init { output } => {
            // Validate output path to prevent path traversal attacks
            let validated_output = validate_output_path(&output, "config output")?;
            generate_sample_config(&validated_output.to_string_lossy())?;
        }

        Commands::BenchFfi => {
            run_ffi_benchmark()?;
        }

        Commands::BenchGen { model } => {
            run_gen_benchmark(&model).await?;
        }

        Commands::Ollama { action } => {
            run_ollama_command(action).await?;
        }

        Commands::Quantize {
            model,
            output,
            imatrix,
            method,
        } => {
            run_quantization(&model, &output, imatrix.as_deref(), &method).await?;
        }
    }

    Ok(())
}

/// Run model quantization.
async fn run_quantization(
    model_path: &str,
    output_path: &str,
    imatrix_path: Option<&str>,
    method: &str,
) -> anyhow::Result<()> {
    use pmetal_gguf::{
        dynamic::{DynamicQuantizationConfig, DynamicQuantizer},
        imatrix::IMatrix,
        quantize::quantize, // Import the function explicitly
        GgmlType,
        GgufBuilder,
    };
    use std::path::{Path, PathBuf};

    println!("========================================");
    println!("  PMetal GGUF Quantization");
    println!("========================================");
    println!("Model:    {}", model_path);
    println!("Output:   {}", output_path);
    println!("Method:   {}", method);
    if let Some(imp) = imatrix_path {
        println!("IMatrix:  {}", imp);
    }
    println!("========================================\n");

    // Resolve HuggingFace model ID to local path
    let resolved_model_path: PathBuf =
        if model_path.contains('/') && !PathBuf::from(model_path).exists() {
            // HuggingFace model ID - download/resolve to cache
            tracing::info!("Resolving HuggingFace model: {}", model_path);
            pmetal_hub::download_model(model_path, None, None).await?
        } else {
            PathBuf::from(model_path)
        };

    // 1. Load IMatrix if provided
    let imatrix = if let Some(path) = imatrix_path {
        tracing::info!("Loading IMatrix from {}", path);
        Some(IMatrix::load(Path::new(path))?)
    } else {
        None
    };

    // 2. Initialize Dynamic Quantizer
    let quantizer = if method == "dynamic" {
        let config = DynamicQuantizationConfig::default();
        DynamicQuantizer::new(config, imatrix)
    } else {
        // Static quantization fallback (mocking dynamic with fixed type)
        // In real impl, we'd parse the method string to GgmlType
        // For now, let's just support dynamic or fallback to Q4_K_M
        let base_type = match method {
            "q8_0" => GgmlType::Q8_0,
            "q4_k_m" => GgmlType::Q4K,
            _ => GgmlType::Q4K,
        };

        let config = DynamicQuantizationConfig {
            base_type,
            high_precision_type: base_type,
            fallback_type: base_type,
            ..Default::default()
        };
        DynamicQuantizer::new(config, None)
    };

    // 3. Load Model Weights
    tracing::info!("Scanning model weights from {:?}...", resolved_model_path);
    // Use the loader from pmetal_models to handle sharded safetensors
    let weights = pmetal_models::loader::load_weights(&resolved_model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?;

    tracing::info!("Loaded {} tensors", weights.len());

    // 4. Detect Architecture
    let config_path = resolved_model_path.join("config.json");
    let mut architecture = "llama".to_string(); // Default fallback

    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(archs) = json.get("architectures").and_then(|v| v.as_array()) {
                    if let Some(arch_str) = archs.first().and_then(|v| v.as_str()) {
                        architecture = match arch_str {
                            "LlamaForCausalLM" => "llama".to_string(),
                            "MistralForCausalLM" => "mistral".to_string(),
                            "Qwen2ForCausalLM" => "qwen2".to_string(),
                            "GemmaForCausalLM" | "Gemma2ForCausalLM" => "gemma".to_string(),
                            "PhiForCausalLM" | "Phi3ForCausalLM" => "phi".to_string(),
                            // Add more mappings as needed
                            _ => {
                                tracing::warn!(
                                    "Unknown architecture '{}', defaulting to 'llama'",
                                    arch_str
                                );
                                "llama".to_string()
                            }
                        };
                        tracing::info!(
                            "Detected architecture: {} (from {})",
                            architecture,
                            arch_str
                        );
                    }
                }
            }
        }
    } else {
        tracing::warn!("config.json not found, defaulting architecture to 'llama'");
    }

    // 5. Initialize GGUF Builder
    let mut builder = GgufBuilder::with_model(&architecture, "quantized-model");

    // 6. Quantize and Write
    tracing::info!("Starting quantization...");

    // Sort keys for deterministic output
    let mut keys: Vec<_> = weights.keys().collect();
    keys.sort();

    for name in keys {
        let tensor = weights.get(name).unwrap();
        // Skip quantization for non-F32/F16 tensors (e.g. integer indices) if any
        // But most LLM weights are floats.

        let shape = tensor.shape();
        let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();

        // Determine target type
        let target_type = quantizer.get_tensor_type(name, &shape_u64);

        // Convert MLX array to host vector
        // Note: This requires evaluating the array and copying data to CPU
        // Ensure tensor is evaluated
        tensor
            .eval()
            .map_err(|e| anyhow::anyhow!("MLX eval error: {}", e))?;

        // We assume weights are float32 for quantization input.
        // If they are float16/bfloat16, we convert them.
        let data_f32: Vec<f32> = match tensor.dtype() {
            pmetal_mlx::Dtype::Float32 => tensor.as_slice::<f32>().to_vec(),
            pmetal_mlx::Dtype::Float16 | pmetal_mlx::Dtype::Bfloat16 => {
                let t_f32 = tensor
                    .as_dtype(pmetal_mlx::Dtype::Float32)
                    .map_err(|e| anyhow::anyhow!("Dtype conversion error: {}", e))?;
                t_f32
                    .eval()
                    .map_err(|e| anyhow::anyhow!("MLX eval error: {}", e))?;
                t_f32.as_slice::<f32>().to_vec()
            }
            _ => {
                tracing::warn!("Skipping non-float tensor: {}", name);
                continue;
            }
        };

        // Quantize
        tracing::info!("Quantizing {} to {:?}", name, target_type);
        let quantized_data = quantize(&data_f32, target_type)
            .map_err(|e| anyhow::anyhow!("Quantization error for {}: {:?}", name, e))?;

        // Add to GGUF
        builder.add_raw_tensor(name, shape_u64, target_type, quantized_data);
    }

    // Write output file
    let mut file = std::fs::File::create(output_path)?;
    builder.write(&mut file)?;

    println!("Quantization complete!");
    Ok(())
}

/// Run LoRA/QLoRA fine-tuning using the new TrainingLoop.
#[allow(clippy::too_many_arguments)]
async fn run_training(
    config_path: Option<String>,
    model_id: Option<String>,
    dataset_path: Option<String>,
    eval_dataset_path: Option<String>,
    output_dir: String,
    lora_r: usize,
    learning_rate: f64,
    batch_size: usize,
    num_epochs: usize,
    max_seq_len: usize,
    gradient_accumulation_steps: usize,
    use_metal_flash_attention: bool,
    max_grad_norm: f64,
    resume: bool,
    quantization: QuantizationMethod,
    quant_block_size: usize,
    double_quant: bool,
    fused: bool,
    use_metal_fused_optimizer: bool,
    use_sequence_packing: bool,
    use_jit_compilation: bool,
    gradient_checkpointing: bool,
    gradient_checkpointing_layers: usize,
    log_metrics: Option<String>,
    embedding_lr: Option<f32>,
) -> anyhow::Result<()> {
    let use_qlora = !matches!(quantization, QuantizationMethod::None);

    // Validate output directory to prevent path traversal attacks
    let validated_output = validate_output_path(&output_dir, "output directory")?;
    let output_dir = validated_output.to_string_lossy().to_string();

    // Load or create configuration
    let mut config = if let Some(ref path) = config_path {
        let content = std::fs::read_to_string(path)?;
        serde_yaml::from_str(&content)?
    } else {
        FullTrainingConfig::default()
    };

    // Override with CLI args if provided
    if let Some(ref model) = model_id {
        config.model.model_id = model.clone();
    }
    if let Some(ref ds) = dataset_path {
        config.dataset.dataset_id = ds.clone();
    }
    config.lora.r = lora_r;
    config.training.learning_rate = learning_rate;
    config.training.batch_size = batch_size;
    config.training.num_epochs = num_epochs;
    config.training.max_seq_len = max_seq_len;
    config.training.gradient_accumulation_steps = gradient_accumulation_steps;
    config.training.max_grad_norm = max_grad_norm;
    config.training.output_dir = output_dir.clone();

    // Validate required fields
    if config.model.model_id.is_empty() {
        anyhow::bail!("Model ID is required. Use --model or specify in config file.");
    }
    if config.dataset.dataset_id.is_empty() {
        anyhow::bail!("Dataset path is required. Use --dataset or specify in config file.");
    }

    println!("========================================");
    println!(
        "  PMetal {} Fine-Tuning",
        if use_qlora { "QLoRA" } else { "LoRA" }
    );
    println!("========================================");
    println!("Model:         {}", config.model.model_id);
    println!("Dataset:       {}", config.dataset.dataset_id);
    if let Some(ref eval_path) = eval_dataset_path {
        println!("Eval Dataset:  {}", eval_path);
    }
    println!("Output:        {}", config.training.output_dir);
    println!("LoRA Rank:     {}", config.lora.r);
    println!("LR:            {}", config.training.learning_rate);
    println!("Batch Size:    {}", config.training.batch_size);
    println!(
        "Grad Accum:    {}",
        config.training.gradient_accumulation_steps
    );
    println!("Epochs:        {}", config.training.num_epochs);
    println!("Max Seq Len:   {}", config.training.max_seq_len);
    println!("Max Grad Norm: {}", config.training.max_grad_norm);
    println!(
        "Metal FA:      {}",
        if use_metal_flash_attention {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!(
        "Fused:         {}",
        if fused { "enabled" } else { "disabled" }
    );
    println!(
        "Metal Opt:     {}",
        if use_metal_fused_optimizer {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!(
        "Packing:       {}",
        if use_sequence_packing {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!(
        "JIT Compile:   {}",
        if use_jit_compilation {
            "enabled"
        } else {
            "disabled"
        }
    );
    if gradient_checkpointing {
        println!(
            "Grad Ckpt:     enabled ({} layers/block)",
            gradient_checkpointing_layers
        );
    } else {
        println!("Grad Ckpt:     disabled");
    }
    if use_qlora {
        println!("Quantization:  {:?}", quantization);
        println!("Block Size:    {}", quant_block_size);
        println!(
            "Double Quant:  {}",
            if double_quant { "enabled" } else { "disabled" }
        );
    }
    if let Some(ref metrics_path) = log_metrics {
        println!("Metrics Log:   {}", metrics_path);
    }
    if let Some(emb_lr) = embedding_lr {
        println!("Embedding LR:  {:.2e}", emb_lr);
    }
    println!("========================================\n");

    // Initialize metrics callback if requested
    let mut metrics_callback = if let Some(ref metrics_path) = log_metrics {
        // Default to output_dir/metrics.jsonl if only filename given
        let path = if metrics_path.contains('/') || metrics_path.contains('\\') {
            PathBuf::from(metrics_path)
        } else {
            PathBuf::from(&output_dir).join(metrics_path)
        };
        let callback = MetricsJsonCallback::new(&path)?
            .with_run_name(format!(
                "{}-{}",
                config.model.model_id.replace('/', "-"),
                chrono::Utc::now().format("%Y%m%d-%H%M%S")
            ))
            .with_config(serde_json::json!({
                "model": config.model.model_id,
                "lora_r": lora_r,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": num_epochs,
                "max_seq_len": max_seq_len,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "gradient_checkpointing": gradient_checkpointing,
                "quantization": format!("{:?}", quantization),
            }));
        use pmetal_core::TrainingCallback;
        let mut cb = callback;
        cb.on_train_start();
        Some(cb)
    } else {
        None
    };

    // Download model if needed
    tracing::info!("Loading model: {}", config.model.model_id);
    let model_path =
        if config.model.model_id.contains('/') && !PathBuf::from(&config.model.model_id).exists() {
            // HuggingFace model ID
            pmetal_hub::download_model(
                &config.model.model_id,
                config.model.revision.as_deref(),
                None,
            )
            .await?
        } else {
            PathBuf::from(&config.model.model_id)
        };

    // Load model config (optional for GGUF - config is extracted from metadata)
    let model_config_path = model_path.join("config.json");
    let llama_config: Option<LlamaConfig> = if model_config_path.exists() {
        let content = std::fs::read_to_string(&model_config_path)?;
        Some(serde_json::from_str(&content)?)
    } else {
        // GGUF files don't have separate config.json
        if WeightFormat::detect(&model_path) != Some(WeightFormat::Gguf) {
            anyhow::bail!(
                "Model config.json not found at {:?}. If using GGUF, pass the .gguf file directly.",
                model_config_path
            );
        }
        None
    };

    if let Some(ref cfg) = llama_config {
        tracing::info!(
            "Model: {} hidden, {} layers, {} heads",
            cfg.hidden_size,
            cfg.num_hidden_layers,
            cfg.num_attention_heads
        );
    } else {
        tracing::info!("Model config will be extracted from GGUF metadata");
    }

    // Load tokenizer
    tracing::info!("Loading tokenizer...");
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        Tokenizer::from_file(&tokenizer_path)?
    } else {
        anyhow::bail!("Tokenizer not found at {:?}", tokenizer_path);
    };

    // Load and tokenize training dataset
    tracing::info!("Loading training dataset: {}", config.dataset.dataset_id);
    let train_dataset = TrainingDataset::from_jsonl_tokenized(
        &config.dataset.dataset_id,
        &tokenizer,
        DatasetFormat::Auto,
        config.training.max_seq_len,
    )?;
    tracing::info!("Training dataset loaded: {} samples", train_dataset.len());

    // Load evaluation dataset if provided
    let eval_dataset = if let Some(ref eval_path) = eval_dataset_path {
        tracing::info!("Loading evaluation dataset: {}", eval_path);
        let ds = TrainingDataset::from_jsonl_tokenized(
            eval_path,
            &tokenizer,
            DatasetFormat::Auto,
            config.training.max_seq_len,
        )?;
        tracing::info!("Evaluation dataset loaded: {} samples", ds.len());
        Some(ds)
    } else {
        None
    };

    // Set up checkpointing
    let checkpoint_dir = PathBuf::from(&output_dir).join("checkpoints");
    let checkpoint_manager = CheckpointManager::new(&checkpoint_dir)?.with_max_checkpoints(3);

    // Create data loader config
    let dataloader_config = DataLoaderConfig {
        batch_size: config.training.batch_size,
        max_seq_len: config.training.max_seq_len,
        shuffle: config.dataset.shuffle,
        seed: config.training.seed,
        pad_token_id: tokenizer.pad_token_id().unwrap_or(0),
        drop_last: false,
    };

    // Create training loop config
    let training_loop_config = TrainingLoopConfig {
        training: config.training.clone(),
        dataloader: dataloader_config.clone(),
        use_metal_flash_attention,
        log_every: config.training.logging_steps,
        checkpoint_every: config.training.save_steps.unwrap_or(500),
        eval_every: if eval_dataset.is_some() { 100 } else { 0 },
        use_jit_compilation,
        use_sequence_packing,
        gradient_checkpointing,
        gradient_checkpointing_layers,
        embedding_lr,
        // Eager evaluation: forces immediate GPU computation after each step.
        // Reduces memory but may lower throughput. Useful for large models/sequences.
        eager_evaluation: false, // Default: disabled for throughput
        use_metal_fused_optimizer,
    };

    // Calculate total steps for progress bar
    let steps_per_epoch = train_dataset.len() / config.training.batch_size;
    let total_steps = if let Some(max) = config.training.max_steps {
        max
    } else {
        steps_per_epoch * config.training.num_epochs
    };

    // Set up progress bar
    let progress = ProgressBar::new(total_steps as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) | Loss: {msg}")
            .unwrap()
            .progress_chars("#>-"),
    );

    // Run training with either LoRA or QLoRA model
    let (final_loss, final_step, total_tokens) = if use_qlora {
        // QLoRA path - quantized base weights
        let quant_scheme = match quantization {
            QuantizationMethod::Nf4 => QuantScheme::NF4,
            QuantizationMethod::Fp4 => QuantScheme::FP4,
            QuantizationMethod::Int8 => QuantScheme::Int8,
            QuantizationMethod::None => unreachable!(),
        };

        let qlora_config = QLoraConfig {
            lora: config.lora.clone(),
            quant_scheme,
            block_size: quant_block_size,
            double_quant,
            compute_in_half: true,
        };

        tracing::info!(
            "Initializing QLoRA model with {:?} quantization...",
            quantization
        );
        // QLoRA currently requires config.json (Llama-only)
        let llama_cfg = llama_config.ok_or_else(|| {
            anyhow::anyhow!(
                "QLoRA requires config.json. GGUF format is only supported with standard LoRA."
            )
        })?;
        let mut model = LlamaQloraForCausalLM::with_qlora_config(llama_cfg, qlora_config)?;

        // Load and quantize base model weights
        tracing::info!(
            "Loading and quantizing base model weights from {:?}...",
            model_path
        );
        model.load_and_quantize_from_dir(&model_path)?;

        // Report memory savings
        let savings = model.memory_savings();
        let (quant_bytes, lora_bytes, total_bytes) = model.memory_usage();
        tracing::info!(
            "Memory usage: {:.2} MB (quantized: {:.2} MB, LoRA: {:.2} MB) - {:.1}% of full precision",
            total_bytes as f64 / 1_000_000.0,
            quant_bytes as f64 / 1_000_000.0,
            lora_bytes as f64 / 1_000_000.0,
            savings * 100.0
        );

        tracing::info!(
            "Trainable parameters: {}",
            format_param_count(model.num_trainable_params())
        );

        // Enable gradient checkpointing if requested
        if gradient_checkpointing {
            use pmetal_lora::TrainableModel;
            if model.supports_gradient_checkpointing() {
                model.enable_gradient_checkpointing(gradient_checkpointing_layers);
                tracing::info!(
                    "Gradient checkpointing enabled ({} layers per block)",
                    gradient_checkpointing_layers
                );
            } else {
                tracing::warn!(
                    "Gradient checkpointing requested but not supported by LlamaQloraForCausalLM. \
                     This feature is currently only supported for Qwen3 models."
                );
            }
        }

        // Create training loop
        let mut training_loop = TrainingLoop::new(training_loop_config);

        // Resume from checkpoint if requested
        if resume {
            if let Some((lora_params, metadata)) = checkpoint_manager.load_latest()? {
                tracing::info!("Resuming from checkpoint at step {}", metadata.step);
                model.set_lora_parameters(&lora_params);
                training_loop.set_step(metadata.step);
                training_loop.set_epoch(metadata.epoch);
            } else {
                tracing::info!("No checkpoint found, starting fresh");
            }
        }

        tracing::info!("Starting QLoRA training...");

        if fused {
            tracing::warn!(
                "Fused training is not yet supported for QLoRA, using standard training"
            );
        }

        // Run training loop
        training_loop.run(
            &mut model,
            train_dataset,
            eval_dataset,
            Some(&checkpoint_manager),
        )?;

        progress.finish_with_message(format!("{:.4}", training_loop.current_loss()));

        // Save final LoRA weights
        let final_path = PathBuf::from(&output_dir).join("lora_weights.safetensors");
        model.save_lora_weights(&final_path)?;
        tracing::info!("Saved LoRA weights to {:?}", final_path);

        (
            training_loop.current_loss(),
            training_loop.current_step(),
            training_loop.total_tokens(),
        )
    } else {
        // Standard LoRA path - full precision base weights with dynamic architecture detection
        tracing::info!("Initializing LoRA model with auto-detected architecture...");

        // Detect weight format and use appropriate loader
        let mut model = match WeightFormat::detect(&model_path) {
            Some(WeightFormat::Gguf) => {
                tracing::info!("Detected GGUF format, loading with dequantization...");
                DynamicLoraModel::from_gguf(&model_path, config.lora.clone())?
            }
            _ => {
                // Default to safetensors (HuggingFace format)
                DynamicLoraModel::from_pretrained(&model_path, config.lora.clone())?
            }
        };
        tracing::info!(
            "Loaded {} model with LoRA adapters",
            model.architecture_name()
        );

        tracing::info!(
            "Trainable parameters: {}",
            format_param_count(model.num_trainable_params())
        );

        // Enable gradient checkpointing if requested
        if gradient_checkpointing {
            if model.supports_gradient_checkpointing() {
                model.enable_gradient_checkpointing(gradient_checkpointing_layers);
                tracing::info!(
                    "Gradient checkpointing enabled ({} layers per block)",
                    gradient_checkpointing_layers
                );
            } else {
                tracing::warn!(
                    "Gradient checkpointing requested but not supported by {} architecture. \
                     This feature is currently only supported for Qwen3 models.",
                    model.architecture_name()
                );
            }
        }

        // Create training loop
        let mut training_loop = TrainingLoop::new(training_loop_config);

        // Resume from checkpoint if requested
        if resume {
            if let Some((lora_params, metadata)) = checkpoint_manager.load_latest()? {
                tracing::info!("Resuming from checkpoint at step {}", metadata.step);
                model.set_lora_parameters(&lora_params);
                training_loop.set_step(metadata.step);
                training_loop.set_epoch(metadata.epoch);
            } else {
                tracing::info!("No checkpoint found, starting fresh");
            }
        }

        tracing::info!("Starting LoRA training...");

        // Run training loop
        // Priority: packed > fused > standard
        if use_sequence_packing {
            // Sequence packing for 2-5x throughput
            let model = training_loop.run_packed(
                model,
                train_dataset.clone(),
                eval_dataset.clone(),
                Some(&checkpoint_manager),
            )?;

            progress.finish_with_message(format!("{:.4}", training_loop.current_loss()));

            // Save final LoRA weights
            let final_path = PathBuf::from(&output_dir).join("lora_weights.safetensors");
            model.save_lora_weights(&final_path)?;
            tracing::info!("Saved LoRA weights to {:?}", final_path);
        } else if (fused || use_jit_compilation) && config.training.gradient_accumulation_steps == 1
        {
            // Fused training step (combines forward/backward/optimizer)
            // JIT compilation requires the fused training path for compile_with_state
            let model = training_loop.run_compiled(
                model,
                train_dataset,
                eval_dataset,
                Some(&checkpoint_manager),
            )?;

            progress.finish_with_message(format!("{:.4}", training_loop.current_loss()));

            // Save final LoRA weights
            let final_path = PathBuf::from(&output_dir).join("lora_weights.safetensors");
            model.save_lora_weights(&final_path)?;
            tracing::info!("Saved LoRA weights to {:?}", final_path);
        } else if use_metal_fused_optimizer {
            // Metal fused optimizer for maximum throughput
            tracing::info!("Using Metal fused optimizer for training");
            training_loop.run_metal_fused(
                &mut model,
                train_dataset,
                eval_dataset,
                Some(&checkpoint_manager),
            )?;

            progress.finish_with_message(format!("{:.4}", training_loop.current_loss()));

            // Save final LoRA weights
            let final_path = PathBuf::from(&output_dir).join("lora_weights.safetensors");
            model.save_lora_weights(&final_path)?;
            tracing::info!("Saved LoRA weights to {:?}", final_path);
        } else {
            if (fused || use_jit_compilation) && config.training.gradient_accumulation_steps != 1 {
                tracing::warn!("Fused/JIT training requires gradient_accumulation_steps=1, falling back to standard training");
            }
            training_loop.run(
                &mut model,
                train_dataset,
                eval_dataset,
                Some(&checkpoint_manager),
            )?;

            progress.finish_with_message(format!("{:.4}", training_loop.current_loss()));

            // Save final LoRA weights
            let final_path = PathBuf::from(&output_dir).join("lora_weights.safetensors");
            model.save_lora_weights(&final_path)?;
            tracing::info!("Saved LoRA weights to {:?}", final_path);
        }

        (
            training_loop.current_loss(),
            training_loop.current_step(),
            training_loop.total_tokens(),
        )
    };

    // Finalize metrics callback
    if let Some(ref mut callback) = metrics_callback {
        use pmetal_core::TrainingCallback;
        // Write final epoch metrics
        let mut custom = std::collections::HashMap::new();
        custom.insert("total_tokens".to_string(), total_tokens as f64);
        custom.insert("total_steps".to_string(), final_step as f64);
        callback.on_epoch_end(
            num_epochs.saturating_sub(1),
            &pmetal_core::EvalMetrics {
                loss: final_loss,
                perplexity: final_loss.exp(),
                accuracy: None,
                custom,
            },
        );
        callback.on_train_end();
        tracing::info!("Metrics saved to {:?}", callback.path());
    }

    println!("\n========================================");
    println!("  Training Complete!");
    println!("========================================");
    println!("Final Loss:   {:.4}", final_loss);
    println!("Total Steps:  {}", final_step);
    println!("Total Tokens: {}", total_tokens);
    println!("Output:       {}", output_dir);
    println!("========================================");

    Ok(())
}

/// Run inference with a model.
///
/// Supports any architecture via DynamicModel, with optional LoRA support
/// for Llama models. Uses KV-cached generation for fast inference.
///
/// Implements SOTA sampling: temperature, top-k, top-p, min-p, repetition penalty,
/// frequency penalty, and presence penalty.
#[allow(clippy::too_many_arguments)]
async fn run_inference(
    model_id: &str,
    lora_path: Option<&str>,
    prompt: &str,
    max_tokens: usize,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    seed: Option<u64>,
    chat: bool,
    system: Option<&str>,
    no_thinking: bool,
    metal_sampler: bool,
    compiled: bool,
    stream: bool,
    minimal: bool,
    show_thinking: bool,
) -> anyhow::Result<()> {
    #[cfg(target_os = "macos")]
    use pmetal_models::generate_cached_metal;
    use pmetal_models::{
        generate_cached_async, generate_cached_compiled, generate_minimal_async, DynamicModel,
        GenerationConfig,
    };

    tracing::info!(model = %model_id, "Loading model for inference");

    // Download model if needed
    let model_path = if model_id.contains('/') && !PathBuf::from(model_id).exists() {
        pmetal_hub::download_model(model_id, None, None).await?
    } else {
        PathBuf::from(model_id)
    };

    // Load tokenizer
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        Tokenizer::from_file(&tokenizer_path)?
    } else {
        anyhow::bail!("Tokenizer not found at {:?}", tokenizer_path);
    };

    // Check if LoRA is requested
    if lora_path.is_some() {
        // LoRA requires architecture-specific handling
        // For now, only Llama is supported with LoRA
        // LoRA uses default temperature if not specified
        return run_inference_with_lora(
            &model_path,
            lora_path.unwrap(),
            &tokenizer,
            prompt,
            max_tokens,
            temperature.unwrap_or(0.7),
        )
        .await;
    }

    // Use DynamicModel for architecture-agnostic inference
    tracing::info!("Loading model with auto-detected architecture...");
    let mut model = DynamicModel::from_pretrained(&model_path)?;

    tracing::info!(
        "Model loaded successfully (architecture: {})",
        model.architecture()
    );

    // Auto-detect if chat mode should be enabled for instruction-tuned models
    let is_instruct_model = is_instruction_tuned(&model_path);
    let use_chat = chat || is_instruct_model;

    if is_instruct_model && !chat {
        tracing::info!("Auto-detected instruction-tuned model, enabling chat template");
    }

    // Load sampling defaults from model's generation_config.json
    let defaults = load_sampling_defaults(&model_path, use_chat && !no_thinking);

    // Apply CLI overrides over model defaults
    let temperature = temperature.unwrap_or(defaults.temperature);
    let top_k = top_k.unwrap_or(defaults.top_k);
    let top_p = top_p.unwrap_or(defaults.top_p);
    let min_p = min_p.unwrap_or(defaults.min_p);
    let repetition_penalty = repetition_penalty.unwrap_or(defaults.repetition_penalty);
    let frequency_penalty = frequency_penalty.unwrap_or(defaults.frequency_penalty);
    let presence_penalty = presence_penalty.unwrap_or(defaults.presence_penalty);

    // Apply chat template if needed
    // The template handles thinking mode - model decides when to think unless --no-thinking
    let final_prompt = if use_chat {
        apply_chat_template(&tokenizer, prompt, system, &model_path, no_thinking)?
    } else {
        prompt.to_string()
    };

    // Tokenize prompt
    let input_ids = tokenizer.encode(&final_prompt)?;
    tracing::info!("Tokenized {} tokens", input_ids.len());

    // Configure stop tokens for chat mode
    // Only stop on <|im_end|> to allow thinking blocks to complete
    let stop_tokens = if use_chat {
        let mut tokens = Vec::new();
        // <|im_end|> token - stops at end of assistant response
        if let Ok(im_end) = tokenizer.encode("<|im_end|>") {
            if let Some(&token) = im_end.last() {
                tokens.push(token);
            }
        }
        // Fallback
        if tokens.is_empty() {
            tokens.push(151645); // Qwen3's <|im_end|> token
        }
        tokens
    } else {
        // Non-chat mode: use EOS tokens from generation_config.json
        get_eos_tokens(&model_path, &tokenizer)
    };

    // Configure generation with user-specified parameters
    let gen_config = if temperature == 0.0 {
        GenerationConfig::greedy(max_tokens).with_stop_tokens(stop_tokens)
    } else {
        let mut config = GenerationConfig::sampling(max_tokens, temperature)
            .with_top_k(top_k)
            .with_top_p(top_p)
            .with_min_p(min_p)
            .with_repetition_penalty(repetition_penalty)
            .with_frequency_penalty(frequency_penalty)
            .with_presence_penalty(presence_penalty)
            .with_stop_tokens(stop_tokens);

        if let Some(s) = seed {
            config = config.with_seed(s);
        }

        config
    };

    // Print configuration
    println!("\n========================================");
    println!("  PMetal Inference");
    println!("========================================");
    println!("Model:       {}", model_id);
    println!("Temperature: {}", gen_config.temperature);
    println!("Top-k:       {}", gen_config.top_k);
    println!("Top-p:       {}", gen_config.top_p);
    println!("Min-p:       {}", gen_config.min_p);
    println!("Rep penalty: {}", gen_config.repetition_penalty);
    println!("Freq penalty:{}", gen_config.frequency_penalty);
    println!("Pres penalty:{}", gen_config.presence_penalty);
    if let Some(s) = gen_config.seed {
        println!("Seed:        {}", s);
    }
    println!("Max tokens:  {}", max_tokens);
    if use_chat && no_thinking {
        println!("Thinking:    disabled");
    }
    println!("========================================\n");

    println!("Prompt: {}\n", prompt);
    println!("Generating...\n");

    // Create KV cache for efficient generation
    // Cache size = prompt_len + max_tokens + buffer
    let max_seq_len = input_ids.len() + max_tokens + 64;
    let mut cache = model.create_cache(max_seq_len);
    tracing::info!("Created KV cache for {} tokens", max_seq_len);

    // Generate with KV cache
    let start = std::time::Instant::now();

    #[cfg(target_os = "macos")]
    let output = if minimal {
        tracing::info!("Using minimal async generation (debugging)");
        generate_minimal_async(
            |input, cache| model.forward_with_cache(input, None, Some(cache)),
            &input_ids,
            gen_config,
            &mut cache,
        )?
    } else if metal_sampler {
        tracing::info!("Using fused Metal sampling kernel");
        generate_cached_metal(
            |input, cache| model.forward_with_cache(input, None, Some(cache)),
            &input_ids,
            gen_config,
            &mut cache,
        )?
    } else if compiled {
        tracing::info!("Using JIT-compiled sampling (mlx_lm style)");
        generate_cached_compiled(
            |input, cache| model.forward_with_cache(input, None, Some(cache)),
            &input_ids,
            gen_config,
            &mut cache,
        )?
    } else {
        // Default: async generation with dedicated stream (matches mlx_lm)
        // Note: --stream flag is now a no-op since async is the default
        if stream {
            tracing::info!("Using async generation with dedicated stream");
        }
        generate_cached_async(
            |input, cache| model.forward_with_cache(input, None, Some(cache)),
            &input_ids,
            gen_config,
            &mut cache,
        )?
    };

    #[cfg(not(target_os = "macos"))]
    let output = {
        let _ = metal_sampler; // Suppress unused warning
        if minimal {
            tracing::info!("Using minimal async generation (debugging)");
            generate_minimal_async(
                |input, cache| model.forward_with_cache(input, None, Some(cache)),
                &input_ids,
                gen_config,
                &mut cache,
            )?
        } else if compiled {
            tracing::info!("Using JIT-compiled sampling (mlx_lm style)");
            generate_cached_compiled(
                |input, cache| model.forward_with_cache(input, None, Some(cache)),
                &input_ids,
                gen_config,
                &mut cache,
            )?
        } else {
            // Default: async generation with dedicated stream (matches mlx_lm)
            // Note: --stream flag is now a no-op since async is the default
            if stream {
                tracing::info!("Using async generation with dedicated stream");
            }
            generate_cached_async(
                |input, cache| model.forward_with_cache(input, None, Some(cache)),
                &input_ids,
                gen_config,
                &mut cache,
            )?
        }
    };
    let elapsed = start.elapsed();

    // Decode only the GENERATED tokens (not the prompt)
    // output.token_ids includes prompt, so slice off the prompt portion
    let prompt_len = input_ids.len();
    let generated_tokens = &output.token_ids[prompt_len..];
    let raw_generated_text = tokenizer.decode(generated_tokens)?;

    // When in thinking mode, we prefill <think>\n in the prompt, so the generated
    // text doesn't include the opening tag. Prepend it for correct extraction.
    let generated_text = if use_chat && !no_thinking {
        format!("<think>{}", raw_generated_text)
    } else {
        raw_generated_text
    };

    // For chat mode, handle thinking content and extract response
    if use_chat {
        if show_thinking {
            // Show thinking separately with banners, then show response
            if let Some(thinking) = extract_thinking_content(&generated_text) {
                println!("=== Thinking ===");
                println!("{}", thinking);
                println!("=== Response ===");
                // Extract just the response (after </think>)
                let response = extract_final_response(&generated_text);
                println!("{}", response);
            } else {
                // No complete thinking found, show truncation or raw content
                let response = extract_final_response(&generated_text);
                println!("{}", response);
            }
        } else {
            // Don't show thinking, just extract final response
            let response = extract_final_response(&generated_text);
            println!("{}", response);
        }
    } else {
        println!("{}", generated_text);
    }

    println!("\n---");
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s)",
        output.num_generated,
        elapsed.as_secs_f64(),
        output.num_generated as f64 / elapsed.as_secs_f64()
    );
    if output.stopped_by_token {
        println!("Stopped by: EOS token");
    } else {
        println!("Stopped by: max length");
    }

    Ok(())
}

/// Get EOS token IDs from model's generation_config.json.
///
/// Many models (like Qwen3) have multiple EOS tokens that should all stop generation.
fn get_eos_tokens(model_path: &Path, tokenizer: &Tokenizer) -> Vec<u32> {
    let mut tokens = Vec::new();

    // Try to read from generation_config.json
    let config_path = model_path.join("generation_config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                // Handle eos_token_id as array or single value
                if let Some(eos) = config.get("eos_token_id") {
                    if let Some(arr) = eos.as_array() {
                        for v in arr {
                            if let Some(id) = v.as_u64() {
                                tokens.push(id as u32);
                            }
                        }
                    } else if let Some(id) = eos.as_u64() {
                        tokens.push(id as u32);
                    }
                }
            }
        }
    }

    // Fallback to tokenizer's eos_token_id
    if tokens.is_empty() {
        tokens.push(tokenizer.eos_token_id().unwrap_or(2));
    }

    tokens
}

/// Sampling hyperparameter defaults loaded from model config.
struct SamplingDefaults {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        // Qwen3 recommended defaults for non-thinking mode
        Self {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

/// Load sampling defaults from model's generation_config.json.
///
/// Uses Qwen3 recommended values as fallback:
/// - Thinking mode: temp=0.6, top_p=0.95, top_k=20, min_p=0
/// - Non-thinking mode: temp=0.7, top_p=0.8, top_k=20, min_p=0
fn load_sampling_defaults(model_path: &Path, thinking_mode: bool) -> SamplingDefaults {
    // Start with mode-appropriate defaults
    let mut defaults = if thinking_mode {
        SamplingDefaults {
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    } else {
        SamplingDefaults::default()
    };

    // Try to load from generation_config.json
    let config_path = model_path.join("generation_config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                // Load each parameter if present
                if let Some(v) = config.get("temperature").and_then(|v| v.as_f64()) {
                    defaults.temperature = v as f32;
                }
                if let Some(v) = config.get("top_k").and_then(|v| v.as_u64()) {
                    defaults.top_k = v as usize;
                }
                if let Some(v) = config.get("top_p").and_then(|v| v.as_f64()) {
                    defaults.top_p = v as f32;
                }
                if let Some(v) = config.get("min_p").and_then(|v| v.as_f64()) {
                    defaults.min_p = v as f32;
                }
                if let Some(v) = config.get("repetition_penalty").and_then(|v| v.as_f64()) {
                    defaults.repetition_penalty = v as f32;
                }
                if let Some(v) = config.get("frequency_penalty").and_then(|v| v.as_f64()) {
                    defaults.frequency_penalty = v as f32;
                }
                if let Some(v) = config.get("presence_penalty").and_then(|v| v.as_f64()) {
                    defaults.presence_penalty = v as f32;
                }
            }
        }
    }

    defaults
}

/// Check if a model is instruction-tuned based on its configuration.
///
/// Looks for indicators like:
/// - chat_template in tokenizer_config.json
/// - "instruct", "chat", "it" in model name
/// - Known instruction-tuned model architectures
fn is_instruction_tuned(model_path: &Path) -> bool {
    // Check model name/path for instruction indicators
    let path_str = model_path.to_string_lossy().to_lowercase();
    if path_str.contains("instruct")
        || path_str.contains("-it")
        || path_str.contains("chat")
        || path_str.contains("qwen3")
        || path_str.contains("llama-3")
    {
        return true;
    }

    // Check for chat_template in tokenizer_config.json
    let config_path = model_path.join("tokenizer_config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                // Has chat template -> instruction tuned
                if config.get("chat_template").is_some() {
                    return true;
                }
            }
        }
    }

    false
}

/// Apply chat template to the prompt.
///
/// If `no_thinking` is true, prefills empty thinking block to disable reasoning.
/// Otherwise, the model decides when to use thinking based on query complexity.
fn apply_chat_template(
    _tokenizer: &Tokenizer,
    user_message: &str,
    system_message: Option<&str>,
    model_path: &Path,
    no_thinking: bool,
) -> anyhow::Result<String> {
    // Try to load chat template from tokenizer_config.json
    let config_path = model_path.join("tokenizer_config.json");
    if config_path.exists() {
        let content = std::fs::read_to_string(&config_path)?;
        if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
            // Check for Qwen3/ChatML style template
            if let Some(template) = config.get("chat_template").and_then(|t| t.as_str()) {
                if template.contains("chatml") || template.contains("<|im_start|>") {
                    return Ok(format_chatml(user_message, system_message, no_thinking));
                }
                if template.contains("llama") || template.contains("<|begin_of_text|>") {
                    return Ok(format_llama3(user_message, system_message));
                }
            }

            // Check model type hints
            if let Some(model_type) = config.get("model_type").and_then(|t| t.as_str()) {
                if model_type.to_lowercase().contains("qwen") {
                    return Ok(format_chatml(user_message, system_message, no_thinking));
                }
            }
        }
    }

    // Default to ChatML format (widely compatible)
    Ok(format_chatml(user_message, system_message, no_thinking))
}

/// Format message using ChatML template (used by Qwen, many others).
fn format_chatml(user_message: &str, system_message: Option<&str>, no_thinking: bool) -> String {
    format_qwen3_chatml(user_message, system_message, no_thinking)
}

/// Format message using Qwen3 ChatML template.
///
/// By default, the model decides when to use `<think>` blocks based on query complexity.
/// If `no_thinking` is true, prefills empty `<think></think>` to force non-thinking mode.
fn format_qwen3_chatml(
    user_message: &str,
    system_message: Option<&str>,
    no_thinking: bool,
) -> String {
    let mut result = String::new();

    if let Some(sys) = system_message {
        result.push_str("<|im_start|>system\n");
        result.push_str(sys);
        result.push_str("<|im_end|>\n");
    }

    result.push_str("<|im_start|>user\n");
    result.push_str(user_message);
    result.push_str("<|im_end|>\n");
    result.push_str("<|im_start|>assistant\n");

    if no_thinking {
        // Force non-thinking: prefill empty think block
        // This is the Qwen3 "hard switch" to disable reasoning
        result.push_str("<think>\n\n</think>\n\n");
    } else {
        // Prefill <think> to ensure clean thinking output
        // Without this, model sometimes generates </think> first or skips thinking
        result.push_str("<think>\n");
    }

    result
}

/// Format message using Llama 3 template.
fn format_llama3(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::from("<|begin_of_text|>");

    if let Some(sys) = system_message {
        result.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
        result.push_str(sys);
        result.push_str("<|eot_id|>");
    }

    result.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
    result.push_str(user_message);
    result.push_str("<|eot_id|>");
    result.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

    result
}

/// Extract the final response after </think> tag, discarding thinking content.
///
/// Handles several cases:
/// 1. Complete thinking: `<think>...</think>response` -> returns `response`
/// 2. Incomplete thinking (hit max tokens): `<think>...` -> returns empty (model didn't finish)
/// 3. No thinking: `response` -> returns `response`
fn extract_final_response(text: &str) -> String {
    // Case 1: Find complete </think> tag
    if let Some(pos) = text.rfind("</think>") {
        let after_think = &text[pos + "</think>".len()..];
        // Clean up any stray <think> tags (small models sometimes output malformed content)
        let cleaned = after_think
            .trim()
            .trim_start_matches("<think>")
            .trim_start_matches('\n')
            .trim_end_matches("<|im_end|>")
            .trim_end_matches("<|endoftext|>")
            .trim();
        return cleaned.to_string();
    }

    // Case 2: Incomplete thinking - model started <think> but never finished
    // Since there's no </think>, the model was still thinking when it hit max tokens
    if text.contains("<think>") {
        return "[Response truncated - model was still thinking. Use --no-thinking or increase --max-tokens]".to_string();
    }

    // Case 3: No thinking block, return as-is
    text.trim_end_matches("<|im_end|>")
        .trim_end_matches("<|endoftext|>")
        .trim()
        .to_string()
}

/// Extract thinking content from response (for display purposes).
///
/// Handles cases where the model generates multiple `<think>` tokens at the start
/// by finding the last complete `<think>...</think>` block.
fn extract_thinking_content(text: &str) -> Option<String> {
    // Find the closing </think> tag first
    let end = text.rfind("</think>")?;

    // Find the last <think> tag before </think> that starts actual content
    // (skip repeated <think> tags at the start)
    let search_region = &text[..end];

    // Find the last <think> that's followed by actual text content, not just more <think> tags
    let mut last_real_start = None;
    let mut pos = 0;
    while let Some(start) = search_region[pos..].find("<think>") {
        let absolute_start = pos + start;
        let after_tag = &search_region[absolute_start + "<think>".len()..];

        // Check if this is followed by real content (not just another <think> or whitespace then <think>)
        let trimmed = after_tag.trim_start();
        if !trimmed.starts_with("<think>") && !trimmed.is_empty() {
            last_real_start = Some(absolute_start);
        }

        pos = absolute_start + "<think>".len();
    }

    if let Some(start) = last_real_start {
        let thinking = &text[start + "<think>".len()..end];
        // Clean up the thinking content
        let cleaned = thinking
            .trim()
            .trim_start_matches("<think>")
            .trim_start_matches('\n')
            .trim();
        if !cleaned.is_empty() {
            return Some(cleaned.to_string());
        }
    }

    // Fallback: simple extraction if the above didn't work
    if let Some(start) = text.find("<think>") {
        if end > start {
            let thinking = &text[start + "<think>".len()..end];
            let cleaned = thinking.trim();
            if !cleaned.is_empty() {
                return Some(cleaned.to_string());
            }
        }
    }

    None
}

/// Run inference with LoRA adapter (supports all architectures via DynamicLoraModel).
async fn run_inference_with_lora(
    model_path: &Path,
    lora_path: &str,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
) -> anyhow::Result<()> {
    use pmetal_core::LoraConfig;
    use pmetal_lora::{DynamicLoraModel, TrainableModel};
    use pmetal_models::{generate_cached_async, GenerationConfig};

    // Create LoRA config - we'll load actual weights which override this
    let lora_config = LoraConfig {
        r: 16, // Will be overridden by loaded weights
        alpha: 16.0,
        ..Default::default()
    };

    // Use DynamicLoraModel for automatic architecture detection
    tracing::info!("Loading model with auto-detected architecture...");
    let mut model = DynamicLoraModel::from_pretrained(model_path, lora_config)?;
    tracing::info!("Detected architecture: {}", model.architecture_name());

    // Load LoRA adapter weights
    tracing::info!("Loading LoRA adapter from {:?}...", lora_path);
    model.load_lora_weights(lora_path)?;

    tracing::info!("Model loaded successfully");

    // Tokenize prompt
    let input_ids = tokenizer.encode(prompt)?;
    tracing::info!("Tokenized {} tokens", input_ids.len());

    // Configure generation
    let gen_config = if temperature > 0.0 {
        GenerationConfig::sampling(max_tokens, temperature)
            .with_stop_tokens(vec![tokenizer.eos_token_id().unwrap_or(2)])
    } else {
        GenerationConfig::greedy(max_tokens)
            .with_stop_tokens(vec![tokenizer.eos_token_id().unwrap_or(2)])
    };

    println!("\nPrompt: {}\n", prompt);
    println!("Generating with KV cache...\n");

    // Create KV cache for efficient generation
    // Cache size = prompt_len + max_tokens + buffer
    let max_seq_len = input_ids.len() + max_tokens + 64;
    let mut cache = model
        .create_cache(max_seq_len)
        .ok_or_else(|| anyhow::anyhow!("Model does not support KV cache"))?;
    tracing::info!("Created KV cache for {} tokens", max_seq_len);

    let start = std::time::Instant::now();

    // Generate with KV cache (O(n+k) complexity - fast!)
    let output = generate_cached_async(
        |input, cache| {
            model
                .forward_with_cache(input, None, Some(cache))
                .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))
        },
        &input_ids,
        gen_config,
        &mut cache,
    )?;

    let elapsed = start.elapsed();

    // Decode only the generated tokens (not the prompt)
    let prompt_len = input_ids.len();
    let generated_tokens = &output.token_ids[prompt_len..];
    let generated_text = tokenizer.decode(generated_tokens)?;
    println!("{}", generated_text);

    let tokens_per_sec = output.num_generated as f64 / elapsed.as_secs_f64();
    println!("\n---");
    println!(
        "Generated {} tokens in {:.2}s ({:.1} tok/s)",
        output.num_generated,
        elapsed.as_secs_f64(),
        tokens_per_sec
    );
    if output.stopped_by_token {
        println!("Stopped by: EOS token");
    } else {
        println!("Stopped by: max length");
    }

    Ok(())
}

/// Run Ollama subcommands.
async fn run_ollama_command(action: OllamaAction) -> anyhow::Result<()> {
    match action {
        OllamaAction::Modelfile {
            base,
            lora,
            output,
            system,
            temperature,
            num_ctx,
            top_k,
            top_p,
            template,
            license,
        } => {
            generate_modelfile(
                &base,
                lora.as_deref(),
                &output,
                system.as_deref(),
                temperature,
                num_ctx,
                top_k,
                top_p,
                template,
                license.as_deref(),
            )?;
        }

        OllamaAction::Create {
            name,
            base,
            lora,
            system,
            temperature,
            num_ctx,
            template,
        } => {
            create_ollama_model(
                &name,
                &base,
                lora.as_deref(),
                system.as_deref(),
                temperature,
                num_ctx,
                template,
            )?;
        }

        OllamaAction::Templates => {
            print_ollama_templates();
        }
    }

    Ok(())
}

/// Generate a Modelfile for Ollama.
fn generate_modelfile(
    base: &str,
    lora: Option<&str>,
    output: &str,
    system: Option<&str>,
    temperature: Option<f32>,
    num_ctx: Option<i32>,
    top_k: Option<i32>,
    top_p: Option<f32>,
    template: Option<OllamaTemplate>,
    license: Option<&str>,
) -> anyhow::Result<()> {
    // Validate output path to prevent path traversal
    let output_path = validate_file_path(output, true)?;

    println!("========================================");
    println!("  PMetal Ollama Export");
    println!("========================================");
    println!("Base Model:  {}", base);
    if let Some(lora_path) = lora {
        println!("LoRA:        {}", lora_path);
    }
    println!("Output:      {}", output_path.display());
    println!("========================================\n");

    // Build Modelfile
    let mut builder = ModelfileBuilder::new().from(base);

    // Add LoRA adapter if specified
    if let Some(lora_path) = lora {
        builder = builder.adapter(lora_path);
    }

    // Add system prompt
    if let Some(sys) = system {
        builder = builder.system(sys);
    }

    // Add parameters
    if let Some(temp) = temperature {
        builder = builder.temperature(temp);
    }
    if let Some(ctx) = num_ctx {
        builder = builder.num_ctx(ctx);
    }
    if let Some(k) = top_k {
        builder = builder.top_k(k);
    }
    if let Some(p) = top_p {
        builder = builder.top_p(p);
    }

    // Add template
    if let Some(tmpl) = template {
        let template_str = get_ollama_template(tmpl);
        builder = builder.template(template_str);
    } else {
        // Try to auto-detect template from base model name
        if let Some(detected_template) = detect_template_from_model(base) {
            builder = builder.template(detected_template);
            println!("Auto-detected template from model name");
        }
    }

    // Add license
    if let Some(lic) = license {
        builder = builder.license(lic);
    }

    // Build and write
    builder.write_to_file(&output_path)?;
    println!("Modelfile written to: {}", output_path.display());

    println!("\nTo create the model in Ollama, run:");
    println!("  ollama create <model-name> -f {}", output_path.display());

    Ok(())
}

/// Validate model name for Ollama (prevent command injection).
fn validate_ollama_model_name(name: &str) -> anyhow::Result<()> {
    // Allow alphanumeric, hyphen, underscore, period, forward slash (for namespaces)
    // Reject anything that could be interpreted as shell metacharacters
    if name.is_empty() {
        anyhow::bail!("Model name cannot be empty");
    }
    if name.len() > 255 {
        anyhow::bail!("Model name too long (max 255 characters)");
    }
    if name.starts_with('.') || name.starts_with('-') {
        anyhow::bail!("Model name cannot start with '.' or '-'");
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | ':'))
    {
        anyhow::bail!(
            "Invalid model name '{}'. Name must contain only alphanumeric characters, \
             hyphens, underscores, periods, colons, and forward slashes.",
            name
        );
    }
    Ok(())
}

/// Validate file path (prevent path traversal).
fn validate_file_path(path: &str, allow_creation: bool) -> anyhow::Result<std::path::PathBuf> {
    let path = std::path::Path::new(path);

    // Prevent path traversal
    if path
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        anyhow::bail!("Invalid path: path traversal detected (.. not allowed)");
    }

    // Get canonical path
    let canonical = if path.exists() {
        path.canonicalize()?
    } else if allow_creation {
        // If file doesn't exist yet, canonicalize parent
        if let Some(parent) = path.parent() {
            if parent.as_os_str().is_empty() {
                std::env::current_dir()?.join(path.file_name().unwrap())
            } else {
                parent.canonicalize()?.join(path.file_name().unwrap())
            }
        } else {
            std::env::current_dir()?.join(path)
        }
    } else {
        anyhow::bail!("Path does not exist: {}", path.display());
    };

    Ok(canonical)
}

/// Create and register a model with Ollama.
fn create_ollama_model(
    name: &str,
    base: &str,
    lora: Option<&str>,
    system: Option<&str>,
    temperature: Option<f32>,
    num_ctx: Option<i32>,
    template: Option<OllamaTemplate>,
) -> anyhow::Result<()> {
    // Validate model name to prevent command injection
    validate_ollama_model_name(name)?;

    // Create secure temporary file
    let temp_dir = std::env::temp_dir();
    // Use a sanitized name for the temp file
    let safe_name: String = name
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .take(64)
        .collect();
    let modelfile_path = temp_dir.join(format!(
        "pmetal-modelfile-{}-{}",
        safe_name,
        std::process::id()
    ));
    let modelfile_str = modelfile_path.to_string_lossy().to_string();

    generate_modelfile(
        base,
        lora,
        &modelfile_str,
        system,
        temperature,
        num_ctx,
        None,
        None,
        template,
        None,
    )?;

    println!("\nCreating Ollama model '{}'...", name);

    // Run ollama create
    let status = std::process::Command::new("ollama")
        .args(["create", name, "-f", &modelfile_str])
        .status();

    match status {
        Ok(exit_status) if exit_status.success() => {
            println!("\nModel '{}' created successfully!", name);
            println!("\nTo use the model, run:");
            println!("  ollama run {}", name);

            // Clean up temp file
            let _ = std::fs::remove_file(&modelfile_path);
        }
        Ok(exit_status) => {
            anyhow::bail!(
                "ollama create failed with exit code: {:?}. \
                 Modelfile saved at: {}",
                exit_status.code(),
                modelfile_str
            );
        }
        Err(e) => {
            if e.kind() == std::io::ErrorKind::NotFound {
                println!("\nOllama not found. Please install Ollama first:");
                println!("  https://ollama.ai/download");
                println!("\nModelfile has been saved to: {}", modelfile_str);
                println!("Once Ollama is installed, run:");
                println!("  ollama create {} -f {}", name, modelfile_str);
            } else {
                anyhow::bail!("Failed to run ollama: {}", e);
            }
        }
    }

    Ok(())
}

/// Print available Ollama templates.
fn print_ollama_templates() {
    println!("Available Ollama Templates:");
    println!("========================================\n");

    println!("llama3 - Llama 3 Chat Format");
    println!("  Uses: <|start_header_id|>...<|end_header_id|> format");
    println!("  Best for: Llama 3, Llama 3.1, Llama 3.2, Llama 4\n");

    println!("qwen3 - Qwen3/ChatML Format");
    println!("  Uses: <|im_start|>...<|im_end|> format");
    println!("  Best for: Qwen 2, Qwen 2.5, Qwen 3\n");

    println!("gemma - Gemma Instruct Format");
    println!("  Uses: <start_of_turn>...<end_of_turn> format");
    println!("  Best for: Gemma 2, Gemma 3\n");

    println!("mistral - Mistral Instruct Format");
    println!("  Uses: [INST]...[/INST] format");
    println!("  Best for: Mistral, Mixtral\n");

    println!("phi3 - Phi-3 Instruct Format");
    println!("  Uses: <|system|>...<|end|> format");
    println!("  Best for: Phi 3, Phi 4\n");

    println!("deepseek - DeepSeek Chat Format");
    println!("  Uses: <|begin_of_sentence|>User:...Assistant: format");
    println!("  Best for: DeepSeek, DeepSeek-V2, DeepSeek-V3\n");

    println!("========================================");
    println!("Usage: pmetal ollama modelfile --base <model> --template <template>");
}

/// Get the Ollama template string for a template type.
fn get_ollama_template(template: OllamaTemplate) -> &'static str {
    match template {
        OllamaTemplate::Llama3 => ollama_templates::LLAMA3_CHAT,
        OllamaTemplate::Qwen3 => ollama_templates::QWEN3_CHAT,
        OllamaTemplate::Gemma => ollama_templates::GEMMA_INSTRUCT,
        OllamaTemplate::Mistral => ollama_templates::MISTRAL_INSTRUCT,
        OllamaTemplate::Phi3 => ollama_templates::PHI3_INSTRUCT,
        OllamaTemplate::DeepSeek => ollama_templates::DEEPSEEK_CHAT,
    }
}

/// Try to detect the appropriate template from the model name.
fn detect_template_from_model(model: &str) -> Option<&'static str> {
    let lower = model.to_lowercase();

    if lower.contains("llama") || lower.contains("meta-llama") {
        Some(ollama_templates::LLAMA3_CHAT)
    } else if lower.contains("qwen") {
        Some(ollama_templates::QWEN3_CHAT)
    } else if lower.contains("gemma") {
        Some(ollama_templates::GEMMA_INSTRUCT)
    } else if lower.contains("mistral") || lower.contains("mixtral") {
        Some(ollama_templates::MISTRAL_INSTRUCT)
    } else if lower.contains("phi") {
        Some(ollama_templates::PHI3_INSTRUCT)
    } else if lower.contains("deepseek") {
        Some(ollama_templates::DEEPSEEK_CHAT)
    } else {
        None
    }
}

/// Run benchmark.
async fn run_benchmark(model: &str, batch_size: usize, seq_len: usize) -> anyhow::Result<()> {
    tracing::info!(
        model = %model,
        batch_size = batch_size,
        seq_len = seq_len,
        "Running benchmark"
    );

    println!("Benchmark Configuration:");
    println!("  Model:      {}", model);
    println!("  Batch Size: {}", batch_size);
    println!("  Seq Length: {}", seq_len);
    println!("\nBenchmarking in progress...");

    // Create dummy config
    let llama_config = LlamaConfig {
        vocab_size: 32000,
        hidden_size: 2048,
        intermediate_size: 5632,
        num_hidden_layers: 22,
        num_attention_heads: 32,
        num_key_value_heads: Some(4),
        max_position_embeddings: 2048,
        ..Default::default()
    };

    let lora_config = LoraConfig {
        r: 16,
        alpha: 32.0,
        ..Default::default()
    };

    let mut model = LlamaLoraForCausalLM::new(llama_config, lora_config)?;

    // Create dummy data
    let input_ids = mlx_rs::Array::zeros::<i32>(&[batch_size as i32, seq_len as i32])?;

    // Warmup
    println!("Warming up...");
    for _ in 0..3 {
        let output = model.forward(&input_ids, None)?;
        output.eval()?;
    }

    // Benchmark
    let iterations = 10;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        let output = model.forward(&input_ids, None)?;
        output.eval()?;
    }

    let elapsed = start.elapsed();
    let avg_ms = elapsed.as_millis() as f64 / iterations as f64;
    let tokens_per_sec = (batch_size * seq_len) as f64 / (avg_ms / 1000.0);

    println!("\nResults:");
    println!("  Avg Time:       {:.2} ms/iteration", avg_ms);
    println!("  Throughput:     {:.0} tokens/sec", tokens_per_sec);

    let stats = pmetal_mlx::memory::get_memory_stats();
    println!("  Memory Used:    {:.2} GB", stats.used_gb());
    println!("  Peak Memory:    {:.2} GB", stats.peak_gb());

    Ok(())
}

/// Generate a sample configuration file.
fn generate_sample_config(output: &str) -> anyhow::Result<()> {
    let config = FullTrainingConfig {
        model: ModelConfig {
            model_id: "meta-llama/Llama-3.2-1B".to_string(),
            max_seq_len: 2048,
            ..Default::default()
        },
        lora: LoraConfig {
            r: 16,
            alpha: 32.0,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            ..Default::default()
        },
        training: TrainingConfig {
            learning_rate: 2e-4,
            batch_size: 4,
            num_epochs: 3,
            warmup_steps: 100,
            max_seq_len: 2048,
            output_dir: "./output".to_string(),
            logging_steps: 10,
            save_steps: Some(500),
            ..Default::default()
        },
        dataset: DatasetConfig {
            dataset_id: "./data/train.jsonl".to_string(),
            shuffle: true,
            ..Default::default()
        },
    };

    let yaml = serde_yaml::to_string(&config)?;
    std::fs::write(output, yaml)?;

    println!("Sample configuration written to: {}", output);
    println!("\nYou can edit this file and run training with:");
    println!("  pmetal train --config {}", output);

    Ok(())
}

/// Format parameter count with suffix (K, M, B).
fn format_param_count(count: usize) -> String {
    if count >= 1_000_000_000 {
        format!("{:.2}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.2}K", count as f64 / 1_000.0)
    } else {
        format!("{}", count)
    }
}

/// Validate and sanitize an output path to prevent path traversal attacks.
///
/// This function:
/// 1. Rejects paths containing ".." components
/// 2. Rejects absolute paths that escape the current working directory
/// 3. Canonicalizes paths to resolve symlinks and normalize components
///
/// # Arguments
/// * `path` - The path to validate
/// * `context` - A description of what this path is for (used in error messages)
///
/// # Returns
/// The validated and canonicalized path, or an error if validation fails.
fn validate_output_path(path: &str, context: &str) -> anyhow::Result<PathBuf> {
    let path = PathBuf::from(path);

    // Check for explicit ".." components in the path
    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            anyhow::bail!(
                "Path traversal detected in {}: '{}' contains '..' component. \
                 Please use a path within the current directory.",
                context,
                path.display()
            );
        }
    }

    // Get the current working directory
    let cwd = std::env::current_dir()?;

    // Resolve the path
    let resolved = if path.is_absolute() {
        path.clone()
    } else {
        cwd.join(&path)
    };

    // Canonicalize after creating parent directories if needed
    // For output paths, the directory may not exist yet
    if let Some(parent) = resolved.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }

    // If the path itself exists, canonicalize it
    // Otherwise, canonicalize the parent and append the filename
    let canonical = if resolved.exists() {
        resolved.canonicalize()?
    } else if let Some(parent) = resolved.parent() {
        let canonical_parent = parent.canonicalize()?;
        if let Some(filename) = resolved.file_name() {
            canonical_parent.join(filename)
        } else {
            canonical_parent
        }
    } else {
        resolved
    };

    // Ensure the canonical path is under the current working directory
    // or is a well-known safe location like /tmp or user home
    let cwd_canonical = cwd.canonicalize()?;
    let home_dir = dirs::home_dir();
    let temp_dir = std::env::temp_dir().canonicalize().ok();

    let is_safe = canonical.starts_with(&cwd_canonical)
        || home_dir
            .as_ref()
            .map(|h| canonical.starts_with(h))
            .unwrap_or(false)
        || temp_dir
            .as_ref()
            .map(|t| canonical.starts_with(t))
            .unwrap_or(false);

    if !is_safe {
        anyhow::bail!(
            "Unsafe output path for {}: '{}' resolves to '{}' which is outside \
             the current directory, home directory, and temp directory. \
             Please use a path within a safe location.",
            context,
            path.display(),
            canonical.display()
        );
    }

    Ok(canonical)
}

/// Benchmark FFI overhead to compare Rust mlx-rs vs Python mlx performance.
///
/// Python baseline: ~7420 argmax ops/sec (~0.135ms per op) on Qwen3 vocab size.
fn run_ffi_benchmark() -> anyhow::Result<()> {
    use mlx_rs::{
        ops::indexing::{argmax, argmax_axis},
        transforms::eval,
        Array,
    };
    use std::time::Instant;

    println!("FFI Overhead Benchmark");
    println!("======================\n");

    // Warmup - ensure Metal is ready
    println!("Warming up Metal...");
    let warmup = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
    let _ = argmax(&warmup, None)?;
    eval([&warmup])?;

    // Create test array similar to Qwen3 logits
    let vocab_size = 151936; // Qwen3 vocab size
    println!("Creating logits array with vocab_size = {}", vocab_size);
    let logits_data: Vec<f32> = (0..vocab_size).map(|i| (i as f32) * 0.001).collect();
    let logits = Array::from_slice(&logits_data, &[1, vocab_size as i32]);
    eval([&logits])?;

    // Benchmark 1: argmax operations
    let n_iters = 1000;
    println!("\n--- Test 1: argmax({} iterations) ---", n_iters);

    let start = Instant::now();
    for _ in 0..n_iters {
        let result = argmax_axis(&logits, -1, None)?;
        eval([&result])?;
    }
    let elapsed = start.elapsed();

    let per_op_us = elapsed.as_micros() as f64 / n_iters as f64;
    let per_op_ms = per_op_us / 1000.0;
    let ops_per_sec = 1_000_000.0 / per_op_us;

    println!("Total time: {:?}", elapsed);
    println!("Per operation: {:.3} ms ({:.1} us)", per_op_ms, per_op_us);
    println!("Operations per second: {:.0}", ops_per_sec);
    println!("Python baseline: ~7420 ops/sec (~0.135 ms/op)");
    println!("Ratio to Python: {:.2}x", ops_per_sec / 7420.0);

    // Benchmark 2: argmax + reshape (generation loop pattern)
    println!("\n--- Test 2: argmax + reshape (generation pattern) ---");

    let token = Array::from_slice(&[42i32], &[1]);
    eval([&token])?;

    let start = Instant::now();
    for _ in 0..n_iters {
        // Reshape token [1] -> [1, 1] (like our generation loop)
        let input = token.reshape(&[1, 1])?;
        // Simulate logits extraction
        let result = argmax_axis(&logits, -1, None)?;
        eval([&input, &result])?;
    }
    let elapsed = start.elapsed();

    let per_op_us = elapsed.as_micros() as f64 / n_iters as f64;
    let per_op_ms = per_op_us / 1000.0;

    println!("Total time: {:?}", elapsed);
    println!("Per operation: {:.3} ms ({:.1} us)", per_op_ms, per_op_us);
    println!("Equivalent tok/s: {:.0}", 1_000_000.0 / per_op_us);

    // Benchmark 3: Multiple small operations (typical sampling overhead)
    println!("\n--- Test 3: Multiple ops per iteration (sampling simulation) ---");

    let start = Instant::now();
    for _ in 0..n_iters {
        // Simulate temperature scaling
        let scaled = logits.multiply(&Array::from_f32(0.7))?;
        // argmax
        let result = argmax_axis(&scaled, -1, None)?;
        eval([&result])?;
    }
    let elapsed = start.elapsed();

    let per_op_us = elapsed.as_micros() as f64 / n_iters as f64;
    let per_op_ms = per_op_us / 1000.0;

    println!("Total time: {:?}", elapsed);
    println!("Per operation: {:.3} ms ({:.1} us)", per_op_ms, per_op_us);
    println!("Equivalent tok/s: {:.0}", 1_000_000.0 / per_op_us);

    println!("\n======================");
    println!("Analysis:");
    println!("- If Rust matches Python (~0.135ms/op), FFI overhead is NOT the issue");
    println!("- If Rust is significantly slower, we may need direct mlx_sys calls");
    println!("- Compare to actual generation: ~5ms/token means overhead is elsewhere");

    // Test 4: Compare sync vs async eval timing
    println!("\n--- Test 4: async_eval timing ---");

    use mlx_rs::transforms::async_eval;

    let start = Instant::now();
    for _ in 0..n_iters {
        // Build graph (should be fast)
        let scaled = logits.multiply(&Array::from_f32(0.7))?;
        let result = argmax_axis(&scaled, -1, None)?;
        // Schedule async (should be where GPU work happens)
        async_eval([&result])?;
    }
    // Final sync
    eval([&logits])?;
    let elapsed = start.elapsed();

    let per_op_us = elapsed.as_micros() as f64 / n_iters as f64;
    println!(
        "Per iteration with async_eval: {:.3} ms ({:.1} us)",
        per_op_us / 1000.0,
        per_op_us
    );

    // Test 5: Measure graph construction vs execution
    println!("\n--- Test 5: Graph construction vs execution timing ---");

    // Just graph construction, no eval
    let start = Instant::now();
    let mut results = Vec::new();
    for _ in 0..100 {
        let scaled = logits.multiply(&Array::from_f32(0.7))?;
        let result = argmax_axis(&scaled, -1, None)?;
        results.push(result);
    }
    let graph_time = start.elapsed();
    println!("Graph construction (100 iters): {:?}", graph_time);
    println!(
        "Per graph: {:.3} ms",
        graph_time.as_micros() as f64 / 100.0 / 1000.0
    );

    // Now evaluate all
    let start = Instant::now();
    for r in &results {
        eval([r])?;
    }
    let exec_time = start.elapsed();
    println!("Execution (100 iters): {:?}", exec_time);
    println!(
        "Per execution: {:.3} ms",
        exec_time.as_micros() as f64 / 100.0 / 1000.0
    );

    Ok(())
}

/// Benchmark generation loop timing with a real model.
///
/// This profiles each step of the generation loop to compare with mlx_lm's timing.
async fn run_gen_benchmark(model_id: &str) -> anyhow::Result<()> {
    use mlx_rs::{
        ops::indexing::{argmax, IndexOp},
        transforms::{async_eval, eval},
        Array,
    };
    use pmetal_models::DynamicModel;
    use std::path::PathBuf;
    use std::time::Instant;

    println!("=== Generation Loop Benchmark ===");
    println!("Model: {}\n", model_id);

    // Download model if needed
    let model_path = if model_id.contains('/') && !PathBuf::from(model_id).exists() {
        pmetal_hub::download_model(model_id, None, None).await?
    } else {
        PathBuf::from(model_id)
    };

    // Load model
    println!("Loading model...");
    let mut model = DynamicModel::from_pretrained(&model_path)?;
    println!("Model loaded.\n");

    // Create KV cache
    let mut cache = model.create_cache(256);

    // Warmup
    println!("Warming up...");
    let token = Array::from_slice(&[42i32], &[1, 1]);
    let _ = model.forward_with_cache(&token, None, Some(&mut cache))?;
    eval([&token])?;
    cache = model.create_cache(256);
    println!("Warmup complete.\n");

    // Profile generation loop
    let n_tokens = 50;
    println!("Profiling {} token generation...\n", n_tokens);

    // Initial forward pass
    let logits = model.forward_with_cache(&token, None, Some(&mut cache))?;
    let mut current_token = argmax(&logits.index((.., -1, ..)), None)?;
    async_eval([&current_token])?;

    let mut times = std::collections::HashMap::new();
    times.insert("reshape", Vec::new());
    times.insert("forward", Vec::new());
    times.insert("extract_logits", Vec::new());
    times.insert("argmax", Vec::new());
    times.insert("async_eval", Vec::new());
    times.insert("item", Vec::new());
    times.insert("total", Vec::new());

    for _ in 0..n_tokens {
        let total_start = Instant::now();

        // Reshape token to [1, 1]
        let t0 = Instant::now();
        let next_input = current_token.reshape(&[1, 1])?;
        times.get_mut("reshape").unwrap().push(t0.elapsed());

        // Forward pass
        let t0 = Instant::now();
        let next_logits = model.forward_with_cache(&next_input, None, Some(&mut cache))?;
        times.get_mut("forward").unwrap().push(t0.elapsed());

        // Extract last logits
        let t0 = Instant::now();
        let last_logits = next_logits.index((.., -1, ..));
        times.get_mut("extract_logits").unwrap().push(t0.elapsed());

        // Argmax
        let t0 = Instant::now();
        let next_token = argmax(&last_logits, None)?;
        times.get_mut("argmax").unwrap().push(t0.elapsed());

        // Async eval for next
        let t0 = Instant::now();
        async_eval([&next_token])?;
        times.get_mut("async_eval").unwrap().push(t0.elapsed());

        // Extract current token (sync point)
        let t0 = Instant::now();
        let _ = current_token.item::<i32>();
        times.get_mut("item").unwrap().push(t0.elapsed());

        times.get_mut("total").unwrap().push(total_start.elapsed());

        current_token = next_token;
    }

    // Print results
    println!("=== Generation Loop Timing ===");
    for (name, durations) in &times {
        let avg_us: f64 =
            durations.iter().map(|d| d.as_micros() as f64).sum::<f64>() / durations.len() as f64;
        let avg_ms = avg_us / 1000.0;
        println!("{:15}: {:7.3}ms", name, avg_ms);
    }

    let total_avg: f64 = times["total"]
        .iter()
        .map(|d| d.as_micros() as f64)
        .sum::<f64>()
        / times["total"].len() as f64;
    println!("\nEffective tok/s: {:.0}", 1_000_000.0 / total_avg);

    println!("\n=== Comparison ===");
    println!("Python mlx_lm reference:");
    println!("  build_graph: 0.570ms");
    println!("  async_eval:  3.111ms");
    println!("  item:        0.004ms");
    println!("  total:       3.686ms (271 tok/s)");

    Ok(())
}
