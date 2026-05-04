//! Dynamic LoRA model dispatch based on architecture.
//!
//! This module provides automatic architecture detection and model loading
//! for LoRA training, eliminating the need for hardcoded model types.
//!
//! # Example
//!
//! ```ignore
//! use pmetal_lora::DynamicLoraModel;
//! use pmetal_core::LoraConfig;
//!
//! // Automatically detect architecture and create LoRA model
//! let mut model = DynamicLoraModel::from_pretrained(
//!     "/path/to/model",
//!     LoraConfig::default(),
//! ).await?;
//!
//! // Training works regardless of architecture
//! let logits = model.forward(&input_ids, None)?;
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use pmetal_bridge::compat::Array;
use pmetal_bridge::compat::Exception;
use pmetal_core::LoraConfig;
use pmetal_mlx::kv_cache::KVCache;
use pmetal_models::{
    GgufModelConfig, ModelArchitecture, WeightFormat, WeightFormatError, WeightLoader,
};

use crate::{
    LoraError, TrainableModel, cohere_lora::CohereLoraForCausalLM,
    deepseek_lora::DeepSeekLoraForCausalLM, gemma_lora::GemmaLoraForCausalLM,
    gemma4_lora::Gemma4LoraForCausalLM, gpt_oss_lora::GptOssLoraForCausalLM,
    granite_lora::GraniteLoraForCausalLM, llama_lora::LlamaLoraForCausalLM,
    llama4_lora::Llama4LoraForCausalLM, mistral_lora::MistralLoraForCausalLM,
    nemotron_h_lora::NemotronHLoraForCausalLM, phi_lora::PhiLoraForCausalLM,
    qwen3_lora::Qwen3LoraForCausalLM, qwen3_moe_lora::Qwen3MoELoraForCausalLM,
    qwen3_next_lora::Qwen3NextLoraForCausalLM,
};

/// Dispatch a method call uniformly across all `DynamicLoraModel` variants.
///
/// Every arm expands to `m.$method($args...)` where `m` is the inner model.
/// Use this only for methods where ALL variants have identical call signatures.
macro_rules! dispatch_lora_uniform {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            Self::Llama(m) => m.$method($($arg),*),
            Self::Mistral(m) => m.$method($($arg),*),
            Self::Qwen3(m) => m.$method($($arg),*),
            Self::Gemma(m) => m.$method($($arg),*),
            Self::Phi(m) => m.$method($($arg),*),
            Self::Qwen3Next(m) => m.$method($($arg),*),
            Self::Qwen3MoE(m) => m.$method($($arg),*),
            Self::Gemma4(m) => m.$method($($arg),*),
            Self::GptOss(m) => m.$method($($arg),*),
            Self::Granite(m) => m.$method($($arg),*),
            Self::Llama4(m) => m.$method($($arg),*),
            Self::DeepSeek(m) => m.$method($($arg),*),
            Self::NemotronH(m) => m.$method($($arg),*),
            Self::Cohere(m) => m.$method($($arg),*),
        }
    };
}

/// Map each `DynamicLoraModel` variant to its corresponding `ModelArchitecture` constant.
macro_rules! dispatch_lora_architecture {
    ($self:expr) => {
        match $self {
            Self::Llama(_) => ModelArchitecture::Llama,
            Self::Mistral(_) => ModelArchitecture::Mistral,
            Self::Qwen3(_) => ModelArchitecture::Qwen3,
            Self::Gemma(_) => ModelArchitecture::Gemma,
            Self::Phi(_) => ModelArchitecture::Phi,
            Self::Qwen3Next(_) => ModelArchitecture::Qwen3Next,
            Self::Qwen3MoE(_) => ModelArchitecture::Qwen3MoE,
            Self::Gemma4(_) => ModelArchitecture::Gemma4,
            Self::GptOss(_) => ModelArchitecture::GptOss,
            Self::Granite(_) => ModelArchitecture::Granite,
            Self::Llama4(_) => ModelArchitecture::Llama4,
            Self::DeepSeek(_) => ModelArchitecture::DeepSeek,
            Self::NemotronH(_) => ModelArchitecture::NemotronH,
            Self::Cohere(_) => ModelArchitecture::Cohere,
        }
    };
}

/// Dynamic LoRA model container using enum dispatch.
///
/// This approach uses static dispatch via enum variants rather than
/// trait objects, which is more efficient while still providing
/// runtime polymorphism based on detected architecture.
///
/// # Supported Architectures
///
/// - Llama (2, 3, 3.1, 3.2, 3.3) — gradient checkpointing supported
/// - Qwen2 (2, 2.5) — uses Qwen3 LoRA implementation internally
/// - Qwen3 — dense attention + RoPE reset support
/// - Qwen3Next (3.5) — hybrid architecture with nested `text_config` handling
/// - Gemma (2, 3) — GeGLU activation, special RMSNorm
/// - Mistral (7B, Mixtral 8x7B) — sliding window attention support
/// - Phi (3, 3.5) — partial RoPE, fused gate_up projection
///
/// Architectures not listed (Llama 4, Qwen3MoE, DeepSeek, Phi4, etc.) return
/// `DynamicLoraError::NotImplemented`.
///
/// # Architecture-Specific Features
///
/// | Feature | Llama | Mistral | Qwen3 | Gemma | Phi | Qwen3Next |
/// |---------|-------|---------|-------|-------|-----|-----------|
/// | LoRA Training | Yes | Yes | Yes | Yes | Yes | Yes |
/// | QLoRA | Yes | Yes | Yes | Yes | — | — |
/// | Gradient Checkpointing | Yes | Yes | No | Yes | Yes | No |
/// | Packed Sequences | Yes | Yes | Yes | Yes | Yes | Yes |
// The Gemma4 variant (per-layer embeddings + KV-sharing bookkeeping) is
// ~1.6 KB vs ~960 B for the next-largest variant. This enum is constructed
// once per training run and dispatched through references, not copied, so the
// size delta is not on any hot path — boxing would just churn the ~20-method
// TrainableModel dispatch sites.
#[allow(clippy::large_enum_variant)]
pub enum DynamicLoraModel {
    /// Llama family with LoRA adapters (supports gradient checkpointing).
    Llama(LlamaLoraForCausalLM),
    /// Mistral family with LoRA adapters (supports gradient checkpointing and SWA).
    Mistral(MistralLoraForCausalLM),
    /// Qwen3 family with LoRA adapters.
    Qwen3(Qwen3LoraForCausalLM),
    /// Gemma family with LoRA adapters (supports GeGLU and special RMSNorm).
    Gemma(GemmaLoraForCausalLM),
    /// Phi family with LoRA adapters (supports partial RoPE and fused gate_up).
    Phi(PhiLoraForCausalLM),
    /// Qwen3.5 (qwen3_next) hybrid family with LoRA adapters.
    Qwen3Next(Qwen3NextLoraForCausalLM),
    /// Qwen3-MoE family with attention + router/dense-MLP adapters.
    Qwen3MoE(Qwen3MoELoraForCausalLM),
    /// Gemma4 family with PLE-aware adapters.
    Gemma4(Gemma4LoraForCausalLM),
    /// GPT-OSS family with attention adapters.
    GptOss(GptOssLoraForCausalLM),
    /// Granite family with attention adapters.
    Granite(GraniteLoraForCausalLM),
    /// Llama 4 family (dense + MoE) with attention adapters.
    Llama4(Llama4LoraForCausalLM),
    /// DeepSeek family (MLA + MoE) with attention adapters.
    DeepSeek(DeepSeekLoraForCausalLM),
    /// NemotronH hybrid family (Mamba + attention) with attention adapters.
    NemotronH(NemotronHLoraForCausalLM),
    /// Cohere / Cohere2 / Command-R family with attention adapters.
    Cohere(CohereLoraForCausalLM),
}

impl std::fmt::Debug for DynamicLoraModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Llama(_) => write!(f, "DynamicLoraModel::Llama"),
            Self::Mistral(_) => write!(f, "DynamicLoraModel::Mistral"),
            Self::Qwen3(_) => write!(f, "DynamicLoraModel::Qwen3"),
            Self::Gemma(_) => write!(f, "DynamicLoraModel::Gemma"),
            Self::Phi(_) => write!(f, "DynamicLoraModel::Phi"),
            Self::Qwen3Next(_) => write!(f, "DynamicLoraModel::Qwen3Next"),
            Self::Qwen3MoE(_) => write!(f, "DynamicLoraModel::Qwen3MoE"),
            Self::Gemma4(_) => write!(f, "DynamicLoraModel::Gemma4"),
            Self::GptOss(_) => write!(f, "DynamicLoraModel::GptOss"),
            Self::Granite(_) => write!(f, "DynamicLoraModel::Granite"),
            Self::Llama4(_) => write!(f, "DynamicLoraModel::Llama4"),
            Self::DeepSeek(_) => write!(f, "DynamicLoraModel::DeepSeek"),
            Self::NemotronH(_) => write!(f, "DynamicLoraModel::NemotronH"),
            Self::Cohere(_) => write!(f, "DynamicLoraModel::Cohere"),
        }
    }
}

impl DynamicLoraModel {
    /// Create a LoRA model from a pretrained model directory.
    ///
    /// This function:
    /// 1. Reads config.json to detect the model architecture
    /// 2. Instantiates the correct LoRA model type
    /// 3. Loads base model weights from safetensors files
    ///
    /// # Arguments
    ///
    /// * `model_dir` - Path to model directory containing config.json and weights
    /// * `lora_config` - LoRA configuration (rank, alpha, target modules)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = LoraConfig::default();
    /// let model = DynamicLoraModel::from_pretrained("/path/to/llama-3.2-1b", config)?;
    /// let model = DynamicLoraModel::from_pretrained("/path/to/qwen3-0.6b", config)?;
    /// ```
    pub fn from_pretrained(
        model_dir: impl AsRef<Path>,
        lora_config: LoraConfig,
    ) -> Result<Self, DynamicLoraError> {
        let model_dir = model_dir.as_ref();

        // Detect architecture
        let arch = ModelArchitecture::detect(model_dir)?;

        tracing::info!("Detected architecture: {}", arch);

        // Read config content
        let config_path = model_dir.join("config.json");
        let config_content = std::fs::read_to_string(&config_path)?;

        // Create and load the appropriate model
        match arch {
            ModelArchitecture::Llama => {
                let llama_config: pmetal_models::architectures::llama::LlamaConfig =
                    serde_json::from_str(&config_content)?;

                let mut model = LlamaLoraForCausalLM::new(llama_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                Ok(DynamicLoraModel::Llama(model))
            }
            ModelArchitecture::Mistral => {
                let mistral_config: pmetal_models::architectures::mistral::MistralConfig =
                    serde_json::from_str(&config_content)?;

                let mut model = MistralLoraForCausalLM::new(mistral_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loaded Mistral LoRA model");
                Ok(DynamicLoraModel::Mistral(model))
            }
            ModelArchitecture::Qwen3 => {
                let qwen_config: pmetal_models::architectures::qwen3::Qwen3Config =
                    serde_json::from_str(&config_content)?;

                let mut model = Qwen3LoraForCausalLM::new(qwen_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                Ok(DynamicLoraModel::Qwen3(model))
            }
            // Qwen2 uses Qwen3 implementation for now (similar architecture)
            ModelArchitecture::Qwen2 => {
                // Qwen2 and Qwen3 share the same base structure
                // We can treat Qwen2 as Qwen3 for LoRA training
                let qwen_config: pmetal_models::architectures::qwen3::Qwen3Config =
                    serde_json::from_str(&config_content)?;

                let mut model = Qwen3LoraForCausalLM::new(qwen_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loading Qwen2 model with Qwen3 LoRA implementation");
                Ok(DynamicLoraModel::Qwen3(model))
            }
            ModelArchitecture::Gemma => {
                let gemma_config: pmetal_models::architectures::gemma::GemmaConfig =
                    serde_json::from_str(&config_content)?;

                let mut model = GemmaLoraForCausalLM::new(gemma_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loaded Gemma LoRA model");
                Ok(DynamicLoraModel::Gemma(model))
            }
            ModelArchitecture::Phi | ModelArchitecture::Phi4 => {
                // Phi4 is the same architecture as Phi3 — just a different config
                // preset (larger hidden size, qkv_bias=true). PhiLoraForCausalLM
                // already honors those flags.
                let phi_config: pmetal_models::architectures::phi::PhiConfig =
                    serde_json::from_str(&config_content)?;

                let mut model = PhiLoraForCausalLM::new(phi_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loaded Phi LoRA model (dispatched from {arch:?})");
                Ok(DynamicLoraModel::Phi(model))
            }
            ModelArchitecture::Qwen3Next => {
                // Qwen 3.5 configs may nest model params inside `text_config`
                let config_json: serde_json::Value = serde_json::from_str(&config_content)?;
                let text_config_str = if config_json.get("text_config").is_some()
                    && config_json.get("hidden_size").is_none()
                {
                    serde_json::to_string(&config_json["text_config"])?
                } else {
                    config_content.clone()
                };
                let mut qwen_config: pmetal_models::architectures::qwen3_next::Qwen3NextConfig =
                    serde_json::from_str(&text_config_str)?;
                qwen_config.apply_rope_parameters();

                let mut model = Qwen3NextLoraForCausalLM::new(qwen_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loaded Qwen3.5 (qwen3_next) LoRA model");
                Ok(DynamicLoraModel::Qwen3Next(model))
            }
            ModelArchitecture::Qwen3MoE => {
                let qwen_config: pmetal_models::architectures::qwen3_moe::Qwen3MoEConfig =
                    serde_json::from_str(&config_content)?;

                let mut model = Qwen3MoELoraForCausalLM::new(qwen_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loaded Qwen3-MoE LoRA model");
                Ok(DynamicLoraModel::Qwen3MoE(model))
            }
            ModelArchitecture::Gemma4 => {
                let config_json: serde_json::Value = serde_json::from_str(&config_content)?;
                let effective = if config_json.get("text_config").is_some()
                    && config_json.get("hidden_size").is_none()
                {
                    serde_json::to_string(&config_json["text_config"])?
                } else {
                    config_content.clone()
                };
                let gemma4_config: pmetal_models::architectures::gemma4::Gemma4Config =
                    serde_json::from_str(&effective)?;

                let mut model = Gemma4LoraForCausalLM::new(gemma4_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loaded Gemma4 LoRA model");
                Ok(DynamicLoraModel::Gemma4(model))
            }
            ModelArchitecture::GptOss => {
                let gpt_oss_config: pmetal_models::architectures::gpt_oss::GptOssConfig =
                    serde_json::from_str(&config_content)?;

                let mut model = GptOssLoraForCausalLM::new(gpt_oss_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loaded GPT-OSS LoRA model");
                Ok(DynamicLoraModel::GptOss(model))
            }
            ModelArchitecture::Granite => {
                let granite_config: pmetal_models::architectures::granite::GraniteConfig =
                    serde_json::from_str(&config_content)?;

                let mut model = GraniteLoraForCausalLM::new(granite_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loaded Granite LoRA model");
                Ok(DynamicLoraModel::Granite(model))
            }
            ModelArchitecture::Llama4 => {
                let config_json: serde_json::Value = serde_json::from_str(&config_content)?;
                let effective = if config_json.get("text_config").is_some()
                    && config_json.get("hidden_size").is_none()
                {
                    serde_json::to_string(&config_json["text_config"])?
                } else {
                    config_content.clone()
                };
                let llama4_config: pmetal_models::architectures::llama4::Llama4TextConfig =
                    serde_json::from_str(&effective)?;

                let mut model = Llama4LoraForCausalLM::new(llama4_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_lora_params()?;

                tracing::info!("Loaded Llama 4 LoRA model");
                Ok(DynamicLoraModel::Llama4(model))
            }
            ModelArchitecture::DeepSeek => {
                let deepseek_config: pmetal_models::architectures::deepseek::DeepSeekConfig =
                    serde_json::from_str(&config_content)?;

                let mut model = DeepSeekLoraForCausalLM::new(deepseek_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loaded DeepSeek LoRA model");
                Ok(DynamicLoraModel::DeepSeek(model))
            }
            ModelArchitecture::NemotronH => {
                let nemotron_config: pmetal_models::architectures::nemotron_h::NemotronHConfig =
                    serde_json::from_str(&config_content)?;

                let mut model = NemotronHLoraForCausalLM::new(nemotron_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loaded NemotronH LoRA model");
                Ok(DynamicLoraModel::NemotronH(model))
            }
            ModelArchitecture::Cohere => {
                let cohere_config: pmetal_models::architectures::cohere::CohereConfig =
                    serde_json::from_str(&config_content)?;

                let mut model = CohereLoraForCausalLM::new(cohere_config, lora_config)?;
                model.load_base_weights_from_dir(model_dir)?;
                model.eval_all()?;

                tracing::info!("Loaded Cohere LoRA model");
                Ok(DynamicLoraModel::Cohere(model))
            }
            // Other architectures not yet supported for LoRA training
            arch => Err(DynamicLoraError::NotImplemented(arch)),
        }
    }

    /// Create a LoRA model from a GGUF file.
    ///
    /// This function:
    /// 1. Reads GGUF metadata to detect the model architecture
    /// 2. Extracts model configuration from GGUF metadata
    /// 3. Dequantizes weights to f32 for training
    /// 4. Loads weights into the appropriate LoRA model
    ///
    /// # Arguments
    ///
    /// * `gguf_path` - Path to .gguf file or directory containing .gguf
    /// * `lora_config` - LoRA configuration (rank, alpha, target modules)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = LoraConfig::default();
    /// let model = DynamicLoraModel::from_gguf("./model.gguf", config)?;
    /// ```
    pub fn from_gguf(
        gguf_path: impl AsRef<Path>,
        lora_config: LoraConfig,
    ) -> Result<Self, DynamicLoraError> {
        let gguf_path = gguf_path.as_ref();

        // Load weights from GGUF (auto-dequantizes and maps tensor names)
        tracing::info!("Loading weights from GGUF: {:?}", gguf_path);
        let weights = WeightLoader::load_gguf(gguf_path)?;

        // Find the actual GGUF file for metadata extraction
        let gguf_file = if gguf_path.is_file() {
            gguf_path.to_path_buf()
        } else {
            // Find .gguf file in directory
            std::fs::read_dir(gguf_path)?
                .filter_map(|e| e.ok())
                .find(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext.to_string_lossy().to_lowercase() == "gguf")
                        .unwrap_or(false)
                })
                .map(|e| e.path())
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("No .gguf file found in {:?}", gguf_path),
                    )
                })?
        };

        // Read GGUF content for config extraction
        let content = pmetal_gguf::GgufContent::from_file(&gguf_file)
            .map_err(|e| WeightFormatError::Gguf(e.to_string()))?;

        // Extract model config from GGUF metadata
        let gguf_config = GgufModelConfig::from_gguf(&content)?;

        tracing::info!(
            "GGUF architecture: {}, hidden_size: {}, layers: {}",
            gguf_config.architecture,
            gguf_config.hidden_size,
            gguf_config.num_hidden_layers
        );

        // Map GGUF architecture name to ModelArchitecture
        let arch = match gguf_config.architecture.to_lowercase().as_str() {
            "llama" => ModelArchitecture::Llama,
            "qwen2" => ModelArchitecture::Qwen2,
            "qwen3" => ModelArchitecture::Qwen3,
            "mistral" => ModelArchitecture::Mistral,
            "gemma" => ModelArchitecture::Gemma,
            "phi" => ModelArchitecture::Phi,
            "phi3" | "phi4" => ModelArchitecture::Phi4,
            _ => {
                return Err(DynamicLoraError::Mlx(Exception::custom(format!(
                    "Unsupported architecture for LoRA: {}",
                    gguf_config.architecture
                ))));
            }
        };

        // Create model with extracted config and load weights
        Self::from_weights_with_arch(weights, arch, gguf_config, lora_config)
    }

    /// Create a LoRA model from pre-loaded weights with known architecture.
    ///
    /// This is the internal implementation used by both `from_pretrained` and `from_gguf`.
    fn from_weights_with_arch(
        weights: HashMap<String, Array>,
        arch: ModelArchitecture,
        gguf_config: GgufModelConfig,
        lora_config: LoraConfig,
    ) -> Result<Self, DynamicLoraError> {
        match arch {
            ModelArchitecture::Llama => {
                let config = gguf_config.to_llama_config();
                let mut model = LlamaLoraForCausalLM::new(config, lora_config)?;
                model.load_base_weights(&weights)?;
                model.eval_all()?;
                Ok(DynamicLoraModel::Llama(model))
            }
            ModelArchitecture::Mistral => {
                let config = gguf_config.to_mistral_config();
                let mut model = MistralLoraForCausalLM::new(config, lora_config)?;
                model.load_base_weights(&weights)?;
                model.eval_all()?;
                Ok(DynamicLoraModel::Mistral(model))
            }
            ModelArchitecture::Qwen3 | ModelArchitecture::Qwen2 => {
                let config = gguf_config.to_qwen3_config();
                let mut model = Qwen3LoraForCausalLM::new(config, lora_config)?;
                model.load_base_weights(&weights)?;
                model.eval_all()?;
                Ok(DynamicLoraModel::Qwen3(model))
            }
            ModelArchitecture::Gemma => {
                let config = gguf_config.to_gemma_config();
                let mut model = GemmaLoraForCausalLM::new(config, lora_config)?;
                model.load_base_weights(&weights)?;
                model.eval_all()?;
                Ok(DynamicLoraModel::Gemma(model))
            }
            ModelArchitecture::Phi | ModelArchitecture::Phi4 => {
                let config = gguf_config.to_phi_config();
                let mut model = PhiLoraForCausalLM::new(config, lora_config)?;
                model.load_base_weights(&weights)?;
                model.eval_all()?;
                Ok(DynamicLoraModel::Phi(model))
            }
            arch => Err(DynamicLoraError::NotImplemented(arch)),
        }
    }

    /// Get the detected architecture.
    pub fn architecture(&self) -> ModelArchitecture {
        dispatch_lora_architecture!(self)
    }

    /// Get the architecture name as a string.
    pub fn architecture_name(&self) -> &'static str {
        match self {
            Self::Llama(_) => "Llama",
            Self::Mistral(_) => "Mistral",
            Self::Qwen3(_) => "Qwen3",
            Self::Gemma(_) => "Gemma",
            Self::Phi(_) => "Phi",
            Self::Qwen3Next(_) => "Qwen3Next",
            Self::Qwen3MoE(_) => "Qwen3MoE",
            Self::Gemma4(_) => "Gemma4",
            Self::GptOss(_) => "GptOss",
            Self::Granite(_) => "Granite",
            Self::Llama4(_) => "Llama4",
            Self::DeepSeek(_) => "DeepSeek",
            Self::NemotronH(_) => "NemotronH",
            Self::Cohere(_) => "Cohere",
        }
    }

    /// Merge LoRA weights into base weights.
    pub fn merge_lora(&mut self) -> Result<(), LoraError> {
        match self {
            Self::Llama(m) => m.merge_lora(),
            Self::Mistral(m) => m.merge_lora(),
            Self::Qwen3(m) => m.merge_lora(),
            Self::Gemma(m) => m.merge_lora(),
            Self::Phi(m) => m.merge_lora(),
            Self::Qwen3Next(m) => m.merge_lora(),
            Self::Qwen3MoE(m) => m.merge_lora(),
            Self::Gemma4(m) => m.merge_lora(),
            Self::GptOss(m) => m.merge_lora(),
            Self::Granite(m) => m.merge_lora(),
            Self::Llama4(m) => m.merge_lora(),
            Self::NemotronH(m) => m.merge_lora(),
            Self::DeepSeek(m) => m.merge_lora(),
            Self::Cohere(m) => m.merge_lora(),
        }
    }

    /// Quantize merged base linear weights to FP8 E4M3 for inference.
    pub fn quantize_fp8(&mut self) -> Result<(), LoraError> {
        pmetal_models::fp8_utils::quantize_model_linears(self)?;
        Ok(())
    }

    /// Unmerge is not supported.
    pub fn unmerge_lora(&mut self) -> Result<(), LoraError> {
        match self {
            Self::Llama(m) => m.unmerge_lora(),
            Self::Mistral(m) => m.unmerge_lora(),
            Self::Qwen3(m) => m.unmerge_lora(),
            Self::Gemma(m) => m.unmerge_lora(),
            Self::Phi(m) => m.unmerge_lora(),
            Self::Qwen3Next(m) => m.unmerge_lora(),
            Self::Qwen3MoE(m) => m.unmerge_lora(),
            Self::Gemma4(m) => m.unmerge_lora(),
            Self::GptOss(m) => m.unmerge_lora(),
            Self::Granite(m) => m.unmerge_lora(),
            Self::Llama4(m) => m.unmerge_lora(),
            Self::NemotronH(m) => m.unmerge_lora(),
            Self::DeepSeek(m) => m.unmerge_lora(),
            Self::Cohere(m) => m.unmerge_lora(),
        }
    }
}

// Implement TrainableModel for DynamicLoraModel via dispatch
impl pmetal_bridge::compat::ModuleParameters for DynamicLoraModel {
    fn num_parameters(&self) -> usize {
        dispatch_lora_uniform!(self, num_trainable_params)
    }

    fn parameters(&self) -> pmetal_bridge::compat::ModuleParamRef<'_> {
        dispatch_lora_uniform!(self, parameters)
    }

    fn parameters_mut(&mut self) -> pmetal_bridge::compat::ModuleParamMut<'_> {
        dispatch_lora_uniform!(self, parameters_mut)
    }

    fn trainable_parameters(&self) -> pmetal_bridge::compat::ModuleParamRef<'_> {
        dispatch_lora_uniform!(self, trainable_parameters)
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        dispatch_lora_uniform!(self, freeze_parameters, recursive)
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        dispatch_lora_uniform!(self, unfreeze_parameters, recursive)
    }

    fn all_frozen(&self) -> Option<bool> {
        dispatch_lora_uniform!(self, all_frozen)
    }

    fn any_frozen(&self) -> Option<bool> {
        dispatch_lora_uniform!(self, any_frozen)
    }
}

impl TrainableModel for DynamicLoraModel {
    fn forward(&mut self, input_ids: &Array, mask: Option<&Array>) -> Result<Array, LoraError> {
        match self {
            Self::Llama(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Mistral(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Qwen3(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Gemma(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Phi(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Qwen3Next(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Qwen3MoE(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Gemma4(m) => TrainableModel::forward(m, input_ids, mask),
            Self::GptOss(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Granite(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Llama4(m) => TrainableModel::forward(m, input_ids, mask),
            Self::DeepSeek(m) => TrainableModel::forward(m, input_ids, mask),
            Self::NemotronH(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Cohere(m) => TrainableModel::forward(m, input_ids, mask),
        }
    }

    fn forward_noised(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        noise_alpha: f32,
    ) -> Result<Array, LoraError> {
        match self {
            Self::Llama(m) => TrainableModel::forward_noised(m, input_ids, mask, noise_alpha),
            Self::Mistral(m) => TrainableModel::forward_noised(m, input_ids, mask, noise_alpha),
            Self::Qwen3(m) => TrainableModel::forward_noised(m, input_ids, mask, noise_alpha),
            Self::Gemma(m) => TrainableModel::forward_noised(m, input_ids, mask, noise_alpha),
            Self::Phi(m) => TrainableModel::forward_noised(m, input_ids, mask, noise_alpha),
            Self::Qwen3Next(m) => TrainableModel::forward_noised(m, input_ids, mask, noise_alpha),
            Self::Qwen3MoE(m) => TrainableModel::forward_noised(m, input_ids, mask, noise_alpha),
            Self::Gemma4(m) => TrainableModel::forward_noised(m, input_ids, mask, noise_alpha),
            Self::GptOss(m) => TrainableModel::forward_noised(m, input_ids, mask, noise_alpha),
            Self::Granite(m) => TrainableModel::forward_noised(m, input_ids, mask, noise_alpha),
            Self::Llama4(m) => TrainableModel::forward_noised(m, input_ids, mask, noise_alpha),
            Self::DeepSeek(m) => TrainableModel::forward_noised(m, input_ids, mask, noise_alpha),
            Self::NemotronH(m) => TrainableModel::forward_noised(m, input_ids, mask, noise_alpha),
            Self::Cohere(m) => TrainableModel::forward_noised(m, input_ids, mask, noise_alpha),
        }
    }

    fn forward_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Result<Array, LoraError> {
        match self {
            Self::Llama(m) => m.forward_with_positions(input_ids, mask, position_ids),
            Self::Mistral(m) => m.forward_with_positions(input_ids, mask, position_ids),
            Self::Qwen3(m) => m.forward_with_positions(input_ids, mask, position_ids),
            // Gemma, Phi, and Qwen3Next don't have forward_with_positions yet, fallback to standard forward
            Self::Gemma(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Phi(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Qwen3Next(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Qwen3MoE(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Gemma4(m) => TrainableModel::forward(m, input_ids, mask),
            Self::GptOss(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Granite(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Llama4(m) => TrainableModel::forward(m, input_ids, mask),
            Self::DeepSeek(m) => TrainableModel::forward(m, input_ids, mask),
            Self::NemotronH(m) => TrainableModel::forward(m, input_ids, mask),
            Self::Cohere(m) => TrainableModel::forward(m, input_ids, mask),
        }
    }

    fn num_trainable_params(&self) -> usize {
        dispatch_lora_uniform!(self, num_trainable_params)
    }

    fn lora_parameters(&self) -> HashMap<Rc<str>, Array> {
        dispatch_lora_uniform!(self, lora_parameters)
    }

    fn set_lora_parameters(&mut self, params: &HashMap<Rc<str>, Array>) {
        dispatch_lora_uniform!(self, set_lora_parameters, params)
    }

    fn save_lora_weights(&self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        dispatch_lora_uniform!(self, save_lora_weights, path)
    }

    fn load_lora_weights(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), LoraError> {
        dispatch_lora_uniform!(self, load_lora_weights, path)
    }

    fn enable_gradient_checkpointing(&mut self, layers_per_block: usize) {
        dispatch_lora_uniform!(self, enable_gradient_checkpointing, layers_per_block)
    }

    fn disable_gradient_checkpointing(&mut self) {
        dispatch_lora_uniform!(self, disable_gradient_checkpointing)
    }

    fn supports_gradient_checkpointing(&self) -> bool {
        dispatch_lora_uniform!(self, supports_gradient_checkpointing)
    }

    fn forward_with_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        cache: Option<&mut KVCache>,
    ) -> Result<Array, LoraError> {
        match self {
            Self::Llama(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Mistral(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Qwen3(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Gemma(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Phi(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Qwen3Next(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Qwen3MoE(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Gemma4(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::GptOss(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Granite(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Llama4(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::DeepSeek(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::NemotronH(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
            Self::Cohere(m) => TrainableModel::forward_with_cache(m, input_ids, mask, cache),
        }
    }

    fn create_cache(&self, max_seq_len: usize) -> Option<KVCache> {
        match self {
            Self::Llama(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Mistral(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Qwen3(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Gemma(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Phi(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Qwen3Next(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Qwen3MoE(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Gemma4(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::GptOss(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Granite(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::Llama4(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::DeepSeek(m) => TrainableModel::create_cache(m, max_seq_len),
            Self::NemotronH(_) => None,
            Self::Cohere(m) => TrainableModel::create_cache(m, max_seq_len),
        }
    }

    fn supports_kv_cache(&self) -> bool {
        match self {
            Self::Llama(_) => true,
            Self::Mistral(_) => true,
            Self::Qwen3(_) => true,
            Self::Gemma(_) => true,
            Self::Phi(_) => true,
            Self::Qwen3Next(_) => false,
            Self::Qwen3MoE(_) => true,
            Self::Gemma4(_) => true,
            Self::GptOss(_) => true,
            Self::Granite(_) => true,
            Self::Llama4(_) => true,
            Self::DeepSeek(_) => true,
            Self::NemotronH(_) => false,
            Self::Cohere(_) => true,
        }
    }

    fn forward_hidden(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
    ) -> Option<Result<Array, LoraError>> {
        dispatch_lora_uniform!(self, forward_hidden, input_ids, mask)
    }

    fn forward_hidden_with_positions(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        position_ids: &Array,
    ) -> Option<Result<Array, LoraError>> {
        dispatch_lora_uniform!(
            self,
            forward_hidden_with_positions,
            input_ids,
            mask,
            position_ids
        )
    }

    fn lm_head_weight(&self) -> Option<Array> {
        dispatch_lora_uniform!(self, lm_head_weight)
    }
}

impl DynamicLoraModel {
    /// Create a Mamba cache for hybrid models (GDN recurrent state).
    pub fn create_mamba_cache(&self) -> Option<pmetal_mlx::kv_cache::MambaCache> {
        match self {
            Self::Qwen3Next(m) => Some(m.create_mamba_cache()),
            _ => None,
        }
    }

    /// Evaluate all model parameters (force GPU computation).
    ///
    /// Call this after loading LoRA weights to ensure the adapter parameters
    /// are materialized on the device before running inference.
    pub fn eval_all(&mut self) -> Result<(), LoraError> {
        match self {
            Self::Llama(m) => m.eval_all(),
            Self::Mistral(m) => m.eval_all(),
            Self::Qwen3(m) => m.eval_all(),
            Self::Gemma(m) => m.eval_all(),
            Self::Phi(m) => m.eval_all(),
            Self::Qwen3Next(m) => m.eval_all(),
            Self::Qwen3MoE(m) => m.eval_all(),
            Self::Gemma4(m) => m.eval_all(),
            Self::GptOss(m) => m.eval_all(),
            Self::Granite(m) => m.eval_all(),
            // Llama4 calls eval_lora_params rather than eval_all
            Self::Llama4(m) => m.eval_all(),
            Self::NemotronH(m) => m.eval_all(),
            Self::DeepSeek(m) => m.eval_all(),
            Self::Cohere(m) => m.eval_all(),
        }
    }

    /// Forward with hybrid cache support (KV cache + Mamba cache).
    ///
    /// For non-hybrid models, the mamba_cache is ignored and standard
    /// KV-cached forward is used.
    pub fn forward_with_hybrid_cache(
        &mut self,
        input_ids: &Array,
        mask: Option<&Array>,
        kv_cache: Option<&mut KVCache>,
        mamba_cache: Option<&mut pmetal_mlx::kv_cache::MambaCache>,
    ) -> Result<Array, LoraError> {
        match self {
            Self::Qwen3Next(m) => m.forward_with_cache(input_ids, mask, kv_cache, mamba_cache),
            _ => {
                // Non-hybrid: delegate to standard forward_with_cache (ignore mamba_cache)
                TrainableModel::forward_with_cache(self, input_ids, mask, kv_cache)
            }
        }
    }
}

/// Errors during dynamic LoRA model loading.
#[derive(Debug, thiserror::Error)]
pub enum DynamicLoraError {
    /// MLX exception.
    #[error("MLX error: {0}")]
    Mlx(#[from] pmetal_bridge::compat::Exception),
    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON parsing error.
    #[error("JSON parsing error: {0}")]
    Json(#[from] serde_json::Error),
    /// LoRA error.
    #[error("LoRA error: {0}")]
    Lora(#[from] LoraError),
    /// Weight format error.
    #[error("Weight format error: {0}")]
    WeightFormat(#[from] WeightFormatError),
    /// Architecture not implemented for LoRA training.
    #[error("Architecture not implemented for LoRA training: {0}")]
    NotImplemented(ModelArchitecture),
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_models::architectures::{
        llama::LlamaConfig, qwen3::Qwen3Config, qwen3_next::Qwen3NextConfig,
    };

    #[test]
    fn test_architecture_dispatch() {
        // Test that the dispatch logic is correct
        assert_eq!(
            ModelArchitecture::from_model_type("llama"),
            Some(ModelArchitecture::Llama)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("mistral"),
            Some(ModelArchitecture::Mistral)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("qwen3"),
            Some(ModelArchitecture::Qwen3)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("qwen2"),
            Some(ModelArchitecture::Qwen2)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("qwen3_next"),
            Some(ModelArchitecture::Qwen3Next)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("qwen3_moe"),
            Some(ModelArchitecture::Qwen3MoE)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("gemma4_text"),
            Some(ModelArchitecture::Gemma4)
        );
        assert_eq!(
            ModelArchitecture::from_model_type("gpt_oss"),
            Some(ModelArchitecture::GptOss)
        );
    }

    #[test]
    fn test_gradient_checkpointing_support_delegates_to_inner_model() {
        let lora_config = LoraConfig {
            r: 4,
            alpha: 8.0,
            dropout: 0.0,
            use_rslora: false,
            ..Default::default()
        };

        let qwen3 = DynamicLoraModel::Qwen3(
            Qwen3LoraForCausalLM::new(
                Qwen3Config {
                    vocab_size: 128,
                    hidden_size: 64,
                    intermediate_size: 128,
                    num_hidden_layers: 2,
                    num_attention_heads: 4,
                    num_key_value_heads: Some(2),
                    head_dim: 16,
                    max_position_embeddings: 128,
                    ..Default::default()
                },
                lora_config.clone(),
            )
            .unwrap(),
        );
        assert!(!qwen3.supports_gradient_checkpointing());

        let qwen3_next = DynamicLoraModel::Qwen3Next(
            Qwen3NextLoraForCausalLM::new(
                Qwen3NextConfig {
                    hidden_size: 32,
                    intermediate_size: 64,
                    num_hidden_layers: 4,
                    num_attention_heads: 2,
                    num_key_value_heads: Some(1),
                    head_dim: Some(16),
                    vocab_size: 100,
                    linear_num_value_heads: 2,
                    linear_num_key_heads: 1,
                    linear_key_head_dim: 16,
                    linear_value_head_dim: 16,
                    linear_conv_kernel_dim: 4,
                    full_attention_interval: 4,
                    num_experts: 0,
                    num_experts_per_tok: 0,
                    decoder_sparse_step: 1,
                    moe_intermediate_size: 16,
                    shared_expert_intermediate_size: 32,
                    mlp_only_layers: vec![],
                    norm_topk_prob: false,
                    tie_word_embeddings: true,
                    ..Default::default()
                },
                lora_config,
            )
            .unwrap(),
        );
        assert!(!qwen3_next.supports_gradient_checkpointing());
    }

    #[test]
    fn test_forward_noised_delegates_to_inner_model() {
        let lora_config = LoraConfig {
            r: 4,
            alpha: 8.0,
            dropout: 0.0,
            use_rslora: false,
            ..Default::default()
        };
        let mut model = DynamicLoraModel::Llama(
            LlamaLoraForCausalLM::new(
                LlamaConfig {
                    vocab_size: 128,
                    hidden_size: 32,
                    intermediate_size: 64,
                    num_hidden_layers: 2,
                    num_attention_heads: 4,
                    num_key_value_heads: Some(2),
                    head_dim: Some(8),
                    max_position_embeddings: 64,
                    ..Default::default()
                },
                lora_config,
            )
            .unwrap(),
        );

        let input_ids = Array::from_i32_slice_shaped(&[1, 2, 3, 4], &[1, 4]);
        let clean = model.forward(&input_ids, None).unwrap();
        let noised = TrainableModel::forward_noised(&mut model, &input_ids, None, 10.0).unwrap();
        let diff = clean.subtract(&noised);
        let sq_sum = diff.multiply(&diff).sum(None).item_f32();
        assert!(
            sq_sum > 0.0,
            "dynamic LoRA wrapper should delegate NEFTune instead of falling back to plain forward"
        );
    }
}
