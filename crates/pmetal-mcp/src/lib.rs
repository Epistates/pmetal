//! PMetal MCP Server — exposes the full PMetal ML toolkit via Model Context Protocol.

pub mod jobs;
pub mod util;

use std::sync::Arc;
use tokio::sync::RwLock;
use turbomcp::prelude::*;

use jobs::JobManager;
use util::{push_bool_flag, push_opt};

/// MCP server exposing all PMetal functionality.
#[derive(Clone)]
pub struct PmetalMcpServer {
    jobs: Arc<RwLock<JobManager>>,
}

impl PmetalMcpServer {
    fn new() -> Self {
        Self {
            jobs: Arc::new(RwLock::new(JobManager::new())),
        }
    }
}

/// Start the MCP server on stdio.
pub async fn run_stdio() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    PmetalMcpServer::new().run_stdio().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
#[turbomcp::server(name = "pmetal", version = "0.4.0", description = "PMetal ML toolkit for Apple Silicon — training, inference, quantization, and model management")]
impl PmetalMcpServer {
    // ── Device & System ───────────────────────────────────────────────────

    /// Get Apple Silicon device information including GPU architecture,
    /// ANE cores, memory bandwidth, NAX support, and unified memory capacity.
    #[tool]
    async fn device_info(&self) -> McpResult<String> {
        let info = util::build_device_info_json()?;
        serde_json::to_string_pretty(&info).map_err(|e| McpError::internal(e.to_string()))
    }

    // ── Hub & Model Management ────────────────────────────────────────────

    /// Search HuggingFace Hub for models. Returns model IDs, sizes,
    /// download counts, and whether each model fits in this device's memory
    /// for inference and training.
    #[tool]
    async fn search_models(
        &self,
        #[description("Search query (e.g. 'qwen3 0.6B', 'llama 8b instruct')")]
        query: String,
        #[description("Maximum number of results (default: 10)")]
        limit: Option<u64>,
    ) -> McpResult<String> {
        let limit = limit.unwrap_or(10) as usize;
        let device_spec = util::build_device_spec().ok();

        let results = pmetal_hub::search_models(&query, limit, None)
            .await
            .map_err(|e| McpError::internal(e.to_string()))?;

        let json_results: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                let fit_info =
                    if let (Some(dev), Some(params_b)) = (&device_spec, r.estimated_params_b) {
                        let quant =
                            pmetal_hub::fit::detect_quantization_from_id(&r.model_id);
                        let model_spec = pmetal_hub::ModelSpec {
                            params_b,
                            quantization: quant,
                            context_length: 4096,
                            num_kv_heads: None,
                            head_dim: None,
                            num_layers: None,
                            is_moe: r.tags.iter().any(|t| t == "moe"),
                            num_experts: None,
                            active_experts: None,
                            kv_cache_bits: None,
                        };
                        let fit = pmetal_hub::estimate_fit(&model_spec, dev);
                        serde_json::json!({
                            "level": fit.fit_level.label(),
                            "total_gb": fit.total_required_gb,
                            "weights_gb": fit.weights_gb,
                            "training_gb": fit.training_memory_gb,
                            "training_fit": fit.training_fit.label(),
                            "estimated_tps": fit.estimated_tps,
                        })
                    } else {
                        serde_json::Value::Null
                    };

                serde_json::json!({
                    "model_id": r.model_id,
                    "downloads": r.downloads,
                    "likes": r.likes,
                    "tags": r.tags,
                    "estimated_params_b": r.estimated_params_b,
                    "fit": fit_info,
                })
            })
            .collect();

        serde_json::to_string_pretty(&json_results)
            .map_err(|e| McpError::internal(e.to_string()))
    }

    /// Download a model from HuggingFace Hub to the local cache.
    /// Returns the local path once complete. May take several minutes for large models.
    #[tool]
    async fn download_model(
        &self,
        #[description("HuggingFace model ID (e.g. 'Qwen/Qwen3-0.6B')")]
        model_id: String,
        #[description("Git revision or branch")]
        revision: Option<String>,
    ) -> McpResult<String> {
        let path = pmetal_hub::download_model(
            &model_id,
            revision.as_deref(),
            None,
        )
        .await
        .map_err(|e| McpError::internal(e.to_string()))?;

        serde_json::to_string_pretty(&serde_json::json!({
            "model_id": model_id,
            "path": path.display().to_string(),
        }))
        .map_err(|e| McpError::internal(e.to_string()))
    }

    /// List models already downloaded to the local HuggingFace cache.
    #[tool]
    async fn list_local_models(&self) -> McpResult<String> {
        let cache_dir = util::hf_cache_dir();
        let mut models = Vec::new();

        if cache_dir.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&cache_dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with("models--") {
                        let model_id = name
                            .trim_start_matches("models--")
                            .replace("--", "/");
                        let model_dir = entry.path();

                        // Check for a valid snapshot
                        let snapshots_dir = model_dir.join("snapshots");
                        let has_snapshot = snapshots_dir.is_dir()
                            && std::fs::read_dir(&snapshots_dir)
                                .map(|mut d| d.next().is_some())
                                .unwrap_or(false);

                        if has_snapshot {
                            // Compute total size
                            let size = dir_size(&model_dir);
                            models.push(serde_json::json!({
                                "model_id": model_id,
                                "path": model_dir.display().to_string(),
                                "size_gb": size as f64 / (1024.0 * 1024.0 * 1024.0),
                            }));
                        }
                    }
                }
            }
        }

        serde_json::to_string_pretty(&models).map_err(|e| McpError::internal(e.to_string()))
    }

    /// Estimate memory requirements and performance for a model on this device.
    /// Shows inference memory, training memory, estimated tok/s, and fit level.
    #[tool]
    async fn model_fit(
        &self,
        #[description("HuggingFace model ID or local path")]
        model: String,
        #[description("Context length in tokens (default: 4096)")]
        context_length: Option<u64>,
    ) -> McpResult<String> {
        let args: Vec<&str> = vec!["search", &model, "--limit", "1", "--json"];
        let output = util::run_pmetal_blocking(&args).await?;

        let results: Vec<serde_json::Value> = serde_json::from_str(&output)
            .map_err(|e| McpError::internal(format!("parse error: {e}")))?;

        let result = results.first().ok_or_else(|| {
            McpError::invalid_params(format!("model not found: {model}"))
        })?;

        let device_spec = util::build_device_spec()?;
        let params_b = result
            .get("estimated_params_b")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        let quant = pmetal_hub::fit::detect_quantization_from_id(&model);
        let is_moe = result
            .get("tags")
            .and_then(|t| t.as_array())
            .map(|tags| tags.iter().any(|t| t.as_str() == Some("moe")))
            .unwrap_or(false);

        let model_spec = pmetal_hub::ModelSpec {
            params_b,
            quantization: quant,
            context_length: context_length.unwrap_or(4096) as u32,
            num_kv_heads: None,
            head_dim: None,
            num_layers: None,
            is_moe,
            num_experts: None,
            active_experts: None,
            kv_cache_bits: None,
        };

        let fit = pmetal_hub::estimate_fit(&model_spec, &device_spec);

        serde_json::to_string_pretty(&serde_json::json!({
            "model": model,
            "params_b": params_b,
            "inference": {
                "fit_level": fit.fit_level.label(),
                "total_required_gb": fit.total_required_gb,
                "weights_gb": fit.weights_gb,
                "kv_cache_gb": fit.kv_cache_gb,
                "overhead_gb": fit.overhead_gb,
                "available_gb": fit.available_gb,
                "utilization_pct": fit.utilization_pct,
                "estimated_tps": fit.estimated_tps,
            },
            "training": {
                "fit_level": fit.training_fit.label(),
                "memory_gb": fit.training_memory_gb,
            },
        }))
        .map_err(|e| McpError::internal(e.to_string()))
    }

    // ── Inference ─────────────────────────────────────────────────────────

    /// Generate text with a model. Blocks until generation completes.
    /// Best for short prompts (< 500 tokens output). For longer sessions,
    /// use start_serve to run an OpenAI-compatible server.
    #[tool]
    async fn generate(
        &self,
        #[description("Model ID or local path")]
        model: String,
        #[description("Input prompt text")]
        prompt: String,
        #[description("Maximum tokens to generate (default: 256)")]
        max_tokens: Option<u64>,
        #[description("Sampling temperature 0.0-2.0")]
        temperature: Option<f64>,
        #[description("Apply chat template (auto-detected from model)")]
        chat: Option<bool>,
        #[description("System message for chat mode")]
        system: Option<String>,
        #[description("LoRA adapter path")]
        lora: Option<String>,
        #[description("Top-k sampling (0 = disabled)")]
        top_k: Option<u64>,
        #[description("Top-p nucleus sampling (0.0-1.0)")]
        top_p: Option<f64>,
        #[description("Min-p dynamic sampling (0.0 = disabled)")]
        min_p: Option<f64>,
        #[description("Repetition penalty (1.0 = disabled, 1.0-1.2 typical)")]
        repetition_penalty: Option<f64>,
        #[description("Frequency penalty (0.0 = disabled, 0.0-2.0 typical)")]
        frequency_penalty: Option<f64>,
        #[description("Presence penalty (0.0 = disabled)")]
        presence_penalty: Option<f64>,
        #[description("Random seed for reproducible generation")]
        seed: Option<u64>,
        #[description("Disable thinking mode for models that support it")]
        no_thinking: Option<bool>,
        #[description("Show thinking content in output")]
        show_thinking: Option<bool>,
        #[description("Use FP8 quantization for weights (~2x memory reduction)")]
        fp8: Option<bool>,
        #[description("Packed expert weights dir for SSD-offloaded MoE")]
        experts_dir: Option<String>,
        #[description("Enable ANE (Apple Neural Engine) inference")]
        ane: Option<bool>,
        #[description("Maximum ANE kernel sequence length")]
        ane_max_seq_len: Option<u64>,
        #[description("Use experimental ANE real-time evaluation path")]
        ane_real_time: Option<bool>,
        #[description("KV cache quantization bits (8=q8, 4=q4, 0=fp16, default: 8)")]
        kv_quant: Option<u64>,
        #[description("Disable KV cache quantization (use fp16)")]
        no_kv_quant: Option<bool>,
    ) -> McpResult<String> {
        let max_tokens_str = max_tokens.unwrap_or(256).to_string();
        let mut args: Vec<&str> = vec![
            "infer", "--model", &model, "--prompt", &prompt,
            "--max-tokens", &max_tokens_str,
        ];

        let temp_str;
        if let Some(t) = temperature {
            temp_str = t.to_string();
            args.extend_from_slice(&["--temperature", &temp_str]);
        }
        if chat.unwrap_or(false) {
            args.push("--chat");
        }
        let system_ref;
        if let Some(ref sys) = system {
            system_ref = sys.as_str();
            args.extend_from_slice(&["--system", system_ref]);
        }
        let lora_ref;
        if let Some(ref l) = lora {
            lora_ref = l.as_str();
            args.extend_from_slice(&["--lora", lora_ref]);
        }
        let top_k_str;
        if let Some(k) = top_k {
            top_k_str = k.to_string();
            args.extend_from_slice(&["--top-k", &top_k_str]);
        }
        let top_p_str;
        if let Some(p) = top_p {
            top_p_str = p.to_string();
            args.extend_from_slice(&["--top-p", &top_p_str]);
        }
        let min_p_str;
        if let Some(p) = min_p {
            min_p_str = p.to_string();
            args.extend_from_slice(&["--min-p", &min_p_str]);
        }
        let rep_pen_str;
        if let Some(r) = repetition_penalty {
            rep_pen_str = r.to_string();
            args.extend_from_slice(&["--repetition-penalty", &rep_pen_str]);
        }
        let freq_pen_str;
        if let Some(f) = frequency_penalty {
            freq_pen_str = f.to_string();
            args.extend_from_slice(&["--frequency-penalty", &freq_pen_str]);
        }
        let pres_pen_str;
        if let Some(p) = presence_penalty {
            pres_pen_str = p.to_string();
            args.extend_from_slice(&["--presence-penalty", &pres_pen_str]);
        }
        let seed_str;
        if let Some(s) = seed {
            seed_str = s.to_string();
            args.extend_from_slice(&["--seed", &seed_str]);
        }
        if no_thinking.unwrap_or(false) {
            args.push("--no-thinking");
        }
        if show_thinking.unwrap_or(false) {
            args.push("--show-thinking");
        }
        if fp8.unwrap_or(false) {
            args.push("--fp8");
        }
        let experts_dir_ref;
        if let Some(ref e) = experts_dir {
            experts_dir_ref = e.as_str();
            args.extend_from_slice(&["--experts-dir", experts_dir_ref]);
        }
        if ane.unwrap_or(false) {
            args.push("--ane");
        }
        let ane_max_seq_len_str;
        if let Some(seq) = ane_max_seq_len {
            ane_max_seq_len_str = seq.to_string();
            args.extend_from_slice(&["--ane-max-seq-len", &ane_max_seq_len_str]);
        }
        if ane_real_time.unwrap_or(false) {
            args.push("--ane-real-time");
        }
        let kv_quant_str;
        if let Some(k) = kv_quant {
            kv_quant_str = k.to_string();
            args.extend_from_slice(&["--kv-quant", &kv_quant_str]);
        }
        if no_kv_quant.unwrap_or(false) {
            args.push("--no-kv-quant");
        }

        util::run_pmetal_blocking(&args).await
    }

    // ── Training (background jobs) ────────────────────────────────────────

    /// Start LoRA/QLoRA fine-tuning on a model. Returns a job ID for
    /// tracking progress. Use job_status/job_logs to monitor.
    #[tool]
    async fn train(
        &self,
        #[description("Model ID or local path")]
        model: String,
        #[description("Training dataset path (JSONL)")]
        dataset: String,
        #[description("Output directory (default: ./output)")]
        output: Option<String>,
        #[description("LoRA rank (default: 16)")]
        lora_r: Option<u64>,
        #[description("LoRA alpha scaling factor (default: 2x rank)")]
        lora_alpha: Option<f64>,
        #[description("Learning rate (default: 2e-4)")]
        learning_rate: Option<f64>,
        #[description("Batch size (default: 1)")]
        batch_size: Option<u64>,
        #[description("Number of epochs (default: 1)")]
        epochs: Option<u64>,
        #[description("Maximum sequence length (0 = auto-detect)")]
        max_seq_len: Option<u64>,
        #[description("Gradient accumulation steps (default: 4)")]
        gradient_accumulation_steps: Option<u64>,
        #[description("QLoRA quantization: none, nf4, fp4, int8")]
        quantization: Option<String>,
        #[description("LR schedule: constant, cosine, linear, wsd")]
        lr_schedule: Option<String>,
        #[description("Evaluation dataset path (JSONL)")]
        eval_dataset: Option<String>,
        #[description("Linear warmup steps (default: 0)")]
        warmup_steps: Option<u64>,
        #[description("AdamW weight decay (default: 0.01)")]
        weight_decay: Option<f64>,
        #[description("Random seed (default: 42)")]
        seed: Option<u64>,
        #[description("Max gradient norm for clipping (default: 1.0)")]
        max_grad_norm: Option<f64>,
        #[description("Loss scaling factor for ANE training (default: 1.0)")]
        loss_scale: Option<f64>,
        #[description("Separate learning rate for embedding layers")]
        embedding_lr: Option<f64>,
        #[description("Resume from checkpoint")]
        resume: Option<bool>,
        #[description("Memory-efficient loss for large-vocab models")]
        cut_cross_entropy: Option<bool>,
        #[description("Disable adaptive LR (let MCP/LLM control LR manually)")]
        no_adaptive_lr: Option<bool>,
        #[description("Disable Metal FlashAttention")]
        no_flash_attention: Option<bool>,
        #[description("Disable sequence packing (2-5x throughput)")]
        no_sequence_packing: Option<bool>,
        #[description("Disable JIT compilation")]
        no_jit_compilation: Option<bool>,
        #[description("Disable gradient checkpointing")]
        no_gradient_checkpointing: Option<bool>,
        #[description("Layers per checkpoint block (default: 4)")]
        gradient_checkpointing_layers: Option<u64>,
        #[description("Disable Metal fused optimizer")]
        no_metal_fused_optimizer: Option<bool>,
        #[description("Enable ANE (Apple Neural Engine) training (experimental, small models only)")]
        ane: Option<bool>,
        #[description("Custom JSONL text column name")]
        text_column: Option<String>,
        #[description("Prompt column for SFT label masking")]
        prompt_column: Option<String>,
        #[description("Response column for SFT label masking")]
        response_column: Option<String>,
        #[description("Path to training config file (YAML)")]
        config: Option<String>,
    ) -> McpResult<String> {
        let mut args = vec![
            "--model".to_string(), model,
            "--dataset".to_string(), dataset,
        ];
        push_opt(&mut args, "--output", &output);
        push_opt(&mut args, "--lora-r", &lora_r);
        push_opt(&mut args, "--lora-alpha", &lora_alpha);
        push_opt(&mut args, "--learning-rate", &learning_rate);
        push_opt(&mut args, "--batch-size", &batch_size);
        push_opt(&mut args, "--epochs", &epochs);
        push_opt(&mut args, "--max-seq-len", &max_seq_len);
        push_opt(&mut args, "--gradient-accumulation-steps", &gradient_accumulation_steps);
        push_opt(&mut args, "--quantization", &quantization);
        push_opt(&mut args, "--lr-schedule", &lr_schedule);
        push_opt(&mut args, "--eval-dataset", &eval_dataset);
        push_opt(&mut args, "--warmup-steps", &warmup_steps);
        push_opt(&mut args, "--weight-decay", &weight_decay);
        push_opt(&mut args, "--seed", &seed);
        push_opt(&mut args, "--max-grad-norm", &max_grad_norm);
        push_opt(&mut args, "--loss-scale", &loss_scale);
        push_opt(&mut args, "--embedding-lr", &embedding_lr);
        push_bool_flag(&mut args, "--resume", &resume);
        push_bool_flag(&mut args, "--cut-cross-entropy", &cut_cross_entropy);
        push_bool_flag(&mut args, "--no-adaptive-lr", &no_adaptive_lr);
        push_bool_flag(&mut args, "--no-flash-attention", &no_flash_attention);
        push_bool_flag(&mut args, "--no-sequence-packing", &no_sequence_packing);
        push_bool_flag(&mut args, "--no-jit-compilation", &no_jit_compilation);
        push_bool_flag(&mut args, "--no-gradient-checkpointing", &no_gradient_checkpointing);
        push_opt(&mut args, "--gradient-checkpointing-layers", &gradient_checkpointing_layers);
        push_bool_flag(&mut args, "--no-metal-fused-optimizer", &no_metal_fused_optimizer);
        push_bool_flag(&mut args, "--ane", &ane);
        push_opt(&mut args, "--text-column", &text_column);
        push_opt(&mut args, "--prompt-column", &prompt_column);
        push_opt(&mut args, "--response-column", &response_column);
        push_opt(&mut args, "--config", &config);

        let mut mgr = self.jobs.write().await;
        let id = mgr.spawn("train", args).await?;
        job_started_response(&id, "train")
    }

    /// Start knowledge distillation from a teacher to a student model.
    /// Returns a job ID for tracking.
    #[tool]
    async fn distill(
        &self,
        #[description("Teacher model ID or path")]
        teacher: String,
        #[description("Student model ID or path")]
        student: String,
        #[description("Training dataset path (JSONL)")]
        dataset: String,
        #[description("Output directory (default: ./output/distilled)")]
        output: Option<String>,
        #[description("Distillation method: online, offline, progressive")]
        method: Option<String>,
        #[description("Softmax temperature (default: 2.0)")]
        temperature: Option<f64>,
        #[description("Hard/soft target blend 0.0-1.0 (default: 0.5)")]
        alpha: Option<f64>,
        #[description("Learning rate (default: 2e-5)")]
        learning_rate: Option<f64>,
        #[description("Number of epochs (default: 1)")]
        epochs: Option<u64>,
        #[description("Maximum sequence length (default: 1024)")]
        max_seq_len: Option<u64>,
        #[description("Loss: kl_divergence, jensen_shannon, soft_cross_entropy, mse")]
        loss_type: Option<String>,
        #[description("Enable reasoning-aware distillation")]
        rationale: Option<bool>,
        #[description("Weight for reasoning tokens (default: 1.0)")]
        rationale_weight: Option<f64>,
        #[description("LoRA rank for student (default: 16)")]
        lora_r: Option<u64>,
        #[description("LoRA alpha (default: 32)")]
        lora_alpha: Option<f64>,
        #[description("Batch size (default: 1)")]
        batch_size: Option<u64>,
        #[description("Random seed (default: 42)")]
        seed: Option<u64>,
        #[description("Custom JSONL text column name")]
        text_column: Option<String>,
        #[description("Prompt column for SFT label masking")]
        prompt_column: Option<String>,
        #[description("Response column for SFT label masking")]
        response_column: Option<String>,
    ) -> McpResult<String> {
        let mut args = vec![
            "--teacher".to_string(), teacher,
            "--student".to_string(), student,
            "--dataset".to_string(), dataset,
        ];
        push_opt(&mut args, "--output", &output);
        push_opt(&mut args, "--method", &method);
        push_opt(&mut args, "--temperature", &temperature);
        push_opt(&mut args, "--alpha", &alpha);
        push_opt(&mut args, "--learning-rate", &learning_rate);
        push_opt(&mut args, "--epochs", &epochs);
        push_opt(&mut args, "--max-seq-len", &max_seq_len);
        push_opt(&mut args, "--loss-type", &loss_type);
        push_bool_flag(&mut args, "--rationale", &rationale);
        push_opt(&mut args, "--rationale-weight", &rationale_weight);
        push_opt(&mut args, "--lora-r", &lora_r);
        push_opt(&mut args, "--lora-alpha", &lora_alpha);
        push_opt(&mut args, "--batch-size", &batch_size);
        push_opt(&mut args, "--seed", &seed);
        push_opt(&mut args, "--text-column", &text_column);
        push_opt(&mut args, "--prompt-column", &prompt_column);
        push_opt(&mut args, "--response-column", &response_column);

        let mut mgr = self.jobs.write().await;
        let id = mgr.spawn("distill", args).await?;
        job_started_response(&id, "distill")
    }

    /// Start Group Relative Policy Optimization (GRPO) for reasoning
    /// model training. Returns a job ID for tracking.
    #[tool]
    async fn grpo(
        &self,
        #[description("Model ID or path")]
        model: String,
        #[description("Dataset path (JSONL with prompts)")]
        dataset: String,
        #[description("Output directory (default: ./output/grpo)")]
        output: Option<String>,
        #[description("Generations per prompt (default: 8)")]
        num_generations: Option<u64>,
        #[description("KL penalty coefficient (default: 0.001)")]
        beta: Option<f64>,
        #[description("Learning rate (default: 5e-6)")]
        learning_rate: Option<f64>,
        #[description("Number of epochs (default: 1)")]
        epochs: Option<u64>,
        #[description("LoRA rank (default: 16)")]
        lora_r: Option<u64>,
        #[description("Enable reasoning-aware rewards")]
        reasoning_rewards: Option<bool>,
        #[description("LoRA alpha (default: 32)")]
        lora_alpha: Option<f64>,
        #[description("Max sequence length (default: 512)")]
        max_seq_len: Option<u64>,
        #[description("Max completion length per generation (default: 512)")]
        max_completion_length: Option<u64>,
        #[description("Random seed (default: 42)")]
        seed: Option<u64>,
        #[description("Enable DAPO (Distribution-Aware Policy Optimization)")]
        dapo: Option<bool>,
        #[description("Disable Metal FlashAttention")]
        no_flash_attention: Option<bool>,
        #[description("Enable VLM mode for image inputs")]
        vlm: Option<bool>,
        #[description("Max image size pixels (default: 336)")]
        max_image_size: Option<u64>,
        #[description("ML reward model path or HF ID")]
        reward_model: Option<String>,
        #[description("Reward model max input tokens (default: 2048)")]
        reward_model_max_length: Option<u64>,
        #[description("Reward model weight in combined score (default: 1.0)")]
        reward_model_weight: Option<f64>,
        #[description("Chat template for reward model")]
        reward_model_template: Option<String>,
        #[description("Enable speculative decoding for rollouts")]
        speculative: Option<bool>,
        #[description("Draft tokens per speculative step (default: 3)")]
        speculative_draft_tokens: Option<u64>,
        #[description("Enable pipelined async reward scoring")]
        async_rewards: Option<bool>,
        #[description("Custom JSONL text column name")]
        text_column: Option<String>,
        #[description("Prompt column for SFT label masking")]
        prompt_column: Option<String>,
        #[description("Response column for SFT label masking")]
        response_column: Option<String>,
    ) -> McpResult<String> {
        let mut args = vec![
            "--model".to_string(), model,
            "--dataset".to_string(), dataset,
        ];
        push_opt(&mut args, "--output", &output);
        push_opt(&mut args, "--num-generations", &num_generations);
        push_opt(&mut args, "--beta", &beta);
        push_opt(&mut args, "--learning-rate", &learning_rate);
        push_opt(&mut args, "--epochs", &epochs);
        push_opt(&mut args, "--lora-r", &lora_r);
        push_bool_flag(&mut args, "--reasoning-rewards", &reasoning_rewards);
        push_opt(&mut args, "--lora-alpha", &lora_alpha);
        push_opt(&mut args, "--max-seq-len", &max_seq_len);
        push_opt(&mut args, "--max-completion-length", &max_completion_length);
        push_opt(&mut args, "--seed", &seed);
        push_bool_flag(&mut args, "--dapo", &dapo);
        push_bool_flag(&mut args, "--no-flash-attention", &no_flash_attention);
        push_bool_flag(&mut args, "--vlm", &vlm);
        push_opt(&mut args, "--max-image-size", &max_image_size);
        push_opt(&mut args, "--reward-model", &reward_model);
        push_opt(&mut args, "--reward-model-max-length", &reward_model_max_length);
        push_opt(&mut args, "--reward-model-weight", &reward_model_weight);
        push_opt(&mut args, "--reward-model-template", &reward_model_template);
        push_bool_flag(&mut args, "--speculative", &speculative);
        push_opt(&mut args, "--speculative-draft-tokens", &speculative_draft_tokens);
        push_bool_flag(&mut args, "--async-rewards", &async_rewards);
        push_opt(&mut args, "--text-column", &text_column);
        push_opt(&mut args, "--prompt-column", &prompt_column);
        push_opt(&mut args, "--response-column", &response_column);

        let mut mgr = self.jobs.write().await;
        let id = mgr.spawn("grpo", args).await?;
        job_started_response(&id, "grpo")
    }

    /// Start Reinforcement Learning from Knowledge Distillation (RLKD),
    /// combining online GRPO with offline distillation. Returns a job ID.
    #[tool]
    async fn rlkd(
        &self,
        #[description("Policy (student) model ID or path")]
        model: String,
        #[description("Teacher model ID or path (frozen)")]
        teacher_model: String,
        #[description("Dataset path (JSONL with prompts)")]
        dataset: String,
        #[description("Output directory (default: ./output/rlkd)")]
        output: Option<String>,
        #[description("Distillation blend 0.0-1.0 (default: 0.3)")]
        distill_alpha: Option<f64>,
        #[description("Final alpha when annealing (default: 0.05)")]
        final_alpha: Option<f64>,
        #[description("Linearly anneal alpha over training")]
        anneal_alpha: Option<bool>,
        #[description("Distillation temperature (default: 2.0)")]
        distill_temperature: Option<f64>,
        #[description("Generations per prompt (default: 8)")]
        num_generations: Option<u64>,
        #[description("KL penalty coefficient (default: 0.001)")]
        beta: Option<f64>,
        #[description("Learning rate (default: 5e-6)")]
        learning_rate: Option<f64>,
        #[description("Number of epochs (default: 1)")]
        epochs: Option<u64>,
        #[description("LoRA rank (default: 16)")]
        lora_r: Option<u64>,
        #[description("LoRA alpha (default: 32)")]
        lora_alpha: Option<f64>,
        #[description("Max sequence length (default: 512)")]
        max_seq_len: Option<u64>,
        #[description("Max completion length (default: 512)")]
        max_completion_length: Option<u64>,
        #[description("Random seed (default: 42)")]
        seed: Option<u64>,
        #[description("Enable reasoning-aware rewards")]
        reasoning_rewards: Option<bool>,
    ) -> McpResult<String> {
        let mut args = vec![
            "--model".to_string(), model,
            "--teacher-model".to_string(), teacher_model,
            "--dataset".to_string(), dataset,
        ];
        push_opt(&mut args, "--output", &output);
        push_opt(&mut args, "--distill-alpha", &distill_alpha);
        push_opt(&mut args, "--final-alpha", &final_alpha);
        push_bool_flag(&mut args, "--anneal-alpha", &anneal_alpha);
        push_opt(&mut args, "--distill-temperature", &distill_temperature);
        push_opt(&mut args, "--num-generations", &num_generations);
        push_opt(&mut args, "--beta", &beta);
        push_opt(&mut args, "--learning-rate", &learning_rate);
        push_opt(&mut args, "--epochs", &epochs);
        push_opt(&mut args, "--lora-r", &lora_r);
        push_opt(&mut args, "--lora-alpha", &lora_alpha);
        push_opt(&mut args, "--max-seq-len", &max_seq_len);
        push_opt(&mut args, "--max-completion-length", &max_completion_length);
        push_opt(&mut args, "--seed", &seed);
        push_bool_flag(&mut args, "--reasoning-rewards", &reasoning_rewards);

        let mut mgr = self.jobs.write().await;
        let id = mgr.spawn("rlkd", args).await?;
        job_started_response(&id, "rlkd")
    }

    /// Train an embedding model for semantic search or similarity tasks.
    /// Supports contrastive, triplet, and other embedding loss functions.
    /// Returns a job ID for tracking.
    #[tool]
    async fn embed_train(
        &self,
        #[description("BERT/encoder model path")]
        model: String,
        #[description("Training dataset (JSONL pairs or triplets)")]
        dataset: String,
        #[description("Output directory (default: ./output-embed)")]
        output: Option<String>,
        #[description("Loss: info_nce, triplet, cosent, cosine_similarity")]
        loss: Option<String>,
        #[description("Pooling: mean, cls, max, last_token, weighted_mean")]
        pooling: Option<String>,
        #[description("Temperature for InfoNCE/CoSENT (default: 0.05)")]
        temperature: Option<f64>,
        #[description("Margin for triplet loss (default: 0.3)")]
        margin: Option<f64>,
        #[description("Learning rate (default: 2e-5)")]
        learning_rate: Option<f64>,
        #[description("Batch size (default: 32)")]
        batch_size: Option<u64>,
        #[description("Number of epochs (default: 3)")]
        epochs: Option<u64>,
        #[description("Max input sequence length (default: 512)")]
        max_seq_len: Option<u64>,
        #[description("AdamW weight decay (default: 0.01)")]
        weight_decay: Option<f64>,
        #[description("Disable L2 normalization of embeddings")]
        no_normalize: Option<bool>,
        #[description("Random seed (default: 42)")]
        seed: Option<u64>,
    ) -> McpResult<String> {
        let mut args = vec![
            "--model".to_string(), model,
            "--dataset".to_string(), dataset,
        ];
        push_opt(&mut args, "--output", &output);
        push_opt(&mut args, "--loss", &loss);
        push_opt(&mut args, "--pooling", &pooling);
        push_opt(&mut args, "--temperature", &temperature);
        push_opt(&mut args, "--margin", &margin);
        push_opt(&mut args, "--learning-rate", &learning_rate);
        push_opt(&mut args, "--batch-size", &batch_size);
        push_opt(&mut args, "--epochs", &epochs);
        push_opt(&mut args, "--max-seq-len", &max_seq_len);
        push_opt(&mut args, "--weight-decay", &weight_decay);
        push_bool_flag(&mut args, "--no-normalize", &no_normalize);
        push_opt(&mut args, "--seed", &seed);

        let mut mgr = self.jobs.write().await;
        let id = mgr.spawn("embed-train", args).await?;
        job_started_response(&id, "embed-train")
    }

    // ── Job Management ────────────────────────────────────────────────────

    /// List all background jobs (training, distillation, GRPO, etc.)
    /// with their current status.
    #[tool]
    async fn list_jobs(&self) -> McpResult<String> {
        let mgr = self.jobs.read().await;
        let summaries = mgr.list_summaries().await;
        serde_json::to_string_pretty(&summaries)
            .map_err(|e| McpError::internal(e.to_string()))
    }

    /// Get detailed status of a background job including current
    /// training metrics (loss, step, learning rate, tok/s).
    #[tool]
    async fn job_status(
        &self,
        #[description("Job ID returned by train/distill/grpo/etc.")]
        job_id: String,
    ) -> McpResult<String> {
        let mgr = self.jobs.read().await;
        let summary = mgr.get_summary(&job_id).await?;
        serde_json::to_string_pretty(&summary).map_err(|e| McpError::internal(e.to_string()))
    }

    /// Get recent stdout/stderr output from a background job.
    #[tool]
    async fn job_logs(
        &self,
        #[description("Job ID returned by train/distill/grpo/etc.")]
        job_id: String,
        #[description("Number of recent lines to return (default: 50)")]
        tail: Option<u64>,
    ) -> McpResult<String> {
        let mgr = self.jobs.read().await;
        let (lines, total) = mgr.get_logs(&job_id, tail.unwrap_or(50) as usize).await?;
        serde_json::to_string_pretty(&serde_json::json!({
            "job_id": job_id,
            "total_lines": total,
            "lines": lines,
        }))
        .map_err(|e| McpError::internal(e.to_string()))
    }

    /// Stop a running background job by sending SIGTERM.
    #[tool]
    async fn stop_job(
        &self,
        #[description("Job ID returned by train/distill/grpo/etc.")]
        job_id: String,
    ) -> McpResult<String> {
        let mgr = self.jobs.read().await;
        mgr.stop(&job_id).await?;
        Ok(format!("Job {job_id} stopped"))
    }

    // ── Dataset Operations ────────────────────────────────────────────────

    /// Analyze a JSONL dataset: sample count, format detection,
    /// character/token length statistics.
    #[tool]
    async fn dataset_analyze(
        &self,
        #[description("Path to dataset file (JSONL)")]
        path: String,
        #[description("Model ID for tokenization stats")]
        model: Option<String>,
        #[description("Show detailed per-sample statistics")]
        detailed: Option<bool>,
    ) -> McpResult<String> {
        let mut args = vec!["dataset", "analyze", "--path", &path];
        let model_ref;
        if let Some(ref m) = model {
            model_ref = m.as_str();
            args.extend_from_slice(&["--model", model_ref]);
        }
        if detailed.unwrap_or(false) {
            args.push("--detailed");
        }
        util::run_pmetal_blocking(&args).await
    }

    /// Preview first N samples from a dataset file or HuggingFace dataset ID.
    #[tool]
    async fn dataset_preview(
        &self,
        #[description("HuggingFace dataset ID (e.g. 'tatsu-lab/alpaca')")]
        dataset_id: String,
        #[description("Number of samples to preview (default: 5)")]
        num: Option<u64>,
        #[description("Dataset split (default: train)")]
        split: Option<String>,
    ) -> McpResult<String> {
        let num_str = num.unwrap_or(5).to_string();
        let mut args = vec!["dataset", "preview", "--path", &dataset_id, "--num", &num_str];
        let split_ref;
        if let Some(ref s) = split {
            split_ref = s.as_str();
            args.extend_from_slice(&["--split", split_ref]);
        }
        util::run_pmetal_blocking(&args).await
    }

    /// Validate a JSONL dataset for training: checks format,
    /// encoding, sequence lengths, and reports issues.
    #[tool]
    async fn dataset_validate(
        &self,
        #[description("Path to dataset file (JSONL)")]
        path: String,
        #[description("Model ID for tokenization validation")]
        model: Option<String>,
        #[description("Max sequence length to check (default: 2048)")]
        max_seq_len: Option<u64>,
    ) -> McpResult<String> {
        let mut args = vec!["dataset", "validate", "--path", &path];
        let model_ref;
        if let Some(ref m) = model {
            model_ref = m.as_str();
            args.extend_from_slice(&["--model", model_ref]);
        }
        let seq_str;
        if let Some(s) = max_seq_len {
            seq_str = s.to_string();
            args.extend_from_slice(&["--max-seq-len", &seq_str]);
        }
        util::run_pmetal_blocking(&args).await
    }

    /// Download a dataset from HuggingFace Hub and convert to JSONL.
    #[tool]
    async fn dataset_download(
        &self,
        #[description("HuggingFace dataset ID")]
        dataset_id: String,
        #[description("Output path for JSONL file")]
        output: Option<String>,
        #[description("Dataset split (default: train)")]
        split: Option<String>,
        #[description("Specific revision/branch")]
        revision: Option<String>,
    ) -> McpResult<String> {
        let mut args = vec!["dataset", "download", "--dataset", &dataset_id];
        let output_ref;
        if let Some(ref o) = output {
            output_ref = o.as_str();
            args.extend_from_slice(&["--output", output_ref]);
        }
        let split_ref;
        if let Some(ref s) = split {
            split_ref = s.as_str();
            args.extend_from_slice(&["--split", split_ref]);
        }
        let revision_ref;
        if let Some(ref r) = revision {
            revision_ref = r.as_str();
            args.extend_from_slice(&["--revision", revision_ref]);
        }
        util::run_pmetal_blocking(&args).await
    }

    /// Convert a dataset file (Parquet, JSON, CSV) to JSONL.
    #[tool]
    async fn dataset_convert(
        &self,
        #[description("Input file path (Parquet, JSON, CSV)")]
        input: String,
        #[description("Output JSONL file path")]
        output: String,
        #[description("Input format: parquet, json, jsonl, csv (auto-detect)")]
        format: Option<String>,
        #[description("Column mapping (e.g. 'text=content,prompt=instruction')")]
        columns: Option<String>,
        #[description("Shuffle output data")]
        shuffle: Option<bool>,
        #[description("Random seed for shuffling (default: 42)")]
        seed: Option<u64>,
    ) -> McpResult<String> {
        let mut args = vec!["dataset", "convert", "--input", &input, "--output", &output];
        let fmt_ref;
        if let Some(ref f) = format {
            fmt_ref = f.as_str();
            args.extend_from_slice(&["--format", fmt_ref]);
        }
        let columns_ref;
        if let Some(ref c) = columns {
            columns_ref = c.as_str();
            args.extend_from_slice(&["--columns", columns_ref]);
        }
        if shuffle.unwrap_or(false) {
            args.push("--shuffle");
        }
        let seed_str;
        if let Some(s) = seed {
            seed_str = s.to_string();
            args.extend_from_slice(&["--seed", &seed_str]);
        }
        util::run_pmetal_blocking(&args).await
    }

    /// Filter a JSONL dataset by various criteria including token length,
    /// deduplication, pattern matching, and quality filters.
    #[tool]
    async fn dataset_filter(
        &self,
        #[description("Input dataset path (JSONL)")]
        input: String,
        #[description("Output dataset path (JSONL)")]
        output: String,
        #[description("Minimum token count")]
        min_tokens: Option<u64>,
        #[description("Maximum token count")]
        max_tokens: Option<u64>,
        #[description("Remove exact-match duplicates")]
        dedup: Option<bool>,
        #[description("Regex pattern (keeps matching samples)")]
        pattern: Option<String>,
        #[description("Model ID for token-based filtering")]
        model: Option<String>,
        #[description("Invert pattern matching")]
        invert: Option<bool>,
        #[description("Require all conversation turns")]
        complete_only: Option<bool>,
    ) -> McpResult<String> {
        let mut args = vec!["dataset", "filter", "--input", &input, "--output", &output];
        let min_str;
        if let Some(min) = min_tokens {
            min_str = min.to_string();
            args.extend_from_slice(&["--min-tokens", &min_str]);
        }
        let max_str;
        if let Some(max) = max_tokens {
            max_str = max.to_string();
            args.extend_from_slice(&["--max-tokens", &max_str]);
        }
        if dedup.unwrap_or(false) {
            args.push("--dedup");
        }
        let pat_ref;
        if let Some(ref p) = pattern {
            pat_ref = p.as_str();
            args.extend_from_slice(&["--pattern", pat_ref]);
        }
        let model_ref;
        if let Some(ref m) = model {
            model_ref = m.as_str();
            args.extend_from_slice(&["--model", model_ref]);
        }
        if invert.unwrap_or(false) {
            args.push("--invert");
        }
        if complete_only.unwrap_or(false) {
            args.push("--complete-only");
        }
        util::run_pmetal_blocking(&args).await
    }

    /// Split a JSONL dataset into train/validation/test sets.
    #[tool]
    async fn dataset_split(
        &self,
        #[description("Input dataset path (JSONL)")]
        input: String,
        #[description("Output directory for split files")]
        output_dir: String,
        #[description("Validation ratio 0.0-1.0 (default: 0.1)")]
        val_ratio: Option<f64>,
        #[description("Test ratio 0.0-1.0 (default: 0.0)")]
        test_ratio: Option<f64>,
        #[description("Random seed (default: 42)")]
        seed: Option<u64>,
        #[description("Stratify by field (e.g. 'category')")]
        stratify: Option<String>,
    ) -> McpResult<String> {
        let mut args = vec!["dataset", "split", "--input", &input, "--output", &output_dir];
        let val_str;
        if let Some(v) = val_ratio {
            val_str = v.to_string();
            args.extend_from_slice(&["--val-ratio", &val_str]);
        }
        let test_str;
        if let Some(t) = test_ratio {
            test_str = t.to_string();
            args.extend_from_slice(&["--test-ratio", &test_str]);
        }
        let seed_str;
        if let Some(s) = seed {
            seed_str = s.to_string();
            args.extend_from_slice(&["--seed", &seed_str]);
        }
        let stratify_ref;
        if let Some(ref s) = stratify {
            stratify_ref = s.as_str();
            args.extend_from_slice(&["--stratify", stratify_ref]);
        }
        util::run_pmetal_blocking(&args).await
    }

    /// Merge multiple JSONL dataset files into one, with optional
    /// shuffling, interleaving, and per-source weighting.
    #[tool]
    async fn dataset_merge(
        &self,
        #[description("Comma-separated input dataset paths (JSONL)")]
        inputs: String,
        #[description("Output dataset path (JSONL)")]
        output: String,
        #[description("Shuffle after merging")]
        shuffle: Option<bool>,
        #[description("Random seed (default: 42)")]
        seed: Option<u64>,
        #[description("Interleave samples from each dataset")]
        interleave: Option<bool>,
        #[description("Weights per dataset (comma-separated, e.g. '1.0,2.0')")]
        weights: Option<String>,
    ) -> McpResult<String> {
        let mut args = vec!["dataset", "merge", "--inputs", &inputs, "--output", &output];
        if shuffle.unwrap_or(false) {
            args.push("--shuffle");
        }
        let seed_str;
        if let Some(s) = seed {
            seed_str = s.to_string();
            args.extend_from_slice(&["--seed", &seed_str]);
        }
        if interleave.unwrap_or(false) {
            args.push("--interleave");
        }
        let weights_ref;
        if let Some(ref w) = weights {
            weights_ref = w.as_str();
            args.extend_from_slice(&["--weights", weights_ref]);
        }
        util::run_pmetal_blocking(&args).await
    }

    /// Sample a random subset of rows from a JSONL dataset.
    #[tool]
    async fn dataset_sample(
        &self,
        #[description("Input dataset path (JSONL)")]
        input: String,
        #[description("Output dataset path (JSONL)")]
        output: String,
        #[description("Number of samples to take")]
        num: u64,
        #[description("Random seed (default: 42)")]
        seed: Option<u64>,
    ) -> McpResult<String> {
        let num_str = num.to_string();
        let mut args = vec!["dataset", "sample", "--input", &input, "--output", &output, "--num", &num_str];
        let seed_str;
        if let Some(s) = seed {
            seed_str = s.to_string();
            args.extend_from_slice(&["--seed", &seed_str]);
        }
        util::run_pmetal_blocking(&args).await
    }

    /// Apply a chat template to a raw dataset, formatting messages
    /// into the prompt format expected by a model.
    #[tool]
    async fn dataset_template(
        &self,
        #[description("Input dataset path (JSONL with conversations)")]
        input: String,
        #[description("Output dataset path (JSONL)")]
        output: String,
        #[description("Chat template: chatml, llama3, llama2, mistral, qwen, phi, gemma, raw, auto")]
        template: Option<String>,
        #[description("Custom system message")]
        system: Option<String>,
        #[description("Model ID for tokenizer-based template")]
        model: Option<String>,
        #[description("Add generation prompt marker at end")]
        add_generation_prompt: Option<bool>,
        #[description("Mask prompt tokens in labels for SFT")]
        mask_prompt: Option<bool>,
    ) -> McpResult<String> {
        let mut args = vec!["dataset", "template", "--input", &input, "--output", &output];
        let template_ref;
        if let Some(ref t) = template {
            template_ref = t.as_str();
            args.extend_from_slice(&["--template", template_ref]);
        }
        let system_ref;
        if let Some(ref s) = system {
            system_ref = s.as_str();
            args.extend_from_slice(&["--system", system_ref]);
        }
        let model_ref;
        if let Some(ref m) = model {
            model_ref = m.as_str();
            args.extend_from_slice(&["--model", model_ref]);
        }
        if add_generation_prompt.unwrap_or(false) {
            args.push("--add-generation-prompt");
        }
        if mask_prompt.unwrap_or(false) {
            args.push("--mask-prompt");
        }
        util::run_pmetal_blocking(&args).await
    }

    /// Download, template, filter, split, and validate a dataset in one step.
    /// Produces a ready-to-train output directory with train/val/test splits.
    #[tool]
    async fn dataset_prepare(
        &self,
        #[description("HuggingFace dataset ID or local path")]
        dataset: String,
        #[description("Output directory")]
        output_dir: String,
        #[description("Model ID for tokenization")]
        model: String,
        #[description("Chat template (default: chatml)")]
        template: Option<String>,
        #[description("Max sequence length filter (default: 2048)")]
        max_seq_len: Option<u64>,
        #[description("Validation split ratio (default: 0.05)")]
        val_ratio: Option<f64>,
        #[description("Random seed (default: 42)")]
        seed: Option<u64>,
        #[description("Skip deduplication")]
        no_dedup: Option<bool>,
        #[description("Column mapping (e.g. 'instruction=problem,output=solution')")]
        columns: Option<String>,
    ) -> McpResult<String> {
        let mut args = vec![
            "dataset", "prepare",
            "--dataset", &dataset,
            "--output", &output_dir,
            "--model", &model,
        ];
        let template_ref;
        if let Some(ref t) = template {
            template_ref = t.as_str();
            args.extend_from_slice(&["--template", template_ref]);
        }
        let seq_str;
        if let Some(s) = max_seq_len {
            seq_str = s.to_string();
            args.extend_from_slice(&["--max-seq-len", &seq_str]);
        }
        let val_str;
        if let Some(v) = val_ratio {
            val_str = v.to_string();
            args.extend_from_slice(&["--val-ratio", &val_str]);
        }
        let seed_str;
        if let Some(s) = seed {
            seed_str = s.to_string();
            args.extend_from_slice(&["--seed", &seed_str]);
        }
        if no_dedup.unwrap_or(false) {
            args.push("--no-dedup");
        }
        let columns_ref;
        if let Some(ref c) = columns {
            columns_ref = c.as_str();
            args.extend_from_slice(&["--columns", columns_ref]);
        }
        util::run_pmetal_blocking(&args).await
    }

    // ── Quantization & Conversion ─────────────────────────────────────────

    /// Quantize a model to GGUF format. Supports Q2_K through Q8_0
    /// and dynamic quantization with importance matrices.
    #[tool]
    async fn quantize(
        &self,
        #[description("Source model path or HuggingFace ID")]
        model: String,
        #[description("Output GGUF file path")]
        output: String,
        #[description("Method: dynamic, q8_0, q6_k, q5_k_m, q4_k_m, q3_k_m, q2_k")]
        method: Option<String>,
        #[description("Path to importance matrix file")]
        imatrix: Option<String>,
        #[description("LoRA adapter to fuse before quantizing")]
        lora: Option<String>,
        #[description("Use KL-divergence calibration per tensor")]
        kl_calibrate: Option<bool>,
        #[description("Target average bits per weight for KL calibration")]
        target_bpw: Option<f64>,
        #[description("KL quality-loss threshold (default: 0.01)")]
        kl_threshold: Option<f64>,
    ) -> McpResult<String> {
        let mut args = vec![
            "--model".to_string(), model,
            "--output".to_string(), output,
        ];
        push_opt(&mut args, "--method", &method);
        push_opt(&mut args, "--imatrix", &imatrix);
        push_opt(&mut args, "--lora", &lora);
        push_bool_flag(&mut args, "--kl-calibrate", &kl_calibrate);
        push_opt(&mut args, "--target-bpw", &target_bpw);
        push_opt(&mut args, "--kl-threshold", &kl_threshold);

        let mut mgr = self.jobs.write().await;
        let id = mgr.spawn("quantize", args).await?;
        job_started_response(&id, "quantize")
    }

    /// Fuse LoRA adapter weights into a base model, producing a
    /// standalone merged model.
    #[tool]
    async fn fuse_lora(
        &self,
        #[description("Base model path or HuggingFace ID")]
        model: String,
        #[description("LoRA adapter path")]
        lora: String,
        #[description("Output directory for fused model")]
        output: String,
        #[description("Use f64-accurate merge path")]
        accurate: Option<bool>,
        #[description("LoRA scaling alpha (default: auto-detect)")]
        alpha: Option<f64>,
        #[description("LoRA rank (default: auto-detect)")]
        rank: Option<u64>,
        #[description("Use tiled low-memory mode with --accurate")]
        low_memory: Option<bool>,
    ) -> McpResult<String> {
        let mut args = vec![
            "--model".to_string(), model,
            "--lora".to_string(), lora,
            "--output".to_string(), output,
        ];
        push_bool_flag(&mut args, "--accurate", &accurate);
        push_opt(&mut args, "--alpha", &alpha);
        push_opt(&mut args, "--rank", &rank);
        push_bool_flag(&mut args, "--low-memory", &low_memory);

        let mut mgr = self.jobs.write().await;
        let id = mgr.spawn("fuse", args).await?;
        job_started_response(&id, "fuse")
    }

    /// Merge two models using SLERP, TIES, DARE, linear, or other merge methods.
    #[tool]
    async fn merge_models(
        &self,
        #[description("First model path or HuggingFace ID")]
        model_a: String,
        #[description("Second model path or HuggingFace ID")]
        model_b: String,
        #[description("Output directory for merged model")]
        output: String,
        #[description("Method: slerp, linear, ties, dare_ties, dare_linear, task_arithmetic")]
        method: Option<String>,
        #[description("SLERP interpolation (0.0=model_a, 1.0=model_b)")]
        t: Option<f64>,
        #[description("Base model for task-vector methods (TIES/DARE)")]
        base: Option<String>,
        #[description("Weight for model_a in linear/ties (default: 0.5)")]
        weight_a: Option<f64>,
        #[description("Weight for model_b in linear/ties (default: 0.5)")]
        weight_b: Option<f64>,
        #[description("Sparsification density for TIES/DARE (default: 0.5)")]
        density: Option<f64>,
        #[description("Output dtype: float32, float16, bfloat16")]
        dtype: Option<String>,
    ) -> McpResult<String> {
        let mut args = vec![
            "--model-a".to_string(), model_a,
            "--model-b".to_string(), model_b,
            "--output".to_string(), output,
        ];
        push_opt(&mut args, "--method", &method);
        push_opt(&mut args, "-t", &t);
        push_opt(&mut args, "--base", &base);
        push_opt(&mut args, "--weight-a", &weight_a);
        push_opt(&mut args, "--weight-b", &weight_b);
        push_opt(&mut args, "--density", &density);
        push_opt(&mut args, "--dtype", &dtype);

        let mut mgr = self.jobs.write().await;
        let id = mgr.spawn("merge", args).await?;
        job_started_response(&id, "merge")
    }

    /// Pack expert weights for SSD-offloaded MoE inference. Enables
    /// running models larger than available memory.
    #[tool]
    async fn pack_experts(
        &self,
        #[description("Model directory")]
        model: String,
        #[description("Output directory for packed experts")]
        output: Option<String>,
        #[description("Quantization bits: 4 or 2")]
        bits: Option<u64>,
    ) -> McpResult<String> {
        let mut args = vec!["--model".to_string(), model];
        push_opt(&mut args, "--output", &output);
        push_opt(&mut args, "--bits", &bits);

        let mut mgr = self.jobs.write().await;
        let id = mgr.spawn("pack-experts", args).await?;
        job_started_response(&id, "pack-experts")
    }

    // ── Evaluation & Benchmarking ─────────────────────────────────────────

    /// Evaluate model perplexity on a dataset. Returns per-sample
    /// and aggregate perplexity metrics.
    #[tool]
    async fn eval_perplexity(
        &self,
        #[description("Model ID or path")]
        model: String,
        #[description("Dataset path (JSONL)")]
        dataset: String,
        #[description("LoRA adapter path")]
        lora: Option<String>,
        #[description("Max sequence length (default: 1024)")]
        max_seq_len: Option<u64>,
        #[description("Number of samples (0 = all)")]
        num_samples: Option<u64>,
    ) -> McpResult<String> {
        let mut args = vec!["eval", "--model", &model, "--dataset", &dataset];
        let lora_ref;
        if let Some(ref l) = lora {
            lora_ref = l.as_str();
            args.extend_from_slice(&["--lora", lora_ref]);
        }
        let seq_str;
        if let Some(s) = max_seq_len {
            seq_str = s.to_string();
            args.extend_from_slice(&["--max-seq-len", &seq_str]);
        }
        let num_str;
        if let Some(n) = num_samples {
            num_str = n.to_string();
            args.extend_from_slice(&["--num-samples", &num_str]);
        }
        util::run_pmetal_blocking(&args).await
    }

    /// Benchmark model inference performance: prefill latency,
    /// decode tok/s, memory usage, and hardware utilization.
    #[tool]
    async fn benchmark(
        &self,
        #[description("Model ID or path")]
        model: String,
        #[description("Prompt text for benchmark")]
        prompt: Option<String>,
        #[description("Tokens to generate (default: 100)")]
        num_tokens: Option<u64>,
        #[description("Decode iterations (default: 5)")]
        benchmark_iters: Option<u64>,
    ) -> McpResult<String> {
        let mut args = vec!["infer", "--model", &model, "--benchmark"];
        let prompt_ref;
        if let Some(ref p) = prompt {
            prompt_ref = p.as_str();
            args.extend_from_slice(&["--prompt", prompt_ref]);
        }
        let tokens_str;
        if let Some(n) = num_tokens {
            tokens_str = n.to_string();
            args.extend_from_slice(&["--max-tokens", &tokens_str]);
        }
        let iters_str;
        if let Some(i) = benchmark_iters {
            iters_str = i.to_string();
            args.extend_from_slice(&["--benchmark-iters", &iters_str]);
        }
        util::run_pmetal_blocking(&args).await
    }

    /// Benchmark training throughput (tok/s, memory, MFU) for a model.
    /// Maps to `pmetal bench`.
    #[tool]
    async fn bench_train(
        &self,
        #[description("Model to benchmark (default: Llama-3.2-1B)")]
        model: Option<String>,
        #[description("Batch size (default: 1)")]
        batch_size: Option<u64>,
        #[description("Sequence length (default: 512)")]
        seq_len: Option<u64>,
    ) -> McpResult<String> {
        let mut args = vec!["bench"];
        let model_ref;
        if let Some(ref m) = model {
            model_ref = m.as_str();
            args.extend_from_slice(&["--model", model_ref]);
        }
        let batch_str;
        if let Some(b) = batch_size {
            batch_str = b.to_string();
            args.extend_from_slice(&["--batch-size", &batch_str]);
        }
        let seq_str;
        if let Some(s) = seq_len {
            seq_str = s.to_string();
            args.extend_from_slice(&["--seq-len", &seq_str]);
        }
        util::run_pmetal_blocking(&args).await
    }

    /// Benchmark generation throughput across a set of standard prompts.
    /// Maps to `pmetal bench-gen`.
    #[tool]
    async fn bench_gen(
        &self,
        #[description("Model to benchmark (default: Qwen3-0.6B)")]
        model: Option<String>,
    ) -> McpResult<String> {
        let mut args = vec!["bench-gen"];
        let model_ref;
        if let Some(ref m) = model {
            model_ref = m.as_str();
            args.extend_from_slice(&["--model", model_ref]);
        }
        util::run_pmetal_blocking(&args).await
    }

    /// Run the full benchmark corpus: evaluates a standard suite of
    /// tasks and reports aggregate scores. Maps to `pmetal bench-corpus`.
    #[tool]
    async fn bench_corpus(
        &self,
        #[description("Use shorter run with fewer iterations")]
        quick: Option<bool>,
        #[description("Output path for JSON report")]
        output: Option<String>,
    ) -> McpResult<String> {
        let mut args = vec!["bench-corpus"];
        if quick.unwrap_or(false) {
            args.push("--quick");
        }
        let output_ref;
        if let Some(ref o) = output {
            output_ref = o.as_str();
            args.extend_from_slice(&["--output", output_ref]);
        }
        util::run_pmetal_blocking(&args).await
    }

    // ── Server Management ─────────────────────────────────────────────────

    /// Start an OpenAI-compatible inference server as a background job.
    /// Returns a job ID. The server listens on the specified port.
    #[tool]
    async fn start_serve(
        &self,
        #[description("Model ID or local path")]
        model: String,
        #[description("Port to listen on (default: 8080)")]
        port: Option<u64>,
        #[description("LoRA adapter path")]
        lora: Option<String>,
        #[description("Host to bind to (default: 0.0.0.0)")]
        host: Option<String>,
        #[description("Max sequence length for KV cache (default: 4096)")]
        max_seq_len: Option<u64>,
        #[description("Packed expert weights dir for SSD-offloaded MoE")]
        experts_dir: Option<String>,
        #[description("Enable ANE (Apple Neural Engine) serving")]
        ane: Option<bool>,
        #[description("Maximum ANE kernel sequence length")]
        ane_max_seq_len: Option<u64>,
        #[description("Use experimental ANE real-time serving path")]
        ane_real_time: Option<bool>,
    ) -> McpResult<String> {
        let mut args = vec!["--model".to_string(), model];
        push_opt(&mut args, "--port", &port);
        push_opt(&mut args, "--lora", &lora);
        push_opt(&mut args, "--host", &host);
        push_opt(&mut args, "--max-seq-len", &max_seq_len);
        push_opt(&mut args, "--experts-dir", &experts_dir);
        push_bool_flag(&mut args, "--ane", &ane);
        push_opt(&mut args, "--ane-max-seq-len", &ane_max_seq_len);
        push_bool_flag(&mut args, "--ane-real-time", &ane_real_time);

        let mut mgr = self.jobs.write().await;
        let id = mgr.spawn("serve", args).await?;
        job_started_response(&id, "serve")
    }

    // ── Ollama Integration ────────────────────────────────────────────────

    /// Create an Ollama model entry from a PMetal model or LoRA adapter.
    /// Registers the model in Ollama so it can be used with `ollama run`.
    #[tool]
    async fn ollama_create(
        &self,
        #[description("Model name for Ollama (e.g. 'my-finetuned-model')")]
        name: String,
        #[description("Base model (GGUF path or Ollama model name)")]
        base: String,
        #[description("LoRA adapter path")]
        lora: Option<String>,
        #[description("System prompt")]
        system: Option<String>,
        #[description("Temperature (0.0-2.0)")]
        temperature: Option<f64>,
        #[description("Context window size")]
        num_ctx: Option<u64>,
        #[description("Model template: llama3, qwen3, gemma, mistral, phi3, deepseek")]
        template: Option<String>,
    ) -> McpResult<String> {
        let mut args = vec!["ollama", "create", "--name", &name, "--base", &base];
        let lora_ref;
        if let Some(ref l) = lora {
            lora_ref = l.as_str();
            args.extend_from_slice(&["--lora", lora_ref]);
        }
        let system_ref;
        if let Some(ref s) = system {
            system_ref = s.as_str();
            args.extend_from_slice(&["--system", system_ref]);
        }
        let temp_str;
        if let Some(t) = temperature {
            temp_str = t.to_string();
            args.extend_from_slice(&["--temperature", &temp_str]);
        }
        let ctx_str;
        if let Some(c) = num_ctx {
            ctx_str = c.to_string();
            args.extend_from_slice(&["--num-ctx", &ctx_str]);
        }
        let template_ref;
        if let Some(ref t) = template {
            template_ref = t.as_str();
            args.extend_from_slice(&["--template", template_ref]);
        }
        util::run_pmetal_blocking(&args).await
    }

    /// Generate a Modelfile for use with Ollama. Prints or writes the
    /// Modelfile content without registering it in Ollama.
    #[tool]
    async fn ollama_modelfile(
        &self,
        #[description("Base model (GGUF path or Ollama model name)")]
        base: String,
        #[description("LoRA adapter path")]
        lora: Option<String>,
        #[description("Output Modelfile path (default: Modelfile)")]
        output: Option<String>,
        #[description("System prompt")]
        system: Option<String>,
        #[description("Temperature (0.0-2.0)")]
        temperature: Option<f64>,
        #[description("Context window size")]
        num_ctx: Option<u64>,
        #[description("Top-k sampling")]
        top_k: Option<u64>,
        #[description("Top-p nucleus sampling")]
        top_p: Option<f64>,
        #[description("Model template: llama3, qwen3, gemma, mistral, phi3, deepseek")]
        template: Option<String>,
        #[description("License text for the model")]
        license: Option<String>,
    ) -> McpResult<String> {
        let mut args = vec!["ollama", "modelfile", "--base", &base];
        let lora_ref;
        if let Some(ref l) = lora {
            lora_ref = l.as_str();
            args.extend_from_slice(&["--lora", lora_ref]);
        }
        let output_ref;
        if let Some(ref o) = output {
            output_ref = o.as_str();
            args.extend_from_slice(&["--output", output_ref]);
        }
        let system_ref;
        if let Some(ref s) = system {
            system_ref = s.as_str();
            args.extend_from_slice(&["--system", system_ref]);
        }
        let temp_str;
        if let Some(t) = temperature {
            temp_str = t.to_string();
            args.extend_from_slice(&["--temperature", &temp_str]);
        }
        let ctx_str;
        if let Some(c) = num_ctx {
            ctx_str = c.to_string();
            args.extend_from_slice(&["--num-ctx", &ctx_str]);
        }
        let top_k_str;
        if let Some(k) = top_k {
            top_k_str = k.to_string();
            args.extend_from_slice(&["--top-k", &top_k_str]);
        }
        let top_p_str;
        if let Some(p) = top_p {
            top_p_str = p.to_string();
            args.extend_from_slice(&["--top-p", &top_p_str]);
        }
        let template_ref;
        if let Some(ref t) = template {
            template_ref = t.as_str();
            args.extend_from_slice(&["--template", template_ref]);
        }
        let license_ref;
        if let Some(ref l) = license {
            license_ref = l.as_str();
            args.extend_from_slice(&["--license", license_ref]);
        }
        util::run_pmetal_blocking(&args).await
    }

    // ── Runtime Training Control ──────────────────────────────────────────

    /// Set the learning rate of a running training job to an absolute value.
    /// The change takes effect within ~10 training steps (control file poll interval).
    #[tool]
    async fn job_set_lr(
        &self,
        #[description("Job ID of the running training job")]
        job_id: String,
        #[description("New learning rate value (e.g. 1e-5)")]
        lr: f64,
    ) -> McpResult<String> {
        let mgr = self.jobs.read().await;
        let json = serde_json::to_string(&serde_json::json!({"action": "set_lr", "value": lr}))
            .map_err(|e| McpError::internal(e.to_string()))?;
        mgr.write_control(&job_id, &json)?;
        Ok(format!("LR set to {lr:.2e} for job {job_id}. Takes effect within ~10 steps."))
    }

    /// Reduce the learning rate of a running training job by a factor.
    /// For example, factor=0.5 halves the current LR.
    #[tool]
    async fn job_reduce_lr(
        &self,
        #[description("Job ID of the running training job")]
        job_id: String,
        #[description("Reduction factor (e.g. 0.5 = halve LR)")]
        factor: f64,
    ) -> McpResult<String> {
        let mgr = self.jobs.read().await;
        let json = serde_json::to_string(&serde_json::json!({"action": "reduce_lr", "factor": factor}))
            .map_err(|e| McpError::internal(e.to_string()))?;
        mgr.write_control(&job_id, &json)?;
        Ok(format!("LR reduced by factor {factor} for job {job_id}. Takes effect within ~10 steps."))
    }

    /// Reset the learning rate of a running training job back to the scheduled value.
    /// Clears all adaptive adjustments (plateau reductions, spike cooldowns, manual overrides).
    #[tool]
    async fn job_reset_lr(
        &self,
        #[description("Job ID of the running training job")]
        job_id: String,
    ) -> McpResult<String> {
        let mgr = self.jobs.read().await;
        let json = r#"{"action":"reset_lr"}"#;
        mgr.write_control(&job_id, json)?;
        Ok(format!("LR reset to schedule for job {job_id}. Takes effect within ~10 steps."))
    }

    /// Trigger an immediate checkpoint save for a running training job.
    /// Training continues after the checkpoint is saved.
    #[tool]
    async fn job_save_checkpoint(
        &self,
        #[description("Job ID of the running training job")]
        job_id: String,
    ) -> McpResult<String> {
        let mgr = self.jobs.read().await;
        let json = r#"{"action":"save_checkpoint"}"#;
        mgr.write_control(&job_id, json)?;
        Ok(format!("Checkpoint save requested for job {job_id}. Will be saved within ~10 steps."))
    }

    /// Gracefully stop a running training job: restore best weights,
    /// save a final checkpoint, then exit cleanly.
    #[tool]
    async fn job_graceful_stop(
        &self,
        #[description("Job ID of the running training job")]
        job_id: String,
    ) -> McpResult<String> {
        let mgr = self.jobs.read().await;
        let json = r#"{"action":"graceful_stop"}"#;
        mgr.write_control(&job_id, json)?;
        Ok(format!("Graceful stop requested for job {job_id}. Will save best weights and checkpoint within ~10 steps."))
    }

    // ── Model Inspection ──────────────────────────────────────────────────

    /// Read a model's config.json and return architecture details:
    /// model type, hidden size, num layers, num heads, vocab size, etc.
    #[tool]
    async fn model_info(
        &self,
        #[description("Model ID or local path")]
        model: String,
    ) -> McpResult<String> {
        // Try local path first, then HF cache
        let config_path = if std::path::Path::new(&model).join("config.json").exists() {
            std::path::Path::new(&model).join("config.json")
        } else {
            // Try to find in HF cache
            let cache_dir = util::hf_cache_dir();
            let model_dir_name = format!("models--{}", model.replace('/', "--"));
            let model_dir = cache_dir.join(&model_dir_name);
            let snapshots_dir = model_dir.join("snapshots");

            if snapshots_dir.is_dir() {
                // Find the latest snapshot (skip dotfiles like .DS_Store)
                let snapshot = std::fs::read_dir(&snapshots_dir)
                    .ok()
                    .and_then(|entries| {
                        entries
                            .filter_map(|e| e.ok())
                            .find(|e| {
                                let name = e.file_name();
                                let name = name.to_string_lossy();
                                !name.starts_with('.') && e.path().is_dir()
                            })
                            .map(|e| e.path())
                    });

                match snapshot {
                    Some(snap_dir) => snap_dir.join("config.json"),
                    None => {
                        return Err(McpError::invalid_params(format!(
                            "model not found locally: {model}"
                        )));
                    }
                }
            } else {
                return Err(McpError::invalid_params(format!(
                    "model not found locally: {model}. Download it first with download_model."
                )));
            }
        };

        let content = std::fs::read_to_string(&config_path).map_err(|e| {
            McpError::internal(format!("failed to read config.json: {e}"))
        })?;

        let config: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
            McpError::internal(format!("invalid config.json: {e}"))
        })?;

        // Extract key fields for a clean summary
        let summary = serde_json::json!({
            "model": model,
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "vocab_size": config.get("vocab_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "rope_theta": config.get("rope_theta"),
            "torch_dtype": config.get("torch_dtype"),
            "num_experts": config.get("num_local_experts").or(config.get("num_experts")),
            "num_experts_per_tok": config.get("num_experts_per_tok"),
            "tie_word_embeddings": config.get("tie_word_embeddings"),
            "full_config": config,
        });

        serde_json::to_string_pretty(&summary).map_err(|e| McpError::internal(e.to_string()))
    }

    // ── Serve Interaction ─────────────────────────────────────────────────

    /// Send a chat completion request to a running pmetal serve instance.
    /// Requires a serve job to be running (started with start_serve).
    #[tool]
    async fn chat(
        &self,
        #[description("User message")]
        message: String,
        #[description("Port of the serve instance (default: 8080)")]
        port: Option<u64>,
        #[description("System prompt")]
        system: Option<String>,
        #[description("Sampling temperature")]
        temperature: Option<f64>,
        #[description("Maximum tokens to generate")]
        max_tokens: Option<u64>,
    ) -> McpResult<String> {
        let port = port.unwrap_or(8080);
        let url = format!("http://localhost:{port}/v1/chat/completions");

        let mut messages = Vec::new();
        if let Some(sys) = &system {
            messages.push(serde_json::json!({"role": "system", "content": sys}));
        }
        messages.push(serde_json::json!({"role": "user", "content": message}));

        let mut body = serde_json::json!({
            "messages": messages,
        });
        if let Some(t) = temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(m) = max_tokens {
            body["max_tokens"] = serde_json::json!(m);
        }

        let client = reqwest::Client::new();
        let response = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| McpError::internal(format!("failed to reach serve at {url}: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(McpError::internal(format!("serve returned {status}: {text}")));
        }

        let result: serde_json::Value = response
            .json()
            .await
            .map_err(|e| McpError::internal(format!("invalid response from serve: {e}")))?;

        // Extract the assistant's message content
        if let Some(content) = result
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
        {
            Ok(content.to_string())
        } else {
            Ok(serde_json::to_string_pretty(&result)
                .map_err(|e| McpError::internal(e.to_string()))?)
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn job_started_response(id: &str, command: &str) -> McpResult<String> {
    serde_json::to_string_pretty(&serde_json::json!({
        "job_id": id,
        "command": command,
        "status": "running",
        "message": format!("Job started. Use job_status with job_id '{id}' to monitor progress."),
    }))
    .map_err(|e| McpError::internal(e.to_string()))
}

/// Recursively compute directory size in bytes.
fn dir_size(path: &std::path::Path) -> u64 {
    let mut size = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let meta = entry.metadata();
            if let Ok(m) = meta {
                if m.is_dir() {
                    size += dir_size(&entry.path());
                } else {
                    size += m.len();
                }
            }
        }
    }
    size
}
