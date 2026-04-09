use crate::{GdnBenchmarkStage, WorkloadBenchmarkPreset, WorkloadInferenceContext};
use anyhow::Context;
use half::f16;
use pmetal::inference_runner::{CacheModeRequest, select_cache_mode_for_model};
use pmetal_bridge::compat::module::ModuleParameters as _;
use pmetal_bridge::compat::{Array, ops};
use pmetal_core::{
    DatasetConfig, LoraConfig, ModelConfig, StepMetrics, TrainingCallback, TrainingConfig,
};
use pmetal_data::{DatasetColumnConfig, DatasetFormat, TextSample, Tokenizer, TrainingDataset};
use pmetal_lora::LlamaLoraForCausalLM;
use pmetal_metal::context::{DeviceTier, MemoryBandwidthSource};
use pmetal_metal::kernels::BatchedCommandBuffer;
use pmetal_metal::kernels::mpp_gemm::{MppGemm, MppGemmConfig};
use pmetal_metal::tuna::MppGemmTuneRequest;
use pmetal_metal::{
    BufferUsage, FlashAttention, FlashAttentionConfig, FusedLinearCrossEntropy,
    FusedLinearCrossEntropyConfig, FusedLora, FusedLoraConfig, FusedMLP, FusedMergeMetal,
    FusedNormLora, FusedNormLoraConfig, FusedSwiGLUConfig, MetalBuffer, MetalContext,
    build_merge_config, build_tensor_info,
};
use pmetal_mlx::kernels::gated_delta_update_with_chunk_size_override;
use pmetal_mlx::kv_cache::CacheMode;
use pmetal_models::architectures::deepseek::{DeepSeekConfig, DeepSeekMoE};
use pmetal_models::architectures::jamba::{JambaConfig, JambaLayer};
use pmetal_models::architectures::llama::LlamaConfig;
use pmetal_models::architectures::llama4::{Llama4MoE, Llama4TextConfig};
use pmetal_models::architectures::qwen3_moe::{Qwen3MoEBlock, Qwen3MoEConfig};
use pmetal_models::architectures::qwen3_next::Qwen3NextGatedDeltaNet;
use pmetal_models::dispatcher::{DynamicModel, DynamicModelLoadOptions};
use pmetal_trainer::orchestrator::{FullTrainingConfig, run_training};
use pmetal_trainer::{DispatchConfig, TrainingJobConfig};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Run benchmark.
pub(crate) async fn run_benchmark(
    model: &str,
    batch_size: usize,
    seq_len: usize,
) -> anyhow::Result<()> {
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

    let mut model_inst = LlamaLoraForCausalLM::new(llama_config, lora_config)?;

    // Create dummy data
    let input_ids = pmetal_bridge::compat::ops::zeros(
        &[batch_size as i32, seq_len as i32],
        pmetal_bridge::compat::Dtype::Int32,
    );

    // Warmup
    println!("Warming up...");
    for _ in 0..3 {
        let output = model_inst.forward(&input_ids, None)?;
        pmetal_bridge::compat::transforms::eval([&output])?;
    }

    // Benchmark
    let iterations = 10;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        let output = model_inst.forward(&input_ids, None)?;
        pmetal_bridge::compat::transforms::eval([&output])?;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WorkloadBenchmarkReport {
    version: String,
    generated_at_unix_ms: u128,
    device: KernelBenchmarkDevice,
    workload: WorkloadBenchmarkConfig,
    inference: WorkloadBenchmarkSection<InferenceWorkloadMetrics>,
    training: WorkloadBenchmarkSection<TrainingWorkloadMetrics>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct GdnDecodeBenchmarkReport {
    version: String,
    generated_at_unix_ms: u128,
    device: KernelBenchmarkDevice,
    stage: String,
    model_id: String,
    resolved_model_path: String,
    layer_idx: usize,
    batch_size: usize,
    seq_len: usize,
    input_dim: usize,
    output_dim: usize,
    warmup_iterations: usize,
    benchmark_iterations: usize,
    reference_backend: String,
    backends: Vec<GdnDecodeBackendResult>,
}

#[derive(Debug, Clone, Serialize)]
struct GdnDecodeBackendResult {
    name: String,
    max_abs_diff_vs_reference: Option<f32>,
    outcome: KernelBenchmarkOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkloadBenchmarkConfig {
    preset: Option<String>,
    model_id: String,
    dataset_id: String,
    experts_dir: Option<String>,
    resolved_model_path: String,
    resolved_dataset_path: String,
    prompt_samples: usize,
    max_prompt_tokens: usize,
    inference_prompt_len: WorkloadPromptLenSelection,
    decode_steps: usize,
    inference_warmup_passes: usize,
    inference_session_repeats: usize,
    inference_repeats: usize,
    train_samples: usize,
    train_steps: usize,
    batch_size: usize,
    max_seq_len: usize,
    training_seq_len: WorkloadTrainingSeqLenSelection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum WorkloadBenchmarkSection<T> {
    Completed(T),
    Skipped { reason: String },
    Failed { error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InferenceWorkloadMetrics {
    session_runs: usize,
    prompt_samples: usize,
    measurement_passes: usize,
    warmup_passes: usize,
    prompt_tokens: usize,
    max_prompt_tokens: usize,
    decode_steps: usize,
    inference_repeats: usize,
    cache_mode: String,
    cache_mode_source: String,
    first_generated_token_id: Option<u32>,
    first_generated_token_text: Option<String>,
    total_prefill_ms: f64,
    prefill_tok_per_sec: f64,
    mean_prefill_tok_per_sec: f64,
    median_prefill_tok_per_sec: f64,
    total_decode_ms: f64,
    decode_tok_per_sec: f64,
    decode_ms_per_token: f64,
    mean_decode_tok_per_sec: f64,
    median_decode_tok_per_sec: f64,
    mean_decode_ms_per_token: f64,
    median_decode_ms_per_token: f64,
    mean_session_prefill_tok_per_sec: f64,
    median_session_prefill_tok_per_sec: f64,
    mean_session_decode_tok_per_sec: f64,
    median_session_decode_tok_per_sec: f64,
    mean_session_decode_ms_per_token: f64,
    median_session_decode_ms_per_token: f64,
    session_prefill_tok_per_sec: Vec<f64>,
    session_decode_tok_per_sec: Vec<f64>,
    session_decode_ms_per_token: Vec<f64>,
    prefetch_hits: Option<usize>,
    prefetch_misses: Option<usize>,
    prefetch_total: Option<usize>,
    prefetch_hit_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkloadPromptLenSelection {
    context_source: String,
    context_source_reason: String,
    requested_max_prompt_tokens: usize,
    effective_max_prompt_tokens: usize,
    max_prompt_tokens_source: String,
    sample_median_prompt_tokens: usize,
    sample_p95_prompt_tokens: usize,
    sample_max_prompt_tokens: usize,
    sample_truncated_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrainingWorkloadMetrics {
    train_samples: usize,
    train_steps: usize,
    batch_size: usize,
    max_seq_len: usize,
    total_tokens: usize,
    total_steps: usize,
    wall_clock_ms: f64,
    median_tok_sec: f64,
    mean_tok_sec: f64,
    median_step_ms: f64,
    mean_step_ms: f64,
    final_loss: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkloadTrainingSeqLenSelection {
    requested_max_seq_len: usize,
    effective_max_seq_len: usize,
    max_seq_len_source: String,
    sample_median_tokens: usize,
    sample_p95_tokens: usize,
    sample_max_tokens: usize,
    sample_truncated_pct: f64,
}

#[derive(Debug, Clone, Copy)]
struct WorkloadPresetConfig {
    preset: WorkloadBenchmarkPreset,
    model_id: &'static str,
    dataset_id: &'static str,
    prompt_samples: usize,
    max_prompt_tokens: usize,
    inference_context: WorkloadInferenceContext,
    decode_steps: usize,
    inference_repeats: usize,
    train_samples: usize,
    train_steps: usize,
    batch_size: usize,
    max_seq_len: usize,
}

#[derive(Clone, Default)]
struct StepMetricsCollector {
    steps: Arc<Mutex<Vec<StepMetrics>>>,
}

impl StepMetricsCollector {
    fn new() -> Self {
        Self::default()
    }

    fn snapshot(&self) -> Vec<StepMetrics> {
        self.steps.lock().expect("step metrics lock").clone()
    }
}

impl TrainingCallback for StepMetricsCollector {
    fn on_step_end_with_metrics(&mut self, metrics: &StepMetrics) {
        self.steps
            .lock()
            .expect("step metrics lock")
            .push(metrics.clone());
    }
}

fn preset_label(preset: WorkloadBenchmarkPreset) -> &'static str {
    match preset {
        WorkloadBenchmarkPreset::DenseQwen3 => "dense-qwen3",
        WorkloadBenchmarkPreset::HybridQwen3Next => "hybrid-qwen3next",
        WorkloadBenchmarkPreset::HybridQwen35Steady => "hybrid-qwen35-steady",
        WorkloadBenchmarkPreset::MoeNemotronH => "moe-nemotronh",
    }
}

fn workload_preset_config(preset: WorkloadBenchmarkPreset) -> WorkloadPresetConfig {
    match preset {
        WorkloadBenchmarkPreset::DenseQwen3 => WorkloadPresetConfig {
            preset,
            model_id: "Qwen/Qwen3-0.6B",
            dataset_id: "TeichAI/gemini-3-pro-preview-high-reasoning-250x",
            prompt_samples: 4,
            max_prompt_tokens: 0,
            inference_context: WorkloadInferenceContext::Auto,
            decode_steps: 16,
            inference_repeats: 1,
            train_samples: 4,
            train_steps: 2,
            batch_size: 1,
            max_seq_len: 0,
        },
        WorkloadBenchmarkPreset::HybridQwen3Next => WorkloadPresetConfig {
            preset,
            model_id: "unsloth/Qwen3.5-0.8B",
            dataset_id: "TeichAI/gemini-3-pro-preview-high-reasoning-250x",
            prompt_samples: 4,
            max_prompt_tokens: 0,
            inference_context: WorkloadInferenceContext::TextPrefix,
            decode_steps: 8,
            inference_repeats: 1,
            train_samples: 0,
            train_steps: 0,
            batch_size: 1,
            max_seq_len: 0,
        },
        WorkloadBenchmarkPreset::HybridQwen35Steady => WorkloadPresetConfig {
            preset,
            model_id: "unsloth/Qwen3.5-0.8B",
            dataset_id: "TeichAI/gemini-3-pro-preview-high-reasoning-250x",
            prompt_samples: 2,
            max_prompt_tokens: 0,
            inference_context: WorkloadInferenceContext::TextPrefix,
            decode_steps: 64,
            inference_repeats: 3,
            train_samples: 0,
            train_steps: 0,
            batch_size: 1,
            max_seq_len: 0,
        },
        WorkloadBenchmarkPreset::MoeNemotronH => WorkloadPresetConfig {
            preset,
            model_id: "unsloth/NVIDIA-Nemotron-3-Nano-4B",
            dataset_id: "TeichAI/gemini-3-pro-preview-high-reasoning-250x",
            prompt_samples: 2,
            max_prompt_tokens: 512,
            inference_context: WorkloadInferenceContext::TextPrefix,
            decode_steps: 4,
            inference_repeats: 1,
            train_samples: 0,
            train_steps: 0,
            batch_size: 1,
            max_seq_len: 0,
        },
    }
}

struct GdnInputProjectionBenchmarkSetup {
    model_path: PathBuf,
    layer_idx: usize,
    batch_size: usize,
    seq_len: usize,
    input_dim: usize,
    output_dim: usize,
    conv_dim: usize,
    value_dim: usize,
    num_v_heads: usize,
    head_v_dim: usize,
    input: Array,
    input_data: Vec<f32>,
    qkv_weight: Array,
    qkv_weight_t: Array,
    z_weight: Array,
    z_weight_t: Array,
    b_weight: Array,
    b_weight_t: Array,
    a_weight: Array,
    a_weight_t: Array,
    qkv_z_combined_weight: Array,
    qkv_z_combined_weight_t: Array,
    combined_weight: Array,
    combined_weight_t: Array,
    combined_weight_data: Vec<f32>,
    reference_output_data: Vec<f32>,
}

struct GdnLinearProjectionBenchmarkSetup {
    model_path: PathBuf,
    layer_idx: usize,
    batch_size: usize,
    seq_len: usize,
    input_dim: usize,
    output_dim: usize,
    input: Array,
    input_data: Vec<f32>,
    weight: Array,
    weight_t: Array,
    weight_data: Vec<f32>,
    reference_output_data: Vec<f32>,
}

struct GdnPrefillBenchmarkSetup {
    model_path: PathBuf,
    layer_idx: usize,
    batch_size: usize,
    seq_len: usize,
    input_dim: usize,
    output_dim: usize,
    q: Array,
    k: Array,
    v: Array,
    a: Array,
    b: Array,
    a_log: Array,
    dt_bias: Array,
    reference_output_data: Vec<f32>,
}

enum GdnBenchmarkSetup {
    InputProj(GdnInputProjectionBenchmarkSetup),
    OutProj(GdnLinearProjectionBenchmarkSetup),
    Prefill(GdnPrefillBenchmarkSetup),
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct KernelBenchmarkReport {
    version: String,
    generated_at_unix_ms: u128,
    mode: &'static str,
    warmup_iterations: usize,
    benchmark_iterations: usize,
    device: KernelBenchmarkDevice,
    summary: KernelBenchmarkSummary,
    cases: Vec<KernelBenchmarkCaseResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KernelBenchmarkDevice {
    name: String,
    tier: String,
    architecture_gen: u32,
    gpu_core_count: u32,
    ane_core_count: u32,
    has_nax: bool,
    is_apple10_or_newer: bool,
    is_ultra_fusion: bool,
    memory_bandwidth_gbps: f64,
    memory_bandwidth_source: String,
}

#[derive(Debug, Clone, Serialize)]
struct KernelBenchmarkSummary {
    completed: usize,
    skipped: usize,
    failed: usize,
}

#[derive(Debug, Clone, Serialize)]
struct KernelBenchmarkCaseResult {
    name: String,
    category: &'static str,
    parameters: BTreeMap<String, String>,
    tuning: BTreeMap<String, String>,
    outcome: KernelBenchmarkOutcome,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum KernelBenchmarkOutcome {
    Completed {
        min_ms: f64,
        median_ms: f64,
        mean_ms: f64,
    },
    Skipped {
        reason: String,
    },
    Failed {
        error: String,
    },
}

#[derive(Debug, Clone, Copy)]
enum KernelBenchmarkCase {
    FlashAttention(FlashAttentionCase),
    FusedLora(FusedLoraCase),
    FusedMlp(FusedMlpCase),
    FusedNormLora(FusedNormLoraCase),
    FusedLinearCrossEntropy(FusedLinearCrossEntropyCase),
    FusedMerge(FusedMergeCase),
    ModelMoe(ModelMoeCase),
    ModelHybrid(ModelHybridCase),
    MppGemm(MppGemmCase),
}

#[derive(Debug, Clone, Copy)]
enum ModelMoeFamily {
    Llama4,
    Qwen3,
    DeepSeek,
}

#[derive(Debug, Clone, Copy)]
struct FlashAttentionCase {
    name: &'static str,
    batch_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
}

#[derive(Debug, Clone, Copy)]
struct FusedLoraCase {
    name: &'static str,
    batch_size: usize,
    in_features: usize,
    out_features: usize,
    rank: usize,
}

#[derive(Debug, Clone, Copy)]
struct FusedMlpCase {
    name: &'static str,
    batch_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
}

#[derive(Debug, Clone, Copy)]
struct FusedNormLoraCase {
    name: &'static str,
    batch_size: usize,
    hidden_size: usize,
    out_features: usize,
    rank: usize,
}

#[derive(Debug, Clone, Copy)]
struct FusedLinearCrossEntropyCase {
    name: &'static str,
    num_tokens: usize,
    hidden_size: usize,
    vocab_size: usize,
}

#[derive(Debug, Clone, Copy)]
struct FusedMergeCase {
    name: &'static str,
    num_models: usize,
    elements_per_model: usize,
}

#[derive(Debug, Clone, Copy)]
struct ModelMoeCase {
    name: &'static str,
    family: ModelMoeFamily,
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
}

#[derive(Debug, Clone, Copy)]
enum ModelHybridFamily {
    Jamba,
}

#[derive(Debug, Clone, Copy)]
struct ModelHybridCase {
    name: &'static str,
    family: ModelHybridFamily,
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    top_k: usize,
}

#[derive(Debug, Clone, Copy)]
struct MppGemmCase {
    name: &'static str,
    m: usize,
    n: usize,
    k: usize,
}

#[derive(Debug, Clone, Copy)]
struct KernelBenchmarkTierProfile {
    flash_attention: FlashAttentionCase,
    fused_lora: FusedLoraCase,
    fused_mlp: FusedMlpCase,
    fused_norm_lora: FusedNormLoraCase,
    fused_linear_cross_entropy: FusedLinearCrossEntropyCase,
    fused_merge: FusedMergeCase,
    llama4_moe: ModelMoeCase,
    qwen3_moe: ModelMoeCase,
    deepseek_moe: ModelMoeCase,
    jamba_hybrid: ModelHybridCase,
    mpp_gemm: MppGemmCase,
}

pub(crate) fn run_kernel_benchmark_corpus(
    quick: bool,
    output: Option<&Path>,
    json: bool,
) -> anyhow::Result<()> {
    let ctx = Arc::new(MetalContext::new().context("failed to initialize Metal context")?);
    let props = ctx.properties();
    let warmup_iterations = if quick { 2 } else { 3 };
    let benchmark_iterations = if quick { 5 } else { 10 };

    let cases = build_benchmark_corpus_for_profile(props.device_tier, props.has_nax(), quick);
    let mut results = Vec::with_capacity(cases.len());
    for case in &cases {
        results.push(run_kernel_benchmark_case(
            &ctx,
            case,
            warmup_iterations,
            benchmark_iterations,
        ));
    }

    let summary = summarize_kernel_benchmark_results(&results);
    let report = KernelBenchmarkReport {
        version: env!("CARGO_PKG_VERSION").to_string(),
        generated_at_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
        mode: if quick { "quick" } else { "standard" },
        warmup_iterations,
        benchmark_iterations,
        device: KernelBenchmarkDevice {
            name: props.name.clone(),
            tier: device_tier_label(props.device_tier).to_string(),
            architecture_gen: props.architecture_gen,
            gpu_core_count: props.gpu_core_count,
            ane_core_count: props.ane_core_count,
            has_nax: props.has_nax(),
            is_apple10_or_newer: props.is_apple10_or_newer(),
            is_ultra_fusion: props.is_ultra_fusion,
            memory_bandwidth_gbps: props.memory_bandwidth_gbps,
            memory_bandwidth_source: memory_bandwidth_source_label(props.memory_bandwidth_source)
                .to_string(),
        },
        summary,
        cases: results,
    };

    let report_json = serde_json::to_string_pretty(&report)?;
    if let Some(output_path) = output {
        std::fs::write(output_path, &report_json).with_context(|| {
            format!(
                "failed to write benchmark corpus to {}",
                output_path.display()
            )
        })?;
    }

    if json {
        println!("{report_json}");
    } else {
        print_kernel_benchmark_report(&report, output);
    }

    Ok(())
}

pub(crate) async fn run_gdn_decode_benchmark(
    model_id: &str,
    stage: GdnBenchmarkStage,
    layer: Option<usize>,
    batch_size: usize,
    seq_len: usize,
    warmup_iterations: usize,
    benchmark_iterations: usize,
    output: Option<&Path>,
    json: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(batch_size > 0, "batch_size must be greater than 0");
    anyhow::ensure!(seq_len > 0, "seq_len must be greater than 0");
    anyhow::ensure!(
        benchmark_iterations > 0,
        "benchmark_iterations must be greater than 0"
    );

    let device = workload_benchmark_device()?;
    let model_path = resolve_workload_model_path(model_id).await?;
    let setup = build_gdn_decode_benchmark_setup(&model_path, stage, layer, batch_size, seq_len)?;

    let report = match setup {
        GdnBenchmarkSetup::InputProj(setup) => {
            let split_outcome = benchmark_operation(
                warmup_iterations,
                benchmark_iterations,
                || -> anyhow::Result<()> {
                    let qkv = mlx_linear_projection(&setup.input, &setup.qkv_weight)?;
                    let z = mlx_linear_projection(&setup.input, &setup.z_weight)?;
                    let b_val = mlx_linear_projection(&setup.input, &setup.b_weight)?;
                    let a = mlx_linear_projection(&setup.input, &setup.a_weight)?;
                    pmetal_bridge::compat::transforms::eval([&qkv])?;
                    pmetal_bridge::compat::transforms::eval([&z])?;
                    pmetal_bridge::compat::transforms::eval([&b_val])?;
                    pmetal_bridge::compat::transforms::eval([&a])?;
                    Ok(())
                },
            )?;

            let split_cached_t_outcome = benchmark_operation(
                warmup_iterations,
                benchmark_iterations,
                || -> anyhow::Result<()> {
                    let projected = mlx_split_projection_rhs_transposed(
                        &setup.input,
                        &setup.qkv_weight_t,
                        &setup.z_weight_t,
                        &setup.b_weight_t,
                        &setup.a_weight_t,
                    )?;
                    pmetal_bridge::compat::transforms::eval([&projected])?;
                    Ok(())
                },
            )?;

            let combined_outcome = benchmark_operation(
                warmup_iterations,
                benchmark_iterations,
                || -> anyhow::Result<()> {
                    let projected = mlx_linear_projection(&setup.input, &setup.combined_weight)?;
                    pmetal_bridge::compat::transforms::eval([&projected])?;
                    Ok(())
                },
            )?;

            let combined_cached_t_outcome = benchmark_operation(
                warmup_iterations,
                benchmark_iterations,
                || -> anyhow::Result<()> {
                    let projected = mlx_linear_projection_rhs_transposed(
                        &setup.input,
                        &setup.combined_weight_t,
                    )?;
                    pmetal_bridge::compat::transforms::eval([&projected])?;
                    Ok(())
                },
            )?;

            let combined_split_outcome = benchmark_operation(
                warmup_iterations,
                benchmark_iterations,
                || -> anyhow::Result<()> {
                    let (qkv, z, b_val, a) = mlx_combined_split_projection(
                        &setup.input,
                        &setup.combined_weight,
                        setup.batch_size,
                        setup.seq_len,
                        setup.conv_dim,
                        setup.value_dim,
                        setup.num_v_heads,
                        setup.head_v_dim,
                    )?;
                    pmetal_bridge::compat::transforms::eval([&qkv])?;
                    pmetal_bridge::compat::transforms::eval([&z])?;
                    pmetal_bridge::compat::transforms::eval([&b_val])?;
                    pmetal_bridge::compat::transforms::eval([&a])?;
                    Ok(())
                },
            )?;

            let combined_split_cached_t_outcome = benchmark_operation(
                warmup_iterations,
                benchmark_iterations,
                || -> anyhow::Result<()> {
                    let (qkv, z, b_val, a) = mlx_combined_split_projection_rhs_transposed(
                        &setup.input,
                        &setup.combined_weight_t,
                        setup.batch_size,
                        setup.seq_len,
                        setup.conv_dim,
                        setup.value_dim,
                        setup.num_v_heads,
                        setup.head_v_dim,
                    )?;
                    pmetal_bridge::compat::transforms::eval([&qkv])?;
                    pmetal_bridge::compat::transforms::eval([&z])?;
                    pmetal_bridge::compat::transforms::eval([&b_val])?;
                    pmetal_bridge::compat::transforms::eval([&a])?;
                    Ok(())
                },
            )?;

            let qkv_z_combined_split_outcome = benchmark_operation(
                warmup_iterations,
                benchmark_iterations,
                || -> anyhow::Result<()> {
                    let (qkv, z, b_val, a) = mlx_qkv_z_combined_split_projection(
                        &setup.input,
                        &setup.qkv_z_combined_weight,
                        &setup.b_weight,
                        &setup.a_weight,
                        setup.batch_size,
                        setup.seq_len,
                        setup.conv_dim,
                        setup.value_dim,
                        setup.num_v_heads,
                        setup.head_v_dim,
                    )?;
                    pmetal_bridge::compat::transforms::eval([&qkv])?;
                    pmetal_bridge::compat::transforms::eval([&z])?;
                    pmetal_bridge::compat::transforms::eval([&b_val])?;
                    pmetal_bridge::compat::transforms::eval([&a])?;
                    Ok(())
                },
            )?;

            let qkv_z_combined_split_cached_t_outcome = benchmark_operation(
                warmup_iterations,
                benchmark_iterations,
                || -> anyhow::Result<()> {
                    let (qkv, z, b_val, a) = mlx_qkv_z_combined_split_projection_rhs_transposed(
                        &setup.input,
                        &setup.qkv_z_combined_weight_t,
                        &setup.b_weight_t,
                        &setup.a_weight_t,
                        setup.batch_size,
                        setup.seq_len,
                        setup.conv_dim,
                        setup.value_dim,
                        setup.num_v_heads,
                        setup.head_v_dim,
                    )?;
                    pmetal_bridge::compat::transforms::eval([&qkv])?;
                    pmetal_bridge::compat::transforms::eval([&z])?;
                    pmetal_bridge::compat::transforms::eval([&b_val])?;
                    pmetal_bridge::compat::transforms::eval([&a])?;
                    Ok(())
                },
            )?;

            let accelerate_outcome = benchmark_operation(
                warmup_iterations,
                benchmark_iterations,
                || -> anyhow::Result<()> {
                    let _ = accelerate_combined_projection(
                        &setup.input_data,
                        &setup.combined_weight_data,
                        batch_size * seq_len,
                        setup.input_dim,
                        setup.output_dim,
                    );
                    Ok(())
                },
            )?;

            let accelerate_roundtrip_outcome = benchmark_operation(
                warmup_iterations,
                benchmark_iterations,
                || -> anyhow::Result<()> {
                    let (qkv, z, b_val, a) = accelerate_roundtrip_split_projection(
                        &setup.input,
                        &setup.combined_weight_data,
                        setup.batch_size,
                        setup.seq_len,
                        setup.input_dim,
                        setup.output_dim,
                        setup.conv_dim,
                        setup.value_dim,
                        setup.num_v_heads,
                        setup.head_v_dim,
                    )?;
                    pmetal_bridge::compat::transforms::eval([&qkv])?;
                    pmetal_bridge::compat::transforms::eval([&z])?;
                    pmetal_bridge::compat::transforms::eval([&b_val])?;
                    pmetal_bridge::compat::transforms::eval([&a])?;
                    Ok(())
                },
            )?;

            let combined_output = mlx_linear_projection(&setup.input, &setup.combined_weight)?;
            pmetal_bridge::compat::transforms::eval([&combined_output])?;
            let combined_output_data = combined_output.as_slice::<f32>().to_vec();
            let (qkv_qz, z_qz, b_qz, a_qz) = mlx_qkv_z_combined_split_projection(
                &setup.input,
                &setup.qkv_z_combined_weight,
                &setup.b_weight,
                &setup.a_weight,
                setup.batch_size,
                setup.seq_len,
                setup.conv_dim,
                setup.value_dim,
                setup.num_v_heads,
                setup.head_v_dim,
            )?;
            let qkv_z_combined_output = ops::concatenate_axis(
                &[
                    &qkv_qz,
                    &z_qz.reshape(&[
                        setup.batch_size as i32,
                        setup.seq_len as i32,
                        setup.value_dim as i32,
                    ]),
                    &b_qz,
                    &a_qz,
                ],
                -1,
            );
            pmetal_bridge::compat::transforms::eval([&qkv_z_combined_output])?;
            let qkv_z_combined_output_data = qkv_z_combined_output.as_slice::<f32>().to_vec();
            let accelerate_output = accelerate_combined_projection(
                &setup.input_data,
                &setup.combined_weight_data,
                batch_size * seq_len,
                setup.input_dim,
                setup.output_dim,
            );

            GdnDecodeBenchmarkReport {
                version: env!("CARGO_PKG_VERSION").to_string(),
                generated_at_unix_ms: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis(),
                device,
                stage: "input-proj".to_string(),
                model_id: model_id.to_string(),
                resolved_model_path: setup.model_path.display().to_string(),
                layer_idx: setup.layer_idx,
                batch_size,
                seq_len,
                input_dim: setup.input_dim,
                output_dim: setup.output_dim,
                warmup_iterations,
                benchmark_iterations,
                reference_backend: "mlx_split".to_string(),
                backends: vec![
                    GdnDecodeBackendResult {
                        name: "mlx_split".to_string(),
                        max_abs_diff_vs_reference: Some(0.0),
                        outcome: split_outcome,
                    },
                    GdnDecodeBackendResult {
                        name: "mlx_split_cached_t".to_string(),
                        max_abs_diff_vs_reference: Some(0.0),
                        outcome: split_cached_t_outcome,
                    },
                    GdnDecodeBackendResult {
                        name: "mlx_combined".to_string(),
                        max_abs_diff_vs_reference: Some(max_abs_diff(
                            &setup.reference_output_data,
                            &combined_output_data,
                        )),
                        outcome: combined_outcome,
                    },
                    GdnDecodeBackendResult {
                        name: "mlx_combined_cached_t".to_string(),
                        max_abs_diff_vs_reference: Some(max_abs_diff(
                            &setup.reference_output_data,
                            &combined_output_data,
                        )),
                        outcome: combined_cached_t_outcome,
                    },
                    GdnDecodeBackendResult {
                        name: "mlx_combined_split".to_string(),
                        max_abs_diff_vs_reference: Some(max_abs_diff(
                            &setup.reference_output_data,
                            &combined_output_data,
                        )),
                        outcome: combined_split_outcome,
                    },
                    GdnDecodeBackendResult {
                        name: "mlx_combined_split_cached_t".to_string(),
                        max_abs_diff_vs_reference: Some(max_abs_diff(
                            &setup.reference_output_data,
                            &combined_output_data,
                        )),
                        outcome: combined_split_cached_t_outcome,
                    },
                    GdnDecodeBackendResult {
                        name: "mlx_qkv_z_combined_split".to_string(),
                        max_abs_diff_vs_reference: Some(max_abs_diff(
                            &setup.reference_output_data,
                            &qkv_z_combined_output_data,
                        )),
                        outcome: qkv_z_combined_split_outcome,
                    },
                    GdnDecodeBackendResult {
                        name: "mlx_qkv_z_combined_split_cached_t".to_string(),
                        max_abs_diff_vs_reference: Some(max_abs_diff(
                            &setup.reference_output_data,
                            &qkv_z_combined_output_data,
                        )),
                        outcome: qkv_z_combined_split_cached_t_outcome,
                    },
                    GdnDecodeBackendResult {
                        name: "accelerate_combined".to_string(),
                        max_abs_diff_vs_reference: Some(max_abs_diff(
                            &setup.reference_output_data,
                            &accelerate_output,
                        )),
                        outcome: accelerate_outcome,
                    },
                    GdnDecodeBackendResult {
                        name: "accelerate_roundtrip_split".to_string(),
                        max_abs_diff_vs_reference: Some(max_abs_diff(
                            &setup.reference_output_data,
                            &accelerate_output,
                        )),
                        outcome: accelerate_roundtrip_outcome,
                    },
                ],
            }
        }
        GdnBenchmarkSetup::OutProj(setup) => {
            let mlx_outcome = benchmark_operation(
                warmup_iterations,
                benchmark_iterations,
                || -> anyhow::Result<()> {
                    let projected = mlx_linear_projection(&setup.input, &setup.weight)?;
                    pmetal_bridge::compat::transforms::eval([&projected])?;
                    Ok(())
                },
            )?;

            let mlx_cached_t_outcome = benchmark_operation(
                warmup_iterations,
                benchmark_iterations,
                || -> anyhow::Result<()> {
                    let projected =
                        mlx_linear_projection_rhs_transposed(&setup.input, &setup.weight_t)?;
                    pmetal_bridge::compat::transforms::eval([&projected])?;
                    Ok(())
                },
            )?;

            let accelerate_outcome = benchmark_operation(
                warmup_iterations,
                benchmark_iterations,
                || -> anyhow::Result<()> {
                    let _ = accelerate_combined_projection(
                        &setup.input_data,
                        &setup.weight_data,
                        batch_size * seq_len,
                        setup.input_dim,
                        setup.output_dim,
                    );
                    Ok(())
                },
            )?;

            let accelerate_roundtrip_outcome = benchmark_operation(
                warmup_iterations,
                benchmark_iterations,
                || -> anyhow::Result<()> {
                    let projected = accelerate_roundtrip_linear_projection(
                        &setup.input,
                        &setup.weight_data,
                        setup.batch_size,
                        setup.seq_len,
                        setup.input_dim,
                        setup.output_dim,
                    )?;
                    pmetal_bridge::compat::transforms::eval([&projected])?;
                    Ok(())
                },
            )?;

            let accelerate_output = accelerate_combined_projection(
                &setup.input_data,
                &setup.weight_data,
                batch_size * seq_len,
                setup.input_dim,
                setup.output_dim,
            );

            GdnDecodeBenchmarkReport {
                version: env!("CARGO_PKG_VERSION").to_string(),
                generated_at_unix_ms: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis(),
                device,
                stage: "out-proj".to_string(),
                model_id: model_id.to_string(),
                resolved_model_path: setup.model_path.display().to_string(),
                layer_idx: setup.layer_idx,
                batch_size,
                seq_len,
                input_dim: setup.input_dim,
                output_dim: setup.output_dim,
                warmup_iterations,
                benchmark_iterations,
                reference_backend: "mlx_linear".to_string(),
                backends: vec![
                    GdnDecodeBackendResult {
                        name: "mlx_linear".to_string(),
                        max_abs_diff_vs_reference: Some(0.0),
                        outcome: mlx_outcome,
                    },
                    GdnDecodeBackendResult {
                        name: "mlx_linear_cached_t".to_string(),
                        max_abs_diff_vs_reference: Some(0.0),
                        outcome: mlx_cached_t_outcome,
                    },
                    GdnDecodeBackendResult {
                        name: "accelerate_combined".to_string(),
                        max_abs_diff_vs_reference: Some(max_abs_diff(
                            &setup.reference_output_data,
                            &accelerate_output,
                        )),
                        outcome: accelerate_outcome,
                    },
                    GdnDecodeBackendResult {
                        name: "accelerate_roundtrip_linear".to_string(),
                        max_abs_diff_vs_reference: Some(max_abs_diff(
                            &setup.reference_output_data,
                            &accelerate_output,
                        )),
                        outcome: accelerate_roundtrip_outcome,
                    },
                ],
            }
        }
        GdnBenchmarkSetup::Prefill(setup) => GdnDecodeBenchmarkReport {
            version: env!("CARGO_PKG_VERSION").to_string(),
            generated_at_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            device,
            stage: "prefill".to_string(),
            model_id: model_id.to_string(),
            resolved_model_path: setup.model_path.display().to_string(),
            layer_idx: setup.layer_idx,
            batch_size,
            seq_len,
            input_dim: setup.input_dim,
            output_dim: setup.output_dim,
            warmup_iterations,
            benchmark_iterations,
            reference_backend: "mlx_sequential".to_string(),
            backends: vec![
                run_gdn_prefill_backend(
                    &setup,
                    "mlx_sequential",
                    Some(0),
                    warmup_iterations,
                    benchmark_iterations,
                )?,
                run_gdn_prefill_backend(
                    &setup,
                    "mlx_chunk_32",
                    Some(32),
                    warmup_iterations,
                    benchmark_iterations,
                )?,
                run_gdn_prefill_backend(
                    &setup,
                    "mlx_chunk_64",
                    Some(64),
                    warmup_iterations,
                    benchmark_iterations,
                )?,
                run_gdn_prefill_backend(
                    &setup,
                    "mlx_chunk_128",
                    Some(128),
                    warmup_iterations,
                    benchmark_iterations,
                )?,
                run_gdn_prefill_backend(
                    &setup,
                    "mlx_chunk_256",
                    Some(256),
                    warmup_iterations,
                    benchmark_iterations,
                )?,
            ],
        },
    };

    let report_json = serde_json::to_string_pretty(&report)?;
    if let Some(output_path) = output {
        std::fs::write(output_path, &report_json).with_context(|| {
            format!("failed to write GDN benchmark to {}", output_path.display())
        })?;
    }

    if json {
        println!("{report_json}");
    } else {
        print_gdn_decode_benchmark_report(&report, output);
    }

    Ok(())
}

fn run_gdn_prefill_backend(
    setup: &GdnPrefillBenchmarkSetup,
    name: &str,
    chunk_size_override: Option<i32>,
    warmup_iterations: usize,
    benchmark_iterations: usize,
) -> anyhow::Result<GdnDecodeBackendResult> {
    if let Some(chunk_size) = chunk_size_override
        && chunk_size > 0
        && setup.seq_len <= chunk_size as usize
    {
        return Ok(GdnDecodeBackendResult {
            name: name.to_string(),
            max_abs_diff_vs_reference: None,
            outcome: KernelBenchmarkOutcome::Skipped {
                reason: format!(
                    "seq_len={} does not exceed forced chunk size {}; chunk path would not engage",
                    setup.seq_len, chunk_size
                ),
            },
        });
    }

    let outcome = benchmark_operation(warmup_iterations, benchmark_iterations, || {
        let (y, state): (Array, Array) = gated_delta_update_with_chunk_size_override(
            &setup.q,
            &setup.k,
            &setup.v,
            &setup.a,
            &setup.b,
            &setup.a_log,
            &setup.dt_bias,
            None,
            None,
            false,
            chunk_size_override,
        )?;
        pmetal_bridge::compat::transforms::eval([&y])?;
        pmetal_bridge::compat::transforms::eval([&state])?;
        Ok(())
    })?;

    let (output, _): (Array, Array) = gated_delta_update_with_chunk_size_override(
        &setup.q,
        &setup.k,
        &setup.v,
        &setup.a,
        &setup.b,
        &setup.a_log,
        &setup.dt_bias,
        None,
        None,
        false,
        chunk_size_override,
    )?;
    pmetal_bridge::compat::transforms::eval([&output])?;

    Ok(GdnDecodeBackendResult {
        name: name.to_string(),
        max_abs_diff_vs_reference: Some(max_abs_diff(
            &setup.reference_output_data,
            output.as_slice::<f32>(),
        )),
        outcome,
    })
}

pub(crate) async fn run_workload_benchmark(
    model_id: &str,
    dataset_id: &str,
    experts_dir: Option<&str>,
    prompt_samples: usize,
    max_prompt_tokens: usize,
    inference_context: WorkloadInferenceContext,
    decode_steps: usize,
    inference_warmup_passes: usize,
    inference_session_repeats: usize,
    inference_repeats: usize,
    train_samples: usize,
    train_steps: usize,
    batch_size: usize,
    max_seq_len: usize,
    output: Option<&Path>,
    json: bool,
) -> anyhow::Result<()> {
    run_workload_benchmark_internal(
        None,
        model_id,
        dataset_id,
        experts_dir,
        prompt_samples,
        max_prompt_tokens,
        inference_context,
        decode_steps,
        inference_warmup_passes,
        inference_session_repeats,
        inference_repeats,
        train_samples,
        train_steps,
        batch_size,
        max_seq_len,
        output,
        json,
    )
    .await
}

pub(crate) async fn run_workload_benchmark_preset(
    preset: WorkloadBenchmarkPreset,
    inference_warmup_passes: usize,
    inference_session_repeats: usize,
    experts_dir: Option<&str>,
    output: Option<&Path>,
    json: bool,
) -> anyhow::Result<()> {
    let config = workload_preset_config(preset);
    run_workload_benchmark_internal(
        Some(preset_label(config.preset).to_string()),
        config.model_id,
        config.dataset_id,
        experts_dir,
        config.prompt_samples,
        config.max_prompt_tokens,
        config.inference_context,
        config.decode_steps,
        inference_warmup_passes,
        inference_session_repeats,
        config.inference_repeats,
        config.train_samples,
        config.train_steps,
        config.batch_size,
        config.max_seq_len,
        output,
        json,
    )
    .await
}

async fn run_workload_benchmark_internal(
    preset: Option<String>,
    model_id: &str,
    dataset_id: &str,
    experts_dir: Option<&str>,
    prompt_samples: usize,
    max_prompt_tokens: usize,
    inference_context: WorkloadInferenceContext,
    decode_steps: usize,
    inference_warmup_passes: usize,
    inference_session_repeats: usize,
    inference_repeats: usize,
    train_samples: usize,
    train_steps: usize,
    batch_size: usize,
    max_seq_len: usize,
    output: Option<&Path>,
    json: bool,
) -> anyhow::Result<()> {
    let device = workload_benchmark_device()?;
    let model_path = resolve_workload_model_path(model_id).await?;
    let dataset_path = pmetal_trainer::resolve_dataset_path(dataset_id).await?;
    let chat_template = pmetal_data::chat_templates::detect_chat_template(&model_path, model_id);
    let text_samples = load_workload_text_samples(&dataset_path, &chat_template)?;

    let selected_inference_samples: Vec<TextSample> = text_samples
        .iter()
        .filter(|sample| !benchmark_prompt_text(sample).is_empty())
        .take(prompt_samples)
        .cloned()
        .collect();
    let selected_training_samples: Vec<TextSample> = text_samples
        .iter()
        .filter(|sample| !sample.text.trim().is_empty())
        .take(train_samples)
        .cloned()
        .collect();

    let inference_prompt_len = resolve_workload_inference_prompt_len(
        &model_path,
        &selected_inference_samples,
        max_prompt_tokens,
        inference_context,
        decode_steps,
    );
    let inference = benchmark_real_inference(
        &model_path,
        experts_dir,
        &selected_inference_samples,
        inference_prompt_len
            .as_ref()
            .map(|selection| selection.effective_max_prompt_tokens)
            .unwrap_or(max_prompt_tokens),
        inference_prompt_len
            .as_ref()
            .map(|selection| {
                resolved_workload_inference_context_from_str(&selection.context_source)
                    .unwrap_or(ResolvedWorkloadInferenceContext::Prompt)
            })
            .unwrap_or(ResolvedWorkloadInferenceContext::Prompt),
        decode_steps,
        inference_warmup_passes,
        inference_session_repeats,
        inference_repeats,
    );
    let training = benchmark_real_training(
        model_id,
        &model_path,
        &selected_training_samples,
        train_steps,
        batch_size,
        max_seq_len,
    )
    .await;
    let (training_seq_len, training) = match training {
        Ok((selection, section)) => (selection, section),
        Err(error) => (
            WorkloadTrainingSeqLenSelection {
                requested_max_seq_len: max_seq_len,
                effective_max_seq_len: max_seq_len,
                max_seq_len_source: if max_seq_len == 0 {
                    "auto-unresolved".to_string()
                } else {
                    "user".to_string()
                },
                sample_median_tokens: 0,
                sample_p95_tokens: 0,
                sample_max_tokens: 0,
                sample_truncated_pct: 0.0,
            },
            WorkloadBenchmarkSection::Failed {
                error: error.to_string(),
            },
        ),
    };
    let inference_prompt_len = match inference_prompt_len {
        Ok(selection) => selection,
        Err(error) => {
            tracing::warn!(
                error = %error,
                requested_max_prompt_tokens = max_prompt_tokens,
                "failed to auto-resolve inference prompt length, falling back to requested limit"
            );
            empty_prompt_len_selection(max_prompt_tokens)
        }
    };

    let report = WorkloadBenchmarkReport {
        version: env!("CARGO_PKG_VERSION").to_string(),
        generated_at_unix_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis(),
        device,
        workload: WorkloadBenchmarkConfig {
            preset,
            model_id: model_id.to_string(),
            dataset_id: dataset_id.to_string(),
            experts_dir: experts_dir.map(ToOwned::to_owned),
            resolved_model_path: model_path.display().to_string(),
            resolved_dataset_path: dataset_path.display().to_string(),
            prompt_samples,
            max_prompt_tokens: inference_prompt_len.effective_max_prompt_tokens,
            inference_prompt_len,
            decode_steps,
            inference_warmup_passes,
            inference_session_repeats,
            inference_repeats,
            train_samples,
            train_steps,
            batch_size,
            max_seq_len: training_seq_len.effective_max_seq_len,
            training_seq_len,
        },
        inference: inference.unwrap_or_else(|error| WorkloadBenchmarkSection::Failed {
            error: error.to_string(),
        }),
        training,
    };

    let report_json = serde_json::to_string_pretty(&report)?;
    if let Some(output_path) = output {
        std::fs::write(output_path, &report_json).with_context(|| {
            format!(
                "failed to write workload benchmark to {}",
                output_path.display()
            )
        })?;
    }

    if json {
        println!("{report_json}");
    } else {
        print_workload_benchmark_report(&report, output);
    }

    Ok(())
}

fn workload_benchmark_device() -> anyhow::Result<KernelBenchmarkDevice> {
    let ctx = MetalContext::new().context("failed to initialize Metal context")?;
    let props = ctx.properties();
    Ok(KernelBenchmarkDevice {
        name: props.name.clone(),
        tier: device_tier_label(props.device_tier).to_string(),
        architecture_gen: props.architecture_gen,
        gpu_core_count: props.gpu_core_count,
        ane_core_count: props.ane_core_count,
        has_nax: props.has_nax(),
        is_apple10_or_newer: props.is_apple10_or_newer(),
        is_ultra_fusion: props.is_ultra_fusion,
        memory_bandwidth_gbps: props.memory_bandwidth_gbps,
        memory_bandwidth_source: memory_bandwidth_source_label(props.memory_bandwidth_source)
            .to_string(),
    })
}

fn build_gdn_decode_benchmark_setup(
    model_path: &Path,
    stage: GdnBenchmarkStage,
    layer: Option<usize>,
    batch_size: usize,
    seq_len: usize,
) -> anyhow::Result<GdnBenchmarkSetup> {
    let model = DynamicModel::load_with_options(
        model_path,
        DynamicModelLoadOptions {
            prefer_expert_offload: true,
        },
    )?;
    let DynamicModel::Qwen3Next(model) = model else {
        anyhow::bail!("GDN benchmark only supports qwen3_next / qwen3_5 model families");
    };

    let layer_idx = resolve_qwen3_next_gdn_layer_index(&model, layer)?;
    let gdn = model.model.layers[layer_idx]
        .linear_attn
        .as_ref()
        .context("selected layer does not contain a GDN block")?;
    match stage {
        GdnBenchmarkStage::InputProj => {
            let input_dim = gdn.hidden_size as usize;
            let conv_dim = gdn.conv_dim as usize;
            let value_dim = gdn.value_dim as usize;
            let num_v_heads = gdn.num_v_heads as usize;
            let head_v_dim = gdn.head_v_dim as usize;
            let output_dim = conv_dim + value_dim + (num_v_heads * 2);
            let input_data: Vec<f32> = (0..batch_size * seq_len * input_dim)
                .map(deterministic_value)
                .collect();
            let input = Array::from_slice(
                &input_data,
                &[batch_size as i32, seq_len as i32, input_dim as i32],
            );

            let qkv_weight = gdn.in_proj_qkv.weight.as_ref().as_type::<f32>();
            let z_weight = gdn.in_proj_z.weight.as_ref().as_type::<f32>();
            let b_weight = gdn.in_proj_b.weight.as_ref().as_type::<f32>();
            let a_weight = gdn.in_proj_a.weight.as_ref().as_type::<f32>();
            let qkv_weight_t = qkv_weight.t();
            let z_weight_t = z_weight.t();
            let b_weight_t = b_weight.t();
            let a_weight_t = a_weight.t();
            pmetal_bridge::compat::transforms::eval([&qkv_weight])?;
            pmetal_bridge::compat::transforms::eval([&z_weight])?;
            pmetal_bridge::compat::transforms::eval([&b_weight])?;
            pmetal_bridge::compat::transforms::eval([&a_weight])?;
            pmetal_bridge::compat::transforms::eval([&qkv_weight_t])?;
            pmetal_bridge::compat::transforms::eval([&z_weight_t])?;
            pmetal_bridge::compat::transforms::eval([&b_weight_t])?;
            pmetal_bridge::compat::transforms::eval([&a_weight_t])?;

            let combined_weight = build_combined_input_projection_weight(gdn)?;
            let qkv_z_combined_weight = build_qkv_z_combined_projection_weight(gdn)?;
            let combined_weight_t = combined_weight.t();
            let qkv_z_combined_weight_t = qkv_z_combined_weight.t();
            pmetal_bridge::compat::transforms::eval([&combined_weight])?;
            pmetal_bridge::compat::transforms::eval([&qkv_z_combined_weight])?;
            pmetal_bridge::compat::transforms::eval([&combined_weight_t])?;
            pmetal_bridge::compat::transforms::eval([&qkv_z_combined_weight_t])?;
            let combined_weight_data = combined_weight.as_slice::<f32>().to_vec();
            let reference_output =
                mlx_split_projection(&input, &qkv_weight, &z_weight, &b_weight, &a_weight)?;
            pmetal_bridge::compat::transforms::eval([&reference_output])?;

            Ok(GdnBenchmarkSetup::InputProj(
                GdnInputProjectionBenchmarkSetup {
                    model_path: model_path.to_path_buf(),
                    layer_idx,
                    batch_size,
                    seq_len,
                    input_dim,
                    output_dim,
                    conv_dim,
                    value_dim,
                    num_v_heads,
                    head_v_dim,
                    input,
                    input_data,
                    qkv_weight,
                    qkv_weight_t,
                    z_weight,
                    z_weight_t,
                    b_weight,
                    b_weight_t,
                    a_weight,
                    a_weight_t,
                    qkv_z_combined_weight,
                    qkv_z_combined_weight_t,
                    combined_weight,
                    combined_weight_t,
                    combined_weight_data,
                    reference_output_data: reference_output.as_slice::<f32>().to_vec(),
                },
            ))
        }
        GdnBenchmarkStage::OutProj => {
            let input_dim = gdn.value_dim as usize;
            let output_dim = gdn.hidden_size as usize;
            let input_data: Vec<f32> = (0..batch_size * seq_len * input_dim)
                .map(deterministic_value)
                .collect();
            let input = Array::from_slice(
                &input_data,
                &[batch_size as i32, seq_len as i32, input_dim as i32],
            );
            let weight = gdn.out_proj.weight.as_ref().as_type::<f32>();
            let weight_t = weight.t();
            pmetal_bridge::compat::transforms::eval([&weight])?;
            pmetal_bridge::compat::transforms::eval([&weight_t])?;
            let weight_data = weight.as_slice::<f32>().to_vec();
            let reference_output = mlx_linear_projection(&input, &weight)?;
            pmetal_bridge::compat::transforms::eval([&reference_output])?;

            Ok(GdnBenchmarkSetup::OutProj(
                GdnLinearProjectionBenchmarkSetup {
                    model_path: model_path.to_path_buf(),
                    layer_idx,
                    batch_size,
                    seq_len,
                    input_dim,
                    output_dim,
                    input,
                    input_data,
                    weight,
                    weight_t,
                    weight_data,
                    reference_output_data: reference_output.as_slice::<f32>().to_vec(),
                },
            ))
        }
        GdnBenchmarkStage::Prefill => {
            let input_dim = gdn.key_dim as usize;
            let output_dim = gdn.value_dim as usize;
            let q_data: Vec<f32> =
                (0..batch_size * seq_len * gdn.num_k_heads as usize * gdn.head_k_dim as usize)
                    .map(deterministic_value)
                    .collect();
            let k_data: Vec<f32> =
                (0..batch_size * seq_len * gdn.num_k_heads as usize * gdn.head_k_dim as usize)
                    .map(|i| deterministic_value(i + 17))
                    .collect();
            let v_data: Vec<f32> =
                (0..batch_size * seq_len * gdn.num_v_heads as usize * gdn.head_v_dim as usize)
                    .map(|i| deterministic_value(i + 29))
                    .collect();
            let a_data: Vec<f32> = (0..batch_size * seq_len * gdn.num_v_heads as usize)
                .map(|i| deterministic_value(i + 43))
                .collect();
            let b_data: Vec<f32> = (0..batch_size * seq_len * gdn.num_v_heads as usize)
                .map(|i| deterministic_value(i + 61))
                .collect();

            let q_raw: Array = Array::from_slice(
                &q_data,
                &[
                    batch_size as i32,
                    seq_len as i32,
                    gdn.num_k_heads,
                    gdn.head_k_dim,
                ],
            );
            let k_raw: Array = Array::from_slice(
                &k_data,
                &[
                    batch_size as i32,
                    seq_len as i32,
                    gdn.num_k_heads,
                    gdn.head_k_dim,
                ],
            );
            let v: Array = Array::from_slice(
                &v_data,
                &[
                    batch_size as i32,
                    seq_len as i32,
                    gdn.num_v_heads,
                    gdn.head_v_dim,
                ],
            );
            let a: Array = Array::from_slice(
                &a_data,
                &[batch_size as i32, seq_len as i32, gdn.num_v_heads],
            );
            let b: Array = Array::from_slice(
                &b_data,
                &[batch_size as i32, seq_len as i32, gdn.num_v_heads],
            );
            let q = l2norm_last_dim(&q_raw, 1e-6)?
                .multiply(&Array::from_f32((gdn.head_k_dim as f32).sqrt().recip()));
            let k = l2norm_last_dim(&k_raw, 1e-6)?;
            let a_log = gdn.a_log.as_ref().as_type::<f32>();
            let dt_bias = gdn.dt_bias.as_ref().as_type::<f32>();
            pmetal_bridge::compat::transforms::eval([&q])?;
            pmetal_bridge::compat::transforms::eval([&k])?;
            pmetal_bridge::compat::transforms::eval([&v])?;
            pmetal_bridge::compat::transforms::eval([&a])?;
            pmetal_bridge::compat::transforms::eval([&b])?;
            pmetal_bridge::compat::transforms::eval([&a_log])?;
            pmetal_bridge::compat::transforms::eval([&dt_bias])?;
            let (reference_output, _): (Array, Array) =
                gated_delta_update_with_chunk_size_override(
                    &q,
                    &k,
                    &v,
                    &a,
                    &b,
                    &a_log,
                    &dt_bias,
                    None,
                    None,
                    false,
                    Some(0),
                )?;
            pmetal_bridge::compat::transforms::eval([&reference_output])?;

            Ok(GdnBenchmarkSetup::Prefill(GdnPrefillBenchmarkSetup {
                model_path: model_path.to_path_buf(),
                layer_idx,
                batch_size,
                seq_len,
                input_dim,
                output_dim,
                q,
                k,
                v,
                a,
                b,
                a_log,
                dt_bias,
                reference_output_data: reference_output.as_slice::<f32>().to_vec(),
            }))
        }
    }
}

fn resolve_qwen3_next_gdn_layer_index(
    model: &pmetal_models::architectures::qwen3_next::Qwen3NextForCausalLM,
    requested_layer: Option<usize>,
) -> anyhow::Result<usize> {
    if let Some(layer_idx) = requested_layer {
        let layer = model
            .model
            .layers
            .get(layer_idx)
            .with_context(|| format!("layer {layer_idx} is out of range"))?;
        anyhow::ensure!(
            layer.linear_attn.is_some(),
            "layer {layer_idx} is not a GDN linear-attention layer"
        );
        return Ok(layer_idx);
    }

    model
        .model
        .layers
        .iter()
        .position(|layer| layer.linear_attn.is_some())
        .context("model does not contain any GDN linear-attention layers")
}

fn slice_last_dim_range(a: &Array, start: i32, end: i32) -> Array {
    let ndim = a.ndim() as usize;
    let mut starts = vec![0; ndim];
    let mut stops: Vec<i32> = a.shape().to_vec();
    starts[ndim - 1] = start;
    stops[ndim - 1] = end;
    a.slice(&starts, &stops)
}

fn slice_last_dim_to(a: &Array, end: i32) -> Array {
    slice_last_dim_range(a, 0, end)
}

fn slice_last_dim_from(a: &Array, start: i32) -> Array {
    let end = a.shape()[a.ndim() as usize - 1];
    slice_last_dim_range(a, start, end)
}

fn build_combined_input_projection_weight(gdn: &Qwen3NextGatedDeltaNet) -> anyhow::Result<Array> {
    let weights = [
        gdn.in_proj_qkv.weight.as_ref().as_type::<f32>(),
        gdn.in_proj_z.weight.as_ref().as_type::<f32>(),
        gdn.in_proj_b.weight.as_ref().as_type::<f32>(),
        gdn.in_proj_a.weight.as_ref().as_type::<f32>(),
    ];
    let refs: Vec<&Array> = weights.iter().collect();
    Ok(ops::concatenate_axis(&refs, 0))
}

fn build_qkv_z_combined_projection_weight(gdn: &Qwen3NextGatedDeltaNet) -> anyhow::Result<Array> {
    let weights = [
        gdn.in_proj_qkv.weight.as_ref().as_type::<f32>(),
        gdn.in_proj_z.weight.as_ref().as_type::<f32>(),
    ];
    let refs: Vec<&Array> = weights.iter().collect();
    Ok(ops::concatenate_axis(&refs, 0))
}

fn mlx_linear_projection(input: &Array, weight: &Array) -> anyhow::Result<Array> {
    Ok(ops::matmul(input, &weight.t()))
}

fn mlx_linear_projection_rhs_transposed(input: &Array, weight_t: &Array) -> anyhow::Result<Array> {
    Ok(ops::matmul(input, weight_t))
}

fn mlx_split_projection(
    input: &Array,
    qkv_weight: &Array,
    z_weight: &Array,
    b_weight: &Array,
    a_weight: &Array,
) -> anyhow::Result<Array> {
    let qkv = mlx_linear_projection(input, qkv_weight)?;
    let z = mlx_linear_projection(input, z_weight)?;
    let b_val = mlx_linear_projection(input, b_weight)?;
    let a = mlx_linear_projection(input, a_weight)?;
    Ok(ops::concatenate_axis(&[&qkv, &z, &b_val, &a], -1))
}

fn mlx_split_projection_rhs_transposed(
    input: &Array,
    qkv_weight_t: &Array,
    z_weight_t: &Array,
    b_weight_t: &Array,
    a_weight_t: &Array,
) -> anyhow::Result<Array> {
    let qkv = mlx_linear_projection_rhs_transposed(input, qkv_weight_t)?;
    let z = mlx_linear_projection_rhs_transposed(input, z_weight_t)?;
    let b_val = mlx_linear_projection_rhs_transposed(input, b_weight_t)?;
    let a = mlx_linear_projection_rhs_transposed(input, a_weight_t)?;
    Ok(ops::concatenate_axis(&[&qkv, &z, &b_val, &a], -1))
}

fn mlx_combined_split_projection(
    input: &Array,
    combined_weight: &Array,
    batch_size: usize,
    seq_len: usize,
    conv_dim: usize,
    value_dim: usize,
    num_v_heads: usize,
    head_v_dim: usize,
) -> anyhow::Result<(Array, Array, Array, Array)> {
    let projected = mlx_linear_projection(input, combined_weight)?;
    split_combined_projection(
        &projected,
        batch_size,
        seq_len,
        conv_dim,
        value_dim,
        num_v_heads,
        head_v_dim,
    )
}

fn mlx_combined_split_projection_rhs_transposed(
    input: &Array,
    combined_weight_t: &Array,
    batch_size: usize,
    seq_len: usize,
    conv_dim: usize,
    value_dim: usize,
    num_v_heads: usize,
    head_v_dim: usize,
) -> anyhow::Result<(Array, Array, Array, Array)> {
    let projected = mlx_linear_projection_rhs_transposed(input, combined_weight_t)?;
    split_combined_projection(
        &projected,
        batch_size,
        seq_len,
        conv_dim,
        value_dim,
        num_v_heads,
        head_v_dim,
    )
}

fn mlx_qkv_z_combined_split_projection(
    input: &Array,
    qkv_z_combined_weight: &Array,
    b_weight: &Array,
    a_weight: &Array,
    batch_size: usize,
    seq_len: usize,
    conv_dim: usize,
    value_dim: usize,
    num_v_heads: usize,
    head_v_dim: usize,
) -> anyhow::Result<(Array, Array, Array, Array)> {
    let projected = mlx_linear_projection(input, qkv_z_combined_weight)?;
    let qkv_end = conv_dim as i32;
    let z_end = qkv_end + value_dim as i32;
    let qkv = slice_last_dim_to(&projected, qkv_end);
    let z = slice_last_dim_range(&projected, qkv_end, z_end).reshape(&[
        batch_size as i32,
        seq_len as i32,
        num_v_heads as i32,
        head_v_dim as i32,
    ]);
    let b_val = mlx_linear_projection(input, b_weight)?;
    let a = mlx_linear_projection(input, a_weight)?;
    Ok((qkv, z, b_val, a))
}

fn mlx_qkv_z_combined_split_projection_rhs_transposed(
    input: &Array,
    qkv_z_combined_weight_t: &Array,
    b_weight_t: &Array,
    a_weight_t: &Array,
    batch_size: usize,
    seq_len: usize,
    conv_dim: usize,
    value_dim: usize,
    num_v_heads: usize,
    head_v_dim: usize,
) -> anyhow::Result<(Array, Array, Array, Array)> {
    let projected = mlx_linear_projection_rhs_transposed(input, qkv_z_combined_weight_t)?;
    let qkv_end = conv_dim as i32;
    let z_end = qkv_end + value_dim as i32;
    let qkv = slice_last_dim_to(&projected, qkv_end);
    let z = slice_last_dim_range(&projected, qkv_end, z_end).reshape(&[
        batch_size as i32,
        seq_len as i32,
        num_v_heads as i32,
        head_v_dim as i32,
    ]);
    let b_val = mlx_linear_projection_rhs_transposed(input, b_weight_t)?;
    let a = mlx_linear_projection_rhs_transposed(input, a_weight_t)?;
    Ok((qkv, z, b_val, a))
}

fn l2norm_last_dim(x: &Array, eps: f32) -> anyhow::Result<Array> {
    let norm = x
        .square()
        .sum_axis(-1, true)
        .add(&Array::from_f32(eps))
        .sqrt();
    Ok(x.divide(&norm))
}

fn accelerate_combined_projection(
    input: &[f32],
    combined_weight: &[f32],
    rows: usize,
    hidden_size: usize,
    total_output_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; rows * total_output_dim];
    pmetal_metal::accelerate::gemm(
        input,
        combined_weight,
        &mut output,
        rows,
        total_output_dim,
        hidden_size,
        1.0,
        0.0,
        false,
        true,
    );
    output
}

fn accelerate_roundtrip_split_projection(
    input: &Array,
    combined_weight: &[f32],
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
    total_output_dim: usize,
    conv_dim: usize,
    value_dim: usize,
    num_v_heads: usize,
    head_v_dim: usize,
) -> anyhow::Result<(Array, Array, Array, Array)> {
    let input_cpu = input.as_slice::<f32>().to_vec();
    let projected = accelerate_combined_projection(
        &input_cpu,
        combined_weight,
        batch_size * seq_len,
        hidden_size,
        total_output_dim,
    );
    let projected = Array::from_slice(
        &projected,
        &[batch_size as i32, seq_len as i32, total_output_dim as i32],
    );
    split_combined_projection(
        &projected,
        batch_size,
        seq_len,
        conv_dim,
        value_dim,
        num_v_heads,
        head_v_dim,
    )
}

fn accelerate_roundtrip_linear_projection(
    input: &Array,
    weight: &[f32],
    batch_size: usize,
    seq_len: usize,
    input_dim: usize,
    output_dim: usize,
) -> anyhow::Result<Array> {
    let input_cpu = input.as_slice::<f32>().to_vec();
    let projected = accelerate_combined_projection(
        &input_cpu,
        weight,
        batch_size * seq_len,
        input_dim,
        output_dim,
    );
    Ok(Array::from_slice(
        &projected,
        &[batch_size as i32, seq_len as i32, output_dim as i32],
    ))
}

fn split_combined_projection(
    projected: &Array,
    batch_size: usize,
    seq_len: usize,
    conv_dim: usize,
    value_dim: usize,
    num_v_heads: usize,
    head_v_dim: usize,
) -> anyhow::Result<(Array, Array, Array, Array)> {
    let qkv_end = conv_dim as i32;
    let z_end = qkv_end + value_dim as i32;
    let b_end = z_end + num_v_heads as i32;
    let qkv = slice_last_dim_to(projected, qkv_end);
    let z = slice_last_dim_range(projected, qkv_end, z_end).reshape(&[
        batch_size as i32,
        seq_len as i32,
        num_v_heads as i32,
        head_v_dim as i32,
    ]);
    let b_val = slice_last_dim_range(projected, z_end, b_end);
    let a = slice_last_dim_from(projected, b_end);
    Ok((qkv, z, b_val, a))
}

async fn resolve_workload_model_path(model_id: &str) -> anyhow::Result<PathBuf> {
    if model_id.contains('/') && !Path::new(model_id).exists() {
        Ok(pmetal_hub::download_model(model_id, None, None).await?)
    } else {
        Ok(PathBuf::from(model_id))
    }
}

fn load_workload_text_samples(
    dataset_path: &Path,
    chat_template: &pmetal_data::chat_templates::ChatTemplate,
) -> anyhow::Result<Vec<TextSample>> {
    if dataset_path.extension().is_some_and(|ext| ext == "parquet") {
        if let Ok(samples) = TrainingDataset::load_parquet_text(dataset_path, "text", None) {
            return Ok(samples);
        }
        if let Ok(samples) = TrainingDataset::load_parquet_text(dataset_path, "content", None) {
            return Ok(samples);
        }

        let tmp_jsonl = write_temp_jsonl_from_parquet(dataset_path)?;
        return Ok(TrainingDataset::load_jsonl_text(
            &tmp_jsonl,
            DatasetFormat::Auto,
            Some(chat_template),
        )?);
    }

    Ok(TrainingDataset::load_jsonl_text(
        dataset_path,
        DatasetFormat::Auto,
        Some(chat_template),
    )?)
}

fn write_temp_jsonl_from_parquet(dataset_path: &Path) -> anyhow::Result<PathBuf> {
    let tmp_dir = std::env::temp_dir().join("pmetal-bench-workload");
    std::fs::create_dir_all(&tmp_dir)?;
    let tmp_jsonl = tmp_dir.join(format!(
        "dataset_{}_{}.jsonl",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    ));

    let file = std::fs::File::open(dataset_path)?;
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let mut out = std::io::BufWriter::new(std::fs::File::create(&tmp_jsonl)?);
    let mut buf = Vec::new();

    for batch in reader {
        let batch = batch?;
        buf.clear();
        let mut json_writer = arrow_json::writer::LineDelimitedWriter::new(&mut buf);
        json_writer.write(&batch)?;
        json_writer.finish()?;
        out.write_all(&buf)?;
    }
    out.flush()?;

    Ok(tmp_jsonl)
}

fn benchmark_prompt_text(sample: &TextSample) -> &str {
    sample
        .prompt
        .as_deref()
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
        .unwrap_or_else(|| sample.text.trim())
}

fn benchmark_full_text(sample: &TextSample) -> &str {
    sample.text.trim()
}

fn benchmark_inference_text(
    sample: &TextSample,
    inference_context: ResolvedWorkloadInferenceContext,
) -> &str {
    match inference_context {
        ResolvedWorkloadInferenceContext::Prompt => benchmark_prompt_text(sample),
        ResolvedWorkloadInferenceContext::TextPrefix => benchmark_full_text(sample),
    }
}

struct InferenceSessionMetrics {
    prompt_samples: usize,
    measurement_passes: usize,
    warmup_passes: usize,
    prompt_tokens: usize,
    max_prompt_tokens: usize,
    decode_steps: usize,
    inference_repeats: usize,
    cache_mode: String,
    cache_mode_source: String,
    first_generated_token_id: Option<u32>,
    first_generated_token_text: Option<String>,
    total_prefill_ms: f64,
    total_decode_ms: f64,
    prefill_tok_sec_samples: Vec<f64>,
    decode_tok_sec_samples: Vec<f64>,
    decode_ms_per_token_samples: Vec<f64>,
    prefetch_hits: Option<usize>,
    prefetch_misses: Option<usize>,
    prefetch_total: Option<usize>,
}

fn benchmark_real_inference(
    model_path: &Path,
    experts_dir: Option<&str>,
    samples: &[TextSample],
    max_prompt_tokens: usize,
    inference_context: ResolvedWorkloadInferenceContext,
    decode_steps: usize,
    inference_warmup_passes: usize,
    inference_session_repeats: usize,
    inference_repeats: usize,
) -> anyhow::Result<WorkloadBenchmarkSection<InferenceWorkloadMetrics>> {
    if samples.is_empty() {
        return Ok(WorkloadBenchmarkSection::Skipped {
            reason: "no non-empty prompt samples available".to_string(),
        });
    }

    let session_runs = inference_session_repeats.max(1);
    let mut sessions = Vec::with_capacity(session_runs);
    for _ in 0..session_runs {
        sessions.push(benchmark_real_inference_session(
            model_path,
            experts_dir,
            samples,
            max_prompt_tokens,
            inference_context,
            decode_steps,
            inference_warmup_passes,
            inference_repeats,
        )?);
    }

    let prompt_samples = sessions[0].prompt_samples;
    let max_prompt_tokens = sessions[0].max_prompt_tokens;
    let decode_steps = sessions[0].decode_steps;
    let inference_repeats = sessions[0].inference_repeats;
    let cache_mode = sessions[0].cache_mode.clone();
    let cache_mode_source = sessions[0].cache_mode_source.clone();
    let first_generated_token_id = sessions[0].first_generated_token_id;
    let first_generated_token_text = sessions[0].first_generated_token_text.clone();

    let mut total_prompt_tokens = 0usize;
    let mut measurement_passes = 0usize;
    let mut warmup_passes = 0usize;
    let mut total_prefill_ms = 0.0;
    let mut total_decode_ms = 0.0;
    let mut prefill_tok_sec_samples = Vec::new();
    let mut decode_tok_sec_samples = Vec::new();
    let mut decode_ms_per_token_samples = Vec::new();
    let mut session_prefill_tok_per_sec = Vec::with_capacity(session_runs);
    let mut session_decode_tok_per_sec = Vec::with_capacity(session_runs);
    let mut session_decode_ms_per_token = Vec::with_capacity(session_runs);
    let mut prefetch_hits = 0usize;
    let mut prefetch_misses = 0usize;
    let mut prefetch_total = 0usize;
    let mut saw_prefetch_stats = true;

    for session in sessions {
        total_prompt_tokens += session.prompt_tokens;
        measurement_passes += session.measurement_passes;
        warmup_passes += session.warmup_passes;
        total_prefill_ms += session.total_prefill_ms;
        total_decode_ms += session.total_decode_ms;
        prefill_tok_sec_samples.extend(session.prefill_tok_sec_samples.iter().copied());
        decode_tok_sec_samples.extend(session.decode_tok_sec_samples.iter().copied());
        decode_ms_per_token_samples.extend(session.decode_ms_per_token_samples.iter().copied());

        let session_decode_tokens = session.measurement_passes * session.decode_steps;
        let session_prefill_rate = if session.total_prefill_ms > 0.0 {
            session.prompt_tokens as f64 / (session.total_prefill_ms / 1000.0)
        } else {
            0.0
        };
        let session_decode_rate = if session_decode_tokens > 0 && session.total_decode_ms > 0.0 {
            session_decode_tokens as f64 / (session.total_decode_ms / 1000.0)
        } else {
            0.0
        };
        let session_decode_ms = if session_decode_tokens > 0 {
            session.total_decode_ms / session_decode_tokens as f64
        } else {
            0.0
        };
        session_prefill_tok_per_sec.push(session_prefill_rate);
        session_decode_tok_per_sec.push(session_decode_rate);
        session_decode_ms_per_token.push(session_decode_ms);

        match (
            session.prefetch_hits,
            session.prefetch_misses,
            session.prefetch_total,
        ) {
            (Some(hits), Some(misses), Some(total)) => {
                prefetch_hits += hits;
                prefetch_misses += misses;
                prefetch_total += total;
            }
            _ => saw_prefetch_stats = false,
        }
    }

    if prompt_samples == 0 || total_prompt_tokens == 0 {
        return Ok(WorkloadBenchmarkSection::Skipped {
            reason: "all selected prompt samples tokenized to empty inputs".to_string(),
        });
    }

    let decode_tokens = measurement_passes * decode_steps;
    let mean_prefill_tok_per_sec = mean(&prefill_tok_sec_samples);
    let mean_decode_tok_per_sec = mean(&decode_tok_sec_samples);
    let mean_decode_ms_per_token = mean(&decode_ms_per_token_samples);
    let median_prefill_tok_per_sec = median(&mut prefill_tok_sec_samples);
    let median_decode_tok_per_sec = median(&mut decode_tok_sec_samples);
    let median_decode_ms_per_token = median(&mut decode_ms_per_token_samples);
    let mean_session_prefill_tok_per_sec = mean(&session_prefill_tok_per_sec);
    let mean_session_decode_tok_per_sec = mean(&session_decode_tok_per_sec);
    let mean_session_decode_ms_per_token = mean(&session_decode_ms_per_token);
    let median_session_prefill_tok_per_sec = median(&mut session_prefill_tok_per_sec);
    let median_session_decode_tok_per_sec = median(&mut session_decode_tok_per_sec);
    let median_session_decode_ms_per_token = median(&mut session_decode_ms_per_token);

    Ok(WorkloadBenchmarkSection::Completed(
        InferenceWorkloadMetrics {
            session_runs,
            prompt_samples,
            measurement_passes,
            warmup_passes,
            prompt_tokens: total_prompt_tokens,
            max_prompt_tokens,
            decode_steps,
            inference_repeats,
            cache_mode,
            cache_mode_source,
            first_generated_token_id,
            first_generated_token_text,
            total_prefill_ms,
            prefill_tok_per_sec: if total_prefill_ms > 0.0 {
                total_prompt_tokens as f64 / (total_prefill_ms / 1000.0)
            } else {
                0.0
            },
            mean_prefill_tok_per_sec,
            median_prefill_tok_per_sec,
            total_decode_ms,
            decode_tok_per_sec: if decode_tokens > 0 && total_decode_ms > 0.0 {
                decode_tokens as f64 / (total_decode_ms / 1000.0)
            } else {
                0.0
            },
            decode_ms_per_token: if decode_tokens > 0 {
                total_decode_ms / decode_tokens as f64
            } else {
                0.0
            },
            mean_decode_tok_per_sec,
            median_decode_tok_per_sec,
            mean_decode_ms_per_token,
            median_decode_ms_per_token,
            mean_session_prefill_tok_per_sec,
            median_session_prefill_tok_per_sec,
            mean_session_decode_tok_per_sec,
            median_session_decode_tok_per_sec,
            mean_session_decode_ms_per_token,
            median_session_decode_ms_per_token,
            session_prefill_tok_per_sec,
            session_decode_tok_per_sec,
            session_decode_ms_per_token,
            prefetch_hits: saw_prefetch_stats.then_some(prefetch_hits),
            prefetch_misses: saw_prefetch_stats.then_some(prefetch_misses),
            prefetch_total: saw_prefetch_stats.then_some(prefetch_total),
            prefetch_hit_rate: if saw_prefetch_stats && prefetch_total > 0 {
                Some(prefetch_hits as f64 / prefetch_total as f64)
            } else {
                None
            },
        },
    ))
}

fn benchmark_real_inference_session(
    model_path: &Path,
    experts_dir: Option<&str>,
    samples: &[TextSample],
    max_prompt_tokens: usize,
    inference_context: ResolvedWorkloadInferenceContext,
    decode_steps: usize,
    inference_warmup_passes: usize,
    inference_repeats: usize,
) -> anyhow::Result<InferenceSessionMetrics> {
    let tokenizer = Tokenizer::from_model_dir(model_path)?;
    let mut model = DynamicModel::load_with_options(
        model_path,
        pmetal_models::dispatcher::DynamicModelLoadOptions {
            prefer_expert_offload: experts_dir.is_some(),
        },
    )?;
    if let Some(experts_dir) = experts_dir {
        model.enable_expert_offloading(Path::new(experts_dir))?;
    } else if model.requires_expert_offloading() {
        anyhow::bail!(
            "this model requires expert offloading; repack routed experts with `pmetal pack-experts` and rerun bench-workload with --experts-dir <packed_dir>"
        );
    }
    let cache_max_seq_len = max_prompt_tokens
        .saturating_add(decode_steps)
        .saturating_add(8);
    let base_cache_config = model.create_cache(cache_max_seq_len).config().clone();
    let cache_selection = select_cache_mode_for_model(
        &base_cache_config,
        model_path,
        model.num_parameters(),
        CacheModeRequest {
            kv_quant: None,
            kv_k_bits: None,
            kv_v_bits: None,
            kv_group_size: 64,
            kv_turboquant: false,
            kv_turboquant_preset: None,
            no_kv_quant: false,
            fp8: false,
        },
    );
    let cache_mode = cache_selection.mode;

    for sample in samples {
        let warmup_ids =
            encode_benchmark_prompt(&tokenizer, sample, max_prompt_tokens, inference_context)?;
        if warmup_ids.is_empty() {
            continue;
        }
        for _ in 0..inference_warmup_passes {
            let _ = run_prefill_decode_pass(&mut model, &warmup_ids, decode_steps, cache_mode)?;
        }
    }
    model.reset_prefetch_stats();

    let inference_repeats = inference_repeats.max(1);
    let mut total_prompt_tokens = 0usize;
    let mut total_prefill = Duration::default();
    let mut total_decode = Duration::default();
    let mut measured_samples = 0usize;
    let mut measurement_passes = 0usize;
    let mut warmup_passes = 0usize;
    let mut first_generated_token_id = None;
    let mut first_generated_token_text = None;
    let mut prefill_tok_sec = Vec::new();
    let mut decode_tok_sec = Vec::new();
    let mut decode_ms_per_token = Vec::new();

    for sample in samples {
        let prompt_ids =
            encode_benchmark_prompt(&tokenizer, sample, max_prompt_tokens, inference_context)?;
        if prompt_ids.is_empty() {
            continue;
        }

        measured_samples += 1;
        warmup_passes += inference_warmup_passes;
        for _ in 0..inference_repeats {
            let pass = run_prefill_decode_pass(&mut model, &prompt_ids, decode_steps, cache_mode)?;
            let prompt_tokens = prompt_ids.len();
            let prefill_ms = duration_to_ms(pass.prefill_elapsed);
            let decode_ms = duration_to_ms(pass.decode_elapsed);

            total_prompt_tokens += prompt_tokens;
            total_prefill += pass.prefill_elapsed;
            total_decode += pass.decode_elapsed;
            measurement_passes += 1;
            if first_generated_token_id.is_none() {
                first_generated_token_id = Some(pass.first_generated_token_id);
                first_generated_token_text =
                    tokenizer.decode(&[pass.first_generated_token_id]).ok();
            }

            if prompt_tokens > 0 && prefill_ms > 0.0 {
                prefill_tok_sec.push(prompt_tokens as f64 / (prefill_ms / 1000.0));
            }
            if decode_steps > 0 && decode_ms > 0.0 {
                decode_tok_sec.push(decode_steps as f64 / (decode_ms / 1000.0));
                decode_ms_per_token.push(decode_ms / decode_steps as f64);
            }
        }
    }

    if measured_samples == 0 || total_prompt_tokens == 0 {
        anyhow::bail!("all selected prompt samples tokenized to empty inputs");
    }

    let prefetch_stats = model.prefetch_stats();

    Ok(InferenceSessionMetrics {
        prompt_samples: measured_samples,
        measurement_passes,
        warmup_passes,
        prompt_tokens: total_prompt_tokens,
        max_prompt_tokens,
        decode_steps,
        inference_repeats,
        cache_mode: cache_mode.describe(),
        cache_mode_source: cache_selection.source.as_str().to_string(),
        first_generated_token_id,
        first_generated_token_text,
        total_prefill_ms: duration_to_ms(total_prefill),
        total_decode_ms: duration_to_ms(total_decode),
        prefill_tok_sec_samples: prefill_tok_sec,
        decode_tok_sec_samples: decode_tok_sec,
        decode_ms_per_token_samples: decode_ms_per_token,
        prefetch_hits: prefetch_stats.as_ref().map(|stats| stats.hits),
        prefetch_misses: prefetch_stats.as_ref().map(|stats| stats.misses),
        prefetch_total: prefetch_stats.as_ref().map(|stats| stats.total),
    })
}

struct PrefillDecodePassResult {
    prefill_elapsed: Duration,
    decode_elapsed: Duration,
    first_generated_token_id: u32,
}

fn encode_benchmark_prompt(
    tokenizer: &Tokenizer,
    sample: &TextSample,
    max_prompt_tokens: usize,
    inference_context: ResolvedWorkloadInferenceContext,
) -> anyhow::Result<Vec<u32>> {
    let mut ids = tokenizer
        .encode_with_special_tokens(benchmark_inference_text(sample, inference_context))?;
    if max_prompt_tokens > 0 && ids.len() > max_prompt_tokens {
        ids.truncate(max_prompt_tokens);
    }
    Ok(ids)
}

fn run_prefill_decode_pass(
    model: &mut DynamicModel,
    prompt_ids: &[u32],
    decode_steps: usize,
    cache_mode: CacheMode,
) -> anyhow::Result<PrefillDecodePassResult> {
    use pmetal_bridge::compat::Array;
    let prompt_tokens: Vec<i32> = prompt_ids.iter().map(|&id| id as i32).collect();
    let prompt = Array::from_slice(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let mut cache = model.create_cache_with_mode(
        prompt_ids
            .len()
            .saturating_add(decode_steps)
            .saturating_add(8),
        cache_mode,
    );
    let mut mamba_cache = model.create_mamba_cache();

    let prefill_start = Instant::now();
    let logits =
        model.forward_with_hybrid_cache(&prompt, None, Some(&mut cache), mamba_cache.as_mut())?;
    pmetal_bridge::compat::transforms::eval([&logits])?;
    let prefill_elapsed = prefill_start.elapsed();

    let first_generated_token = pmetal_bridge::compat::ops::argmax(&logits.index((.., -1, ..)), -1);
    let first_generated_token_id = first_generated_token.item::<u32>();
    let mut next_token = first_generated_token_id as i32;
    let decode_start = Instant::now();
    for _ in 0..decode_steps {
        let decode_input = Array::from_slice(&[next_token], &[1, 1]);
        let decode_logits = model.forward_with_hybrid_cache(
            &decode_input,
            None,
            Some(&mut cache),
            mamba_cache.as_mut(),
        )?;
        let next_token_array =
            pmetal_bridge::compat::ops::argmax(&decode_logits.index((.., -1, ..)), -1);
        next_token = next_token_array.item::<u32>() as i32;
    }
    let decode_elapsed = decode_start.elapsed();

    Ok(PrefillDecodePassResult {
        prefill_elapsed,
        decode_elapsed,
        first_generated_token_id,
    })
}

async fn benchmark_real_training(
    model_id: &str,
    model_path: &Path,
    samples: &[TextSample],
    train_steps: usize,
    batch_size: usize,
    requested_max_seq_len: usize,
) -> anyhow::Result<(
    WorkloadTrainingSeqLenSelection,
    WorkloadBenchmarkSection<TrainingWorkloadMetrics>,
)> {
    if train_steps == 0 {
        return Ok((
            empty_training_seq_len_selection(requested_max_seq_len),
            WorkloadBenchmarkSection::Skipped {
                reason: "train_steps set to 0".to_string(),
            },
        ));
    }
    if samples.is_empty() {
        return Ok((
            empty_training_seq_len_selection(requested_max_seq_len),
            WorkloadBenchmarkSection::Skipped {
                reason: "no non-empty training samples available".to_string(),
            },
        ));
    }

    let bench_dir = std::env::temp_dir().join(format!(
        "pmetal-bench-workload-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    ));
    std::fs::create_dir_all(&bench_dir)?;
    let dataset_path = bench_dir.join("sampled_train.jsonl");
    write_sampled_workload_dataset(samples, &dataset_path)?;
    let training_seq_len = resolve_workload_training_seq_len(
        model_path,
        model_id,
        &dataset_path,
        requested_max_seq_len,
    )?;
    let max_seq_len = training_seq_len.effective_max_seq_len;

    let collector = StepMetricsCollector::new();
    let callback = collector.clone();

    let mut lora = LoraConfig::default();
    if lora.target_modules.is_empty() {
        lora.target_modules = vec![
            "q_proj".to_string(),
            "k_proj".to_string(),
            "v_proj".to_string(),
            "o_proj".to_string(),
        ];
    }

    let training_start = Instant::now();
    let result = run_training(
        TrainingJobConfig {
            model_id: model_id.to_string(),
            dataset: dataset_path.display().to_string(),
            eval_dataset: None,
            output_dir: bench_dir.join("output").display().to_string(),
            lora,
            qlora: None,
            training: TrainingConfig {
                batch_size,
                gradient_accumulation_steps: 1,
                num_epochs: 1,
                max_steps: Some(train_steps),
                warmup_steps: 0,
                logging_steps: 1,
                eval_steps: None,
                save_steps: None,
                max_seq_len,
                ..Default::default()
            },
            columns: Some(DatasetColumnConfig {
                text_column: Some("text".to_string()),
                prompt_column: Some("prompt".to_string()),
                ..Default::default()
            }),
            dispatch: DispatchConfig {
                flash_attention: true,
                sequence_packing: true,
                pack_max_seq_len: None,
                jit_compilation: true,
                fused: true,
                metal_fused_optimizer: true,
                gradient_checkpointing: true,
                gradient_checkpointing_layers: 4,
                cut_cross_entropy: true,
                ane: false,
                loss_scale: 1.0,
                no_adaptive_lr: true,
                #[cfg(feature = "distributed")]
                distributed: None,
            },
            config_path: None,
            log_metrics: None,
            resume: false,
            seed: 42,
            emit_console_output: false,
        },
        None,
        vec![Box::new(callback)],
    )
    .await?;
    let wall_clock_ms = duration_to_ms(training_start.elapsed());
    let steps = collector.snapshot();
    if steps.is_empty() {
        return Ok((
            training_seq_len,
            WorkloadBenchmarkSection::Skipped {
                reason: "training completed without step metrics".to_string(),
            },
        ));
    }

    let mut tok_sec: Vec<f64> = steps.iter().map(|step| step.tok_sec).collect();
    let mut step_ms: Vec<f64> = steps.iter().map(|step| step.total_ms).collect();

    Ok((
        training_seq_len,
        WorkloadBenchmarkSection::Completed(TrainingWorkloadMetrics {
            train_samples: samples.len(),
            train_steps,
            batch_size,
            max_seq_len,
            total_tokens: result.total_tokens,
            total_steps: result.total_steps,
            wall_clock_ms,
            median_tok_sec: median(&mut tok_sec),
            mean_tok_sec: mean(&tok_sec),
            median_step_ms: median(&mut step_ms),
            mean_step_ms: mean(&step_ms),
            final_loss: result.final_loss,
        }),
    ))
}

const AUTO_BENCH_WORKLOAD_MAX_SEQ_LEN_CAP: usize = 2048;
const AUTO_BENCH_WORKLOAD_MAX_PROMPT_TOKENS_CAP: usize = 1024;
const AUTO_BENCH_WORKLOAD_MIN_PROMPT_TOKENS: usize = 64;

#[derive(Debug, Clone, Copy)]
enum ResolvedWorkloadInferenceContext {
    Prompt,
    TextPrefix,
}

impl ResolvedWorkloadInferenceContext {
    fn as_str(self) -> &'static str {
        match self {
            Self::Prompt => "prompt",
            Self::TextPrefix => "text-prefix",
        }
    }
}

fn empty_prompt_len_selection(requested_max_prompt_tokens: usize) -> WorkloadPromptLenSelection {
    WorkloadPromptLenSelection {
        context_source: "prompt".to_string(),
        context_source_reason: "unresolved".to_string(),
        requested_max_prompt_tokens,
        effective_max_prompt_tokens: requested_max_prompt_tokens,
        max_prompt_tokens_source: if requested_max_prompt_tokens == 0 {
            "auto-unresolved".to_string()
        } else {
            "user".to_string()
        },
        sample_median_prompt_tokens: 0,
        sample_p95_prompt_tokens: 0,
        sample_max_prompt_tokens: 0,
        sample_truncated_pct: 0.0,
    }
}

fn empty_training_seq_len_selection(
    requested_max_seq_len: usize,
) -> WorkloadTrainingSeqLenSelection {
    WorkloadTrainingSeqLenSelection {
        requested_max_seq_len,
        effective_max_seq_len: requested_max_seq_len,
        max_seq_len_source: if requested_max_seq_len == 0 {
            "auto-unresolved".to_string()
        } else {
            "user".to_string()
        },
        sample_median_tokens: 0,
        sample_p95_tokens: 0,
        sample_max_tokens: 0,
        sample_truncated_pct: 0.0,
    }
}

fn resolve_workload_inference_prompt_len(
    model_path: &Path,
    samples: &[TextSample],
    requested_max_prompt_tokens: usize,
    requested_inference_context: WorkloadInferenceContext,
    decode_steps: usize,
) -> anyhow::Result<WorkloadPromptLenSelection> {
    let tokenizer = Tokenizer::from_model_dir(model_path)?;
    let prompt_lengths = collect_inference_context_lengths(
        &tokenizer,
        samples,
        ResolvedWorkloadInferenceContext::Prompt,
    )?;
    let text_prefix_lengths = collect_inference_context_lengths(
        &tokenizer,
        samples,
        ResolvedWorkloadInferenceContext::TextPrefix,
    )?;

    if prompt_lengths.is_empty() && text_prefix_lengths.is_empty() {
        return Ok(empty_prompt_len_selection(requested_max_prompt_tokens));
    }

    let auto_cap = workload_auto_inference_prompt_len_cap(model_path, decode_steps);
    let prompt_summary = summarize_prompt_lengths(&prompt_lengths, auto_cap);
    let (resolved_context, context_source_reason) = resolve_requested_workload_inference_context(
        requested_inference_context,
        prompt_summary.sample_p95_prompt_tokens,
    );
    let raw_lengths = match resolved_context {
        ResolvedWorkloadInferenceContext::Prompt => &prompt_lengths,
        ResolvedWorkloadInferenceContext::TextPrefix => &text_prefix_lengths,
    };
    let effective_max_prompt_tokens = if requested_max_prompt_tokens > 0 {
        requested_max_prompt_tokens
    } else {
        summarize_prompt_lengths(raw_lengths, auto_cap)
            .sample_p95_prompt_tokens
            .min(auto_cap)
            .max(32)
    };
    let summary = summarize_prompt_lengths(raw_lengths, effective_max_prompt_tokens);

    Ok(WorkloadPromptLenSelection {
        context_source: resolved_context.as_str().to_string(),
        context_source_reason,
        requested_max_prompt_tokens,
        effective_max_prompt_tokens,
        max_prompt_tokens_source: if requested_max_prompt_tokens > 0 {
            "user".to_string()
        } else {
            format!("auto-sampled-p95-capped-{auto_cap}")
        },
        ..summary
    })
}

fn collect_inference_context_lengths(
    tokenizer: &Tokenizer,
    samples: &[TextSample],
    inference_context: ResolvedWorkloadInferenceContext,
) -> anyhow::Result<Vec<usize>> {
    Ok(samples
        .iter()
        .map(|sample| {
            tokenizer
                .encode_with_special_tokens(benchmark_inference_text(sample, inference_context))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|ids| ids.len())
        .filter(|&len| len > 0)
        .collect())
}

fn resolve_requested_workload_inference_context(
    requested_inference_context: WorkloadInferenceContext,
    prompt_p95_tokens: usize,
) -> (ResolvedWorkloadInferenceContext, String) {
    match requested_inference_context {
        WorkloadInferenceContext::Prompt => (
            ResolvedWorkloadInferenceContext::Prompt,
            "user-prompt".to_string(),
        ),
        WorkloadInferenceContext::TextPrefix => (
            ResolvedWorkloadInferenceContext::TextPrefix,
            "user-text-prefix".to_string(),
        ),
        WorkloadInferenceContext::Auto => {
            if prompt_p95_tokens >= AUTO_BENCH_WORKLOAD_MIN_PROMPT_TOKENS {
                (
                    ResolvedWorkloadInferenceContext::Prompt,
                    format!(
                        "auto-prompt-p95-ge-{}",
                        AUTO_BENCH_WORKLOAD_MIN_PROMPT_TOKENS
                    ),
                )
            } else {
                (
                    ResolvedWorkloadInferenceContext::TextPrefix,
                    format!(
                        "auto-short-prompt-p95-{}-lt-{}",
                        prompt_p95_tokens, AUTO_BENCH_WORKLOAD_MIN_PROMPT_TOKENS
                    ),
                )
            }
        }
    }
}

fn resolved_workload_inference_context_from_str(
    value: &str,
) -> Option<ResolvedWorkloadInferenceContext> {
    match value {
        "prompt" => Some(ResolvedWorkloadInferenceContext::Prompt),
        "text-prefix" => Some(ResolvedWorkloadInferenceContext::TextPrefix),
        _ => None,
    }
}

fn resolve_workload_training_seq_len(
    model_path: &Path,
    model_id: &str,
    sampled_dataset_path: &Path,
    requested_max_seq_len: usize,
) -> anyhow::Result<WorkloadTrainingSeqLenSelection> {
    let tokenizer = Tokenizer::from_model_dir(model_path)?;
    let chat_template = pmetal_data::chat_templates::detect_chat_template(model_path, model_id);

    let auto_cap = workload_auto_training_seq_len_cap(model_path);
    let effective_max_seq_len = if requested_max_seq_len > 0 {
        requested_max_seq_len
    } else {
        let probe_stats = analyze_workload_training_dataset(
            sampled_dataset_path,
            &tokenizer,
            &chat_template,
            auto_cap,
        )?;
        probe_stats.suggested_max_seq_len.max(64)
    };

    let stats = analyze_workload_training_dataset(
        sampled_dataset_path,
        &tokenizer,
        &chat_template,
        effective_max_seq_len,
    )?;

    Ok(WorkloadTrainingSeqLenSelection {
        requested_max_seq_len,
        effective_max_seq_len,
        max_seq_len_source: if requested_max_seq_len > 0 {
            "user".to_string()
        } else {
            format!("auto-sampled-p95-capped-{auto_cap}")
        },
        sample_median_tokens: stats.median_length,
        sample_p95_tokens: stats.p95_length,
        sample_max_tokens: stats.max_length,
        sample_truncated_pct: stats.truncated_pct,
    })
}

fn summarize_prompt_lengths(
    lengths: &[usize],
    max_prompt_tokens: usize,
) -> WorkloadPromptLenSelection {
    if lengths.is_empty() {
        return empty_prompt_len_selection(max_prompt_tokens);
    }

    let mut sorted = lengths.to_vec();
    sorted.sort_unstable();

    let median_idx = sorted.len() / 2;
    let p95_idx = ((sorted.len() * 95).div_ceil(100)).saturating_sub(1);
    let truncated = if max_prompt_tokens == 0 {
        0
    } else {
        lengths
            .iter()
            .filter(|&&len| len > max_prompt_tokens)
            .count()
    };

    WorkloadPromptLenSelection {
        context_source: "prompt".to_string(),
        context_source_reason: "derived".to_string(),
        requested_max_prompt_tokens: max_prompt_tokens,
        effective_max_prompt_tokens: max_prompt_tokens,
        max_prompt_tokens_source: "derived".to_string(),
        sample_median_prompt_tokens: sorted[median_idx],
        sample_p95_prompt_tokens: sorted[p95_idx],
        sample_max_prompt_tokens: *sorted.last().unwrap_or(&0),
        sample_truncated_pct: (truncated as f64 * 100.0) / lengths.len() as f64,
    }
}

fn analyze_workload_training_dataset(
    sampled_dataset_path: &Path,
    tokenizer: &Tokenizer,
    chat_template: &pmetal_data::chat_templates::ChatTemplate,
    max_seq_len: usize,
) -> anyhow::Result<pmetal_data::DatasetStatistics> {
    let dataset = TrainingDataset::from_jsonl_tokenized(
        sampled_dataset_path,
        tokenizer,
        DatasetFormat::Auto,
        max_seq_len,
        Some(chat_template),
        Some(&DatasetColumnConfig {
            text_column: Some("text".to_string()),
            prompt_column: Some("prompt".to_string()),
            ..Default::default()
        }),
    )?;
    Ok(dataset.compute_statistics(max_seq_len))
}

fn workload_auto_training_seq_len_cap(model_path: &Path) -> usize {
    workload_model_max_seq_len(model_path)
        .unwrap_or(AUTO_BENCH_WORKLOAD_MAX_SEQ_LEN_CAP)
        .clamp(64, AUTO_BENCH_WORKLOAD_MAX_SEQ_LEN_CAP)
}

fn workload_auto_inference_prompt_len_cap(model_path: &Path, decode_steps: usize) -> usize {
    let requested_window = workload_model_max_seq_len(model_path)
        .map(|max_seq_len| max_seq_len.saturating_sub(decode_steps.saturating_add(8)))
        .unwrap_or(AUTO_BENCH_WORKLOAD_MAX_PROMPT_TOKENS_CAP);
    requested_window.clamp(32, AUTO_BENCH_WORKLOAD_MAX_PROMPT_TOKENS_CAP)
}

fn workload_model_max_seq_len(model_path: &Path) -> Option<usize> {
    let config_path = model_path.join("config.json");
    let config = std::fs::read(&config_path).ok()?;
    let json: serde_json::Value = serde_json::from_slice(&config).ok()?;
    [
        "/max_position_embeddings",
        "/text_config/max_position_embeddings",
        "/seq_length",
        "/text_config/seq_length",
        "/max_sequence_length",
        "/text_config/max_sequence_length",
        "/n_positions",
        "/text_config/n_positions",
    ]
    .into_iter()
    .find_map(|pointer| json.pointer(pointer).and_then(|v| v.as_u64()))
    .map(|value| value as usize)
}

fn write_sampled_workload_dataset(samples: &[TextSample], path: &Path) -> anyhow::Result<()> {
    let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
    for sample in samples {
        let row = if let Some(prompt) = sample.prompt.as_ref() {
            serde_json::json!({
                "text": sample.text,
                "prompt": prompt,
            })
        } else {
            serde_json::json!({
                "text": sample.text,
            })
        };
        writeln!(writer, "{}", serde_json::to_string(&row)?)?;
    }
    writer.flush()?;
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

fn max_abs_diff(reference: &[f32], candidate: &[f32]) -> f32 {
    reference
        .iter()
        .zip(candidate.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max)
}

fn print_workload_benchmark_report(report: &WorkloadBenchmarkReport, output: Option<&Path>) {
    println!("Workload Benchmark");
    println!("  Device:  {}", report.device.name);
    if let Some(preset) = report.workload.preset.as_deref() {
        println!("  Preset:  {}", preset);
    }
    println!("  Model:   {}", report.workload.model_id);
    println!("  Dataset: {}", report.workload.dataset_id);
    println!();

    match &report.inference {
        WorkloadBenchmarkSection::Completed(metrics) => {
            println!("Inference");
            println!("  Sessions: {}", metrics.session_runs);
            println!(
                "  Context: {} ({})",
                report.workload.inference_prompt_len.context_source,
                report.workload.inference_prompt_len.context_source_reason
            );
            println!(
                "  Prompt len: {} ({}, sample p95 {}, {:.1}% truncated)",
                report
                    .workload
                    .inference_prompt_len
                    .effective_max_prompt_tokens,
                report
                    .workload
                    .inference_prompt_len
                    .max_prompt_tokens_source,
                report
                    .workload
                    .inference_prompt_len
                    .sample_p95_prompt_tokens,
                report.workload.inference_prompt_len.sample_truncated_pct
            );
            println!(
                "  KV cache: {} ({})",
                metrics.cache_mode, metrics.cache_mode_source
            );
            if let Some(first_token_id) = metrics.first_generated_token_id {
                println!(
                    "  First token: {} ({:?})",
                    first_token_id,
                    metrics
                        .first_generated_token_text
                        .as_deref()
                        .unwrap_or("<decode failed>")
                );
            }
            println!(
                "  Prefill: {:.0} tok/s aggregate, {:.0} tok/s median over {} prompt tokens ({} samples, {} passes, {:.2} ms total)",
                metrics.prefill_tok_per_sec,
                metrics.median_prefill_tok_per_sec,
                metrics.prompt_tokens,
                metrics.prompt_samples,
                metrics.measurement_passes,
                metrics.total_prefill_ms
            );
            println!(
                "  Warmup: {} untimed pass(es) per sample ({} total)",
                report.workload.inference_warmup_passes, metrics.warmup_passes
            );
            println!(
                "  Decode:  {:.0} tok/s aggregate, {:.0} tok/s median ({:.2} ms/token aggregate, {:.2} ms/token median over {} steps/sample x {} repeats)",
                metrics.decode_tok_per_sec,
                metrics.median_decode_tok_per_sec,
                metrics.decode_ms_per_token,
                metrics.median_decode_ms_per_token,
                metrics.decode_steps,
                metrics.inference_repeats
            );
            if metrics.session_runs > 1 {
                println!(
                    "  Session medians: prefill {:.0} tok/s, decode {:.0} tok/s ({:.2} ms/token)",
                    metrics.median_session_prefill_tok_per_sec,
                    metrics.median_session_decode_tok_per_sec,
                    metrics.median_session_decode_ms_per_token
                );
            }
        }
        WorkloadBenchmarkSection::Skipped { reason } => {
            println!("Inference");
            println!("  Skipped: {reason}");
        }
        WorkloadBenchmarkSection::Failed { error } => {
            println!("Inference");
            println!("  Failed: {error}");
        }
    }

    match &report.training {
        WorkloadBenchmarkSection::Completed(metrics) => {
            println!("Training");
            println!(
                "  Seq len: {} ({}, sample p95 {}, {:.1}% truncated)",
                report.workload.training_seq_len.effective_max_seq_len,
                report.workload.training_seq_len.max_seq_len_source,
                report.workload.training_seq_len.sample_p95_tokens,
                report.workload.training_seq_len.sample_truncated_pct
            );
            println!(
                "  Median step: {:.2} ms, median throughput: {:.0} tok/s",
                metrics.median_step_ms, metrics.median_tok_sec
            );
            println!(
                "  Final loss: {:.4} across {} steps / {} tokens",
                metrics.final_loss, metrics.total_steps, metrics.total_tokens
            );
        }
        WorkloadBenchmarkSection::Skipped { reason } => {
            println!("Training");
            println!("  Skipped: {reason}");
        }
        WorkloadBenchmarkSection::Failed { error } => {
            println!("Training");
            println!("  Failed: {error}");
        }
    }

    if let Some(path) = output {
        println!();
        println!("Report written to {}", path.display());
    }
}

fn print_gdn_decode_benchmark_report(report: &GdnDecodeBenchmarkReport, output: Option<&Path>) {
    println!("GDN Benchmark");
    println!("  Device:  {}", report.device.name);
    println!("  Stage:   {}", report.stage);
    println!("  Model:   {}", report.model_id);
    println!("  Layer:   {}", report.layer_idx);
    println!(
        "  Shape:   batch={} seq={} input={} output={}",
        report.batch_size, report.seq_len, report.input_dim, report.output_dim
    );
    println!("  Reference: {}", report.reference_backend);
    println!();

    for backend in &report.backends {
        print!("{}: ", backend.name);
        match &backend.outcome {
            KernelBenchmarkOutcome::Completed {
                min_ms,
                median_ms,
                mean_ms,
            } => {
                println!(
                    "min {:.3} ms, median {:.3} ms, mean {:.3} ms",
                    min_ms, median_ms, mean_ms
                );
                if let Some(max_abs_diff) = backend.max_abs_diff_vs_reference {
                    println!("  max abs diff vs reference: {:.6}", max_abs_diff);
                }
            }
            KernelBenchmarkOutcome::Skipped { reason } => {
                println!("skipped ({reason})");
            }
            KernelBenchmarkOutcome::Failed { error } => {
                println!("failed ({error})");
            }
        }
    }

    if let Some(path) = output {
        println!();
        println!("Report written to {}", path.display());
    }
}

fn run_kernel_benchmark_case(
    ctx: &Arc<MetalContext>,
    case: &KernelBenchmarkCase,
    warmup_iterations: usize,
    benchmark_iterations: usize,
) -> KernelBenchmarkCaseResult {
    match case {
        KernelBenchmarkCase::FlashAttention(case) => {
            let parameters = flash_attention_parameters(*case);
            let config = FlashAttentionConfig::inference(
                case.batch_size,
                case.num_heads,
                case.num_kv_heads,
                case.seq_len,
                case.head_dim,
            );
            let tuning = match ctx.tuner().tune_flash_attention(ctx, &config) {
                Ok(tuned) => btree_map([
                    ("block_q", tuned.block_q.to_string()),
                    ("block_k", tuned.block_k.to_string()),
                ]),
                Err(error) => btree_map([("selection_error", error.to_string())]),
            };

            let outcome = (|| -> anyhow::Result<KernelBenchmarkOutcome> {
                let kernel = FlashAttention::new(ctx.clone(), config.clone())?;
                let queries = alloc_f16_buffer(ctx, config.query_size())?;
                let keys = alloc_f16_buffer(ctx, config.kv_size())?;
                let values = alloc_f16_buffer(ctx, config.kv_size())?;

                benchmark_operation(warmup_iterations, benchmark_iterations, || {
                    let output = kernel.forward(&queries, &keys, &values)?;
                    std::hint::black_box(output);
                    Ok(())
                })
            })();

            build_case_result(case.name, "flash_attention", parameters, tuning, outcome)
        }
        KernelBenchmarkCase::FusedLora(case) => {
            let parameters = fused_lora_parameters(*case);
            let config = FusedLoraConfig::new(
                case.batch_size,
                case.in_features,
                case.out_features,
                case.rank,
                2.0,
            );
            let tuning = match ctx.tuner().tune_lora_forward(
                ctx,
                case.batch_size,
                case.in_features,
                case.out_features,
                case.rank,
            ) {
                Ok(tuned) => btree_map([
                    ("tile_m", tuned.tile_m.to_string()),
                    ("tile_n", tuned.tile_n.to_string()),
                    ("tile_k", tuned.tile_k.to_string()),
                ]),
                Err(error) => btree_map([("selection_error", error.to_string())]),
            };

            let outcome = (|| -> anyhow::Result<KernelBenchmarkOutcome> {
                let kernel = FusedLora::new(ctx.clone(), config)?;
                let x = alloc_f16_buffer(ctx, case.batch_size * case.in_features)?;
                let weight = alloc_f16_buffer(ctx, case.out_features * case.in_features)?;
                let lora_a = alloc_f16_buffer(ctx, case.rank * case.in_features)?;
                let lora_b = alloc_f16_buffer(ctx, case.out_features * case.rank)?;

                benchmark_operation(warmup_iterations, benchmark_iterations, || {
                    let output = kernel.forward_inference(&x, &weight, &lora_a, &lora_b)?;
                    std::hint::black_box(output);
                    Ok(())
                })
            })();

            build_case_result(case.name, "fused_lora", parameters, tuning, outcome)
        }
        KernelBenchmarkCase::FusedMlp(case) => {
            let parameters = fused_mlp_parameters(*case);
            let tuning = match ctx.tuner().tune_swiglu(
                ctx,
                case.batch_size,
                case.hidden_size,
                case.intermediate_size,
            ) {
                Ok(tuned) => btree_map([
                    ("threads_per_token", tuned.threads_per_token.to_string()),
                    ("chunk_size", tuned.chunk_size.to_string()),
                ]),
                Err(error) => btree_map([("selection_error", error.to_string())]),
            };
            let config =
                FusedSwiGLUConfig::new(case.batch_size, case.hidden_size, case.intermediate_size);

            let outcome = (|| -> anyhow::Result<KernelBenchmarkOutcome> {
                let kernel = FusedMLP::new(ctx.clone(), config)?;
                let input = alloc_f32_buffer(ctx, case.batch_size * case.hidden_size)?;
                let gate_weight = alloc_f32_buffer(ctx, case.intermediate_size * case.hidden_size)?;
                let up_weight = alloc_f32_buffer(ctx, case.intermediate_size * case.hidden_size)?;
                let down_weight = alloc_f32_buffer(ctx, case.hidden_size * case.intermediate_size)?;

                benchmark_operation(warmup_iterations, benchmark_iterations, || {
                    let output = kernel.forward(&input, &gate_weight, &up_weight, &down_weight)?;
                    std::hint::black_box(output);
                    Ok(())
                })
            })();

            build_case_result(case.name, "fused_mlp", parameters, tuning, outcome)
        }
        KernelBenchmarkCase::FusedNormLora(case) => {
            let parameters = fused_norm_lora_parameters(*case);
            let tuning = match ctx.tuner().tune_norm_lora(
                ctx,
                case.batch_size,
                case.hidden_size,
                case.out_features,
                case.rank,
            ) {
                Ok(tuned) => btree_map([
                    ("threads_per_token", tuned.threads_per_token.to_string()),
                    ("use_tiled", tuned.use_tiled.to_string()),
                ]),
                Err(error) => btree_map([("selection_error", error.to_string())]),
            };
            let config = FusedNormLoraConfig::new(
                case.batch_size,
                case.hidden_size,
                case.out_features,
                case.rank,
                16.0,
            );

            let outcome = (|| -> anyhow::Result<KernelBenchmarkOutcome> {
                let kernel = FusedNormLora::new(ctx.clone(), config)?;
                let input = alloc_f32_buffer(ctx, case.batch_size * case.hidden_size)?;
                let gamma = alloc_f32_buffer(ctx, case.hidden_size)?;
                let weight = alloc_f32_buffer(ctx, case.out_features * case.hidden_size)?;
                let lora_a = alloc_f32_buffer(ctx, case.rank * case.hidden_size)?;
                let lora_b = alloc_f32_buffer(ctx, case.out_features * case.rank)?;

                benchmark_operation(warmup_iterations, benchmark_iterations, || {
                    let output = kernel.forward(&input, &gamma, &weight, &lora_a, &lora_b)?;
                    std::hint::black_box(output);
                    Ok(())
                })
            })();

            build_case_result(case.name, "fused_norm_lora", parameters, tuning, outcome)
        }
        KernelBenchmarkCase::FusedLinearCrossEntropy(case) => {
            let parameters = fused_linear_cross_entropy_parameters(*case);
            let config = FusedLinearCrossEntropyConfig::new(
                case.num_tokens,
                case.hidden_size,
                case.vocab_size,
            )
            .with_fp16();
            let tuning = match ctx.tuner().tune_fused_linear_cross_entropy(ctx, &config) {
                Ok(tuned) => btree_map([
                    ("threadgroup_size", tuned.threadgroup_size.to_string()),
                    ("chunk_size", tuned.chunk_size.to_string()),
                ]),
                Err(error) => btree_map([("selection_error", error.to_string())]),
            };

            let outcome = (|| -> anyhow::Result<KernelBenchmarkOutcome> {
                let kernel = FusedLinearCrossEntropy::new(ctx.clone(), config)?;
                let hidden_states = alloc_f16_buffer(ctx, case.num_tokens * case.hidden_size)?;
                let lm_head_weight = alloc_f16_buffer(ctx, case.vocab_size * case.hidden_size)?;
                let targets = alloc_i32_targets(ctx, case.num_tokens, case.vocab_size)?;

                benchmark_operation(warmup_iterations, benchmark_iterations, || {
                    let output = kernel.forward_f16(&hidden_states, &lm_head_weight, &targets)?;
                    std::hint::black_box(output);
                    Ok(())
                })
            })();

            build_case_result(
                case.name,
                "fused_linear_cross_entropy",
                parameters,
                tuning,
                outcome,
            )
        }
        KernelBenchmarkCase::FusedMerge(case) => {
            let parameters = fused_merge_parameters(*case);
            let total_elements = case.num_models * case.elements_per_model;
            let mut kernel = FusedMergeMetal::new(ctx.clone());
            let tuning = match kernel.tune_for_problem_size(total_elements, case.num_models) {
                Ok(tuned) => btree_map([
                    ("threads_per_group", tuned.threads_per_group.to_string()),
                    ("elements_per_thread", tuned.elements_per_thread.to_string()),
                    ("use_simd", tuned.use_simd.to_string()),
                ]),
                Err(error) => btree_map([("selection_error", error.to_string())]),
            };

            let outcome = (|| -> anyhow::Result<KernelBenchmarkOutcome> {
                let sizes = vec![case.elements_per_model; case.num_models];
                let densities = vec![0.5; case.num_models];
                let tensor_info = build_tensor_info(&sizes, &densities);
                let config = build_merge_config(&tensor_info, 1e-8);
                let input = alloc_f32_buffer(ctx, total_elements)?;
                let output = MetalBuffer::zeros(ctx, total_elements, BufferUsage::Shared)?;
                let tensor_info_buffer =
                    MetalBuffer::from_slice(ctx, &tensor_info, BufferUsage::Shared)?;

                benchmark_operation(warmup_iterations, benchmark_iterations, || {
                    let mut batch = BatchedCommandBuffer::new(ctx.clone())?;
                    kernel.queue_compute_magnitudes(
                        &mut batch,
                        &input,
                        &output,
                        &tensor_info_buffer,
                        &config,
                    )?;
                    batch.execute()?;
                    Ok(())
                })
            })();

            build_case_result(case.name, "fused_merge", parameters, tuning, outcome)
        }
        KernelBenchmarkCase::ModelMoe(case) => {
            let parameters = model_moe_parameters(*case);
            let tuning = btree_map([
                (
                    "dispatch",
                    match case.family {
                        ModelMoeFamily::Llama4 => "per_expert_gather_scatter",
                        ModelMoeFamily::Qwen3 | ModelMoeFamily::DeepSeek => "stacked_gather_mm",
                    }
                    .to_string(),
                ),
                (
                    "router_topk",
                    match case.family {
                        ModelMoeFamily::Llama4 => "argpartition+shared_expert",
                        ModelMoeFamily::Qwen3 => "softmax+argpartition",
                        ModelMoeFamily::DeepSeek => "sigmoid+bias+argpartition",
                    }
                    .to_string(),
                ),
            ]);

            let outcome = (|| -> anyhow::Result<KernelBenchmarkOutcome> {
                let input = pmetal_bridge::compat::random::normal(
                    &[
                        case.batch_size as i32,
                        case.seq_len as i32,
                        case.hidden_size as i32,
                    ],
                    pmetal_bridge::compat::Dtype::Float32,
                );

                match case.family {
                    ModelMoeFamily::Llama4 => {
                        let config = Llama4TextConfig {
                            hidden_size: case.hidden_size as i32,
                            intermediate_size: case.intermediate_size as i32,
                            intermediate_size_mlp: case.intermediate_size as i32,
                            num_local_experts: case.num_experts as i32,
                            num_experts_per_tok: case.top_k as i32,
                            ..Default::default()
                        };
                        let mut moe = Llama4MoE::new(&config)?;
                        benchmark_operation(warmup_iterations, benchmark_iterations, || {
                            let output = moe.forward(&input)?;
                            pmetal_bridge::compat::transforms::eval([&output])?;
                            std::hint::black_box(output);
                            Ok(())
                        })
                    }
                    ModelMoeFamily::Qwen3 => {
                        let num_attention_heads = (case.hidden_size / 128).max(1) as i32;
                        let config = Qwen3MoEConfig {
                            hidden_size: case.hidden_size as i32,
                            intermediate_size: case.intermediate_size as i32,
                            moe_intermediate_size: Some(case.intermediate_size as i32),
                            num_experts: case.num_experts as i32,
                            num_experts_per_tok: case.top_k as i32,
                            num_attention_heads,
                            num_key_value_heads: Some(num_attention_heads),
                            head_dim: (case.hidden_size / num_attention_heads as usize) as i32,
                            ..Default::default()
                        };
                        let mut moe = Qwen3MoEBlock::new(&config)?;
                        benchmark_operation(warmup_iterations, benchmark_iterations, || {
                            let output = moe.forward(&input)?;
                            pmetal_bridge::compat::transforms::eval([&output])?;
                            std::hint::black_box(output);
                            Ok(())
                        })
                    }
                    ModelMoeFamily::DeepSeek => {
                        let config = DeepSeekConfig {
                            hidden_size: case.hidden_size as i32,
                            moe_intermediate_size: case.intermediate_size as i32,
                            n_shared_experts: Some(1),
                            n_routed_experts: Some(case.num_experts as i32),
                            num_experts_per_tok: case.top_k as i32,
                            ..Default::default()
                        };
                        let mut moe = DeepSeekMoE::new(&config)?;
                        benchmark_operation(warmup_iterations, benchmark_iterations, || {
                            let output = moe.forward(&input)?;
                            pmetal_bridge::compat::transforms::eval([&output])?;
                            std::hint::black_box(output);
                            Ok(())
                        })
                    }
                }
            })();

            build_case_result(
                case.name,
                match case.family {
                    ModelMoeFamily::Llama4 => "llama4_moe",
                    ModelMoeFamily::Qwen3 => "qwen3_moe",
                    ModelMoeFamily::DeepSeek => "deepseek_moe",
                },
                parameters,
                tuning,
                outcome,
            )
        }
        KernelBenchmarkCase::ModelHybrid(case) => {
            let parameters = model_hybrid_parameters(*case);
            let tuning = btree_map([
                ("dispatch", "causal_depthwise_conv".to_string()),
                ("feed_forward", "moe_argpartition".to_string()),
            ]);

            let outcome = (|| -> anyhow::Result<KernelBenchmarkOutcome> {
                let input = pmetal_bridge::compat::random::normal(
                    &[
                        case.batch_size as i32,
                        case.seq_len as i32,
                        case.hidden_size as i32,
                    ],
                    pmetal_bridge::compat::Dtype::Float32,
                );

                match case.family {
                    ModelHybridFamily::Jamba => {
                        let num_attention_heads = (case.hidden_size / 64).max(1) as i32;
                        let config = JambaConfig {
                            hidden_size: case.hidden_size as i32,
                            intermediate_size: case.intermediate_size as i32,
                            num_hidden_layers: 2,
                            num_attention_heads,
                            num_key_value_heads: num_attention_heads,
                            vocab_size: 32_768,
                            rms_norm_eps: 1e-5,
                            num_experts: case.num_experts as i32,
                            num_experts_per_tok: case.top_k as i32,
                            layers_per_block: 2,
                            attn_layer_offset: 0,
                            mamba_conv_kernel_size: 4,
                        };
                        let mut layer = JambaLayer::new(&config, 1)?;
                        benchmark_operation(warmup_iterations, benchmark_iterations, || {
                            let output = layer.forward(&input)?;
                            pmetal_bridge::compat::transforms::eval([&output])?;
                            std::hint::black_box(output);
                            Ok(())
                        })
                    }
                }
            })();

            build_case_result(
                case.name,
                match case.family {
                    ModelHybridFamily::Jamba => "jamba_hybrid",
                },
                parameters,
                tuning,
                outcome,
            )
        }
        KernelBenchmarkCase::MppGemm(case) => {
            let parameters = mpp_gemm_parameters(*case);
            let config = MppGemmConfig::new(case.m, case.n, case.k);
            let gemm = MppGemm::new(ctx.clone(), config);
            if !gemm.is_available() {
                return KernelBenchmarkCaseResult {
                    name: case.name.to_string(),
                    category: "mpp_gemm",
                    parameters,
                    tuning: BTreeMap::new(),
                    outcome: KernelBenchmarkOutcome::Skipped {
                        reason: "MPP GEMM not available on this device".to_string(),
                    },
                };
            }

            let tuning = match ctx.tuner().tune_mpp_gemm(
                ctx,
                MppGemmTuneRequest {
                    m: case.m,
                    n: case.n,
                    k: case.k,
                    batch_size: 1,
                    use_fp16: false,
                    accumulate: false,
                },
            ) {
                Ok(tuned) => btree_map([
                    ("variant", format!("{:?}", tuned.variant)),
                    ("use_morton", tuned.use_morton.to_string()),
                ]),
                Err(error) => btree_map([("selection_error", error.to_string())]),
            };

            let outcome = (|| -> anyhow::Result<KernelBenchmarkOutcome> {
                let a = alloc_f32_buffer(ctx, case.m * case.k)?;
                let b = alloc_f32_buffer(ctx, case.n * case.k)?;
                let d = MetalBuffer::zeros(ctx, case.m * case.n, BufferUsage::Shared)?;

                benchmark_operation(warmup_iterations, benchmark_iterations, || {
                    gemm.execute_f32(&a, &b, &d)?;
                    Ok(())
                })
            })();

            build_case_result(case.name, "mpp_gemm", parameters, tuning, outcome)
        }
    }
}

fn build_case_result(
    name: &str,
    category: &'static str,
    parameters: BTreeMap<String, String>,
    tuning: BTreeMap<String, String>,
    outcome: anyhow::Result<KernelBenchmarkOutcome>,
) -> KernelBenchmarkCaseResult {
    KernelBenchmarkCaseResult {
        name: name.to_string(),
        category,
        parameters,
        tuning,
        outcome: match outcome {
            Ok(outcome) => outcome,
            Err(error) => KernelBenchmarkOutcome::Failed {
                error: error.to_string(),
            },
        },
    }
}

fn summarize_kernel_benchmark_results(
    results: &[KernelBenchmarkCaseResult],
) -> KernelBenchmarkSummary {
    let mut summary = KernelBenchmarkSummary {
        completed: 0,
        skipped: 0,
        failed: 0,
    };
    for result in results {
        match result.outcome {
            KernelBenchmarkOutcome::Completed { .. } => summary.completed += 1,
            KernelBenchmarkOutcome::Skipped { .. } => summary.skipped += 1,
            KernelBenchmarkOutcome::Failed { .. } => summary.failed += 1,
        }
    }
    summary
}

fn build_benchmark_corpus_for_profile(
    tier: DeviceTier,
    _has_nax: bool,
    quick: bool,
) -> Vec<KernelBenchmarkCase> {
    let profile = benchmark_tier_profile(tier, quick);
    vec![
        KernelBenchmarkCase::FlashAttention(profile.flash_attention),
        KernelBenchmarkCase::FusedLora(profile.fused_lora),
        KernelBenchmarkCase::FusedMlp(profile.fused_mlp),
        KernelBenchmarkCase::FusedNormLora(profile.fused_norm_lora),
        KernelBenchmarkCase::FusedLinearCrossEntropy(profile.fused_linear_cross_entropy),
        KernelBenchmarkCase::FusedMerge(profile.fused_merge),
        KernelBenchmarkCase::ModelMoe(profile.llama4_moe),
        KernelBenchmarkCase::ModelMoe(profile.qwen3_moe),
        KernelBenchmarkCase::ModelMoe(profile.deepseek_moe),
        KernelBenchmarkCase::ModelHybrid(profile.jamba_hybrid),
        KernelBenchmarkCase::MppGemm(profile.mpp_gemm),
    ]
}

fn benchmark_tier_profile(tier: DeviceTier, quick: bool) -> KernelBenchmarkTierProfile {
    let scale = if quick { 1 } else { 2 };
    match tier {
        DeviceTier::Base => KernelBenchmarkTierProfile {
            flash_attention: FlashAttentionCase {
                name: "flash_attention_prefill",
                batch_size: 1,
                num_heads: 8,
                num_kv_heads: 8,
                seq_len: 128 * scale,
                head_dim: 64,
            },
            fused_lora: FusedLoraCase {
                name: "fused_lora_forward",
                batch_size: 64 * scale,
                in_features: 1024,
                out_features: 1024,
                rank: 16,
            },
            fused_mlp: FusedMlpCase {
                name: "fused_mlp",
                batch_size: 32 * scale,
                hidden_size: 1024,
                intermediate_size: 2816,
            },
            fused_norm_lora: FusedNormLoraCase {
                name: "fused_norm_lora",
                batch_size: 32 * scale,
                hidden_size: 1024,
                out_features: 1024,
                rank: 8,
            },
            fused_linear_cross_entropy: FusedLinearCrossEntropyCase {
                name: "fused_linear_cross_entropy",
                num_tokens: 32 * scale,
                hidden_size: 1024,
                vocab_size: 32_768,
            },
            fused_merge: FusedMergeCase {
                name: "fused_merge_magnitudes",
                num_models: 4,
                elements_per_model: 65_536 * scale,
            },
            llama4_moe: ModelMoeCase {
                name: "llama4_moe_dispatch",
                family: ModelMoeFamily::Llama4,
                batch_size: 1,
                seq_len: 32 * scale,
                hidden_size: 512,
                intermediate_size: 1536,
                num_experts: 8,
                top_k: 1,
            },
            qwen3_moe: ModelMoeCase {
                name: "qwen3_moe_dispatch",
                family: ModelMoeFamily::Qwen3,
                batch_size: 1,
                seq_len: 32 * scale,
                hidden_size: 512,
                intermediate_size: 768,
                num_experts: 8,
                top_k: 2,
            },
            deepseek_moe: ModelMoeCase {
                name: "deepseek_moe_dispatch",
                family: ModelMoeFamily::DeepSeek,
                batch_size: 1,
                seq_len: 32 * scale,
                hidden_size: 512,
                intermediate_size: 768,
                num_experts: 8,
                top_k: 2,
            },
            jamba_hybrid: ModelHybridCase {
                name: "jamba_hybrid_layer",
                family: ModelHybridFamily::Jamba,
                batch_size: 1,
                seq_len: 32 * scale,
                hidden_size: 512,
                intermediate_size: 1536,
                num_experts: 8,
                top_k: 2,
            },
            mpp_gemm: MppGemmCase {
                name: "mpp_gemm_prefill",
                m: 64 * scale,
                n: 1024,
                k: 1024,
            },
        },
        DeviceTier::Pro => KernelBenchmarkTierProfile {
            flash_attention: FlashAttentionCase {
                name: "flash_attention_prefill",
                batch_size: 1,
                num_heads: 16,
                num_kv_heads: 8,
                seq_len: 256 * scale,
                head_dim: 128,
            },
            fused_lora: FusedLoraCase {
                name: "fused_lora_forward",
                batch_size: 64 * scale,
                in_features: 1536,
                out_features: 1536,
                rank: 16,
            },
            fused_mlp: FusedMlpCase {
                name: "fused_mlp",
                batch_size: 32 * scale,
                hidden_size: 1536,
                intermediate_size: 4224,
            },
            fused_norm_lora: FusedNormLoraCase {
                name: "fused_norm_lora",
                batch_size: 32 * scale,
                hidden_size: 1536,
                out_features: 1536,
                rank: 16,
            },
            fused_linear_cross_entropy: FusedLinearCrossEntropyCase {
                name: "fused_linear_cross_entropy",
                num_tokens: 64 * scale,
                hidden_size: 1536,
                vocab_size: 32_768,
            },
            fused_merge: FusedMergeCase {
                name: "fused_merge_magnitudes",
                num_models: 6,
                elements_per_model: 65_536 * scale,
            },
            llama4_moe: ModelMoeCase {
                name: "llama4_moe_dispatch",
                family: ModelMoeFamily::Llama4,
                batch_size: 1,
                seq_len: 48 * scale,
                hidden_size: 768,
                intermediate_size: 2048,
                num_experts: 8,
                top_k: 1,
            },
            qwen3_moe: ModelMoeCase {
                name: "qwen3_moe_dispatch",
                family: ModelMoeFamily::Qwen3,
                batch_size: 1,
                seq_len: 48 * scale,
                hidden_size: 768,
                intermediate_size: 1024,
                num_experts: 16,
                top_k: 4,
            },
            deepseek_moe: ModelMoeCase {
                name: "deepseek_moe_dispatch",
                family: ModelMoeFamily::DeepSeek,
                batch_size: 1,
                seq_len: 48 * scale,
                hidden_size: 768,
                intermediate_size: 1024,
                num_experts: 16,
                top_k: 4,
            },
            jamba_hybrid: ModelHybridCase {
                name: "jamba_hybrid_layer",
                family: ModelHybridFamily::Jamba,
                batch_size: 1,
                seq_len: 48 * scale,
                hidden_size: 768,
                intermediate_size: 2048,
                num_experts: 16,
                top_k: 2,
            },
            mpp_gemm: MppGemmCase {
                name: "mpp_gemm_prefill",
                m: 64 * scale,
                n: 1536,
                k: 1536,
            },
        },
        DeviceTier::Max | DeviceTier::Ultra => KernelBenchmarkTierProfile {
            flash_attention: FlashAttentionCase {
                name: "flash_attention_prefill",
                batch_size: 1,
                num_heads: 32,
                num_kv_heads: 8,
                seq_len: 256 * scale,
                head_dim: 128,
            },
            fused_lora: FusedLoraCase {
                name: "fused_lora_forward",
                batch_size: 64 * scale,
                in_features: 2048,
                out_features: 2048,
                rank: 16,
            },
            fused_mlp: FusedMlpCase {
                name: "fused_mlp",
                batch_size: 32 * scale,
                hidden_size: 2048,
                intermediate_size: 5632,
            },
            fused_norm_lora: FusedNormLoraCase {
                name: "fused_norm_lora",
                batch_size: 32 * scale,
                hidden_size: 2048,
                out_features: 2048,
                rank: 16,
            },
            fused_linear_cross_entropy: FusedLinearCrossEntropyCase {
                name: "fused_linear_cross_entropy",
                num_tokens: 64 * scale,
                hidden_size: 2048,
                vocab_size: 65_536,
            },
            fused_merge: FusedMergeCase {
                name: "fused_merge_magnitudes",
                num_models: 8,
                elements_per_model: 131_072 * scale,
            },
            llama4_moe: ModelMoeCase {
                name: "llama4_moe_dispatch",
                family: ModelMoeFamily::Llama4,
                batch_size: 1,
                seq_len: 64 * scale,
                hidden_size: 1024,
                intermediate_size: 2816,
                num_experts: 16,
                top_k: 1,
            },
            qwen3_moe: ModelMoeCase {
                name: "qwen3_moe_dispatch",
                family: ModelMoeFamily::Qwen3,
                batch_size: 1,
                seq_len: 64 * scale,
                hidden_size: 1024,
                intermediate_size: 1408,
                num_experts: 32,
                top_k: 4,
            },
            deepseek_moe: ModelMoeCase {
                name: "deepseek_moe_dispatch",
                family: ModelMoeFamily::DeepSeek,
                batch_size: 1,
                seq_len: 64 * scale,
                hidden_size: 1024,
                intermediate_size: 1408,
                num_experts: 32,
                top_k: 4,
            },
            jamba_hybrid: ModelHybridCase {
                name: "jamba_hybrid_layer",
                family: ModelHybridFamily::Jamba,
                batch_size: 1,
                seq_len: 64 * scale,
                hidden_size: 1024,
                intermediate_size: 2816,
                num_experts: 16,
                top_k: 2,
            },
            mpp_gemm: MppGemmCase {
                name: "mpp_gemm_prefill",
                m: 64 * scale,
                n: 2048,
                k: 2048,
            },
        },
    }
}

fn benchmark_operation<F>(
    warmup_iterations: usize,
    benchmark_iterations: usize,
    mut op: F,
) -> anyhow::Result<KernelBenchmarkOutcome>
where
    F: FnMut() -> anyhow::Result<()>,
{
    for _ in 0..warmup_iterations {
        op()?;
    }

    let mut times = Vec::with_capacity(benchmark_iterations);
    for _ in 0..benchmark_iterations {
        let start = Instant::now();
        op()?;
        times.push(start.elapsed());
    }

    times.sort();
    let min_time = times[0];
    let median_time = times[times.len() / 2];
    let mean_time = Duration::from_secs_f64(
        times.iter().map(|time| time.as_secs_f64()).sum::<f64>() / times.len() as f64,
    );

    Ok(KernelBenchmarkOutcome::Completed {
        min_ms: duration_to_ms(min_time),
        median_ms: duration_to_ms(median_time),
        mean_ms: duration_to_ms(mean_time),
    })
}

fn alloc_f16_buffer(ctx: &Arc<MetalContext>, len: usize) -> anyhow::Result<MetalBuffer<f16>> {
    let data: Vec<f16> = (0..len)
        .map(|i| f16::from_f32(deterministic_value(i)))
        .collect();
    Ok(MetalBuffer::from_slice(ctx, &data, BufferUsage::Shared)?)
}

fn alloc_f32_buffer(ctx: &Arc<MetalContext>, len: usize) -> anyhow::Result<MetalBuffer<f32>> {
    let data: Vec<f32> = (0..len).map(deterministic_value).collect();
    Ok(MetalBuffer::from_slice(ctx, &data, BufferUsage::Shared)?)
}

fn alloc_i32_targets(
    ctx: &Arc<MetalContext>,
    len: usize,
    vocab_size: usize,
) -> anyhow::Result<MetalBuffer<i32>> {
    let data: Vec<i32> = (0..len).map(|i| (i % vocab_size.max(1)) as i32).collect();
    Ok(MetalBuffer::from_slice(ctx, &data, BufferUsage::Shared)?)
}

fn deterministic_value(index: usize) -> f32 {
    (((index.wrapping_mul(1103515245).wrapping_add(12345) >> 16) & 1023) as f32 / 512.0) - 1.0
}

fn btree_map<const N: usize>(pairs: [(&str, String); N]) -> BTreeMap<String, String> {
    pairs
        .into_iter()
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

fn flash_attention_parameters(case: FlashAttentionCase) -> BTreeMap<String, String> {
    btree_map([
        ("batch_size", case.batch_size.to_string()),
        ("num_heads", case.num_heads.to_string()),
        ("num_kv_heads", case.num_kv_heads.to_string()),
        ("seq_len", case.seq_len.to_string()),
        ("head_dim", case.head_dim.to_string()),
    ])
}

fn fused_lora_parameters(case: FusedLoraCase) -> BTreeMap<String, String> {
    btree_map([
        ("batch_size", case.batch_size.to_string()),
        ("in_features", case.in_features.to_string()),
        ("out_features", case.out_features.to_string()),
        ("rank", case.rank.to_string()),
    ])
}

fn fused_mlp_parameters(case: FusedMlpCase) -> BTreeMap<String, String> {
    btree_map([
        ("batch_size", case.batch_size.to_string()),
        ("hidden_size", case.hidden_size.to_string()),
        ("intermediate_size", case.intermediate_size.to_string()),
    ])
}

fn fused_norm_lora_parameters(case: FusedNormLoraCase) -> BTreeMap<String, String> {
    btree_map([
        ("batch_size", case.batch_size.to_string()),
        ("hidden_size", case.hidden_size.to_string()),
        ("out_features", case.out_features.to_string()),
        ("rank", case.rank.to_string()),
    ])
}

fn fused_linear_cross_entropy_parameters(
    case: FusedLinearCrossEntropyCase,
) -> BTreeMap<String, String> {
    btree_map([
        ("num_tokens", case.num_tokens.to_string()),
        ("hidden_size", case.hidden_size.to_string()),
        ("vocab_size", case.vocab_size.to_string()),
    ])
}

fn fused_merge_parameters(case: FusedMergeCase) -> BTreeMap<String, String> {
    btree_map([
        ("num_models", case.num_models.to_string()),
        ("elements_per_model", case.elements_per_model.to_string()),
        (
            "total_elements",
            (case.num_models * case.elements_per_model).to_string(),
        ),
    ])
}

fn model_moe_parameters(case: ModelMoeCase) -> BTreeMap<String, String> {
    btree_map([
        (
            "family",
            match case.family {
                ModelMoeFamily::Llama4 => "llama4",
                ModelMoeFamily::Qwen3 => "qwen3",
                ModelMoeFamily::DeepSeek => "deepseek",
            }
            .to_string(),
        ),
        ("batch_size", case.batch_size.to_string()),
        ("seq_len", case.seq_len.to_string()),
        ("hidden_size", case.hidden_size.to_string()),
        ("intermediate_size", case.intermediate_size.to_string()),
        ("num_experts", case.num_experts.to_string()),
        ("top_k", case.top_k.to_string()),
    ])
}

fn model_hybrid_parameters(case: ModelHybridCase) -> BTreeMap<String, String> {
    btree_map([
        (
            "family",
            match case.family {
                ModelHybridFamily::Jamba => "jamba",
            }
            .to_string(),
        ),
        ("batch_size", case.batch_size.to_string()),
        ("seq_len", case.seq_len.to_string()),
        ("hidden_size", case.hidden_size.to_string()),
        ("intermediate_size", case.intermediate_size.to_string()),
        ("num_experts", case.num_experts.to_string()),
        ("top_k", case.top_k.to_string()),
    ])
}

fn mpp_gemm_parameters(case: MppGemmCase) -> BTreeMap<String, String> {
    btree_map([
        ("m", case.m.to_string()),
        ("n", case.n.to_string()),
        ("k", case.k.to_string()),
    ])
}

fn duration_to_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn device_tier_label(tier: DeviceTier) -> &'static str {
    match tier {
        DeviceTier::Base => "base",
        DeviceTier::Pro => "pro",
        DeviceTier::Max => "max",
        DeviceTier::Ultra => "ultra",
    }
}

fn memory_bandwidth_source_label(source: MemoryBandwidthSource) -> &'static str {
    match source {
        MemoryBandwidthSource::MeasuredGpuCopy => "measured_gpu_copy",
        MemoryBandwidthSource::SpecTableFallback => "spec_table_fallback",
    }
}

fn print_kernel_benchmark_report(report: &KernelBenchmarkReport, output: Option<&Path>) {
    println!("Kernel Benchmark Corpus");
    println!("  Device: {}", report.device.name);
    println!("  Tier:   {}", report.device.tier);
    println!("  Mode:   {}", report.mode);
    println!(
        "  Warmup / Iterations: {} / {}",
        report.warmup_iterations, report.benchmark_iterations
    );
    println!(
        "  Cases: completed={} skipped={} failed={}",
        report.summary.completed, report.summary.skipped, report.summary.failed
    );
    println!();

    for case in &report.cases {
        match &case.outcome {
            KernelBenchmarkOutcome::Completed {
                median_ms, mean_ms, ..
            } => {
                println!(
                    "{:<30} {:<24} median={:>8.2} ms mean={:>8.2} ms",
                    case.name, case.category, median_ms, mean_ms
                );
            }
            KernelBenchmarkOutcome::Skipped { reason } => {
                println!(
                    "{:<30} {:<24} skipped ({})",
                    case.name, case.category, reason
                );
            }
            KernelBenchmarkOutcome::Failed { error } => {
                println!("{:<30} {:<24} failed ({})", case.name, case.category, error);
            }
        }
    }

    if let Some(path) = output {
        println!();
        println!("Report written to {}", path.display());
    }
}

/// Generate a sample configuration file.
pub(crate) fn generate_sample_config(output: &str) -> anyhow::Result<()> {
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

/// Benchmark FFI overhead to compare Rust mlx-rs vs Python mlx performance.
///
/// Python baseline: ~7420 argmax ops/sec (~0.135ms per op) on Qwen3 vocab size.
pub(crate) fn run_ffi_benchmark() -> anyhow::Result<()> {
    use pmetal_bridge::compat::{
        Array,
        indexing::{argmax, argmax_axis},
        transforms::eval,
    };
    use std::time::Instant;

    println!("FFI Overhead Benchmark");
    println!("======================\n");

    // Warmup - ensure Metal is ready
    println!("Warming up Metal...");
    let warmup = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
    let _ = argmax(&warmup);
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
        let result = argmax_axis(&logits, -1, false);
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
        let input = token.reshape(&[1, 1]);
        // Simulate logits extraction
        let result = argmax_axis(&logits, -1, false);
        eval([&input, &result])?;
    }
    let elapsed = start.elapsed();

    let per_op_us = elapsed.as_micros() as f64 / n_iters as f64;
    let per_op_ms = per_op_us / 1000.0;

    println!("Total time: {:?}", elapsed);
    println!("Per operation: {:.3} ms ({:.1} us)", per_op_ms, per_op_us);
    println!("Equivalent tok/s: {:.0}", 1_000_000.0 / per_op_us);

    Ok(())
}

/// Benchmark generation loop timing with a real model.
///
/// This profiles each step of the generation loop to compare with mlx_lm's timing.
pub(crate) async fn run_gen_benchmark(model_id: &str) -> anyhow::Result<()> {
    use pmetal_bridge::compat::{Array, indexing::argmax, ops::async_eval, transforms::eval};
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
    let mut model = DynamicModel::load(&model_path)?;
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
    let mut current_token = argmax(&logits.index((.., -1, ..)));
    async_eval([&current_token]);

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
        let next_input = current_token.reshape(&[1, 1]);
        times.entry("reshape").or_default().push(t0.elapsed());

        // Forward pass
        let t0 = Instant::now();
        let next_logits = model.forward_with_cache(&next_input, None, Some(&mut cache))?;
        times.entry("forward").or_default().push(t0.elapsed());

        // Extract last logits
        let t0 = Instant::now();
        let last_logits = next_logits.index((.., -1, ..));
        times
            .entry("extract_logits")
            .or_default()
            .push(t0.elapsed());

        // Argmax
        let t0 = Instant::now();
        let next_token = argmax(&last_logits);
        times.entry("argmax").or_default().push(t0.elapsed());

        // Async eval for next
        let t0 = Instant::now();
        async_eval([&next_token]);
        times.entry("async_eval").or_default().push(t0.elapsed());

        // Extract current token (sync point)
        let t0 = Instant::now();
        let _ = current_token.item::<u32>();
        times.entry("item").or_default().push(t0.elapsed());

        times
            .entry("total")
            .or_default()
            .push(total_start.elapsed());

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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn benchmark_tier_profile_scales_by_tier() {
        let base = benchmark_tier_profile(DeviceTier::Base, true);
        let pro = benchmark_tier_profile(DeviceTier::Pro, true);
        let max = benchmark_tier_profile(DeviceTier::Max, true);

        assert!(base.fused_mlp.hidden_size < pro.fused_mlp.hidden_size);
        assert!(pro.fused_mlp.hidden_size < max.fused_mlp.hidden_size);
        assert!(base.flash_attention.seq_len < pro.flash_attention.seq_len);
        assert!(pro.flash_attention.seq_len <= max.flash_attention.seq_len);
        assert!(base.llama4_moe.hidden_size < pro.llama4_moe.hidden_size);
        assert!(pro.llama4_moe.hidden_size < max.llama4_moe.hidden_size);
        assert!(base.qwen3_moe.num_experts <= pro.qwen3_moe.num_experts);
        assert!(pro.qwen3_moe.num_experts <= max.qwen3_moe.num_experts);
        assert!(base.deepseek_moe.hidden_size < pro.deepseek_moe.hidden_size);
        assert!(pro.deepseek_moe.hidden_size < max.deepseek_moe.hidden_size);
        assert!(base.jamba_hybrid.hidden_size < pro.jamba_hybrid.hidden_size);
        assert!(pro.jamba_hybrid.hidden_size < max.jamba_hybrid.hidden_size);
    }

    #[test]
    fn benchmark_corpus_has_expected_categories() {
        let cases = build_benchmark_corpus_for_profile(DeviceTier::Base, false, true);
        assert_eq!(cases.len(), 11);
        assert!(matches!(cases[0], KernelBenchmarkCase::FlashAttention(_)));
        assert!(matches!(cases[1], KernelBenchmarkCase::FusedLora(_)));
        assert!(matches!(cases[2], KernelBenchmarkCase::FusedMlp(_)));
        assert!(matches!(cases[3], KernelBenchmarkCase::FusedNormLora(_)));
        assert!(matches!(
            cases[4],
            KernelBenchmarkCase::FusedLinearCrossEntropy(_)
        ));
        assert!(matches!(cases[5], KernelBenchmarkCase::FusedMerge(_)));
        assert!(matches!(cases[6], KernelBenchmarkCase::ModelMoe(_)));
        assert!(matches!(cases[7], KernelBenchmarkCase::ModelMoe(_)));
        assert!(matches!(cases[8], KernelBenchmarkCase::ModelMoe(_)));
        assert!(matches!(cases[9], KernelBenchmarkCase::ModelHybrid(_)));
        assert!(matches!(cases[10], KernelBenchmarkCase::MppGemm(_)));
    }

    #[test]
    fn benchmark_report_serializes_to_json() {
        let report = KernelBenchmarkReport {
            version: "test".to_string(),
            generated_at_unix_ms: 1,
            mode: "quick",
            warmup_iterations: 2,
            benchmark_iterations: 5,
            device: KernelBenchmarkDevice {
                name: "Test".to_string(),
                tier: "base".to_string(),
                architecture_gen: 7,
                gpu_core_count: 8,
                ane_core_count: 16,
                has_nax: false,
                is_apple10_or_newer: false,
                is_ultra_fusion: false,
                memory_bandwidth_gbps: 100.0,
                memory_bandwidth_source: "spec_table_fallback".to_string(),
            },
            summary: KernelBenchmarkSummary {
                completed: 1,
                skipped: 1,
                failed: 0,
            },
            cases: vec![KernelBenchmarkCaseResult {
                name: "flash_attention_prefill".to_string(),
                category: "flash_attention",
                parameters: btree_map([("seq_len", "128".to_string())]),
                tuning: btree_map([("block_q", "64".to_string())]),
                outcome: KernelBenchmarkOutcome::Completed {
                    min_ms: 1.0,
                    median_ms: 1.2,
                    mean_ms: 1.3,
                },
            }],
        };

        let json = serde_json::to_string_pretty(&report).expect("serialize");
        assert!(json.contains("\"flash_attention_prefill\""));
        assert!(json.contains("\"status\": \"completed\""));
    }

    #[test]
    fn workload_report_serializes_to_json() {
        let report = WorkloadBenchmarkReport {
            version: "test".to_string(),
            generated_at_unix_ms: 1,
            device: KernelBenchmarkDevice {
                name: "Apple M4 Max".to_string(),
                tier: "max".to_string(),
                architecture_gen: 9,
                gpu_core_count: 40,
                ane_core_count: 16,
                has_nax: false,
                is_apple10_or_newer: false,
                is_ultra_fusion: false,
                memory_bandwidth_gbps: 546.0,
                memory_bandwidth_source: "measured_gpu_copy".to_string(),
            },
            workload: WorkloadBenchmarkConfig {
                preset: Some("dense-qwen3".to_string()),
                model_id: "Qwen/Qwen3-0.6B".to_string(),
                dataset_id: "TeichAI/gemini-3-pro-preview-high-reasoning-250x".to_string(),
                experts_dir: None,
                resolved_model_path: "/tmp/model".to_string(),
                resolved_dataset_path: "/tmp/dataset".to_string(),
                prompt_samples: 8,
                max_prompt_tokens: 768,
                inference_prompt_len: WorkloadPromptLenSelection {
                    context_source: "prompt".to_string(),
                    context_source_reason: "auto-prompt-p95-ge-64".to_string(),
                    requested_max_prompt_tokens: 0,
                    effective_max_prompt_tokens: 768,
                    max_prompt_tokens_source: "auto-sampled-p95-capped-1024".to_string(),
                    sample_median_prompt_tokens: 602,
                    sample_p95_prompt_tokens: 741,
                    sample_max_prompt_tokens: 780,
                    sample_truncated_pct: 12.5,
                },
                decode_steps: 32,
                inference_warmup_passes: 2,
                inference_session_repeats: 3,
                inference_repeats: 3,
                train_samples: 8,
                train_steps: 4,
                batch_size: 1,
                max_seq_len: 1792,
                training_seq_len: WorkloadTrainingSeqLenSelection {
                    requested_max_seq_len: 0,
                    effective_max_seq_len: 1792,
                    max_seq_len_source: "auto-sampled-p95-capped-2048".to_string(),
                    sample_median_tokens: 1721,
                    sample_p95_tokens: 1770,
                    sample_max_tokens: 1770,
                    sample_truncated_pct: 0.0,
                },
            },
            inference: WorkloadBenchmarkSection::Completed(InferenceWorkloadMetrics {
                session_runs: 3,
                prompt_samples: 8,
                measurement_passes: 24,
                warmup_passes: 16,
                prompt_tokens: 12288,
                max_prompt_tokens: 768,
                decode_steps: 32,
                inference_repeats: 3,
                cache_mode: "fp16".to_string(),
                cache_mode_source: "auto-fp16".to_string(),
                first_generated_token_id: Some(8160),
                first_generated_token_text: Some("Here".to_string()),
                total_prefill_ms: 300.0,
                prefill_tok_per_sec: 40960.0,
                mean_prefill_tok_per_sec: 40480.0,
                median_prefill_tok_per_sec: 41000.0,
                total_decode_ms: 240.0,
                decode_tok_per_sec: 3200.0,
                decode_ms_per_token: 2.5,
                mean_decode_tok_per_sec: 3190.0,
                median_decode_tok_per_sec: 3210.0,
                mean_decode_ms_per_token: 2.52,
                median_decode_ms_per_token: 2.49,
                mean_session_prefill_tok_per_sec: 40800.0,
                median_session_prefill_tok_per_sec: 40900.0,
                mean_session_decode_tok_per_sec: 3185.0,
                median_session_decode_tok_per_sec: 3205.0,
                mean_session_decode_ms_per_token: 2.53,
                median_session_decode_ms_per_token: 2.50,
                session_prefill_tok_per_sec: vec![40600.0, 40900.0, 40940.0],
                session_decode_tok_per_sec: vec![3170.0, 3205.0, 3180.0],
                session_decode_ms_per_token: vec![2.52, 2.50, 2.57],
                prefetch_hits: None,
                prefetch_misses: None,
                prefetch_total: None,
                prefetch_hit_rate: None,
            }),
            training: WorkloadBenchmarkSection::Completed(TrainingWorkloadMetrics {
                train_samples: 8,
                train_steps: 4,
                batch_size: 1,
                max_seq_len: 1792,
                total_tokens: 2048,
                total_steps: 4,
                wall_clock_ms: 1200.0,
                median_tok_sec: 1800.0,
                mean_tok_sec: 1750.0,
                median_step_ms: 45.0,
                mean_step_ms: 47.0,
                final_loss: 1.23,
            }),
        };

        let json = serde_json::to_string_pretty(&report).expect("serialize");
        assert!(json.contains("\"Qwen/Qwen3-0.6B\""));
        assert!(json.contains("\"preset\": \"dense-qwen3\""));
        assert!(json.contains("\"prefill_tok_per_sec\""));
        assert!(json.contains("\"median_step_ms\""));
        assert!(json.contains("\"inference_prompt_len\""));
        assert!(json.contains("\"training_seq_len\""));
        assert!(json.contains("\"first_generated_token_id\": 8160"));
        assert!(json.contains("\"session_runs\": 3"));
        assert!(json.contains("\"inference_session_repeats\": 3"));
        assert!(json.contains("\"effective_max_seq_len\": 1792"));
        assert!(json.contains("\"effective_max_prompt_tokens\": 768"));
    }

    #[test]
    fn gdn_decode_report_serializes_to_json() {
        let report = GdnDecodeBenchmarkReport {
            version: "test".to_string(),
            generated_at_unix_ms: 1,
            device: KernelBenchmarkDevice {
                name: "Apple M4 Max".to_string(),
                tier: "max".to_string(),
                architecture_gen: 9,
                gpu_core_count: 40,
                ane_core_count: 16,
                has_nax: false,
                is_apple10_or_newer: false,
                is_ultra_fusion: false,
                memory_bandwidth_gbps: 546.0,
                memory_bandwidth_source: "measured_gpu_copy".to_string(),
            },
            stage: "input-proj".to_string(),
            model_id: "unsloth/Qwen3.5-0.8B".to_string(),
            resolved_model_path: "/tmp/model".to_string(),
            layer_idx: 0,
            batch_size: 1,
            seq_len: 1,
            input_dim: 2048,
            output_dim: 2320,
            warmup_iterations: 10,
            benchmark_iterations: 50,
            reference_backend: "mlx_split".to_string(),
            backends: vec![GdnDecodeBackendResult {
                name: "accelerate_combined".to_string(),
                max_abs_diff_vs_reference: Some(1e-6),
                outcome: KernelBenchmarkOutcome::Completed {
                    min_ms: 0.10,
                    median_ms: 0.12,
                    mean_ms: 0.13,
                },
            }],
        };

        let json = serde_json::to_string_pretty(&report).expect("serialize");
        assert!(json.contains("\"stage\": \"input-proj\""));
        assert!(json.contains("\"reference_backend\": \"mlx_split\""));
        assert!(json.contains("\"accelerate_combined\""));
        assert!(json.contains("\"max_abs_diff_vs_reference\""));
    }

    #[test]
    fn accelerate_combined_projection_matches_mlx_reference() {
        let batch_size = 1usize;
        let seq_len = 1usize;
        let hidden_size = 8usize;
        let total_output_dim = 13usize;
        let input_data: Vec<f32> = (0..batch_size * seq_len * hidden_size)
            .map(deterministic_value)
            .collect();
        let weight_data: Vec<f32> = (0..total_output_dim * hidden_size)
            .map(|index| deterministic_value(index + 97))
            .collect();
        let input = Array::from_slice(
            &input_data,
            &[batch_size as i32, seq_len as i32, hidden_size as i32],
        );
        let weight =
            Array::from_slice(&weight_data, &[total_output_dim as i32, hidden_size as i32]);

        let reference = mlx_linear_projection(&input, &weight).expect("mlx projection");
        reference.eval().expect("eval");
        let reference_data = reference.as_slice::<f32>().to_vec();
        let accelerate = accelerate_combined_projection(
            &input_data,
            &weight_data,
            batch_size * seq_len,
            hidden_size,
            total_output_dim,
        );

        assert!(max_abs_diff(&reference_data, &accelerate) < 1e-4);
    }

    #[test]
    fn mlx_combined_split_projection_matches_reference_layout() {
        let batch_size = 1usize;
        let seq_len = 1usize;
        let hidden_size = 8usize;
        let conv_dim = 8usize;
        let value_dim = 3usize;
        let num_v_heads = 1usize;
        let head_v_dim = 3usize;
        let total_output_dim = conv_dim + value_dim + (num_v_heads * 2);
        let input_data: Vec<f32> = (0..batch_size * seq_len * hidden_size)
            .map(deterministic_value)
            .collect();
        let weight_data: Vec<f32> = (0..total_output_dim * hidden_size)
            .map(|index| deterministic_value(index + 211))
            .collect();
        let input = Array::from_slice(
            &input_data,
            &[batch_size as i32, seq_len as i32, hidden_size as i32],
        );
        let combined_weight =
            Array::from_slice(&weight_data, &[total_output_dim as i32, hidden_size as i32]);

        let reference = mlx_linear_projection(&input, &combined_weight).expect("mlx projection");
        reference.eval().expect("eval");
        let (qkv, z, b_val, a) = mlx_combined_split_projection(
            &input,
            &combined_weight,
            batch_size,
            seq_len,
            conv_dim,
            value_dim,
            num_v_heads,
            head_v_dim,
        )
        .expect("split projection");
        let z_flat = z
            .reshape(&[batch_size as i32, seq_len as i32, value_dim as i32])
            .expect("reshape");
        let stitched = ops::concatenate_axis(&[&qkv, &z_flat, &b_val, &a], -1).expect("concat");
        stitched.eval().expect("eval");

        assert!(max_abs_diff(reference.as_slice::<f32>(), stitched.as_slice::<f32>()) < 1e-4);
    }

    #[test]
    fn mlx_qkv_z_combined_split_projection_matches_reference_layout() {
        let batch_size = 1usize;
        let seq_len = 1usize;
        let hidden_size = 8usize;
        let conv_dim = 8usize;
        let value_dim = 3usize;
        let num_v_heads = 1usize;
        let head_v_dim = 3usize;
        let total_output_dim = conv_dim + value_dim + (num_v_heads * 2);
        let input_data: Vec<f32> = (0..batch_size * seq_len * hidden_size)
            .map(deterministic_value)
            .collect();
        let qkv_z_weight_data: Vec<f32> = (0..(conv_dim + value_dim) * hidden_size)
            .map(|index| deterministic_value(index + 263))
            .collect();
        let b_weight_data: Vec<f32> = (0..num_v_heads * hidden_size)
            .map(|index| deterministic_value(index + 911))
            .collect();
        let a_weight_data: Vec<f32> = (0..num_v_heads * hidden_size)
            .map(|index| deterministic_value(index + 1223))
            .collect();
        let input = Array::from_slice(
            &input_data,
            &[batch_size as i32, seq_len as i32, hidden_size as i32],
        );
        let qkv_z_weight = Array::from_slice(
            &qkv_z_weight_data,
            &[(conv_dim + value_dim) as i32, hidden_size as i32],
        );
        let b_weight = Array::from_slice(&b_weight_data, &[num_v_heads as i32, hidden_size as i32]);
        let a_weight = Array::from_slice(&a_weight_data, &[num_v_heads as i32, hidden_size as i32]);

        let reference = mlx_split_projection(
            &input,
            &qkv_z_weight.index((..conv_dim as i32, ..)),
            &qkv_z_weight.index((conv_dim as i32.., ..)),
            &b_weight,
            &a_weight,
        )
        .expect("mlx split");
        reference.eval().expect("eval");
        let (qkv, z, b_val, a) = mlx_qkv_z_combined_split_projection(
            &input,
            &qkv_z_weight,
            &b_weight,
            &a_weight,
            batch_size,
            seq_len,
            conv_dim,
            value_dim,
            num_v_heads,
            head_v_dim,
        )
        .expect("split projection");
        let z_flat = z
            .reshape(&[batch_size as i32, seq_len as i32, value_dim as i32])
            .expect("reshape");
        let stitched = ops::concatenate_axis(&[&qkv, &z_flat, &b_val, &a], -1).expect("concat");
        stitched.eval().expect("eval");

        assert!(max_abs_diff(reference.as_slice::<f32>(), stitched.as_slice::<f32>()) < 1e-4);
        assert_eq!(
            stitched.shape(),
            &[batch_size as i32, seq_len as i32, total_output_dim as i32]
        );
    }

    #[test]
    fn mlx_split_projection_rhs_transposed_matches_reference_layout() {
        let batch_size = 1usize;
        let seq_len = 1usize;
        let hidden_size = 8usize;
        let conv_dim = 8usize;
        let value_dim = 3usize;
        let num_v_heads = 1usize;
        let total_output_dim = conv_dim + value_dim + (num_v_heads * 2);
        let input_data: Vec<f32> = (0..batch_size * seq_len * hidden_size)
            .map(deterministic_value)
            .collect();
        let qkv_weight_data: Vec<f32> = (0..conv_dim * hidden_size)
            .map(|index| deterministic_value(index + 521))
            .collect();
        let z_weight_data: Vec<f32> = (0..value_dim * hidden_size)
            .map(|index| deterministic_value(index + 683))
            .collect();
        let b_weight_data: Vec<f32> = (0..num_v_heads * hidden_size)
            .map(|index| deterministic_value(index + 811))
            .collect();
        let a_weight_data: Vec<f32> = (0..num_v_heads * hidden_size)
            .map(|index| deterministic_value(index + 947))
            .collect();
        let input = Array::from_slice(
            &input_data,
            &[batch_size as i32, seq_len as i32, hidden_size as i32],
        );
        let qkv_weight =
            Array::from_slice(&qkv_weight_data, &[conv_dim as i32, hidden_size as i32]);
        let z_weight = Array::from_slice(&z_weight_data, &[value_dim as i32, hidden_size as i32]);
        let b_weight = Array::from_slice(&b_weight_data, &[num_v_heads as i32, hidden_size as i32]);
        let a_weight = Array::from_slice(&a_weight_data, &[num_v_heads as i32, hidden_size as i32]);

        let reference = mlx_split_projection(&input, &qkv_weight, &z_weight, &b_weight, &a_weight)
            .expect("mlx split");
        reference.eval().expect("eval");
        let projected = mlx_split_projection_rhs_transposed(
            &input,
            &qkv_weight.t(),
            &z_weight.t(),
            &b_weight.t(),
            &a_weight.t(),
        )
        .expect("transposed split");
        projected.eval().expect("eval");

        assert!(max_abs_diff(reference.as_slice::<f32>(), projected.as_slice::<f32>()) < 1e-4);
        assert_eq!(
            projected.shape(),
            &[batch_size as i32, seq_len as i32, total_output_dim as i32]
        );
    }

    #[test]
    fn mlx_combined_split_projection_rhs_transposed_matches_reference_layout() {
        let batch_size = 1usize;
        let seq_len = 1usize;
        let hidden_size = 8usize;
        let conv_dim = 8usize;
        let value_dim = 3usize;
        let num_v_heads = 1usize;
        let head_v_dim = 3usize;
        let total_output_dim = conv_dim + value_dim + (num_v_heads * 2);
        let input_data: Vec<f32> = (0..batch_size * seq_len * hidden_size)
            .map(deterministic_value)
            .collect();
        let weight_data: Vec<f32> = (0..total_output_dim * hidden_size)
            .map(|index| deterministic_value(index + 1051))
            .collect();
        let input = Array::from_slice(
            &input_data,
            &[batch_size as i32, seq_len as i32, hidden_size as i32],
        );
        let combined_weight =
            Array::from_slice(&weight_data, &[total_output_dim as i32, hidden_size as i32]);

        let reference = mlx_linear_projection(&input, &combined_weight).expect("mlx projection");
        reference.eval().expect("eval");
        let (qkv, z, b_val, a) = mlx_combined_split_projection_rhs_transposed(
            &input,
            &combined_weight.t(),
            batch_size,
            seq_len,
            conv_dim,
            value_dim,
            num_v_heads,
            head_v_dim,
        )
        .expect("split projection");
        let z_flat = z
            .reshape(&[batch_size as i32, seq_len as i32, value_dim as i32])
            .expect("reshape");
        let stitched = ops::concatenate_axis(&[&qkv, &z_flat, &b_val, &a], -1).expect("concat");
        stitched.eval().expect("eval");

        assert!(max_abs_diff(reference.as_slice::<f32>(), stitched.as_slice::<f32>()) < 1e-4);
    }

    #[test]
    fn mlx_qkv_z_combined_split_projection_rhs_transposed_matches_reference_layout() {
        let batch_size = 1usize;
        let seq_len = 1usize;
        let hidden_size = 8usize;
        let conv_dim = 8usize;
        let value_dim = 3usize;
        let num_v_heads = 1usize;
        let head_v_dim = 3usize;
        let input_data: Vec<f32> = (0..batch_size * seq_len * hidden_size)
            .map(deterministic_value)
            .collect();
        let qkv_z_weight_data: Vec<f32> = (0..(conv_dim + value_dim) * hidden_size)
            .map(|index| deterministic_value(index + 1171))
            .collect();
        let b_weight_data: Vec<f32> = (0..num_v_heads * hidden_size)
            .map(|index| deterministic_value(index + 1301))
            .collect();
        let a_weight_data: Vec<f32> = (0..num_v_heads * hidden_size)
            .map(|index| deterministic_value(index + 1459))
            .collect();
        let input = Array::from_slice(
            &input_data,
            &[batch_size as i32, seq_len as i32, hidden_size as i32],
        );
        let qkv_z_weight = Array::from_slice(
            &qkv_z_weight_data,
            &[(conv_dim + value_dim) as i32, hidden_size as i32],
        );
        let b_weight = Array::from_slice(&b_weight_data, &[num_v_heads as i32, hidden_size as i32]);
        let a_weight = Array::from_slice(&a_weight_data, &[num_v_heads as i32, hidden_size as i32]);

        let reference = mlx_split_projection(
            &input,
            &qkv_z_weight.index((..conv_dim as i32, ..)),
            &qkv_z_weight.index((conv_dim as i32.., ..)),
            &b_weight,
            &a_weight,
        )
        .expect("mlx split");
        reference.eval().expect("eval");
        let (qkv, z, b_val, a) = mlx_qkv_z_combined_split_projection_rhs_transposed(
            &input,
            &qkv_z_weight.t(),
            &b_weight.t(),
            &a_weight.t(),
            batch_size,
            seq_len,
            conv_dim,
            value_dim,
            num_v_heads,
            head_v_dim,
        )
        .expect("split projection");
        let z_flat = z
            .reshape(&[batch_size as i32, seq_len as i32, value_dim as i32])
            .expect("reshape");
        let stitched = ops::concatenate_axis(&[&qkv, &z_flat, &b_val, &a], -1).expect("concat");
        stitched.eval().expect("eval");

        assert!(max_abs_diff(reference.as_slice::<f32>(), stitched.as_slice::<f32>()) < 1e-4);
    }

    #[test]
    fn mlx_linear_projection_rhs_transposed_matches_reference() {
        let batch_size = 1usize;
        let seq_len = 1usize;
        let input_dim = 6usize;
        let output_dim = 10usize;
        let input_data: Vec<f32> = (0..batch_size * seq_len * input_dim)
            .map(deterministic_value)
            .collect();
        let weight_data: Vec<f32> = (0..output_dim * input_dim)
            .map(|index| deterministic_value(index + 1601))
            .collect();
        let input = Array::from_slice(
            &input_data,
            &[batch_size as i32, seq_len as i32, input_dim as i32],
        );
        let weight = Array::from_slice(&weight_data, &[output_dim as i32, input_dim as i32]);

        let reference = mlx_linear_projection(&input, &weight).expect("mlx projection");
        reference.eval().expect("eval");
        let projected =
            mlx_linear_projection_rhs_transposed(&input, &weight.t()).expect("transposed linear");
        projected.eval().expect("eval");

        assert!(max_abs_diff(reference.as_slice::<f32>(), projected.as_slice::<f32>()) < 1e-4);
    }

    #[test]
    fn accelerate_roundtrip_split_projection_matches_reference_layout() {
        let batch_size = 1usize;
        let seq_len = 1usize;
        let hidden_size = 8usize;
        let conv_dim = 8usize;
        let value_dim = 3usize;
        let num_v_heads = 1usize;
        let head_v_dim = 3usize;
        let total_output_dim = conv_dim + value_dim + (num_v_heads * 2);
        let input_data: Vec<f32> = (0..batch_size * seq_len * hidden_size)
            .map(deterministic_value)
            .collect();
        let weight_data: Vec<f32> = (0..total_output_dim * hidden_size)
            .map(|index| deterministic_value(index + 307))
            .collect();
        let input = Array::from_slice(
            &input_data,
            &[batch_size as i32, seq_len as i32, hidden_size as i32],
        );
        let weight =
            Array::from_slice(&weight_data, &[total_output_dim as i32, hidden_size as i32]);

        let reference = mlx_linear_projection(&input, &weight).expect("mlx projection");
        reference.eval().expect("eval");
        let (qkv, z, b_val, a) = accelerate_roundtrip_split_projection(
            &input,
            &weight_data,
            batch_size,
            seq_len,
            hidden_size,
            total_output_dim,
            conv_dim,
            value_dim,
            num_v_heads,
            head_v_dim,
        )
        .expect("accelerate roundtrip");
        let z_flat = z
            .reshape(&[batch_size as i32, seq_len as i32, value_dim as i32])
            .expect("reshape");
        let stitched = ops::concatenate_axis(&[&qkv, &z_flat, &b_val, &a], -1).expect("concat");
        stitched.eval().expect("eval");

        assert!(max_abs_diff(reference.as_slice::<f32>(), stitched.as_slice::<f32>()) < 1e-4);
    }

    #[test]
    fn accelerate_roundtrip_linear_projection_matches_reference() {
        let batch_size = 1usize;
        let seq_len = 1usize;
        let input_dim = 6usize;
        let output_dim = 10usize;
        let input_data: Vec<f32> = (0..batch_size * seq_len * input_dim)
            .map(deterministic_value)
            .collect();
        let weight_data: Vec<f32> = (0..output_dim * input_dim)
            .map(|index| deterministic_value(index + 401))
            .collect();
        let input = Array::from_slice(
            &input_data,
            &[batch_size as i32, seq_len as i32, input_dim as i32],
        );
        let weight = Array::from_slice(&weight_data, &[output_dim as i32, input_dim as i32]);

        let reference = mlx_linear_projection(&input, &weight).expect("mlx projection");
        reference.eval().expect("eval");
        let projected = accelerate_roundtrip_linear_projection(
            &input,
            &weight_data,
            batch_size,
            seq_len,
            input_dim,
            output_dim,
        )
        .expect("roundtrip");
        projected.eval().expect("eval");

        assert!(max_abs_diff(reference.as_slice::<f32>(), projected.as_slice::<f32>()) < 1e-4);
    }

    #[test]
    fn hybrid_qwen3next_preset_uses_conservative_training_shape() {
        let preset = workload_preset_config(WorkloadBenchmarkPreset::HybridQwen3Next);

        assert_eq!(preset.model_id, "unsloth/Qwen3.5-0.8B");
        assert!(matches!(
            preset.inference_context,
            WorkloadInferenceContext::TextPrefix
        ));
        assert_eq!(preset.inference_repeats, 1);
        assert_eq!(preset.max_seq_len, 0);
        assert_eq!(preset.train_steps, 0);
        assert_eq!(preset.train_samples, 0);
    }

    #[test]
    fn hybrid_qwen35_steady_preset_prefers_longer_decode_measurement() {
        let preset = workload_preset_config(WorkloadBenchmarkPreset::HybridQwen35Steady);

        assert_eq!(preset.model_id, "unsloth/Qwen3.5-0.8B");
        assert!(matches!(
            preset.inference_context,
            WorkloadInferenceContext::TextPrefix
        ));
        assert_eq!(preset.prompt_samples, 2);
        assert_eq!(preset.decode_steps, 64);
        assert_eq!(preset.inference_repeats, 3);
        assert_eq!(preset.train_steps, 0);
        assert_eq!(preset.train_samples, 0);
    }

    #[test]
    fn moe_nemotronh_preset_is_inference_only() {
        let preset = workload_preset_config(WorkloadBenchmarkPreset::MoeNemotronH);

        assert_eq!(preset.model_id, "unsloth/NVIDIA-Nemotron-3-Nano-4B");
        assert!(matches!(
            preset.inference_context,
            WorkloadInferenceContext::TextPrefix
        ));
        assert_eq!(preset.max_prompt_tokens, 512);
        assert_eq!(preset.decode_steps, 4);
        assert_eq!(preset.inference_repeats, 1);
        assert_eq!(preset.train_steps, 0);
        assert_eq!(preset.train_samples, 0);
    }

    #[test]
    fn workload_model_max_seq_len_reads_config_fields() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"text_config":{"max_position_embeddings":4096}}"#,
        )
        .expect("write config");

        assert_eq!(workload_model_max_seq_len(dir.path()), Some(4096));
        assert_eq!(workload_auto_training_seq_len_cap(dir.path()), 2048);
        assert_eq!(workload_auto_inference_prompt_len_cap(dir.path(), 32), 1024);
    }

    #[test]
    fn summarize_prompt_lengths_tracks_percentiles_and_truncation() {
        let summary = summarize_prompt_lengths(&[128, 256, 384, 640, 1024], 512);

        assert_eq!(summary.sample_median_prompt_tokens, 384);
        assert_eq!(summary.sample_p95_prompt_tokens, 1024);
        assert_eq!(summary.sample_max_prompt_tokens, 1024);
        assert_eq!(summary.sample_truncated_pct, 40.0);
        assert_eq!(summary.effective_max_prompt_tokens, 512);
    }

    #[test]
    fn auto_inference_context_promotes_short_prompts_to_text_prefix() {
        let (context, reason) =
            resolve_requested_workload_inference_context(WorkloadInferenceContext::Auto, 28);

        assert!(matches!(
            context,
            ResolvedWorkloadInferenceContext::TextPrefix
        ));
        assert!(reason.contains("auto-short-prompt-p95-28-lt-64"));
    }

    #[test]
    fn explicit_inference_context_is_respected() {
        let (prompt_context, prompt_reason) =
            resolve_requested_workload_inference_context(WorkloadInferenceContext::Prompt, 1);
        let (text_prefix_context, text_prefix_reason) =
            resolve_requested_workload_inference_context(WorkloadInferenceContext::TextPrefix, 999);

        assert!(matches!(
            prompt_context,
            ResolvedWorkloadInferenceContext::Prompt
        ));
        assert_eq!(prompt_reason, "user-prompt");
        assert!(matches!(
            text_prefix_context,
            ResolvedWorkloadInferenceContext::TextPrefix
        ));
        assert_eq!(text_prefix_reason, "user-text-prefix");
    }

    #[test]
    fn sampled_workload_dataset_writes_prompt_column() {
        let output = NamedTempFile::new().expect("temp output");
        let samples = vec![
            TextSample {
                text: "prompt\n\nresponse".to_string(),
                prompt: Some("prompt".to_string()),
            },
            TextSample {
                text: "plain text".to_string(),
                prompt: None,
            },
        ];

        write_sampled_workload_dataset(&samples, output.path()).expect("write dataset");
        let content = std::fs::read_to_string(output.path()).expect("read dataset");

        assert!(content.contains("\"prompt\":\"prompt\""));
        assert!(content.contains("\"text\":\"plain text\""));
    }

    #[test]
    #[ignore = "requires local Metal hardware and unsandboxed cargo test"]
    fn benchmark_corpus_smoke_writes_json_report() {
        let output = NamedTempFile::new().expect("temp output");
        run_kernel_benchmark_corpus(true, Some(output.path()), false).expect("benchmark corpus");

        let report_json = std::fs::read_to_string(output.path()).expect("read report");
        let report: serde_json::Value = serde_json::from_str(&report_json).expect("parse report");

        assert_eq!(report["mode"], "quick");
        assert!(report["summary"]["completed"].as_u64().unwrap_or(0) > 0);
        assert!(
            report["cases"]
                .as_array()
                .is_some_and(|cases| !cases.is_empty())
        );
    }
}
