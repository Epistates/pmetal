use anyhow::Context;
use crate::{WorkloadBenchmarkPreset, WorkloadInferenceContext};
use half::f16;
use mlx_rs::module::ModuleParameters as _;
use pmetal::inference_runner::{CacheModeRequest, select_cache_mode_for_model};
use pmetal_core::{DatasetConfig, LoraConfig, ModelConfig, StepMetrics, TrainingCallback, TrainingConfig};
use pmetal_data::{DatasetColumnConfig, DatasetFormat, TextSample, Tokenizer, TrainingDataset};
use pmetal_lora::LlamaLoraForCausalLM;
use pmetal_metal::context::{DeviceTier, MemoryBandwidthSource};
use pmetal_metal::kernels::BatchedCommandBuffer;
use pmetal_metal::kernels::mpp_gemm::{MppGemm, MppGemmConfig};
use pmetal_metal::tuna::MppGemmTuneRequest;
use pmetal_mlx::kv_cache::CacheMode;
use pmetal_metal::{
    BufferUsage, FlashAttention, FlashAttentionConfig, FusedLinearCrossEntropy,
    FusedLinearCrossEntropyConfig, FusedLora, FusedLoraConfig, FusedMLP, FusedMergeMetal,
    FusedNormLora, FusedNormLoraConfig, FusedSwiGLUConfig, MetalBuffer, MetalContext,
    build_merge_config, build_tensor_info,
};
use pmetal_models::architectures::deepseek::{DeepSeekConfig, DeepSeekMoE};
use pmetal_models::architectures::jamba::{JambaConfig, JambaLayer};
use pmetal_models::dispatcher::DynamicModel;
use pmetal_models::architectures::llama::LlamaConfig;
use pmetal_models::architectures::llama4::{Llama4MoE, Llama4TextConfig};
use pmetal_models::architectures::qwen3_moe::{Qwen3MoEBlock, Qwen3MoEConfig};
use pmetal_trainer::orchestrator::{FullTrainingConfig, run_training};
use pmetal_trainer::{DispatchConfig, TrainingJobConfig};
use serde::Serialize;
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
    let input_ids = mlx_rs::Array::zeros::<i32>(&[batch_size as i32, seq_len as i32])?;

    // Warmup
    println!("Warming up...");
    for _ in 0..3 {
        let output = model_inst.forward(&input_ids, None)?;
        output.eval()?;
    }

    // Benchmark
    let iterations = 10;
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        let output = model_inst.forward(&input_ids, None)?;
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

#[derive(Debug, Clone, Serialize)]
pub(crate) struct WorkloadBenchmarkReport {
    version: String,
    generated_at_unix_ms: u128,
    device: KernelBenchmarkDevice,
    workload: WorkloadBenchmarkConfig,
    inference: WorkloadBenchmarkSection<InferenceWorkloadMetrics>,
    training: WorkloadBenchmarkSection<TrainingWorkloadMetrics>,
}

#[derive(Debug, Clone, Serialize)]
struct WorkloadBenchmarkConfig {
    preset: Option<String>,
    model_id: String,
    dataset_id: String,
    resolved_model_path: String,
    resolved_dataset_path: String,
    prompt_samples: usize,
    max_prompt_tokens: usize,
    inference_prompt_len: WorkloadPromptLenSelection,
    decode_steps: usize,
    train_samples: usize,
    train_steps: usize,
    batch_size: usize,
    max_seq_len: usize,
    training_seq_len: WorkloadTrainingSeqLenSelection,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum WorkloadBenchmarkSection<T> {
    Completed(T),
    Skipped { reason: String },
    Failed { error: String },
}

#[derive(Debug, Clone, Serialize)]
struct InferenceWorkloadMetrics {
    prompt_samples: usize,
    prompt_tokens: usize,
    max_prompt_tokens: usize,
    decode_steps: usize,
    cache_mode: String,
    cache_mode_source: String,
    total_prefill_ms: f64,
    prefill_tok_per_sec: f64,
    total_decode_ms: f64,
    decode_tok_per_sec: f64,
    decode_ms_per_token: f64,
}

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
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
            train_samples: 4,
            train_steps: 2,
            batch_size: 1,
            max_seq_len: 0,
        },
        WorkloadBenchmarkPreset::HybridQwen3Next => WorkloadPresetConfig {
            preset,
            model_id: "unsloth/Qwen3.5-0.8B-Base",
            dataset_id: "TeichAI/gemini-3-pro-preview-high-reasoning-250x",
            prompt_samples: 4,
            max_prompt_tokens: 0,
            inference_context: WorkloadInferenceContext::TextPrefix,
            decode_steps: 8,
            train_samples: 0,
            train_steps: 0,
            batch_size: 1,
            max_seq_len: 0,
        },
    }
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

#[derive(Debug, Clone, Serialize)]
struct KernelBenchmarkDevice {
    name: String,
    tier: &'static str,
    architecture_gen: u32,
    gpu_core_count: u32,
    ane_core_count: u32,
    has_nax: bool,
    is_apple10_or_newer: bool,
    is_ultra_fusion: bool,
    memory_bandwidth_gbps: f64,
    memory_bandwidth_source: &'static str,
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
            tier: device_tier_label(props.device_tier),
            architecture_gen: props.architecture_gen,
            gpu_core_count: props.gpu_core_count,
            ane_core_count: props.ane_core_count,
            has_nax: props.has_nax(),
            is_apple10_or_newer: props.is_apple10_or_newer(),
            is_ultra_fusion: props.is_ultra_fusion,
            memory_bandwidth_gbps: props.memory_bandwidth_gbps,
            memory_bandwidth_source: memory_bandwidth_source_label(props.memory_bandwidth_source),
        },
        summary,
        cases: results,
    };

    let report_json = serde_json::to_string_pretty(&report)?;
    if let Some(output_path) = output {
        std::fs::write(output_path, &report_json)
            .with_context(|| format!("failed to write benchmark corpus to {}", output_path.display()))?;
    }

    if json {
        println!("{report_json}");
    } else {
        print_kernel_benchmark_report(&report, output);
    }

    Ok(())
}

pub(crate) async fn run_workload_benchmark(
    model_id: &str,
    dataset_id: &str,
    prompt_samples: usize,
    max_prompt_tokens: usize,
    inference_context: WorkloadInferenceContext,
    decode_steps: usize,
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
        prompt_samples,
        max_prompt_tokens,
        inference_context,
        decode_steps,
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
    output: Option<&Path>,
    json: bool,
) -> anyhow::Result<()> {
    let config = workload_preset_config(preset);
    run_workload_benchmark_internal(
        Some(preset_label(config.preset).to_string()),
        config.model_id,
        config.dataset_id,
        config.prompt_samples,
        config.max_prompt_tokens,
        config.inference_context,
        config.decode_steps,
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
    prompt_samples: usize,
    max_prompt_tokens: usize,
    inference_context: WorkloadInferenceContext,
    decode_steps: usize,
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
    let chat_template =
        pmetal_data::chat_templates::detect_chat_template(&model_path, model_id);
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

    let inference_prompt_len =
        resolve_workload_inference_prompt_len(
            &model_path,
            &selected_inference_samples,
            max_prompt_tokens,
            inference_context,
            decode_steps,
        );
    let inference = benchmark_real_inference(
        &model_path,
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
            resolved_model_path: model_path.display().to_string(),
            resolved_dataset_path: dataset_path.display().to_string(),
            prompt_samples,
            max_prompt_tokens: inference_prompt_len.effective_max_prompt_tokens,
            inference_prompt_len,
            decode_steps,
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
        std::fs::write(output_path, &report_json)
            .with_context(|| format!("failed to write workload benchmark to {}", output_path.display()))?;
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
        tier: device_tier_label(props.device_tier),
        architecture_gen: props.architecture_gen,
        gpu_core_count: props.gpu_core_count,
        ane_core_count: props.ane_core_count,
        has_nax: props.has_nax(),
        is_apple10_or_newer: props.is_apple10_or_newer(),
        is_ultra_fusion: props.is_ultra_fusion,
        memory_bandwidth_gbps: props.memory_bandwidth_gbps,
        memory_bandwidth_source: memory_bandwidth_source_label(props.memory_bandwidth_source),
    })
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

fn benchmark_real_inference(
    model_path: &Path,
    samples: &[TextSample],
    max_prompt_tokens: usize,
    inference_context: ResolvedWorkloadInferenceContext,
    decode_steps: usize,
) -> anyhow::Result<WorkloadBenchmarkSection<InferenceWorkloadMetrics>> {
    if samples.is_empty() {
        return Ok(WorkloadBenchmarkSection::Skipped {
            reason: "no non-empty prompt samples available".to_string(),
        });
    }

    let tokenizer = Tokenizer::from_model_dir(model_path)?;
    let mut model = DynamicModel::load(model_path)?;
    let cache_max_seq_len = max_prompt_tokens.saturating_add(decode_steps).saturating_add(8);
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
            no_kv_quant: false,
            fp8: false,
        },
    );
    let cache_mode = cache_selection.mode;

    // Warm up model load / kernels on the first prompt so the timed run is steadier.
    let warmup_ids =
        encode_benchmark_prompt(&tokenizer, &samples[0], max_prompt_tokens, inference_context)?;
    if !warmup_ids.is_empty() {
        let _ = run_prefill_decode_pass(&mut model, &warmup_ids, decode_steps, cache_mode)?;
    }

    let mut total_prompt_tokens = 0usize;
    let mut total_prefill = Duration::default();
    let mut total_decode = Duration::default();
    let mut measured_samples = 0usize;

    for sample in samples {
        let prompt_ids =
            encode_benchmark_prompt(&tokenizer, sample, max_prompt_tokens, inference_context)?;
        if prompt_ids.is_empty() {
            continue;
        }

        let (prefill_time, decode_time) =
            run_prefill_decode_pass(&mut model, &prompt_ids, decode_steps, cache_mode)?;
        total_prompt_tokens += prompt_ids.len();
        total_prefill += prefill_time;
        total_decode += decode_time;
        measured_samples += 1;
    }

    if measured_samples == 0 || total_prompt_tokens == 0 {
        return Ok(WorkloadBenchmarkSection::Skipped {
            reason: "all selected prompt samples tokenized to empty inputs".to_string(),
        });
    }

    let prefill_ms = duration_to_ms(total_prefill);
    let decode_ms = duration_to_ms(total_decode);
    let decode_tokens = measured_samples * decode_steps;

    Ok(WorkloadBenchmarkSection::Completed(InferenceWorkloadMetrics {
        prompt_samples: measured_samples,
        prompt_tokens: total_prompt_tokens,
        max_prompt_tokens,
        decode_steps,
        cache_mode: cache_mode.describe(),
        cache_mode_source: cache_selection.source.as_str().to_string(),
        total_prefill_ms: prefill_ms,
        prefill_tok_per_sec: if prefill_ms > 0.0 {
            total_prompt_tokens as f64 / (prefill_ms / 1000.0)
        } else {
            0.0
        },
        total_decode_ms: decode_ms,
        decode_tok_per_sec: if decode_tokens > 0 && decode_ms > 0.0 {
            decode_tokens as f64 / (decode_ms / 1000.0)
        } else {
            0.0
        },
        decode_ms_per_token: if decode_tokens > 0 {
            decode_ms / decode_tokens as f64
        } else {
            0.0
        },
    }))
}

fn encode_benchmark_prompt(
    tokenizer: &Tokenizer,
    sample: &TextSample,
    max_prompt_tokens: usize,
    inference_context: ResolvedWorkloadInferenceContext,
) -> anyhow::Result<Vec<u32>> {
    let mut ids = tokenizer.encode_with_special_tokens(
        benchmark_inference_text(sample, inference_context),
    )?;
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
) -> anyhow::Result<(Duration, Duration)> {
    use mlx_rs::Array;
    use mlx_rs::ops::indexing::{IndexOp, argmax};

    let prompt_tokens: Vec<i32> = prompt_ids.iter().map(|&id| id as i32).collect();
    let prompt = Array::from_slice(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let mut cache =
        model.create_cache_with_mode(prompt_ids.len().saturating_add(decode_steps).saturating_add(8), cache_mode);
    let mut mamba_cache = model.create_mamba_cache();

    let prefill_start = Instant::now();
    let logits = model.forward_with_hybrid_cache(
        &prompt,
        None,
        Some(&mut cache),
        mamba_cache.as_mut(),
    )?;
    logits.eval()?;
    let prefill_elapsed = prefill_start.elapsed();

    let mut next_token = argmax(&logits.index((.., -1, ..)), None)?.item::<u32>() as i32;
    let decode_start = Instant::now();
    for _ in 0..decode_steps {
        let decode_input = Array::from_slice(&[next_token], &[1, 1]);
        let decode_logits = model.forward_with_hybrid_cache(
            &decode_input,
            None,
            Some(&mut cache),
            mamba_cache.as_mut(),
        )?;
        next_token = argmax(&decode_logits.index((.., -1, ..)), None)?.item::<u32>() as i32;
    }
    let decode_elapsed = decode_start.elapsed();

    Ok((prefill_elapsed, decode_elapsed))
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

fn empty_training_seq_len_selection(requested_max_seq_len: usize) -> WorkloadTrainingSeqLenSelection {
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
                        prompt_p95_tokens,
                        AUTO_BENCH_WORKLOAD_MIN_PROMPT_TOKENS
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
    let chat_template =
        pmetal_data::chat_templates::detect_chat_template(model_path, model_id);

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

fn summarize_prompt_lengths(lengths: &[usize], max_prompt_tokens: usize) -> WorkloadPromptLenSelection {
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
            println!(
                "  Context: {} ({})",
                report.workload.inference_prompt_len.context_source,
                report.workload.inference_prompt_len.context_source_reason
            );
            println!(
                "  Prompt len: {} ({}, sample p95 {}, {:.1}% truncated)",
                report.workload.inference_prompt_len.effective_max_prompt_tokens,
                report.workload.inference_prompt_len.max_prompt_tokens_source,
                report.workload.inference_prompt_len.sample_p95_prompt_tokens,
                report.workload.inference_prompt_len.sample_truncated_pct
            );
            println!(
                "  KV cache: {} ({})",
                metrics.cache_mode,
                metrics.cache_mode_source
            );
            println!(
                "  Prefill: {:.0} tok/s over {} prompt tokens ({} samples, {:.2} ms total)",
                metrics.prefill_tok_per_sec,
                metrics.prompt_tokens,
                metrics.prompt_samples,
                metrics.total_prefill_ms
            );
            println!(
                "  Decode:  {:.0} tok/s ({:.2} ms/token over {} steps/sample)",
                metrics.decode_tok_per_sec,
                metrics.decode_ms_per_token,
                metrics.decode_steps
            );
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
                metrics.median_step_ms,
                metrics.median_tok_sec
            );
            println!(
                "  Final loss: {:.4} across {} steps / {} tokens",
                metrics.final_loss,
                metrics.total_steps,
                metrics.total_tokens
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
            let tuning = match ctx
                .tuner()
                .tune_swiglu(ctx, case.batch_size, case.hidden_size, case.intermediate_size)
            {
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
                let down_weight =
                    alloc_f32_buffer(ctx, case.hidden_size * case.intermediate_size)?;

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
            let config =
                FusedNormLoraConfig::new(case.batch_size, case.hidden_size, case.out_features, case.rank, 16.0);

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
            let tuning = match ctx
                .tuner()
                .tune_fused_linear_cross_entropy(ctx, &config)
            {
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
                    let output =
                        kernel.forward_f16(&hidden_states, &lm_head_weight, &targets)?;
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
                let input = mlx_rs::random::normal::<f32>(
                    &[case.batch_size as i32, case.seq_len as i32, case.hidden_size as i32],
                    None,
                    None,
                    None,
                )?;

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
                            output.eval()?;
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
                            output.eval()?;
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
                            output.eval()?;
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
                let input = mlx_rs::random::normal::<f32>(
                    &[case.batch_size as i32, case.seq_len as i32, case.hidden_size as i32],
                    None,
                    None,
                    None,
                )?;

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
                            output.eval()?;
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
    let data: Vec<i32> = (0..len)
        .map(|i| (i % vocab_size.max(1)) as i32)
        .collect();
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
                println!(
                    "{:<30} {:<24} failed ({})",
                    case.name, case.category, error
                );
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
    use mlx_rs::{
        Array,
        ops::indexing::{argmax, argmax_axis},
        transforms::eval,
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

    Ok(())
}

/// Benchmark generation loop timing with a real model.
///
/// This profiles each step of the generation loop to compare with mlx_lm's timing.
pub(crate) async fn run_gen_benchmark(model_id: &str) -> anyhow::Result<()> {
    use mlx_rs::{
        Array,
        ops::indexing::{IndexOp, argmax},
        transforms::{async_eval, eval},
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
        let next_token = argmax(&last_logits, None)?;
        times.entry("argmax").or_default().push(t0.elapsed());

        // Async eval for next
        let t0 = Instant::now();
        async_eval([&next_token])?;
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
                tier: "base",
                architecture_gen: 7,
                gpu_core_count: 8,
                ane_core_count: 16,
                has_nax: false,
                is_apple10_or_newer: false,
                is_ultra_fusion: false,
                memory_bandwidth_gbps: 100.0,
                memory_bandwidth_source: "spec_table_fallback",
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
                tier: "max",
                architecture_gen: 9,
                gpu_core_count: 40,
                ane_core_count: 16,
                has_nax: false,
                is_apple10_or_newer: false,
                is_ultra_fusion: false,
                memory_bandwidth_gbps: 546.0,
                memory_bandwidth_source: "measured_gpu_copy",
            },
            workload: WorkloadBenchmarkConfig {
                preset: Some("dense-qwen3".to_string()),
                model_id: "Qwen/Qwen3-0.6B".to_string(),
                dataset_id: "TeichAI/gemini-3-pro-preview-high-reasoning-250x".to_string(),
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
                prompt_samples: 8,
                prompt_tokens: 4096,
                max_prompt_tokens: 768,
                decode_steps: 32,
                cache_mode: "fp16".to_string(),
                cache_mode_source: "auto-fp16".to_string(),
                total_prefill_ms: 100.0,
                prefill_tok_per_sec: 40960.0,
                total_decode_ms: 80.0,
                decode_tok_per_sec: 3200.0,
                decode_ms_per_token: 2.5,
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
        assert!(json.contains("\"effective_max_seq_len\": 1792"));
        assert!(json.contains("\"effective_max_prompt_tokens\": 768"));
    }

    #[test]
    fn hybrid_qwen3next_preset_uses_conservative_training_shape() {
        let preset = workload_preset_config(WorkloadBenchmarkPreset::HybridQwen3Next);

        assert_eq!(preset.model_id, "unsloth/Qwen3.5-0.8B-Base");
        assert!(matches!(
            preset.inference_context,
            WorkloadInferenceContext::TextPrefix
        ));
        assert_eq!(preset.max_seq_len, 0);
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
        assert!(report["cases"].as_array().is_some_and(|cases| !cases.is_empty()));
    }
}
