use anyhow::Context;
use half::f16;
use pmetal_core::{DatasetConfig, LoraConfig, ModelConfig, TrainingConfig};
use pmetal_lora::LlamaLoraForCausalLM;
use pmetal_metal::context::{DeviceTier, MemoryBandwidthSource};
use pmetal_metal::kernels::mpp_gemm::{MppGemm, MppGemmConfig};
use pmetal_metal::tuna::MppGemmTuneRequest;
use pmetal_metal::{
    BufferUsage, FlashAttention, FlashAttentionConfig, FusedLinearCrossEntropy,
    FusedLinearCrossEntropyConfig, FusedLora, FusedLoraConfig, FusedMLP,
    FusedNormLora, FusedNormLoraConfig, FusedSwiGLUConfig, MetalBuffer, MetalContext,
};
use pmetal_models::architectures::llama::LlamaConfig;
use pmetal_trainer::orchestrator::FullTrainingConfig;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
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
    MppGemm(MppGemmCase),
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
    }

    #[test]
    fn benchmark_corpus_has_expected_categories() {
        let cases = build_benchmark_corpus_for_profile(DeviceTier::Base, false, true);
        assert_eq!(cases.len(), 6);
        assert!(matches!(cases[0], KernelBenchmarkCase::FlashAttention(_)));
        assert!(matches!(cases[1], KernelBenchmarkCase::FusedLora(_)));
        assert!(matches!(cases[2], KernelBenchmarkCase::FusedMlp(_)));
        assert!(matches!(cases[3], KernelBenchmarkCase::FusedNormLora(_)));
        assert!(matches!(
            cases[4],
            KernelBenchmarkCase::FusedLinearCrossEntropy(_)
        ));
        assert!(matches!(cases[5], KernelBenchmarkCase::MppGemm(_)));
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
