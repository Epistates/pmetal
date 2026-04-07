use std::path::{Path, PathBuf};

use anyhow::Context;
use pmetal_mlx::{Array, Exception};
use pmetal_models::{
    DynamicModel,
    architectures::{Qwen3NextForwardProfile, Qwen3NextLayerProfile},
};
use std::collections::BTreeMap;

use pmetal::response_parser::{extract_final_response, extract_thinking_content};

#[derive(Debug, serde::Serialize, PartialEq, Eq)]
struct HybridProfileSectionSummary {
    name: String,
    total_us: u64,
}

#[derive(Debug, serde::Serialize, PartialEq, Eq)]
struct HybridProfileKindSummary {
    layer_kind: String,
    layer_count: usize,
    total_us: u64,
    top_sections: Vec<HybridProfileSectionSummary>,
}

#[derive(Debug, serde::Serialize, PartialEq, Eq)]
struct HybridPhaseProfileSummary {
    layer_total_us: u64,
    non_layer_us: u64,
    layer_kind_totals: Vec<HybridProfileKindSummary>,
    top_sections: Vec<HybridProfileSectionSummary>,
}

#[derive(Debug, serde::Serialize)]
struct HybridLayerProfileReport {
    model: String,
    architecture: String,
    prompt_tokens: usize,
    decode_token_id: u32,
    decode_token_text: String,
    prefill: Qwen3NextForwardProfile,
    prefill_summary: HybridPhaseProfileSummary,
    decode: Qwen3NextForwardProfile,
    decode_summary: HybridPhaseProfileSummary,
}

fn build_qwen3_next_phase_summary(profile: &Qwen3NextForwardProfile) -> HybridPhaseProfileSummary {
    let mut by_kind: BTreeMap<String, (usize, u64, BTreeMap<String, u64>)> = BTreeMap::new();
    let mut overall_sections: BTreeMap<String, u64> = BTreeMap::new();
    let mut layer_total_us = 0u64;

    for layer in &profile.layers {
        layer_total_us += layer.total_us;
        let entry = by_kind
            .entry(layer.layer_kind.clone())
            .or_insert_with(|| (0, 0, BTreeMap::new()));
        entry.0 += 1;
        entry.1 += layer.total_us;
        for section in &layer.sections {
            *entry.2.entry(section.name.clone()).or_default() += section.elapsed_us;
            *overall_sections.entry(section.name.clone()).or_default() += section.elapsed_us;
        }
    }

    let mut layer_kind_totals: Vec<_> = by_kind
        .into_iter()
        .map(|(layer_kind, (layer_count, total_us, sections))| {
            let mut top_sections: Vec<_> = sections
                .into_iter()
                .map(|(name, total_us)| HybridProfileSectionSummary { name, total_us })
                .collect();
            top_sections.sort_by_key(|section| std::cmp::Reverse(section.total_us));
            top_sections.truncate(6);
            HybridProfileKindSummary {
                layer_kind,
                layer_count,
                total_us,
                top_sections,
            }
        })
        .collect();
    layer_kind_totals.sort_by_key(|kind| std::cmp::Reverse(kind.total_us));

    let mut top_sections: Vec<_> = overall_sections
        .into_iter()
        .map(|(name, total_us)| HybridProfileSectionSummary { name, total_us })
        .collect();
    top_sections.sort_by_key(|section| std::cmp::Reverse(section.total_us));
    top_sections.truncate(10);

    HybridPhaseProfileSummary {
        layer_total_us,
        non_layer_us: profile.total_us.saturating_sub(layer_total_us),
        layer_kind_totals,
        top_sections,
    }
}

fn print_qwen3_next_profile_summary(
    label: &str,
    profile: &Qwen3NextForwardProfile,
    summary: &HybridPhaseProfileSummary,
) {
    println!("\n=== {label} Profile ===");
    println!(
        "Total: {:.3} ms | Embed: {:.3} ms | Final norm: {:.3} ms | LM head: {:.3} ms",
        profile.total_us as f64 / 1000.0,
        profile.embedding_us as f64 / 1000.0,
        profile.final_norm_us as f64 / 1000.0,
        profile.lm_head_us as f64 / 1000.0
    );
    println!(
        "Layer total: {:.3} ms | Non-layer / glue: {:.3} ms",
        summary.layer_total_us as f64 / 1000.0,
        summary.non_layer_us as f64 / 1000.0
    );

    if !summary.layer_kind_totals.is_empty() {
        println!("By layer kind:");
        for kind in &summary.layer_kind_totals {
            println!(
                "  {:>16}: {:7.3} ms across {} layer(s)",
                kind.layer_kind,
                kind.total_us as f64 / 1000.0,
                kind.layer_count
            );
            for section in kind.top_sections.iter().take(4) {
                println!(
                    "    {:>22}: {:7.3} ms",
                    section.name,
                    section.total_us as f64 / 1000.0
                );
            }
        }
    }

    if !summary.top_sections.is_empty() {
        println!("Top sections overall:");
        for section in summary.top_sections.iter().take(8) {
            println!(
                "  {:>24}: {:7.3} ms",
                section.name,
                section.total_us as f64 / 1000.0
            );
        }
    }

    let mut layers: Vec<&Qwen3NextLayerProfile> = profile.layers.iter().collect();
    layers.sort_by_key(|layer| std::cmp::Reverse(layer.total_us));
    for layer in layers.into_iter().take(6) {
        println!(
            "Layer {:>2} {:>16}: {:7.3} ms",
            layer.layer_idx,
            layer.layer_kind,
            layer.total_us as f64 / 1000.0
        );
        for section in layer.sections.iter().take(8) {
            println!(
                "  {:>24}: {:7.3} ms",
                section.name,
                section.elapsed_us as f64 / 1000.0
            );
        }
    }
}

#[allow(clippy::needless_option_as_deref)]
fn run_qwen3_next_layer_profile(
    runner: &mut pmetal::inference_runner::InferenceRunner,
    model_id: &str,
    profile_output: Option<&Path>,
) -> anyhow::Result<()> {
    let prompt_token_ids = runner.state.input_ids().to_vec();
    let prompt_ids_i32: Vec<i32> = prompt_token_ids.iter().map(|id| *id as i32).collect();
    let prompt_input = Array::from_slice(&prompt_ids_i32, &[1, prompt_ids_i32.len() as i32]);
    let tokenizer = &runner.tokenizer;

    let (architecture, prefill_profile, decode_profile, decode_token_id) = runner
        .state
        .run_standard_model_with_state(|model, cache, mamba_cache| {
            let mut mamba_cache = mamba_cache;
            let architecture = model.architecture();
            let DynamicModel::Qwen3Next(qwen) = model else {
                return Err(Exception::custom(format!(
                    "--profile-layers currently supports Qwen 3.5 / qwen3_next only, got {architecture}"
                )));
            };

            let prefill_mamba = mamba_cache.as_deref_mut();
            let (prefill_logits, prefill_profile) = qwen.forward_with_cache_profiled(
                &prompt_input,
                None,
                Some(cache),
                prefill_mamba,
                "prefill",
            )?;

            let last_logits = pmetal_bridge::compat::ops::select_axis(&prefill_logits, -1, 1);
            let next_token = pmetal_bridge::compat::ops::argmax(&last_logits, -1);
            let next_token = next_token;
            next_token.eval();
            let decode_token_id = next_token.item::<u32>();
            let decode_input = Array::from_slice(&[decode_token_id as i32], &[1, 1]);

            let decode_mamba = mamba_cache.as_deref_mut();
            let (_decode_logits, decode_profile) = qwen.forward_with_cache_profiled(
                &decode_input,
                None,
                Some(cache),
                decode_mamba,
                "decode",
            )?;

            Ok((architecture.to_string(), prefill_profile, decode_profile, decode_token_id))
        })
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let decode_token_text = tokenizer
        .decode(&[decode_token_id])
        .unwrap_or_else(|_| "<decode failed>".to_string());
    let prefill_summary = build_qwen3_next_phase_summary(&prefill_profile);
    let decode_summary = build_qwen3_next_phase_summary(&decode_profile);
    let report = HybridLayerProfileReport {
        model: model_id.to_string(),
        architecture,
        prompt_tokens: prompt_token_ids.len(),
        decode_token_id,
        decode_token_text,
        prefill: prefill_profile,
        prefill_summary,
        decode: decode_profile,
        decode_summary,
    };

    println!("\n========================================");
    println!("  PMetal Hybrid Layer Profile");
    println!("========================================");
    println!("Model:         {}", report.model);
    println!("Architecture:  {}", report.architecture);
    println!("Prompt tokens: {}", report.prompt_tokens);
    println!(
        "Decode token:  {} ({:?})",
        report.decode_token_id, report.decode_token_text
    );
    print_qwen3_next_profile_summary("Prefill", &report.prefill, &report.prefill_summary);
    print_qwen3_next_profile_summary("Decode", &report.decode, &report.decode_summary);

    if let Some(output_path) = profile_output {
        let report_json = serde_json::to_string_pretty(&report)?;
        std::fs::write(output_path, report_json).with_context(|| {
            format!(
                "failed to write profile report to {}",
                output_path.display()
            )
        })?;
        println!("\nProfile JSON written to {}", output_path.display());
    }

    Ok(())
}

/// Run inference with a model.
///
/// Uses the shared `InferenceRunner` for model loading, tokenization, chat
/// template, sampling config, and cache creation. CLI-specific features
/// (ANE, metal sampler, compiled, benchmark) use the runner's prepared state.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_inference(
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
    _stream: bool,
    minimal: bool,
    show_thinking: bool,
    fp8: bool,
    tools: Option<&[pmetal_data::chat_templates::ToolDefinition]>,
    ane: bool,
    ane_max_seq_len: usize,
    ane_real_time: bool,
    benchmark: bool,
    benchmark_iters: usize,
    benchmark_prompt_tokens: Option<usize>,
    profile_layers: bool,
    profile_output: Option<&Path>,
    kv_quant: Option<u8>,
    kv_k_bits: Option<u8>,
    kv_v_bits: Option<u8>,
    kv_group_size: usize,
    kv_turboquant: bool,
    kv_turboquant_preset: Option<pmetal::inference_runner::TurboQuantPreset>,
    kv_quant_preset: Option<String>,
    no_kv_quant: bool,
    kv_qjl: bool,
    experts_dir: Option<&str>,
) -> anyhow::Result<()> {
    #[cfg(not(feature = "ane"))]
    if ane {
        anyhow::bail!("ANE inference requires the 'ane' feature: cargo build --features ane");
    }
    #[cfg(target_os = "macos")]
    use pmetal_models::generate_cached_metal;
    use pmetal_models::{GenerationOutput, generate_cached_compiled, generate_minimal_async};

    tracing::info!(model = %model_id, "Loading model for inference");

    // Download model if needed (HuggingFace repo ID contains '/')
    let model_path = if model_id.contains('/') && !PathBuf::from(model_id).exists() {
        tracing::info!("Model not found locally, downloading from HuggingFace Hub...");
        let path = pmetal_hub::download_model(model_id, None, None).await?;
        tracing::info!("Model downloaded successfully to {:?}", path);
        path
    } else {
        PathBuf::from(model_id)
    };

    // ── Prepare inference via shared runner ──────────────────────────────
    use pmetal::inference_runner::{InferenceRunner, InferenceRunnerConfig};

    let runner_config = InferenceRunnerConfig {
        model_path: model_path.clone(),
        lora_path: lora_path.map(|s| s.to_string()),
        experts_dir: experts_dir.map(|s| s.to_string()),
        fp8,
        prompt: prompt.to_string(),
        chat_messages: None,
        system_message: system.map(|s| s.to_string()),
        chat,
        no_thinking,
        tools: tools.map(|t| t.to_vec()),
        temperature,
        top_k,
        top_p,
        min_p,
        max_tokens,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        seed,
        kv_quant,
        kv_k_bits,
        kv_v_bits,
        kv_group_size,
        kv_turboquant,
        kv_turboquant_preset,
        kv_quant_preset,
        no_kv_quant,
        kv_qjl,
    };

    let mut runner = InferenceRunner::prepare(runner_config)?;
    let use_chat = runner.is_chat();
    let gen_config = runner.state.gen_config();

    if profile_layers {
        if benchmark {
            anyhow::bail!(
                "--profile-layers and --benchmark are separate modes; run them separately"
            );
        }
        if lora_path.is_some() {
            anyhow::bail!("--profile-layers is only supported for standard models right now");
        }
        return run_qwen3_next_layer_profile(&mut runner, model_id, profile_output);
    }

    if benchmark {
        let prompt_tokens = benchmark_prompt_tokens
            .ok_or_else(|| anyhow::anyhow!("--benchmark requires --benchmark-prompt-tokens"))?;
        println!("Running warmup..");
        println!(
            "Timing with prompt_tokens={prompt_tokens}, generation_tokens={max_tokens}, batch_size=1."
        );
        let trials = runner.benchmark_mlx_lm(
            prompt_tokens,
            max_tokens,
            benchmark_iters,
            seed.unwrap_or(0),
        )?;

        for (i, trial) in trials.iter().enumerate() {
            println!(
                "Trial {}:  prompt_tps={:.3}, generation_tps={:.3}, peak_memory={:.3}",
                i + 1,
                trial.prompt_tps,
                trial.generation_tps,
                trial.peak_memory_gb
            );
        }

        if !trials.is_empty() {
            let avg = |f: fn(&pmetal::native_inference::MlxLmBenchmarkTrial) -> f64| -> f64 {
                trials.iter().map(|trial| f(trial)).sum::<f64>() / trials.len() as f64
            };
            println!(
                "Averages: prompt_tps={:.3}, generation_tps={:.3}, peak_memory={:.3}",
                avg(|t| t.prompt_tps),
                avg(|t| t.generation_tps),
                avg(|t| t.peak_memory_gb),
            );
        }

        return Ok(());
    }

    // Print configuration
    println!("\n========================================");
    println!("  PMetal Inference");
    println!("========================================");
    println!("Model:       {}", model_id);
    if lora_path.is_some() {
        println!("LoRA:        {}", lora_path.unwrap());
    }
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
    println!("========================================");
    println!();
    println!("Generating...");

    // Extract refs for generation dispatch (split borrow)
    let input_ids = runner.state.input_ids().to_vec();
    let mut gen_config = runner.state.gen_config().clone();
    gen_config.ane_real_time = ane_real_time;

    // ── Generation dispatch ────────────────────────────────────────────────
    let start = std::time::Instant::now();
    let mut already_streamed = false;

    #[cfg(target_os = "macos")]
    let output = {
        // ANE branch: separate engine with its own weight loading and KV cache
        #[cfg(feature = "ane")]
        let ane_output: Option<GenerationOutput> = if ane {
            let param_count_too_small =
                match std::fs::read_to_string(model_path.join("config.json")) {
                    Ok(config_text) => {
                        match serde_json::from_str::<serde_json::Value>(&config_text) {
                            Ok(config_json) => {
                                let hidden = config_json
                                    .get("hidden_size")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                let layers = config_json
                                    .get("num_hidden_layers")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                let vocab = config_json
                                    .get("vocab_size")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);
                                let est_params = 12 * hidden * hidden * layers + hidden * vocab;
                                est_params < 2_000_000_000
                            }
                            Err(_) => false,
                        }
                    }
                    Err(_) => false,
                };

            if param_count_too_small {
                tracing::info!("Small model (<2B) — using GPU path for faster decode");
                None
            } else {
                let ane_compatible = match std::fs::read_to_string(model_path.join("config.json")) {
                    Ok(config_text) => {
                        match serde_json::from_str::<serde_json::Value>(&config_text) {
                            Ok(config_json) => {
                                use pmetal_metal::ane::dynamic_trainer::DynamicAneTrainerConfig;
                                match DynamicAneTrainerConfig::is_ane_compatible(&config_json) {
                                    Ok(()) => true,
                                    Err(reason) => {
                                        tracing::info!("Skipping ANE inference: {}", reason);
                                        false
                                    }
                                }
                            }
                            Err(_) => true,
                        }
                    }
                    Err(_) => true,
                };

                if ane_compatible {
                    tracing::info!("Attempting ANE-hybrid inference engine");
                    match pmetal_models::generate_cached_ane(
                        &model_path,
                        &input_ids,
                        &gen_config,
                        ane_max_seq_len,
                    ) {
                        Ok(output) => Some(output),
                        Err(e) => {
                            tracing::warn!("ANE inference failed ({}), falling back to GPU", e);
                            None
                        }
                    }
                } else {
                    None
                }
            }
        } else {
            None
        };
        #[cfg(not(feature = "ane"))]
        let ane_output: Option<GenerationOutput> = None;

        #[cfg(feature = "ane")]
        let cpu_hybrid_output: Option<GenerationOutput> = if ane_output.is_none() && ane {
            match std::fs::read_to_string(model_path.join("config.json")) {
                Ok(config_text) => match serde_json::from_str::<serde_json::Value>(&config_text) {
                    Ok(config_json) => {
                        use pmetal_models::is_hybrid_cpu_compatible;
                        match is_hybrid_cpu_compatible(&config_json) {
                            Ok(()) => {
                                tracing::info!("Attempting CPU GEMV hybrid engine");
                                match pmetal_models::generate_cached_hybrid_cpu(
                                    &model_path,
                                    &input_ids,
                                    &gen_config,
                                ) {
                                    Ok(output) => Some(output),
                                    Err(e) => {
                                        tracing::warn!(
                                            "CPU hybrid engine failed ({}), falling back to GPU",
                                            e
                                        );
                                        None
                                    }
                                }
                            }
                            Err(_) => None,
                        }
                    }
                    Err(_) => None,
                },
                Err(_) => None,
            }
        } else {
            None
        };
        #[cfg(not(feature = "ane"))]
        let cpu_hybrid_output: Option<GenerationOutput> = None;

        if let Some(output) = ane_output {
            output
        } else if let Some(output) = cpu_hybrid_output {
            output
        } else if minimal {
            tracing::info!("Using minimal async generation (debugging)");
            runner
                .state
                .run_with(|fwd, cache| generate_minimal_async(fwd, &input_ids, gen_config, cache))?
        } else if metal_sampler {
            tracing::info!("Using fused Metal sampling kernel");
            runner
                .state
                .run_with(|fwd, cache| generate_cached_metal(fwd, &input_ids, gen_config, cache))?
        } else if compiled {
            tracing::info!("Using JIT-compiled sampling");
            runner.state.run_with(|fwd, cache| {
                generate_cached_compiled(fwd, &input_ids, gen_config, cache)
            })?
        } else {
            already_streamed = true;
            let tokenizer = &runner.tokenizer;
            let mut token_buf: Vec<u32> = Vec::new();
            let mut streamed_text = String::new();
            runner.state.generate_streaming(|token_id| {
                use std::io::Write;
                token_buf.push(token_id);
                if let Ok(text) = tokenizer.decode(&token_buf) {
                    if text.len() > streamed_text.len() {
                        let delta = &text[streamed_text.len()..];
                        let _ = std::io::stdout().write_all(delta.as_bytes());
                        let _ = std::io::stdout().flush();
                    }
                    streamed_text = text;
                }
                true
            })?
        }
    };

    #[cfg(not(target_os = "macos"))]
    let output = {
        let _ = metal_sampler;
        let _ = ane;
        if minimal {
            runner
                .state
                .run_with(|fwd, cache| generate_minimal_async(fwd, &input_ids, gen_config, cache))?
        } else if compiled {
            runner.state.run_with(|fwd, cache| {
                generate_cached_compiled(fwd, &input_ids, gen_config, cache)
            })?
        } else {
            already_streamed = true;
            let tokenizer = &runner.tokenizer;
            let mut token_buf: Vec<u32> = Vec::new();
            let mut streamed_text = String::new();
            runner.state.generate_streaming(|token_id| {
                use std::io::Write;
                token_buf.push(token_id);
                if let Ok(text) = tokenizer.decode(&token_buf) {
                    if text.len() > streamed_text.len() {
                        let delta = &text[streamed_text.len()..];
                        let _ = std::io::stdout().write_all(delta.as_bytes());
                        let _ = std::io::stdout().flush();
                    }
                    streamed_text = text;
                }
                true
            })?
        }
    };
    let elapsed = start.elapsed();

    // For non-streaming paths, decode and print the generated text now
    if !already_streamed {
        let generated_tokens = &output.token_ids[input_ids.len()..];
        let raw_text = runner.tokenizer.decode(generated_tokens)?;
        let text = if use_chat && !no_thinking {
            format!("<think>{}", raw_text)
        } else {
            raw_text
        };
        if use_chat && show_thinking {
            if let Some(thinking) = extract_thinking_content(&text) {
                println!("=== Thinking ===");
                println!("{}", thinking);
                println!("=== Response ===");
            }
            println!("{}", extract_final_response(&text));
        } else if use_chat {
            println!("{}", extract_final_response(&text));
        } else {
            println!("{}", text);
        }
    } else {
        println!();
    }

    // Ensure a clean newline between streamed output and stats
    println!();
    println!("---");
    println!(
        "Generated {} tokens in {:.2}s",
        output.num_generated,
        elapsed.as_secs_f64(),
    );
    if output.stopped_by_token {
        println!("Stopped by: EOS token");
    } else {
        println!("Stopped by: max length");
    }
    if let Some(m) = runner.state.last_decode_metrics {
        println!(
            "Decode: {:.0} tok/s (avg={:.2}ms p50={:.2}ms over {} steps)",
            m.tok_per_sec, m.avg_step_ms, m.p50_step_ms, m.measured_steps,
        );
    }

    // Print expert prefetch stats if offloading was active
    if let Some(model) = runner.state.dynamic_model() {
        if let Some(stats) = model.prefetch_stats() {
            eprintln!(
                "Expert prefetch: {:.1}% hit rate ({} hits / {} total)",
                stats.hit_rate() * 100.0,
                stats.hits,
                stats.total,
            );
        }
    }

    Ok(())
}

// NOTE: run_inference_with_lora has been removed — LoRA loading is now handled
// by InferenceRunner::prepare() via the lora_path config field.

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_models::architectures::Qwen3NextProfileSection;

    #[test]
    fn qwen3_next_phase_summary_groups_layer_kinds_and_sections() {
        let profile = Qwen3NextForwardProfile {
            phase: "prefill".to_string(),
            input_shape: vec![1, 128],
            embedding_us: 10,
            layers: vec![
                Qwen3NextLayerProfile {
                    layer_idx: 0,
                    layer_kind: "linear_attention".to_string(),
                    sections: vec![
                        Qwen3NextProfileSection {
                            name: "gdn_input_qkv".to_string(),
                            elapsed_us: 100,
                        },
                        Qwen3NextProfileSection {
                            name: "gdn_recurrence".to_string(),
                            elapsed_us: 80,
                        },
                    ],
                    total_us: 220,
                },
                Qwen3NextLayerProfile {
                    layer_idx: 1,
                    layer_kind: "full_attention".to_string(),
                    sections: vec![
                        Qwen3NextProfileSection {
                            name: "attn_sdpa".to_string(),
                            elapsed_us: 140,
                        },
                        Qwen3NextProfileSection {
                            name: "attn_out_proj".to_string(),
                            elapsed_us: 40,
                        },
                    ],
                    total_us: 210,
                },
                Qwen3NextLayerProfile {
                    layer_idx: 2,
                    layer_kind: "linear_attention".to_string(),
                    sections: vec![Qwen3NextProfileSection {
                        name: "gdn_input_qkv".to_string(),
                        elapsed_us: 90,
                    }],
                    total_us: 120,
                },
            ],
            final_norm_us: 20,
            lm_head_us: 30,
            total_us: 650,
        };

        let summary = build_qwen3_next_phase_summary(&profile);
        assert_eq!(summary.layer_total_us, 550);
        assert_eq!(summary.non_layer_us, 100);
        assert_eq!(summary.layer_kind_totals.len(), 2);
        assert_eq!(summary.layer_kind_totals[0].layer_kind, "linear_attention");
        assert_eq!(summary.layer_kind_totals[0].layer_count, 2);
        assert_eq!(summary.layer_kind_totals[0].total_us, 340);
        assert_eq!(
            summary.layer_kind_totals[0].top_sections[0],
            HybridProfileSectionSummary {
                name: "gdn_input_qkv".to_string(),
                total_us: 190,
            }
        );
        assert_eq!(summary.top_sections[0].name, "gdn_input_qkv");
        assert_eq!(summary.top_sections[0].total_us, 190);
    }
}
