use std::path::Path;

use anyhow::Context;
use pmetal_core::jobs::InferSpec;
use pmetal_models::{
    DynamicModel,
    architectures::{Qwen3NextForwardProfile, Qwen3NextLayerProfile},
};
use pmetal_runtime::compat::{Array, Exception};
use std::collections::BTreeMap;

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

            let last_logits = pmetal_runtime::compat::ops::select_axis(&prefill_logits, -1, 1);
            let next_token = pmetal_runtime::compat::ops::argmax(&last_logits, -1);
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

/// Continue the conversation in `--messages-file` instead of answering a
/// prompt: read and checked (`pmetal_core::jobs::parse_messages`), then handed
/// to the runner as its history, the last user message being the prompt, so a
/// one-message conversation is exactly the prompt it stands for.
fn apply_messages_file(
    config: &mut pmetal::inference_runner::InferenceRunnerConfig,
    path: &str,
) -> anyhow::Result<()> {
    use pmetal_core::jobs::MAX_MESSAGES_BYTES;

    let size = std::fs::metadata(path)
        .map_err(|e| anyhow::anyhow!("--messages-file: could not read '{path}': {e}"))?
        .len();
    if size > MAX_MESSAGES_BYTES as u64 {
        anyhow::bail!(
            "--messages-file: '{path}' is larger than {} MiB",
            MAX_MESSAGES_BYTES >> 20
        );
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("--messages-file: could not read '{path}': {e}"))?;
    let messages = pmetal_core::jobs::parse_messages(&text)
        .map_err(|e| anyhow::anyhow!("--messages-file: '{path}' {e}"))?;
    config.chat = true;
    (config.chat_messages, config.prompt) = conversation(messages, config.system_message.is_some())
        .map_err(|e| anyhow::anyhow!("--messages-file: '{path}': {e}"))?;
    Ok(())
}

/// A checked conversation as the runner takes it: the history, and the prompt
/// (the last message if it is a user's, else empty and the history ends with a
/// tool's result). `has_system` is whether `--system` was also given.
fn conversation(
    messages: Vec<pmetal_core::jobs::ChatMessage>,
    has_system: bool,
) -> Result<(Option<Vec<pmetal_data::chat_templates::Message>>, String), String> {
    use pmetal_data::chat_templates::{Message, ToolCall};

    if has_system && messages.first().is_some_and(|m| m.role == "system") {
        return Err(
            "it has a system message and --system was given too; use one of them".to_string(),
        );
    }
    let mut history = Vec::with_capacity(messages.len());
    for (i, wire) in messages.into_iter().enumerate() {
        let mut message = Message::new(wire.role, wire.content);
        message.reasoning_content = wire.reasoning_content;
        message.tool_call_id = wire.tool_call_id;
        if let Some(calls) = wire.tool_calls {
            let calls: Vec<ToolCall> = serde_json::from_value(serde_json::Value::Array(calls))
                .map_err(|e| format!("message {}: tool_calls are not tool calls: {e}", i + 1))?;
            message.tool_calls = Some(calls);
        }
        history.push(message);
    }
    let prompt = match history.last() {
        Some(last) if last.role == "user" => history.pop().map(|m| m.content).unwrap_or_default(),
        _ => String::new(),
    };
    Ok(((!history.is_empty()).then_some(history), prompt))
}

/// Run inference with a model.
///
/// Uses the shared `InferenceRunner` for model loading, tokenization, chat
/// template, sampling config, and cache creation. CLI-specific features
/// (ANE, benchmark) use the runner's prepared state.
pub(crate) async fn run_inference(spec: InferSpec) -> anyhow::Result<()> {
    use pmetal::inference_runner::{InferenceRunner, InferenceRunnerConfig};
    use pmetal_data::inference_config::InferenceBackend;

    let tool_defs: Option<Vec<pmetal_data::chat_templates::ToolDefinition>> =
        match spec.tools.as_deref() {
            Some(tools_path) => {
                let json = std::fs::read_to_string(tools_path)
                    .map_err(|e| anyhow::anyhow!("--tools: could not read '{tools_path}': {e}"))?;
                let defs: Vec<pmetal_data::chat_templates::ToolDefinition> =
                    serde_json::from_str(&json).map_err(|e| {
                        anyhow::anyhow!("--tools: '{tools_path}' is not a tool list: {e}")
                    })?;
                tracing::info!("Loaded {} tool definitions from {tools_path}", defs.len());
                Some(defs)
            }
            None => None,
        };
    let validated_profile_output = spec
        .profile_output
        .as_deref()
        .map(|path| crate::validate_output_path(path, "infer profile output"))
        .transpose()?;
    // The runner's configuration comes from the spec before anything is
    // fetched, so a bad option fails at once; the model paths are filled in
    // once they are resolved.
    let mut runner_config = InferenceRunnerConfig::from_spec(&spec).map_err(anyhow::Error::msg)?;
    runner_config.tools = tool_defs;
    if let Some(path) = spec.messages_file.as_deref() {
        apply_messages_file(&mut runner_config, path)?;
    }
    let backend: InferenceBackend = spec.backend.parse().map_err(anyhow::Error::msg)?;

    let InferSpec {
        model: ref model_id,
        lora: ref lora_path,
        image: ref images,
        video: ref videos,
        json,
        max_tokens,
        seed,
        hide_thinking,
        ref draft_model,
        mtp,
        ref mtp_model,
        mtp_draft_tokens,
        fp8,
        ref experts_dir,
        ane,
        ane_max_seq_len,
        benchmark,
        benchmark_iters,
        benchmark_prompt_tokens,
        profile_layers,
        ..
    } = spec;
    let model_id = model_id.as_str();
    let lora_path = lora_path.as_deref();
    let draft_model = draft_model.as_deref();
    let mtp_model = mtp_model.as_deref();
    let experts_dir = experts_dir.as_deref();
    let profile_output = validated_profile_output.as_deref();
    // With --json, stdout is the one JSON object; everything a person reads
    // (the banner, the statistics) goes to stderr.
    macro_rules! say {
        ($($arg:tt)*) => {
            if json { eprintln!($($arg)*) } else { println!($($arg)*) }
        };
    }
    #[cfg(not(feature = "ane"))]
    {
        // Only the ANE engine reads it.
        let _ = ane_max_seq_len;
        if ane {
            anyhow::bail!("ANE inference requires the 'ane' feature: cargo build --features ane");
        }
    }
    // `--chat-template-kwargs '{"enable_thinking": false}'` is --no-thinking.
    let no_thinking = runner_config.no_thinking;
    let reasoning_effort = runner_config
        .chat_template_kwargs
        .reasoning_effort()
        .map(str::to_string);
    use pmetal_models::GenerationOutput;

    // ── Resolve --backend into the generation path ────────────────────────
    //
    // `auto` streams on the GPU, or runs on the ANE when `--ane` is set;
    // `standard` and `ane` pin their path and override `--ane`.
    let ane = match backend {
        InferenceBackend::Auto => ane,
        InferenceBackend::Standard => false,
        InferenceBackend::Ane => {
            #[cfg(not(feature = "ane"))]
            {
                anyhow::bail!(
                    "--backend ane requires the 'ane' feature: cargo build --features ane"
                );
            }
            #[cfg(feature = "ane")]
            {
                true
            }
        }
    };

    if !images.is_empty() || !videos.is_empty() {
        // Images and videos go through the native engine's multimodal
        // prefill; every other generation path would ignore them.
        let media = if videos.is_empty() {
            "--image"
        } else if images.is_empty() {
            "--video"
        } else {
            "--image / --video"
        };
        let other = [
            (benchmark, "--benchmark"),
            (profile_layers, "--profile-layers"),
            (mtp, "--mtp"),
            (draft_model.is_some(), "--draft-model"),
            (lora_path.is_some(), "--lora"),
            (fp8, "--fp8"),
            (experts_dir.is_some(), "--experts-dir"),
            (ane, "--ane / --backend ane"),
        ];
        if let Some((_, flag)) = other.iter().find(|(set, _)| *set) {
            anyhow::bail!("{media} cannot be combined with {flag}");
        }
    }

    tracing::info!(model = %model_id, "Loading model for inference");
    if mtp && draft_model.is_some() {
        anyhow::bail!(
            "Use either --mtp for bundled Qwen MTP or --draft-model for Gemma 4 MTP, not both"
        );
    }
    if mtp_model.is_some() && !mtp {
        anyhow::bail!("--mtp-model requires --mtp");
    }
    // A draft model is a Gemma 4 MTP assistant, or a DFlash draft model for
    // the ANE engine.
    let draft_path = match draft_model {
        Some(draft) => Some(pmetal_hub::resolve_model_path(draft, None, None).await?),
        None => None,
    };
    let dflash_draft = draft_path
        .as_deref()
        .is_some_and(pmetal_models::dflash_drafter::is_dflash_draft);
    if dflash_draft && !ane {
        anyhow::bail!(
            "{} is a DFlash draft model, which drafts for the ANE engine: add --ane, or run \
             `pmetal dflash` for the GPU",
            draft_model.unwrap_or_default()
        );
    }
    let mtp_owns_generation = mtp || (draft_model.is_some() && !dflash_draft);
    // Only the ANE path reads `ane` from here on.
    #[cfg_attr(not(feature = "ane"), allow(unused_variables))]
    let ane = if mtp_owns_generation {
        if ane {
            tracing::info!(
                "MTP uses its own verifier/drafter generation backend; ignoring the requested non-MTP backend selector"
            );
        }
        false
    } else {
        ane
    };

    // Download model if needed (HuggingFace repo ID contains '/')
    let model_path = pmetal_hub::resolve_model_path(model_id, None, None).await?;
    tracing::info!("Model ready at {:?}", model_path);
    let mtp_assistant_path = match &draft_path {
        Some(path) if !dflash_draft => {
            tracing::info!("Gemma 4 MTP assistant ready at {:?}", path);
            Some(path.clone())
        }
        _ => None,
    };
    let qwen_mtp_path = if let Some(mtp_model) = mtp_model {
        let path = pmetal_hub::resolve_model_path(mtp_model, None, None).await?;
        tracing::info!("Qwen MTP checkpoint ready at {:?}", path);
        Some(path)
    } else {
        None
    };

    // ── Prepare inference via shared runner ──────────────────────────────
    runner_config.model_path = model_path.clone();
    runner_config.mtp_assistant_path = mtp_assistant_path.clone();
    runner_config.qwen_mtp_path = qwen_mtp_path.clone();

    let mut runner = InferenceRunner::prepare(runner_config)?;
    let use_chat = runner.is_chat();
    // Whether the output opens with thinking: asked for, or the chat
    // template's own default (Gemma 4 and Qwen3.5-0.8B don't).
    let thinking_requested_off = no_thinking;
    let no_thinking = !runner.thinks();
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
        // A benchmark times a fixed generation length, not the model's budget.
        let max_tokens = max_tokens.unwrap_or(pmetal_data::inference_config::FALLBACK_MAX_TOKENS);
        println!("Running warmup..");
        println!(
            "Timing with prompt_tokens={prompt_tokens}, generation_tokens={max_tokens}, batch_size=1."
        );
        let trials = runner.benchmark(
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
            let avg = |f: fn(&pmetal::native_inference::BenchmarkTrialMetrics) -> f64| -> f64 {
                trials.iter().map(f).sum::<f64>() / trials.len() as f64
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
    say!("\n========================================");
    say!("  PMetal Inference");
    say!("========================================");
    say!("Model:       {}", model_id);
    if let Some(path) = lora_path {
        say!("LoRA:        {}", path);
    }
    if let Some(path) = mtp_assistant_path.as_ref() {
        say!("MTP:         {}", path.display());
    }
    if mtp {
        if let Some(path) = qwen_mtp_path.as_ref() {
            say!(
                "MTP:         {} ({})",
                path.display(),
                mtp_draft_tokens.max(1)
            );
        } else {
            say!("MTP:         bundled Qwen ({})", mtp_draft_tokens.max(1));
        }
    }
    say!("Temperature: {}", gen_config.temperature);
    say!("Top-k:       {}", gen_config.top_k);
    say!("Top-p:       {}", gen_config.top_p);
    say!("Min-p:       {}", gen_config.min_p);
    say!("Rep penalty: {}", gen_config.repetition_penalty);
    say!("Freq penalty:{}", gen_config.frequency_penalty);
    say!("Pres penalty:{}", gen_config.presence_penalty);
    if let Some(s) = gen_config.seed {
        say!("Seed:        {}", s);
    }
    say!(
        "Max tokens:  {}{}",
        gen_config.max_new_tokens,
        if max_tokens.is_none() {
            " (model default)"
        } else {
            ""
        }
    );
    if use_chat && thinking_requested_off {
        say!("Thinking:    disabled");
    } else if use_chat && !no_thinking {
        say!(
            "Thinking:    on{}",
            reasoning_effort
                .as_deref()
                .map(|effort| format!(" (reasoning effort {effort})"))
                .unwrap_or_default()
        );
    }
    say!("========================================");
    say!();
    if runner.state.is_native_only() {
        say!("Loading model...");
    } else {
        say!("Generating...");
    }

    // Extract refs for generation dispatch (split borrow)
    let input_ids = runner.state.input_ids().to_vec();
    #[cfg(feature = "ane")]
    let gen_config = runner.state.gen_config().clone();

    // ── Generation dispatch ────────────────────────────────────────────────
    let start = std::time::Instant::now();
    // Ctrl-C, or the SIGTERM a supervisor stops a job with (the desktop app's
    // Stop button), ends the generation after the token in flight, so the text
    // so far is complete and the statistics are printed; the exit status is 130.
    let interrupted = super::stop::StopFlag::watch().handle();
    // The text: on stdout as it is generated, or kept for the JSON object.
    let collected = std::cell::RefCell::new(String::new());
    let emit = |delta: &str| {
        if delta.is_empty() {
            return;
        }
        if json {
            collected.borrow_mut().push_str(delta);
        } else {
            use std::io::Write;
            let _ = std::io::stdout().write_all(delta.as_bytes());
            let _ = std::io::stdout().flush();
        }
    };
    // ⚠️ The native engines load their weights inside the dispatch below, so
    // `start` measures load + prefill + decode. Reporting that as generation
    // time made a 1 s generation read as 30 s.
    let mut first_token_at: Option<std::time::Instant> = None;

    let output = {
        // ANE branch: separate engine with its own weight loading and KV cache
        #[cfg(feature = "ane")]
        let ane_output: Option<GenerationOutput> = if ane {
            // --ane is an instruction: a model the ANE cannot run, or a run
            // that fails, is an error. Falling back to the GPU would answer
            // slower than asked, hide a broken ANE path, and (after a
            // failure part-way) print the reply twice.
            let config_json: serde_json::Value =
                std::fs::read_to_string(model_path.join("config.json"))
                    .map_err(|e| anyhow::anyhow!("--ane: can't read config.json: {e}"))
                    .and_then(|text| {
                        serde_json::from_str(&text)
                            .map_err(|e| anyhow::anyhow!("--ane: config.json does not parse: {e}"))
                    })?;
            pmetal_models::is_ane_inference_compatible(&config_json).map_err(|reason| {
                anyhow::anyhow!("--ane: the ANE cannot run this model: {reason}")
            })?;
            let get = |key: &str| config_json.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
            let est_params =
                12 * get("hidden_size") * get("hidden_size") * get("num_hidden_layers")
                    + get("hidden_size") * get("vocab_size");
            if est_params < 2_000_000_000 {
                tracing::warn!(
                    "models under ~2B parameters decode faster on the GPU than on the ANE; \
                     running on the ANE as asked"
                );
            }
            tracing::info!("Generating on the ANE");
            let tokenizer = &runner.tokenizer;
            let show_thinking = use_chat && !hide_thinking && !no_thinking;
            let mut formatter =
                pmetal_data::stream_format::StreamFormatter::new(tokenizer, show_thinking);
            let output = pmetal_models::generate_cached_ane_streaming(
                &model_path,
                draft_path.as_deref().filter(|_| dflash_draft),
                &input_ids,
                &gen_config,
                ane_max_seq_len,
                |token_id| {
                    first_token_at.get_or_insert_with(std::time::Instant::now);
                    if gen_config.stop_tokens.contains(&token_id) {
                        return true;
                    }
                    let delta = formatter.push_token(tokenizer, token_id);
                    emit(&delta);
                    !interrupted.load(std::sync::atomic::Ordering::Relaxed)
                },
            )
            .map_err(|e| anyhow::anyhow!("ANE inference failed: {e}"))?;
            Some(output)
        } else {
            None
        };
        #[cfg(not(feature = "ane"))]
        let ane_output: Option<GenerationOutput> = None;

        if let Some(output) = ane_output {
            output
        } else {
            let tokenizer = &runner.tokenizer;
            let show_thinking = use_chat && !hide_thinking && !no_thinking;
            let mut formatter =
                pmetal_data::stream_format::StreamFormatter::new(tokenizer, show_thinking);
            runner.state.generate_streaming(|token_id| {
                first_token_at.get_or_insert_with(std::time::Instant::now);
                let delta = formatter.push_token(tokenizer, token_id);
                emit(&delta);
                !interrupted.load(std::sync::atomic::Ordering::Relaxed)
            })?
        }
    };
    let elapsed = start.elapsed();

    // The streamed text ends without a newline, and the statistics follow.
    if !json {
        println!();
    }
    say!();
    say!("---");
    match first_token_at {
        Some(first) => say!(
            "Generated {} tokens in {:.2}s ({:.2}s to the first token, model load included)",
            output.num_generated,
            (elapsed - (first - start)).as_secs_f64(),
            (first - start).as_secs_f64(),
        ),
        None => say!(
            "Generated {} tokens in {:.2}s",
            output.num_generated,
            elapsed.as_secs_f64(),
        ),
    }
    if output.stopped_by_token {
        say!("Stopped by: EOS token");
    } else {
        say!("Stopped by: max length");
    }
    if let Some(m) = runner.state.last_decode_metrics {
        let chip = if let Ok(ctx) = pmetal_metal::context::MetalContext::global() {
            let name = &ctx.properties().name;
            name.strip_prefix("Apple ").unwrap_or(name).to_string()
        } else {
            String::new()
        };
        if chip.is_empty() {
            say!(
                "Decode: {:.0} tok/s (avg={:.2}ms p50={:.2}ms over {} steps)",
                m.tok_per_sec,
                m.avg_step_ms,
                m.p50_step_ms,
                m.measured_steps,
            );
        } else {
            say!(
                "Decode: {:.0} tok/s (avg={:.2}ms p50={:.2}ms over {} steps) [{}]",
                m.tok_per_sec,
                m.avg_step_ms,
                m.p50_step_ms,
                m.measured_steps,
                chip,
            );
        }
    }
    if let Some(m) = output.speculative_metrics.as_ref() {
        say!(
            "Speculative: {:.1}% accepted ({} / {} draft tokens, avg {:.2}/verify, {} bonus, {} correction)",
            m.acceptance_rate() * 100.0,
            m.accepted_draft_tokens,
            m.drafted_tokens,
            m.avg_accepted_per_verify(),
            m.bonus_tokens,
            m.correction_tokens,
        );
    }

    if json {
        let report = inference_report(&InferenceFacts {
            model: model_id,
            text: &collected.borrow(),
            prompt_tokens: input_ids.len(),
            generated_tokens: output.num_generated,
            stopped_by_token: output.stopped_by_token,
            elapsed_seconds: elapsed.as_secs_f64(),
            first_token_seconds: first_token_at.map(|first| (first - start).as_secs_f64()),
            decode_tokens_per_second: runner.state.last_decode_metrics.map(|m| m.tok_per_sec),
            speculative_acceptance: output
                .speculative_metrics
                .as_ref()
                .map(|m| f64::from(m.acceptance_rate())),
            interrupted: interrupted.load(std::sync::atomic::Ordering::Relaxed),
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    }

    // Print what the expert files were asked for if offloading was active
    if let Some(model) = runner.state.dynamic_model() {
        if let Some(stats) = model.expert_read_stats() {
            eprintln!(
                "Expert reads: {} experts, {:.1} MB in {:.1} ms{}",
                stats.experts,
                stats.bytes as f64 / 1e6,
                stats.read_us as f64 / 1000.0,
                stats
                    .bytes_per_second()
                    .map_or(String::new(), |rate| format!(" ({:.2} GB/s)", rate / 1e9)),
            );
        }
    }

    if interrupted.load(std::sync::atomic::Ordering::Relaxed) {
        eprintln!("Interrupted.");
        return Err(crate::Interrupted.into());
    }
    Ok(())
}

/// What a finished generation reports in `--json`.
struct InferenceFacts<'a> {
    model: &'a str,
    text: &'a str,
    prompt_tokens: usize,
    generated_tokens: usize,
    stopped_by_token: bool,
    elapsed_seconds: f64,
    first_token_seconds: Option<f64>,
    decode_tokens_per_second: Option<f64>,
    speculative_acceptance: Option<f64>,
    interrupted: bool,
}

/// The `--json` object. `seconds` is the generation time (the model's load
/// is in `time_to_first_token_seconds` for the engines that load inside the
/// run).
fn inference_report(facts: &InferenceFacts<'_>) -> serde_json::Value {
    let stopped_by = if facts.interrupted {
        "interrupted"
    } else if facts.stopped_by_token {
        "eos"
    } else {
        "length"
    };
    let seconds = facts.elapsed_seconds - facts.first_token_seconds.unwrap_or(0.0);
    serde_json::json!({
        "model": facts.model,
        "text": facts.text,
        "prompt_tokens": facts.prompt_tokens,
        "generated_tokens": facts.generated_tokens,
        "stopped_by": stopped_by,
        "seconds": seconds.max(0.0),
        "time_to_first_token_seconds": facts.first_token_seconds,
        "decode_tokens_per_second": facts.decode_tokens_per_second,
        "speculative_acceptance": facts.speculative_acceptance,
    })
}

// NOTE: run_inference_with_lora has been removed — LoRA loading is now handled
// by InferenceRunner::prepare() via the lora_path config field.

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_models::architectures::Qwen3NextProfileSection;

    #[test]
    fn the_json_report_says_why_generation_stopped_and_what_it_cost() {
        let mut facts = InferenceFacts {
            model: "m",
            text: "hello",
            prompt_tokens: 5,
            generated_tokens: 3,
            stopped_by_token: true,
            elapsed_seconds: 2.5,
            first_token_seconds: Some(1.0),
            decode_tokens_per_second: Some(42.0),
            speculative_acceptance: None,
            interrupted: false,
        };
        let report = inference_report(&facts);
        assert_eq!(report["text"], "hello");
        assert_eq!(report["stopped_by"], "eos");
        assert_eq!(report["seconds"], 1.5);
        assert_eq!(report["time_to_first_token_seconds"], 1.0);
        assert!(report["speculative_acceptance"].is_null());
        facts.stopped_by_token = false;
        assert_eq!(inference_report(&facts)["stopped_by"], "length");
        facts.interrupted = true;
        assert_eq!(inference_report(&facts)["stopped_by"], "interrupted");
    }

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

#[cfg(test)]
mod messages_file_tests {
    use super::*;
    use pmetal_core::jobs::{ChatMessage, parse_messages};

    fn chat(json: &str) -> Vec<ChatMessage> {
        parse_messages(json).unwrap()
    }

    /// A one-message conversation is the prompt it stands for, so the two
    /// produce the same text.
    #[test]
    fn a_single_user_message_is_the_prompt() {
        let (history, prompt) =
            conversation(chat(r#"[{"role":"user","content":"hello"}]"#), false).unwrap();
        assert!(history.is_none());
        assert_eq!(prompt, "hello");
    }

    #[test]
    fn earlier_turns_are_history_and_the_last_user_message_is_the_prompt() {
        let (history, prompt) = conversation(
            chat(
                r#"[{"role":"system","content":"be brief"},{"role":"user","content":"a"},
                    {"role":"assistant","content":"b","reasoning_content":"thought"},
                    {"role":"user","content":"c"}]"#,
            ),
            false,
        )
        .unwrap();
        let history = history.unwrap();
        let roles: Vec<&str> = history.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user", "assistant"]);
        assert_eq!(history[2].reasoning_content.as_deref(), Some("thought"));
        assert_eq!(prompt, "c");
    }

    #[test]
    fn a_conversation_ending_in_a_tool_result_has_no_prompt() {
        let (history, prompt) = conversation(
            chat(
                r#"[{"role":"user","content":"weather?"},
                    {"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"SF\"}"}}]},
                    {"role":"tool","content":"sunny","tool_call_id":"c1"}]"#,
            ),
            false,
        )
        .unwrap();
        assert_eq!(prompt, "");
        let history = history.unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(
            history[1].tool_calls.as_ref().unwrap()[0].function.name,
            "get_weather"
        );
        assert_eq!(history[2].tool_call_id.as_deref(), Some("c1"));
    }

    #[test]
    fn a_system_message_and_the_system_flag_are_one_too_many() {
        let err = conversation(
            chat(r#"[{"role":"system","content":"s"},{"role":"user","content":"a"}]"#),
            true,
        )
        .unwrap_err();
        assert!(err.contains("--system"), "{err}");
        assert!(
            conversation(chat(r#"[{"role":"user","content":"a"}]"#), true).is_ok(),
            "the flag alone is fine"
        );
    }

    #[test]
    fn a_bad_tool_call_names_its_message() {
        let err = conversation(
            chat(r#"[{"role":"user","content":"a"},{"role":"assistant","content":"","tool_calls":[{"nope":1}]},{"role":"tool","content":"x"}]"#),
            false,
        )
        .unwrap_err();
        assert!(err.contains("message 2"), "{err}");
    }
}
