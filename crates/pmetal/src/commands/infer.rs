use std::path::{Path, PathBuf};

use pmetal_data::Tokenizer;

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
    kv_quant: u8,
    kv_k_bits: Option<u8>,
    kv_v_bits: Option<u8>,
    kv_group_size: usize,
    no_kv_quant: bool,
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
        no_kv_quant,
    };

    let mut runner = InferenceRunner::prepare(runner_config)?;
    let use_chat = runner.is_chat();
    let gen_config = runner.state.gen_config();

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
    println!("========================================\n");

    println!("Prompt: {}\n", prompt);
    println!("Generating...\n");

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
                                use pmetal_trainer::DynamicAneTrainerConfig;
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

    println!("---");
    let tokens_per_sec = output.num_generated as f64 / elapsed.as_secs_f64();
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

    // ── Benchmark mode ────────────────────────────────────────────────────────
    if benchmark {
        use std::time::Instant;

        println!(
            "\n=== Benchmark Mode ({} decode iterations) ===",
            benchmark_iters
        );

        let last_token_id = output.token_ids.last().copied().unwrap_or(1);
        let mut decode_times_ms: Vec<f64> = Vec::with_capacity(benchmark_iters);

        // Warm-up + timed decode via run_with to avoid borrow conflicts
        runner.state.run_with(|fwd, cache| {
            // Warm-up
            {
                let input = mlx_rs::Array::from_slice(&[last_token_id as i32], &[1, 1]);
                if let Ok(ref logits_w) = fwd(&input, cache) {
                    let _ = logits_w.eval();
                }
            }

            for i in 0..benchmark_iters {
                let input = mlx_rs::Array::from_slice(&[last_token_id as i32], &[1, 1]);
                let t0 = Instant::now();
                let logits = fwd(&input, cache);
                if let Ok(ref l) = logits {
                    let _ = l.eval();
                }
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                let tps = 1000.0 / ms;
                decode_times_ms.push(ms);
                let status = if logits.is_ok() { "ok" } else { "err" };
                println!("  [{i}] {ms:.1} ms ({tps:.2} tok/s)  [{status}]");
            }
        });

        if !decode_times_ms.is_empty() {
            let mut sorted = decode_times_ms.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = sorted.len();
            let mean = sorted.iter().sum::<f64>() / n as f64;
            let min = sorted[0];
            let p50 = sorted[n / 2];
            let p99 = sorted[(n * 99 / 100).min(n - 1)];
            println!("[mean]    {mean:.1} ms ({:.2} tok/s)", 1000.0 / mean);
            println!("[min]     {min:.1} ms ({:.2} tok/s)", 1000.0 / min);
            println!("[p50]     {p50:.1} ms ({:.2} tok/s)", 1000.0 / p50);
            println!("[p99]     {p99:.1} ms ({:.2} tok/s)", 1000.0 / p99);
        }

        let mem_stats = pmetal_mlx::memory::get_memory_stats();
        println!("[memory]  {:.1} GB resident", mem_stats.used_gb());
        println!("[peak]    {:.1} GB peak", mem_stats.peak_gb());
    }

    Ok(())
}

// NOTE: get_eos_tokens, is_instruction_tuned, apply_chat_template, and 15
// per-template format_* functions have been removed. Their logic is now in
// InferenceRunner::prepare() via the generic ChatTemplate::apply path.

// NOTE: is_instruction_tuned moved to inference_runner.rs

#[allow(dead_code)]
pub(crate) fn is_instruction_tuned(model_path: &Path) -> bool {
    // Primary: check for chat_template in tokenizer_config.json (authoritative)
    let config_path = model_path.join("tokenizer_config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                if config.get("chat_template").is_some() {
                    return true;
                }
            }
        }
    }

    // Fallback: only explicit instruct markers in model name
    let path_str = model_path.to_string_lossy().to_lowercase();
    path_str.contains("instruct")
        || path_str.contains("-it-")
        || path_str.contains("-it/")
        || path_str.ends_with("-it")
        || path_str.contains("chat")
}

/// Apply chat template to the prompt using unified detection.
///
/// Returns the formatted prompt string and the detected `ChatTemplateType` so the caller
/// can select the right stop tokens.
///
/// If `no_thinking` is true, prefills empty thinking block to disable reasoning (ChatML/Phi4 only).
/// Otherwise, the model decides when to use thinking based on query complexity.
pub(crate) fn apply_chat_template(
    _tokenizer: &Tokenizer,
    user_message: &str,
    system_message: Option<&str>,
    model_path: &Path,
    no_thinking: bool,
    tools: Option<&[pmetal_data::chat_templates::ToolDefinition]>,
) -> anyhow::Result<(String, pmetal_data::chat_templates::ChatTemplateType)> {
    use pmetal_data::chat_templates::{ChatTemplateType, Message};

    let detected = pmetal_data::chat_templates::detect_chat_template(
        model_path,
        &model_path.to_string_lossy(),
    );

    // When tools are provided, use the structured template system which handles
    // tool injection into system prompts in the model-native format
    if tools.is_some() {
        let mut messages = Vec::new();

        // Build system message with optional thinking control
        let sys_content = match (system_message, no_thinking) {
            (Some(sys), true) => Some(format!("{}\n/no_think", sys)),
            (Some(sys), false) => Some(sys.to_string()),
            (None, true) => Some("/no_think".to_string()),
            (None, false) => None,
        };
        if let Some(sys) = sys_content {
            messages.push(Message::system(sys));
        }

        messages.push(Message::user(user_message));

        let formatted = detected.apply_with_tools(&messages, tools);

        // The formatted text includes the assistant generation prompt
        return Ok((formatted.text, detected.template_type));
    }

    // No tools — use the existing per-template formatting functions
    let formatted = match detected.template_type {
        ChatTemplateType::ChatMl | ChatTemplateType::Qwen => {
            format_chatml(user_message, system_message, no_thinking)
        }
        ChatTemplateType::Llama3 => format_llama3(user_message, system_message),
        ChatTemplateType::Llama2 => format_llama2_inference(user_message, system_message),
        ChatTemplateType::Gemma => format_gemma_inference(user_message, system_message),
        ChatTemplateType::Mistral => format_mistral_inference(user_message, system_message),
        ChatTemplateType::Phi3 => format_phi3_inference(user_message, system_message),
        ChatTemplateType::Phi4 => format_phi4_inference(user_message, system_message, no_thinking),
        ChatTemplateType::GptOss => format_gpt_oss_inference(user_message, system_message),
        ChatTemplateType::Llama4 => format_llama4_inference(user_message, system_message),
        ChatTemplateType::DeepSeek => {
            format_deepseek_inference(user_message, system_message, no_thinking)
        }
        ChatTemplateType::Cohere => format_cohere_inference(user_message, system_message),
        // Alpaca, Vicuna, Zephyr, Custom — fall back to ChatML for inference
        _ => format_chatml(user_message, system_message, no_thinking),
    };

    Ok((formatted, detected.template_type))
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

    // Always include system block (can be empty per NemotronH template)
    result.push_str("<|im_start|>system\n");
    if let Some(sys) = system_message {
        result.push_str(sys);
    }
    result.push_str("<|im_end|>\n");

    result.push_str("<|im_start|>user\n");
    result.push_str(user_message);
    result.push_str("<|im_end|>\n");
    result.push_str("<|im_start|>assistant\n");

    if no_thinking {
        // Force non-thinking: prefill empty think block without newlines
        // This matches NemotronH's expected format
        result.push_str("<think></think>");
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

/// Format message using Llama-2 template for inference.
fn format_llama2_inference(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::from("<s>[INST] ");
    if let Some(sys) = system_message {
        result.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", sys));
    }
    result.push_str(user_message);
    result.push_str(" [/INST] ");
    result
}

/// Format message using Gemma template for inference.
fn format_gemma_inference(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::new();
    if let Some(sys) = system_message {
        // Gemma folds system into a user turn
        result.push_str(&format!(
            "<start_of_turn>user\n{}\n\n{}<end_of_turn>\n",
            sys, user_message
        ));
    } else {
        result.push_str(&format!(
            "<start_of_turn>user\n{}<end_of_turn>\n",
            user_message
        ));
    }
    result.push_str("<start_of_turn>model\n");
    result
}

/// Format message using Mistral template for inference.
fn format_mistral_inference(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::from("[INST] ");
    if let Some(sys) = system_message {
        result.push_str(sys);
        result.push_str("\n\n");
    }
    result.push_str(user_message);
    result.push_str(" [/INST]");
    result
}

/// Format message using Phi-3 template for inference.
fn format_phi3_inference(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::new();
    if let Some(sys) = system_message {
        result.push_str("<|system|>\n");
        result.push_str(sys);
        result.push_str("<|end|>\n");
    }
    result.push_str("<|user|>\n");
    result.push_str(user_message);
    result.push_str("<|end|>\n");
    result.push_str("<|assistant|>\n");
    result
}

/// Format message using Phi-4 template for inference.
///
/// Phi-4 uses `<|im_sep|>` instead of the newline separator in standard ChatML.
fn format_phi4_inference(
    user_message: &str,
    system_message: Option<&str>,
    no_thinking: bool,
) -> String {
    let mut result = String::new();

    if let Some(sys) = system_message {
        result.push_str("<|im_start|>system<|im_sep|>");
        result.push_str(sys);
        result.push_str("<|im_end|>");
    }

    result.push_str("<|im_start|>user<|im_sep|>");
    result.push_str(user_message);
    result.push_str("<|im_end|>");
    result.push_str("<|im_start|>assistant<|im_sep|>");

    if no_thinking {
        result.push_str("<think></think>");
    }

    result
}

/// Format message using GPT-OSS Harmony template for inference.
fn format_gpt_oss_inference(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::new();
    if let Some(sys) = system_message {
        result.push_str("<|start|>system<|message|>");
        result.push_str(sys);
        result.push_str("<|end|>");
    }
    result.push_str("<|start|>user<|message|>");
    result.push_str(user_message);
    result.push_str("<|end|>");
    result.push_str("<|start|>assistant<|channel|>final<|message|>");
    result
}

/// Format message using Llama 4 template for inference.
///
/// Llama 4 uses `<|header_start|>`/`<|header_end|>` and `<|eot|>` (not Llama 3's tokens).
fn format_llama4_inference(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::from("<|begin_of_text|>");

    if let Some(sys) = system_message {
        result.push_str("<|header_start|>system<|header_end|>\n\n");
        result.push_str(sys);
        result.push_str("<|eot|>");
    }

    result.push_str("<|header_start|>user<|header_end|>\n\n");
    result.push_str(user_message);
    result.push_str("<|eot|>");
    result.push_str("<|header_start|>assistant<|header_end|>\n\n");

    result
}

/// Format message using DeepSeek template for inference.
///
/// Uses full-width unicode characters in token names.
/// V3.1+ supports thinking mode via `<think>` / `</think>` prefill.
fn format_deepseek_inference(
    user_message: &str,
    system_message: Option<&str>,
    no_thinking: bool,
) -> String {
    let mut result = String::from("<｜begin▁of▁sentence｜>");

    if let Some(sys) = system_message {
        result.push_str(sys);
    }

    result.push_str("<｜User｜>");
    result.push_str(user_message);
    result.push_str("<｜Assistant｜>");

    if no_thinking {
        result.push_str("</think>");
    } else {
        result.push_str("<think>\n");
    }

    result
}

/// Format message using Cohere Command R template for inference.
fn format_cohere_inference(user_message: &str, system_message: Option<&str>) -> String {
    let mut result = String::from("<BOS_TOKEN>");

    if let Some(sys) = system_message {
        result.push_str("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>");
        result.push_str(sys);
        result.push_str("<|END_OF_TURN_TOKEN|>");
    }

    result.push_str("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>");
    result.push_str(user_message);
    result.push_str("<|END_OF_TURN_TOKEN|>");
    result.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");

    result
}

/// Get stop tokens appropriate for a given chat template type.
///
/// Encodes the template's EOS token via the tokenizer; falls back to the generic
/// `get_eos_tokens` if encoding fails.
#[allow(dead_code)]
pub(crate) fn get_chat_stop_tokens(
    template_type: pmetal_data::chat_templates::ChatTemplateType,
    tokenizer: &Tokenizer,
) -> Vec<u32> {
    let eos_str = template_type.eos_token();
    let mut tokens = Vec::new();

    // Template-specific EOS
    if let Ok(encoded) = tokenizer.encode(eos_str) {
        if encoded.len() == 1 {
            tokens.push(encoded[0]);
        }
    }

    // Hardcoded fallbacks for common models
    if tokens.is_empty() {
        match template_type {
            pmetal_data::chat_templates::ChatTemplateType::ChatMl
            | pmetal_data::chat_templates::ChatTemplateType::Qwen
            | pmetal_data::chat_templates::ChatTemplateType::Phi4 => {
                tokens.push(151645); // <|im_end|>
            }
            pmetal_data::chat_templates::ChatTemplateType::Llama3 => {
                tokens.push(128009); // <|eot_id|>
            }
            _ => {
                if let Ok(encoded) = tokenizer.encode("</s>") {
                    if encoded.len() == 1 {
                        tokens.push(encoded[0]);
                    }
                }
                if tokens.is_empty() {
                    tokens.push(2);
                }
            }
        }
    }

    // Also include the tokenizer's native EOS — critical for base models
    // fine-tuned with LoRA that might emit either the chat EOS or the base EOS.
    if let Some(eos) = tokenizer.eos_token_id() {
        if !tokens.contains(&eos) {
            tokens.push(eos);
        }
    }

    // Probe well-known special tokens in vocabulary
    let candidates = [
        "<|im_end|>",
        "<|eot_id|>",
        "<|eot|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<end_of_turn>",
        "<|end|>",
        "<|return|>",
        "<|END_OF_TURN_TOKEN|>",
        "<｜end▁of▁sentence｜>",
        "</s>",
    ];
    for candidate in &candidates {
        if let Ok(encoded) = tokenizer.encode(candidate) {
            if encoded.len() == 1 && !tokens.contains(&encoded[0]) {
                tokens.push(encoded[0]);
            }
        }
    }

    tokens
}

/// Extract the final response after </think> tag, discarding thinking content.
///
/// Handles several cases:
/// 1. Complete thinking: `<think>...</think>response` -> returns `response`
/// 2. Incomplete thinking (hit max tokens): `<think>...` -> returns empty (model didn't finish)
/// 3. No thinking: `response` -> returns `response`
pub(crate) fn extract_final_response(text: &str) -> String {
    // Case 1: Find complete </think> tag
    if let Some(pos) = text.rfind("</think>") {
        let after_think = &text[pos + "</think>".len()..];
        // Clean up any stray <think> tags (small models sometimes output malformed content)
        let cleaned = after_think
            .trim()
            .trim_start_matches("<think>")
            .trim_start_matches('\n');
        return strip_eos_tokens(cleaned).to_string();
    }

    // Case 2: Incomplete thinking - model started <think> but never finished
    // Since there's no </think>, the model was still thinking when it hit max tokens
    if text.contains("<think>") {
        return "[Response truncated - model was still thinking. Use --no-thinking or increase --max-tokens]".to_string();
    }

    // Case 3: No thinking block, return as-is
    strip_eos_tokens(text).to_string()
}

/// Strip any known EOS / stop tokens from the end of generated text.
fn strip_eos_tokens(text: &str) -> &str {
    // Order: longest tokens first to avoid partial matches
    const EOS_TOKENS: &[&str] = &[
        "<|endoftext|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|eot|>",
        "<end_of_turn>",
        "<|END_OF_TURN_TOKEN|>",
        "<｜end▁of▁sentence｜>",
        "<|return|>",
        "<|end|>",
        "</s>",
    ];

    let mut s = text.trim();
    // Loop in case multiple EOS tokens are concatenated
    loop {
        let before = s;
        for tok in EOS_TOKENS {
            s = s.trim_end_matches(tok).trim();
        }
        if s == before {
            break;
        }
    }
    s
}

/// Extract thinking content from response (for display purposes).
///
/// Handles cases where the model generates multiple `<think>` tokens at the start
/// by finding the last complete `<think>...</think>` block.
pub(crate) fn extract_thinking_content(text: &str) -> Option<String> {
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

// NOTE: run_inference_with_lora has been removed — LoRA loading is now handled
// by InferenceRunner::prepare() via the lora_path config field.
