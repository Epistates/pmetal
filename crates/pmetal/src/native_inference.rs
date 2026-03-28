//! Native inference — all models through pmetal-bridge, zero mlx-rs.
//!
//! Each architecture has its own `run_{arch}` function that owns the full
//! pipeline: load config → load weights → prefill → decode.  The module is
//! intentionally self-contained; the only external dependencies are
//! `pmetal_bridge` and `serde_json`.

use std::path::Path;

use pmetal_bridge::InlineArray;

// ============================================================================
// Architecture enum
// ============================================================================

/// Supported model architectures for native inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeArch {
    /// Dense Qwen3 (`model_type = "qwen3"`).
    Qwen3,
    /// Qwen3.5 dense or MoE (`model_type = "qwen3_5"` / `"qwen3_5_text"`).
    Qwen3_5,
    /// Llama 4 (`model_type = "llama4"` / `"llama4_text"`).
    Llama4,
    /// DeepSeek V3/R1 (`model_type = "deepseek_v3"`).
    DeepSeek,
    /// GPT-OSS (`model_type = "gpt_oss"`).
    GptOss,
}

// ============================================================================
// Architecture detection
// ============================================================================

/// Detect architecture from `config.json`.
///
/// Checks `text_config.model_type` first (multi-modal configs), then falls
/// back to the top-level `model_type` field.
pub fn detect_arch(model_path: &Path) -> Option<NativeArch> {
    let data = std::fs::read_to_string(model_path.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&data).ok()?;

    let mt = v
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .or_else(|| v.get("model_type"))
        .and_then(|mv| mv.as_str())?;

    match mt {
        "qwen3" | "qwen3dense" => Some(NativeArch::Qwen3),
        "qwen3_5" | "qwen3_5_text" => Some(NativeArch::Qwen3_5),
        "llama4" | "llama4_text" => Some(NativeArch::Llama4),
        "deepseek_v3" => Some(NativeArch::DeepSeek),
        "gpt_oss" => Some(NativeArch::GptOss),
        _ => None,
    }
}

// ============================================================================
// Output type
// ============================================================================

/// Output produced by a native generation run.
pub struct NativeGenerationOutput {
    /// All token IDs: prompt + generated.
    pub token_ids: Vec<u32>,
    /// Number of tokens generated (excludes prompt).
    pub num_generated: usize,
    /// True when generation stopped because `on_token` returned `false`
    /// (i.e. an EOS or stop token was hit).
    pub stopped_by_token: bool,
    /// True when generation stopped because `max_tokens` was exhausted.
    pub stopped_by_length: bool,
}

// ============================================================================
// Top-level dispatch
// ============================================================================

/// Run native inference end-to-end: load → prefill → generate.
///
/// `on_token(id)` is called for every generated token; return `false` to stop
/// early (EOS, stop token, or user cancel).
///
/// Returns `Err` if the architecture is unsupported or if loading fails.
pub fn run_native_inference(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    tq_bits: Option<u8>,
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    let arch = detect_arch(model_path)
        .ok_or_else(|| "unsupported architecture for native inference".to_string())?;

    match arch {
        NativeArch::Qwen3 | NativeArch::Qwen3_5 => {
            run_qwen3(model_path, input_ids, max_tokens, temperature, tq_bits, &mut on_token)
        }
        NativeArch::Llama4 => {
            run_llama4(model_path, input_ids, max_tokens, temperature, &mut on_token)
        }
        NativeArch::DeepSeek => {
            run_deepseek(model_path, input_ids, max_tokens, temperature, &mut on_token)
        }
        NativeArch::GptOss => {
            run_gpt_oss(model_path, input_ids, max_tokens, temperature, &mut on_token)
        }
    }
}

// ============================================================================
// Shared helper
// ============================================================================

/// Convert a `&[u32]` prompt to an `InlineArray` of shape `[1, T]` (i32 dtype).
///
/// All `forward_step` implementations expect `[B, T]` int32 token IDs.
fn prompt_to_input(input_ids: &[u32]) -> InlineArray {
    let ids_i32: Vec<i32> = input_ids.iter().map(|&t| t as i32).collect();
    InlineArray::from_i32_slice(&ids_i32).reshape(&[1, ids_i32.len() as i32])
}

/// Extract last-token logits from a `[B, T, vocab]` logits tensor.
///
/// Returns a `[1, vocab]` slice, ready for `sample_token`.
fn last_token_logits(logits: &InlineArray) -> InlineArray {
    // logits: [B, T, vocab]
    let b = logits.dim(0);
    let t = logits.dim(1);
    let vocab = logits.dim(2);
    // Slice out the last sequence position: [B, T-1:T, vocab] → reshape [B, vocab].
    logits
        .slice(&[0, t - 1, 0], &[b, t, vocab])
        .reshape(&[b, vocab])
}

// ============================================================================
// Qwen3 / Qwen3.5
// ============================================================================

fn run_qwen3(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    tq_bits: Option<u8>,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::qwen3_native;

    let config = qwen3_native::load_config(model_path)?;
    eprintln!(
        "[NATIVE] Qwen3{}: {} layers, hidden={}{}",
        if config.is_moe() { " MoE" } else { "" },
        config.num_hidden_layers,
        config.hidden_size,
        if config.is_qwen3_dense() { " (Qwen3 dense)" } else { "" },
    );

    let t0 = std::time::Instant::now();
    let weights = qwen3_native::load_model(model_path, &config)?;
    eprintln!(
        "[NATIVE] Loaded in {:.1}s, active={:.0}MB",
        t0.elapsed().as_secs_f64(),
        pmetal_bridge::inline_array::get_active_memory() as f64 / 1e6,
    );

    let mut cache = if let Some(bits) = tq_bits {
        let tq_config = pmetal_bridge::turboquant::TurboQuantConfig::uniform(bits, bits);
        qwen3_native::NativeCache::new_with_turboquant(&weights, Some(tq_config))
    } else {
        qwen3_native::NativeCache::new_empty(&weights)
    };

    // Prefill
    let input = prompt_to_input(input_ids);
    let logits = qwen3_native::forward_step(&weights, &input, &mut cache);
    let last_logits = last_token_logits(&logits); // [1, vocab]

    let mut tok_arr = qwen3_native::sample_token(&last_logits, temperature);
    tok_arr.eval();
    let first_tok = tok_arr.item_u32();

    finish_generation(
        input_ids,
        first_tok,
        max_tokens,
        on_token,
        |cur_tok, cache| {
            let input = prompt_to_input(&[cur_tok]);
            let logits = qwen3_native::forward_step(&weights, &input, cache);
            let last_logits = last_token_logits(&logits);
            let mut tok_arr = qwen3_native::sample_token(&last_logits, temperature);
            tok_arr.eval();
            tok_arr.item_u32()
        },
        &mut cache,
    )
}

// ============================================================================
// Llama 4
// ============================================================================

fn run_llama4(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::llama4_native;

    let config = llama4_native::load_config(model_path)?;
    let tc = config.text();
    eprintln!(
        "[NATIVE] Llama4 MoE: {} layers, hidden={}, experts={}/tok={}",
        tc.num_hidden_layers,
        tc.hidden_size,
        tc.num_local_experts,
        tc.num_experts_per_tok,
    );

    let t0 = std::time::Instant::now();
    let weights = llama4_native::load_model(model_path, &config)?;
    eprintln!(
        "[NATIVE] Loaded in {:.1}s, active={:.0}MB",
        t0.elapsed().as_secs_f64(),
        pmetal_bridge::inline_array::get_active_memory() as f64 / 1e6,
    );

    let mut cache = llama4_native::NativeCache::new_empty(&weights);

    // Prefill
    let input = prompt_to_input(input_ids);
    let logits = llama4_native::forward_step(&weights, &input, &mut cache);
    let last_logits = last_token_logits(&logits); // [1, vocab]

    let mut tok_arr = llama4_native::sample_token(&last_logits, temperature);
    tok_arr.eval();
    let first_tok = tok_arr.item_u32();

    finish_generation(
        input_ids,
        first_tok,
        max_tokens,
        on_token,
        |cur_tok, cache| {
            let input = prompt_to_input(&[cur_tok]);
            let logits = llama4_native::forward_step(&weights, &input, cache);
            let last_logits = last_token_logits(&logits);
            let mut tok_arr = llama4_native::sample_token(&last_logits, temperature);
            tok_arr.eval();
            tok_arr.item_u32()
        },
        &mut cache,
    )
}

// ============================================================================
// DeepSeek V3/R1
// ============================================================================

fn run_deepseek(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::deepseek_native;

    let config = deepseek_native::load_config(model_path)?;
    eprintln!(
        "[NATIVE] DeepSeek V3: {} layers, hidden={}, experts={}/tok={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.n_routed_experts.unwrap_or(0),
        config.num_experts_per_tok,
    );

    let t0 = std::time::Instant::now();
    let weights = deepseek_native::load_model(model_path, &config)?;
    eprintln!(
        "[NATIVE] Loaded in {:.1}s, active={:.0}MB",
        t0.elapsed().as_secs_f64(),
        pmetal_bridge::inline_array::get_active_memory() as f64 / 1e6,
    );

    let num_layers = config.num_hidden_layers as usize;
    let mut cache = deepseek_native::NativeCache::new_empty(num_layers);

    // Prefill
    let input = prompt_to_input(input_ids);
    let logits = deepseek_native::forward_step(&weights, &input, &mut cache);
    let last_logits = last_token_logits(&logits); // [1, vocab]

    let mut tok_arr = deepseek_native::sample_token(&last_logits, temperature);
    tok_arr.eval();
    let first_tok = tok_arr.item_u32();

    finish_generation(
        input_ids,
        first_tok,
        max_tokens,
        on_token,
        |cur_tok, cache| {
            let input = prompt_to_input(&[cur_tok]);
            let logits = deepseek_native::forward_step(&weights, &input, cache);
            let last_logits = last_token_logits(&logits);
            let mut tok_arr = deepseek_native::sample_token(&last_logits, temperature);
            tok_arr.eval();
            tok_arr.item_u32()
        },
        &mut cache,
    )
}

// ============================================================================
// GPT-OSS
// ============================================================================

fn run_gpt_oss(
    model_path: &Path,
    input_ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    on_token: &mut dyn FnMut(u32) -> bool,
) -> Result<NativeGenerationOutput, String> {
    use pmetal_bridge::gpt_oss_native;

    let config = gpt_oss_native::load_config(model_path)?;
    eprintln!(
        "[NATIVE] GPT-OSS: {} layers, hidden={}, experts={}/tok={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.num_local_experts,
        config.experts_per_tok(),
    );

    let t0 = std::time::Instant::now();
    let weights = gpt_oss_native::load_model(model_path, &config)?;
    eprintln!(
        "[NATIVE] Loaded in {:.1}s, active={:.0}MB",
        t0.elapsed().as_secs_f64(),
        pmetal_bridge::inline_array::get_active_memory() as f64 / 1e6,
    );

    let mut cache = gpt_oss_native::NativeCache::new_empty(&weights);

    // Prefill
    let input = prompt_to_input(input_ids);
    let logits = gpt_oss_native::forward_step(&weights, &input, &mut cache);
    let last_logits = last_token_logits(&logits); // [1, vocab]

    let mut tok_arr = gpt_oss_native::sample_token(&last_logits, temperature);
    tok_arr.eval();
    let first_tok = tok_arr.item_u32();

    finish_generation(
        input_ids,
        first_tok,
        max_tokens,
        on_token,
        |cur_tok, cache| {
            let input = prompt_to_input(&[cur_tok]);
            let logits = gpt_oss_native::forward_step(&weights, &input, cache);
            let last_logits = last_token_logits(&logits);
            let mut tok_arr = gpt_oss_native::sample_token(&last_logits, temperature);
            tok_arr.eval();
            tok_arr.item_u32()
        },
        &mut cache,
    )
}

// ============================================================================
// Shared decode-loop driver
// ============================================================================

/// Drive the post-prefill decode loop for any architecture.
///
/// This eliminates the repetition of the "emit first token → decode loop →
/// build output" bookkeeping across the four `run_*` functions.
///
/// # Parameters
/// - `prompt`: original prompt token IDs (used only to compute `num_generated`)
/// - `first_tok`: the token sampled from the prefill logits
/// - `max_tokens`: total budget (including `first_tok`)
/// - `on_token`: streaming callback; `false` return stops generation early
/// - `step`: closure that takes `(current_token_id, &mut Cache)` and returns
///   the next sampled token
/// - `cache`: the architecture-specific cache (passed through to `step`)
///
/// The internal decode loop delegates to the architecture's own `generate()`
/// function, which handles GPU stream management, wired memory, and pipelining.
/// We only call `generate()` through the step closure so that the hot path
/// stays inside the native module with its optimised async schedule.
fn finish_generation<Cache>(
    prompt: &[u32],
    first_tok: u32,
    max_tokens: usize,
    on_token: &mut dyn FnMut(u32) -> bool,
    mut step: impl FnMut(u32, &mut Cache) -> u32,
    cache: &mut Cache,
) -> Result<NativeGenerationOutput, String> {
    let prompt_len = prompt.len();
    let mut all_tokens: Vec<u32> = prompt.to_vec();
    all_tokens.push(first_tok);

    // Notify the caller about the first generated token; stop if requested.
    if !on_token(first_tok) {
        let num_generated = all_tokens.len() - prompt_len;
        return Ok(NativeGenerationOutput {
            token_ids: all_tokens,
            num_generated,
            stopped_by_token: true,
            stopped_by_length: false,
        });
    }

    // Remaining decode budget (first_tok already consumed one slot).
    let remaining = max_tokens.saturating_sub(1);
    let mut cur_tok = first_tok;
    let mut stopped_by_token = false;

    for _ in 0..remaining {
        cur_tok = step(cur_tok, cache);
        all_tokens.push(cur_tok);
        if !on_token(cur_tok) {
            stopped_by_token = true;
            break;
        }
    }

    let num_generated = all_tokens.len() - prompt_len;
    Ok(NativeGenerationOutput {
        token_ids: all_tokens,
        num_generated,
        stopped_by_token,
        stopped_by_length: !stopped_by_token && num_generated >= max_tokens,
    })
}
