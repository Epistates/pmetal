//! Prefill / prime / generate loops, MLX-LM-compatible benchmark trials, and
//! the optional C++ decode fast path (CppForwardState / CppDecodeSession).

use crate::InlineArray;
use crate::inline_array as bridge;
use crate::inline_array::RawBuf;

use super::Qwen3Config;
use super::cache::NativeCache;
use super::forward::forward_step;
use super::weights::NativeWeights;

pub fn prefill_first_token(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    input_ids: &[u32],
    temperature: f32,
) -> u32 {
    crate::decode::prefill_first_token(weights, cache, input_ids, temperature, forward_step)
}

// ============================================================================
// Generation loop
// ============================================================================

/// Run the full generation loop with async GPU pipelining.
///
/// `first_token` is the token at the end of the prompt (already prefilled into
/// `cache`). Each call to `on_token` receives the sampled token ID and returns
/// `false` to stop early (e.g. on EOS).
///
/// Returns all generated token IDs (not including `first_token`).
fn prepare_generation_cache(cache: &mut NativeCache, reserve_decode_inputs: i32, model_dtype: i32) {
    let trace_qwen35 = std::env::var_os("PMETAL_TRACE_QWEN35").is_some();
    if trace_qwen35 {
        eprintln!("[QWEN35 TRACE] begin_generation_session before_eval_and_detach");
    }
    cache.eval_and_detach_states();
    if trace_qwen35 {
        eprintln!("[QWEN35 TRACE] begin_generation_session after_eval_and_detach");
    }
    cache.reserve_decode_inputs(reserve_decode_inputs, model_dtype);
    if trace_qwen35 {
        eprintln!(
            "[QWEN35 TRACE] begin_generation_session after_reserve decode_inputs={reserve_decode_inputs}"
        );
    }
    if std::env::var_os("PMETAL_SKIP_CLEAR_CACHE").is_none() {
        bridge::clear_cache();
        if trace_qwen35 {
            eprintln!("[QWEN35 TRACE] begin_generation_session after_clear_cache");
        }
    } else if trace_qwen35 {
        eprintln!("[QWEN35 TRACE] begin_generation_session skipped_clear_cache");
    }
}

fn prime_generation_impl(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    reserve_decode_inputs: usize,
    temperature: f32,
    reset_peak_memory: bool,
    log_session: bool,
) -> InlineArray {
    let reserve_decode_inputs = reserve_decode_inputs.min(i32::MAX as usize) as i32;
    crate::decode::prime_generation(
        "NATIVE",
        weights.model_dtype,
        weights,
        cache,
        first_token,
        temperature,
        reset_peak_memory,
        log_session,
        |cache| prepare_generation_cache(cache, reserve_decode_inputs, weights.model_dtype),
        forward_step,
    )
}

fn generate_from_primed_sample_impl(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    current_y: InlineArray,
    max_tokens: usize,
    params: crate::decode::SamplingParams,
    log_stats: bool,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    crate::decode::generate_from_primed_sample_with_params(
        "NATIVE",
        weights,
        cache,
        current_y,
        max_tokens,
        params,
        log_stats,
        on_token,
        forward_step,
    )
}

/// Prime the canonical decode loop without resetting peak memory.
///
/// This is used by the MLX-LM parity benchmark so the timing path shares the
/// same bridge decode implementation as live inference.
pub fn prime_generation_preserve_peak(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    reserve_decode_inputs: usize,
    temperature: f32,
) -> InlineArray {
    prime_generation_impl(
        weights,
        cache,
        first_token,
        reserve_decode_inputs,
        temperature,
        false,
        true,
    )
}

pub fn prime_generation_preserve_peak_silent(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    reserve_decode_inputs: usize,
    temperature: f32,
) -> InlineArray {
    prime_generation_impl(
        weights,
        cache,
        first_token,
        reserve_decode_inputs,
        temperature,
        false,
        false,
    )
}

/// Continue generation from an already-primed async sample.
///
/// `current_y` must come from [`prime_generation_preserve_peak`] or the
/// equivalent internal priming path.
pub fn generate_from_primed_sample(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    current_y: InlineArray,
    max_tokens: usize,
    temperature: f32,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    generate_from_primed_sample_impl(
        weights,
        cache,
        current_y,
        max_tokens,
        crate::decode::SamplingParams::new(temperature),
        true,
        on_token,
    )
}

/// Run one MLX-LM-style benchmark trial on the canonical Qwen native path.
///
/// The timing split matches `mlx_lm.benchmark`: prompt timing includes prefill,
/// first-token sampling, and priming the next decode step; generation timing
/// covers only the remaining decode loop.
pub fn benchmark_mlx_lm_trial(
    weights: &NativeWeights,
    prompt_ids: &[u32],
    generation_tokens: usize,
    turboquant: Option<crate::turboquant::TurboQuantConfig>,
) -> crate::decode::BenchmarkTrial {
    crate::inline_array::reset_peak_memory();
    let mut cache = NativeCache::new_with_turboquant(weights, turboquant);

    let prompt_tic = std::time::Instant::now();
    let first_tok = prefill_first_token(weights, &mut cache, prompt_ids, 0.0);
    let current_y = prime_generation_preserve_peak_silent(
        weights,
        &mut cache,
        first_tok,
        generation_tokens.saturating_sub(1),
        0.0,
    );
    let prompt_secs = prompt_tic.elapsed().as_secs_f64();

    let generation_secs = if generation_tokens > 1 {
        let generation_tic = std::time::Instant::now();
        let generated_tail = generate_from_primed_sample_silent(
            weights,
            &mut cache,
            current_y,
            generation_tokens - 1,
            0.0,
            |_| true,
        );
        debug_assert_eq!(generated_tail.len(), generation_tokens - 1);
        generation_tic.elapsed().as_secs_f64()
    } else {
        crate::inline_array::synchronize();
        f64::MIN_POSITIVE
    };

    let trial = crate::decode::BenchmarkTrial {
        prompt_secs,
        generation_secs,
        peak_memory_bytes: crate::inline_array::get_peak_memory(),
    };

    crate::inline_array::synchronize();
    crate::inline_array::clear_cache();
    trial
}

pub fn generate_from_primed_sample_silent(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    current_y: InlineArray,
    max_tokens: usize,
    temperature: f32,
    on_token: impl FnMut(u32) -> bool,
) -> Vec<u32> {
    generate_from_primed_sample_impl(
        weights,
        cache,
        current_y,
        max_tokens,
        crate::decode::SamplingParams::new(temperature),
        false,
        on_token,
    )
    .0
}

pub fn generate(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    max_tokens: usize,
    params: crate::decode::SamplingParams,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    let current_y = prime_generation_impl(
        weights,
        cache,
        first_token,
        max_tokens,
        params.temperature,
        true,
        true,
    );
    generate_from_primed_sample_impl(
        weights, cache, current_y, max_tokens, params, true, on_token,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenDecodeBackend {
    RustBridge,
}

pub fn canonical_decode_backend(
    _config: &Qwen3Config,
    _turboquant: Option<crate::turboquant::TurboQuantConfig>,
) -> QwenDecodeBackend {
    QwenDecodeBackend::RustBridge
}

#[allow(clippy::too_many_arguments)]
pub fn generate_canonical(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    config: &Qwen3Config,
    first_token: u32,
    max_tokens: usize,
    params: crate::decode::SamplingParams,
    turboquant: Option<crate::turboquant::TurboQuantConfig>,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    match canonical_decode_backend(config, turboquant) {
        QwenDecodeBackend::RustBridge => {
            generate(weights, cache, first_token, max_tokens, params, on_token)
        }
    }
}

pub fn benchmark_mlx_lm_trial_canonical(
    weights: &NativeWeights,
    config: &Qwen3Config,
    prompt_ids: &[u32],
    generation_tokens: usize,
    turboquant: Option<crate::turboquant::TurboQuantConfig>,
) -> crate::decode::BenchmarkTrial {
    match canonical_decode_backend(config, turboquant) {
        QwenDecodeBackend::RustBridge => {
            benchmark_mlx_lm_trial(weights, prompt_ids, generation_tokens, turboquant)
        }
    }
}

pub fn generate_preserve_peak(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    first_token: u32,
    max_tokens: usize,
    temperature: f32,
    on_token: impl FnMut(u32) -> bool,
) -> (Vec<u32>, Option<crate::decode::DecodeMetrics>) {
    let current_y = prime_generation_impl(
        weights,
        cache,
        first_token,
        max_tokens,
        temperature,
        false,
        true,
    );
    generate_from_primed_sample_impl(
        weights,
        cache,
        current_y,
        max_tokens,
        crate::decode::SamplingParams::new(temperature),
        true,
        on_token,
    )
}

// ============================================================================
// C++ monolithic per-token generation loop
// ============================================================================

fn begin_cpp_generation_session<'a>(
    weights: &'a NativeWeights,
    cache: &'a mut NativeCache,
    config: &Qwen3Config,
    first_token: u32,
    temperature: f32,
    reset_peak_memory: bool,
    log_session: bool,
) -> (CppDecodeSession<'a>, InlineArray) {
    if reset_peak_memory {
        crate::decode::begin_generation_session("NATIVE-CPP", weights.model_dtype);
    } else if log_session {
        crate::decode::begin_generation_session_preserve_peak("NATIVE-CPP", weights.model_dtype);
    } else {
        crate::decode::begin_generation_session_preserve_peak_silent(
            "NATIVE-CPP",
            weights.model_dtype,
        );
    }

    cache.eval_and_detach_states();
    bridge::clear_cache();

    let mut session = start_cpp_decode_session(weights, cache, config);
    let logits = session.step(first_token);
    let logits_2d = logits.squeeze(1);
    let current_y = crate::decode::sample_token(&logits_2d, temperature);
    current_y.async_eval_ref();
    (session, current_y)
}

fn generate_from_primed_cpp_session(
    mut session: CppDecodeSession<'_>,
    mut current_y: InlineArray,
    max_tokens: usize,
    temperature: f32,
    log_stats: bool,
    mut on_token: impl FnMut(u32) -> bool,
) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut step_times: Vec<f64> = Vec::new();

    for step in 0..max_tokens {
        if step == 0 {
            current_y.eval();
        }
        let token_val = current_y.item_u32();

        tokens.push(token_val);
        if !on_token(token_val) {
            break;
        }
        if step + 1 >= max_tokens {
            break;
        }

        let t_step = std::time::Instant::now();
        let next_logits = session.step(token_val);
        let next_logits_2d = next_logits.squeeze(1);
        current_y = crate::decode::sample_token(&next_logits_2d, temperature);
        current_y.async_eval_ref();

        step_times.push(t_step.elapsed().as_secs_f64() * 1000.0);

        if step % 256 == 255 {
            bridge::clear_cache();
        }
    }

    if log_stats && step_times.len() > 20 {
        step_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let skip = 10;
        let avg = step_times[skip..].iter().sum::<f64>() / (step_times.len() - skip) as f64;
        let p50 = step_times[step_times.len() / 2];
        eprintln!(
            "[NATIVE-CPP] per-step: avg={avg:.2}ms p50={p50:.2}ms = {:.0} tok/s",
            1000.0 / avg
        );
    }

    drop(session);
    bridge::synchronize();
    tokens
}

/// Generation loop using the C++ monolithic per-token forward path.
///
/// Equivalent to [`generate`] but each decode step executes all per-layer ops
/// inside a single C++ function call (`mlx_inline_qwen35_decode_step`), which
/// removes per-op FFI overhead while still using the same bridge-native MLX
/// tensors and cache ownership as the Rust path.
#[allow(dead_code)]
fn generate_cpp(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    config: &Qwen3Config,
    first_token: u32,
    max_tokens: usize,
    temperature: f32,
    on_token: impl FnMut(u32) -> bool,
) -> Vec<u32> {
    if !supports_cpp_decode(config) {
        return generate(
            weights,
            cache,
            first_token,
            max_tokens,
            crate::decode::SamplingParams::new(temperature),
            on_token,
        )
        .0;
    }

    let (session, current_y) =
        begin_cpp_generation_session(weights, cache, config, first_token, temperature, true, true);
    generate_from_primed_cpp_session(session, current_y, max_tokens, temperature, true, on_token)
}

#[allow(dead_code)]
fn benchmark_mlx_lm_trial_cpp(
    weights: &NativeWeights,
    config: &Qwen3Config,
    prompt_ids: &[u32],
    generation_tokens: usize,
) -> crate::decode::BenchmarkTrial {
    crate::inline_array::reset_peak_memory();
    let mut cache = NativeCache::new_empty(weights);

    let prompt_tic = std::time::Instant::now();
    let first_tok = prefill_first_token(weights, &mut cache, prompt_ids, 0.0);
    let (session, current_y) =
        begin_cpp_generation_session(weights, &mut cache, config, first_tok, 0.0, false, false);
    let prompt_secs = prompt_tic.elapsed().as_secs_f64();

    let generation_secs = if generation_tokens > 1 {
        let generation_tic = std::time::Instant::now();
        let generated_tail = generate_from_primed_cpp_session(
            session,
            current_y,
            generation_tokens - 1,
            0.0,
            false,
            |_| true,
        );
        debug_assert_eq!(generated_tail.len(), generation_tokens - 1);
        generation_tic.elapsed().as_secs_f64()
    } else {
        crate::inline_array::synchronize();
        f64::MIN_POSITIVE
    };

    let trial = crate::decode::BenchmarkTrial {
        prompt_secs,
        generation_secs,
        peak_memory_bytes: crate::inline_array::get_peak_memory(),
    };

    crate::inline_array::synchronize();
    crate::inline_array::clear_cache();
    trial
}

#[allow(dead_code)]
fn supports_cpp_decode(config: &Qwen3Config) -> bool {
    let is_quantized = config.quantization_config.is_some();
    !config.is_qwen3_dense() && !is_quantized
}

fn sync_cpp_state_back(cache: &mut NativeCache, state: &CppForwardState) {
    cache.rope_offset = state.rope_offset;
    for (layer_cache, offset) in cache.kv_caches.iter_mut().zip(state.attn_kv_offsets.iter()) {
        layer_cache.offset = *offset;
    }
}

// ============================================================================
// C++ monolithic per-token path
// ============================================================================
//
// `CppForwardState` packages the flat weight pointer arrays, config int/float
// arrays, and the mutable cache pointer arrays required by
// `mlx_inline_qwen35_decode_step`. It is built once from `NativeWeights` +
// `NativeCache` and then passed to `forward_step_cpp_with_token` on every
// decode step.
//
// Layout matches the documentation in `bridge.h`:
//
//   weight_ptrs:  [embed_w, final_norm_w, lm_head_w, layer_0_block, ..., layer_N-1_block]
//                 where each layer block is QWEN35_WEIGHTS_PER_LAYER pointers.
//
//   cache_ptrs:   [gdn_0_conv, gdn_0_ssm, ..., gdn_{n_gdn-1}_ssm,
//                  attn_0_keys, attn_0_vals, ..., attn_{n_attn-1}_vals]
//                 n_attn cache slots = n_attn * 4 (keys + vals + 2 reserved/future slots)
//                 Actually: n_gdn*2 + n_attn*4 — but we only use keys+vals (2 slots each).
//                 NOTE: the bridge contract uses n_attn*4; slots +2 and +3 are zero-init
//                 sentinels included so that the cache pointer array is uniformly spaced.
//
// IMPORTANT: `CppForwardState` stores RAW POINTERS into `NativeWeights` and
// `NativeCache` arrays.  The caller MUST ensure both outlive the state.

const WEIGHTS_PER_LAYER: usize = 21;
#[allow(dead_code)]
pub struct CppForwardState {
    // Flat weight pointer array (const *const RawBuf).
    // All None slots (attention layers' GDN slots, etc.) are filled with a
    // dummy sentinel InlineArray that the C++ side never dereferences.
    weight_storage: Vec<InlineArray>, // owns sentinel arrays (indices where weight is absent)
    weight_ptrs: Vec<*const RawBuf>, // flat pointer array, length = 3 + num_layers * WEIGHTS_PER_LAYER

    // Flat cache pointer array (mutable, in/out).
    // n_gdn*2 slots for GDN + n_attn*4 slots for attn (keys, vals, sent, sent).
    cache_ptrs: Vec<*mut RawBuf>,

    // Scalar cache — updated by C++ in-place.
    pub attn_kv_offsets: Vec<i32>, // [n_attn]
    pub rope_offset: i32,

    // Config arrays
    config_ints: Vec<i32>,
    config_floats: Vec<f32>,

    // Counts for bounds checking / documentation
    n_gdn: usize,
    n_attn: usize,
    num_layers: usize,
}

// SAFETY: CppForwardState is only used from a single thread per generation
// step (the Rust caller holds &mut NativeCache).  Raw pointers into
// NativeWeights/NativeCache are stable because those structures never
// reallocate their InlineArray storage once constructed.
unsafe impl Send for CppForwardState {}
unsafe impl Sync for CppForwardState {}

pub struct CppDecodeSession<'a> {
    state: CppForwardState,
    cache: &'a mut NativeCache,
    _weights: std::marker::PhantomData<&'a NativeWeights>,
}

impl CppDecodeSession<'_> {
    pub fn step(&mut self, token_id: u32) -> InlineArray {
        // SAFETY: `start_cpp_decode_session` ties the session lifetime to the
        // borrowed weights/cache, so the raw pointers captured in `state`
        // remain valid for the whole session.
        unsafe { forward_step_cpp_with_token(&mut self.state, token_id) }
    }
}

impl Drop for CppDecodeSession<'_> {
    fn drop(&mut self) {
        sync_cpp_state_back(self.cache, &self.state);
    }
}

#[allow(dead_code)]
fn start_cpp_decode_session<'a>(
    weights: &'a NativeWeights,
    cache: &'a mut NativeCache,
    config: &Qwen3Config,
) -> CppDecodeSession<'a> {
    // SAFETY: the returned session borrows both `weights` and `cache` for its
    // lifetime, so neither can be moved or dropped while the raw-pointer state
    // is in use.
    let state = unsafe { build_cpp_forward_state(weights, cache, config) };
    CppDecodeSession {
        state,
        cache,
        _weights: std::marker::PhantomData,
    }
}

// Sentinel zero array used to fill unused weight slots.
fn sentinel() -> InlineArray {
    InlineArray::from_f32(0.0)
}

/// Build a `CppForwardState` from weights + cache.
///
/// This is called ONCE after `load_model` + `NativeCache::new_empty`. The
/// returned state holds raw pointers into `weights` and `cache`; both must
/// remain live and un-moved for the state's lifetime.
///
/// # Safety
/// `weights` and `cache` must not be moved or dropped while `CppForwardState`
/// is alive. In practice both live in the same generation loop scope.
pub unsafe fn build_cpp_forward_state(
    weights: &NativeWeights,
    cache: &mut NativeCache,
    config: &Qwen3Config,
) -> CppForwardState {
    let num_layers = weights.layers.len();
    let n_gdn = cache.gdn_caches.len();
    let n_attn = cache.kv_caches.len();

    // Compute counts for the config arrays.
    let n_config_floats = 4 + num_layers * 2 + n_gdn + n_attn * 2;
    let n_config_ints = 20 + num_layers;

    // ── Build config_ints ──────────────────────────────────────────────────
    let gdn_nv = config.gdn_nv();
    let gdn_nk = config.gdn_nk();
    let gdn_dk = config.gdn_dk();
    let gdn_dv = config.gdn_dv();
    let ck = config.linear_conv_kernel_dim;
    let kd = gdn_nk * gdn_dk;
    let cd = kd * 2 + gdn_nv * gdn_dv;
    let n_heads = config.num_attention_heads;
    let n_kv = config.get_num_kv_heads();
    let head_dim = config.get_head_dim();
    let rope_dims = config.rope_dims();

    let mut config_ints = Vec::with_capacity(n_config_ints);
    config_ints.extend_from_slice(&[
        num_layers as i32,                               // [0]
        config.hidden_size,                              // [1]
        weights.model_dtype,                             // [2]
        n_gdn as i32,                                    // [3]
        n_attn as i32,                                   // [4]
        gdn_nv,                                          // [5]
        gdn_nk,                                          // [6]
        gdn_dk,                                          // [7]
        gdn_dv,                                          // [8]
        cd,                                              // [9]  gdn_cd
        ck,                                              // [10] gdn_ck
        kd,                                              // [11] gdn_kd
        n_heads,                                         // [12]
        n_kv,                                            // [13]
        head_dim,                                        // [14]
        rope_dims,                                       // [15]
        config.full_attention_interval,                  // [16]
        if weights.tie_word_embeddings { 1 } else { 0 }, // [17]
        config.num_experts_per_tok,                      // [18]
        if config.norm_topk_prob { 1 } else { 0 },       // [19]
    ]);
    for lw in &weights.layers {
        config_ints.push(if lw.is_moe_layer { 1 } else { 0 });
    }

    // ── Build config_floats ────────────────────────────────────────────────
    let attn_scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut config_floats = Vec::with_capacity(n_config_floats);
    config_floats.push(weights.final_norm_eps); // [0]
    config_floats.push(attn_scale); // [1]
    config_floats.push(config.rope_theta as f32); // [2]
    config_floats.push(1.0_f32); // [3] rope_scale

    // Per-layer norm eps (input + post)
    for lw in &weights.layers {
        config_floats.push(lw.input_ln_eps);
        config_floats.push(lw.post_ln_eps);
    }
    // GDN norm eps
    for lw in &weights.layers {
        if lw.is_linear {
            config_floats.push(lw.gdn_norm_eps);
        }
    }
    // Attention Q/K norm eps
    for lw in &weights.layers {
        if !lw.is_linear {
            config_floats.push(lw.attn_q_norm_eps);
            config_floats.push(lw.attn_k_norm_eps);
        }
    }

    // ── Build weight_ptrs ──────────────────────────────────────────────────
    let total_weight_slots = 3 + num_layers * WEIGHTS_PER_LAYER;
    let mut weight_storage: Vec<InlineArray> = Vec::new();
    let mut weight_ptrs: Vec<*const RawBuf> = Vec::with_capacity(total_weight_slots);

    let push_real = |ptrs: &mut Vec<*const RawBuf>, w: &InlineArray| {
        ptrs.push(w.as_raw_ptr());
    };
    let push_sent = |ptrs: &mut Vec<*const RawBuf>, storage: &mut Vec<InlineArray>| {
        storage.push(sentinel());
        ptrs.push(storage.last().unwrap().as_raw_ptr());
    };

    // Global weights [0..3)
    push_real(&mut weight_ptrs, &weights.embed_w);
    push_real(&mut weight_ptrs, &weights.final_norm_w);
    if let Some(ref lm) = weights.lm_head_w {
        push_real(&mut weight_ptrs, lm.weight_arr());
    } else {
        push_sent(&mut weight_ptrs, &mut weight_storage);
    }

    // Per-layer weight blocks [3 + li*WEIGHTS_PER_LAYER .. 3 + (li+1)*WEIGHTS_PER_LAYER)
    // Slot layout (21 per layer):
    //   0: input_ln_w
    //   1: post_ln_w
    //   2: dense mlp_gate_w / moe_router_w
    //   3: dense mlp_up_w   / moe_gate_w
    //   4: dense mlp_down_w / moe_up_w
    //   5: attn_q_w   / gdn_qkv_w
    //   6: attn_k_w   / gdn_z_w
    //   7: attn_v_w   / gdn_b_w
    //   8: attn_o_w   / gdn_a_w
    //   9: attn_q_norm_w / gdn_conv_w
    //  10: attn_k_norm_w / gdn_q_nw
    //  11: gdn_k_nw
    //  12: gdn_a_log
    //  13: gdn_dt_bias
    //  14: gdn_norm_w
    //  15: gdn_out_w
    //  16: moe_down_w
    //  17: shared_gate_w
    //  18: shared_up_w
    //  19: shared_down_w
    //  20: shared_expert_gate_w
    for lw in &weights.layers {
        push_real(&mut weight_ptrs, &lw.input_ln_w);
        push_real(&mut weight_ptrs, &lw.post_ln_w);

        // MLP prefix slots. Dense layers expose gate/up/down; MoE layers expose
        // router/gate/up so the C++ path can execute either post-attention block.
        if lw.is_moe_layer {
            if let Some(w) = &lw.moe_router_w {
                push_real(&mut weight_ptrs, w);
            } else {
                push_sent(&mut weight_ptrs, &mut weight_storage);
            }
            for opt in [&lw.moe_gate_w, &lw.moe_up_w] {
                if let Some(w) = opt {
                    push_real(&mut weight_ptrs, w.weight_arr());
                } else {
                    push_sent(&mut weight_ptrs, &mut weight_storage);
                }
            }
        } else {
            for opt in [&lw.mlp_gate_w, &lw.mlp_up_w, &lw.mlp_down_w] {
                if let Some(w) = opt {
                    push_real(&mut weight_ptrs, w.weight_arr());
                } else {
                    push_sent(&mut weight_ptrs, &mut weight_storage);
                }
            }
        }

        if lw.is_linear {
            // GDN slots — mixed types: LayerWeight for projections, InlineArray for small tensors.
            // Projections (LayerWeight):
            for opt in [&lw.gdn_qkv_w, &lw.gdn_z_w, &lw.gdn_b_w, &lw.gdn_a_w] {
                if let Some(w) = opt {
                    push_real(&mut weight_ptrs, w.weight_arr());
                } else {
                    push_sent(&mut weight_ptrs, &mut weight_storage);
                }
            }
            // Small tensors (InlineArray):
            for opt in [
                &lw.gdn_conv_w,
                &lw.gdn_q_nw,
                &lw.gdn_k_nw,
                &lw.gdn_a_log,
                &lw.gdn_dt_bias,
                &lw.gdn_norm_w,
            ] {
                if let Some(w) = opt {
                    push_real(&mut weight_ptrs, w);
                } else {
                    push_sent(&mut weight_ptrs, &mut weight_storage);
                }
            }
            // out_proj (LayerWeight):
            if let Some(w) = &lw.gdn_out_w {
                push_real(&mut weight_ptrs, w.weight_arr());
            } else {
                push_sent(&mut weight_ptrs, &mut weight_storage);
            }
        } else {
            // Attention projection slots (LayerWeight):
            for opt in [&lw.attn_q_w, &lw.attn_k_w, &lw.attn_v_w, &lw.attn_o_w] {
                if let Some(w) = opt {
                    push_real(&mut weight_ptrs, w.weight_arr());
                } else {
                    push_sent(&mut weight_ptrs, &mut weight_storage);
                }
            }
            // Norm slots (InlineArray):
            for opt in [&lw.attn_q_norm_w, &lw.attn_k_norm_w] {
                if let Some(w) = opt {
                    push_real(&mut weight_ptrs, w);
                } else {
                    push_sent(&mut weight_ptrs, &mut weight_storage);
                }
            }
            // Attention layers do not use the GDN-only slots.
            for _ in 0..5 {
                push_sent(&mut weight_ptrs, &mut weight_storage);
            }
        }

        if lw.is_moe_layer {
            if let Some(w) = &lw.moe_down_w {
                push_real(&mut weight_ptrs, w.weight_arr());
            } else {
                push_sent(&mut weight_ptrs, &mut weight_storage);
            }
            for opt in [&lw.shared_gate_w, &lw.shared_up_w, &lw.shared_down_w] {
                if let Some(w) = opt {
                    push_real(&mut weight_ptrs, w.weight_arr());
                } else {
                    push_sent(&mut weight_ptrs, &mut weight_storage);
                }
            }
            if let Some(w) = &lw.shared_expert_gate_w {
                push_real(&mut weight_ptrs, w);
            } else {
                push_sent(&mut weight_ptrs, &mut weight_storage);
            }
        } else {
            for _ in 0..5 {
                push_sent(&mut weight_ptrs, &mut weight_storage);
            }
        }
    }

    // ── Build cache_ptrs ───────────────────────────────────────────────────
    let total_cache_slots = n_gdn * 2 + n_attn * 4;
    let mut cache_ptrs: Vec<*mut RawBuf> = Vec::with_capacity(total_cache_slots);

    for gc in &mut cache.gdn_caches {
        if let Some(ref mut s) = gc.conv_state {
            cache_ptrs.push(s.as_raw_ptr_mut());
        } else {
            weight_storage.push(sentinel());
            cache_ptrs.push(weight_storage.last_mut().unwrap().as_raw_ptr_mut());
        }
        if let Some(ref mut s) = gc.ssm_state {
            cache_ptrs.push(s.as_raw_ptr_mut());
        } else {
            weight_storage.push(sentinel());
            cache_ptrs.push(weight_storage.last_mut().unwrap().as_raw_ptr_mut());
        }
    }

    let attn_kv_offsets: Vec<i32> = cache.kv_caches.iter().map(|c| c.offset).collect();
    for kvc in &mut cache.kv_caches {
        if let Some(ref mut k) = kvc.keys {
            cache_ptrs.push(k.as_raw_ptr_mut());
        } else {
            weight_storage.push(sentinel());
            cache_ptrs.push(weight_storage.last_mut().unwrap().as_raw_ptr_mut());
        }
        if let Some(ref mut v) = kvc.values {
            cache_ptrs.push(v.as_raw_ptr_mut());
        } else {
            weight_storage.push(sentinel());
            cache_ptrs.push(weight_storage.last_mut().unwrap().as_raw_ptr_mut());
        }
        // 2 sentinel padding slots (bridge contract: 4 slots per attn layer)
        for _ in 0..2 {
            weight_storage.push(sentinel());
            cache_ptrs.push(weight_storage.last_mut().unwrap().as_raw_ptr_mut());
        }
    }

    CppForwardState {
        weight_storage,
        weight_ptrs,
        cache_ptrs,
        attn_kv_offsets,
        rope_offset: cache.rope_offset,
        config_ints,
        config_floats,
        n_gdn,
        n_attn,
        num_layers,
    }
}

/// Run one forward step using the C++ monolithic per-token path.
///
/// This still builds the MLX graph for the full model on each call; it only
/// avoids Rust-side per-op FFI traffic by doing the work inside one C++ entry
/// point.
///
/// # Safety
///
/// The `state` must have been created by `build_cpp_forward_state` with valid
/// weight and cache pointers that outlive this call.
#[allow(dead_code)]
pub unsafe fn forward_step_cpp_with_token(
    state: &mut CppForwardState,
    token_id: u32,
) -> InlineArray {
    let token_ids = InlineArray::from_i32(token_id as i32).reshape(&[1, 1]);
    // SAFETY: caller guarantees weight/cache pointers are valid (upheld by build_cpp_forward_state).
    unsafe {
        bridge::qwen35_decode_step(
            &token_ids,
            &state.weight_ptrs,
            &mut state.cache_ptrs,
            &mut state.attn_kv_offsets,
            &mut state.rope_offset,
            &state.config_ints,
            &state.config_floats,
        )
    }
}

