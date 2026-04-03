//! Zero-allocation MLX array — stores `mlx::core::array` inline on the Rust stack.
//!
//! This eliminates ALL per-op heap allocation, matching Python/nanobind's direct
//! C++ binding performance. Each op is a single `extern "C"` call with placement-new
//! into a caller-provided buffer.

use std::mem::MaybeUninit;

use memmap2::Mmap;
use safetensors::{Dtype as SafeDtype, SafeTensors};

/// Size of `mlx::core::array` in bytes. Must match MLX_ARRAY_SIZE in bridge.h.
const ARRAY_BUF_SIZE: usize = 128;
/// Alignment of `mlx::core::array`.
const ARRAY_BUF_ALIGN: usize = 8;

/// Raw inline array buffer — matches `mlx_inline_array` in C.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub(crate) struct RawBuf {
    pub(crate) buf: [u8; ARRAY_BUF_SIZE],
}

#[allow(dead_code)]
unsafe extern "C" {
    fn mlx_inline_destroy(a: *mut RawBuf);
    fn mlx_inline_init_copy(dst: *mut RawBuf, src: *const RawBuf);
    fn mlx_inline_from_handle(dst: *mut RawBuf, handle_ctx: *mut std::ffi::c_void);
    fn mlx_inline_to_handle(src: *const RawBuf) -> *mut std::ffi::c_void;

    fn mlx_inline_matmul(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    fn mlx_inline_add(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    fn mlx_inline_multiply(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    fn mlx_inline_subtract(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    fn mlx_inline_divide(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    fn mlx_inline_negative(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_exp(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_sigmoid(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_silu(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_softmax(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    fn mlx_inline_sqrt(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_transpose(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_reshape(dst: *mut RawBuf, a: *const RawBuf, shape: *const i32, ndim: i32);
    fn mlx_inline_sum_axis(dst: *mut RawBuf, a: *const RawBuf, axis: i32, keepdims: bool);
    fn mlx_inline_astype(dst: *mut RawBuf, a: *const RawBuf, dtype: i32);

    fn mlx_inline_gather_mm(
        dst: *mut RawBuf,
        a: *const RawBuf,
        b: *const RawBuf,
        lhs: *const RawBuf,
        rhs: *const RawBuf,
        sorted: bool,
    );
    fn mlx_inline_rms_norm(dst: *mut RawBuf, x: *const RawBuf, w: *const RawBuf, eps: f32);
    fn mlx_inline_rope(
        dst: *mut RawBuf,
        x: *const RawBuf,
        dims: i32,
        trad: bool,
        base: f32,
        scale: f32,
        off: i32,
    );
    fn mlx_inline_sdpa(
        dst: *mut RawBuf,
        q: *const RawBuf,
        k: *const RawBuf,
        v: *const RawBuf,
        scale: f32,
        mode: *const std::ffi::c_char,
    );

    fn mlx_inline_split(
        input: *const RawBuf,
        indices: *const i32,
        n: i32,
        axis: i32,
        out: *mut RawBuf,
    );
    fn mlx_inline_concatenate(dst: *mut RawBuf, arrays: *const RawBuf, num: i32, axis: i32);
    fn mlx_inline_argpartition(dst: *mut RawBuf, a: *const RawBuf, kth: i32, axis: i32);
    fn mlx_inline_take_along_axis(
        dst: *mut RawBuf,
        a: *const RawBuf,
        idx: *const RawBuf,
        axis: i32,
    );

    fn mlx_inline_eval(a: *mut RawBuf);
    fn mlx_inline_async_eval(a: *mut RawBuf);
    fn mlx_inline_from_f32(dst: *mut RawBuf, val: f32);
    fn mlx_inline_from_i32(dst: *mut RawBuf, val: i32);

    fn mlx_inline_ndim(a: *const RawBuf) -> i32;
    fn mlx_inline_dim(a: *const RawBuf, axis: i32) -> i32;
    fn mlx_inline_shape(a: *const RawBuf) -> *const i32;
    fn mlx_inline_dtype(a: *const RawBuf) -> i32;
    fn mlx_inline_item_f32(a: *mut RawBuf) -> f32;
    fn mlx_inline_item_u32(a: *mut RawBuf) -> u32;

    fn mlx_inline_sign(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_dequantize(
        dst: *mut RawBuf,
        w: *const RawBuf,
        scales: *const RawBuf,
        biases: *const RawBuf,
        group_size: i32,
        bits: i32,
    );
    fn mlx_inline_from_f32_slice(dst: *mut RawBuf, data: *const f32, shape: *const i32, ndim: i32);
    fn mlx_inline_from_u32_slice(dst: *mut RawBuf, data: *const u32, shape: *const i32, ndim: i32);
    fn mlx_inline_from_u8_slice(dst: *mut RawBuf, data: *const u8, shape: *const i32, ndim: i32);
    fn mlx_inline_from_u16_bits_slice(
        dst: *mut RawBuf,
        data: *const u16,
        shape: *const i32,
        ndim: i32,
        dtype: i32,
    );
    fn mlx_inline_to_f32_slice(a: *mut RawBuf, out: *mut f32, n: usize) -> i32;
    fn mlx_inline_stack(dst: *mut RawBuf, arrays: *const RawBuf, num: i32, axis: i32);
    fn mlx_inline_norm_l2(dst: *mut RawBuf, a: *const RawBuf, axis: i32, keepdims: bool);

    fn mlx_inline_conv1d(
        dst: *mut RawBuf,
        input: *const RawBuf,
        weight: *const RawBuf,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    );

    fn mlx_inline_array_size() -> usize;
    fn mlx_inline_array_align() -> usize;

    fn mlx_inline_gdn_update(
        dst_y: *mut RawBuf,
        dst_state: *mut RawBuf,
        q: *const RawBuf,
        k: *const RawBuf,
        v: *const RawBuf,
        a: *const RawBuf,
        b: *const RawBuf,
        a_log: *const RawBuf,
        dt_bias: *const RawBuf,
        state_in: *const RawBuf,
        training: bool,
    ) -> i32;

    fn mlx_inline_set_wired_limit(limit: usize) -> usize;
    fn mlx_inline_get_max_recommended_size() -> usize;
    fn mlx_inline_new_stream() -> i32;
    fn mlx_inline_set_default_stream(index: i32);
    fn mlx_inline_synchronize();
    fn mlx_inline_clear_cache();
    fn mlx_inline_set_cache_limit(limit: usize) -> usize;
    fn mlx_inline_enable_compile();
    fn mlx_inline_disable_compile();
    fn mlx_inline_graph_node_count(a: *const RawBuf) -> usize;
    fn mlx_inline_graph_desc_count(a: *const RawBuf) -> usize;
    fn mlx_inline_graph_dump(a: *const RawBuf);
    fn mlx_inline_metal_start_capture(path: *const std::ffi::c_char) -> i32;
    fn mlx_inline_metal_stop_capture();

    // Compiled GDN layer — entire layer as single compiled function
    fn mlx_inline_compiled_gdn_layer(
        dst_out: *mut RawBuf,
        dst_conv: *mut RawBuf,
        dst_ssm: *mut RawBuf,
        normed: *const RawBuf,
        qkv_w: *const RawBuf,
        z_w: *const RawBuf,
        b_w: *const RawBuf,
        a_w: *const RawBuf,
        conv_w: *const RawBuf,
        q_nw: *const RawBuf,
        k_nw: *const RawBuf,
        a_log: *const RawBuf,
        dt_bias: *const RawBuf,
        norm_w: *const RawBuf,
        out_w: *const RawBuf,
        conv_state: *const RawBuf,
        ssm_state: *const RawBuf,
        nv: i32,
        nk: i32,
        dk: i32,
        dv: i32,
        cd: i32,
        ck: i32,
        kd: i32,
        norm_eps: f32,
    );

    // Fixed-shape compiled GDN layer (shapeless=false, works with ALL primitives)
    fn mlx_inline_compiled_gdn_layer_fixed(
        dst_out: *mut RawBuf,
        dst_conv: *mut RawBuf,
        dst_ssm: *mut RawBuf,
        normed: *const RawBuf,
        qkv_w: *const RawBuf,
        z_w: *const RawBuf,
        b_w: *const RawBuf,
        a_w: *const RawBuf,
        conv_w: *const RawBuf,
        q_nw: *const RawBuf,
        k_nw: *const RawBuf,
        a_log: *const RawBuf,
        dt_bias: *const RawBuf,
        norm_w: *const RawBuf,
        out_w: *const RawBuf,
        conv_state: *const RawBuf,
        ssm_state: *const RawBuf,
        nv: i32,
        nk: i32,
        dk: i32,
        dv: i32,
        cd: i32,
        ck: i32,
        kd: i32,
        norm_eps: f32,
    );

    fn mlx_inline_compiled_attn_layer_fixed(
        dst_out: *mut RawBuf,
        dst_cache_keys: *mut RawBuf,
        dst_cache_vals: *mut RawBuf,
        normed: *const RawBuf,
        q_w: *const RawBuf,
        k_w: *const RawBuf,
        v_w: *const RawBuf,
        o_w: *const RawBuf,
        q_nw: *const RawBuf,
        k_nw: *const RawBuf,
        cache_keys_in: *const RawBuf,
        cache_vals_in: *const RawBuf,
        kv_offset: i32,
        rope_offset: i32,
        n_heads: i32,
        n_kv: i32,
        head_dim: i32,
        scale: f32,
        rope_dims: i32,
        rope_base: f32,
        rope_scale: f32,
        q_norm_eps: f32,
        k_norm_eps: f32,
        gated: bool,
    );

    fn mlx_inline_compiled_moe_layer_fixed(
        dst_out: *mut RawBuf,
        x: *const RawBuf,
        router_w: *const RawBuf,
        moe_gate_w: *const RawBuf,
        moe_up_w: *const RawBuf,
        moe_down_w: *const RawBuf,
        shared_gate_w: *const RawBuf,
        shared_up_w: *const RawBuf,
        shared_down_w: *const RawBuf,
        shared_expert_gate_w: *const RawBuf,
        top_k: i32,
        norm_topk_prob: bool,
    );

    // Arange — non-broadcast tensor creation
    fn mlx_inline_arange(dst: *mut RawBuf, n: i32, dtype: i32);
    fn mlx_inline_load_safetensors_key(
        dst: *mut RawBuf,
        path: *const std::ffi::c_char,
        key: *const std::ffi::c_char,
    ) -> i32;

    // Graph detach — severs computation graph references
    fn mlx_inline_detach(a: *mut RawBuf);

    // Batch eval — single GPU submission for multiple arrays
    fn mlx_inline_eval_many(arrays: *mut *mut RawBuf, count: i32);
    fn mlx_inline_async_eval_many(arrays: *mut *mut RawBuf, count: i32);

    // Metal memory instrumentation
    fn mlx_inline_get_active_memory() -> usize;
    fn mlx_inline_get_cache_memory() -> usize;
    fn mlx_inline_get_peak_memory() -> usize;
    fn mlx_inline_reset_peak_memory();

    // ── FFT ops ──
    fn mlx_inline_rfft(dst: *mut RawBuf, a: *const RawBuf, n_fft: i32, axis: i32);
    fn mlx_inline_irfft(dst: *mut RawBuf, a: *const RawBuf, n_fft: i32, axis: i32);

    // ── leaky_relu ──
    fn mlx_inline_leaky_relu(dst: *mut RawBuf, a: *const RawBuf, neg_slope: f32);

    // ── squeeze_all (remove all size-1 axes) ──
    fn mlx_inline_squeeze_all(dst: *mut RawBuf, a: *const RawBuf);

    // ── pad ──
    fn mlx_inline_pad(
        dst: *mut RawBuf,
        a: *const RawBuf,
        pad_widths: *const i32,
        ndim: i32,
        fill_value: f32,
    );

    // ── Additional ops for complete model inference ──
    fn mlx_inline_concatenate_2(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf, axis: i32);
    fn mlx_inline_softplus(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_where(dst: *mut RawBuf, cond: *const RawBuf, a: *const RawBuf, b: *const RawBuf);
    fn mlx_inline_maximum(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    fn mlx_inline_zeros(dst: *mut RawBuf, shape: *const i32, ndim: i32, dtype: i32);
    fn mlx_inline_ones(dst: *mut RawBuf, shape: *const i32, ndim: i32, dtype: i32);
    fn mlx_inline_slice(
        dst: *mut RawBuf,
        a: *const RawBuf,
        start: *const i32,
        stop: *const i32,
        ndim: i32,
    );
    fn mlx_inline_slice_set(
        dst: *mut RawBuf,
        a: *const RawBuf,
        val: *const RawBuf,
        start: *const i32,
        stop: *const i32,
        ndim: i32,
    );
    fn mlx_inline_repeat(dst: *mut RawBuf, a: *const RawBuf, repeats: i32, axis: i32);
    fn mlx_inline_squeeze(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    fn mlx_inline_expand_dims(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    fn mlx_inline_transpose_axes(dst: *mut RawBuf, a: *const RawBuf, axes: *const i32, ndim: i32);
    fn mlx_inline_cumsum(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    fn mlx_inline_log(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_tril(dst: *mut RawBuf, a: *const RawBuf, k: i32);
    fn mlx_inline_index(dst: *mut RawBuf, a: *const RawBuf, indices: *const RawBuf);
    fn mlx_inline_softmax_precise(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    fn mlx_inline_sdpa_with_mask(
        dst: *mut RawBuf,
        q: *const RawBuf,
        k: *const RawBuf,
        v: *const RawBuf,
        scale: f32,
        mask: *const RawBuf,
    );
    fn mlx_inline_eval_2(a: *mut RawBuf, b: *mut RawBuf);
    fn mlx_inline_quantized_matmul(
        dst: *mut RawBuf,
        x: *const RawBuf,
        w: *const RawBuf,
        scales: *const RawBuf,
        biases: *const RawBuf,
        transpose: bool,
        group_size: i32,
        bits: i32,
    );
    fn mlx_inline_gather_qmm(
        dst: *mut RawBuf,
        x: *const RawBuf,
        w: *const RawBuf,
        scales: *const RawBuf,
        biases: *const RawBuf,
        lhs: *const RawBuf,
        rhs: *const RawBuf,
        transpose: bool,
        group_size: i32,
        bits: i32,
        sorted: bool,
    );

    // ── Sampling ──
    fn mlx_inline_argmax(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    fn mlx_inline_argmin(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    fn mlx_inline_logsumexp(dst: *mut RawBuf, a: *const RawBuf, axis: i32, keepdims: bool);
    fn mlx_inline_categorical(dst: *mut RawBuf, logits: *const RawBuf);

    // ── Element-wise math ──
    fn mlx_inline_abs(dst: *mut RawBuf, a: *const RawBuf);

    // ── Embedding / KV cache ──
    fn mlx_inline_take_axis(dst: *mut RawBuf, a: *const RawBuf, indices: *const RawBuf, axis: i32);
    fn mlx_inline_kv_cache_append(
        dst: *mut RawBuf,
        cached: *const RawBuf,
        new_kv: *const RawBuf,
        axis: i32,
    );
    fn mlx_inline_async_eval_arr(a: *const RawBuf);

    // ── GDN Metal kernel step with pre-computed g/beta ──
    fn mlx_inline_gdn_metal_step(
        dst_y: *mut RawBuf,
        dst_state: *mut RawBuf,
        q: *const RawBuf,
        k: *const RawBuf,
        v: *const RawBuf,
        g: *const RawBuf,
        beta: *const RawBuf,
        state_in: *const RawBuf,
        t: i32,
    );

    // ── TurboQuant fused Metal kernels ──
    //
    // Encode: nearest-centroid search; eliminates the [N,D,C] intermediate.
    // input: [N,D] f32 (normalised+rotated).  codebook: [C] f32 (C <= 16).
    // out_indices: [N,D] uint32.  out_norms: reserved (pass null ptr).
    // Returns 0 on success, 1 if Metal unavailable.
    fn mlx_inline_turboquant_encode(
        out_indices: *mut RawBuf,
        out_norms: *mut RawBuf, // reserved — pass std::ptr::null_mut()
        input: *const RawBuf,
        codebook: *const RawBuf,
        dim: u32,
        n_centroids: u32,
        n_rows: u32,
    ) -> i32;

    // Decode: codebook lookup → [N,D] f32 centroid values (un-scaled).
    // indices: [N,D] uint32.  norms: reserved (pass null ptr).  codebook: [C] f32.
    // Returns 0 on success, 1 if Metal unavailable.
    fn mlx_inline_turboquant_decode(
        out: *mut RawBuf,
        indices: *const RawBuf,
        norms: *const RawBuf, // reserved — pass std::ptr::null_mut()
        codebook: *const RawBuf,
        dim: u32,
        n_centroids: u32,
        n_rows: u32,
    ) -> i32;

    fn mlx_inline_turboquant_score(
        out_scores: *mut RawBuf,
        query_rot: *const RawBuf,
        query_proj: *const RawBuf,
        indices: *const RawBuf,
        qjl_signs: *const RawBuf,
        norms: *const RawBuf,
        residual_norms: *const RawBuf,
        codebook: *const RawBuf,
        dim: u32,
        qjl_words: u32,
        n_centroids: u32,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale_bits: u32,
    ) -> i32;

    fn mlx_inline_turboquant_score_q8_d256(
        out_scores: *mut RawBuf,
        query_rot: *const RawBuf,
        query_proj: *const RawBuf,
        indices: *const RawBuf,
        qjl_signs: *const RawBuf,
        norms: *const RawBuf,
        residual_norms: *const RawBuf,
        codebook: *const RawBuf,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale_bits: u32,
    ) -> i32;

    fn mlx_inline_turboquant_mixed_score(
        out_scores: *mut RawBuf,
        regular_query_rot: *const RawBuf,
        regular_query_proj: *const RawBuf,
        regular_indices: *const RawBuf,
        regular_qjl_signs: *const RawBuf,
        regular_norms: *const RawBuf,
        regular_residual_norms: *const RawBuf,
        regular_codebook: *const RawBuf,
        outlier_query_rot: *const RawBuf,
        outlier_query_proj: *const RawBuf,
        outlier_indices: *const RawBuf,
        outlier_qjl_signs: *const RawBuf,
        outlier_norms: *const RawBuf,
        outlier_residual_norms: *const RawBuf,
        outlier_codebook: *const RawBuf,
        regular_dim: u32,
        regular_qjl_words: u32,
        regular_n_centroids: u32,
        outlier_dim: u32,
        outlier_qjl_words: u32,
        outlier_n_centroids: u32,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale_bits: u32,
    ) -> i32;

    fn mlx_inline_turboquant_pack_sign_bits(
        out: *mut RawBuf,
        projected: *const RawBuf,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
    ) -> i32;

    fn mlx_inline_turboquant_pack_q8_keybytes(
        out: *mut RawBuf,
        indices: *const RawBuf,
        qjl_signs: *const RawBuf,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
        cache_seq_capacity: u32,
    ) -> i32;

    fn mlx_inline_turboquant_pack_q8_keybytes_seq(
        out: *mut RawBuf,
        indices: *const RawBuf,
        qjl_signs: *const RawBuf,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
        cache_seq_capacity: u32,
    ) -> i32;

    fn mlx_inline_turboquant_pack_q8_kvbytes_seq(
        out: *mut RawBuf,
        indices: *const RawBuf,
        qjl_signs: *const RawBuf,
        value_indices: *const RawBuf,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
        cache_seq_capacity: u32,
    ) -> i32;

    fn mlx_inline_turboquant_unpack_sign_bits(
        out: *mut RawBuf,
        packed: *const RawBuf,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
    ) -> i32;

    fn mlx_inline_turboquant_weighted_decode(
        out: *mut RawBuf,
        weights: *const RawBuf,
        indices: *const RawBuf,
        norms: *const RawBuf,
        codebook: *const RawBuf,
        dim: u32,
        n_centroids: u32,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
    ) -> i32;

    fn mlx_inline_turboquant_attention_q8_d256_2pass(
        out: *mut RawBuf,
        query_rot: *const RawBuf,
        query_proj: *const RawBuf,
        key_indices: *const RawBuf,
        key_qjl_signs: *const RawBuf,
        key_norms: *const RawBuf,
        key_residual_norms: *const RawBuf,
        key_codebook: *const RawBuf,
        value_indices: *const RawBuf,
        value_norms: *const RawBuf,
        value_codebook: *const RawBuf,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale_bits: u32,
    ) -> i32;

    fn mlx_inline_turboquant_attention_q8_d256_packed_keys_2pass(
        out: *mut RawBuf,
        query_rot: *const RawBuf,
        query_proj: *const RawBuf,
        key_bytes: *const RawBuf,
        slot_scales: *const RawBuf,
        key_codebook: *const RawBuf,
        value_indices: *const RawBuf,
        value_codebook: *const RawBuf,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale_bits: u32,
    ) -> i32;

    fn mlx_inline_turboquant_attention_q8_d256_packed_kv_2pass(
        out: *mut RawBuf,
        query_rot: *const RawBuf,
        query_proj: *const RawBuf,
        kv_bytes: *const RawBuf,
        slot_scales: *const RawBuf,
        key_codebook: *const RawBuf,
        value_codebook: *const RawBuf,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale_bits: u32,
    ) -> i32;

    fn mlx_inline_turboquant_attention_q8_d256_packed_kv_dense_values_2pass(
        out: *mut RawBuf,
        query_rot: *const RawBuf,
        query_proj: *const RawBuf,
        kv_bytes: *const RawBuf,
        slot_scales: *const RawBuf,
        key_codebook: *const RawBuf,
        value_dense: *const RawBuf,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale_bits: u32,
    ) -> i32;

    fn mlx_inline_turboquant_attention_q8_d128_2pass(
        out: *mut RawBuf,
        query_rot: *const RawBuf,
        query_proj: *const RawBuf,
        key_indices: *const RawBuf,
        key_qjl_signs: *const RawBuf,
        key_norms: *const RawBuf,
        key_residual_norms: *const RawBuf,
        key_codebook: *const RawBuf,
        value_indices: *const RawBuf,
        value_norms: *const RawBuf,
        value_codebook: *const RawBuf,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale_bits: u32,
    ) -> i32;

    fn mlx_inline_turboquant_attention_q8_d128_packed_keys_2pass(
        out: *mut RawBuf,
        query_rot: *const RawBuf,
        query_proj: *const RawBuf,
        key_bytes: *const RawBuf,
        key_norms: *const RawBuf,
        key_residual_norms: *const RawBuf,
        key_codebook: *const RawBuf,
        value_indices: *const RawBuf,
        value_norms: *const RawBuf,
        value_codebook: *const RawBuf,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale_bits: u32,
    ) -> i32;

    fn mlx_inline_turboquant_gather_last_dim(
        out: *mut RawBuf,
        input: *const RawBuf,
        positions: *const RawBuf,
        full_dim: u32,
        out_dim: u32,
        n_rows: u32,
    ) -> i32;

    fn mlx_inline_turboquant_scatter_last_dim(
        out: *mut RawBuf,
        regular: *const RawBuf,
        outlier: *const RawBuf,
        regular_positions: *const RawBuf,
        outlier_positions: *const RawBuf,
        full_dim: u32,
        regular_dim: u32,
        outlier_dim: u32,
        n_rows: u32,
    ) -> i32;

    // ── Training ops: random ──
    fn mlx_inline_random_normal(dst: *mut RawBuf, shape: *const i32, ndim: i32, dtype: i32);
    fn mlx_inline_random_uniform(dst: *mut RawBuf, shape: *const i32, ndim: i32, dtype: i32);
    fn mlx_inline_random_bernoulli(
        dst: *mut RawBuf,
        p: *const RawBuf,
        shape: *const i32,
        ndim: i32,
    );
    fn mlx_inline_random_seed(seed: u64);
    fn mlx_inline_random_randint(
        dst: *mut RawBuf,
        low: i32,
        high: i32,
        shape: *const i32,
        ndim: i32,
        dtype: i32,
    );

    // ── Training ops: math ──
    fn mlx_inline_mean_axis(dst: *mut RawBuf, a: *const RawBuf, axis: i32, keepdims: bool);
    fn mlx_inline_mean_all(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_var_axis(dst: *mut RawBuf, a: *const RawBuf, axis: i32, keepdims: bool);
    fn mlx_inline_pow(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    fn mlx_inline_reciprocal(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_sin(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_cos(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_clip(dst: *mut RawBuf, a: *const RawBuf, lo: *const RawBuf, hi: *const RawBuf);
    fn mlx_inline_log_softmax(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    fn mlx_inline_cross_entropy(
        dst: *mut RawBuf,
        logits: *const RawBuf,
        targets: *const RawBuf,
        axis: i32,
    );
    fn mlx_inline_square(dst: *mut RawBuf, a: *const RawBuf);

    // ── Training ops: creation ──
    fn mlx_inline_full(dst: *mut RawBuf, shape: *const i32, ndim: i32, val: f32, dtype: i32);
    fn mlx_inline_eye(dst: *mut RawBuf, n: i32, dtype: i32);
    fn mlx_inline_tri(dst: *mut RawBuf, n: i32, m: i32, k: i32, dtype: i32);

    // ── Training ops: shape ──
    fn mlx_inline_broadcast_to(dst: *mut RawBuf, a: *const RawBuf, shape: *const i32, ndim: i32);
    fn mlx_inline_flatten(dst: *mut RawBuf, a: *const RawBuf, start_axis: i32, end_axis: i32);

    // ── Training ops: sort/reduction ──
    fn mlx_inline_argsort(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    fn mlx_inline_sum_all(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_max_axis(dst: *mut RawBuf, a: *const RawBuf, axis: i32, keepdims: bool);
    fn mlx_inline_min_axis(dst: *mut RawBuf, a: *const RawBuf, axis: i32, keepdims: bool);
    fn mlx_inline_minimum(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);

    // ── Training ops: activation ──
    fn mlx_inline_relu(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_gelu(dst: *mut RawBuf, a: *const RawBuf);

    // ── Training ops: comparison ──
    fn mlx_inline_equal(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    fn mlx_inline_not_equal(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    fn mlx_inline_greater(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    fn mlx_inline_less(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    fn mlx_inline_greater_equal(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    fn mlx_inline_less_equal(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);

    // ── Training ops: serialization ──
    fn mlx_inline_save_safetensors(
        path: *const std::ffi::c_char,
        keys: *const *const std::ffi::c_char,
        arrays: *const RawBuf,
        count: i32,
    );

    // ── Training ops: quantize ──
    fn mlx_inline_quantize(
        dst_w: *mut RawBuf,
        dst_scales: *mut RawBuf,
        dst_biases: *mut RawBuf,
        a: *const RawBuf,
        group_size: i32,
        bits: i32,
    );

    // ── Training ops: multi-axis ──
    fn mlx_inline_sum_axes(
        dst: *mut RawBuf,
        a: *const RawBuf,
        axes: *const i32,
        num_axes: i32,
        keepdims: bool,
    );
    fn mlx_inline_mean_axes(
        dst: *mut RawBuf,
        a: *const RawBuf,
        axes: *const i32,
        num_axes: i32,
        keepdims: bool,
    );

    // ── Training ops: misc ──
    fn mlx_inline_size(a: *const RawBuf) -> usize;
    fn mlx_inline_nbytes(a: *const RawBuf) -> usize;
    fn mlx_inline_data_ptr(a: *const RawBuf, out_ptr: *mut *const std::ffi::c_void) -> i32;
    fn mlx_inline_stop_gradient(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_tri_inv(dst: *mut RawBuf, a: *const RawBuf, upper: bool, use_cpu: bool);

    // ── Autograd: value_and_grad ──
    fn mlx_inline_value_and_grad(
        forward_fn: unsafe extern "C" fn(
            *const *const RawBuf,
            i32,
            *mut RawBuf,
            *mut std::ffi::c_void,
        ),
        ctx: *mut std::ffi::c_void,
        all_arrays: *const *const RawBuf,
        n_params: i32,
        n_total: i32,
        loss_out: *mut RawBuf,
        grads_out: *mut *mut RawBuf,
    );

    // ── Fused compiled ops (match Python's @mx.compile) ──
    fn mlx_inline_fused_swiglu(dst: *mut RawBuf, gate: *const RawBuf, up: *const RawBuf);
    fn mlx_inline_fused_silu(dst: *mut RawBuf, x: *const RawBuf);
    fn mlx_inline_fused_compute_g(
        dst: *mut RawBuf,
        a_log: *const RawBuf,
        a: *const RawBuf,
        dt_bias: *const RawBuf,
    );
    fn mlx_inline_fused_precise_swiglu(dst: *mut RawBuf, x: *const RawBuf, gate: *const RawBuf);

    // Batch safetensors load — parses the file once and fills caller-provided buffers.
    // Returns number of entries written, or -1 on error.
    fn mlx_inline_load_safetensors_all(
        path: *const std::ffi::c_char,
        key_buf: *mut *mut std::ffi::c_char,
        arr_buf: *mut RawBuf,
        max_entries: i32,
    ) -> i32;

    // Free key strings allocated by mlx_inline_load_safetensors_all.
    fn mlx_inline_free_key_strings(keys: *mut *mut std::ffi::c_char, count: i32);

    // Create a 1-D int32 array from a Rust slice.
    fn mlx_inline_from_i32_slice(dst: *mut RawBuf, data: *const i32, len: i32);

    // ── Linalg: SVD ──
    fn mlx_inline_svd(
        dst_u: *mut RawBuf,
        dst_s: *mut RawBuf,
        dst_vt: *mut RawBuf,
        a: *const RawBuf,
    );

    // ── Missing ops for pmetal-models migration ──
    fn mlx_inline_rsqrt(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_zeros_like(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_ones_like(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_tile(dst: *mut RawBuf, a: *const RawBuf, reps: *const i32, ndim: i32);
    fn mlx_inline_linspace(dst: *mut RawBuf, start: f32, stop: f32, n: i32, dtype: i32);
    fn mlx_inline_split_sections(
        dst_arr: *mut RawBuf,
        a: *const RawBuf,
        sections: i32,
        axis: i32,
        out_count: *mut i32,
    );
    fn mlx_inline_scatter_add(
        dst: *mut RawBuf,
        a: *const RawBuf,
        indices: *const RawBuf,
        updates: *const RawBuf,
        axis: i32,
    );
    fn mlx_inline_topk(dst: *mut RawBuf, a: *const RawBuf, k: i32, axis: i32);
    fn mlx_inline_put_along_axis(
        dst: *mut RawBuf,
        a: *const RawBuf,
        indices: *const RawBuf,
        values: *const RawBuf,
        axis: i32,
    );
    fn mlx_inline_layer_norm(
        dst: *mut RawBuf,
        x: *const RawBuf,
        weight: *const RawBuf,
        bias: *const RawBuf,
        eps: f32,
    );
    fn mlx_inline_addmm(dst: *mut RawBuf, c: *const RawBuf, a: *const RawBuf, b: *const RawBuf);
    fn mlx_inline_conv2d(
        dst: *mut RawBuf,
        input: *const RawBuf,
        weight: *const RawBuf,
        stride_h: i32,
        stride_w: i32,
        pad_h: i32,
        pad_w: i32,
        dil_h: i32,
        dil_w: i32,
        groups: i32,
    );

    // ── Full Qwen3.5 forward pass — single C++ function, zero FFI overhead ──
    // See bridge.h for the complete weight/cache/config layout documentation.
    fn mlx_inline_qwen35_decode_step(
        dst_logits: *mut RawBuf,
        token_ids: *const RawBuf,
        weight_ptrs: *const *const RawBuf,
        num_weights: i32,
        cache_ptrs: *mut *mut RawBuf,
        num_cache: i32,
        attn_kv_offsets: *mut i32,
        rope_offset: *mut i32,
        config_ints: *const i32,
        num_config_ints: i32,
        config_floats: *const f32,
        num_config_floats: i32,
    );
}

// ── Graph / compile helpers ───────────────────────────────────────────────

/// Count the number of unique nodes in the computation graph rooted at this
/// array.  Useful for diagnosing performance — each node becomes a Metal
/// kernel dispatch during eval.
pub fn graph_node_count(arr: &InlineArray) -> usize {
    unsafe { mlx_inline_graph_node_count(&arr.raw) }
}

/// Count unique ArrayDesc nodes (the REAL graph nodes that map to dispatches).
pub fn graph_desc_count(arr: &InlineArray) -> usize {
    unsafe { mlx_inline_graph_desc_count(&arr.raw) }
}

/// Dump the graph topology to stderr: print every node's primitive type and shape.
pub fn graph_dump(arr: &InlineArray) {
    unsafe { mlx_inline_graph_dump(&arr.raw) }
}

/// Start a Metal GPU capture to the given .gputrace path.
/// Must run with MTL_CAPTURE_ENABLED=1 environment variable.
pub fn metal_start_capture(path: &str) -> bool {
    let c_path = std::ffi::CString::new(path).unwrap();
    unsafe { mlx_inline_metal_start_capture(c_path.as_ptr()) == 0 }
}

/// Stop the Metal GPU capture.
pub fn metal_stop_capture() {
    unsafe { mlx_inline_metal_stop_capture() }
}

/// Set wired memory limit to maximum recommended — CRITICAL for GPU performance.
/// Without this, Metal buffers may be paged out causing massive overhead.
/// Returns previous limit.
pub fn set_wired_limit_max() -> usize {
    let max_size = unsafe { mlx_inline_get_max_recommended_size() };
    if max_size > 0 {
        unsafe { mlx_inline_set_wired_limit(max_size) }
    } else {
        0
    }
}

/// Set the wired memory limit explicitly. Returns the previous limit.
pub fn set_wired_limit(limit: usize) -> usize {
    unsafe { mlx_inline_set_wired_limit(limit) }
}

/// Get the device's maximum recommended working set size (GPU memory limit).
pub fn get_max_recommended_size() -> usize {
    unsafe { mlx_inline_get_max_recommended_size() }
}

/// Create a new GPU stream and set it as default for all subsequent ops.
/// Matches Python's `generation_stream = mx.new_stream(mx.default_device())`.
pub fn new_generation_stream() {
    unsafe {
        mlx_inline_new_stream();
    }
}

/// Set the generation stream as the default stream for all ops.
pub fn set_generation_stream() {
    unsafe {
        mlx_inline_set_default_stream(0);
    }
}

/// Synchronize the generation stream (wait for all pending GPU work).
pub fn synchronize() {
    unsafe {
        mlx_inline_synchronize();
    }
}

/// Clear the Metal buffer cache — frees unused GPU memory.
/// Call periodically during generation to prevent memory accumulation.
pub fn clear_cache() {
    unsafe { mlx_inline_clear_cache() }
}

/// Set the Metal cache size limit (in bytes). Returns the previous limit.
pub fn set_cache_limit(limit: usize) -> usize {
    unsafe { mlx_inline_set_cache_limit(limit) }
}

/// Enable MLX global compilation — fuses ops across the entire computation
/// graph.
pub fn enable_compile() {
    unsafe { mlx_inline_enable_compile() }
}

/// Disable MLX global compilation.
pub fn disable_compile() {
    unsafe { mlx_inline_disable_compile() }
}

/// Eval a batch of arrays in a SINGLE GPU submission, then detach each one.
/// This is critical for cache arrays: eval+detach severs the computation
/// graph chain across decode steps without per-array sync barriers.
pub fn eval_and_detach_many(arrays: &mut [&mut InlineArray]) {
    if arrays.is_empty() {
        return;
    }
    let mut ptrs: Vec<*mut RawBuf> = arrays
        .iter_mut()
        .map(|a| &mut a.raw as *mut RawBuf)
        .collect();
    unsafe {
        mlx_inline_eval_many(ptrs.as_mut_ptr(), ptrs.len() as i32);
    }
    for a in arrays.iter_mut() {
        unsafe {
            mlx_inline_detach(&mut a.raw);
        }
    }
}

/// Metal memory: bytes currently in use by live arrays.
pub fn get_active_memory() -> usize {
    unsafe { mlx_inline_get_active_memory() }
}

/// Metal memory: bytes freed but held in buffer cache for reuse.
pub fn get_cache_memory() -> usize {
    unsafe { mlx_inline_get_cache_memory() }
}

/// Metal memory: high-water mark of active memory.
pub fn get_peak_memory() -> usize {
    unsafe { mlx_inline_get_peak_memory() }
}

/// Reset the peak memory tracker.
pub fn reset_peak_memory() {
    unsafe { mlx_inline_reset_peak_memory() }
}

// ── Full Qwen3.5 forward pass ─────────────────────────────────────────────

/// Run the entire Qwen3.5 forward pass (all N layers) as a single C++ call,
/// eliminating per-op FFI overhead (~1800 round trips per decode step).
///
/// # Safety
/// All raw pointers in `weight_ptrs` and `cache_ptrs` must point to live,
/// placement-new'd `mlx::core::array` objects (i.e. valid `InlineArray.raw`
/// fields).  The arrays must remain live for the duration of this call.
///
/// `attn_kv_offsets` and `rope_offset` are updated in-place by C++.
pub(crate) unsafe fn qwen35_decode_step(
    token_ids: &InlineArray,
    weight_ptrs: &[*const RawBuf],
    cache_ptrs: &mut [*mut RawBuf],
    attn_kv_offsets: &mut [i32],
    rope_offset: &mut i32,
    config_ints: &[i32],
    config_floats: &[f32],
) -> InlineArray {
    let mut dst = InlineArray::uninit();
    unsafe {
        mlx_inline_qwen35_decode_step(
            dst.as_raw_ptr_mut(),
            token_ids.as_raw_ptr(),
            weight_ptrs.as_ptr(),
            weight_ptrs.len() as i32,
            cache_ptrs.as_mut_ptr(),
            cache_ptrs.len() as i32,
            attn_kv_offsets.as_mut_ptr(),
            rope_offset,
            config_ints.as_ptr(),
            config_ints.len() as i32,
            config_floats.as_ptr(),
            config_floats.len() as i32,
        );
    }
    dst
}

// ── Batch safetensors loader ──────────────────────────────────────────────

/// Load all arrays from a safetensors shard in a single parse.
///
/// This is substantially faster than calling `InlineArray::load_safetensors`
/// per key because the file is parsed exactly once.  A typical model shard
/// has ~300 tensors; `MAX_ENTRIES` (2048) comfortably covers any realistic
/// shard.
///
/// Returns `None` on I/O or parse error.  Individual key allocation failures
/// (malformed UTF-8 key) are silently skipped.
pub fn load_safetensors_shard(path: &str) -> Option<Vec<(String, InlineArray)>> {
    const MAX_ENTRIES: usize = 2048;

    let c_path = std::ffi::CString::new(path).ok()?;

    // Allocate key-pointer buffer.  C++ will strdup into each slot.
    let mut key_ptrs: Vec<*mut std::ffi::c_char> = vec![std::ptr::null_mut(); MAX_ENTRIES];

    // Allocate uninitialised array slots.  C++ does placement new into each
    // occupied slot; only the first `count` slots are initialised.
    let mut arr_slots: Vec<MaybeUninit<RawBuf>> = (0..MAX_ENTRIES)
        .map(|_| MaybeUninit::<RawBuf>::uninit())
        .collect();

    let count = unsafe {
        mlx_inline_load_safetensors_all(
            c_path.as_ptr(),
            key_ptrs.as_mut_ptr(),
            // Cast *mut MaybeUninit<RawBuf> → *mut RawBuf.  This is safe
            // because MaybeUninit<T> has the same layout as T.
            arr_slots.as_mut_ptr() as *mut RawBuf,
            MAX_ENTRIES as i32,
        )
    };

    if count < 0 {
        // Fallback: recover tensor names from the safetensors header, then load
        // each tensor through the single-key bridge path. This preserves a
        // correct native load path even when the batched C++ loader fails.
        return load_safetensors_shard_fallback(path);
    }

    let count = count as usize;

    // Convert the count valid slots into owned InlineArrays + String keys.
    // We must adopt each initialised array slot so its destructor runs on drop.
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        // SAFETY: C++ placement-new'd into slots [0, count).
        let array = InlineArray {
            raw: unsafe { arr_slots[i].assume_init() },
        };

        // key_ptrs[i] is a strdup'd C string.  Convert to Rust String and
        // free the C allocation immediately — the String owns the data.
        let key = unsafe {
            let s = std::ffi::CStr::from_ptr(key_ptrs[i])
                .to_string_lossy()
                .into_owned();
            // Free the strdup allocation.
            libc_free(key_ptrs[i] as *mut std::ffi::c_void);
            s
        };

        result.push((key, array));
    }

    Some(result)
}

fn load_safetensors_shard_fallback(path: &str) -> Option<Vec<(String, InlineArray)>> {
    let mapped = map_safetensors_file(path)?;
    let tensors = SafeTensors::deserialize(&mapped).ok()?;
    let names = tensors.names();
    let mut result = Vec::with_capacity(names.len());
    for key in names {
        let tensor = tensors.tensor(key).ok()?;
        let array = inline_array_from_tensor_view(&tensor)?;
        result.push((key.to_string(), array));
    }
    Some(result)
}

fn map_safetensors_file(path: &str) -> Option<Mmap> {
    let file = std::fs::File::open(path).ok()?;
    unsafe { Mmap::map(&file).ok() }
}

fn as_typed_slice<T>(data: &[u8]) -> Option<&[T]> {
    // SAFETY: We only accept fully aligned, remainder-free views.
    let (prefix, values, suffix) = unsafe { data.align_to::<T>() };
    if prefix.is_empty() && suffix.is_empty() {
        Some(values)
    } else {
        None
    }
}

fn shape_to_i32(shape: &[usize]) -> Option<Vec<i32>> {
    shape.iter().map(|&dim| i32::try_from(dim).ok()).collect()
}

fn inline_array_from_tensor_view(
    tensor: &safetensors::tensor::TensorView<'_>,
) -> Option<InlineArray> {
    let shape = shape_to_i32(tensor.shape())?;
    match tensor.dtype() {
        SafeDtype::F32 => Some(InlineArray::from_f32_slice(
            as_typed_slice::<f32>(tensor.data())?,
            &shape,
        )),
        SafeDtype::I32 => Some(InlineArray::from_i32_slice_shaped(
            as_typed_slice::<i32>(tensor.data())?,
            &shape,
        )),
        SafeDtype::U32 => Some(InlineArray::from_u32_slice(
            as_typed_slice::<u32>(tensor.data())?,
            &shape,
        )),
        SafeDtype::U8 => Some(InlineArray::from_u8_slice(
            as_typed_slice::<u8>(tensor.data())?,
            &shape,
        )),
        SafeDtype::F16 => Some(InlineArray::from_u16_bits_slice(
            as_typed_slice::<u16>(tensor.data())?,
            &shape,
            1,
        )),
        SafeDtype::BF16 => Some(InlineArray::from_u16_bits_slice(
            as_typed_slice::<u16>(tensor.data())?,
            &shape,
            11,
        )),
        SafeDtype::I64 => {
            let values = as_typed_slice::<i64>(tensor.data())?;
            let cast: Vec<i32> = values.iter().map(|&value| value as i32).collect();
            Some(InlineArray::from_i32_slice_shaped(&cast, &shape))
        }
        _ => None,
    }
}

/// Thin wrapper around libc free so we can call it without a libc dependency.
/// `strdup` allocates with the C allocator; we must free with the same.
unsafe fn libc_free(ptr: *mut std::ffi::c_void) {
    unsafe extern "C" {
        fn free(ptr: *mut std::ffi::c_void);
    }
    unsafe { free(ptr) }
}

// ── Random seed (global, not per-array) ───────────────────────────────────

/// Set the global MLX random seed for reproducibility.
pub fn random_seed(seed: u64) {
    unsafe { mlx_inline_random_seed(seed) }
}

// ── InlineArray ───────────────────────────────────────────────────────────

/// Stack-allocated MLX array. Zero heap allocation per op.
pub struct InlineArray {
    raw: RawBuf,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EvalToken;

impl EvalToken {
    #[inline]
    pub fn unwrap(self) {}

    #[inline]
    pub fn expect(self, _msg: &str) {}
}

impl Drop for InlineArray {
    #[inline]
    fn drop(&mut self) {
        unsafe { mlx_inline_destroy(&mut self.raw) };
    }
}

impl Clone for InlineArray {
    fn clone(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_init_copy(dst.as_mut_ptr(), &self.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }
}

unsafe impl Send for InlineArray {}
unsafe impl Sync for InlineArray {}

impl std::fmt::Debug for InlineArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "InlineArray(ndim={}, shape={:?})",
            self.ndim(),
            self.shape()
        )
    }
}

macro_rules! binop {
    ($name:ident, $cfn:ident) => {
        #[inline]
        pub fn $name(&self, other: &Self) -> Self {
            let mut dst = MaybeUninit::<RawBuf>::uninit();
            unsafe {
                $cfn(dst.as_mut_ptr(), &self.raw, &other.raw);
                Self {
                    raw: dst.assume_init(),
                }
            }
        }
    };
}

macro_rules! unop {
    ($name:ident, $cfn:ident) => {
        #[inline]
        pub fn $name(&self) -> Self {
            let mut dst = MaybeUninit::<RawBuf>::uninit();
            unsafe {
                $cfn(dst.as_mut_ptr(), &self.raw);
                Self {
                    raw: dst.assume_init(),
                }
            }
        }
    };
}

impl InlineArray {
    // ── Interop with mlx-rs (transition period) ──────────────────────────

    /// Create from an opaque mlx-rs array context pointer.
    ///
    /// # Safety
    /// `ctx` must be a valid `mlx::core::array*` as returned by
    /// `mlx_array { ctx }` from the mlx-c / mlx-rs layer.  The C++ side
    /// copies (ref-counts) the array, so the caller retains ownership of the
    /// original handle.
    ///
    /// Typical usage during migration:
    /// ```ignore
    /// let inline = InlineArray::from_raw_ctx(arr.as_ptr().ctx);
    /// ```
    pub unsafe fn from_raw_ctx(ctx: *mut std::ffi::c_void) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_handle(dst.as_mut_ptr(), ctx);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Export as an opaque heap-allocated `mlx::core::array*` context pointer.
    ///
    /// The caller is responsible for freeing the returned pointer via the
    /// mlx-c `mlx_array_free` mechanism (or by wrapping in an `mlx_array`
    /// handle and passing to `mlx_array_free`).
    ///
    /// Typical usage during migration:
    /// ```ignore
    /// let ctx = inline.to_raw_ctx();
    /// let handle = mlx_sys::mlx_array { ctx };
    /// let arr = unsafe { mlx_rs::Array::from_ptr(handle) };
    /// ```
    pub fn to_raw_ctx(&self) -> *mut std::ffi::c_void {
        unsafe { mlx_inline_to_handle(&self.raw) }
    }

    // ── Raw pointer access (crate-internal, for C++ bridge) ──────────────

    /// Return a const raw pointer to the inline buffer (for C++ bridge calls).
    #[inline]
    pub(crate) fn as_raw_ptr(&self) -> *const RawBuf {
        &self.raw
    }

    /// Return a mutable raw pointer to the inline buffer (for C++ bridge calls).
    #[inline]
    pub(crate) fn as_raw_ptr_mut(&mut self) -> *mut RawBuf {
        &mut self.raw
    }

    // ── Factory ──────────────────────────────────────────────────────────

    /// Create an uninitialised slot — caller MUST ensure C++ does placement-new
    /// into `self.raw` before this is read or dropped.
    ///
    /// Used as the destination buffer for C++ functions that return arrays via
    /// placement-new (e.g. `mlx_inline_qwen35_decode_step`).
    pub(crate) fn uninit() -> Self {
        // We initialise to a scalar 0.0 so the Drop impl always runs a valid
        // destructor even if the C++ side never fills the slot.
        Self::from_f32(0.0)
    }

    /// Identity constructor — clone an existing array.
    ///
    /// Compatible with mlx-rs `Array::from_array(arr)` which was a no-op copy.
    /// Since `Array = InlineArray` in this bridge, this is just `.clone()`.
    #[inline]
    pub fn from_array(other: &Self) -> Self {
        other.clone()
    }

    /// Scalar integer array constructor.
    ///
    /// Compatible with mlx-rs `Array::from_int(val)`.
    #[inline]
    pub fn from_int(val: i32) -> Self {
        Self::from_i32(val)
    }

    /// Construct an array from an iterator of integers with an explicit shape.
    ///
    /// Compatible with mlx-rs `Array::from_iter(iter, shape)`.
    /// The iterator is collected into a `Vec<i32>` and shaped.
    ///
    /// # Example
    /// ```ignore
    /// let a = Array::from_iter(0..n, &[n]);
    /// ```
    pub fn from_iter(iter: impl IntoIterator<Item = i32>, shape: &[i32]) -> Self {
        let v: Vec<i32> = iter.into_iter().collect();
        Self::from_i32_slice_shaped(&v, shape)
    }

    pub fn from_f32(val: f32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_f32(dst.as_mut_ptr(), val);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn from_i32(val: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_i32(dst.as_mut_ptr(), val);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn zeros(shape: &[i32], dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_zeros(dst.as_mut_ptr(), shape.as_ptr(), shape.len() as i32, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn ones(shape: &[i32], dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_ones(dst.as_mut_ptr(), shape.as_ptr(), shape.len() as i32, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Binary ops ───────────────────────────────────────────────────────
    binop!(matmul, mlx_inline_matmul);
    binop!(add, mlx_inline_add);
    binop!(multiply, mlx_inline_multiply);
    binop!(subtract, mlx_inline_subtract);
    binop!(divide, mlx_inline_divide);
    binop!(maximum, mlx_inline_maximum);

    // ── Unary ops ────────────────────────────────────────────────────────
    unop!(negative, mlx_inline_negative);
    unop!(exp, mlx_inline_exp);
    unop!(sigmoid, mlx_inline_sigmoid);
    unop!(silu, mlx_inline_silu);
    unop!(sqrt, mlx_inline_sqrt);
    unop!(t, mlx_inline_transpose);
    unop!(softplus, mlx_inline_softplus);
    unop!(log, mlx_inline_log);
    unop!(sign, mlx_inline_sign);

    /// Dequantize packed integer weights using per-group scales and biases.
    pub fn dequantize(&self, scales: &Self, biases: &Self, group_size: i32, bits: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_dequantize(
                dst.as_mut_ptr(),
                &self.raw,
                &scales.raw,
                &biases.raw,
                group_size,
                bits,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// L2 norm along an axis.
    pub fn norm_l2(&self, axis: i32, keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_norm_l2(dst.as_mut_ptr(), &self.raw, axis, keepdims);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn softmax(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_softmax(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn softmax_precise(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_softmax_precise(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn reshape(&self, shape: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_reshape(
                dst.as_mut_ptr(),
                &self.raw,
                shape.as_ptr(),
                shape.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn sum_axis(&self, axis: i32, keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_sum_axis(dst.as_mut_ptr(), &self.raw, axis, keepdims);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn as_dtype(&self, dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_astype(dst.as_mut_ptr(), &self.raw, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Cast to a Rust primitive type `T` — compatible with mlx-rs `as_type::<T>()`.
    ///
    /// Uses the [`AsDtype`] sealed trait to map Rust types to MLX dtypes.
    #[inline]
    pub fn as_type<T: AsDtype>(&self) -> Self {
        self.as_dtype(T::DTYPE_ID)
    }

    // ── Gather / MoE ─────────────────────────────────────────────────────

    pub fn gather_mm(
        &self,
        b: &Self,
        lhs: Option<&Self>,
        rhs: Option<&Self>,
        sorted: bool,
    ) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_gather_mm(
                dst.as_mut_ptr(),
                &self.raw,
                &b.raw,
                lhs.map_or(std::ptr::null(), |a| &a.raw),
                rhs.map_or(std::ptr::null(), |a| &a.raw),
                sorted,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn argpartition(&self, kth: i32, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_argpartition(dst.as_mut_ptr(), &self.raw, kth, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn take_along_axis(&self, indices: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_take_along_axis(dst.as_mut_ptr(), &self.raw, &indices.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Fast ops ─────────────────────────────────────────────────────────

    pub fn rms_norm(&self, weight: Option<&Self>, eps: f32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_rms_norm(
                dst.as_mut_ptr(),
                &self.raw,
                weight.map_or(std::ptr::null(), |w| &w.raw),
                eps,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn rope(&self, dims: i32, traditional: bool, base: f32, scale: f32, offset: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_rope(
                dst.as_mut_ptr(),
                &self.raw,
                dims,
                traditional,
                base,
                scale,
                offset,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn sdpa(&self, k: &Self, v: &Self, scale: f32, mask_mode: &str) -> Self {
        let c = std::ffi::CString::new(mask_mode).unwrap();
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_sdpa(
                dst.as_mut_ptr(),
                &self.raw,
                &k.raw,
                &v.raw,
                scale,
                c.as_ptr(),
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// SDPA with optional mask array. Pass `None` for no mask.
    #[inline]
    pub fn sdpa_with_mask(&self, k: &Self, v: &Self, scale: f32, mask: Option<&Self>) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        let mask_ptr = mask
            .map(|m| &m.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        unsafe {
            mlx_inline_sdpa_with_mask(dst.as_mut_ptr(), &self.raw, &k.raw, &v.raw, scale, mask_ptr);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn split(&self, indices: &[i32], axis: i32) -> Vec<Self> {
        let n = indices.len() + 1;
        let mut bufs: Vec<MaybeUninit<RawBuf>> = (0..n).map(|_| MaybeUninit::uninit()).collect();
        unsafe {
            mlx_inline_split(
                &self.raw,
                indices.as_ptr(),
                indices.len() as i32,
                axis,
                bufs.as_mut_ptr() as *mut RawBuf,
            );
            bufs.into_iter()
                .map(|b| Self {
                    raw: b.assume_init(),
                })
                .collect()
        }
    }

    pub fn conv1d(
        &self,
        weight: &Self,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    ) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_conv1d(
                dst.as_mut_ptr(),
                &self.raw,
                &weight.raw,
                stride,
                padding,
                dilation,
                groups,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Shape / dtype query ───────────────────────────────────────────────

    pub fn ndim(&self) -> i32 {
        unsafe { mlx_inline_ndim(&self.raw) }
    }
    pub fn dim(&self, axis: i32) -> i32 {
        unsafe { mlx_inline_dim(&self.raw, axis) }
    }
    pub fn shape(&self) -> &[i32] {
        unsafe { std::slice::from_raw_parts(mlx_inline_shape(&self.raw), self.ndim() as usize) }
    }
    pub fn dtype_raw(&self) -> i32 {
        unsafe { mlx_inline_dtype(&self.raw) }
    }

    /// Returns the dtype as a [`crate::compat::Dtype`] enum.
    ///
    /// Equivalent to mlx-rs `Array::dtype()`.
    #[inline]
    pub fn dtype(&self) -> crate::compat::Dtype {
        crate::compat::Dtype::from_raw(self.dtype_raw())
    }

    // ── Eval ─────────────────────────────────────────────────────────────

    pub fn eval(&self) -> EvalToken {
        // MLX array handles are internally mutable; eval materializes the backing
        // graph state but does not change the logical Rust ownership model.
        unsafe { mlx_inline_eval(std::ptr::from_ref(&self.raw).cast_mut()) }
        EvalToken
    }
    pub fn async_eval(&self) -> EvalToken {
        unsafe { mlx_inline_async_eval(std::ptr::from_ref(&self.raw).cast_mut()) }
        EvalToken
    }

    /// Eval two arrays in one call (avoids two FFI round-trips).
    #[inline]
    pub fn eval_2(a: &mut Self, b: &mut Self) {
        unsafe { mlx_inline_eval_2(&mut a.raw, &mut b.raw) }
    }

    /// Compiled GDN layer: entire forward pass as a single compiled function.
    /// Uses 4 separate projection weights matching Python's in_proj_qkv/z/b/a.
    /// Returns (output, new_conv_state, new_ssm_state).
    #[allow(clippy::too_many_arguments)]
    pub fn compiled_gdn_layer(
        normed: &Self,
        qkv_w: &Self,
        z_w: &Self,
        b_w: &Self,
        a_w: &Self,
        conv_w: &Self,
        q_nw: &Self,
        k_nw: &Self,
        a_log: &Self,
        dt_bias: &Self,
        norm_w: &Self,
        out_w: &Self,
        conv_state: &Self,
        ssm_state: &Self,
        nv: i32,
        nk: i32,
        dk: i32,
        dv: i32,
        cd: i32,
        ck: i32,
        kd: i32,
        norm_eps: f32,
    ) -> (Self, Self, Self) {
        let mut out = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut conv = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut ssm = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_compiled_gdn_layer(
                out.as_mut_ptr(),
                conv.as_mut_ptr(),
                ssm.as_mut_ptr(),
                &normed.raw,
                &qkv_w.raw,
                &z_w.raw,
                &b_w.raw,
                &a_w.raw,
                &conv_w.raw,
                &q_nw.raw,
                &k_nw.raw,
                &a_log.raw,
                &dt_bias.raw,
                &norm_w.raw,
                &out_w.raw,
                &conv_state.raw,
                &ssm_state.raw,
                nv,
                nk,
                dk,
                dv,
                cd,
                ck,
                kd,
                norm_eps,
            );
            (
                Self {
                    raw: out.assume_init(),
                },
                Self {
                    raw: conv.assume_init(),
                },
                Self {
                    raw: ssm.assume_init(),
                },
            )
        }
    }

    /// Fixed-shape compiled GDN layer (shapeless=false).
    /// Works with ALL primitives. Traces on first T=1 call, replays tape on subsequent.
    /// Eliminates graph traversal overhead for ~10ms savings per step.
    #[allow(clippy::too_many_arguments)]
    pub fn compiled_gdn_layer_fixed(
        normed: &Self,
        qkv_w: &Self,
        z_w: &Self,
        b_w: &Self,
        a_w: &Self,
        conv_w: &Self,
        q_nw: &Self,
        k_nw: &Self,
        a_log: &Self,
        dt_bias: &Self,
        norm_w: &Self,
        out_w: &Self,
        conv_state: &Self,
        ssm_state: &Self,
        nv: i32,
        nk: i32,
        dk: i32,
        dv: i32,
        cd: i32,
        ck: i32,
        kd: i32,
        norm_eps: f32,
    ) -> (Self, Self, Self) {
        let mut out = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut conv = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut ssm = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_compiled_gdn_layer_fixed(
                out.as_mut_ptr(),
                conv.as_mut_ptr(),
                ssm.as_mut_ptr(),
                &normed.raw,
                &qkv_w.raw,
                &z_w.raw,
                &b_w.raw,
                &a_w.raw,
                &conv_w.raw,
                &q_nw.raw,
                &k_nw.raw,
                &a_log.raw,
                &dt_bias.raw,
                &norm_w.raw,
                &out_w.raw,
                &conv_state.raw,
                &ssm_state.raw,
                nv,
                nk,
                dk,
                dv,
                cd,
                ck,
                kd,
                norm_eps,
            );
            (
                Self {
                    raw: out.assume_init(),
                },
                Self {
                    raw: conv.assume_init(),
                },
                Self {
                    raw: ssm.assume_init(),
                },
            )
        }
    }

    /// Fixed-shape compiled attention decode layer (shapeless=false).
    /// Traces per cache-capacity bucket on first T=1 call, then replays.
    #[allow(clippy::too_many_arguments)]
    pub fn compiled_attn_layer_fixed(
        normed: &Self,
        q_w: &Self,
        k_w: &Self,
        v_w: &Self,
        o_w: &Self,
        q_nw: &Self,
        k_nw: &Self,
        cache_keys_in: &Self,
        cache_vals_in: &Self,
        kv_offset: i32,
        rope_offset: i32,
        n_heads: i32,
        n_kv: i32,
        head_dim: i32,
        scale: f32,
        rope_dims: i32,
        rope_base: f32,
        rope_scale: f32,
        q_norm_eps: f32,
        k_norm_eps: f32,
        gated: bool,
    ) -> (Self, Self, Self) {
        let mut out = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut cache_keys = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut cache_vals = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_compiled_attn_layer_fixed(
                out.as_mut_ptr(),
                cache_keys.as_mut_ptr(),
                cache_vals.as_mut_ptr(),
                &normed.raw,
                &q_w.raw,
                &k_w.raw,
                &v_w.raw,
                &o_w.raw,
                &q_nw.raw,
                &k_nw.raw,
                &cache_keys_in.raw,
                &cache_vals_in.raw,
                kv_offset,
                rope_offset,
                n_heads,
                n_kv,
                head_dim,
                scale,
                rope_dims,
                rope_base,
                rope_scale,
                q_norm_eps,
                k_norm_eps,
                gated,
            );
            (
                Self {
                    raw: out.assume_init(),
                },
                Self {
                    raw: cache_keys.assume_init(),
                },
                Self {
                    raw: cache_vals.assume_init(),
                },
            )
        }
    }

    /// Fixed-shape compiled dense MoE decode block (shapeless=false).
    /// Replays the routed-expert + shared-expert graph for T=1 decode.
    #[allow(clippy::too_many_arguments)]
    pub fn compiled_moe_layer_fixed(
        x: &Self,
        router_w: &Self,
        moe_gate_w: &Self,
        moe_up_w: &Self,
        moe_down_w: &Self,
        shared_gate_w: &Self,
        shared_up_w: &Self,
        shared_down_w: &Self,
        shared_expert_gate_w: &Self,
        top_k: i32,
        norm_topk_prob: bool,
    ) -> Self {
        let mut out = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_compiled_moe_layer_fixed(
                out.as_mut_ptr(),
                &x.raw,
                &router_w.raw,
                &moe_gate_w.raw,
                &moe_up_w.raw,
                &moe_down_w.raw,
                &shared_gate_w.raw,
                &shared_up_w.raw,
                &shared_down_w.raw,
                &shared_expert_gate_w.raw,
                top_k,
                norm_topk_prob,
            );
            Self {
                raw: out.assume_init(),
            }
        }
    }

    /// Load a single array from a safetensors file by key name.
    /// Uses pmetal-bridge's MLX instance (not mlx-rs) — critical for avoiding
    /// dual-allocator interference.
    pub fn load_safetensors(path: &str, key: &str) -> Option<Self> {
        let c_path = std::ffi::CString::new(path).ok()?;
        let c_key = std::ffi::CString::new(key).ok()?;
        let mut dst = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            if mlx_inline_load_safetensors_key(dst.as_mut_ptr(), c_path.as_ptr(), c_key.as_ptr())
                == 0
            {
                Some(Self {
                    raw: dst.assume_init(),
                })
            } else {
                let mapped = map_safetensors_file(path)?;
                let tensors = SafeTensors::deserialize(&mapped).ok()?;
                let tensor = tensors.tensor(key).ok()?;
                inline_array_from_tensor_view(&tensor)
            }
        }
    }

    /// Create an array from a flat f32 slice with an explicit shape.
    ///
    /// Zero-copy on the C++ side: MLX creates the array pointing at the data
    /// which is then eval'd into a Metal buffer.  `shape` must satisfy
    /// `shape.iter().product() == data.len()`.
    pub fn from_f32_slice(data: &[f32], shape: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_f32_slice(
                dst.as_mut_ptr(),
                data.as_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn from_u32_slice(data: &[u32], shape: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_u32_slice(
                dst.as_mut_ptr(),
                data.as_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn from_u8_slice(data: &[u8], shape: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_u8_slice(
                dst.as_mut_ptr(),
                data.as_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn from_u16_bits_slice(data: &[u16], shape: &[i32], dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_u16_bits_slice(
                dst.as_mut_ptr(),
                data.as_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
                dtype,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Copy all f32 values out of this array into a `Vec<f32>`.
    ///
    /// The array is cast to f32 and evaluated (GPU → CPU sync) before copying.
    /// Returns `None` when the element count doesn't match `n` or on dtype error.
    pub fn to_f32_vec(&mut self, n: usize) -> Option<Vec<f32>> {
        let mut out = vec![0.0f32; n];
        let rc = unsafe { mlx_inline_to_f32_slice(&mut self.raw, out.as_mut_ptr(), n) };
        if rc == 0 { Some(out) } else { None }
    }

    /// Create a 1-D int32 array from a Rust slice — zero copy for token IDs.
    ///
    /// Typical use: prefill token IDs for `embedding.take_axis(ids, 0)`.
    pub fn from_i32_slice(data: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_from_i32_slice(dst.as_mut_ptr(), data.as_ptr(), data.len() as i32);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Create a shaped int32 array from a Rust slice.
    ///
    /// Shape must satisfy `shape.iter().product::<i32>() == data.len() as i32`.
    pub fn from_i32_slice_shaped(data: &[i32], shape: &[i32]) -> Self {
        Self::from_i32_slice(data).reshape(shape)
    }

    /// Generic `from_slice` compatible with mlx-rs `Array::from_slice::<T>(data, shape)`.
    ///
    /// Supports `i32`, `f32`, and `u32` element types via the [`ArrayElement`] trait.
    /// Typical usage:
    /// ```ignore
    /// let arr = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
    /// let arr = Array::from_slice(&[0.1f32, 0.2], &[2]);
    /// ```
    pub fn from_slice<T: ArrayElement>(data: &[T], shape: &[i32]) -> Self {
        T::into_array(data, shape)
    }

    /// Create a range [0, 1, ..., n-1] with full Metal buffer (no broadcast).
    /// Useful for benchmarks — ensures matmuls read real data from GPU memory.
    pub fn arange(n: i32, dtype: i32) -> Self {
        let mut dst = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_arange(dst.as_mut_ptr(), n, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Sever the computation graph, freeing all input references.
    /// CRITICAL for cache arrays: without this, cache updates chain across
    /// decode steps, keeping ALL previous steps' Metal buffers alive.
    /// Call on cache arrays after each eval to prevent memory accumulation.
    #[inline]
    pub fn detach(&mut self) {
        unsafe { mlx_inline_detach(&mut self.raw) }
    }

    // ── Sampling ──────────────────────────────────────────────────────

    #[inline]
    pub fn argmax(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_argmax(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn argmin(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_argmin(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Element-wise absolute value.
    #[inline]
    pub fn abs(&self) -> Self {
        self.abs_val()
    }

    /// Element-wise absolute value (alias to avoid f32::abs conflict in some contexts).
    #[inline]
    pub fn abs_val(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_abs(dst.as_mut_ptr(), &self.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn logsumexp(&self, axis: i32, keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_logsumexp(dst.as_mut_ptr(), &self.raw, axis, keepdims);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn categorical(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_categorical(dst.as_mut_ptr(), &self.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Embedding / KV cache ────────────────────────────────────────────

    /// Take rows along axis (embedding lookup: `take(weight, indices, axis=0)`).
    #[inline]
    pub fn take_axis(&self, indices: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_take_axis(dst.as_mut_ptr(), &self.raw, &indices.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Concatenate cached and new K/V along the sequence axis.
    #[inline]
    pub fn kv_cache_append(&self, new_kv: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_kv_cache_append(dst.as_mut_ptr(), &self.raw, &new_kv.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Async eval (non-blocking).
    #[inline]
    pub fn async_eval_ref(&self) {
        unsafe { mlx_inline_async_eval_arr(&self.raw) }
    }

    // ── GDN Metal kernel step ───────────────────────────────────────────

    /// GDN recurrence with pre-computed g and beta. Uses Metal kernel (1 dispatch)
    /// when dk%32==0 && dk<=256, otherwise falls back to ops.
    #[inline]
    pub fn gdn_metal_step(
        q: &Self,
        k: &Self,
        v: &Self,
        g: &Self,
        beta: &Self,
        state: &Self,
        t: i32,
    ) -> (Self, Self) {
        let mut dst_y = MaybeUninit::<RawBuf>::uninit();
        let mut dst_state = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_gdn_metal_step(
                dst_y.as_mut_ptr(),
                dst_state.as_mut_ptr(),
                &q.raw,
                &k.raw,
                &v.raw,
                &g.raw,
                &beta.raw,
                &state.raw,
                t,
            );
            (
                Self {
                    raw: dst_y.assume_init(),
                },
                Self {
                    raw: dst_state.assume_init(),
                },
            )
        }
    }

    // ── TurboQuant fused Metal kernels ──────────────────────────────────

    /// Fused TurboQuant encode: nearest-centroid search over a tiny codebook.
    ///
    /// Replaces the expand_dims+subtract+square+argmin chain that allocates a
    /// huge `[N, D, C]` intermediate tensor.  For D=128, C=8 (3-bit MSE), N=100
    /// the old intermediate is 409 600 f32 elements per call; this kernel uses
    /// only registers (n_centroids ≤ 16).
    ///
    /// - `input`: `[N, D]` f32 — already normalised onto the unit sphere AND
    ///   rotated by the orthogonal projection matrix.
    /// - `codebook`: `[C]` f32, C ≤ 16.
    /// - Returns `indices [N, D]` uint32 on success, `None` if Metal unavailable.
    ///
    /// Norm computation (`keys.norm_l2(-1, true)`) and the rotation matmul are
    /// handled by the caller before calling this function.
    pub fn turboquant_encode(
        input: &Self,
        codebook: &Self,
        dim: u32,
        n_centroids: u32,
        n_rows: u32,
    ) -> Option<Self> {
        let mut out_indices = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_encode(
                out_indices.as_mut_ptr(),
                std::ptr::null_mut(), // norms: reserved
                &input.raw,
                &codebook.raw,
                dim,
                n_centroids,
                n_rows,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out_indices.assume_init() },
            })
        } else {
            None
        }
    }

    /// Fused TurboQuant decode: codebook lookup producing `[N, D]` f32 centroid
    /// values in the rotated domain.
    ///
    /// Replaces: `take(codebook, flat_indices, 0).reshape(orig_shape)`.
    /// The result is **un-scaled** (no norm multiplication) and in the *rotated*
    /// domain.  The caller multiplies by norms and matmuls with the rotation
    /// matrix to recover the original input-space vectors.
    ///
    /// - `indices`: `[N, D]` uint32.
    /// - `codebook`: `[C]` f32, C ≤ 16.
    /// - Returns `output [N, D]` f32 on success, `None` if Metal unavailable.
    pub fn turboquant_decode(
        indices: &Self,
        codebook: &Self,
        dim: u32,
        n_centroids: u32,
        n_rows: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_decode(
                out.as_mut_ptr(),
                &indices.raw,
                std::ptr::null_mut(), // norms: reserved
                &codebook.raw,
                dim,
                n_centroids,
                n_rows,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Fused TurboQuant key scoring directly from compressed indices/signs.
    ///
    /// Inputs:
    /// - `query_rot` / `query_proj`: `[N, D]` f32
    /// - `indices`: `[N, D, S]` transposed uint8 key indices
    /// - `qjl_signs`: `[N, ceil(D/32), S]` packed uint32 sign words
    /// - `norms` / `residual_norms`: `[N, S]` f32
    /// - `codebook`: `[C]` f32
    ///
    /// Returns `scores [N, S]` f32 on success.
    pub fn turboquant_score(
        query_rot: &Self,
        query_proj: &Self,
        indices: &Self,
        qjl_signs: &Self,
        norms: &Self,
        residual_norms: &Self,
        codebook: &Self,
        dim: u32,
        qjl_words: u32,
        n_centroids: u32,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_score(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &indices.raw,
                &qjl_signs.raw,
                &norms.raw,
                &residual_norms.raw,
                &codebook.raw,
                dim,
                qjl_words,
                n_centroids,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized q8 key scoring for D=256 on the seq-major transposed cache layout.
    pub fn turboquant_score_q8_d256(
        query_rot: &Self,
        query_proj: &Self,
        indices: &Self,
        qjl_signs: &Self,
        norms: &Self,
        residual_norms: &Self,
        codebook: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_score_q8_d256(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &indices.raw,
                &qjl_signs.raw,
                &norms.raw,
                &residual_norms.raw,
                &codebook.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Fused mixed TurboQuant key scoring directly from regular/outlier
    /// compressed subspaces.
    pub fn turboquant_mixed_score(
        regular_query_rot: &Self,
        regular_query_proj: &Self,
        regular_indices: &Self,
        regular_qjl_signs: &Self,
        regular_norms: &Self,
        regular_residual_norms: &Self,
        regular_codebook: &Self,
        outlier_query_rot: &Self,
        outlier_query_proj: &Self,
        outlier_indices: &Self,
        outlier_qjl_signs: &Self,
        outlier_norms: &Self,
        outlier_residual_norms: &Self,
        outlier_codebook: &Self,
        regular_dim: u32,
        regular_qjl_words: u32,
        regular_n_centroids: u32,
        outlier_dim: u32,
        outlier_qjl_words: u32,
        outlier_n_centroids: u32,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_mixed_score(
                out.as_mut_ptr(),
                &regular_query_rot.raw,
                &regular_query_proj.raw,
                &regular_indices.raw,
                &regular_qjl_signs.raw,
                &regular_norms.raw,
                &regular_residual_norms.raw,
                &regular_codebook.raw,
                &outlier_query_rot.raw,
                &outlier_query_proj.raw,
                &outlier_indices.raw,
                &outlier_qjl_signs.raw,
                &outlier_norms.raw,
                &outlier_residual_norms.raw,
                &outlier_codebook.raw,
                regular_dim,
                regular_qjl_words,
                regular_n_centroids,
                outlier_dim,
                outlier_qjl_words,
                outlier_n_centroids,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Pack `sign(projected >= 0)` along the last dimension into uint32 words.
    ///
    /// - `projected`: `[N, D]` f32
    /// - Returns packed `[N, ceil(D/32)]` uint32 on success.
    pub fn turboquant_pack_sign_bits(
        projected: &Self,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_pack_sign_bits(
                out.as_mut_ptr(),
                &projected.raw,
                dim,
                packed_dim,
                n_rows,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Pack q8 key bytes from centroid indices and packed QJL signs.
    ///
    /// - `indices`: `[N, D, S_cap]` uint8
    /// - `qjl_signs`: `[N, ceil(D/32), S_cap]` uint32
    /// - Returns `[N, D, S_cap]` uint8 where low 7 bits are the centroid index
    ///   and the high bit is the QJL sign.
    pub fn turboquant_pack_q8_keybytes(
        indices: &Self,
        qjl_signs: &Self,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
        cache_seq_capacity: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_pack_q8_keybytes(
                out.as_mut_ptr(),
                &indices.raw,
                &qjl_signs.raw,
                dim,
                packed_dim,
                n_rows,
                cache_seq_capacity,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Pack q8 key bytes directly into a seq-major shadow layout.
    ///
    /// - `indices`: `[N, D, S_cap]` uint8
    /// - `qjl_signs`: `[N, ceil(D/32), S_cap]` uint32
    /// - Returns `[N, S_cap, D]` uint8 where low 7 bits are the centroid index
    ///   and the high bit is the QJL sign.
    pub fn turboquant_pack_q8_keybytes_seq(
        indices: &Self,
        qjl_signs: &Self,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
        cache_seq_capacity: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_pack_q8_keybytes_seq(
                out.as_mut_ptr(),
                &indices.raw,
                &qjl_signs.raw,
                dim,
                packed_dim,
                n_rows,
                cache_seq_capacity,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Pack q8 key bytes and q8 value indices into one seq-major shadow.
    ///
    /// - `indices`: `[N, D, S_cap]` uint8
    /// - `qjl_signs`: `[N, ceil(D/32), S_cap]` uint32
    /// - `value_indices`: `[N, S_cap, D]` uint8
    /// - Returns `[N, S_cap, D]` uint16 where:
    ///   low byte = key byte (low 7 bits centroid index, high bit QJL sign)
    ///   high byte = value centroid index
    pub fn turboquant_pack_q8_kvbytes_seq(
        indices: &Self,
        qjl_signs: &Self,
        value_indices: &Self,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
        cache_seq_capacity: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_pack_q8_kvbytes_seq(
                out.as_mut_ptr(),
                &indices.raw,
                &qjl_signs.raw,
                &value_indices.raw,
                dim,
                packed_dim,
                n_rows,
                cache_seq_capacity,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Unpack uint32 sign words back into `{-1,+1}` float32 signs.
    ///
    /// - `packed`: `[N, ceil(D/32)]` uint32
    /// - Returns unpacked `[N, D]` f32 on success.
    pub fn turboquant_unpack_sign_bits(
        packed: &Self,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_unpack_sign_bits(
                out.as_mut_ptr(),
                &packed.raw,
                dim,
                packed_dim,
                n_rows,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Fused TurboQuant value aggregation in the rotated domain.
    ///
    /// Inputs:
    /// - `weights`: `[N, S]` f32
    /// - `indices`: `[N, D, S_cap]` uint8
    /// - `norms`: `[N, S]` f32
    /// - `codebook`: `[C]` f32
    ///
    /// Returns `output [N, D]` f32 on success.
    pub fn turboquant_weighted_decode(
        weights: &Self,
        indices: &Self,
        norms: &Self,
        codebook: &Self,
        dim: u32,
        n_centroids: u32,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_weighted_decode(
                out.as_mut_ptr(),
                &weights.raw,
                &indices.raw,
                &norms.raw,
                &codebook.raw,
                dim,
                n_centroids,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized long-context q8 TurboQuant decode for D=256/V=256.
    ///
    /// Returns the rotated aggregated values `[N, 256]` on success.
    pub fn turboquant_attention_q8_d256_2pass(
        query_rot: &Self,
        query_proj: &Self,
        key_indices: &Self,
        key_qjl_signs: &Self,
        key_norms: &Self,
        key_residual_norms: &Self,
        key_codebook: &Self,
        value_indices: &Self,
        value_norms: &Self,
        value_codebook: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d256_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &key_indices.raw,
                &key_qjl_signs.raw,
                &key_norms.raw,
                &key_residual_norms.raw,
                &key_codebook.raw,
                &value_indices.raw,
                &value_norms.raw,
                &value_codebook.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized long-context q8 TurboQuant decode for D=256/V=256 over
    /// combined slot-major storage:
    /// - packed key bytes `[N, S_cap, D]`
    /// - value indices `[N, S_cap, D]`
    ///
    /// Returns the rotated aggregated values `[N, 256]` on success.
    pub fn turboquant_attention_q8_d256_packed_keys_2pass(
        query_rot: &Self,
        query_proj: &Self,
        key_bytes: &Self,
        slot_scales: &Self,
        key_codebook: &Self,
        value_indices: &Self,
        value_codebook: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d256_packed_keys_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &key_bytes.raw,
                &slot_scales.raw,
                &key_codebook.raw,
                &value_indices.raw,
                &value_codebook.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized long-context q8 TurboQuant decode for D=256/V=256 over
    /// a seq-major packed `{key,value}` shadow:
    /// - `kv_bytes`: `[N, S_cap, D]` uint16
    ///   low byte = key byte
    ///   high byte = value centroid index
    /// - `slot_scales`: `[N, S_cap, 4]` f32
    ///
    /// Returns the rotated aggregated values `[N, 256]` on success.
    pub fn turboquant_attention_q8_d256_packed_kv_2pass(
        query_rot: &Self,
        query_proj: &Self,
        kv_bytes: &Self,
        slot_scales: &Self,
        key_codebook: &Self,
        value_codebook: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d256_packed_kv_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &kv_bytes.raw,
                &slot_scales.raw,
                &key_codebook.raw,
                &value_codebook.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized long-context q8 TurboQuant decode for D=256/V=256 over
    /// a seq-major packed key shadow plus dense rotated values:
    /// - `kv_bytes`: `[N, S_cap, D]` uint16, low byte = key byte
    /// - `value_dense`: `[N, S_cap, D]` bf16/f32 rotated dense values
    ///
    /// Returns the rotated aggregated values `[N, 256]` on success.
    pub fn turboquant_attention_q8_d256_packed_kv_dense_values_2pass(
        query_rot: &Self,
        query_proj: &Self,
        kv_bytes: &Self,
        slot_scales: &Self,
        key_codebook: &Self,
        value_dense: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d256_packed_kv_dense_values_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &kv_bytes.raw,
                &slot_scales.raw,
                &key_codebook.raw,
                &value_dense.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized long-context q8 TurboQuant decode for D=128/V=128.
    ///
    /// Returns the rotated aggregated values `[N, 128]` on success.
    pub fn turboquant_attention_q8_d128_2pass(
        query_rot: &Self,
        query_proj: &Self,
        key_indices: &Self,
        key_qjl_signs: &Self,
        key_norms: &Self,
        key_residual_norms: &Self,
        key_codebook: &Self,
        value_indices: &Self,
        value_norms: &Self,
        value_codebook: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d128_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &key_indices.raw,
                &key_qjl_signs.raw,
                &key_norms.raw,
                &key_residual_norms.raw,
                &key_codebook.raw,
                &value_indices.raw,
                &value_norms.raw,
                &value_codebook.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Specialized long-context q8 TurboQuant decode for D=128/V=128 over
    /// packed key bytes stored as `[N, D, S_cap]`.
    ///
    /// Returns the rotated aggregated values `[N, 128]` on success.
    pub fn turboquant_attention_q8_d128_packed_keys_2pass(
        query_rot: &Self,
        query_proj: &Self,
        key_bytes: &Self,
        key_norms: &Self,
        key_residual_norms: &Self,
        key_codebook: &Self,
        value_indices: &Self,
        value_norms: &Self,
        value_codebook: &Self,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale: f32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_attention_q8_d128_packed_keys_2pass(
                out.as_mut_ptr(),
                &query_rot.raw,
                &query_proj.raw,
                &key_bytes.raw,
                &key_norms.raw,
                &key_residual_norms.raw,
                &key_codebook.raw,
                &value_indices.raw,
                &value_norms.raw,
                &value_codebook.raw,
                n_rows,
                n_seq,
                cache_seq_capacity,
                q_heads,
                kv_heads,
                attn_scale.to_bits(),
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Gather selected coordinates from a `[N, D]` f32 tensor.
    pub fn turboquant_gather_last_dim(
        input: &Self,
        positions: &Self,
        full_dim: u32,
        out_dim: u32,
        n_rows: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_gather_last_dim(
                out.as_mut_ptr(),
                &input.raw,
                &positions.raw,
                full_dim,
                out_dim,
                n_rows,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    /// Scatter regular/outlier component rows back into `[N, D]` f32 rows.
    pub fn turboquant_scatter_last_dim(
        regular: &Self,
        outlier: &Self,
        regular_positions: &Self,
        outlier_positions: &Self,
        full_dim: u32,
        regular_dim: u32,
        outlier_dim: u32,
        n_rows: u32,
    ) -> Option<Self> {
        let mut out = MaybeUninit::<RawBuf>::uninit();
        let rc = unsafe {
            mlx_inline_turboquant_scatter_last_dim(
                out.as_mut_ptr(),
                &regular.raw,
                &outlier.raw,
                &regular_positions.raw,
                &outlier_positions.raw,
                full_dim,
                regular_dim,
                outlier_dim,
                n_rows,
            )
        };
        if rc == 0 {
            Some(Self {
                raw: unsafe { out.assume_init() },
            })
        } else {
            None
        }
    }

    // ── Fused compiled ops (match Python's @mx.compile) ─────────────────

    /// Fused SwiGLU: `silu(gate) * up` → 1 compiled dispatch instead of 3.
    #[inline]
    pub fn fused_swiglu(gate: &Self, up: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_fused_swiglu(dst.as_mut_ptr(), &gate.raw, &up.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Fused SiLU: `x * sigmoid(x)` → 1 compiled dispatch instead of 2.
    #[inline]
    pub fn fused_silu(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_fused_silu(dst.as_mut_ptr(), &self.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Fused compute_g: `exp(-exp(A_log.f32()) * softplus(a + dt_bias))` → 1 compiled dispatch instead of 6.
    #[inline]
    pub fn fused_compute_g(a_log: &Self, a: &Self, dt_bias: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_fused_compute_g(dst.as_mut_ptr(), &a_log.raw, &a.raw, &dt_bias.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Fused precise SwiGLU: `(silu(gate.f32()) * x.f32()).as(x.dtype)` → 1 compiled dispatch instead of 5.
    #[inline]
    pub fn fused_precise_swiglu(x: &Self, gate: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_fused_precise_swiglu(dst.as_mut_ptr(), &x.raw, &gate.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Slice access (requires prior eval) ───────────────────────────────

    /// Return a borrowed slice of the array's f32 data.
    ///
    /// # Panics
    /// Panics if the array has not been evaluated (GPU→CPU sync), if the
    /// dtype is not Float32, or if the data pointer is null.
    pub fn as_slice<T: crate::inline_array::BridgeScalar>(&self) -> &[T] {
        let ptr = self.data_ptr() as *const T;
        assert!(
            !ptr.is_null(),
            "as_slice: array not evaluated (null data ptr)"
        );
        let n = self.size();
        // SAFETY: `data_ptr` returns a valid pointer into MLX's heap allocation
        // for the lifetime of `self`. The array must have been `eval()`d first
        // so the pointer is on the CPU (not on the GPU).
        unsafe { std::slice::from_raw_parts(ptr, n) }
    }

    // ── Item extraction ───────────────────────────────────────────────────

    pub fn item_f32(&self) -> f32 {
        let mut owned = self.clone();
        owned.eval();
        unsafe { mlx_inline_item_f32(&mut owned.raw) }
    }
    pub fn item_u32(&self) -> u32 {
        let mut owned = self.clone();
        owned.eval();
        unsafe { mlx_inline_item_u32(&mut owned.raw) }
    }

    // ── Indexing / slicing ────────────────────────────────────────────────

    #[inline]
    pub fn concatenate_2(&self, other: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_concatenate_2(dst.as_mut_ptr(), &self.raw, &other.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn where_cond(&self, a: &Self, b: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_where(dst.as_mut_ptr(), &self.raw, &a.raw, &b.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn slice(&self, start: &[i32], stop: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_slice(
                dst.as_mut_ptr(),
                &self.raw,
                start.as_ptr(),
                stop.as_ptr(),
                start.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// In-place slice update. Consumes self so MLX sees refcount=1 and can
    /// mutate the buffer directly (zero allocation). Matches Python's
    /// `self.keys[..., prev:offset, :] = keys` pattern.
    #[inline]
    pub fn slice_set(&self, value: &Self, start: &[i32], stop: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_slice_set(
                dst.as_mut_ptr(),
                &self.raw,
                &value.raw,
                start.as_ptr(),
                stop.as_ptr(),
                start.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn repeat(&self, repeats: i32, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_repeat(dst.as_mut_ptr(), &self.raw, repeats, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn squeeze(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_squeeze(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Squeeze all size-1 dimensions (multi-axis compat alias).
    #[inline]
    pub fn squeeze_axes(&self, axes: &[i32]) -> Self {
        let mut result = self.clone();
        // Process axes in descending order to maintain correct indices
        let mut sorted = axes.to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        for &ax in &sorted {
            result = result.squeeze(ax);
        }
        result
    }

    #[inline]
    pub fn expand_dims(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_expand_dims(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Multi-axis expand_dims — insert a new size-1 axis at each position.
    ///
    /// Compatible with mlx-rs `expand_dims_axes(&[ax1, ax2, ...])`.
    #[inline]
    pub fn expand_dims_axes(&self, axes: &[i32]) -> Self {
        let mut result = self.clone();
        // Insert axes in ascending order (each insertion shifts subsequent axes)
        let mut sorted = axes.to_vec();
        sorted.sort_unstable();
        for &ax in &sorted {
            result = result.expand_dims(ax);
        }
        result
    }

    #[inline]
    pub fn transpose_axes(&self, axes: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_transpose_axes(
                dst.as_mut_ptr(),
                &self.raw,
                axes.as_ptr(),
                axes.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn cumsum(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_cumsum(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub fn tril(&self, k: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_tril(dst.as_mut_ptr(), &self.raw, k);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    #[inline]
    pub(crate) fn index_array(&self, indices: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_index(dst.as_mut_ptr(), &self.raw, &indices.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Index or slice this array using the compatibility bridge.
    ///
    /// Supports gather indexing with an index array as well as mlx-rs style
    /// integer and tuple/range slicing via `compat::indexing::IndexOp`.
    #[inline]
    pub fn index<Idx>(&self, idx: Idx) -> Self
    where
        Self: crate::compat::indexing::IndexOp<Idx>,
    {
        <Self as crate::compat::indexing::IndexOp<Idx>>::index(self, idx)
    }

    // ── GDN recurrence ────────────────────────────────────────────────────

    /// GDN recurrence step (gated delta network) — dispatches to the fused
    /// Metal kernel when possible (inference, `Dk % 32 == 0`, `Dk <= 256`),
    /// otherwise falls back to an ops-based sequential loop.
    ///
    /// Returns `(y, new_state)`.
    #[inline]
    pub fn gdn_update(
        q: &Self,
        k: &Self,
        v: &Self,
        a: &Self,
        b: &Self,
        a_log: &Self,
        dt_bias: &Self,
        state: &Self,
        training: bool,
    ) -> (Self, Self) {
        let mut dst_y = MaybeUninit::<RawBuf>::uninit();
        let mut dst_state = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_gdn_update(
                dst_y.as_mut_ptr(),
                dst_state.as_mut_ptr(),
                &q.raw,
                &k.raw,
                &v.raw,
                &a.raw,
                &b.raw,
                &a_log.raw,
                &dt_bias.raw,
                &state.raw,
                training,
            );
            (
                Self {
                    raw: dst_y.assume_init(),
                },
                Self {
                    raw: dst_state.assume_init(),
                },
            )
        }
    }

    // ── Quantized matmul ──────────────────────────────────────────────────

    /// Quantized matmul: `x @ dequantize(w, scales, biases)`.
    #[inline]
    pub fn quantized_matmul(
        &self,
        w: &Self,
        scales: &Self,
        biases: Option<&Self>,
        transpose: bool,
        group_size: i32,
        bits: i32,
    ) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        let b_ptr = biases
            .map(|b| &b.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        unsafe {
            mlx_inline_quantized_matmul(
                dst.as_mut_ptr(),
                &self.raw,
                &w.raw,
                &scales.raw,
                b_ptr,
                transpose,
                group_size,
                bits,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Gather quantized matmul (MoE expert routing).
    #[inline]
    pub fn gather_qmm(
        &self,
        w: &Self,
        scales: &Self,
        biases: Option<&Self>,
        lhs_indices: Option<&Self>,
        rhs_indices: Option<&Self>,
        transpose: bool,
        group_size: i32,
        bits: i32,
        sorted: bool,
    ) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        let b_ptr = biases
            .map(|b| &b.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        let l_ptr = lhs_indices
            .map(|l| &l.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        let r_ptr = rhs_indices
            .map(|r| &r.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        unsafe {
            mlx_inline_gather_qmm(
                dst.as_mut_ptr(),
                &self.raw,
                &w.raw,
                &scales.raw,
                b_ptr,
                l_ptr,
                r_ptr,
                transpose,
                group_size,
                bits,
                sorted,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Training ops: random ──────────────────────────────────────────────

    /// Sample from N(0,1) with given shape and dtype.
    pub fn random_normal(shape: &[i32], dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_random_normal(dst.as_mut_ptr(), shape.as_ptr(), shape.len() as i32, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Sample from U(0,1) with given shape and dtype.
    pub fn random_uniform(shape: &[i32], dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_random_uniform(dst.as_mut_ptr(), shape.as_ptr(), shape.len() as i32, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Sample Bernoulli with given probability and shape.
    pub fn random_bernoulli(p: &Self, shape: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_random_bernoulli(
                dst.as_mut_ptr(),
                &p.raw,
                shape.as_ptr(),
                shape.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Random integers in [low, high) with given shape and dtype.
    pub fn random_randint(low: i32, high: i32, shape: &[i32], dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_random_randint(
                dst.as_mut_ptr(),
                low,
                high,
                shape.as_ptr(),
                shape.len() as i32,
                dtype,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Training ops: math ────────────────────────────────────────────────

    pub fn mean_axis(&self, axis: i32, keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_mean_axis(dst.as_mut_ptr(), &self.raw, axis, keepdims);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn mean_all(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_mean_all(dst.as_mut_ptr(), &self.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn var_axis(&self, axis: i32, keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_var_axis(dst.as_mut_ptr(), &self.raw, axis, keepdims);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    binop!(pow, mlx_inline_pow);
    unop!(reciprocal, mlx_inline_reciprocal);
    unop!(sin, mlx_inline_sin);
    unop!(cos, mlx_inline_cos);
    unop!(rsqrt, mlx_inline_rsqrt);
    unop!(zeros_like, mlx_inline_zeros_like);
    unop!(ones_like, mlx_inline_ones_like);
    unop!(square, mlx_inline_square);
    unop!(relu, mlx_inline_relu);
    unop!(gelu, mlx_inline_gelu);
    unop!(stop_gradient, mlx_inline_stop_gradient);

    /// Compute the inverse of a triangular matrix (batched over leading dims).
    ///
    /// `upper=false` (default) inverts a lower-triangular matrix.
    /// `use_cpu=true` dispatches on the CPU stream — matches mlx-lm's GDN
    /// WY factorization which calls `tri_inv(StreamOrDevice::cpu())` because
    /// `tri_inv` has no registered VJP and must stay off the autograd tape.
    pub fn tri_inv(&self, upper: bool, use_cpu: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_tri_inv(dst.as_mut_ptr(), &self.raw, upper, use_cpu);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Singular Value Decomposition — returns `(U, S, Vt)`.
    ///
    /// Economy/thin SVD: `U` is `[m, k]`, `S` is `[k]`, `Vt` is `[k, n]`
    /// where `k = min(m, n)`.  Always runs on the CPU stream.
    pub fn svd(&self) -> (Self, Self, Self) {
        let mut u = MaybeUninit::<RawBuf>::uninit();
        let mut s = MaybeUninit::<RawBuf>::uninit();
        let mut vt = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_svd(u.as_mut_ptr(), s.as_mut_ptr(), vt.as_mut_ptr(), &self.raw);
            (
                Self {
                    raw: u.assume_init(),
                },
                Self {
                    raw: s.assume_init(),
                },
                Self {
                    raw: vt.assume_init(),
                },
            )
        }
    }

    /// Clip values to [lo, hi]. Either bound can be None.
    pub fn clip(&self, lo: Option<&Self>, hi: Option<&Self>) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        let lo_ptr = lo
            .map(|x| &x.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        let hi_ptr = hi
            .map(|x| &x.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        unsafe {
            mlx_inline_clip(dst.as_mut_ptr(), &self.raw, lo_ptr, hi_ptr);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn log_softmax(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_log_softmax(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Cross-entropy: -sum(targets * log_softmax(self), axis).
    pub fn cross_entropy(&self, targets: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_cross_entropy(dst.as_mut_ptr(), &self.raw, &targets.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Training ops: creation ────────────────────────────────────────────

    /// Constant-filled array.
    pub fn full(shape: &[i32], val: f32, dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_full(
                dst.as_mut_ptr(),
                shape.as_ptr(),
                shape.len() as i32,
                val,
                dtype,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Identity matrix [n, n].
    pub fn eye(n: i32, dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_eye(dst.as_mut_ptr(), n, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Triangular matrix [n, m] with diagonal offset k.
    pub fn tri(n: i32, m: i32, k: i32, dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_tri(dst.as_mut_ptr(), n, m, k, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Training ops: shape ───────────────────────────────────────────────

    pub fn broadcast_to(&self, shape: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_broadcast_to(
                dst.as_mut_ptr(),
                &self.raw,
                shape.as_ptr(),
                shape.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn flatten(&self, start_axis: i32, end_axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_flatten(dst.as_mut_ptr(), &self.raw, start_axis, end_axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Training ops: sort/reduction ──────────────────────────────────────

    pub fn argsort(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_argsort(dst.as_mut_ptr(), &self.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Sum all elements to a scalar.
    pub fn sum_all(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_sum_all(dst.as_mut_ptr(), &self.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn max_axis(&self, axis: i32, keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_max_axis(dst.as_mut_ptr(), &self.raw, axis, keepdims);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn min_axis(&self, axis: i32, keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_min_axis(dst.as_mut_ptr(), &self.raw, axis, keepdims);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    binop!(minimum, mlx_inline_minimum);

    /// Sum over multiple axes.
    pub fn sum_axes(&self, axes: &[i32], keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_sum_axes(
                dst.as_mut_ptr(),
                &self.raw,
                axes.as_ptr(),
                axes.len() as i32,
                keepdims,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Mean over multiple axes.
    pub fn mean_axes(&self, axes: &[i32], keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_mean_axes(
                dst.as_mut_ptr(),
                &self.raw,
                axes.as_ptr(),
                axes.len() as i32,
                keepdims,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── Training ops: comparison ──────────────────────────────────────────

    binop!(equal, mlx_inline_equal);
    binop!(not_equal, mlx_inline_not_equal);
    binop!(greater, mlx_inline_greater);
    binop!(less, mlx_inline_less);
    binop!(greater_equal, mlx_inline_greater_equal);
    binop!(less_equal, mlx_inline_less_equal);

    // ── Training ops: serialization ───────────────────────────────────────

    /// Save arrays to safetensors format.
    pub fn save_safetensors(path: &str, entries: &[(&str, &InlineArray)]) {
        let c_path = std::ffi::CString::new(path).expect("null byte in path");
        let c_keys: Vec<std::ffi::CString> = entries
            .iter()
            .map(|(k, _)| std::ffi::CString::new(*k).expect("null byte in key"))
            .collect();
        let key_ptrs: Vec<*const std::ffi::c_char> = c_keys.iter().map(|k| k.as_ptr()).collect();
        // Build a contiguous array of RawBufs (copy refs, not move)
        let raw_arrays: Vec<RawBuf> = entries.iter().map(|(_, a)| a.raw).collect();
        unsafe {
            mlx_inline_save_safetensors(
                c_path.as_ptr(),
                key_ptrs.as_ptr(),
                raw_arrays.as_ptr(),
                entries.len() as i32,
            );
        }
    }

    // ── Training ops: quantize ────────────────────────────────────────────

    /// Quantize: returns (packed_weights, scales, biases).
    pub fn quantize_weights(&self, group_size: i32, bits: i32) -> (Self, Self, Self) {
        let mut w = MaybeUninit::<RawBuf>::uninit();
        let mut s = MaybeUninit::<RawBuf>::uninit();
        let mut b = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_quantize(
                w.as_mut_ptr(),
                s.as_mut_ptr(),
                b.as_mut_ptr(),
                &self.raw,
                group_size,
                bits,
            );
            (
                Self {
                    raw: w.assume_init(),
                },
                Self {
                    raw: s.assume_init(),
                },
                Self {
                    raw: b.assume_init(),
                },
            )
        }
    }

    // ── Training ops: misc ────────────────────────────────────────────────

    /// Alias for pow (mlx-rs compat).
    pub fn power(&self, other: &Self) -> Self {
        self.pow(other)
    }

    /// gt/lt/ge/le aliases for compat with mlx-rs naming.
    pub fn eq(&self, other: &Self) -> Self {
        self.equal(other)
    }
    pub fn ne(&self, other: &Self) -> Self {
        self.not_equal(other)
    }
    pub fn gt(&self, other: &Self) -> Self {
        self.greater(other)
    }
    pub fn lt(&self, other: &Self) -> Self {
        self.less(other)
    }
    pub fn ge(&self, other: &Self) -> Self {
        self.greater_equal(other)
    }
    pub fn le(&self, other: &Self) -> Self {
        self.less_equal(other)
    }

    /// Swap two axes.
    pub fn swap_axes(&self, a: i32, b: i32) -> Self {
        let ndim = self.ndim();
        let mut perm: Vec<i32> = (0..ndim).map(|i| i as i32).collect();
        let a_idx = if a < 0 { ndim as i32 + a } else { a } as usize;
        let b_idx = if b < 0 { ndim as i32 + b } else { b } as usize;
        perm.swap(a_idx, b_idx);
        self.transpose_axes(&perm)
    }

    /// Total element count.
    pub fn size(&self) -> usize {
        unsafe { mlx_inline_size(&self.raw) }
    }

    /// Total byte count.
    pub fn nbytes(&self) -> usize {
        unsafe { mlx_inline_nbytes(&self.raw) }
    }

    /// Get a raw const pointer to the evaluated data.
    /// Array must be evaluated first.
    pub fn data_ptr(&self) -> *const std::ffi::c_void {
        let mut ptr: *const std::ffi::c_void = std::ptr::null();
        unsafe { mlx_inline_data_ptr(&self.raw, &mut ptr) };
        ptr
    }

    // ── FFT ───────────────────────────────────────────────────────────────

    /// Real-valued FFT along `axis`. Pass `n_fft = -1` to use the full axis length.
    #[inline]
    pub fn rfft(&self, n_fft: i32, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_rfft(dst.as_mut_ptr(), &self.raw, n_fft, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Inverse real-valued FFT along `axis`. Pass `n_fft = -1` to infer from input.
    #[inline]
    pub fn irfft(&self, n_fft: i32, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_irfft(dst.as_mut_ptr(), &self.raw, n_fft, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── leaky_relu ────────────────────────────────────────────────────────

    /// Leaky ReLU: `max(neg_slope * x, x)`.
    #[inline]
    pub fn leaky_relu(&self, neg_slope: f32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_leaky_relu(dst.as_mut_ptr(), &self.raw, neg_slope);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── squeeze all ───────────────────────────────────────────────────────

    /// Remove all size-1 dimensions.
    #[inline]
    pub fn squeeze_all(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_squeeze_all(dst.as_mut_ptr(), &self.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── pad ───────────────────────────────────────────────────────────────

    /// Pad array with constant value.
    ///
    /// `pad_widths`: slice of `(before, after)` pairs for each axis, flattened:
    /// `[before_0, after_0, before_1, after_1, ...]`.  Length must be `2 * ndim`.
    /// Tile `self` by `reps` along each axis.
    pub fn tile(&self, reps: &[i32]) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_tile(
                dst.as_mut_ptr(),
                &self.raw,
                reps.as_ptr(),
                reps.len() as i32,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Linspace scalar creation — returns a 1-D array of `n` evenly-spaced values.
    pub fn linspace(start: f32, stop: f32, n: i32, dtype: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_linspace(dst.as_mut_ptr(), start, stop, n, dtype);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Split into `sections` equal parts along `axis`.
    pub fn split_sections(&self, sections: i32, axis: i32) -> Vec<Self> {
        // Allocate enough output slots.
        let max = sections as usize;
        let mut buf: Vec<MaybeUninit<RawBuf>> = (0..max).map(|_| MaybeUninit::uninit()).collect();
        let mut out_count: i32 = 0;
        unsafe {
            mlx_inline_split_sections(
                buf[0].as_mut_ptr(),
                &self.raw,
                sections,
                axis,
                &mut out_count,
            );
        }
        (0..out_count as usize)
            .map(|i| unsafe {
                Self {
                    raw: buf[i].assume_init(),
                }
            })
            .collect()
    }

    /// Scatter-add: `self[indices] += updates` along `axis`.
    pub fn scatter_add_axis(&self, indices: &Self, updates: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_scatter_add(
                dst.as_mut_ptr(),
                &self.raw,
                &indices.raw,
                &updates.raw,
                axis,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Top-k values along `axis`.
    pub fn topk(&self, k: i32, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_topk(dst.as_mut_ptr(), &self.raw, k, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Put values at `indices` along `axis` (in-place scatter).
    pub fn put_along_axis_op(&self, indices: &Self, values: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_put_along_axis(dst.as_mut_ptr(), &self.raw, &indices.raw, &values.raw, axis);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// Layer normalisation. `weight` and `bias` may be null (use `std::ptr::null()`).
    pub fn layer_norm(&self, weight: Option<&Self>, bias: Option<&Self>, eps: f32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        let w_ptr = weight
            .map(|w| &w.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        let b_ptr = bias
            .map(|b| &b.raw as *const RawBuf)
            .unwrap_or(std::ptr::null());
        unsafe {
            mlx_inline_layer_norm(dst.as_mut_ptr(), &self.raw, w_ptr, b_ptr, eps);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// `c + a @ b` (addmm).
    pub fn addmm(c: &Self, a: &Self, b: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_addmm(dst.as_mut_ptr(), &c.raw, &a.raw, &b.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    /// 2-D convolution (NHWC format, MLX standard).
    pub fn conv2d(
        &self,
        weight: &Self,
        stride_h: i32,
        stride_w: i32,
        pad_h: i32,
        pad_w: i32,
        dil_h: i32,
        dil_w: i32,
        groups: i32,
    ) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_conv2d(
                dst.as_mut_ptr(),
                &self.raw,
                &weight.raw,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dil_h,
                dil_w,
                groups,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn pad_constant(&self, pad_widths_flat: &[i32], fill_value: f32) -> Self {
        debug_assert_eq!(pad_widths_flat.len(), 2 * self.ndim() as usize);
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_pad(
                dst.as_mut_ptr(),
                &self.raw,
                pad_widths_flat.as_ptr(),
                (pad_widths_flat.len() / 2) as i32,
                fill_value,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    // ── item generic ──────────────────────────────────────────────────────

    /// Extract the scalar value from a 0-d array. Evaluates lazily if needed.
    /// `T` must be `f32` or `u32` (the only types exported by the bridge).
    pub fn item<T: BridgeScalar>(&self) -> T {
        let mut owned = self.clone();
        owned.eval();
        T::extract(&mut owned)
    }

    // ── max / min scalar reductions ───────────────────────────────────────

    /// Reduce to the global maximum (returns scalar array).
    pub fn max(&self, _axis: Option<i32>) -> Self {
        // The vocoder code uses `.max(None)` for global max.
        // We reduce all axes by flattening first.
        let flat = self.flatten(0, -1);
        flat.max_axis(0, false)
    }

    /// Reduce to the global minimum (returns scalar array).
    pub fn min(&self, _axis: Option<i32>) -> Self {
        let flat = self.flatten(0, -1);
        flat.min_axis(0, false)
    }

    /// Reduce to the global sum (returns scalar array).
    pub fn sum(&self, _axis: Option<i32>) -> Self {
        self.sum_all()
    }

    /// Reduce to the global mean (returns scalar array).
    pub fn mean(&self, _axis: Option<i32>) -> Self {
        self.mean_all()
    }

    /// mlx-rs compat alias.
    pub fn logsumexp_axis(&self, axis: i32, keepdims: bool) -> Self {
        self.logsumexp(axis, keepdims)
    }

    // ── mlx-rs compat constructors ────────────────────────────────────────

    /// Convenience constructor: zeros with float32 dtype.
    /// Matches mlx-rs `Array::zeros::<f32>(&[n])`.
    #[inline]
    pub fn zeros_f32(shape: &[i32]) -> Self {
        Self::zeros(shape, crate::compat::Dtype::Float32.as_i32())
    }

    /// Convenience constructor: ones with float32 dtype.
    /// Matches mlx-rs `Array::ones::<f32>(&[n])`.
    #[inline]
    pub fn ones_f32(shape: &[i32]) -> Self {
        Self::ones(shape, crate::compat::Dtype::Float32.as_i32())
    }

    /// Convenience constructor: zeros with int32 dtype.
    /// Matches mlx-rs `Array::zeros::<i32>(&[n])`.
    #[inline]
    pub fn zeros_i32(shape: &[i32]) -> Self {
        Self::zeros(shape, crate::compat::Dtype::Int32.as_i32())
    }

    /// Cast to the specified dtype enum value.
    /// Matches mlx-rs `as_dtype(Dtype::X)` — bridge normally takes `i32`.
    #[inline]
    pub fn cast(&self, dtype: crate::compat::Dtype) -> Self {
        self.as_dtype(dtype.as_i32())
    }

    /// Cast to the same dtype as another array.
    /// Convenient replacement for `arr.as_dtype(other.dtype_raw())`.
    #[inline]
    pub fn cast_like(&self, other: &Self) -> Self {
        self.as_dtype(other.dtype_raw())
    }

    #[inline]
    pub fn unwrap(self) -> Self {
        self
    }

    #[inline]
    pub fn expect(self, _msg: &str) -> Self {
        self
    }
}

// ── Trait impls ────────────────────────────────────────────────────────────

impl AsRef<InlineArray> for InlineArray {
    #[inline]
    fn as_ref(&self) -> &InlineArray {
        self
    }
}

// ── Autograd ──────────────────────────────────────────────────────────────

/// Compute loss + gradients via callback-based autograd.
///
/// `loss_fn` receives all arrays (params first, then inputs) and must return
/// a scalar loss. Gradients are computed w.r.t. the first `params.len()` arrays.
///
/// Returns `(loss, gradients)` where `gradients[i]` is `dloss/dparams[i]`.
pub fn value_and_grad<F>(
    mut loss_fn: F,
    params: &[InlineArray],
    inputs: &[InlineArray],
) -> (InlineArray, Vec<InlineArray>)
where
    F: FnMut(&[InlineArray]) -> InlineArray,
{
    // Trampoline: C++ calls this with InlineArray-sized buffers
    unsafe extern "C" fn trampoline<F: FnMut(&[InlineArray]) -> InlineArray>(
        all_arrays: *const *const RawBuf,
        n_total: i32,
        loss_out: *mut RawBuf,
        ctx: *mut std::ffi::c_void,
    ) {
        let f = unsafe { &mut *(ctx as *mut F) };
        // Wrap raw pointers as borrowed InlineArrays (no ownership transfer)
        let arrays: Vec<InlineArray> = (0..n_total as usize)
            .map(|i| {
                let ptr = unsafe { *all_arrays.add(i) };
                let mut dst = MaybeUninit::<RawBuf>::uninit();
                unsafe { mlx_inline_init_copy(dst.as_mut_ptr(), ptr) };
                InlineArray {
                    raw: unsafe { dst.assume_init() },
                }
            })
            .collect();
        let loss = f(&arrays);
        // Write loss into output buffer (placement-copy)
        unsafe { mlx_inline_init_copy(loss_out, &loss.raw) };
        // arrays and loss drop here (calling mlx_inline_destroy for each)
    }

    let n_params = params.len();
    let n_total = n_params + inputs.len();

    // Build flat pointer array: [param0, param1, ..., input0, input1, ...]
    let all_ptrs: Vec<*const RawBuf> = params
        .iter()
        .chain(inputs.iter())
        .map(|a| &a.raw as *const RawBuf)
        .collect();

    let mut loss = InlineArray::from_f32(0.0);
    let mut grads: Vec<InlineArray> = (0..n_params).map(|_| InlineArray::from_f32(0.0)).collect();
    let mut grad_ptrs: Vec<*mut RawBuf> = grads
        .iter_mut()
        .map(|g| &mut g.raw as *mut RawBuf)
        .collect();

    unsafe {
        mlx_inline_value_and_grad(
            trampoline::<F>,
            &mut loss_fn as *mut F as *mut std::ffi::c_void,
            all_ptrs.as_ptr(),
            n_params as i32,
            n_total as i32,
            &mut loss.raw,
            grad_ptrs.as_mut_ptr(),
        );
    }

    (loss, grads)
}

// ── Crate-internal helpers for compat.rs ─────────────────────────────────

/// Copy-construct a RawBuf — equivalent to `mlx::core::array` copy constructor.
/// Used by `compat::ops` when it needs to build a contiguous slice of buffers.
#[inline]
pub(crate) unsafe fn raw_copy_buf(dst: *mut RawBuf, src: *const RawBuf) {
    unsafe { mlx_inline_init_copy(dst, src) }
}

/// Destroy a raw buffer — calls the `mlx::core::array` destructor.
#[inline]
pub(crate) unsafe fn raw_destroy(a: *mut RawBuf) {
    unsafe { mlx_inline_destroy(a) }
}

/// Concatenate a contiguous slice of RawBufs along `axis`.
#[inline]
pub(crate) unsafe fn raw_concatenate(dst: *mut RawBuf, arrays: *const RawBuf, num: i32, axis: i32) {
    unsafe { mlx_inline_concatenate(dst, arrays, num, axis) }
}

/// Stack a contiguous slice of RawBufs along a new `axis`.
#[inline]
pub(crate) unsafe fn raw_stack(dst: *mut RawBuf, arrays: *const RawBuf, num: i32, axis: i32) {
    unsafe { mlx_inline_stack(dst, arrays, num, axis) }
}

/// Wrap a raw RawBuf (already placement-new'd by C++) into an `InlineArray`.
///
/// # Safety
/// `raw` must have been initialised by a C++ placement-new (e.g. via one of
/// the `mlx_inline_*` FFI functions).  Ownership is transferred: `Drop` will
/// call the C++ destructor.
#[inline]
pub(crate) unsafe fn from_raw_buf(raw: RawBuf) -> InlineArray {
    InlineArray { raw }
}

// ── Verify buffer dimensions at startup ──────────────────────────────────

/// Panic at runtime if the Rust buffer constants don't match the C++ values.
/// Call once at program startup (or in a test).
pub fn verify_buffer_layout() {
    let sz = unsafe { mlx_inline_array_size() };
    let al = unsafe { mlx_inline_array_align() };
    assert!(
        sz <= ARRAY_BUF_SIZE,
        "mlx::core::array is {sz} bytes but ARRAY_BUF_SIZE={ARRAY_BUF_SIZE}"
    );
    assert!(
        al <= ARRAY_BUF_ALIGN,
        "mlx::core::array alignment is {al} but ARRAY_BUF_ALIGN={ARRAY_BUF_ALIGN}"
    );
}

// ── AsDtype: sealed trait for as_type<T>() ──────────────────────────────

/// Sealed trait mapping Rust primitive types to MLX dtype IDs.
///
/// Used by [`InlineArray::as_type::<T>()`] to cast arrays by Rust type.
pub trait AsDtype {
    const DTYPE_ID: i32;
}

impl AsDtype for f32 {
    const DTYPE_ID: i32 = 10;
} // Float32
impl AsDtype for f16 {
    const DTYPE_ID: i32 = 9;
} // Float16 (using half::f16 or similar)
impl AsDtype for u8 {
    const DTYPE_ID: i32 = 1;
} // Uint8
impl AsDtype for u16 {
    const DTYPE_ID: i32 = 2;
} // Uint16
impl AsDtype for u32 {
    const DTYPE_ID: i32 = 3;
} // Uint32
impl AsDtype for u64 {
    const DTYPE_ID: i32 = 4;
} // Uint64
impl AsDtype for i8 {
    const DTYPE_ID: i32 = 5;
} // Int8
impl AsDtype for i16 {
    const DTYPE_ID: i32 = 6;
} // Int16
impl AsDtype for i32 {
    const DTYPE_ID: i32 = 7;
} // Int32
impl AsDtype for i64 {
    const DTYPE_ID: i32 = 8;
} // Int64
impl AsDtype for bool {
    const DTYPE_ID: i32 = 0;
} // Bool

/// Half-precision float marker type for `as_type::<f16>()`.
/// Use `half::f16` from the `half` crate, or this zero-sized stub.
#[allow(non_camel_case_types)]
pub struct f16;

/// Bfloat16 marker type for `as_type::<bf16>()`.
#[allow(non_camel_case_types)]
pub struct bf16;

impl AsDtype for bf16 {
    const DTYPE_ID: i32 = 11;
} // Bfloat16

// ── ArrayElement: trait for from_slice<T>() ──────────────────────────────

/// Trait for element types supported by [`InlineArray::from_slice`].
///
/// Implemented for `f32`, `i32`, `u32`, and `i64`.
pub trait ArrayElement {
    fn into_array(data: &[Self], shape: &[i32]) -> InlineArray
    where
        Self: Sized;
}

impl ArrayElement for f32 {
    fn into_array(data: &[f32], shape: &[i32]) -> InlineArray {
        InlineArray::from_f32_slice(data, shape)
    }
}

impl ArrayElement for i32 {
    fn into_array(data: &[i32], shape: &[i32]) -> InlineArray {
        InlineArray::from_i32_slice_shaped(data, shape)
    }
}

impl ArrayElement for u32 {
    fn into_array(data: &[u32], shape: &[i32]) -> InlineArray {
        InlineArray::from_u32_slice(data, shape)
    }
}

impl ArrayElement for i64 {
    fn into_array(data: &[i64], shape: &[i32]) -> InlineArray {
        let i32_data: Vec<i32> = data.iter().map(|&x| x as i32).collect();
        InlineArray::from_i32_slice_shaped(&i32_data, shape)
    }
}

impl ArrayElement for usize {
    fn into_array(data: &[usize], shape: &[i32]) -> InlineArray {
        let i32_data: Vec<i32> = data.iter().map(|&x| x as i32).collect();
        InlineArray::from_i32_slice_shaped(&i32_data, shape)
    }
}

// ── BridgeScalar: sealed trait for item<T>() ─────────────────────────────

/// Sealed trait for extracting a scalar value from an [`InlineArray`].
///
/// Only `f32` and `u32` are supported — they are the types the bridge FFI
/// exposes via `mlx_inline_item_f32` / `mlx_inline_item_u32`.
pub trait BridgeScalar: private::Sealed {
    fn extract(arr: &mut InlineArray) -> Self;
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
}

impl BridgeScalar for f32 {
    fn extract(arr: &mut InlineArray) -> f32 {
        arr.item_f32()
    }
}

impl BridgeScalar for u32 {
    fn extract(arr: &mut InlineArray) -> u32 {
        arr.item_u32()
    }
}

impl BridgeScalar for i32 {
    fn extract(arr: &mut InlineArray) -> i32 {
        arr.item_u32() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_slice_set_round_trip(mut base: InlineArray, value: InlineArray, expected: &[f32]) {
        let start = [0, 1];
        let stop = [2, 3];
        base = base.slice_set(&value, &start, &stop);
        let got = base.to_f32_vec(expected.len()).expect("to_f32_vec");
        assert_eq!(got, expected);
    }

    fn assert_tail_write_after_kv_cache_append(
        mut base: InlineArray,
        zeros: InlineArray,
        value: InlineArray,
        expected: &[f32],
    ) {
        base = base.kv_cache_append(&zeros, 2);
        let start = [0, 0, 3, 0];
        let stop = [1, 1, 4, 2];
        base = base.slice_set(&value, &start, &stop);
        let got = base.to_f32_vec(expected.len()).expect("to_f32_vec");
        assert_eq!(got, expected);
    }

    #[test]
    fn test_buffer_layout() {
        verify_buffer_layout();
    }

    #[test]
    fn test_scalar_roundtrip() {
        let mut a = InlineArray::from_f32(3.14);
        a.eval();
        let v = a.item_f32();
        assert!((v - 3.14).abs() < 1e-5, "got {v}");
    }

    #[test]
    fn test_add_scalars() {
        let a = InlineArray::from_f32(2.0);
        let b = InlineArray::from_f32(3.0);
        let mut c = a.add(&b);
        c.eval();
        let v = c.item_f32();
        assert!((v - 5.0).abs() < 1e-6, "expected 5.0, got {v}");
    }

    #[test]
    fn test_slice_set_f32() {
        let base = InlineArray::zeros(&[2, 4], crate::compat::Dtype::Float32.as_i32());
        let value = InlineArray::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_slice_set_round_trip(base, value, &[0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0]);
    }

    #[test]
    fn test_slice_set_u8() {
        let base = InlineArray::zeros(&[2, 4], crate::compat::Dtype::Uint8.as_i32());
        let value = InlineArray::from_u8_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_slice_set_round_trip(base, value, &[0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0]);
    }

    #[test]
    fn test_slice_set_u32() {
        let base = InlineArray::zeros(&[2, 4], crate::compat::Dtype::Uint32.as_i32());
        let value = InlineArray::from_u32_slice(&[1, 2, 3, 4], &[2, 2]);
        assert_slice_set_round_trip(base, value, &[0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0]);
    }

    #[test]
    fn test_tail_slice_set_after_kv_cache_append_f32() {
        let base = InlineArray::from_f32_slice(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0], &[1, 1, 3, 2]);
        let zeros = InlineArray::zeros(&[1, 1, 1, 2], crate::compat::Dtype::Float32.as_i32());
        let value = InlineArray::from_f32_slice(&[20.0, 21.0], &[1, 1, 1, 2]);
        assert_tail_write_after_kv_cache_append(
            base,
            zeros,
            value,
            &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0],
        );
    }

    #[test]
    fn test_tail_slice_set_after_kv_cache_append_u8() {
        let base = InlineArray::from_u8_slice(&[10, 11, 12, 13, 14, 15], &[1, 1, 3, 2]);
        let zeros = InlineArray::zeros(&[1, 1, 1, 2], crate::compat::Dtype::Uint8.as_i32());
        let value = InlineArray::from_u8_slice(&[20, 21], &[1, 1, 1, 2]);
        assert_tail_write_after_kv_cache_append(
            base,
            zeros,
            value,
            &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0],
        );
    }

    #[test]
    fn test_tail_slice_set_after_kv_cache_append_u32() {
        let base = InlineArray::from_u32_slice(&[10, 11, 12, 13, 14, 15], &[1, 1, 3, 2]);
        let zeros = InlineArray::zeros(&[1, 1, 1, 2], crate::compat::Dtype::Uint32.as_i32());
        let value = InlineArray::from_u32_slice(&[20, 21], &[1, 1, 1, 2]);
        assert_tail_write_after_kv_cache_append(
            base,
            zeros,
            value,
            &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0],
        );
    }
}
