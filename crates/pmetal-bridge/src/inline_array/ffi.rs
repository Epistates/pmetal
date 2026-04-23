//! Central FFI surface for the inline-array bridge.
//!
//! All `mlx_inline_*` extern declarations live here with `pub(super)` visibility
//! so each submodule can `use super::ffi::*;` to pick up exactly what it needs.

use super::RawBuf;

#[allow(dead_code)]
unsafe extern "C" {
    pub(super) fn mlx_inline_destroy(a: *mut RawBuf);
    pub(super) fn mlx_inline_init_copy(dst: *mut RawBuf, src: *const RawBuf);
    pub(super) fn mlx_inline_from_handle(dst: *mut RawBuf, handle_ctx: *mut std::ffi::c_void);
    pub(super) fn mlx_inline_to_handle(src: *const RawBuf) -> *mut std::ffi::c_void;

    pub(super) fn mlx_inline_matmul(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    pub(super) fn mlx_inline_add(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    pub(super) fn mlx_inline_multiply(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    pub(super) fn mlx_inline_subtract(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    pub(super) fn mlx_inline_divide(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    pub(super) fn mlx_inline_negative(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_exp(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_sigmoid(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_silu(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_softmax(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    pub(super) fn mlx_inline_sqrt(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_transpose(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_reshape(
        dst: *mut RawBuf,
        a: *const RawBuf,
        shape: *const i32,
        ndim: i32,
    );
    pub(super) fn mlx_inline_sum_axis(
        dst: *mut RawBuf,
        a: *const RawBuf,
        axis: i32,
        keepdims: bool,
    );
    pub(super) fn mlx_inline_astype(dst: *mut RawBuf, a: *const RawBuf, dtype: i32);

    pub(super) fn mlx_inline_gather_mm(
        dst: *mut RawBuf,
        a: *const RawBuf,
        b: *const RawBuf,
        lhs: *const RawBuf,
        rhs: *const RawBuf,
        sorted: bool,
    );
    pub(super) fn mlx_inline_rms_norm(
        dst: *mut RawBuf,
        x: *const RawBuf,
        w: *const RawBuf,
        eps: f32,
    );
    pub(super) fn mlx_inline_rope(
        dst: *mut RawBuf,
        x: *const RawBuf,
        dims: i32,
        trad: bool,
        base: f32,
        scale: f32,
        off: i32,
    );
    pub(super) fn mlx_inline_rope_with_freqs(
        dst: *mut RawBuf,
        x: *const RawBuf,
        dims: i32,
        trad: bool,
        scale: f32,
        off: i32,
        freqs: *const RawBuf,
    );
    pub(super) fn mlx_inline_rope_with_pos_ids(
        dst: *mut RawBuf,
        x: *const RawBuf,
        dims: i32,
        trad: bool,
        base: f32,
        scale: f32,
        offset_arr: *const RawBuf,
    );
    pub(super) fn mlx_inline_sdpa(
        dst: *mut RawBuf,
        q: *const RawBuf,
        k: *const RawBuf,
        v: *const RawBuf,
        scale: f32,
        mode: *const std::ffi::c_char,
    );

    pub(super) fn mlx_inline_split(
        input: *const RawBuf,
        indices: *const i32,
        n: i32,
        axis: i32,
        out: *mut RawBuf,
    );
    pub(super) fn mlx_inline_concatenate(
        dst: *mut RawBuf,
        arrays: *const RawBuf,
        num: i32,
        axis: i32,
    );
    pub(super) fn mlx_inline_argpartition(dst: *mut RawBuf, a: *const RawBuf, kth: i32, axis: i32);
    pub(super) fn mlx_inline_take_along_axis(
        dst: *mut RawBuf,
        a: *const RawBuf,
        idx: *const RawBuf,
        axis: i32,
    );

    pub(super) fn mlx_inline_eval(a: *mut RawBuf);
    pub(super) fn mlx_inline_async_eval(a: *mut RawBuf);
    pub(super) fn mlx_inline_from_f32(dst: *mut RawBuf, val: f32);
    pub(super) fn mlx_inline_from_i32(dst: *mut RawBuf, val: i32);

    pub(super) fn mlx_inline_ndim(a: *const RawBuf) -> i32;
    pub(super) fn mlx_inline_dim(a: *const RawBuf, axis: i32) -> i32;
    pub(super) fn mlx_inline_shape(a: *const RawBuf) -> *const i32;
    pub(super) fn mlx_inline_dtype(a: *const RawBuf) -> i32;
    pub(super) fn mlx_inline_item_f32(a: *mut RawBuf) -> f32;
    pub(super) fn mlx_inline_item_u32(a: *mut RawBuf) -> u32;

    pub(super) fn mlx_inline_sign(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_dequantize(
        dst: *mut RawBuf,
        w: *const RawBuf,
        scales: *const RawBuf,
        biases: *const RawBuf,
        group_size: i32,
        bits: i32,
    );
    pub(super) fn mlx_inline_from_f32_slice(
        dst: *mut RawBuf,
        data: *const f32,
        shape: *const i32,
        ndim: i32,
    );
    pub(super) fn mlx_inline_from_u32_slice(
        dst: *mut RawBuf,
        data: *const u32,
        shape: *const i32,
        ndim: i32,
    );
    pub(super) fn mlx_inline_from_u8_slice(
        dst: *mut RawBuf,
        data: *const u8,
        shape: *const i32,
        ndim: i32,
    );
    pub(super) fn mlx_inline_from_u16_bits_slice(
        dst: *mut RawBuf,
        data: *const u16,
        shape: *const i32,
        ndim: i32,
        dtype: i32,
    );
    pub(super) fn mlx_inline_to_f32_slice(a: *mut RawBuf, out: *mut f32, n: usize) -> i32;
    pub(super) fn mlx_inline_stack(dst: *mut RawBuf, arrays: *const RawBuf, num: i32, axis: i32);
    pub(super) fn mlx_inline_norm_l2(dst: *mut RawBuf, a: *const RawBuf, axis: i32, keepdims: bool);

    pub(super) fn mlx_inline_conv1d(
        dst: *mut RawBuf,
        input: *const RawBuf,
        weight: *const RawBuf,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
    );

    pub(super) fn mlx_inline_array_size() -> usize;
    pub(super) fn mlx_inline_array_align() -> usize;

    pub(super) fn mlx_inline_gdn_update(
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

    pub(super) fn mlx_inline_set_wired_limit(limit: usize) -> usize;
    pub(super) fn mlx_inline_get_max_recommended_size() -> usize;
    pub(super) fn mlx_inline_new_stream() -> i32;
    pub(super) fn mlx_inline_set_default_stream(index: i32);
    pub(super) fn mlx_inline_reset_default_stream();
    pub(super) fn mlx_inline_synchronize();
    pub(super) fn mlx_inline_clear_cache();
    pub(super) fn mlx_inline_enable_compile();
    pub(super) fn mlx_inline_disable_compile();
    pub(super) fn mlx_inline_graph_node_count(a: *const RawBuf) -> usize;
    pub(super) fn mlx_inline_graph_desc_count(a: *const RawBuf) -> usize;
    pub(super) fn mlx_inline_graph_dump(a: *const RawBuf);
    pub(super) fn mlx_inline_metal_start_capture(path: *const std::ffi::c_char) -> i32;
    pub(super) fn mlx_inline_metal_stop_capture();

    // Fixed-shape compiled GDN layer (shapeless=false, works with ALL primitives)
    pub(super) fn mlx_inline_compiled_gdn_layer_fixed(
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

    pub(super) fn mlx_inline_compiled_attn_layer_fixed(
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

    pub(super) fn mlx_inline_compiled_moe_layer_fixed(
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

    pub(super) fn mlx_inline_compiled_gemma4_attn_block(
        dst_out: *mut RawBuf,
        dst_cache_keys: *mut RawBuf,
        dst_cache_vals: *mut RawBuf,
        x: *const RawBuf,
        in_norm_w: *const RawBuf,
        q_w: *const RawBuf,
        k_w: *const RawBuf,
        v_w: *const RawBuf,
        o_w: *const RawBuf,
        q_norm_w: *const RawBuf,
        k_norm_w: *const RawBuf,
        post_norm_w: *const RawBuf,
        rope_freqs: *const RawBuf,
        cache_keys_in: *const RawBuf,
        cache_vals_in: *const RawBuf,
        kv_offset: i32,
        n_heads: i32,
        n_kv: i32,
        head_dim: i32,
        in_norm_eps: f32,
        qk_norm_eps: f32,
        post_norm_eps: f32,
        sliding_window: i32,
        use_k_eq_v: bool,
        rope_base: f32,
        rope_dims: i32,
    );

    pub(super) fn mlx_inline_compiled_gemma4_shared_attn_decode(
        dst_out: *mut RawBuf,
        x: *const RawBuf,
        in_norm_w: *const RawBuf,
        q_w: *const RawBuf,
        o_w: *const RawBuf,
        q_norm_w: *const RawBuf,
        post_norm_w: *const RawBuf,
        rope_freqs: *const RawBuf,
        cache_keys_in: *const RawBuf,
        cache_vals_in: *const RawBuf,
        valid_kv_len: i32,
        rope_offset: i32,
        n_heads: i32,
        n_kv: i32,
        head_dim: i32,
        in_norm_eps: f32,
        q_norm_eps: f32,
        post_norm_eps: f32,
        sliding_window: i32,
        rope_base: f32,
        rope_dims: i32,
    );

    pub(super) fn mlx_inline_compiled_gemma4_mlp_block(
        dst_out: *mut RawBuf,
        x: *const RawBuf,
        pre_norm_w: *const RawBuf,
        gate_w: *const RawBuf,
        up_w: *const RawBuf,
        down_w: *const RawBuf,
        post_norm_w: *const RawBuf,
        pre_norm_eps: f32,
        post_norm_eps: f32,
    );

    pub(super) fn mlx_inline_compiled_gemma4_per_layer_input_block(
        dst_out: *mut RawBuf,
        x: *const RawBuf,
        layer_input: *const RawBuf,
        gate_w: *const RawBuf,
        projection_w: *const RawBuf,
        post_norm_w: *const RawBuf,
        post_norm_eps: f32,
    );

    // Arange — non-broadcast tensor creation
    pub(super) fn mlx_inline_arange(dst: *mut RawBuf, n: i32, dtype: i32);
    pub(super) fn mlx_inline_load_safetensors_key(
        dst: *mut RawBuf,
        path: *const std::ffi::c_char,
        key: *const std::ffi::c_char,
    ) -> i32;

    // Graph detach — severs computation graph references
    pub(super) fn mlx_inline_detach(a: *mut RawBuf);

    // Batch eval — single GPU submission for multiple arrays
    pub(super) fn mlx_inline_eval_many(arrays: *mut *mut RawBuf, count: i32);
    pub(super) fn mlx_inline_async_eval_many(arrays: *mut *mut RawBuf, count: i32);

    // Metal memory instrumentation
    pub(super) fn mlx_inline_get_active_memory() -> usize;
    pub(super) fn mlx_inline_get_cache_memory() -> usize;
    pub(super) fn mlx_inline_get_peak_memory() -> usize;
    pub(super) fn mlx_inline_reset_peak_memory();

    // ── FFT ops ──
    pub(super) fn mlx_inline_rfft(dst: *mut RawBuf, a: *const RawBuf, n_fft: i32, axis: i32);
    pub(super) fn mlx_inline_irfft(dst: *mut RawBuf, a: *const RawBuf, n_fft: i32, axis: i32);

    // ── leaky_relu ──
    pub(super) fn mlx_inline_leaky_relu(dst: *mut RawBuf, a: *const RawBuf, neg_slope: f32);

    // ── squeeze_all (remove all size-1 axes) ──
    pub(super) fn mlx_inline_squeeze_all(dst: *mut RawBuf, a: *const RawBuf);

    // ── pad ──
    pub(super) fn mlx_inline_pad(
        dst: *mut RawBuf,
        a: *const RawBuf,
        pad_widths: *const i32,
        ndim: i32,
        fill_value: f32,
    );

    // ── Additional ops for complete model inference ──
    pub(super) fn mlx_inline_concatenate_2(
        dst: *mut RawBuf,
        a: *const RawBuf,
        b: *const RawBuf,
        axis: i32,
    );
    pub(super) fn mlx_inline_softplus(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_where(
        dst: *mut RawBuf,
        cond: *const RawBuf,
        a: *const RawBuf,
        b: *const RawBuf,
    );
    pub(super) fn mlx_inline_maximum(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    pub(super) fn mlx_inline_zeros(dst: *mut RawBuf, shape: *const i32, ndim: i32, dtype: i32);
    pub(super) fn mlx_inline_ones(dst: *mut RawBuf, shape: *const i32, ndim: i32, dtype: i32);
    pub(super) fn mlx_inline_slice(
        dst: *mut RawBuf,
        a: *const RawBuf,
        start: *const i32,
        stop: *const i32,
        ndim: i32,
    );
    pub(super) fn mlx_inline_slice_set(
        dst: *mut RawBuf,
        a: *const RawBuf,
        val: *const RawBuf,
        start: *const i32,
        stop: *const i32,
        ndim: i32,
    );
    pub(super) fn mlx_inline_repeat(dst: *mut RawBuf, a: *const RawBuf, repeats: i32, axis: i32);
    pub(super) fn mlx_inline_squeeze(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    pub(super) fn mlx_inline_expand_dims(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    pub(super) fn mlx_inline_transpose_axes(
        dst: *mut RawBuf,
        a: *const RawBuf,
        axes: *const i32,
        ndim: i32,
    );
    pub(super) fn mlx_inline_cumsum(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    pub(super) fn mlx_inline_log(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_tril(dst: *mut RawBuf, a: *const RawBuf, k: i32);
    pub(super) fn mlx_inline_index(dst: *mut RawBuf, a: *const RawBuf, indices: *const RawBuf);
    pub(super) fn mlx_inline_softmax_precise(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    pub(super) fn mlx_inline_sdpa_with_mask(
        dst: *mut RawBuf,
        q: *const RawBuf,
        k: *const RawBuf,
        v: *const RawBuf,
        scale: f32,
        mask: *const RawBuf,
    );
    pub(super) fn mlx_inline_eval_2(a: *mut RawBuf, b: *mut RawBuf);
    pub(super) fn mlx_inline_quantized_matmul(
        dst: *mut RawBuf,
        x: *const RawBuf,
        w: *const RawBuf,
        scales: *const RawBuf,
        biases: *const RawBuf,
        transpose: bool,
        group_size: i32,
        bits: i32,
    );
    pub(super) fn mlx_inline_gather_qmm(
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
    pub(super) fn mlx_inline_argmax(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    pub(super) fn mlx_inline_argmin(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    pub(super) fn mlx_inline_logsumexp(
        dst: *mut RawBuf,
        a: *const RawBuf,
        axis: i32,
        keepdims: bool,
    );
    pub(super) fn mlx_inline_categorical(dst: *mut RawBuf, logits: *const RawBuf);

    // ── Element-wise math ──
    pub(super) fn mlx_inline_abs(dst: *mut RawBuf, a: *const RawBuf);

    // ── Embedding / KV cache ──
    pub(super) fn mlx_inline_take_axis(
        dst: *mut RawBuf,
        a: *const RawBuf,
        indices: *const RawBuf,
        axis: i32,
    );
    pub(super) fn mlx_inline_kv_cache_append(
        dst: *mut RawBuf,
        cached: *const RawBuf,
        new_kv: *const RawBuf,
        axis: i32,
    );
    pub(super) fn mlx_inline_async_eval_arr(a: *const RawBuf);

    // ── GDN Metal kernel step with pre-computed g/beta ──
    pub(super) fn mlx_inline_gdn_metal_step(
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

    // ── GDN state-only advance for speculative-decoding rollback replay ──
    //
    // Same contract as `mlx_inline_gdn_metal_step` minus the `q`/`y` channels:
    // advances the recurrent state through `t` tokens without computing the
    // attention output. Roughly 2× faster per step than dispatching the
    // full step kernel and discarding its output.
    pub(super) fn mlx_inline_gdn_metal_state_update(
        dst_state: *mut RawBuf,
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
    pub(super) fn mlx_inline_turboquant_encode(
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
    pub(super) fn mlx_inline_turboquant_decode(
        out: *mut RawBuf,
        indices: *const RawBuf,
        norms: *const RawBuf, // reserved — pass std::ptr::null_mut()
        codebook: *const RawBuf,
        dim: u32,
        n_centroids: u32,
        n_rows: u32,
    ) -> i32;

    pub(super) fn mlx_inline_turboquant_score(
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

    pub(super) fn mlx_inline_turboquant_score_q8_d256(
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

    pub(super) fn mlx_inline_turboquant_mixed_score(
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

    pub(super) fn mlx_inline_turboquant_pack_sign_bits(
        out: *mut RawBuf,
        projected: *const RawBuf,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
    ) -> i32;

    pub(super) fn mlx_inline_turboquant_pack_q8_keybytes(
        out: *mut RawBuf,
        indices: *const RawBuf,
        qjl_signs: *const RawBuf,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
        cache_seq_capacity: u32,
    ) -> i32;

    pub(super) fn mlx_inline_turboquant_pack_q8_keybytes_seq(
        out: *mut RawBuf,
        indices: *const RawBuf,
        qjl_signs: *const RawBuf,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
        cache_seq_capacity: u32,
    ) -> i32;

    pub(super) fn mlx_inline_turboquant_pack_q8_kvbytes_seq(
        out: *mut RawBuf,
        indices: *const RawBuf,
        qjl_signs: *const RawBuf,
        value_indices: *const RawBuf,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
        cache_seq_capacity: u32,
    ) -> i32;

    pub(super) fn mlx_inline_turboquant_unpack_sign_bits(
        out: *mut RawBuf,
        packed: *const RawBuf,
        dim: u32,
        packed_dim: u32,
        n_rows: u32,
    ) -> i32;

    pub(super) fn mlx_inline_turboquant_signed_fwht_256_rows(
        out: *mut RawBuf,
        input: *const RawBuf,
        left_signs: *const RawBuf,
        right_signs: *const RawBuf,
        n_rows: u32,
    ) -> i32;

    pub(super) fn mlx_inline_turboquant_weighted_decode(
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

    pub(super) fn mlx_inline_turboquant_attention_q8_d256_2pass(
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

    pub(super) fn mlx_inline_turboquant_attention_q8_d256_packed_keys_2pass(
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

    pub(super) fn mlx_inline_turboquant_attention_q8_d256_packed_keys_dense_values_2pass(
        out: *mut RawBuf,
        query_rot: *const RawBuf,
        query_proj: *const RawBuf,
        key_bytes: *const RawBuf,
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

    pub(super) fn mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass(
        out: *mut RawBuf,
        query_rot: *const RawBuf,
        key_indices: *const RawBuf,
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

    pub(super) fn mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_state(
        out_partials: *mut RawBuf,
        out_sums: *mut RawBuf,
        out_maxs: *mut RawBuf,
        query_rot: *const RawBuf,
        key_indices: *const RawBuf,
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

    pub(super) fn mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_pass1(
        out: *mut RawBuf,
        query_rot: *const RawBuf,
        key_indices: *const RawBuf,
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

    pub(super) fn mlx_inline_turboquant_attention_q8_d256_pass2_merge(
        out: *mut RawBuf,
        partials: *const RawBuf,
        sums: *const RawBuf,
        maxs: *const RawBuf,
        n_rows: u32,
        blocks: u32,
    ) -> i32;

    pub(super) fn mlx_inline_turboquant_attention_q8_d256_fullbyte_dense_values_2pass_localsoftmax(
        out: *mut RawBuf,
        query_rot: *const RawBuf,
        key_indices: *const RawBuf,
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

    pub(super) fn mlx_inline_turboquant_score_q8_d256_fullbyte(
        out: *mut RawBuf,
        query_rot: *const RawBuf,
        key_indices: *const RawBuf,
        slot_scales: *const RawBuf,
        key_codebook: *const RawBuf,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
        attn_scale_bits: u32,
    ) -> i32;

    pub(super) fn mlx_inline_turboquant_weighted_sum_d256_dense_values(
        out: *mut RawBuf,
        weights: *const RawBuf,
        value_dense: *const RawBuf,
        n_rows: u32,
        n_seq: u32,
        cache_seq_capacity: u32,
        q_heads: u32,
        kv_heads: u32,
    ) -> i32;

    pub(super) fn mlx_inline_turboquant_attention_q8_d256_packed_kv_2pass(
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

    pub(super) fn mlx_inline_turboquant_attention_q8_d256_packed_kv_dense_values_2pass(
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

    pub(super) fn mlx_inline_turboquant_attention_q8_d128_2pass(
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

    pub(super) fn mlx_inline_turboquant_attention_q8_d128_packed_keys_2pass(
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

    pub(super) fn mlx_inline_turboquant_gather_last_dim(
        out: *mut RawBuf,
        input: *const RawBuf,
        positions: *const RawBuf,
        full_dim: u32,
        out_dim: u32,
        n_rows: u32,
    ) -> i32;

    pub(super) fn mlx_inline_turboquant_scatter_last_dim(
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
    pub(super) fn mlx_inline_random_normal(
        dst: *mut RawBuf,
        shape: *const i32,
        ndim: i32,
        dtype: i32,
    );
    pub(super) fn mlx_inline_random_uniform(
        dst: *mut RawBuf,
        shape: *const i32,
        ndim: i32,
        dtype: i32,
    );
    pub(super) fn mlx_inline_random_bernoulli(
        dst: *mut RawBuf,
        p: *const RawBuf,
        shape: *const i32,
        ndim: i32,
    );
    pub(super) fn mlx_inline_random_seed(seed: u64);
    pub(super) fn mlx_inline_random_randint(
        dst: *mut RawBuf,
        low: i32,
        high: i32,
        shape: *const i32,
        ndim: i32,
        dtype: i32,
    );

    // ── Training ops: math ──
    pub(super) fn mlx_inline_mean_axis(
        dst: *mut RawBuf,
        a: *const RawBuf,
        axis: i32,
        keepdims: bool,
    );
    pub(super) fn mlx_inline_mean_all(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_pow(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    pub(super) fn mlx_inline_reciprocal(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_sin(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_cos(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_clip(
        dst: *mut RawBuf,
        a: *const RawBuf,
        lo: *const RawBuf,
        hi: *const RawBuf,
    );
    pub(super) fn mlx_inline_log_softmax(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    pub(super) fn mlx_inline_cross_entropy(
        dst: *mut RawBuf,
        logits: *const RawBuf,
        targets: *const RawBuf,
        axis: i32,
    );
    pub(super) fn mlx_inline_cross_entropy_sparse(
        dst: *mut RawBuf,
        logits: *const RawBuf,
        indices: *const RawBuf,
        axis: i32,
    );
    pub(super) fn mlx_inline_square(dst: *mut RawBuf, a: *const RawBuf);

    // ── Training ops: creation ──
    pub(super) fn mlx_inline_full(
        dst: *mut RawBuf,
        shape: *const i32,
        ndim: i32,
        val: f32,
        dtype: i32,
    );
    pub(super) fn mlx_inline_eye(dst: *mut RawBuf, n: i32, dtype: i32);
    pub(super) fn mlx_inline_tri(dst: *mut RawBuf, n: i32, m: i32, k: i32, dtype: i32);

    // ── Training ops: shape ──
    pub(super) fn mlx_inline_broadcast_to(
        dst: *mut RawBuf,
        a: *const RawBuf,
        shape: *const i32,
        ndim: i32,
    );
    pub(super) fn mlx_inline_flatten(
        dst: *mut RawBuf,
        a: *const RawBuf,
        start_axis: i32,
        end_axis: i32,
    );

    // ── Training ops: sort/reduction ──
    pub(super) fn mlx_inline_argsort(dst: *mut RawBuf, a: *const RawBuf, axis: i32);
    pub(super) fn mlx_inline_sum_all(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_max_axis(
        dst: *mut RawBuf,
        a: *const RawBuf,
        axis: i32,
        keepdims: bool,
    );
    pub(super) fn mlx_inline_min_axis(
        dst: *mut RawBuf,
        a: *const RawBuf,
        axis: i32,
        keepdims: bool,
    );
    pub(super) fn mlx_inline_minimum(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);

    // ── Training ops: activation ──
    pub(super) fn mlx_inline_relu(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_gelu(dst: *mut RawBuf, a: *const RawBuf);

    // ── Training ops: comparison ──
    pub(super) fn mlx_inline_equal(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    pub(super) fn mlx_inline_not_equal(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    pub(super) fn mlx_inline_greater(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    pub(super) fn mlx_inline_less(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    pub(super) fn mlx_inline_greater_equal(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);
    pub(super) fn mlx_inline_less_equal(dst: *mut RawBuf, a: *const RawBuf, b: *const RawBuf);

    // ── Training ops: serialization ──
    pub(super) fn mlx_inline_save_safetensors(
        path: *const std::ffi::c_char,
        keys: *const *const std::ffi::c_char,
        arrays: *const RawBuf,
        count: i32,
    );

    // ── Training ops: quantize ──
    pub(super) fn mlx_inline_quantize(
        dst_w: *mut RawBuf,
        dst_scales: *mut RawBuf,
        dst_biases: *mut RawBuf,
        a: *const RawBuf,
        group_size: i32,
        bits: i32,
    );

    // ── Training ops: multi-axis ──
    pub(super) fn mlx_inline_sum_axes(
        dst: *mut RawBuf,
        a: *const RawBuf,
        axes: *const i32,
        num_axes: i32,
        keepdims: bool,
    );
    pub(super) fn mlx_inline_mean_axes(
        dst: *mut RawBuf,
        a: *const RawBuf,
        axes: *const i32,
        num_axes: i32,
        keepdims: bool,
    );

    // ── Training ops: misc ──
    pub(super) fn mlx_inline_size(a: *const RawBuf) -> usize;
    pub(super) fn mlx_inline_nbytes(a: *const RawBuf) -> usize;
    pub(super) fn mlx_inline_data_ptr(
        a: *const RawBuf,
        out_ptr: *mut *const std::ffi::c_void,
    ) -> i32;
    pub(super) fn mlx_inline_array_id(a: *const RawBuf) -> usize;
    pub(super) fn mlx_inline_stop_gradient(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_tri_inv(
        dst: *mut RawBuf,
        a: *const RawBuf,
        upper: bool,
        use_cpu: bool,
    );

    // ── Autograd: value_and_grad ──
    pub(super) fn mlx_inline_value_and_grad(
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

    // ── Gradient checkpointing ──
    // Wraps `forward_fn` with mlx::core::checkpoint() so that activations
    // are discarded after the forward pass and recomputed during backward.
    // `n_outputs_max` is the capacity of `dst_outputs`; the actual count
    // produced is written into `*n_outputs_written`.
    pub(super) fn mlx_inline_checkpoint(
        forward_fn: unsafe extern "C" fn(
            *const *const RawBuf, // all_arrays
            i32,                  // n_total
            *mut RawBuf,          // outputs_out  (flat array, capacity = n_outputs_max)
            *mut i32,             // n_outputs_out (written by callback)
            *mut std::ffi::c_void,
        ),
        ctx: *mut std::ffi::c_void,
        all_arrays: *const *const RawBuf,
        n_total: i32,
        n_outputs_max: i32,
        dst_outputs: *mut RawBuf,
        n_outputs_written: *mut i32,
    );

    // ── Fused compiled ops (match Python's @mx.compile) ──
    pub(super) fn mlx_inline_fused_swiglu(dst: *mut RawBuf, gate: *const RawBuf, up: *const RawBuf);
    pub(super) fn mlx_inline_fused_geglu_tanh(
        dst: *mut RawBuf,
        gate: *const RawBuf,
        up: *const RawBuf,
    );
    pub(super) fn mlx_inline_fused_silu(dst: *mut RawBuf, x: *const RawBuf);
    pub(super) fn mlx_inline_fused_compute_g(
        dst: *mut RawBuf,
        a_log: *const RawBuf,
        a: *const RawBuf,
        dt_bias: *const RawBuf,
    );
    pub(super) fn mlx_inline_fused_precise_swiglu(
        dst: *mut RawBuf,
        x: *const RawBuf,
        gate: *const RawBuf,
    );

    // Batch safetensors load — parses the file once and fills caller-provided buffers.
    // Returns number of entries written, or -1 on error.
    pub(super) fn mlx_inline_load_safetensors_all(
        path: *const std::ffi::c_char,
        key_buf: *mut *mut std::ffi::c_char,
        arr_buf: *mut RawBuf,
        max_entries: i32,
    ) -> i32;

    // Free key strings allocated by mlx_inline_load_safetensors_all.
    pub(super) fn mlx_inline_free_key_strings(keys: *mut *mut std::ffi::c_char, count: i32);

    // Create a 1-D int32 array from a Rust slice.
    pub(super) fn mlx_inline_from_i32_slice(dst: *mut RawBuf, data: *const i32, len: i32);

    // ── Linalg: SVD ──
    pub(super) fn mlx_inline_svd(
        dst_u: *mut RawBuf,
        dst_s: *mut RawBuf,
        dst_vt: *mut RawBuf,
        a: *const RawBuf,
    );

    // ── Missing ops for pmetal-models migration ──
    pub(super) fn mlx_inline_rsqrt(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_zeros_like(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_ones_like(dst: *mut RawBuf, a: *const RawBuf);
    pub(super) fn mlx_inline_tile(dst: *mut RawBuf, a: *const RawBuf, reps: *const i32, ndim: i32);
    pub(super) fn mlx_inline_linspace(dst: *mut RawBuf, start: f32, stop: f32, n: i32, dtype: i32);
    pub(super) fn mlx_inline_split_sections(
        dst_arr: *mut RawBuf,
        a: *const RawBuf,
        sections: i32,
        axis: i32,
        out_count: *mut i32,
    );
    pub(super) fn mlx_inline_scatter_add(
        dst: *mut RawBuf,
        a: *const RawBuf,
        indices: *const RawBuf,
        updates: *const RawBuf,
        axis: i32,
    );
    pub(super) fn mlx_inline_topk(dst: *mut RawBuf, a: *const RawBuf, k: i32, axis: i32);
    pub(super) fn mlx_inline_put_along_axis(
        dst: *mut RawBuf,
        a: *const RawBuf,
        indices: *const RawBuf,
        values: *const RawBuf,
        axis: i32,
    );
    pub(super) fn mlx_inline_layer_norm(
        dst: *mut RawBuf,
        x: *const RawBuf,
        weight: *const RawBuf,
        bias: *const RawBuf,
        eps: f32,
    );
    pub(super) fn mlx_inline_addmm(
        dst: *mut RawBuf,
        c: *const RawBuf,
        a: *const RawBuf,
        b: *const RawBuf,
    );
    pub(super) fn mlx_inline_conv2d(
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
    pub(super) fn mlx_inline_qwen35_decode_step(
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
