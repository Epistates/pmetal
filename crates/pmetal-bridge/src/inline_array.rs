//! Zero-allocation MLX array — stores `mlx::core::array` inline on the Rust stack.
//!
//! This eliminates ALL per-op heap allocation, matching Python/nanobind's direct
//! C++ binding performance. Each op is a single `extern "C"` call with placement-new
//! into a caller-provided buffer.

use std::mem::MaybeUninit;

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
    fn mlx_inline_rms_norm(
        dst: *mut RawBuf,
        x: *const RawBuf,
        w: *const RawBuf,
        eps: f32,
    );
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
    fn mlx_inline_from_f32_slice(dst: *mut RawBuf, data: *const f32, shape: *const i32, ndim: i32);
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
        dst_out: *mut RawBuf, dst_conv: *mut RawBuf, dst_ssm: *mut RawBuf,
        normed: *const RawBuf,
        qkv_w: *const RawBuf, z_w: *const RawBuf,
        b_w: *const RawBuf, a_w: *const RawBuf,
        conv_w: *const RawBuf,
        q_nw: *const RawBuf, k_nw: *const RawBuf,
        a_log: *const RawBuf, dt_bias: *const RawBuf,
        norm_w: *const RawBuf, out_w: *const RawBuf,
        conv_state: *const RawBuf, ssm_state: *const RawBuf,
        nv: i32, nk: i32, dk: i32, dv: i32, cd: i32, ck: i32, kd: i32, norm_eps: f32,
    );

    // Fixed-shape compiled GDN layer (shapeless=false, works with ALL primitives)
    fn mlx_inline_compiled_gdn_layer_fixed(
        dst_out: *mut RawBuf, dst_conv: *mut RawBuf, dst_ssm: *mut RawBuf,
        normed: *const RawBuf,
        qkv_w: *const RawBuf, z_w: *const RawBuf,
        b_w: *const RawBuf, a_w: *const RawBuf,
        conv_w: *const RawBuf,
        q_nw: *const RawBuf, k_nw: *const RawBuf,
        a_log: *const RawBuf, dt_bias: *const RawBuf,
        norm_w: *const RawBuf, out_w: *const RawBuf,
        conv_state: *const RawBuf, ssm_state: *const RawBuf,
        nv: i32, nk: i32, dk: i32, dv: i32, cd: i32, ck: i32, kd: i32, norm_eps: f32,
    );

    // Arange — non-broadcast tensor creation
    fn mlx_inline_arange(dst: *mut RawBuf, n: i32, dtype: i32);
    fn mlx_inline_load_safetensors_key(dst: *mut RawBuf, path: *const std::ffi::c_char, key: *const std::ffi::c_char) -> i32;

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

    // ── Additional ops for complete model inference ──
    fn mlx_inline_concatenate_2(
        dst: *mut RawBuf,
        a: *const RawBuf,
        b: *const RawBuf,
        axis: i32,
    );
    fn mlx_inline_softplus(dst: *mut RawBuf, a: *const RawBuf);
    fn mlx_inline_where(
        dst: *mut RawBuf,
        cond: *const RawBuf,
        a: *const RawBuf,
        b: *const RawBuf,
    );
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
    fn mlx_inline_transpose_axes(
        dst: *mut RawBuf,
        a: *const RawBuf,
        axes: *const i32,
        ndim: i32,
    );
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
    fn mlx_inline_kv_cache_append(dst: *mut RawBuf, cached: *const RawBuf, new_kv: *const RawBuf, axis: i32);
    fn mlx_inline_async_eval_arr(a: *const RawBuf);

    // ── GDN Metal kernel step with pre-computed g/beta ──
    fn mlx_inline_gdn_metal_step(
        dst_y: *mut RawBuf, dst_state: *mut RawBuf,
        q: *const RawBuf, k: *const RawBuf, v: *const RawBuf,
        g: *const RawBuf, beta: *const RawBuf,
        state_in: *const RawBuf, t: i32,
    );

    // ── Fused compiled ops (match Python's @mx.compile) ──
    fn mlx_inline_fused_swiglu(dst: *mut RawBuf, gate: *const RawBuf, up: *const RawBuf);
    fn mlx_inline_fused_silu(dst: *mut RawBuf, x: *const RawBuf);
    fn mlx_inline_fused_compute_g(dst: *mut RawBuf, a_log: *const RawBuf, a: *const RawBuf, dt_bias: *const RawBuf);
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

    // ── Full Qwen3.5 forward pass — single C++ function, zero FFI overhead ──
    // See bridge.h for the complete weight/cache/config layout documentation.
    fn mlx_inline_qwen35_decode_step(
        dst_logits:       *mut RawBuf,
        token_ids:        *const RawBuf,
        weight_ptrs:      *const *const RawBuf,
        num_weights:      i32,
        cache_ptrs:       *mut *mut RawBuf,
        num_cache:        i32,
        attn_kv_offsets:  *mut i32,
        rope_offset:      *mut i32,
        config_ints:      *const i32,
        num_config_ints:  i32,
        config_floats:    *const f32,
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
    unsafe { mlx_inline_new_stream(); }
}

/// Set the generation stream as the default stream for all ops.
pub fn set_generation_stream() {
    unsafe { mlx_inline_set_default_stream(0); }
}

/// Synchronize the generation stream (wait for all pending GPU work).
pub fn synchronize() {
    unsafe { mlx_inline_synchronize(); }
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
    if arrays.is_empty() { return; }
    let mut ptrs: Vec<*mut RawBuf> = arrays.iter_mut().map(|a| &mut a.raw as *mut RawBuf).collect();
    unsafe { mlx_inline_eval_many(ptrs.as_mut_ptr(), ptrs.len() as i32); }
    for a in arrays.iter_mut() {
        unsafe { mlx_inline_detach(&mut a.raw); }
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
        // Error: no arrays were placement-new'd, nothing to destroy.
        return None;
    }

    let count = count as usize;

    // Convert the count valid slots into owned InlineArrays + String keys.
    // We must adopt each initialised array slot so its destructor runs on drop.
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        // SAFETY: C++ placement-new'd into slots [0, count).
        let array = InlineArray { raw: unsafe { arr_slots[i].assume_init() } };

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

/// Thin wrapper around libc free so we can call it without a libc dependency.
/// `strdup` allocates with the C allocator; we must free with the same.
unsafe fn libc_free(ptr: *mut std::ffi::c_void) {
    unsafe extern "C" {
        fn free(ptr: *mut std::ffi::c_void);
    }
    unsafe { free(ptr) }
}

// ── InlineArray ───────────────────────────────────────────────────────────

/// Stack-allocated MLX array. Zero heap allocation per op.
pub struct InlineArray {
    raw: RawBuf,
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
        write!(f, "InlineArray(ndim={}, shape={:?})", self.ndim(), self.shape())
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

    /// L2 norm along an axis.
    pub fn norm_l2(&self, axis: i32, keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_norm_l2(dst.as_mut_ptr(), &self.raw, axis, keepdims);
            Self { raw: dst.assume_init() }
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

    pub fn rope(
        &self,
        dims: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset: i32,
    ) -> Self {
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
            mlx_inline_sdpa_with_mask(
                dst.as_mut_ptr(),
                &self.raw,
                &k.raw,
                &v.raw,
                scale,
                mask_ptr,
            );
            Self {
                raw: dst.assume_init(),
            }
        }
    }

    pub fn split(&self, indices: &[i32], axis: i32) -> Vec<Self> {
        let n = indices.len() + 1;
        let mut bufs: Vec<MaybeUninit<RawBuf>> =
            (0..n).map(|_| MaybeUninit::uninit()).collect();
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

    // ── Eval ─────────────────────────────────────────────────────────────

    pub fn eval(&mut self) {
        unsafe { mlx_inline_eval(&mut self.raw) }
    }
    pub fn async_eval(&mut self) {
        unsafe { mlx_inline_async_eval(&mut self.raw) }
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
        qkv_w: &Self, z_w: &Self, b_w: &Self, a_w: &Self,
        conv_w: &Self,
        q_nw: &Self, k_nw: &Self, a_log: &Self, dt_bias: &Self,
        norm_w: &Self, out_w: &Self, conv_state: &Self, ssm_state: &Self,
        nv: i32, nk: i32, dk: i32, dv: i32, cd: i32, ck: i32, kd: i32, norm_eps: f32,
    ) -> (Self, Self, Self) {
        let mut out = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut conv = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut ssm = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_compiled_gdn_layer(
                out.as_mut_ptr(), conv.as_mut_ptr(), ssm.as_mut_ptr(),
                &normed.raw,
                &qkv_w.raw, &z_w.raw, &b_w.raw, &a_w.raw,
                &conv_w.raw,
                &q_nw.raw, &k_nw.raw, &a_log.raw, &dt_bias.raw,
                &norm_w.raw, &out_w.raw, &conv_state.raw, &ssm_state.raw,
                nv, nk, dk, dv, cd, ck, kd, norm_eps,
            );
            (Self { raw: out.assume_init() }, Self { raw: conv.assume_init() }, Self { raw: ssm.assume_init() })
        }
    }

    /// Fixed-shape compiled GDN layer (shapeless=false).
    /// Works with ALL primitives. Traces on first T=1 call, replays tape on subsequent.
    /// Eliminates graph traversal overhead for ~10ms savings per step.
    #[allow(clippy::too_many_arguments)]
    pub fn compiled_gdn_layer_fixed(
        normed: &Self,
        qkv_w: &Self, z_w: &Self, b_w: &Self, a_w: &Self,
        conv_w: &Self,
        q_nw: &Self, k_nw: &Self, a_log: &Self, dt_bias: &Self,
        norm_w: &Self, out_w: &Self, conv_state: &Self, ssm_state: &Self,
        nv: i32, nk: i32, dk: i32, dv: i32, cd: i32, ck: i32, kd: i32, norm_eps: f32,
    ) -> (Self, Self, Self) {
        let mut out = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut conv = std::mem::MaybeUninit::<RawBuf>::uninit();
        let mut ssm = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_compiled_gdn_layer_fixed(
                out.as_mut_ptr(), conv.as_mut_ptr(), ssm.as_mut_ptr(),
                &normed.raw,
                &qkv_w.raw, &z_w.raw, &b_w.raw, &a_w.raw,
                &conv_w.raw,
                &q_nw.raw, &k_nw.raw, &a_log.raw, &dt_bias.raw,
                &norm_w.raw, &out_w.raw, &conv_state.raw, &ssm_state.raw,
                nv, nk, dk, dv, cd, ck, kd, norm_eps,
            );
            (Self { raw: out.assume_init() }, Self { raw: conv.assume_init() }, Self { raw: ssm.assume_init() })
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
            if mlx_inline_load_safetensors_key(dst.as_mut_ptr(), c_path.as_ptr(), c_key.as_ptr()) == 0 {
                Some(Self { raw: dst.assume_init() })
            } else {
                None
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
            Self { raw: dst.assume_init() }
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
            Self { raw: dst.assume_init() }
        }
    }

    /// Create a range [0, 1, ..., n-1] with full Metal buffer (no broadcast).
    /// Useful for benchmarks — ensures matmuls read real data from GPU memory.
    pub fn arange(n: i32, dtype: i32) -> Self {
        let mut dst = std::mem::MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_arange(dst.as_mut_ptr(), n, dtype);
            Self { raw: dst.assume_init() }
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
        unsafe { mlx_inline_argmax(dst.as_mut_ptr(), &self.raw, axis); Self { raw: dst.assume_init() } }
    }

    #[inline]
    pub fn argmin(&self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe { mlx_inline_argmin(dst.as_mut_ptr(), &self.raw, axis); Self { raw: dst.assume_init() } }
    }

    /// Element-wise absolute value.
    #[inline]
    pub fn abs_val(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe { mlx_inline_abs(dst.as_mut_ptr(), &self.raw); Self { raw: dst.assume_init() } }
    }

    #[inline]
    pub fn logsumexp(&self, axis: i32, keepdims: bool) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe { mlx_inline_logsumexp(dst.as_mut_ptr(), &self.raw, axis, keepdims); Self { raw: dst.assume_init() } }
    }

    #[inline]
    pub fn categorical(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe { mlx_inline_categorical(dst.as_mut_ptr(), &self.raw); Self { raw: dst.assume_init() } }
    }

    // ── Embedding / KV cache ────────────────────────────────────────────

    /// Take rows along axis (embedding lookup: `take(weight, indices, axis=0)`).
    #[inline]
    pub fn take_axis(&self, indices: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe { mlx_inline_take_axis(dst.as_mut_ptr(), &self.raw, &indices.raw, axis); Self { raw: dst.assume_init() } }
    }

    /// Concatenate cached and new K/V along the sequence axis.
    #[inline]
    pub fn kv_cache_append(&self, new_kv: &Self, axis: i32) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe { mlx_inline_kv_cache_append(dst.as_mut_ptr(), &self.raw, &new_kv.raw, axis); Self { raw: dst.assume_init() } }
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
        q: &Self, k: &Self, v: &Self,
        g: &Self, beta: &Self,
        state: &Self, t: i32,
    ) -> (Self, Self) {
        let mut dst_y = MaybeUninit::<RawBuf>::uninit();
        let mut dst_state = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_gdn_metal_step(
                dst_y.as_mut_ptr(), dst_state.as_mut_ptr(),
                &q.raw, &k.raw, &v.raw,
                &g.raw, &beta.raw,
                &state.raw, t,
            );
            (Self { raw: dst_y.assume_init() }, Self { raw: dst_state.assume_init() })
        }
    }

    // ── Fused compiled ops (match Python's @mx.compile) ─────────────────

    /// Fused SwiGLU: `silu(gate) * up` → 1 compiled dispatch instead of 3.
    #[inline]
    pub fn fused_swiglu(gate: &Self, up: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe { mlx_inline_fused_swiglu(dst.as_mut_ptr(), &gate.raw, &up.raw); Self { raw: dst.assume_init() } }
    }

    /// Fused SiLU: `x * sigmoid(x)` → 1 compiled dispatch instead of 2.
    #[inline]
    pub fn fused_silu(&self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe { mlx_inline_fused_silu(dst.as_mut_ptr(), &self.raw); Self { raw: dst.assume_init() } }
    }

    /// Fused compute_g: `exp(-exp(A_log.f32()) * softplus(a + dt_bias))` → 1 compiled dispatch instead of 6.
    #[inline]
    pub fn fused_compute_g(a_log: &Self, a: &Self, dt_bias: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe { mlx_inline_fused_compute_g(dst.as_mut_ptr(), &a_log.raw, &a.raw, &dt_bias.raw); Self { raw: dst.assume_init() } }
    }

    /// Fused precise SwiGLU: `(silu(gate.f32()) * x.f32()).as(x.dtype)` → 1 compiled dispatch instead of 5.
    #[inline]
    pub fn fused_precise_swiglu(x: &Self, gate: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe { mlx_inline_fused_precise_swiglu(dst.as_mut_ptr(), &x.raw, &gate.raw); Self { raw: dst.assume_init() } }
    }

    // ── Item extraction ───────────────────────────────────────────────────

    pub fn item_f32(&mut self) -> f32 {
        unsafe { mlx_inline_item_f32(&mut self.raw) }
    }
    pub fn item_u32(&mut self) -> u32 {
        unsafe { mlx_inline_item_u32(&mut self.raw) }
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

    /// Embedding/gather lookup: `self[indices]`
    #[inline]
    pub fn index(&self, indices: &Self) -> Self {
        let mut dst = MaybeUninit::<RawBuf>::uninit();
        unsafe {
            mlx_inline_index(dst.as_mut_ptr(), &self.raw, &indices.raw);
            Self {
                raw: dst.assume_init(),
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
