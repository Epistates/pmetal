//! Dynamic weight kernel generators for ANE.
//!
//! Generates 9 MIL programs that compile once at startup and accept weights
//! packed alongside activations in the IOSurface spatial dimension.
//!
//! Instead of `conv(const_weight, x)`, each kernel uses:
//! ```text
//! input: [1, IC, 1, SEQ + weight_cols] fp32
//!   sp[0:SEQ]         = activations
//!   sp[SEQ:SEQ+W]     = weight matrix columns
//! ```
//!
//! MIL pattern: cast fp32→fp16, slice act+weight, reshape, transpose, matmul,
//! cast fp16→fp32, output.
//!
//! # Kernel Table
//!
//! | # | Kernel | IC | Spatial | Weights Packed |
//! |---|--------|---|----|---|
//! | 1 | `sdpa_fwd` | DIM | SEQ + 4*DIM | Wq, Wk, Wv, Wo |
//! | 2 | `ffn_w13` | DIM | SEQ + 2*HIDDEN | W1, W3 (+ SiLU/gate inside) |
//! | 3 | `ffn_w2` | HIDDEN | SEQ + DIM | W2 |
//! | 4 | `ffn_bwd_w2t` | DIM | SEQ + HIDDEN | W2^T |
//! | 5 | `ffn_bwd_w13t` | HIDDEN | 2*SEQ + 2*DIM | W1^T, W3^T |
//! | 6 | `wo_bwd` | DIM | SEQ + Q_DIM | Wo^T |
//! | 7 | `sdpa_bwd1` | 2*Q_DIM+2*KV_DIM | SEQ | None (mask const only) |
//! | 8 | `sdpa_bwd2` | 2*SCORE+Q_DIM+KV_DIM | SEQ | None (weight-free) |
//! | 9a | `qkv_bwd_q` | Q_DIM | SEQ + DIM | Wq^T |
//! | 9b | `qkv_bwd_kv` | KV_DIM | 2*SEQ + 2*DIM | Wk^T, Wv^T |
//! | 9 | `qkv_bwd` | Q_DIM | 3*SEQ + 3*DIM | Wq^T, Wk^T, Wv^T (MHA only) |
//! | 10 | `rmsnorm_bwd` | 2*DIM | SEQ + 1 | RMSNorm weight (1 spatial col) |
//!
//! **GQA support:** Kernels 7-8 handle GQA natively via tile+reduce_sum.
//! Kernel 9 is split into 9a+9b for GQA (mixed IC dimensions).

use crate::ane::kernel::{TransformerKernelConfig, WeightBlob};
use crate::ane::mil::MilProgram;
use crate::ane::runtime::WeightDict;

/// Configuration wrapper for dynamic kernel generation.
#[derive(Debug, Clone)]
pub struct DynamicKernelConfig {
    /// Underlying transformer kernel config.
    pub cfg: TransformerKernelConfig,
}

impl DynamicKernelConfig {
    /// Create a new dynamic kernel config wrapper.
    pub fn new(cfg: TransformerKernelConfig) -> Self {
        Self { cfg }
    }
}

/// Spatial layout descriptor for a dynamic kernel's IOSurface.
#[derive(Debug, Clone)]
pub struct SpatialLayout {
    /// Number of input channels.
    pub ic: usize,
    /// Sequence length portion of spatial dimension.
    pub seq_len: usize,
    /// Total spatial dimension (seq_len + sum of weight columns).
    pub total_spatial: usize,
    /// Output channels.
    pub oc: usize,
    /// Output spatial dimension.
    pub out_spatial: usize,
}

/// Result of dynamic kernel generation.
pub struct DynamicKernelOutput {
    /// MIL program text (UTF-8).
    pub mil_text: String,
    /// Static weights (causal mask, RoPE tables — things that don't change).
    pub static_weights: WeightDict,
    /// Input layout descriptor.
    pub input_layout: SpatialLayout,
    /// Output layout descriptor.
    pub output_layout: SpatialLayout,
}

// ============================================================================
// Helper: emit a dynamic matmul block
// ============================================================================

/// Emit MIL ops for a single dynamic matmul within a larger kernel.
///
/// Given an fp16 input tensor, slices activations and weights from the spatial
/// dimension, reshapes/transposes, performs matmul, and reshapes back.
///
/// Returns the name of the output variable `[1, oc, 1, seq]` fp16.
fn emit_dyn_matmul(
    p: &mut MilProgram,
    prefix: &str,
    input: &str,
    ic: usize,
    oc: usize,
    seq: usize,
    act_sp_off: usize,
    w_sp_off: usize,
) -> String {
    // Slice activations: [1, ic, 1, seq] from spatial offset
    let act_begin = p.next_var(&format!("{prefix}_ab"));
    p.emit_tensor_const(
        &act_begin,
        &[4],
        "int32",
        &format!("[0,0,0,{}]", act_sp_off),
    );
    let act_size = p.next_var(&format!("{prefix}_as"));
    p.emit_tensor_const(&act_size, &[4], "int32", &format!("[1,{ic},1,{seq}]"));
    let act = p.next_var(&format!("{prefix}_act"));
    p.emit_slice_by_size(&act, &[1, ic, 1, seq], input, &act_begin, &act_size);

    // Slice weights: [1, ic, 1, oc] from spatial offset
    let w_begin = p.next_var(&format!("{prefix}_wb"));
    p.emit_tensor_const(&w_begin, &[4], "int32", &format!("[0,0,0,{}]", w_sp_off));
    let w_size = p.next_var(&format!("{prefix}_ws"));
    p.emit_tensor_const(&w_size, &[4], "int32", &format!("[1,{ic},1,{oc}]"));
    let w = p.next_var(&format!("{prefix}_w"));
    p.emit_slice_by_size(&w, &[1, ic, 1, oc], input, &w_begin, &w_size);

    // Reshape act: [1, ic, 1, seq] → [1, 1, ic, seq]
    let rsh1 = p.next_var(&format!("{prefix}_rs1"));
    p.emit_tensor_const(&rsh1, &[4], "int32", &format!("[1,1,{ic},{seq}]"));
    let act_r = p.next_var(&format!("{prefix}_ar"));
    p.emit_reshape(&act_r, &[1, 1, ic, seq], &rsh1, &act);

    // Transpose act: [1, 1, ic, seq] → [1, 1, seq, ic]
    let perm_23 = p.next_var(&format!("{prefix}_p23"));
    p.emit_tensor_const(&perm_23, &[4], "int32", "[0,1,3,2]");
    let act_t = p.next_var(&format!("{prefix}_at"));
    p.emit_transpose(&act_t, &[1, 1, seq, ic], &perm_23, &act_r);

    // Reshape weight: [1, ic, 1, oc] → [1, 1, ic, oc]
    let rsh2 = p.next_var(&format!("{prefix}_rs2"));
    p.emit_tensor_const(&rsh2, &[4], "int32", &format!("[1,1,{ic},{oc}]"));
    let w_r = p.next_var(&format!("{prefix}_wr"));
    p.emit_reshape(&w_r, &[1, 1, ic, oc], &rsh2, &w);

    // Matmul: [1, 1, seq, ic] @ [1, 1, ic, oc] → [1, 1, seq, oc]
    let mm_false = p.next_var(&format!("{prefix}_mf"));
    p.emit_scalar_const(&mm_false, "bool", "false");
    let mm = p.next_var(&format!("{prefix}_mm"));
    p.emit_matmul(&mm, &[1, 1, seq, oc], &mm_false, &mm_false, &act_t, &w_r);

    // Transpose back: [1, 1, seq, oc] → [1, 1, oc, seq]
    let perm_back = p.next_var(&format!("{prefix}_pb"));
    p.emit_tensor_const(&perm_back, &[4], "int32", "[0,1,3,2]");
    let mm_t = p.next_var(&format!("{prefix}_mt"));
    p.emit_transpose(&mm_t, &[1, 1, oc, seq], &perm_back, &mm);

    // Reshape: [1, 1, oc, seq] → [1, oc, 1, seq]
    let rsh3 = p.next_var(&format!("{prefix}_rs3"));
    p.emit_tensor_const(&rsh3, &[4], "int32", &format!("[1,{oc},1,{seq}]"));
    let out = p.next_var(&format!("{prefix}_out"));
    p.emit_reshape(&out, &[1, oc, 1, seq], &rsh3, &mm_t);

    out
}

/// Expand a KV tensor from [1, nkv, hd, S] to [1, nh, hd, S] for GQA using concat.
///
/// ANE's tile op is unreliable, so we use concat to replicate each KV head
/// `groups` times. Pattern:
/// - Merge hd*S dims: [1, nkv, hd, S] → [1, nkv, 1, hd*S]
/// - Concat `groups` copies along axis=2: [1, nkv, groups, hd*S]
/// - Reshape back: [1, nh, hd, S]
///
/// `kv_var` must already be [1, nkv, hd, S]. Returns the name of the
/// expanded tensor [1, nh, hd, S].
fn emit_gqa_expand(
    p: &mut MilProgram,
    prefix: &str,
    kv_var: &str,
    nkv: usize,
    nh: usize,
    hd: usize,
    s: usize,
    groups: usize,
) -> String {
    let hds = hd * s;

    // Merge last 2 dims: [1, nkv, hd, S] → [1, nkv, 1, hd*S]
    let merge_rsh = p.next_var(&format!("{prefix}_mrs"));
    p.emit_tensor_const(&merge_rsh, &[4], "int32", &format!("[1,{nkv},1,{hds}]"));
    let merged = p.next_var(&format!("{prefix}_mg"));
    p.emit_reshape(&merged, &[1, nkv, 1, hds], &merge_rsh, kv_var);

    // Concat `groups` copies along axis=2
    let cat_ax = p.next_var(&format!("{prefix}_ca"));
    p.emit_scalar_const(&cat_ax, "int32", "2");
    let cat_il = p.next_var(&format!("{prefix}_ci"));
    p.emit_scalar_const(&cat_il, "bool", "false");
    let copies: Vec<&str> = (0..groups).map(|_| merged.as_str()).collect();
    let expanded = p.next_var(&format!("{prefix}_ex"));
    p.emit_concat(
        &expanded,
        &[1, nkv, groups, hds],
        &cat_ax,
        &cat_il,
        &copies,
    );

    // Reshape to [1, nh, hd, S]
    let nh_rsh = p.next_var(&format!("{prefix}_nr"));
    p.emit_tensor_const(&nh_rsh, &[4], "int32", &format!("[1,{nh},{hd},{s}]"));
    let result = p.next_var(&format!("{prefix}_fn"));
    p.emit_reshape(&result, &[1, nh, hd, s], &nh_rsh, &expanded);

    result
}

/// Reduce a gradient tensor from [1, nh, hd, S] to [1, nkv, hd, S] for GQA.
///
/// Groups consecutive heads and sums them:
/// - Reshape [1, nh, hd, S] → [1, nkv, groups, hd*S] (4D, merge hd+S)
/// - reduce_sum(axis=2, keepdim=true) → [1, nkv, 1, hd*S]
/// - Reshape → [1, nkv, hd, S] → [1, kvd, 1, S]
///
/// `grad_var` must be [1, nh, hd, S]. Returns name of [1, kvd, 1, S] result.
fn emit_gqa_reduce(
    p: &mut MilProgram,
    prefix: &str,
    grad_var: &str,
    nkv: usize,
    groups: usize,
    hd: usize,
    s: usize,
    kvd: usize,
) -> String {
    let hds = hd * s;

    // Reshape [1, nh, hd, S] → [1, nkv, groups, hd*S]
    let grp_rsh = p.next_var(&format!("{prefix}_gr"));
    p.emit_tensor_const(
        &grp_rsh,
        &[4],
        "int32",
        &format!("[1,{nkv},{groups},{hds}]"),
    );
    let grp = p.next_var(&format!("{prefix}_g"));
    p.emit_reshape(&grp, &[1, nkv, groups, hds], &grp_rsh, grad_var);

    // reduce_sum(axis=2, keepdim=true) → [1, nkv, 1, hd*S]
    let ax2 = p.next_var(&format!("{prefix}_a2"));
    p.emit_tensor_const(&ax2, &[1], "int32", "[2]");
    let kd = p.next_var(&format!("{prefix}_kd"));
    p.emit_scalar_const(&kd, "bool", "true");
    let reduced = p.next_var(&format!("{prefix}_rd"));
    p.emit_reduce_sum(&reduced, &[1, nkv, 1, hds], &grp, &ax2, &kd);

    // Reshape [1, nkv, 1, hd*S] → [1, nkv, hd, S]
    let kv_head_rsh = p.next_var(&format!("{prefix}_hr"));
    p.emit_tensor_const(&kv_head_rsh, &[4], "int32", &format!("[1,{nkv},{hd},{s}]"));
    let heads = p.next_var(&format!("{prefix}_hd"));
    p.emit_reshape(&heads, &[1, nkv, hd, s], &kv_head_rsh, &reduced);

    // Flatten to [1, kvd, 1, S]
    let flat_rsh = p.next_var(&format!("{prefix}_fr"));
    p.emit_tensor_const(&flat_rsh, &[4], "int32", &format!("[1,{kvd},1,{s}]"));
    let result = p.next_var(&format!("{prefix}_fl"));
    p.emit_reshape(&result, &[1, kvd, 1, s], &flat_rsh, &heads);

    result
}

// ============================================================================
// Decomposed kernels: single-projection + attention-only
// ============================================================================

/// Generate a standalone single linear projection kernel.
///
/// This is the fundamental building block for the decomposed ANE pipeline.
/// Each projection (Q, K, V, Wo, W1, W2, W3, and their transposes for
/// backward) gets its own compiled kernel. This keeps IOSurface sizes
/// manageable and avoids the "too complex" MIL compilation failures that
/// occur when multiple matmuls are packed into a single kernel.
///
/// Input: `[1, IC, 1, SEQ + OC]` fp32
/// - `sp[0:SEQ]`      = activations `[IC, SEQ]`
/// - `sp[SEQ:SEQ+OC]` = weight matrix columns `[IC, OC]`
///
/// Output: `[1, OC, 1, SEQ]` fp32
/// - `y = act @ W`
pub fn gen_dynamic_projection(ic: usize, oc: usize, seq: usize) -> DynamicKernelOutput {
    let sp = seq + oc;
    let mut p = MilProgram::new_fp32(ic, sp);

    // Cast fp32 → fp16
    p.emit_cast("x16", &[1, ic, 1, sp], "x", "fp16");

    // Single matmul: act[ic, seq] @ W[ic, oc] → out[oc, seq]
    let out16 = emit_dyn_matmul(&mut p, "mm", "x16", ic, oc, seq, 0, seq);

    // Cast fp16 → fp32
    let out32 = p.next_var("out32");
    p.emit_cast(&out32, &[1, oc, 1, seq], &out16, "fp32");

    let mil_text = p.finalize(&out32);

    DynamicKernelOutput {
        mil_text,
        static_weights: WeightDict::new(),
        input_layout: SpatialLayout {
            ic,
            seq_len: seq,
            total_spatial: sp,
            oc,
            out_spatial: seq,
        },
        output_layout: SpatialLayout {
            ic: oc,
            seq_len: seq,
            total_spatial: seq,
            oc,
            out_spatial: seq,
        },
    }
}

/// Generate the attention-only SDPA kernel (no weight projections).
///
/// Takes pre-projected Q, K, V concatenated along channels and computes
/// scaled dot-product attention with causal mask.
///
/// Input: `[1, QD + 2*KVD, 1, SEQ]` fp32
/// - `ch[0:QD]`                = Q (pre-projected)
/// - `ch[QD:QD+KVD]`           = K (pre-projected)
/// - `ch[QD+KVD:QD+2*KVD]`     = V (pre-projected)
///
/// Output: `[1, QD, 1, SEQ]` fp32
/// - attention output (before Wo projection)
///
/// Supports both MHA and GQA. For GQA, K/V are expanded via concat
/// to full head count before attention computation.
pub fn gen_dynamic_sdpa_attn(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let qd = c.q_dim();
    let kvd = c.kv_dim();
    let s = c.seq_len;
    let nh = c.n_heads;
    let nkv = c.n_kv_heads;
    let groups = c.n_groups();
    let hd = c.head_dim;
    let in_ch = qd + 2 * kvd;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut p = MilProgram::new_fp32(in_ch, s);

    // Cast fp32 → fp16
    p.emit_cast("x16", &[1, in_ch, 1, s], "x", "fp16");

    // Slice Q, K, V from channels
    let sb = |p: &mut MilProgram, name: &str, ch_off: usize, ch: usize| -> String {
        let begin = p.next_var(&format!("{name}_b"));
        p.emit_tensor_const(&begin, &[4], "int32", &format!("[0,{ch_off},0,0]"));
        let size = p.next_var(&format!("{name}_s"));
        p.emit_tensor_const(&size, &[4], "int32", &format!("[1,{ch},1,{s}]"));
        let out = p.next_var(name);
        p.emit_slice_by_size(&out, &[1, ch, 1, s], "x16", &begin, &size);
        out
    };

    let q_flat = sb(&mut p, "qf", 0, qd);
    let k_flat = sb(&mut p, "kf", qd, kvd);
    let v_flat = sb(&mut p, "vf", qd + kvd, kvd);

    // Reshape Q: [1, qd, 1, S] → [1, nh, hd, S]
    let q_rsh = p.next_var("qrs");
    p.emit_tensor_const(&q_rsh, &[4], "int32", &format!("[1,{nh},{hd},{s}]"));
    let q_heads = p.next_var("qh");
    p.emit_reshape(&q_heads, &[1, nh, hd, s], &q_rsh, &q_flat);

    // Transpose Q: [1, nh, hd, S] → [1, nh, S, hd]
    let perm23 = p.next_var("p23");
    p.emit_tensor_const(&perm23, &[4], "int32", "[0,1,3,2]");
    let qt = p.next_var("qt");
    p.emit_transpose(&qt, &[1, nh, s, hd], &perm23, &q_heads);

    // K/V: reshape to [nkv, hd, S], expand to [nh, hd, S] if GQA
    let kv_rsh = p.next_var("kvrs");
    p.emit_tensor_const(&kv_rsh, &[4], "int32", &format!("[1,{nkv},{hd},{s}]"));
    let k_kv = p.next_var("kkv");
    p.emit_reshape(&k_kv, &[1, nkv, hd, s], &kv_rsh, &k_flat);

    let (k_heads, v_h) = if groups > 1 {
        let k_final = emit_gqa_expand(&mut p, "ke", &k_kv, nkv, nh, hd, s, groups);
        let v_kv = p.next_var("vkv");
        p.emit_reshape(&v_kv, &[1, nkv, hd, s], &kv_rsh, &v_flat);
        let v_final = emit_gqa_expand(&mut p, "ve", &v_kv, nkv, nh, hd, s, groups);
        (k_final, v_final)
    } else {
        let k_h = p.next_var("kh");
        p.emit_reshape(&k_h, &[1, nh, hd, s], &q_rsh, &k_flat);
        let v_h = p.next_var("vh");
        p.emit_reshape(&v_h, &[1, nh, hd, s], &q_rsh, &v_flat);
        (k_h, v_h)
    };

    // scores = Q @ K^T: [1, nh, S, hd] @ [1, nh, hd, S] → [1, nh, S, S]
    let mm_false = p.next_var("mmf");
    p.emit_scalar_const(&mm_false, "bool", "false");
    let scores_raw = p.next_var("sr");
    p.emit_matmul(
        &scores_raw,
        &[1, nh, s, s],
        &mm_false,
        &mm_false,
        &qt,
        &k_heads,
    );

    // Scale
    let scale_var = p.next_var("sc");
    p.emit_scalar_const(&scale_var, "fp16", &format!("{scale}"));
    let scores_scaled = p.next_var("ss");
    p.emit_mul(&scores_scaled, &[1, nh, s, s], &scores_raw, &scale_var);

    // Causal mask
    let mask_path = "@model_path/weights/mask.bin";
    p.emit_weight_const("mask", &[1, 1, s, s], mask_path);
    let scores_masked = p.next_var("sm");
    p.emit_add(&scores_masked, &[1, nh, s, s], &scores_scaled, "mask");

    // Softmax
    let ax3 = p.next_var("ax3");
    p.emit_scalar_const(&ax3, "int32", "3");
    let probs = p.next_var("probs");
    p.emit_softmax(&probs, &[1, nh, s, s], &ax3, &scores_masked);

    // Transpose V: [1, nh, hd, S] → [1, nh, S, hd]
    let vt = p.next_var("vt");
    p.emit_transpose(&vt, &[1, nh, s, hd], &perm23, &v_h);

    // attn = probs @ V_t: [1, nh, S, S] @ [1, nh, S, hd] → [1, nh, S, hd]
    let attn = p.next_var("attn");
    p.emit_matmul(&attn, &[1, nh, s, hd], &mm_false, &mm_false, &probs, &vt);

    // Transpose: [1, nh, S, hd] → [1, nh, hd, S]
    let attn_t = p.next_var("at");
    p.emit_transpose(&attn_t, &[1, nh, hd, s], &perm23, &attn);

    // Reshape: [1, nh, hd, S] → [1, qd, 1, S]
    let qd_rsh = p.next_var("qdrs");
    p.emit_tensor_const(&qd_rsh, &[4], "int32", &format!("[1,{qd},1,{s}]"));
    let attn_flat = p.next_var("af");
    p.emit_reshape(&attn_flat, &[1, qd, 1, s], &qd_rsh, &attn_t);

    // Cast fp16 → fp32
    let out32 = p.next_var("out32");
    p.emit_cast(&out32, &[1, qd, 1, s], &attn_flat, "fp32");

    let mil_text = p.finalize(&out32);

    let mut static_weights = WeightDict::new();
    static_weights.add(mask_path, build_causal_mask(s));

    DynamicKernelOutput {
        mil_text,
        static_weights,
        input_layout: SpatialLayout {
            ic: in_ch,
            seq_len: s,
            total_spatial: s,
            oc: in_ch,
            out_spatial: s,
        },
        output_layout: SpatialLayout {
            ic: qd,
            seq_len: s,
            total_spatial: s,
            oc: qd,
            out_spatial: s,
        },
    }
}

/// Build the causal mask blob (fp16). Same format as kernel.rs.
fn build_causal_mask(seq_len: usize) -> Vec<u8> {
    let n = seq_len * seq_len;
    let mut mask = vec![0.0f32; n];
    for t in 0..seq_len {
        for t2 in 0..seq_len {
            mask[t * seq_len + t2] = if t2 <= t { 0.0 } else { -65504.0 };
        }
    }
    WeightBlob::from_f32(&mask, seq_len, seq_len)
}

// ============================================================================
// Kernel 1: SDPA Forward (dynamic Wq, Wk, Wv, Wo)
// ============================================================================

/// Generate the dynamic SDPA forward kernel.
///
/// Input: `[1, DIM, 1, SEQ + 4*DIM]` fp32
/// - `sp[0:SEQ]` = xnorm (post-RMSNorm activations)
/// - `sp[SEQ:SEQ+DIM]` = Wq
/// - `sp[SEQ+DIM:SEQ+2*DIM]` = Wk
/// - `sp[SEQ+2*DIM:SEQ+3*DIM]` = Wv
/// - `sp[SEQ+3*DIM:SEQ+4*DIM]` = Wo
///
/// Output: `[1, 6*DIM, 1, SEQ]` fp32 (assumes MHA: q_dim == kv_dim == dim)
/// - `concat(o_out[DIM], Q[DIM], K[DIM], V[DIM], attn_out[DIM], xnorm[DIM])`
pub fn gen_dynamic_sdpa_fwd(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let s = c.seq_len;
    let nh = c.n_heads;
    let nkv = c.n_kv_heads;
    let groups = c.n_groups();
    let hd = c.head_dim;
    let qd = c.q_dim(); // n_heads * head_dim
    let kvd = c.kv_dim(); // n_kv_heads * head_dim

    // Spatial: xnorm(s) + Wq(qd cols) + Wk(kvd cols) + Wv(kvd cols) + Wo(qd cols)
    let wo_offset = s + qd + kvd + kvd;
    let sp = wo_offset + qd;
    // Output taps: [o_out(d), Q(qd), K(kvd), V(kvd), attn(qd), xnorm(d)]
    let out_ch = 2 * d + 2 * qd + 2 * kvd;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut p = MilProgram::new_fp32(d, sp);

    // Cast input fp32 → fp16
    p.emit_cast("x16", &[1, d, 1, sp], "x", "fp16");

    // Q projection: xnorm[d, s] @ Wq[d, qd] → Q[qd, s]
    let q = emit_dyn_matmul(&mut p, "q", "x16", d, qd, s, 0, s);
    // K projection: xnorm[d, s] @ Wk[d, kvd] → K[kvd, s]
    let k = emit_dyn_matmul(&mut p, "k", "x16", d, kvd, s, 0, s + qd);
    // V projection: xnorm[d, s] @ Wv[d, kvd] → V[kvd, s]
    let v = emit_dyn_matmul(&mut p, "v", "x16", d, kvd, s, 0, s + qd + kvd);

    // Save xnorm (slice from input)
    let xn_begin = p.next_var("xnb");
    p.emit_tensor_const(&xn_begin, &[4], "int32", "[0,0,0,0]");
    let xn_size = p.next_var("xns");
    p.emit_tensor_const(&xn_size, &[4], "int32", &format!("[1,{d},1,{s}]"));
    let xnorm = p.next_var("xnorm");
    p.emit_slice_by_size(&xnorm, &[1, d, 1, s], "x16", &xn_begin, &xn_size);

    // Reshape Q to multi-head: [1, qd, 1, S] → [1, nh, hd, S]
    let q_rsh = p.next_var("qrs");
    p.emit_tensor_const(&q_rsh, &[4], "int32", &format!("[1,{nh},{hd},{s}]"));
    let q_heads = p.next_var("qh");
    p.emit_reshape(&q_heads, &[1, nh, hd, s], &q_rsh, &q);

    // Transpose Q: [1, nh, hd, S] → [1, nh, S, hd]
    let perm23 = p.next_var("p23");
    p.emit_tensor_const(&perm23, &[4], "int32", "[0,1,3,2]");
    let qt = p.next_var("qt");
    p.emit_transpose(&qt, &[1, nh, s, hd], &perm23, &q_heads);

    // Reshape K to [nkv, hd, S], then expand to [nh, hd, S] if GQA
    let kv_rsh = p.next_var("kvrs");
    p.emit_tensor_const(&kv_rsh, &[4], "int32", &format!("[1,{nkv},{hd},{s}]"));
    let k_kv = p.next_var("kkv");
    p.emit_reshape(&k_kv, &[1, nkv, hd, s], &kv_rsh, &k);

    let (k_heads, v_h) = if groups > 1 {
        // GQA K/V expansion using concat (tile op unreliable on ANE)
        let k_final = emit_gqa_expand(&mut p, "ke", &k_kv, nkv, nh, hd, s, groups);

        let v_kv = p.next_var("vkv");
        p.emit_reshape(&v_kv, &[1, nkv, hd, s], &kv_rsh, &v);
        let v_final = emit_gqa_expand(&mut p, "ve", &v_kv, nkv, nh, hd, s, groups);

        (k_final, v_final)
    } else {
        // MHA: K and V are already at full head count
        let k_h = p.next_var("kh");
        p.emit_reshape(&k_h, &[1, nh, hd, s], &q_rsh, &k);
        let v_h = p.next_var("vh");
        p.emit_reshape(&v_h, &[1, nh, hd, s], &q_rsh, &v);
        (k_h, v_h)
    };

    // scores = Q @ K^T: [1, nh, S, hd] @ [1, nh, hd, S] → [1, nh, S, S]
    let mm_false = p.next_var("mmf");
    p.emit_scalar_const(&mm_false, "bool", "false");
    let scores_raw = p.next_var("sr");
    p.emit_matmul(
        &scores_raw,
        &[1, nh, s, s],
        &mm_false,
        &mm_false,
        &qt,
        &k_heads,
    );

    // Scale scores
    let scale_var = p.next_var("sc");
    p.emit_scalar_const(&scale_var, "fp16", &format!("{scale}"));
    let scores_scaled = p.next_var("ss");
    p.emit_mul(&scores_scaled, &[1, nh, s, s], &scores_raw, &scale_var);

    // Add causal mask (static const BLOBFILE)
    let mask_path = "@model_path/weights/mask.bin";
    p.emit_weight_const("mask", &[1, 1, s, s], mask_path);
    let scores_masked = p.next_var("sm");
    p.emit_add(&scores_masked, &[1, nh, s, s], &scores_scaled, "mask");

    // Softmax over last axis (axis=3)
    let ax3 = p.next_var("ax3");
    p.emit_scalar_const(&ax3, "int32", "3");
    let probs = p.next_var("probs");
    p.emit_softmax(&probs, &[1, nh, s, s], &ax3, &scores_masked);

    // Transpose V: [1, nh, hd, S] → [1, nh, S, hd]
    let vt = p.next_var("vt");
    p.emit_transpose(&vt, &[1, nh, s, hd], &perm23, &v_h);

    // attn_out = probs @ V_t: [1, nh, S, S] @ [1, nh, S, hd] → [1, nh, S, hd]
    let attn = p.next_var("attn");
    p.emit_matmul(&attn, &[1, nh, s, hd], &mm_false, &mm_false, &probs, &vt);

    // Transpose back: [1, nh, S, hd] → [1, nh, hd, S]
    let attn_t = p.next_var("at");
    p.emit_transpose(&attn_t, &[1, nh, hd, s], &perm23, &attn);

    // Reshape: [1, nh, hd, S] → [1, qd, 1, S]
    let qd_rsh = p.next_var("qdrs");
    p.emit_tensor_const(&qd_rsh, &[4], "int32", &format!("[1,{qd},1,{s}]"));
    let attn_flat = p.next_var("af");
    p.emit_reshape(&attn_flat, &[1, qd, 1, s], &qd_rsh, &attn_t);

    // Wo projection: attn_flat[qd, S] @ Wo → o_out[d, S]
    // Wo is packed in input as [1, d, 1, qd] at offset wo_offset.
    // Wo weight matrix is [d, qd] (d outputs, qd inputs) stored row-major.
    // Packed column-wise: qd columns of d channels each.
    // Reshape to [1, 1, d, qd] then matmul: attn[1,1,s,qd] @ Wo^T[1,1,qd,d] → [1,1,s,d]
    let wo_begin = p.next_var("wob");
    p.emit_tensor_const(
        &wo_begin,
        &[4],
        "int32",
        &format!("[0,0,0,{}]", wo_offset),
    );
    let wo_size = p.next_var("wos");
    p.emit_tensor_const(&wo_size, &[4], "int32", &format!("[1,{d},1,{qd}]"));
    let wo_raw = p.next_var("wor");
    p.emit_slice_by_size(&wo_raw, &[1, d, 1, qd], "x16", &wo_begin, &wo_size);

    // Reshape attn_flat for matmul: [1, qd, 1, S] → [1, 1, qd, S] → [1, 1, S, qd]
    let af_rsh = p.next_var("afrs");
    p.emit_tensor_const(&af_rsh, &[4], "int32", &format!("[1,1,{qd},{s}]"));
    let af_r = p.next_var("afr");
    p.emit_reshape(&af_r, &[1, 1, qd, s], &af_rsh, &attn_flat);
    let af_t = p.next_var("aft");
    p.emit_transpose(&af_t, &[1, 1, s, qd], &perm23, &af_r);

    // Reshape Wo: [1, d, 1, qd] → [1, 1, d, qd] → transpose → [1, 1, qd, d]
    let wo_rsh = p.next_var("wrs");
    p.emit_tensor_const(&wo_rsh, &[4], "int32", &format!("[1,1,{d},{qd}]"));
    let wo_r = p.next_var("wrr");
    p.emit_reshape(&wo_r, &[1, 1, d, qd], &wo_rsh, &wo_raw);
    let wo_t = p.next_var("wot");
    p.emit_transpose(&wo_t, &[1, 1, qd, d], &perm23, &wo_r);

    // matmul: [1, 1, S, qd] @ [1, 1, qd, d] → [1, 1, S, d]
    let oo_mm = p.next_var("oom");
    p.emit_matmul(&oo_mm, &[1, 1, s, d], &mm_false, &mm_false, &af_t, &wo_t);

    // Transpose + reshape back: [1, 1, S, d] → [1, 1, d, S] → [1, d, 1, S]
    let out_rsh = p.next_var("ors");
    p.emit_tensor_const(&out_rsh, &[4], "int32", &format!("[1,{d},1,{s}]"));
    let oo_t = p.next_var("oot");
    p.emit_transpose(&oo_t, &[1, 1, d, s], &perm23, &oo_mm);
    let o_out = p.next_var("oo");
    p.emit_reshape(&o_out, &[1, d, 1, s], &out_rsh, &oo_t);

    // Concat taps: [o_out(d), Q(qd), K(kvd), V(kvd), attn_flat(qd), xnorm(d)]
    let cat_ax = p.next_var("cax");
    p.emit_scalar_const(&cat_ax, "int32", "1");
    let cat_il = p.next_var("cil");
    p.emit_scalar_const(&cat_il, "bool", "false");
    let taps16 = p.next_var("taps16");
    p.emit_concat(
        &taps16,
        &[1, out_ch, 1, s],
        &cat_ax,
        &cat_il,
        &[&o_out, &q, &k, &v, &attn_flat, &xnorm],
    );

    // Cast output fp16 → fp32
    let taps32 = p.next_var("taps32");
    p.emit_cast(&taps32, &[1, out_ch, 1, s], &taps16, "fp32");

    let mil_text = p.finalize(&taps32);

    // Static weight: causal mask only
    let mut static_weights = WeightDict::new();
    static_weights.add(mask_path, build_causal_mask(s));

    DynamicKernelOutput {
        mil_text,
        static_weights,
        input_layout: SpatialLayout {
            ic: d,
            seq_len: s,
            total_spatial: sp,
            oc: d,
            out_spatial: s,
        },
        output_layout: SpatialLayout {
            ic: out_ch,
            seq_len: s,
            total_spatial: s,
            oc: out_ch,
            out_spatial: s,
        },
    }
}

// ============================================================================
// Kernel 2: FFN W1+W3 Forward (dynamic W1, W3 + SiLU gate)
// ============================================================================

/// Generate the dynamic FFN W1+W3 forward kernel.
///
/// Input: `[1, DIM, 1, SEQ + 2*HIDDEN]` fp32
/// - `sp[0:SEQ]` = x2norm
/// - `sp[SEQ:SEQ+HIDDEN]` = W1
/// - `sp[SEQ+HIDDEN:SEQ+2*HIDDEN]` = W3
///
/// Output: `[1, 3*HIDDEN, 1, SEQ]` fp32
/// - `concat(h1, h3, gate)`
pub fn gen_dynamic_ffn_w13(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let h = c.hidden_dim;
    let s = c.seq_len;
    let sp = s + 2 * h;
    let out_ch = 3 * h;

    let mut p = MilProgram::new_fp32(d, sp);

    // Cast fp32 → fp16
    p.emit_cast("x16", &[1, d, 1, sp], "x", "fp16");

    // h1 = xnorm @ W1: dynamic matmul
    let h1 = emit_dyn_matmul(&mut p, "w1", "x16", d, h, s, 0, s);
    // h3 = xnorm @ W3: dynamic matmul
    let h3 = emit_dyn_matmul(&mut p, "w3", "x16", d, h, s, 0, s + h);

    // SiLU: sig = sigmoid(h1), silu = h1 * sig
    let sig = p.next_var("sig");
    p.emit_sigmoid(&sig, &[1, h, 1, s], &h1);
    let silu = p.next_var("silu");
    p.emit_mul(&silu, &[1, h, 1, s], &h1, &sig);

    // gate = silu * h3
    let gate = p.next_var("gate");
    p.emit_mul(&gate, &[1, h, 1, s], &silu, &h3);

    // Concat output: [h1, h3, gate]
    let cat_ax = p.next_var("cax");
    p.emit_scalar_const(&cat_ax, "int32", "1");
    let cat_il = p.next_var("cil");
    p.emit_scalar_const(&cat_il, "bool", "false");
    let out16 = p.next_var("out16");
    p.emit_concat(
        &out16,
        &[1, out_ch, 1, s],
        &cat_ax,
        &cat_il,
        &[&h1, &h3, &gate],
    );

    // Cast fp16 → fp32
    let out32 = p.next_var("out32");
    p.emit_cast(&out32, &[1, out_ch, 1, s], &out16, "fp32");

    let mil_text = p.finalize(&out32);

    DynamicKernelOutput {
        mil_text,
        static_weights: WeightDict::new(),
        input_layout: SpatialLayout {
            ic: d,
            seq_len: s,
            total_spatial: sp,
            oc: h,
            out_spatial: s,
        },
        output_layout: SpatialLayout {
            ic: out_ch,
            seq_len: s,
            total_spatial: s,
            oc: out_ch,
            out_spatial: s,
        },
    }
}

// ============================================================================
// Kernel 3: FFN W2 Forward (dynamic W2)
// ============================================================================

/// Generate the dynamic FFN W2 forward kernel.
///
/// Input: `[1, HIDDEN, 1, SEQ + DIM]` fp32
/// - `sp[0:SEQ]` = gate (SiLU output)
/// - `sp[SEQ:SEQ+DIM]` = W2
///
/// Output: `[1, DIM, 1, SEQ]` fp32
pub fn gen_dynamic_ffn_w2(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let h = c.hidden_dim;
    let s = c.seq_len;
    let sp = s + d;

    let mut p = MilProgram::new_fp32(h, sp);

    // Cast fp32 → fp16
    p.emit_cast("x16", &[1, h, 1, sp], "x", "fp16");

    // y = gate @ W2
    let y = emit_dyn_matmul(&mut p, "w2", "x16", h, d, s, 0, s);

    // Cast fp16 → fp32
    let out32 = p.next_var("out32");
    p.emit_cast(&out32, &[1, d, 1, s], &y, "fp32");

    let mil_text = p.finalize(&out32);

    DynamicKernelOutput {
        mil_text,
        static_weights: WeightDict::new(),
        input_layout: SpatialLayout {
            ic: h,
            seq_len: s,
            total_spatial: sp,
            oc: d,
            out_spatial: s,
        },
        output_layout: SpatialLayout {
            ic: d,
            seq_len: s,
            total_spatial: s,
            oc: d,
            out_spatial: s,
        },
    }
}

// ============================================================================
// Kernel 4: FFN Backward W2^T (dynamic W2^T)
// ============================================================================

/// Generate the dynamic FFN backward W2^T kernel.
///
/// Input: `[1, DIM, 1, SEQ + HIDDEN]` fp32
/// - `sp[0:SEQ]` = dffn
/// - `sp[SEQ:SEQ+HIDDEN]` = W2^T
///
/// Output: `[1, HIDDEN, 1, SEQ]` fp32
pub fn gen_dynamic_ffn_bwd_w2t(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let h = c.hidden_dim;
    let s = c.seq_len;
    let sp = s + h;

    let mut p = MilProgram::new_fp32(d, sp);

    p.emit_cast("x16", &[1, d, 1, sp], "x", "fp16");

    let dsilu = emit_dyn_matmul(&mut p, "w2t", "x16", d, h, s, 0, s);

    let out32 = p.next_var("out32");
    p.emit_cast(&out32, &[1, h, 1, s], &dsilu, "fp32");

    let mil_text = p.finalize(&out32);

    DynamicKernelOutput {
        mil_text,
        static_weights: WeightDict::new(),
        input_layout: SpatialLayout {
            ic: d,
            seq_len: s,
            total_spatial: sp,
            oc: h,
            out_spatial: s,
        },
        output_layout: SpatialLayout {
            ic: h,
            seq_len: s,
            total_spatial: s,
            oc: h,
            out_spatial: s,
        },
    }
}

// ============================================================================
// Kernel 5: FFN Backward W1^T + W3^T (dynamic W1^T, W3^T)
// ============================================================================

/// Generate the dynamic FFN backward W1^T + W3^T kernel.
///
/// Input: `[1, HIDDEN, 1, 2*SEQ + 2*DIM]` fp32
/// - `sp[0:SEQ]` = dh1
/// - `sp[SEQ:2*SEQ]` = dh3
/// - `sp[2*SEQ:2*SEQ+DIM]` = W1^T
/// - `sp[2*SEQ+DIM:2*SEQ+2*DIM]` = W3^T
///
/// Output: `[1, DIM, 1, SEQ]` fp32 = dx1 + dx3
pub fn gen_dynamic_ffn_bwd_w13t(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let h = c.hidden_dim;
    let s = c.seq_len;
    let sp = 2 * s + 2 * d;

    let mut p = MilProgram::new_fp32(h, sp);

    p.emit_cast("x16", &[1, h, 1, sp], "x", "fp16");

    // dx1 = dh1 @ W1^T
    let dx1 = emit_dyn_matmul(&mut p, "w1t", "x16", h, d, s, 0, 2 * s);
    // dx3 = dh3 @ W3^T
    let dx3 = emit_dyn_matmul(&mut p, "w3t", "x16", h, d, s, s, 2 * s + d);

    // dx = dx1 + dx3
    let dx = p.next_var("dx");
    p.emit_add(&dx, &[1, d, 1, s], &dx1, &dx3);

    let out32 = p.next_var("out32");
    p.emit_cast(&out32, &[1, d, 1, s], &dx, "fp32");

    let mil_text = p.finalize(&out32);

    DynamicKernelOutput {
        mil_text,
        static_weights: WeightDict::new(),
        input_layout: SpatialLayout {
            ic: h,
            seq_len: s,
            total_spatial: sp,
            oc: d,
            out_spatial: s,
        },
        output_layout: SpatialLayout {
            ic: d,
            seq_len: s,
            total_spatial: s,
            oc: d,
            out_spatial: s,
        },
    }
}

// ============================================================================
// Kernel 6: Wo^T Backward (dynamic Wo^T)
// ============================================================================

/// Generate the dynamic Wo^T backward kernel.
///
/// Input: `[1, DIM, 1, SEQ + Q_DIM]` fp32
/// - `sp[0:SEQ]` = dy (dim channels)
/// - `sp[SEQ:SEQ+Q_DIM]` = Wo^T columns (dim → q_dim)
///
/// Output: `[1, Q_DIM, 1, SEQ]` fp32
///
/// Wo is [dim, q_dim], so Wo^T is [q_dim, dim].
/// dy[s, dim] @ Wo^T[dim, q_dim] → da[s, q_dim].
pub fn gen_dynamic_wo_bwd(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let qd = c.q_dim(); // n_heads * head_dim
    let s = c.seq_len;
    let sp = s + qd; // Wo^T has q_dim output columns

    let mut p = MilProgram::new_fp32(d, sp);

    p.emit_cast("x16", &[1, d, 1, sp], "x", "fp16");

    // dy[s, d] @ Wo^T[d, qd] → da[s, qd]
    let da = emit_dyn_matmul(&mut p, "wot", "x16", d, qd, s, 0, s);

    let out32 = p.next_var("out32");
    p.emit_cast(&out32, &[1, qd, 1, s], &da, "fp32");

    let mil_text = p.finalize(&out32);

    DynamicKernelOutput {
        mil_text,
        static_weights: WeightDict::new(),
        input_layout: SpatialLayout {
            ic: d,
            seq_len: s,
            total_spatial: sp,
            oc: d,
            out_spatial: s,
        },
        output_layout: SpatialLayout {
            ic: qd,
            seq_len: s,
            total_spatial: s,
            oc: qd,
            out_spatial: s,
        },
    }
}

// ============================================================================
// Kernel 7: SDPA Backward Part 1 (no dynamic weights, mask const only)
// ============================================================================

/// Generate the SDPA backward part 1 kernel (no dynamic weights).
///
/// Input: `[1, Q_DIM+2*KV_DIM+Q_DIM, 1, SEQ]` fp16
/// - concat(Q(q_dim), K(kv_dim), V(kv_dim), da(q_dim))
///
/// Output: `[1, KV_DIM + 2*SCORE_CH, 1, SEQ]` fp16
/// - concat(dV(kv_dim), probs, dp)
///
/// Supports both MHA (`n_kv_heads == n_heads`) and GQA (`n_kv_heads < n_heads`).
/// For GQA, K/V are expanded via tile to full head count before attention,
/// and dV_full is reduced (group-summed) back to `n_kv_heads` dimension.
pub fn gen_dynamic_sdpa_bwd1(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let qd = c.q_dim(); // n_heads * head_dim
    let kvd = c.kv_dim(); // n_kv_heads * head_dim
    let s = c.seq_len;
    let nh = c.n_heads;
    let nkv = c.n_kv_heads;
    let groups = c.n_groups(); // nh / nkv
    let hd = c.head_dim;
    let score_ch = nh * s;
    let in_ch = 2 * qd + 2 * kvd; // Q(qd) + K(kvd) + V(kvd) + da(qd)
    let out_ch = kvd + 2 * score_ch; // dV(kvd) + probs + dp
    let scale = 1.0 / (hd as f32).sqrt();

    let mut p = MilProgram::new(in_ch, s);

    // Slice Q(qd), K(kvd), V(kvd), da(qd) from input
    let sb = |p: &mut MilProgram, name: &str, ch_off: usize, ch: usize| -> String {
        let begin = p.next_var(&format!("{name}_b"));
        p.emit_tensor_const(&begin, &[4], "int32", &format!("[0,{ch_off},0,0]"));
        let size = p.next_var(&format!("{name}_s"));
        p.emit_tensor_const(&size, &[4], "int32", &format!("[1,{ch},1,{s}]"));
        let out = p.next_var(name);
        p.emit_slice_by_size(&out, &[1, ch, 1, s], "x", &begin, &size);
        out
    };

    let q_flat = sb(&mut p, "qf", 0, qd);
    let k_flat = sb(&mut p, "kf", qd, kvd);
    let v_flat = sb(&mut p, "vf", qd + kvd, kvd);
    let da_flat = sb(&mut p, "daf", qd + 2 * kvd, qd);

    // Reshape Q to multi-head: [1, qd, 1, S] → [1, nh, hd, S]
    let q_rsh = p.next_var("qrs");
    p.emit_tensor_const(&q_rsh, &[4], "int32", &format!("[1,{nh},{hd},{s}]"));
    let perm23 = p.next_var("p23");
    p.emit_tensor_const(&perm23, &[4], "int32", "[0,1,3,2]");
    let mm_false = p.next_var("mmf");
    p.emit_scalar_const(&mm_false, "bool", "false");
    let mm_true = p.next_var("mmt");
    p.emit_scalar_const(&mm_true, "bool", "true");

    let q_h = p.next_var("qh");
    p.emit_reshape(&q_h, &[1, nh, hd, s], &q_rsh, &q_flat);
    let qt = p.next_var("qt");
    p.emit_transpose(&qt, &[1, nh, s, hd], &perm23, &q_h);

    // K/V: reshape to [1, nkv, hd, S], then expand to [1, nh, hd, S] if GQA
    let kv_rsh = p.next_var("kvrs");
    p.emit_tensor_const(&kv_rsh, &[4], "int32", &format!("[1,{nkv},{hd},{s}]"));

    let k_kv = p.next_var("kkv");
    p.emit_reshape(&k_kv, &[1, nkv, hd, s], &kv_rsh, &k_flat);
    let v_kv = p.next_var("vkv");
    p.emit_reshape(&v_kv, &[1, nkv, hd, s], &kv_rsh, &v_flat);

    // Expand K/V to full head count for GQA
    let (k_h, v_h) = if groups > 1 {
        let k_final = emit_gqa_expand(&mut p, "ke", &k_kv, nkv, nh, hd, s, groups);
        let v_final = emit_gqa_expand(&mut p, "ve", &v_kv, nkv, nh, hd, s, groups);
        (k_final, v_final)
    } else {
        // MHA: K/V already at full head count
        let k_h = p.next_var("kh");
        p.emit_reshape(&k_h, &[1, nh, hd, s], &q_rsh, &k_flat);
        let v_h = p.next_var("vvh");
        p.emit_reshape(&v_h, &[1, nh, hd, s], &q_rsh, &v_flat);
        (k_h, v_h)
    };

    // V transposed for dp computation
    let vt = p.next_var("vt");
    p.emit_transpose(&vt, &[1, nh, s, hd], &perm23, &v_h);

    // Recompute attention: scores = Q @ K^T * scale + mask → softmax
    let scores = p.next_var("scr");
    p.emit_matmul(&scores, &[1, nh, s, s], &mm_false, &mm_false, &qt, &k_h);

    let scale_var = p.next_var("sc");
    p.emit_scalar_const(&scale_var, "fp16", &format!("{scale}"));
    let ss = p.next_var("ss");
    p.emit_mul(&ss, &[1, nh, s, s], &scores, &scale_var);

    let mask_path = "@model_path/weights/mask.bin";
    p.emit_weight_const("mask", &[1, 1, s, s], mask_path);
    let sm = p.next_var("sm");
    p.emit_add(&sm, &[1, nh, s, s], &ss, "mask");

    let ax3 = p.next_var("ax3");
    p.emit_scalar_const(&ax3, "int32", "3");
    let probs_h = p.next_var("ph");
    p.emit_softmax(&probs_h, &[1, nh, s, s], &ax3, &sm);

    // da → [1, nh, hd, S] → [1, nh, S, hd]
    let da_h = p.next_var("dah");
    p.emit_reshape(&da_h, &[1, nh, hd, s], &q_rsh, &da_flat);
    let da_t = p.next_var("dat");
    p.emit_transpose(&da_t, &[1, nh, s, hd], &perm23, &da_h);

    // dV_full = probs^T @ da: [1, nh, S, S]^T @ [1, nh, S, hd] → [1, nh, S, hd]
    let dv_full = p.next_var("dvfl");
    p.emit_matmul(
        &dv_full,
        &[1, nh, s, hd],
        &mm_true,
        &mm_false,
        &probs_h,
        &da_t,
    );

    // dV: transpose [1, nh, S, hd] → [1, nh, hd, S], then reduce groups for GQA
    let dv_t = p.next_var("dvt");
    p.emit_transpose(&dv_t, &[1, nh, hd, s], &perm23, &dv_full);

    let dv_flat = if groups > 1 {
        emit_gqa_reduce(&mut p, "dv", &dv_t, nkv, groups, hd, s, kvd)
    } else {
        let flat_rsh = p.next_var("frs");
        p.emit_tensor_const(&flat_rsh, &[4], "int32", &format!("[1,{kvd},1,{s}]"));
        let dv_out = p.next_var("dvf");
        p.emit_reshape(&dv_out, &[1, kvd, 1, s], &flat_rsh, &dv_t);
        dv_out
    };

    // dp = da @ V^T: [1, nh, S, hd] @ [1, nh, hd, S] → [1, nh, S, S]
    let dp_h = p.next_var("dph");
    p.emit_matmul(&dp_h, &[1, nh, s, s], &mm_false, &mm_false, &da_t, &v_h);

    // Reshape probs and dp to flat: [1, nh, S, S] → [1, nh*S, 1, S]
    let score_rsh = p.next_var("srs");
    p.emit_tensor_const(&score_rsh, &[4], "int32", &format!("[1,{score_ch},1,{s}]"));
    let probs_flat = p.next_var("pf");
    p.emit_reshape(&probs_flat, &[1, score_ch, 1, s], &score_rsh, &probs_h);
    let dp_flat = p.next_var("dpf");
    p.emit_reshape(&dp_flat, &[1, score_ch, 1, s], &score_rsh, &dp_h);

    // Concat output: [dV, probs, dp]
    let cat_ax = p.next_var("cax");
    p.emit_scalar_const(&cat_ax, "int32", "1");
    let cat_il = p.next_var("cil");
    p.emit_scalar_const(&cat_il, "bool", "false");
    let out = p.next_var("out");
    p.emit_concat(
        &out,
        &[1, out_ch, 1, s],
        &cat_ax,
        &cat_il,
        &[&dv_flat, &probs_flat, &dp_flat],
    );

    let mil_text = p.finalize(&out);

    let mut static_weights = WeightDict::new();
    static_weights.add(mask_path, build_causal_mask(s));

    DynamicKernelOutput {
        mil_text,
        static_weights,
        input_layout: SpatialLayout {
            ic: in_ch,
            seq_len: s,
            total_spatial: s,
            oc: in_ch,
            out_spatial: s,
        },
        output_layout: SpatialLayout {
            ic: out_ch,
            seq_len: s,
            total_spatial: s,
            oc: out_ch,
            out_spatial: s,
        },
    }
}

// ============================================================================
// Kernel 8: SDPA Backward Part 2 (pure computation, weight-free)
// ============================================================================

/// Generate the SDPA backward part 2 kernel (weight-free).
///
/// Input: `[1, 2*SCORE_CH + Q_DIM + KV_DIM, 1, SEQ]` fp16
/// - concat(probs, dp, Q(q_dim), K(kv_dim))
///
/// Output: `[1, Q_DIM + KV_DIM, 1, SEQ]` fp16
/// - concat(dQ(q_dim), dK(kv_dim))
///
/// Computes softmax backward: `dS = probs * (dp - sum(probs*dp, axis=-1))`
/// Then: `dQ = dS @ K_expanded`, `dK_full = dS^T @ Q`, reduced for GQA.
///
/// Supports both MHA and GQA. For GQA, K is expanded via tile and
/// dK_full is reduced (group-summed) back to `n_kv_heads` dimension.
pub fn gen_dynamic_sdpa_bwd2(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let qd = c.q_dim(); // n_heads * head_dim
    let kvd = c.kv_dim(); // n_kv_heads * head_dim
    let s = c.seq_len;
    let nh = c.n_heads;
    let nkv = c.n_kv_heads;
    let groups = c.n_groups();
    let hd = c.head_dim;
    let score_ch = nh * s;
    let in_ch = 2 * score_ch + qd + kvd;
    let out_ch = qd + kvd;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut p = MilProgram::new(in_ch, s);

    // Slice inputs: probs(score_ch), dp(score_ch), Q(qd), K(kvd)
    let sb = |p: &mut MilProgram, name: &str, ch_off: usize, ch: usize| -> String {
        let begin = p.next_var(&format!("{name}_b"));
        p.emit_tensor_const(&begin, &[4], "int32", &format!("[0,{ch_off},0,0]"));
        let size = p.next_var(&format!("{name}_s"));
        p.emit_tensor_const(&size, &[4], "int32", &format!("[1,{ch},1,{s}]"));
        let out = p.next_var(name);
        p.emit_slice_by_size(&out, &[1, ch, 1, s], "x", &begin, &size);
        out
    };

    let probs_flat = sb(&mut p, "pf", 0, score_ch);
    let dp_flat = sb(&mut p, "dpf", score_ch, score_ch);
    let q_flat = sb(&mut p, "qf", 2 * score_ch, qd);
    let k_flat = sb(&mut p, "kf", 2 * score_ch + qd, kvd);

    // Reshape score matrices to multi-head
    let score_rsh = p.next_var("srs");
    p.emit_tensor_const(&score_rsh, &[4], "int32", &format!("[1,{nh},{s},{s}]"));
    let q_rsh = p.next_var("qrs");
    p.emit_tensor_const(&q_rsh, &[4], "int32", &format!("[1,{nh},{hd},{s}]"));
    let perm23 = p.next_var("p23");
    p.emit_tensor_const(&perm23, &[4], "int32", "[0,1,3,2]");
    let mm_false = p.next_var("mmf");
    p.emit_scalar_const(&mm_false, "bool", "false");
    let mm_true = p.next_var("mmt");
    p.emit_scalar_const(&mm_true, "bool", "true");

    let probs_h = p.next_var("ph");
    p.emit_reshape(&probs_h, &[1, nh, s, s], &score_rsh, &probs_flat);
    let dp_h = p.next_var("dph");
    p.emit_reshape(&dp_h, &[1, nh, s, s], &score_rsh, &dp_flat);

    // Q → [1, nh, hd, S] → [1, nh, S, hd]
    let q_h = p.next_var("qh");
    p.emit_reshape(&q_h, &[1, nh, hd, s], &q_rsh, &q_flat);
    let qt = p.next_var("qt");
    p.emit_transpose(&qt, &[1, nh, s, hd], &perm23, &q_h);

    // K: reshape to [1, nkv, hd, S], then expand to [1, nh, hd, S] if GQA
    let kv_rsh = p.next_var("kvrs");
    p.emit_tensor_const(&kv_rsh, &[4], "int32", &format!("[1,{nkv},{hd},{s}]"));
    let k_kv = p.next_var("kkv");
    p.emit_reshape(&k_kv, &[1, nkv, hd, s], &kv_rsh, &k_flat);

    let k_h = if groups > 1 {
        emit_gqa_expand(&mut p, "ke", &k_kv, nkv, nh, hd, s, groups)
    } else {
        let k_h = p.next_var("kh");
        p.emit_reshape(&k_h, &[1, nh, hd, s], &q_rsh, &k_flat);
        k_h
    };

    // Softmax backward: dS = probs * (dp - sum(probs * dp, axis=-1))
    let pd = p.next_var("pd");
    p.emit_mul(&pd, &[1, nh, s, s], &probs_h, &dp_h);

    let ax3 = p.next_var("ax3");
    p.emit_tensor_const(&ax3, &[1], "int32", "[3]");
    let kd_true = p.next_var("kdt");
    p.emit_scalar_const(&kd_true, "bool", "true");
    let sum_pd = p.next_var("spd");
    p.emit_reduce_sum(&sum_pd, &[1, nh, s, 1], &pd, &ax3, &kd_true);

    let dp_sub = p.next_var("dps");
    p.emit_sub(&dp_sub, &[1, nh, s, s], &dp_h, &sum_pd);

    let ds_raw = p.next_var("dsr");
    p.emit_mul(&ds_raw, &[1, nh, s, s], &probs_h, &dp_sub);

    let scale_var = p.next_var("sc");
    p.emit_scalar_const(&scale_var, "fp16", &format!("{scale}"));
    let ds = p.next_var("ds");
    p.emit_mul(&ds, &[1, nh, s, s], &ds_raw, &scale_var);

    // K expanded → transpose for dQ computation
    let kt = p.next_var("kt");
    p.emit_transpose(&kt, &[1, nh, s, hd], &perm23, &k_h);

    // dQ = dS @ K_t: [1, nh, S, S] @ [1, nh, S, hd] → [1, nh, S, hd]
    let dq_h = p.next_var("dqh");
    p.emit_matmul(&dq_h, &[1, nh, s, hd], &mm_false, &mm_false, &ds, &kt);

    // dK_full = dS^T @ Q: [1, nh, S, S]^T @ [1, nh, S, hd] → [1, nh, S, hd]
    let dk_full = p.next_var("dkfl");
    p.emit_matmul(&dk_full, &[1, nh, s, hd], &mm_true, &mm_false, &ds, &qt);

    // Reshape dQ to flat: [1, nh, S, hd] → [1, nh, hd, S] → [1, Q_DIM, 1, S]
    let dq_t = p.next_var("dqt");
    p.emit_transpose(&dq_t, &[1, nh, hd, s], &perm23, &dq_h);
    let flat_rsh_q = p.next_var("frsq");
    p.emit_tensor_const(&flat_rsh_q, &[4], "int32", &format!("[1,{qd},1,{s}]"));
    let dq_flat = p.next_var("dqf");
    p.emit_reshape(&dq_flat, &[1, qd, 1, s], &flat_rsh_q, &dq_t);

    // dK: transpose + reduce groups for GQA
    let dk_t = p.next_var("dkt");
    p.emit_transpose(&dk_t, &[1, nh, hd, s], &perm23, &dk_full);

    let dk_flat = if groups > 1 {
        emit_gqa_reduce(&mut p, "dk", &dk_t, nkv, groups, hd, s, kvd)
    } else {
        let flat_rsh_kv = p.next_var("frskv");
        p.emit_tensor_const(&flat_rsh_kv, &[4], "int32", &format!("[1,{kvd},1,{s}]"));
        let dk_out = p.next_var("dkf");
        p.emit_reshape(&dk_out, &[1, kvd, 1, s], &flat_rsh_kv, &dk_t);
        dk_out
    };

    // Concat output: [dQ, dK]
    let cat_ax = p.next_var("cax");
    p.emit_scalar_const(&cat_ax, "int32", "1");
    let cat_il = p.next_var("cil");
    p.emit_scalar_const(&cat_il, "bool", "false");
    let out = p.next_var("out");
    p.emit_concat(
        &out,
        &[1, out_ch, 1, s],
        &cat_ax,
        &cat_il,
        &[&dq_flat, &dk_flat],
    );

    let mil_text = p.finalize(&out);

    DynamicKernelOutput {
        mil_text,
        static_weights: WeightDict::new(),
        input_layout: SpatialLayout {
            ic: in_ch,
            seq_len: s,
            total_spatial: s,
            oc: in_ch,
            out_spatial: s,
        },
        output_layout: SpatialLayout {
            ic: out_ch,
            seq_len: s,
            total_spatial: s,
            oc: out_ch,
            out_spatial: s,
        },
    }
}

// ============================================================================
// Kernel 9: QKV Backward (dynamic Wq^T, Wk^T, Wv^T)
// ============================================================================

/// Generate the dynamic QKV backward Q kernel.
///
/// Input: `[1, Q_DIM, 1, SEQ + DIM]` fp32
/// - `sp[0:SEQ]` = dQ (IC=q_dim channels)
/// - `sp[SEQ:SEQ+DIM]` = Wq^T columns (q_dim→dim)
///
/// Output: `[1, DIM, 1, SEQ]` fp32
/// - dxq = dQ @ Wq^T
///
/// This is the Q-only portion of the QKV backward. For MHA it produces
/// the same result as the combined kernel; for GQA it is paired with
/// `gen_dynamic_qkv_bwd_kv` which handles the different KV channel count.
pub fn gen_dynamic_qkv_bwd_q(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let qd = c.q_dim();
    let s = c.seq_len;
    let sp = s + d;

    let mut p = MilProgram::new_fp32(qd, sp);

    p.emit_cast("x16", &[1, qd, 1, sp], "x", "fp16");

    let dxq = emit_dyn_matmul(&mut p, "wqt", "x16", qd, d, s, 0, s);

    let out32 = p.next_var("out32");
    p.emit_cast(&out32, &[1, d, 1, s], &dxq, "fp32");

    let mil_text = p.finalize(&out32);

    DynamicKernelOutput {
        mil_text,
        static_weights: WeightDict::new(),
        input_layout: SpatialLayout {
            ic: qd,
            seq_len: s,
            total_spatial: sp,
            oc: qd,
            out_spatial: s,
        },
        output_layout: SpatialLayout {
            ic: d,
            seq_len: s,
            total_spatial: s,
            oc: d,
            out_spatial: s,
        },
    }
}

/// Generate the dynamic QKV backward KV kernel.
///
/// Input: `[1, KV_DIM, 1, 2*SEQ + 2*DIM]` fp32
/// - `sp[0:SEQ]` = dK (IC=kv_dim channels)
/// - `sp[SEQ:2*SEQ]` = dV (IC=kv_dim channels)
/// - `sp[2*SEQ:2*SEQ+DIM]` = Wk^T columns (kv_dim→dim)
/// - `sp[2*SEQ+DIM:2*SEQ+2*DIM]` = Wv^T columns (kv_dim→dim)
///
/// Output: `[1, DIM, 1, SEQ]` fp32
/// - dxkv = dK @ Wk^T + dV @ Wv^T
///
/// For MHA (kv_dim == q_dim), this can be combined with `qkv_bwd_q` output
/// via simple addition. For GQA, the IC (kv_dim) differs from Q's IC (q_dim),
/// requiring a separate kernel.
pub fn gen_dynamic_qkv_bwd_kv(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let kvd = c.kv_dim();
    let s = c.seq_len;
    let sp = 2 * s + 2 * d;

    let mut p = MilProgram::new_fp32(kvd, sp);

    p.emit_cast("x16", &[1, kvd, 1, sp], "x", "fp16");

    // dxk = dK[s, kvd] @ Wk^T[kvd, d] → [s, d]
    let dxk = emit_dyn_matmul(&mut p, "wkt", "x16", kvd, d, s, 0, 2 * s);
    // dxv = dV[s, kvd] @ Wv^T[kvd, d] → [s, d]
    let dxv = emit_dyn_matmul(&mut p, "wvt", "x16", kvd, d, s, s, 2 * s + d);

    // dxkv = dxk + dxv
    let dxkv = p.next_var("dxkv");
    p.emit_add(&dxkv, &[1, d, 1, s], &dxk, &dxv);

    let out32 = p.next_var("out32");
    p.emit_cast(&out32, &[1, d, 1, s], &dxkv, "fp32");

    let mil_text = p.finalize(&out32);

    DynamicKernelOutput {
        mil_text,
        static_weights: WeightDict::new(),
        input_layout: SpatialLayout {
            ic: kvd,
            seq_len: s,
            total_spatial: sp,
            oc: kvd,
            out_spatial: s,
        },
        output_layout: SpatialLayout {
            ic: d,
            seq_len: s,
            total_spatial: s,
            oc: d,
            out_spatial: s,
        },
    }
}

/// Generate the combined QKV backward kernel (MHA-only convenience).
///
/// For models where `n_kv_heads == n_heads`, this packs all three gradients
/// into a single kernel with shared IC=q_dim. For GQA models, use the split
/// `gen_dynamic_qkv_bwd_q` + `gen_dynamic_qkv_bwd_kv` instead.
///
/// Input: `[1, Q_DIM, 1, 3*SEQ + 3*DIM]` fp32
/// Output: `[1, DIM, 1, SEQ]` fp32 = dQ@Wq^T + dK@Wk^T + dV@Wv^T
pub fn gen_dynamic_qkv_bwd(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let qd = c.q_dim();
    let s = c.seq_len;
    let sp = 3 * s + 3 * d;

    debug_assert_eq!(
        c.n_kv_heads, c.n_heads,
        "Combined QKV backward requires MHA. Use split qkv_bwd_q + qkv_bwd_kv for GQA."
    );

    let mut p = MilProgram::new_fp32(qd, sp);

    p.emit_cast("x16", &[1, qd, 1, sp], "x", "fp16");

    let dxq = emit_dyn_matmul(&mut p, "wqt", "x16", qd, d, s, 0, 3 * s);
    let dxk = emit_dyn_matmul(&mut p, "wkt", "x16", qd, d, s, s, 3 * s + d);
    let dxv = emit_dyn_matmul(&mut p, "wvt", "x16", qd, d, s, 2 * s, 3 * s + 2 * d);

    let dx12 = p.next_var("dx12");
    p.emit_add(&dx12, &[1, d, 1, s], &dxq, &dxk);
    let dx = p.next_var("dx");
    p.emit_add(&dx, &[1, d, 1, s], &dx12, &dxv);

    let out32 = p.next_var("out32");
    p.emit_cast(&out32, &[1, d, 1, s], &dx, "fp32");

    let mil_text = p.finalize(&out32);

    DynamicKernelOutput {
        mil_text,
        static_weights: WeightDict::new(),
        input_layout: SpatialLayout {
            ic: qd,
            seq_len: s,
            total_spatial: sp,
            oc: qd,
            out_spatial: s,
        },
        output_layout: SpatialLayout {
            ic: d,
            seq_len: s,
            total_spatial: s,
            oc: d,
            out_spatial: s,
        },
    }
}

// ============================================================================
// Kernel 10 (Phase 3): RMSNorm Backward Offload
// ============================================================================

/// Generate the dynamic RMSNorm backward kernel.
///
/// Input: `[1, 2*DIM, 1, SEQ + 1]` fp32
/// - `ch[0:DIM]` sp `[0:SEQ]` = dy
/// - `ch[DIM:2*DIM]` sp `[0:SEQ]` = x
/// - `sp[SEQ]` = RMSNorm weight per channel (packed as 1 spatial column)
///
/// Output: `[1, DIM, 1, SEQ]` fp32 = dx
///
/// Computes: `rrms = 1/sqrt(mean(x^2) + eps)`, then
/// `dx = (dy*w - x * dot(dy*w, x) * rrms^2 / D) * rrms`
pub fn gen_dynamic_rmsnorm_bwd(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let s = c.seq_len;
    let in_ch = 2 * d;
    let sp = s + 1; // 1 spatial column for RMSNorm weight

    let mut p = MilProgram::new_fp32(in_ch, sp);

    p.emit_cast("x16", &[1, in_ch, 1, sp], "x", "fp16");

    // Slice dy: ch[0:D], sp[0:S]
    let dy_begin = p.next_var("dyb");
    p.emit_tensor_const(&dy_begin, &[4], "int32", "[0,0,0,0]");
    let dy_size = p.next_var("dys");
    p.emit_tensor_const(&dy_size, &[4], "int32", &format!("[1,{d},1,{s}]"));
    let dy = p.next_var("dy");
    p.emit_slice_by_size(&dy, &[1, d, 1, s], "x16", &dy_begin, &dy_size);

    // Slice x_in: ch[D:2D], sp[0:S]
    let xin_begin = p.next_var("xib");
    p.emit_tensor_const(&xin_begin, &[4], "int32", &format!("[0,{d},0,0]"));
    let xin_size = p.next_var("xis");
    p.emit_tensor_const(&xin_size, &[4], "int32", &format!("[1,{d},1,{s}]"));
    let x_in = p.next_var("xin");
    p.emit_slice_by_size(&x_in, &[1, d, 1, s], "x16", &xin_begin, &xin_size);

    // Slice RMSNorm weight: ch[0:D], sp[S:S+1] — but only first D channels
    let w_begin = p.next_var("wb");
    p.emit_tensor_const(&w_begin, &[4], "int32", &format!("[0,0,0,{s}]"));
    let w_size = p.next_var("wss");
    p.emit_tensor_const(&w_size, &[4], "int32", &format!("[1,{d},1,1]"));
    let w = p.next_var("rw");
    p.emit_slice_by_size(&w, &[1, d, 1, 1], "x16", &w_begin, &w_size);

    // dy_w = dy * w (broadcast w over spatial)
    let dy_w = p.next_var("dyw");
    p.emit_mul(&dy_w, &[1, d, 1, s], &dy, &w);

    // x_sq = x * x
    let x_sq = p.next_var("xsq");
    p.emit_mul(&x_sq, &[1, d, 1, s], &x_in, &x_in);

    // mean_sq = reduce_sum(x_sq, axis=1) / D → [1, 1, 1, S]
    let ax1 = p.next_var("ax1");
    p.emit_tensor_const(&ax1, &[1], "int32", "[1]");
    let kd_true = p.next_var("kdt");
    p.emit_scalar_const(&kd_true, "bool", "true");
    let ss = p.next_var("ss");
    p.emit_reduce_sum(&ss, &[1, 1, 1, s], &x_sq, &ax1, &kd_true);

    let inv_d = p.next_var("invd");
    p.emit_scalar_const(&inv_d, "fp16", &format!("{}", 1.0 / d as f32));
    let mean_sq = p.next_var("msq");
    p.emit_mul(&mean_sq, &[1, 1, 1, s], &ss, &inv_d);

    // rrms = pow(mean_sq + eps, -0.5)
    let eps_var = p.next_var("eps");
    p.emit_scalar_const(&eps_var, "fp16", &format!("{}", c.rms_norm_eps));
    let ms_eps = p.next_var("mse");
    p.emit_add(&ms_eps, &[1, 1, 1, s], &mean_sq, &eps_var);
    let neg_half = p.next_var("nh");
    p.emit_scalar_const(&neg_half, "fp16", "-0.5");
    let rrms = p.next_var("rrms");
    p.emit_pow(&rrms, &[1, 1, 1, s], &ms_eps, &neg_half);

    // dot = reduce_sum(dy_w * x, axis=1) → [1, 1, 1, S]
    let dyw_x = p.next_var("dwx");
    p.emit_mul(&dyw_x, &[1, d, 1, s], &dy_w, &x_in);
    let dot = p.next_var("dot");
    p.emit_reduce_sum(&dot, &[1, 1, 1, s], &dyw_x, &ax1, &kd_true);

    // correction = x * dot * rrms^2 / D = x * dot * inv_d / (mean_sq + eps)
    // Actually: dx = (dy_w - x * dot * inv_d * rrms^2) * rrms
    // = (dy_w - x * dot / D / (mean_sq + eps)) * rrms
    let dot_invd = p.next_var("did");
    p.emit_mul(&dot_invd, &[1, 1, 1, s], &dot, &inv_d);

    // rrms2 = rrms * rrms = 1 / (mean_sq + eps)
    let rrms2 = p.next_var("rr2");
    p.emit_mul(&rrms2, &[1, 1, 1, s], &rrms, &rrms);

    let corr_sc = p.next_var("crs");
    p.emit_mul(&corr_sc, &[1, 1, 1, s], &dot_invd, &rrms2);

    let corr = p.next_var("corr");
    p.emit_mul(&corr, &[1, d, 1, s], &x_in, &corr_sc);

    // dx = (dy_w - corr) * rrms
    let diff = p.next_var("diff");
    p.emit_sub(&diff, &[1, d, 1, s], &dy_w, &corr);

    let dx16 = p.next_var("dx16");
    p.emit_mul(&dx16, &[1, d, 1, s], &diff, &rrms);

    // Cast to fp32
    let dx32 = p.next_var("dx32");
    p.emit_cast(&dx32, &[1, d, 1, s], &dx16, "fp32");

    let mil_text = p.finalize(&dx32);

    DynamicKernelOutput {
        mil_text,
        static_weights: WeightDict::new(),
        input_layout: SpatialLayout {
            ic: in_ch,
            seq_len: s,
            total_spatial: sp,
            oc: d,
            out_spatial: s,
        },
        output_layout: SpatialLayout {
            ic: d,
            seq_len: s,
            total_spatial: s,
            oc: d,
            out_spatial: s,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> DynamicKernelConfig {
        DynamicKernelConfig::new(TransformerKernelConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 16,
            seq_len: 32,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        })
    }

    #[test]
    fn test_dynamic_sdpa_fwd_generates_valid_mil() {
        let dkc = test_config();
        let out = gen_dynamic_sdpa_fwd(&dkc);
        assert!(out.mil_text.contains("program(1.3)"));
        assert!(out.mil_text.contains("tensor<fp32"));
        assert!(out.mil_text.contains("cast(dtype=string(\"fp16\")"));
        assert!(out.mil_text.contains("cast(dtype=string(\"fp32\")"));
        assert!(out.mil_text.contains("matmul("));
        assert!(out.mil_text.contains("softmax("));
        assert!(out.mil_text.contains("BLOBFILE")); // causal mask
        assert_eq!(out.input_layout.ic, 64);
        assert_eq!(out.input_layout.total_spatial, 32 + 4 * 64);
    }

    #[test]
    fn test_dynamic_ffn_w13_generates_valid_mil() {
        let dkc = test_config();
        let out = gen_dynamic_ffn_w13(&dkc);
        assert!(out.mil_text.contains("program(1.3)"));
        assert!(out.mil_text.contains("sigmoid("));
        assert_eq!(out.input_layout.ic, 64);
        assert_eq!(out.input_layout.total_spatial, 32 + 2 * 128);
        assert_eq!(out.output_layout.ic, 3 * 128);
    }

    #[test]
    fn test_dynamic_ffn_w2_generates_valid_mil() {
        let dkc = test_config();
        let out = gen_dynamic_ffn_w2(&dkc);
        assert!(out.mil_text.contains("program(1.3)"));
        assert_eq!(out.input_layout.ic, 128);
        assert_eq!(out.input_layout.total_spatial, 32 + 64);
    }

    #[test]
    fn test_dynamic_ffn_bwd_w2t() {
        let dkc = test_config();
        let out = gen_dynamic_ffn_bwd_w2t(&dkc);
        assert!(out.mil_text.contains("program(1.3)"));
        assert_eq!(out.output_layout.ic, 128);
    }

    #[test]
    fn test_dynamic_ffn_bwd_w13t() {
        let dkc = test_config();
        let out = gen_dynamic_ffn_bwd_w13t(&dkc);
        assert!(out.mil_text.contains("add("));
        assert_eq!(out.input_layout.total_spatial, 2 * 32 + 2 * 64);
    }

    #[test]
    fn test_dynamic_wo_bwd() {
        let dkc = test_config();
        let out = gen_dynamic_wo_bwd(&dkc);
        assert!(out.mil_text.contains("matmul("));
        assert_eq!(out.input_layout.total_spatial, 32 + 64);
    }

    #[test]
    fn test_dynamic_sdpa_bwd1() {
        let dkc = test_config();
        let out = gen_dynamic_sdpa_bwd1(&dkc);
        assert!(out.mil_text.contains("softmax("));
        assert!(out.mil_text.contains("BLOBFILE")); // causal mask
        assert_eq!(out.input_layout.ic, 4 * 64);
    }

    #[test]
    fn test_dynamic_sdpa_bwd2() {
        let dkc = test_config();
        let out = gen_dynamic_sdpa_bwd2(&dkc);
        assert!(out.mil_text.contains("reduce_sum("));
        assert_eq!(out.output_layout.ic, 2 * 64);
    }

    #[test]
    fn test_dynamic_qkv_bwd() {
        let dkc = test_config();
        let out = gen_dynamic_qkv_bwd(&dkc);
        assert!(out.mil_text.contains("add("));
        assert_eq!(out.input_layout.total_spatial, 3 * 32 + 3 * 64);
        assert_eq!(out.output_layout.ic, 64);
    }

    #[test]
    fn test_dynamic_rmsnorm_bwd() {
        let dkc = test_config();
        let out = gen_dynamic_rmsnorm_bwd(&dkc);
        assert!(out.mil_text.contains("pow("));
        assert!(out.mil_text.contains("reduce_sum("));
        assert_eq!(out.input_layout.ic, 2 * 64);
        assert_eq!(out.input_layout.total_spatial, 32 + 1);
    }

    // ================================================================
    // Non-standard head_dim tests (Qwen3-style: q_dim != dim)
    // ================================================================

    /// Qwen3-0.6B style config: dim=1024, n_heads=16, head_dim=128 → q_dim=2048.
    fn qwen3_config() -> DynamicKernelConfig {
        DynamicKernelConfig::new(TransformerKernelConfig {
            dim: 64, // small dim for fast tests
            hidden_dim: 128,
            n_heads: 4,
            n_kv_heads: 4, // MHA
            head_dim: 32,  // head_dim > dim/n_heads (64/4=16), so q_dim = 4*32 = 128 != 64
            seq_len: 16,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        })
    }

    #[test]
    fn test_qwen3_dynamic_config_invariant() {
        let dkc = qwen3_config();
        let c = &dkc.cfg;
        assert_eq!(c.q_dim(), 128); // 4 * 32
        assert_eq!(c.kv_dim(), 128); // 4 * 32 (MHA)
        assert_ne!(c.q_dim(), c.dim); // Key: q_dim (128) != dim (64)
    }

    #[test]
    fn test_qwen3_dynamic_sdpa_fwd() {
        let dkc = qwen3_config();
        let out = gen_dynamic_sdpa_fwd(&dkc);
        assert!(out.mil_text.contains("program(1.3)"));
        // IC = dim (64), spatial includes weight columns for Wq(q_dim=128) etc.
        assert_eq!(out.input_layout.ic, 64);
    }

    #[test]
    fn test_qwen3_dynamic_wo_bwd() {
        let dkc = qwen3_config();
        let qd = dkc.cfg.q_dim(); // 128
        let d = dkc.cfg.dim; // 64
        let s = dkc.cfg.seq_len; // 16

        let out = gen_dynamic_wo_bwd(&dkc);
        assert!(out.mil_text.contains("matmul("));
        // Input: [1, dim, 1, s + q_dim]
        assert_eq!(out.input_layout.ic, d);
        assert_eq!(out.input_layout.total_spatial, s + qd); // 16 + 128 = 144
        // Output: [1, q_dim, 1, s]
        assert_eq!(out.output_layout.ic, qd);
        assert_eq!(out.output_layout.out_spatial, s);
    }

    #[test]
    fn test_qwen3_dynamic_sdpa_bwd1() {
        let dkc = qwen3_config();
        let qd = dkc.cfg.q_dim(); // 128
        let kvd = dkc.cfg.kv_dim(); // 128 (MHA)
        let score_ch = dkc.cfg.n_heads * dkc.cfg.seq_len; // 4 * 16 = 64

        let out = gen_dynamic_sdpa_bwd1(&dkc);
        assert!(out.mil_text.contains("softmax("));
        // Input IC = 2*qd + 2*kvd = 512
        assert_eq!(out.input_layout.ic, 2 * qd + 2 * kvd);
        // Output IC = kvd + 2*score_ch = 128 + 128 = 256
        assert_eq!(out.output_layout.ic, kvd + 2 * score_ch);
    }

    #[test]
    fn test_qwen3_dynamic_sdpa_bwd2() {
        let dkc = qwen3_config();
        let qd = dkc.cfg.q_dim();
        let kvd = dkc.cfg.kv_dim();
        let score_ch = dkc.cfg.n_heads * dkc.cfg.seq_len;

        let out = gen_dynamic_sdpa_bwd2(&dkc);
        assert!(out.mil_text.contains("reduce_sum("));
        // Input IC = 2*score_ch + qd + kvd
        assert_eq!(out.input_layout.ic, 2 * score_ch + qd + kvd);
        // Output IC = qd + kvd
        assert_eq!(out.output_layout.ic, qd + kvd);
    }

    #[test]
    fn test_qwen3_dynamic_qkv_bwd() {
        let dkc = qwen3_config();
        let qd = dkc.cfg.q_dim(); // 128
        let d = dkc.cfg.dim; // 64
        let s = dkc.cfg.seq_len; // 16

        let out = gen_dynamic_qkv_bwd(&dkc);
        assert!(out.mil_text.contains("add("));
        // Input IC = q_dim (128, not dim=64!)
        assert_eq!(out.input_layout.ic, qd);
        // Spatial: 3*s + 3*d (weights have d output columns each)
        assert_eq!(out.input_layout.total_spatial, 3 * s + 3 * d);
        // Output IC = dim (64)
        assert_eq!(out.output_layout.ic, d);
    }

    // ================================================================
    // GQA tests (n_kv_heads != n_heads)
    // ================================================================

    /// GQA config: n_heads=4, n_kv_heads=2 (groups=2), head_dim=16.
    /// q_dim = 4*16 = 64, kv_dim = 2*16 = 32.
    fn gqa_config() -> DynamicKernelConfig {
        DynamicKernelConfig::new(TransformerKernelConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            n_kv_heads: 2, // GQA: 2 KV heads, 4 Q heads
            head_dim: 16,
            seq_len: 32,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        })
    }

    #[test]
    fn test_gqa_config_invariants() {
        let dkc = gqa_config();
        let c = &dkc.cfg;
        assert_eq!(c.q_dim(), 64);  // 4 * 16
        assert_eq!(c.kv_dim(), 32); // 2 * 16
        assert_eq!(c.n_groups(), 2);
        assert_ne!(c.q_dim(), c.kv_dim()); // Key: GQA means q_dim != kv_dim
    }

    #[test]
    fn test_gqa_sdpa_fwd() {
        let dkc = gqa_config();
        let qd = dkc.cfg.q_dim();  // 64
        let kvd = dkc.cfg.kv_dim(); // 32
        let d = dkc.cfg.dim;        // 64
        let s = dkc.cfg.seq_len;    // 32

        let out = gen_dynamic_sdpa_fwd(&dkc);
        assert!(out.mil_text.contains("program(1.3)"));
        // Should have concat for GQA K/V expansion in forward
        assert!(out.mil_text.contains("concat("));
        // IC is dim
        assert_eq!(out.input_layout.ic, d);
        // Spatial: s + 2*qd + 2*kvd = 32 + 128 + 64 = 224
        assert_eq!(out.input_layout.total_spatial, s + 2 * qd + 2 * kvd);
        // Output channels: 2*d + 2*qd + 2*kvd = 128 + 128 + 64 = 320
        assert_eq!(out.output_layout.ic, 2 * d + 2 * qd + 2 * kvd);
    }

    #[test]
    fn test_gqa_sdpa_bwd1() {
        let dkc = gqa_config();
        let qd = dkc.cfg.q_dim();  // 64
        let kvd = dkc.cfg.kv_dim(); // 32
        let score_ch = dkc.cfg.n_heads * dkc.cfg.seq_len; // 4 * 32 = 128

        let out = gen_dynamic_sdpa_bwd1(&dkc);
        // Should have concat for GQA K/V expansion
        assert!(out.mil_text.contains("concat("));
        // Should have reduce_sum for dV group reduction
        assert!(out.mil_text.contains("reduce_sum("));
        // IC = 2*qd + 2*kvd = 2*64 + 2*32 = 192
        assert_eq!(out.input_layout.ic, 2 * qd + 2 * kvd);
        // Output IC = kvd + 2*score_ch = 32 + 256 = 288
        assert_eq!(out.output_layout.ic, kvd + 2 * score_ch);
    }

    #[test]
    fn test_gqa_sdpa_bwd2() {
        let dkc = gqa_config();
        let qd = dkc.cfg.q_dim();  // 64
        let kvd = dkc.cfg.kv_dim(); // 32
        let score_ch = dkc.cfg.n_heads * dkc.cfg.seq_len; // 4 * 32 = 128

        let out = gen_dynamic_sdpa_bwd2(&dkc);
        // Should have concat for GQA K expansion
        assert!(out.mil_text.contains("concat("));
        // Should have reduce_sum for dK group reduction
        assert!(out.mil_text.contains("reduce_sum("));
        // Input IC = 2*score_ch + qd + kvd = 256 + 64 + 32 = 352
        assert_eq!(out.input_layout.ic, 2 * score_ch + qd + kvd);
        // Output IC = qd + kvd = 64 + 32 = 96
        assert_eq!(out.output_layout.ic, qd + kvd);
    }

    #[test]
    fn test_gqa_qkv_bwd_q() {
        let dkc = gqa_config();
        let qd = dkc.cfg.q_dim(); // 64
        let d = dkc.cfg.dim;      // 64
        let s = dkc.cfg.seq_len;  // 32

        let out = gen_dynamic_qkv_bwd_q(&dkc);
        assert!(out.mil_text.contains("matmul("));
        // Input IC = q_dim (64)
        assert_eq!(out.input_layout.ic, qd);
        // Spatial: s + d = 32 + 64 = 96
        assert_eq!(out.input_layout.total_spatial, s + d);
        // Output IC = dim (64)
        assert_eq!(out.output_layout.ic, d);
    }

    #[test]
    fn test_gqa_qkv_bwd_kv() {
        let dkc = gqa_config();
        let kvd = dkc.cfg.kv_dim(); // 32
        let d = dkc.cfg.dim;        // 64
        let s = dkc.cfg.seq_len;    // 32

        let out = gen_dynamic_qkv_bwd_kv(&dkc);
        assert!(out.mil_text.contains("add("));
        // Input IC = kv_dim (32)
        assert_eq!(out.input_layout.ic, kvd);
        // Spatial: 2*s + 2*d = 64 + 128 = 192
        assert_eq!(out.input_layout.total_spatial, 2 * s + 2 * d);
        // Output IC = dim (64)
        assert_eq!(out.output_layout.ic, d);
    }

    #[test]
    fn test_gqa_split_kernels_valid_mil() {
        let dkc = gqa_config();

        // Both split kernels should produce valid MIL programs
        let q_out = gen_dynamic_qkv_bwd_q(&dkc);
        assert!(q_out.mil_text.contains("program(1.3)"));
        assert!(q_out.mil_text.contains("cast(dtype=string(\"fp16\")"));
        assert!(q_out.mil_text.contains("cast(dtype=string(\"fp32\")"));

        let kv_out = gen_dynamic_qkv_bwd_kv(&dkc);
        assert!(kv_out.mil_text.contains("program(1.3)"));
        assert!(kv_out.mil_text.contains("cast(dtype=string(\"fp16\")"));
        assert!(kv_out.mil_text.contains("cast(dtype=string(\"fp32\")"));

        // Both produce the same output dim
        assert_eq!(q_out.output_layout.ic, kv_out.output_layout.ic);
        assert_eq!(q_out.output_layout.ic, dkc.cfg.dim);
    }

    #[test]
    #[should_panic(expected = "Combined QKV backward requires MHA")]
    fn test_gqa_combined_qkv_bwd_panics() {
        let dkc = gqa_config();
        // Combined kernel should panic for GQA configs (debug_assert)
        let _ = gen_dynamic_qkv_bwd(&dkc);
    }

    // ── Tests for decomposed single-projection kernels ──

    #[test]
    fn test_dynamic_projection_basic() {
        let out = gen_dynamic_projection(64, 128, 16);
        assert!(out.mil_text.contains("program(1.3)"));
        assert!(out.mil_text.contains("matmul("));
        // fp32→fp16→matmul→fp32
        assert!(out.mil_text.contains("cast(dtype=string(\"fp16\")"));
        assert!(out.mil_text.contains("cast(dtype=string(\"fp32\")"));
        // Input: [1, IC=64, 1, S+OC=16+128=144]
        assert_eq!(out.input_layout.ic, 64);
        assert_eq!(out.input_layout.total_spatial, 16 + 128);
        // Output: [1, OC=128, 1, S=16]
        assert_eq!(out.output_layout.ic, 128);
        assert_eq!(out.output_layout.out_spatial, 16);
    }

    #[test]
    fn test_dynamic_projection_qwen3_shapes() {
        // Qwen3-0.6B: dim=1024, q_dim=2048, kv_dim=1024, hidden=3072
        let shapes = [(1024, 2048), (1024, 1024), (2048, 1024), (1024, 3072), (3072, 1024)];
        let seq = 64;
        for (ic, oc) in shapes {
            let out = gen_dynamic_projection(ic, oc, seq);
            assert!(out.mil_text.contains("program(1.3)"));
            assert_eq!(out.input_layout.ic, ic);
            assert_eq!(out.input_layout.total_spatial, seq + oc);
            assert_eq!(out.output_layout.ic, oc);
            assert_eq!(out.output_layout.out_spatial, seq);
            assert!(out.static_weights.entries.is_empty());
        }
    }

    #[test]
    fn test_dynamic_sdpa_attn_mha() {
        let dkc = test_config();
        let s = dkc.cfg.seq_len; // 16
        let qd = dkc.cfg.q_dim(); // 64 (MHA: n_heads * head_dim = 4 * 16)
        let kvd = dkc.cfg.kv_dim(); // 64

        let out = gen_dynamic_sdpa_attn(&dkc);
        assert!(out.mil_text.contains("program(1.3)"));
        assert!(out.mil_text.contains("matmul("));
        assert!(out.mil_text.contains("softmax("));
        // Input: [1, QD + 2*KVD, 1, S]
        assert_eq!(out.input_layout.ic, qd + 2 * kvd);
        assert_eq!(out.input_layout.total_spatial, s);
        // Output: [1, QD, 1, S]
        assert_eq!(out.output_layout.ic, qd);
        assert_eq!(out.output_layout.out_spatial, s);
        // Should have causal mask static weight
        assert!(!out.static_weights.entries.is_empty());
    }

    #[test]
    fn test_dynamic_sdpa_attn_gqa() {
        let dkc = gqa_config();
        let qd = dkc.cfg.q_dim(); // 64
        let kvd = dkc.cfg.kv_dim(); // 32
        let s = dkc.cfg.seq_len; // 32

        let out = gen_dynamic_sdpa_attn(&dkc);
        assert!(out.mil_text.contains("program(1.3)"));
        assert!(out.mil_text.contains("concat(")); // GQA expansion
        // Input: [1, QD + 2*KVD, 1, S] = [1, 128, 1, 32]
        assert_eq!(out.input_layout.ic, qd + 2 * kvd);
        assert_eq!(out.input_layout.total_spatial, s);
        // Output: [1, QD, 1, S] = [1, 64, 1, 32]
        assert_eq!(out.output_layout.ic, qd);
        assert_eq!(out.output_layout.out_spatial, s);
    }
}
