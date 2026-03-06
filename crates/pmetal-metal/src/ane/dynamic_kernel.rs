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
//! | 1 | `sdpa_fwd` | DIM | SEQ + 1 + 2*Q_DIM + 2*KV_DIM | x, rms_w, Wq, Wk, Wv, Wo |
//! | 2 | `ffn_w13` | DIM | SEQ + 1 + 2*HIDDEN | x, rms_w, W1, W3 (+ SiLU/gate inside) |
//! | 3 | `ffn_w2` | HIDDEN | SEQ + DIM | gate, W2 |
//! | 4 | `ffn_bwd_w2t` | DIM | SEQ + HIDDEN | dffn, W2^T |
//! | 5 | `ffn_bwd_w13t` | HIDDEN | 2*SEQ + 2*DIM | dh1, dh3, W1^T, W3^T |
//! | 6 | `wo_bwd` | DIM | SEQ + Q_DIM | dy, Wo^T (deprecated by fusion) |
//! | 7 | `sdpa_bwd1` | Q_DIM+2*KV_DIM+DIM | SEQ + Q_DIM | Q, K, V, dy, Wo^T (Fused) |
//! | 8 | `sdpa_bwd2` | 2*SCORE+Q_DIM+KV_DIM | SEQ | probs, dp, Q, K (weight-free) |
//! | 9a | `qkv_bwd_q` | Q_DIM | SEQ + DIM | dQ, Wq^T |
//! | 9b | `qkv_bwd_kv` | KV_DIM | 2*SEQ + 2*DIM | dK, dV, Wk^T, Wv^T |
//! | 9 | `qkv_bwd` | Q_DIM | 3*SEQ + 3*DIM | dQ, dK, dV, Wq^T, Wk^T, Wv^T (MHA only) |
//! | 10 | `rmsnorm_bwd` | 2*DIM | SEQ + 1 | dy, x, RMSNorm weight (1 spatial col) |
//! | 11 | `rmsnorm_fwd` | DIM | SEQ + 1 | x, RMSNorm weight (1 spatial col) |
//! | 12 | `softmax` | VOCAB | SEQ | None (weight-free, fp16 in/out) |
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
// Helpers: MIL program fragments
// ============================================================================

/// Helper for fused RMSNorm.
fn emit_rmsnorm_fuse(
    p: &mut MilProgram,
    prefix: &str,
    x_in: &str,
    w_in: &str,
    d: usize,
    s: usize,
    eps_val: f32,
) -> String {
    let sq = p.next_var(&format!("{prefix}_sq"));
    p.emit_mul(&sq, &[1, d, 1, s], x_in, x_in);

    let rax = p.next_var(&format!("{prefix}_rax"));
    p.emit_tensor_const(&rax, &[1], "int32", "[1]");
    let kd = p.next_var(&format!("{prefix}_kd"));
    p.emit_scalar_const(&kd, "bool", "true");
    let ss = p.next_var(&format!("{prefix}_ss"));
    p.emit_reduce_sum(&ss, &[1, 1, 1, s], &sq, &rax, &kd);

    let inv_d = p.next_var(&format!("{prefix}_id"));
    let inv_d_val = 1.0 / (d as f32);
    p.emit_scalar_const(&inv_d, "fp16", &format!("{inv_d_val}"));
    let ss2 = p.next_var(&format!("{prefix}_ss2"));
    p.emit_mul(&ss2, &[1, 1, 1, s], &ss, &inv_d);

    let eps = p.next_var(&format!("{prefix}_eps"));
    p.emit_scalar_const(&eps, "fp16", &format!("{eps_val}"));
    let ss3 = p.next_var(&format!("{prefix}_ss3"));
    p.emit_add(&ss3, &[1, 1, 1, s], &ss2, &eps);

    let nhalf = p.next_var(&format!("{prefix}_nh"));
    p.emit_scalar_const(&nhalf, "fp16", "-0.5");
    let rrms = p.next_var(&format!("{prefix}_rrm"));
    p.emit_pow(&rrms, &[1, 1, 1, s], &ss3, &nhalf);

    let xr = p.next_var(&format!("{prefix}_xr"));
    p.emit_mul(&xr, &[1, d, 1, s], x_in, &rrms);

    let xn = p.next_var(&format!("{prefix}_xn"));
    p.emit_mul(&xn, &[1, d, 1, s], &xr, w_in);

    xn
}

/// Helper for matmul using pre-computed activations and sliced weights from spatial dimension.
#[allow(clippy::too_many_arguments)]
fn emit_dyn_matmul_with_act(
    p: &mut MilProgram,
    prefix: &str,
    act: &str,
    input_full: &str,
    ic: usize,
    oc: usize,
    seq: usize,
    w_sp_off: usize,
) -> String {
    let w_begin = p.next_var(&format!("{prefix}_wb"));
    p.emit_tensor_const(&w_begin, &[4], "int32", &format!("[0,0,0,{}]", w_sp_off));
    let w_size = p.next_var(&format!("{prefix}_ws"));
    p.emit_tensor_const(&w_size, &[4], "int32", &format!("[1,{ic},1,{oc}]"));
    let w = p.next_var(&format!("{prefix}_w"));
    p.emit_slice_by_size(&w, &[1, ic, 1, oc], input_full, &w_begin, &w_size);

    let rsh1 = p.next_var(&format!("{prefix}_rs1"));
    p.emit_tensor_const(&rsh1, &[4], "int32", &format!("[1,1,{ic},{seq}]"));
    let act_r = p.next_var(&format!("{prefix}_ar"));
    p.emit_reshape(&act_r, &[1, 1, ic, seq], &rsh1, act);

    let perm_23 = p.next_var(&format!("{prefix}_p23"));
    p.emit_tensor_const(&perm_23, &[4], "int32", "[0,1,3,2]");
    let act_t = p.next_var(&format!("{prefix}_at"));
    p.emit_transpose(&act_t, &[1, 1, seq, ic], &perm_23, &act_r);

    let rsh2 = p.next_var(&format!("{prefix}_rs2"));
    p.emit_tensor_const(&rsh2, &[4], "int32", &format!("[1,1,{ic},{oc}]"));
    let w_r = p.next_var(&format!("{prefix}_wr"));
    p.emit_reshape(&w_r, &[1, 1, ic, oc], &rsh2, &w);

    let mm_false = p.next_var(&format!("{prefix}_mf"));
    p.emit_scalar_const(&mm_false, "bool", "false");
    let mm = p.next_var(&format!("{prefix}_mm"));
    p.emit_matmul(&mm, &[1, 1, seq, oc], &mm_false, &mm_false, &act_t, &w_r);

    let perm_back = p.next_var(&format!("{prefix}_pb"));
    p.emit_tensor_const(&perm_back, &[4], "int32", "[0,1,3,2]");
    let mm_t = p.next_var(&format!("{prefix}_mt"));
    p.emit_transpose(&mm_t, &[1, 1, oc, seq], &perm_back, &mm);

    let rsh3 = p.next_var(&format!("{prefix}_rs3"));
    p.emit_tensor_const(&rsh3, &[4], "int32", &format!("[1,{oc},1,{seq}]"));
    let out = p.next_var(&format!("{prefix}_out"));
    p.emit_reshape(&out, &[1, oc, 1, seq], &rsh3, &mm_t);

    out
}

/// Helper for matmul where input activations are at a specific channel offset.
#[allow(clippy::too_many_arguments)]
fn emit_dyn_matmul_at_ch(
    p: &mut MilProgram,
    prefix: &str,
    input: &str,
    ic: usize,
    oc: usize,
    seq: usize,
    act_sp_off: usize,
    w_sp_off: usize,
    act_ch_off: usize,
) -> String {
    // Slice activations: [1, ic, 1, seq] from spatial AND channel offset
    let act_begin = p.next_var(&format!("{prefix}_ab"));
    p.emit_tensor_const(
        &act_begin,
        &[4],
        "int32",
        &format!("[0,{},0,{}]", act_ch_off, act_sp_off),
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

/// Emit MIL ops for a single dynamic matmul within a larger kernel.
#[allow(clippy::too_many_arguments)]
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
#[allow(clippy::too_many_arguments)]
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
    let merge_rsh = p.next_var(&format!("{prefix}_mrs"));
    p.emit_tensor_const(&merge_rsh, &[4], "int32", &format!("[1,{nkv},1,{hds}]"));
    let merged = p.next_var(&format!("{prefix}_mg"));
    p.emit_reshape(&merged, &[1, nkv, 1, hds], &merge_rsh, kv_var);

    let cat_ax = p.next_var(&format!("{prefix}_ca"));
    p.emit_scalar_const(&cat_ax, "int32", "2");
    let cat_il = p.next_var(&format!("{prefix}_ci"));
    p.emit_scalar_const(&cat_il, "bool", "false");
    let copies: Vec<&str> = (0..groups).map(|_| merged.as_str()).collect();
    let expanded = p.next_var(&format!("{prefix}_ex"));
    p.emit_concat(&expanded, &[1, nkv, groups, hds], &cat_ax, &cat_il, &copies);

    let nh_rsh = p.next_var(&format!("{prefix}_nr"));
    p.emit_tensor_const(&nh_rsh, &[4], "int32", &format!("[1,{nh},{hd},{s}]"));
    let result = p.next_var(&format!("{prefix}_fn"));
    p.emit_reshape(&result, &[1, nh, hd, s], &nh_rsh, &expanded);
    result
}

/// Reduce a gradient tensor from [1, nh, hd, S] to [1, nkv, hd, S] for GQA.
#[allow(clippy::too_many_arguments)]
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
    let grp_rsh = p.next_var(&format!("{prefix}_gr"));
    p.emit_tensor_const(
        &grp_rsh,
        &[4],
        "int32",
        &format!("[1,{nkv},{groups},{hds}]"),
    );
    let grp = p.next_var(&format!("{prefix}_g"));
    p.emit_reshape(&grp, &[1, nkv, groups, hds], &grp_rsh, grad_var);

    let ax2 = p.next_var(&format!("{prefix}_a2"));
    p.emit_tensor_const(&ax2, &[1], "int32", "[2]");
    let kd = p.next_var(&format!("{prefix}_kd"));
    p.emit_scalar_const(&kd, "bool", "true");
    let reduced = p.next_var(&format!("{prefix}_rd"));
    p.emit_reduce_sum(&reduced, &[1, nkv, 1, hds], &grp, &ax2, &kd);

    let kv_head_rsh = p.next_var(&format!("{prefix}_hr"));
    p.emit_tensor_const(&kv_head_rsh, &[4], "int32", &format!("[1,{nkv},{hd},{s}]"));
    let heads = p.next_var(&format!("{prefix}_hd"));
    p.emit_reshape(&heads, &[1, nkv, hd, s], &kv_head_rsh, &reduced);

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
pub fn gen_dynamic_projection(ic: usize, oc: usize, seq: usize) -> DynamicKernelOutput {
    let sp = seq + oc;
    let mut p = MilProgram::new_fp32(ic, sp);
    p.emit_cast("x16", &[1, ic, 1, sp], "x", "fp16");
    let out16 = emit_dyn_matmul(&mut p, "mm", "x16", ic, oc, seq, 0, seq);
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
    p.emit_cast("x16", &[1, in_ch, 1, s], "x", "fp16");

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

    let q_rsh = p.next_var("qrs");
    p.emit_tensor_const(&q_rsh, &[4], "int32", &format!("[1,{nh},{hd},{s}]"));
    let q_heads = p.next_var("qh");
    p.emit_reshape(&q_heads, &[1, nh, hd, s], &q_rsh, &q_flat);

    let perm23 = p.next_var("p23");
    p.emit_tensor_const(&perm23, &[4], "int32", "[0,1,3,2]");
    let qt = p.next_var("qt");
    p.emit_transpose(&qt, &[1, nh, s, hd], &perm23, &q_heads);

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

    let scale_var = p.next_var("sc");
    p.emit_scalar_const(&scale_var, "fp16", &format!("{scale}"));
    let scores_scaled = p.next_var("ss");
    p.emit_mul(&scores_scaled, &[1, nh, s, s], &scores_raw, &scale_var);

    let mask_path = "@model_path/weights/mask.bin";
    p.emit_weight_const("mask", &[1, 1, s, s], mask_path);
    let scores_masked = p.next_var("sm");
    p.emit_add(&scores_masked, &[1, nh, s, s], &scores_scaled, "mask");

    let ax3 = p.next_var("ax3");
    p.emit_scalar_const(&ax3, "int32", "3");
    let probs = p.next_var("probs");
    p.emit_softmax(&probs, &[1, nh, s, s], &ax3, &scores_masked);

    let vt = p.next_var("vt");
    p.emit_transpose(&vt, &[1, nh, s, hd], &perm23, &v_h);
    let attn = p.next_var("attn");
    p.emit_matmul(&attn, &[1, nh, s, hd], &mm_false, &mm_false, &probs, &vt);

    let attn_t = p.next_var("at");
    p.emit_transpose(&attn_t, &[1, nh, hd, s], &perm23, &attn);

    let qd_rsh = p.next_var("qdrs");
    p.emit_tensor_const(&qd_rsh, &[4], "int32", &format!("[1,{qd},1,{s}]"));
    let attn_flat = p.next_var("af");
    p.emit_reshape(&attn_flat, &[1, qd, 1, s], &qd_rsh, &attn_t);

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

/// Generate a standalone softmax kernel over the channel (vocab) dimension.
pub fn gen_dynamic_softmax(vocab: usize, seq: usize) -> DynamicKernelOutput {
    let mut p = MilProgram::new(vocab, seq);
    let ax = p.next_var("ax");
    p.emit_scalar_const(&ax, "int32", "1");
    let out = p.next_var("sm");
    p.emit_softmax(&out, &[1, vocab, 1, seq], &ax, "x");
    let mil_text = p.finalize(&out);

    DynamicKernelOutput {
        mil_text,
        static_weights: WeightDict::new(),
        input_layout: SpatialLayout {
            ic: vocab,
            seq_len: seq,
            total_spatial: seq,
            oc: vocab,
            out_spatial: seq,
        },
        output_layout: SpatialLayout {
            ic: vocab,
            seq_len: seq,
            total_spatial: seq,
            oc: vocab,
            out_spatial: seq,
        },
    }
}

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
// Kernel 1: SDPA Forward (dynamic Wq, Wk, Wv, Wo + Fused RMSNorm)
// ============================================================================

/// Generate the dynamic SDPA forward kernel with fused RMSNorm and tap outputs.
pub fn gen_dynamic_sdpa_fwd(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let s = c.seq_len;
    let nh = c.n_heads;
    let nkv = c.n_kv_heads;
    let groups = c.n_groups();
    let hd = c.head_dim;
    let qd = c.q_dim();
    let kvd = c.kv_dim();

    let wq_off = s + 1;
    let wk_off = wq_off + qd;
    let wv_off = wk_off + kvd;
    let wo_off = wv_off + kvd;
    let sp = wo_off + qd;

    let out_ch = 2 * d + 2 * qd + 2 * kvd;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut p = MilProgram::new_fp32(d, sp);
    p.emit_cast("x16", &[1, d, 1, sp], "x", "fp16");

    let x_begin = p.next_var("xb");
    p.emit_tensor_const(&x_begin, &[4], "int32", "[0,0,0,0]");
    let x_size = p.next_var("xs");
    p.emit_tensor_const(&x_size, &[4], "int32", &format!("[1,{d},1,{s}]"));
    let x_raw = p.next_var("xr");
    p.emit_slice_by_size(&x_raw, &[1, d, 1, s], "x16", &x_begin, &x_size);

    let rw_begin = p.next_var("rwb");
    p.emit_tensor_const(&rw_begin, &[4], "int32", &format!("[0,0,0,{s}]"));
    let rw_size = p.next_var("rws");
    p.emit_tensor_const(&rw_size, &[4], "int32", &format!("[1,{d},1,1]"));
    let rms_w = p.next_var("rw");
    p.emit_slice_by_size(&rms_w, &[1, d, 1, 1], "x16", &rw_begin, &rw_size);

    let xnorm = emit_rmsnorm_fuse(&mut p, "rn", &x_raw, &rms_w, d, s, c.rms_norm_eps);

    let q = emit_dyn_matmul_with_act(&mut p, "q", &xnorm, "x16", d, qd, s, wq_off);
    let k = emit_dyn_matmul_with_act(&mut p, "k", &xnorm, "x16", d, kvd, s, wk_off);
    let v = emit_dyn_matmul_with_act(&mut p, "v", &xnorm, "x16", d, kvd, s, wv_off);

    let q_rsh = p.next_var("qrs");
    p.emit_tensor_const(&q_rsh, &[4], "int32", &format!("[1,{nh},{hd},{s}]"));
    let q_heads = p.next_var("qh");
    p.emit_reshape(&q_heads, &[1, nh, hd, s], &q_rsh, &q);

    let perm23 = p.next_var("p23");
    p.emit_tensor_const(&perm23, &[4], "int32", "[0,1,3,2]");
    let qt = p.next_var("qt");
    p.emit_transpose(&qt, &[1, nh, s, hd], &perm23, &q_heads);

    let kv_rsh = p.next_var("kvrs");
    p.emit_tensor_const(&kv_rsh, &[4], "int32", &format!("[1,{nkv},{hd},{s}]"));
    let k_kv = p.next_var("kkv");
    p.emit_reshape(&k_kv, &[1, nkv, hd, s], &kv_rsh, &k);

    let (k_heads, v_h) = if groups > 1 {
        let k_final = emit_gqa_expand(&mut p, "ke", &k_kv, nkv, nh, hd, s, groups);
        let v_kv = p.next_var("vkv");
        p.emit_reshape(&v_kv, &[1, nkv, hd, s], &kv_rsh, &v);
        let v_final = emit_gqa_expand(&mut p, "ve", &v_kv, nkv, nh, hd, s, groups);
        (k_final, v_final)
    } else {
        let k_h = p.next_var("kh");
        p.emit_reshape(&k_h, &[1, nh, hd, s], &q_rsh, &k);
        let v_h = p.next_var("vh");
        p.emit_reshape(&v_h, &[1, nh, hd, s], &q_rsh, &v);
        (k_h, v_h)
    };

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

    let scale_var = p.next_var("sc");
    p.emit_scalar_const(&scale_var, "fp16", &format!("{scale}"));
    let scores_scaled = p.next_var("ss");
    p.emit_mul(&scores_scaled, &[1, nh, s, s], &scores_raw, &scale_var);

    let mask_path = "@model_path/weights/mask.bin";
    p.emit_weight_const("mask", &[1, 1, s, s], mask_path);
    let scores_masked = p.next_var("sm");
    p.emit_add(&scores_masked, &[1, nh, s, s], &scores_scaled, "mask");

    let ax3 = p.next_var("ax3");
    p.emit_scalar_const(&ax3, "int32", "3");
    let probs = p.next_var("probs");
    p.emit_softmax(&probs, &[1, nh, s, s], &ax3, &scores_masked);

    let vt = p.next_var("vt");
    p.emit_transpose(&vt, &[1, nh, s, hd], &perm23, &v_h);
    let attn = p.next_var("attn");
    p.emit_matmul(&attn, &[1, nh, s, hd], &mm_false, &mm_false, &probs, &vt);

    let attn_t = p.next_var("at");
    p.emit_transpose(&attn_t, &[1, nh, hd, s], &perm23, &attn);

    let qd_rsh = p.next_var("qdrs");
    p.emit_tensor_const(&qd_rsh, &[4], "int32", &format!("[1,{qd},1,{s}]"));
    let attn_flat = p.next_var("af");
    p.emit_reshape(&attn_flat, &[1, qd, 1, s], &qd_rsh, &attn_t);

    let wo_begin = p.next_var("wob");
    p.emit_tensor_const(&wo_begin, &[4], "int32", &format!("[0,0,0,{}]", wo_off));
    let wo_size = p.next_var("wos");
    p.emit_tensor_const(&wo_size, &[4], "int32", &format!("[1,{d},1,{qd}]"));
    let wo_raw = p.next_var("wor");
    p.emit_slice_by_size(&wo_raw, &[1, d, 1, qd], "x16", &wo_begin, &wo_size);

    let af_rsh = p.next_var("afrs");
    p.emit_tensor_const(&af_rsh, &[4], "int32", &format!("[1,1,{qd},{s}]"));
    let af_r = p.next_var("afr");
    p.emit_reshape(&af_r, &[1, 1, qd, s], &af_rsh, &attn_flat);
    let af_t = p.next_var("aft");
    p.emit_transpose(&af_t, &[1, 1, s, qd], &perm23, &af_r);

    let wo_rsh = p.next_var("wrs");
    p.emit_tensor_const(&wo_rsh, &[4], "int32", &format!("[1,1,{d},{qd}]"));
    let wo_r = p.next_var("wrr");
    p.emit_reshape(&wo_r, &[1, 1, d, qd], &wo_rsh, &wo_raw);
    let wo_t = p.next_var("wot");
    p.emit_transpose(&wo_t, &[1, 1, qd, d], &perm23, &wo_r);

    let oo_mm = p.next_var("oom");
    p.emit_matmul(&oo_mm, &[1, 1, s, d], &mm_false, &mm_false, &af_t, &wo_t);

    let out_rsh = p.next_var("ors");
    p.emit_tensor_const(&out_rsh, &[4], "int32", &format!("[1,{d},1,{s}]"));
    let oo_t = p.next_var("oot");
    p.emit_transpose(&oo_t, &[1, 1, d, s], &perm23, &oo_mm);
    let o_out = p.next_var("oo");
    p.emit_reshape(&o_out, &[1, d, 1, s], &out_rsh, &oo_t);

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

    let taps32 = p.next_var("taps32");
    p.emit_cast(&taps32, &[1, out_ch, 1, s], &taps16, "fp32");

    let mil_text = p.finalize(&taps32);
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
// Kernel 2: FFN W1+W3 Forward (dynamic W1, W3 + SiLU gate + Fused RMSNorm)
// ============================================================================

/// Generate the dynamic FFN forward kernel with fused RMSNorm.
pub fn gen_dynamic_ffn_w13(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let h = c.hidden_dim;
    let s = c.seq_len;
    let sp = s + 1 + 2 * h;
    let out_ch = 3 * h;

    let mut p = MilProgram::new_fp32(d, sp);
    p.emit_cast("x16", &[1, d, 1, sp], "x", "fp16");

    let x_begin = p.next_var("xb");
    p.emit_tensor_const(&x_begin, &[4], "int32", "[0,0,0,0]");
    let x_size = p.next_var("xs");
    p.emit_tensor_const(&x_size, &[4], "int32", &format!("[1,{d},1,{s}]"));
    let x_raw = p.next_var("xr");
    p.emit_slice_by_size(&x_raw, &[1, d, 1, s], "x16", &x_begin, &x_size);

    let rw_begin = p.next_var("rwb");
    p.emit_tensor_const(&rw_begin, &[4], "int32", &format!("[0,0,0,{s}]"));
    let rw_size = p.next_var("rws");
    p.emit_tensor_const(&rw_size, &[4], "int32", &format!("[1,{d},1,1]"));
    let rms_w = p.next_var("rw");
    p.emit_slice_by_size(&rms_w, &[1, d, 1, 1], "x16", &rw_begin, &rw_size);

    let xnorm = emit_rmsnorm_fuse(&mut p, "rn", &x_raw, &rms_w, d, s, c.rms_norm_eps);

    let h1 = emit_dyn_matmul_with_act(&mut p, "w1", &xnorm, "x16", d, h, s, s + 1);
    let h3 = emit_dyn_matmul_with_act(&mut p, "w3", &xnorm, "x16", d, h, s, s + 1 + h);

    let sig = p.next_var("sig");
    p.emit_sigmoid(&sig, &[1, h, 1, s], &h1);
    let silu = p.next_var("silu");
    p.emit_mul(&silu, &[1, h, 1, s], &h1, &sig);

    let gate = p.next_var("gate");
    p.emit_mul(&gate, &[1, h, 1, s], &silu, &h3);

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
pub fn gen_dynamic_ffn_w2(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let h = c.hidden_dim;
    let s = c.seq_len;
    let sp = s + d;

    let mut p = MilProgram::new_fp32(h, sp);
    p.emit_cast("x16", &[1, h, 1, sp], "x", "fp16");
    let y = emit_dyn_matmul(&mut p, "w2", "x16", h, d, s, 0, s);
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
pub fn gen_dynamic_ffn_bwd_w13t(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let h = c.hidden_dim;
    let s = c.seq_len;
    let sp = 2 * s + 2 * d;

    let mut p = MilProgram::new_fp32(h, sp);
    p.emit_cast("x16", &[1, h, 1, sp], "x", "fp16");
    let dx1 = emit_dyn_matmul(&mut p, "w1t", "x16", h, d, s, 0, 2 * s);
    let dx3 = emit_dyn_matmul(&mut p, "w3t", "x16", h, d, s, s, 2 * s + d);
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
pub fn gen_dynamic_wo_bwd(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let qd = c.q_dim();
    let s = c.seq_len;
    let sp = s + qd;

    let mut p = MilProgram::new_fp32(d, sp);
    p.emit_cast("x16", &[1, d, 1, sp], "x", "fp16");
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
// Kernel 7: SDPA Backward Part 1 (Fused with Wo^T)
// ============================================================================

/// Generate the dynamic SDPA backward kernel part 1 fused with Wo^T projection.
pub fn gen_dynamic_sdpa_bwd1(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let qd = c.q_dim();
    let kvd = c.kv_dim();
    let s = c.seq_len;
    let nh = c.n_heads;
    let nkv = c.n_kv_heads;
    let groups = c.n_groups();
    let hd = c.head_dim;
    let score_ch = nh * s;
    let sp = s + qd;
    let in_ch = qd + 2 * kvd + d;
    let out_ch = kvd + 2 * score_ch;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut p = MilProgram::new_fp32(in_ch, sp);
    p.emit_cast("x16", &[1, in_ch, 1, sp], "x", "fp16");

    let da_raw = emit_dyn_matmul_at_ch(&mut p, "wot", "x16", d, qd, s, 0, s, qd + 2 * kvd);

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

    let q_rsh = p.next_var("qrs");
    p.emit_tensor_const(&q_rsh, &[4], "int32", &format!("[1,{nh},{hd},{s}]"));
    let q_h = p.next_var("qh");
    p.emit_reshape(&q_h, &[1, nh, hd, s], &q_rsh, &q_flat);
    let perm23 = p.next_var("p23");
    p.emit_tensor_const(&perm23, &[4], "int32", "[0,1,3,2]");
    let qt = p.next_var("qt");
    p.emit_transpose(&qt, &[1, nh, s, hd], &perm23, &q_h);

    let kv_rsh = p.next_var("kvrs");
    p.emit_tensor_const(&kv_rsh, &[4], "int32", &format!("[1,{nkv},{hd},{s}]"));
    let k_kv = p.next_var("kkv");
    p.emit_reshape(&k_kv, &[1, nkv, hd, s], &kv_rsh, &k_flat);
    let v_kv = p.next_var("vkv");
    p.emit_reshape(&v_kv, &[1, nkv, hd, s], &kv_rsh, &v_flat);

    let (k_h, v_h) = if groups > 1 {
        let ke = emit_gqa_expand(&mut p, "ke", &k_kv, nkv, nh, hd, s, groups);
        let ve = emit_gqa_expand(&mut p, "ve", &v_kv, nkv, nh, hd, s, groups);
        (ke, ve)
    } else {
        let k_h = p.next_var("kh");
        p.emit_reshape(&k_h, &[1, nh, hd, s], &q_rsh, &k_flat);
        let v_h = p.next_var("vvh");
        p.emit_reshape(&v_h, &[1, nh, hd, s], &q_rsh, &v_flat);
        (k_h, v_h)
    };

    let vt = p.next_var("vt");
    p.emit_transpose(&vt, &[1, nh, s, hd], &perm23, &v_h);

    let mm_false = p.next_var("mmf");
    p.emit_scalar_const(&mm_false, "bool", "false");
    let mm_true = p.next_var("mmt");
    p.emit_scalar_const(&mm_true, "bool", "true");

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

    let da_h = p.next_var("dah");
    p.emit_reshape(&da_h, &[1, nh, hd, s], &q_rsh, &da_raw);
    let da_t = p.next_var("dat");
    p.emit_transpose(&da_t, &[1, nh, s, hd], &perm23, &da_h);

    let dv_full = p.next_var("dvfl");
    p.emit_matmul(
        &dv_full,
        &[1, nh, s, hd],
        &mm_true,
        &mm_false,
        &probs_h,
        &da_t,
    );

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

    let dp_h = p.next_var("dph");
    p.emit_matmul(&dp_h, &[1, nh, s, s], &mm_false, &mm_false, &da_t, &v_h);

    let score_rsh = p.next_var("srs");
    p.emit_tensor_const(&score_rsh, &[4], "int32", &format!("[1,{score_ch},1,{s}]"));
    let probs_flat = p.next_var("pf");
    p.emit_reshape(&probs_flat, &[1, score_ch, 1, s], &score_rsh, &probs_h);
    let dp_flat = p.next_var("dpf");
    p.emit_reshape(&dp_flat, &[1, score_ch, 1, s], &score_rsh, &dp_h);

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
            total_spatial: sp,
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

/// Generate the dynamic SDPA backward kernel part 2 (weight-free).
pub fn gen_dynamic_sdpa_bwd2(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let qd = c.q_dim();
    let kvd = c.kv_dim();
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

    let q_h = p.next_var("qh");
    p.emit_reshape(&q_h, &[1, nh, hd, s], &q_rsh, &q_flat);
    let qt = p.next_var("qt");
    p.emit_transpose(&qt, &[1, nh, s, hd], &perm23, &q_h);

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

    let kt = p.next_var("kt");
    p.emit_transpose(&kt, &[1, nh, s, hd], &perm23, &k_h);

    let dq_h = p.next_var("dqh");
    p.emit_matmul(&dq_h, &[1, nh, s, hd], &mm_false, &mm_false, &ds, &kt);

    let dk_full = p.next_var("dkfl");
    p.emit_matmul(&dk_full, &[1, nh, s, hd], &mm_true, &mm_false, &ds, &qt);

    let dq_t = p.next_var("dqt");
    p.emit_transpose(&dq_t, &[1, nh, hd, s], &perm23, &dq_h);
    let flat_rsh_q = p.next_var("frsq");
    p.emit_tensor_const(&flat_rsh_q, &[4], "int32", &format!("[1,{qd},1,{s}]"));
    let dq_flat = p.next_var("dqf");
    p.emit_reshape(&dq_flat, &[1, qd, 1, s], &flat_rsh_q, &dq_t);

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
pub fn gen_dynamic_qkv_bwd_kv(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let kvd = c.kv_dim();
    let s = c.seq_len;
    let sp = 2 * s + 2 * d;
    let mut p = MilProgram::new_fp32(kvd, sp);
    p.emit_cast("x16", &[1, kvd, 1, sp], "x", "fp16");
    let dxk = emit_dyn_matmul(&mut p, "wkt", "x16", kvd, d, s, 0, 2 * s);
    let dxv = emit_dyn_matmul(&mut p, "wvt", "x16", kvd, d, s, s, 2 * s + d);
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
pub fn gen_dynamic_qkv_bwd(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let qd = c.q_dim();
    let s = c.seq_len;
    let sp = 3 * s + 3 * d;
    assert_eq!(
        c.n_kv_heads, c.n_heads,
        "Combined QKV backward requires MHA."
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

/// Generate the dynamic RMSNorm forward kernel.
pub fn gen_dynamic_rmsnorm_fwd(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let s = c.seq_len;
    let sp = s + 1;
    let mut p = MilProgram::new_fp32(d, sp);
    p.emit_cast("x16", &[1, d, 1, sp], "x", "fp16");
    let x_begin = p.next_var("xb");
    p.emit_tensor_const(&x_begin, &[4], "int32", "[0,0,0,0]");
    let x_size = p.next_var("xs");
    p.emit_tensor_const(&x_size, &[4], "int32", &format!("[1,{d},1,{s}]"));
    let x_in = p.next_var("xi");
    p.emit_slice_by_size(&x_in, &[1, d, 1, s], "x16", &x_begin, &x_size);
    let w_begin = p.next_var("wb");
    p.emit_tensor_const(&w_begin, &[4], "int32", &format!("[0,0,0,{s}]"));
    let w_size = p.next_var("ws");
    p.emit_tensor_const(&w_size, &[4], "int32", &format!("[1,{d},1,1]"));
    let w_in = p.next_var("wi");
    p.emit_slice_by_size(&w_in, &[1, d, 1, 1], "x16", &w_begin, &w_size);
    let xn = emit_rmsnorm_fuse(&mut p, "rn", &x_in, &w_in, d, s, c.rms_norm_eps);
    let out32 = p.next_var("out32");
    p.emit_cast(&out32, &[1, d, 1, s], &xn, "fp32");
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
            ic: d,
            seq_len: s,
            total_spatial: s,
            oc: d,
            out_spatial: s,
        },
    }
}

/// Generate the dynamic RMSNorm backward kernel.
pub fn gen_dynamic_rmsnorm_bwd(dkc: &DynamicKernelConfig) -> DynamicKernelOutput {
    let c = &dkc.cfg;
    let d = c.dim;
    let s = c.seq_len;
    let in_ch = 2 * d;
    let sp = s + 1;
    let mut p = MilProgram::new_fp32(in_ch, sp);
    p.emit_cast("x16", &[1, in_ch, 1, sp], "x", "fp16");
    let dy_begin = p.next_var("dyb");
    p.emit_tensor_const(&dy_begin, &[4], "int32", "[0,0,0,0]");
    let dy_size = p.next_var("dys");
    p.emit_tensor_const(&dy_size, &[4], "int32", &format!("[1,{d},1,{s}]"));
    let dy = p.next_var("dy");
    p.emit_slice_by_size(&dy, &[1, d, 1, s], "x16", &dy_begin, &dy_size);
    let xin_begin = p.next_var("xib");
    p.emit_tensor_const(&xin_begin, &[4], "int32", &format!("[0,{d},0,0]"));
    let xin_size = p.next_var("xis");
    p.emit_tensor_const(&xin_size, &[4], "int32", &format!("[1,{d},1,{s}]"));
    let x_in = p.next_var("xin");
    p.emit_slice_by_size(&x_in, &[1, d, 1, s], "x16", &xin_begin, &xin_size);
    let w_begin = p.next_var("wb");
    p.emit_tensor_const(&w_begin, &[4], "int32", &format!("[0,0,0,{s}]"));
    let w_size = p.next_var("wss");
    p.emit_tensor_const(&w_size, &[4], "int32", &format!("[1,{d},1,1]"));
    let w = p.next_var("rw");
    p.emit_slice_by_size(&w, &[1, d, 1, 1], "x16", &w_begin, &w_size);
    let dy_w = p.next_var("dyw");
    p.emit_mul(&dy_w, &[1, d, 1, s], &dy, &w);
    let x_sq = p.next_var("xsq");
    p.emit_mul(&x_sq, &[1, d, 1, s], &x_in, &x_in);
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
    let eps_var = p.next_var("eps");
    p.emit_scalar_const(&eps_var, "fp16", &format!("{}", c.rms_norm_eps));
    let ms_eps = p.next_var("mse");
    p.emit_add(&ms_eps, &[1, 1, 1, s], &mean_sq, &eps_var);
    let neg_half = p.next_var("nh");
    p.emit_scalar_const(&neg_half, "fp16", "-0.5");
    let rrms = p.next_var("rrms");
    p.emit_pow(&rrms, &[1, 1, 1, s], &ms_eps, &neg_half);
    let dyw_x = p.next_var("dwx");
    p.emit_mul(&dyw_x, &[1, d, 1, s], &dy_w, &x_in);
    let dot = p.next_var("dot");
    p.emit_reduce_sum(&dot, &[1, 1, 1, s], &dyw_x, &ax1, &kd_true);
    let dot_invd = p.next_var("did");
    p.emit_mul(&dot_invd, &[1, 1, 1, s], &dot, &inv_d);
    let rrms2 = p.next_var("rr2");
    p.emit_mul(&rrms2, &[1, 1, 1, s], &rrms, &rrms);
    let corr_sc = p.next_var("crs");
    p.emit_mul(&corr_sc, &[1, 1, 1, s], &dot_invd, &rrms2);
    let corr = p.next_var("corr");
    p.emit_mul(&corr, &[1, d, 1, s], &x_in, &corr_sc);
    let diff = p.next_var("diff");
    p.emit_sub(&diff, &[1, d, 1, s], &dy_w, &corr);
    let dx16 = p.next_var("dx16");
    p.emit_mul(&dx16, &[1, d, 1, s], &diff, &rrms);
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
        assert!(out.mil_text.contains("softmax("));
        assert!(out.mil_text.contains("BLOBFILE"));
        assert_eq!(out.input_layout.ic, 64);
        assert_eq!(out.output_layout.ic, 2 * 64 + 2 * 64 + 2 * 64);
    }

    #[test]
    fn test_dynamic_ffn_w13_generates_valid_mil() {
        let dkc = test_config();
        let out = gen_dynamic_ffn_w13(&dkc);
        assert!(out.mil_text.contains("sigmoid("));
        assert_eq!(out.input_layout.ic, 64);
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
        assert!(out.mil_text.contains("BLOBFILE"));
        assert_eq!(out.input_layout.ic, 64 + 2 * 64 + 64);
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
    fn test_dynamic_rmsnorm_fwd() {
        let dkc = test_config();
        let out = gen_dynamic_rmsnorm_fwd(&dkc);
        assert!(out.mil_text.contains("pow("));
        assert_eq!(out.input_layout.total_spatial, 32 + 1);
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

    /// Qwen3-0.6B style config: dim=1024, n_heads=16, head_dim=128 → q_dim=2048.
    fn qwen3_config() -> DynamicKernelConfig {
        DynamicKernelConfig::new(TransformerKernelConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 32,
            seq_len: 16,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        })
    }

    #[test]
    fn test_qwen3_dynamic_sdpa_fwd() {
        let dkc = qwen3_config();
        let out = gen_dynamic_sdpa_fwd(&dkc);
        assert!(out.mil_text.contains("program(1.3)"));
        assert_eq!(out.input_layout.ic, 64);
    }

    #[test]
    fn test_qwen3_dynamic_sdpa_bwd1() {
        let dkc = qwen3_config();
        let qd = dkc.cfg.q_dim(); // 128
        let kvd = dkc.cfg.kv_dim(); // 128
        let d = dkc.cfg.dim; // 64
        let score_ch = dkc.cfg.n_heads * dkc.cfg.seq_len; // 4 * 16 = 64

        let out = gen_dynamic_sdpa_bwd1(&dkc);
        assert!(out.mil_text.contains("softmax("));
        assert_eq!(out.input_layout.ic, qd + 2 * kvd + d);
        assert_eq!(out.output_layout.ic, kvd + 2 * score_ch);
    }

    /// GQA config: n_heads=4, n_kv_heads=2 (groups=2), head_dim=16.
    fn gqa_config() -> DynamicKernelConfig {
        DynamicKernelConfig::new(TransformerKernelConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            seq_len: 32,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        })
    }

    #[test]
    fn test_gqa_sdpa_fwd() {
        let dkc = gqa_config();
        let qd = dkc.cfg.q_dim(); // 64
        let kvd = dkc.cfg.kv_dim(); // 32
        let d = dkc.cfg.dim; // 64
        let out = gen_dynamic_sdpa_fwd(&dkc);
        assert!(out.mil_text.contains("concat("));
        assert_eq!(out.input_layout.ic, d);
        assert_eq!(out.output_layout.ic, 2 * d + 2 * qd + 2 * kvd);
    }

    #[test]
    fn test_gqa_sdpa_bwd1() {
        let dkc = gqa_config();
        let qd = dkc.cfg.q_dim(); // 64
        let kvd = dkc.cfg.kv_dim(); // 32
        let score_ch = dkc.cfg.n_heads * dkc.cfg.seq_len; // 4 * 32 = 128
        let d = dkc.cfg.dim; // 64

        let out = gen_dynamic_sdpa_bwd1(&dkc);
        assert!(out.mil_text.contains("reduce_sum("));
        assert_eq!(out.input_layout.ic, qd + 2 * kvd + d);
        assert_eq!(out.output_layout.ic, kvd + 2 * score_ch);
    }
}
