//! Extend kernels: run `W` new tokens through a chunk of transformer layers
//! against a KV cache held in IOSurfaces.
//!
//! One kernel family covers prefill (a wide `W`), decode and speculative
//! verification: on the ANE a pass over `W` tokens costs about what a pass
//! over one does, because reading the weights dominates. Each program holds
//! several layers so a token makes few round trips, and its weights can be
//! int8, dequantized by the ANE as it reads them.
//!
//! Per layer (Qwen3-shaped: per-head q/k RMSNorm, split-half RoPE, GQA,
//! SiLU-gated FFN):
//!
//! ```text
//! inputs   a_x [1,D,1,W]  b_cos, c_sin [1,hd/2,1,W]  d_mask [1,1,1,L]
//!          k_NN [1,n_kv,hd,L], v_NN [1,n_kv,L,hd]  (the cache, per layer)
//! outputs  o_kv [1, layers*2*kv_dim, 1, W]  (each layer's new K, V)
//!          o_x  [1,D,1,W]                    (the residual stream after)
//! ```
//!
//! Attention runs as two blocks, the cache and the new tokens, under one
//! softmax, so the cache is never copied to append to it. The host writes the
//! new K/V into the cache after the call. V is cached token-major so the
//! product with the attention weights needs no transpose: the ANE compiler
//! can't schedule a transpose of an input ("Couldn't do topological sort").

use std::ops::Range;

use crate::ane::iosurface::IoSurface;
use crate::ane::kernel::{KernelOutput, WeightBlob, emit_rope_with};
use crate::ane::mil::MilProgram;
use crate::ane::runtime::{AneModel, AneRuntime, WeightDict};
use crate::error::{MetalError, Result};

/// fp16's most negative finite value, the additive mask for a slot a query
/// can't see.
const MASKED: f32 = -65504.0;

/// How much smaller the down projection's input is computed (see
/// `emit_layer`).
const DOWN_SCALE: f32 = 16.0;

/// How a kernel's projection weights are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightFormat {
    /// fp16, as the checkpoint has them (rounded).
    Fp16,
    /// int8 per output channel, dequantized on the ANE: half the bytes, and
    /// decode reads the weights, so about half the time.
    Int8,
}

/// Shape of an extend kernel.
#[derive(Debug, Clone)]
pub struct ExtendConfig {
    /// Model dimension.
    pub dim: usize,
    /// FFN hidden dimension.
    pub hidden_dim: usize,
    /// Query heads.
    pub n_heads: usize,
    /// Key/value heads (GQA).
    pub n_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Tokens per call (`W`).
    pub width: usize,
    /// KV cache slots per layer (`L`): the context the kernel can attend to.
    pub capacity: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Projection weight storage.
    pub weights: WeightFormat,
}

impl ExtendConfig {
    /// Query projection width.
    pub fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    /// Key/value projection width.
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Query heads per key/value head.
    pub fn groups(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    fn validate(&self) -> Result<()> {
        let bad = |why: &str| Err(MetalError::InvalidConfig(format!("extend kernel: {why}")));
        if self.n_kv_heads == 0 || self.n_heads % self.n_kv_heads != 0 {
            return bad("n_heads must be a multiple of n_kv_heads");
        }
        if self.head_dim % 2 != 0 {
            return bad("head_dim must be even for RoPE");
        }
        if self.width == 0 || self.capacity == 0 {
            return bad("width and capacity must be positive");
        }
        Ok(())
    }
}

/// One layer's weights, `[out, in]` row-major f32 as checkpoints store them.
pub struct LayerTensors<'a> {
    /// Attention input RMSNorm, `[dim]`.
    pub rms_att: &'a [f32],
    /// Query projection, `[q_dim, dim]`.
    pub wq: &'a [f32],
    /// Key projection, `[kv_dim, dim]`.
    pub wk: &'a [f32],
    /// Value projection, `[kv_dim, dim]`.
    pub wv: &'a [f32],
    /// Output projection, `[dim, q_dim]`.
    pub wo: &'a [f32],
    /// Per-head query RMSNorm, `[head_dim]`.
    pub q_norm: &'a [f32],
    /// Per-head key RMSNorm, `[head_dim]`.
    pub k_norm: &'a [f32],
    /// FFN input RMSNorm, `[dim]`.
    pub rms_ffn: &'a [f32],
    /// Gate projection, `[hidden, dim]`.
    pub w_gate: &'a [f32],
    /// Up projection, `[hidden, dim]`.
    pub w_up: &'a [f32],
    /// Down projection, `[dim, hidden]`.
    pub w_down: &'a [f32],
}

/// One layer's weights, owned: what [`ExtendModel::compile`] asks for a layer
/// at a time, so a model is never held whole as f32.
#[derive(Clone)]
pub struct LayerWeights {
    /// Attention input RMSNorm, `[dim]`.
    pub rms_att: Vec<f32>,
    /// Query projection, `[q_dim, dim]`.
    pub wq: Vec<f32>,
    /// Key projection, `[kv_dim, dim]`.
    pub wk: Vec<f32>,
    /// Value projection, `[kv_dim, dim]`.
    pub wv: Vec<f32>,
    /// Output projection, `[dim, q_dim]`.
    pub wo: Vec<f32>,
    /// Per-head query RMSNorm, `[head_dim]`.
    pub q_norm: Vec<f32>,
    /// Per-head key RMSNorm, `[head_dim]`.
    pub k_norm: Vec<f32>,
    /// FFN input RMSNorm, `[dim]`.
    pub rms_ffn: Vec<f32>,
    /// Gate projection, `[hidden, dim]`.
    pub w_gate: Vec<f32>,
    /// Up projection, `[hidden, dim]`.
    pub w_up: Vec<f32>,
    /// Down projection, `[dim, hidden]`.
    pub w_down: Vec<f32>,
}

impl LayerWeights {
    /// Borrow as the kernel generator's input.
    pub fn tensors(&self) -> LayerTensors<'_> {
        LayerTensors {
            rms_att: &self.rms_att,
            wq: &self.wq,
            wk: &self.wk,
            wv: &self.wv,
            wo: &self.wo,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            rms_ffn: &self.rms_ffn,
            w_gate: &self.w_gate,
            w_up: &self.w_up,
            w_down: &self.w_down,
        }
    }
}

/// The name of layer `i`'s (within its chunk) cache input for `kind` (`k` or
/// `v`). Zero-padded so the ANE's alphabetical binding order is layer order.
fn cache_input(kind: char, i: usize) -> String {
    format!("{kind}_{i:02}")
}

/// Generate the extend kernel for `layers`, consecutive layers of the model.
pub fn gen_extend_chunk(cfg: &ExtendConfig, layers: &[LayerTensors<'_>]) -> Result<KernelOutput> {
    cfg.validate()?;
    if layers.is_empty() || layers.len() > 100 {
        return Err(MetalError::InvalidConfig(
            "extend kernel: a chunk holds 1 to 100 layers".into(),
        ));
    }
    let (d, w, l) = (cfg.dim, cfg.width, cfg.capacity);
    let (kvd, half) = (cfg.kv_dim(), cfg.head_dim / 2);

    let mut inputs: Vec<(String, Vec<usize>)> = vec![
        ("a_x".into(), vec![1, d, 1, w]),
        ("b_cos".into(), vec![1, half, 1, w]),
        ("c_sin".into(), vec![1, half, 1, w]),
        ("d_mask".into(), vec![1, 1, 1, l]),
    ];
    let (nkv, hd) = (cfg.n_kv_heads, cfg.head_dim);
    for i in 0..layers.len() {
        inputs.push((cache_input('k', i), vec![1, nkv, hd, l]));
    }
    for i in 0..layers.len() {
        inputs.push((cache_input('v', i), vec![1, nkv, l, hd]));
    }
    let input_refs: Vec<(&str, &[usize])> = inputs
        .iter()
        .map(|(n, s)| (n.as_str(), s.as_slice()))
        .collect();
    let mut p = MilProgram::with_inputs(&input_refs);
    let mut weights = WeightDict::new();
    p.emit_conv_constants();

    // Shared constants.
    p.emit_tensor_const("ax1", &[1], "int32", "[1]");
    p.emit_tensor_const("ax2", &[1], "int32", "[2]");
    p.emit_tensor_const("ax3", &[1], "int32", "[3]");
    p.emit_scalar_const("kd", "bool", "true");
    p.emit_scalar_const("tf", "bool", "false");
    p.emit_scalar_const("tt", "bool", "true");
    p.emit_scalar_const("cat1", "int32", "1");
    p.emit_scalar_const("ax_last", "int32", "-1");
    p.emit_tensor_const("p0132", &[4], "int32", "[0,1,3,2]");
    p.emit_tensor_const("rs_rope", &[4], "int32", &format!("[1,1,{half},{w}]"));
    p.emit_reshape("cos4", &[1, 1, half, w], "rs_rope", "b_cos");
    p.emit_reshape("sin4", &[1, 1, half, w], "rs_rope", "c_sin");
    p.emit_tensor_const("rs_mask", &[4], "int32", &format!("[1,1,1,{l}]"));
    p.emit_reshape("mask4", &[1, 1, 1, l], "rs_mask", "d_mask");
    let gw = cfg.groups() * w;
    p.emit_weight_const("causal", &[1, 1, gw, w], "@model_path/weights/causal.bin");
    weights.add(
        "@model_path/weights/causal.bin",
        causal_block(cfg.groups(), w),
    );

    let mut x = "a_x".to_string();
    let mut kv_taps = Vec::with_capacity(2 * layers.len());
    for (i, layer) in layers.iter().enumerate() {
        let (x_next, k_tap, v_tap) = emit_layer(&mut p, &mut weights, cfg, i, layer, &x);
        x = x_next;
        kv_taps.push(k_tap);
        kv_taps.push(v_tap);
    }

    let taps: Vec<&str> = kv_taps.iter().map(String::as_str).collect();
    p.emit_concat(
        "o_kv",
        &[1, 2 * kvd * layers.len(), 1, w],
        "cat1",
        "tf",
        &taps,
    );
    // An output can't be an input, so even a no-op chunk produces a new x.
    p.emit_scalar_const("one", "fp16", "1.0");
    p.emit_mul("o_x", &[1, d, 1, w], &x, "one");
    let mil_text = p.finalize_multi(&["o_kv", "o_x"]);

    Ok(KernelOutput {
        mil_text,
        weights,
        input_bytes: d * w * 2,
        output_bytes: d * w * 2,
    })
}

/// The additive causal mask among the `w` new tokens, rows repeated for each
/// of the `groups` query heads sharing a key head: `[1, 1, groups * w, w]`.
fn causal_block(groups: usize, w: usize) -> Vec<u8> {
    let mut mask = Vec::with_capacity(groups * w * w);
    for _ in 0..groups {
        for t in 0..w {
            for u in 0..w {
                let v = if u <= t { 0.0 } else { MASKED };
                mask.push(half::f16::from_f32(v).to_bits());
            }
        }
    }
    WeightBlob::from_fp16(&mask)
}

/// Emit `rows x cols` weights (`[out, in]`) as the constant `name`, stored as
/// `cfg.weights` says.
fn emit_projection_weight(
    p: &mut MilProgram,
    weights: &mut WeightDict,
    cfg: &ExtendConfig,
    name: &str,
    w: &[f32],
    rows: usize,
    cols: usize,
) {
    let path = format!("@model_path/weights/{name}.bin");
    match cfg.weights {
        WeightFormat::Fp16 => {
            p.emit_weight_const(name, &[rows, cols, 1, 1], &path);
            weights.add(&path, WeightBlob::from_f32(w, rows, cols));
        }
        WeightFormat::Int8 => {
            let scale_path = format!("@model_path/weights/{name}_s.bin");
            p.emit_int8_weight_const(name, &[rows, cols, 1, 1], &path, &scale_path);
            let (data, scales) = WeightBlob::int8_per_row(w, rows, cols);
            weights.add(&path, data);
            weights.add(&scale_path, scales);
        }
    }
}

/// Emit RMSNorm of `x` (shape `shape`) over `axis`, times `weight` (already
/// a program constant broadcastable to `shape`), as `out`.
///
/// `x` is first divided by its largest magnitude along `axis`, so the sum of
/// squares is at most the axis length. The plain sum overflows fp16 once the
/// RMS passes `sqrt(65504 / n)`, 5 for a 2560-wide model, and a real model's
/// residual stream goes far past that. ε is carried exactly, as `ε / m²`.
#[allow(clippy::too_many_arguments)]
fn emit_rmsnorm(
    p: &mut MilProgram,
    x: &str,
    out: &str,
    shape: &[usize],
    axis: usize,
    eps: f32,
    weight: &str,
) {
    let mut stat = shape.to_vec();
    stat[axis] = 1;
    let n = shape[axis];
    let axes = format!("ax{axis}");
    let v = |p: &mut MilProgram, s: &str| p.next_var(&format!("{out}_{s}"));

    let ab = v(p, "abs");
    p.emit_unary("abs", &ab, shape, x);
    let mx = v(p, "max");
    p.emit_reduce("reduce_max", &mx, &stat, &ab, &axes, "kd");
    let floor = v(p, "floor");
    p.emit_scalar_const(&floor, "fp16", "0.0001");
    let m = v(p, "m");
    p.emit_binary("maximum", &m, &stat, &mx, &floor);
    // 1 / m, once: the compiler rejected programs where m itself fed both
    // the division and the ε term.
    let inv_eps = v(p, "ieps");
    p.emit_scalar_const(&inv_eps, "fp16", "0.0");
    let inv_m = v(p, "invm");
    p.emit_raw(&format!(
        "        tensor<fp16, {}> {inv_m} = inverse(x={m},epsilon={inv_eps})[name=string(\"{inv_m}\")];",
        format!("{stat:?}").replace(' ', "")
    ));
    let xs = v(p, "xs");
    p.emit_mul(&xs, shape, x, &inv_m);
    let sq = v(p, "sq");
    p.emit_mul(&sq, shape, &xs, &xs);
    let ss = v(p, "ss");
    p.emit_reduce("reduce_sum", &ss, &stat, &sq, &axes, "kd");
    let inv_n = v(p, "invn");
    p.emit_scalar_const(&inv_n, "fp16", &format!("{}", 1.0 / n as f32));
    let ms = v(p, "ms");
    p.emit_mul(&ms, &stat, &ss, &inv_n);
    // √ε / m.
    let root_eps = v(p, "reps");
    p.emit_scalar_const(&root_eps, "fp16", &format!("{}", eps.sqrt()));
    let t = v(p, "t");
    p.emit_mul(&t, &stat, &inv_m, &root_eps);
    let t2 = v(p, "t2");
    p.emit_mul(&t2, &stat, &t, &t);
    let den = v(p, "den");
    p.emit_add(&den, &stat, &ms, &t2);
    let nhalf = v(p, "nh");
    p.emit_scalar_const(&nhalf, "fp16", "-0.5");
    let r = v(p, "r");
    p.emit_pow(&r, &stat, &den, &nhalf);
    let xr = v(p, "xr");
    p.emit_mul(&xr, shape, &xs, &r);
    p.emit_mul(out, shape, &xr, weight);
}

/// Emit one layer reading the residual stream `x`. Returns the residual
/// stream after it and the layer's new K and V, each `[1, kv_dim, 1, W]`.
fn emit_layer(
    p: &mut MilProgram,
    weights: &mut WeightDict,
    cfg: &ExtendConfig,
    i: usize,
    t: &LayerTensors<'_>,
    x: &str,
) -> (String, String, String) {
    let (d, h, w, l) = (cfg.dim, cfg.hidden_dim, cfg.width, cfg.capacity);
    let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let (qd, kvd, gw) = (cfg.q_dim(), cfg.kv_dim(), cfg.groups() * w);
    let n = |s: &str| format!("l{i}_{s}");
    let small =
        |p: &mut MilProgram, wd: &mut WeightDict, name: &str, data: &[f32], shape: &[usize]| {
            let path = format!("@model_path/weights/{name}.bin");
            p.emit_weight_const(name, shape, &path);
            wd.add(&path, WeightBlob::from_rms_weights(data));
        };

    // Attention.
    small(p, weights, &n("rms_att"), t.rms_att, &[1, d, 1, 1]);
    emit_rmsnorm(
        p,
        x,
        &n("xn"),
        &[1, d, 1, w],
        1,
        cfg.rms_norm_eps,
        &n("rms_att"),
    );
    emit_projection_weight(p, weights, cfg, &n("wq"), t.wq, qd, d);
    emit_projection_weight(p, weights, cfg, &n("wk"), t.wk, kvd, d);
    emit_projection_weight(p, weights, cfg, &n("wv"), t.wv, kvd, d);
    p.emit_conv(&n("q"), &[1, qd, 1, w], &n("wq"), &n("xn"));
    p.emit_conv(&n("k"), &[1, kvd, 1, w], &n("wk"), &n("xn"));
    p.emit_conv(&n("v"), &[1, kvd, 1, w], &n("wv"), &n("xn"));

    p.emit_tensor_const(&n("rs_q"), &[4], "int32", &format!("[1,{nh},{hd},{w}]"));
    p.emit_tensor_const(&n("rs_kv"), &[4], "int32", &format!("[1,{nkv},{hd},{w}]"));
    p.emit_reshape(&n("q4"), &[1, nh, hd, w], &n("rs_q"), &n("q"));
    p.emit_reshape(&n("k4"), &[1, nkv, hd, w], &n("rs_kv"), &n("k"));
    p.emit_reshape(&n("v4"), &[1, nkv, hd, w], &n("rs_kv"), &n("v"));
    small(p, weights, &n("qn_w"), t.q_norm, &[1, 1, hd, 1]);
    small(p, weights, &n("kn_w"), t.k_norm, &[1, 1, hd, 1]);
    emit_rmsnorm(
        p,
        &n("q4"),
        &n("qn"),
        &[1, nh, hd, w],
        2,
        cfg.rms_norm_eps,
        &n("qn_w"),
    );
    emit_rmsnorm(
        p,
        &n("k4"),
        &n("kn"),
        &[1, nkv, hd, w],
        2,
        cfg.rms_norm_eps,
        &n("kn_w"),
    );
    emit_rope_with(p, &n("qn"), &n("qr"), nh, hd, w, "cos4", "sin4");
    emit_rope_with(p, &n("kn"), &n("kr"), nkv, hd, w, "cos4", "sin4");

    // Queries grouped by key head: [1, nkv, groups * W, hd].
    p.emit_scalar_const(
        &n("scale"),
        "fp16",
        &format!("{}", 1.0 / (hd as f32).sqrt()),
    );
    p.emit_mul(&n("qs"), &[1, nh, hd, w], &n("qr"), &n("scale"));
    p.emit_transpose(&n("qt"), &[1, nh, w, hd], "p0132", &n("qs"));
    p.emit_tensor_const(&n("rs_qg"), &[4], "int32", &format!("[1,{nkv},{gw},{hd}]"));
    p.emit_reshape(&n("qg"), &[1, nkv, gw, hd], &n("rs_qg"), &n("qt"));

    // Scores against the cache and against the new tokens.
    let (kc, vc) = (cache_input('k', i), cache_input('v', i));
    p.emit_matmul(&n("sc"), &[1, nkv, gw, l], "tf", "tf", &n("qg"), &kc);
    p.emit_add(&n("scm"), &[1, nkv, gw, l], &n("sc"), "mask4");
    p.emit_matmul(&n("sn"), &[1, nkv, gw, w], "tf", "tf", &n("qg"), &n("kr"));
    p.emit_add(&n("snm"), &[1, nkv, gw, w], &n("sn"), "causal");

    // One softmax over both blocks, as two: each block's own softmax times
    // V, mixed by the block's share of the total weight, which from the two
    // log-sum-exps is sigmoid(lse_c - lse_n). Exact, no sum can overflow, and
    // an empty cache (every slot masked) gets a share of 0. Normalizing the
    // exponentials by a hand-built sum instead made the compiler's graph
    // cyclic ("Couldn't do topological sort").
    let stat = [1, nkv, gw, 1];
    p.emit_softmax(&n("pc"), &[1, nkv, gw, l], "ax_last", &n("scm"));
    p.emit_softmax(&n("pn"), &[1, nkv, gw, w], "ax_last", &n("snm"));
    p.emit_matmul(&n("oc"), &[1, nkv, gw, hd], "tf", "tf", &n("pc"), &vc);
    p.emit_matmul(&n("on"), &[1, nkv, gw, hd], "tf", "tt", &n("pn"), &n("v4"));
    // Each log-sum-exp as max + log(sum(exp(s - max))). The ANE's own
    // reduce_log_sum_exp overflows fp16 once a score passes ~11 (a real
    // model's reach 40+), which left the cache's share NaN.
    for (block, scores, shape) in [("c", "scm", [1, nkv, gw, l]), ("n", "snm", [1, nkv, gw, w])] {
        let v = |s: &str| n(&format!("{s}{block}"));
        p.emit_reduce("reduce_max", &v("mx"), &stat, &n(scores), "ax3", "kd");
        p.emit_sub(&v("sh"), &shape, &n(scores), &v("mx"));
        p.emit_unary("exp", &v("ex"), &shape, &v("sh"));
        p.emit_reduce("reduce_sum", &v("z"), &stat, &v("ex"), "ax3", "kd");
        // MIL's log needs its epsilon spelled out (z >= 1 here anyway).
        p.emit_scalar_const(&v("leps"), "fp16", "0.0");
        p.emit_raw(&format!(
            "        tensor<fp16, [1, {nkv}, {gw}, 1]> {lz} = log(x={z},epsilon={eps})[name=string(\"{lz}\")];",
            lz = v("lz"),
            z = v("z"),
            eps = v("leps"),
        ));
        p.emit_add(&v("lse"), &stat, &v("mx"), &v("lz"));
    }
    p.emit_sub(&n("dl"), &stat, &n("lsec"), &n("lsen"));
    p.emit_sigmoid(&n("wc"), &stat, &n("dl"));
    p.emit_scalar_const(&n("one_w"), "fp16", "1.0");
    p.emit_sub(&n("wn"), &stat, &n("one_w"), &n("wc"));
    p.emit_mul(&n("ocw"), &[1, nkv, gw, hd], &n("oc"), &n("wc"));
    p.emit_mul(&n("onw"), &[1, nkv, gw, hd], &n("on"), &n("wn"));
    p.emit_add(&n("o"), &[1, nkv, gw, hd], &n("ocw"), &n("onw"));

    // Back to [1, q_dim, 1, W] and out through Wo.
    p.emit_tensor_const(&n("rs_oh"), &[4], "int32", &format!("[1,{nh},{w},{hd}]"));
    p.emit_reshape(&n("oh"), &[1, nh, w, hd], &n("rs_oh"), &n("o"));
    p.emit_transpose(&n("ot"), &[1, nh, hd, w], "p0132", &n("oh"));
    p.emit_tensor_const(&n("rs_of"), &[4], "int32", &format!("[1,{qd},1,{w}]"));
    p.emit_reshape(&n("of"), &[1, qd, 1, w], &n("rs_of"), &n("ot"));
    emit_projection_weight(p, weights, cfg, &n("wo"), t.wo, d, qd);
    p.emit_conv(&n("ao"), &[1, d, 1, w], &n("wo"), &n("of"));
    p.emit_add(&n("x2"), &[1, d, 1, w], x, &n("ao"));

    // FFN.
    small(p, weights, &n("rms_ffn"), t.rms_ffn, &[1, d, 1, 1]);
    emit_rmsnorm(
        p,
        &n("x2"),
        &n("xn2"),
        &[1, d, 1, w],
        1,
        cfg.rms_norm_eps,
        &n("rms_ffn"),
    );
    emit_projection_weight(p, weights, cfg, &n("w1"), t.w_gate, h, d);
    // The ANE's int8 convolution appears to sum the int8 values times the
    // input in fp16 and apply each row's scale after, so its sum is the
    // output over the scale: hundreds of times the output. At an attention
    // sink token the down projection's output reaches the thousands, and the
    // sum overflowed (Qwen3-4B's chat prompts came out as end-of-text). With
    // int8 weights, up is scaled down by DOWN_SCALE, exactly (its rows carry
    // their own scale), and the down projection's output back up. fp16
    // weights don't need it, and small models' activations lose precision to
    // it.
    let down_scale = match cfg.weights {
        WeightFormat::Int8 => DOWN_SCALE,
        WeightFormat::Fp16 => 1.0,
    };
    let up_scaled: Vec<f32> = t.w_up.iter().map(|v| v / down_scale).collect();
    emit_projection_weight(p, weights, cfg, &n("w3"), &up_scaled, h, d);
    drop(up_scaled);
    emit_projection_weight(p, weights, cfg, &n("w2"), t.w_down, d, h);
    p.emit_conv(&n("h1"), &[1, h, 1, w], &n("w1"), &n("xn2"));
    p.emit_conv(&n("h3"), &[1, h, 1, w], &n("w3"), &n("xn2"));
    p.emit_sigmoid(&n("sg"), &[1, h, 1, w], &n("h1"));
    p.emit_mul(&n("silu"), &[1, h, 1, w], &n("h1"), &n("sg"));
    p.emit_mul(&n("g"), &[1, h, 1, w], &n("silu"), &n("h3"));
    p.emit_conv(&n("ys"), &[1, d, 1, w], &n("w2"), &n("g"));
    p.emit_scalar_const(&n("down_scale"), "fp16", &format!("{down_scale}"));
    p.emit_mul(&n("y"), &[1, d, 1, w], &n("ys"), &n("down_scale"));
    p.emit_add(&n("x3"), &[1, d, 1, w], &n("x2"), &n("y"));

    p.emit_tensor_const(&n("rs_kvf"), &[4], "int32", &format!("[1,{kvd},1,{w}]"));
    p.emit_reshape(&n("ktap"), &[1, kvd, 1, w], &n("rs_kvf"), &n("kr"));
    (n("x3"), n("ktap"), n("v"))
}

/// The vocabulary rows each LM head output covers, as `(start, rows)`, for
/// a model `dim` wide.
///
/// The ANE reads a convolution's weights slowly once its outputs outnumber
/// its inputs by much more than four: Qwen3-0.6B's head (1024 wide) as
/// 16384-row pieces read 26 GB/s, as 4096-row pieces 97 GB/s. A piece is
/// also kept to 16384 rows, the widest tensor dimension the ANE handles
/// natively.
pub fn lm_head_pieces(vocab: usize, dim: usize) -> Vec<(usize, usize)> {
    let piece = (4 * dim).clamp(1024, 16384);
    (0..vocab)
        .step_by(piece)
        .map(|start| (start, piece.min(vocab - start)))
        .collect()
}

/// Generate the LM head: the final RMSNorm of `a_x` `[1, dim, 1, W]` and the
/// logits, one output per [`lm_head_pieces`] piece, `o_logits_NN`
/// `[1, rows, 1, W]`, from `weight` `[vocab, dim]` stored as `cfg.weights`
/// says.
pub fn gen_lm_head(
    cfg: &ExtendConfig,
    final_norm: &[f32],
    weight: &[f32],
    vocab: usize,
) -> Result<KernelOutput> {
    cfg.validate()?;
    let (d, w) = (cfg.dim, cfg.width);
    if final_norm.len() != d || weight.len() != vocab * d {
        return Err(MetalError::InvalidConfig(format!(
            "LM head: expected a [{d}] norm and [{vocab}, {d}] weights"
        )));
    }
    let mut p = MilProgram::with_inputs(&[("a_x", &[1, d, 1, w])]);
    let mut weights = WeightDict::new();
    p.emit_conv_constants();
    p.emit_tensor_const("ax1", &[1], "int32", "[1]");
    p.emit_scalar_const("kd", "bool", "true");
    let path = "@model_path/weights/final_norm.bin";
    p.emit_weight_const("final_norm", &[1, d, 1, 1], path);
    weights.add(path, WeightBlob::from_rms_weights(final_norm));
    emit_rmsnorm(
        &mut p,
        "a_x",
        "xn",
        &[1, d, 1, w],
        1,
        cfg.rms_norm_eps,
        "final_norm",
    );

    let mut outputs = Vec::new();
    for (i, (start, rows)) in lm_head_pieces(vocab, d).into_iter().enumerate() {
        let name = format!("head{i}");
        emit_projection_weight(
            &mut p,
            &mut weights,
            cfg,
            &name,
            &weight[start * d..(start + rows) * d],
            rows,
            d,
        );
        // Zero-padded: outputs bind in name order.
        let out = format!("o_logits_{i:02}");
        p.emit_conv(&out, &[1, rows, 1, w], &name, "xn");
        outputs.push(out);
    }
    let refs: Vec<&str> = outputs.iter().map(String::as_str).collect();
    Ok(KernelOutput {
        mil_text: p.finalize_multi(&refs),
        weights,
        input_bytes: d * w * 2,
        output_bytes: vocab * w * 2,
    })
}

/// cos and sin of RoPE for positions `start..start + width`, each
/// channel-first `[head_dim / 2, width]`.
pub fn rope_tables(
    head_dim: usize,
    start: usize,
    width: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0; half * width];
    let mut sin = vec![0.0; half * width];
    for i in 0..half {
        let freq = 1.0 / (theta as f64).powf(2.0 * i as f64 / head_dim as f64);
        for t in 0..width {
            let (s, c) = (((start + t) as f64) * freq).sin_cos();
            cos[i * width + t] = c as f32;
            sin[i * width + t] = s as f32;
        }
    }
    (cos, sin)
}

struct Chunk {
    model: AneModel,
    layers: Range<usize>,
    /// The chunk's new K/V, `[layers * 2 * kv_dim, W]`.
    kv_out: IoSurface,
}

/// A model's layers compiled into extend kernels, with its KV cache.
///
/// [`extend`](Self::extend) runs up to `W` tokens' hidden states through
/// every layer, appends their keys and values to the cache, and returns the
/// residual stream after the last layer. Embedding, final norm and logits are
/// the caller's.
pub struct ExtendModel {
    cfg: ExtendConfig,
    rope_theta: f32,
    chunks: Vec<Chunk>,
    k_cache: Vec<IoSurface>,
    v_cache: Vec<IoSurface>,
    /// The residual stream, ping-ponged between chunks.
    x: [IoSurface; 2],
    cos: IoSurface,
    sin: IoSurface,
    mask: IoSurface,
    pos: usize,
}

impl ExtendModel {
    /// Compile `n_layers` layers in chunks of at most `layers_per_chunk`,
    /// getting each layer's weights from `layer`.
    pub fn compile(
        cfg: ExtendConfig,
        rope_theta: f32,
        n_layers: usize,
        layers_per_chunk: usize,
        mut layer: impl FnMut(usize) -> Result<LayerWeights>,
    ) -> Result<Self> {
        cfg.validate()?;
        let rt = AneRuntime::global()?;
        let (d, w, l, kvd) = (cfg.dim, cfg.width, cfg.capacity, cfg.kv_dim());
        let per_chunk = layers_per_chunk.max(1);
        let mut chunks = Vec::new();
        for start in (0..n_layers).step_by(per_chunk) {
            let range = start..(start + per_chunk).min(n_layers);
            let owned = range.clone().map(&mut layer).collect::<Result<Vec<_>>>()?;
            let tensors: Vec<LayerTensors<'_>> = owned.iter().map(LayerWeights::tensors).collect();
            let kernel = gen_extend_chunk(&cfg, &tensors)?;
            drop(tensors);
            drop(owned);
            let model = rt.compile(kernel.mil_text.as_bytes(), Some(&kernel.weights))?;
            chunks.push(Chunk {
                model,
                kv_out: IoSurface::for_tensor(2 * kvd * range.len(), w)?,
                layers: range,
            });
        }
        let caches = |_| IoSurface::for_tensor(kvd, l);
        Ok(Self {
            k_cache: (0..n_layers).map(caches).collect::<Result<_>>()?,
            v_cache: (0..n_layers).map(caches).collect::<Result<_>>()?,
            x: [IoSurface::for_tensor(d, w)?, IoSurface::for_tensor(d, w)?],
            cos: IoSurface::for_tensor(cfg.head_dim / 2, w)?,
            sin: IoSurface::for_tensor(cfg.head_dim / 2, w)?,
            mask: IoSurface::for_tensor(1, l)?,
            cfg,
            rope_theta,
            chunks,
            pos: 0,
        })
    }

    /// The kernel shape.
    pub fn config(&self) -> &ExtendConfig {
        &self.cfg
    }

    /// Tokens in the cache.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Forget the cache from `pos` on (rejected speculative tokens, a new
    /// prompt). The slots stay; the mask hides them until they're rewritten.
    pub fn truncate(&mut self, pos: usize) {
        self.pos = self.pos.min(pos);
    }

    /// Run `n` tokens' hidden states, channel-first `[dim, n]` in `x`,
    /// through every layer at positions `position()..position() + n`, and
    /// return the residual stream after the last layer, `[dim, n]`.
    pub fn extend(&mut self, x: &[f32], n: usize) -> Result<Vec<f32>> {
        let (d, w, l, kvd) = (
            self.cfg.dim,
            self.cfg.width,
            self.cfg.capacity,
            self.cfg.kv_dim(),
        );
        if n == 0 || n > w || x.len() != d * n {
            return Err(MetalError::InvalidConfig(format!(
                "extend takes 1 to {w} tokens of [{d}, n]; got {} values for {n}",
                x.len()
            )));
        }
        if self.pos + n > l {
            return Err(MetalError::InvalidConfig(format!(
                "KV cache full: {} + {n} tokens past its {l} slots",
                self.pos
            )));
        }

        let mut padded = vec![0.0f32; d * w];
        for c in 0..d {
            padded[c * w..c * w + n].copy_from_slice(&x[c * n..(c + 1) * n]);
        }
        self.x[0].write_f32_as_fp16(&padded, d, w);
        let (cos, sin) = rope_tables(self.cfg.head_dim, self.pos, w, self.rope_theta);
        self.cos.write_f32_as_fp16(&cos, self.cfg.head_dim / 2, w);
        self.sin.write_f32_as_fp16(&sin, self.cfg.head_dim / 2, w);
        let pos = self.pos;
        let (zero, masked) = (
            half::f16::ZERO.to_bits(),
            half::f16::from_f32(MASKED).to_bits(),
        );
        self.mask.with_fp16_mut(|m| {
            for (j, v) in m[..l].iter_mut().enumerate() {
                *v = if j < pos { zero } else { masked };
            }
        });

        let mut cur = 0;
        for chunk in &self.chunks {
            let mut inputs = vec![
                self.x[cur].as_ptr(),
                self.cos.as_ptr(),
                self.sin.as_ptr(),
                self.mask.as_ptr(),
            ];
            inputs.extend(chunk.layers.clone().map(|i| self.k_cache[i].as_ptr()));
            inputs.extend(chunk.layers.clone().map(|i| self.v_cache[i].as_ptr()));
            chunk
                .model
                .evaluate(&inputs, &[chunk.kv_out.as_ptr(), self.x[1 - cur].as_ptr()])?;
            cur = 1 - cur;

            // Append this chunk's new keys and values at pos..pos + n. The taps
            // are channel-first [kv_dim, W]; K is cached [n_kv, hd, L] (the
            // same layout) and V token-major, [n_kv, L, hd].
            let hd = self.cfg.head_dim;
            chunk.kv_out.with_fp16(|kv| {
                for (j, layer) in chunk.layers.clone().enumerate() {
                    let (k_tap, v_tap) = (2 * j * kvd, (2 * j + 1) * kvd);
                    self.k_cache[layer].with_fp16_mut(|c| {
                        for ch in 0..kvd {
                            let src = &kv[(k_tap + ch) * w..(k_tap + ch) * w + n];
                            c[ch * l + pos..ch * l + pos + n].copy_from_slice(src);
                        }
                    });
                    self.v_cache[layer].with_fp16_mut(|c| {
                        for ch in 0..kvd {
                            let (head, i) = (ch / hd, ch % hd);
                            for t in 0..n {
                                c[(head * l + pos + t) * hd + i] = kv[(v_tap + ch) * w + t];
                            }
                        }
                    });
                }
            });
        }
        self.pos += n;

        let mut out = vec![0.0f32; d * w];
        self.x[cur].read_fp16_as_f32(&mut out, 0, d, w);
        let mut hidden = vec![0.0f32; d * n];
        for c in 0..d {
            hidden[c * n..(c + 1) * n].copy_from_slice(&out[c * w..c * w + n]);
        }
        Ok(hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> ExtendConfig {
        ExtendConfig {
            dim: 64,
            hidden_dim: 128,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            width: 8,
            capacity: 64,
            rms_norm_eps: 1e-6,
            weights: WeightFormat::Int8,
        }
    }

    #[test]
    fn inputs_are_declared_in_binding_order() {
        let cfg = tiny();
        // Sized for the largest projection; each takes its own [out, in].
        let w = vec![0.01f32; 128 * 64];
        let norm = vec![1.0f32; 64];
        let t = LayerTensors {
            rms_att: &norm,
            wq: &w[..64 * 64],
            wk: &w[..32 * 64],
            wv: &w[..32 * 64],
            wo: &w[..64 * 64],
            q_norm: &norm[..16],
            k_norm: &norm[..16],
            rms_ffn: &norm,
            w_gate: &w,
            w_up: &w,
            w_down: &w,
        };
        let out = gen_extend_chunk(&cfg, &[t]).unwrap();
        let sig = out
            .mil_text
            .lines()
            .find(|l| l.contains("func main"))
            .unwrap();
        let names: Vec<&str> = sig
            .split("]> ")
            .skip(1)
            .map(|p| p.split([',', ')']).next().unwrap())
            .collect();
        assert_eq!(names, ["a_x", "b_cos", "c_sin", "d_mask", "k_00", "v_00"]);
        assert!(out.mil_text.contains("constexpr_blockwise_shift_scale"));
        assert!(out.mil_text.contains("} -> (o_kv, o_x);"));
    }

    #[test]
    fn rope_tables_start_at_the_position() {
        let (cos, sin) = rope_tables(4, 3, 2, 10_000.0);
        // Pair 0 rotates at frequency 1: positions 3 and 4.
        assert!((cos[0] - 3f32.cos()).abs() < 1e-6 && (sin[1] - 4f32.sin()).abs() < 1e-6);
    }
}
