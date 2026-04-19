//! Feed-forward + GDN path: dense SwiGLU, SwitchGLU MoE (with shared expert),
//! and the Gated Delta Network step shared across Qwen3-5 / Qwen3-Next.

use crate::InlineArray;

use super::cache::GdnCache;
use super::weights::{LayerWeight, LayerWeights};


// ============================================================================
// Dense MLP forward
// ============================================================================

#[inline(always)]
pub(super) fn dense_mlp_forward(lw: &LayerWeights, mlp_in: &InlineArray) -> InlineArray {
    let gate = lw.mlp_gate_w.as_ref().unwrap().matmul_from(mlp_in);
    let up = lw.mlp_up_w.as_ref().unwrap().matmul_from(mlp_in);
    let activated = InlineArray::fused_swiglu(&gate, &up);
    lw.mlp_down_w.as_ref().unwrap().matmul_from(&activated)
}

// ============================================================================
// MoE forward
// ============================================================================
//
// Mirrors Python's Qwen3NextSparseMoeBlock.__call__:
//
//   gates = softmax(gate(x), axis=-1, precise=True)
//   inds  = argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
//   scores = take_along_axis(gates, inds, axis=-1)
//   if norm_topk_prob: scores /= scores.sum(-1, keepdims=True)
//   y = switch_mlp(x, inds)                        # gather_mm
//   y = (y * scores[..., None]).sum(-2)
//   shared_y = shared_expert(x)
//   shared_y = sigmoid(shared_expert_gate(x)) * shared_y
//   return y + shared_y
//
// Input x: [B, T, hidden].  For decode T=1, B=1 → x is [1, 1, hidden].
// We work in [B*T, hidden] = [S, hidden] throughout, then reshape back.

#[inline]
pub(super) fn moe_switch_glu_input(x_flat: &InlineArray) -> InlineArray {
    debug_assert_eq!(x_flat.ndim(), 2);
    // MLX SwitchGLU does `mx.expand_dims(x, (-2, -3))` before gather_mm. Use
    // positive axes here so insertion order is unambiguous and yields the same
    // `[S, 1, 1, hidden]` layout for flattened `[S, hidden]` inputs.
    x_flat.expand_dims(1).expand_dims(2)
}

#[inline]
fn moe_routed_forward(lw: &LayerWeights, x_flat: &InlineArray) -> InlineArray {
    let s = x_flat.dim(0);
    let top_k = lw.moe_top_k;

    // ── Router ──────────────────────────────────────────────────────────────
    // gates: [S, num_experts]
    let gates = x_flat
        .matmul(lw.moe_router_w.as_ref().unwrap())
        .softmax_precise(-1);

    // Top-k selection: argpartition returns full permutation, take last top_k.
    // inds: [S, num_experts] → slice to [S, top_k]
    let all_inds = gates.argpartition(-top_k, -1);
    let num_experts_dim = gates.dim(1);
    let inds = all_inds.slice(&[0, num_experts_dim - top_k], &[s, num_experts_dim]);

    // Gather expert scores: [S, top_k]
    let mut scores = gates.take_along_axis(&inds, -1);
    if lw.moe_norm_topk_prob {
        let score_sum = scores.sum_axis(-1, true);
        scores = scores.divide(&score_sum);
    }

    // ── Expert dispatch via gather_mm / gather_qmm ─────────────────────────
    //
    // Mirror MLX SwitchGLU rank semantics exactly:
    //   x: [S, hidden] -> [S, 1, 1, hidden]
    //   up/gate gather_mm -> [S, top_k, 1, moe_intermediate]
    //   down gather_mm -> [S, top_k, 1, hidden]
    //   squeeze(-2) -> [S, top_k, hidden]
    //
    // Without these singleton axes, the down projection can reinterpret the
    // sequence axis as an additional batch dimension and produce
    // `[S, top_k, S, hidden]`, which then breaks score broadcasting.
    let switch_in = moe_switch_glu_input(x_flat);
    let x_gate_exp =
        lw.moe_gate_w
            .as_ref()
            .unwrap()
            .gather_mm_from(&switch_in, None, Some(&inds), false);
    let x_up_exp =
        lw.moe_up_w
            .as_ref()
            .unwrap()
            .gather_mm_from(&switch_in, None, Some(&inds), false);

    // Fused swiglu: silu(gate) * up
    let x_act = InlineArray::fused_swiglu(&x_gate_exp, &x_up_exp);

    // gather_mm for down projection: [S, top_k, 1, moe_intermediate] ×
    // [E, moe_intermediate, hidden] → [S, top_k, 1, hidden]
    let y_exp = lw
        .moe_down_w
        .as_ref()
        .unwrap()
        .gather_mm_from(&x_act, None, Some(&inds), false)
        .squeeze(-2);

    // Weighted sum over top_k: [S, top_k, hidden] * [S, top_k, 1] →
    // sum(-2) → [S, hidden]
    let scores_exp = scores.reshape(&[s, top_k, 1]);
    y_exp.multiply(&scores_exp).sum_axis(-2, false)
}

pub(super) fn moe_forward(lw: &LayerWeights, x: &InlineArray) -> InlineArray {
    let b = x.dim(0);
    let t = x.dim(1);
    let h = x.dim(2);
    let s = b * t; // flattened sequence length

    if s == 1 {
        if let (
            Some(router_w),
            Some(LayerWeight::Dense(moe_gate_w)),
            Some(LayerWeight::Dense(moe_up_w)),
            Some(LayerWeight::Dense(moe_down_w)),
            Some(LayerWeight::Dense(shared_gate_w)),
            Some(LayerWeight::Dense(shared_up_w)),
            Some(LayerWeight::Dense(shared_down_w)),
            Some(shared_expert_gate_w),
        ) = (
            lw.moe_router_w.as_ref(),
            lw.moe_gate_w.as_ref(),
            lw.moe_up_w.as_ref(),
            lw.moe_down_w.as_ref(),
            lw.shared_gate_w.as_ref(),
            lw.shared_up_w.as_ref(),
            lw.shared_down_w.as_ref(),
            lw.shared_expert_gate_w.as_ref(),
        ) {
            return InlineArray::compiled_moe_layer_fixed(
                x,
                router_w,
                moe_gate_w,
                moe_up_w,
                moe_down_w,
                shared_gate_w,
                shared_up_w,
                shared_down_w,
                shared_expert_gate_w,
                lw.moe_top_k,
                lw.moe_norm_topk_prob,
            );
        }
    }

    // Flatten to [S, hidden].
    let x_flat = x.reshape(&[s, h]);
    let y_routed = moe_routed_forward(lw, &x_flat);

    // ── Shared expert ────────────────────────────────────────────────────────
    //
    // shared_expert(x): standard SwiGLU MLP with its own gate/up/down weights.
    // shared_expert_gate: Linear(hidden, 1) → sigmoid → scales shared output.
    let sh_gate = lw.shared_gate_w.as_ref().unwrap().matmul_from(&x_flat);
    let sh_up = lw.shared_up_w.as_ref().unwrap().matmul_from(&x_flat);
    let sh_act = InlineArray::fused_swiglu(&sh_gate, &sh_up);
    let sh_out = lw.shared_down_w.as_ref().unwrap().matmul_from(&sh_act);

    // shared_expert_gate: [S, 1] sigmoid gate
    let sh_scale = x_flat
        .matmul(lw.shared_expert_gate_w.as_ref().unwrap())
        .sigmoid();
    let y_shared = sh_out.multiply(&sh_scale);

    // ── Combine ──────────────────────────────────────────────────────────────
    y_routed.add(&y_shared).reshape(&[b, t, h])
}

// ============================================================================
// GDN layer forward
// ============================================================================

pub(super) fn gdn_forward(
    lw: &LayerWeights,
    normed: &InlineArray,
    _b: i32,
    _s: i32,
    cache: &mut GdnCache,
    dtype: i32,
) -> InlineArray {
    let nv = lw.gdn_nv;
    let nk = lw.gdn_nk;
    let dk = lw.gdn_dk;
    let dv = lw.gdn_dv;
    let kd = lw.gdn_kd;
    let cd = lw.gdn_cd;
    let ck = lw.gdn_ck;
    let b = normed.dim(0);
    let s = normed.dim(1);

    // For decode-time T=1 on dense checkpoints, replay the fixed-shape compiled
    // GDN tape instead of rebuilding the full op graph every step.
    if s == 1 {
        if let (
            Some(LayerWeight::Dense(qkv_w)),
            Some(LayerWeight::Dense(z_w)),
            Some(LayerWeight::Dense(b_w)),
            Some(LayerWeight::Dense(a_w)),
            Some(LayerWeight::Dense(out_w)),
        ) = (
            &lw.gdn_qkv_w,
            &lw.gdn_z_w,
            &lw.gdn_b_w,
            &lw.gdn_a_w,
            &lw.gdn_out_w,
        ) {
            let conv_state = cache
                .conv_state
                .take()
                .unwrap_or_else(|| InlineArray::zeros(&[b, ck - 1, cd], dtype));
            let ssm_state = cache
                .ssm_state
                .take()
                .unwrap_or_else(|| InlineArray::zeros(&[b, nv, dv, dk], 10));

            let (output, new_conv, new_state) = InlineArray::compiled_gdn_layer_fixed(
                normed,
                qkv_w,
                z_w,
                b_w,
                a_w,
                lw.gdn_conv_w.as_ref().unwrap(),
                lw.gdn_q_nw.as_ref().unwrap(),
                lw.gdn_k_nw.as_ref().unwrap(),
                lw.gdn_a_log.as_ref().unwrap(),
                lw.gdn_dt_bias.as_ref().unwrap(),
                lw.gdn_norm_w.as_ref().unwrap(),
                out_w,
                &conv_state,
                &ssm_state,
                nv,
                nk,
                dk,
                dv,
                cd,
                ck,
                kd,
                lw.gdn_norm_eps,
            );

            cache.conv_state = Some(new_conv);
            cache.ssm_state = Some(new_state);
            return output;
        }
    }

    // Unified path for all T (T=1 decode and T>1 prefill).
    // Structure mirrors Python's gated_delta_update exactly:
    //   1. 4 separate matmul projections (qkv, z, b, a)
    //   2. Conv1d with fused silu activation
    //   3. split → q/k/v + rms_norm on q/k
    //   4. fused_compute_g (shapeless=True compiled — opaque Compiled node)
    //   5. gdn_metal_step (CustomKernel, outside any compile boundary)
    //   6. fused_precise_swiglu (shapeless=True compiled — opaque Compiled node)
    //   7. out_proj matmul
    let qkv = lw.gdn_qkv_w.as_ref().unwrap().matmul_from(normed);
    let z = lw
        .gdn_z_w
        .as_ref()
        .unwrap()
        .matmul_from(normed)
        .reshape(&[b, s, nv, dv]);
    let b_val = lw.gdn_b_w.as_ref().unwrap().matmul_from(normed);
    let a_val = lw.gdn_a_w.as_ref().unwrap().matmul_from(normed);

    // Conv state: concat previous state + new QKV, take new state, apply conv1d + silu
    let conv_state = cache
        .conv_state
        .take()
        .unwrap_or_else(|| InlineArray::zeros(&[b, ck - 1, cd], dtype));
    let conv_in = conv_state.concatenate_2(&qkv, 1);

    let new_conv = conv_in.slice(&[0, 1, 0], &[b, ck, cd]);
    let conv_out = conv_in
        .conv1d(lw.gdn_conv_w.as_ref().unwrap(), 1, 0, 1, cd)
        .fused_silu();

    // Split conv_out → q [B,S,nk,dk], k [B,S,nk,dk], v [B,S,nv,dv]
    let mut conv_parts = conv_out.split(&[kd, kd * 2], -1);
    let v = conv_parts.pop().unwrap().reshape(&[b, s, nv, dv]);
    let k = conv_parts.pop().unwrap().reshape(&[b, s, nk, dk]);
    let q = conv_parts.pop().unwrap().reshape(&[b, s, nk, dk]);

    // Q/K normalization
    let q = q.rms_norm(lw.gdn_q_nw.as_ref(), 1e-6);
    let k = k.rms_norm(lw.gdn_k_nw.as_ref(), 1e-6);

    // Decay gate: fused compute_g
    let g = InlineArray::fused_compute_g(
        lw.gdn_a_log.as_ref().unwrap(),
        &a_val,
        lw.gdn_dt_bias.as_ref().unwrap(),
    );
    let beta = b_val.sigmoid();

    // GDN Metal kernel recurrence
    let ssm_state = cache
        .ssm_state
        .take()
        .unwrap_or_else(|| InlineArray::zeros(&[b, nv, dv, dk], 10));
    let (out, new_state) = InlineArray::gdn_metal_step(&q, &k, &v, &g, &beta, &ssm_state, s);

    cache.conv_state = Some(new_conv);
    cache.ssm_state = Some(new_state);

    // Output: rms_norm → precise_swiglu → reshape → out_proj
    let out_n = out.rms_norm(lw.gdn_norm_w.as_ref(), lw.gdn_norm_eps);
    let gated = InlineArray::fused_precise_swiglu(&out_n, &z);
    let flat = gated.reshape(&[b, s, -1]);
    lw.gdn_out_w.as_ref().unwrap().matmul_from(&flat)
}
