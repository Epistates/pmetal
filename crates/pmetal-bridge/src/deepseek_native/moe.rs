//! Dense SwiGLU MLP and DeepSeek's routed MoE: the router (V3's `noaux_tc`
//! and V2's softmax, plain or group-limited) and the routed experts.

use serde::Deserialize;

use crate::InlineArray;
use crate::dtype;

use super::weights::{Mlp, MoeWeights};

/// How the router turns its logits into expert scores: the config's
/// `scoring_func`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoringFunc {
    /// V3: an independent sigmoid per expert.
    Sigmoid,
    /// V2: a softmax over all experts.
    Softmax,
}

/// How the router picks the top-k experts: the config's `topk_method`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TopkMethod {
    /// V3: the bias-corrected scores, restricted to the `topk_group` groups
    /// whose two best experts score highest; the weights are the uncorrected
    /// scores.
    NoauxTc,
    /// V2: the `top_k` best scores.
    Greedy,
    /// V2 (the 236B): the `top_k` best scores among the `topk_group` groups
    /// whose best expert scores highest.
    GroupLimitedGreedy,
}

pub(super) fn mlp_forward(mlp: &Mlp, x: &InlineArray) -> InlineArray {
    let gate = mlp.gate.matmul_from(x);
    let up = mlp.up.matmul_from(x);
    mlp.down.matmul_from(&InlineArray::fused_swiglu(&gate, &up))
}

/// DeepSeek MoE forward pass: the routed experts, each row through its
/// `top_k` of them weighted by the router, plus the shared expert.
pub(super) fn moe_forward(moe: &MoeWeights, x: &InlineArray, b: i32, s: i32) -> InlineArray {
    let hidden = x.dim(2);
    let x_flat = x.reshape(&[b * s, hidden]);
    let (inds, weights) = moe.router.route(&x_flat);
    let weights = weights.as_dtype(x.dtype_raw());
    let routed = crate::native_moe::switch_glu(
        &x_flat,
        &moe.gate,
        &moe.up,
        &moe.down,
        &inds,
        &weights,
        InlineArray::fused_swiglu,
    );
    let y = match &moe.shared {
        Some(shared) => routed.add(&mlp_forward(shared, &x_flat)),
        None => routed,
    };
    y.reshape(&[b, s, hidden])
}

/// The router: gate weights and the settings of one MoE layer's top-k.
pub(super) struct Router {
    /// `[hidden, experts]`, `f32`: the gate is applied to the `f32` hidden
    /// state, as transformers does, so near-ties between experts resolve as
    /// they do there.
    pub(super) weight: InlineArray,
    /// `[experts]`, `f32`: V3's `e_score_correction_bias`, which steers the
    /// choice and never the weights.
    pub(super) bias: Option<InlineArray>,
    pub(super) scoring: ScoringFunc,
    pub(super) method: TopkMethod,
    pub(super) n_group: i32,
    pub(super) topk_group: i32,
    pub(super) top_k: i32,
    pub(super) routed_scaling_factor: f32,
    pub(super) norm_topk_prob: bool,
}

impl Router {
    /// `[rows, hidden]` to each row's `top_k` experts, `[rows, top_k]` `int32`,
    /// and their weights, `[rows, top_k]` `f32`.
    ///
    /// ```text
    ///   scores  = sigmoid(x @ W) | softmax(x @ W)           (f32)
    ///   choice  = scores + bias                              (noaux_tc only)
    ///   choice  = choice, limited to the best groups         (grouped methods)
    ///   experts = top_k(choice)
    ///   weights = scores[experts], normalised to sum 1 when norm_topk_prob,
    ///             times routed_scaling_factor
    /// ```
    pub(super) fn route(&self, x_flat: &InlineArray) -> (InlineArray, InlineArray) {
        let logits = x_flat.as_dtype(dtype::F32).matmul(&self.weight);
        let scores = match self.scoring {
            ScoringFunc::Sigmoid => logits.sigmoid(),
            ScoringFunc::Softmax => logits.softmax_precise(-1),
        };
        let choice = match &self.bias {
            Some(bias) => scores.add(bias),
            None => scores.clone(),
        };
        let choice = match self.method {
            TopkMethod::Greedy => choice,
            TopkMethod::NoauxTc | TopkMethod::GroupLimitedGreedy => limit_to_groups(
                &choice,
                self.method,
                self.n_group,
                self.topk_group,
                // transformers masks V3's choice to -inf and V2's scores to 0.
                if self.method == TopkMethod::NoauxTc {
                    f32::NEG_INFINITY
                } else {
                    0.0
                },
            ),
        };
        let rows = choice.dim(0);
        // `argpartition(-x, k - 1)[:, :k]` is the top k of `x`. The indices
        // carry no gradient: MLX rejects a gather VJP taken with respect to
        // indices that descend from differentiable scores.
        let inds = choice
            .negative()
            .argpartition(self.top_k - 1, -1)
            .slice(&[0, 0], &[rows, self.top_k])
            .stop_gradient();
        let weights = scores.take_along_axis(&inds, -1);
        let weights = if self.norm_topk_prob {
            let total = weights
                .sum_axis(-1, true)
                .add(&InlineArray::scalar_like(1e-20, &weights));
            weights.divide(&total)
        } else {
            weights
        };
        let scale = InlineArray::scalar_like(self.routed_scaling_factor, &weights);
        (inds, weights.multiply(&scale))
    }
}

/// `choice` `[rows, experts]` with every expert outside the `topk_group` best
/// of `n_group` groups replaced by `fill`.
///
/// A group's score is the sum of its two best experts (`NoauxTc`) or its best
/// expert (`GroupLimitedGreedy`).
fn limit_to_groups(
    choice: &InlineArray,
    method: TopkMethod,
    n_group: i32,
    topk_group: i32,
    fill: f32,
) -> InlineArray {
    let (rows, experts) = (choice.dim(0), choice.dim(1));
    let per_group = experts / n_group;
    let grouped = choice.reshape(&[rows, n_group, per_group]);
    let group_scores = match method {
        TopkMethod::GroupLimitedGreedy => grouped.max_axis(-1, false),
        _ => grouped.topk(per_group.min(2), -1).sum_axis(-1, false),
    };
    // The best `topk_group` groups of each row, as a `[rows, n_group]` mask.
    let best = group_scores
        .negative()
        .argpartition(topk_group - 1, -1)
        .slice(&[0, 0], &[rows, topk_group])
        .stop_gradient();
    let group_ids = InlineArray::arange(n_group, dtype::I32);
    let kept = best
        .reshape(&[rows, topk_group, 1])
        .eq(&group_ids)
        .any(Some(&[1]), false)
        .expand_dims(-1);
    let fill = InlineArray::scalar_like(fill, choice);
    kept.where_cond(&grouped, &fill).reshape(&[rows, experts])
}
