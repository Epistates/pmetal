//! Dense SwiGLU MLP and DeepSeek's routed MoE: the router (V3's `noaux_tc`
//! and V2's softmax, plain or group-limited) and the routed experts.

use serde::{Deserialize, Serialize};

use crate::InlineArray;
use crate::dtype;

use super::weights::{Mlp, MoeWeights};

/// How the router turns its logits into expert scores: the config's
/// `scoring_func`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoringFunc {
    /// V3: an independent sigmoid per expert.
    Sigmoid,
    /// V2: a softmax over all experts.
    Softmax,
}

/// How the router picks the top-k experts: the config's `topk_method`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

impl ScoringFunc {
    /// What a config without `scoring_func` means: V2 softmax, V3 sigmoid.
    pub fn default_for(model_type: &str) -> Self {
        if model_type == "deepseek_v2" {
            Self::Softmax
        } else {
            Self::Sigmoid
        }
    }
}

impl TopkMethod {
    /// What a config without `topk_method` means: V2 `greedy`, V3 `noaux_tc`.
    pub fn default_for(model_type: &str) -> Self {
        if model_type == "deepseek_v2" {
            Self::Greedy
        } else {
            Self::NoauxTc
        }
    }
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

/// The settings of one MoE layer's top-k, and the top-k itself: DeepSeek's
/// router from its logits on.
///
/// This is the one implementation. The native engine's [`Router`] and the
/// shared-path model (`pmetal_models`' `DeepSeekMoEGate`, which training and
/// the adapters run) both route through [`DeepseekRouting::route`], so the
/// model that is trained picks the experts the model that is served picks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeepseekRouting {
    /// How the logits become scores.
    pub scoring: ScoringFunc,
    /// How the top-k is chosen from them.
    pub method: TopkMethod,
    /// Expert groups the grouped methods pick among (`Greedy` ignores it).
    pub n_group: i32,
    /// Groups the grouped methods keep (`Greedy` ignores it).
    pub topk_group: i32,
    /// Experts per token.
    pub top_k: i32,
    /// Multiplies the final weights.
    pub routed_scaling_factor: f32,
    /// Renormalise the chosen scores to sum 1 before scaling.
    pub norm_topk_prob: bool,
}

impl DeepseekRouting {
    /// Refuse settings that do not fit `experts` routed experts: a top-k that
    /// exceeds them, groups that do not divide them, or kept groups too small
    /// to hold `top_k` experts. The ops would otherwise return placeholder
    /// arrays rather than fail.
    pub fn validate(&self, experts: i32) -> Result<(), String> {
        let (groups, kept, top_k) = (self.n_group, self.topk_group, self.top_k);
        if experts < 1 || top_k < 1 || top_k > experts {
            return Err(format!(
                "num_experts_per_tok {top_k} does not fit n_routed_experts {experts}"
            ));
        }
        if self.method != TopkMethod::Greedy {
            if groups < 1 || experts % groups != 0 {
                return Err(format!(
                    "n_routed_experts {experts} does not divide into n_group {groups}"
                ));
            }
            if kept < 1 || kept > groups {
                return Err(format!("topk_group {kept} is not within n_group {groups}"));
            }
            if top_k > kept * (experts / groups) {
                return Err(format!(
                    "num_experts_per_tok {top_k} exceeds the {} experts of the {kept} kept groups",
                    kept * (experts / groups)
                ));
            }
        }
        Ok(())
    }

    /// `[rows, experts]` `f32` router logits to each row's `top_k` experts,
    /// `[rows, top_k]` `int32`, and their weights, `[rows, top_k]` `f32`.
    /// `bias` is V3's `e_score_correction_bias`, `[experts]` `f32`; it steers
    /// the choice of `NoauxTc` and nothing else.
    ///
    /// ```text
    ///   scores  = sigmoid(logits) | softmax(logits)          (f32)
    ///   choice  = scores + bias                              (noaux_tc only)
    ///   choice  = choice, limited to the best groups         (grouped methods)
    ///   experts = top_k(choice)
    ///   weights = scores[experts], normalised to sum 1 when norm_topk_prob,
    ///             times routed_scaling_factor
    /// ```
    ///
    /// The logits are expected in `f32`, which is how transformers computes
    /// them (`F.linear(x.float(), W.float())`): bf16 logits would round away
    /// the near-ties between experts the choice turns on.
    pub fn route(
        &self,
        logits: &InlineArray,
        bias: Option<&InlineArray>,
    ) -> (InlineArray, InlineArray) {
        let scores = match self.scoring {
            ScoringFunc::Sigmoid => logits.sigmoid(),
            ScoringFunc::Softmax => logits.softmax_precise(-1),
        };
        let choice = match (self.method, bias) {
            (TopkMethod::NoauxTc, Some(bias)) => scores.add(bias),
            _ => scores.clone(),
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

/// The router: the gate weight and the layer's [`DeepseekRouting`].
pub(super) struct Router {
    /// `[hidden, experts]`, `f32`: the gate is applied to the `f32` hidden
    /// state, as transformers does, so near-ties between experts resolve as
    /// they do there.
    pub(super) weight: InlineArray,
    /// `[experts]`, `f32`: V3's `e_score_correction_bias`, which steers the
    /// choice and never the weights.
    pub(super) bias: Option<InlineArray>,
    pub(super) routing: DeepseekRouting,
}

impl Router {
    /// `[rows, hidden]` to each row's experts and their weights, see
    /// [`DeepseekRouting::route`].
    pub(super) fn route(&self, x_flat: &InlineArray) -> (InlineArray, InlineArray) {
        let logits = x_flat.as_dtype(dtype::F32).matmul(&self.weight);
        self.routing.route(&logits, self.bias.as_ref())
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
