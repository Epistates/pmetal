//! Architecture-agnostic tensor parallelism plan builders.
//!
//! These functions generate [`ShardingPlan`]s by inspecting model configuration
//! (as `serde_json::Value`) without depending on pmetal-models types. This keeps
//! the dependency direction clean: pmetal-distributed → (no model deps).
//!
//! # Supported Block Types
//!
//! - Standard multi-head attention (Llama, Mistral, Gemma, etc.)
//! - SwiGLU FFN blocks
//! - GDN (gated delta net) blocks (Qwen 3.5)
//! - MoE (shared expert + routed experts)
//!
//! # Reference
//!
//! The Qwen 3.5 `shard()` method in mlx-lm serves as the authoritative
//! reference for hybrid GDN+Attention+MoE tensor parallelism.

use super::sharding::{ShardingDirective, ShardingPlan};
use anyhow::Result;

/// Build a TP plan for a standard multi-head attention block.
///
/// # Convention
///
/// MLX `nn.Linear` stores weights as `[out_features, in_features]`:
///
/// - **Column-shard** (`AllToSharded`) splits `out_features` (axis 0).
///   Each rank produces a slice of the output; no communication needed.
/// - **Row-shard** (`ShardedToAll`) splits `in_features` (axis 1).
///   Each rank produces a partial sum; an `all_sum` reduces to the
///   full output.
///
/// For attention:
/// - Q / K / V projections: column-sharded (axis 0) — each rank owns a
///   slice of the heads.
/// - O projection: row-sharded (axis 1) — each rank consumes its head
///   slice and contributes a partial hidden-dim output; allreduce sums.
/// - Biases on column-sharded layers: sharded (axis 0, matches weight).
/// - Bias on the row-sharded O projection: **Replicated** — it is added
///   after the allreduce, so every rank needs the full vector.
///
/// # GQA
///
/// When `n_kv_heads < n_heads`, the KV projections have
/// `out_features = n_kv_heads * head_dim`. As long as
/// `world_size <= n_kv_heads` the split stays aligned. When
/// `world_size > n_kv_heads` the caller must replicate KV heads before
/// sharding (MLX's standard GQA handling).
pub fn plan_attention(
    prefix: &str,
    _n_heads: usize,
    _n_kv_heads: usize,
) -> Vec<(String, ShardingDirective)> {
    vec![
        (
            format!("{prefix}.q_proj.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.k_proj.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.v_proj.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.o_proj.weight"),
            ShardingDirective::ShardedToAll { axis: 1 },
        ),
        // Biases on the column-sharded projections follow the weight
        // sharding (axis 0, = out_features per rank).
        (
            format!("{prefix}.q_proj.bias"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.k_proj.bias"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.v_proj.bias"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        // O bias is applied *after* allreduce — it must be the full
        // hidden-dim vector on every rank.
        (format!("{prefix}.o_proj.bias"), ShardingDirective::Replicated),
    ]
}

/// Build a TP plan for a DeepSeek-V2/V3 Multi-Latent Attention (MLA) block.
///
/// MLA factors Q and KV through small latent spaces instead of projecting
/// directly. The column/row-shard shape is therefore different from
/// standard attention:
///
/// ```text
///                                shape                  sharding
/// q_a_proj         (optional)   [q_lora, hidden]       Replicated
/// q_a_layernorm    (optional)   [q_lora]               Replicated
/// q_b_proj         (optional)   [H*q_head, q_lora]     AllToSharded a=0
/// q_proj           (when no-lora) [H*q_head, hidden]   AllToSharded a=0
/// kv_a_proj_with_mqa            [kv_lora+rope, hidden] Replicated
/// kv_a_layernorm                [kv_lora]              Replicated
/// kv_b_proj                     [H*(nope+v), kv_lora]  AllToSharded a=0
/// o_proj                        [hidden, H*v_head]     ShardedToAll a=1
/// ```
///
/// # Why replicate the `_a_` down-projections?
///
/// `q_a_proj` and `kv_a_proj_with_mqa` compress from `hidden` to a small
/// latent (`q_lora_rank`, `kv_lora_rank`). The `_b_` up-projections then
/// expand into per-head outputs. Sharding the down-projection would
/// require a synchronous reduction before the up-projection, destroying
/// the point of MLA (small latents shouldn't cost communication).
/// Replicating them is cheap — `hidden * q_lora` is typically < 1% of
/// the attention weight budget.
///
/// `kv_a_proj_with_mqa` also carries the `qk_rope_head_dim` single-head
/// (MQA) rotary channel; splitting that single head across ranks is
/// meaningless, so we keep the whole projection replicated.
///
/// # q_lora_rank handling
///
/// `DeepSeek-V2-Lite` and some V3 variants skip the Q factorization and
/// use a direct `q_proj`. Pass `q_lora_rank = None` in that case and the
/// plan will emit a single column-sharded `q_proj.weight` entry instead
/// of the `q_a_*` / `q_b_*` pair.
///
/// DeepSeek Linear layers are bias-less, so no `.bias` directives are
/// emitted.
///
/// # Reference
///
/// `pmetal-models/src/architectures/deepseek.rs::DeepSeekAttention` —
/// the ground-truth weight layout.
pub fn plan_mla(
    prefix: &str,
    q_lora_rank: Option<usize>,
) -> Vec<(String, ShardingDirective)> {
    let mut directives = Vec::new();

    match q_lora_rank {
        Some(_) => {
            directives.push((
                format!("{prefix}.q_a_proj.weight"),
                ShardingDirective::Replicated,
            ));
            directives.push((
                format!("{prefix}.q_a_layernorm.weight"),
                ShardingDirective::Replicated,
            ));
            directives.push((
                format!("{prefix}.q_b_proj.weight"),
                ShardingDirective::AllToSharded { axis: 0 },
            ));
        }
        None => {
            directives.push((
                format!("{prefix}.q_proj.weight"),
                ShardingDirective::AllToSharded { axis: 0 },
            ));
        }
    }

    directives.push((
        format!("{prefix}.kv_a_proj_with_mqa.weight"),
        ShardingDirective::Replicated,
    ));
    directives.push((
        format!("{prefix}.kv_a_layernorm.weight"),
        ShardingDirective::Replicated,
    ));
    directives.push((
        format!("{prefix}.kv_b_proj.weight"),
        ShardingDirective::AllToSharded { axis: 0 },
    ));
    directives.push((
        format!("{prefix}.o_proj.weight"),
        ShardingDirective::ShardedToAll { axis: 1 },
    ));

    directives
}

/// Build a TP plan for a SwiGLU FFN block.
///
/// `gate_proj` and `up_proj` are column-sharded (axis 0 = out_features),
/// `down_proj` is row-sharded (axis 1 = in_features). See
/// [`plan_attention`] for the full weight-layout convention.
pub fn plan_ffn(prefix: &str) -> Vec<(String, ShardingDirective)> {
    vec![
        (
            format!("{prefix}.gate_proj.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.up_proj.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.down_proj.weight"),
            ShardingDirective::ShardedToAll { axis: 1 },
        ),
    ]
}

/// Build a TP plan for a GDN (gated delta net) block.
///
/// Follows the Qwen 3.5 shard() pattern:
/// - All input projections (qkv, z, b, a) are column-sharded
/// - Conv1d is custom-sharded by key_dim
/// - Output projection is row-sharded
/// - Per-head parameters (dt_bias, A_log) are split by heads
///
/// # Reference
///
/// mlx-lm `qwen3_5.py` lines 393-440
pub fn plan_gdn(
    prefix: &str,
    _key_dim: usize,
    _n_heads: usize,
) -> Vec<(String, ShardingDirective)> {
    vec![
        // Input projections: column-shard
        (
            format!("{prefix}.in_proj_qkv.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.in_proj_z.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.in_proj_b.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.in_proj_a.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        // Conv1d (depthwise): shard along output channels (= key_dim * n_heads)
        (
            format!("{prefix}.conv1d.weight"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        // Output projection: row-shard on in_features (axis 1). GDN's
        // o_proj mirrors attention's o_proj — each rank holds a slice of
        // the input (head) axis and an allreduce sums the partial hidden
        // outputs.
        (
            format!("{prefix}.o_proj.weight"),
            ShardingDirective::ShardedToAll { axis: 1 },
        ),
        // Per-head parameters: shard along head dimension (axis 0)
        (
            format!("{prefix}.dt_bias"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
        (
            format!("{prefix}.A_log"),
            ShardingDirective::AllToSharded { axis: 0 },
        ),
    ]
}

/// Build a TP plan for a DeepSeek V2/V3 MoE block.
///
/// DeepSeek differs from Qwen3-style MoE in three ways:
///
/// 1. **Shared experts** is plural (`shared_experts.*`) — Qwen3 uses
///    singular `shared_expert.*`.
/// 2. **No `shared_expert_gate`** — DeepSeek's shared branch is always
///    mixed in without a gating scalar.
/// 3. **Stacked routed experts live under `switch_mlp.*`** (the
///    pmetal bridge loader rewrites `experts.{e}.{m}.weight` into
///    `switch_mlp.{m}.weight` with leading axis 0 = expert). The plan
///    targets these stacked names so the expert dimension aligns with
///    [`ShardingDirective::ExpertSharded`].
///
/// `gate.weight` (the router) is replicated: every rank needs the full
/// router to score all experts before dispatch.
pub fn plan_deepseek_moe(
    prefix: &str,
    total_experts: usize,
    has_shared_experts: bool,
) -> Vec<(String, ShardingDirective)> {
    let mut directives = Vec::new();

    // Router: replicated.
    directives.push((
        format!("{prefix}.gate.weight"),
        ShardingDirective::Replicated,
    ));

    // Shared experts: standard FFN sharding under `shared_experts.*`.
    if has_shared_experts {
        let shared = format!("{prefix}.shared_experts");
        directives.extend(plan_ffn(&shared));
    }

    // Routed experts: stacked under `switch_mlp.*` with leading expert
    // axis. Shard along expert dim — each rank owns a subset of experts.
    for suffix in &["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
        directives.push((
            format!("{prefix}.switch_mlp.{suffix}"),
            ShardingDirective::ExpertSharded { total_experts },
        ));
    }

    directives
}

/// Build a TP plan for a Qwen2/Qwen3-MoE / Qwen3-Next MoE block.
///
/// Qwen-family runtime key layout (after `qwen3_native` loader stacks experts):
///
/// ```text
/// {prefix}.gate.weight                         [E, hidden]      Replicated (router)
/// {prefix}.switch_mlp.gate_proj.weight         [E, I, hidden]   ExpertSharded
/// {prefix}.switch_mlp.up_proj.weight           [E, I, hidden]   ExpertSharded
/// {prefix}.switch_mlp.down_proj.weight         [E, hidden, I]   ExpertSharded
/// {prefix}.shared_expert.{gate,up,down}_proj.weight              plan_ffn
/// {prefix}.shared_expert_gate.weight           [1, hidden]      Replicated (scalar gate)
/// ```
///
/// # Reference
///
/// `pmetal-bridge/src/qwen3_native/load.rs` — ground-truth weight
/// stacking. The loader rewrites per-expert HF keys into
/// `switch_mlp.{proj}.weight` so the expert dimension is axis 0.
pub fn plan_qwen_moe(prefix: &str, total_experts: usize) -> Vec<(String, ShardingDirective)> {
    let mut directives = Vec::new();

    // Router: replicated.
    directives.push((
        format!("{prefix}.gate.weight"),
        ShardingDirective::Replicated,
    ));

    // Shared expert: standard FFN sharding.
    let shared = format!("{prefix}.shared_expert");
    directives.extend(plan_ffn(&shared));

    // Shared expert gate scalar: replicated (applied per token after allreduce).
    directives.push((
        format!("{prefix}.shared_expert_gate.weight"),
        ShardingDirective::Replicated,
    ));

    // Routed experts (stacked): shard along expert dim (axis 0).
    for suffix in &["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
        directives.push((
            format!("{prefix}.switch_mlp.{suffix}"),
            ShardingDirective::ExpertSharded { total_experts },
        ));
    }

    directives
}

/// Build a TP plan for a Llama 4 MoE block.
///
/// Llama 4 runtime key layout:
///
/// ```text
/// {prefix}.router.weight                       [E, hidden]      Replicated
/// {prefix}.experts.gate_proj.weight            [E, hidden, I]   ExpertSharded
/// {prefix}.experts.up_proj.weight              [E, hidden, I]   ExpertSharded
/// {prefix}.experts.down_proj.weight            [E, I, hidden]   ExpertSharded
/// {prefix}.shared_expert.{gate,up,down}_proj.weight              plan_ffn
/// ```
///
/// Differences from Qwen:
///
/// - Block prefix is `.feed_forward.*` (Qwen uses `.mlp.*`). The caller
///   passes the right prefix; this helper only cares about the sub-keys.
/// - Router name is `router` (Qwen uses `gate`).
/// - Routed experts stack under `.experts.*` (Qwen uses `.switch_mlp.*`).
/// - No `shared_expert_gate` scalar — the shared branch is summed in
///   without a per-token gate.
///
/// # Reference
///
/// `pmetal-bridge/src/llama4_native/weights.rs`.
pub fn plan_llama4_moe(
    prefix: &str,
    total_experts: usize,
) -> Vec<(String, ShardingDirective)> {
    let mut directives = Vec::new();

    // Router: replicated.
    directives.push((
        format!("{prefix}.router.weight"),
        ShardingDirective::Replicated,
    ));

    // Shared expert: standard FFN sharding.
    let shared = format!("{prefix}.shared_expert");
    directives.extend(plan_ffn(&shared));

    // Routed experts (stacked): shard along expert dim (axis 0).
    for suffix in &["gate_proj.weight", "up_proj.weight", "down_proj.weight"] {
        directives.push((
            format!("{prefix}.experts.{suffix}"),
            ShardingDirective::ExpertSharded { total_experts },
        ));
    }

    directives
}

/// Build a TP plan for a GPT-OSS MoE block.
///
/// GPT-OSS runtime key layout (pure routed, no shared expert):
///
/// ```text
/// {prefix}.router.weight                       [hidden, E]      Replicated
/// {prefix}.experts.gate_proj.{weight,bias}                      ExpertSharded
/// {prefix}.experts.up_proj.{weight,bias}                        ExpertSharded
/// {prefix}.experts.down_proj.{weight,bias}                      ExpertSharded
/// ```
///
/// Differences from Qwen/Llama4:
///
/// - No shared expert at all.
/// - Per-expert biases are present (clamped SwiGLU) and must shard on
///   axis 0 alongside the weights so each rank has the biases for the
///   experts it owns.
/// - Router weight is stored as `[hidden, E]` (GPT-OSS does not
///   pre-transpose; see `gpt_oss_native/weights.rs`).
///
/// # Reference
///
/// `pmetal-bridge/src/gpt_oss_native/weights.rs`.
pub fn plan_gpt_oss_moe(
    prefix: &str,
    total_experts: usize,
) -> Vec<(String, ShardingDirective)> {
    let mut directives = Vec::new();

    // Router: replicated.
    directives.push((
        format!("{prefix}.router.weight"),
        ShardingDirective::Replicated,
    ));

    // Routed experts (stacked) with per-expert biases: shard along
    // expert dim (axis 0) — biases are [E, out] so axis 0 is still
    // the expert axis.
    for proj in &["gate_proj", "up_proj", "down_proj"] {
        for kind in &["weight", "bias"] {
            directives.push((
                format!("{prefix}.experts.{proj}.{kind}"),
                ShardingDirective::ExpertSharded { total_experts },
            ));
        }
    }

    directives
}

/// Build a full model sharding plan from architecture type and config.json.
///
/// Reads the architecture type and config fields to produce a complete
/// sharding plan without depending on pmetal-models types.
///
/// # Supported Architectures
///
/// - `"Qwen3NextForCausalLM"` / `"qwen3_next"` — Hybrid GDN+Attention+MoE
/// - `"LlamaForCausalLM"` / `"llama"` — Standard transformer
/// - `"MistralForCausalLM"` / `"mistral"` — Standard transformer + sliding window
/// - `"Qwen2MoeForCausalLM"` / `"qwen2_moe"` — MoE transformer
/// - `"DeepseekV3ForCausalLM"` / `"deepseek_v3"` — MoE with shared expert
/// - Generic fallback for standard transformer architectures
pub fn build_plan(
    arch: &str,
    config: &serde_json::Value,
    world_size: usize,
) -> Result<ShardingPlan> {
    if world_size <= 1 {
        return Ok(ShardingPlan::new());
    }

    let num_layers = config
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;

    let n_heads = config
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;

    let n_kv_heads = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;

    // DeepSeek-V2/V3 route counts experts under `n_routed_experts`.
    let num_experts = config
        .get("num_experts")
        .or_else(|| config.get("num_local_experts"))
        .or_else(|| config.get("n_routed_experts"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    let key_dim = config
        .get("key_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(128) as usize;

    // Detect hybrid GDN+Attention models by checking for GDN-specific config
    let is_hybrid_gdn = config.get("gdn_heads").is_some()
        || config.get("mixer_pattern").is_some()
        || arch.contains("qwen3_next")
        || arch.contains("Qwen3Next");

    // Detect DeepSeek-V2/V3 (MLA attention + specialized MoE naming).
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let is_deepseek = arch.contains("Deepseek")
        || arch.contains("deepseek")
        || model_type == "deepseek_v2"
        || model_type == "deepseek_v3";

    // MoE family detection. Every family has its own block name, router
    // key, and expert-stacking convention — see the per-arch planners.
    let is_llama4 = arch.contains("Llama4") || model_type == "llama4";
    let is_gpt_oss = arch.contains("GptOss")
        || arch.contains("gpt_oss")
        || model_type == "gpt_oss";
    // Qwen-family MoE covers Qwen2-MoE, Qwen3-MoE, and Qwen3-Next's sparse
    // block. `is_hybrid_gdn` already routes Qwen3-Next's recurrent layers.
    let is_qwen_moe = arch.contains("Qwen2Moe")
        || arch.contains("qwen2_moe")
        || arch.contains("Qwen3Moe")
        || arch.contains("qwen3_moe")
        || is_hybrid_gdn;

    // MoE layer frequency. Qwen-family uses `decoder_sparse_step` (1-based:
    // a layer is MoE when `(layer + 1) % step == 0`). Llama4 uses
    // `interleave_moe_layer_step` (0-based: `layer % step == 0`). GPT-OSS
    // is MoE on every layer.
    let qwen_sparse_step = config
        .get("decoder_sparse_step")
        .and_then(|v| v.as_u64())
        .unwrap_or(1)
        .max(1) as usize;
    let llama4_interleave = config
        .get("interleave_moe_layer_step")
        .and_then(|v| v.as_u64())
        .unwrap_or(1)
        .max(1) as usize;

    // DeepSeek MLA parameters (only relevant when is_deepseek).
    let deepseek_q_lora_rank = config
        .get("q_lora_rank")
        .and_then(|v| v.as_u64())
        .map(|r| r as usize);
    let deepseek_has_shared = config
        .get("n_shared_experts")
        .and_then(|v| v.as_u64())
        .map(|n| n > 0)
        .unwrap_or(false);
    let deepseek_first_dense = config
        .get("first_k_dense_replace")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let deepseek_moe_freq = config
        .get("moe_layer_freq")
        .and_then(|v| v.as_u64())
        .unwrap_or(1)
        .max(1) as usize;

    // Qwen3-Next layer-type resolution.
    //
    // The arch exposes two knobs (see `qwen3_next.rs::is_linear_attention_layer`):
    // 1. An explicit `layer_types: ["linear_attention" | "full_attention", ...]`
    //    array that names each layer by index. When present, it wins.
    // 2. A `full_attention_interval: N` scalar: layer `L` is a full-attention
    //    layer iff `(L + 1) % N == 0`; every other layer uses linear attention
    //    (GDN). Older / community configs sometimes ship the legacy key
    //    `attention_interval` or `attn_layer_period`, both with the same
    //    semantics, so we fall through to those for compatibility.
    //
    // Both defaults match the upstream default (N=4).
    let hybrid_layer_types: Option<Vec<String>> = config
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        });
    let attn_interval = config
        .get("full_attention_interval")
        .or_else(|| config.get("attention_interval"))
        .or_else(|| config.get("attn_layer_period"))
        .and_then(|v| v.as_u64())
        .unwrap_or(4) as usize;

    if is_hybrid_gdn
        && hybrid_layer_types
            .as_ref()
            .is_some_and(|lts| lts.len() != num_layers)
    {
        tracing::warn!(
            "build_plan: Qwen3-Next `layer_types` length ({}) != num_hidden_layers ({}) — \
             falling back to full_attention_interval={attn_interval}",
            hybrid_layer_types.as_ref().unwrap().len(),
            num_layers
        );
    }

    let mut plan = ShardingPlan::new();

    // Embedding: replicated (each rank needs full vocab)
    plan.add("model.embed_tokens.weight", ShardingDirective::Replicated);

    for layer in 0..num_layers {
        let layer_prefix = format!("model.layers.{layer}");

        // ── Attention / recurrent block ─────────────────────────────
        if is_hybrid_gdn {
            // Hybrid GDN + full-attention model. `layer_types` wins over
            // the modular `full_attention_interval` when it's present and
            // the right length (see the validation / warning above).
            let is_attention_layer = match hybrid_layer_types.as_ref() {
                Some(lts) if lts.len() == num_layers => {
                    // Known names: "full_attention" (attention), "linear_attention" (GDN).
                    // Treat anything unrecognised as linear_attention (safer default
                    // — attention sharding would require GQA-compatible heads).
                    lts[layer].as_str() == "full_attention"
                }
                _ => (layer + 1) % attn_interval == 0,
            };

            if is_attention_layer {
                // Full attention layer: standard Q/K/V/O sharding.
                let attn = format!("{layer_prefix}.self_attn");
                for (name, dir) in plan_attention(&attn, n_heads, n_kv_heads) {
                    plan.add(name, dir);
                }
            } else {
                // GDN linear-attention layer. The real weight path is
                // `model.layers.{N}.linear_attn.*` (see
                // `qwen3_next.rs::Qwen3NextDecoderLayer.linear_attn`). The
                // old `.gdn.` block name didn't match any checkpoint key,
                // so the directives were emitted into a void.
                let gdn = format!("{layer_prefix}.linear_attn");
                for (name, dir) in plan_gdn(&gdn, key_dim, n_heads) {
                    plan.add(name, dir);
                }
            }
        } else if is_deepseek {
            // DeepSeek uses Multi-Latent Attention, not standard Q/K/V.
            let attn = format!("{layer_prefix}.self_attn");
            for (name, dir) in plan_mla(&attn, deepseek_q_lora_rank) {
                plan.add(name, dir);
            }
        } else {
            // Standard attention layer
            let attn = format!("{layer_prefix}.self_attn");
            for (name, dir) in plan_attention(&attn, n_heads, n_kv_heads) {
                plan.add(name, dir);
            }
        }

        // ── FFN / MoE block ────────────────────────────────────────
        if is_deepseek {
            // DeepSeek: dense FFN for the first `first_k_dense_replace`
            // layers, MoE for layers where (layer % moe_layer_freq == 0)
            // AFTER the dense prefix.
            let is_moe_layer =
                layer >= deepseek_first_dense && layer % deepseek_moe_freq == 0;
            let block = format!("{layer_prefix}.mlp");
            if is_moe_layer && num_experts > 0 {
                for (name, dir) in
                    plan_deepseek_moe(&block, num_experts, deepseek_has_shared)
                {
                    plan.add(name, dir);
                }
            } else {
                for (name, dir) in plan_ffn(&block) {
                    plan.add(name, dir);
                }
            }
        } else if is_llama4 && num_experts > 0 {
            // Llama4: `{layer}.feed_forward.*` on MoE layers,
            // `{layer}.mlp.*` on dense layers, chosen by
            // `interleave_moe_layer_step`.
            let is_moe_layer = layer % llama4_interleave == 0;
            if is_moe_layer {
                let block = format!("{layer_prefix}.feed_forward");
                for (name, dir) in plan_llama4_moe(&block, num_experts) {
                    plan.add(name, dir);
                }
            } else {
                let ffn = format!("{layer_prefix}.mlp");
                for (name, dir) in plan_ffn(&ffn) {
                    plan.add(name, dir);
                }
            }
        } else if is_gpt_oss && num_experts > 0 {
            // GPT-OSS: every layer is MoE, block prefix is `.mlp`.
            let block = format!("{layer_prefix}.mlp");
            for (name, dir) in plan_gpt_oss_moe(&block, num_experts) {
                plan.add(name, dir);
            }
        } else if is_qwen_moe && num_experts > 0 {
            // Qwen2/Qwen3 MoE and Qwen3-Next: MoE on layers where
            // `(layer + 1) % decoder_sparse_step == 0`; dense otherwise.
            // Block prefix is always `.mlp`.
            let is_moe_layer = (layer + 1) % qwen_sparse_step == 0;
            let block = format!("{layer_prefix}.mlp");
            if is_moe_layer {
                for (name, dir) in plan_qwen_moe(&block, num_experts) {
                    plan.add(name, dir);
                }
            } else {
                for (name, dir) in plan_ffn(&block) {
                    plan.add(name, dir);
                }
            }
        } else {
            let ffn = format!("{layer_prefix}.mlp");
            for (name, dir) in plan_ffn(&ffn) {
                plan.add(name, dir);
            }
        }

        // Layer norms: replicated (small, not worth sharding)
        plan.add(
            format!("{layer_prefix}.input_layernorm.weight"),
            ShardingDirective::Replicated,
        );
        plan.add(
            format!("{layer_prefix}.post_attention_layernorm.weight"),
            ShardingDirective::Replicated,
        );
    }

    // Final norm and lm_head: replicated
    plan.add("model.norm.weight", ShardingDirective::Replicated);
    plan.add("lm_head.weight", ShardingDirective::Replicated);

    tracing::info!(
        "Built TP plan for arch={arch}, layers={num_layers}, heads={n_heads}, \
         kv_heads={n_kv_heads}, experts={num_experts}, hybrid_gdn={is_hybrid_gdn}, \
         sharded_weights={}, world_size={world_size}",
        plan.num_sharded()
    );

    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Look up a directive by name, panicking with context if missing.
    fn expect<'a>(
        dirs: &'a [(String, ShardingDirective)],
        name: &str,
    ) -> &'a ShardingDirective {
        dirs.iter()
            .find_map(|(n, d)| (n == name).then_some(d))
            .unwrap_or_else(|| {
                let names: Vec<&str> = dirs.iter().map(|(n, _)| n.as_str()).collect();
                panic!("missing directive for {name}; have {names:?}")
            })
    }

    #[test]
    fn plan_attention_axes_are_correct() {
        let directives = plan_attention("layer.0.self_attn", 32, 8);
        assert_eq!(directives.len(), 8); // q,k,v,o weights + q,k,v,o biases

        // Column-shard: Q/K/V out_features on axis 0.
        for name in [
            "layer.0.self_attn.q_proj.weight",
            "layer.0.self_attn.k_proj.weight",
            "layer.0.self_attn.v_proj.weight",
        ] {
            match expect(&directives, name) {
                ShardingDirective::AllToSharded { axis } => assert_eq!(*axis, 0),
                other => panic!("{name} expected AllToSharded axis=0, got {other:?}"),
            }
        }

        // Row-shard: O weight on axis 1 (in_features).
        match expect(&directives, "layer.0.self_attn.o_proj.weight") {
            ShardingDirective::ShardedToAll { axis } => assert_eq!(*axis, 1),
            other => panic!("o_proj.weight expected ShardedToAll axis=1, got {other:?}"),
        }

        // O bias is Replicated — applied after allreduce.
        match expect(&directives, "layer.0.self_attn.o_proj.bias") {
            ShardingDirective::Replicated => {}
            other => panic!("o_proj.bias expected Replicated, got {other:?}"),
        }
    }

    #[test]
    fn plan_ffn_has_row_sharded_down_proj_on_axis_1() {
        let directives = plan_ffn("layer.0.mlp");
        assert_eq!(directives.len(), 3);
        match expect(&directives, "layer.0.mlp.down_proj.weight") {
            ShardingDirective::ShardedToAll { axis } => assert_eq!(*axis, 1),
            other => panic!("down_proj expected ShardedToAll axis=1, got {other:?}"),
        }
    }

    #[test]
    fn plan_gdn_has_row_sharded_o_proj_on_axis_1() {
        let directives = plan_gdn("layer.0.gdn", 128, 32);
        assert_eq!(directives.len(), 8);
        match expect(&directives, "layer.0.gdn.o_proj.weight") {
            ShardingDirective::ShardedToAll { axis } => assert_eq!(*axis, 1),
            other => panic!("gdn.o_proj expected ShardedToAll axis=1, got {other:?}"),
        }
    }

    #[test]
    fn plan_qwen_moe_uses_switch_mlp_and_shared_expert_gate() {
        let directives = plan_qwen_moe("layer.0.mlp", 512);
        // gate (1) + shared_expert FFN (3) + shared_expert_gate (1) + switch_mlp (3) = 8
        assert_eq!(directives.len(), 8);

        match expect(&directives, "layer.0.mlp.gate.weight") {
            ShardingDirective::Replicated => {}
            other => panic!("gate expected Replicated, got {other:?}"),
        }
        match expect(&directives, "layer.0.mlp.shared_expert_gate.weight") {
            ShardingDirective::Replicated => {}
            other => panic!("shared_expert_gate expected Replicated, got {other:?}"),
        }
        for name in [
            "layer.0.mlp.switch_mlp.gate_proj.weight",
            "layer.0.mlp.switch_mlp.up_proj.weight",
            "layer.0.mlp.switch_mlp.down_proj.weight",
        ] {
            match expect(&directives, name) {
                ShardingDirective::ExpertSharded { total_experts } => {
                    assert_eq!(*total_experts, 512)
                }
                other => panic!("{name} expected ExpertSharded, got {other:?}"),
            }
        }
        // Singular `shared_expert`, not plural — Qwen-family convention.
        match expect(&directives, "layer.0.mlp.shared_expert.down_proj.weight") {
            ShardingDirective::ShardedToAll { axis } => assert_eq!(*axis, 1),
            other => panic!("shared_expert.down_proj expected ShardedToAll axis=1, got {other:?}"),
        }
        assert!(directives.iter().all(|(n, _)| !n.contains("shared_experts.")));
    }

    #[test]
    fn plan_llama4_moe_uses_feed_forward_router_and_shared_expert() {
        let directives = plan_llama4_moe("layer.0.feed_forward", 16);
        // router (1) + shared_expert FFN (3) + experts (3) = 7
        assert_eq!(directives.len(), 7);

        // Router key is `router`, NOT `gate`.
        match expect(&directives, "layer.0.feed_forward.router.weight") {
            ShardingDirective::Replicated => {}
            other => panic!("router expected Replicated, got {other:?}"),
        }

        // Routed experts stack under `experts.*` (not `switch_mlp.*`).
        for name in [
            "layer.0.feed_forward.experts.gate_proj.weight",
            "layer.0.feed_forward.experts.up_proj.weight",
            "layer.0.feed_forward.experts.down_proj.weight",
        ] {
            match expect(&directives, name) {
                ShardingDirective::ExpertSharded { total_experts } => {
                    assert_eq!(*total_experts, 16)
                }
                other => panic!("{name} expected ExpertSharded, got {other:?}"),
            }
        }

        // No shared_expert_gate scalar — Llama4 lacks it.
        assert!(
            directives
                .iter()
                .all(|(n, _)| !n.contains("shared_expert_gate"))
        );
        // No `gate.weight` router name (that's Qwen).
        assert!(
            directives
                .iter()
                .all(|(n, _)| n != "layer.0.feed_forward.gate.weight")
        );
    }

    #[test]
    fn plan_gpt_oss_moe_has_per_expert_biases_and_no_shared() {
        let directives = plan_gpt_oss_moe("layer.0.mlp", 32);
        // router (1) + 3 projections × (weight + bias) = 7
        assert_eq!(directives.len(), 7);

        // Every routed expert projection has BOTH weight and bias
        // sharded over the expert dim.
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            for kind in ["weight", "bias"] {
                let key = format!("layer.0.mlp.experts.{proj}.{kind}");
                match expect(&directives, &key) {
                    ShardingDirective::ExpertSharded { total_experts } => {
                        assert_eq!(*total_experts, 32)
                    }
                    other => panic!("{key} expected ExpertSharded, got {other:?}"),
                }
            }
        }

        // No shared expert at all.
        assert!(directives.iter().all(|(n, _)| !n.contains("shared_expert")));
    }

    #[test]
    fn plan_mla_with_q_lora_covers_factored_q_and_kv() {
        let directives = plan_mla("layer.0.self_attn", Some(1536));
        // q_a_proj + q_a_layernorm + q_b_proj
        // + kv_a_proj_with_mqa + kv_a_layernorm + kv_b_proj + o_proj = 7
        assert_eq!(directives.len(), 7);

        // Down projections: replicated.
        for name in [
            "layer.0.self_attn.q_a_proj.weight",
            "layer.0.self_attn.q_a_layernorm.weight",
            "layer.0.self_attn.kv_a_proj_with_mqa.weight",
            "layer.0.self_attn.kv_a_layernorm.weight",
        ] {
            match expect(&directives, name) {
                ShardingDirective::Replicated => {}
                other => panic!("{name} expected Replicated, got {other:?}"),
            }
        }

        // Up projections: column-shard on out_features (axis 0).
        for name in [
            "layer.0.self_attn.q_b_proj.weight",
            "layer.0.self_attn.kv_b_proj.weight",
        ] {
            match expect(&directives, name) {
                ShardingDirective::AllToSharded { axis } => assert_eq!(*axis, 0),
                other => panic!("{name} expected AllToSharded axis=0, got {other:?}"),
            }
        }

        // O projection: row-shard on in_features (axis 1).
        match expect(&directives, "layer.0.self_attn.o_proj.weight") {
            ShardingDirective::ShardedToAll { axis } => assert_eq!(*axis, 1),
            other => panic!("o_proj expected ShardedToAll axis=1, got {other:?}"),
        }
    }

    #[test]
    fn plan_mla_without_q_lora_uses_direct_q_proj() {
        let directives = plan_mla("layer.0.self_attn", None);
        // q_proj + kv_a_proj_with_mqa + kv_a_layernorm + kv_b_proj + o_proj = 5
        assert_eq!(directives.len(), 5);

        match expect(&directives, "layer.0.self_attn.q_proj.weight") {
            ShardingDirective::AllToSharded { axis } => assert_eq!(*axis, 0),
            other => panic!("q_proj expected AllToSharded axis=0, got {other:?}"),
        }

        // No factored-Q keys should be present.
        assert!(
            directives
                .iter()
                .all(|(n, _)| !n.contains(".q_a_") && !n.contains(".q_b_"))
        );
    }

    #[test]
    fn plan_deepseek_moe_uses_switch_mlp_and_shared_experts() {
        let directives = plan_deepseek_moe("layer.0.mlp", 256, true);
        // gate (1) + shared_experts FFN (3) + switch_mlp (3) = 7
        assert_eq!(directives.len(), 7);

        match expect(&directives, "layer.0.mlp.gate.weight") {
            ShardingDirective::Replicated => {}
            other => panic!("gate expected Replicated, got {other:?}"),
        }

        // Routed experts: expert-sharded over switch_mlp's stacked axis.
        for name in [
            "layer.0.mlp.switch_mlp.gate_proj.weight",
            "layer.0.mlp.switch_mlp.up_proj.weight",
            "layer.0.mlp.switch_mlp.down_proj.weight",
        ] {
            match expect(&directives, name) {
                ShardingDirective::ExpertSharded { total_experts } => {
                    assert_eq!(*total_experts, 256)
                }
                other => panic!("{name} expected ExpertSharded, got {other:?}"),
            }
        }

        // Shared experts follow FFN sharding.
        match expect(&directives, "layer.0.mlp.shared_experts.down_proj.weight") {
            ShardingDirective::ShardedToAll { axis } => assert_eq!(*axis, 1),
            other => panic!("shared_experts.down_proj expected ShardedToAll axis=1, got {other:?}"),
        }

        // DeepSeek does NOT use shared_expert_gate.
        assert!(
            directives
                .iter()
                .all(|(n, _)| !n.contains("shared_expert_gate"))
        );
    }

    #[test]
    fn plan_deepseek_moe_without_shared_skips_shared_block() {
        let directives = plan_deepseek_moe("layer.0.mlp", 256, false);
        // gate (1) + switch_mlp (3) = 4
        assert_eq!(directives.len(), 4);
        assert!(
            directives
                .iter()
                .all(|(n, _)| !n.contains("shared_experts"))
        );
    }

    #[test]
    fn build_plan_world_size_1_is_empty() {
        let config = serde_json::json!({ "num_hidden_layers": 4 });
        let plan = build_plan("llama", &config, 1).unwrap();
        assert_eq!(plan.num_sharded(), 0);
    }

    #[test]
    fn build_plan_standard_transformer() {
        let config = serde_json::json!({
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
        });
        let plan = build_plan("llama", &config, 2).unwrap();
        assert!(plan.num_sharded() > 0);
    }

    #[test]
    fn build_plan_hybrid_gdn_moe_uses_real_linear_attn_block_name() {
        let config = serde_json::json!({
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_experts": 512,
            "gdn_heads": 32,
            // Upstream Qwen3-Next config key.
            "full_attention_interval": 4,
            "key_dim": 128,
        });
        let plan = build_plan("Qwen3NextForCausalLM", &config, 4).unwrap();
        assert!(plan.num_sharded() > 0);

        // GDN layers (0,1,2) must emit directives against the real
        // `linear_attn` block — not the stale `.gdn.` prefix.
        assert!(
            plan.directives
                .contains_key("model.layers.0.linear_attn.in_proj_qkv.weight"),
            "GDN block should be `linear_attn`, not `gdn`"
        );
        assert!(
            !plan.directives
                .contains_key("model.layers.0.gdn.in_proj_qkv.weight"),
            "`gdn` block name is stale and must not appear in the plan"
        );
        // Layer 3 (index 3, i.e. (3+1) % 4 == 0) is full attention.
        assert!(
            plan.directives
                .contains_key("model.layers.3.self_attn.q_proj.weight")
        );
        assert!(
            !plan.directives
                .contains_key("model.layers.3.linear_attn.in_proj_qkv.weight")
        );
    }

    #[test]
    fn build_plan_hybrid_honors_layer_types_override() {
        // Explicit layer_types should override `full_attention_interval`.
        // Here layer 1 is full attention even though interval math says
        // only layer 3 would be.
        let config = serde_json::json!({
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "num_experts": 64,
            "gdn_heads": 16,
            "full_attention_interval": 4,
            "key_dim": 128,
            "layer_types": [
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
            ],
        });
        let plan = build_plan("Qwen3NextForCausalLM", &config, 2).unwrap();

        // Layer 1 picked up attention sharding.
        assert!(
            plan.directives
                .contains_key("model.layers.1.self_attn.q_proj.weight")
        );
        assert!(
            !plan.directives
                .contains_key("model.layers.1.linear_attn.in_proj_qkv.weight")
        );
        // Layer 3, which would be attention by interval math, is now
        // linear because the override says so.
        assert!(
            plan.directives
                .contains_key("model.layers.3.linear_attn.in_proj_qkv.weight")
        );
        assert!(
            !plan.directives
                .contains_key("model.layers.3.self_attn.q_proj.weight")
        );
    }

    #[test]
    fn build_plan_hybrid_layer_types_wrong_length_falls_back_to_interval() {
        // Mismatched `layer_types` length must not silently break the
        // plan — the builder should fall through to the interval path.
        let config = serde_json::json!({
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "num_experts": 64,
            "gdn_heads": 16,
            "full_attention_interval": 4,
            "layer_types": ["linear_attention"],  // wrong length (1 != 4)
        });
        let plan = build_plan("Qwen3NextForCausalLM", &config, 2).unwrap();

        // Interval math: layer 3 is attention, 0/1/2 are linear.
        assert!(
            plan.directives
                .contains_key("model.layers.3.self_attn.q_proj.weight")
        );
        assert!(
            plan.directives
                .contains_key("model.layers.0.linear_attn.in_proj_qkv.weight")
        );
    }

    #[test]
    fn build_plan_deepseek_v3_uses_mla_and_deepseek_moe() {
        // Layer 0: dense (first_k_dense_replace=1). Layers 1..4: MoE.
        let config = serde_json::json!({
            "model_type": "deepseek_v3",
            "num_hidden_layers": 4,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
        });
        let plan = build_plan("DeepseekV3ForCausalLM", &config, 4).unwrap();
        assert!(plan.num_sharded() > 0);

        // MLA attention keys present; no standard q_proj.
        assert!(
            plan.directives
                .contains_key("model.layers.0.self_attn.q_b_proj.weight")
        );
        assert!(
            plan.directives
                .contains_key("model.layers.0.self_attn.kv_b_proj.weight")
        );
        assert!(
            !plan.directives
                .contains_key("model.layers.0.self_attn.k_proj.weight")
        );

        // Layer 0: dense FFN (no switch_mlp).
        assert!(plan.directives.contains_key("model.layers.0.mlp.down_proj.weight"));
        assert!(
            !plan.directives
                .contains_key("model.layers.0.mlp.switch_mlp.gate_proj.weight")
        );

        // Layer 1: MoE (switch_mlp + shared_experts).
        assert!(
            plan.directives
                .contains_key("model.layers.1.mlp.switch_mlp.gate_proj.weight")
        );
        assert!(
            plan.directives
                .contains_key("model.layers.1.mlp.shared_experts.down_proj.weight")
        );
        assert!(plan.directives.contains_key("model.layers.1.mlp.gate.weight"));
    }

    #[test]
    fn build_plan_deepseek_v2_lite_without_q_lora() {
        // DeepSeek-V2-Lite: no q_lora_rank → direct q_proj path.
        let config = serde_json::json!({
            "model_type": "deepseek_v2",
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "first_k_dense_replace": 0,
            "moe_layer_freq": 1,
        });
        let plan = build_plan("DeepseekV2ForCausalLM", &config, 2).unwrap();

        // Direct q_proj (no q_a_proj / q_b_proj).
        assert!(
            plan.directives
                .contains_key("model.layers.0.self_attn.q_proj.weight")
        );
        assert!(
            !plan.directives
                .contains_key("model.layers.0.self_attn.q_a_proj.weight")
        );
    }

    #[test]
    fn build_plan_qwen3_moe_alternates_with_decoder_sparse_step() {
        // Qwen3-MoE with sparse_step=2: only odd layers (1, 3, …) are MoE.
        let config = serde_json::json!({
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_experts": 64,
            "decoder_sparse_step": 2,
        });
        let plan = build_plan("Qwen3MoeForCausalLM", &config, 2).unwrap();

        // Layer 0: dense FFN.
        assert!(
            plan.directives
                .contains_key("model.layers.0.mlp.down_proj.weight")
        );
        assert!(
            !plan.directives
                .contains_key("model.layers.0.mlp.switch_mlp.gate_proj.weight")
        );

        // Layer 1: MoE — switch_mlp + shared_expert + shared_expert_gate.
        assert!(
            plan.directives
                .contains_key("model.layers.1.mlp.switch_mlp.gate_proj.weight")
        );
        assert!(
            plan.directives
                .contains_key("model.layers.1.mlp.shared_expert.down_proj.weight")
        );
        assert!(
            plan.directives
                .contains_key("model.layers.1.mlp.shared_expert_gate.weight")
        );
    }

    #[test]
    fn build_plan_llama4_uses_feed_forward_and_interleave() {
        // Llama4 with interleave=2: only even layers are MoE, others dense.
        let config = serde_json::json!({
            "model_type": "llama4",
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_local_experts": 16,
            "interleave_moe_layer_step": 2,
        });
        let plan = build_plan("Llama4ForCausalLM", &config, 4).unwrap();

        // Layer 0: MoE block at `feed_forward.*`.
        assert!(
            plan.directives
                .contains_key("model.layers.0.feed_forward.router.weight")
        );
        assert!(
            plan.directives
                .contains_key("model.layers.0.feed_forward.experts.gate_proj.weight")
        );
        assert!(
            plan.directives
                .contains_key("model.layers.0.feed_forward.shared_expert.down_proj.weight")
        );
        // No Qwen-style `gate.weight` router, no switch_mlp, no scalar gate.
        assert!(
            !plan.directives
                .contains_key("model.layers.0.feed_forward.gate.weight")
        );
        assert!(
            !plan.directives
                .contains_key("model.layers.0.feed_forward.switch_mlp.gate_proj.weight")
        );
        assert!(
            !plan.directives
                .contains_key("model.layers.0.feed_forward.shared_expert_gate.weight")
        );

        // Layer 1: dense FFN under `mlp.*`.
        assert!(
            plan.directives
                .contains_key("model.layers.1.mlp.down_proj.weight")
        );
        assert!(
            !plan.directives
                .contains_key("model.layers.1.feed_forward.router.weight")
        );
    }

    #[test]
    fn build_plan_gpt_oss_shards_per_expert_biases() {
        // GPT-OSS: every layer is MoE, pure routed, per-expert biases.
        let config = serde_json::json!({
            "model_type": "gpt_oss",
            "num_hidden_layers": 2,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "num_local_experts": 32,
        });
        let plan = build_plan("GptOssForCausalLM", &config, 4).unwrap();

        for proj in ["gate_proj", "up_proj", "down_proj"] {
            for kind in ["weight", "bias"] {
                let key = format!("model.layers.0.mlp.experts.{proj}.{kind}");
                assert!(
                    plan.directives.contains_key(&key),
                    "missing {key} in plan"
                );
                assert!(
                    matches!(
                        plan.directives[&key],
                        ShardingDirective::ExpertSharded { total_experts: 32 }
                    ),
                    "{key} wrong directive"
                );
            }
        }

        // No shared expert / shared_expert_gate anywhere.
        assert!(
            plan.directives
                .keys()
                .all(|k| !k.contains("shared_expert"))
        );
        // Router name is `router`, not `gate`.
        assert!(
            plan.directives
                .contains_key("model.layers.0.mlp.router.weight")
        );
        assert!(
            !plan.directives
                .contains_key("model.layers.0.mlp.gate.weight")
        );
    }
}
