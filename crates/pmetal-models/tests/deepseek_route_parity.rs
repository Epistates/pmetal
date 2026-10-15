//! Numerical-parity test for DeepSeek-V3 group-limited expert routing.
//!
//! Replays a reference from HuggingFace `transformers` 5.10.0.dev0
//! `DeepseekV3MoE::route_tokens_to_experts` (torch 2.12, seed 33) with `n_group=4`,
//! `topk_group=2`, so the group-masking branch is exercised. The reference is a
//! DENSE `[N, num_experts]` weight vector (top-k weights scattered to their
//! expert columns), which is invariant to top-k ordering — the two
//! implementations need not agree on tie-break order to agree on routing.
//!
//! The Rust side is [`DeepseekRouting`], the one router the native engine and
//! the shared-path model both run, applied to the fixture's gate logits; its
//! output is scattered into the same dense form.

mod common;

use common::{fixture_path, load_shard, ref_tensor};

use pmetal_runtime::compat::ops;
use pmetal_runtime::deepseek_native::{DeepseekRouting, ScoringFunc, TopkMethod};

use pmetal_cache::test_utils::{ParityReport, Tolerance, print_report_table};

#[test]
fn deepseek_group_routing_synthetic_parity() {
    // Matches the fixture's `.meta.json`.
    const N_TOKENS: i32 = 5;
    const NUM_EXPERTS: i32 = 16;

    let shard = load_shard(&fixture_path("deepseek_route_reference.safetensors"));
    let gates = ref_tensor(&shard, "gates").clone();
    let bias = ref_tensor(&shard, "e_score_correction_bias").clone();

    let routing = DeepseekRouting {
        scoring: ScoringFunc::Sigmoid,
        method: TopkMethod::NoauxTc,
        n_group: 4,
        topk_group: 2,
        top_k: 4,
        routed_scaling_factor: 1.5,
        norm_topk_prob: true,
    };
    let (top_indices, top_weights) = routing.route(&gates, Some(&bias));
    pmetal_runtime::check_unobserved_error().expect("route");

    // Scatter top-k weights into a dense [N, E] vector, matching the oracle.
    let dense = ops::zeros_dtype(&[N_TOKENS, NUM_EXPERTS], top_weights.dtype());
    let dense = ops::put_along_axis(&dense, &top_indices, &top_weights, -1);

    let report = ParityReport::compute_with_per_position(
        "dense_routing_weights",
        &dense,
        ref_tensor(&shard, "dense_weights"),
        Tolerance::new(1e-5, 1e-5),
    );

    println!("\n== DeepSeek group-limited routing parity report ==");
    print_report_table(std::slice::from_ref(&report));

    assert!(report.passed(), "DeepSeek group-routing parity failed");
}
