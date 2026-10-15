//! MoE expert-permutation sensitivity tests for full-model merging.
//!
//! These tests pin down the known limitation documented in
//! [`pmetal_merge::moe_merge_caveat`]: full-model merge methods operate on
//! tensor names only, so merging MoE checkpoints whose routed experts have
//! been specialised in different orders across training runs produces a
//! different — and generally incoherent — merged expert bank.
//!
//! We verify two things:
//!
//! 1. `contains_moe_experts` correctly detects the routed-expert naming
//!    patterns used by every supported MoE arch (DeepSeek, Qwen3MoE,
//!    Qwen3Next, GPT-OSS, Llama 4, Granite MoE).
//!
//! 2. A naive TIES merge is permutation-sensitive: swapping two experts in
//!    one checkpoint before merging changes the result. This is the
//!    observable evidence of the caveat — if a future change makes merge
//!    expert-aware, this test will flip and should be updated.

use pmetal_merge::{TiesMerge, contains_moe_experts, moe_merge_caveat};
use pmetal_runtime::compat::Array;

#[test]
fn detects_moe_expert_names_across_supported_archs() {
    // Representative tensor names from every MoE arch we support.
    let moe_names = vec![
        // DeepSeek
        "model.layers.0.mlp.experts.3.gate_proj.weight".to_string(),
        // Qwen3MoE / Qwen3Next
        "model.layers.12.mlp.experts.0.up_proj.weight".to_string(),
        // GPT-OSS
        "model.layers.4.block_sparse_moe.experts.1.w1.weight".to_string(),
        // Llama 4
        "language_model.model.layers.7.feed_forward.experts.2.gate_proj.weight".to_string(),
        // Granite MoE
        "model.layers.5.block_sparse_moe.experts.6.input_linear.weight".to_string(),
    ];
    assert!(contains_moe_experts(&moe_names));

    // Dense-only names must NOT match.
    let dense_names = vec![
        "model.embed_tokens.weight".to_string(),
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        "model.layers.0.mlp.gate_proj.weight".to_string(),
        "lm_head.weight".to_string(),
    ];
    assert!(!contains_moe_experts(&dense_names));
}

#[test]
fn caveat_docstring_is_populated() {
    // Sanity: the caveat string exists and is non-trivial, so TUI / log
    // surfaces can print it without checking for empty content.
    let msg = moe_merge_caveat();
    assert!(msg.len() > 32, "caveat message should be informative");
    assert!(
        msg.contains("MoE") || msg.contains("expert"),
        "caveat should reference MoE/experts: {msg}"
    );
}

/// Build a rank-1 tensor filled with a scalar — stand-in for an expert's
/// collapsed weight vector. Different values mark different "expert identities".
fn expert_stub(value: f32) -> Array {
    Array::from_f32_slice(&[value; 8], &[8])
}

#[test]
fn ties_merge_is_permutation_sensitive_across_experts() {
    // Construct two "checkpoints" with 2 experts each, branching from a
    // common base. In checkpoint A, expert 0 learned `a0`, expert 1 learned
    // `a1`. In checkpoint B, expert 0 learned `b0`, expert 1 learned `b1`.
    // A "permuted B" has its experts swapped — same semantics, different
    // index order.
    let base0 = expert_stub(0.0);
    let base1 = expert_stub(0.0);

    let a0 = expert_stub(1.0);
    let a1 = expert_stub(2.0);

    let b0 = expert_stub(3.0);
    let b1 = expert_stub(4.0);

    let densities = [1.0_f32, 1.0_f32];
    let weights = [0.5_f32, 0.5_f32];

    // Merge expert 0 slot across (A, B).
    let task_a0 = a0.subtract(&base0);
    let task_b0 = b0.subtract(&base0);
    let merged_0_ordered = TiesMerge::merge_task_vectors(
        &[task_a0.clone(), task_b0.clone()],
        &densities,
        &weights,
        1.0,
        true,
    )
    .expect("ordered merge");

    // Merge expert 0 slot across (A, B_permuted) — i.e. A's expert 0 is now
    // paired with B's *expert 1*.
    let task_b1 = b1.subtract(&base0);
    let merged_0_permuted =
        TiesMerge::merge_task_vectors(&[task_a0, task_b1], &densities, &weights, 1.0, true)
            .expect("permuted merge");

    merged_0_ordered.eval();
    merged_0_permuted.eval();
    let ordered: Vec<f32> = merged_0_ordered.as_slice::<f32>().to_vec();
    let permuted: Vec<f32> = merged_0_permuted.as_slice::<f32>().to_vec();

    let max_abs_diff = ordered
        .iter()
        .zip(permuted.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // If this assertion ever fails, merge has become expert-aware. Update
    // docs / caveat / callers accordingly.
    assert!(
        max_abs_diff > 0.1,
        "TIES merge was insensitive to expert permutation (diff={max_abs_diff}); \
         if merge became expert-aware, update the MoE caveat in merge.rs"
    );

    // Also verify symmetry: merging the "other" expert slot with the same
    // permutation reproduces the complementary difference — not mandatory,
    // just a coherence check.
    let task_a1 = a1.subtract(&base1);
    let merged_1_ordered = TiesMerge::merge_task_vectors(
        &[task_a1.clone(), b1.subtract(&base1)],
        &densities,
        &weights,
        1.0,
        true,
    )
    .expect("ordered merge 1");
    let merged_1_permuted = TiesMerge::merge_task_vectors(
        &[task_a1, b0.subtract(&base1)],
        &densities,
        &weights,
        1.0,
        true,
    )
    .expect("permuted merge 1");
    merged_1_ordered.eval();
    merged_1_permuted.eval();
    let ordered1: Vec<f32> = merged_1_ordered.as_slice::<f32>().to_vec();
    let permuted1: Vec<f32> = merged_1_permuted.as_slice::<f32>().to_vec();
    let diff1 = ordered1
        .iter()
        .zip(permuted1.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        diff1 > 0.1,
        "symmetry check: expert-1 slot also permutation-sensitive"
    );
}
