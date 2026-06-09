//! Numerical-parity test for DeepSeek-V3 group-limited expert routing.
//!
//! Replays a reference dumped from `mlx_lm.models.deepseek_v3.group_expert_select`
//! (`.strategy/parity/dump_deepseek_route_reference.py`) with `n_group=4`,
//! `topk_group=2`, so the group-masking branch is exercised. The reference is a
//! DENSE `[N, num_experts]` weight vector (top-k weights scattered to their
//! expert columns), which is invariant to top-k ordering.
//!
//! The Rust side runs `noaux_tc_topk` on `sigmoid(gates)` and scatters its
//! output into the same dense form. Before the fix `noaux_tc_topk` ignored
//! `n_group`/`topk_group` and selected experts globally, so experts from masked
//! groups would leak into the dense vector and diverge.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pmetal_bridge::compat::{Array, ops};

use pmetal_mlx::test_utils::{ParityReport, Tolerance, print_report_table};

use pmetal_models::moe_routing::noaux_tc_topk;

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

fn load_shard(path: &Path) -> HashMap<String, Array> {
    let path_str = path.to_str().expect("utf8 path");
    pmetal_bridge::inline_array::load_safetensors_shard(path_str)
        .unwrap_or_else(|| panic!("failed to load safetensors shard at {path_str:?}"))
        .into_iter()
        .collect()
}

fn ref_tensor<'a>(shard: &'a HashMap<String, Array>, key: &str) -> &'a Array {
    shard
        .get(key)
        .unwrap_or_else(|| panic!("reference shard missing key {key:?}"))
}

#[test]
fn deepseek_group_routing_synthetic_parity() {
    // Matches the dumper meta.
    const N_TOKENS: i32 = 5;
    const NUM_EXPERTS: i32 = 16;
    const TOP_K: i32 = 4;
    const N_GROUP: i32 = 4;
    const TOPK_GROUP: i32 = 2;
    const SCALING: f32 = 1.5;

    let shard = load_shard(&fixture_path("deepseek_route_reference.safetensors"));
    let gates = ref_tensor(&shard, "gates").clone();
    let bias = ref_tensor(&shard, "e_score_correction_bias").clone();

    let scores = ops::sigmoid(&gates);
    let (top_indices, top_weights) =
        noaux_tc_topk(&scores, &bias, TOP_K, true, SCALING, N_GROUP, TOPK_GROUP)
            .expect("noaux_tc_topk");

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
