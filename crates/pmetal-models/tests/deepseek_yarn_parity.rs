//! Numerical-parity test for DeepSeek MLA attention WITH YARN rope_scaling.
//!
//! Replays references dumped from `mlx_lm.models.deepseek_v2.DeepseekV2Attention`
//! (`.strategy/parity/dump_deepseek_yarn_reference.py`) for two cases:
//!
//!   * "scale" — mscale_all_dim=1.0 (V3-realistic): mscale² folded into the
//!     softmax scale + YARN frequency blending; embedding mscale == 1.0.
//!   * "emb"   — mscale=2.0, mscale_all_dim=0.0: q/k pre-scaled by the embedding
//!     mscale before rotation; softmax scale unchanged.
//!
//! Together they pin the YARN inverse-frequency computation, the embedding
//! mscale, and the mscale² scale adjustment.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use pmetal_bridge::compat::{Array, Param};

use pmetal_mlx::test_utils::{ParityReport, Tolerance, print_report_table};

use pmetal_models::architectures::deepseek::{DeepSeekAttention, DeepSeekConfig};

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

fn config_with_scaling(rope_scaling: serde_json::Value) -> DeepSeekConfig {
    DeepSeekConfig {
        hidden_size: 128,
        num_attention_heads: 4,
        q_lora_rank: Some(48),
        kv_lora_rank: 32,
        qk_rope_head_dim: 16,
        qk_nope_head_dim: 16,
        v_head_dim: 24,
        max_position_embeddings: 2048,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        rope_scaling: Some(rope_scaling),
        attention_bias: false,
        ..DeepSeekConfig::default()
    }
}

#[test]
fn deepseek_yarn_synthetic_parity() {
    let shard = load_shard(&fixture_path("deepseek_yarn_reference.safetensors"));
    let mask = ref_tensor(&shard, "mask").clone();

    let cases = [
        (
            "scale",
            serde_json::json!({
                "type": "yarn", "factor": 4.0, "beta_fast": 32, "beta_slow": 1,
                "mscale": 1.0, "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 32
            }),
        ),
        (
            "emb",
            serde_json::json!({
                "type": "yarn", "factor": 4.0, "beta_fast": 32, "beta_slow": 1,
                "mscale": 2.0, "mscale_all_dim": 0.0,
                "original_max_position_embeddings": 32
            }),
        ),
    ];

    let mut reports = Vec::new();
    for (tag, rope_scaling) in cases {
        let config = config_with_scaling(rope_scaling);
        let mut attn = DeepSeekAttention::new(&config, 0).expect("build attention");

        attn.q_a_proj.as_mut().unwrap().weight =
            Param::new(ref_tensor(&shard, &format!("{tag}.q_a_proj")).clone());
        attn.q_a_layernorm.as_mut().unwrap().weight =
            Param::new(ref_tensor(&shard, &format!("{tag}.q_a_layernorm")).clone());
        attn.q_b_proj.as_mut().unwrap().weight =
            Param::new(ref_tensor(&shard, &format!("{tag}.q_b_proj")).clone());
        attn.kv_a_proj_with_mqa.weight =
            Param::new(ref_tensor(&shard, &format!("{tag}.kv_a_proj_with_mqa")).clone());
        attn.kv_a_layernorm.weight =
            Param::new(ref_tensor(&shard, &format!("{tag}.kv_a_layernorm")).clone());
        attn.kv_b_proj.weight = Param::new(ref_tensor(&shard, &format!("{tag}.kv_b_proj")).clone());
        attn.o_proj.weight = Param::new(ref_tensor(&shard, &format!("{tag}.o_proj")).clone());

        let x = ref_tensor(&shard, &format!("{tag}.x")).clone();
        let y = attn
            .forward(&x, Some(&mask), None)
            .expect("attention forward");

        reports.push(ParityReport::compute_with_per_position(
            &format!("yarn_{tag}_output"),
            &y,
            ref_tensor(&shard, &format!("{tag}.y")),
            Tolerance::new(2e-4, 1e-3),
        ));
    }

    println!("\n== DeepSeek YARN attention synthetic parity report ==");
    print_report_table(&reports);

    let failures: Vec<_> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.name.clone())
        .collect();
    assert!(
        failures.is_empty(),
        "DeepSeek YARN parity failed at: {failures:?}"
    );
}
