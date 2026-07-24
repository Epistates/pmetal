//! Numerical-parity test for the Rust DeepSeek MLA attention block.
//!
//! Replays a reference dumped from `mlx_lm.models.deepseek_v2.DeepseekV2Attention`
//! (`.strategy/parity/dump_deepseek_attn_reference.py`), whose explicit
//! `kv_b_proj` formulation matches pmetal's `DeepSeekAttention`. The reference
//! uses `rope_scaling={"factor": 1.0}`, which makes YARN degenerate to standard
//! RoPE and mscale -> 1.0, so this isolates the one always-wrong bug: DeepSeek
//! MLA uses traditional (interleaved) RoPE, not the split-half variant.
//!
//! YARN frequency correction + mscale (only active when a real `rope_scaling`
//! block is present) are a separate follow-up and not exercised here.

mod common;

use common::{fixture_path, load_shard, ref_tensor};

use pmetal_bridge::compat::Param;

use pmetal_mlx::test_utils::{ParityReport, Tolerance, print_report_table};

use pmetal_models::architectures::deepseek::{DeepSeekAttention, DeepSeekConfig};

fn synthetic_config() -> DeepSeekConfig {
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
        rope_scaling: None,
        attention_bias: false,
        ..DeepSeekConfig::default()
    }
}

#[test]
fn deepseek_attention_synthetic_parity() {
    let shard = load_shard(&fixture_path("deepseek_attn_reference.safetensors"));
    let config = synthetic_config();

    let mut attn = DeepSeekAttention::new(&config, 0).expect("build attention");

    // Load projection + norm weights from the oracle.
    attn.q_a_proj.as_mut().unwrap().weight = Param::new(ref_tensor(&shard, "q_a_proj").clone());
    attn.q_a_layernorm.as_mut().unwrap().weight =
        Param::new(ref_tensor(&shard, "q_a_layernorm").clone());
    attn.q_b_proj.as_mut().unwrap().weight = Param::new(ref_tensor(&shard, "q_b_proj").clone());
    attn.kv_a_proj_with_mqa.weight = Param::new(ref_tensor(&shard, "kv_a_proj_with_mqa").clone());
    attn.kv_a_layernorm.weight = Param::new(ref_tensor(&shard, "kv_a_layernorm").clone());
    attn.kv_b_proj.weight = Param::new(ref_tensor(&shard, "kv_b_proj").clone());
    attn.o_proj.weight = Param::new(ref_tensor(&shard, "o_proj").clone());

    let x = ref_tensor(&shard, "x").clone();
    let mask = ref_tensor(&shard, "mask").clone();
    let y = attn
        .forward(&x, Some(&mask), None)
        .expect("attention forward");

    let report = ParityReport::compute_with_per_position(
        "deepseek_attn_output",
        &y,
        ref_tensor(&shard, "y"),
        Tolerance::new(2e-4, 1e-3),
    );

    println!("\n== DeepSeek MLA attention synthetic parity report ==");
    print_report_table(std::slice::from_ref(&report));

    assert!(report.passed(), "DeepSeek attention parity failed");
}
