//! Numerical-parity test for the Rust DeepSeek MLA attention block.
//!
//! Replays a reference dumped from the authoritative HuggingFace `transformers`
//! `DeepseekV3Attention` (`.strategy/parity/dump_deepseek_attn_reference.py`),
//! whose explicit `kv_b_proj` formulation matches pmetal's `DeepSeekAttention`.
//! The v3 module is the oracle rather than v2 because v2 drops YARN's `mscale²`
//! softmax term entirely — see the dumper for the full reasoning.
//!
//! With `rope_type: "default"` there is no YARN frequency correction and no
//! mscale, so this fixture isolates the one thing that is wrong regardless of
//! config: DeepSeek rotates the RoPE head-dims *interleaved*, not split-half.
//! YARN is covered by `deepseek_yarn_parity.rs`.

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
        .forward(&x, Some(&mask), None, None)
        .expect("attention forward");

    let report = ParityReport::compute_with_per_position(
        "deepseek_attn_output",
        &y,
        ref_tensor(&shard, "y"),
        Tolerance::new(1e-5, 1e-5),
    );

    println!("\n== DeepSeek MLA attention synthetic parity report ==");
    print_report_table(std::slice::from_ref(&report));

    assert!(report.passed(), "DeepSeek attention parity failed");
}
