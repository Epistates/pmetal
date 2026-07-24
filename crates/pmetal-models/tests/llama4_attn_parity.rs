//! Numerical-parity test for the Rust Llama 4 attention block.
//!
//! Isolates `Llama4Attention` (no MoE, so no SwitchGLU expert-splitting needed)
//! and replays a reference dumped from `mlx_lm.models.llama4.Attention`
//! (`.strategy/parity/dump_llama4_attn_reference.py`). It pins the interacting
//! iRoPE fixes:
//!
//!   * traditional (interleaved) RoPE,
//!   * weightless QK-norm applied AFTER RoPE on RoPE layers only (eps 1e-6),
//!   * the NoPE temperature-tuning path on a no-rope layer.
//!
//! Layer 0 exercises RoPE + QK-norm; layer 3 (where `(3 + 1) % 4 == 0`)
//! exercises NoPE + temperature tuning. Each replays the SAME input and additive
//! causal mask the oracle saw. Before the fixes the layer-0 output diverges
//! (split-half RoPE, pre-RoPE/weighted/eps-1e-5 QK-norm) and the layer mapping
//! flips which layers are NoPE.

mod common;

use common::{fixture_path, load_shard, ref_tensor};

use pmetal_bridge::compat::Param;

use pmetal_mlx::test_utils::{ParityReport, Tolerance, print_report_table};

use pmetal_models::architectures::llama4::{Llama4Attention, Llama4TextConfig};

fn synthetic_config() -> Llama4TextConfig {
    Llama4TextConfig {
        vocab_size: 512,
        hidden_size: 128,
        intermediate_size: 512,
        intermediate_size_mlp: 512,
        num_hidden_layers: 8,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 8192,
        use_qk_norm: true,
        attn_temperature_tuning: true,
        ..Llama4TextConfig::default()
    }
}

#[test]
fn llama4_attention_synthetic_parity() {
    let shard = load_shard(&fixture_path("llama4_attn_reference.safetensors"));
    let mask = ref_tensor(&shard, "mask").clone();
    let config = synthetic_config();

    let mut reports = Vec::new();
    for layer_idx in [0usize, 3usize] {
        let tag = format!("l{layer_idx}");
        let mut attn = Llama4Attention::new(&config, layer_idx).expect("build attention");

        // Load the four projection weights; q_norm/k_norm stay at their default
        // ones (== mlx weightless `rms_norm(x, None, eps)`).
        attn.q_proj.weight = Param::new(ref_tensor(&shard, &format!("{tag}.q_proj")).clone());
        attn.k_proj.weight = Param::new(ref_tensor(&shard, &format!("{tag}.k_proj")).clone());
        attn.v_proj.weight = Param::new(ref_tensor(&shard, &format!("{tag}.v_proj")).clone());
        attn.o_proj.weight = Param::new(ref_tensor(&shard, &format!("{tag}.o_proj")).clone());

        // Sanity-check the iRoPE layer mapping that the dumper reported.
        let expects_rope = layer_idx == 0;
        assert_eq!(
            attn.uses_rope, expects_rope,
            "layer {layer_idx}: uses_rope should be {expects_rope}"
        );

        let x = ref_tensor(&shard, &format!("{tag}.x")).clone();
        let y = attn
            .forward(&x, Some(&mask), None)
            .expect("attention forward");

        reports.push(ParityReport::compute_with_per_position(
            &format!("{tag}_output"),
            &y,
            ref_tensor(&shard, &format!("{tag}.y")),
            Tolerance::new(2e-4, 1e-3),
        ));
    }

    println!("\n== Llama 4 attention synthetic parity report ==");
    print_report_table(&reports);

    let failures: Vec<_> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.name.clone())
        .collect();
    assert!(
        failures.is_empty(),
        "Llama 4 attention parity failed at: {failures:?}"
    );
}
