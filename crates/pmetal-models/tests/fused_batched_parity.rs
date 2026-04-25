//! Parity test for the fused `[N_active, 1]` batched-decode path.
//!
//! Drives a tiny Llama model through two equivalent generation schedules:
//!
//! 1. **Serial path** — per-slot `forward_with_cache` over one `KVCache`.
//! 2. **Fused path** — `forward_batched_impl` over a shared
//!    [`FusedBatchKVCache`] with `N_active = 1`.
//!
//! For identical weights + identical token streams, the last-position
//! logits produced by both paths must match to within tight tolerance.
//! Single-slot parity is the strictest check we can get without
//! multi-batch reference dumps: any bug in RoPE position handling, KV
//! cache write/read, mask construction, or projection reshape would
//! surface as a divergence on the very first decode token.

use pmetal_bridge::compat::Array;
use pmetal_mlx::kv_cache::{FusedBatchKVCache, KVCache, KVCacheConfig};

use pmetal_models::architectures::llama::{LlamaConfig, LlamaForCausalLM};
use pmetal_models::architectures::mistral::{MistralConfig, MistralForCausalLM};
use pmetal_models::architectures::qwen2::{Qwen2Config, Qwen2ForCausalLM};
use pmetal_models::architectures::qwen3::{Qwen3Config, Qwen3ForCausalLM};
use pmetal_models::architectures::gemma::{GemmaConfig, GemmaForCausalLM};
use pmetal_models::architectures::gpt_oss::{AttentionType, GptOssConfig, GptOssForCausalLM};
use pmetal_models::architectures::phi::{PhiConfig, PhiForCausalLM};
use pmetal_models::architectures::cohere::{CohereConfig, CohereForCausalLM};
use pmetal_models::architectures::granite::{GraniteConfig, GraniteForCausalLM};
use pmetal_models::architectures::qwen3_moe::{Qwen3MoE, Qwen3MoEConfig};
use pmetal_models::traits::ModelConfig;

fn tiny_config() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: Some(8),
        max_position_embeddings: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        hidden_act: "silu".to_string(),
        tie_word_embeddings: true,
        ..Default::default()
    }
}

/// Compute last-position f32 logits from a `[B, S, V]` tensor.
fn last_logits(logits: &Array) -> Vec<f32> {
    let shape = logits.shape();
    assert!(
        shape.len() >= 2,
        "expected rank>=2 logits, got {shape:?}"
    );
    let seq_axis = shape.len() - 2;
    let seq_len = shape[seq_axis];
    let idx = Array::from_i32_slice(&[seq_len - 1]);
    let last = logits.take_axis(&idx, seq_axis as i32).squeeze_axes(&[seq_axis as i32]);
    // Flatten to a Vec<f32> for comparison. last has shape [B, V]; we
    // evaluate and copy out.
    let v = shape[shape.len() - 1] as usize;
    let batch = shape[0] as usize;
    let mut out = vec![0.0f32; batch * v];
    last.eval();
    let slice = last.as_slice::<f32>();
    out.copy_from_slice(slice);
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "shape mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, |acc, d| acc.max(d))
}

#[test]
fn serial_cache_per_layer_offsets_after_forward() {
    // Verify what rope_offset() returns in the middle of a multi-layer
    // forward. This is a diagnostic to validate my assumption about
    // per-layer offset semantics.
    let config = tiny_config();
    let _model = LlamaForCausalLM::new(config.clone()).unwrap();
    let mut cache = KVCache::new(KVCacheConfig::new(
        config.num_hidden_layers as usize,
        config.max_position_embeddings as usize,
        config.num_kv_heads() as usize,
        config.get_head_dim() as usize,
    ));
    // Fresh cache: layer 0 offset should be 0, rope_offset() should be 0.
    assert_eq!(cache.seq_len(), 0);
    assert_eq!(cache.rope_offset(), 0);
    // Simulate layer 0 writing one token.
    let h = config.num_kv_heads();
    let d = config.get_head_dim();
    let new_k = Array::zeros_f32(&[1, h, 1, d]);
    let new_v = Array::zeros_f32(&[1, h, 1, d]);
    cache.update_and_fetch(0, &new_k, &new_v).unwrap();
    // Now layer 0's offset is 1, so rope_offset returns 1.
    assert_eq!(cache.seq_len(), 1);
    assert_eq!(cache.rope_offset(), 1);
}

#[test]
fn q_proj_stable_across_calls() {
    let config = tiny_config();
    let mut model = LlamaForCausalLM::new(config.clone()).unwrap();
    let x = Array::from_i32_slice(&[7_i32]).reshape(&[1, 1]);
    let emb1 = pmetal_bridge::compat::Module::forward(&mut model.model.embed_tokens, &x).unwrap();
    let out1 = pmetal_bridge::compat::Module::forward(
        &mut model.model.layers[0].self_attn.q_proj,
        &emb1,
    )
    .unwrap();
    out1.eval();
    let emb2 = pmetal_bridge::compat::Module::forward(&mut model.model.embed_tokens, &x).unwrap();
    let out2 = pmetal_bridge::compat::Module::forward(
        &mut model.model.layers[0].self_attn.q_proj,
        &emb2,
    )
    .unwrap();
    out2.eval();
    let v1 = out1.as_slice::<f32>();
    let v2 = out2.as_slice::<f32>();
    let d: f32 = v1
        .iter()
        .zip(v2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(d < 1e-6, "q_proj output unstable: {d}");
}

#[test]
fn attn_layer0_matches_at_position_zero() {
    use pmetal_bridge::compat::Module;
    use pmetal_mlx::kv_cache::FusedBatchKVCache as FBC;
    use pmetal_models::common::{BatchedGqaAttnCfg, batched_gqa_attn};

    let config = tiny_config();
    let mut model = LlamaForCausalLM::new(config.clone()).unwrap();

    let input = Array::from_i32_slice(&[7_i32]).reshape(&[1, 1]);
    let emb = Module::forward(&mut model.model.embed_tokens, &input).unwrap();
    let normed = Module::forward(&mut model.model.layers[0].input_layernorm, &emb).unwrap();

    // Serial: LlamaAttention::forward_with_cache with fresh KVCache.
    let mut cache_s = KVCache::new(KVCacheConfig::new(
        config.num_hidden_layers as usize,
        config.max_position_embeddings as usize,
        config.num_kv_heads() as usize,
        config.get_head_dim() as usize,
    ));
    let out_s = model.model.layers[0]
        .self_attn
        .forward_with_cache(&normed, None, Some((&mut cache_s, 0)))
        .unwrap();
    out_s.eval();
    let sv = out_s.as_slice::<f32>().to_vec();

    // Fused: batched_gqa_attn directly on the same q/k/v/o projections.
    let head_dim = config.get_head_dim();
    let attn_cfg = BatchedGqaAttnCfg::new(
        config.num_attention_heads,
        config.num_kv_heads(),
        head_dim,
        model.model.layers[0].self_attn.effective_base,
        model.model.layers[0].self_attn.rope_scale,
    );
    let mut fused = FBC::new(
        KVCacheConfig::new(
            config.num_hidden_layers as usize,
            config.max_position_embeddings as usize,
            config.num_kv_heads() as usize,
            config.get_head_dim() as usize,
        ),
        1,
    )
    .unwrap();
    fused.admit(0).unwrap();
    let layer = &mut model.model.layers[0];
    let out_f = batched_gqa_attn(
        &normed,
        &mut layer.self_attn.q_proj,
        &mut layer.self_attn.k_proj,
        &mut layer.self_attn.v_proj,
        &mut layer.self_attn.o_proj,
        None,
        None,
        &attn_cfg,
        &mut fused,
        &[0],
        0,
    )
    .unwrap();
    out_f.eval();
    let fv = out_f.as_slice::<f32>().to_vec();

    let diff = max_abs_diff(&sv, &fv);
    assert!(
        diff < 5e-4,
        "attn layer 0 divergence: max_abs_diff={diff}\nserial[0..4]={:?}\nfused[0..4]={:?}",
        &sv[..4.min(sv.len())],
        &fv[..4.min(fv.len())]
    );
}

#[test]
fn fused_batched_single_token_matches_serial() {
    let config = tiny_config();
    let mut model = LlamaForCausalLM::new(config.clone()).unwrap();

    let input = Array::from_i32_slice(&[7_i32]).reshape(&[1, 1]);

    // Serial: fresh KVCache, forward once.
    let mut cache = KVCache::new(KVCacheConfig::new(
        config.num_hidden_layers as usize,
        config.max_position_embeddings as usize,
        config.num_kv_heads() as usize,
        config.get_head_dim() as usize,
    ));
    let serial = model
        .forward_with_cache(&input, None, Some(&mut cache))
        .unwrap();
    let serial_v = last_logits(&serial);

    // Fused: fresh FusedBatchKVCache, forward_batched once.
    let mut fused_cache = FusedBatchKVCache::new(
        KVCacheConfig::new(
            config.num_hidden_layers as usize,
            config.max_position_embeddings as usize,
            config.num_kv_heads() as usize,
            config.get_head_dim() as usize,
        ),
        1,
    )
    .unwrap();
    fused_cache.admit(0).unwrap();
    let fused = model
        .forward_batched_impl(&input, &[0], &mut fused_cache)
        .unwrap();
    let fused_v = last_logits(&fused);

    let diff = max_abs_diff(&serial_v, &fused_v);
    assert!(
        diff < 5e-4,
        "single-token fused vs serial divergence: max_abs_diff={diff}\nserial[0..4]={:?}\nfused[0..4]={:?}",
        &serial_v[..4],
        &fused_v[..4]
    );
}

#[test]
fn fused_batched_matches_serial_on_llama_n1() {
    // One model; both paths share its weights. Running two forwards
    // over it is safe because the model's parameters aren't mutated by
    // a forward (only caches change).
    let config = tiny_config();
    let mut model = LlamaForCausalLM::new(config.clone()).unwrap();

    // Prompt: a short sequence. We replay it one token at a time in
    // both paths so layer offsets advance identically — bulk-prefill
    // against fused-replay is NOT apples-to-apples because the cached
    // forward applies RoPE with layer-0's offset, and layer 0 advances
    // by seq_len in bulk mode vs by 1 per token in replay mode.
    let prompt = vec![3_i32, 5, 7, 11];
    let next_tok = 17_i32;

    // --- Serial path: prompt replay + decode via forward_with_cache. ---
    let mut serial_cache = KVCache::new(KVCacheConfig::new(
        config.num_hidden_layers as usize,
        config.max_position_embeddings as usize,
        config.num_kv_heads() as usize,
        config.get_head_dim() as usize,
    ));
    for &tok in &prompt {
        let ids = Array::from_i32_slice(&[tok]).reshape(&[1, 1]);
        let out = model
            .forward_with_cache(&ids, None, Some(&mut serial_cache))
            .unwrap();
        out.eval();
    }
    let decode_in = Array::from_i32_slice(&[next_tok]).reshape(&[1, 1]);
    let serial_decode = model
        .forward_with_cache(&decode_in, None, Some(&mut serial_cache))
        .unwrap();
    let serial_last = last_logits(&serial_decode);

    // --- Fused path: replay the prompt one token at a time through
    // forward_batched_impl (N_active=1), then decode the same next_tok.
    // Prompt replay exercises the fused K/V write path end-to-end. ---
    let fused_cfg = KVCacheConfig::new(
        config.num_hidden_layers as usize,
        config.max_position_embeddings as usize,
        config.num_kv_heads() as usize,
        config.get_head_dim() as usize,
    );
    let mut fused_cache = FusedBatchKVCache::new(fused_cfg, 1).unwrap();
    fused_cache.admit(0).unwrap();

    for &tok in &prompt {
        let ids = Array::from_i32_slice(&[tok]).reshape(&[1, 1]);
        let out = model
            .forward_batched_impl(&ids, &[0], &mut fused_cache)
            .unwrap();
        out.eval();
    }

    // Decode one more token — this should match the serial decode.
    let ids = Array::from_i32_slice(&[next_tok]).reshape(&[1, 1]);
    let fused_decode = model
        .forward_batched_impl(&ids, &[0], &mut fused_cache)
        .unwrap();
    let fused_last = last_logits(&fused_decode);

    // Tight tolerance: both paths use the same weights, same floating-
    // point ops, so differences should be well under 5e-4 even with
    // bf16/fp16 flush-to-zero quirks.
    let diff = max_abs_diff(&serial_last, &fused_last);
    assert!(
        diff < 5e-4,
        "fused vs serial last-logits divergence: max_abs_diff={diff}"
    );
    let _ = (serial_decode, fused_decode);
}

fn tiny_qwen2_config() -> Qwen2Config {
    Qwen2Config {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: 8,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        use_sliding_window: false,
        sliding_window: None,
        tie_word_embeddings: true,
        ..Default::default()
    }
}

fn tiny_qwen3_config() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: 8,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        use_sliding_window: false,
        sliding_window: None,
        max_window_layers: None,
        layer_types: None,
        tie_word_embeddings: true,
        hidden_act: "silu".to_string(),
        rope_scaling: None,
        model_type: "qwen3".to_string(),
    }
}

fn tiny_mistral_config() -> MistralConfig {
    MistralConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: Some(8),
        max_position_embeddings: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        sliding_window: None,
        tie_word_embeddings: true,
        ..Default::default()
    }
}

fn kv_cfg(num_layers: usize, max_seq: usize, heads: usize, d: usize) -> KVCacheConfig {
    KVCacheConfig::new(num_layers, max_seq, heads, d)
}

#[test]
fn fused_vs_serial_mistral_single_token() {
    let config = tiny_mistral_config();
    let mut model = MistralForCausalLM::new(config.clone()).unwrap();
    let input = Array::from_i32_slice(&[9_i32]).reshape(&[1, 1]);
    let max_seq = config.max_position_embeddings as usize;
    let hkv = ModelConfig::num_kv_heads(&config) as usize;
    let hd = config.get_head_dim() as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cache = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let serial = model
        .forward_with_cache(&input, None, Some(&mut cache))
        .unwrap();
    let sv = last_logits(&serial);

    let mut fused = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    fused.admit(0).unwrap();
    let f = model.forward_batched_impl(&input, &[0], &mut fused).unwrap();
    let fv = last_logits(&f);

    let diff = max_abs_diff(&sv, &fv);
    assert!(diff < 5e-4, "Mistral single-token divergence: {diff}");
}

#[test]
fn fused_vs_serial_qwen2_single_token() {
    let config = tiny_qwen2_config();
    let mut model = Qwen2ForCausalLM::new(config.clone()).unwrap();
    let input = Array::from_i32_slice(&[9_i32]).reshape(&[1, 1]);
    let max_seq = config.max_position_embeddings as usize;
    let hkv = ModelConfig::num_kv_heads(&config) as usize;
    let hd = config.get_head_dim() as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cache = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let serial = model
        .forward_with_cache(&input, None, Some(&mut cache))
        .unwrap();
    let sv = last_logits(&serial);

    let mut fused = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    fused.admit(0).unwrap();
    let f = model.forward_batched_impl(&input, &[0], &mut fused).unwrap();
    let fv = last_logits(&f);

    let diff = max_abs_diff(&sv, &fv);
    assert!(diff < 5e-4, "Qwen2 single-token divergence: {diff}");
}

/// The "real" win of fused decode: two slots share one forward pass.
/// Each slot runs serially on its own KVCache to produce a reference,
/// then we replay the same two token streams through a shared
/// FusedBatchKVCache in a single batched call and compare last-logits
/// for each row.
#[test]
fn fused_n2_matches_two_independent_serial_decodes() {
    let config = tiny_config();
    let mut serial = LlamaForCausalLM::new(config.clone()).unwrap();
    // Second model with *independent* weights would defeat the test —
    // reuse `serial` for both serial runs and for the fused run.
    let max_seq = config.max_position_embeddings as usize;
    let hkv = config.num_kv_heads() as usize;
    let hd = config.get_head_dim() as usize;
    let nl = config.num_hidden_layers as usize;

    // Serial slot 0: token 3.
    let mut cache_a = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let input_a = Array::from_i32_slice(&[3_i32]).reshape(&[1, 1]);
    let s_a = serial
        .forward_with_cache(&input_a, None, Some(&mut cache_a))
        .unwrap();
    let sv_a = last_logits(&s_a);

    // Serial slot 1: token 7.
    let mut cache_b = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let input_b = Array::from_i32_slice(&[7_i32]).reshape(&[1, 1]);
    let s_b = serial
        .forward_with_cache(&input_b, None, Some(&mut cache_b))
        .unwrap();
    let sv_b = last_logits(&s_b);

    // Fused: both slots in one forward. batch axis 0 corresponds to
    // active_indices ordering.
    let mut fused = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 4).unwrap();
    fused.admit(0).unwrap();
    fused.admit(2).unwrap();
    let input_fused = Array::from_i32_slice(&[3_i32, 7_i32]).reshape(&[2, 1]);
    let f = serial
        .forward_batched_impl(&input_fused, &[0, 2], &mut fused)
        .unwrap();
    let fv = last_logits(&f);

    // fv is [2*vocab]; split into per-row slices.
    let v = config.vocab_size as usize;
    let fv_a = &fv[..v];
    let fv_b = &fv[v..];
    assert_eq!(fv_a.len(), sv_a.len());
    assert_eq!(fv_b.len(), sv_b.len());

    let diff_a = max_abs_diff(&sv_a, fv_a);
    let diff_b = max_abs_diff(&sv_b, fv_b);
    assert!(diff_a < 5e-4, "N=2 slot 0 divergence: {diff_a}");
    assert!(diff_b < 5e-4, "N=2 slot 1 divergence: {diff_b}");
}

#[test]
fn fused_vs_serial_qwen3_single_token() {
    let config = tiny_qwen3_config();
    let mut model = Qwen3ForCausalLM::new(config.clone()).unwrap();
    let input = Array::from_i32_slice(&[9_i32]).reshape(&[1, 1]);
    let max_seq = config.max_position_embeddings as usize;
    let hkv = ModelConfig::num_kv_heads(&config) as usize;
    let hd = config.get_head_dim() as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cache = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let serial = model
        .forward_with_cache(&input, None, Some(&mut cache))
        .unwrap();
    let sv = last_logits(&serial);

    let mut fused = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    fused.admit(0).unwrap();
    let f = model.forward_batched_impl(&input, &[0], &mut fused).unwrap();
    let fv = last_logits(&f);

    let diff = max_abs_diff(&sv, &fv);
    assert!(diff < 5e-4, "Qwen3 single-token divergence: {diff}");
}

fn tiny_qwen3_moe_config() -> Qwen3MoEConfig {
    Qwen3MoEConfig {
        hidden_size: 32,
        intermediate_size: 64,
        moe_intermediate_size: Some(32),
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: 8,
        vocab_size: 64,
        max_position_embeddings: 64,
        num_experts: 4,
        num_experts_per_tok: 2,
        decoder_sparse_step: 1,
        tie_word_embeddings: true,
        rope_theta: 10000.0,
        ..Default::default()
    }
}

#[test]
fn fused_vs_serial_qwen3_moe_single_token() {
    let config = tiny_qwen3_moe_config();
    let mut model = Qwen3MoE::new(config.clone()).unwrap();
    // Eagerly populate stacked expert cache so both paths hit the same
    // stacked-routing kernel; otherwise the first forward builds the cache
    // while the second reuses it, which does not change numerics but keeps
    // the parity invariant obvious.
    model.init_stacked_moe().unwrap();

    let input = Array::from_i32_slice(&[9_i32]).reshape(&[1, 1]);
    let max_seq = config.max_position_embeddings as usize;
    let hkv = config.num_kv_heads() as usize;
    let hd = config.head_dim as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cache = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let serial = model.forward(&input, None, Some(&mut cache)).unwrap();
    let sv = last_logits(&serial);

    let mut fused = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    fused.admit(0).unwrap();
    let f = model.forward_batched_impl(&input, &[0], &mut fused).unwrap();
    let fv = last_logits(&f);

    let diff = max_abs_diff(&sv, &fv);
    assert!(diff < 5e-4, "Qwen3MoE single-token divergence: {diff}");
}

fn tiny_gpt_oss_config() -> GptOssConfig {
    GptOssConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 32,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        max_position_embeddings: 64,
        initial_context_length: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        rope_scaling: None,
        attention_bias: true,
        attention_dropout: 0.0,
        tie_word_embeddings: false,
        num_local_experts: 4,
        experts_per_token: 2,
        num_experts_per_tok: None,
        router_aux_loss_coef: 0.9,
        output_router_logits: false,
        sliding_window: 2,
        layer_types: vec![
            AttentionType::SlidingAttention,
            AttentionType::FullAttention,
            AttentionType::SlidingAttention,
            AttentionType::FullAttention,
        ],
        swiglu_limit: 7.0,
        hidden_act: "silu".to_string(),
        eos_token_id: 0,
        pad_token_id: 0,
        model_type: "gpt_oss".to_string(),
    }
}

/// Replay tokens through the serial and fused decode paths side by side,
/// confirming parity after `sliding_window`+1 steps — the regime where
/// the sliding lower-bound mask becomes load-bearing.
#[test]
fn fused_vs_serial_gpt_oss_respects_sliding_window() {
    let config = tiny_gpt_oss_config();
    assert!(config.sliding_window >= 1);
    let window = config.sliding_window as usize;

    let mut model = GptOssForCausalLM::new(config.clone()).unwrap();
    model.init_stacked_moe().unwrap();

    let max_seq = config.max_position_embeddings as usize;
    let hkv = config.num_key_value_heads as usize;
    let hd = config.head_dim as usize;
    let nl = config.num_hidden_layers as usize;

    // Drive enough steps (window + 2) that the sliding mask has to drop at
    // least one historical K position — otherwise sliding_window is a no-op
    // and the test wouldn't exercise the new mask overlay.
    let steps = window + 2;
    let token_stream: Vec<i32> = (0..steps as i32).map(|i| (i * 3 + 5) % 37).collect();

    let mut cache_serial = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let mut cache_fused = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    cache_fused.admit(0).unwrap();

    for (step, &tok) in token_stream.iter().enumerate() {
        let serial_input = Array::from_i32_slice(&[tok]).reshape(&[1, 1]);
        let fused_input = Array::from_i32_slice(&[tok]).reshape(&[1, 1]);

        let s = model
            .forward(&serial_input, None, Some(&mut cache_serial))
            .unwrap();
        let sv = last_logits(&s);

        let f = model
            .forward_batched_impl(&fused_input, &[0], &mut cache_fused)
            .unwrap();
        let fv = last_logits(&f);

        let diff = max_abs_diff(&sv, &fv);
        assert!(
            diff < 5e-4,
            "GPT-OSS step {step} (token {tok}) divergence: {diff}"
        );
    }
}

fn tiny_gemma_config() -> GemmaConfig {
    GemmaConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: Some(8),
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        hidden_act: "gelu".to_string(),
        attn_logit_softcapping: None,
        sliding_window: None,
        is_gemma2: false,
        is_gemma3: false,
        rope_scaling: None,
        ..Default::default()
    }
}

#[test]
fn fused_vs_serial_gemma_single_token() {
    let config = tiny_gemma_config();
    let mut model = GemmaForCausalLM::new(config.clone()).unwrap();
    let input = Array::from_i32_slice(&[9_i32]).reshape(&[1, 1]);
    let max_seq = config.max_position_embeddings as usize;
    let hkv = config.num_kv_heads() as usize;
    let hd = config.get_head_dim() as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cache = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let serial = model
        .forward_with_cache(&input, None, Some(&mut cache))
        .unwrap();
    let sv = last_logits(&serial);

    let mut fused = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    fused.admit(0).unwrap();
    let f = model.forward_batched_impl(&input, &[0], &mut fused).unwrap();
    let fv = last_logits(&f);

    let diff = max_abs_diff(&sv, &fv);
    assert!(diff < 5e-4, "Gemma single-token divergence: {diff}");
}

/// Multi-step Gemma1 decode: verifies the embedding-scale + tied-LM-head
/// path holds parity past offset=0 (where RoPE is identity). The
/// `GemmaRmsNorm`'s `+1` weight offset and the `embedding_scale = sqrt(H)`
/// multiplier both apply to non-zero values from step 1 onward.
#[test]
fn fused_vs_serial_gemma_multi_step() {
    let config = tiny_gemma_config();
    let mut model = GemmaForCausalLM::new(config.clone()).unwrap();
    let max_seq = config.max_position_embeddings as usize;
    let hkv = config.num_kv_heads() as usize;
    let hd = config.get_head_dim() as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cs = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let mut cf = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    cf.admit(0).unwrap();

    for (step, tok) in [5_i32, 8, 11, 14].iter().enumerate() {
        let inp = Array::from_i32_slice(&[*tok]).reshape(&[1, 1]);
        let s = model
            .forward_with_cache(&inp, None, Some(&mut cs))
            .unwrap();
        let f = model.forward_batched_impl(&inp, &[0], &mut cf).unwrap();
        let d = max_abs_diff(&last_logits(&s), &last_logits(&f));
        assert!(
            d < 5e-4,
            "Gemma step {step} (token {tok}) divergence: {d}"
        );
    }
}

fn tiny_phi_config() -> PhiConfig {
    PhiConfig {
        model_type: "phi3".to_string(),
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        max_position_embeddings: 64,
        rope_theta: 10000.0,
        partial_rotary_factor: 0.5,
        rms_norm_eps: 1e-5,
        qkv_bias: false,
        sliding_window: None,
        original_max_position_embeddings: None,
        rope_scaling: None,
        tie_word_embeddings: false,
        ..Default::default()
    }
}

fn tiny_phi4_config() -> PhiConfig {
    PhiConfig {
        qkv_bias: true,
        ..tiny_phi_config()
    }
}

#[test]
fn fused_vs_serial_phi_single_token() {
    let config = tiny_phi_config();
    let mut model = PhiForCausalLM::new(config.clone()).unwrap();
    let input = Array::from_i32_slice(&[9_i32]).reshape(&[1, 1]);
    let max_seq = config.max_position_embeddings as usize;
    let hkv = config.num_key_value_heads as usize;
    let hd = config.head_dim() as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cache = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let serial = model
        .forward_with_cache(&input, None, Some(&mut cache))
        .unwrap();
    let sv = last_logits(&serial);

    let mut fused = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    fused.admit(0).unwrap();
    let f = model.forward_batched_impl(&input, &[0], &mut fused).unwrap();
    let fv = last_logits(&f);

    let diff = max_abs_diff(&sv, &fv);
    assert!(diff < 5e-4, "Phi single-token divergence: {diff}");
}

/// Multi-step Phi decode — exercises partial-RoPE at offset > 0, which
/// is exactly the regime where the prior axis-mismatched serial path
/// silently misrotated heads instead of seq positions.
#[test]
fn fused_vs_serial_phi_multi_step() {
    let config = tiny_phi_config();
    let mut model = PhiForCausalLM::new(config.clone()).unwrap();
    let max_seq = config.max_position_embeddings as usize;
    let hkv = config.num_key_value_heads as usize;
    let hd = config.head_dim() as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cs = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let mut cf = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    cf.admit(0).unwrap();

    for (step, tok) in [4_i32, 7, 11, 13, 19].iter().enumerate() {
        let inp = Array::from_i32_slice(&[*tok]).reshape(&[1, 1]);
        let s = model
            .forward_with_cache(&inp, None, Some(&mut cs))
            .unwrap();
        let f = model.forward_batched_impl(&inp, &[0], &mut cf).unwrap();
        let d = max_abs_diff(&last_logits(&s), &last_logits(&f));
        assert!(d < 5e-4, "Phi step {step} (token {tok}) divergence: {d}");
    }
}

/// Phi-4 has `qkv_bias: true`. The bias lives on the `nn::Linear`
/// projections themselves; both fused and serial paths invoke
/// `Module::forward(linear, ...)`, so this test mainly guards against a
/// regression where the fused path bypasses the linear's `forward` and
/// drops the bias on the floor.
fn tiny_gemma2_config() -> GemmaConfig {
    GemmaConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: Some(8),
        max_position_embeddings: 64,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        hidden_act: "gelu".to_string(),
        attn_logit_softcapping: Some(50.0),
        sliding_window: Some(4),
        is_gemma2: true,
        is_gemma3: false,
        rope_scaling: None,
        ..Default::default()
    }
}

fn tiny_gemma3_config() -> GemmaConfig {
    GemmaConfig {
        // Gemma3 — every (idx+1) % 6 != 0 is local (sliding); others global.
        // With 6 layers we hit at least one global (idx 5) and several locals.
        num_hidden_layers: 6,
        is_gemma2: false,
        is_gemma3: true,
        ..tiny_gemma2_config()
    }
}

#[test]
fn fused_vs_serial_gemma2_single_token() {
    let config = tiny_gemma2_config();
    let mut model = GemmaForCausalLM::new(config.clone()).unwrap();
    let input = Array::from_i32_slice(&[9_i32]).reshape(&[1, 1]);
    let max_seq = config.max_position_embeddings as usize;
    let hkv = config.num_kv_heads() as usize;
    let hd = config.get_head_dim() as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cache = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let serial = model
        .forward_with_cache(&input, None, Some(&mut cache))
        .unwrap();
    let sv = last_logits(&serial);

    let mut fused = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    fused.admit(0).unwrap();
    let f = model.forward_batched_impl(&input, &[0], &mut fused).unwrap();
    let fv = last_logits(&f);

    let diff = max_abs_diff(&sv, &fv);
    assert!(diff < 5e-4, "Gemma2 single-token divergence: {diff}");
}

/// Multi-step Gemma2 — exercises the 4-norm peri-norm helper, attn logit
/// softcap, AND per-layer sliding window past `sliding_window` tokens.
#[test]
fn fused_vs_serial_gemma2_multi_step() {
    let config = tiny_gemma2_config();
    let window = config.sliding_window.unwrap() as usize;
    let mut model = GemmaForCausalLM::new(config.clone()).unwrap();
    let max_seq = config.max_position_embeddings as usize;
    let hkv = config.num_kv_heads() as usize;
    let hd = config.get_head_dim() as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cs = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let mut cf = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    cf.admit(0).unwrap();

    // Drive enough tokens that the sliding-window boundary on even layers
    // becomes load-bearing (window + 2 like the GPT-OSS test).
    let steps = window + 2;
    let token_stream: Vec<i32> = (0..steps as i32).map(|i| (i * 5 + 3) % 41).collect();

    for (step, &tok) in token_stream.iter().enumerate() {
        let inp = Array::from_i32_slice(&[tok]).reshape(&[1, 1]);
        let s = model
            .forward_with_cache(&inp, None, Some(&mut cs))
            .unwrap();
        let f = model.forward_batched_impl(&inp, &[0], &mut cf).unwrap();
        let d = max_abs_diff(&last_logits(&s), &last_logits(&f));
        assert!(d < 5e-4, "Gemma2 step {step} (token {tok}) divergence: {d}");
    }
}

#[test]
fn fused_vs_serial_gemma3_multi_step() {
    let config = tiny_gemma3_config();
    let window = config.sliding_window.unwrap() as usize;
    let mut model = GemmaForCausalLM::new(config.clone()).unwrap();
    let max_seq = config.max_position_embeddings as usize;
    let hkv = config.num_kv_heads() as usize;
    let hd = config.get_head_dim() as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cs = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let mut cf = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    cf.admit(0).unwrap();

    let steps = window + 2;
    let token_stream: Vec<i32> = (0..steps as i32).map(|i| (i * 7 + 2) % 47).collect();

    for (step, &tok) in token_stream.iter().enumerate() {
        let inp = Array::from_i32_slice(&[tok]).reshape(&[1, 1]);
        let s = model
            .forward_with_cache(&inp, None, Some(&mut cs))
            .unwrap();
        let f = model.forward_batched_impl(&inp, &[0], &mut cf).unwrap();
        let d = max_abs_diff(&last_logits(&s), &last_logits(&f));
        assert!(d < 5e-4, "Gemma3 step {step} (token {tok}) divergence: {d}");
    }
}

fn tiny_cohere_config() -> CohereConfig {
    CohereConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        max_position_embeddings: 64,
        rope_theta: 10000.0,
        layer_norm_eps: 1e-5,
        tie_word_embeddings: false,
        use_sliding_window: false,
        sliding_window: 4096,
        global_attention_layers: None,
    }
}

#[test]
fn fused_vs_serial_cohere_single_token() {
    let config = tiny_cohere_config();
    let mut model = CohereForCausalLM::new(config.clone()).unwrap();
    let input = Array::from_i32_slice(&[9_i32]).reshape(&[1, 1]);
    let max_seq = config.max_position_embeddings as usize;
    let hkv = config.num_key_value_heads as usize;
    let hd = config.head_dim as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cache = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let serial = model
        .forward_with_cache(&input, None, Some(&mut cache))
        .unwrap();
    let sv = last_logits(&serial);

    let mut fused = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    fused.admit(0).unwrap();
    let f = model.forward_batched_impl(&input, &[0], &mut fused).unwrap();
    let fv = last_logits(&f);

    let diff = max_abs_diff(&sv, &fv);
    assert!(diff < 5e-4, "Cohere single-token divergence: {diff}");
}

/// Multi-step Cohere — exercises the parallel-block helper at offset > 0
/// and confirms the post-transpose RoPE fix in serial path matches the
/// fused path's per-batch-row position handling.
#[test]
fn fused_vs_serial_cohere_multi_step() {
    let config = tiny_cohere_config();
    let mut model = CohereForCausalLM::new(config.clone()).unwrap();
    let max_seq = config.max_position_embeddings as usize;
    let hkv = config.num_key_value_heads as usize;
    let hd = config.head_dim as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cs = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let mut cf = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    cf.admit(0).unwrap();

    for (step, tok) in [3_i32, 9, 12, 18, 23].iter().enumerate() {
        let inp = Array::from_i32_slice(&[*tok]).reshape(&[1, 1]);
        let s = model
            .forward_with_cache(&inp, None, Some(&mut cs))
            .unwrap();
        let f = model.forward_batched_impl(&inp, &[0], &mut cf).unwrap();
        let d = max_abs_diff(&last_logits(&s), &last_logits(&f));
        assert!(d < 5e-4, "Cohere step {step} (token {tok}) divergence: {d}");
    }
}

#[test]
fn fused_vs_serial_phi4_qkv_bias_multi_step() {
    let config = tiny_phi4_config();
    let mut model = PhiForCausalLM::new(config.clone()).unwrap();
    let max_seq = config.max_position_embeddings as usize;
    let hkv = config.num_key_value_heads as usize;
    let hd = config.head_dim() as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cs = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let mut cf = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    cf.admit(0).unwrap();

    for (step, tok) in [2_i32, 6, 10, 15].iter().enumerate() {
        let inp = Array::from_i32_slice(&[*tok]).reshape(&[1, 1]);
        let s = model
            .forward_with_cache(&inp, None, Some(&mut cs))
            .unwrap();
        let f = model.forward_batched_impl(&inp, &[0], &mut cf).unwrap();
        let d = max_abs_diff(&last_logits(&s), &last_logits(&f));
        assert!(d < 5e-4, "Phi-4 step {step} (token {tok}) divergence: {d}");
    }
}

fn tiny_granite_config() -> GraniteConfig {
    GraniteConfig {
        vocab_size: 64,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        max_position_embeddings: 64,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: true,
        is_hybrid: false,
        layer_types: None,
        mamba_state_dim: 128,
        mamba_conv_dim: 4,
        is_moe: false,
        num_experts: 8,
        num_experts_per_tok: 2,
        use_shared_expert: true,
    }
}

#[test]
fn fused_vs_serial_granite_single_token() {
    let config = tiny_granite_config();
    let mut model = GraniteForCausalLM::new(config.clone()).unwrap();
    let input = Array::from_i32_slice(&[9_i32]).reshape(&[1, 1]);
    let max_seq = config.max_position_embeddings as usize;
    let hkv = config.num_key_value_heads as usize;
    let hd = config.head_dim as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cache = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let serial = model
        .forward_with_cache(&input, None, Some(&mut cache))
        .unwrap();
    let sv = last_logits(&serial);

    let mut fused = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    fused.admit(0).unwrap();
    let f = model.forward_batched_impl(&input, &[0], &mut fused).unwrap();
    let fv = last_logits(&f);

    let diff = max_abs_diff(&sv, &fv);
    assert!(diff < 5e-4, "Granite single-token divergence: {diff}");
}

/// Multi-step Granite — exercises RoPE at offset > 0 across the new
/// serial cached path (previously a `// RoPE would be applied here`
/// placeholder + hand-rolled softmax with no cache integration). Tied
/// embeddings + GQA path covered.
#[test]
fn fused_vs_serial_granite_multi_step() {
    let config = tiny_granite_config();
    let mut model = GraniteForCausalLM::new(config.clone()).unwrap();
    let max_seq = config.max_position_embeddings as usize;
    let hkv = config.num_key_value_heads as usize;
    let hd = config.head_dim as usize;
    let nl = config.num_hidden_layers as usize;

    let mut cs = KVCache::new(kv_cfg(nl, max_seq, hkv, hd));
    let mut cf = FusedBatchKVCache::new(kv_cfg(nl, max_seq, hkv, hd), 1).unwrap();
    cf.admit(0).unwrap();

    for (step, tok) in [3_i32, 5, 11, 17, 23].iter().enumerate() {
        let inp = Array::from_i32_slice(&[*tok]).reshape(&[1, 1]);
        let s = model
            .forward_with_cache(&inp, None, Some(&mut cs))
            .unwrap();
        let f = model.forward_batched_impl(&inp, &[0], &mut cf).unwrap();
        let d = max_abs_diff(&last_logits(&s), &last_logits(&f));
        assert!(d < 5e-4, "Granite step {step} (token {tok}) divergence: {d}");
    }
}
