//! A prefill or a decode step that samples from the last position computes the
//! LM head for that position alone.
//!
//! `forward_with_cache_last` and `forward_with_hybrid_cache_last` return
//! `[B, 1, vocab]`, equal to the last row of the full forward's logits, from a
//! cache that decodes the next token exactly as the full forward's does, and
//! without ever forming the `[B, T, vocab]` logits: a 2048-token chunk of a
//! 152k vocabulary is 600 MB of them.

use pmetal_models::DynamicModel;
use pmetal_models::architectures::cohere::{CohereConfig, CohereForCausalLM};
use pmetal_models::architectures::granite::{GraniteConfig, GraniteForCausalLM};
use pmetal_models::architectures::llama::{LlamaConfig, LlamaForCausalLM};
use pmetal_models::architectures::mistral::{MistralConfig, MistralForCausalLM};
use pmetal_models::architectures::phi::{PhiConfig, PhiForCausalLM};
use pmetal_models::architectures::qwen2::{Qwen2Config, Qwen2ForCausalLM};
use pmetal_models::architectures::qwen3::{Qwen3Config, Qwen3ForCausalLM};
use pmetal_models::architectures::qwen3_moe::{Qwen3MoE, Qwen3MoEConfig};
use pmetal_runtime::compat::{Array, random};
use pmetal_runtime::inline_array::{get_active_memory, get_peak_memory, reset_peak_memory};
use serial_test::serial;

/// Large enough that the logits of the whole prompt dwarf everything else the
/// forward holds.
const VOCAB: i32 = 32768;
const TOKENS: usize = 256;
const MAX_SEQ: usize = 320;

fn prompt() -> Array {
    let ids: Vec<i32> = (0..TOKENS as i32).map(|i| (i * 37 + 11) % VOCAB).collect();
    Array::from_slice(&ids, &[1, TOKENS as i32])
}

fn next_token() -> Array {
    Array::from_slice(&[17i32], &[1, 1])
}

fn host(a: &Array) -> Vec<f32> {
    let mut a = a.clone();
    a.eval();
    let n = a.size();
    a.to_f32_vec(n).expect("to host")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f32, |worst, (x, y)| {
        let d = (x - y).abs();
        assert!(!d.is_nan(), "NaN in logits");
        worst.max(d)
    })
}

/// Bytes the forward holds above what was resident before it.
fn peak_above_baseline(run: impl FnOnce() -> Array) -> usize {
    pmetal_runtime::inline_array::clear_cache();
    let baseline = get_active_memory();
    reset_peak_memory();
    let out = run();
    out.eval();
    let peak = get_peak_memory();
    drop(out);
    peak.saturating_sub(baseline)
}

fn check(label: &str, mut model: DynamicModel) {
    let ids = prompt();
    let vocab = VOCAB as usize;

    // Full forward: the reference, and its cache.
    let mut full_cache = model.create_cache(MAX_SEQ);
    let mut mamba = model.create_mamba_cache();
    let full = model
        .forward_with_hybrid_cache(&ids, None, Some(&mut full_cache), mamba.as_mut())
        .expect("full forward");
    let full_last = host(&full)[(TOKENS - 1) * vocab..TOKENS * vocab].to_vec();

    // Last position only, through the hybrid entry point and the plain one.
    let mut last_cache = model.create_cache(MAX_SEQ);
    let mut last_mamba = model.create_mamba_cache();
    let last = model
        .forward_with_hybrid_cache_last(&ids, None, Some(&mut last_cache), last_mamba.as_mut())
        .expect("last forward");
    assert_eq!(last.shape(), &[1, 1, VOCAB], "{label}: shape");
    let diff = max_abs_diff(&host(&last), &full_last);
    assert!(diff < 1e-4, "{label}: last row differs by {diff:e}");

    let mut plain_cache = model.create_cache(MAX_SEQ);
    if last_mamba.is_none() {
        let plain = model
            .forward_with_cache_last(&ids, None, Some(&mut plain_cache))
            .expect("plain last forward");
        assert_eq!(plain.shape(), &[1, 1, VOCAB], "{label}: plain shape");
        let diff = max_abs_diff(&host(&plain), &full_last);
        assert!(diff < 1e-4, "{label}: plain last row differs by {diff:e}");
    }

    // The caches decode alike.
    let want = model
        .forward_with_hybrid_cache(&next_token(), None, Some(&mut full_cache), mamba.as_mut())
        .expect("decode after full");
    let got = model
        .forward_with_hybrid_cache_last(
            &next_token(),
            None,
            Some(&mut last_cache),
            last_mamba.as_mut(),
        )
        .expect("decode after last");
    let diff = max_abs_diff(&host(&got), &host(&want));
    assert!(
        diff < 1e-4,
        "{label}: decode after a last-only prefill differs by {diff:e}"
    );
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");

    // It never forms the logits of the prompt.
    let logits_bytes = TOKENS * vocab * 4;
    let mut cache = model.create_cache(MAX_SEQ);
    let mut mamba = model.create_mamba_cache();
    let full_peak = peak_above_baseline(|| {
        model
            .forward_with_hybrid_cache(&ids, None, Some(&mut cache), mamba.as_mut())
            .unwrap()
    });
    let mut cache = model.create_cache(MAX_SEQ);
    let mut mamba = model.create_mamba_cache();
    let last_peak = peak_above_baseline(|| {
        model
            .forward_with_hybrid_cache_last(&ids, None, Some(&mut cache), mamba.as_mut())
            .unwrap()
    });
    assert!(
        full_peak >= logits_bytes,
        "{label}: the full forward peaks at {full_peak} bytes, under its {logits_bytes} bytes of \
         logits, so this measurement cannot see them"
    );
    assert!(
        last_peak < logits_bytes / 2,
        "{label}: the last-position forward peaks at {last_peak} bytes, over half of the \
         {logits_bytes} bytes the prompt's logits take"
    );
}

#[test]
#[serial]
fn llama() {
    random::seed(1);
    let model = LlamaForCausalLM::new(LlamaConfig {
        vocab_size: VOCAB,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: Some(8),
        tie_word_embeddings: false,
        ..Default::default()
    })
    .unwrap();
    check("llama", DynamicModel::Llama(model));
}

#[test]
#[serial]
fn llama_tied() {
    random::seed(1);
    let model = LlamaForCausalLM::new(LlamaConfig {
        vocab_size: VOCAB,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: Some(8),
        tie_word_embeddings: true,
        ..Default::default()
    })
    .unwrap();
    check("llama (tied)", DynamicModel::Llama(model));
}

#[test]
#[serial]
fn qwen2() {
    random::seed(2);
    let model = Qwen2ForCausalLM::new(Qwen2Config {
        vocab_size: VOCAB,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        use_sliding_window: false,
        sliding_window: None,
        ..Default::default()
    })
    .unwrap();
    check("qwen2", DynamicModel::Qwen2(model));
}

#[test]
#[serial]
fn qwen3() {
    random::seed(3);
    let model = Qwen3ForCausalLM::new(Qwen3Config {
        vocab_size: VOCAB,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: 8,
        use_sliding_window: false,
        sliding_window: None,
        ..Default::default()
    })
    .unwrap();
    check("qwen3", DynamicModel::Qwen3(model));
}

#[test]
#[serial]
fn qwen3_moe() {
    random::seed(4);
    let model = Qwen3MoE::new(Qwen3MoEConfig {
        vocab_size: VOCAB,
        hidden_size: 32,
        intermediate_size: 64,
        moe_intermediate_size: Some(32),
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: 8,
        num_experts: 4,
        num_experts_per_tok: 2,
        ..Default::default()
    })
    .unwrap();
    check("qwen3_moe", DynamicModel::Qwen3MoE(model));
}

#[test]
#[serial]
fn mistral() {
    random::seed(5);
    let model = MistralForCausalLM::new(MistralConfig {
        vocab_size: VOCAB,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: Some(8),
        sliding_window: None,
        ..Default::default()
    })
    .unwrap();
    check("mistral", DynamicModel::Mistral(model));
}

#[test]
#[serial]
fn phi() {
    random::seed(6);
    let model = PhiForCausalLM::new(PhiConfig {
        vocab_size: VOCAB,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        ..Default::default()
    })
    .unwrap();
    check("phi", DynamicModel::Phi(model));
}

#[test]
#[serial]
fn cohere() {
    random::seed(7);
    let model = CohereForCausalLM::new(CohereConfig {
        vocab_size: VOCAB,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        ..Default::default()
    })
    .unwrap();
    check("cohere", DynamicModel::Cohere(model));
}

#[test]
#[serial]
fn granite() {
    random::seed(8);
    let model = GraniteForCausalLM::new(GraniteConfig {
        vocab_size: VOCAB,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: Some(8),
        ..Default::default()
    })
    .unwrap();
    check("granite", DynamicModel::Granite(model));
}

/// Nemotron-H, hybrid: Mamba-2, attention, MLP and routed experts. The final
/// norm and the head run on the last position, whichever mixers came before.
#[test]
#[serial]
fn nemotron_h() {
    random::seed(9);
    let config = serde_json::json!({
        "model_type": "nemotron_h",
        "vocab_size": VOCAB,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "mamba_num_heads": 4,
        "mamba_head_dim": 16,
        "ssm_state_size": 8,
        "n_groups": 2,
        "chunk_size": 32,
        "hybrid_override_pattern": "M*-EM",
        "moe_intermediate_size": 32,
        "moe_shared_expert_intermediate_size": 32,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
    });
    let model = DynamicModel::from_config(&config.to_string()).unwrap();
    check("nemotron_h", model);
}
