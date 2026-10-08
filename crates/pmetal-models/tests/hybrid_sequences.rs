//! A hybrid model decodes every sequence against that sequence's own state.
//!
//! Qwen 3.5's decode step keeps state of its own beside the caller's KV and
//! recurrent caches. It kept one for the whole model, bootstrapped by the
//! first sequence ever decoded, so every later sequence (the second request a
//! server answered, the second slot of a batch) decoded against the first
//! one's attention keys and recurrent state. One sequence per process, which
//! is what a single generation and every parity test run, never shows it.
//!
//! Tiny random-init model through `DynamicModel`, the path serving takes.

use pmetal_bridge::compat::Array;
use pmetal_mlx::kv_cache::{KVCache, MambaCache};
use pmetal_models::dispatcher::DynamicModel;

const QWEN3_NEXT: &str = r#"{
    "model_type": "qwen3_next",
    "vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
    "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 16, "linear_num_value_heads": 2, "linear_num_key_heads": 1,
    "linear_key_head_dim": 32, "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
    "num_experts": 0, "num_experts_per_tok": 0, "decoder_sparse_step": 1,
    "moe_intermediate_size": 16, "shared_expert_intermediate_size": 32,
    "mlp_only_layers": [], "norm_topk_prob": false, "tie_word_embeddings": true,
    "max_position_embeddings": 256, "rms_norm_eps": 1e-6, "rope_theta": 10000.0
}"#;

/// Granite 4.0-H's layout: Mamba-2 and NoPE attention, routed experts plus a
/// shared MLP. Its recurrent state lives entirely in the caller's
/// `MambaCache`, so this holds the serving contract from the other side: no
/// state of the model's own for a second sequence to inherit.
const GRANITE_HYBRID: &str = r#"{
    "model_type": "granitemoehybrid",
    "vocab_size": 64, "hidden_size": 32, "intermediate_size": 16,
    "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
    "layer_types": ["mamba", "attention", "mamba", "mamba"],
    "position_embedding_type": "nope",
    "num_local_experts": 4, "num_experts_per_tok": 2, "shared_intermediate_size": 24,
    "mamba_n_heads": 8, "mamba_d_state": 8, "mamba_chunk_size": 4,
    "embedding_multiplier": 12.0, "residual_multiplier": 0.22,
    "logits_scaling": 6.0, "attention_multiplier": 0.125,
    "tie_word_embeddings": true,
    "max_position_embeddings": 256, "rms_norm_eps": 1e-5
}"#;

const HYBRIDS: &[(&str, &str)] = &[
    ("qwen3_next", QWEN3_NEXT),
    ("granitemoehybrid", GRANITE_HYBRID),
];

const STEPS: usize = 4;

struct Sequence {
    tokens: Vec<i32>,
    kv: KVCache,
    recurrent: MambaCache,
    /// Last-position logits of the latest forward.
    last: Vec<f32>,
}

impl Sequence {
    /// Prefill `prompt` into fresh caches.
    fn start(model: &mut DynamicModel, prompt: &[i32]) -> Self {
        let mut kv = model.create_cache(prompt.len() + STEPS + 1);
        let mut recurrent = model.create_mamba_cache().expect("hybrid model");
        let logits = model
            .forward_with_hybrid_cache(&ids(prompt), None, Some(&mut kv), Some(&mut recurrent))
            .expect("prefill");
        Self {
            tokens: prompt.to_vec(),
            kv,
            recurrent,
            last: last_row(&logits),
        }
    }

    /// Decode one greedy token through the cached single-token path.
    fn step(&mut self, model: &mut DynamicModel) {
        let next = argmax(&self.last) as i32;
        self.tokens.push(next);
        let logits = model
            .forward_with_hybrid_cache(
                &ids(&[next]),
                None,
                Some(&mut self.kv),
                Some(&mut self.recurrent),
            )
            .expect("decode step");
        self.last = last_row(&logits);
    }

    /// The cached logits have to be what recomputing the whole sequence with
    /// no cache gives.
    fn assert_matches_recompute(&self, model: &mut DynamicModel, what: &str) {
        let fresh = last_row(&model.forward(&ids(&self.tokens), None).expect("recompute"));
        let diff = self
            .last
            .iter()
            .zip(&fresh)
            .fold(0.0f32, |worst, (a, b)| worst.max((a - b).abs()));
        assert!(
            diff < 1e-3,
            "{what}: cached decode of {:?} is {diff:e} from recomputing it",
            self.tokens
        );
    }
}

/// A sequence decoded after another one on the same model.
#[test]
fn second_sequence_decodes_against_its_own_state() {
    for (name, config) in HYBRIDS {
        let mut model = DynamicModel::from_config(config).expect("hybrid builds");

        let mut first = Sequence::start(&mut model, &[3, 11, 7, 2, 9]);
        for _ in 0..STEPS {
            first.step(&mut model);
        }
        first.assert_matches_recompute(&mut model, &format!("{name}: first sequence"));
        drop(first);

        let mut second = Sequence::start(&mut model, &[40, 5, 17]);
        for step in 0..STEPS {
            second.step(&mut model);
            second.assert_matches_recompute(
                &mut model,
                &format!("{name}: second sequence, step {step}"),
            );
        }
        pmetal_bridge::check_last_error().expect("no bridge op failed");
    }
}

/// Two sequences decoded in alternation on one model, as continuous batching
/// does with its slots: each step has to see only its own sequence.
#[test]
fn interleaved_sequences_keep_their_own_state() {
    for (name, config) in HYBRIDS {
        let mut model = DynamicModel::from_config(config).expect("hybrid builds");

        let mut a = Sequence::start(&mut model, &[3, 11, 7, 2, 9]);
        let mut b = Sequence::start(&mut model, &[40, 5, 17]);
        for step in 0..STEPS {
            a.step(&mut model);
            b.step(&mut model);
            a.assert_matches_recompute(&mut model, &format!("{name}: sequence a, step {step}"));
            b.assert_matches_recompute(&mut model, &format!("{name}: sequence b, step {step}"));
        }
        pmetal_bridge::check_last_error().expect("no bridge op failed");
    }
}

fn ids(tokens: &[i32]) -> Array {
    Array::from_i32_slice_shaped(tokens, &[1, tokens.len() as i32])
}

/// Logits at the last position, as f32.
fn last_row(logits: &Array) -> Vec<f32> {
    let shape = logits.shape();
    let (seq, vocab) = (shape[1], shape[2]);
    let mut row = logits.slice(&[0, seq - 1, 0], &[1, seq, vocab]);
    row.eval();
    row.to_f32_vec(vocab as usize).expect("to_f32_vec")
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |best, (index, value)| {
            if *value > best.1 {
                (index, *value)
            } else {
                best
            }
        })
        .0
}
