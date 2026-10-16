//! A sliding-window layer attends no further back than its window inside a
//! packed row.
//!
//! Packed training hands every layer a block-diagonal causal mask. A layer that
//! takes a caller's mask as the whole story attends across its whole sequence,
//! so a sliding-window model trained packed is a different model from the one
//! served. Each case packs two sequences longer than the window and compares
//! every one with the same sequence run alone (where the layer builds its own
//! window), and a second model, the same weights with a window wider than any
//! sequence, shows the window changes the logits at all, so the comparison is
//! not between two runs of the same thing.

use std::collections::HashMap;

use pmetal_models::architectures::gpt_oss::{GptOssConfig, GptOssForCausalLM};
use pmetal_models::architectures::mistral::{MistralConfig, MistralForCausalLM};
use pmetal_models::architectures::qwen2::{Qwen2Config, Qwen2ForCausalLM};
use pmetal_models::architectures::qwen3::{Qwen3Config, Qwen3ForCausalLM};
use pmetal_runtime::compat::{Array, Dtype, Exception, ModuleParametersExt, random};
use serial_test::serial;

const VOCAB: usize = 64;
const FIRST: usize = 7;
const SECOND: usize = 9;
const WINDOW: i32 = 4;

fn tokens() -> Vec<i32> {
    (0..(FIRST + SECOND) as i32)
        .map(|i| (i * 19 + 5) % VOCAB as i32)
        .collect()
}

/// Additive block-diagonal causal mask.
fn block_diagonal_mask() -> Array {
    let n = FIRST + SECOND;
    let mut mask = vec![f32::NEG_INFINITY; n * n];
    for (start, len) in [(0, FIRST), (FIRST, SECOND)] {
        for i in 0..len {
            for j in 0..=i {
                mask[(start + i) * n + start + j] = 0.0;
            }
        }
    }
    Array::from_f32_slice(&mask, &[1, 1, n as i32, n as i32])
}

fn positions() -> Array {
    let values: Vec<i32> = (0..FIRST as i32).chain(0..SECOND as i32).collect();
    Array::from_slice(&values, &[(FIRST + SECOND) as i32])
}

fn ids(tokens: &[i32]) -> Array {
    Array::from_slice(tokens, &[1, tokens.len() as i32])
}

fn host(logits: &Array, rows: usize) -> Vec<f32> {
    let mut logits = logits.as_dtype(Dtype::Float32.as_i32());
    logits.eval();
    logits.to_f32_vec(rows * VOCAB).expect("logits to host")
}

/// Fill every parameter of `model` with draws from [-0.5, 0.5).
fn randomize(model: &mut impl ModuleParametersExt) {
    for (_, p) in model.flatten_params_mut() {
        *p = random::uniform_range(-0.5, 0.5, p.shape(), Dtype::Float32);
    }
}

/// Give `to` the weights of `from`.
fn copy_weights(from: &impl ModuleParametersExt, to: &mut impl ModuleParametersExt) {
    let weights: HashMap<String, Array> = from
        .flatten_params()
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    for (key, p) in to.flatten_params_mut() {
        *p = weights
            .get(&key)
            .unwrap_or_else(|| panic!("no weight {key}"))
            .clone();
    }
}

type Forward<'a> =
    dyn FnMut(&Array, Option<&Array>, Option<&Array>) -> Result<Array, Exception> + 'a;

fn check(label: &str, narrow: &mut Forward<'_>, wide: &mut Forward<'_>) {
    let tokens = tokens();
    let mask = block_diagonal_mask();
    let positions = positions();

    let together = host(
        &narrow(&ids(&tokens), Some(&mask), Some(&positions)).expect("packed forward"),
        tokens.len(),
    );

    let mut worst = 0.0f32;
    let mut window_effect = 0.0f32;
    for (start, len) in [(0, FIRST), (FIRST, SECOND)] {
        let alone_ids = ids(&tokens[start..start + len]);
        let alone = host(&narrow(&alone_ids, None, None).expect("forward"), len);
        let unwindowed = host(&wide(&alone_ids, None, None).expect("forward"), len);
        let packed = &together[start * VOCAB..(start + len) * VOCAB];
        for ((p, a), u) in packed.iter().zip(&alone).zip(&unwindowed) {
            let d = (p - a).abs();
            assert!(!d.is_nan(), "{label}: NaN");
            worst = worst.max(d);
            window_effect = window_effect.max((a - u).abs());
        }
    }
    pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
    assert!(
        window_effect > 1e-2,
        "{label}: a window of {WINDOW} moves the logits by only {window_effect:e}, so this \
         comparison cannot tell a layer that honours it from one that does not"
    );
    assert!(
        worst < 1e-3,
        "{label}: a packed sequence attends past its window and is off by {worst:e}"
    );
}

#[test]
#[serial]
fn mistral_packed_sequences_keep_their_sliding_window() {
    let config = |window| MistralConfig {
        vocab_size: VOCAB as i32,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: Some(8),
        sliding_window: Some(window),
        ..Default::default()
    };
    random::seed(5);
    let mut narrow = MistralForCausalLM::new(config(WINDOW)).unwrap();
    randomize(&mut narrow);
    let mut wide = MistralForCausalLM::new(config(64)).unwrap();
    copy_weights(&narrow, &mut wide);
    check(
        "mistral",
        &mut |t, m, p| narrow.forward_with_positions(t, m, p),
        &mut |t, m, p| wide.forward_with_positions(t, m, p),
    );
}

#[test]
#[serial]
fn qwen2_packed_sequences_keep_their_sliding_window() {
    let config = |window| Qwen2Config {
        vocab_size: VOCAB as i32,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        use_sliding_window: true,
        sliding_window: Some(window),
        ..Default::default()
    };
    random::seed(6);
    let mut narrow = Qwen2ForCausalLM::new(config(WINDOW)).unwrap();
    randomize(&mut narrow);
    let mut wide = Qwen2ForCausalLM::new(config(64)).unwrap();
    copy_weights(&narrow, &mut wide);
    check(
        "qwen2",
        &mut |t, m, p| narrow.forward_with_positions(t, m, p),
        &mut |t, m, p| wide.forward_with_positions(t, m, p),
    );
}

#[test]
#[serial]
fn qwen3_packed_sequences_keep_their_sliding_window() {
    let config = |window| Qwen3Config {
        vocab_size: VOCAB as i32,
        hidden_size: 32,
        intermediate_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        head_dim: 8,
        use_sliding_window: true,
        sliding_window: Some(window),
        max_window_layers: Some(0),
        ..Default::default()
    };
    random::seed(7);
    let mut narrow = Qwen3ForCausalLM::new(config(WINDOW)).unwrap();
    randomize(&mut narrow);
    let mut wide = Qwen3ForCausalLM::new(config(64)).unwrap();
    copy_weights(&narrow, &mut wide);
    check(
        "qwen3",
        &mut |t, m, p| narrow.forward_with_positions(t, m, p),
        &mut |t, m, p| wide.forward_with_positions(t, m, p),
    );
}

#[test]
#[serial]
fn gpt_oss_packed_sequences_keep_their_sliding_window() {
    let config = |window| GptOssConfig {
        vocab_size: VOCAB as i32,
        hidden_size: 32,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        num_local_experts: 4,
        experts_per_token: 2,
        num_experts_per_tok: Some(2),
        sliding_window: window,
        ..Default::default()
    };
    random::seed(8);
    let mut narrow = GptOssForCausalLM::new(config(WINDOW)).unwrap();
    randomize(&mut narrow);
    let mut wide = GptOssForCausalLM::new(config(64)).unwrap();
    copy_weights(&narrow, &mut wide);
    check(
        "gpt_oss",
        &mut |t, m, p| narrow.forward_with_positions(t, m, p),
        &mut |t, m, p| wide.forward_with_positions(t, m, p),
    );
}
