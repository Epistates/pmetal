//! `LoadedModel` scores and generates what the model itself says.
//!
//! A tiny random model is the reference: each request's log-likelihood is
//! computed here one sequence at a time from the model's own forward pass, and
//! the batched, padded and shared-context paths of `LoadedModel` must agree.

use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::{Array, Dtype, ops, random};
use pmetal_trainer::eval::{EvalModel, LlRequest, LoadedModel};

const LLAMA: &str = r#"{
    "model_type": "llama",
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "tie_word_embeddings": false
}"#;

fn model() -> DynamicModel {
    random::seed(5);
    DynamicModel::from_config(LLAMA).expect("builds")
}

/// The log-probabilities the model gives every next token after each prefix of
/// `tokens`: `[len, vocab]`, as f32.
fn next_token_log_probs(model: &mut DynamicModel, tokens: &[u32]) -> Array {
    let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    let input = Array::from_slice(&ids, &[1, ids.len() as i32]);
    let logits = model.forward(&input, None).expect("forward");
    let logits = logits.squeeze_axes(&[0]).as_dtype(Dtype::Float32.as_i32());
    pmetal_runtime::check_unobserved_error().expect("no runtime error");
    logits.subtract(&ops::logsumexp_axis_keepdims(&logits, -1, true))
}

/// The log-likelihood of `continuation` after `context` and whether greedy
/// decoding would produce it, by the plainest means.
fn reference(model: &mut DynamicModel, context: &[u32], continuation: &[u32]) -> (f64, bool) {
    let mut full = context.to_vec();
    full.extend(continuation);
    let log_probs = next_token_log_probs(model, &full[..full.len() - 1]);
    log_probs.eval();
    let table = log_probs.as_slice::<f32>();
    let vocab = 64;
    let (mut total, mut greedy) = (0.0f64, true);
    for (i, &token) in continuation.iter().enumerate() {
        let row = &table[(context.len() - 1 + i) * vocab..][..vocab];
        total += f64::from(row[token as usize]);
        let best = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(t, _)| t)
            .unwrap();
        greedy &= best == token as usize;
    }
    (total, greedy)
}

fn tokens(seed: u32, len: usize) -> Vec<u32> {
    (0..len as u32)
        .map(|i| (seed.wrapping_mul(7919).wrapping_add(i * 31 + 5)) % 64)
        .collect()
}

#[test]
fn batched_padded_scoring_agrees_with_one_sequence_at_a_time() {
    let mut reference_model = model();
    // Different lengths, so the batches pad; multi-token continuations.
    let requests: Vec<LlRequest> = (1..=9)
        .map(|i| LlRequest {
            context: tokens(i, 4 + 3 * i as usize),
            continuation: tokens(i + 100, 1 + (i as usize % 4)),
        })
        .collect();
    let mut scored = LoadedModel::base(model()).with_batch_size(4);
    let got = scored.loglikelihood(&requests).unwrap();
    assert_eq!(got.len(), requests.len());
    for (request, result) in requests.iter().zip(&got) {
        let (expected, _) = reference(
            &mut reference_model,
            &request.context,
            &request.continuation,
        );
        assert!(
            (result.logprob - expected).abs() < 2e-3,
            "{} vs {expected}",
            result.logprob
        );
    }
}

#[test]
fn choices_of_one_token_after_a_shared_context_are_scored_from_one_forward() {
    let mut reference_model = model();
    let context = tokens(3, 20);
    // Four one-token choices after the same context, and a request that shares nothing.
    let mut requests: Vec<LlRequest> = (0..4)
        .map(|c| LlRequest {
            context: context.clone(),
            continuation: vec![10 + c],
        })
        .collect();
    requests.push(LlRequest {
        context: tokens(4, 11),
        continuation: vec![9],
    });
    // Another shared context of another length, in the same call.
    let other = tokens(5, 14);
    requests.extend((0..2).map(|c| LlRequest {
        context: other.clone(),
        continuation: vec![20 + c],
    }));
    let mut scored = LoadedModel::base(model()).with_batch_size(2);
    let got = scored.loglikelihood(&requests).unwrap();
    for (request, result) in requests.iter().zip(&got) {
        let (expected, greedy) = reference(
            &mut reference_model,
            &request.context,
            &request.continuation,
        );
        assert!(
            (result.logprob - expected).abs() < 2e-3,
            "{:?}: {} vs {expected}",
            request.continuation,
            result.logprob
        );
        assert_eq!(result.greedy, greedy, "{:?}", request.continuation);
    }
}

#[test]
fn greedy_is_whether_the_model_would_say_the_continuation() {
    let mut reference_model = model();
    let context = tokens(9, 12);
    // What the model says next, token by token.
    let mut said = Vec::new();
    for _ in 0..3 {
        let mut full = context.clone();
        full.extend(&said);
        let log_probs = next_token_log_probs(&mut reference_model, &full);
        let last = log_probs.shape()[0] - 1;
        let row = ops::select_axis(&log_probs, last, 0);
        let best = ops::argmax_axis(&row, -1).item_u32();
        said.push(best);
    }
    let wrong = vec![(said[0] + 1) % 64, said[1], said[2]];
    let mut scored = LoadedModel::base(model());
    let got = scored
        .loglikelihood(&[
            LlRequest {
                context: context.clone(),
                continuation: said.clone(),
            },
            LlRequest {
                context,
                continuation: wrong,
            },
        ])
        .unwrap();
    assert!(got[0].greedy && !got[1].greedy);
    assert!(got[0].logprob > got[1].logprob);
}

#[test]
fn a_sequence_longer_than_the_limit_loses_the_start_of_its_context() {
    let mut reference_model = model();
    let context = tokens(2, 40);
    let continuation = tokens(3, 3);
    let mut scored = LoadedModel::base(model()).with_max_len(20);
    let got = scored
        .loglikelihood(&[LlRequest {
            context: context.clone(),
            continuation: continuation.clone(),
        }])
        .unwrap();
    // 20 tokens: the last 17 of the context, and the continuation.
    let kept = &context[context.len() - 17..];
    let (expected, _) = reference(&mut reference_model, kept, &continuation);
    assert!((got[0].logprob - expected).abs() < 2e-3);
    // A continuation that alone overflows the limit cannot be scored.
    let too_long = LlRequest {
        context: tokens(1, 4),
        continuation: tokens(2, 30),
    };
    assert!(scored.loglikelihood(&[too_long]).is_err());
    assert!(
        scored
            .loglikelihood(&[LlRequest {
                context: Vec::new(),
                continuation: vec![1]
            }])
            .is_err()
    );
}

#[test]
fn generation_is_the_greedy_continuation_and_stops_when_asked() {
    let mut reference_model = model();
    let prompt = tokens(6, 10);
    let mut expected = Vec::new();
    for _ in 0..6 {
        let mut full = prompt.clone();
        full.extend(&expected);
        let log_probs = next_token_log_probs(&mut reference_model, &full);
        let row = ops::select_axis(&log_probs, log_probs.shape()[0] - 1, 0);
        expected.push(ops::argmax_axis(&row, -1).item_u32());
    }
    let mut generating = LoadedModel::base(model());
    let got = generating.generate(&prompt, 6, &mut |_| false).unwrap();
    assert_eq!(got, expected);
    // A stop request ends it after the token that asked.
    let stopped = generating
        .generate(&prompt, 6, &mut |so_far| so_far.len() == 3)
        .unwrap();
    assert_eq!(stopped, expected[..3]);
    assert!(generating.generate(&[], 4, &mut |_| false).is_err());
}
