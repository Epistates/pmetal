//! A real checkpoint decodes the same greedy tokens through the last-position
//! head as through the full one.
//!
//! `last_position_logits` holds `forward_with_cache_last` to the last row of the
//! full forward on tiny random models. This runs a released checkpoint's
//! weights, where the softcap saturates, bf16 rounds forty layers deep and a
//! prompt is past a sliding window: a greedy decode through
//! `forward_with_cache` (the head on every position, then the last row) and one
//! through `forward_with_cache_last` (the head on the last position) must emit
//! the same tokens, and their prefill logits must agree.
//!
//! `#[ignore]`d and gated on `PMETAL_LAST_REAL_DIR`, a checkpoint directory
//! the dispatcher loads.

use std::path::PathBuf;

use pmetal_cache::test_utils::argmax_last_axis;
use pmetal_models::DynamicModel;
use pmetal_runtime::compat::Array;
use pmetal_runtime::decode::last_token_hidden;

const PROMPT_TOKENS: usize = 96;
const NEW_TOKENS: usize = 24;

fn argmax_last(logits: &Array) -> i32 {
    argmax_last_axis(&last_token_hidden(logits))[0]
}

fn host(a: &Array) -> Vec<f32> {
    let mut a = a.as_dtype(pmetal_runtime::compat::Dtype::Float32);
    a.eval();
    let n = a.size();
    a.to_f32_vec(n).expect("to host")
}

#[test]
#[ignore = "needs PMETAL_LAST_REAL_DIR"]
fn greedy_decode_is_the_same_through_the_last_position_head() {
    let Some(dir) = std::env::var_os("PMETAL_LAST_REAL_DIR").map(PathBuf::from) else {
        eprintln!("PMETAL_LAST_REAL_DIR is not set; nothing to load");
        return;
    };
    let mut model = DynamicModel::load(&dir).expect("checkpoint loads");
    let vocab = model.vocab_size();
    let ids: Vec<i32> = (0..PROMPT_TOKENS as i32)
        .map(|i| 200 + (i * 53) % (vocab - 400).max(1))
        .collect();
    let prompt = Array::from_slice(&ids, &[1, PROMPT_TOKENS as i32]);
    let max_seq = PROMPT_TOKENS + NEW_TOKENS + 8;

    let run = |model: &mut DynamicModel, last_only: bool| -> (Vec<i32>, Vec<f32>) {
        let mut cache = model.create_cache(max_seq);
        let step = |model: &mut DynamicModel, ids: &Array, cache: &mut _| {
            if last_only {
                model.forward_with_cache_last(ids, None, Some(cache))
            } else {
                model
                    .forward_with_cache(ids, None, Some(cache))
                    .map(|logits| last_token_hidden(&logits))
            }
            .expect("forward")
        };
        let prefill = step(model, &prompt, &mut cache);
        let prefill_row = host(&prefill);
        let mut tokens = vec![argmax_last(&prefill)];
        for _ in 1..NEW_TOKENS {
            let next = Array::from_slice(&[*tokens.last().unwrap()], &[1, 1]);
            let logits = step(model, &next, &mut cache);
            tokens.push(argmax_last(&logits));
        }
        pmetal_runtime::check_unobserved_error().expect("a runtime op threw");
        (tokens, prefill_row)
    };

    let (full_tokens, full_row) = run(&mut model, false);
    let (last_tokens, last_row) = run(&mut model, true);
    println!("full: {full_tokens:?}\nlast: {last_tokens:?}");
    assert_eq!(last_tokens, full_tokens, "greedy tokens differ");
    assert_eq!(last_row.len(), full_row.len());
    let scale = full_row.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
    let worst = last_row
        .iter()
        .zip(&full_row)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    // The head runs on one row instead of the prompt's, and a bf16 matmul of
    // another shape reduces in another order: logits of magnitude 16 to 32 sit
    // 0.125 apart in bf16, so a rounding of one or two steps (about 1% of the
    // largest logit) is the same number. The greedy tokens above are the
    // behavioural check; this bounds the numbers to bf16 rounding.
    assert!(
        worst <= 1e-2 * scale,
        "prefill logits differ by {worst} over a scale of {scale}"
    );
}
