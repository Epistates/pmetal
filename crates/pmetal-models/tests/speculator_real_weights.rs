//! A real Qwen3.5 checkpoint with its bundled MTP head: speculative greedy
//! decoding gives plain greedy decoding's tokens. Opt-in:
//!
//! ```text
//! PMETAL_QWEN35_DIR=<snapshot> cargo test --release -p pmetal-models \
//!     --test speculator_real_weights -- --nocapture
//! ```
//!
//! The model runs bf16, and the verify's block-shaped forward rounds
//! differently from one-token decoding. Where the two decodings part, the
//! target's top two logits must be a rounding apart (bf16 steps are 1/8 for
//! logits between 16 and 32), and the two tokens must be those two.

use std::path::PathBuf;

use pmetal_data::Tokenizer;
use pmetal_models::DynamicModel;
use pmetal_models::generation::GenerationConfig;
use pmetal_models::speculator::{Speculator, SpeculatorSpec};
use pmetal_runtime::decode::{last_token_logits, prompt_tokens_to_input};
use serial_test::serial;

const NEW_TOKENS: usize = 160;
/// Two bf16 logits in [16, 32) are 0.125 apart; allow two steps.
const TIE: f32 = 0.25;

fn plain(model: &mut DynamicModel, prompt: &[u32], config: &GenerationConfig) -> Vec<u32> {
    let mut kv = model.create_cache(prompt.len() + NEW_TOKENS + 8);
    let mut mamba = model.create_mamba_cache();
    let mut sampler = config.sampler(prompt);
    let mut input = prompt_tokens_to_input(prompt);
    let mut out = Vec::new();
    while out.len() < NEW_TOKENS {
        let logits = model
            .forward_with_hybrid_cache(&input, None, Some(&mut kv), mamba.as_mut())
            .unwrap();
        let token = sampler.sample(&last_token_logits(&logits));
        token.eval();
        let token = token.item_u32();
        if config.is_stop_token(token) {
            break;
        }
        out.push(token);
        input = prompt_tokens_to_input(&[token]);
    }
    out
}

/// The target's two likeliest tokens after `prefix`, with their logits.
fn top_two(model: &mut DynamicModel, prefix: &[u32]) -> [(u32, f32); 2] {
    let mut kv = model.create_cache(prefix.len() + 8);
    let mut mamba = model.create_mamba_cache();
    let logits = model
        .forward_with_hybrid_cache(
            &prompt_tokens_to_input(prefix),
            None,
            Some(&mut kv),
            mamba.as_mut(),
        )
        .unwrap();
    let row = last_token_logits(&logits)
        .as_dtype(pmetal_runtime::compat::Dtype::Float32.as_i32())
        .reshape(&[-1]);
    row.eval();
    let values = row.as_slice::<f32>();
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]));
    [
        (order[0] as u32, values[order[0]]),
        (order[1] as u32, values[order[1]]),
    ]
}

#[test]
#[serial]
fn a_real_qwen_mtp_decode_is_plain_greedy_up_to_a_rounding() {
    let Some(dir) = std::env::var_os("PMETAL_QWEN35_DIR").map(PathBuf::from) else {
        eprintln!("PMETAL_QWEN35_DIR is not set; skipping");
        return;
    };
    let tokenizer = Tokenizer::from_model_dir(&dir).unwrap();
    let mut model = DynamicModel::load(&dir).unwrap();
    let mut speculator = Speculator::load(
        &model,
        &SpeculatorSpec::QwenMtp {
            checkpoint: dir.clone(),
            draft_tokens: 3,
        },
        4096,
    )
    .expect("the bundled MTP head loads");

    let mut parted = 0;
    for text in [
        "The capital of France is",
        "Explain how a binary search tree works.",
        "Once upon a time, in a small lighthouse,",
    ] {
        let prompt = tokenizer.encode(text).unwrap();
        let config = GenerationConfig::greedy(NEW_TOKENS);
        let want = plain(&mut model, &prompt, &config);
        let mut got = Vec::new();
        let run = speculator
            .generate(&mut model, &prompt, &config, config.sampler(&prompt), |t| {
                got.push(t);
                true
            })
            .unwrap();
        pmetal_runtime::check_unobserved_error().unwrap();
        eprintln!(
            "{text:?}: {} tokens, drafted {}, accepted {} ({:.0}%)",
            got.len(),
            run.drafted_tokens,
            run.accepted_tokens,
            100.0 * run.acceptance_rate()
        );
        match got.iter().zip(&want).position(|(a, b)| a != b) {
            None => assert_eq!(
                got.len(),
                want.len(),
                "{text:?}: one is a prefix of the other"
            ),
            Some(at) => {
                parted += 1;
                let mut prefix = prompt.clone();
                prefix.extend_from_slice(&want[..at]);
                let [first, second] = top_two(&mut model, &prefix);
                let mut chosen = [want[at], got[at]];
                chosen.sort_unstable();
                let mut top = [first.0, second.0];
                top.sort_unstable();
                eprintln!(
                    "  parts at token {at}: plain {} speculative {}; the target's top two are \
                     {} at {:.3} and {} at {:.3}",
                    want[at], got[at], first.0, first.1, second.0, second.1
                );
                assert_eq!(chosen, top, "{text:?}: the two decodings chose the top two");
                assert!(
                    first.1 - second.1 <= TIE,
                    "{text:?}: the decodings part at token {at} where the target's top two \
                     logits are {} apart: more than a rounding",
                    first.1 - second.1
                );
            }
        }
    }
    eprintln!("{parted} of 3 prompts parted, each at a tie");
}
