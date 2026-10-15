//! DFlash drafting (GPU) for a model on the ANE, on Qwen3-4B.
//!
//!     PMETAL_TEST_QWEN3_4B=<dir> PMETAL_TEST_QWEN3_4B_DFLASH=<dir> \
//!     cargo test -p pmetal-models --features ane --release --test ane_dflash -- --ignored --nocapture
//!
//! The draft model is z-lab/Qwen3-4B-DFlash-b16.

#![cfg(all(target_os = "macos", feature = "ane"))]

use std::path::PathBuf;

use pmetal_metal::ane::lm::{AneLm, AneLmOptions, GenerateOptions, Greedy, NoDraft};
use pmetal_models::dflash_drafter::DFlashDrafter;
use pmetal_models::generation::{GenerationConfig, HostLogitsSampler};
use pmetal_runtime::QuantizedMode;
use pmetal_runtime::native_weight::QuantParams;

/// Chat prompts (thinking off).
const PROMPTS: [(&str, &[u32]); 3] = [
    (
        "fibonacci",
        &[
            151644, 872, 198, 7985, 264, 13027, 729, 429, 4675, 279, 55129, 79683, 1372, 13,
            151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271,
        ],
    ),
    (
        "quicksort",
        &[
            151644, 872, 198, 7985, 264, 13027, 3974, 6860, 8129, 448, 11682, 6042, 13, 151645,
            198, 151644, 77091, 198, 151667, 271, 151668, 271,
        ],
    ),
    (
        "hash map",
        &[
            151644, 872, 198, 840, 20772, 1246, 264, 5175, 2415, 4278, 323, 979, 311, 990, 825, 13,
            151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271,
        ],
    ),
];

fn dirs() -> Option<(PathBuf, PathBuf)> {
    let target = std::env::var_os("PMETAL_TEST_QWEN3_4B")?;
    let draft = std::env::var_os("PMETAL_TEST_QWEN3_4B_DFLASH")?;
    Some((target.into(), draft.into()))
}

/// Drafting changes how many passes generation takes, not what it generates:
/// greedy output with DFlash is greedy output without it. Prints the tokens
/// per pass, and holds them to a floor a drafter that saw only part of its
/// context (about 2) misses.
#[test]
#[ignore = "requires ANE hardware, PMETAL_TEST_QWEN3_4B and PMETAL_TEST_QWEN3_4B_DFLASH"]
fn dflash_keeps_greedy_output_on_the_ane() {
    let Some((target, draft)) = dirs() else {
        eprintln!("PMETAL_TEST_QWEN3_4B / PMETAL_TEST_QWEN3_4B_DFLASH not set; skipping");
        return;
    };
    let capacity = 1024;
    let eight_bit = QuantParams {
        group_size: 64,
        bits: 8,
        mode: QuantizedMode::Affine,
    };
    let mut drafters = [
        (
            "bf16",
            DFlashDrafter::load(&draft, &target, capacity, None).expect("load draft"),
        ),
        (
            "8-bit",
            DFlashDrafter::load(&draft, &target, capacity, Some(eight_bit)).expect("load draft"),
        ),
    ];
    let opts = AneLmOptions {
        capacity,
        taps: drafters[0].1.target_layer_ids().to_vec(),
        ..AneLmOptions::default()
    };
    let mut lm = AneLm::load(&target, &opts).expect("load target");
    let settings = GenerateOptions {
        max_new: 192,
        stop: vec![151645],
    };
    let mut per_pass = Vec::new();
    for (name, prompt) in PROMPTS {
        let (plain, p) = lm
            .generate(prompt, &settings, &mut Greedy, &mut NoDraft, |_| true)
            .expect("plain");
        eprintln!(
            "{name}: plain {:.1} tok/s",
            p.generated_tokens as f64 / p.decode_secs
        );
        for (precision, drafter) in &mut drafters {
            let (spec, s) = lm
                .generate(prompt, &settings, &mut Greedy, drafter, |_| true)
                .expect("dflash");
            let tokens_per_pass = s.generated_tokens as f64 / (s.passes + 1) as f64;
            eprintln!(
                "  {precision} draft: {:.1} tok/s, {tokens_per_pass:.2} tokens/pass, {:.1} ms/pass \
                 of which {:.1} drafting ({} tokens, {} passes, {}/{} guesses kept)",
                s.generated_tokens as f64 / s.decode_secs,
                s.decode_secs * 1e3 / (s.passes + 1) as f64,
                s.draft_secs * 1e3 / (s.passes + 1) as f64,
                s.generated_tokens,
                s.passes + 1,
                s.accepted_tokens,
                s.drafted_tokens,
            );
            assert_eq!(
                spec, plain,
                "{name}: the {precision} draft changed the greedy output"
            );
            per_pass.push(tokens_per_pass);
        }
    }
    // The fibonacci prompt, either draft.
    assert!(
        per_pass[0] > 4.0 && per_pass[1] > 4.0,
        "fibonacci: {:.2} and {:.2} tokens per pass; expected above 4",
        per_pass[0],
        per_pass[1]
    );
}

/// Under the penalties, and sampling with a seed, DFlash still changes only
/// the speed: the sampler decides each verified position in turn, penalized
/// for the drafts before it, so the output is that of decoding one token at a
/// time with the same sampler.
#[test]
#[ignore = "requires ANE hardware, PMETAL_TEST_QWEN3_4B and PMETAL_TEST_QWEN3_4B_DFLASH"]
fn dflash_keeps_penalized_and_sampled_output_on_the_ane() {
    let Some((target, draft)) = dirs() else {
        eprintln!("PMETAL_TEST_QWEN3_4B / PMETAL_TEST_QWEN3_4B_DFLASH not set; skipping");
        return;
    };
    let capacity = 1024;
    let eight_bit = QuantParams {
        group_size: 64,
        bits: 8,
        mode: QuantizedMode::Affine,
    };
    let mut drafter =
        DFlashDrafter::load(&draft, &target, capacity, Some(eight_bit)).expect("load draft");
    let opts = AneLmOptions {
        capacity,
        taps: drafter.target_layer_ids().to_vec(),
        ..AneLmOptions::default()
    };
    let mut lm = AneLm::load(&target, &opts).expect("load target");
    let settings = GenerateOptions {
        max_new: 128,
        stop: vec![151645],
    };
    let configs = [
        (
            "greedy + penalties",
            GenerationConfig::greedy(128)
                .with_repetition_penalty(1.1)
                .with_presence_penalty(1.5),
        ),
        (
            "seeded top-k/top-p + penalties",
            GenerationConfig::sampling(128, 0.7)
                .with_top_k(20)
                .with_top_p(0.8)
                .with_repetition_penalty(1.1)
                .with_presence_penalty(1.5)
                .with_seed(3),
        ),
    ];
    let (_, prompt) = PROMPTS[0];
    let (unpenalized, _) = lm
        .generate(prompt, &settings, &mut Greedy, &mut NoDraft, |_| true)
        .expect("unpenalized");
    for (name, config) in &configs {
        let mut sampler = HostLogitsSampler::new(config, prompt);
        let (plain, p) = lm
            .generate(prompt, &settings, &mut sampler, &mut NoDraft, |_| true)
            .expect("plain");
        let mut sampler = HostLogitsSampler::new(config, prompt);
        let (spec, s) = lm
            .generate(prompt, &settings, &mut sampler, &mut drafter, |_| true)
            .expect("dflash");
        eprintln!(
            "{name}: plain {:.1} tok/s; DFlash {:.1} tok/s, {:.2} tokens/pass ({}/{} guesses kept)",
            p.generated_tokens as f64 / p.decode_secs,
            s.generated_tokens as f64 / s.decode_secs,
            s.generated_tokens as f64 / (s.passes + 1) as f64,
            s.accepted_tokens,
            s.drafted_tokens,
        );
        assert_ne!(plain, unpenalized, "{name} changed nothing");
        assert_eq!(spec, plain, "{name}: DFlash changed the output");
        assert!(s.accepted_tokens > 0, "{name}: no draft was kept");
    }
}
