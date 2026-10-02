//! ANE text generation on a real checkpoint.
//!
//!     cargo test -p pmetal-metal --test ane_lm --release -- --ignored --nocapture
//!
//! Uses Qwen3-0.6B from `PMETAL_TEST_QWEN3_0_6B` (a model directory), and
//! skips when it isn't set.

#![cfg(target_os = "macos")]

use pmetal_metal::ane::extend::WeightFormat;
use pmetal_metal::ane::lm::{AneLm, AneLmOptions, Drafter, GenerateOptions, NoDraft, PromptLookup};

fn model_dir() -> Option<std::path::PathBuf> {
    let dir = std::env::var_os("PMETAL_TEST_QWEN3_0_6B")?;
    Some(dir.into())
}

/// Greedy continuations from mlx_lm (bf16) for Qwen3-0.6B.
const CASES: [(&[u32], &[u32]); 2] = [
    // "The capital of France is" -> " Paris. The capital of France is also the"
    (
        &[785, 6722, 315, 9625, 374],
        &[12095, 13, 576, 6722, 315, 9625, 374, 1083, 279],
    ),
    // "def fibonacci(n):\n    " -> " if n == 0:\n         return 0"
    (
        &[750, 75698, 1445, 982, 257],
        &[421, 308, 621, 220, 15, 510, 260, 470, 220, 15],
    ),
];

#[test]
#[ignore = "requires ANE hardware and PMETAL_TEST_QWEN3_0_6B"]
fn greedy_generation_matches_mlx_lm() {
    let Some(dir) = model_dir() else {
        eprintln!("PMETAL_TEST_QWEN3_0_6B not set; skipping");
        return;
    };
    for weights in [WeightFormat::Fp16, WeightFormat::Int8] {
        let opts = AneLmOptions {
            capacity: 512,
            weights,
            ..AneLmOptions::default()
        };
        let mut lm = AneLm::load(&dir, &opts).expect("load");
        for (prompt, want) in CASES {
            let opts = GenerateOptions {
                max_new: want.len(),
                temperature: 0.0,
                top_k: 0,
                stop: Vec::new(),
            };
            let (got, stats) = lm
                .generate(prompt, &opts, &mut NoDraft, |_| true)
                .expect("generate");
            eprintln!(
                "{weights:?}: {got:?} ({} tokens in {:.0} ms decode)",
                stats.generated_tokens,
                stats.decode_secs * 1e3
            );
            match weights {
                WeightFormat::Fp16 => assert_eq!(got, want, "fp16 greedy continuation"),
                // int8 per output channel is lossy on a model this small; it
                // agrees for the first several tokens.
                WeightFormat::Int8 => assert_eq!(got[..5], want[..5], "int8 greedy continuation"),
            }
        }
    }
}

/// Verifying guesses changes how fast tokens come, not which: greedy output
/// with prompt-lookup guesses is greedy output without them. The prompt
/// repeats itself, so guesses are made and some kept.
#[test]
#[ignore = "requires ANE hardware and PMETAL_TEST_QWEN3_0_6B"]
fn speculation_keeps_greedy_output() {
    let Some(dir) = model_dir() else {
        eprintln!("PMETAL_TEST_QWEN3_0_6B not set; skipping");
        return;
    };
    let opts = AneLmOptions {
        capacity: 512,
        weights: WeightFormat::Fp16,
        ..AneLmOptions::default()
    };
    let mut lm = AneLm::load(&dir, &opts).expect("load");
    // "def fibonacci(n):\n    " twice over, then its start again.
    let mut prompt: Vec<u32> = Vec::new();
    for _ in 0..2 {
        prompt.extend([
            750, 75698, 1445, 982, 257, 421, 308, 621, 220, 15, 510, 260, 470, 220, 15, 198,
        ]);
    }
    prompt.extend([750, 75698, 1445, 982, 257]);
    let settings = GenerateOptions {
        max_new: 40,
        temperature: 0.0,
        top_k: 0,
        stop: Vec::new(),
    };
    let (plain, plain_stats) = lm
        .generate(&prompt, &settings, &mut NoDraft, |_| true)
        .unwrap();
    let mut lookup = PromptLookup::default();
    let drafter: &mut dyn Drafter = &mut lookup;
    let (spec, stats) = lm.generate(&prompt, &settings, drafter, |_| true).unwrap();
    eprintln!(
        "plain: {} tokens in {} passes, {:.0} ms; lookup: {} passes, {}/{} guesses kept, {:.0} ms",
        plain.len(),
        plain_stats.passes + 1,
        plain_stats.decode_secs * 1e3,
        stats.passes + 1,
        stats.accepted_tokens,
        stats.drafted_tokens,
        stats.decode_secs * 1e3,
    );
    assert_eq!(spec, plain);
    assert!(
        stats.accepted_tokens > 0,
        "the repeated prompt should yield kept guesses"
    );
}

/// Load time and decode speed on Qwen3-4B (`PMETAL_TEST_QWEN3_4B`), plain
/// and with prompt lookup. Prints; checks only that both agree.
#[test]
#[ignore = "requires ANE hardware and PMETAL_TEST_QWEN3_4B; benchmark"]
fn qwen3_4b_throughput() {
    let Some(dir) = std::env::var_os("PMETAL_TEST_QWEN3_4B") else {
        eprintln!("PMETAL_TEST_QWEN3_4B not set; skipping");
        return;
    };
    let started = std::time::Instant::now();
    let opts = AneLmOptions {
        capacity: 1024,
        ..AneLmOptions::default()
    };
    let mut lm = AneLm::load(std::path::Path::new(&dir), &opts).expect("load");
    eprintln!("load {:.1?}", started.elapsed());
    // The fibonacci prompt from above, twice, then its start (Qwen3 shares
    // the tokenizer).
    let mut prompt: Vec<u32> = Vec::new();
    for _ in 0..2 {
        prompt.extend([
            750, 75698, 1445, 982, 257, 421, 308, 621, 220, 15, 510, 260, 470, 220, 15, 198,
        ]);
    }
    prompt.extend([750, 75698, 1445, 982, 257]);
    let settings = GenerateOptions {
        max_new: 64,
        temperature: 0.0,
        top_k: 0,
        stop: Vec::new(),
    };
    let (plain, p) = lm
        .generate(&prompt, &settings, &mut NoDraft, |_| true)
        .unwrap();
    let mut lookup = PromptLookup::default();
    let (spec, s) = lm
        .generate(&prompt, &settings, &mut lookup, |_| true)
        .unwrap();
    for (name, st) in [("plain", &p), ("lookup", &s)] {
        eprintln!(
            "{name}: prefill {} tokens {:.0} ms; decode {} tokens in {} passes, {:.1} ms/pass, {:.1} tok/s ({}/{} guesses kept)",
            st.prompt_tokens,
            st.prefill_secs * 1e3,
            st.generated_tokens,
            st.passes + 1,
            st.decode_secs * 1e3 / (st.passes + 1) as f64,
            st.generated_tokens as f64 / st.decode_secs,
            st.accepted_tokens,
            st.drafted_tokens,
        );
    }
    assert_eq!(spec, plain);
}
