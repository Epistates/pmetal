//! ANE text generation on a real checkpoint.
//!
//!     cargo test -p pmetal-metal --test ane_lm --release -- --ignored --nocapture
//!
//! Uses Qwen3-0.6B from `PMETAL_TEST_QWEN3_0_6B` (a model directory), and
//! skips when it isn't set.

#![cfg(target_os = "macos")]

use pmetal_metal::ane::extend::WeightFormat;
use pmetal_metal::ane::lm::{AneLm, AneLmOptions};

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
            let (got, stats) = lm
                .generate(prompt, want.len(), 0.0, 0, &[], |_| true)
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
