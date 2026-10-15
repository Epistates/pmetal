//! Sampling on the ANE is the GPU path's sampling, penalties included.
//!
//!     PMETAL_TEST_QWEN3_0_6B=<dir> [PMETAL_TEST_QWEN3_4B=<dir>] \
//!     cargo test -p pmetal-models --features ane --release --test ane_penalties -- --ignored --nocapture
//!
//! The ANE engine sampled greedily or top-k on its own and dropped the
//! repetition, frequency and presence penalties (and top-p and min-p). It now
//! hands each position's logits to a [`HostLogitsSampler`], the GPU path's
//! sampler and penalties, so greedy decoding under a penalty picks the tokens
//! `generate_cached` picks on the GPU, and verifying guesses, which asks the
//! sampler for each position only while it agrees with the guess before it,
//! still changes only the speed.

#![cfg(all(target_os = "macos", feature = "ane"))]

use std::path::PathBuf;
use std::time::Instant;

use pmetal_metal::ane::extend::WeightFormat;
use pmetal_metal::ane::lm::{
    AneLm, AneLmOptions, GenerateOptions, Greedy, NoDraft, PromptLookup, TokenSampler,
};
use pmetal_models::DynamicModel;
use pmetal_models::generation::{GenerationConfig, HostLogitsSampler, generate_cached};

/// "The capital of France is"
const PROMPT: [u32; 5] = [785, 6722, 315, 9625, 374];
const NEW_TOKENS: usize = 10;

fn qwen3_0_6b() -> Option<PathBuf> {
    std::env::var_os("PMETAL_TEST_QWEN3_0_6B").map(PathBuf::from)
}

fn penalized(max_new: usize) -> GenerationConfig {
    GenerationConfig::greedy(max_new)
        .with_repetition_penalty(1.3)
        .with_frequency_penalty(0.4)
        .with_presence_penalty(0.3)
}

fn on_ane(lm: &mut AneLm, prompt: &[u32], config: &GenerationConfig) -> Vec<u32> {
    let opts = GenerateOptions {
        max_new: config.max_new_tokens,
        stop: Vec::new(),
    };
    let mut sampler = HostLogitsSampler::new(config, prompt);
    lm.generate(prompt, &opts, &mut sampler, &mut NoDraft, |_| true)
        .expect("generate")
        .0
}

fn load_fp16(dir: &std::path::Path) -> AneLm {
    let opts = AneLmOptions {
        capacity: 512,
        weights: WeightFormat::Fp16,
        ..AneLmOptions::default()
    };
    AneLm::load(dir, &opts).expect("load")
}

/// Greedy decoding under the penalties picks the same tokens on the ANE
/// (fp16 weights) as `generate_cached` on the GPU, and other tokens than
/// without them.
#[test]
#[ignore = "requires ANE hardware and PMETAL_TEST_QWEN3_0_6B"]
fn penalized_greedy_decoding_matches_the_gpu() {
    let Some(dir) = qwen3_0_6b() else {
        eprintln!("PMETAL_TEST_QWEN3_0_6B not set; skipping");
        return;
    };
    let config = penalized(NEW_TOKENS);
    let mut model = DynamicModel::load(&dir).expect("load on the GPU");
    let mut cache = model.create_cache(PROMPT.len() + NEW_TOKENS + 8);
    let want = generate_cached(
        |input, cache| model.forward_with_cache(input, None, Some(cache)),
        &PROMPT,
        config.clone(),
        &mut cache,
    )
    .expect("generate on the GPU")
    .token_ids[PROMPT.len()..]
        .to_vec();
    pmetal_runtime::check_unobserved_error().expect("GPU decode: no bridge op failed");

    let mut lm = load_fp16(&dir);
    let got = on_ane(&mut lm, &PROMPT, &config);
    let plain = on_ane(&mut lm, &PROMPT, &GenerationConfig::greedy(NEW_TOKENS));
    eprintln!("GPU {want:?}\nANE {got:?}\nANE without penalties {plain:?}");
    assert_ne!(
        got, plain,
        "the penalties changed nothing, so this checks nothing"
    );
    assert_eq!(got, want);
}

/// Verifying prompt-lookup guesses under the penalties keeps the tokens of
/// decoding one at a time under them: a guess is penalized for the guesses
/// before it, and a rejected one never reaches the history.
#[test]
#[ignore = "requires ANE hardware and PMETAL_TEST_QWEN3_0_6B"]
fn speculation_keeps_penalized_output() {
    let Some(dir) = qwen3_0_6b() else {
        eprintln!("PMETAL_TEST_QWEN3_0_6B not set; skipping");
        return;
    };
    let mut lm = load_fp16(&dir);
    // "def fibonacci(n):\n    " and its body twice over, then its start
    // again: prompt lookup guesses the body.
    let mut prompt: Vec<u32> = Vec::new();
    for _ in 0..2 {
        prompt.extend([
            750, 75698, 1445, 982, 257, 421, 308, 621, 220, 15, 510, 260, 470, 220, 15, 198,
        ]);
    }
    prompt.extend([750, 75698, 1445, 982, 257]);
    let opts = GenerateOptions {
        max_new: 40,
        stop: Vec::new(),
    };
    let config = penalized(40);
    let mut sampler = HostLogitsSampler::new(&config, &prompt);
    let (plain, _) = lm
        .generate(&prompt, &opts, &mut sampler, &mut NoDraft, |_| true)
        .expect("plain");
    let mut sampler = HostLogitsSampler::new(&config, &prompt);
    let (spec, stats) = lm
        .generate(
            &prompt,
            &opts,
            &mut sampler,
            &mut PromptLookup::default(),
            |_| true,
        )
        .expect("lookup");
    let (unpenalized, _) = lm
        .generate(&prompt, &opts, &mut Greedy, &mut NoDraft, |_| true)
        .expect("unpenalized");
    eprintln!(
        "{}/{} guesses kept\npenalized   {plain:?}\nunpenalized {unpenalized:?}",
        stats.accepted_tokens, stats.drafted_tokens
    );
    // Early enough that most positions decide under a history the
    // penalties changed.
    let diverges = plain.iter().zip(&unpenalized).position(|(a, b)| a != b);
    assert!(
        diverges.is_some_and(|at| at < 20),
        "the penalties change the output from {diverges:?} on; expected within 20 tokens"
    );
    assert_eq!(spec, plain);
    assert!(stats.accepted_tokens > 0, "no guess was kept");
}

/// Times a sampler's calls.
struct Timed<S> {
    inner: S,
    secs: f64,
    calls: usize,
}

impl<S: TokenSampler> TokenSampler for Timed<S> {
    fn sample(&mut self, logits: &[f32]) -> pmetal_metal::error::Result<u32> {
        let started = Instant::now();
        let token = self.inner.sample(logits);
        self.secs += started.elapsed().as_secs_f64();
        self.calls += 1;
        token
    }
}

/// What sampling through the GPU sampler costs per token on Qwen3-4B
/// (`PMETAL_TEST_QWEN3_4B`, int8), against a host argmax. Prints.
#[test]
#[ignore = "requires ANE hardware and PMETAL_TEST_QWEN3_4B; benchmark"]
fn sampling_overhead_on_qwen3_4b() {
    let Some(dir) = std::env::var_os("PMETAL_TEST_QWEN3_4B").map(PathBuf::from) else {
        eprintln!("PMETAL_TEST_QWEN3_4B not set; skipping");
        return;
    };
    let mut lm = AneLm::load(
        &dir,
        &AneLmOptions {
            capacity: 1024,
            ..AneLmOptions::default()
        },
    )
    .expect("load");
    // "<|im_start|>user\nWrite a Python function that returns the nth
    // Fibonacci number.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    let prompt = [
        151644, 872, 198, 7985, 264, 13027, 729, 429, 4675, 279, 55129, 79683, 1372, 13, 151645,
        198, 151644, 77091, 198, 151667, 271, 151668, 271,
    ];
    let opts = GenerateOptions {
        max_new: 128,
        stop: vec![151645],
    };
    let sampled = GenerationConfig::sampling(128, 0.7)
        .with_top_k(20)
        .with_top_p(0.8)
        .with_repetition_penalty(1.1)
        .with_presence_penalty(0.5)
        .with_seed(1);
    let configs: [(&str, Option<GenerationConfig>); 4] = [
        ("host argmax", None),
        ("greedy", Some(GenerationConfig::greedy(128))),
        ("greedy + penalties", Some(penalized(128))),
        ("top-k/top-p + penalties", Some(sampled)),
    ];
    // Warm up the compiled programs and the GPU.
    let _ = lm.generate(&prompt, &opts, &mut Greedy, &mut NoDraft, |_| true);
    for round in 0..2 {
        for (name, config) in &configs {
            let (secs, calls, stats) = match config {
                None => {
                    let mut timed = Timed {
                        inner: Greedy,
                        secs: 0.0,
                        calls: 0,
                    };
                    let (_, stats) = lm
                        .generate(&prompt, &opts, &mut timed, &mut NoDraft, |_| true)
                        .expect("generate");
                    (timed.secs, timed.calls, stats)
                }
                Some(config) => {
                    let mut timed = Timed {
                        inner: HostLogitsSampler::new(config, &prompt),
                        secs: 0.0,
                        calls: 0,
                    };
                    let (_, stats) = lm
                        .generate(&prompt, &opts, &mut timed, &mut NoDraft, |_| true)
                        .expect("generate");
                    (timed.secs, timed.calls, stats)
                }
            };
            if round == 1 {
                eprintln!(
                    "{name}: {:.3} ms/token sampling of {:.2} ms/token decode ({} tokens, {:.1} tok/s)",
                    secs * 1e3 / calls as f64,
                    stats.decode_secs * 1e3 / stats.generated_tokens as f64,
                    stats.generated_tokens,
                    stats.generated_tokens as f64 / stats.decode_secs,
                );
            }
        }
    }
}
