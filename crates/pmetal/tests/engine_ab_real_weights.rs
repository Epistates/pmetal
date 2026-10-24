//! The native engine against the shared model on one real checkpoint: the
//! same prompt, greedy, decoded by each in turn, so that whatever else the
//! machine is doing lands on both. Opt-in: needs a checkpoint of an
//! architecture that has a native engine, and the GPU to itself.
//!
//! ```text
//! PMETAL_ENGINE_AB_MODEL_DIR=<checkpoint directory> \
//!     cargo test --release -p pmetal --test engine_ab_real_weights -- --nocapture
//! ```
//!
//! Two processes run back to back see two machines: a GPU that has been
//! busy for a couple of seconds decodes at about half the speed of one that
//! has rested, and what else shares it moves the number too, which is more
//! than the engines differ by. Here both engines are resident, and a round is
//! one decode of each, native first on even rounds and shared first on odd
//! ones, each from the same rest. The verdict is the median of the per-round
//! ratios, not a ratio of two runs.
//!
//! `PMETAL_ENGINE_AB_ROUNDS` (default 6), `PMETAL_ENGINE_AB_TOKENS` (default
//! 64) and `PMETAL_ENGINE_AB_COOLDOWN_SECS` (default 0; 20 gives the GPU its
//! rest before each decode) size the measurement. The test fails when the engine
//! [`select_engine`](pmetal::native_inference::select_engine) picks decodes
//! more than 10% slower than the other, or when the two engines' greedy
//! outputs part at a token the model does not score as a near-tie.

use std::path::{Path, PathBuf};

use pmetal::native_inference::{Engine, EngineRequest, NativeArch, detect_arch, select_engine};
use pmetal_data::chat_templates::{ChatTemplateKwargs, Message, detect_chat_template};
use pmetal_models::dispatcher::{DynamicModel, DynamicModelLoadOptions};
use pmetal_models::generation::{GenerationConfig, generate_cached_async_streaming};
use pmetal_runtime::decode::{DecodeMetrics, SamplingParams};

const PROMPT: &str = "Write a short story about a lighthouse keeper.";

/// What one engine produced for one decode.
struct Decode {
    tokens: Vec<u32>,
    metrics: DecodeMetrics,
}

/// A native engine with its weights resident, ready to decode a prompt.
trait NativeDecoder {
    fn decode(&self, prompt: &[u32], tokens: usize) -> Decode;
}

struct Gemma4(pmetal_runtime::gemma4_native::NativeWeights);

impl NativeDecoder for Gemma4 {
    fn decode(&self, prompt: &[u32], tokens: usize) -> Decode {
        use pmetal_runtime::gemma4_native as g;
        let mut cache = g::build_cache(&self.0, &self.0.config);
        let (first, sampler) =
            g::prefill_first_token(&self.0, &mut cache, prompt, SamplingParams::new(0.0));
        let mut out = vec![first];
        let (rest, metrics) = g::generate(
            &self.0,
            &self.0.config,
            &mut cache,
            first,
            tokens - 1,
            sampler,
            &mut |t| {
                let _ = t;
                true
            },
        );
        out.extend(rest);
        Decode {
            tokens: out,
            metrics: metrics.expect("decode metrics"),
        }
    }
}

struct Qwen3(pmetal_runtime::qwen3_native::NativeWeights);

impl NativeDecoder for Qwen3 {
    fn decode(&self, prompt: &[u32], tokens: usize) -> Decode {
        use pmetal_runtime::qwen3_native as q;
        let mut cache = q::NativeCache::new_empty(&self.0);
        let (first, sampler) =
            q::prefill_first_token(&self.0, &mut cache, prompt, SamplingParams::new(0.0));
        let mut out = vec![first];
        let (rest, metrics) =
            q::generate(&self.0, &mut cache, first, tokens - 1, sampler, &mut |_| {
                true
            });
        out.extend(rest);
        Decode {
            tokens: out,
            metrics: metrics.expect("decode metrics"),
        }
    }
}

fn load_native(dir: &Path, arch: NativeArch) -> Box<dyn NativeDecoder> {
    match arch {
        NativeArch::Gemma4 => {
            let config = pmetal_runtime::gemma4_native::load_config(dir).expect("config");
            Box::new(Gemma4(
                pmetal_runtime::gemma4_native::load_model(dir, &config).expect("weights"),
            ))
        }
        NativeArch::Qwen3 | NativeArch::Qwen3_5 => {
            let config = pmetal_runtime::qwen3_native::load_config(dir).expect("config");
            Box::new(Qwen3(
                pmetal_runtime::qwen3_native::load_model(dir, &config).expect("weights"),
            ))
        }
        other => panic!("{} has no decoder in this test yet", other.label()),
    }
}

fn decode_shared(model: &mut DynamicModel, prompt: &[u32], tokens: usize) -> Decode {
    let mut cache = model.create_cache(prompt.len() + tokens + 64);
    let mut mamba = model.create_mamba_cache();
    let out = generate_cached_async_streaming(
        |input, cache| {
            model.forward_with_hybrid_cache_last(input, None, Some(cache), mamba.as_mut())
        },
        prompt,
        GenerationConfig::greedy(tokens),
        &mut cache,
        |_| true,
    )
    .expect("shared decode");
    Decode {
        // The output carries the prompt first.
        tokens: out.token_ids[prompt.len()..].to_vec(),
        metrics: out.decode_metrics.expect("decode metrics"),
    }
}

/// The shared model's logits for two candidate next tokens after `prefix`, and
/// its largest logit, to tell a near-tie that bf16 arithmetic can flip from a
/// real disagreement.
fn logits_of(
    model: &mut DynamicModel,
    prompt: &[u32],
    prefix: &[u32],
    candidates: [u32; 2],
) -> ([f32; 2], f32) {
    let mut ids = prompt.to_vec();
    ids.extend_from_slice(prefix);
    let mut cache = model.create_cache(ids.len() + 8);
    let mut mamba = model.create_mamba_cache();
    let logits = model
        .forward_with_hybrid_cache_last(
            &pmetal_runtime::decode::prompt_tokens_to_input(&ids),
            None,
            Some(&mut cache),
            mamba.as_mut(),
        )
        .expect("logits")
        .as_dtype(pmetal_runtime::compat::Dtype::Float32.as_i32());
    let n = logits.size();
    let mut flat = logits.reshape(&[n as i32]);
    let values = flat.to_f32_vec(n).expect("read logits");
    let top = values.iter().copied().fold(f32::MIN, f32::max);
    (candidates.map(|t| values[t as usize]), top)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
fn the_engine_selection_picks_the_faster_engine_on_a_real_model() {
    let Some(dir) = std::env::var_os("PMETAL_ENGINE_AB_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("PMETAL_ENGINE_AB_MODEL_DIR is not set; skipping");
        return;
    };
    let rounds: usize = std::env::var("PMETAL_ENGINE_AB_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);
    let tokens: usize = std::env::var("PMETAL_ENGINE_AB_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let cooldown: u64 = std::env::var("PMETAL_ENGINE_AB_COOLDOWN_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let arch = detect_arch(&dir).expect("a checkpoint of an architecture with a native engine");
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(&dir).expect("tokenizer");
    // Through the model's chat template, as `pmetal infer --chat` has it: an
    // instruction-tuned model asked to continue raw text gives two engines
    // near-ties at every step, and Gemma takes its BOS from the template.
    let mut kwargs = ChatTemplateKwargs::new();
    kwargs.set(ChatTemplateKwargs::ENABLE_THINKING, false);
    let template = detect_chat_template(&dir, &dir.to_string_lossy());
    let rendered = template
        .apply_inference_with_kwargs(&[Message::user(PROMPT)], None, &kwargs)
        .expect("chat template")
        .text;
    let prompt: Vec<u32> = tokenizer.encode(&rendered).expect("encode");

    let native = load_native(&dir, arch);
    let mut shared = DynamicModel::load_with_options(
        &dir,
        DynamicModelLoadOptions {
            prefer_expert_offload: false,
            keep_quantized: true,
        },
    )
    .expect("shared model");

    // Warm both: the first decode of each compiles its kernels.
    native.decode(&prompt, 16);
    decode_shared(&mut shared, &prompt, 16);

    let mut ratios = Vec::new();
    let (mut native_tps, mut shared_tps) = (Vec::new(), Vec::new());
    let mut common_prefix = usize::MAX;
    // Where the engines' greedy outputs first part, and how far apart the
    // shared model scores the two tokens: (token, gap, largest logit).
    let mut disagreement: Option<(usize, f32, f32)> = None;
    for round in 0..rounds {
        // A GPU that has just been busy decodes at about half the speed of one
        // that has rested, whichever engine is on it, so each decode starts
        // from rest and the two of a round start from the same state.
        let rest = || std::thread::sleep(std::time::Duration::from_secs(cooldown));
        let (n, s) = if round % 2 == 0 {
            rest();
            let n = native.decode(&prompt, tokens);
            rest();
            (n, decode_shared(&mut shared, &prompt, tokens))
        } else {
            rest();
            let s = decode_shared(&mut shared, &prompt, tokens);
            rest();
            (native.decode(&prompt, tokens), s)
        };
        let agree = n
            .tokens
            .iter()
            .zip(&s.tokens)
            .take_while(|(a, b)| a == b)
            .count();
        common_prefix = common_prefix.min(agree);
        if round == 0 {
            let show = |t: &[u32]| tokenizer.decode(&t[..t.len().min(16)]).unwrap_or_default();
            eprintln!(
                "native: {:?}\nshared: {:?}",
                show(&n.tokens),
                show(&s.tokens)
            );
            if agree < n.tokens.len().min(s.tokens.len()) {
                let ([native_logit, shared_logit], top) = logits_of(
                    &mut shared,
                    &prompt,
                    &s.tokens[..agree],
                    [n.tokens[agree], s.tokens[agree]],
                );
                let gap = (native_logit - shared_logit).abs();
                eprintln!(
                    "first disagreement at token {agree}: the shared model scores native's pick {native_logit:.3} and its own {shared_logit:.3} (largest logit {top:.3}, gap {gap:.3})"
                );
                disagreement = Some((agree, gap, top));
            }
        }
        // The median step, so a stall that hit one engine's decode does not
        // set the round's speed.
        let (nt, st) = (
            1000.0 / n.metrics.p50_step_ms,
            1000.0 / s.metrics.p50_step_ms,
        );
        eprintln!(
            "round {round}: native {nt:.1} tok/s (mean {:.1}), shared {st:.1} tok/s (mean {:.1}), agree on {agree}/{} tokens",
            n.metrics.tok_per_sec, s.metrics.tok_per_sec, tokens
        );
        ratios.push(nt / st);
        native_tps.push(nt);
        shared_tps.push(st);
    }
    let ratio = median(ratios);
    eprintln!(
        "{arch:?}: native/shared = {ratio:.2} (median of {rounds} rounds; native {:.1}, shared {:.1} tok/s median); greedy outputs agree on the first {common_prefix} tokens",
        median(native_tps),
        median(shared_tps)
    );

    let choice = select_engine(&dir, &EngineRequest::default());
    eprintln!("selected engine: {:?} ({})", choice.engine, choice.reason);
    let chosen_over_other = match choice.engine {
        Engine::Native => ratio,
        Engine::Shared => 1.0 / ratio,
    };
    assert!(
        chosen_over_other >= 0.9,
        "the selected engine ({:?}) decodes at {:.2}x the other's speed",
        choice.engine,
        chosen_over_other
    );
    // Two engines that round differently part at a token the model scores
    // alike, to within a few bf16 steps of its largest logit (8 bits of
    // mantissa: 3%). A gap of whole logits is a different computation.
    if let Some((at, gap, top)) = disagreement {
        assert!(
            gap <= 0.03 * top.abs().max(1.0),
            "the engines' greedy outputs part at token {at}, where the shared model scores \
             the two picks {gap:.3} apart (largest logit {top:.3}): not a rounding tie"
        );
    }
}
