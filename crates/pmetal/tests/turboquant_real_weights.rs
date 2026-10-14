//! The native engine's TurboQuant KV cache on a real checkpoint: how much
//! each width costs in perplexity and in agreement with the plain bf16 cache,
//! on natural text, through both the prefill and the single-token decode
//! paths. Opt-in: needs a dense Qwen3 or a Qwen3.5-family checkpoint
//! (Qwen3-0.6B or Qwen3.5-0.8B is enough), and the GPU to itself.
//!
//! ```text
//! PMETAL_TQ_MODEL_DIR=<checkpoint directory> \
//!     cargo test --release -p pmetal --test turboquant_real_weights -- --nocapture
//! ```
//!
//! The text is the changelog's first 1,600 tokens. The first 256 go through
//! as one prefill chunk and the rest one token at a time; every position's
//! log-probability of the true next token is taken from the model's logits.
//! The TurboQuant caches keep 4 sinks and a 64-token recent window, so
//! nearly all of the text's keys and values are quantized: that is a
//! harsher setting than the default 8,192-token window, which leaves a text
//! this short untouched.

use std::path::PathBuf;

use pmetal_bridge::InlineArray;
use pmetal_bridge::qwen3_native::{self, NativeCache, NativeWeights};
use pmetal_bridge::turboquant::TurboQuantConfig;

const TEXT_TOKENS: usize = 1_600;
const PREFILL: usize = 256;

struct Run {
    /// Negative log-probability of each true next token.
    nll: Vec<f64>,
    /// The model's top token at each position.
    top1: Vec<u32>,
}

impl Run {
    fn perplexity(&self) -> f64 {
        (self.nll.iter().sum::<f64>() / self.nll.len() as f64).exp()
    }
}

fn run(weights: &NativeWeights, tokens: &[i32], quantized: Option<TurboQuantConfig>) -> Run {
    let mut cache = match quantized {
        Some(config) => NativeCache::new_with_turboquant(weights, Some(config)),
        None => NativeCache::new_empty(weights),
    };
    let vocab = weights.embed_w.dim(0);
    let mut nll = Vec::new();
    let mut top1 = Vec::new();
    let mut at = 0usize;
    while at + 1 < tokens.len() {
        let step = if at == 0 { PREFILL } else { 1 }.min(tokens.len() - 1 - at);
        let input = InlineArray::from_i32_slice(&tokens[at..at + step]).reshape(&[1, step as i32]);
        let logits = qwen3_native::forward_step(weights, &input, &mut cache)
            .reshape(&[step as i32, vocab])
            .as_dtype(pmetal_bridge::compat::Dtype::Float32.as_i32());
        let targets =
            InlineArray::from_i32_slice_shaped(&tokens[at + 1..at + 1 + step], &[step as i32, 1]);
        let mut picked = logits.log_softmax(-1).take_along_axis(&targets, -1);
        let best = logits.argmax(-1);
        picked.eval();
        best.eval();
        let picked = picked.reshape(&[step as i32]).to_f32_vec(step).unwrap();
        nll.extend(picked.iter().map(|lp| -f64::from(*lp)));
        top1.extend_from_slice(best.as_slice::<u32>());
        at += step;
    }
    pmetal_bridge::check_unobserved_error().expect("a bridge op threw");
    Run { nll, top1 }
}

#[test]
fn turboquant_costs_little_perplexity_on_a_real_model() {
    let Some(dir) = std::env::var_os("PMETAL_TQ_MODEL_DIR").map(PathBuf::from) else {
        eprintln!("PMETAL_TQ_MODEL_DIR is not set; skipping");
        return;
    };
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(&dir).expect("tokenizer");
    let text = include_str!("../../../CHANGELOG.md");
    let tokens: Vec<i32> = tokenizer
        .encode(text)
        .expect("encode")
        .into_iter()
        .take(TEXT_TOKENS)
        .map(|t| t as i32)
        .collect();
    assert_eq!(tokens.len(), TEXT_TOKENS, "the text is too short");

    let config = qwen3_native::load_config(&dir).expect("config");
    let weights = qwen3_native::load_model(&dir, &config).expect("weights");

    let base = run(&weights, &tokens, None);
    let base_ppl = base.perplexity();
    eprintln!("bf16 cache: perplexity {base_ppl:.4}");

    // PMETAL_TQ_EXPLORE=key_bits:value_bits:window:sinks[:center],... prints
    // those configurations' numbers and asserts nothing (`center` is 0 or 1,
    // default 1).
    if let Ok(list) = std::env::var("PMETAL_TQ_EXPLORE") {
        for spec in list.split(',') {
            let p: Vec<usize> = spec.split(':').map(|v| v.parse().unwrap()).collect();
            let center = p.get(4).is_none_or(|c| *c != 0);
            let scale = p.get(5).is_none_or(|c| *c != 0);
            let tq = TurboQuantConfig::uniform(p[0] as u8, p[1] as u8)
                .with_recent_window(Some(p[2]))
                .with_sink_tokens(p[3])
                .with_center_keys(center)
                .with_scale_key_channels(scale);
            let quantized = run(&weights, &tokens, Some(tq));
            let agree = base
                .top1
                .iter()
                .zip(&quantized.top1)
                .filter(|(a, b)| a == b)
                .count() as f64
                / base.top1.len() as f64;
            eprintln!(
                "k{}v{} window {} sinks {} centre {} scale {}: perplexity {:.4} ({:+.2}%), top-1 agreement {:.1}%",
                p[0],
                p[1],
                p[2],
                p[3],
                center,
                scale,
                quantized.perplexity(),
                100.0 * (quantized.perplexity() / base_ppl - 1.0),
                100.0 * agree
            );
        }
        return;
    }

    // (bits, largest perplexity increase, smallest top-1 agreement). On
    // Qwen3-0.6B the increases measure -0.1%, -0.1%, +0.3% and +2.3%, with
    // 98%, 98%, 96% and 92% agreement; without the key frame the 4-bit cache
    // costs 273% and the 3-bit one 999%.
    let widths = [
        (8u8, 0.01, 0.97),
        (6, 0.01, 0.96),
        (4, 0.02, 0.92),
        (3, 0.06, 0.88),
    ];
    for (bits, max_increase, min_agreement) in widths {
        let tq = TurboQuantConfig::uniform(bits, bits)
            .with_recent_window(Some(64))
            .with_sink_tokens(4);
        let quantized = run(&weights, &tokens, Some(tq));
        let ppl = quantized.perplexity();
        let agree = base
            .top1
            .iter()
            .zip(&quantized.top1)
            .filter(|(a, b)| a == b)
            .count() as f64
            / base.top1.len() as f64;
        let increase = ppl / base_ppl - 1.0;
        eprintln!(
            "{bits}-bit cache: perplexity {ppl:.4} ({:+.2}%), top-1 agreement {:.1}%",
            100.0 * increase,
            100.0 * agree
        );
        assert!(
            increase < max_increase,
            "{bits}-bit: perplexity rose {:.2}% (limit {:.0}%)",
            100.0 * increase,
            100.0 * max_increase
        );
        assert!(
            agree > min_agreement,
            "{bits}-bit: top-1 agreement {agree:.3} (floor {min_agreement})"
        );
    }
}
