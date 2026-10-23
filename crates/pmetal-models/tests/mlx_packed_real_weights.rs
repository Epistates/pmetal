//! A real MLX-quantized checkpoint, run packed and run unpacked.
//!
//! The packed model reads MLX's quantized matmul; the unpacked one reads the
//! same bytes dequantized to bf16 and multiplied densely. They compute the same
//! function with different rounding, so a greedy continuation can part where two
//! tokens are within bf16 of each other. This test feeds both the same text (the
//! unpacked model's own greedy continuation) and compares what each predicts at
//! every position: the argmax, and how far the two distributions are apart. A
//! disagreement is explained when the unpacked model's top two tokens were a
//! bf16 step apart.
//!
//! Checkpoints are not committed, so this is `#[ignore]`d:
//!
//! ```bash
//! PMETAL_PACKED_MODEL=/path/to/mlx-4bit-checkpoint \
//!     cargo test -p pmetal-models --test mlx_packed_real_weights -- --ignored --nocapture
//! ```

use pmetal_data::Tokenizer;
use pmetal_models::DynamicModel;
use pmetal_models::dispatcher::DynamicModelLoadOptions;
use pmetal_runtime::compat::Array;

const PROMPT: &str = "The lighthouse keeper climbed the spiral stairs every evening, and the sea";
const CONTINUATION: usize = 48;

fn ids(tokens: &[u32]) -> Array {
    let tokens: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    Array::from_i32_slice_shaped(&tokens, &[1, tokens.len() as i32])
}

/// Logits of every position as `[positions][vocab]` in f32.
fn rows(logits: &Array) -> Vec<Vec<f32>> {
    let shape = logits.shape();
    let (seq, vocab) = (shape[1] as usize, shape[2] as usize);
    let mut flat = logits.clone();
    flat.eval();
    let all = flat.to_f32_vec(seq * vocab).expect("logits to host");
    all.chunks(vocab).map(<[f32]>::to_vec).collect()
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = row.iter().map(|&x| (x as f64 - max).exp()).sum();
    let log_sum = sum.ln() + max;
    row.iter().map(|&x| x as f64 - log_sum).collect()
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 { (i, v) } else { best }
        })
        .0
}

#[test]
#[ignore = "needs a real checkpoint; set PMETAL_PACKED_MODEL"]
fn a_real_checkpoint_packed_agrees_with_the_same_bytes_unpacked() {
    let Ok(dir) = std::env::var("PMETAL_PACKED_MODEL") else {
        panic!("set PMETAL_PACKED_MODEL to an MLX-quantized checkpoint directory");
    };
    let tokenizer = Tokenizer::from_file(std::path::Path::new(&dir).join("tokenizer.json"))
        .expect("tokenizer.json");
    let prompt = tokenizer.encode(PROMPT).expect("encode");

    let load = |keep_quantized| {
        DynamicModel::load_with_options(
            &dir,
            DynamicModelLoadOptions {
                prefer_expert_offload: false,
                keep_quantized,
            },
        )
        .expect("checkpoint loads")
    };
    // The unpacked model's greedy continuation of the prompt, and its logits
    // over the whole text. It is dropped before the packed model loads, so the
    // two never share memory.
    let mut tokens = prompt.clone();
    let dense_rows = {
        let mut dense = load(false);
        // Uncached, so the hybrid architectures (which need a recurrent cache
        // beside the attention one) take the same path as the rest.
        for _ in 0..CONTINUATION {
            let logits = dense.forward(&ids(&tokens), None).expect("dense forward");
            let (seq, vocab) = (logits.dim(1), logits.dim(2));
            let last = logits.slice(&[0, seq - 1, 0], &[1, seq, vocab]);
            let next = argmax(&rows(&last.reshape(&[1, 1, vocab])).remove(0)) as u32;
            tokens.push(next);
        }
        rows(&dense.forward(&ids(&tokens), None).expect("dense forward"))
    };
    let packed_rows = {
        let mut packed = load(true);
        rows(&packed.forward(&ids(&tokens), None).expect("packed forward"))
    };
    pmetal_runtime::check_unobserved_error().expect("no runtime error");

    let mut agree = 0usize;
    let (mut kl_sum, mut kl_max) = (0.0f64, 0.0f64);
    let mut flips = Vec::new();
    let positions = prompt.len() - 1..tokens.len() - 1;
    for p in positions.clone() {
        let (d, q) = (log_softmax(&dense_rows[p]), log_softmax(&packed_rows[p]));
        let kl: f64 = d
            .iter()
            .zip(&q)
            .map(|(&ld, &lq)| ld.exp() * (ld - lq))
            .sum();
        kl_sum += kl;
        kl_max = kl_max.max(kl);
        let (a, b) = (argmax(&dense_rows[p]), argmax(&packed_rows[p]));
        if a == b {
            agree += 1;
        } else {
            // How far ahead the unpacked model's choice was, in logits.
            flips.push((p, dense_rows[p][a] - dense_rows[p][b]));
        }
    }
    let n = positions.len();
    println!(
        "{n} positions: argmax agrees on {agree} ({:.1}%), KL(dense || packed) mean {:.2e}, max {:.2e}",
        100.0 * agree as f64 / n as f64,
        kl_sum / n as f64,
        kl_max
    );
    for (p, gap) in &flips {
        println!(
            "  position {p}: the unpacked model's top token leads the packed one's by {gap:.3} logits"
        );
    }
    // The bounds are for a mixture of experts, where a router's near-tie sends
    // a token to a different expert and the two paths part further than a dense
    // model's do; the dense models measured stay under 1e-2 mean KL.
    assert!(
        agree * 100 >= n * 85,
        "argmax agrees on only {agree} of {n}"
    );
    assert!(
        kl_sum / (n as f64) < 0.1,
        "mean KL {:.3e}",
        kl_sum / n as f64
    );
}
