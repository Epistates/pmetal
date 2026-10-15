//! What a user of gradient compression can rely on, through the public API.
//!
//! Error feedback (Seide et al. 2014; Lin et al. 2018) promises that nothing
//! is lost, only delayed: what a step fails to transmit is carried into the
//! next one, so over many steps the transmitted total equals the true total.

use pmetal_distributed::{CompressionStrategy, GradientCompressor, QuantizationType};

fn values(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12345);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32 - 0.5) * 3.0
        })
        .collect()
}

/// Compress `first`, then keep compressing zeros so error feedback can flush
/// its residual, and return everything that was transmitted, summed.
fn transmitted_total(
    strategy: CompressionStrategy,
    error_feedback: bool,
    first: &[f32],
    flush_steps: usize,
) -> Vec<f32> {
    let mut compressor = GradientCompressor::new(strategy, error_feedback);
    let mut total = vec![0f32; first.len()];
    let zeros = vec![0f32; first.len()];
    for step in 0..=flush_steps {
        let input = if step == 0 { first } else { &zeros };
        let packet = compressor.compress(input);
        let sent = compressor.decompress(&packet);
        for (t, s) in total.iter_mut().zip(&sent) {
            *t += s;
        }
    }
    total
}

#[test]
fn error_feedback_recovers_what_quantization_dropped() {
    let g = values(256, 7);
    for qtype in [
        QuantizationType::FP16,
        QuantizationType::BF16,
        QuantizationType::INT8,
    ] {
        let strategy = || CompressionStrategy::Quantize(qtype);
        let without = transmitted_total(strategy(), false, &g, 40);
        let with = transmitted_total(strategy(), true, &g, 40);
        let err = |total: &[f32]| {
            total
                .iter()
                .zip(&g)
                .map(|(t, g)| (t - g).abs())
                .fold(0f32, f32::max)
        };
        // Without feedback the quantization error stays; with it, it is
        // transmitted a step later and the total converges on the truth.
        assert!(
            err(&with) < err(&without) * 0.05 + 1e-6,
            "{qtype:?}: error with feedback {} vs without {}",
            err(&with),
            err(&without)
        );
    }
}

#[test]
fn sparsification_with_error_feedback_neither_loses_nor_inflates() {
    for strategy in [
        CompressionStrategy::TopK { ratio: 0.25 },
        CompressionStrategy::Random { probability: 0.25 },
    ] {
        let g = vec![1.0f32; 64];
        let steps = 4000;
        let mut compressor = GradientCompressor::new(strategy.clone(), true);
        let mut total = vec![0f64; g.len()];
        for _ in 0..steps {
            let packet = compressor.compress(&g);
            let sent = compressor.decompress(&packet);
            for (t, s) in total.iter_mut().zip(&sent) {
                *t += f64::from(*s);
            }
        }
        // Everything fed in was `steps * 1.0` per coordinate; what has been
        // transmitted plus what is still owed must equal it, so the average
        // per step is 1 to within the (bounded) outstanding residual.
        let mean = total.iter().sum::<f64>() / (total.len() * steps) as f64;
        assert!(
            (mean - 1.0).abs() < 0.05,
            "{strategy:?}: transmitted {mean:.3}x the gradient on average"
        );
    }
}

#[test]
fn random_sparsification_without_feedback_is_unbiased() {
    let g = values(32, 3);
    let mut compressor =
        GradientCompressor::new(CompressionStrategy::Random { probability: 0.25 }, false);
    let steps = 20_000;
    let mut mean = vec![0f64; g.len()];
    for _ in 0..steps {
        let packet = compressor.compress(&g);
        for (m, s) in mean.iter_mut().zip(compressor.decompress(&packet)) {
            *m += f64::from(s) / steps as f64;
        }
    }
    for (i, (m, g)) in mean.iter().zip(&g).enumerate() {
        assert!(
            (m - f64::from(*g)).abs() < 0.1 * f64::from(g.abs()).max(0.2),
            "element {i}: mean {m:.4} vs {g:.4}"
        );
    }
}
