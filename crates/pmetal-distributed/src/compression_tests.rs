use super::*;
use crate::ring::RingBackend;
use crate::testing::{fresh_security, tcp_ring};

fn noise(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 4.0
        })
        .collect()
}

fn all_strategies() -> Vec<CompressionStrategy> {
    vec![
        CompressionStrategy::TopK { ratio: 0.1 },
        CompressionStrategy::Random { probability: 0.3 },
        CompressionStrategy::Quantize(QuantizationType::FP16),
        CompressionStrategy::Quantize(QuantizationType::BF16),
        CompressionStrategy::Quantize(QuantizationType::INT8),
        CompressionStrategy::Quantize(QuantizationType::OneBit),
    ]
}

#[test]
fn test_no_compression() {
    let mut compressor = GradientCompressor::new(CompressionStrategy::None, false);
    let grads = vec![1.0, 2.0, 3.0, 4.0];

    let compressed = compressor.compress(&grads);
    let decompressed = compressor.decompress(&compressed);

    assert_eq!(grads, decompressed);
    assert!((compressed.compression_ratio() - 1.0).abs() < 0.01);
}

#[test]
fn test_topk() {
    let mut compressor = GradientCompressor::new(CompressionStrategy::TopK { ratio: 0.5 }, false);
    let grads = vec![1.0, 4.0, 2.0, 3.0];

    let compressed = compressor.compress(&grads);
    let decompressed = compressor.decompress(&compressed);

    // Top 50% should be 4.0 and 3.0
    assert_eq!(decompressed, vec![0.0, 4.0, 0.0, 3.0]);
    assert!(compressed.compression_ratio() >= 1.0);
}

/// Top-k picks the k largest magnitudes, ties to the lower index, exactly as
/// a full sort would.
#[test]
fn topk_matches_a_full_sort() {
    for (seed, len, ratio) in [(1, 1, 0.5), (2, 10, 0.3), (3, 1000, 0.01), (4, 1000, 0.5)] {
        let mut grads = noise(seed, len);
        // Plenty of exact ties.
        for g in grads.iter_mut().step_by(3) {
            *g = g.signum();
        }
        let k = ((len as f64 * f64::from(ratio)) as usize).max(1).min(len);
        let mut order: Vec<usize> = (0..len).collect();
        order.sort_by(|&a, &b| grads[b].abs().total_cmp(&grads[a].abs()).then(a.cmp(&b)));
        let mut want: Vec<u32> = order[..k].iter().map(|&i| i as u32).collect();
        want.sort_unstable();

        let mut compressor = GradientCompressor::new(CompressionStrategy::TopK { ratio }, false);
        match compressor.compress(&grads).data {
            CompressedData::Sparse { indices, values } => {
                assert_eq!(indices, want, "len {len} ratio {ratio}");
                assert!(
                    indices
                        .iter()
                        .zip(&values)
                        .all(|(&i, &v)| v == grads[i as usize])
                );
            }
            other => panic!("expected sparse, got {other:?}"),
        }
    }
}

#[test]
fn test_fp16_quantization() {
    let mut compressor =
        GradientCompressor::new(CompressionStrategy::Quantize(QuantizationType::FP16), false);
    let grads = vec![1.0, 2.5, 3.125, 4.0];

    let compressed = compressor.compress(&grads);
    let decompressed = compressor.decompress(&compressed);

    for (orig, decomp) in grads.iter().zip(decompressed.iter()) {
        assert!((orig - decomp).abs() < 0.01);
    }
    assert!((compressed.compression_ratio() - 2.0).abs() < 0.1);
}

#[test]
fn fp16_saturates_instead_of_overflowing_to_infinity() {
    let mut compressor =
        GradientCompressor::new(CompressionStrategy::Quantize(QuantizationType::FP16), false);
    let out = compressor.compress(&[1.0e6, -1.0e6, 3.0]).to_dense();
    assert_eq!(out, vec![65504.0, -65504.0, 3.0]);
}

#[test]
fn test_int8_quantization() {
    let mut compressor =
        GradientCompressor::new(CompressionStrategy::Quantize(QuantizationType::INT8), false);
    let grads = vec![1.0, 2.0, 3.0, 4.0];

    let compressed = compressor.compress(&grads);
    let decompressed = compressor.decompress(&compressed);

    for (orig, decomp) in grads.iter().zip(decompressed.iter()) {
        assert!((orig - decomp).abs() < 0.04);
    }
    assert!(compressed.compression_ratio() >= 2.0);
}

/// Rounding to nearest, not truncating toward zero, so the error is at most
/// half a step and has no sign.
#[test]
fn int8_rounds_to_nearest() {
    let mut compressor =
        GradientCompressor::new(CompressionStrategy::Quantize(QuantizationType::INT8), false);
    let g = vec![127.0, 0.6, -0.6, 0.4, -0.4];
    let out = compressor.compress(&g).to_dense();
    assert_eq!(out, vec![127.0, 1.0, -1.0, 0.0, 0.0]);
}

/// A NaN or infinity must reach the other ranks rather than being quietly
/// replaced by a number.
#[test]
fn a_non_finite_gradient_poisons_int8_rather_than_vanishing() {
    let mut compressor =
        GradientCompressor::new(CompressionStrategy::Quantize(QuantizationType::INT8), false);
    let out = compressor.compress(&[1.0, f32::NAN, 2.0]).to_dense();
    assert!(out.iter().all(|x| x.is_nan()), "{out:?}");
}

/// The error-feedback invariant, for every method: what has been transmitted
/// plus what is still owed equals what was fed in.
#[test]
fn error_feedback_conserves_the_gradient_sum_for_every_method() {
    for strategy in all_strategies() {
        let len = 200;
        let mut compressor = GradientCompressor::new(strategy.clone(), true);
        let mut fed = vec![0f64; len];
        let mut sent = vec![0f64; len];
        for step in 0..50 {
            let g = noise(step + 1, len);
            let packet = compressor.compress(&g);
            for i in 0..len {
                fed[i] += f64::from(g[i]);
            }
            for (s, x) in sent.iter_mut().zip(compressor.decompress(&packet)) {
                *s += f64::from(x);
            }
        }
        let residual = compressor.residual().expect("residual kept");
        for i in 0..len {
            let owed = f64::from(residual[i]);
            assert!(
                (fed[i] - sent[i] - owed).abs() < 1e-3,
                "{strategy:?} element {i}: fed {} sent {} owed {}",
                fed[i],
                sent[i],
                owed
            );
        }
    }
}

/// Dropping the residual of a differently sized tensor is right; adding it
/// to this one would be wrong.
#[test]
fn a_residual_for_another_length_is_discarded() {
    let mut compressor = GradientCompressor::new(CompressionStrategy::TopK { ratio: 0.25 }, true);
    compressor.compress(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(compressor.residual().unwrap().len(), 4);
    let out = compressor.compress(&[5.0, 6.0]);
    assert_eq!(out.original_size, 2);
    assert_eq!(compressor.residual().unwrap().len(), 2);
}

#[test]
fn bad_fractions_cannot_drop_the_whole_gradient() {
    for bad in [0.0, -1.0, f32::NAN, f32::INFINITY, 7.0] {
        let mut compressor =
            GradientCompressor::new(CompressionStrategy::TopK { ratio: bad }, false);
        let out = compressor.compress(&[1.0, 2.0, 3.0]);
        let n = out.to_dense().iter().filter(|&&x| x != 0.0).count();
        assert!(n >= 1, "ratio {bad} kept nothing");
    }
}

#[test]
fn random_masks_follow_the_seed() {
    let g = vec![1.0f32; 400];
    let mask = |seed| {
        let mut c = GradientCompressor::with_seed(
            CompressionStrategy::Random { probability: 0.3 },
            false,
            seed,
        );
        match c.compress(&g).data {
            CompressedData::Sparse { indices, .. } => indices,
            _ => unreachable!(),
        }
    };
    assert_eq!(mask(5), mask(5));
    assert_ne!(mask(5), mask(6));
    let kept = mask(5).len() as f64 / 400.0;
    assert!((kept - 0.3).abs() < 0.08, "kept fraction {kept}");
}

#[test]
fn every_format_survives_a_serialization_round_trip() {
    let g = noise(11, 77);
    for strategy in std::iter::once(CompressionStrategy::None).chain(all_strategies()) {
        let mut compressor = GradientCompressor::new(strategy.clone(), false);
        let packet = compressor.compress(&g);
        let bytes = serialize_compressed(&packet);
        let restored = deserialize_compressed(&bytes).unwrap_or_else(|| panic!("{strategy:?}"));
        assert_eq!(restored.original_size, 77);
        assert_eq!(
            restored.to_dense(),
            packet.to_dense(),
            "{strategy:?} changed in transit"
        );
    }
}

#[test]
fn malformed_packets_are_rejected_without_panicking_or_allocating() {
    let g = noise(12, 40);
    let mut compressor = GradientCompressor::new(CompressionStrategy::TopK { ratio: 0.5 }, false);
    let good = serialize_compressed(&compressor.compress(&g));
    assert!(deserialize_compressed(&good).is_some());

    // Every truncation.
    for cut in 0..good.len() {
        assert!(deserialize_compressed(&good[..cut]).is_none(), "cut {cut}");
    }
    // Trailing bytes.
    let mut longer = good.clone();
    longer.push(0);
    assert!(deserialize_compressed(&longer).is_none());
    // Wrong magic, wrong version, unknown tag.
    for (at, val) in [(0, 0), (1, 99), (2, 200)] {
        let mut bad = good.clone();
        bad[at] = val;
        assert!(deserialize_compressed(&bad).is_none(), "byte {at}");
    }
    // A header that claims 2^63 elements over a tiny body.
    for tag in 0..=5u8 {
        let mut bad = vec![WIRE_MAGIC, WIRE_VERSION, tag];
        bad.extend_from_slice(&(1u64 << 63).to_le_bytes());
        bad.extend_from_slice(&[0u8; 16]);
        assert!(deserialize_compressed(&bad).is_none(), "tag {tag}");
    }
    // A sparse packet with more entries than elements, or an index out of range.
    let mut bad = vec![WIRE_MAGIC, WIRE_VERSION, 1];
    bad.extend_from_slice(&2u64.to_le_bytes()); // 2 elements
    bad.extend_from_slice(&1u64.to_le_bytes()); // 1 entry
    bad.extend_from_slice(&5u32.to_le_bytes()); // index 5 >= 2
    bad.extend_from_slice(&1.0f32.to_le_bytes());
    assert!(deserialize_compressed(&bad).is_none());
}

#[test]
fn sparse_decompress_ignores_out_of_range_indices() {
    let compressed = CompressedGradient {
        original_size: 2,
        strategy: CompressionStrategy::TopK { ratio: 0.5 },
        data: CompressedData::Sparse {
            indices: vec![99],
            values: vec![1.0],
        },
    };
    assert_eq!(compressed.to_dense(), vec![0.0, 0.0]);
}

/// Ranks that compress their gradients and exchange the packets end up with
/// identical results equal to the mean of what each rank's packet decodes to.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compressed_all_reduce_agrees_across_ranks_and_matches_the_decoded_mean() {
    for strategy in all_strategies() {
        let n = 3;
        let len = 257;
        let inputs: Vec<Vec<f32>> = (0..n).map(|r| noise(100 + r as u64, len)).collect();

        // What the packets decode to, computed independently of the ring.
        let mut want = vec![0f32; len];
        for (rank, input) in inputs.iter().enumerate() {
            let mut c = GradientCompressor::with_seed(strategy.clone(), true, rank as u64);
            c.compress(input).add_to(&mut want);
        }
        want.iter_mut().for_each(|x| *x /= n as f32);

        let ring = tcp_ring(n, fresh_security()).await;
        let mut tasks = Vec::new();
        for (rank, collective) in ring.into_iter().enumerate() {
            let backend = RingBackend::from_collective(collective);
            let strategy = strategy.clone();
            let mut data = inputs[rank].clone();
            tasks.push(tokio::spawn(async move {
                let mut c = GradientCompressor::with_seed(strategy, true, rank as u64);
                all_reduce_compressed(&backend, &mut c, &mut data, ReduceOp::Mean)
                    .await
                    .expect("compressed all_reduce");
                data
            }));
        }
        for (rank, task) in tasks.into_iter().enumerate() {
            let got = task.await.expect("task");
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{strategy:?}: rank {rank} differs from the decoded mean"
            );
        }
    }
}

/// The compressed path moves less than the ring would for a sparsifier.
#[test]
fn a_sparse_packet_is_smaller_than_the_gradient() {
    let mut compressor = GradientCompressor::new(CompressionStrategy::TopK { ratio: 0.01 }, true);
    let packet = compressor.compress(&noise(3, 10_000));
    assert!(serialize_compressed(&packet).len() * 10 < 10_000 * 4);
}
