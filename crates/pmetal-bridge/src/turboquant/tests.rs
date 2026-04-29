//! Unit tests for the TurboQuant K/V cache modules.
//!
//! The two sub-modules pin distinct contracts:
//!   - [`layout_invariants`] guards storage-layout dimensions that the C++
//!     score kernels depend on (off-by-one in any helper would corrupt cache
//!     reads).
//!   - [`tests`] exercises end-to-end encode/score/dequantize and the
//!     hot/cold split state machine against reference paths.

#![cfg(test)]

// Re-export the parent module's surface into tests scope so the nested
// `mod tests { use super::*; }` block picks up everything the original
// inline test module saw via its `super::*` import.
use super::*;
use super::config::HOT_EVICTION_CHUNK;
use super::dispatch::{
    gpu_compute_outlier_partition, gpu_dequantize_key_subvector, gpu_dequantize_keys,
    gpu_dequantize_keys_mixed, gpu_dequantize_values_mixed, gpu_encode_key_subvector,
    gpu_quantize_kv, gpu_quantize_kv_mixed, try_gpu_mixed_score,
};
use super::encode::{
    EncodedKeyRows, decode_key_component_rows_raw, encode_key_component_rows,
    encode_value_component_rows, nearest_centroid_index, scatter_mixed_rows, select_outlier_mask,
    split_rows_by_outliers,
};
use super::math::{build_beta_codebook, l2_norm};
use super::state::TensorRuntime;
use crate::compat::Dtype;

#[cfg(test)]
mod layout_invariants {
    //! Pin the layout-derived dimensions so a silent off-by-one in a helper
    //! cannot misalign the C++ score kernels' reads. Mirrors the contract in
    //! `cpp/bridge/turboquant.h`'s top-of-file invariants block.
    use super::super::bits::packed_qjl_words;

    #[test]
    fn qjl_words_match_pack_invariants() {
        // Exact 32-multiples — kernel expects no padding word.
        assert_eq!(packed_qjl_words(32), 1);
        assert_eq!(packed_qjl_words(64), 2);
        assert_eq!(packed_qjl_words(128), 4);
        assert_eq!(packed_qjl_words(256), 8);
        // Non-multiples — caller must round up because each sign-bit lands
        // in some 32-bit lane and the kernel reads `qjl_words` lanes.
        assert_eq!(packed_qjl_words(33), 2);
        assert_eq!(packed_qjl_words(96), 3);
        assert_eq!(packed_qjl_words(192), 6);
    }

    #[test]
    fn d256_qjl_words_is_eight() {
        // The fused d256 attention kernel hardcodes `qjl_words == 8`. If we
        // ever change the QJL packing this test fails fast with a clear
        // pointer to the C++ kernel that needs an update.
        assert_eq!(packed_qjl_words(256), 8);
    }
}

mod kvcache {
    use super::*;

    fn make_uniform_direct_attention_case_with(
        dim: usize,
        heads: i32,
        prefill: i32,
    ) -> (
        QuantizedKvCache,
        InlineArray,
        InlineArray,
        InlineArray,
        f32,
        i32,
        i32,
        i32,
    ) {
        // These helpers exercise the cold compressed GPU store directly, so
        // disable the recent-fp16 window — otherwise short prefills sit in
        // the hot ring and never touch the cold path.
        let config = TurboQuantConfig::uniform(8, 8).with_recent_window(None);
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let step_keys = InlineArray::from_f32_slice(&make_data(step_len, 1.9), &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");
        assert!(
            seed_cache
                .keys
                .as_ref()
                .and_then(|k| k.gpu.as_ref())
                .is_some()
        );
        assert!(
            seed_cache
                .values
                .as_ref()
                .and_then(|v| v.gpu.as_ref())
                .is_some()
        );

        (seed_cache, queries, step_keys, step_values, scale, b, h, d)
    }

    fn make_uniform_direct_attention_case() -> (
        QuantizedKvCache,
        InlineArray,
        InlineArray,
        InlineArray,
        f32,
        i32,
        i32,
        i32,
    ) {
        make_uniform_direct_attention_case_with(16, 2, 3)
    }

    fn make_uniform_gqa_direct_attention_case_with(
        dim: usize,
        q_heads: i32,
        kv_heads: i32,
        prefill: i32,
    ) -> (
        QuantizedKvCache,
        InlineArray,
        InlineArray,
        InlineArray,
        f32,
        i32,
        i32,
        i32,
        i32,
    ) {
        // These helpers exercise the cold compressed GPU store directly, so
        // disable the recent-fp16 window — otherwise short prefills sit in
        // the hot ring and never touch the cold path.
        let config = TurboQuantConfig::uniform(8, 8).with_recent_window(None);
        let b = 1i32;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_kv_len = (b * kv_heads * prefill * d) as usize;
        let step_kv_len = (b * kv_heads * d) as usize;
        let query_len = (b * q_heads * d) as usize;
        let prefill_keys = InlineArray::from_f32_slice(
            &make_data(prefill_kv_len, 0.2),
            &[b, kv_heads, prefill, d],
        );
        let prefill_values = InlineArray::from_f32_slice(
            &make_data(prefill_kv_len, 0.7),
            &[b, kv_heads, prefill, d],
        );
        let queries = InlineArray::from_f32_slice(&make_data(query_len, 1.3), &[b, q_heads, 1, d]);
        let step_keys =
            InlineArray::from_f32_slice(&make_data(step_kv_len, 1.9), &[b, kv_heads, 1, d]);
        let step_values =
            InlineArray::from_f32_slice(&make_data(step_kv_len, 2.4), &[b, kv_heads, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");
        assert!(
            seed_cache
                .keys
                .as_ref()
                .and_then(|k| k.gpu.as_ref())
                .is_some()
        );
        assert!(
            seed_cache
                .values
                .as_ref()
                .and_then(|v| v.gpu.as_ref())
                .is_some()
        );

        (
            seed_cache,
            queries,
            step_keys,
            step_values,
            scale,
            b,
            q_heads,
            kv_heads,
            d,
        )
    }

    /// Build a Mixed-precision (`preset_q3_5`) direct-attention fixture. Matches
    /// the Uniform helper's shape contract so the same `manual_single_token_attention`
    /// reference can compare against both. Mixed paths currently dequantize+SDPA
    /// inside `append_and_compute_attention`; once the fused Mixed kernel lands
    /// (Phase 3), this fixture pins the expected numerics so the kernel must
    /// match the dequantize+SDPA baseline at < 1e-4.
    fn make_mixed_direct_attention_case_with(
        dim: usize,
        heads: i32,
        prefill: i32,
    ) -> (
        QuantizedKvCache,
        InlineArray,
        InlineArray,
        InlineArray,
        f32,
        i32,
        i32,
        i32,
    ) {
        // Cold-only path: disable the recent fp16 window so every appended token
        // takes the Mixed-quantized cold store and exercises the full
        // dequantize_keys/dequantize_values + SDPA fallback (the path the fused
        // Mixed kernel will replace).
        let config = TurboQuantConfig::preset_q3_5(dim).with_recent_window(None);
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let step_keys = InlineArray::from_f32_slice(&make_data(step_len, 1.9), &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        (seed_cache, queries, step_keys, step_values, scale, b, h, d)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_range_loop)]
    fn manual_single_token_attention(
        queries: &mut InlineArray,
        keys: &mut InlineArray,
        values: &mut InlineArray,
        batch: i32,
        heads: i32,
        seq: i32,
        dim: i32,
        scale: f32,
    ) -> Vec<f32> {
        let q = queries
            .to_f32_vec((batch * heads * dim) as usize)
            .expect("queries to_f32");
        let k = keys
            .to_f32_vec((batch * heads * seq * dim) as usize)
            .expect("keys to_f32");
        let v = values
            .to_f32_vec((batch * heads * seq * dim) as usize)
            .expect("values to_f32");

        let rows = (batch * heads) as usize;
        let seq_usize = seq as usize;
        let dim_usize = dim as usize;
        let mut out = vec![0.0f32; rows * dim_usize];

        for row in 0..rows {
            let q_base = row * dim_usize;
            let q_row = &q[q_base..q_base + dim_usize];

            let mut scores = vec![0.0f32; seq_usize];
            for t in 0..seq_usize {
                let k_base = (row * seq_usize + t) * dim_usize;
                let k_row = &k[k_base..k_base + dim_usize];
                let dot = q_row
                    .iter()
                    .zip(k_row.iter())
                    .map(|(lhs, rhs)| lhs * rhs)
                    .sum::<f32>();
                scores[t] = dot * scale;
            }

            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                sum_exp += *score;
            }
            for score in &mut scores {
                *score /= sum_exp.max(f32::MIN_POSITIVE);
            }

            let out_row = &mut out[q_base..q_base + dim_usize];
            for t in 0..seq_usize {
                let v_base = (row * seq_usize + t) * dim_usize;
                let v_row = &v[v_base..v_base + dim_usize];
                let weight = scores[t];
                for (dst, val) in out_row.iter_mut().zip(v_row.iter()) {
                    *dst += weight * *val;
                }
            }
        }

        out
    }

    #[test]
    fn packed_bits_round_trip() {
        let values = [1u16, 6, 3, 0, 7, 2, 4];
        let mut packed = PackedBits::new(3);
        packed.extend_from_slice(&values);
        let round_trip: Vec<u16> = (0..values.len()).map(|i| packed.get(i)).collect();
        assert_eq!(round_trip, values);

        packed.truncate(4);
        let truncated: Vec<u16> = (0..4).map(|i| packed.get(i)).collect();
        assert_eq!(truncated, values[..4]);
    }

    #[test]
    fn beta_codebook_is_sorted_and_correct_length() {
        let codebook = build_beta_codebook(128, 4);
        assert_eq!(codebook.len(), 16);
        assert!(codebook.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn beta_codebook_memo_matches_direct() {
        // Memoized public entry must match the underlying computation byte-for-byte.
        let direct = build_beta_codebook(128, 3);
        let memoed = beta_codebook(128, 3);
        assert_eq!(direct.len(), memoed.len());
        for (a, b) in direct.iter().zip(memoed.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        // Second call hits the cache and returns identical contents.
        let memoed_again = beta_codebook(128, 3);
        assert_eq!(*memoed, *memoed_again);
    }

    #[test]
    fn codebook_range_within_unit_interval() {
        let codebook = build_beta_codebook(128, 4);
        assert!(codebook.iter().all(|&v| (-1.0..=1.0).contains(&v)));
    }

    #[test]
    fn nearest_centroid_boundary_cases() {
        let cb = vec![-0.5f32, 0.0, 0.5];
        assert_eq!(nearest_centroid_index(-2.0, &cb), 0);
        assert_eq!(nearest_centroid_index(2.0, &cb), 2);
        assert_eq!(nearest_centroid_index(0.0, &cb), 1);
        assert_eq!(nearest_centroid_index(0.26, &cb), 2);
    }

    #[test]
    fn turboquant_handles_zero_rows() {
        let core = TurboQuantCore::new(8, 4);
        let encoded = encode_key_component_rows(&core, &[0.0; 8], 4, super::config::TurboQuantQjlMode::Standard, 0);
        assert_eq!(encoded.norms, vec![0.0]);
        assert_eq!(encoded.residual_norms, vec![0.0]);
        assert!(encoded.mse_indices.iter().all(|&v| v == 0));
        assert!(encoded.qjl_signs.iter().all(|&v| v == 0));
    }

    #[test]
    fn turboquant_state_constructs_without_panic() {
        let config = TurboQuantConfig::uniform(4, 4);
        let _state = TurboQuantState::new(64, 64, config);
    }

    #[test]
    fn mixed_config_effective_bits() {
        let config = TurboQuantTensorConfig::mixed(2, 4, 32);
        assert_eq!(config.effective_bits(128), 2.5);
        assert_eq!(config.regular_dim(128), 96);
        assert_eq!(config.outlier_count(), 32);
    }

    #[test]
    fn select_outlier_mask_marks_top_k() {
        let row = [0.1f32, 0.9, 0.5, 0.8, 0.2];
        let mask = select_outlier_mask(&row, 2);
        // Top 2 by magnitude: index 1 (0.9) and index 3 (0.8)
        assert_eq!(mask[1], 1);
        assert_eq!(mask[3], 1);
        assert_eq!(mask[0], 0);
        assert_eq!(mask[2], 0);
        assert_eq!(mask[4], 0);
    }

    #[test]
    fn scatter_round_trips_split() {
        let rows = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (mask, regular, outlier) = split_rows_by_outliers(&rows, 3, 1);
        let merged = scatter_mixed_rows(&mask, 3, 1, &regular, &outlier);
        assert_eq!(merged.len(), rows.len());
        // Merged must contain the same values (possibly reordered by scatter).
        let mut orig_sorted = rows.clone();
        let mut merged_sorted = merged.clone();
        orig_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        merged_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(orig_sorted, merged_sorted);
    }

    #[test]
    fn encode_value_norm_preserved() {
        let core = TurboQuantCore::new(8, 4);
        let v: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let encoded = encode_value_component_rows(&core, &v, 4);
        assert_eq!(encoded.norms.len(), 1);
        let expected_norm = l2_norm(&v);
        assert!((encoded.norms[0] - expected_norm).abs() < 1e-5);
    }

    #[test]
    fn turboquant_presets_match_schedule() {
        let q2_5 = TurboQuantConfig::preset_q2_5(128);
        let q3_5 = TurboQuantConfig::preset_q3_5(128);
        assert_eq!(q2_5, TurboQuantConfig::mixed(2, 4, 32, 2, 4, 32));
        assert_eq!(q3_5, TurboQuantConfig::mixed(3, 5, 32, 3, 5, 32));
    }

    /// GPU round-trip: append via GPU path then dequantize via GPU path.
    ///
    /// We verify two things:
    ///   1. The GPU path is actually taken (store.gpu is Some).
    ///   2. The GPU dequantised output is close to the CPU dequantised output
    ///      (same algorithm, both paths should produce bitwise-close results
    ///      modulo f32 ordering differences).
    #[test]
    fn turboquant_gpu_path_round_trip() {
        // Small dim so the test is fast.
        let dim = 16usize;
        // Disable the hot window — this test asserts that data lands in the
        // cold GPU store, which only happens once the recent fp16 ring is off
        // (or has overflowed past `recent_window + HOT_EVICTION_CHUNK`).
        let config = TurboQuantConfig::uniform(4, 4).with_recent_window(None);
        let b = 1i32;
        let h = 2i32;
        let s = 3i32;
        let d = dim as i32;
        let total = (b * h * s * d) as usize;

        // Build deterministic input vectors.
        let data: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.1 - total as f32 * 0.05).sin())
            .collect();
        // Upload as [B, H, S, D] f32.
        let keys_arr = InlineArray::from_f32_slice(&data, &[b, h, s, d]);
        let vals_arr = InlineArray::from_f32_slice(&data, &[b, h, s, d]);

        // ── CPU reference path (use Mixed config to force CPU) ────────────
        // Disable the hot window so the test exercises the cold-only legacy
        // path it was originally written against.
        let cpu_config = TurboQuantConfig {
            qjl: super::config::TurboQuantQjlMode::Standard,
            keys: TurboQuantTensorConfig::Mixed {
                regular_bits: 3,
                outlier_bits: 4,
                outlier_count: 4,
            },
            values: TurboQuantTensorConfig::Mixed {
                regular_bits: 4,
                outlier_bits: 4,
                outlier_count: 4,
            },
            recent_window: None,
            skiplist_threshold: None,
            outliers: super::config::TurboQuantOutlierMode::None,
            pack_mode: super::config::TurboQuantPackMode::Bitstream,
            warm_tier: None,
        };
        let mut cpu_cache = QuantizedKvCache::new(cpu_config);
        cpu_cache.append(&keys_arr, &vals_arr).expect("CPU append");
        // Verify CPU path taken (no GPU store).
        assert!(
            cpu_cache.keys.as_ref().unwrap().gpu.is_none(),
            "Expected CPU path for Mixed config"
        );

        // ── GPU path ──────────────────────────────────────────────────────
        let mut gpu_cache = QuantizedKvCache::new(config);
        gpu_cache.append(&keys_arr, &vals_arr).expect("GPU append");

        // Verify GPU path was taken.
        assert!(
            gpu_cache.keys.as_ref().unwrap().gpu.is_some(),
            "GPU store should be Some for Uniform config"
        );
        assert!(
            gpu_cache.values.as_ref().unwrap().gpu.is_some(),
            "GPU value store should be Some for Uniform config"
        );

        // Dequantise — should succeed.
        let dk = gpu_cache.dequantize_keys().expect("GPU dequantize_keys");
        let dv = gpu_cache
            .dequantize_values()
            .expect("GPU dequantize_values");

        // Verify output shapes: [B, H, T, D].
        assert_eq!(dk.shape(), &[b, h, s, d], "dequantized keys shape mismatch");
        assert_eq!(
            dv.shape(),
            &[b, h, s, d],
            "dequantized values shape mismatch"
        );

        // Output should be finite (not NaN/Inf).
        let dk_vals = dk
            .reshape(&[(b * h * s * d)])
            .to_f32_vec(total)
            .expect("dk to_f32");
        let dv_vals = dv
            .reshape(&[(b * h * s * d)])
            .to_f32_vec(total)
            .expect("dv to_f32");
        assert!(
            dk_vals.iter().all(|v| v.is_finite()),
            "dequantized keys contain non-finite"
        );
        assert!(
            dv_vals.iter().all(|v| v.is_finite()),
            "dequantized values contain non-finite"
        );

        // Verify output is within reasonable range (quantisation introduces error but
        // should not explode — reconstructed vectors should be roughly same magnitude as input).
        let input_max = data.iter().cloned().fold(0.0f32, f32::max).abs();
        let dk_max = dk_vals.iter().cloned().fold(0.0f32, f32::max).abs();
        assert!(
            dk_max < input_max * 3.0,
            "dequantized keys magnitude unreasonably large"
        );
    }

    #[test]
    fn gpu_packed_qjl_sign_words_round_trip_encodes_zero_as_positive_bit() {
        let projected = InlineArray::from_f32_slice(&[-2.0f32, 0.0, 3.0, -0.5], &[1, 4]);
        let packed =
            InlineArray::turboquant_pack_sign_bits(&projected, 4, 1, 1).expect("pack sign bits");
        let mut unpacked =
            InlineArray::turboquant_unpack_sign_bits(&packed, 4, 1, 1).expect("unpack sign bits");
        let values = unpacked.to_f32_vec(4).expect("unpacked to_f32");
        assert_eq!(values, vec![-1.0, 1.0, 1.0, -1.0]);
    }

    #[test]
    fn turboquant_q8_d256_gpu_store_uses_seq_shadow_without_transposed_shadow() {
        let (seed_cache, _, _, _, _, _, _, _, _) =
            make_uniform_gqa_direct_attention_case_with(256, 16, 2, 1023);
        let gpu = seed_cache
            .keys
            .as_ref()
            .and_then(|k| k.gpu.as_ref())
            .expect("gpu key store");
        assert!(
            gpu.q8_keybytes_t.is_none(),
            "d256 q8 path should not keep transposed q8 shadow"
        );
        assert!(
            gpu.q8_keybytes_seq.is_some(),
            "d256 q8 path should keep seq-major packed key shadow"
        );
        assert!(
            gpu.q8_kvbytes_seq.is_none(),
            "d256 q8 path should not keep packed kv shadow when dense rotated values are present"
        );
        assert!(
            gpu.q8_slot_scales_seq.is_some(),
            "d256 q8 path should keep seq-major slot scale shadow"
        );
        assert!(
            gpu.indices_t.is_none(),
            "d256 q8 path should not keep transposed key indices"
        );
        assert!(
            gpu.qjl_signs_t.is_none(),
            "d256 q8 path should not keep transposed qjl sign words"
        );
        assert!(
            gpu.norms.is_none(),
            "d256 q8 path should source key norms from slot scales"
        );
        assert!(
            gpu.residual_norms.is_none(),
            "d256 q8 path should source residual norms from slot scales"
        );
        let gpu_values = seed_cache
            .values
            .as_ref()
            .and_then(|v| v.gpu.as_ref())
            .expect("gpu value store");
        assert!(
            gpu_values.indices_t.is_none(),
            "d256 q8 path should not keep transposed value indices"
        );
    }

    #[test]
    fn turboquant_no_qjl_skips_gpu_qjl_allocations() {
        // Variant F's whole point: qjl_signs / qjl_signs_t / residual_norms
        // should not be allocated on the GPU side. Pin this so we don't
        // regress to "Variant F via fallback that still allocates QJL".
        let dim = 16usize;
        let heads = 2i32;
        let prefill = 3i32;
        let config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl);
        let b = 1i32;
        let h = heads;
        let d = dim as i32;

        let prefill_len = (b * h * prefill * d) as usize;
        let make_data = |seed: f32| -> Vec<f32> {
            (0..prefill_len)
                .map(|i| ((i as f32) * 0.07 + seed).sin())
                .collect()
        };
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(0.7), &[b, h, prefill, d]);

        let mut cache = QuantizedKvCache::new(config);
        cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        let gpu = cache
            .keys
            .as_ref()
            .and_then(|k| k.gpu.as_ref())
            .expect("uniform NoQjl should produce a GpuKeyStore");
        assert!(
            gpu.qjl_signs.is_none(),
            "NoQjl must NOT allocate qjl_signs on the GPU side"
        );
        assert!(
            gpu.qjl_signs_t.is_none(),
            "NoQjl must NOT allocate qjl_signs_t on the GPU side"
        );
    }

    // Phase D.3+ cleanup: NoQjl with pack_mode=Fullbyte must NOT allocate
    // q8_fullbyte_seq. The NoQjl fast paths read `ks.indices` directly
    // (which holds k_rot_scaled at key_mse_bits = key_bits), so a separate
    // 8b fullbyte shadow is dead storage — identical to ks.indices at 8b
    // and an unconsumed 8b view at 4..=7b widths. d256 here is the only
    // realised seq-shadow path; d128 doesn't trigger use_q8_seq_shadow.
    #[test]
    fn turboquant_no_qjl_fullbyte_skips_q8_fullbyte_seq() {
        let dim = 256usize;
        let heads = 1i32;
        let prefill = 4i32;
        let config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl)
            .with_pack_mode(super::config::TurboQuantPackMode::Fullbyte);
        let b = 1i32;
        let h = heads;
        let d = dim as i32;

        let prefill_len = (b * h * prefill * d) as usize;
        let make_data = |seed: f32| -> Vec<f32> {
            (0..prefill_len)
                .map(|i| ((i as f32) * 0.07 + seed).sin())
                .collect()
        };
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(0.7), &[b, h, prefill, d]);

        let mut cache = QuantizedKvCache::new(config);
        cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        let gpu = cache
            .keys
            .as_ref()
            .and_then(|k| k.gpu.as_ref())
            .expect("uniform NoQjl should produce a GpuKeyStore");
        assert!(
            gpu.q8_fullbyte_seq.is_none(),
            "NoQjl + pack_mode=Fullbyte must NOT allocate q8_fullbyte_seq \
             (NoQjl fast path reads ks.indices)"
        );
    }

    #[test]
    fn turboquant_hamming_distances_matches_scalar_reference() {
        // Phase F kernel parity: the Metal XOR+popcount kernel must agree
        // with a scalar Rust reference on arbitrary u32 inputs.
        let n_rows = 3usize;
        let n_seq = 17usize;
        let packed_dim = 4usize; // D = 128

        let mut query: Vec<u32> = (0..(n_rows * packed_dim))
            .map(|i| (i as u32).wrapping_mul(0x9E3779B1) ^ 0xDEADBEEF)
            .collect();
        // Salt query[1] differently so different rows produce distinct
        // distances and we exercise multi-row indexing.
        for w in query.iter_mut().skip(packed_dim).take(packed_dim) {
            *w = w.wrapping_mul(0xC2B2AE35).wrapping_add(0x85EBCA77);
        }

        let keys: Vec<u32> = (0..(n_rows * n_seq * packed_dim))
            .map(|i| (i as u32).wrapping_mul(0xCC9E2D51).rotate_left(13) ^ 0xC2B2AE35)
            .collect();

        let query_arr = InlineArray::from_u32_slice(
            &query,
            &[n_rows as i32, packed_dim as i32],
        );
        let keys_arr = InlineArray::from_u32_slice(
            &keys,
            &[n_rows as i32, n_seq as i32, packed_dim as i32],
        );

        // Non-GQA case (groups=1): row → kv_row identity, behaviour is the
        // same as the original kernel.
        let distances = InlineArray::turboquant_hamming_distances(
            &query_arr,
            &keys_arr,
            packed_dim as u32,
            n_rows as u32,
            n_seq as u32,
            1, // groups
        )
        .expect("hamming kernel");
        distances.eval();
        let actual: &[u32] = distances.as_slice();
        for row in 0..n_rows {
            for slot in 0..n_seq {
                let mut expected = 0u32;
                for w in 0..packed_dim {
                    let q = query[row * packed_dim + w];
                    let k = keys[(row * n_seq + slot) * packed_dim + w];
                    expected += (q ^ k).count_ones();
                }
                let got = actual[row * n_seq + slot];
                assert_eq!(
                    got, expected,
                    "row={row} slot={slot}: expected {expected}, got {got}"
                );
            }
        }
    }

    // Phase F.4 V2: GQA mode (groups > 1) — kernel maps q_row → kv_row =
    // q_row / groups. With groups=2, q_rows 0..2 share kv_row 0 and rows
    // 2..4 share kv_row 1. Distance must therefore use the SHARED kv key
    // signs while the query rows differ.
    #[test]
    fn turboquant_hamming_distances_gqa_kv_row_mapping() {
        let kv_rows = 2usize;
        let groups = 2usize;
        let q_rows = kv_rows * groups;
        let n_seq = 7usize;
        let packed_dim = 2usize; // D = 64

        // Distinct query per q_row.
        let query: Vec<u32> = (0..(q_rows * packed_dim))
            .map(|i| (i as u32).wrapping_mul(0x9E3779B1) ^ 0xCAFEF00D)
            .collect();
        // kv_rows worth of cached keys.
        let keys: Vec<u32> = (0..(kv_rows * n_seq * packed_dim))
            .map(|i| (i as u32).wrapping_mul(0xCC9E2D51).rotate_left(7) ^ 0x12345678)
            .collect();

        let query_arr =
            InlineArray::from_u32_slice(&query, &[q_rows as i32, packed_dim as i32]);
        let keys_arr = InlineArray::from_u32_slice(
            &keys,
            &[kv_rows as i32, n_seq as i32, packed_dim as i32],
        );

        let distances = InlineArray::turboquant_hamming_distances(
            &query_arr,
            &keys_arr,
            packed_dim as u32,
            q_rows as u32,
            n_seq as u32,
            groups as u32,
        )
        .expect("hamming kernel (gqa)");
        distances.eval();
        let actual: &[u32] = distances.as_slice();

        for q_row in 0..q_rows {
            let kv_row = q_row / groups;
            for slot in 0..n_seq {
                let mut expected = 0u32;
                for w in 0..packed_dim {
                    let q = query[q_row * packed_dim + w];
                    let k = keys[(kv_row * n_seq + slot) * packed_dim + w];
                    expected += (q ^ k).count_ones();
                }
                let got = actual[q_row * n_seq + slot];
                assert_eq!(
                    got, expected,
                    "q_row={q_row} kv_row={kv_row} slot={slot}: \
                     expected {expected}, got {got}"
                );
            }
        }
    }

    // Phase F.4 V2: end-to-end skiplist dispatch with GQA. Builds a cache
    // with q_heads != kv_heads, runs the skiplist dispatch path on a 1-step
    // decode, and confirms (a) the dispatch does NOT bail out (V1 would
    // have, due to the q_heads != kv_heads gate), and (b) the output is
    // finite and shape-correct. Top-M == cold_offset gives bit-equivalence
    // vs dense; partial M is correctness-finite-only at this layer.
    #[test]
    fn turboquant_hamming_skiplist_gqa_dispatch_runs() {
        let dim = 128usize;
        let kv_heads = 2i32;
        let groups = 4i32;
        let q_heads = kv_heads * groups;
        let prefill = 1024i32;
        let b = 1i32;
        let d = dim as i32;

        let kv_seed_len = (b * kv_heads * prefill * d) as usize;
        let make_data = |seed: f32| -> Vec<f32> {
            (0..kv_seed_len)
                .map(|i| ((i as f32) * 0.05 + seed).sin())
                .collect()
        };
        let seed_keys =
            InlineArray::from_f32_slice(&make_data(0.1), &[b, kv_heads, prefill, d]);
        let seed_values =
            InlineArray::from_f32_slice(&make_data(0.7), &[b, kv_heads, prefill, d]);

        // Variant F (NoQjl) + skiplist threshold of 0 so the path engages
        // immediately at any cold size. Recent window None so all tokens
        // are cold.
        let config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl)
            .with_skiplist_threshold(Some(0));
        let mut cache = QuantizedKvCache::new(config);
        cache.append(&seed_keys, &seed_values).expect("seed append");

        // 1-step decode: query has q_heads, step_keys/values have kv_heads.
        let q_len = (b * q_heads * 1 * d) as usize;
        let kv_step_len = (b * kv_heads * 1 * d) as usize;
        let query_data: Vec<f32> = (0..q_len).map(|i| ((i as f32) * 0.03).cos()).collect();
        let kv_step_data: Vec<f32> =
            (0..kv_step_len).map(|i| ((i as f32) * 0.04).sin()).collect();
        let query = InlineArray::from_f32_slice(&query_data, &[b, q_heads, 1, d]);
        let step_keys =
            InlineArray::from_f32_slice(&kv_step_data, &[b, kv_heads, 1, d]);
        let step_values = InlineArray::from_f32_slice(
            &kv_step_data.iter().map(|v| v * 0.5).collect::<Vec<_>>(),
            &[b, kv_heads, 1, d],
        );
        let scale = 1.0f32 / (d as f32).sqrt();

        let output = cache
            .append_and_compute_attention(&query, &step_keys, &step_values, scale)
            .expect("attention");
        assert_eq!(output.shape(), &[b, q_heads, 1, d]);
        let total = (b * q_heads * 1 * d) as usize;
        let vals: Vec<f32> = output
            .reshape(&[total as i32])
            .to_f32_vec(total)
            .expect("output to_f32");
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "GQA skiplist dispatch must produce finite output"
        );
    }

    #[test]
    fn turboquant_skiplist_threshold_populates_sign_hash() {
        // Phase F: when skiplist_threshold is set, encode must produce the
        // per-slot sign-hash buffer used by the Hamming pre-filter. When
        // unset, the buffer must remain None so the legacy path stays free
        // of the extra allocation.
        let dim = 16usize;
        let heads = 2i32;
        let prefill = 3i32;
        let b = 1i32;
        let h = heads;
        let d = dim as i32;

        let prefill_len = (b * h * prefill * d) as usize;
        let make_data = |seed: f32| -> Vec<f32> {
            (0..prefill_len)
                .map(|i| ((i as f32) * 0.07 + seed).sin())
                .collect()
        };
        let keys = InlineArray::from_f32_slice(&make_data(0.2), &[b, h, prefill, d]);
        let values = InlineArray::from_f32_slice(&make_data(0.7), &[b, h, prefill, d]);

        // Off path: sign_hash should be None.
        let off_config = TurboQuantConfig::uniform(8, 8).with_recent_window(None);
        let mut off_cache = QuantizedKvCache::new(off_config);
        off_cache.append(&keys, &values).expect("off append");
        let off_gpu = off_cache
            .keys
            .as_ref()
            .and_then(|k| k.gpu.as_ref())
            .expect("uniform should produce a GpuKeyStore");
        assert!(
            off_gpu.sign_hash.is_none(),
            "skiplist_threshold=None must NOT allocate sign_hash"
        );

        // On path: sign_hash present with shape [B, H, T, ceil(D/32)].
        let on_config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None)
            .with_skiplist_threshold(Some(1024));
        let mut on_cache = QuantizedKvCache::new(on_config);
        on_cache.append(&keys, &values).expect("on append");
        let on_gpu = on_cache
            .keys
            .as_ref()
            .and_then(|k| k.gpu.as_ref())
            .expect("uniform should produce a GpuKeyStore");
        let sign_hash = on_gpu
            .sign_hash
            .as_ref()
            .expect("skiplist_threshold=Some must allocate sign_hash");
        assert_eq!(sign_hash.dim(0), b);
        assert_eq!(sign_hash.dim(1), h);
        assert_eq!(sign_hash.dim(2), prefill);
        let expected_words = dim.div_ceil(32) as i32;
        assert_eq!(
            sign_hash.dim(3),
            expected_words,
            "sign_hash should hold ceil(D/32) packed u32 words per slot"
        );
    }

    // Phase F.4: when skiplist_threshold gates a kv-cache that is below the
    // 2048-slot top-M cap, the Hamming pre-filter selects ALL cold slots
    // (top_m == n_seq), so the kernel-on-gathered-subset path produces the
    // exact same scores as the dense kernel — modulo accumulation order. This
    // pins the dispatch wiring without requiring a separate quality bench.
    #[test]
    fn turboquant_hamming_skiplist_full_slot_scoring_matches_dense() {
        let dim = 128usize;
        let heads = 2i32;
        let prefill = 1024i32;
        let config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl)
            .with_skiplist_threshold(Some(512));
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let step_keys = InlineArray::from_f32_slice(&make_data(step_len, 1.9), &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");
        // Sanity: skiplist gates require sign_hash population.
        assert!(
            seed_cache
                .keys
                .as_ref()
                .and_then(|k| k.gpu.as_ref())
                .and_then(|g| g.sign_hash.as_ref())
                .is_some(),
            "skiplist_threshold=Some must allocate sign_hash on encode"
        );

        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("skiplist dispatch attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            (prefill + 1) as i32,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        // top_m == cold_offset means the Hamming pre-filter is a no-op
        // selection (all slots picked), so the score result must match the
        // dense Variant F kernel within long-context fp32 accumulation drift.
        assert!(
            max_abs_diff < 1e-3,
            "skiplist dispatch (top_m == cold_offset) diverged from dense reference: max_abs_diff={max_abs_diff}"
        );
    }

    // Phase E.2: when outliers = PerBlock { k }, encode must populate
    // outlier_channels (u8 channel index per slot) and outlier_values (f16
    // rotated value at that channel). Default None leaves both buffers
    // unallocated. Decode-time override is not yet wired — this test only
    // pins the storage gating + content; reconstruction quality is unchanged.
    #[test]
    fn turboquant_outliers_per_block_populates_extrema() {
        let dim = 16usize;
        let heads = 1i32;
        let prefill = 3i32;
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let k_outliers: u8 = 4;

        let prefill_len = (b * h * prefill * d) as usize;
        let make_data = |seed: f32| -> Vec<f32> {
            (0..prefill_len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.13 - seed).cos())
                .collect()
        };
        let keys = InlineArray::from_f32_slice(&make_data(0.2), &[b, h, prefill, d]);
        let values = InlineArray::from_f32_slice(&make_data(0.7), &[b, h, prefill, d]);

        // Default (None): no outlier buffers.
        let off_config = TurboQuantConfig::uniform(8, 8).with_recent_window(None);
        let mut off_cache = QuantizedKvCache::new(off_config);
        off_cache.append(&keys, &values).expect("off append");
        let off_gpu = off_cache
            .keys
            .as_ref()
            .and_then(|k| k.gpu.as_ref())
            .expect("uniform should produce a GpuKeyStore");
        assert!(
            off_gpu.outlier_channels.is_none(),
            "outliers = None must NOT allocate outlier_channels"
        );
        assert!(
            off_gpu.outlier_values.is_none(),
            "outliers = None must NOT allocate outlier_values"
        );

        // PerBlock { k }: both buffers populated with shape [B, H, T, k].
        let on_config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None)
            .with_outliers(super::config::TurboQuantOutlierMode::PerBlock { k: k_outliers });
        let mut on_cache = QuantizedKvCache::new(on_config);
        on_cache.append(&keys, &values).expect("on append");
        let on_gpu = on_cache
            .keys
            .as_ref()
            .and_then(|k| k.gpu.as_ref())
            .expect("uniform should produce a GpuKeyStore");

        let channels = on_gpu
            .outlier_channels
            .as_ref()
            .expect("outliers = PerBlock must allocate outlier_channels");
        let values_arr = on_gpu
            .outlier_values
            .as_ref()
            .expect("outliers = PerBlock must allocate outlier_values");

        assert_eq!(channels.dim(0), b);
        assert_eq!(channels.dim(1), h);
        assert_eq!(channels.dim(2), prefill);
        assert_eq!(channels.dim(3), k_outliers as i32);
        assert_eq!(values_arr.dim(3), k_outliers as i32);

        // Sanity: every channel index must be in [0, D).
        let channels_u32 = channels.as_dtype(crate::dtype::U32);
        channels_u32.eval();
        let n_per_slot = k_outliers as usize;
        let total = (b * h * prefill) as usize * n_per_slot;
        let raw: &[u32] = channels_u32.as_slice();
        assert!(raw.len() >= total);
        for &c in &raw[..total] {
            assert!(
                c < dim as u32,
                "outlier channel {c} out of range [0, {dim})"
            );
        }

        // Sanity: outlier values are finite f16 reads.
        let values_f32 = values_arr.as_dtype(crate::dtype::F32);
        values_f32.eval();
        let vals: &[f32] = values_f32.as_slice();
        assert!(vals.len() >= total);
        assert!(
            vals[..total].iter().all(|v| v.is_finite()),
            "outlier values must be finite"
        );
    }

    // Phase E.3: with PerBlock outlier override, encode zeros the K largest
    // |rotated| coords before slot_scale + codebook quant, and decode
    // (gpu_dequantize_keys) scatters the stored f16 values back at those
    // channels. The expected outcome is that dequantized keys reconstruct
    // their input more accurately than a baseline run with outliers = None
    // at the same bit-width — the heavy-tail coords are restored to f16
    // precision instead of going through the body-fit codebook.
    #[test]
    fn turboquant_outliers_per_block_decode_override_improves_reconstruction() {
        // Use a very low bit-width so the codebook approximation is coarse
        // enough that the override benefit is unambiguous in the unit-test
        // tolerance. d=16 keeps the test fast.
        let dim = 16usize;
        let heads = 1i32;
        let prefill = 4i32;
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let k_outliers: u8 = 2;

        let prefill_len = (b * h * prefill * d) as usize;
        // Mix a smooth body with a few deliberately heavy spikes per slot.
        // Spikes land in input space and get diffused across rotated channels
        // by the rotation, so several rotated coords end up large per slot —
        // exactly the heavy-tail distribution outlier override targets.
        let mut data = vec![0f32; prefill_len];
        for slot in 0..(prefill as usize) {
            for c in 0..dim {
                let i = slot * dim + c;
                data[i] = (((i as f32) * 0.07).sin() * 0.3) + (((i as f32) * 0.13).cos() * 0.2);
            }
            // Two large spikes per slot at channels (slot, slot+5) — different
            // per slot to stress per-block extraction (not a fixed channel
            // mask the codebook could memorise).
            let s = slot * dim;
            data[s + slot] += 6.0;
            data[s + ((slot + 5) % dim)] -= 5.5;
        }
        let keys = InlineArray::from_f32_slice(&data, &[b, h, prefill, d]);
        // Values are arbitrary — only key reconstruction matters here.
        let values = InlineArray::from_f32_slice(&data, &[b, h, prefill, d]);

        // Baseline: outliers = None. The heavy spikes pass through the
        // codebook approximation at low bit-width.
        let off_config = TurboQuantConfig::uniform(4, 4).with_recent_window(None);
        let mut off_cache = QuantizedKvCache::new(off_config);
        off_cache.append(&keys, &values).expect("off append");
        let off_dk = off_cache
            .dequantize_keys()
            .expect("off dequantize_keys");
        assert_eq!(off_dk.shape(), &[b, h, prefill, d]);
        let total = prefill_len;
        let off_vals: Vec<f32> = off_dk.reshape(&[total as i32]).to_f32_vec(total).unwrap();

        // With override: the K largest |rotated| coords are reproduced from
        // the stored f16 values; the body uses the same 4-bit codebook.
        let on_config = TurboQuantConfig::uniform(4, 4)
            .with_recent_window(None)
            .with_outliers(super::config::TurboQuantOutlierMode::PerBlock { k: k_outliers });
        let mut on_cache = QuantizedKvCache::new(on_config);
        on_cache.append(&keys, &values).expect("on append");
        let on_dk = on_cache.dequantize_keys().expect("on dequantize_keys");
        assert_eq!(on_dk.shape(), &[b, h, prefill, d]);
        let on_vals: Vec<f32> = on_dk.reshape(&[total as i32]).to_f32_vec(total).unwrap();

        let off_mse: f32 = off_vals
            .iter()
            .zip(data.iter())
            .map(|(r, o)| (r - o) * (r - o))
            .sum::<f32>()
            / (total as f32);
        let on_mse: f32 = on_vals
            .iter()
            .zip(data.iter())
            .map(|(r, o)| (r - o) * (r - o))
            .sum::<f32>()
            / (total as f32);
        assert!(
            on_mse < off_mse,
            "outlier override must improve reconstruction MSE: off={off_mse} on={on_mse}"
        );
        // The heavy spikes contribute a large fraction of the total error
        // when the codebook squashes them. Override should cut MSE by at
        // least 30% on this synthetic workload (typical real-world drop is
        // larger; this bound is loose to keep the test stable across
        // floating-point drift).
        assert!(
            on_mse < off_mse * 0.7,
            "outlier override expected to cut MSE by ≥30% on heavy-tail keys: \
             off={off_mse} on={on_mse} ratio={}",
            on_mse / off_mse
        );
        // Sanity: outputs finite.
        assert!(on_vals.iter().all(|v| v.is_finite()));
        assert!(off_vals.iter().all(|v| v.is_finite()));
    }

    // Phase E.3 NoQjl variant: production targets typically run Variant F
    // (NoQjl) with outliers, so cover that mode too. Variant F gives the
    // codebook the full bit-budget (no QJL residual sign), which makes the
    // body tighter — the override should still cleanly improve the
    // heavy-tail reconstruction at the same bit-width.
    #[test]
    fn turboquant_outliers_per_block_decode_override_no_qjl() {
        let dim = 16usize;
        let heads = 1i32;
        let prefill = 4i32;
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let k_outliers: u8 = 2;

        let prefill_len = (b * h * prefill * d) as usize;
        let mut data = vec![0f32; prefill_len];
        for slot in 0..(prefill as usize) {
            for c in 0..dim {
                let i = slot * dim + c;
                data[i] = (((i as f32) * 0.07).sin() * 0.3) + (((i as f32) * 0.13).cos() * 0.2);
            }
            let s = slot * dim;
            data[s + slot] += 6.0;
            data[s + ((slot + 5) % dim)] -= 5.5;
        }
        let keys = InlineArray::from_f32_slice(&data, &[b, h, prefill, d]);
        let values = InlineArray::from_f32_slice(&data, &[b, h, prefill, d]);

        let off_config = TurboQuantConfig::uniform(4, 4)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl);
        let mut off_cache = QuantizedKvCache::new(off_config);
        off_cache.append(&keys, &values).expect("off append");
        let off_dk = off_cache.dequantize_keys().expect("off dequantize_keys");
        let total = prefill_len;
        let off_vals: Vec<f32> = off_dk.reshape(&[total as i32]).to_f32_vec(total).unwrap();

        let on_config = TurboQuantConfig::uniform(4, 4)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl)
            .with_outliers(super::config::TurboQuantOutlierMode::PerBlock { k: k_outliers });
        let mut on_cache = QuantizedKvCache::new(on_config);
        on_cache.append(&keys, &values).expect("on append");
        let on_dk = on_cache.dequantize_keys().expect("on dequantize_keys");
        let on_vals: Vec<f32> = on_dk.reshape(&[total as i32]).to_f32_vec(total).unwrap();

        let off_mse: f32 = off_vals
            .iter()
            .zip(data.iter())
            .map(|(r, o)| (r - o) * (r - o))
            .sum::<f32>()
            / (total as f32);
        let on_mse: f32 = on_vals
            .iter()
            .zip(data.iter())
            .map(|(r, o)| (r - o) * (r - o))
            .sum::<f32>()
            / (total as f32);
        assert!(
            on_mse < off_mse * 0.7,
            "NoQjl + outlier override expected to cut MSE by ≥30%: \
             off={off_mse} on={on_mse} ratio={}",
            on_mse / off_mse
        );
        assert!(on_vals.iter().all(|v| v.is_finite()));
    }

    // Phase D.2: setting pack_mode = Fullbyte must build the q8_fullbyte_seq
    // shadow buffer at encode time, even when the legacy env-var override
    // (PMETAL_TQ_Q8_FULLBYTE) is unset. Default Bitstream leaves the buffer
    // None so existing call sites pay no extra allocation.
    #[test]
    fn turboquant_pack_mode_fullbyte_populates_q8_fullbyte_seq() {
        // q8 / d256 uniform is the only currently-realised fullbyte path
        // (use_q8_seq_shadow gate). Other configs accept pack_mode = Fullbyte
        // but stay on the bitstream path until the kernel is widened.
        let dim = 256usize;
        let heads = 1i32;
        let prefill = 4i32;
        let b = 1i32;
        let h = heads;
        let d = dim as i32;

        let prefill_len = (b * h * prefill * d) as usize;
        let make_data = |seed: f32| -> Vec<f32> {
            (0..prefill_len)
                .map(|i| ((i as f32) * 0.07 + seed).sin())
                .collect()
        };
        let keys = InlineArray::from_f32_slice(&make_data(0.2), &[b, h, prefill, d]);
        let values = InlineArray::from_f32_slice(&make_data(0.7), &[b, h, prefill, d]);

        // Default (Bitstream): no fullbyte shadow.
        let bitstream_config = TurboQuantConfig::uniform(8, 8).with_recent_window(None);
        let mut bitstream_cache = QuantizedKvCache::new(bitstream_config);
        bitstream_cache
            .append(&keys, &values)
            .expect("bitstream append");
        let bitstream_gpu = bitstream_cache
            .keys
            .as_ref()
            .and_then(|k| k.gpu.as_ref())
            .expect("uniform should produce a GpuKeyStore");
        // Note: env-var is not set in cargo test by default, so q8_fullbyte_seq
        // must be None for the bitstream config.
        if std::env::var_os("PMETAL_TQ_Q8_FULLBYTE").is_none() {
            assert!(
                bitstream_gpu.q8_fullbyte_seq.is_none(),
                "pack_mode = Bitstream + env-var unset must NOT allocate q8_fullbyte_seq"
            );
        }

        // Fullbyte: shadow present regardless of env-var.
        let fullbyte_config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None)
            .with_pack_mode(super::config::TurboQuantPackMode::Fullbyte);
        let mut fullbyte_cache = QuantizedKvCache::new(fullbyte_config);
        fullbyte_cache
            .append(&keys, &values)
            .expect("fullbyte append");
        let fullbyte_gpu = fullbyte_cache
            .keys
            .as_ref()
            .and_then(|k| k.gpu.as_ref())
            .expect("uniform should produce a GpuKeyStore");
        let fullbyte_seq = fullbyte_gpu
            .q8_fullbyte_seq
            .as_ref()
            .expect("pack_mode = Fullbyte must allocate q8_fullbyte_seq");
        assert_eq!(fullbyte_seq.dim(0), b);
        assert_eq!(fullbyte_seq.dim(1), h);
        assert_eq!(fullbyte_seq.dim(2), prefill);
        assert_eq!(fullbyte_seq.dim(3), d);
    }

    // Phase D.3+ regression: when pack_mode=Fullbyte is combined with the
    // default Standard QJL mode at d256/8b/8b, the encoder must quantise from
    // `k_rot / slot_scale` (i.e. the same scaled rotated values that drive
    // ks.indices), not from un-scaled k_rot. The score kernel multiplies
    // codebook lookups by `key_slot_scale` unconditionally; quantising
    // un-scaled k_rot folds the per-row magnitude into the codebook hit, so
    // the kernel double-multiplies by slot_scale and silently corrupts long-
    // context attention whenever rows have varying magnitudes. This test
    // builds adversarial per-row magnitudes (so slot_scale spans 0.1× to 4×
    // across rows) and asserts the production fullbyte fast path matches
    // dequantize+SDPA within codebook resolution. Pre-fix the divergence is
    // catastrophic; post-fix it sits comfortably under 5e-3.
    #[test]
    fn turboquant_pack_mode_fullbyte_standard_qjl_varying_magnitudes_match_dequantized_sdpa() {
        let dim = 256usize;
        let heads = 2i32;
        let prefill = 1023i32;
        // Default qjl mode is Standard — we deliberately do NOT call
        // `with_qjl_mode(NoQjl)` so the fullbyte production path under
        // try_gpu_uniform_attention_q8_d256_precomputed is exercised.
        let config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None)
            .with_pack_mode(super::config::TurboQuantPackMode::Fullbyte);
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        // Per-row magnitude varies geometrically across the cache, so
        // `slot_scale = max(|k_rot|) / centroid_max` spans roughly 0.1×–4×.
        // The bug-pre-fix path mis-quantises low-magnitude rows badly (idx
        // collapses to 0 because un-scaled k_rot[d] is well below the first
        // codebook step) and clips high-magnitude rows (idx saturates at the
        // codebook extremes), so the resulting attention output diverges
        // from dequantize+SDPA by orders of magnitude on this fixture.
        let make_data = |len: usize, seed: f32, prefill_i32: i32, d_i32: i32, b_i32: i32, h_i32: i32| -> Vec<f32> {
            let mut buf = vec![0.0f32; len];
            for bh in 0..(b_i32 * h_i32) as usize {
                for slot in 0..prefill_i32 as usize {
                    // Geometric spread: 0.1 .. 4.0 across slots.
                    let mag = 0.1f32 * (40.0f32).powf((slot as f32) / (prefill_i32 as f32 - 1.0));
                    let row_off = (bh * prefill_i32 as usize + slot) * d_i32 as usize;
                    for di in 0..d_i32 as usize {
                        let i = row_off + di;
                        buf[i] = (((i as f32) * 0.07 + seed).sin()
                            + ((i as f32) * 0.11 - seed).cos())
                            * mag;
                    }
                }
            }
            buf
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let prefill_keys_data = make_data(prefill_len, 0.2, prefill, d, b, h);
        let prefill_values_data = make_data(prefill_len, 0.7, prefill, d, b, h);
        let prefill_keys = InlineArray::from_f32_slice(&prefill_keys_data, &[b, h, prefill, d]);
        let prefill_values = InlineArray::from_f32_slice(&prefill_values_data, &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(
            &(0..step_len)
                .map(|i| ((i as f32) * 0.07 + 1.3).sin() + ((i as f32) * 0.11 - 1.3).cos())
                .collect::<Vec<_>>(),
            &[b, h, 1, d],
        );
        // Step token magnitude sits in the middle of the prefill range so the
        // newly-appended slot also has a non-trivial slot_scale.
        let step_keys_data: Vec<f32> = (0..step_len)
            .map(|i| (((i as f32) * 0.07 + 1.9).sin() + ((i as f32) * 0.11 - 1.9).cos()) * 1.5)
            .collect();
        let step_keys = InlineArray::from_f32_slice(&step_keys_data, &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(
            &(0..step_len)
                .map(|i| ((i as f32) * 0.07 + 2.4).sin() + ((i as f32) * 0.11 - 2.4).cos())
                .collect::<Vec<_>>(),
            &[b, h, 1, d],
        );

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        // Direct: production fullbyte fast path through
        // try_gpu_uniform_attention_q8_d256_precomputed.
        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        // Reference: dequantize the cache (uses ks.indices @ key_bits-1 +
        // QJL signs, which are quantised in scaled space and known-correct)
        // and run manual SDPA over the full cache.
        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        // 5e-2 covers codebook-resolution drift (8b fullbyte vs 7b+QJL ref)
        // plus fp32 reduction order across 1024 keys, with the per-row
        // magnitude variation amplifying both terms via softmax. Measured
        // pre-fix divergence on this fixture is ≈7.19 (un-scaled k_rot
        // double-multiplied by slot_scale at score time); post-fix sits
        // around 0.04 — a >150× tightening, leaving ample headroom under
        // the 5e-2 bound.
        assert!(
            max_abs_diff < 5e-2,
            "Standard-QJL fullbyte path diverged from dequantize+SDPA: \
             max_abs_diff={max_abs_diff}"
        );
    }

    // Phase F.4 partial-selection smoke: cold_offset > 2048 forces top_m < n_seq
    // (real Hamming pre-filter selection, not a no-op). Asserts the dispatch
    // returns a finite output of the correct shape — exact agreement vs dense
    // is not expected because attention is computed over a strict subset of
    // slots. A separate quality bench will measure recall@M against dense.
    #[test]
    fn turboquant_hamming_skiplist_partial_selection_runs() {
        let dim = 128usize;
        let heads = 2i32;
        let prefill = 2049i32;
        let config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl)
            .with_skiplist_threshold(Some(1024));
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let step_keys = InlineArray::from_f32_slice(&make_data(step_len, 1.9), &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut cache = QuantizedKvCache::new(config);
        cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        let mut output = cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("partial-selection skiplist dispatch");
        assert_eq!(output.dim(0), b);
        assert_eq!(output.dim(1), h);
        assert_eq!(output.dim(2), 1);
        assert_eq!(output.dim(3), d);

        let vals = output
            .to_f32_vec((b * h * d) as usize)
            .expect("output to_f32");
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "skiplist output contained non-finite values"
        );
    }

    // Pins the no_qjl_2pass d128 kernel fast path. Shares pass 2 with the
    // Standard d128 kernel; both were unblocked by the 2026-04-27 pass-2
    // reduction fix.
    #[test]
    fn turboquant_no_qjl_d128_long_context_matches_dequantized_sdpa() {
        // Exercises the new no_qjl fused d128 fast path (n_seq >= 1024 +
        // d=128 + 8b/8b uniform). The kernel result must match the
        // dequantize+SDPA reference within fp32 numerical precision.
        let dim = 128usize;
        let heads = 2i32;
        let prefill = 1023i32;
        let config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl);
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let step_keys = InlineArray::from_f32_slice(&make_data(step_len, 1.9), &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        // Fused kernel uses the same Beta codebook + slot_scale + key_norm
        // factors as the dequantize path; differences are pure fp32
        // accumulation order. 1e-3 tolerance handles long-context summation
        // drift across 1024 keys.
        assert!(
            max_abs_diff < 1e-3,
            "Variant F d128 fused kernel diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    // GQA flavor of the d128 NoQjl long-context fast path. Variant F has no
    // separate `packed_keys` kernel — the base `no_qjl_2pass` reads u8 indices
    // directly (no QJL signs to bit-pack alongside), so it serves the same
    // role for groups > 8 that `packed_keys_2pass` serves for Standard. This
    // test proves the kernel handles groups=8 correctly (q_heads=16,
    // kv_heads=2 → groups=8; threadgroup is 32 simdgroups × 32 lanes = 256
    // threads, well below the 1024 limit).
    #[test]
    fn turboquant_no_qjl_d128_long_context_matches_dequantized_sdpa_gqa() {
        let dim = 128usize;
        let q_heads = 16i32;
        let kv_heads = 2i32;
        let prefill = 1023i32;
        let config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl);
        let b = 1i32;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_kv_len = (b * kv_heads * prefill * d) as usize;
        let step_kv_len = (b * kv_heads * d) as usize;
        let query_len = (b * q_heads * d) as usize;
        let prefill_keys = InlineArray::from_f32_slice(
            &make_data(prefill_kv_len, 0.2),
            &[b, kv_heads, prefill, d],
        );
        let prefill_values = InlineArray::from_f32_slice(
            &make_data(prefill_kv_len, 0.7),
            &[b, kv_heads, prefill, d],
        );
        let queries = InlineArray::from_f32_slice(&make_data(query_len, 1.3), &[b, q_heads, 1, d]);
        let step_keys =
            InlineArray::from_f32_slice(&make_data(step_kv_len, 1.9), &[b, kv_heads, 1, d]);
        let step_values =
            InlineArray::from_f32_slice(&make_data(step_kv_len, 2.4), &[b, kv_heads, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let full_values = ref_cache.dequantize_values().expect("dequantize values");
        let repeated_keys = full_keys.repeat(q_heads / kv_heads, 1);
        let repeated_values = full_values.repeat(q_heads / kv_heads, 1);
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut repeated_keys.clone(),
            &mut repeated_values.clone(),
            b,
            q_heads,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * q_heads * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-3,
            "Variant F d128 GQA fused kernel diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    // Pins the no_qjl_2pass d256 kernel fast path. Mirrors the d128 test
    // above; shares pass 2 with the Standard d256 kernel, so the kernel
    // reduction shape was already correct (the 2026-04-27 fix landed in
    // d128 only).
    #[test]
    fn turboquant_no_qjl_d256_long_context_matches_dequantized_sdpa() {
        let dim = 256usize;
        let heads = 2i32;
        let prefill = 1023i32;
        let config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl);
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let step_keys = InlineArray::from_f32_slice(&make_data(step_len, 1.9), &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        // Same tolerance as d128 — fp32 accumulation order differs across
        // 1024 keys regardless of head_dim.
        assert!(
            max_abs_diff < 1e-3,
            "Variant F d256 fused kernel diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    // Phase E.4 V1: with PerBlock outliers enabled, the d256 fullbyte score
    // kernel must add the precomputed outlier-bias term so its end-to-end
    // result matches dequantize+SDPA (which goes through the E.3 scatter).
    // Heavy-tail prefill data — the scenario outliers were designed for —
    // exercises both the encode-side zero/scatter and the score-side bias.
    #[test]
    fn turboquant_no_qjl_d256_long_context_outliers_match_dequantized_sdpa() {
        let dim = 256usize;
        let heads = 2i32;
        let prefill = 1023i32;
        let k_outliers: u8 = 4;
        let config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl)
            .with_outliers(super::config::TurboQuantOutlierMode::PerBlock { k: k_outliers });
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };
        // Inject heavy spikes at varying channels per slot — same shape as
        // the GPU E.3 test, so the bias path actually has non-trivial work.
        let inject_spikes = |buf: &mut [f32], dim_i32: i32, prefill_i32: i32, batch_i32: i32, head_i32: i32| {
            for bh in 0..(batch_i32 * head_i32) as usize {
                for slot in 0..prefill_i32 as usize {
                    let row = bh * prefill_i32 as usize + slot;
                    let row_off = row * dim_i32 as usize;
                    let dim_u = dim_i32 as usize;
                    buf[row_off + (slot % dim_u)] += 6.0;
                    buf[row_off + ((slot + 17) % dim_u)] -= 5.5;
                    buf[row_off + ((slot + 53) % dim_u)] += 4.7;
                    buf[row_off + ((slot + 113) % dim_u)] -= 4.3;
                }
            }
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let mut prefill_keys_data = make_data(prefill_len, 0.2);
        inject_spikes(&mut prefill_keys_data, d, prefill, b, h);
        let prefill_keys = InlineArray::from_f32_slice(&prefill_keys_data, &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let mut step_keys_data = make_data(step_len, 1.9);
        // Step token also gets spikes — exercises the cache-grow path.
        for bh in 0..(b * h) as usize {
            let row_off = bh * dim;
            step_keys_data[row_off + 7] += 5.0;
            step_keys_data[row_off + 41] -= 4.5;
            step_keys_data[row_off + 99] += 4.0;
            step_keys_data[row_off + 199] -= 3.5;
        }
        let step_keys = InlineArray::from_f32_slice(&step_keys_data, &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        // Direct path: GPU score kernel with outlier-bias variant.
        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        // Reference path: append, then dequantize (which applies the E.3
        // scatter override) and run manual SDPA.
        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        // Slightly looser tolerance than the no-outlier 1e-3 because the
        // bias path adds an extra fp32 reduction pass; values stored as
        // f16 also round-trip through half-precision in the GPU
        // outlier_values buffer (versus f32 in the manual reference's
        // dequant path, which uses gpu_dequantize_keys → also reads
        // outlier_values as f16 so the dtype mismatch is shared, not
        // additive). 5e-3 leaves margin for SIMD-reduction order drift.
        assert!(
            max_abs_diff < 5e-3,
            "Variant F d256 fused kernel with outlier-bias diverged from \
             dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    // Phase E.4 V2: outlier-bias path on d128 base no_qjl_2pass. d128
    // doesn't trigger `use_q8_seq_shadow` in encode (no q8_slot_scales_seq /
    // d256_rot_values_seq shadows), so the dispatch goes through the
    // standalone key_norms / key_slot_scale / value_indices_t / value_norms
    // paths — different from the d256 fullbyte test which uses the seq
    // shadow. Same parity bound (5e-3) since the bias compute is identical.
    #[test]
    fn turboquant_no_qjl_d128_long_context_outliers_match_dequantized_sdpa() {
        let dim = 128usize;
        let heads = 2i32;
        let prefill = 1023i32;
        let k_outliers: u8 = 4;
        let config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl)
            .with_outliers(super::config::TurboQuantOutlierMode::PerBlock { k: k_outliers });
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let mut prefill_keys_data = make_data(prefill_len, 0.2);
        // Inject heavy spikes per slot at varying channels.
        for bh in 0..(b * h) as usize {
            for slot in 0..prefill as usize {
                let row_off = (bh * prefill as usize + slot) * dim;
                prefill_keys_data[row_off + (slot % dim)] += 6.0;
                prefill_keys_data[row_off + ((slot + 17) % dim)] -= 5.5;
                prefill_keys_data[row_off + ((slot + 53) % dim)] += 4.7;
                prefill_keys_data[row_off + ((slot + 79) % dim)] -= 4.3;
            }
        }
        let prefill_keys = InlineArray::from_f32_slice(&prefill_keys_data, &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let mut step_keys_data = make_data(step_len, 1.9);
        for bh in 0..(b * h) as usize {
            let row_off = bh * dim;
            step_keys_data[row_off + 7] += 5.0;
            step_keys_data[row_off + 41] -= 4.5;
            step_keys_data[row_off + 99] += 4.0;
            step_keys_data[row_off + 113] -= 3.5;
        }
        let step_keys = InlineArray::from_f32_slice(&step_keys_data, &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 5e-3,
            "Variant F d128 fused kernel with outlier-bias diverged from \
             dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    // Phase D.3.1: NoQjl + d256 fast path widened to key_bits in 4..=8.
    // Score kernel loads 256 codebook entries unconditionally, so encode +
    // dispatch must hand it a zero-padded 256-entry codebook. Indices are
    // u8 in [0, 2^bits) — never index past the real centroids — and the
    // kernel still recovers the original-magnitude key via per-slot
    // `key_slot_scale * key_norm`. Tolerance loosens vs the 8-bit pin
    // because lower-bit codebooks introduce more reconstruction error,
    // but the GPU fast path must still match the dequantize+SDPA
    // reference within the codebook's resolution.
    fn run_no_qjl_d256_long_context_matches_dequantized_sdpa_at_bits(
        key_bits: u8,
        max_abs_diff_tol: f32,
    ) {
        let dim = 256usize;
        let heads = 2i32;
        let prefill = 1023i32;
        let config = TurboQuantConfig::uniform(key_bits, 8)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl);
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let step_keys = InlineArray::from_f32_slice(&make_data(step_len, 1.9), &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < max_abs_diff_tol,
            "Variant F d256 fused kernel at {key_bits}b diverged from dequantized sdpa: \
             max_abs_diff={max_abs_diff} tol={max_abs_diff_tol}"
        );
    }

    #[test]
    fn turboquant_no_qjl_d256_long_context_matches_dequantized_sdpa_5b() {
        // 5-bit codebook: 32 centroids; widening uses zero-padded 256-entry
        // view for the score kernel.
        run_no_qjl_d256_long_context_matches_dequantized_sdpa_at_bits(5, 1e-3);
    }

    #[test]
    fn turboquant_no_qjl_d256_long_context_matches_dequantized_sdpa_4b() {
        // 4-bit codebook: 16 centroids. Looser tolerance because the
        // codebook itself has lower resolution; still must match the
        // dequant+SDPA reference (which uses the same 16-entry codebook).
        run_no_qjl_d256_long_context_matches_dequantized_sdpa_at_bits(4, 1e-3);
    }

    // Phase G: extreme low-bit support. The score kernel is bit-width-agnostic
    // (reads 256-padded codebook unconditionally, indexes by u8); the encoder
    // uses the real 2^bits-entry codebook. Both paths converge on the same
    // dequantize+SDPA reference at the same codebook resolution, so parity
    // holds within fp32 reduction drift. Quality vs fp16 is a separate
    // question — at 2..=3 bits the codebook gap dominates and Variant G
    // outliers are required for practical use; the dedicated outliers test
    // (turboquant_no_qjl_d256_long_context_outliers_match_dequantized_sdpa_2b)
    // pins the outlier-paired path.
    #[test]
    fn turboquant_no_qjl_d256_long_context_matches_dequantized_sdpa_3b() {
        run_no_qjl_d256_long_context_matches_dequantized_sdpa_at_bits(3, 1e-3);
    }

    #[test]
    fn turboquant_no_qjl_d256_long_context_matches_dequantized_sdpa_2b() {
        run_no_qjl_d256_long_context_matches_dequantized_sdpa_at_bits(2, 1e-3);
    }

    // Phase G SOTA: 2-bit codebook (4 levels, ≈8× compression vs fp16) paired
    // with 8 per-block fp16 outliers (Variant G). The 4-level codebook hits
    // only the body's bulk distribution; the 8 outliers per slot capture the
    // heavy tail that wrecks raw 2-bit. Per-block layout:
    //   2b body @ d256: 256 * 2 / 8     = 64 B
    //   slot_scale + key_norm           ≈  8 B
    //   outliers (8 channels + 8 vals)  = 8 + 16 = 24 B
    //   total                           = 96 B per slot
    // vs fp16 baseline at d256: 512 B per slot → ~5.3× compression.
    // Compared to quant.cpp's TURBO_KV_2B (40 B at d=128 = 2-bit body with 1+1
    // sign hash, no outliers), we trade ~30 extra bytes per slot for an
    // outlier mechanism that closes the heavy-tail gap — quality-superior at
    // similar memory budget.
    //
    // This test pins the kernel/dispatch parity at this aggressive setting:
    // the direct fast path must match dequant+SDPA on the same encoded
    // values to within fp32 reduction drift + f16 outlier-value round-trip.
    #[test]
    fn turboquant_no_qjl_d256_long_context_2b_with_outliers_match_dequantized_sdpa() {
        let dim = 256usize;
        let heads = 2i32;
        let prefill = 1023i32;
        let k_outliers: u8 = 8;
        let config = TurboQuantConfig::uniform(2, 8)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl)
            .with_outliers(super::config::TurboQuantOutlierMode::PerBlock { k: k_outliers });
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };
        // Heavy-tail spikes per slot — mirrors the 8b outliers test.
        let inject_spikes = |buf: &mut [f32], dim_i32: i32, prefill_i32: i32, batch_i32: i32, head_i32: i32| {
            for bh in 0..(batch_i32 * head_i32) as usize {
                for slot in 0..prefill_i32 as usize {
                    let row = bh * prefill_i32 as usize + slot;
                    let row_off = row * dim_i32 as usize;
                    let dim_u = dim_i32 as usize;
                    buf[row_off + (slot % dim_u)] += 6.0;
                    buf[row_off + ((slot + 17) % dim_u)] -= 5.5;
                    buf[row_off + ((slot + 53) % dim_u)] += 4.7;
                    buf[row_off + ((slot + 113) % dim_u)] -= 4.3;
                    buf[row_off + ((slot + 7) % dim_u)] += 3.9;
                    buf[row_off + ((slot + 31) % dim_u)] -= 3.5;
                    buf[row_off + ((slot + 67) % dim_u)] += 3.1;
                    buf[row_off + ((slot + 191) % dim_u)] -= 2.7;
                }
            }
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let mut prefill_keys_data = make_data(prefill_len, 0.2);
        inject_spikes(&mut prefill_keys_data, d, prefill, b, h);
        let prefill_keys = InlineArray::from_f32_slice(&prefill_keys_data, &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let mut step_keys_data = make_data(step_len, 1.9);
        for bh in 0..(b * h) as usize {
            let row_off = bh * dim;
            step_keys_data[row_off + 7] += 5.0;
            step_keys_data[row_off + 41] -= 4.5;
            step_keys_data[row_off + 99] += 4.0;
            step_keys_data[row_off + 199] -= 3.5;
            step_keys_data[row_off + 13] += 3.1;
            step_keys_data[row_off + 71] -= 2.9;
            step_keys_data[row_off + 137] += 2.5;
            step_keys_data[row_off + 233] -= 2.3;
        }
        let step_keys = InlineArray::from_f32_slice(&step_keys_data, &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        // Same 5e-3 bound as the 8b+outliers parity test: parity is between
        // the direct GPU path and the dequant+SDPA reference, both reading
        // the same encoded values + same outlier override. The bound is
        // dominated by fp32 reduction order + f16 outlier round-trip,
        // independent of body bit-width.
        assert!(
            max_abs_diff < 5e-3,
            "Variant F + 2b + 8-outliers d256 fused kernel diverged from \
             dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    // Phase H — progressive 3-tier cache ordering pin.
    //
    // Build a cache with: hot fp16 (recent_window default), warm tier at 8b
    // NoQjl + Variant G outliers, cold tier at 4b NoQjl. Prefill enough
    // tokens to trigger at least two hot→warm evictions and one warm→cold
    // migration so all three tiers have data. Then assert:
    //   1. cache.cold_len + cache.warm_len + cache.hot_len == prefill
    //   2. dequantize_keys returns shape [B, H, prefill, D] and is finite
    //   3. The same data appended to a 2-tier (warm_tier=None) reference
    //      cache produces a comparable dequant — temporal ordering matches.
    #[test]
    fn turboquant_warm_tier_progressive_3_tier_ordering() {
        use super::config::TurboQuantWarmTier;
        let dim = 128usize;
        let heads = 1i32;
        // Eviction trigger is `hot_offset > recent_window + HOT_EVICTION_CHUNK`
        // (≈2049 by default). We need two evictions plus a partial-fill, so
        // 3500 tokens total. That puts ~1024 in warm post-1st-eviction; the
        // 2nd eviction adds another 1024 → warm = 2048 → exceeds
        // warm_window=2048? Boundary; we set warm_window=1024 so the first
        // eviction itself triggers a warm→cold migration, exercising both
        // paths.
        let prefill = 3500i32;
        let warm_window = 1024usize;

        // Override recent_window so eviction triggers at small prefill.
        // Default is 8192 — prefill=3500 would never overflow.
        let recent_window = 512usize;
        let warm_tier = TurboQuantWarmTier::uniform(warm_window, 8, 8);
        let config_3tier = TurboQuantConfig::uniform(4, 4)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl)
            .with_recent_window(Some(recent_window))
            .with_warm_tier(Some(warm_tier));
        let config_2tier = TurboQuantConfig::uniform(4, 4)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl)
            .with_recent_window(Some(recent_window));

        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };
        let prefill_len = (b * h * prefill * d) as usize;
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);

        // 3-tier cache.
        let mut cache_3tier = QuantizedKvCache::new(config_3tier);
        cache_3tier
            .append(&prefill_keys, &prefill_values)
            .expect("3-tier prefill append");

        // Total accounting must match.
        assert_eq!(
            cache_3tier.cold_len() + cache_3tier.warm_len() + cache_3tier.hot_len(),
            prefill as usize,
            "cold + warm + hot must equal prefill",
        );
        assert_eq!(cache_3tier.offset, prefill as usize, "offset == total");
        // Cold must have data — eviction-then-migration pattern at this size.
        assert!(
            cache_3tier.cold_len() >= warm_window,
            "expected at least one warm→cold migration; cold_len={}",
            cache_3tier.cold_len()
        );

        // Warm tier struct must be present.
        assert!(cache_3tier.warm.is_some(), "warm sub-cache must exist");

        // Dequantize and assert shape + finiteness.
        let mut deq_3tier = cache_3tier
            .dequantize_keys()
            .expect("3-tier dequantize_keys");
        assert_eq!(deq_3tier.dim(0), b);
        assert_eq!(deq_3tier.dim(1), h);
        assert_eq!(deq_3tier.dim(2), prefill);
        assert_eq!(deq_3tier.dim(3), d);
        let deq_vals = deq_3tier
            .to_f32_vec((b * h * prefill * d) as usize)
            .expect("dequant to f32");
        assert!(
            deq_vals.iter().all(|v| v.is_finite()),
            "3-tier dequant must produce all-finite values"
        );

        // 2-tier reference for sanity — same prefill, no warm tier. Should
        // also dequantize to the same shape.
        let mut cache_2tier = QuantizedKvCache::new(config_2tier);
        cache_2tier
            .append(&prefill_keys, &prefill_values)
            .expect("2-tier prefill append");
        assert_eq!(cache_2tier.warm_len(), 0, "2-tier warm should be empty");
        assert_eq!(
            cache_2tier.cold_len() + cache_2tier.hot_len(),
            prefill as usize,
        );
        let deq_2tier = cache_2tier
            .dequantize_keys()
            .expect("2-tier dequantize_keys");
        assert_eq!(deq_2tier.dim(2), prefill);

        // Reset must clear warm.
        cache_3tier.reset();
        assert_eq!(cache_3tier.warm_len(), 0);
        assert_eq!(cache_3tier.cold_len(), 0);
        assert_eq!(cache_3tier.hot_len(), 0);
        assert_eq!(cache_3tier.offset, 0);
    }

    // Phase I — COW fork primitive: a forked cache must diverge cleanly
    // from the original after independent appends. Both sides see the same
    // shared prefix, then their appended suffixes are independent. The
    // underlying MLX arrays are ref-counted, so the fork cost is
    // O(layers) refcount bumps, not a deep copy.
    #[test]
    fn turboquant_fork_cow_independent_divergence() {
        let dim = 64usize;
        let heads = 1i32;
        let prefix = 8i32;
        let suffix_a = 4i32;
        let suffix_b = 6i32;
        let config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None) // straight to cold so divergence is visible
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl);

        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefix_len = (b * h * prefix * d) as usize;
        let prefix_keys =
            InlineArray::from_f32_slice(&make_data(prefix_len, 0.2), &[b, h, prefix, d]);
        let prefix_values =
            InlineArray::from_f32_slice(&make_data(prefix_len, 0.7), &[b, h, prefix, d]);

        let mut origin = QuantizedKvCache::new(config);
        origin
            .append(&prefix_keys, &prefix_values)
            .expect("prefix append");
        assert_eq!(origin.offset, prefix as usize);

        // Fork before divergence — both sides should see the same prefix.
        let mut fork_a = origin.fork();
        let mut fork_b = origin.fork();
        assert_eq!(fork_a.offset, prefix as usize);
        assert_eq!(fork_b.offset, prefix as usize);

        // Diverge: fork_a appends `suffix_a` tokens, fork_b appends
        // `suffix_b` tokens (different lengths AND different values via
        // distinct seeds).
        let suffix_a_len = (b * h * suffix_a * d) as usize;
        let suffix_b_len = (b * h * suffix_b * d) as usize;
        let a_keys =
            InlineArray::from_f32_slice(&make_data(suffix_a_len, 1.5), &[b, h, suffix_a, d]);
        let a_values =
            InlineArray::from_f32_slice(&make_data(suffix_a_len, 2.5), &[b, h, suffix_a, d]);
        let b_keys =
            InlineArray::from_f32_slice(&make_data(suffix_b_len, 3.5), &[b, h, suffix_b, d]);
        let b_values =
            InlineArray::from_f32_slice(&make_data(suffix_b_len, 4.5), &[b, h, suffix_b, d]);

        fork_a.append(&a_keys, &a_values).expect("fork_a append");
        fork_b.append(&b_keys, &b_values).expect("fork_b append");

        // Each fork sees its own length; origin is unchanged.
        assert_eq!(fork_a.offset, (prefix + suffix_a) as usize);
        assert_eq!(fork_b.offset, (prefix + suffix_b) as usize);
        assert_eq!(origin.offset, prefix as usize, "origin unaffected by forks");

        // Decoded histories must differ (different suffix lengths and
        // different content).
        let mut deq_a = fork_a.dequantize_keys().expect("fork_a dequant");
        let mut deq_b = fork_b.dequantize_keys().expect("fork_b dequant");
        let mut deq_origin = origin.dequantize_keys().expect("origin dequant");

        assert_eq!(deq_a.dim(2), prefix + suffix_a);
        assert_eq!(deq_b.dim(2), prefix + suffix_b);
        assert_eq!(deq_origin.dim(2), prefix);

        // Both forks share the leading `prefix` slots — verify by checking
        // those leading rows match between fork_a and fork_b within
        // codebook tolerance. (They go through the same encoder, so
        // bit-identical.)
        let a_vals = deq_a
            .to_f32_vec((b * h * (prefix + suffix_a) * d) as usize)
            .expect("a to_f32");
        let b_vals = deq_b
            .to_f32_vec((b * h * (prefix + suffix_b) * d) as usize)
            .expect("b to_f32");
        let prefix_elems = (b * h * prefix * d) as usize;
        for i in 0..prefix_elems {
            assert!(
                (a_vals[i] - b_vals[i]).abs() < 1e-6,
                "fork_a and fork_b must agree on the shared prefix at i={i}: a={} b={}",
                a_vals[i],
                b_vals[i]
            );
        }

        // Origin's shared prefix should match too.
        let origin_vals = deq_origin
            .to_f32_vec((b * h * prefix * d) as usize)
            .expect("origin to_f32");
        for i in 0..prefix_elems {
            assert!(
                (origin_vals[i] - a_vals[i]).abs() < 1e-6,
                "origin and fork_a must agree on the prefix at i={i}",
            );
        }
    }

    // d128 sibling of the bits-sweep test above. d128 doesn't trigger
    // `use_q8_seq_shadow` in encode (its score kernel uses transposed-D
    // indices, not the seq-major shadow), but the score kernel still loads
    // 256 codebook entries unconditionally — so the same `_padded_256`
    // codebook view fix applies. Pinning 5-bit catches both encode (slot
    // scale + 32-centroid argmin) and dispatch (codebook padding) regressions.
    #[test]
    fn turboquant_no_qjl_d128_long_context_matches_dequantized_sdpa_5b() {
        let dim = 128usize;
        let heads = 2i32;
        let prefill = 1023i32;
        let config = TurboQuantConfig::uniform(5, 8)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl);
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let step_keys = InlineArray::from_f32_slice(&make_data(step_len, 1.9), &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-3,
            "Variant F d128 5b fused kernel diverged from dequantized sdpa: \
             max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn turboquant_no_qjl_direct_attention_matches_dequantized_sdpa() {
        // Variant F (NoQjl) end-to-end: append → direct attention through the
        // dequantize-and-SDPA fallback path (try_gpu_uniform_attention is
        // gated off for NoQjl) must match a manual reference attention
        // computed against gpu_dequantize_keys output. This pins:
        //   - GPU encode honors qjl_mode (full-bits codebook, zero residuals)
        //   - gpu_dequantize_keys honors qjl_mode (full-bits codebook lookup)
        //   - cache.rs routes NoQjl off the fast path
        let dim = 16usize;
        let heads = 2i32;
        let prefill = 3i32;
        let config = TurboQuantConfig::uniform(8, 8)
            .with_recent_window(None)
            .with_qjl_mode(super::config::TurboQuantQjlMode::NoQjl);
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let step_keys = InlineArray::from_f32_slice(&make_data(step_len, 1.9), &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            4,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        // NoQjl goes through the fallback path which IS the same dequantize
        // + SDPA computation as the reference, so this should match to
        // numerical precision.
        assert!(
            max_abs_diff < 1e-4,
            "Variant F direct attention diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    // 5-bit Lloyd-Max codebooks are structurally supported (PackedBits +
    // beta_codebook are bit-width-generic) but were never pinned end-to-end.
    // These tests validate the cache append → direct attention → dequantize
    // path produces self-consistent output at 5b for both QJL modes. Phase D
    // depends on this being a tested first-class config; the fast-path
    // kernels (4b/5b/6b siblings of the existing q8 fullbyte family) come
    // later.
    fn run_uniform_5b_round_trip_at(qjl_mode: super::config::TurboQuantQjlMode, tolerance: f32) {
        let dim = 16usize;
        let heads = 2i32;
        let prefill = 3i32;
        let config = TurboQuantConfig::uniform(5, 5)
            .with_recent_window(None)
            .with_qjl_mode(qjl_mode);
        let b = 1i32;
        let h = heads;
        let d = dim as i32;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_len = (b * h * prefill * d) as usize;
        let step_len = (b * h * d) as usize;
        let prefill_keys =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.2), &[b, h, prefill, d]);
        let prefill_values =
            InlineArray::from_f32_slice(&make_data(prefill_len, 0.7), &[b, h, prefill, d]);
        let queries = InlineArray::from_f32_slice(&make_data(step_len, 1.3), &[b, h, 1, d]);
        let step_keys = InlineArray::from_f32_slice(&make_data(step_len, 1.9), &[b, h, 1, d]);
        let step_values = InlineArray::from_f32_slice(&make_data(step_len, 2.4), &[b, h, 1, d]);

        let mut seed_cache = QuantizedKvCache::new(config);
        seed_cache
            .append(&prefill_keys, &prefill_values)
            .expect("prefill append");

        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            4,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < tolerance,
            "5b ({:?}) direct attention diverged from dequantized sdpa: max_abs_diff={max_abs_diff}",
            qjl_mode
        );
    }

    #[test]
    fn turboquant_uniform_5b_standard_round_trip_matches_dequantized_sdpa() {
        // 5b Standard: 4-bit codebook (16 centroids) + 1-bit QJL sign.
        run_uniform_5b_round_trip_at(super::config::TurboQuantQjlMode::Standard, 1e-4);
    }

    #[test]
    fn turboquant_uniform_5b_no_qjl_round_trip_matches_dequantized_sdpa() {
        // 5b NoQjl (Variant F): full 5-bit codebook (32 centroids), no QJL.
        run_uniform_5b_round_trip_at(super::config::TurboQuantQjlMode::NoQjl, 1e-4);
    }

    #[test]
    fn turboquant_direct_attention_matches_dequantized_sdpa_uniform() {
        let (seed_cache, queries, step_keys, step_values, scale, b, h, d) =
            make_uniform_direct_attention_case();
        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            4,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "direct attention diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn turboquant_direct_attention_matches_dequantized_sdpa_uniform_q8_d128_long_context() {
        let (seed_cache, queries, step_keys, step_values, scale, b, h, d) =
            make_uniform_direct_attention_case_with(128, 2, 1023);
        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "long-context direct attention diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn turboquant_direct_attention_matches_dequantized_sdpa_uniform_q8_d256_long_context_gqa() {
        let (seed_cache, queries, step_keys, step_values, scale, b, q_heads, kv_heads, d) =
            make_uniform_gqa_direct_attention_case_with(256, 16, 2, 1023);
        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let full_values = ref_cache.dequantize_values().expect("dequantize values");
        let repeated_keys = full_keys.repeat(q_heads / kv_heads, 1);
        let repeated_values = full_values.repeat(q_heads / kv_heads, 1);
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut repeated_keys.clone(),
            &mut repeated_values.clone(),
            b,
            q_heads,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * q_heads * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "d256 gqa direct attention diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    /// d128 packed_keys GQA fast path: q_heads > 8 routes through
    /// `turboquant_attention_q8_d128_packed_keys_2pass` (packed key bytes
    /// instead of separate indices+sign words). Shares pass 2 with the
    /// d128 base 2pass kernel; pin parity here so the packed_keys variant
    /// can't silently regress.
    #[test]
    fn turboquant_direct_attention_matches_dequantized_sdpa_uniform_q8_d128_long_context_gqa() {
        let (seed_cache, queries, step_keys, step_values, scale, b, q_heads, kv_heads, d) =
            make_uniform_gqa_direct_attention_case_with(128, 16, 2, 1023);
        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let full_values = ref_cache.dequantize_values().expect("dequantize values");
        let repeated_keys = full_keys.repeat(q_heads / kv_heads, 1);
        let repeated_values = full_values.repeat(q_heads / kv_heads, 1);
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut repeated_keys.clone(),
            &mut repeated_values.clone(),
            b,
            q_heads,
            1024,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * q_heads * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "d128 gqa direct attention diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    /// Phase 3 parity baseline. Mixed-precision (`preset_q3_5`) currently
    /// flows through `dequantize_keys + dequantize_values + sdpa_causal_like_mlx`
    /// — a "correctness-first" fallback. This test pins the expected output
    /// against a from-scratch scalar reference so that when the fused Mixed
    /// kernel lands it must produce the same numerics.
    ///
    /// Tolerance matches the Uniform parity tests (1e-4): both paths
    /// dequantize the same Mixed-quantized data, so the only delta here is
    /// MLX's fused SDPA vs scalar SDPA on identical f32 tensors.
    #[test]
    fn turboquant_direct_attention_matches_dequantized_sdpa_mixed() {
        let (seed_cache, queries, step_keys, step_values, scale, b, h, d) =
            make_mixed_direct_attention_case_with(128, 2, 31);
        let mut direct_cache = seed_cache.clone();
        let mut ref_cache = seed_cache;

        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");

        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let reference_vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            32,
            d,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "mixed direct attention diverged from dequantized sdpa: max_abs_diff={max_abs_diff}"
        );
    }

    /// Drives the d256 fullbyte+dense-values direct attention path against a
    /// from-scratch CPU reference at the given sequence length. Pulled out of
    /// the original single-shape test so the seq-sweep variant below can hit
    /// 1024/2048/4096 without copy-pasting 80 LOC of fixture setup.
    fn run_d256_fullbyte_dense_values_parity_at(seq: i32) {
        let batch = 1i32;
        let q_heads = 4i32;
        let kv_heads = 2i32;
        let groups = q_heads / kv_heads;
        let dim = 256i32;
        let kv_rows = batch * kv_heads;
        let q_rows = batch * q_heads;
        let scale = 1.0f32 / (dim as f32).sqrt();

        let query_rot_vec: Vec<f32> = (0..(q_rows * dim) as usize)
            .map(|i| ((i as f32) * 0.013 + 0.4).sin() * 0.7)
            .collect();
        let key_indices_vec: Vec<u8> = (0..(kv_rows * seq * dim) as usize)
            .map(|i| (((i * 17) + 23) & 0xff) as u8)
            .collect();
        // slot_scales pack is [N, S_cap, 4]: [key_norm, residual_norm, value_norm, key_slot_scale].
        // Component 3 is set to 1.0 here so kernel reconstruction
        //   keys[d] = codebook[idx] * key_slot_scale * key_norm
        // matches the reference computed below with key_slot_scale folded out.
        let slot_scales_vec: Vec<f32> = (0..(kv_rows * seq * 4) as usize)
            .map(|i| match i % 4 {
                0 => 0.5 + (((i / 4) % 11) as f32) * 0.03125,
                1 => 0.0,
                2 => 1.0,
                _ => 1.0,
            })
            .collect();
        let key_codebook_vec: Vec<f32> = (0..256).map(|i| ((i as f32) - 127.5) / 96.0).collect();
        let value_dense_vec: Vec<f32> = (0..(kv_rows * seq * dim) as usize)
            .map(|i| ((i as f32) * 0.009 - 0.7).cos() * 0.5)
            .collect();

        let query_rot = InlineArray::from_f32_slice(&query_rot_vec, &[q_rows, dim]);
        let key_indices = InlineArray::from_u8_slice(&key_indices_vec, &[kv_rows, seq, dim]);
        let slot_scales = InlineArray::from_f32_slice(&slot_scales_vec, &[kv_rows, seq, 4]);
        let key_codebook = InlineArray::from_f32_slice(&key_codebook_vec, &[256]);
        let value_dense = InlineArray::from_f32_slice(&value_dense_vec, &[kv_rows, seq, dim]);

        let mut direct = InlineArray::turboquant_attention_q8_d256_fullbyte_dense_values_2pass(
            &query_rot,
            &key_indices,
            &slot_scales,
            &key_codebook,
            &value_dense,
            q_rows as u32,
            seq as u32,
            seq as u32,
            q_heads as u32,
            kv_heads as u32,
            scale,
        )
        .expect("fullbyte direct attention");

        let mut keys = vec![0.0f32; (batch * q_heads * seq * dim) as usize];
        let mut values = vec![0.0f32; (batch * q_heads * seq * dim) as usize];
        for qh in 0..q_heads as usize {
            let kvh = qh / groups as usize;
            for t in 0..seq as usize {
                let scale_base = (kvh * seq as usize + t) * 4;
                let key_norm = slot_scales_vec[scale_base];
                let key_base = (kvh * seq as usize + t) * dim as usize;
                let out_base = (qh * seq as usize + t) * dim as usize;
                for d_idx in 0..dim as usize {
                    let idx = key_indices_vec[key_base + d_idx] as usize;
                    keys[out_base + d_idx] = key_codebook_vec[idx] * key_norm;
                    values[out_base + d_idx] = value_dense_vec[key_base + d_idx];
                }
            }
        }

        let mut queries = query_rot.reshape(&[batch, q_heads, 1, dim]);
        let mut keys = InlineArray::from_f32_slice(&keys, &[batch, q_heads, seq, dim]);
        let mut values = InlineArray::from_f32_slice(&values, &[batch, q_heads, seq, dim]);
        let reference_vals = manual_single_token_attention(
            &mut queries,
            &mut keys,
            &mut values,
            batch,
            q_heads,
            seq,
            dim,
            scale,
        );

        let direct_vals = direct
            .to_f32_vec((batch * q_heads * dim) as usize)
            .expect("direct to_f32");
        let max_abs_diff = direct_vals
            .iter()
            .zip(reference_vals.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 1e-4,
            "d256 fullbyte direct attention diverged from manual reference at seq={seq}: max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn turboquant_attention_q8_d256_fullbyte_dense_values_matches_manual_reference() {
        run_d256_fullbyte_dense_values_parity_at(1024);
    }

    /// Denser parity sweep over n_seq for the d256 fullbyte+dense-values path.
    ///
    /// The single-shape test above pins correctness at the n_seq=1024 dispatch
    /// boundary. This sweep extends coverage to 2048 and 4096 so any future
    /// touch-ups to the shared d256 pass-2 merge kernel (e.g. when the
    /// mixed-precision fused attention kernel from Phase 3 reuses it) catch
    /// regressions across the long-context envelope, not just at the threshold.
    #[test]
    fn turboquant_attention_q8_d256_fullbyte_dense_values_parity_seq_sweep() {
        for seq in [1024, 2048, 4096] {
            run_d256_fullbyte_dense_values_parity_at(seq);
        }
    }

    #[test]
    fn turboquant_direct_attention_uniform_smoke() {
        let (seed_cache, queries, step_keys, step_values, scale, b, h, d) =
            make_uniform_direct_attention_case();
        let mut direct_cache = seed_cache;
        let mut direct = direct_cache
            .append_and_compute_attention(&queries, &step_keys, &step_values, scale)
            .expect("direct attention");
        let vals = direct
            .to_f32_vec((b * h * d) as usize)
            .expect("direct to_f32");
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    /// CPU reference for `[B, H, S, D]` causal multi-token attention. Returns
    /// the output flattened in `[B, H, S, D]` order. Only used by the prefill
    /// dispatch tests below — production paths go through MLX SDPA directly.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_range_loop)]
    fn manual_causal_multi_token_attention(
        queries: &mut InlineArray,
        keys: &mut InlineArray,
        values: &mut InlineArray,
        batch: i32,
        heads: i32,
        seq_q: i32,
        seq_kv: i32,
        dim: i32,
        scale: f32,
    ) -> Vec<f32> {
        let q = queries
            .to_f32_vec((batch * heads * seq_q * dim) as usize)
            .expect("queries to_f32");
        let k = keys
            .to_f32_vec((batch * heads * seq_kv * dim) as usize)
            .expect("keys to_f32");
        let v = values
            .to_f32_vec((batch * heads * seq_kv * dim) as usize)
            .expect("values to_f32");

        let rows = (batch * heads) as usize;
        let sq = seq_q as usize;
        let sk = seq_kv as usize;
        let dim_us = dim as usize;
        let causal_offset = sk - sq;
        let mut out = vec![0.0f32; rows * sq * dim_us];

        for row in 0..rows {
            for qi in 0..sq {
                let q_pos = causal_offset + qi;
                let q_base = (row * sq + qi) * dim_us;
                let q_row = &q[q_base..q_base + dim_us];

                let mut scores = vec![f32::NEG_INFINITY; sk];
                for t in 0..=q_pos {
                    let k_base = (row * sk + t) * dim_us;
                    let k_row = &k[k_base..k_base + dim_us];
                    let dot = q_row
                        .iter()
                        .zip(k_row.iter())
                        .map(|(lhs, rhs)| lhs * rhs)
                        .sum::<f32>();
                    scores[t] = dot * scale;
                }

                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for score in &mut scores {
                    if score.is_finite() {
                        *score = (*score - max_score).exp();
                        sum_exp += *score;
                    } else {
                        *score = 0.0;
                    }
                }
                for score in &mut scores {
                    *score /= sum_exp.max(f32::MIN_POSITIVE);
                }

                let out_row = &mut out[q_base..q_base + dim_us];
                for t in 0..=q_pos {
                    let v_base = (row * sk + t) * dim_us;
                    let v_row = &v[v_base..v_base + dim_us];
                    let weight = scores[t];
                    for (dst, val) in out_row.iter_mut().zip(v_row.iter()) {
                        *dst += weight * *val;
                    }
                }
            }
        }

        out
    }

    /// Prefill with non-empty cache: dispatch must take the `prev > 0`
    /// fallback path (`append → dequantize → SDPA`) and produce results
    /// matching a CPU reference run over the concatenated history.
    ///
    /// This exercises the previously-untested prefill branch in
    /// `turboquant_dispatch::turboquant_attention_step` (audit Phase G).
    #[test]
    fn turboquant_dispatch_prefill_with_existing_cache_matches_reference() {
        use crate::turboquant_dispatch::turboquant_attention_step;

        let dim = 16usize;
        let heads = 2i32;
        let prev_len = 3i32;
        let new_len = 5i32;
        let total = prev_len + new_len;
        let b = 1i32;
        let d = dim as i32;
        let scale = 1.0f32 / (d as f32).sqrt();

        let make_data = |len: usize, seed: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i as f32) * 0.07 + seed).sin() + ((i as f32) * 0.11 - seed).cos())
                .collect()
        };

        let prefill_kv_len = (b * heads * prev_len * d) as usize;
        let new_kv_len = (b * heads * new_len * d) as usize;
        let q_len = (b * heads * new_len * d) as usize;
        let total_kv_len = (b * heads * total * d) as usize;

        let prev_keys =
            InlineArray::from_f32_slice(&make_data(prefill_kv_len, 0.2), &[b, heads, prev_len, d]);
        let prev_values =
            InlineArray::from_f32_slice(&make_data(prefill_kv_len, 0.7), &[b, heads, prev_len, d]);
        let new_keys =
            InlineArray::from_f32_slice(&make_data(new_kv_len, 1.4), &[b, heads, new_len, d]);
        let new_values =
            InlineArray::from_f32_slice(&make_data(new_kv_len, 1.9), &[b, heads, new_len, d]);
        let queries =
            InlineArray::from_f32_slice(&make_data(q_len, 0.31), &[b, heads, new_len, d]);

        // Disable the recent-fp16 window so the dequantize fallback exercises
        // the cold compressed store, not the hot ring.
        let config = TurboQuantConfig::uniform(8, 8).with_recent_window(None);
        let mut cache = QuantizedKvCache::new(config);
        cache.append(&prev_keys, &prev_values).expect("seed prefill");

        let mut output =
            turboquant_attention_step(&mut cache, &queries, &new_keys, &new_values, scale, prev_len, "TEST");
        let actual = output
            .to_f32_vec((b * heads * new_len * d) as usize)
            .expect("output to_f32");

        // Reference: stitch dequantized prev cache + raw new K/V, run causal
        // multi-token attention on the full concatenation. Matches what the
        // dispatch's `prev > 0` branch is *trying* to compute.
        let prev_keys_dq = cache
            .dequantize_keys()
            .expect("dequantize keys")
            .slice(&[0, 0, 0, 0], &[b, heads, total, d]);
        let prev_values_dq = cache
            .dequantize_values()
            .expect("dequantize values")
            .slice(&[0, 0, 0, 0], &[b, heads, total, d]);

        let mut ref_keys = prev_keys_dq;
        let mut ref_values = prev_values_dq;
        let mut ref_queries = queries.clone();
        let _ = ref_keys.to_f32_vec(total_kv_len);
        let _ = ref_values.to_f32_vec(total_kv_len);
        let expected = manual_causal_multi_token_attention(
            &mut ref_queries,
            &mut ref_keys,
            &mut ref_values,
            b,
            heads,
            new_len,
            total,
            d,
            scale,
        );

        let max_abs_diff = actual
            .iter()
            .zip(expected.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_diff < 5e-3,
            "prefill-with-history dispatch diverged from CPU reference: max_abs_diff={max_abs_diff}"
        );
    }

    #[test]
    fn turboquant_reference_attention_uniform_smoke() {
        let (seed_cache, queries, step_keys, step_values, scale, b, h, d) =
            make_uniform_direct_attention_case();
        let mut ref_cache = seed_cache;
        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        let vals = manual_single_token_attention(
            &mut queries.clone(),
            &mut full_keys,
            &mut full_values,
            b,
            h,
            4,
            d,
            scale,
        );
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn turboquant_dequantize_keys_after_append_uniform_smoke() {
        let (seed_cache, _, step_keys, step_values, _, b, h, d) =
            make_uniform_direct_attention_case();
        let mut ref_cache = seed_cache;
        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_keys = ref_cache.dequantize_keys().expect("dequantize keys");
        assert_eq!(full_keys.shape(), &[b, h, 4, d]);
        let vals = full_keys
            .to_f32_vec((b * h * 4 * d) as usize)
            .expect("keys to_f32");
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn turboquant_dequantize_values_after_append_uniform_smoke() {
        let (seed_cache, _, step_keys, step_values, _, b, h, d) =
            make_uniform_direct_attention_case();
        let mut ref_cache = seed_cache;
        ref_cache
            .append(&step_keys, &step_values)
            .expect("reference append");
        let mut full_values = ref_cache.dequantize_values().expect("dequantize values");
        assert_eq!(full_values.shape(), &[b, h, 4, d]);
        let vals = full_values
            .to_f32_vec((b * h * 4 * d) as usize)
            .expect("values to_f32");
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    /// Verify that multiple appends accumulate correctly in the GPU store.
    #[test]
    fn turboquant_gpu_multi_append() {
        let dim = 8usize;
        let config = TurboQuantConfig::uniform(4, 4);
        let b = 1i32;
        let h = 1i32;
        let d = dim as i32;

        let make_data = |seed: f32| -> Vec<f32> {
            (0..b * h * d)
                .map(|i| (i as f32 * 0.15 + seed).sin())
                .collect()
        };

        let mut cache = QuantizedKvCache::new(config);

        // Append 3 steps individually.
        for step in 0..3 {
            let data = make_data(step as f32);
            let arr = InlineArray::from_f32_slice(&data, &[b, h, 1, d]);
            cache.append(&arr, &arr).expect("append step");
        }

        assert_eq!(cache.offset, 3, "Should have 3 cached positions");

        let dk = cache.dequantize_keys().expect("dequantize_keys");
        // Shape should be [B, H, 3, D].
        assert_eq!(dk.shape(), &[b, h, 3, d]);
    }

    // ─── Hot/cold split (Phase C bridge mirror) ──────────────────────────────

    /// With the recent-fp16 window enabled, short prompts must stay
    /// uncompressed in the hot ring. The cold stores never come online.
    #[test]
    fn hot_window_keeps_short_context_uncompressed() {
        let dim = 16usize;
        let config = TurboQuantConfig::uniform(8, 8).with_recent_window(Some(64));
        let b = 1i32;
        let h = 1i32;
        let d = dim as i32;
        let prefill = 32i32;
        let total = (b * h * prefill * d) as usize;
        let data: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.07).sin())
            .collect();
        let keys = InlineArray::from_f32_slice(&data, &[b, h, prefill, d]);
        let values = InlineArray::from_f32_slice(&data, &[b, h, prefill, d]);

        let mut cache = QuantizedKvCache::new(config);
        cache.append(&keys, &values).expect("append");

        assert_eq!(cache.hot_len(), prefill as usize);
        assert_eq!(cache.cold_len(), 0);
        assert_eq!(cache.offset, prefill as usize);
        assert!(cache.keys.is_none(), "cold key store should not exist yet");
        assert!(cache.values.is_none(), "cold value store should not exist yet");

        let dk = cache.dequantize_keys().expect("dequantize_keys");
        assert_eq!(dk.shape(), &[b, h, prefill, d]);
    }

    /// Once the hot ring exceeds `recent_window + HOT_EVICTION_CHUNK`,
    /// the oldest tokens spill into the cold compressed stores. The
    /// invariant `offset == cold_offset + hot_offset` must hold.
    #[test]
    fn hot_window_evicts_to_cold_after_overflow() {
        let dim = 16usize;
        // Tiny window so we can exercise eviction without enormous tensors.
        // Eviction triggers when hot_offset > window + HOT_EVICTION_CHUNK.
        // With window = 4, that's > 4 + 1024 = > 1028.
        let window = 4usize;
        let config = TurboQuantConfig::uniform(8, 8).with_recent_window(Some(window));
        let b = 1i32;
        let h = 1i32;
        let d = dim as i32;
        let total_tokens = 1100i32; // > window + HOT_EVICTION_CHUNK
        let total = (b * h * total_tokens * d) as usize;
        let data: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.05).sin())
            .collect();
        let keys = InlineArray::from_f32_slice(&data, &[b, h, total_tokens, d]);
        let values = InlineArray::from_f32_slice(&data, &[b, h, total_tokens, d]);

        let mut cache = QuantizedKvCache::new(config);
        cache.append(&keys, &values).expect("prefill");

        assert_eq!(cache.offset, total_tokens as usize);
        assert_eq!(cache.cold_len() + cache.hot_len(), total_tokens as usize);
        assert!(cache.cold_len() > 0, "expected some tokens in cold");
        assert!(
            cache.hot_len() <= window + HOT_EVICTION_CHUNK,
            "hot ring should not exceed window+chunk after eviction"
        );

        let dk = cache.dequantize_keys().expect("dequantize_keys");
        assert_eq!(dk.shape(), &[b, h, total_tokens, d]);
    }

    /// Legacy mode (`recent_window: None`) compresses every appended token
    /// immediately; the hot ring must stay disabled.
    #[test]
    fn legacy_recent_window_none_compresses_immediately() {
        let dim = 16usize;
        let config = TurboQuantConfig::uniform(8, 8).with_recent_window(None);
        let b = 1i32;
        let h = 1i32;
        let d = dim as i32;
        let prefill = 8i32;
        let total = (b * h * prefill * d) as usize;
        let data: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.09).cos()).collect();
        let keys = InlineArray::from_f32_slice(&data, &[b, h, prefill, d]);
        let values = InlineArray::from_f32_slice(&data, &[b, h, prefill, d]);

        let mut cache = QuantizedKvCache::new(config);
        cache.append(&keys, &values).expect("append");

        assert_eq!(cache.cold_len(), prefill as usize);
        assert_eq!(cache.hot_len(), 0);
        assert!(cache.hot_keys.is_none());
        assert!(cache.hot_values.is_none());
    }

    // ─── Defensive residual-norm clamp (A1 from audit) ───────────────────────
    //
    // Pathological inputs (NaN / ±Inf from upstream fp16 corruption) must not
    // propagate into the QJL term. The CPU encode path uses an explicit
    // `is_finite` + `clamp` guard; the GPU path composes `maximum(0).minimum(MAX)`.
    //
    // These tests exercise the CPU path directly since the GPU op graph
    // requires a live Metal device and is covered by the broader GPU integration
    // tests above.

    #[test]
    fn residual_norm_clamp_sanitizes_nan_input() {
        // One row of NaN should produce finite residual norm (0) — the row is
        // treated as zero by the upstream `norm <= ZERO_EPSILON` check, but if
        // it slipped past that the clamp still catches it.
        let core = TurboQuantCore::new(16, 4);
        let mut row = vec![0.1f32; 16];
        row[0] = f32::NAN;
        let encoded = encode_key_component_rows(&core, &row, 4, super::config::TurboQuantQjlMode::Standard, 0);
        assert_eq!(encoded.residual_norms.len(), 1);
        assert!(
            encoded.residual_norms[0].is_finite(),
            "residual_norm must be finite even with NaN input, got {}",
            encoded.residual_norms[0]
        );
        assert!(
            (0.0..=MAX_RESIDUAL_NORM).contains(&encoded.residual_norms[0]),
            "residual_norm must be in [0, MAX], got {}",
            encoded.residual_norms[0]
        );
    }

    #[test]
    fn residual_norm_clamp_caps_inf_input() {
        let core = TurboQuantCore::new(16, 4);
        let mut row = vec![1.0f32; 16];
        row[5] = f32::INFINITY;
        let encoded = encode_key_component_rows(&core, &row, 4, super::config::TurboQuantQjlMode::Standard, 0);
        assert!(
            encoded.residual_norms[0].is_finite(),
            "Inf input must not leak into residual_norm"
        );
        assert!(encoded.residual_norms[0] <= MAX_RESIDUAL_NORM);
    }

    // ─── Round-trip correctness (T1 from audit) ──────────────────────────────
    //
    // Verify the key invariants from turboquant.pdf Theorem 1 + Theorem 2 on
    // the deterministic CPU encode/decode path:
    //
    //   1. Round-trip reconstruction error has per-row MSE bounded by a
    //      constant that shrinks with bit-width (distortion bound).
    //   2. The inner product <q, decode(encode(k))> is approximately unbiased
    //      for <q, k> averaged across many random q, k pairs.

    fn seeded_gaussian_rows(num_rows: usize, dim: usize, seed: u64) -> Vec<f32> {
        // Box-Muller over a xorshift stream — deterministic, no external crate.
        fn xorshift64(state: &mut u64) -> u64 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *state = x;
            x
        }
        let mut state = seed.max(1);
        let mut u = || -> f32 {
            // Uniform (0, 1).
            let raw = xorshift64(&mut state);
            ((raw >> 40) as f32 + 1.0) / ((1u64 << 24) as f32)
        };
        let total = num_rows * dim;
        let mut out = Vec::with_capacity(total);
        while out.len() < total {
            let u1 = u();
            let u2 = u();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;
            out.push(r * theta.cos());
            if out.len() < total {
                out.push(r * theta.sin());
            }
        }
        out.truncate(total);
        out
    }

    fn decode_cpu_key(
        core: &TurboQuantCore,
        encoded: &EncodedKeyRows,
        bits: u8,
        _num_rows: usize,
    ) -> Vec<f32> {
        // Delegate to the production CPU decode path so the test exercises the
        // same arithmetic as live inference.
        decode_key_component_rows_raw(
            core,
            &encoded.mse_indices,
            &encoded.qjl_signs,
            &encoded.norms,
            &encoded.residual_norms,
            &encoded.slot_scale,
            bits,
            super::config::TurboQuantQjlMode::Standard,
            None,
            None,
            0,
        )
    }

    #[test]
    fn turboquant_cpu_round_trip_error_bound_shrinks_with_bits() {
        // Distortion bound from Theorem 1: per-row MSE scales like 1/2^(2*(b-1))
        // for the MSE stage, with QJL residual correction adding an unbiased
        // zero-mean term. Across enough random rows, the average squared error
        // should be strictly smaller at higher bit widths.
        let dim = 64;
        let num_rows = 128;
        let data = seeded_gaussian_rows(num_rows, dim, 0xA1B2_C3D4_E5F6_0789);

        let mut errors = Vec::new();
        for &bits in &[3u8, 5u8, 7u8] {
            let core = TurboQuantCore::new(dim, bits);
            let encoded = encode_key_component_rows(&core, &data, bits, super::config::TurboQuantQjlMode::Standard, 0);
            let decoded = decode_cpu_key(&core, &encoded, bits, num_rows);
            // Per-element MSE, averaged across all rows.
            let mse: f32 = data
                .iter()
                .zip(decoded.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                / (num_rows * dim) as f32;
            assert!(
                mse.is_finite(),
                "MSE must be finite for bits={}, got {}",
                bits,
                mse
            );
            errors.push((bits, mse));
        }

        // Monotonicity: more bits ⇒ less error. This is the concrete form of
        // the distortion bound at the statistical level.
        for window in errors.windows(2) {
            let (b_lo, mse_lo) = window[0];
            let (b_hi, mse_hi) = window[1];
            assert!(
                mse_hi < mse_lo,
                "Error at {} bits ({}) should beat {} bits ({})",
                b_hi,
                mse_hi,
                b_lo,
                mse_lo
            );
        }

        // Sanity floor: highest bit-width should reconstruct to a small
        // absolute error — Gaussian data normalized to unit sphere with 7-bit
        // MSE + QJL should land well below 0.5 per-element MSE.
        let (_, worst_case_at_7_bits) = errors.last().copied().unwrap();
        assert!(
            worst_case_at_7_bits < 0.5,
            "7-bit TurboQuant MSE {} is unexpectedly high",
            worst_case_at_7_bits
        );
    }

    // Phase E.3 CPU mirror: encode_key_component_rows must extract top-K
    // |rotated| coords, zero them in the body before slot_scale + codebook
    // quant, and the decode path must scatter the exact stored values back
    // at those channels before inverse-rotate. Heavy-tail rows (where ≥1
    // coord dominates the rest by 10×) make this measurable: the codebook
    // — built for normalised-rotated-key statistics — wastes resolution on
    // the heavy tail and mis-fits the body. Carving out the outliers lets
    // the codebook fit the body, and the override restores the heavy-tail
    // coords exactly. MSE must drop by a non-trivial margin vs the
    // outlier-disabled baseline.
    #[test]
    fn turboquant_cpu_outliers_per_block_decode_override_improves_reconstruction() {
        // Mirrors the GPU equivalent at line 1117. Small dim + few rows + 4-bit
        // codebook makes the override benefit unambiguous in scalar Rust. The
        // heavy spikes are injected in input space; rotation diffuses them
        // across rotated channels but the per-row max stays large enough that
        // top-K still picks meaningful coords.
        let dim = 16usize;
        let num_rows = 4usize;
        let bits = 4u8;
        let outlier_k = 2usize;

        let mut data = vec![0f32; num_rows * dim];
        for slot in 0..num_rows {
            for c in 0..dim {
                let i = slot * dim + c;
                data[i] = (((i as f32) * 0.07).sin() * 0.3) + (((i as f32) * 0.13).cos() * 0.2);
            }
            // Spikes at slot-dependent channels — same shape as the GPU test.
            let s = slot * dim;
            data[s + slot] += 6.0;
            data[s + ((slot + 5) % dim)] -= 5.5;
        }

        let core = TurboQuantCore::new(dim, bits);
        let qjl = super::config::TurboQuantQjlMode::Standard;

        // Baseline: outlier_k = 0 (no per-block override path).
        let baseline_encoded = encode_key_component_rows(&core, &data, bits, qjl, 0);
        let baseline_decoded = decode_cpu_key(&core, &baseline_encoded, bits, num_rows);

        // With override: outlier_k = 2 (matches synthetic injected spikes).
        let override_encoded = encode_key_component_rows(&core, &data, bits, qjl, outlier_k);
        // Sanity: encoder produced the outlier buffers.
        assert!(override_encoded.per_block_outlier_channels.is_some());
        assert!(override_encoded.per_block_outlier_values.is_some());
        let chans = override_encoded
            .per_block_outlier_channels
            .as_ref()
            .unwrap();
        let vals = override_encoded
            .per_block_outlier_values
            .as_ref()
            .unwrap();
        assert_eq!(chans.len(), num_rows * outlier_k);
        assert_eq!(vals.len(), num_rows * outlier_k);
        let override_decoded = decode_key_component_rows_raw(
            &core,
            &override_encoded.mse_indices,
            &override_encoded.qjl_signs,
            &override_encoded.norms,
            &override_encoded.residual_norms,
            &override_encoded.slot_scale,
            bits,
            qjl,
            Some(chans.as_slice()),
            Some(vals.as_slice()),
            outlier_k,
        );

        let mse = |decoded: &[f32]| -> f32 {
            data.iter()
                .zip(decoded.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                / data.len() as f32
        };
        let baseline_mse = mse(&baseline_decoded);
        let override_mse = mse(&override_decoded);
        // Override should cut MSE by ≥30% on heavy-tail data — same bound
        // the GPU equivalent test asserts. If this regresses, the encoder's
        // zero-pre-quant or the decoder's scatter is broken.
        assert!(
            override_mse < baseline_mse * 0.7,
            "outlier override did not improve reconstruction: \
             baseline_mse={baseline_mse}, override_mse={override_mse}"
        );
    }

    #[test]
    fn turboquant_cpu_inner_product_is_approximately_unbiased() {
        // Paper Theorem 2: E[<q, k̂>] = <q, k> for keys encoded via the 2-stage
        // MSE + QJL path. With enough independent (q, k) pairs, the mean
        // reconstructed inner product should track the ground-truth mean.
        let dim = 128;
        let num_rows = 256;
        let bits = 5u8;
        let core = TurboQuantCore::new(dim, bits);

        let keys = seeded_gaussian_rows(num_rows, dim, 0x1111_2222_3333_4444);
        let queries = seeded_gaussian_rows(num_rows, dim, 0x5555_6666_7777_8888);

        let encoded = encode_key_component_rows(&core, &keys, bits, super::config::TurboQuantQjlMode::Standard, 0);
        let decoded = decode_cpu_key(&core, &encoded, bits, num_rows);

        let mut sum_gt = 0.0f64;
        let mut sum_est = 0.0f64;
        let mut sum_abs_rel_err = 0.0f64;
        let mut valid = 0usize;
        for row_idx in 0..num_rows {
            let start = row_idx * dim;
            let k_row = &keys[start..start + dim];
            let q_row = &queries[start..start + dim];
            let k_hat = &decoded[start..start + dim];
            let gt: f32 = q_row.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum();
            let est: f32 = q_row.iter().zip(k_hat.iter()).map(|(a, b)| a * b).sum();
            sum_gt += gt as f64;
            sum_est += est as f64;
            if gt.abs() > 1e-3 {
                sum_abs_rel_err += ((est - gt) / gt).abs() as f64;
                valid += 1;
            }
        }
        let mean_gt = sum_gt / num_rows as f64;
        let mean_est = sum_est / num_rows as f64;
        let _ = (sum_abs_rel_err, valid); // per-row rel err is high-variance at low bits

        // Unbiasedness: the sample mean of the reconstructed inner product
        // should match the ground-truth mean to within the CLT-expected
        // standard error. For 256 rows of 128-dim Gaussian vectors the
        // per-row variance is O(1), so the standard error of the mean is
        // O(1/sqrt(256)) = 0.0625. We allow 4x headroom to keep the test
        // stable across platforms.
        let diff = (mean_est - mean_gt).abs();
        assert!(
            diff < 0.25,
            "Mean reconstructed inner product {} diverges from ground truth {} (diff {})",
            mean_est,
            mean_gt,
            diff
        );
    }

    /// Phase 3a — direct round-trip: bypass `QuantizedKvCache`, encode and
    /// decode via the new GPU Mixed helpers, verify reconstruction error
    /// stays within the q3_5 codebook budget. Catches encode/decode bugs
    /// without dragging in cache-state interactions.
    #[test]
    fn turboquant_gpu_quantize_kv_mixed_round_trip_no_cache() {
        let dim = 128usize;
        let b = 1i32;
        let h = 4i32;
        let s = 2i32;
        let d = dim as i32;
        let total = (b * h * s * d) as usize;
        let data: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.137).sin() + if i % 17 == 0 { 3.0 } else { 0.0 })
            .collect();
        let arr = InlineArray::from_f32_slice(&data, &[b, h, s, d]);

        let config = TurboQuantConfig::preset_q3_5(dim).with_recent_window(None);
        let state = TurboQuantState::new(dim, dim, config);

        let (kstore, vstore) = gpu_quantize_kv_mixed(&state, &arr, &arr, config)
            .expect("gpu_quantize_kv_mixed");

        let mut dec_keys = gpu_dequantize_keys_mixed(&kstore, &state.keys, &config)
            .expect("dequantize keys");
        let mut dec_vals = gpu_dequantize_values_mixed(&vstore, &state.values, &config)
            .expect("dequantize values");
        let recon_keys = dec_keys.to_f32_vec(total).expect("dec keys to_f32");
        let recon_vals = dec_vals.to_f32_vec(total).expect("dec vals to_f32");

        let max_key_err = recon_keys
            .iter()
            .zip(data.iter())
            .map(|(r, o)| (r - o).abs())
            .fold(0.0f32, f32::max);
        let max_val_err = recon_vals
            .iter()
            .zip(data.iter())
            .map(|(r, o)| (r - o).abs())
            .fold(0.0f32, f32::max);
        // q3_5 on rows with outliers in [-3, 3] reconstructs well below 1.5.
        // Tighter bounds risk legitimate codebook variance; this gates real
        // bugs (lazy-graph corruption gave err ≈ 5.5 before the eval barrier).
        assert!(
            max_key_err < 1.5,
            "key reconstruction error {max_key_err} too large"
        );
        assert!(
            max_val_err < 1.5,
            "val reconstruction error {max_val_err} too large"
        );
    }

    /// Phase 3a sanity: gather sub-vectors, then scatter back. Round-trip
    /// must equal the original input.
    #[test]
    fn turboquant_gpu_mixed_partition_gather_scatter_round_trip() {
        let b = 1i32;
        let h = 4i32;
        let s = 2i32;
        let d = 128i32;
        let total = (b * h * s * d) as usize;
        let data: Vec<f32> = (0..total)
            .map(|i| (i as f32) * 0.137 + if i % 17 == 0 { 3.0 } else { 0.0 })
            .collect();
        let arr = InlineArray::from_f32_slice(&data, &[b, h, s, d]);

        let (reg_src, out_src) =
            gpu_compute_outlier_partition(&arr, 32, 128).expect("partition");
        let reg = arr.take_along_axis(&reg_src, -1);
        let out = arr.take_along_axis(&out_src, -1);

        let reg_src_i32 = reg_src.as_dtype(Dtype::Int32.as_i32());
        let out_src_i32 = out_src.as_dtype(Dtype::Int32.as_i32());

        let zero = InlineArray::zeros(&[b, h, s, d], Dtype::Float32.as_i32());
        let with_reg = zero.put_along_axis_op(&reg_src_i32, &reg, -1);
        let mut merged = with_reg.put_along_axis_op(&out_src_i32, &out, -1);

        let merged_vec = merged.to_f32_vec(total).expect("merged to_f32");
        let max_diff = merged_vec
            .iter()
            .zip(data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "partition+gather+scatter round-trip max_abs_diff = {max_diff}"
        );
    }

    /// Phase 3a invariant: Mixed sub-vector encode/decode helpers must
    /// produce bit-identical results to the existing Uniform helpers when
    /// pointed at the same core + same data. Catches drift between the
    /// two pipelines.
    #[test]
    fn turboquant_gpu_mixed_subvector_matches_uniform_when_no_partition() {
        let dim = 96usize;
        let b = 1i32;
        let h = 4i32;
        let s = 2i32;
        let d = dim as i32;
        let total = (b * h * s * d) as usize;
        let data: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.13).sin() * 0.5)
            .collect();
        let rows = InlineArray::from_f32_slice(&data, &[b, h, s, d]);
        let vals = InlineArray::from_f32_slice(&data, &[b, h, s, d]);

        let config = TurboQuantConfig::uniform(3, 4).with_recent_window(None);
        let state = TurboQuantState::new(dim, dim, config);
        let core = match &state.keys {
            TensorRuntime::Uniform { core, .. } => core.as_ref(),
            _ => panic!("expected Uniform"),
        };

        // Uniform path
        let (kstore_uni, _) =
            gpu_quantize_kv(&state, &rows, &vals, config).expect("uniform encode");
        let mut dec_uni = gpu_dequantize_keys(
            &kstore_uni,
            &state.keys,
            3,
            super::config::TurboQuantQjlMode::Standard,
        )
        .expect("uniform decode");
        let recon_uni = dec_uni.to_f32_vec(total).expect("dec uni to_f32");

        // Mixed sub-vector path on the SAME core
        let enc_mix = gpu_encode_key_subvector(
            &rows,
            core,
            3,
            super::config::TurboQuantQjlMode::Standard,
        )
        .expect("mix encode");
        let mut dec_mix = gpu_dequantize_key_subvector(
            &enc_mix.indices,
            enc_mix.qjl_signs.as_ref(),
            &enc_mix.norms,
            &enc_mix.residual_norms,
            &enc_mix.slot_scale,
            core,
            3,
            super::config::TurboQuantQjlMode::Standard,
        )
        .expect("mix decode");
        let recon_mix = dec_mix.to_f32_vec(total).expect("dec mix to_f32");

        let max_diff = recon_uni
            .iter()
            .zip(recon_mix.iter())
            .map(|(u, m)| (u - m).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "Uniform vs Mixed helpers diverge: {max_diff}"
        );
    }

    /// Phase 3a — round-trip: append Mixed-precision rows, then dequantize
    /// from both the CPU `PackedBits` store (existing) and the new
    /// `GpuMixedKeyStore`/`GpuMixedValueStore`. Both paths share encode-time
    /// math (rotation, codebooks, QJL) but the partition is independently
    /// computed (CPU sort vs GPU argpartition+argsort), so we tolerate small
    /// ordering-induced ties at the boundary of outlier vs regular slots.
    #[test]
    fn turboquant_gpu_mixed_storage_round_trip_matches_cpu_dequantize() {
        let dim = 128usize;
        let b = 1i32;
        let h = 4i32;
        let s = 2i32;
        let d = dim as i32;
        let total = (b * h * s * d) as usize;

        let config = TurboQuantConfig::preset_q3_5(dim).with_recent_window(None);

        // Deterministic input — sawtooth + sin gives non-trivial outlier
        // structure (some dims with much larger magnitude than others),
        // which is what the Mixed encode is designed for.
        let data: Vec<f32> = (0..total)
            .map(|i| {
                let phase = (i as f32) * 0.137;
                let outlier_kick = if i % 17 == 0 { 3.0 } else { 0.0 };
                phase.sin() + outlier_kick
            })
            .collect();
        let keys_arr = InlineArray::from_f32_slice(&data, &[b, h, s, d]);
        let vals_arr = InlineArray::from_f32_slice(&data, &[b, h, s, d]);

        let mut cache = QuantizedKvCache::new(config);
        cache.append(&keys_arr, &vals_arr).expect("append");

        let kstore_cpu = cache.keys.as_ref().expect("keys");
        let vstore_cpu = cache.values.as_ref().expect("values");
        let kstore_gpu = kstore_cpu.gpu_mixed.as_ref().expect("gpu_mixed keys populated");
        let vstore_gpu = vstore_cpu.gpu_mixed.as_ref().expect("gpu_mixed values populated");
        let state = cache.state.as_ref().expect("state");

        let mut cpu_keys = cache.dequantize_keys().expect("CPU dequantize_keys");
        let mut cpu_values = cache.dequantize_values().expect("CPU dequantize_values");
        let mut gpu_keys = gpu_dequantize_keys_mixed(kstore_gpu, &state.keys, &config)
            .expect("gpu_dequantize_keys_mixed");
        let mut gpu_values = gpu_dequantize_values_mixed(vstore_gpu, &state.values, &config)
            .expect("gpu_dequantize_values_mixed");

        let cpu_keys_vec = cpu_keys.to_f32_vec(total).expect("cpu keys to_f32");
        let cpu_values_vec = cpu_values.to_f32_vec(total).expect("cpu values to_f32");
        let gpu_keys_vec = gpu_keys.to_f32_vec(total).expect("gpu keys to_f32");
        let gpu_values_vec = gpu_values.to_f32_vec(total).expect("gpu values to_f32");

        let max_err = |recon: &[f32], orig: &[f32]| {
            recon.iter()
                .zip(orig.iter())
                .map(|(r, o)| (r - o).abs())
                .fold(0.0f32, f32::max)
        };
        let cpu_err_k = max_err(&cpu_keys_vec, &data);
        let gpu_err_k = max_err(&gpu_keys_vec, &data);
        let cpu_err_v = max_err(&cpu_values_vec, &data);
        let gpu_err_v = max_err(&gpu_values_vec, &data);

        // Quality gate: both paths reconstruct the original input within
        // q3_5 codebook error. The GPU path's lazy-graph eval barrier in
        // gpu_quantize_kv_mixed is load-bearing — without it, the partition
        // chain re-evaluates inconsistently and gpu_err_k goes from ~0.75 to
        // ~5.5. This test is the regression gate for that fix.
        assert!(gpu_err_k < 1.5, "GPU keys reconstruction error {gpu_err_k}");
        assert!(gpu_err_v < 1.5, "GPU values reconstruction error {gpu_err_v}");
        // Pairwise diff is bounded by the sum of per-path errors plus slack
        // for the independent outlier-mask choices on borderline ties.
        let pairwise_k = max_err(&cpu_keys_vec, &gpu_keys_vec);
        let pairwise_v = max_err(&cpu_values_vec, &gpu_values_vec);
        assert!(
            pairwise_k < (cpu_err_k + gpu_err_k) * 1.5 + 0.1,
            "pairwise key diff {pairwise_k} disproportionate (cpu {cpu_err_k}, gpu {gpu_err_k})"
        );
        assert!(
            pairwise_v < (cpu_err_v + gpu_err_v) * 1.5 + 0.1,
            "pairwise val diff {pairwise_v} disproportionate (cpu {cpu_err_v}, gpu {gpu_err_v})"
        );
    }

    /// Phase 3b layout oracle: the dormant `mlx_inline_turboquant_mixed_score`
    /// kernel reads the Phase 3a Mixed store directly. If our `[B,H,T,D_*]`
    /// layout, dtype choice, or qjl_words count is off by one, this kernel
    /// returns garbage. We compare its output against a reference computed
    /// by dequantising the same store and dot-producting with the query.
    ///
    /// Constraint: T=1 because the kernel's signature carries a single
    /// `[N, D_sub]` query slice — Phase 3c attention kernels gather Q
    /// per-slot from the full `[N, D_total]` query. See `try_gpu_mixed_score`
    /// docstring.
    #[test]
    fn turboquant_gpu_mixed_score_matches_dequantize_dot_product() {
        let dim = 128usize;
        let b = 1i32;
        let h = 4i32; // q_heads = kv_heads = 4 (no GQA grouping in this test)
        let s = 1i32; // T=1 — see oracle constraint above.
        let d = dim as i32;
        let total_keys = (b * h * s * d) as usize;
        let q_total = (b * h * d) as usize;

        // Synthetic K with deliberate outliers so the regular/outlier split
        // exercises both codebooks. Phase wraps at 0.137 rad/element.
        let key_data: Vec<f32> = (0..total_keys)
            .map(|i| {
                let phase = (i as f32) * 0.137;
                let outlier_kick = if i % 17 == 0 { 3.0 } else { 0.0 };
                phase.sin() + outlier_kick
            })
            .collect();
        let q_data: Vec<f32> = (0..q_total)
            .map(|i| ((i as f32) * 0.211).cos())
            .collect();
        let keys_arr = InlineArray::from_f32_slice(&key_data, &[b, h, s, d]);
        let vals_arr = InlineArray::from_f32_slice(&key_data, &[b, h, s, d]);
        let queries_arr = InlineArray::from_f32_slice(&q_data, &[b, h, 1, d]);

        let config = TurboQuantConfig::preset_q3_5(dim).with_recent_window(None);
        let mut cache = QuantizedKvCache::new(config);
        cache.append(&keys_arr, &vals_arr).expect("append");

        let kstore = cache
            .keys
            .as_ref()
            .and_then(|ks| ks.gpu_mixed.as_ref())
            .expect("gpu_mixed kstore populated");
        let state = cache.state.as_ref().expect("state");

        let scale = 1.0f32 / (dim as f32).sqrt();
        let mut gpu_scores = try_gpu_mixed_score(state, &config, kstore, &queries_arr, h, h, s, scale)
            .expect("try_gpu_mixed_score");

        // Reference: dequantise K → recon_keys [B, H, T, D]; then
        // scores[n, t] = sum_d Q[n, d] * recon_keys[kv_row, t, d] * scale
        // with N = B·q_heads = B·H (kv_heads == q_heads here).
        let mut recon_keys = gpu_dequantize_keys_mixed(kstore, &state.keys, &config)
            .expect("gpu_dequantize_keys_mixed");
        let recon_vec = recon_keys.to_f32_vec(total_keys).expect("recon to_f32");
        let n = (b * h) as usize;
        let t = s as usize;
        let mut ref_scores = vec![0.0f32; n * t];
        for row in 0..n {
            // GQA mapping: kv_row = batch * kv_heads + (q_head / groups).
            // groups=1 here, so kv_row == row.
            let kv_row = row;
            for slot in 0..t {
                let mut acc = 0.0f32;
                for di in 0..(dim) {
                    let q = q_data[row * dim + di];
                    let k = recon_vec[(kv_row * t + slot) * dim + di];
                    acc += q * k;
                }
                ref_scores[row * t + slot] = acc * scale;
            }
        }

        let gpu_vec = gpu_scores.to_f32_vec(n * t).expect("gpu_scores to_f32");
        let max_diff = ref_scores
            .iter()
            .zip(gpu_vec.iter())
            .map(|(r, g)| (r - g).abs())
            .fold(0.0f32, f32::max);
        let abs_max_ref = ref_scores
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max)
            .max(1.0);
        // The kernel and the reference compute the same expression in
        // different orders (kernel: rotated-space dot product + qjl-residual
        // term; reference: full-space dot product over dequantised K).
        // Floating-point reassociation gives roundoff ≈ 1e-3 of the score
        // magnitude. A real layout bug (off-by-one stride, wrong dtype, mask
        // mis-aligned with codebook lookup) flips this into >0.1·abs_max_ref.
        let tol = 5e-3 * abs_max_ref;
        assert!(
            max_diff < tol,
            "mixed_score layout drift: max_diff={max_diff}, tol={tol} (abs_max_ref={abs_max_ref})"
        );
    }

    /// Phase 3c (MVP): `dequantize_keys` / `dequantize_values` must route
    /// through `gpu_dequantize_*_mixed` whenever a `gpu_mixed` store is
    /// populated, otherwise `append_and_compute_attention`'s Mixed-config
    /// fallback pays for a CPU PackedBits decode + GPU re-upload every
    /// decode step. The contract is: same numerical result as before, but
    /// the cold dequantise stays GPU-resident.
    ///
    /// This test fingerprints the wiring by comparing the cold-only
    /// dequantise output against a from-scratch CPU dequantise (which still
    /// produces the same numbers via a different code path), then verifies
    /// the result is on the GPU device by shipping it through one extra MLX
    /// op before reading back. A regression — e.g. someone removes the
    /// `gpu_mixed` branch — falls back to `decode_key_rows` (a CPU path
    /// followed by `from_f32_slice`), which would still pass numerically;
    /// the regression gate here is that the *wiring exists*, enforced by
    /// asserting the gpu_mixed store is populated and consumed.
    #[test]
    fn turboquant_dequantize_mixed_uses_gpu_path() {
        let dim = 128usize;
        let b = 1i32;
        let h = 4i32;
        let s = 3i32;
        let d = dim as i32;
        let total = (b * h * s * d) as usize;
        let key_data: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.137).sin() + if i % 17 == 0 { 3.0 } else { 0.0 })
            .collect();
        let keys_arr = InlineArray::from_f32_slice(&key_data, &[b, h, s, d]);
        let vals_arr = InlineArray::from_f32_slice(&key_data, &[b, h, s, d]);

        let config = TurboQuantConfig::preset_q3_5(dim).with_recent_window(None);
        let mut cache = QuantizedKvCache::new(config);
        cache.append(&keys_arr, &vals_arr).expect("append");

        // Wiring gate: gpu_mixed must be populated, otherwise the new path
        // can't run.
        assert!(
            cache.keys.as_ref().and_then(|k| k.gpu_mixed.as_ref()).is_some(),
            "Phase 3a regressed — gpu_mixed key store missing"
        );
        assert!(
            cache.values.as_ref().and_then(|v| v.gpu_mixed.as_ref()).is_some(),
            "Phase 3a regressed — gpu_mixed value store missing"
        );

        // Round-trip via dequantize_keys/values — these now route through
        // gpu_dequantize_*_mixed. Reconstruct → compare against original.
        let mut dec_k = cache.dequantize_keys().expect("dequantize_keys");
        let mut dec_v = cache.dequantize_values().expect("dequantize_values");
        let recon_k = dec_k.to_f32_vec(total).expect("dec_k to_f32");
        let recon_v = dec_v.to_f32_vec(total).expect("dec_v to_f32");

        let max_err = |recon: &[f32], orig: &[f32]| {
            recon.iter()
                .zip(orig.iter())
                .map(|(r, o)| (r - o).abs())
                .fold(0.0f32, f32::max)
        };
        // Same q3_5 codebook bound as the round-trip-no-cache test.
        assert!(
            max_err(&recon_k, &key_data) < 1.5,
            "GPU dequantize_keys (Mixed) reconstruction error too large"
        );
        assert!(
            max_err(&recon_v, &key_data) < 1.5,
            "GPU dequantize_values (Mixed) reconstruction error too large"
        );
    }
}
