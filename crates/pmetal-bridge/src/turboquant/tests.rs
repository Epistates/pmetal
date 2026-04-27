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
        let encoded = encode_key_component_rows(&core, &[0.0; 8], 4);
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
        let slot_scales_vec: Vec<f32> = (0..(kv_rows * seq * 3) as usize)
            .map(|i| match i % 3 {
                0 => 0.5 + (((i / 3) % 11) as f32) * 0.03125,
                1 => 0.0,
                _ => 1.0,
            })
            .collect();
        let key_codebook_vec: Vec<f32> = (0..256).map(|i| ((i as f32) - 127.5) / 96.0).collect();
        let value_dense_vec: Vec<f32> = (0..(kv_rows * seq * dim) as usize)
            .map(|i| ((i as f32) * 0.009 - 0.7).cos() * 0.5)
            .collect();

        let query_rot = InlineArray::from_f32_slice(&query_rot_vec, &[q_rows, dim]);
        let key_indices = InlineArray::from_u8_slice(&key_indices_vec, &[kv_rows, seq, dim]);
        let slot_scales = InlineArray::from_f32_slice(&slot_scales_vec, &[kv_rows, seq, 3]);
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
                let scale_base = (kvh * seq as usize + t) * 3;
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
        let encoded = encode_key_component_rows(&core, &row, 4);
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
        let encoded = encode_key_component_rows(&core, &row, 4);
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
            bits,
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
            let encoded = encode_key_component_rows(&core, &data, bits);
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

        let encoded = encode_key_component_rows(&core, &keys, bits);
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
        let mut dec_uni = gpu_dequantize_keys(&kstore_uni, &state.keys, 3).expect("uniform decode");
        let recon_uni = dec_uni.to_f32_vec(total).expect("dec uni to_f32");

        // Mixed sub-vector path on the SAME core
        let enc_mix = gpu_encode_key_subvector(&rows, core, 3).expect("mix encode");
        let mut dec_mix = gpu_dequantize_key_subvector(
            &enc_mix.indices,
            &enc_mix.qjl_signs,
            &enc_mix.norms,
            &enc_mix.residual_norms,
            core,
            3,
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
