//! Smoke test over real GGUF files.
//!
//! Ignored by default. Set `PMETAL_GGUF_FILES` to a path-separated list of
//! GGUF files (any shard of a split set opens the set) and run
//! `cargo test -p pmetal-gguf --test real_files -- --ignored --nocapture`.
//!
//! For every file this opens the whole model, checks that each tensor maps
//! inside its shard, decodes the first rows of one tensor of every distinct
//! type, and checks that the weights have a plausible distribution: finite,
//! non-degenerate, and not wildly larger than a trained weight can be. A
//! decoder with a wrong bit layout produces values that fail these checks.

use std::collections::BTreeMap;

use pmetal_gguf::{GgmlType, GgufModel, dequant};

/// Decode at most this many elements of a tensor.
const SAMPLE_ELEMENTS: usize = 1 << 16;

fn files() -> Vec<std::path::PathBuf> {
    std::env::var_os("PMETAL_GGUF_FILES")
        .map(|v| std::env::split_paths(&v).collect())
        .unwrap_or_default()
}

#[test]
#[ignore = "needs PMETAL_GGUF_FILES"]
fn real_files_open_and_decode_plausibly() {
    let files = files();
    assert!(!files.is_empty(), "set PMETAL_GGUF_FILES");

    for path in files {
        let model = GgufModel::open(&path).expect("open");
        println!(
            "{}: {} shard(s), {} tensors, arch {:?}",
            path.display(),
            model.num_shards(),
            model.num_tensors(),
            model.architecture()
        );

        // One representative (the largest, to get a real weight matrix) per type.
        let mut by_type: BTreeMap<String, (&str, u64)> = BTreeMap::new();
        for name in model.tensor_names() {
            let info = model.tensor_info(name).expect("info");
            let slot = by_type
                .entry(format!("{:?}", info.dtype))
                .or_insert((name, 0));
            let n = info.n_elements() as u64;
            if n > slot.1 {
                *slot = (name, n);
            }
            // Mapping every tensor proves the extents against the real file.
            model.tensor_data(name).expect("tensor maps");
        }

        for (dtype_name, (name, n)) in by_type {
            let info = model.tensor_info(name).unwrap();
            if !dequant::is_supported(info.dtype) {
                println!("  {dtype_name:>8}: {name} ({n} elements) not decodable");
                continue;
            }
            // Whole rows only: take as many rows as fit in the sample budget.
            let row = *info.dimensions.last().unwrap_or(&1) as usize;
            let rows = (SAMPLE_ELEMENTS / row.max(1)).clamp(1, info.n_elements() / row.max(1));
            let elements = rows * row;
            let bytes = &model.tensor_data(name).unwrap()[..info.dtype.tensor_size(elements)];
            let mut values = vec![0.0f32; elements];
            dequant::dequantize_into(bytes, info.dtype, &[rows as i32, row as i32], &mut values)
                .expect("decode");

            let finite = values.iter().all(|v| v.is_finite());
            let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / elements as f64;
            let var = values
                .iter()
                .map(|&v| (f64::from(v) - mean).powi(2))
                .sum::<f64>()
                / elements as f64;
            let max = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let zeros = values.iter().filter(|v| **v == 0.0).count() as f64 / elements as f64;
            println!(
                "  {dtype_name:>8}: {name} [{elements} sampled] std {:.4} max {:.3} zeros {:.3}",
                var.sqrt(),
                max,
                zeros
            );
            assert!(finite, "{name}: non-finite values");
            if matches!(
                info.dtype,
                GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 | GgmlType::I32
            ) {
                continue;
            }
            // Norm vectors and small tensors can legitimately look odd, but a
            // weight matrix is a near-zero-mean, modest-variance bulk.
            if elements >= 4096 && name.ends_with(".weight") && !name.contains("norm") {
                assert!(var.sqrt() > 1e-5, "{name}: degenerate (all ~zero)");
                assert!(var.sqrt() < 10.0, "{name}: implausibly large spread");
                assert!(mean.abs() < var.sqrt(), "{name}: mean dominates the spread");
                assert!(zeros < 0.6, "{name}: mostly zeros");
            }
        }
    }
}
