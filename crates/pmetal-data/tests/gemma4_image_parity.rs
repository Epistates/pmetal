//! Numerical-parity test for [`Gemma4ImageProcessor`] — the front half of
//! Gemma 4's vision path, which turns a raw image into the
//! `pixel_values [B, N, 3·patch²]` + `image_position_ids [B, N, 2]` pair
//! `Gemma4VisionModel::forward` consumes.
//!
//! The oracle is the authoritative HuggingFace `transformers`
//! `Gemma4ImageProcessorPil`. `.strategy/parity/dump_gemma4_image_reference.py`
//! runs it over four deterministic test patterns and commits both the raw uint8
//! pixels and the processor's output, so no image codec sits between the two
//! implementations.
//!
//! **The split matters.** Only the resize is inexact, and it is inexact for a
//! reason no tolerance tweak will fix: Pillow clips bicubic overshoot between
//! its two resampling passes, the `image` crate does not, so the two part
//! company at hard edges. Everything downstream — rescale, the
//! `(row, col, channel)` patch flattening, the `(x, y)` position ids, and the
//! zero/`-1` padding to the patch budget — is exact integer bookkeeping, and a
//! bug there would silently feed the tower transposed or misaligned patches
//! without ever tripping a loose pixel tolerance.
//!
//! Case 0 is therefore sized so the aspect-ratio-preserving resize is a no-op,
//! pinning that bookkeeping at **atol 0**. Cases 1-3 add downscale, upscale, and
//! the degenerate-axis fallback under the resampler tolerance. Position ids are
//! compared exactly in every case.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use image::{DynamicImage, RgbImage};
use pmetal_bridge::compat::Array;
use pmetal_data::image_processing::{Gemma4ImageProcessor, Gemma4ImageProcessorConfig};
use pmetal_mlx::test_utils::{ParityReport, Tolerance, print_report_table, to_f32_vec_eval};

const FIXTURE: &str = "gemma4_image_reference.safetensors";

/// Exact match — for the checkpoints that are integer bookkeeping, not
/// arithmetic.
const EXACT: Tolerance = Tolerance::new(0.0, 0.0);

/// Resized-pixel tolerance. Observed max is `5.49e-2` = 14/255, all of it in
/// the resize: Pillow clips bicubic overshoot between its two passes and the
/// `image` crate does not, so the gap lives at hard edges (mean diff is
/// `0.17/255`). `fast_image_resize` measured no closer, so this is not slack
/// waiting to be tightened by a better library — closing it would take a
/// Pillow-exact resampler. Set from the observed diff with ~1.5× headroom.
const RESAMPLED: Tolerance = Tolerance::new(8e-2, 8e-2);

/// Mirrors `PROC_ARGS` in `.strategy/parity/dump_gemma4_image_reference.py`.
/// The tiny patch size keeps the fixture at ~600 KB while leaving every branch
/// of the geometry reachable.
fn fixture_config() -> Gemma4ImageProcessorConfig {
    Gemma4ImageProcessorConfig {
        patch_size: 4,
        max_soft_tokens: 70,
        pooling_kernel_size: 3,
        ..Default::default()
    }
}

/// One fixture image and what it is there to exercise.
struct Case {
    /// Fixture key suffix (`image_{index}`).
    index: usize,
    /// `(height, width)` of the raw image.
    input: (u32, u32),
    /// `(height, width)` the aspect-ratio-preserving resize should pick.
    target: (u32, u32),
    /// Expected unpadded soft-token count.
    soft_tokens: usize,
    /// Tolerance for `pixel_values` — [`EXACT`] when the resize is a no-op.
    pixel_tol: Tolerance,
    /// What this case covers, for the report table.
    covers: &'static str,
}

const CASES: [Case; 4] = [
    Case {
        index: 0,
        input: (72, 120),
        target: (72, 120),
        soft_tokens: 60,
        pixel_tol: EXACT,
        covers: "resize no-op (exact) + padding",
    },
    Case {
        index: 1,
        input: (100, 150),
        target: (72, 120),
        soft_tokens: 60,
        pixel_tol: RESAMPLED,
        covers: "downscale",
    },
    Case {
        index: 2,
        input: (20, 30),
        target: (72, 120),
        soft_tokens: 60,
        pixel_tol: RESAMPLED,
        covers: "upscale",
    },
    Case {
        index: 3,
        input: (8, 1200),
        target: (12, 840),
        soft_tokens: 70,
        pixel_tol: RESAMPLED,
        covers: "degenerate axis + max-side clamp, full budget",
    },
];

/// Rebuild the exact `DynamicImage` the dumper fed `transformers` from the
/// fixture's `[H, W, 3]` uint8 tensor.
fn fixture_image(shard: &HashMap<String, Array>, key: &str) -> DynamicImage {
    let arr = ref_tensor(shard, key);
    let shape = arr.shape().to_vec();
    assert_eq!(shape.len(), 3, "{key}: expected [H, W, 3], got {shape:?}");
    assert_eq!(shape[2], 3, "{key}: expected 3 channels, got {shape:?}");
    // The fixture stores uint8; read it back through f32, which is lossless for
    // `0..=255` and avoids depending on a uint8 `as_slice` on the bridge side.
    let bytes: Vec<u8> = to_f32_vec_eval(arr).into_iter().map(|v| v as u8).collect();
    let buf = RgbImage::from_raw(shape[1] as u32, shape[0] as u32, bytes)
        .unwrap_or_else(|| panic!("{key}: raw bytes do not fill {shape:?}"));
    DynamicImage::ImageRgb8(buf)
}

#[test]
fn gemma4_image_processor_parity() {
    let shard = load_shard(&fixture_path(FIXTURE));
    let processor = Gemma4ImageProcessor::new(fixture_config()).expect("processor builds");
    let mut reports = Vec::new();

    for case in &CASES {
        let image = fixture_image(&shard, &format!("image_{}", case.index));
        assert_eq!(
            (image.height(), image.width()),
            case.input,
            "case {}: fixture image size drifted from the case table",
            case.index
        );
        assert_eq!(
            processor
                .target_size(case.input.0, case.input.1)
                .expect("target size resolves"),
            case.target,
            "case {} ({}): resize geometry",
            case.index,
            case.covers
        );

        let batch = processor
            .preprocess(std::slice::from_ref(&image))
            .expect("preprocess runs");
        assert_eq!(
            batch.num_soft_tokens_per_image,
            vec![case.soft_tokens],
            "case {} ({}): soft-token count",
            case.index,
            case.covers
        );

        reports.push(ParityReport::compute(
            &format!("image_{}_pixels", case.index),
            &batch.pixel_values,
            ref_tensor(&shard, &format!("image_{}_pixel_values", case.index)),
            case.pixel_tol,
        ));
        reports.push(ParityReport::compute(
            &format!("image_{}_positions", case.index),
            &batch.image_position_ids,
            ref_tensor(&shard, &format!("image_{}_position_ids", case.index)),
            EXACT,
        ));
    }

    println!("\n== Gemma4 image-processor parity report ==");
    print_report_table(&reports);

    for r in &reports {
        assert!(
            r.passed(),
            "checkpoint {} failed parity (max abs {:.3e})",
            r.name,
            r.max_abs_diff
        );
    }
}

/// A batched call must be exactly the per-image calls stacked along axis 0.
///
/// The oracle only ever sees one image at a time in the fixture, so this pins
/// the batching itself: images of different aspect ratios are resized
/// independently and only then padded to the shared budget, which is the whole
/// reason `pixel_values` is padded rather than ragged.
#[test]
fn gemma4_image_batch_matches_per_image_calls() {
    let shard = load_shard(&fixture_path(FIXTURE));
    let processor = Gemma4ImageProcessor::new(fixture_config()).expect("processor builds");

    let images: Vec<DynamicImage> = CASES
        .iter()
        .map(|c| fixture_image(&shard, &format!("image_{}", c.index)))
        .collect();

    let batched = processor
        .preprocess(&images)
        .expect("batched preprocess runs");
    let max_patches = processor.max_patches() as i32;
    let patch_dim = processor.patch_dim() as i32;
    assert_eq!(
        batched.pixel_values.shape(),
        &[images.len() as i32, max_patches, patch_dim]
    );
    assert_eq!(
        batched.image_position_ids.shape(),
        &[images.len() as i32, max_patches, 2]
    );

    let mut expected_pixels = Vec::new();
    let mut expected_positions = Vec::new();
    let mut expected_soft_tokens = Vec::new();
    for image in &images {
        let one = processor
            .preprocess(std::slice::from_ref(image))
            .expect("single preprocess runs");
        expected_pixels.extend(to_f32_vec_eval(&one.pixel_values));
        expected_positions.extend(to_f32_vec_eval(&one.image_position_ids));
        expected_soft_tokens.extend(one.num_soft_tokens_per_image);
    }

    assert_eq!(batched.num_soft_tokens_per_image, expected_soft_tokens);
    assert_eq!(to_f32_vec_eval(&batched.pixel_values), expected_pixels);
    assert_eq!(
        to_f32_vec_eval(&batched.image_position_ids),
        expected_positions
    );
}
