//! Numerical-parity test for [`MllamaImageProcessor`] — Llama 3.2 Vision's
//! tiled preprocessing.
//!
//! The oracle is the authoritative HuggingFace `transformers`
//! `MllamaImageProcessorPil`. `.strategy/parity/dump_mllama_image_reference.py`
//! commits the raw uint8 pixels beside the processor's output, so no image codec
//! sits between the two implementations.
//!
//! **Why this is compared at atol 0.** Mllama's geometry is not a preprocessing
//! nicety: the chosen tile arrangement comes back as `aspect_ratio_ids`, and the
//! vision tower indexes a precomputed tile-embedding table with it. An
//! off-by-one there silently selects a different embedding — the model still
//! runs, and still returns plausible features. Nothing downstream would catch
//! it, so it is caught here, exactly.
//!
//! The batch covers each branch that decides the geometry: the 1x1 / 1x2 / 2x1 /
//! 4x1 arrangements, an upscale (where the *smallest* upscale wins), an extreme
//! aspect ratio where a side floors and padding dominates, a multi-image sample,
//! and a ragged batch so both image-slot and tile padding are exercised.
//!
//! One subtlety the padding assertions depend on: the reference pads **before**
//! it normalises, so padded pixels leave the processor at `-mean/std`, not zero.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use image::{DynamicImage, RgbImage};
use pmetal_bridge::compat::Array;
use pmetal_data::image_processing::{
    MllamaImageProcessor, MllamaImageProcessorConfig, supported_aspect_ratios,
};
use pmetal_mlx::test_utils::{ParityReport, Tolerance, print_report_table, to_f32_vec_eval};

const FIXTURE: &str = "mllama_image_reference.safetensors";

/// Exact. Tiling is integer geometry over a bit-exact resampler.
const EXACT: Tolerance = Tolerance::new(0.0, 0.0);

/// Mirrors `PROC_ARGS` in `.strategy/parity/dump_mllama_image_reference.py`.
/// The tile size is shrunk from 560 to keep the fixture small; the geometry is
/// scale-free and `pillow_resample_parity.rs` pins the resampler at real sizes.
fn fixture_config() -> MllamaImageProcessorConfig {
    MllamaImageProcessorConfig {
        tile_size: 28,
        max_image_tiles: 4,
        ..Default::default()
    }
}

/// Mirrors `BATCH` in the dumper: per sample, the `(height, width)` of each
/// image, and the tile count the reference chose for it.
const BATCH: [&[((u32, u32), usize)]; 3] = [
    &[((28, 28), 1), ((100, 50), 2)],
    &[((50, 100), 2)],
    &[((9, 400), 4), ((13, 11), 1)],
];

/// Rebuild an image from the fixture's `[H, W, 3]` uint8 tensor.
fn fixture_image(shard: &HashMap<String, Array>, key: &str) -> DynamicImage {
    let arr = ref_tensor(shard, key);
    let shape = arr.shape().to_vec();
    assert_eq!(shape.len(), 3, "{key}: expected [H, W, 3], got {shape:?}");
    let bytes: Vec<u8> = to_f32_vec_eval(arr).into_iter().map(|v| v as u8).collect();
    let buf = RgbImage::from_raw(shape[1] as u32, shape[0] as u32, bytes)
        .unwrap_or_else(|| panic!("{key}: raw bytes do not fill {shape:?}"));
    DynamicImage::ImageRgb8(buf)
}

fn fixture_batch(shard: &HashMap<String, Array>) -> Vec<Vec<DynamicImage>> {
    BATCH
        .iter()
        .enumerate()
        .map(|(sample_idx, images)| {
            (0..images.len())
                .map(|image_idx| fixture_image(shard, &format!("image_{sample_idx}_{image_idx}")))
                .collect()
        })
        .collect()
}

#[test]
fn mllama_image_processor_parity() {
    let shard = load_shard(&fixture_path(FIXTURE));
    let processor = MllamaImageProcessor::new(fixture_config()).expect("processor builds");
    let samples = fixture_batch(&shard);

    let batch = processor.preprocess(&samples).expect("preprocess runs");

    // Tile counts first: they decide the whole geometry, so a mismatch here
    // explains any pixel mismatch below rather than being buried under it.
    let expected_tiles: Vec<Vec<usize>> = BATCH
        .iter()
        .map(|images| images.iter().map(|&(_, tiles)| tiles).collect())
        .collect();
    assert_eq!(
        batch.num_tiles, expected_tiles,
        "tile counts diverged from the reference"
    );

    let reports = vec![
        ParityReport::compute(
            "pixel_values",
            &batch.pixel_values,
            ref_tensor(&shard, "pixel_values"),
            EXACT,
        ),
        ParityReport::compute(
            "aspect_ratio_ids",
            &batch.aspect_ratio_ids,
            ref_tensor(&shard, "aspect_ratio_ids"),
            EXACT,
        ),
        ParityReport::compute(
            "aspect_ratio_mask",
            &batch.aspect_ratio_mask,
            ref_tensor(&shard, "aspect_ratio_mask"),
            EXACT,
        ),
    ];

    println!("\n== Mllama image-processor parity report ==");
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

/// The arrangement table's *order* is what `aspect_ratio_id` means, so it is
/// pinned independently of any image.
#[test]
fn supported_aspect_ratio_order_matches_reference() {
    assert_eq!(
        supported_aspect_ratios(4),
        vec![
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 1),
            (2, 2),
            (3, 1),
            (4, 1)
        ]
    );
    assert_eq!(supported_aspect_ratios(2), vec![(1, 1), (1, 2), (2, 1)]);
    assert_eq!(supported_aspect_ratios(1), vec![(1, 1)]);
}

/// Tiling choices at Llama 3.2 Vision's real 560px tile size, against a table
/// taken from the reference `get_optimal_tiled_canvas`.
///
/// The fixture runs at tile size 28 to stay small; this checks that nothing in
/// the selection depends on that.
#[test]
fn optimal_tiling_matches_reference_at_production_tile_size() {
    let processor =
        MllamaImageProcessor::new(MllamaImageProcessorConfig::default()).expect("builds");
    // (height, width) -> (tiles_high, tiles_wide) at tile_size 560, max 4.
    let table = [
        ((560, 560), (1, 1)),
        ((2500, 1250), (2, 1)),
        ((1250, 2500), (1, 2)),
        // Smaller than one tile: every arrangement can hold it, so the smallest
        // upscale wins, and the area tie-break picks 1x1 over 2x2.
        ((250, 250), (1, 1)),
        ((5000, 250), (4, 1)),
        ((560, 2240), (1, 4)),
        ((750, 1750), (1, 3)),
        ((2800, 2800), (2, 2)),
        ((125, 7500), (1, 4)),
    ];
    for ((height, width), expected) in table {
        assert_eq!(
            processor.optimal_tiling(height, width).expect("resolves"),
            expected,
            "{height}x{width}"
        );
    }
    assert!(processor.optimal_tiling(0, 560).is_err());
}

/// Padded tile slots must be zero, and the padded *region inside* a real tile
/// must sit at `-mean/std` — the reference pads before it normalises, so a port
/// that pads afterwards leaves zeros there and would still pass a shape check.
#[test]
fn padding_is_normalised_not_zeroed() {
    let shard = load_shard(&fixture_path(FIXTURE));
    let config = fixture_config();
    let processor = MllamaImageProcessor::new(config.clone()).expect("processor builds");
    let samples = fixture_batch(&shard);
    let batch = processor.preprocess(&samples).expect("preprocess runs");

    let tile = config.tile_size as usize;
    let tile_len = 3 * tile * tile;
    let max_tiles = config.max_image_tiles;
    let max_images = BATCH.iter().map(|s| s.len()).max().unwrap();
    let pixels = to_f32_vec_eval(&batch.pixel_values);

    // Sample 1 has one image, so its second image slot is entirely padding.
    let empty_slot = (max_images + 1) * max_tiles * tile_len;
    assert!(
        pixels[empty_slot..empty_slot + max_tiles * tile_len]
            .iter()
            .all(|&v| v == 0.0),
        "an unused image slot should be zero, not normalised"
    );

    // Sample 2 image 0 is 9x400: a 1x4 arrangement, so the canvas is 28x112 and
    // the image is fitted to just 2x112. Rows 2.. of every tile are padding that
    // went through the normalisation, so they must sit at `padding_value`.
    let first_tile = 2 * max_images * max_tiles * tile_len;
    for ch in 0..3 {
        let want = processor.padding_value(ch);
        // Row 10 is comfortably below the 2-row image band.
        let value = pixels[first_tile + ch * tile * tile + 10 * tile];
        assert!(
            (value - want).abs() < 1e-9,
            "padded pixel in channel {ch}: got {value}, want {want}"
        );
        // ...and it must not be zero, which is the mistake this guards.
        assert_ne!(
            value, 0.0,
            "channel {ch} padding was zeroed, not normalised"
        );
    }
}
