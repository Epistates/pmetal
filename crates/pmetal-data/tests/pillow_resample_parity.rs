//! Bit-exactness test for [`pmetal_data::pillow_resample`] against Pillow.
//!
//! The oracle here is **Pillow itself**, not `transformers`: every HuggingFace
//! image processor resamples through Pillow (the fast torchvision backends
//! reimplement its filters rather than the other way round), so Pillow's output
//! is what a released VLM's pixel input is *defined* as.
//!
//! `.strategy/parity/dump_pillow_resample_reference.py` commits the source
//! images and `PIL.Image.resize` results for a grid chosen to break naive
//! implementations: pure up- and downscale, mixed axes, single-axis resizes that
//! exercise the skip-the-other-pass shortcut, >20x ratios where the stretched
//! kernel spans many input pixels, and coprime sizes so no output centre ever
//! lands on an input centre. The source patterns carry hard edges, because
//! *where* bicubic overshoot gets clamped is precisely what separated pmetal's
//! previous `image`-crate resize from Pillow.
//!
//! Every comparison is `assert_eq!` on raw bytes. There is no tolerance to
//! loosen: the implementation reproduces Pillow's fixed-point arithmetic, so a
//! single differing byte means a real divergence, not accumulated float error.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use image::RgbImage;
use pmetal_bridge::compat::Array;
use pmetal_data::pillow_resample::{ResampleFilter, resize_rgb8};
use pmetal_mlx::test_utils::to_f32_vec_eval;

const FIXTURE: &str = "pillow_resample_reference.safetensors";

/// Mirrors `CASES` in `.strategy/parity/dump_pillow_resample_reference.py`:
/// `(source index, [(target height, target width), ...])`.
const CASES: [(usize, &[(u32, u32)]); 3] = [
    (0, &[(24, 32), (96, 128), (17, 53), (48, 21), (9, 64)]),
    (1, &[(13, 5), (128, 51), (91, 200), (7, 111)]),
    (2, &[(12, 840), (4, 9)]),
];

const FILTERS: [(&str, ResampleFilter); 2] = [
    ("bilinear", ResampleFilter::Bilinear),
    ("bicubic", ResampleFilter::Bicubic),
];

/// Rebuild an `RgbImage` from the fixture's `[H, W, 3]` uint8 tensor.
fn fixture_image(shard: &HashMap<String, Array>, key: &str) -> RgbImage {
    let arr = ref_tensor(shard, key);
    let shape = arr.shape().to_vec();
    assert_eq!(shape.len(), 3, "{key}: expected [H, W, 3], got {shape:?}");
    assert_eq!(shape[2], 3, "{key}: expected 3 channels, got {shape:?}");
    // Stored as uint8; read back through f32, which is lossless for `0..=255`.
    let bytes: Vec<u8> = to_f32_vec_eval(arr).into_iter().map(|v| v as u8).collect();
    RgbImage::from_raw(shape[1] as u32, shape[0] as u32, bytes)
        .unwrap_or_else(|| panic!("{key}: raw bytes do not fill {shape:?}"))
}

/// Where the two images first differ, as `(x, y, channel, ours, pillow)`.
fn first_difference(ours: &RgbImage, reference: &RgbImage) -> Option<(u32, u32, usize, u8, u8)> {
    let width = ours.width();
    ours.as_raw()
        .iter()
        .zip(reference.as_raw())
        .enumerate()
        .find(|(_, (a, b))| a != b)
        .map(|(i, (&a, &b))| {
            let pixel = i / 3;
            ((pixel as u32) % width, (pixel as u32) / width, i % 3, a, b)
        })
}

#[test]
fn pillow_resample_is_bit_exact() {
    let shard = load_shard(&fixture_path(FIXTURE));
    let mut compared = 0usize;

    for (src_idx, targets) in CASES {
        let source = fixture_image(&shard, &format!("src_{src_idx}"));
        for &(height, width) in targets {
            for (name, filter) in FILTERS {
                let key = format!("out_{src_idx}_{name}_{height}x{width}");
                let expected = fixture_image(&shard, &key);
                assert_eq!(
                    (expected.width(), expected.height()),
                    (width, height),
                    "{key}: fixture size drifted from the case table"
                );

                let ours = resize_rgb8(&source, width, height, filter);
                assert_eq!(
                    (ours.width(), ours.height()),
                    (width, height),
                    "{key}: shape"
                );

                if let Some((x, y, ch, got, want)) = first_difference(&ours, &expected) {
                    let mismatches = ours
                        .as_raw()
                        .iter()
                        .zip(expected.as_raw())
                        .filter(|(a, b)| a != b)
                        .count();
                    panic!(
                        "{key}: {mismatches} of {} bytes differ; first at \
                         (x={x}, y={y}, ch={ch}): got {got}, Pillow says {want}",
                        ours.as_raw().len()
                    );
                }
                compared += 1;
            }
        }
    }

    // Guards against the loop silently covering nothing if a key naming scheme
    // changes — a test that compares zero images passes very convincingly.
    let expected_cases: usize = CASES.iter().map(|(_, t)| t.len()).sum::<usize>() * FILTERS.len();
    assert_eq!(compared, expected_cases);
    println!("{compared} Pillow resize cases matched bit-for-bit");
}

/// An identity resize must return the source untouched — Pillow skips both
/// passes, and so do we.
#[test]
fn identity_resize_returns_the_source() {
    let shard = load_shard(&fixture_path(FIXTURE));
    let source = fixture_image(&shard, "src_0");
    for (_, filter) in FILTERS {
        let out = resize_rgb8(&source, source.width(), source.height(), filter);
        assert_eq!(out.as_raw(), source.as_raw());
    }
}
