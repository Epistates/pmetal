//! Numerical-parity test for [`FixedSizeImageProcessor`] and
//! [`SiglipImageProcessor`]: resize straight to `(height, width)`, rescale,
//! normalise.
//!
//! The oracle is HuggingFace `transformers` `SiglipImageProcessorPil` (the
//! processor that stretches to a fixed size; the fixture's `.meta.json` names
//! the versions). The fixture holds each case's raw uint8 source beside the
//! processor's output, so no image codec sits between the two. Compared at
//! atol 0: the resampler is Pillow-exact (`pillow_resample_parity.rs`) and
//! rescale-then-normalise is staged as the reference stages it (an f64
//! multiply narrowed to f32, then a real f32 divide), so any difference is a
//! bug, not rounding.
//!
//! Cases: a downscale, a non-square target (height and width are different
//! axes), an upscale, an image already at the target size, RGBA and gray
//! inputs (`do_convert_rgb`), and a bilinear resample.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use image::{DynamicImage, GrayImage, RgbImage, RgbaImage};
use pmetal_data::image_processing::{
    FixedSizeImageProcessor, FixedSizeImageProcessorConfig, SiglipImageProcessor,
};
use pmetal_data::pillow_resample::ResampleFilter;
use pmetal_mlx::test_utils::{ParityReport, Tolerance, print_report_table, to_f32_vec_eval};
use pmetal_runtime::compat::Array;

const FIXTURE: &str = "fixed_size_image_reference.safetensors";
const EXACT: Tolerance = Tolerance::new(0.0, 0.0);

/// name, target (height, width), resample: as the reference was generated.
const CASES: [(&str, (u32, u32), ResampleFilter); 7] = [
    ("down_square", (28, 28), ResampleFilter::Bicubic),
    ("wide_to_tall", (32, 24), ResampleFilter::Bicubic),
    ("upscale", (40, 36), ResampleFilter::Bicubic),
    ("same_size", (28, 28), ResampleFilter::Bicubic),
    ("rgba", (24, 20), ResampleFilter::Bicubic),
    ("gray", (16, 16), ResampleFilter::Bicubic),
    ("bilinear_down", (28, 28), ResampleFilter::Bilinear),
];

/// Rebuild the source image from the fixture's `[H, W, C]` uint8 tensor; the
/// channel count says the mode.
fn source_image(shard: &HashMap<String, Array>, key: &str) -> DynamicImage {
    let arr = ref_tensor(shard, key);
    let shape = arr.shape().to_vec();
    assert_eq!(shape.len(), 3, "{key}: expected [H, W, C], got {shape:?}");
    let (h, w, c) = (shape[0] as u32, shape[1] as u32, shape[2]);
    let bytes: Vec<u8> = to_f32_vec_eval(arr).into_iter().map(|v| v as u8).collect();
    match c {
        1 => DynamicImage::ImageLuma8(GrayImage::from_raw(w, h, bytes).expect("gray bytes")),
        3 => DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, bytes).expect("rgb bytes")),
        4 => DynamicImage::ImageRgba8(RgbaImage::from_raw(w, h, bytes).expect("rgba bytes")),
        other => panic!("{key}: {other} channels"),
    }
}

fn run(case: &str) -> (Array, Array) {
    let shard = load_shard(&fixture_path(FIXTURE));
    let (_, (height, width), resample) = *CASES.iter().find(|c| c.0 == case).expect("a case");
    let image = source_image(&shard, &format!("{case}_source"));
    let processor = FixedSizeImageProcessor::new(FixedSizeImageProcessorConfig {
        size: (width, height),
        mean: [0.5; 3],
        std: [0.5; 3],
        rescale_factor: 1.0 / 255.0,
        resample,
    });
    let got = processor.process_image(&image).expect("process");
    pmetal_runtime::check_unobserved_error().expect("a bridge op threw");
    let want = ref_tensor(&shard, &format!("{case}_pixel_values")).clone();
    (got, want)
}

#[test]
fn fixed_size_processor_matches_the_reference_exactly() {
    let mut reports = Vec::new();
    for (case, _, _) in CASES {
        let (got, want) = run(case);
        reports.push(ParityReport::compute(case, &got, &want, EXACT));
    }
    println!("\n== Fixed-size image-processor parity report ==");
    print_report_table(&reports);
    for r in &reports {
        assert!(
            r.passed(),
            "case {} failed parity (max abs {:.3e})",
            r.name,
            r.max_abs_diff
        );
    }
}

/// `SiglipImageProcessor` is the same chain with SigLIP's constants.
#[test]
fn siglip_processor_matches_the_reference_exactly() {
    let shard = load_shard(&fixture_path(FIXTURE));
    // The square, bicubic cases are what `SiglipImageProcessor::new(size)`
    // builds: (28, 28) down and same-size.
    for case in ["down_square", "same_size"] {
        let image = source_image(&shard, &format!("{case}_source"));
        let got = SiglipImageProcessor::new((28, 28))
            .process_image(&image)
            .expect("process");
        pmetal_runtime::check_unobserved_error().expect("a bridge op threw");
        let report = ParityReport::compute(
            case,
            &got,
            ref_tensor(&shard, &format!("{case}_pixel_values")),
            EXACT,
        );
        assert!(
            report.passed(),
            "{case}: max abs {:.3e}",
            report.max_abs_diff
        );
    }
}
