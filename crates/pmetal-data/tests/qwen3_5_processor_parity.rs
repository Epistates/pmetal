//! Parity for the Qwen 3.5-family image and video preprocessing
//! ([`pmetal_data::qwen_vl_processing`]) and the torchvision resampler it uses
//! ([`pmetal_data::pillow_resample::resize_rgb8_torchvision`]).
//!
//! Oracle: Hugging Face transformers 5.19 on the torchvision backend, the one
//! `AutoProcessor` picks when torchvision is installed. The fixture is dumped
//! by `.strategy/parity/dump_qwen3_5_processor_reference.py` from synthetic
//! images and covers:
//!
//! * torchvision's antialiased `uint8` resize over a grid of ratios, both
//!   filters, compared byte for byte;
//! * `smart_resize` for images and videos under the released budgets, on sizes
//!   from 16 px to 8000 px including both extremes of aspect ratio;
//! * `pixel_values` and `image_grid_thw` for images under budgets that
//!   upscale, downscale and leave the size alone;
//! * videos: frame sampling from a stated and an unstated frame rate, temporal
//!   padding of an odd frame count, `pixel_values_videos`, `video_grid_thw`;
//! * the prompt expansion for an image and a video, timestamps included, and,
//!   when `PMETAL_QWEN3_5_DIR` names a Qwen3.5-family checkpoint, the token ids
//!   and `mm_token_type_ids` its tokenizer gives.
//!
//! Pixel values are compared exactly: the arithmetic is the reference's, op for
//! op, so any difference at all is a real divergence.

mod common;

use std::collections::HashMap;

use common::{fixture_path, load_shard, ref_tensor};
use image::RgbImage;
use pmetal_bridge::compat::Array;
use pmetal_data::pillow_resample::{ResampleFilter, resize_rgb8_torchvision};
use pmetal_data::qwen_vl_processing::{
    PixelBudget, ProcessedMedia, QwenVlProcessor, QwenVlProcessorConfig, VideoFrames,
    expand_placeholders, smart_resize, smart_resize_video,
};
use pmetal_mlx::test_utils::to_f32_vec_eval;
use serde_json::Value;

const FIXTURE: &str = "qwen3_5_processor_reference.safetensors";

struct Fixture {
    shard: HashMap<String, Array>,
    meta: Value,
}

fn fixture() -> Fixture {
    let shard = load_shard(&fixture_path(FIXTURE));
    let meta_path = fixture_path(&format!("{FIXTURE}.meta.json"));
    let meta = serde_json::from_str(&std::fs::read_to_string(meta_path).unwrap()).unwrap();
    Fixture { shard, meta }
}

impl Fixture {
    fn floats(&self, key: &str) -> (Vec<f32>, Vec<i32>) {
        let arr = ref_tensor(&self.shard, key);
        (to_f32_vec_eval(arr), arr.shape().to_vec())
    }

    /// A `[H, W, 3]` uint8 tensor as an image.
    fn image(&self, key: &str) -> RgbImage {
        let (values, shape) = self.floats(key);
        assert_eq!(shape.len(), 3, "{key}: {shape:?}");
        RgbImage::from_raw(
            shape[1] as u32,
            shape[0] as u32,
            values.into_iter().map(|v| v as u8).collect(),
        )
        .unwrap()
    }

    /// A `[T, H, W, 3]` uint8 tensor as frames.
    fn frames(&self, key: &str) -> Vec<RgbImage> {
        let (values, shape) = self.floats(key);
        assert_eq!(shape.len(), 4, "{key}: {shape:?}");
        let frame_len = (shape[1] * shape[2] * 3) as usize;
        values
            .chunks(frame_len)
            .map(|chunk| {
                RgbImage::from_raw(
                    shape[2] as u32,
                    shape[1] as u32,
                    chunk.iter().map(|&v| v as u8).collect(),
                )
                .unwrap()
            })
            .collect()
    }

    fn grid(&self, key: &str) -> [usize; 3] {
        let (values, _) = self.floats(key);
        [values[0] as usize, values[1] as usize, values[2] as usize]
    }
}

/// The released configs (Qwen3.8-27B and Clef ship the same values).
fn released_processor() -> QwenVlProcessor {
    let image: QwenVlProcessorConfig = serde_json::from_value(serde_json::json!({
        "size": {"longest_edge": 16777216, "shortest_edge": 65536},
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]
    }))
    .unwrap();
    let video: QwenVlProcessorConfig = serde_json::from_value(serde_json::json!({
        "size": {"longest_edge": 25165824, "shortest_edge": 4096},
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "fps": 2, "min_frames": 4, "max_frames": 768
    }))
    .unwrap();
    QwenVlProcessor { image, video }
}

fn budget(value: &Value) -> PixelBudget {
    serde_json::from_value(value.clone()).unwrap()
}

/// Exact comparison, reporting the first and worst difference.
fn assert_pixels_equal(name: &str, got: &ProcessedMedia, want: &[f32]) {
    assert_eq!(got.pixel_values.len(), want.len(), "{name}: pixel count");
    let mut worst = (0usize, 0f32);
    for (i, (a, b)) in got.pixel_values.iter().zip(want).enumerate() {
        let diff = (a - b).abs();
        if diff > worst.1 {
            worst = (i, diff);
        }
    }
    assert_eq!(
        worst.1, 0.0,
        "{name}: pixel {} differs by {} ({} vs {})",
        worst.0, worst.1, got.pixel_values[worst.0], want[worst.0]
    );
}

#[test]
fn torchvision_resample_is_bit_exact() {
    let fx = fixture();
    let mut compared = 0;
    for case in fx.meta["resample_cases"].as_array().unwrap() {
        let index = case["source"].as_u64().unwrap();
        let source = fx.image(&format!("tv_src_{index}"));
        for target in case["targets"].as_array().unwrap() {
            let (h, w) = (target[0].as_u64().unwrap(), target[1].as_u64().unwrap());
            for (name, filter) in [
                ("bilinear", ResampleFilter::Bilinear),
                ("bicubic", ResampleFilter::Bicubic),
            ] {
                let key = format!("tv_{index}_{h}x{w}_{name}");
                let want = fx.image(&key);
                let got = resize_rgb8_torchvision(&source, w as u32, h as u32, filter);
                let differing = got
                    .as_raw()
                    .iter()
                    .zip(want.as_raw())
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(differing, 0, "{key}: {differing} bytes differ");
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 22);
}

#[test]
fn smart_resize_matches_both_released_budgets() {
    let fx = fixture();
    let image = budget(&serde_json::json!({"shortest_edge": 65536, "longest_edge": 16777216}));
    let video = budget(&serde_json::json!({"shortest_edge": 4096, "longest_edge": 25165824}));
    for row in fx.meta["smart_resize"]["image"].as_array().unwrap() {
        let v: Vec<usize> = row
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(
            smart_resize(v[0], v[1], 32, image).unwrap(),
            (v[2], v[3]),
            "image {}x{}",
            v[0],
            v[1]
        );
    }
    for row in fx.meta["smart_resize"]["video"].as_array().unwrap() {
        let v: Vec<usize> = row
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(
            smart_resize_video(v[0], v[1], v[2], 2, 32, video).unwrap(),
            (v[3], v[4]),
            "video {}x{}x{}",
            v[0],
            v[1],
            v[2]
        );
    }
}

#[test]
fn images_match_the_reference_processor() {
    let fx = fixture();
    for case in fx.meta["image_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mut processor = released_processor();
        processor.image.size = budget(&case["size"]);
        let source = fx.image(&format!("image_{name}_source"));
        let got = processor.preprocess_image(&source).unwrap();
        assert_eq!(
            got.grid_thw,
            fx.grid(&format!("image_{name}_grid_thw")),
            "{name}: grid"
        );
        let key = format!("image_{name}_pixel_values");
        if fx.shard.contains_key(&key) {
            let (want, _) = fx.floats(&key);
            assert_pixels_equal(name, &got, &want);
        }
    }
}

#[test]
fn videos_match_the_reference_processor() {
    let fx = fixture();
    let processor = released_processor();
    for case in fx.meta["video_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let video = VideoFrames {
            frames: fx.frames(&format!("video_{name}_frames")),
            fps: case["fps"].as_f64(),
        };
        let got = processor.preprocess_video(&video).unwrap();
        let want_indices: Vec<usize> = case["frames_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(got.frame_indices, want_indices, "{name}: sampled frames");
        assert_eq!(
            got.grid_thw,
            fx.grid(&format!("video_{name}_grid_thw")),
            "{name}: grid"
        );
        let (want, _) = fx.floats(&format!("video_{name}_pixel_values"));
        assert_pixels_equal(name, &got, &want);
    }
}

#[test]
fn timestamps_format_like_python() {
    let fx = fixture();
    for pair in fx.meta["timestamp_format"].as_array().unwrap() {
        let value = pair[0].as_f64().unwrap();
        assert_eq!(format!("{value:.1}"), pair[1].as_str().unwrap(), "{value}");
    }
}

/// The image and video the prompt case used, processed.
fn prompt_media(fx: &Fixture) -> (ProcessedMedia, ProcessedMedia) {
    let processor = released_processor();
    let image = processor
        .preprocess_image(&fx.image("image_released_upscale_source"))
        .unwrap();
    let video = processor
        .preprocess_video(&VideoFrames {
            frames: fx.frames("video_fps6_frames"),
            fps: Some(6.0),
        })
        .unwrap();
    (image, video)
}

#[test]
fn prompt_expansion_matches_the_reference_processor() {
    let fx = fixture();
    let prompt = &fx.meta["prompt"];
    let (image, video) = prompt_media(&fx);
    let grid = |v: &Value| -> [usize; 3] {
        let row = &v.as_array().unwrap()[0];
        [0, 1, 2].map(|i| row[i].as_u64().unwrap() as usize)
    };
    assert_eq!(image.grid_thw, grid(&prompt["image_grid_thw"]));
    assert_eq!(video.grid_thw, grid(&prompt["video_grid_thw"]));
    let expanded =
        expand_placeholders(prompt["text"].as_str().unwrap(), &[image], &[video], 2).unwrap();
    assert_eq!(expanded, prompt["expanded"].as_str().unwrap());
}

/// Token ids through a real Qwen3.5-family tokenizer. Opt-in, since the
/// tokenizer is not committed: set `PMETAL_QWEN3_5_DIR` to a checkpoint.
#[test]
fn prompt_tokens_match_the_reference_processor() {
    let Some(dir) = std::env::var_os("PMETAL_QWEN3_5_DIR") else {
        eprintln!("skipping: PMETAL_QWEN3_5_DIR is not set");
        return;
    };
    let fx = fixture();
    let prompt = &fx.meta["prompt"];
    let (image, video) = prompt_media(&fx);
    let expanded =
        expand_placeholders(prompt["text"].as_str().unwrap(), &[image], &[video], 2).unwrap();
    let tokenizer = pmetal_data::Tokenizer::from_model_dir(std::path::Path::new(&dir)).unwrap();
    let ids = tokenizer.encode(&expanded).unwrap();
    let want: Vec<u32> = prompt["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(ids, want);
}
