//! Numerical parity for Qwen 3.5-family vision against Hugging Face
//! transformers' `Qwen3_5ForConditionalGeneration`, on a tiny seeded model
//! dumped by `.strategy/parity/dump_qwen3_5_vision_reference.py`.
//!
//! The text half is the `swish` profile of `qwen3_5_parity.rs`; the vision
//! tower is shrunk but keeps every structural feature (two blocks, rotary
//! quarters, a 4x4 position table each grid is resampled from, the tanh-gelu
//! MLP and the exact-gelu merger). The prompt holds one image (6x10 patches)
//! and a three-frame video (padded to two temporal groups of 4x6 patches) laid
//! out as the processor lays them out, and the reference ran it together with
//! a six-token continuation, so the rows after the prompt are what a cached
//! decode must reproduce.
//!
//! Checked, in order:
//!
//! * the processor on the fixture's source image and frames gives the exact
//!   pixels the reference processor gave;
//! * the vision tower's merged features for the image and for the video;
//! * `rope_index` against `get_rope_index`, for the prompt and for prompt plus
//!   continuation, including where decoding resumes (one past the largest
//!   position, 16 short of the prompt length here);
//! * both engines, `DynamicModel` and the native bridge (`pmetal infer`):
//!   prefill logits and hidden states for the merged prompt, then the
//!   continuation decoded one token at a time against the cache.
//!
//! Every forward is drained for bridge errors.

mod common;

use std::collections::HashMap;
use std::path::Path;

use common::{fixture_path, load_shard, ref_tensor};
use image::RgbImage;
use pmetal_bridge::compat::{Array, Module, ops::slice_axis};
use pmetal_data::qwen_vl_processing::{
    ProcessedMedia, QwenVlProcessor, QwenVlProcessorConfig, VideoFrames,
};
use pmetal_mlx::test_utils::{
    ParityReport, Tolerance, argmax_last_axis, print_report_table, to_f32_vec_eval,
};
use pmetal_models::DynamicModel;
use pmetal_models::architectures::qwen3_5_vision::{
    MropePositions, Qwen3_5MultimodalConfig, Qwen3_5VisionModel, merge_media_features, rope_index,
};
use serial_test::serial;

/// fp32 against fp32; the reference's own fp32-vs-fp64 noise on the logits is
/// 5.2e-6 (in the fixture's meta).
const TOL: Tolerance = Tolerance::new(5e-5, 0.0);

struct Fixture {
    dir: tempfile::TempDir,
    reference: HashMap<String, Array>,
    meta: serde_json::Value,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy(
        fixture_path("qwen3_5_vision_config.json"),
        dir.path().join("config.json"),
    )
    .unwrap();
    std::fs::copy(
        fixture_path("qwen3_5_vision_weights.safetensors"),
        dir.path().join("model.safetensors"),
    )
    .unwrap();
    let reference = load_shard(&fixture_path("qwen3_5_vision_reference.safetensors"));
    let meta = serde_json::from_str(
        &std::fs::read_to_string(fixture_path(
            "qwen3_5_vision_reference.safetensors.meta.json",
        ))
        .unwrap(),
    )
    .unwrap();
    Fixture {
        dir,
        reference,
        meta,
    }
}

impl Fixture {
    fn path(&self) -> &Path {
        self.dir.path()
    }

    fn tensor(&self, key: &str) -> Array {
        ref_tensor(&self.reference, key).clone()
    }

    fn ints(&self, key: &str) -> Vec<i64> {
        to_f32_vec_eval(&self.tensor(key))
            .into_iter()
            .map(|v| v as i64)
            .collect()
    }

    fn grid(&self, key: &str) -> [usize; 3] {
        let v = self.ints(key);
        [v[0] as usize, v[1] as usize, v[2] as usize]
    }

    fn prompt_len(&self) -> usize {
        self.meta["prompt_len"].as_u64().unwrap() as usize
    }

    fn input_ids(&self) -> Vec<u32> {
        self.ints("input_ids")
            .into_iter()
            .map(|v| v as u32)
            .collect()
    }

    fn media(&self, pixels: &str, grid: &str) -> ProcessedMedia {
        ProcessedMedia {
            pixel_values: to_f32_vec_eval(&self.tensor(pixels)),
            grid_thw: self.grid(grid),
            timestamps: Vec::new(),
            frame_indices: Vec::new(),
        }
    }

    fn mm_config(&self) -> Qwen3_5MultimodalConfig {
        Qwen3_5MultimodalConfig::from_model_dir(self.path()).unwrap()
    }
}

fn drain(op: &str) {
    pmetal_bridge::check_last_error().unwrap_or_else(|e| panic!("{op}: bridge error: {e}"));
}

fn rows(a: &Array, start: i32, end: i32) -> Array {
    slice_axis(a, 1, start, end)
}

fn assert_all_pass(title: &str, reports: &[ParityReport]) {
    println!("\n== {title} ==");
    print_report_table(reports);
    let failed: Vec<&str> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.name.as_str())
        .collect();
    assert!(failed.is_empty(), "{title}: {failed:?} out of tolerance");
}

fn image_from(values: &[f32], height: usize, width: usize) -> RgbImage {
    RgbImage::from_raw(
        width as u32,
        height as u32,
        values.iter().map(|&v| v as u8).collect(),
    )
    .unwrap()
}

/// The processors the fixture's pixels came from (patch 4, small budgets).
fn processor(fx: &Fixture) -> QwenVlProcessor {
    let config = |size: &serde_json::Value| -> QwenVlProcessorConfig {
        serde_json::from_value(serde_json::json!({
            "size": size, "patch_size": 4, "temporal_patch_size": 2, "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "fps": 2
        }))
        .unwrap()
    };
    QwenVlProcessor {
        image: config(&fx.meta["image_size"]),
        video: config(&fx.meta["video_size"]),
    }
}

#[test]
#[serial]
fn processor_reproduces_the_fixture_pixels() {
    let fx = fixture();
    let processor = processor(&fx);
    let source = fx.tensor("image_source");
    let (h, w) = (source.dim(0) as usize, source.dim(1) as usize);
    let image = processor
        .preprocess_image(&image_from(&to_f32_vec_eval(&source), h, w))
        .unwrap();
    assert_eq!(image.grid_thw, fx.grid("image_grid_thw"));
    assert_eq!(
        image.pixel_values,
        to_f32_vec_eval(&fx.tensor("image_pixel_values"))
    );

    let frames = fx.tensor("video_frames");
    let (t, fh, fw) = (
        frames.dim(0) as usize,
        frames.dim(1) as usize,
        frames.dim(2) as usize,
    );
    let values = to_f32_vec_eval(&frames);
    let frames = values
        .chunks(fh * fw * 3)
        .take(t)
        .map(|chunk| image_from(chunk, fh, fw))
        .collect();
    let video = processor
        .preprocess_video(&VideoFrames {
            frames,
            fps: fx.meta["video_fps"].as_f64(),
        })
        .unwrap();
    assert_eq!(video.grid_thw, fx.grid("video_grid_thw"));
    assert_eq!(
        video.pixel_values,
        to_f32_vec_eval(&fx.tensor("video_pixel_values"))
    );
}

#[test]
#[serial]
fn vision_tower_matches_transformers() {
    let fx = fixture();
    let tower = Qwen3_5VisionModel::load(fx.path()).unwrap();
    let image = tower
        .forward(
            &fx.tensor("image_pixel_values"),
            &[fx.grid("image_grid_thw")],
        )
        .unwrap();
    let video = tower
        .forward(
            &fx.tensor("video_pixel_values"),
            &[fx.grid("video_grid_thw")],
        )
        .unwrap();
    drain("vision tower");
    let reports = vec![
        ParityReport::compute("image_features", &image, &fx.tensor("image_features"), TOL),
        ParityReport::compute("video_features", &video, &fx.tensor("video_features"), TOL),
    ];
    assert_all_pass("vision tower", &reports);
}

fn positions(fx: &Fixture, ids: &[u32]) -> MropePositions {
    let config = fx.mm_config();
    rope_index(
        ids,
        &[fx.grid("image_grid_thw")],
        &[fx.grid("video_grid_thw")],
        config.vision.spatial_merge_size,
        config.image_token_id,
        config.video_token_id,
    )
    .unwrap()
}

#[test]
#[serial]
fn rope_index_matches_transformers() {
    let fx = fixture();
    let ids = fx.input_ids();
    let len = fx.prompt_len();
    let full = positions(&fx, &ids);
    let want: Vec<i32> = fx
        .ints("position_ids")
        .into_iter()
        .map(|v| v as i32)
        .collect();
    assert_eq!(full.positions, want, "prompt + continuation");
    let prompt = positions(&fx, &ids[..len]);
    let want: Vec<i32> = fx
        .ints("prompt_position_ids")
        .into_iter()
        .map(|v| v as i32)
        .collect();
    assert_eq!(prompt.positions, want, "prompt");
    let delta = fx.meta["prompt_rope_delta"].as_i64().unwrap() as i32;
    assert_eq!(prompt.next_position, len as i32 + delta);
}

/// The prompt's merged input embeddings, given the engine's token embedding.
fn merged_prompt(fx: &Fixture, embed: impl FnOnce(&Array) -> Array) -> Array {
    let config = fx.mm_config();
    let tower = Qwen3_5VisionModel::load(fx.path()).unwrap();
    let image = tower
        .encode(&[fx.media("image_pixel_values", "image_grid_thw")])
        .unwrap();
    let video = tower
        .encode(&[fx.media("video_pixel_values", "video_grid_thw")])
        .unwrap();
    let ids = &fx.input_ids()[..fx.prompt_len()];
    let id_array = Array::from_i32_slice_shaped(
        &ids.iter().map(|&i| i as i32).collect::<Vec<_>>(),
        &[1, ids.len() as i32],
    );
    let text = embed(&id_array);
    merge_media_features(
        &text,
        ids,
        image.as_ref(),
        video.as_ref(),
        config.image_token_id,
        config.video_token_id,
    )
    .unwrap()
}

fn token(id: u32) -> Array {
    Array::from_i32_slice_shaped(&[id as i32], &[1, 1])
}

#[test]
#[serial]
fn dynamic_engine_matches_transformers() {
    let fx = fixture();
    let len = fx.prompt_len() as i32;
    let ids = fx.input_ids();
    let want = fx.tensor("logits");
    let mut model = DynamicModel::load(fx.path()).expect("checkpoint loads");
    let qwen = model
        .as_qwen3_next_mut()
        .expect("qwen3_5 routes to Qwen3Next");
    let embeddings = merged_prompt(&fx, |ids| {
        Module::forward(&mut qwen.model.embed_tokens, ids).unwrap()
    });
    let prompt = positions(&fx, &ids[..len as usize]);

    // Uncached prefill.
    let (hidden, logits) = qwen
        .forward_embeddings(&embeddings, &prompt.array(), None, None)
        .unwrap();
    drain("dynamic prefill");
    let mut reports = vec![
        ParityReport::compute(
            "final_hidden",
            &hidden,
            &rows(&fx.tensor("final_hidden"), 0, len),
            TOL,
        ),
        ParityReport::compute_with_per_position("logits", &logits, &rows(&want, 0, len), TOL),
    ];
    assert_eq!(
        argmax_last_axis(&logits),
        argmax_last_axis(&rows(&want, 0, len))
    );

    // Cached prefill, then the continuation one token at a time.
    let mut kv = model.create_cache(ids.len() + 1);
    let mut mamba = model.create_mamba_cache().unwrap();
    let qwen = model.as_qwen3_next_mut().unwrap();
    let (_, logits) = qwen
        .forward_embeddings(
            &embeddings,
            &prompt.array(),
            Some(&mut kv),
            Some(&mut mamba),
        )
        .unwrap();
    drain("dynamic cached prefill");
    reports.push(ParityReport::compute(
        "cached_prefill_logits",
        &logits,
        &rows(&want, 0, len),
        TOL,
    ));
    for (step, &id) in ids[len as usize..].iter().enumerate() {
        let at = Array::from_i32_slice(&[prompt.next_position + step as i32]);
        let logits = qwen
            .forward_with_cache_at(&token(id), &at, Some(&mut kv), Some(&mut mamba))
            .unwrap();
        drain("dynamic decode");
        let pos = len + step as i32;
        reports.push(ParityReport::compute(
            &format!("step_{pos}"),
            &logits,
            &rows(&want, pos, pos + 1),
            TOL,
        ));
    }
    assert_all_pass("dynamic engine", &reports);
}

#[test]
#[serial]
fn native_engine_matches_transformers() {
    use pmetal_bridge::qwen3_native::{
        NativeCache, embed_tokens, forward_embeddings_hidden, forward_step_hidden, load_config,
        load_model,
    };

    let fx = fixture();
    let len = fx.prompt_len() as i32;
    let ids = fx.input_ids();
    let want = fx.tensor("logits");
    let config = load_config(fx.path()).unwrap();
    let weights = load_model(fx.path(), &config).unwrap();
    drain("native load");
    let embeddings = merged_prompt(&fx, |ids| embed_tokens(&weights, ids));
    let prompt = positions(&fx, &ids[..len as usize]);
    let mrope = config.mrope_tables(&prompt.array());

    let mut cache = NativeCache::new_empty(&weights);
    let (hidden, logits) = forward_embeddings_hidden(
        &weights,
        &embeddings,
        &mrope,
        prompt.next_position,
        &mut cache,
    );
    drain("native prefill");
    let mut reports = vec![
        ParityReport::compute(
            "final_hidden",
            &hidden,
            &rows(&fx.tensor("final_hidden"), 0, len),
            TOL,
        ),
        ParityReport::compute_with_per_position("logits", &logits, &rows(&want, 0, len), TOL),
    ];
    assert_eq!(cache.rope_offset, prompt.next_position);
    for (step, &id) in ids[len as usize..].iter().enumerate() {
        let (_, logits) = forward_step_hidden(&weights, &token(id), &mut cache);
        drain("native decode");
        let pos = len + step as i32;
        reports.push(ParityReport::compute(
            &format!("step_{pos}"),
            &logits,
            &rows(&want, pos, pos + 1),
            TOL,
        ));
    }
    assert_all_pass("native engine", &reports);
}
