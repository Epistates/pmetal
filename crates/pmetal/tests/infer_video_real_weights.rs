//! `infer --video` end to end on a real Qwen3.5-family checkpoint: a video
//! given as a directory of frames goes through the shared runner (frame
//! sampling, timestamps, prompt expansion, the vision tower and the native
//! engine) and the model says which way the ball in it moves. Opt-in:
//!
//! ```text
//! PMETAL_QWEN3_5_DIR=<Qwen3.5-0.8B snapshot> \
//!     cargo test -p pmetal --test infer_video_real_weights -- --nocapture
//! ```

use std::path::{Path, PathBuf};

use pmetal::inference_runner::{InferenceRunner, InferenceRunnerConfig, VideoInput};
use pmetal_data::qwen_vl_processing::RgbImage;

const FRAMES: u32 = 16;
const FPS: f64 = 4.0;

/// A red ball crossing a sky-and-grass scene, one frame per file, named
/// `frame1.png` to `frame16.png` so the order depends on natural sorting.
fn write_ball_video(dir: &Path, left_to_right: bool) {
    std::fs::create_dir_all(dir).unwrap();
    let (width, height, radius) = (320u32, 240u32, 25.0f64);
    for frame in 0..FRAMES {
        let step = if left_to_right {
            frame
        } else {
            FRAMES - 1 - frame
        };
        let cx = 30.0 + step as f64 * (width as f64 - 60.0) / (FRAMES - 1) as f64;
        let image = RgbImage::from_fn(width, height, |x, y| {
            let (dx, dy) = (x as f64 - cx, y as f64 - 150.0);
            if dx * dx + dy * dy <= radius * radius {
                [220, 20, 20].into()
            } else if y > 175 {
                [40, 160, 40].into()
            } else {
                [150, 200, 250].into()
            }
        });
        image
            .save(dir.join(format!("frame{}.png", frame + 1)))
            .unwrap();
    }
}

fn describe(model: &Path, video: PathBuf) -> String {
    let mut runner = InferenceRunner::prepare(InferenceRunnerConfig {
        model_path: model.to_path_buf(),
        prompt: "Describe what happens in this video. Which direction does the ball move?".into(),
        chat: true,
        no_thinking: true,
        videos: vec![VideoInput {
            frames_dir: video,
            fps: Some(FPS),
        }],
        temperature: Some(0.0),
        max_tokens: Some(40),
        ..Default::default()
    })
    .expect("prepare");
    let prompt_len = runner.state.input_ids().len();
    let output = runner.state.generate_streaming(|_| true).expect("generate");
    pmetal_bridge::check_last_error().expect("bridge error");
    runner
        .tokenizer
        .decode(&output.token_ids[prompt_len..])
        .unwrap()
        .to_lowercase()
}

#[test]
fn the_model_sees_which_way_the_ball_moves() {
    let Some(model) = std::env::var_os("PMETAL_QWEN3_5_DIR").map(PathBuf::from) else {
        eprintln!("PMETAL_QWEN3_5_DIR is not set; skipping");
        return;
    };
    let root = std::env::temp_dir().join(format!("pmetal-video-{}", std::process::id()));
    let (rightwards, leftwards) = (root.join("rightwards"), root.join("leftwards"));
    write_ball_video(&rightwards, true);
    write_ball_video(&leftwards, false);

    let order = |answer: &str| -> Option<bool> {
        let (left, right) = (answer.find("left")?, answer.find("right")?);
        Some(left < right)
    };
    let answer = describe(&model, rightwards);
    eprintln!("left to right: {answer}");
    assert_eq!(order(&answer), Some(true), "{answer}");
    let answer = describe(&model, leftwards);
    eprintln!("right to left: {answer}");
    assert_eq!(order(&answer), Some(false), "{answer}");

    std::fs::remove_dir_all(root).unwrap();
}
