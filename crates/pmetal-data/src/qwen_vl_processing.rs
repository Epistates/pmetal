//! Image and video preprocessing for the Qwen 3.5 vision family (Qwen3.5,
//! Qwen3.6, Qwen3.8 and the Clef decision models built on them).
//!
//! Reproduces what a Qwen3.5 checkpoint's processor does in Hugging Face
//! transformers: `Qwen2VLImageProcessor` for images, `Qwen3VLVideoProcessor`
//! for videos and `Qwen3VLProcessor` for the prompt, on the torchvision backend
//! that `AutoProcessor` picks whenever torchvision is installed:
//!
//! 1. **`smart_resize`**: scale to multiples of `patch_size · merge_size` with
//!    the pixel count inside the `[shortest_edge, longest_edge]` budget,
//!    aspect ratio kept. Python's `round` is half-to-even, and so is this.
//! 2. **Resize** with torchvision's antialiased `uint8` bicubic
//!    ([`pillow_resample::resize_rgb8_torchvision`]), bit-exact.
//! 3. **Rescale and normalise fused**, as the torchvision backend fuses them:
//!    `(x − mean/rescale) / (std/rescale)` in f32.
//! 4. **Patchify** into `[patches, C·T·patch²]` rows, patches in
//!    spatial-merge-block order (each `merge × merge` block contiguous), an
//!    image repeated `temporal_patch_size` times along T, a video's frames
//!    grouped `temporal_patch_size` at a time with the last frame repeated to
//!    fill the final group.
//!
//! Videos are sampled to the processor's `fps` (2 for the released configs)
//! from the frame rate the caller states, 24 when it states none, exactly as
//! the reference samples a list of frames; each temporal group then gets the
//! `<{t:.1f} seconds>` timestamp the prompt carries before its frame.
//!
//! [`expand_placeholders`] turns each `<|image_pad|>` / `<|video_pad|>` in a
//! prompt into the run of tokens the vision tower's output fills.

use std::path::Path;

use base64::Engine as _;
use image::RgbImage;
use pmetal_bridge::compat::Array;
use serde::Deserialize;

use crate::pillow_resample::{self, ResampleFilter};

/// The image placeholder token a prompt carries once per image.
pub const IMAGE_PAD: &str = "<|image_pad|>";
/// The video placeholder token a prompt carries once per video.
pub const VIDEO_PAD: &str = "<|video_pad|>";
/// Opens every image and every video frame.
pub const VISION_START: &str = "<|vision_start|>";
/// Closes every image and every video frame.
pub const VISION_END: &str = "<|vision_end|>";

/// Why preprocessing failed.
#[derive(Debug, thiserror::Error)]
pub enum MediaError {
    /// The input is unusable; the message is the caller's to read.
    #[error("{0}")]
    Invalid(String),
    /// The processor config could not be read.
    #[error("processor config: {0}")]
    Config(String),
}

type Result<T> = std::result::Result<T, MediaError>;

fn invalid(message: impl Into<String>) -> MediaError {
    MediaError::Invalid(message.into())
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// The `size` budget, in pixels per image (or per video, all frames together).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct PixelBudget {
    /// Fewest pixels (`min_pixels`).
    pub shortest_edge: u64,
    /// Most pixels (`max_pixels`).
    pub longest_edge: u64,
}

fn default_rescale_factor() -> f64 {
    1.0 / 255.0
}
fn default_resample() -> ResampleFilter {
    ResampleFilter::Bicubic
}
fn default_true() -> bool {
    true
}
fn default_image_budget() -> PixelBudget {
    PixelBudget {
        shortest_edge: 56 * 56,
        longest_edge: 28 * 28 * 1280,
    }
}
fn default_video_budget() -> PixelBudget {
    PixelBudget {
        shortest_edge: 128 * 32 * 32,
        longest_edge: 32 * 32 * 768,
    }
}
fn default_patch_size() -> usize {
    16
}
fn default_temporal_patch_size() -> usize {
    2
}
fn default_merge_size() -> usize {
    2
}
fn default_half() -> [f32; 3] {
    [0.5; 3]
}
fn default_fps() -> f64 {
    2.0
}
fn default_min_frames() -> usize {
    4
}
fn default_max_frames() -> usize {
    768
}

/// The fields a `preprocessor_config.json` / `video_preprocessor_config.json`
/// carries, with the reference classes' defaults for absent keys (the video
/// class's here; [`QwenVlProcessor::from_model_dir`] fills the image class's).
#[derive(Debug, Clone, Deserialize)]
pub struct QwenVlProcessorConfig {
    /// Pixel budget (`min_pixels` / `max_pixels`).
    #[serde(default = "default_video_budget")]
    pub size: PixelBudget,
    /// Side of a vision patch, in pixels.
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    /// Frames per temporal patch.
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    /// Side of the block of patches merged into one prompt token.
    #[serde(default = "default_merge_size")]
    pub merge_size: usize,
    /// Per-channel normalisation mean, in `[0, 1]` units.
    #[serde(default = "default_half")]
    pub image_mean: [f32; 3],
    /// Per-channel normalisation std, in `[0, 1]` units.
    #[serde(default = "default_half")]
    pub image_std: [f32; 3],
    /// Multiplier taking `u8` samples to `[0, 1]`.
    #[serde(default = "default_rescale_factor")]
    pub rescale_factor: f64,
    /// Resampling filter (PIL code).
    #[serde(default = "default_resample")]
    pub resample: ResampleFilter,
    /// Whether to `smart_resize`.
    #[serde(default = "default_true")]
    pub do_resize: bool,
    /// Whether to apply `rescale_factor`.
    #[serde(default = "default_true")]
    pub do_rescale: bool,
    /// Whether to normalise by mean and std.
    #[serde(default = "default_true")]
    pub do_normalize: bool,
    /// Video only: the rate frames are sampled at.
    #[serde(default = "default_fps")]
    pub fps: f64,
    /// Video only.
    #[serde(default = "default_min_frames")]
    pub min_frames: usize,
    /// Video only.
    #[serde(default = "default_max_frames")]
    pub max_frames: usize,
    /// Video only.
    #[serde(default = "default_true")]
    pub do_sample_frames: bool,
}

impl QwenVlProcessorConfig {
    fn parse(value: &serde_json::Value, video: bool) -> Result<Self> {
        let mut value = value.clone();
        if !video && let Some(map) = value.as_object_mut() {
            // `Qwen2VLImageProcessor`'s own defaults, where they differ from
            // the video processor's.
            let budget = default_image_budget();
            for (key, default) in [
                (
                    "size",
                    serde_json::json!({
                        "shortest_edge": budget.shortest_edge,
                        "longest_edge": budget.longest_edge,
                    }),
                ),
                ("patch_size", serde_json::json!(14)),
                (
                    "image_mean",
                    serde_json::json!([0.481_454_66, 0.457_827_5, 0.408_210_73]),
                ),
                (
                    "image_std",
                    serde_json::json!([0.268_629_54, 0.261_302_58, 0.275_777_11]),
                ),
            ] {
                map.entry(key).or_insert(default);
            }
        }
        let config: Self =
            serde_json::from_value(value).map_err(|e| MediaError::Config(e.to_string()))?;
        if config.patch_size == 0 || config.merge_size == 0 || config.temporal_patch_size == 0 {
            return Err(MediaError::Config(
                "patch_size, merge_size and temporal_patch_size must be positive".into(),
            ));
        }
        Ok(config)
    }

    /// The resize granularity, `patch_size · merge_size`.
    pub fn factor(&self) -> usize {
        self.patch_size * self.merge_size
    }

    /// The width of one `pixel_values` row, `C · T · patch²`.
    pub fn patch_dim(&self) -> usize {
        3 * self.temporal_patch_size * self.patch_size * self.patch_size
    }
}

// ---------------------------------------------------------------------------
// smart_resize
// ---------------------------------------------------------------------------

/// Python's `round`: half to even.
fn py_round(x: f64) -> f64 {
    x.round_ties_even()
}

/// `Qwen2VLImageProcessor`'s `smart_resize`: the `(height, width)` an image is
/// resized to.
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    budget: PixelBudget,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        return Err(invalid("image has no pixels"));
    }
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err(invalid(format!(
            "absolute aspect ratio must be smaller than 200, got {}",
            h.max(w) / h.min(w)
        )));
    }
    let mut h_bar = py_round(h / f) * f;
    let mut w_bar = py_round(w / f) * f;
    let (min_pixels, max_pixels) = (budget.shortest_edge as f64, budget.longest_edge as f64);
    if h_bar * w_bar > max_pixels {
        let beta = ((h * w) / max_pixels).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels / (h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// `Qwen3VLVideoProcessor`'s `smart_resize`: the per-frame `(height, width)`
/// for `num_frames` frames sharing one budget.
pub fn smart_resize_video(
    num_frames: usize,
    height: usize,
    width: usize,
    temporal_factor: usize,
    factor: usize,
    budget: PixelBudget,
) -> Result<(usize, usize)> {
    if num_frames < temporal_factor {
        return Err(invalid(format!(
            "t:{num_frames} must be larger than temporal_factor:{temporal_factor}"
        )));
    }
    if height == 0 || width == 0 {
        return Err(invalid("video frame has no pixels"));
    }
    let f = factor as f64;
    let (mut h, mut w) = (height as f64, width as f64);
    if height < factor || width < factor {
        let scale = (f / h).max(f / w);
        h = (h * scale).trunc();
        w = (w * scale).trunc();
    }
    if h.max(w) / h.min(w) > 200.0 {
        return Err(invalid(format!(
            "absolute aspect ratio must be smaller than 200, got {}",
            h.max(w) / h.min(w)
        )));
    }
    let t = num_frames as f64;
    let mut h_bar = py_round(h / f) * f;
    let mut w_bar = py_round(w / f) * f;
    let t_bar = py_round(t / temporal_factor as f64) * temporal_factor as f64;
    let (min_pixels, max_pixels) = (budget.shortest_edge as f64, budget.longest_edge as f64);
    if t_bar * h_bar * w_bar > max_pixels {
        let beta = ((t * h * w) / max_pixels).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if t_bar * h_bar * w_bar < min_pixels {
        let beta = (min_pixels / (t * h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

// ---------------------------------------------------------------------------
// Processed media
// ---------------------------------------------------------------------------

/// One image's or video's patches, ready for the vision tower.
#[derive(Debug, Clone)]
pub struct ProcessedMedia {
    /// `[patches, C·T·patch²]` f32, row-major.
    pub pixel_values: Vec<f32>,
    /// `(t, h, w)` in patches: `t` temporal groups of an `h × w` patch grid.
    pub grid_thw: [usize; 3],
    /// One timestamp per temporal group, in seconds (videos only).
    pub timestamps: Vec<f64>,
    /// The source frames the video was sampled at (videos only).
    pub frame_indices: Vec<usize>,
}

impl ProcessedMedia {
    /// Rows of [`pixel_values`](Self::pixel_values).
    pub fn num_patches(&self) -> usize {
        self.grid_thw.iter().product()
    }

    /// Tokens this media occupies in the prompt, `t·h·w / merge²`.
    pub fn num_tokens(&self, merge_size: usize) -> usize {
        self.num_patches() / (merge_size * merge_size)
    }

    /// [`pixel_values`](Self::pixel_values) as a `[patches, patch_dim]` array.
    pub fn pixel_array(&self) -> Array {
        let rows = self.num_patches() as i32;
        let cols = (self.pixel_values.len() / self.num_patches().max(1)) as i32;
        Array::from_f32_slice(&self.pixel_values, &[rows, cols])
    }
}

/// A video as the caller hands it over: decoded frames and their frame rate.
#[derive(Debug, Clone)]
pub struct VideoFrames {
    /// Decoded frames, in order.
    pub frames: Vec<RgbImage>,
    /// Frames per second of `frames`; the reference assumes 24 when unknown.
    pub fps: Option<f64>,
}

// ---------------------------------------------------------------------------
// The processor
// ---------------------------------------------------------------------------

/// A Qwen 3.5-family checkpoint's image and video processors.
#[derive(Debug, Clone)]
pub struct QwenVlProcessor {
    /// `Qwen2VLImageProcessor`'s settings.
    pub image: QwenVlProcessorConfig,
    /// `Qwen3VLVideoProcessor`'s settings.
    pub video: QwenVlProcessorConfig,
}

/// The source frame rate the reference assumes for frames given without one.
pub const DEFAULT_VIDEO_FPS: f64 = 24.0;

impl QwenVlProcessor {
    /// Read a checkpoint's processor configs: `processor_config.json` with
    /// nested `image_processor` / `video_processor` sections, or the separate
    /// `preprocessor_config.json` and `video_preprocessor_config.json`.
    pub fn from_model_dir(dir: &Path) -> Result<Self> {
        let read = |name: &str| -> Result<Option<serde_json::Value>> {
            let path = dir.join(name);
            if !path.is_file() {
                return Ok(None);
            }
            let text = std::fs::read_to_string(&path)
                .map_err(|e| MediaError::Config(format!("{}: {e}", path.display())))?;
            serde_json::from_str(&text)
                .map(Some)
                .map_err(|e| MediaError::Config(format!("{}: {e}", path.display())))
        };
        let nested = read("processor_config.json")?;
        let section = |key: &str| nested.as_ref().and_then(|v| v.get(key)).cloned();
        let image = section("image_processor")
            .or(read("preprocessor_config.json")?)
            .ok_or_else(|| {
                MediaError::Config(format!(
                    "{} has no image processor config (processor_config.json or \
                     preprocessor_config.json)",
                    dir.display()
                ))
            })?;
        let video = section("video_processor").or(read("video_preprocessor_config.json")?);
        Ok(Self {
            image: QwenVlProcessorConfig::parse(&image, false)?,
            video: QwenVlProcessorConfig::parse(
                video.as_ref().unwrap_or(&serde_json::json!({})),
                true,
            )?,
        })
    }

    /// Preprocess one image.
    pub fn preprocess_image(&self, image: &RgbImage) -> Result<ProcessedMedia> {
        let config = &self.image;
        let (height, width) = (image.height() as usize, image.width() as usize);
        let resized = if config.do_resize {
            let (h, w) = smart_resize(height, width, config.factor(), config.size)?;
            resize(image, h, w, config.resample)
        } else {
            image.clone()
        };
        let plane = normalize(&resized, config);
        let (h, w) = (resized.height() as usize, resized.width() as usize);
        let frames = vec![&plane; config.temporal_patch_size];
        let (pixel_values, grid_h, grid_w) = patchify(&frames, h, w, config)?;
        Ok(ProcessedMedia {
            pixel_values,
            grid_thw: [1, grid_h, grid_w],
            timestamps: Vec::new(),
            frame_indices: Vec::new(),
        })
    }

    /// The source frames a video of `total` frames at `fps` is sampled at.
    pub fn sample_frame_indices(&self, total: usize, fps: Option<f64>) -> Vec<usize> {
        let config = &self.video;
        if !config.do_sample_frames {
            return (0..total).collect();
        }
        let source_fps = fps.unwrap_or(DEFAULT_VIDEO_FPS);
        let wanted = (total as f64 / source_fps * config.fps) as usize;
        let count = wanted
            .max(config.min_frames)
            .min(config.max_frames)
            .min(total);
        linspace_rounded(total.saturating_sub(1), count)
    }

    /// Preprocess one video.
    pub fn preprocess_video(&self, video: &VideoFrames) -> Result<ProcessedMedia> {
        let config = &self.video;
        let first = video
            .frames
            .first()
            .ok_or_else(|| invalid("video has no frames"))?;
        let (height, width) = (first.height() as usize, first.width() as usize);
        if video
            .frames
            .iter()
            .any(|f| (f.height() as usize, f.width() as usize) != (height, width))
        {
            return Err(invalid("every frame of a video must be the same size"));
        }
        if let Some(fps) = video.fps
            && !(fps.is_finite() && fps > 0.0)
        {
            return Err(invalid(format!("video fps must be positive, got {fps}")));
        }
        let indices = self.sample_frame_indices(video.frames.len(), video.fps);
        let (h, w) = if config.do_resize {
            smart_resize_video(
                indices.len(),
                height,
                width,
                config.temporal_patch_size,
                config.factor(),
                config.size,
            )?
        } else {
            (height, width)
        };
        let planes: Vec<Vec<f32>> = indices
            .iter()
            .map(|&i| {
                let frame = &video.frames[i];
                let resized = if config.do_resize {
                    resize(frame, h, w, config.resample)
                } else {
                    frame.clone()
                };
                normalize(&resized, config)
            })
            .collect();
        // Pad to whole temporal groups with the last frame.
        let mut frames: Vec<&Vec<f32>> = planes.iter().collect();
        while frames.len() % config.temporal_patch_size != 0 {
            frames.push(planes.last().expect("sampled at least one frame"));
        }
        let (pixel_values, grid_h, grid_w) = patchify(&frames, h, w, config)?;
        let grid_t = frames.len() / config.temporal_patch_size;
        Ok(ProcessedMedia {
            pixel_values,
            grid_thw: [grid_t, grid_h, grid_w],
            timestamps: timestamps(
                &indices,
                video.fps.unwrap_or(DEFAULT_VIDEO_FPS),
                config.temporal_patch_size,
            ),
            frame_indices: indices,
        })
    }
}

/// `np.linspace(0, last, count).round().astype(int)`.
fn linspace_rounded(last: usize, count: usize) -> Vec<usize> {
    match count {
        0 => Vec::new(),
        1 => vec![0],
        _ => {
            let step = last as f64 / (count - 1) as f64;
            (0..count)
                .map(|i| {
                    if i == count - 1 {
                        last
                    } else {
                        py_round(i as f64 * step) as usize
                    }
                })
                .collect()
        }
    }
}

/// `Qwen3VLProcessor._calculate_timestamps`: one timestamp per temporal group,
/// the mean of its first and last frame's time.
fn timestamps(indices: &[usize], fps: f64, merge: usize) -> Vec<f64> {
    let mut indices = indices.to_vec();
    if let Some(&last) = indices.last() {
        while indices.len() % merge != 0 {
            indices.push(last);
        }
    }
    let seconds: Vec<f64> = indices.iter().map(|&i| i as f64 / fps).collect();
    seconds
        .chunks(merge)
        .map(|group| (group[0] + group[group.len() - 1]) / 2.0)
        .collect()
}

fn resize(image: &RgbImage, height: usize, width: usize, filter: ResampleFilter) -> RgbImage {
    pillow_resample::resize_rgb8_torchvision(image, width as u32, height as u32, filter)
}

/// `[3, H, W]` f32, rescaled and normalised the way the torchvision backend
/// fuses the two: mean and std divided by the rescale factor, then one
/// subtract and one divide.
fn normalize(image: &RgbImage, config: &QwenVlProcessorConfig) -> Vec<f32> {
    let (h, w) = (image.height() as usize, image.width() as usize);
    let inv_rescale = (1.0 / config.rescale_factor) as f32;
    let (mean, std): ([f32; 3], [f32; 3]) = match (config.do_rescale, config.do_normalize) {
        (true, true) => (
            config.image_mean.map(|m| m * inv_rescale),
            config.image_std.map(|s| s * inv_rescale),
        ),
        (false, true) => (config.image_mean, config.image_std),
        (true, false) => ([0.0; 3], [inv_rescale; 3]),
        (false, false) => ([0.0; 3], [1.0; 3]),
    };
    let raw = image.as_raw();
    let mut out = vec![0f32; 3 * h * w];
    for c in 0..3 {
        let plane = &mut out[c * h * w..(c + 1) * h * w];
        for (i, value) in plane.iter_mut().enumerate() {
            let x = raw[i * 3 + c] as f32;
            *value = if config.do_normalize {
                (x - mean[c]) / std[c]
            } else if config.do_rescale {
                x * config.rescale_factor as f32
            } else {
                x
            };
        }
    }
    out
}

/// Lay `frames` (each `[3, H, W]`, `frames.len()` a multiple of T) out as
/// `[groups · gh · gw, C · T · p · p]` rows, patches in merge-block order.
fn patchify(
    frames: &[&Vec<f32>],
    height: usize,
    width: usize,
    config: &QwenVlProcessorConfig,
) -> Result<(Vec<f32>, usize, usize)> {
    let (p, m, t) = (
        config.patch_size,
        config.merge_size,
        config.temporal_patch_size,
    );
    if height % (p * m) != 0 || width % (p * m) != 0 {
        return Err(invalid(format!(
            "a {height}x{width} frame does not tile into {}-pixel merge blocks",
            p * m
        )));
    }
    let (grid_h, grid_w) = (height / p, width / p);
    let groups = frames.len() / t;
    let patch_dim = 3 * t * p * p;
    let mut out = Vec::with_capacity(groups * grid_h * grid_w * patch_dim);
    for group in frames.chunks(t) {
        for block_row in 0..grid_h / m {
            for block_col in 0..grid_w / m {
                for in_row in 0..m {
                    for in_col in 0..m {
                        let row0 = (block_row * m + in_row) * p;
                        let col0 = (block_col * m + in_col) * p;
                        for c in 0..3 {
                            for frame in group {
                                let plane = &frame[c * height * width..(c + 1) * height * width];
                                for y in 0..p {
                                    let start = (row0 + y) * width + col0;
                                    out.extend_from_slice(&plane[start..start + p]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(out.len(), groups * grid_h * grid_w * patch_dim);
    Ok((out, grid_h, grid_w))
}

// ---------------------------------------------------------------------------
// Prompt placeholders
// ---------------------------------------------------------------------------

/// What `<|image_pad|>` becomes for this image.
pub fn image_replacement(media: &ProcessedMedia, merge_size: usize) -> String {
    IMAGE_PAD.repeat(media.num_tokens(merge_size))
}

/// What `<|video_pad|>` becomes for this video: per temporal group, its
/// timestamp, then the group's frame between its own vision markers.
pub fn video_replacement(media: &ProcessedMedia, merge_size: usize) -> String {
    let [t, h, w] = media.grid_thw;
    let per_frame = h * w / (merge_size * merge_size);
    let mut out = String::new();
    for group in 0..t {
        let seconds = media.timestamps.get(group).copied().unwrap_or(0.0);
        out.push_str(&format!("<{seconds:.1} seconds>"));
        out.push_str(VISION_START);
        out.push_str(&VIDEO_PAD.repeat(per_frame));
        out.push_str(VISION_END);
    }
    out
}

/// Replace each `<|image_pad|>` and `<|video_pad|>` in `text`, in order of
/// appearance, with its media's token run. The placeholder counts must match
/// the media counts.
pub fn expand_placeholders(
    text: &str,
    images: &[ProcessedMedia],
    videos: &[ProcessedMedia],
    merge_size: usize,
) -> Result<String> {
    let mut out = String::with_capacity(text.len());
    let (mut next_image, mut next_video) = (images.iter(), videos.iter());
    let mut rest = text;
    loop {
        let image_at = rest.find(IMAGE_PAD);
        let video_at = rest.find(VIDEO_PAD);
        let (at, is_image) = match (image_at, video_at) {
            (None, None) => break,
            (Some(i), None) => (i, true),
            (None, Some(v)) => (v, false),
            (Some(i), Some(v)) => (i.min(v), i < v),
        };
        out.push_str(&rest[..at]);
        if is_image {
            let media = next_image.next().ok_or_else(|| {
                invalid(format!(
                    "the prompt has more {IMAGE_PAD} placeholders than the {} images given",
                    images.len()
                ))
            })?;
            out.push_str(&image_replacement(media, merge_size));
            rest = &rest[at + IMAGE_PAD.len()..];
        } else {
            let media = next_video.next().ok_or_else(|| {
                invalid(format!(
                    "the prompt has more {VIDEO_PAD} placeholders than the {} videos given",
                    videos.len()
                ))
            })?;
            out.push_str(&video_replacement(media, merge_size));
            rest = &rest[at + VIDEO_PAD.len()..];
        }
    }
    out.push_str(rest);
    if next_image.next().is_some() || next_video.next().is_some() {
        return Err(invalid(format!(
            "{} images and {} videos were given, but the prompt has fewer placeholders",
            images.len(),
            videos.len()
        )));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Image sources
// ---------------------------------------------------------------------------

/// Decode an image from a string the way the reference processor reads one:
/// a path to an existing file, or a base64 string, optionally a
/// `data:image/...;base64,` URI. URLs are refused: fetch the image and send
/// its bytes as base64.
pub fn load_image_source(source: &str) -> Result<RgbImage> {
    if source.starts_with("http://") || source.starts_with("https://") {
        return Err(invalid(
            "image URLs are not fetched; send the image as a file path or base64 instead",
        ));
    }
    let bytes = if Path::new(source).is_file() {
        std::fs::read(source).map_err(|e| invalid(format!("{source}: {e}")))?
    } else {
        let data = match source.strip_prefix("data:image/") {
            Some(uri) => uri.split_once(',').map_or("", |(_, data)| data),
            None => source,
        };
        decode_base64(data).ok_or_else(|| {
            invalid(format!(
                "incorrect image source: must be a path to an image file or a base64 encoded \
                 image, got {}",
                truncate_for_error(source)
            ))
        })?
    };
    decode_image(&bytes)
}

/// Decode image bytes (PNG, JPEG, WebP, ...) to RGB8.
pub fn decode_image(bytes: &[u8]) -> Result<RgbImage> {
    image::load_from_memory(bytes)
        .map(|image| image.to_rgb8())
        .map_err(|e| invalid(format!("could not decode image: {e}")))
}

/// Python's `base64.decodebytes`: whitespace is skipped.
fn decode_base64(data: &str) -> Option<Vec<u8>> {
    let compact: String = data.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.is_empty() {
        return None;
    }
    base64::engine::general_purpose::STANDARD
        .decode(compact.as_bytes())
        .ok()
}

fn truncate_for_error(source: &str) -> String {
    const LIMIT: usize = 64;
    match source.char_indices().nth(LIMIT) {
        Some((at, _)) => format!("{}...", &source[..at]),
        None => source.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget(min: u64, max: u64) -> PixelBudget {
        PixelBudget {
            shortest_edge: min,
            longest_edge: max,
        }
    }

    #[test]
    fn python_round_is_half_to_even() {
        // 48 / 32 = 1.5 rounds to 2, 80 / 32 = 2.5 rounds to 2.
        assert_eq!(
            smart_resize(48, 80, 32, budget(0, u64::MAX)).unwrap(),
            (64, 64)
        );
    }

    #[test]
    fn linspace_matches_numpy() {
        assert_eq!(linspace_rounded(8, 4), vec![0, 3, 5, 8]);
        assert_eq!(linspace_rounded(4, 4), vec![0, 1, 3, 4]);
        assert_eq!(linspace_rounded(2, 3), vec![0, 1, 2]);
    }

    #[test]
    fn timestamps_pad_with_the_last_frame() {
        assert_eq!(
            timestamps(&[0, 1, 2], 24.0, 2),
            vec![0.5 / 24.0, 2.0 / 24.0]
        );
        assert_eq!(timestamps(&[0, 1, 2, 3], 2.0, 2), vec![0.25, 1.25]);
    }

    #[test]
    fn placeholder_counts_must_match() {
        let media = ProcessedMedia {
            pixel_values: vec![0.0; 4],
            grid_thw: [1, 2, 2],
            timestamps: vec![],
            frame_indices: vec![],
        };
        assert_eq!(
            expand_placeholders("a<|image_pad|>b", std::slice::from_ref(&media), &[], 2).unwrap(),
            "a<|image_pad|>b"
        );
        assert!(expand_placeholders("a", std::slice::from_ref(&media), &[], 2).is_err());
        assert!(expand_placeholders("<|image_pad|><|image_pad|>", &[media], &[], 2).is_err());
    }

    /// Clef nests both processors in `processor_config.json`; Qwen3.8 ships
    /// two files. Both read the same.
    #[test]
    fn both_config_layouts_read() {
        let image = serde_json::json!({
            "size": {"longest_edge": 16777216, "shortest_edge": 65536},
            "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]
        });
        let video = serde_json::json!({
            "size": {"longest_edge": 25165824, "shortest_edge": 4096},
            "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2, "fps": 2
        });
        let nested = tempfile::tempdir().unwrap();
        std::fs::write(
            nested.path().join("processor_config.json"),
            serde_json::json!({"image_processor": image, "video_processor": video}).to_string(),
        )
        .unwrap();
        let split = tempfile::tempdir().unwrap();
        std::fs::write(
            split.path().join("preprocessor_config.json"),
            image.to_string(),
        )
        .unwrap();
        std::fs::write(
            split.path().join("video_preprocessor_config.json"),
            video.to_string(),
        )
        .unwrap();
        for dir in [nested.path(), split.path()] {
            let processor = QwenVlProcessor::from_model_dir(dir).unwrap();
            assert_eq!(processor.image.size, budget(65536, 16777216));
            assert_eq!(processor.video.size, budget(4096, 25165824));
            assert_eq!(processor.image.patch_size, 16);
            assert_eq!(processor.video.image_mean, [0.5; 3]);
        }
        assert!(QwenVlProcessor::from_model_dir(tempfile::tempdir().unwrap().path()).is_err());
    }

    #[test]
    fn base64_and_data_uris_decode() {
        let image = RgbImage::from_pixel(3, 2, image::Rgb([10, 20, 30]));
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(image.clone())
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        let encoded = base64::engine::general_purpose::STANDARD.encode(&png);
        assert_eq!(load_image_source(&encoded).unwrap(), image);
        let uri = format!("data:image/png;base64,{encoded}");
        assert_eq!(load_image_source(&uri).unwrap(), image);
        assert!(load_image_source("https://example.com/a.png").is_err());
        assert!(load_image_source("not an image").is_err());
    }
}
