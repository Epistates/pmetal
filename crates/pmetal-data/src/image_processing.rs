//! Image processing utilities for Vision Language Models.
//!
//! Two families live here:
//!
//! * [`MllamaImageProcessor`] / [`SiglipImageProcessor`] — fixed-size,
//!   CLIP-style `[N, 3, H, W]` preprocessing for Llama 3.2 Vision and SigLIP
//!   towers.
//! * [`Gemma4ImageProcessor`] — Gemma 4's patch-budget preprocessing, which
//!   resizes to preserve aspect ratio, patchifies to
//!   `[B, max_patches, 3·patch²]`, and emits the `(x, y)` patch coordinates the
//!   vision tower's 2-D position table and pooler are indexed by.

use image::{DynamicImage, RgbImage, imageops::FilterType};
use pmetal_bridge::compat::{Array, Exception};
use serde::Deserialize;
use std::path::Path;

/// Configuration for Mllama image processing.
#[derive(Debug, Clone)]
pub struct MllamaImageProcessorConfig {
    /// Target image size (width, height).
    pub size: (u32, u32),
    /// Normalization mean (RGB).
    pub mean: [f32; 3],
    /// Normalization standard deviation (RGB).
    pub std: [f32; 3],
    /// Rescaling factor (e.g., 1/255.0).
    pub rescale_factor: f32,
}

impl Default for MllamaImageProcessorConfig {
    fn default() -> Self {
        Self {
            size: (560, 560), // Default for Llama 3.2 11B Vision
            // CLIP stats (canonical values from OpenAI CLIP)
            #[allow(clippy::excessive_precision)]
            mean: [0.48145466, 0.4578275, 0.40821073],
            #[allow(clippy::excessive_precision)]
            std: [0.26862954, 0.26130258, 0.27577711],
            rescale_factor: 1.0 / 255.0,
        }
    }
}

/// Image processor for Mllama.
///
/// Supports:
/// - Single image preprocessing
/// - Batch preprocessing
/// - GPU-accelerated normalization via MLX
#[derive(Debug, Clone)]
pub struct MllamaImageProcessor {
    config: MllamaImageProcessorConfig,
    /// Pre-computed normalization arrays for GPU processing.
    mean_array: Option<Array>,
    std_array: Option<Array>,
}

impl MllamaImageProcessor {
    /// Create a new processor.
    pub fn new(config: MllamaImageProcessorConfig) -> Self {
        Self {
            config,
            mean_array: None,
            std_array: None,
        }
    }

    /// Initialize GPU arrays for normalization.
    /// Call this once before processing many images for better performance.
    pub fn init_gpu_arrays(&mut self) -> Result<(), Exception> {
        // Mean: [1, 3, 1, 1] for broadcasting over [N, C, H, W]
        self.mean_array = Some(Array::from_f32_slice(&self.config.mean, &[1, 3, 1, 1]));
        // Std: [1, 3, 1, 1]
        self.std_array = Some(Array::from_f32_slice(&self.config.std, &[1, 3, 1, 1]));
        Ok(())
    }

    /// Load and preprocess an image from file.
    ///
    /// Returns a tensor of shape [1, 3, H, W] (NCHW format).
    pub fn preprocess(&self, image_path: impl AsRef<Path>) -> Result<Array, Exception> {
        let img = image::open(image_path)
            .map_err(|e| Exception::custom(format!("Failed to open image: {}", e)))?;

        self.process_image(img)
    }

    /// Process a loaded DynamicImage.
    ///
    /// Optimized implementation that:
    /// 1. Resizes image to target size
    /// 2. Converts to NCHW float32 layout
    /// 3. Applies rescaling and normalization
    ///
    /// Returns: Array of shape [1, 3, H, W]
    pub fn process_image(&self, img: DynamicImage) -> Result<Array, Exception> {
        // 1. Resize with bilinear interpolation
        let resized =
            img.resize_exact(self.config.size.0, self.config.size.1, FilterType::Triangle);
        let rgb = resized.to_rgb8();

        let width = rgb.width() as usize;
        let height = rgb.height() as usize;
        let num_pixels = height * width;
        let pixels = rgb.as_raw();

        // 2. Convert to NCHW format with single-pass processing
        // Pre-allocate exact size needed: 3 channels * height * width
        let total_size = 3 * num_pixels;
        let mut data = Vec::with_capacity(total_size);

        // Process each channel using iterator for better vectorization
        for c in 0..3 {
            let mean = self.config.mean[c];
            let std = self.config.std[c];
            let scale = self.config.rescale_factor;

            // Extract channel c from interleaved RGB data
            // Pixels are stored as [R, G, B, R, G, B, ...]
            // We want [R0, R1, R2, ...] for channel 0
            data.extend((0..num_pixels).map(|i| {
                let pixel_val = pixels[i * 3 + c] as f32;
                (pixel_val * scale - mean) / std
            }));
        }

        // Create Array: [1, C, H, W]
        let shape = &[1, 3i32, height as i32, width as i32];
        let array = Array::from_f32_slice(&data, shape);

        Ok(array)
    }

    /// Process image with GPU-accelerated normalization.
    ///
    /// More efficient for large batches as normalization happens on GPU.
    /// Requires `init_gpu_arrays()` to be called first.
    pub fn process_image_gpu(&self, img: DynamicImage) -> Result<Array, Exception> {
        let mean = self.mean_array.as_ref().ok_or_else(|| {
            Exception::custom("GPU arrays not initialized. Call init_gpu_arrays() first.")
        })?;
        let std = self.std_array.as_ref().ok_or_else(|| {
            Exception::custom("GPU arrays not initialized. Call init_gpu_arrays() first.")
        })?;

        // 1. Resize
        let resized =
            img.resize_exact(self.config.size.0, self.config.size.1, FilterType::Triangle);
        let rgb = resized.to_rgb8();

        let width = rgb.width() as usize;
        let height = rgb.height() as usize;
        let num_pixels = height * width;
        let pixels = rgb.as_raw();

        // 2. Convert to NCHW uint8 first (just layout conversion)
        let mut data = Vec::with_capacity(3 * num_pixels);
        for c in 0..3 {
            data.extend((0..num_pixels).map(|i| pixels[i * 3 + c] as f32));
        }

        // 3. Create array and do normalization on GPU
        let shape = &[1, 3i32, height as i32, width as i32];
        let arr = Array::from_f32_slice(&data, shape);

        // GPU operations: rescale then normalize
        let rescale = Array::from_f32(self.config.rescale_factor);
        let scaled = arr.multiply(&rescale);
        let centered = scaled.subtract(mean);
        let normalized = centered.divide(std);

        Ok(normalized)
    }

    /// Process a batch of images.
    ///
    /// Returns: Array of shape [batch, 3, H, W]
    pub fn process_batch(&self, images: &[DynamicImage]) -> Result<Array, Exception> {
        if images.is_empty() {
            return Err(Exception::custom("Empty image batch"));
        }

        let mut batch_data = Vec::new();

        for img in images {
            let processed = self.process_image(img.clone())?;
            batch_data.push(processed);
        }

        // Stack along batch dimension
        let batch_refs: Vec<&Array> = batch_data.iter().collect();
        Ok(pmetal_bridge::compat::ops::concatenate_axis(&batch_refs, 0))
    }

    /// Process a batch from file paths.
    pub fn process_batch_from_paths(&self, paths: &[impl AsRef<Path>]) -> Result<Array, Exception> {
        let images: Result<Vec<_>, _> = paths
            .iter()
            .map(|p| {
                image::open(p)
                    .map_err(|e| Exception::custom(format!("Failed to open image: {}", e)))
            })
            .collect();

        self.process_batch(&images?)
    }

    /// Get the config.
    pub fn config(&self) -> &MllamaImageProcessorConfig {
        &self.config
    }
}

/// SigLIP-style image processor with different normalization.
#[derive(Debug, Clone)]
pub struct SiglipImageProcessor {
    config: MllamaImageProcessorConfig,
}

impl SiglipImageProcessor {
    /// Create a new SigLIP processor.
    pub fn new(size: (u32, u32)) -> Self {
        Self {
            config: MllamaImageProcessorConfig {
                size,
                // SigLIP uses different normalization
                mean: [0.5, 0.5, 0.5],
                std: [0.5, 0.5, 0.5],
                rescale_factor: 1.0 / 255.0,
            },
        }
    }

    /// Process an image.
    pub fn process_image(&self, img: DynamicImage) -> Result<Array, Exception> {
        let processor = MllamaImageProcessor::new(self.config.clone());
        processor.process_image(img)
    }
}

/// Soft-token budgets Gemma 4 was trained with. Anything else is rejected, as
/// in the reference `Gemma4ImageProcessor`.
pub const GEMMA4_SOFT_TOKEN_BUDGETS: [usize; 5] = [70, 140, 280, 560, 1120];

/// Configuration for [`Gemma4ImageProcessor`].
///
/// Field names and defaults mirror a Gemma 4 checkpoint's
/// `preprocessor_config.json`, so the struct deserializes straight from it.
/// Note the unusual pixel policy: Gemma 4 was trained on plain `[0, 1]` pixels,
/// so `do_normalize` is off and mean/std are the identity.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Gemma4ImageProcessorConfig {
    /// Patch side length in pixels. Each patch flattens to `3 · patch_size²`.
    pub patch_size: usize,
    /// Soft-token budget per image; must be one of [`GEMMA4_SOFT_TOKEN_BUDGETS`].
    pub max_soft_tokens: usize,
    /// Spatial pooling kernel the vision tower applies after patchification.
    /// Also constrains the resize: both sides land on a multiple of
    /// `pooling_kernel_size · patch_size` so each pooling block is full.
    pub pooling_kernel_size: usize,
    /// Whether to run the aspect-ratio-preserving resize. Off means the image
    /// must already be patch- and pooling-aligned.
    pub do_resize: bool,
    /// Whether to scale raw `[0, 255]` samples by [`Self::rescale_factor`].
    pub do_rescale: bool,
    /// Pixel scale, `1/255` for Gemma 4. `f64` rather than `f32` on purpose: the
    /// reference rescales via `image.astype(np.float64) * scale` before casting
    /// back down, so an `f32` factor lands 1 ULP off on some samples.
    pub rescale_factor: f64,
    /// Whether to apply mean/std normalisation after rescaling.
    pub do_normalize: bool,
    /// Per-channel mean, only used when [`Self::do_normalize`] is set.
    pub image_mean: [f32; 3],
    /// Per-channel standard deviation, only used when [`Self::do_normalize`] is set.
    pub image_std: [f32; 3],
}

impl Default for Gemma4ImageProcessorConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            max_soft_tokens: 280,
            pooling_kernel_size: 3,
            do_resize: true,
            do_rescale: true,
            rescale_factor: 1.0 / 255.0,
            do_normalize: false,
            image_mean: [0.0; 3],
            image_std: [1.0; 3],
        }
    }
}

/// Preprocessed images in the layout `Gemma4VisionModel::forward` consumes.
#[derive(Debug, Clone)]
pub struct Gemma4ImageBatch {
    /// `[B, max_patches, 3 · patch_size²]`. Within a patch the layout is
    /// row-major with the channel innermost (`(row, col, channel)`), and rows
    /// past an image's real patch count are zero.
    pub pixel_values: Array,
    /// `[B, max_patches, 2]` — `(x, y)` = `(column, row)` patch coordinates,
    /// `(-1, -1)` on padding rows. The tower keys its position table, RoPE, and
    /// pooling blocks off these, and treats `(-1, -1)` as "ignore".
    pub image_position_ids: Array,
    /// Real (unpadded) soft-token count per image,
    /// `num_patches / pooling_kernel_size²`.
    pub num_soft_tokens_per_image: Vec<usize>,
}

/// Gemma 4 image processor: aspect-ratio-preserving resize into a patch budget,
/// then patchify + position ids + padding.
///
/// Unlike the fixed-size VLM processors above, Gemma 4 spends a *patch budget*
/// rather than a fixed resolution: an image is scaled so that it produces at
/// most `max_soft_tokens · pooling_kernel_size²` patches, with both sides
/// divisible by `pooling_kernel_size · patch_size`. Tall and wide images
/// therefore keep their aspect ratio and simply use a different patch grid,
/// which is why the tower needs explicit `(x, y)` coordinates instead of
/// inferring them from a known grid shape.
///
/// **Resampler caveat — the one inexact step.** The reference resamples through
/// Pillow (transformers' torchvision backend tracks it to within `1/255`,
/// because PyTorch's antialiased path reimplements Pillow's filters). We use
/// `image`'s `CatmullRom`, which is the same cubic kernel (`a = -0.5`, the
/// `B = 0, C = 0.5` spline), the same half-pixel alignment, and the same
/// downscale support scaling — but it samples vertical-before-horizontal
/// through an *unclamped* f32 intermediate where Pillow goes horizontal-first
/// and clips to `[0, 255]` between passes. Bicubic overshoot at a hard edge
/// therefore survives pmetal's first pass and is clipped in Pillow's, which is
/// where the two diverge: **up to 14/255 on a synthetic hard-edge pattern,
/// mean 0.17/255**. `fast_image_resize` was measured against the same fixture
/// and is no closer (15/255, mean 0.28), so this is the state of the art for an
/// off-the-shelf Rust resampler rather than a poor choice of one.
///
/// Everything downstream of the resize — rescale, patchify, position ids,
/// padding — is exact. `crates/pmetal-data/tests/gemma4_image_parity.rs` pins
/// the two halves separately: **atol 0** on an image the resize passes through
/// untouched, measured tolerance on the resampled ones.
#[derive(Debug, Clone)]
pub struct Gemma4ImageProcessor {
    config: Gemma4ImageProcessorConfig,
}

impl Gemma4ImageProcessor {
    /// Create a processor, validating the soft-token budget and geometry.
    pub fn new(config: Gemma4ImageProcessorConfig) -> Result<Self, Exception> {
        if !GEMMA4_SOFT_TOKEN_BUDGETS.contains(&config.max_soft_tokens) {
            return Err(Exception::custom(format!(
                "max_soft_tokens must be one of {:?}, got {}",
                GEMMA4_SOFT_TOKEN_BUDGETS, config.max_soft_tokens
            )));
        }
        if config.patch_size == 0 || config.pooling_kernel_size == 0 {
            return Err(Exception::custom(
                "patch_size and pooling_kernel_size must be non-zero",
            ));
        }
        Ok(Self { config })
    }

    /// Get the config.
    pub fn config(&self) -> &Gemma4ImageProcessorConfig {
        &self.config
    }

    /// Patch budget per image: `max_soft_tokens · pooling_kernel_size²`.
    pub fn max_patches(&self) -> usize {
        self.config.max_soft_tokens * self.config.pooling_kernel_size.pow(2)
    }

    /// Flattened patch width: `3 · patch_size²`.
    pub fn patch_dim(&self) -> usize {
        3 * self.config.patch_size.pow(2)
    }

    /// Largest `(height, width)` that preserves the aspect ratio, fits the
    /// patch budget, and has both sides divisible by
    /// `pooling_kernel_size · patch_size`.
    ///
    /// Mirrors the reference `get_aspect_ratio_preserving_size`, including its
    /// degenerate-side fallback: when one ideal side floors to zero, that axis
    /// is pinned to a single pooling block and the other is derived from the
    /// *integer* aspect ratio, clamped to the budget's longest side.
    pub fn target_size(&self, height: u32, width: u32) -> Result<(u32, u32), Exception> {
        if height == 0 || width == 0 {
            return Err(Exception::custom("cannot preprocess a zero-sized image"));
        }
        let patch = self.config.patch_size;
        let side = self.config.pooling_kernel_size * patch;
        let target_px = self.max_patches() * patch.pow(2);

        let factor = (target_px as f64 / (height as f64 * width as f64)).sqrt();
        let mut target_h = (factor * height as f64 / side as f64).floor() as usize * side;
        let mut target_w = (factor * width as f64 / side as f64).floor() as usize * side;

        if target_h == 0 && target_w == 0 {
            return Err(Exception::custom(format!(
                "resizing {height}x{width} into {target_px} pixels rounds to 0x0; both sides \
                 must reach pooling_kernel_size · patch_size = {side}"
            )));
        }
        // One pooling block on the short axis, integer aspect ratio on the long
        // one — capped so a pathological ratio cannot blow the patch budget.
        let max_side = self.config.max_soft_tokens * side;
        if target_h == 0 {
            target_h = side;
            target_w = ((width / height) as usize * side).min(max_side);
        } else if target_w == 0 {
            target_w = side;
            target_h = ((height / width) as usize * side).min(max_side);
        }

        if target_h * target_w > target_px {
            return Err(Exception::custom(format!(
                "resizing {height}x{width} to {target_h}x{target_w} exceeds the \
                 {} patch budget at patch_size {patch}",
                self.max_patches()
            )));
        }
        Ok((target_h as u32, target_w as u32))
    }

    /// Per-channel `(scale, shift)` folding rescale and normalise into one
    /// affine pass, evaluated in f64: `out = raw · scale + shift`.
    ///
    /// Mirrors the reference `rescale_and_normalize`, which folds
    /// `rescale_factor` into mean/std when both steps are on and otherwise
    /// applies whichever single step is enabled. For Gemma 4's shipped config
    /// (rescale only, identity mean/std) the f64 evaluation reproduces the
    /// reference's `astype(float64) · scale → float32` bit-for-bit. With
    /// `do_normalize` on the fold skips an intermediate f32 rounding the
    /// reference performs between the two steps — sub-ULP, and no released
    /// Gemma 4 checkpoint enables it.
    fn channel_affine(&self) -> [(f64, f64); 3] {
        let scale = if self.config.do_rescale {
            self.config.rescale_factor
        } else {
            1.0
        };
        if !self.config.do_normalize {
            return [(scale, 0.0); 3];
        }
        let mut affine = [(scale, 0.0); 3];
        for (ch, slot) in affine.iter_mut().enumerate() {
            let std = self.config.image_std[ch] as f64;
            *slot = (scale / std, -(self.config.image_mean[ch] as f64) / std);
        }
        affine
    }

    /// Resize, rescale, patchify and pad one image, appending to the batch
    /// buffers. Returns the image's real soft-token count.
    fn append_image(
        &self,
        img: &DynamicImage,
        pixels: &mut Vec<f32>,
        positions: &mut Vec<i32>,
    ) -> Result<usize, Exception> {
        // `do_convert_rgb`: alpha is dropped rather than composited, matching
        // PIL's `convert("RGB")`.
        let rgb = img.to_rgb8();
        let (height, width) = (rgb.height(), rgb.width());
        let (target_h, target_w) = if self.config.do_resize {
            self.target_size(height, width)?
        } else {
            (height, width)
        };
        let resized: RgbImage = if (target_h, target_w) == (height, width) {
            rgb
        } else {
            image::imageops::resize(&rgb, target_w, target_h, FilterType::CatmullRom)
        };

        let patch = self.config.patch_size;
        if target_h as usize % patch != 0 || target_w as usize % patch != 0 {
            return Err(Exception::custom(format!(
                "image {target_h}x{target_w} is not divisible by patch_size {patch}; enable \
                 `do_resize` so the aspect-ratio-preserving resize can align it"
            )));
        }
        let (rows, cols) = (target_h as usize / patch, target_w as usize / patch);
        let kernel = self.config.pooling_kernel_size;
        if rows % kernel != 0 || cols % kernel != 0 {
            return Err(Exception::custom(format!(
                "patch grid {rows}x{cols} is not divisible by pooling_kernel_size {kernel}, so \
                 the tower's pooling blocks would be partial; enable `do_resize`"
            )));
        }
        let num_patches = rows * cols;
        let max_patches = self.max_patches();
        if num_patches > max_patches {
            return Err(Exception::custom(format!(
                "image {target_h}x{target_w} yields {num_patches} patches, over the \
                 {max_patches} budget"
            )));
        }

        let affine = self.channel_affine();
        let raw = resized.as_raw();
        let stride = target_w as usize * 3;
        for patch_row in 0..rows {
            for patch_col in 0..cols {
                // `(row, col, channel)` within the patch: the flattening order
                // the tower's `input_proj` was trained against.
                for r in 0..patch {
                    let row_base = (patch_row * patch + r) * stride + patch_col * patch * 3;
                    for c in 0..patch {
                        let px = row_base + c * 3;
                        for (ch, &(scale, shift)) in affine.iter().enumerate() {
                            pixels.push((raw[px + ch] as f64 * scale + shift) as f32);
                        }
                    }
                }
                positions.push(patch_col as i32);
                positions.push(patch_row as i32);
            }
        }

        let pad = max_patches - num_patches;
        pixels.resize(pixels.len() + pad * self.patch_dim(), 0.0);
        positions.resize(positions.len() + pad * 2, -1);
        Ok(num_patches / kernel.pow(2))
    }

    /// Preprocess a batch of images. Each is resized independently — they may
    /// have different aspect ratios — then padded to the shared patch budget.
    pub fn preprocess(&self, images: &[DynamicImage]) -> Result<Gemma4ImageBatch, Exception> {
        if images.is_empty() {
            return Err(Exception::custom("Empty image batch"));
        }
        let max_patches = self.max_patches();
        let patch_dim = self.patch_dim();
        let mut pixels = Vec::with_capacity(images.len() * max_patches * patch_dim);
        let mut positions = Vec::with_capacity(images.len() * max_patches * 2);
        let mut num_soft_tokens_per_image = Vec::with_capacity(images.len());
        for img in images {
            num_soft_tokens_per_image.push(self.append_image(img, &mut pixels, &mut positions)?);
        }

        let batch = images.len() as i32;
        Ok(Gemma4ImageBatch {
            pixel_values: Array::from_f32_slice(
                &pixels,
                &[batch, max_patches as i32, patch_dim as i32],
            ),
            image_position_ids: Array::from_i32_slice_shaped(
                &positions,
                &[batch, max_patches as i32, 2],
            ),
            num_soft_tokens_per_image,
        })
    }

    /// Preprocess a batch from file paths.
    pub fn preprocess_paths(
        &self,
        paths: &[impl AsRef<Path>],
    ) -> Result<Gemma4ImageBatch, Exception> {
        let images: Result<Vec<_>, _> = paths
            .iter()
            .map(|p| {
                image::open(p)
                    .map_err(|e| Exception::custom(format!("Failed to open image: {}", e)))
            })
            .collect();
        self.preprocess(&images?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_creation() {
        let config = MllamaImageProcessorConfig::default();
        let processor = MllamaImageProcessor::new(config);

        assert_eq!(processor.config().size, (560, 560));
    }

    #[test]
    fn test_normalization_values() {
        let config = MllamaImageProcessorConfig::default();

        // CLIP stats should be correct
        assert!((config.mean[0] - 0.48145466).abs() < 1e-6);
        assert!((config.std[0] - 0.26862954).abs() < 1e-6);
    }

    #[test]
    fn test_siglip_processor() {
        let processor = SiglipImageProcessor::new((384, 384));

        // SigLIP uses 0.5 mean/std
        assert_eq!(processor.config.mean, [0.5, 0.5, 0.5]);
        assert_eq!(processor.config.std, [0.5, 0.5, 0.5]);
    }

    #[test]
    #[ignore = "requires a functional MLX array backend"]
    fn test_synthetic_image_processing() {
        let config = MllamaImageProcessorConfig {
            size: (4, 4), // Small for testing
            ..Default::default()
        };
        let processor = MllamaImageProcessor::new(config);

        // Create a simple synthetic image
        let img_buf = image::RgbImage::from_fn(4, 4, |_x, _y| image::Rgb([128u8, 64, 192]));
        let img = DynamicImage::ImageRgb8(img_buf);

        let mut result = processor.process_image(img).unwrap();
        result.eval();

        // Check shape: [1, 3, 4, 4]
        assert_eq!(result.shape(), &[1, 3, 4, 4]);

        // Check normalization was applied (values should not be 0-255)
        let n = result.shape().iter().product::<i32>() as usize;
        let vals: Vec<f32> = result.to_f32_vec(n).unwrap();

        // With CLIP normalization, 128 in red channel becomes:
        // (128/255 - 0.48145466) / 0.26862954 ≈ 0.082
        let expected_r = (128.0 / 255.0 - 0.48145466) / 0.26862954;
        assert!((vals[0] - expected_r).abs() < 0.01);
    }

    // ── Gemma 4 ─────────────────────────────────────────────────────────────

    fn gemma4_processor(config: Gemma4ImageProcessorConfig) -> Gemma4ImageProcessor {
        Gemma4ImageProcessor::new(config).expect("config is valid")
    }

    #[test]
    fn gemma4_rejects_untrained_soft_token_budget() {
        let config = Gemma4ImageProcessorConfig {
            max_soft_tokens: 300,
            ..Default::default()
        };
        assert!(Gemma4ImageProcessor::new(config).is_err());
        for budget in GEMMA4_SOFT_TOKEN_BUDGETS {
            let config = Gemma4ImageProcessorConfig {
                max_soft_tokens: budget,
                ..Default::default()
            };
            assert!(Gemma4ImageProcessor::new(config).is_ok(), "budget {budget}");
        }
    }

    #[test]
    fn gemma4_default_geometry_matches_checkpoints() {
        let processor = gemma4_processor(Gemma4ImageProcessorConfig::default());
        assert_eq!(processor.max_patches(), 2520); // 280 · 3²
        assert_eq!(processor.patch_dim(), 768); // 3 · 16²
    }

    /// Values produced by the reference `get_aspect_ratio_preserving_size` at
    /// `patch_size=16, max_soft_tokens=280, pooling_kernel_size=3`
    /// (`side_mult = 48`, budget 645120 px).
    #[test]
    fn gemma4_target_size_matches_reference_table() {
        let processor = gemma4_processor(Gemma4ImageProcessorConfig::default());
        let table = [
            ((896, 896), (768, 768)),
            ((1024, 768), (912, 672)),
            ((480, 640), (672, 912)),
            ((3000, 2000), (960, 624)),
            // Below the budget: the resize scales *up* to spend it.
            ((64, 64), (768, 768)),
            ((48, 48), (768, 768)),
            // Degenerate axis: the short side floors to 0, so it is pinned to
            // one pooling block and the long side is clamped to 280 · 48.
            ((4, 4000), (48, 13440)),
            ((4000, 4), (13440, 48)),
            ((12, 10000), (48, 13440)),
        ];
        for ((height, width), expected) in table {
            assert_eq!(
                processor.target_size(height, width).expect("resolves"),
                expected,
                "{height}x{width}"
            );
        }
        assert!(processor.target_size(0, 32).is_err());
    }

    /// The patch flattening, position ids and padding are exact bookkeeping, so
    /// they can be asserted by construction: each pixel is stamped with its own
    /// coordinates, and every output slot is checked against where it came from.
    #[test]
    fn gemma4_patch_layout_positions_and_padding() {
        // 12x24 with patch 4 / kernel 3 → a 3x6 patch grid = 18 of 630 patches,
        // so the padding tail is exercised too. `do_resize` off keeps the
        // resampler out of it.
        let config = Gemma4ImageProcessorConfig {
            patch_size: 4,
            max_soft_tokens: 70,
            pooling_kernel_size: 3,
            do_resize: false,
            ..Default::default()
        };
        let processor = gemma4_processor(config);
        let (height, width) = (12u32, 24u32);
        // Stamp each pixel with its coordinates: red = x, green = y, blue = a
        // mix, so a transposed or channel-swapped read cannot pass.
        let img = DynamicImage::ImageRgb8(RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([x as u8, y as u8, (x * 3 + y * 7) as u8])
        }));

        let batch = processor.preprocess(&[img]).expect("preprocess runs");
        assert_eq!(batch.num_soft_tokens_per_image, vec![2]); // 18 / 3²
        assert_eq!(batch.pixel_values.shape(), &[1, 630, 48]);
        assert_eq!(batch.image_position_ids.shape(), &[1, 630, 2]);

        let pixels = batch.pixel_values.clone();
        let mut pixels_eval = pixels;
        pixels_eval.eval();
        let pixels = pixels_eval.to_f32_vec(630 * 48).expect("materialises");
        let mut positions_eval = batch.image_position_ids.clone();
        positions_eval.eval();
        let positions = positions_eval.to_f32_vec(630 * 2).expect("materialises");

        let (rows, cols) = (3usize, 6usize);
        for patch_row in 0..rows {
            for patch_col in 0..cols {
                let p = patch_row * cols + patch_col;
                // `(x, y)` = `(column, row)`, not the other way round.
                assert_eq!(positions[p * 2], patch_col as f32, "patch {p} x");
                assert_eq!(positions[p * 2 + 1], patch_row as f32, "patch {p} y");
                for r in 0..4usize {
                    for c in 0..4usize {
                        let (x, y) = (patch_col * 4 + c, patch_row * 4 + r);
                        // Within a patch: row-major with the channel innermost.
                        let base = p * 48 + (r * 4 + c) * 3;
                        let expected = [x as f32, y as f32, ((x * 3 + y * 7) % 256) as f32];
                        for (ch, want) in expected.iter().enumerate() {
                            // `do_rescale` is on by default: [0, 255] → [0, 1].
                            let got = pixels[base + ch] * 255.0;
                            assert!(
                                (got - want).abs() < 1e-3,
                                "patch {p} ({r},{c}) ch{ch}: got {got}, want {want}"
                            );
                        }
                    }
                }
            }
        }
        // Padding: zeroed pixels, `(-1, -1)` positions.
        assert!(pixels[rows * cols * 48..].iter().all(|&v| v == 0.0));
        assert!(positions[rows * cols * 2..].iter().all(|&v| v == -1.0));
    }

    #[test]
    fn gemma4_unaligned_image_is_rejected_without_resize() {
        let config = Gemma4ImageProcessorConfig {
            patch_size: 4,
            max_soft_tokens: 70,
            pooling_kernel_size: 3,
            do_resize: false,
            ..Default::default()
        };
        let processor = gemma4_processor(config);
        // 10 is not a multiple of patch_size.
        let unaligned = DynamicImage::ImageRgb8(RgbImage::new(24, 10));
        assert!(processor.preprocess(&[unaligned]).is_err());
        // 4x4 patches, but the 1x1 patch grid is not divisible by the pooling
        // kernel, so the tower's pooling blocks would be partial.
        let unpooled = DynamicImage::ImageRgb8(RgbImage::new(4, 4));
        assert!(processor.preprocess(&[unpooled]).is_err());
    }

    #[test]
    fn gemma4_channel_affine_folds_rescale_and_normalize() {
        let config = Gemma4ImageProcessorConfig {
            do_normalize: true,
            image_mean: [0.1, 0.2, 0.3],
            image_std: [0.5, 0.25, 0.125],
            ..Default::default()
        };
        let processor = gemma4_processor(config);
        let affine = processor.channel_affine();
        for (ch, &(scale, shift)) in affine.iter().enumerate() {
            let (mean, std) = (
                processor.config.image_mean[ch] as f64,
                processor.config.image_std[ch] as f64,
            );
            // The reference applies `(raw / 255 − mean) / std`.
            let expected = (200.0 / 255.0 - mean) / std;
            assert!((200.0 * scale + shift - expected).abs() < 1e-9);
        }
        // Gemma 4's shipped config: rescale only, identity mean/std.
        let plain = gemma4_processor(Gemma4ImageProcessorConfig::default());
        assert_eq!(plain.channel_affine(), [(1.0f64 / 255.0, 0.0); 3]);
    }
}
