//! Image processing utilities for Vision Language Models.
//!
//! Two families live here:
//!
//! * [`FixedSizeImageProcessor`] / [`SiglipImageProcessor`] — plain
//!   resize-to-a-fixed-square `[N, 3, H, W]` preprocessing, used for generic
//!   image training data.
//! * [`Gemma4ImageProcessor`] — Gemma 4's patch-budget preprocessing, which
//!   resizes to preserve aspect ratio, patchifies to
//!   `[B, max_patches, 3·patch²]`, and emits the `(x, y)` patch coordinates the
//!   vision tower's 2-D position table and pooler are indexed by.

use image::{DynamicImage, RgbImage};
use pmetal_bridge::compat::{Array, Exception};
use serde::Deserialize;
use std::path::Path;

use crate::pillow_resample::{self, ResampleFilter};

/// Configuration for [`FixedSizeImageProcessor`].
#[derive(Debug, Clone)]
pub struct FixedSizeImageProcessorConfig {
    /// Target image size (width, height).
    pub size: (u32, u32),
    /// Normalization mean (RGB).
    pub mean: [f32; 3],
    /// Normalization standard deviation (RGB).
    pub std: [f32; 3],
    /// Rescaling factor (e.g., 1/255.0).
    pub rescale_factor: f32,
    /// Resampling filter, corresponding to `preprocessor_config.json`'s
    /// `resample`. Llama 3.2 Vision ships `2` = bilinear.
    pub resample: ResampleFilter,
}

impl Default for FixedSizeImageProcessorConfig {
    fn default() -> Self {
        Self {
            // Llama 3.2 11B Vision's tile size and stats. NOTE: the released
            // checkpoint does *not* simply resize to this — it fits the image
            // into a tiled canvas of up to `max_image_tiles` 560x560 tiles.
            // These defaults reproduce its colour handling, not its geometry.
            size: (560, 560),
            // CLIP stats (canonical values from OpenAI CLIP)
            #[allow(clippy::excessive_precision)]
            mean: [0.48145466, 0.4578275, 0.40821073],
            #[allow(clippy::excessive_precision)]
            std: [0.26862954, 0.26130258, 0.27577711],
            rescale_factor: 1.0 / 255.0,
            resample: ResampleFilter::Bilinear,
        }
    }
}

/// Fixed-size CLIP-style image processor: stretch to a fixed square, rescale,
/// normalise.
///
/// This is the generic training-data path — it is *not* any released VLM's
/// preprocessing, because every one of them preserves aspect ratio somehow.
/// Llama 3.2 Vision in particular tiles (see the caveat on the defaults below).
///
/// Supports:
/// - Single image preprocessing
/// - Batch preprocessing
/// - GPU-accelerated normalization via MLX
#[derive(Debug, Clone)]
pub struct FixedSizeImageProcessor {
    config: FixedSizeImageProcessorConfig,
    /// Pre-computed normalization arrays for GPU processing.
    mean_array: Option<Array>,
    std_array: Option<Array>,
}

impl FixedSizeImageProcessor {
    /// Create a new processor.
    pub fn new(config: FixedSizeImageProcessorConfig) -> Self {
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
        // 1. Convert to RGB, then resize — the reference order (`do_convert_rgb`
        // runs before the resize), and the resampler is Pillow-exact.
        let rgb = pillow_resample::resize_rgb8(
            &img.to_rgb8(),
            self.config.size.0,
            self.config.size.1,
            self.config.resample,
        );

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

        // 1. Convert to RGB, then resize (see `process_image`).
        let rgb = pillow_resample::resize_rgb8(
            &img.to_rgb8(),
            self.config.size.0,
            self.config.size.1,
            self.config.resample,
        );

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
    pub fn config(&self) -> &FixedSizeImageProcessorConfig {
        &self.config
    }
}

/// SigLIP-style image processor with different normalization.
#[derive(Debug, Clone)]
pub struct SiglipImageProcessor {
    config: FixedSizeImageProcessorConfig,
}

impl SiglipImageProcessor {
    /// Create a new SigLIP processor.
    pub fn new(size: (u32, u32)) -> Self {
        Self {
            config: FixedSizeImageProcessorConfig {
                size,
                // SigLIP normalises with `IMAGENET_STANDARD_MEAN/STD`, not
                // CLIP's stats, and resamples bicubic (`"resample": 3`) where
                // Llama 3.2 Vision uses bilinear.
                mean: [0.5, 0.5, 0.5],
                std: [0.5, 0.5, 0.5],
                rescale_factor: 1.0 / 255.0,
                resample: ResampleFilter::Bicubic,
            },
        }
    }

    /// Process an image.
    pub fn process_image(&self, img: DynamicImage) -> Result<Array, Exception> {
        let processor = FixedSizeImageProcessor::new(self.config.clone());
        processor.process_image(img)
    }
}

/// The reference's rescale-then-normalise chain, reproduced step for step.
///
/// Both halves have a precision quirk that matters if you want to assert
/// bit-exactness rather than a tolerance:
///
/// * the rescale is `image.astype(np.float64) * scale` — done in **f64** and
///   only then narrowed, so an f32 `rescale_factor` lands 1 ULP off;
/// * the normalisation is `(image - mean) / std` in **f32**, a real divide.
///   Folding it into `x · (1/std) + shift` costs another ULP.
///
/// Folding both into a single affine pass is tempting and measurably wrong, so
/// this keeps the reference's staging. Shared by every processor here, because
/// the arithmetic is the same wherever `do_rescale` / `do_normalize` appear.
#[derive(Debug, Clone, Copy)]
struct PixelNormalizer {
    /// `Some(factor)` when `do_rescale`.
    rescale: Option<f64>,
    /// `Some((mean, std))` when `do_normalize`.
    normalize: Option<([f32; 3], [f32; 3])>,
}

impl PixelNormalizer {
    fn new(
        do_rescale: bool,
        rescale_factor: f64,
        do_normalize: bool,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Self {
        Self {
            rescale: do_rescale.then_some(rescale_factor),
            normalize: do_normalize.then_some((mean, std)),
        }
    }

    /// Map one raw sample in `channel` to its model-ready value.
    #[inline]
    fn apply(&self, raw: u8, channel: usize) -> f32 {
        let value = match self.rescale {
            Some(factor) => (raw as f64 * factor) as f32,
            None => raw as f32,
        };
        match self.normalize {
            Some((mean, std)) => (value - mean[channel]) / std[channel],
            None => value,
        }
    }

    /// The value a zero-valued (padding) pixel takes after the chain — the
    /// reference pads *before* normalising, so padded regions are not zero.
    #[inline]
    fn padding_value(&self, channel: usize) -> f32 {
        self.apply(0, channel)
    }
}

/// Every `(tiles_high, tiles_wide)` arrangement that fits inside `max_tiles`.
///
/// The order is load-bearing: an image's `aspect_ratio_id` is its index in this
/// list **plus one** (0 is reserved for batch padding), and the model looks that
/// id up in a precomputed embedding table. Generated exactly as the reference
/// does — height-major within a width loop — so the ids agree.
///
/// For `max_tiles = 4`:
/// `[(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (3,1), (4,1)]`.
pub fn supported_aspect_ratios(max_tiles: usize) -> Vec<(usize, usize)> {
    let mut ratios = Vec::new();
    for width in 1..=max_tiles {
        for height in 1..=max_tiles {
            if width * height <= max_tiles {
                ratios.push((width, height));
            }
        }
    }
    ratios
}

/// Configuration for [`MllamaImageProcessor`].
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MllamaImageProcessorConfig {
    /// Tile side length in pixels. Must be square — the reference validates it.
    pub tile_size: u32,
    /// Most tiles one image may be split into.
    pub max_image_tiles: usize,
    /// Whether to scale raw `[0, 255]` samples by [`Self::rescale_factor`].
    pub do_rescale: bool,
    /// Pixel scale, `1/255`. `f64` for the reason spelled out on
    /// [`PixelNormalizer`]: the reference rescales in f64, so an f32 factor is
    /// already rounded before the multiply and lands a ULP off.
    pub rescale_factor: f64,
    /// Whether to apply mean/std normalisation after rescaling.
    pub do_normalize: bool,
    /// Per-channel mean.
    pub image_mean: [f32; 3],
    /// Per-channel standard deviation.
    pub image_std: [f32; 3],
    /// Resampling filter; Llama 3.2 Vision ships `2` = bilinear.
    pub resample: ResampleFilter,
}

impl Default for MllamaImageProcessorConfig {
    fn default() -> Self {
        Self {
            tile_size: 560,
            max_image_tiles: 4,
            do_rescale: true,
            rescale_factor: 1.0 / 255.0,
            do_normalize: true,
            // CLIP stats, as the released Llama 3.2 Vision checkpoint ships.
            #[allow(clippy::excessive_precision)]
            image_mean: [0.48145466, 0.4578275, 0.40821073],
            #[allow(clippy::excessive_precision)]
            image_std: [0.26862954, 0.26130258, 0.27577711],
            resample: ResampleFilter::Bilinear,
        }
    }
}

/// Preprocessed images in the layout `MllamaVisionModel::forward` consumes.
#[derive(Debug, Clone)]
pub struct MllamaImageBatch {
    /// `[B, max_num_images, max_image_tiles, 3, tile, tile]`. Unused image and
    /// tile slots are zero.
    pub pixel_values: Array,
    /// `[B, max_num_images]` — index into [`supported_aspect_ratios`] plus one,
    /// `0` where a batch row has fewer images than the widest one. The vision
    /// tower indexes its tile-embedding tables by this.
    pub aspect_ratio_ids: Array,
    /// `[B, max_num_images, max_image_tiles]` — `1` for real tiles, `0` for
    /// padding, so attention can ignore the padded tiles.
    pub aspect_ratio_mask: Array,
    /// Real tile count per image, per batch row.
    pub num_tiles: Vec<Vec<usize>>,
}

/// Llama 3.2 Vision (Mllama) image processor: fit into a tiled canvas, pad,
/// split into tiles.
///
/// Mllama does not resize to a fixed square. It picks the arrangement of up to
/// `max_image_tiles` `tile_size²` tiles whose canvas best matches the image's
/// aspect ratio, scales the image to fit inside that canvas *without* distorting
/// it, pads the remainder, and splits the result into tiles the vision tower
/// encodes independently before a global transformer attends across them. The
/// chosen arrangement is reported back as `aspect_ratio_ids`, which the tower
/// uses to look up per-tile position embeddings — so the geometry is not merely
/// a preprocessing detail, it is an input the model reads.
///
/// One ordering detail worth knowing: the reference **pads before it
/// normalises**, so padded pixels leave the processor at `-mean/std`, not at
/// zero. `aspect_ratio_mask` is what marks them, not their value.
#[derive(Debug, Clone)]
pub struct MllamaImageProcessor {
    config: MllamaImageProcessorConfig,
    /// `supported_aspect_ratios(max_image_tiles)`, cached for id lookup.
    ratios: Vec<(usize, usize)>,
}

impl MllamaImageProcessor {
    /// Create a processor, validating the tiling geometry.
    pub fn new(config: MllamaImageProcessorConfig) -> Result<Self, Exception> {
        if config.tile_size == 0 || config.max_image_tiles == 0 {
            return Err(Exception::custom(
                "tile_size and max_image_tiles must be non-zero",
            ));
        }
        let ratios = supported_aspect_ratios(config.max_image_tiles);
        Ok(Self { config, ratios })
    }

    /// Get the config.
    pub fn config(&self) -> &MllamaImageProcessorConfig {
        &self.config
    }

    /// The tile arrangements this processor can emit, in `aspect_ratio_id`
    /// order (id = index + 1).
    pub fn aspect_ratios(&self) -> &[(usize, usize)] {
        &self.ratios
    }

    fn normalizer(&self) -> PixelNormalizer {
        PixelNormalizer::new(
            self.config.do_rescale,
            self.config.rescale_factor,
            self.config.do_normalize,
            self.config.image_mean,
            self.config.image_std,
        )
    }

    /// The value padded pixels carry once the batch is built. Exposed because
    /// it is not zero and callers reasonably expect it to be.
    pub fn padding_value(&self, channel: usize) -> f32 {
        self.normalizer().padding_value(channel)
    }

    /// Best `(tiles_high, tiles_wide)` arrangement for an image.
    ///
    /// Mirrors the reference `get_optimal_tiled_canvas`: score every arrangement
    /// by `min(canvas_h / h, canvas_w / w)` — the factor that makes the image
    /// fit — then prefer the *smallest* upscale if any arrangement can hold the
    /// image at full size, else the *largest* downscale. Ties break toward the
    /// smallest canvas area, so a square image gets 1x1 rather than a padded
    /// 2x2.
    pub fn optimal_tiling(&self, height: u32, width: u32) -> Result<(usize, usize), Exception> {
        if height == 0 || width == 0 {
            return Err(Exception::custom("cannot preprocess a zero-sized image"));
        }
        let tile = self.config.tile_size as f64;
        // The reference compares scales with `==` against the selected value, so
        // these must be computed exactly as it computes them.
        let scale_of = |&(tiles_h, tiles_w): &(usize, usize)| {
            let scale_h = (tiles_h as f64 * tile) / height as f64;
            let scale_w = (tiles_w as f64 * tile) / width as f64;
            if scale_w > scale_h { scale_h } else { scale_w }
        };

        let scales: Vec<f64> = self.ratios.iter().map(scale_of).collect();
        let upscales: Vec<f64> = scales.iter().copied().filter(|&s| s >= 1.0).collect();
        let selected = if !upscales.is_empty() {
            upscales.iter().copied().fold(f64::INFINITY, f64::min)
        } else {
            scales
                .iter()
                .copied()
                .filter(|&s| s < 1.0)
                .fold(f64::NEG_INFINITY, f64::max)
        };

        // `argmin` over ties keeps the first, as numpy does.
        self.ratios
            .iter()
            .zip(&scales)
            .filter(|&(_, &s)| s == selected)
            .map(|(&ratio, _)| ratio)
            .min_by_key(|&(tiles_h, tiles_w)| tiles_h * tiles_w)
            .ok_or_else(|| Exception::custom("no tile arrangement matched"))
    }

    /// Size the image should be scaled to inside `canvas`, preserving aspect
    /// ratio. Mirrors the reference `get_image_size_fit_to_canvas`.
    fn fit_to_canvas(&self, height: u32, width: u32, canvas: (u32, u32)) -> (u32, u32) {
        let tile = self.config.tile_size;
        let target_h = height.clamp(tile, canvas.0) as f64;
        let target_w = width.clamp(tile, canvas.1) as f64;
        let scale_h = target_h / height as f64;
        let scale_w = target_w / width as f64;

        if scale_w < scale_h {
            // Width is the binding axis; derive the height and never exceed the
            // canvas. The `.max(1)` mirrors the reference's `or 1` guard against
            // an extreme aspect ratio flooring a side to zero.
            let new_h = ((height as f64 * scale_w).floor() as u32).max(1);
            (new_h.min(target_h as u32), target_w as u32)
        } else {
            let new_w = ((width as f64 * scale_h).floor() as u32).max(1);
            (target_h as u32, new_w.min(target_w as u32))
        }
    }

    /// Resize, pad, normalise and split one image into `[num_tiles, 3, t, t]`
    /// worth of f32, appended to `out` in tile-row-major order.
    fn append_tiles(
        &self,
        img: &DynamicImage,
        out: &mut Vec<f32>,
    ) -> Result<(usize, usize), Exception> {
        let rgb = img.to_rgb8();
        let (height, width) = (rgb.height(), rgb.width());
        let (tiles_h, tiles_w) = self.optimal_tiling(height, width)?;
        let tile = self.config.tile_size;
        let canvas = (tiles_h as u32 * tile, tiles_w as u32 * tile);
        let (fit_h, fit_w) = self.fit_to_canvas(height, width, canvas);

        let resized = pillow_resample::resize_rgb8(&rgb, fit_w, fit_h, self.config.resample);
        let raw = resized.as_raw();

        let normalizer = self.normalizer();

        // Tiles are emitted row-major, each as a contiguous `[3, t, t]` block.
        // The image occupies the top-left `fit_h x fit_w` of the canvas and the
        // rest is padding — zero *before* the normalisation, so it comes out at
        // `padding_value`, not at zero.
        let tile = tile as usize;
        for tile_row in 0..tiles_h {
            for tile_col in 0..tiles_w {
                for ch in 0..3 {
                    let pad = normalizer.padding_value(ch);
                    for r in 0..tile {
                        let y = tile_row * tile + r;
                        for c in 0..tile {
                            let x = tile_col * tile + c;
                            out.push(if y < fit_h as usize && x < fit_w as usize {
                                normalizer.apply(raw[(y * fit_w as usize + x) * 3 + ch], ch)
                            } else {
                                pad
                            });
                        }
                    }
                }
            }
        }
        Ok((tiles_h, tiles_w))
    }

    /// Preprocess a batch of samples, each carrying one or more images.
    ///
    /// Samples may hold different numbers of images and images may tile
    /// differently; everything is padded up to the widest sample and to
    /// `max_image_tiles`.
    pub fn preprocess(&self, samples: &[Vec<DynamicImage>]) -> Result<MllamaImageBatch, Exception> {
        if samples.is_empty() || samples.iter().all(|s| s.is_empty()) {
            return Err(Exception::custom("Empty image batch"));
        }
        let batch = samples.len();
        let max_images = samples.iter().map(|s| s.len()).max().unwrap_or(0);
        let max_tiles = self.config.max_image_tiles;
        let tile = self.config.tile_size as usize;
        let tile_len = 3 * tile * tile;

        let mut pixels = vec![0f32; batch * max_images * max_tiles * tile_len];
        let mut ids = vec![0i32; batch * max_images];
        let mut mask = vec![0i32; batch * max_images * max_tiles];
        let mut num_tiles = Vec::with_capacity(batch);

        // Scratch reused per image so a 4-tile 560px image doesn't reallocate.
        let mut scratch = Vec::with_capacity(max_tiles * tile_len);

        for (sample_idx, sample) in samples.iter().enumerate() {
            let mut sample_tiles = Vec::with_capacity(sample.len());
            for (image_idx, img) in sample.iter().enumerate() {
                scratch.clear();
                let (tiles_h, tiles_w) = self.append_tiles(img, &mut scratch)?;
                let count = tiles_h * tiles_w;

                let base = ((sample_idx * max_images) + image_idx) * max_tiles * tile_len;
                pixels[base..base + scratch.len()].copy_from_slice(&scratch);

                let ratio_index = self
                    .ratios
                    .iter()
                    .position(|&r| r == (tiles_h, tiles_w))
                    .ok_or_else(|| {
                        Exception::custom(format!(
                            "tiling {tiles_h}x{tiles_w} is not a supported arrangement for \
                             max_image_tiles {max_tiles}"
                        ))
                    })?;
                ids[sample_idx * max_images + image_idx] = ratio_index as i32 + 1;

                // The reference marks tile 0 valid for *every* slot, including
                // padded ones, then fills in each image's real tiles.
                let mask_base = ((sample_idx * max_images) + image_idx) * max_tiles;
                for slot in 0..count {
                    mask[mask_base + slot] = 1;
                }
                sample_tiles.push(count);
            }
            for image_idx in 0..max_images {
                mask[((sample_idx * max_images) + image_idx) * max_tiles] = 1;
            }
            num_tiles.push(sample_tiles);
        }

        let tile_i32 = tile as i32;
        Ok(MllamaImageBatch {
            pixel_values: Array::from_f32_slice(
                &pixels,
                &[
                    batch as i32,
                    max_images as i32,
                    max_tiles as i32,
                    3,
                    tile_i32,
                    tile_i32,
                ],
            ),
            aspect_ratio_ids: Array::from_i32_slice_shaped(
                &ids,
                &[batch as i32, max_images as i32],
            ),
            aspect_ratio_mask: Array::from_i32_slice_shaped(
                &mask,
                &[batch as i32, max_images as i32, max_tiles as i32],
            ),
            num_tiles,
        })
    }

    /// Preprocess a single image as a one-sample, one-image batch.
    pub fn preprocess_one(&self, img: &DynamicImage) -> Result<MllamaImageBatch, Exception> {
        self.preprocess(std::slice::from_ref(&vec![img.clone()]))
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
/// **Bit-exact against the reference, resize included.** The resize goes through
/// [`crate::pillow_resample`], which reproduces Pillow's fixed-point resampling
/// rather than approximating it — off-the-shelf Rust resamplers (`image`'s
/// `CatmullRom`, `fast_image_resize`) get the kernel right but clamp bicubic
/// overshoot in the wrong place and land 14-15/255 off at hard edges. Rescale is
/// evaluated in f64 for the same reason the reference does. Patchify, position
/// ids and padding are integer bookkeeping.
///
/// So `crates/pmetal-data/tests/gemma4_image_parity.rs` asserts **atol 0** on
/// every checkpoint of every case, and there is no tolerance in it to loosen.
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

    /// The reference's rescale/normalise chain for this config.
    ///
    /// Gemma 4 ships `do_normalize = false` with `rescale_factor = 1/255`, so in
    /// practice this is a plain `[0, 255] -> [0, 1]` — but it goes through the
    /// shared [`PixelNormalizer`] so the f64 rescale and f32 divide match the
    /// reference exactly either way.
    fn normalizer(&self) -> PixelNormalizer {
        PixelNormalizer::new(
            self.config.do_rescale,
            self.config.rescale_factor,
            self.config.do_normalize,
            self.config.image_mean,
            self.config.image_std,
        )
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
            pillow_resample::resize_rgb8(&rgb, target_w, target_h, ResampleFilter::Bicubic)
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

        let normalizer = self.normalizer();
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
                        for ch in 0..3 {
                            pixels.push(normalizer.apply(raw[px + ch], ch));
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
        let config = FixedSizeImageProcessorConfig::default();
        let processor = FixedSizeImageProcessor::new(config);

        assert_eq!(processor.config().size, (560, 560));
    }

    #[test]
    fn test_normalization_values() {
        let config = FixedSizeImageProcessorConfig::default();

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
        // ...and bicubic resampling, where Llama 3.2 Vision ships bilinear.
        assert_eq!(processor.config.resample, ResampleFilter::Bicubic);
        assert_eq!(
            FixedSizeImageProcessorConfig::default().resample,
            ResampleFilter::Bilinear
        );
    }

    #[test]
    fn test_synthetic_image_processing() {
        let config = FixedSizeImageProcessorConfig {
            size: (4, 4), // Small for testing
            ..Default::default()
        };
        let processor = FixedSizeImageProcessor::new(config);

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

    /// The rescale/normalise chain must be staged exactly as the reference
    /// stages it: f64 rescale, narrow to f32, *then* `(x - mean) / std` in f32.
    /// Folding it into one affine is a ULP off, which is the whole distance
    /// between an atol-0 parity assertion and one carrying slack.
    #[test]
    fn pixel_normalizer_matches_reference_staging() {
        let mean = [0.1f32, 0.2, 0.3];
        let std = [0.5f32, 0.25, 0.125];
        let normalizer = PixelNormalizer::new(true, 1.0 / 255.0, true, mean, std);
        for raw in [0u8, 1, 37, 128, 200, 254, 255] {
            for ch in 0..3 {
                let staged = ((raw as f64 * (1.0 / 255.0)) as f32 - mean[ch]) / std[ch];
                assert_eq!(normalizer.apply(raw, ch), staged, "raw {raw} ch {ch}");
            }
        }
        // Padding enters as a zero *pixel*, so it inherits the normalisation.
        for ch in 0..3 {
            assert_eq!(normalizer.padding_value(ch), -mean[ch] / std[ch]);
        }

        // Gemma 4's shipped config: rescale only, identity mean/std.
        let plain = PixelNormalizer::new(true, 1.0 / 255.0, false, [0.0; 3], [1.0; 3]);
        assert_eq!(plain.apply(200, 0), (200.0f64 / 255.0) as f32);
        assert_eq!(plain.padding_value(2), 0.0);

        // Neither step: a straight widening.
        let raw_only = PixelNormalizer::new(false, 1.0 / 255.0, false, [0.0; 3], [1.0; 3]);
        assert_eq!(raw_only.apply(200, 1), 200.0);
    }
}
