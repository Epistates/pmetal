//! Image processing utilities for Vision Language Models.
//!
//! Provides efficient image preprocessing for VLMs like Llama 3.2 Vision.
//! Uses CLIP-style normalization and supports batch processing.

use image::{imageops::FilterType, DynamicImage};
use mlx_rs::{error::Exception, Array};
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
        self.mean_array = Some(Array::from_slice(&self.config.mean, &[1, 3, 1, 1]));
        // Std: [1, 3, 1, 1]
        self.std_array = Some(Array::from_slice(&self.config.std, &[1, 3, 1, 1]));
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
        let array = Array::from_slice(&data, shape);

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
        let arr = Array::from_slice(&data, shape);

        // GPU operations: rescale then normalize
        let rescale = Array::from_f32(self.config.rescale_factor);
        let scaled = arr.multiply(&rescale)?;
        let centered = scaled.subtract(mean)?;
        let normalized = centered.divide(std)?;

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
        mlx_rs::ops::concatenate_axis(&batch_refs, 0)
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
    fn test_synthetic_image_processing() {
        let config = MllamaImageProcessorConfig {
            size: (4, 4), // Small for testing
            ..Default::default()
        };
        let processor = MllamaImageProcessor::new(config);

        // Create a simple synthetic image
        let img_buf = image::RgbImage::from_fn(4, 4, |_x, _y| image::Rgb([128u8, 64, 192]));
        let img = DynamicImage::ImageRgb8(img_buf);

        let result = processor.process_image(img).unwrap();
        result.eval().unwrap();

        // Check shape: [1, 3, 4, 4]
        assert_eq!(result.shape(), &[1, 3, 4, 4]);

        // Check normalization was applied (values should not be 0-255)
        let vals: Vec<f32> = result.as_slice::<f32>().to_vec();

        // With CLIP normalization, 128 in red channel becomes:
        // (128/255 - 0.48145466) / 0.26862954 ≈ 0.082
        let expected_r = (128.0 / 255.0 - 0.48145466) / 0.26862954;
        assert!((vals[0] - expected_r).abs() < 0.01);
    }
}
