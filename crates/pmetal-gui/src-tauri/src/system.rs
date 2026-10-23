//! This Mac: its chip, memory, and whether a model fits.

use pmetal::hub::{DeviceSpec, FitEstimate, ModelSpec};
use serde::Serialize;

use crate::hub::fit_name;

/// The Mac, as the Settings page and the dashboard show it.
#[derive(Debug, Serialize)]
pub struct SystemInfo {
    pub version: String,
    pub platform: String,
    pub arch: String,
    pub is_apple_silicon: bool,
    pub gpu_name: String,
    pub chip_tier: Option<String>,
    pub total_memory: u64,
    pub available_memory: u64,
    pub total_memory_formatted: String,
    pub available_memory_formatted: String,
    pub gpu_cores: Option<u32>,
    pub ane_cores: Option<u32>,
    pub memory_bandwidth_gbps: Option<f64>,
    pub has_ane: bool,
    pub has_nax: bool,
}

pub fn system_info() -> SystemInfo {
    let platform = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();
    let is_apple_silicon = arch == "aarch64" && platform == "macos";
    let device = pmetal::version::device_info();
    let gb = 1024.0_f64.powi(3);
    let total_memory = (device.memory_total_gb * gb) as u64;
    let available_memory = (device.memory_available_gb * gb) as u64;

    let mut info = SystemInfo {
        version: pmetal::version::VERSION.to_string(),
        platform,
        arch,
        is_apple_silicon,
        gpu_name: "Apple GPU".to_string(),
        chip_tier: None,
        total_memory,
        available_memory,
        total_memory_formatted: pmetal::core::format::human_memory(total_memory),
        available_memory_formatted: pmetal::core::format::human_memory(available_memory),
        gpu_cores: None,
        ane_cores: None,
        memory_bandwidth_gbps: None,
        has_ane: is_apple_silicon,
        has_nax: false,
    };
    if let Ok(ctx) = pmetal::metal::MetalContext::global() {
        let props = ctx.properties();
        info.gpu_name = props.name.clone();
        info.gpu_cores = Some(props.gpu_core_count);
        info.ane_cores = Some(props.ane_core_count);
        info.memory_bandwidth_gbps = Some(props.memory_bandwidth_gbps);
        info.has_ane = props.ane_core_count > 0;
        info.has_nax = props.has_nax;
        info.chip_tier = Some(format!("{:?}", props.device_tier).to_lowercase());
    }
    info
}

/// What the GPU can address and how fast, from the device's own report: never
/// a share of the installed memory, which is a different number on every Mac.
pub fn device_spec() -> Option<DeviceSpec> {
    let ctx = pmetal::metal::MetalContext::global().ok()?;
    let props = ctx.properties();
    Some(DeviceSpec::from_device(
        props.recommended_working_set_size,
        props.memory_bandwidth_gbps,
        props.has_unified_memory,
    ))
}

/// How a model would run, as the Models page shows it.
#[derive(Debug, Serialize, PartialEq)]
pub struct ModelFitInfo {
    pub inference_fit: String,
    pub training_fit: String,
    pub weights_gb: f64,
    pub inference_memory_gb: f64,
    pub training_memory_gb: f64,
    pub available_memory_gb: f64,
    pub estimated_tps: Option<f64>,
    pub recommended_batch_size: u32,
    /// Where the numbers came from, when it matters.
    pub notes: Vec<String>,
    /// Why pmetal cannot run the model, when its architecture is one pmetal
    /// declines. The memory figures are still given.
    pub unsupported: Option<String>,
}

impl ModelFitInfo {
    pub fn of(estimate: &FitEstimate) -> Self {
        Self {
            inference_fit: fit_name(estimate.fit_level).to_string(),
            training_fit: fit_name(estimate.training_fit).to_string(),
            weights_gb: estimate.weights_gb,
            inference_memory_gb: estimate.total_required_gb,
            training_memory_gb: estimate.training_memory_gb,
            available_memory_gb: estimate.available_gb,
            estimated_tps: (estimate.estimated_tps > 0.0).then_some(estimate.estimated_tps),
            recommended_batch_size: u32::try_from(estimate.recommended_batch_size)
                .unwrap_or(u32::MAX),
            notes: estimate.notes.clone(),
            unsupported: None,
        }
    }
}

/// How the model in `spec` runs on `device`.
pub fn fit(spec: &ModelSpec, device: &DeviceSpec) -> ModelFitInfo {
    ModelFitInfo {
        unsupported: spec.unsupported.clone(),
        ..ModelFitInfo::of(&pmetal::hub::estimate_fit(spec, device))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_that_needs_more_than_the_gpu_can_address_does_not_fit() {
        let device = DeviceSpec::from_device(8 << 30, 200.0, true);
        let small = fit(&ModelSpec::new(0.5, "fp16"), &device);
        let huge = fit(&ModelSpec::new(70.0, "fp16"), &device);
        assert_eq!(small.inference_fit, "fits");
        assert_eq!(huge.inference_fit, "too_large");
        assert!(huge.weights_gb > small.weights_gb);
        // The memory it is compared with is the device's, not the installed RAM.
        assert!((small.available_memory_gb - (8u64 << 30) as f64 / 1e9).abs() < 0.01);
    }

    #[test]
    fn a_declined_architecture_is_reported_beside_its_fit() {
        let device = DeviceSpec::from_device(64 << 30, 400.0, true);
        let config = serde_json::json!({
            "model_type": "qwen3_vl", "hidden_size": 4096, "num_hidden_layers": 36,
            "intermediate_size": 12288, "num_attention_heads": 32,
            "num_key_value_heads": 8, "vocab_size": 151936,
        });
        let spec = pmetal::hub::model_spec_from_config(&config, None);
        let info = fit(&spec, &device);
        assert!(
            info.unsupported
                .as_deref()
                .is_some_and(|r| r.contains("Qwen3-VL"))
        );
        assert!(
            fit(&ModelSpec::new(0.5, "fp16"), &device)
                .unsupported
                .is_none()
        );
    }
}
