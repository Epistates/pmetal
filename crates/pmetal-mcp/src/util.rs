use turbomcp::prelude::*;

/// Query total system memory via sysctl (macOS).
pub fn get_system_memory() -> Option<u64> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let mem_str = String::from_utf8(output.stdout).ok()?;
    mem_str.trim().parse().ok()
}

pub use pmetal_supervisor::{redact_secrets, resolve_pmetal_binary};

/// Build device info JSON by calling MetalContext directly.
pub fn build_device_info_json() -> McpResult<serde_json::Value> {
    let ctx = pmetal_metal::context::MetalContext::global()
        .map_err(|e| McpError::internal(format!("Metal not available: {e}")))?;

    let props = ctx.properties();
    let system_memory = get_system_memory();

    const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;

    Ok(serde_json::json!({
        "device_name": props.name,
        "gpu_family": format!("{:?}", props.gpu_family),
        "architecture_gen": props.architecture_gen,
        "has_nax": props.has_nax,
        "gpu_cores": props.gpu_core_count,
        "ane_cores": props.ane_core_count,
        "memory_total_gb": system_memory.map(|b| b as f64 / BYTES_PER_GB),
        "recommended_working_set_gb": props.recommended_working_set_size as f64 / BYTES_PER_GB,
        "memory_bandwidth_gbps": props.memory_bandwidth_gbps,
        "memory_bandwidth_source": format!("{:?}", props.memory_bandwidth_source),
        "has_unified_memory": props.has_unified_memory,
        "metal_available": true,
    }))
}

/// Build a DeviceSpec for fit estimation.
pub fn build_device_spec() -> McpResult<pmetal_hub::DeviceSpec> {
    let ctx = pmetal_metal::context::MetalContext::global()
        .map_err(|e| McpError::internal(format!("Metal not available: {e}")))?;
    let props = ctx.properties();

    Ok(pmetal_hub::DeviceSpec::from_device(
        props.recommended_working_set_size,
        props.memory_bandwidth_gbps,
        props.has_unified_memory,
    ))
}
