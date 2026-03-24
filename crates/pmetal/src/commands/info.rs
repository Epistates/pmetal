/// Show device information: GPU architecture, ANE cores, bandwidth, NAX, and unified memory.
pub(crate) async fn run_info(json_output: bool) -> anyhow::Result<()> {
    let ctx_result = pmetal_metal::context::MetalContext::global();

    if json_output {
        let obj = match ctx_result {
            Ok(ctx) => {
                let props = ctx.properties();
                build_info_json(props, pmetal_mlx::memory::get_system_memory())
            }
            Err(e) => serde_json::json!({
                "metal_available": false,
                "error": e.to_string(),
            }),
        };
        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("PMetal Device Information");
        println!("=========================");
        match ctx_result {
            Ok(ctx) => {
                let props = ctx.properties();
                for line in build_info_lines(props, pmetal_mlx::memory::get_system_memory()) {
                    println!("{line}");
                }
            }
            Err(e) => {
                println!("Metal:          unavailable ({})", e);
            }
        }
        println!("PMetal Version: {}", env!("CARGO_PKG_VERSION"));
    }
    Ok(())
}

const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;

fn bytes_to_gb(bytes: u64) -> f64 {
    bytes as f64 / BYTES_PER_GB
}

fn build_info_json(
    props: &pmetal_metal::context::DeviceProperties,
    system_memory: Option<u64>,
) -> serde_json::Value {
    #[cfg(feature = "distributed")]
    let ultrafusion_local_executor = build_ultrafusion_info_json(props, system_memory);
    #[cfg(not(feature = "distributed"))]
    let ultrafusion_local_executor = serde_json::Value::Null;

    serde_json::json!({
        "device_name": props.name,
        "gpu_family": format!("{:?}", props.gpu_family),
        "architecture_gen": props.architecture_gen,
        "has_nax": props.has_nax,
        "gpu_cores": props.gpu_core_count,
        "ane_cores": props.ane_core_count,
        "memory_total_gb": system_memory.map(bytes_to_gb),
        "recommended_working_set_gb": bytes_to_gb(props.recommended_working_set_size),
        "memory_bandwidth_gbps": props.memory_bandwidth_gbps,
        "memory_bandwidth_source": format!("{:?}", props.memory_bandwidth_source),
        "has_unified_memory": props.has_unified_memory,
        "is_ultra_fusion": props.is_ultra_fusion,
        "die_count": props.die_count,
        "ultrafusion_local_executor": ultrafusion_local_executor,
        "metal_available": true,
    })
}

fn build_info_lines(
    props: &pmetal_metal::context::DeviceProperties,
    system_memory: Option<u64>,
) -> Vec<String> {
    let mut lines = vec![
        format!("Device:         {}", props.name),
        format!("GPU Family:     {:?}", props.gpu_family),
        format!("Architecture:   gen {}", props.architecture_gen),
        format!("GPU Cores:      {}", props.gpu_core_count),
        format!("ANE Cores:      {}", props.ane_core_count),
        format!(
            "Unified Memory: {}",
            if props.has_unified_memory {
                "yes"
            } else {
                "no"
            }
        ),
        format!(
            "Recommended WS: {:.0} GB",
            bytes_to_gb(props.recommended_working_set_size)
        ),
        format!(
            "Bandwidth:      {:.0} GB/s ({:?})",
            props.memory_bandwidth_gbps, props.memory_bandwidth_source
        ),
        format!(
            "NAX (Neural):   {}",
            if props.has_nax { "yes" } else { "no" }
        ),
        "Metal:          available".to_string(),
    ];

    lines.insert(
        6,
        match system_memory {
            Some(total_bytes) => format!("System Memory:  {:.0} GB", bytes_to_gb(total_bytes)),
            None => "System Memory:  unknown".to_string(),
        },
    );

    if props.is_ultra_fusion {
        lines.insert(9, format!("UltraFusion:    yes ({} dies)", props.die_count));

        #[cfg(feature = "distributed")]
        if let Some(executor_lines) = build_ultrafusion_info_lines(props, system_memory) {
            for line in executor_lines.into_iter().rev() {
                lines.insert(10, line);
            }
        }
    }

    lines
}

#[cfg(feature = "distributed")]
fn build_ultrafusion_executor_config(
    props: &pmetal_metal::context::DeviceProperties,
    system_memory: Option<u64>,
) -> Option<pmetal_distributed::UltraFusionExecutionConfig> {
    if !props.is_ultra_fusion || props.die_count < 2 {
        return None;
    }

    let total_available_ram = system_memory.unwrap_or(props.recommended_working_set_size);
    Some(
        pmetal_distributed::UltraFusionExecutionConfig::from_uniform_hardware(
            total_available_ram,
            props.gpu_core_count,
            props.ane_core_count,
            props.die_count as usize,
        ),
    )
}

#[cfg(feature = "distributed")]
fn build_ultrafusion_info_json(
    props: &pmetal_metal::context::DeviceProperties,
    system_memory: Option<u64>,
) -> serde_json::Value {
    let Some(config) = build_ultrafusion_executor_config(props, system_memory) else {
        return serde_json::Value::Null;
    };

    serde_json::json!({
        "supported": true,
        "local_stage_count": config.dies.len(),
        "channel_capacity": config.channel_capacity,
        "interconnect_bandwidth_bytes_per_sec": config.interconnect_bandwidth_bytes_per_sec,
        "interconnect_bandwidth_tbps": config.interconnect_bandwidth_bytes_per_sec as f64 / 1_000_000_000_000.0,
        "per_die": config.dies.iter().map(|die| serde_json::json!({
            "die_id": die.die_id,
            "available_ram_gb": bytes_to_gb(die.available_ram_bytes),
            "gpu_cores": die.gpu_cores,
            "ane_cores": die.ane_cores,
        })).collect::<Vec<_>>(),
    })
}

#[cfg(feature = "distributed")]
fn build_ultrafusion_info_lines(
    props: &pmetal_metal::context::DeviceProperties,
    system_memory: Option<u64>,
) -> Option<Vec<String>> {
    let config = build_ultrafusion_executor_config(props, system_memory)?;
    let per_die = config.dies.first()?;
    Some(vec![
        format!(
            "Local Exec:     {}-stage in-memory pipeline available",
            config.dies.len()
        ),
        format!(
            "Interconnect:   {:.1} TB/s heuristic",
            config.interconnect_bandwidth_bytes_per_sec as f64 / 1_000_000_000_000.0
        ),
        format!(
            "Per-Die Slice:  {:.0} GB / {} GPU / {} ANE",
            bytes_to_gb(per_die.available_ram_bytes),
            per_die.gpu_cores,
            per_die.ane_cores
        ),
    ])
}

#[cfg(test)]
mod tests {
    use pmetal_metal::context::{
        AppleGPUFamily, DeviceProperties, DeviceTier, MemoryBandwidthSource,
    };

    use super::*;

    fn test_properties() -> DeviceProperties {
        DeviceProperties {
            name: "Apple M5 Pro".to_string(),
            max_threads_per_threadgroup: 1024,
            max_threadgroup_memory_length: 32 * 1024,
            has_unified_memory: true,
            recommended_working_set_size: 48 * 1024 * 1024 * 1024,
            max_buffer_length: 256 * 1024 * 1024,
            gpu_family: AppleGPUFamily::Apple10,
            device_tier: DeviceTier::Pro,
            has_dynamic_caching: true,
            has_hardware_ray_tracing: true,
            has_mesh_shaders: true,
            has_nax: true,
            architecture_gen: 17,
            memory_bandwidth_gbps: 273.0,
            memory_bandwidth_source: MemoryBandwidthSource::SpecTableFallback,
            gpu_core_count: 20,
            ane_core_count: 16,
            is_ultra_fusion: false,
            die_count: 1,
        }
    }

    #[cfg(feature = "distributed")]
    fn test_ultra_properties() -> DeviceProperties {
        DeviceProperties {
            name: "Apple M5 Ultra".to_string(),
            max_threads_per_threadgroup: 1024,
            max_threadgroup_memory_length: 32 * 1024,
            has_unified_memory: true,
            recommended_working_set_size: 192 * 1024 * 1024 * 1024,
            max_buffer_length: 256 * 1024 * 1024,
            gpu_family: AppleGPUFamily::Apple10,
            device_tier: DeviceTier::Ultra,
            has_dynamic_caching: true,
            has_hardware_ray_tracing: true,
            has_mesh_shaders: true,
            has_nax: true,
            architecture_gen: 17,
            memory_bandwidth_gbps: 800.0,
            memory_bandwidth_source: MemoryBandwidthSource::SpecTableFallback,
            gpu_core_count: 80,
            ane_core_count: 32,
            is_ultra_fusion: true,
            die_count: 2,
        }
    }

    #[test]
    fn test_build_info_json_reports_total_and_working_set_separately() {
        let props = test_properties();
        let json = build_info_json(&props, Some(64 * 1024 * 1024 * 1024));

        assert_eq!(json["memory_total_gb"], serde_json::json!(64.0));
        assert_eq!(json["recommended_working_set_gb"], serde_json::json!(48.0));
        assert_eq!(json["has_unified_memory"], serde_json::json!(true));
        assert_eq!(json["metal_available"], serde_json::json!(true));
        assert_eq!(json["is_ultra_fusion"], serde_json::json!(false));
        assert_eq!(json["die_count"], serde_json::json!(1));
        assert_eq!(
            json["memory_bandwidth_source"],
            serde_json::json!("SpecTableFallback")
        );
    }

    #[test]
    fn test_build_info_lines_include_system_memory_and_working_set() {
        let props = test_properties();
        let lines = build_info_lines(&props, Some(64 * 1024 * 1024 * 1024));

        assert!(lines.iter().any(|line| line == "Unified Memory: yes"));
        assert!(lines.iter().any(|line| line == "System Memory:  64 GB"));
        assert!(lines.iter().any(|line| line == "Recommended WS: 48 GB"));
        assert!(
            lines
                .iter()
                .any(|line| line == "Bandwidth:      273 GB/s (SpecTableFallback)")
        );
    }

    #[cfg(feature = "distributed")]
    #[test]
    fn test_build_info_json_includes_ultrafusion_executor() {
        let props = test_ultra_properties();
        let json = build_info_json(&props, Some(256 * 1024 * 1024 * 1024));

        assert_eq!(json["is_ultra_fusion"], serde_json::json!(true));
        assert_eq!(json["die_count"], serde_json::json!(2));
        assert_eq!(
            json["ultrafusion_local_executor"]["local_stage_count"],
            serde_json::json!(2)
        );
        assert_eq!(
            json["ultrafusion_local_executor"]["per_die"][0]["gpu_cores"],
            serde_json::json!(40)
        );
        assert_eq!(
            json["ultrafusion_local_executor"]["per_die"][0]["ane_cores"],
            serde_json::json!(16)
        );
    }

    #[cfg(feature = "distributed")]
    #[test]
    fn test_build_info_lines_include_ultrafusion_executor() {
        let props = test_ultra_properties();
        let lines = build_info_lines(&props, Some(256 * 1024 * 1024 * 1024));

        assert!(
            lines
                .iter()
                .any(|line| line == "UltraFusion:    yes (2 dies)")
        );
        assert!(
            lines
                .iter()
                .any(|line| line == "Local Exec:     2-stage in-memory pipeline available")
        );
        assert!(
            lines
                .iter()
                .any(|line| line == "Per-Die Slice:  128 GB / 40 GPU / 16 ANE")
        );
    }

    #[test]
    fn test_build_info_lines_handle_unknown_system_memory() {
        let props = test_properties();
        let lines = build_info_lines(&props, None);

        assert!(lines.iter().any(|line| line == "System Memory:  unknown"));
    }
}
