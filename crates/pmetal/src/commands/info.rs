/// Show device information: GPU architecture, ANE cores, bandwidth, NAX, and unified memory.
pub(crate) async fn run_info(json_output: bool) -> anyhow::Result<()> {
    let ctx_result = pmetal_metal::context::MetalContext::global();

    if json_output {
        let obj = match ctx_result {
            Ok(ctx) => {
                let props = ctx.properties();
                serde_json::json!({
                    "device_name": props.name,
                    "gpu_family": format!("{:?}", props.gpu_family),
                    "architecture_gen": props.architecture_gen,
                    "has_nax": props.has_nax,
                    "gpu_cores": props.gpu_core_count,
                    "ane_cores": props.ane_core_count,
                    "memory_total_gb": props.recommended_working_set_size as f64 / (1024.0 * 1024.0 * 1024.0),
                    "memory_bandwidth_gbps": props.memory_bandwidth_gbps,
                    "has_unified_memory": props.has_unified_memory,
                    "metal_available": true,
                })
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
                let mem_gb = props.recommended_working_set_size as f64 / (1024.0 * 1024.0 * 1024.0);
                println!("Device:         {}", props.name);
                println!("GPU Family:     {:?}", props.gpu_family);
                println!("Architecture:   gen {}", props.architecture_gen);
                println!("GPU Cores:      {}", props.gpu_core_count);
                println!("ANE Cores:      {}", props.ane_core_count);
                println!("Unified Memory: {:.0} GB", mem_gb);
                println!("Bandwidth:      {:.0} GB/s", props.memory_bandwidth_gbps);
                println!(
                    "NAX (Neural):   {}",
                    if props.has_nax { "yes" } else { "no" }
                );
                println!("Metal:          available");
            }
            Err(e) => {
                println!("Metal:          unavailable ({})", e);
            }
        }
        println!("PMetal Version: {}", env!("CARGO_PKG_VERSION"));
    }
    Ok(())
}
