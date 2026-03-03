//! Version introspection and device information.

/// PMetal version string, sourced from Cargo.toml at compile time.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Information about the current device and PMetal installation.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// PMetal version.
    pub pmetal_version: String,
    /// Operating system version string.
    pub os_version: String,
    /// CPU architecture (e.g. `"aarch64"`).
    pub arch: &'static str,
    /// Whether Metal GPU acceleration is available.
    pub metal_available: bool,
    /// Total system memory in GB.
    pub memory_total_gb: f64,
    /// Available (free) memory in GB.
    pub memory_available_gb: f64,
}

impl std::fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "PMetal v{}", self.pmetal_version)?;
        writeln!(f, "  OS:     {}", self.os_version)?;
        writeln!(f, "  Arch:   {}", self.arch)?;
        writeln!(
            f,
            "  Metal:  {}",
            if self.metal_available { "yes" } else { "no" }
        )?;
        writeln!(
            f,
            "  Memory: {:.1} GB total, {:.1} GB available",
            self.memory_total_gb, self.memory_available_gb
        )?;
        Ok(())
    }
}

/// Query device information for the current system.
///
/// Returns details about the PMetal version, OS, architecture, Metal
/// availability, and memory. Useful for diagnostics and compatibility checks.
///
/// # Example
/// ```no_run
/// let info = pmetal::version::device_info();
/// println!("{}", info);
/// ```
pub fn device_info() -> DeviceInfo {
    let (memory_total_gb, memory_available_gb) = get_memory_info();

    DeviceInfo {
        pmetal_version: VERSION.to_string(),
        os_version: get_os_version(),
        arch: std::env::consts::ARCH,
        metal_available: cfg!(target_os = "macos") && std::env::consts::ARCH == "aarch64",
        memory_total_gb,
        memory_available_gb,
    }
}

/// Get total and available memory in GB.
fn get_memory_info() -> (f64, f64) {
    #[cfg(feature = "mlx")]
    {
        let stats = pmetal_mlx::memory::get_memory_stats();
        let total_gb = stats.total_gb();
        let available_gb = total_gb - stats.used_gb();
        (total_gb, available_gb.max(0.0))
    }

    #[cfg(not(feature = "mlx"))]
    {
        let total = get_system_memory_bytes().unwrap_or(0) as f64 / (1024.0 * 1024.0 * 1024.0);
        (total, 0.0) // Without MLX, we can't determine available memory
    }
}

/// Fallback system memory query (used when mlx feature is disabled).
#[cfg(not(feature = "mlx"))]
fn get_system_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        let mem_str = String::from_utf8(output.stdout).ok()?;
        mem_str.trim().parse().ok()
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Get the OS version string.
fn get_os_version() -> String {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sw_vers")
            .arg("-productVersion")
            .output();
        match output {
            Ok(o) => format!("macOS {}", String::from_utf8_lossy(&o.stdout).trim()),
            Err(_) => "macOS (unknown version)".to_string(),
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        format!("{} {}", std::env::consts::OS, std::env::consts::ARCH)
    }
}
