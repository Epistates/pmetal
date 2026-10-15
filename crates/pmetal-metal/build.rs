//! Build script for compiling Metal shaders.
//!
//! This script compiles .metal shader files into .metallib libraries
//! that can be embedded into the binary at compile time.
//!
//! Supports dual compilation:
//! - Metal 3 (macOS 14.0+): All shaders in src/kernels/metal/
//! - Metal 4 (macOS 26.0+): MPP/NAX shaders in src/kernels/metal4/
//!   Only compiled when Metal compiler version >= 400 and SDK >= 26.0
//!
//! Each shader is compiled on its own thread, and a shader whose source, shared
//! headers and compiler flags have not changed since the last build in this
//! target directory is not compiled again: editing one kernel recompiles that
//! kernel, not the library.

use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::SystemTime;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shaders_dir = manifest_dir.join("src").join("kernels").join("metal");
    let metal4_dir = manifest_dir.join("src").join("kernels").join("metal4");

    // Declare has_metal4 cfg for check-cfg lint
    println!("cargo::rustc-check-cfg=cfg(has_metal4)");

    // Check if we're on macOS
    if env::var("CARGO_CFG_TARGET_OS").unwrap() != "macos" {
        println!("cargo:warning=Metal shaders only compile on macOS");
        return;
    }

    // Re-run if shaders change, or the toolchain they are built with moves
    println!("cargo:rerun-if-changed=src/kernels/metal");
    println!("cargo:rerun-if-changed=src/kernels/metal4");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=SDKROOT");

    // Verify Metal toolchain is available before attempting compilation
    check_metal_toolchain();

    // Detect Metal compiler version and SDK version
    let metal_version = detect_metal_version();
    let sdk_version = detect_sdk_version();
    let compiler_id = compiler_identity();

    // Compile Metal 3 shaders (always)
    if shaders_dir.exists() {
        compile_metal_shaders(
            &shaders_dir,
            &out_dir,
            "air64-apple-macos14.0",
            "pmetal_kernels",
            &compiler_id,
        );
    } else {
        println!(
            "cargo:warning=No metal shaders directory found at {:?}",
            shaders_dir
        );
    }

    // Compile Metal 4 / MPP shaders (conditional on toolchain support)
    let has_metal4 = metal_version >= 400 && sdk_version >= 26.0;
    if has_metal4 && metal4_dir.exists() && metal_sources(&metal4_dir).next().is_some() {
        compile_metal_shaders(
            &metal4_dir,
            &out_dir,
            "air64-apple-macos26.0",
            "pmetal_kernels_metal4",
            &compiler_id,
        );
        println!("cargo:rustc-cfg=has_metal4");
        println!(
            "cargo:warning=Metal 4 / MPP shaders compiled (Metal version {metal_version}, SDK {sdk_version})"
        );
    }

    // Link against Accelerate.framework for vDSP vector operations
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // Conditionally link IOSurface.framework when ANE feature is active
    if env::var("CARGO_FEATURE_ANE").is_ok() {
        println!("cargo:rustc-link-lib=framework=IOSurface");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        // AppleNeuralEngine.framework is loaded at runtime via dlopen, not linked here
    }
}

/// Run `xcrun -sdk macosx metal <args>`, feeding `stdin` if given.
fn run_metal(args: &[&str], stdin: Option<&str>) -> Option<std::process::Output> {
    let mut command = Command::new("xcrun");
    command.args(["-sdk", "macosx", "metal"]).args(args);
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    command.stdin(if stdin.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    let mut child = command.spawn().ok()?;
    if let (Some(text), Some(mut pipe)) = (stdin, child.stdin.take()) {
        pipe.write_all(text.as_bytes()).ok()?;
    }
    child.wait_with_output().ok()
}

/// Detect the maximum Metal language version the toolchain supports.
///
/// `__METAL_VERSION__` is determined by the chosen `-std=` flag. Without one,
/// the compiler falls back to its default (e.g. `metal3.1` on many SDKs even
/// when `metal4.0` is available), so a bare `metal -E` returns 310 on SDK 26+.
/// We probe by requesting `-std=metal4.0` explicitly: if the compiler accepts,
/// it preprocesses `__METAL_VERSION__` to `400`; otherwise it errors and we
/// fall back to probing `metal3.1`. Returns 0 when nothing works (should be
/// rare — means the toolchain is broken, not just old).
fn detect_metal_version() -> u32 {
    for std_flag in ["metal4.0", "metal3.1"] {
        let std_arg = format!("-std={std_flag}");
        let output = run_metal(
            &[&std_arg, "-E", "-x", "metal", "-P", "-"],
            Some("__METAL_VERSION__\n"),
        );
        if let Some(o) = output
            && o.status.success()
        {
            let text = String::from_utf8_lossy(&o.stdout);
            if let Some(n) = text
                .lines()
                .rev()
                .find_map(|line| line.trim().parse::<u32>().ok())
                && n > 0
            {
                return n;
            }
        }
    }
    0
}

/// The compiler's own version banner. A shader compiled by another compiler is
/// not the same shader, so a change in this invalidates every object.
fn compiler_identity() -> String {
    run_metal(&["--version"], None)
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

/// Detect macOS SDK version (e.g. 14.5, 26.2).
fn detect_sdk_version() -> f64 {
    let output = Command::new("xcrun")
        .args(["-sdk", "macosx", "--show-sdk-version"])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let version_str = String::from_utf8_lossy(&o.stdout).trim().to_string();
            version_str.parse::<f64>().unwrap_or(0.0)
        }
        _ => 0.0,
    }
}

/// Verify that the Metal compiler toolchain is available via xcrun.
///
/// Provides actionable installation instructions on failure instead of
/// a cryptic "Failed to run Metal compiler" panic.
fn check_metal_toolchain() {
    // Check xcrun itself
    let xcrun_ok = Command::new("xcrun").args(["--find", "metal"]).output();

    match xcrun_ok {
        Ok(output) if output.status.success() => {} // Metal compiler found
        Ok(_) => {
            // xcrun exists but can't find metal
            panic!(
                "\n\
                 ╔══════════════════════════════════════════════════════════════════╗\n\
                 ║  Metal compiler not found                                       ║\n\
                 ╚══════════════════════════════════════════════════════════════════╝\n\n\
                 PMetal requires the Metal shader compiler to build.\n\n\
                 To install:\n\
                 1. Install Xcode from the App Store (or Command Line Tools):\n\
                    xcode-select --install\n\n\
                 2. Accept the Xcode license:\n\
                    sudo xcodebuild -license accept\n\n\
                 3. Download the Metal toolchain:\n\
                    xcodebuild -downloadComponent MetalToolchain\n\n\
                 4. Restart your terminal (or reboot) after installation.\n\n\
                 If you have Xcode installed but metal is not found, try:\n\
                    sudo xcode-select -s /Applications/Xcode.app/Contents/Developer\n"
            );
        }
        Err(e) => {
            // xcrun itself not found
            panic!(
                "\n\
                 ╔══════════════════════════════════════════════════════════════════╗\n\
                 ║  xcrun not found — Xcode Command Line Tools required            ║\n\
                 ╚══════════════════════════════════════════════════════════════════╝\n\n\
                 PMetal requires Xcode Command Line Tools to compile Metal shaders.\n\n\
                 To install:\n\
                    xcode-select --install\n\n\
                 After installation, restart your terminal and try again.\n\n\
                 Error: {e}\n"
            );
        }
    }
}

/// The `.metal` files directly in `dir`.
fn metal_sources(dir: &Path) -> impl Iterator<Item = PathBuf> {
    fs::read_dir(dir).into_iter().flatten().filter_map(|entry| {
        let path = entry.ok()?.path();
        (path.extension()?.to_str()? == "metal").then_some(path)
    })
}

fn modified(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// The newest modification time among every file in `dirs` (shaders and the
/// headers they share).
fn newest_input(dirs: &[&Path]) -> Option<SystemTime> {
    dirs.iter()
        .flat_map(|dir| fs::read_dir(dir).into_iter().flatten())
        .filter_map(|entry| modified(&entry.ok()?.path()))
        .max()
}

/// Compile `shaders_dir`'s `.metal` files into `<lib_name>.metallib` in
/// `out_dir`, and return that path.
fn compile_metal_shaders(
    shaders_dir: &Path,
    out_dir: &Path,
    target: &str,
    lib_name: &str,
    compiler_id: &str,
) -> PathBuf {
    let cache_root = out_dir.join(".cache");
    fs::create_dir_all(&cache_root).expect("Failed to create shader compiler cache");

    // Determine metal language standard from target
    let std_flag = if target.contains("macos26") {
        "-std=metal4.0"
    } else {
        "-std=metal3.1"
    };

    let mut metal_files: Vec<PathBuf> = metal_sources(shaders_dir).collect();
    metal_files.sort();

    if metal_files.is_empty() {
        println!("cargo:warning=No .metal files found in {:?}", shaders_dir);
    }

    // Shaders include headers from their own directory and from `metal/`, so a
    // change to any file in either invalidates every object (the headers are
    // shared and the include graph is not tracked).
    let shared_dir = shaders_dir.parent().unwrap().join("metal");
    let newest_source = newest_input(&[shaders_dir, &shared_dir]);

    // The flags, compiler and shader set a cached library was built with. A
    // mismatch rebuilds everything (a deleted shader would otherwise live on
    // in the old library).
    let names: Vec<String> = metal_files
        .iter()
        .map(|file| file.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    let flags = format!(
        "{std_flag} -O3 -ffast-math -target {target}\n{}\n{compiler_id}",
        names.join(",")
    );
    let stamp_path = out_dir.join(format!("{lib_name}.flags"));
    let flags_unchanged = fs::read_to_string(&stamp_path).is_ok_and(|old| old == flags);

    let air_files: Vec<PathBuf> = metal_files
        .iter()
        .map(|metal_file| {
            let stem = metal_file.file_stem().unwrap().to_str().unwrap();
            out_dir.join(format!("{lib_name}_{stem}.air"))
        })
        .collect();

    let stale: Vec<usize> = (0..metal_files.len())
        .filter(|&i| {
            !flags_unchanged
                || match (modified(&air_files[i]), newest_source) {
                    (Some(air), Some(source)) => air < source,
                    _ => true,
                }
        })
        .collect();

    // Compile the stale shaders concurrently, as many at a time as cargo's
    // job count (the compiler is the whole cost of a clean build).
    let jobs = env::var("NUM_JOBS")
        .ok()
        .and_then(|n| n.parse::<usize>().ok())
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1)
        .clamp(1, 8);
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..jobs.min(stale.len()) {
            scope.spawn(|| {
                loop {
                    let slot = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&i) = stale.get(slot) else {
                        break;
                    };
                    compile_to_air(
                        &metal_files[i],
                        &air_files[i],
                        shaders_dir,
                        &shared_dir,
                        std_flag,
                        target,
                        out_dir,
                        &cache_root,
                    );
                }
            });
        }
    });

    let metallib_file = out_dir.join(format!("{lib_name}.metallib"));
    let library_stale = !stale.is_empty() || !metallib_file.exists();

    // Link all .air files into a single .metallib
    if library_stale {
        let mut cmd = Command::new("xcrun");
        cmd.env("HOME", out_dir);
        cmd.env("XDG_CACHE_HOME", &cache_root);
        cmd.args(["-sdk", "macosx", "metallib"]);

        for air_file in &air_files {
            cmd.arg(air_file);
        }

        cmd.arg("-o").arg(&metallib_file);

        let output = cmd.output().expect("Failed to run metallib linker");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "Failed to link Metal library {}\n\n--- linker output ---\n{}",
                lib_name, stderr
            );
        }
        // Recorded only now that every object built: an interrupted build
        // leaves no stamp, so the next one starts over.
        fs::write(&stamp_path, &flags).expect("Failed to record shader flags");
    }

    // Set env var for embedding — uppercase lib name
    let env_name = format!(
        "{}_PATH",
        lib_name.to_uppercase().replace("pmetal_", "PMETAL_")
    );
    println!("cargo:rustc-env={}={}", env_name, metallib_file.display());

    println!(
        "Compiled {} of {} Metal shaders to {:?} (target: {})",
        stale.len(),
        metal_files.len(),
        metallib_file,
        target
    );
    metallib_file
}

/// Compile one `.metal` file to `.air`, panicking with the compiler's output
/// if it fails.
#[allow(clippy::too_many_arguments)]
fn compile_to_air(
    metal_file: &Path,
    air_file: &Path,
    shaders_dir: &Path,
    shared_dir: &Path,
    std_flag: &str,
    target: &str,
    out_dir: &Path,
    cache_root: &Path,
) {
    let output = Command::new("xcrun")
        .env("HOME", out_dir)
        .env("XDG_CACHE_HOME", cache_root)
        .args([
            "-sdk",
            "macosx",
            "metal",
            // Metal language standard
            std_flag,
            // Optimization flags
            "-O3",
            // Enable fast math for ML workloads
            "-ffast-math",
            // Target Apple Silicon
            "-target",
            target,
        ])
        // Include path for shared headers (both metal/ and metal4/)
        .arg("-I")
        .arg(shaders_dir)
        .arg("-I")
        .arg(shared_dir)
        // Compile to AIR
        .arg("-c")
        .arg(metal_file)
        .arg("-o")
        .arg(air_file)
        .output()
        .expect("Failed to run Metal compiler (xcrun verified but execution failed)");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "Failed to compile Metal shader: {}\n\n--- compiler output ---\n{}",
            metal_file.display(),
            stderr
        );
    }
}
