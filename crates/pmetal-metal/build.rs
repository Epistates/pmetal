//! Build script for compiling Metal shaders.
//!
//! This script compiles .metal shader files into .metallib libraries
//! that can be embedded into the binary at compile time.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let shaders_dir = manifest_dir.join("src").join("kernels").join("metal");

    // Check if we're on macOS
    if env::var("CARGO_CFG_TARGET_OS").unwrap() != "macos" {
        println!("cargo:warning=Metal shaders only compile on macOS");
        return;
    }

    // Find all .metal files
    if shaders_dir.exists() {
        compile_metal_shaders(&shaders_dir, &out_dir);
    } else {
        println!(
            "cargo:warning=No metal shaders directory found at {:?}",
            shaders_dir
        );
    }

    // Re-run if shaders change
    println!("cargo:rerun-if-changed=src/kernels/metal");
}

fn compile_metal_shaders(shaders_dir: &Path, out_dir: &Path) {
    let metal_files: Vec<_> = std::fs::read_dir(shaders_dir)
        .expect("Failed to read shaders directory")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension()?.to_str()? == "metal" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if metal_files.is_empty() {
        println!("cargo:warning=No .metal files found in {:?}", shaders_dir);
        return;
    }

    // Compile each .metal file to .air (intermediate representation)
    let mut air_files = Vec::new();
    for metal_file in &metal_files {
        let stem = metal_file.file_stem().unwrap().to_str().unwrap();
        let air_file = out_dir.join(format!("{}.air", stem));

        println!("cargo:rerun-if-changed={}", metal_file.display());

        let status = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                // Optimization flags
                "-O3",
                // Enable fast math for ML workloads
                "-ffast-math",
                // Target Apple Silicon
                "-target",
                "air64-apple-macos14.0",
                // Include path for shared headers
                "-I",
                shaders_dir.to_str().unwrap(),
                // Compile to AIR
                "-c",
                metal_file.to_str().unwrap(),
                "-o",
                air_file.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to run Metal compiler");

        if !status.success() {
            panic!("Failed to compile Metal shader: {}", metal_file.display());
        }

        air_files.push(air_file);
    }

    // Link all .air files into a single .metallib
    let metallib_file = out_dir.join("pmetal_kernels.metallib");

    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);

    for air_file in &air_files {
        cmd.arg(air_file.to_str().unwrap());
    }

    cmd.args(["-o", metallib_file.to_str().unwrap()]);

    let status = cmd.status().expect("Failed to run metallib linker");

    if !status.success() {
        panic!("Failed to link Metal library");
    }

    println!(
        "cargo:rustc-env=PMETAL_METALLIB_PATH={}",
        metallib_file.display()
    );

    println!("Successfully compiled Metal shaders to {:?}", metallib_file);
}
