//! Build script for pmetal-cli.
//!
//! Locates `mlx.metallib` produced by the mlx-sys build, compresses it with
//! gzip, and writes it to OUT_DIR so it can be embedded into the binary via
//! `include_bytes!`. At runtime the binary decompresses it on first use.
//!
//! Raw metallib is ~102MB; gzip-compressed is ~31MB — keeps the binary lean
//! while ensuring `cargo install pmetal-cli` is fully self-contained.

use std::env;
use std::path::PathBuf;

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").unwrap() != "macos" {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dest = out_dir.join("mlx.metallib.gz");

    // If we already have the compressed blob (incremental rebuild), skip.
    if dest.is_file() {
        emit(&dest);
        return;
    }

    if let Some(src) = find_metallib() {
        compress_metallib(&src, &dest);
        emit(&dest);
    } else {
        println!(
            "cargo:warning=mlx.metallib not found at build time — \
             embedded metallib will not be available. \
             The binary will fall back to runtime discovery/download."
        );
    }
}

fn emit(path: &std::path::Path) {
    println!(
        "cargo:rustc-env=MLX_METALLIB_EMBED_PATH={}",
        path.display()
    );
}

fn compress_metallib(src: &std::path::Path, dest: &std::path::Path) {
    use std::io::Write;

    let raw = std::fs::read(src).unwrap_or_else(|e| {
        panic!("Failed to read mlx.metallib from {}: {e}", src.display())
    });

    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
    encoder.write_all(&raw).expect("gzip compression failed");
    let compressed = encoder.finish().expect("gzip finish failed");

    std::fs::write(dest, &compressed).unwrap_or_else(|e| {
        panic!(
            "Failed to write compressed metallib to {}: {e}",
            dest.display()
        )
    });

    let raw_mb = raw.len() as f64 / 1_048_576.0;
    let gz_mb = compressed.len() as f64 / 1_048_576.0;
    println!(
        "cargo:warning=Compressed mlx.metallib: {raw_mb:.1}MB → {gz_mb:.1}MB ({:.0}% reduction)",
        (1.0 - gz_mb / raw_mb) * 100.0
    );
}

/// Search for mlx.metallib in order:
/// 1. Sibling mlx-sys build output directories (same target/profile/build/)
/// 2. ~/.cache/pmetal/lib/ (cached by mlx-sys build.rs)
fn find_metallib() -> Option<PathBuf> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").ok()?);

    // OUT_DIR is typically: target/<profile>/build/<crate>-<hash>/out
    // We want:              target/<profile>/build/pmetal-mlx-sys-<hash>/out/build/lib/mlx.metallib
    if let Some(build_dir) = out_dir.parent().and_then(|p| p.parent()) {
        if let Ok(entries) = std::fs::read_dir(build_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("pmetal-mlx-sys-") && entry.path().is_dir() {
                    let candidate = entry.path().join("out/build/lib/mlx.metallib");
                    if candidate.is_file() {
                        println!(
                            "cargo:warning=Found mlx.metallib in build dir: {}",
                            candidate.display()
                        );
                        return Some(candidate);
                    }
                }
            }
        }
    }

    // Fallback: check the cache directory
    if let Ok(home) = env::var("HOME") {
        let cached = PathBuf::from(home).join(".cache/pmetal/lib/mlx.metallib");
        if cached.is_file() {
            println!(
                "cargo:warning=Found mlx.metallib in cache: {}",
                cached.display()
            );
            return Some(cached);
        }
    }

    None
}
