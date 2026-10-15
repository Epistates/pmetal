//! Build script for pmetal.
//!
//! Embeds the `mlx.metallib` built with the MLX this binary links, gzip
//! compressed, so an installed `pmetal` carries its own kernels (~31 MB of
//! ~102 MB). On first use the binary writes it to the per-version metallib
//! cache (`pmetal_runtime::metallib::Cache`). Its SHA-256 is compiled in as
//! `PMETAL_EMBEDDED_METALLIB_SHA256`, which names the cache entry and checks
//! the bytes written there.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_PMETAL_RUNTIME_MLX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=DEP_PMETAL_RUNTIME_MLX_METALLIB");
    println!("cargo:rerun-if-env-changed=DEP_PMETAL_RUNTIME_MIN_MACOS");

    if env::var("CARGO_CFG_TARGET_OS").unwrap() != "macos" {
        return;
    }

    emit_deployment_target();
    emit_runtime_rpath();

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dest = out_dir.join("mlx.metallib.gz");

    // Only the bridge's own metallib, the one built with the MLX this links.
    // This used to fall back to scanning sibling build directories and to an
    // unversioned copy in ~/.cache, either of which could hold another MLX
    // revision's kernels.
    let metallib = env::var("DEP_PMETAL_RUNTIME_MLX_METALLIB")
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.is_file());
    match metallib {
        Some(src) => {
            // Re-embed when the file changes, as when the MLX prefix slot it
            // lives in is rebuilt.
            println!("cargo:rerun-if-changed={}", src.display());
            let sha256 = compress_metallib(&src, &dest);
            println!("cargo:rustc-env=PMETAL_EMBEDDED_METALLIB_SHA256={sha256}");
        }
        None => {
            // An empty blob keeps `include_bytes!` compiling; at run time the
            // binary then looks for a metallib instead of extracting one.
            std::fs::write(&dest, b"").unwrap();
            println!(
                "cargo:warning=pmetal-runtime published no mlx.metallib (built without the \
                 `metal` feature?), so none is embedded. The binary will look for one at run time."
            );
        }
    }
}

/// Stamp the binary with the macOS floor the bridge was built against, so it
/// advertises what it actually needs instead of rustc's default of 11.0.
///
/// This has to be a link arg rather than `MACOSX_DEPLOYMENT_TARGET` in
/// `.cargo/config.toml`: `[env]` reaches host proc macros too, and stamping
/// those breaks the build outright under Xcode 27 (#29). A link arg from here
/// applies to this package's artifacts and nothing else.
fn emit_deployment_target() {
    let Ok(min_macos) = env::var("DEP_PMETAL_RUNTIME_MIN_MACOS") else {
        return;
    };
    println!("cargo:rustc-link-arg=-mmacosx-version-min={min_macos}");
}

fn emit_runtime_rpath() {
    match find_mlx_linkage() {
        MlxLinkage::Dynamic(mlx_lib_dir) => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", mlx_lib_dir.display());
            println!(
                "cargo:warning=Embedding libmlx.dylib runtime search path into pmetal: {}",
                mlx_lib_dir.display()
            );
        }
        MlxLinkage::Static => {}
        MlxLinkage::Missing => {
            println!("cargo:warning=Could not determine MLX library runtime search path");
        }
    }
}

/// Gzip `src` into `dest` and return the SHA-256 of the uncompressed bytes.
fn compress_metallib(src: &Path, dest: &Path) -> String {
    use sha2::Digest as _;
    use std::io::Write;

    let raw = std::fs::read(src)
        .unwrap_or_else(|e| panic!("Failed to read mlx.metallib from {}: {e}", src.display()));
    let sha256: String = sha2::Sha256::digest(&raw)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
    encoder.write_all(&raw).expect("gzip compression failed");
    let compressed = encoder.finish().expect("gzip finish failed");

    std::fs::write(dest, &compressed).unwrap_or_else(|e| {
        panic!(
            "Failed to write compressed metallib to {}: {e}",
            dest.display()
        )
    });
    sha256
}

enum MlxLinkage {
    Dynamic(PathBuf),
    Static,
    Missing,
}

fn linkage_in_dir(candidate: PathBuf) -> Option<MlxLinkage> {
    if candidate.join("libmlx.dylib").is_file() {
        Some(MlxLinkage::Dynamic(candidate))
    } else if candidate.join("libmlx.a").is_file() {
        Some(MlxLinkage::Static)
    } else {
        None
    }
}

fn find_mlx_linkage() -> MlxLinkage {
    if let Ok(path) = env::var("DEP_PMETAL_RUNTIME_MLX_LIB_DIR") {
        if let Some(linkage) = linkage_in_dir(PathBuf::from(path)) {
            return linkage;
        }
    }

    if let Ok(path) = env::var("PMETAL_MLX_LIB_DIR") {
        if let Some(linkage) = linkage_in_dir(PathBuf::from(path)) {
            return linkage;
        }
    }

    let Some(out_dir) = env::var("OUT_DIR").ok().map(PathBuf::from) else {
        return MlxLinkage::Missing;
    };
    let Some(build_dir) = out_dir.parent().and_then(|p| p.parent()) else {
        return MlxLinkage::Missing;
    };
    let Ok(entries) = std::fs::read_dir(build_dir) else {
        return MlxLinkage::Missing;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("pmetal-runtime-") {
            continue;
        }
        let candidate = path.join("out/build/lib");
        if let Some(linkage) = linkage_in_dir(candidate) {
            return linkage;
        }
    }

    MlxLinkage::Missing
}
