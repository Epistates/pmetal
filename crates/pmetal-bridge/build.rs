use cmake::Config;
use std::{env, path::PathBuf, process::Command};

const BUNDLED_MLX_GIT_TAG: &str = "v0.31.1";

// ── Deployment target ──────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn resolve_deployment_target() -> String {
    const MLX_MIN_MACOS: (u32, u32) = (14, 0);
    if let Ok(val) = env::var("MACOSX_DEPLOYMENT_TARGET") {
        let parts: Vec<u32> = val.split('.').filter_map(|s| s.parse().ok()).collect();
        let major = parts.first().copied().unwrap_or(0);
        let minor = parts.get(1).copied().unwrap_or(0);
        if (major, minor) >= MLX_MIN_MACOS {
            return val;
        }
    }
    format!("{}.{}", MLX_MIN_MACOS.0, MLX_MIN_MACOS.1)
}

// ── Clang runtime (___isPlatformVersionAtLeast) ───────────────────────────

fn find_clang_rt_path() -> Option<String> {
    let output = Command::new("xcode-select")
        .args(["--print-path"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let developer_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let toolchain_base = format!(
        "{}/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang",
        developer_dir
    );
    let clang_dir = std::fs::read_dir(&toolchain_base).ok()?;
    for entry in clang_dir.flatten() {
        let darwin_path = entry.path().join("lib/darwin");
        if darwin_path.join("libclang_rt.osx.a").exists() {
            return Some(darwin_path.to_string_lossy().to_string());
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn emit_mlx_rpath(path: &str) {
    println!("cargo:rustc-link-arg=-Wl,-rpath,{path}");
}

fn emit_bridge_metadata(key: &str, value: impl AsRef<str>) {
    println!("cargo:metadata={key}={}", value.as_ref());
}

// ── Patches (embedded as string constants) ────────────────────────────────

/// The metallib search-path patch: adds PMETAL_METALLIB_PATH env-var override
/// and ~/.cache/pmetal/lib/ user-cache lookups to MLX's Metal device loader.
const METALLIB_SEARCH_PATH_PATCH: &str = include_str!("patches/metallib-search-path.patch");

/// The slice-output-shapes patch: adds Slice::output_shapes() and
/// CustomKernel::set_output_shapes() so MLX compile() works with our kernels.
const SLICE_OUTPUT_SHAPES_PATCH: &str = include_str!("patches/slice-output-shapes.patch");

// ── Staging + CMake build ─────────────────────────────────────────────────

/// Stage the minimal CMakeLists.txt into OUT_DIR, inject the patch commands,
/// and write the patch files. Returns the staged directory path.
fn prepare_cmake_source() -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let staged = out_dir.join("bridge-cmake-staged");

    // Always re-stage so patch edits are picked up on rebuild
    if staged.exists() {
        std::fs::remove_dir_all(&staged).expect("Failed to clean staged cmake dir");
    }
    std::fs::create_dir_all(&staged).expect("Failed to create staged cmake dir");

    // Stage CMakeLists.txt
    let cmake_src =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("cmake/CMakeLists.txt");
    let cmake_content =
        std::fs::read_to_string(&cmake_src).expect("Failed to read cmake/CMakeLists.txt");

    // Inject PATCH_COMMAND into the FetchContent_Declare for MLX (same pattern as mlx-sys)
    let patched = cmake_content.replace(
        "GIT_TAG v0.31.1)",
        concat!(
            "GIT_TAG v0.31.1\n",
            "  PATCH_COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/metallib-search-path.patch || true",
            " && git apply ${CMAKE_CURRENT_SOURCE_DIR}/patches/slice-output-shapes.patch || true)"
        ),
    );
    std::fs::write(staged.join("CMakeLists.txt"), patched)
        .expect("Failed to write patched CMakeLists.txt");

    // Write patch files into staged/patches/
    let patches_dir = staged.join("patches");
    std::fs::create_dir_all(&patches_dir).expect("Failed to create patches dir");
    std::fs::write(
        patches_dir.join("metallib-search-path.patch"),
        METALLIB_SEARCH_PATH_PATCH,
    )
    .expect("Failed to write metallib patch");
    std::fs::write(
        patches_dir.join("slice-output-shapes.patch"),
        SLICE_OUTPUT_SHAPES_PATCH,
    )
    .expect("Failed to write slice-output-shapes patch");

    staged
}

// ── Main build ────────────────────────────────────────────────────────────

fn build_and_link() {
    // Enforce macOS deployment target >= 14.0 before cmake or cc see CFLAGS
    #[cfg(target_os = "macos")]
    {
        let target = resolve_deployment_target();
        // SAFETY: build scripts are single-threaded; no other threads exist yet.
        unsafe { env::set_var("MACOSX_DEPLOYMENT_TARGET", &target) };
    }

    let cmake_src = prepare_cmake_source();

    let mut config = Config::new(&cmake_src);
    config.very_verbose(false);
    config.define("CMAKE_INSTALL_PREFIX", ".");
    config.define("CMAKE_C_COMPILER", "/usr/bin/cc");
    config.define("CMAKE_CXX_COMPILER", "/usr/bin/c++");

    #[cfg(target_os = "macos")]
    {
        let target = resolve_deployment_target();
        config.define("CMAKE_OSX_DEPLOYMENT_TARGET", &target);
        // macOS ar doesn't support -D (deterministic mode). Override CMake's
        // default archive commands to avoid "illegal option" warnings.
        config.define("CMAKE_C_ARCHIVE_CREATE", "<CMAKE_AR> cr <TARGET> <OBJECTS>");
        config.define("CMAKE_CXX_ARCHIVE_CREATE", "<CMAKE_AR> cr <TARGET> <OBJECTS>");
    }

    // Metal + Accelerate mirror the features this crate exposes
    #[cfg(feature = "metal")]
    config.define("MLX_BUILD_METAL", "ON");
    // CRITICAL: Enable JIT compilation for Metal kernels.
    // Without this, MLX uses pre-compiled shaders (air64_v26 / Metal 3.2).
    // With JIT, kernels are compiled at RUNTIME using the system's Metal compiler
    // (Metal 4.0 / air64_v28 on macOS 16+), enabling NAX kernels and better
    // code generation. This is what Python's pip wheel uses, and it's the
    // primary cause of a ~3x performance difference.
    config.define("MLX_METAL_JIT", "ON");
    #[cfg(not(feature = "metal"))]
    config.define("MLX_BUILD_METAL", "OFF");

    #[cfg(feature = "accelerate")]
    config.define("MLX_BUILD_ACCELERATE", "ON");
    #[cfg(not(feature = "accelerate"))]
    config.define("MLX_BUILD_ACCELERATE", "OFF");

    #[cfg(debug_assertions)]
    config.define("CMAKE_BUILD_TYPE", "Debug");
    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
        config.define("BUILD_SHARED_LIBS", "ON");
        config.define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "ON");
    }

    let dst = config.build();

    // ── Link libmlx.a ──
    // cmake installs libmlx.a into <dst>/build/lib (cmake crate convention)
    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    // The fetched MLX target also produces libmlx.a; cmake may put it in _deps
    // or build/lib depending on build system; add both search paths.
    println!(
        "cargo:rustc-link-search=native={}/build/_deps/mlx-build",
        dst.display()
    );
    // Link MLX — use PMETAL_MLX_LIB_DIR to override with Python's libmlx.dylib
    let mlx_lib_dir = if let Ok(mlx_dir) = env::var("PMETAL_MLX_LIB_DIR") {
        println!("cargo:rustc-link-search=native={mlx_dir}");
        println!("cargo:rustc-link-lib=dylib=mlx");
        #[cfg(target_os = "macos")]
        emit_mlx_rpath(&mlx_dir);
        println!("cargo:warning=Using external libmlx.dylib from {mlx_dir}");
        println!("cargo:rustc-env=PMETAL_BRIDGE_MLX_KIND=external");
        PathBuf::from(mlx_dir)
    } else {
        println!("cargo:rustc-link-lib=dylib=mlx");
        println!("cargo:rustc-link-search={}/build/lib", dst.display());
        #[cfg(target_os = "macos")]
        emit_mlx_rpath(&format!("{}/build/lib", dst.display()));
        println!("cargo:rustc-env=PMETAL_BRIDGE_MLX_KIND=bundled-upstream");
        dst.join("build/lib")
    };
    println!("cargo:rustc-env=PMETAL_BRIDGE_MLX_GIT_TAG={BUNDLED_MLX_GIT_TAG}");
    emit_bridge_metadata("mlx_lib_dir", mlx_lib_dir.display().to_string());

    // ── Compile bridge.cpp via cc::Build ──
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let bridge_src = manifest_dir.join("cpp/bridge.cpp");
    let mlx_include = dst.join("build/_deps/mlx-src");

    #[cfg(target_os = "macos")]
    let deploy_target = resolve_deployment_target();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++20")
        .file(&bridge_src)
        // MLX C++ headers (mlx/mlx.h)
        .include(&mlx_include)
        // bridge.h lives alongside bridge.cpp
        .include(manifest_dir.join("cpp"))
        .flag("-w") // suppress warnings from MLX internals
        .flag("-flto=thin") // LTO matches Python's MLX pip wheel build
        .opt_level(3);

    #[cfg(target_os = "macos")]
    build.flag(&format!("-mmacosx-version-min={deploy_target}"));

    build.compile("pmetal_bridge");
    println!("cargo:rustc-link-lib=static=pmetal_bridge");

    // ── System frameworks and runtime libs ──
    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    println!("cargo:rustc-link-lib=framework=Metal");

    #[cfg(feature = "accelerate")]
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // clang runtime for ___isPlatformVersionAtLeast (macOS 26 / Xcode 27)
    if let Some(clang_rt_path) = find_clang_rt_path() {
        println!("cargo:rustc-link-search={}", clang_rt_path);
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
        println!("cargo:warning=clang_rt found: {}", clang_rt_path);
    } else {
        let fallback = "/Applications/Xcode.app/Contents/Developer/Toolchains/\
            XcodeDefault.xctoolchain/usr/lib/clang/17/lib/darwin";
        if std::path::Path::new(fallback)
            .join("libclang_rt.osx.a")
            .exists()
        {
            println!("cargo:rustc-link-search={}", fallback);
            println!("cargo:rustc-link-lib=static=clang_rt.osx");
            println!("cargo:warning=clang_rt fallback: {}", fallback);
        } else {
            println!(
                "cargo:warning=clang_rt NOT FOUND — ___isPlatformVersionAtLeast may be missing"
            );
        }
    }

    // ── Cache mlx.metallib ──
    #[cfg(feature = "metal")]
    {
        // CMake installs it to build/lib/mlx.metallib
        let metallib = dst.join("build/lib/mlx.metallib");
        if metallib.exists() {
            emit_bridge_metadata("mlx_metallib", metallib.display().to_string());
        }
        if metallib.exists() {
            if let Ok(home) = env::var("HOME") {
                let cache_dir = PathBuf::from(home).join(".cache/pmetal/lib");
                let dest = cache_dir.join("mlx.metallib");
                let should_copy = if dest.exists() {
                    dest.metadata()
                        .and_then(|d| {
                            metallib.metadata().map(|s| {
                                s.modified()
                                    .ok()
                                    .zip(d.modified().ok())
                                    .is_some_and(|(src_t, dst_t)| src_t > dst_t)
                            })
                        })
                        .unwrap_or(false)
                } else {
                    true
                };
                if should_copy {
                    let _ = std::fs::create_dir_all(&cache_dir);
                    match std::fs::copy(&metallib, &dest) {
                        Ok(_) => {
                            println!("cargo:warning=Cached mlx.metallib to {}", dest.display())
                        }
                        Err(e) => println!("cargo:warning=Failed to cache mlx.metallib: {}", e),
                    }
                }
            }
        }
    }

    // ── Rerun triggers ──
    println!("cargo:rerun-if-changed=cpp/bridge.cpp");
    println!("cargo:rerun-if-changed=cpp/bridge.h");
    println!("cargo:rerun-if-changed=cmake/CMakeLists.txt");
    println!("cargo:rerun-if-changed=patches/metallib-search-path.patch");
    println!("cargo:rerun-if-changed=patches/slice-output-shapes.patch");
}

fn main() {
    // Ensure Cargo re-runs build.rs when C++ sources change.
    // Without this, edits to bridge.cpp/bridge.h produce stale binaries.
    println!("cargo:rerun-if-changed=cpp/bridge.cpp");
    println!("cargo:rerun-if-changed=cpp/bridge.h");
    build_and_link();
}
