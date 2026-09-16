use cmake::Config;
use std::{env, path::PathBuf, process::Command};

/// The MLX revision this crate builds against, tag or commit.
///
/// Currently a commit: see `cmake/CMakeLists.txt` for which kernels and
/// quantized-matmul fixes v0.32.2 is missing.
const BUNDLED_MLX_GIT_TAG: &str = "7241f12e631440387a2156ec1018d1c9fe8e56b9";

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

/// Rewrite the `LC_ID_DYLIB` install name of a Mach-O dylib to an absolute
/// path so downstream binaries record that absolute path in their
/// `LC_LOAD_DYLIB` and dyld resolves without any `@rpath` / `DYLD_*` help.
///
/// Cargo's `cargo:rustc-link-arg=-Wl,-rpath,…` applies only to this crate's
/// own artifacts — it does not propagate to downstream bins. Rewriting the
/// dylib's own install name is the one setting that every dependent binary
/// inherits.
#[cfg(target_os = "macos")]
fn set_dylib_install_name(dylib: &std::path::Path) {
    if !dylib.exists() {
        return;
    }
    let status = Command::new("install_name_tool")
        .args([
            "-id",
            &dylib.display().to_string(),
            &dylib.display().to_string(),
        ])
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => println!(
            "cargo:warning=install_name_tool -id {} failed: {s}",
            dylib.display()
        ),
        Err(e) => println!(
            "cargo:warning=install_name_tool -id {} errored: {e}",
            dylib.display()
        ),
    }
}

fn emit_bridge_metadata(key: &str, value: impl AsRef<str>) {
    println!("cargo:metadata={key}={}", value.as_ref());
}

// ── Staging + CMake build ─────────────────────────────────────────────────
//
// Since MLX v0.32 the metallib override is upstream API
// (mlx::core::metal::set_metallib_path), called from the bridge at startup —
// the vendored MLX checkout is built exactly as shipped, no patches.

/// Stage the minimal CMakeLists.txt into OUT_DIR. Returns the staged directory path.
fn prepare_cmake_source() -> PathBuf {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let staged = out_dir.join("bridge-cmake-staged");

    // Always re-stage so edits are picked up on rebuild
    if staged.exists() {
        std::fs::remove_dir_all(&staged).expect("Failed to clean staged cmake dir");
    }
    std::fs::create_dir_all(&staged).expect("Failed to create staged cmake dir");

    // Stage CMakeLists.txt
    let cmake_src =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("cmake/CMakeLists.txt");
    let cmake_content =
        std::fs::read_to_string(&cmake_src).expect("Failed to read cmake/CMakeLists.txt");

    // Guard: the cmake tag must match BUNDLED_MLX_GIT_TAG so version bumps
    // cannot silently build a different MLX than the one this crate targets.
    let expected = format!("GIT_TAG {BUNDLED_MLX_GIT_TAG})");
    assert!(
        cmake_content.contains(&expected),
        "cmake/CMakeLists.txt GIT_TAG does not match BUNDLED_MLX_GIT_TAG {BUNDLED_MLX_GIT_TAG}"
    );
    std::fs::write(staged.join("CMakeLists.txt"), cmake_content)
        .expect("Failed to write staged CMakeLists.txt");

    staged
}

// ── PMETAL_MLX_PREFIX: per-tag persistent MLX build cache ─────────────────
//
// When PMETAL_MLX_PREFIX is set to a directory, a fingerprint-matched
// pre-built MLX in that directory is reused and cmake is skipped entirely.
// Otherwise MLX is built as usual, and the result is copied into the
// prefix for reuse on the next invocation (intended for CI where the
// prefix is mounted via actions/cache keyed on BUNDLED_MLX_GIT_TAG).
//
// Layout inside the prefix:
//   .mlx-version         — fingerprint string (tag + feature flags)
//   lib/libmlx.dylib     — MLX dynamic library (or libmlx.so elsewhere)
//   lib/mlx.metallib     — compiled Metal kernels (when metal feature on)
//   include/mlx/**/*.h   — MLX public + internal headers
//
// The fingerprint bakes in feature flags and debug/release so a
// (metal+release) build never aliases a (no-metal+debug) build.

fn mlx_fingerprint() -> String {
    let metal = cfg!(feature = "metal") as u8;
    let accelerate = cfg!(feature = "accelerate") as u8;
    let debug = cfg!(debug_assertions) as u8;
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown-target".to_string());
    let deployment = if cfg!(target_os = "macos") {
        resolve_deployment_target()
    } else {
        "n/a".to_string()
    };
    let os = if cfg!(target_os = "macos") {
        "macos"
    } else {
        "other"
    };
    // Static vs shared changes which artifact lands in the prefix (libmlx.a vs
    // libmlx.dylib), so it has to be part of the key or a cached prefix from
    // the other mode gets reused.
    let static_mlx = mlx_static_requested() as u8;
    format!(
        "{tag};target={target};macos_deployment={deployment};metal={metal};accelerate={accelerate};debug={debug};static={static_mlx};os={os}",
        tag = BUNDLED_MLX_GIT_TAG
    )
}

fn mlx_dylib_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "libmlx.dylib"
    } else {
        "libmlx.so"
    }
}

fn try_reuse_cached_mlx(prefix: &std::path::Path) -> Option<(PathBuf, PathBuf)> {
    let marker = std::fs::read_to_string(prefix.join(".mlx-version")).ok()?;
    if marker.trim() != mlx_fingerprint() {
        return None;
    }
    let lib_dir = prefix.join("lib");
    let include_dir = prefix.join("include");
    // Accept either the shared build (libmlx.dylib, release) or the static
    // build (libmlx.a, debug). Profile is baked into the fingerprint above,
    // so whichever shows up here matches what this build expects.
    let has_lib = lib_dir.join(mlx_dylib_name()).exists() || lib_dir.join("libmlx.a").exists();
    if !has_lib {
        return None;
    }
    #[cfg(feature = "metal")]
    if !lib_dir.join("mlx.metallib").exists() {
        return None;
    }
    if !include_dir.join("mlx").join("mlx.h").exists() {
        return None;
    }
    Some((lib_dir, include_dir))
}

fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else if ty.is_file() {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

fn populate_mlx_prefix(dst: &std::path::Path, prefix: &std::path::Path) -> std::io::Result<()> {
    let lib_dir = prefix.join("lib");
    let include_dir = prefix.join("include");
    std::fs::create_dir_all(&lib_dir)?;
    std::fs::create_dir_all(&include_dir)?;

    // Copy whichever MLX library artifact exists for this profile.
    for name in [mlx_dylib_name(), "libmlx.a"] {
        let src = dst.join("build/lib").join(name);
        if src.exists() {
            std::fs::copy(&src, lib_dir.join(name))?;
        }
    }

    #[cfg(feature = "metal")]
    {
        let src_metallib = dst.join("build/lib/mlx.metallib");
        if src_metallib.exists() {
            std::fs::copy(&src_metallib, lib_dir.join("mlx.metallib"))?;
        }
    }

    let src_headers = dst.join("build/_deps/mlx-src/mlx");
    let dst_headers = include_dir.join("mlx");
    if dst_headers.exists() {
        std::fs::remove_dir_all(&dst_headers)?;
    }
    copy_dir_recursive(&src_headers, &dst_headers)?;

    std::fs::write(prefix.join(".mlx-version"), mlx_fingerprint())?;
    Ok(())
}

fn run_cmake_build() -> PathBuf {
    let cmake_src = prepare_cmake_source();

    let mut config = Config::new(&cmake_src);
    config.very_verbose(false);
    config.define("CMAKE_INSTALL_PREFIX", ".");
    config.define("CMAKE_C_COMPILER", "/usr/bin/cc");
    config.define("CMAKE_CXX_COMPILER", "/usr/bin/c++");

    // Allow offline/sandbox builds (e.g., MacPorts): forward any
    // FETCHCONTENT_SOURCE_DIR_* and FETCHCONTENT_FULLY_DISCONNECTED env vars
    // to cmake so pre-fetched source trees are used instead of git clones.
    for (key, val) in env::vars() {
        if key.starts_with("FETCHCONTENT_SOURCE_DIR_") || key == "FETCHCONTENT_FULLY_DISCONNECTED" {
            config.define(&key, &val);
        }
    }

    #[cfg(target_os = "macos")]
    {
        let target = resolve_deployment_target();
        config.define("CMAKE_OSX_DEPLOYMENT_TARGET", &target);
        // macOS ar doesn't support -D (deterministic mode). Override CMake's
        // default archive commands to avoid "illegal option" warnings.
        config.define("CMAKE_C_ARCHIVE_CREATE", "<CMAKE_AR> cr <TARGET> <OBJECTS>");
        config.define("CMAKE_C_ARCHIVE_APPEND", "<CMAKE_AR> r <TARGET> <OBJECTS>");
        config.define(
            "CMAKE_CXX_ARCHIVE_CREATE",
            "<CMAKE_AR> cr <TARGET> <OBJECTS>",
        );
        config.define(
            "CMAKE_CXX_ARCHIVE_APPEND",
            "<CMAKE_AR> r <TARGET> <OBJECTS>",
        );
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
        config.define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "ON");
        // Release normally builds MLX as a dylib: `libmlx.a` is ~200 MB, and a
        // release `cargo test` links ~96 test binaries, so a static MLX there
        // is not viable.
        //
        // A *distributed* binary is the opposite case — one binary, and it has
        // to run on a machine that has never seen this build tree. Set
        // PMETAL_MLX_STATIC=1 for those (see .github/workflows/release.yml) and
        // the artifact carries MLX inside it: no dylib to locate, no install
        // name, no rpath, nothing to bundle. Dead-stripping at link time keeps
        // the result near the size of the dynamic build.
        if !mlx_static_requested() {
            config.define("BUILD_SHARED_LIBS", "ON");
        }
    }

    config.build()
}

/// Whether to link MLX statically even in release. See the call site.
///
/// Debug already builds static unconditionally, so this only steers release.
fn mlx_static_requested() -> bool {
    env::var("PMETAL_MLX_STATIC").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
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

    let mlx_prefix = env::var("PMETAL_MLX_PREFIX").ok().map(PathBuf::from);

    // Resolve MLX artifacts: prefer a matching cache, otherwise build and
    // (if a prefix is configured) populate the cache for next time.
    let (mlx_lib_dir_built, mlx_include): (PathBuf, PathBuf) =
        match mlx_prefix.as_deref().and_then(try_reuse_cached_mlx) {
            Some((lib, inc)) => {
                println!(
                    "cargo:warning=Reusing cached MLX build from {} (tag {BUNDLED_MLX_GIT_TAG})",
                    mlx_prefix.as_ref().unwrap().display()
                );
                // Re-stamp the install name against the current cache path
                // — safe if the cache was moved since the last build.
                #[cfg(target_os = "macos")]
                set_dylib_install_name(&lib.join(mlx_dylib_name()));
                (lib, inc)
            }
            None => {
                let dst = run_cmake_build();
                // Legacy search path — cmake puts the fetched target's libs in _deps too.
                println!(
                    "cargo:rustc-link-search=native={}/build/_deps/mlx-build",
                    dst.display()
                );

                // Rewrite LC_ID_DYLIB on both copies of libmlx.dylib that live
                // in this build tree. CMake emits the dylib into
                // `build/lib/` (canonical) and — via FetchContent — also into
                // `build/_deps/mlx-build/`. The linker picks whichever it
                // finds first in the search path; if we miss one, downstream
                // bins get an @rpath install name and fail to load without
                // DYLD_LIBRARY_PATH. Do this BEFORE cache population so the
                // archived copy has the final install name baked in.
                #[cfg(target_os = "macos")]
                {
                    let built = dst.join("build/lib").join(mlx_dylib_name());
                    let deps = dst.join("build/_deps/mlx-build").join(mlx_dylib_name());
                    set_dylib_install_name(&built);
                    set_dylib_install_name(&deps);
                }

                if let Some(prefix) = &mlx_prefix {
                    match populate_mlx_prefix(&dst, prefix) {
                        Ok(()) => println!(
                            "cargo:warning=Populated MLX cache at {} (tag {BUNDLED_MLX_GIT_TAG})",
                            prefix.display()
                        ),
                        Err(e) => println!(
                            "cargo:warning=Failed to populate PMETAL_MLX_PREFIX cache: {e}"
                        ),
                    }
                    // The cache copy has the build-tree install name baked in.
                    // Rewrite it to point at the cache location so downstream
                    // bins built against the cache also load without DYLD help.
                    #[cfg(target_os = "macos")]
                    set_dylib_install_name(&prefix.join("lib").join(mlx_dylib_name()));
                }
                // When FETCHCONTENT_SOURCE_DIR_MLX is set (e.g., MacPorts sandbox),
                // CMake uses that path directly and never creates _deps/mlx-src.
                // Fall back to the env var so the bridge C++ is compiled against
                // the same headers that libmlx.dylib was built from.
                let mlx_src = dst.join("build/_deps/mlx-src");
                let mlx_inc = if mlx_src.exists() {
                    mlx_src
                } else if let Ok(dir) = env::var("FETCHCONTENT_SOURCE_DIR_MLX") {
                    PathBuf::from(dir)
                } else {
                    dst.join("build/_deps/mlx-src")
                };
                (dst.join("build/lib"), mlx_inc)
            }
        };

    // ── Link libmlx ──
    println!(
        "cargo:rustc-link-search=native={}",
        mlx_lib_dir_built.display()
    );
    // Link MLX — use PMETAL_MLX_LIB_DIR to override with an external libmlx.dylib.
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
        #[cfg(target_os = "macos")]
        emit_mlx_rpath(&mlx_lib_dir_built.display().to_string());
        println!("cargo:rustc-env=PMETAL_BRIDGE_MLX_KIND=bundled-upstream");
        mlx_lib_dir_built.clone()
    };
    println!("cargo:rustc-env=PMETAL_BRIDGE_MLX_GIT_TAG={BUNDLED_MLX_GIT_TAG}");
    emit_bridge_metadata("mlx_lib_dir", mlx_lib_dir.display().to_string());

    // ── Compile bridge C++ sources via cc::Build ──
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    #[cfg(target_os = "macos")]
    let deploy_target = resolve_deployment_target();

    let bridge_sources = [
        "bridge.cpp",
        "bridge_inference.cpp",
        "bridge_turboquant_encode.cpp",
        "bridge_turboquant_score.cpp",
        "bridge_turboquant_pack.cpp",
        "bridge_turboquant_weighted.cpp",
        "bridge_turboquant_attn_d256.cpp",
        "bridge_turboquant_attn_d128.cpp",
        "bridge_turboquant_hamming.cpp",
        "bridge_compiled.cpp",
        "bridge_native.cpp",
        "bridge_training.cpp",
        "bridge_gdn.cpp",
    ];

    let mut build = cc::Build::new();
    build.cpp(true).std("c++20");

    for src in &bridge_sources {
        build.file(manifest_dir.join("cpp").join(src));
    }

    build
        // MLX C++ headers (mlx/mlx.h)
        .include(&mlx_include)
        // bridge.h / bridge_internal.h live alongside the source files
        .include(manifest_dir.join("cpp"))
        .flag("-w") // suppress warnings from MLX internals
        .flag("-flto=thin") // LTO matches Python's MLX pip wheel build
        .opt_level(3);

    #[cfg(target_os = "macos")]
    build.flag(format!("-mmacosx-version-min={deploy_target}"));

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
        // Either the freshly built tree or the reused cache has it at <lib>/mlx.metallib
        let metallib = mlx_lib_dir_built.join("mlx.metallib");
        if metallib.exists() {
            emit_bridge_metadata("mlx_metallib", metallib.display().to_string());
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
    for src in &bridge_sources {
        println!("cargo:rerun-if-changed=cpp/{src}");
    }
    println!("cargo:rerun-if-changed=cpp/bridge.h");
    println!("cargo:rerun-if-changed=cpp/bridge_internal.h");
    println!("cargo:rerun-if-changed=cmake/CMakeLists.txt");
    println!("cargo:rerun-if-env-changed=PMETAL_MLX_PREFIX");
    println!("cargo:rerun-if-env-changed=PMETAL_MLX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=PMETAL_MLX_STATIC");
}

fn main() {
    // Ensure Cargo re-runs build.rs when C++ sources change.
    // Without this, edits to bridge.cpp/bridge.h produce stale binaries.
    println!("cargo:rerun-if-changed=cpp/"); // any change in cpp/ dir
    build_and_link();
}
