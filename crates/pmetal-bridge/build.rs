use cmake::Config;
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

/// The MLX revision this crate builds against, tag or commit.
///
/// Currently a commit on upstream main: see `cmake/CMakeLists.txt` for why.
const BUNDLED_MLX_GIT_TAG: &str = "99f109b567b969ef06974f1e0450ce0e2eeb4de2";

// ── Deployment target ──────────────────────────────────────────────────────

/// The macOS floor for everything this crate produces.
///
/// This is the only place it is set. It deliberately does *not* live in
/// `.cargo/config.toml` under `[env]`, which would also stamp host proc-macro
/// dylibs and make them unloadable under Xcode 27 (#29). An ambient
/// `MACOSX_DEPLOYMENT_TARGET` may still raise it, never lower it.
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
///
/// A dylib that already carries the name is left alone. The build script
/// watches the library it links (see `emit_link_inputs_rerun`), so rewriting
/// it on every run would make every build rerun this script.
#[cfg(target_os = "macos")]
fn set_dylib_install_name(dylib: &std::path::Path) {
    if !dylib.exists() {
        return;
    }
    let current = Command::new("otool")
        .arg("-D")
        .arg(dylib)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).into_owned());
    if current.is_some_and(|text| text.lines().nth(1) == Some(&*dylib.display().to_string())) {
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

/// Publish a key dependents read as `DEP_PMETAL_BRIDGE_<KEY>`.
///
/// The separator has to be `::`. `cargo:metadata=k=v` is the *old* one-key
/// form, so cargo reads the key as `metadata` and the value as the whole
/// `k=v` string: every call lands on `DEP_PMETAL_BRIDGE_METADATA` and the last
/// one wins. That is why `pmetal/build.rs` never found `DEP_PMETAL_BRIDGE_*`
/// and always fell through to scanning sibling build directories.
fn emit_bridge_metadata(key: &str, value: impl AsRef<str>) {
    println!("cargo::metadata={key}={}", value.as_ref());
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
// prefix for reuse on the next invocation. CI mounts it via actions/cache
// keyed on BUNDLED_MLX_GIT_TAG; locally .cargo/config.toml points it at
// target/mlx-prefix, so every pmetal-bridge variant shares one build.
//
// Each fingerprint gets its own slot, `<prefix>/<mlx_prefix_slot()>/`:
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

/// The prefix subdirectory for this build's fingerprint: its 64-bit FNV-1a
/// hash in hex. Stable across runs and Rust versions, which `DefaultHasher`
/// doesn't promise.
fn mlx_prefix_slot() -> String {
    let hash = mlx_fingerprint()
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    format!("{hash:016x}")
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
    // The slot must hold the kind this build links and not the other: a
    // static slot that also held a dylib (populated from a stale tree) linked
    // dynamically, so it's rebuilt instead of reused.
    let (wanted, other) = mlx_lib_names();
    if !lib_dir.join(wanted).exists() || lib_dir.join(other).exists() {
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

    // Only the kind this fingerprint promises. The build tree can hold a
    // stale one of the other kind, and copying it made a static slot link
    // dynamically.
    let (wanted, other) = mlx_lib_names();
    let _ = std::fs::remove_file(lib_dir.join(other));
    std::fs::copy(dst.join("build/lib").join(wanted), lib_dir.join(wanted))?;

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
        // A shipped binary has to run on a machine that has never seen this
        // build tree, so release links MLX statically by default
        // (`mlx_static_requested`): the artifact carries MLX inside it, with
        // no dylib to locate, no install name, no rpath, nothing to bundle.
        // Dead-stripping at link time keeps it near the size of the dynamic
        // build.
        //
        // This repository's own release builds opt out (PMETAL_MLX_STATIC=0
        // in .cargo/config.toml): `libmlx.a` is ~200 MB, and a release
        // `cargo test` links ~96 test binaries. Its release jobs set
        // PMETAL_MLX_STATIC=1 back for the artifacts they ship.
    }
    // Always explicit, both ways: CMake caches it in the build tree, so
    // leaving it unset for a static build meant a directory that last built
    // shared (a release `cargo test`, then PMETAL_MLX_STATIC=1) built shared
    // again, and the "static" binary linked this tree's libmlx.dylib.
    config.define(
        "BUILD_SHARED_LIBS",
        if mlx_builds_shared() { "ON" } else { "OFF" },
    );

    config.build()
}

/// Whether to link MLX statically in release. See the call site.
///
/// Static unless `PMETAL_MLX_STATIC` says otherwise. A build with no say in
/// the matter is a `cargo install pmetal` or a crate depending on this one,
/// and a shared MLX there is a dylib whose install name points into a build
/// directory that `cargo install` deletes. This repository's
/// `.cargo/config.toml` sets `PMETAL_MLX_STATIC=0` for its own release
/// builds, where ~96 test binaries each linking a static MLX would not be
/// viable. Debug builds static unconditionally, so this only steers release.
fn mlx_static_requested() -> bool {
    env::var("PMETAL_MLX_STATIC").map_or(true, |v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Whether this build produces `libmlx.dylib` (else `libmlx.a`): only a
/// release build with PMETAL_MLX_STATIC=0.
fn mlx_builds_shared() -> bool {
    !cfg!(debug_assertions) && !mlx_static_requested()
}

/// The MLX library file this build links, and the one it must not find.
///
/// The link is `dylib=mlx`, which resolves to whichever of the two the linker
/// meets first on the search path, so a leftover of the other kind decides
/// the linkage. Static linking relies on no `libmlx.dylib` being visible.
fn mlx_lib_names() -> (&'static str, &'static str) {
    if mlx_builds_shared() {
        (mlx_dylib_name(), "libmlx.a")
    } else {
        ("libmlx.a", mlx_dylib_name())
    }
}

// ── What the link depends on ──────────────────────────────────────────────

/// Rerun this script when the MLX it linked against changes or disappears.
///
/// That MLX can live in a prefix slot shared by many build directories, and
/// Cargo only reruns a build script for the files it names. Deleting or
/// repopulating the slot while this script's output stayed cached left later
/// builds linking against files that were gone or replaced, and the test
/// binaries they produced computed on placeholder arrays. A watched file that
/// changes or goes missing makes Cargo rerun the script, which rebuilds or
/// reuses the slot as it finds it.
fn emit_link_inputs_rerun(lib_dir: &Path, slot: Option<&Path>) {
    let (wanted, _) = mlx_lib_names();
    let mut watched = vec![lib_dir.join(wanted), lib_dir.join("mlx.metallib")];
    if let Some(slot) = slot {
        watched.push(slot.join(".mlx-version"));
    }
    for path in watched.into_iter().filter(|p| p.exists()) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

/// SHA-256 of `bytes`, lower-case hex.
fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Publish the `mlx.metallib` built with the MLX this crate links: its path
/// and SHA-256, to this crate (`PMETAL_BRIDGE_MLX_METALLIB{,_SHA256}`, read by
/// `pmetal_bridge::metallib`) and to dependents
/// (`DEP_PMETAL_BRIDGE_MLX_METALLIB{,_SHA256}`).
///
/// Nothing is written outside OUT_DIR. Cargo asks build scripts not to, and a
/// sandboxed build (Homebrew's) can't. This script used to copy the file to an
/// unversioned `~/.cache/pmetal/lib/mlx.metallib`, which every build of every
/// MLX version overwrote for every installed binary. Binaries built here find
/// the file at this path while it exists; an application that outlives its
/// build tree carries its own copy (the CLI embeds one, the GUI bundles one)
/// and caches it by MLX revision and content hash at run time.
fn publish_metallib(lib_dir: &Path) -> Option<PathBuf> {
    if !cfg!(feature = "metal") {
        return None;
    }
    let metallib = lib_dir.join("mlx.metallib");
    let bytes = std::fs::read(&metallib).ok()?;
    let sha256 = sha256_hex(&bytes);
    emit_bridge_metadata("mlx_metallib", metallib.display().to_string());
    emit_bridge_metadata("mlx_metallib_sha256", &sha256);
    println!(
        "cargo:rustc-env=PMETAL_BRIDGE_MLX_METALLIB={}",
        metallib.display()
    );
    println!("cargo:rustc-env=PMETAL_BRIDGE_MLX_METALLIB_SHA256={sha256}");
    Some(metallib)
}

/// Write `pmetal_metallib_config.h` for the bridge's C++, naming the metallib
/// [`publish_metallib`] found (empty when there is none). Returns the
/// directory to put on the include path.
fn write_metallib_config(metallib: Option<&Path>) -> PathBuf {
    let dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("generated");
    std::fs::create_dir_all(&dir).expect("Failed to create OUT_DIR/generated");
    let path = metallib
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    let mut literal = String::with_capacity(path.len() + 2);
    for byte in path.bytes() {
        match byte {
            b'"' | b'\\' => {
                literal.push('\\');
                literal.push(byte as char);
            }
            0x20..=0x7e => literal.push(byte as char),
            // Octal escapes keep any other byte (UTF-8 included) exact.
            _ => literal.push_str(&format!("\\{byte:03o}")),
        }
    }
    let header = format!(
        "// Generated by pmetal-bridge's build.rs.\n\
         #pragma once\n\n\
         // The mlx.metallib built with the MLX this bridge links, or \"\".\n\
         #define PMETAL_BUILT_METALLIB \"{literal}\"\n"
    );
    let file = dir.join("pmetal_metallib_config.h");
    // Unchanged contents keep their mtime, so the C++ doesn't rebuild for nothing.
    if std::fs::read_to_string(&file).ok().as_deref() != Some(header.as_str()) {
        std::fs::write(&file, header).expect("Failed to write pmetal_metallib_config.h");
    }
    dir
}

// ── Main build ────────────────────────────────────────────────────────────

fn build_and_link() {
    // Enforce macOS deployment target >= 14.0 before cmake or cc see CFLAGS.
    // Setting it here keeps it inside this build script's own process, so no
    // host proc macro is ever linked with it.
    #[cfg(target_os = "macos")]
    {
        let target = resolve_deployment_target();
        // SAFETY: build scripts are single-threaded; no other threads exist yet.
        unsafe { env::set_var("MACOSX_DEPLOYMENT_TARGET", &target) };
        // Dependents link against objects built for this floor, so publish it.
        emit_bridge_metadata("min_macos", &target);
    }

    // One subdirectory per fingerprint. With a single slot, a debug and a
    // release build overwrote each other's MLX, so switching profile rebuilt
    // it, and binaries whose libmlx.dylib install name points into the prefix
    // started loading the other profile's library.
    let mlx_prefix = env::var("PMETAL_MLX_PREFIX")
        .ok()
        .map(|root| PathBuf::from(root).join(mlx_prefix_slot()));

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
                // This tree may have built the other kind before (the env var
                // changed, not the OUT_DIR). Both paths below go on the link
                // search path, so drop the leftover or it decides the linkage.
                let (_, other) = mlx_lib_names();
                for dir in ["build/lib", "build/_deps/mlx-build"] {
                    let _ = std::fs::remove_file(dst.join(dir).join(other));
                }
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

    emit_link_inputs_rerun(&mlx_lib_dir_built, mlx_prefix.as_deref());

    // ── mlx.metallib ──
    let built_metallib = publish_metallib(&mlx_lib_dir_built);

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
        "bridge_cce.cpp",
        "bridge_gdn.cpp",
        "bridge_distributed.cpp",
    ];

    let mut build = cc::Build::new();
    build.cpp(true).std("c++20");

    for src in &bridge_sources {
        build.file(manifest_dir.join("cpp").join(src));
    }

    build
        // MLX C++ headers (mlx/mlx.h)
        .include(&mlx_include)
        // pmetal_metallib_config.h, written by `write_metallib_config`
        .include(write_metallib_config(built_metallib.as_deref()))
        // bridge.h / bridge_internal.h live alongside the source files
        .include(manifest_dir.join("cpp"))
        .flag("-w") // suppress warnings from MLX internals
        .flag("-flto=thin") // LTO matches Python's MLX pip wheel build
        .opt_level(3);

    #[cfg(target_os = "macos")]
    build.flag(format!("-mmacosx-version-min={deploy_target}"));

    // The error counter `strict-errors` reads. Without the feature the bridge
    // compiles exactly as it did before it existed.
    if env::var_os("CARGO_FEATURE_STRICT_ERRORS").is_some() {
        build.define("PMETAL_BRIDGE_STRICT_ERRORS", None);
    }

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
