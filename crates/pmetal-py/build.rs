use std::env;

fn main() {
    pyo3_build_config::add_extension_module_link_args();
    println!("cargo:rerun-if-env-changed=DEP_PMETAL_RUNTIME_MIN_MACOS");
    emit_deployment_target();
}

/// Stamp the extension with the macOS floor the bridge was built against, so it
/// advertises what it actually needs (MLX does not run before macOS 14)
/// instead of rustc's default of 11.0.
///
/// A link arg and not `MACOSX_DEPLOYMENT_TARGET` in `.cargo/config.toml`:
/// `[env]` reaches host proc macros too, and stamping those breaks the build
/// under Xcode 27 (#29). A link arg from here applies to this package's
/// artifacts and nothing else, as in the `pmetal` crate's own build script.
fn emit_deployment_target() {
    let Ok(min_macos) = env::var("DEP_PMETAL_RUNTIME_MIN_MACOS") else {
        return;
    };
    println!("cargo:rustc-link-arg=-mmacosx-version-min={min_macos}");
}
