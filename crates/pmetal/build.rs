fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "macos" {
        panic!(
            "pmetal requires macOS (Apple Silicon). \
             Cross-platform components are available separately via `pmetal-distributed`."
        );
    }
}
