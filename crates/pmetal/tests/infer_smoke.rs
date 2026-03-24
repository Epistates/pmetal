#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::process::Command;

fn cached_qwen3_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let path = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca");
    path.is_dir().then_some(path)
}

#[test]
#[ignore = "requires local Metal hardware, cached Qwen3 weights, and unsandboxed cargo test"]
fn qwen3_cached_infer_smoke_exits_successfully() {
    let Some(model_path) = cached_qwen3_path() else {
        eprintln!("Skipping infer smoke: cached Qwen3 snapshot not found");
        return;
    };

    let binary = env!("CARGO_BIN_EXE_pmetal");
    let output = Command::new(binary)
        .args([
            "infer",
            "--model",
            model_path.to_str().expect("utf-8 model path"),
            "--prompt",
            "test",
            "--max-tokens",
            "1",
            "--temperature",
            "0",
        ])
        .env("RUST_LOG", "warn")
        .output()
        .expect("spawn pmetal infer");

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "pmetal infer smoke failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status, stdout, stderr
        );
    }
}
