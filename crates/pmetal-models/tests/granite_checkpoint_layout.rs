//! Released Granite checkpoints against the model their config builds, by
//! tensor name and shape, without downloading a weight.
//!
//! The synthetic parity fixtures are written under the key names the dumper
//! believes checkpoints use. This holds that belief against the real thing:
//! for every checkpoint directory, the config builds a model through the
//! production path, and a zero tensor for every entry of the checkpoint's
//! safetensors headers is handed to the same all-or-nothing assignment
//! `DynamicModel::load` runs. Any tensor the model has no place for, any
//! parameter the checkpoint does not supply, and any shape mismatch fails.
//!
//! Directories holding only a `config.json` are Granite variants pmetal must
//! refuse (sliding-window, `granite_switch`, speech, vision); detection has to
//! say so by name.
//!
//! Headers are fetched with HTTP range requests (the first `8 + n` bytes of
//! each shard), so the whole sweep is a few megabytes. Populate a directory of
//! `<slug>/{config.json, header.json}` (`header.json`: tensor name to
//! `{"dtype", "shape"}` across every shard) and run:
//!
//! ```bash
//! PMETAL_GRANITE_HEADERS=/Volumes/AmBa/tmp/granite-headers \
//!     cargo test -p pmetal-models --test granite_checkpoint_layout -- --ignored --nocapture
//! ```
//!
//! MLX-quantized conversions are covered too: a packed weight's unpacked
//! width is its scales' width times the group size, which is what the shared
//! reader produces before any Granite code sees the tensors.

use std::collections::HashMap;
use std::path::Path;

use pmetal_bridge::compat::Array;
use pmetal_models::architectures::granite_hybrid::assign_granite_weights;
use pmetal_models::dispatcher::{DynamicModel, ModelArchitecture};

fn json(path: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    pmetal_models::dispatcher::config_value(&text).unwrap_or_else(|e| panic!("{path:?}: {e}"))
}

/// Group size an MLX-quantized module was packed with.
fn group_size(config: &serde_json::Value, module: &str) -> i64 {
    let quant = config.get("quantization").expect("quantized config");
    quant
        .get(module)
        .and_then(|m| m.get("group_size"))
        .or_else(|| quant.get("group_size"))
        .and_then(|v| v.as_i64())
        .expect("group_size")
}

/// Zero tensors shaped as the checkpoint stores them once read: packed MLX
/// weights unpacked, their `.scales` / `.biases` consumed.
fn checkpoint_tensors(
    config: &serde_json::Value,
    header: &serde_json::Value,
) -> HashMap<String, Array> {
    let header = header.as_object().expect("header object");
    let mut out = HashMap::new();
    for (key, entry) in header {
        if key.ends_with(".scales") || key.ends_with(".biases") {
            let base = key.rsplit_once('.').unwrap().0;
            if header.contains_key(&format!("{base}.weight")) {
                continue;
            }
        }
        let mut shape: Vec<i32> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_i64().unwrap() as i32)
            .collect();
        if let Some(module) = key.strip_suffix(".weight") {
            if let Some(scales) = header.get(&format!("{module}.scales")) {
                let groups = scales["shape"]
                    .as_array()
                    .unwrap()
                    .last()
                    .unwrap()
                    .as_i64()
                    .unwrap();
                *shape.last_mut().unwrap() = (groups * group_size(config, module)) as i32;
            }
        }
        out.insert(key.clone(), Array::zeros_f32(&shape));
    }
    out
}

#[test]
#[ignore = "requires PMETAL_GRANITE_HEADERS; see the module docs"]
fn released_granite_checkpoints_match_the_model_their_config_builds() {
    let root = std::env::var("PMETAL_GRANITE_HEADERS")
        .expect("set PMETAL_GRANITE_HEADERS to a directory of <slug>/{config.json,header.json}");
    let mut dirs: Vec<_> = std::fs::read_dir(&root)
        .expect("readable header directory")
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("config.json").exists())
        .collect();
    dirs.sort();
    assert!(!dirs.is_empty(), "no checkpoints under {root}");

    let mut failures = Vec::new();
    for dir in &dirs {
        let slug = dir.file_name().unwrap().to_string_lossy().to_string();
        let config_path = dir.join("config.json");
        let header_path = dir.join("header.json");

        if !header_path.exists() {
            match ModelArchitecture::detect(dir) {
                Ok(arch) => failures.push(format!("{slug}: should be refused, detected {arch:?}")),
                Err(e) => println!("  refused  {slug:48}  {e}"),
            }
            continue;
        }

        let config = json(&config_path);
        let text = std::fs::read_to_string(&config_path).unwrap();
        let model = match DynamicModel::from_config(&text) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("{slug}: config does not build: {e}"));
                continue;
            }
        };
        let DynamicModel::Granite(mut granite) = model else {
            failures.push(format!("{slug}: did not route to Granite"));
            continue;
        };
        let tensors = checkpoint_tensors(&config, &json(&header_path));
        let count = tensors.len();
        match assign_granite_weights(&mut granite, tensors) {
            Ok(report) => println!(
                "  ok       {slug:48}  {} {:>4} tensors placed, {} ignored, mamba={} experts={}",
                granite.config.model_type,
                report.loaded,
                report.ignored.len(),
                granite.config.has_mamba_layers(),
                granite.config.num_experts(),
            ),
            Err(e) => failures.push(format!("{slug} ({count} tensors): {e}")),
        }
    }
    pmetal_bridge::check_last_error().expect("no bridge op failed");
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
