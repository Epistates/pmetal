//! Released Nemotron-H and Nemotron 3 checkpoints against the model their
//! config builds, by tensor name and shape, without downloading a weight.
//!
//! The synthetic parity fixtures are written under the key names their
//! generator believes checkpoints use. This holds that belief against the real
//! thing: for every checkpoint directory, the config builds a model through the
//! production path, and a zero tensor for every entry of the checkpoint's
//! safetensors headers (as the shared reader leaves them: MLX-packed weights
//! unpacked, FP8 and NVFP4 sidecars consumed) is handed to the same
//! all-or-nothing assignment `DynamicModel::load` runs. Any tensor the model
//! has no place for, any parameter the checkpoint does not supply and any
//! shape mismatch fails.
//!
//! Headers are fetched with HTTP range requests (the first `8 + n` bytes of
//! each shard), so the whole sweep is a few megabytes. Populate a directory of
//! `<slug>/{config.json, header.json}` (`header.json`: tensor name to
//! `{"dtype", "shape"}` across every shard) and run:
//!
//! ```bash
//! PMETAL_NEMOTRON_HEADERS=<dir of checkpoint headers> \
//!     cargo test -p pmetal-models --test nemotron_h_checkpoint_layout -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::path::Path;

use pmetal_models::architectures::nemotron_h::assign_nemotron_h_weights;
use pmetal_models::dispatcher::DynamicModel;
use pmetal_runtime::compat::Array;

fn json(path: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    pmetal_models::dispatcher::config_value(&text).unwrap_or_else(|e| panic!("{path:?}: {e}"))
}

/// Group size an MLX-quantized module was packed with.
fn group_size(config: &serde_json::Value, module: &str) -> i64 {
    let quant = config
        .get("quantization")
        .or_else(|| config.get("quantization_config"))
        .expect("quantized config");
    quant
        .get(module)
        .and_then(|m| m.get("group_size"))
        .or_else(|| quant.get("group_size"))
        .and_then(|v| v.as_i64())
        .expect("group_size")
}

/// Zero tensors shaped as the shared reader leaves the checkpoint: packed MLX
/// weights unpacked, their `.scales` / `.biases` consumed, and the per-tensor
/// sidecars of ModelOpt FP8 / NVFP4 (`weight_scale`, `weight_scale_2`,
/// `input_scale`) consumed.
fn checkpoint_tensors(
    config: &serde_json::Value,
    header: &serde_json::Value,
) -> HashMap<String, Array> {
    let header = header.as_object().expect("header object");
    let mut out = HashMap::new();
    for (key, entry) in header {
        if key.ends_with(".weight_scale")
            || key.ends_with(".weight_scale_2")
            || key.ends_with(".input_scale")
        {
            continue;
        }
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
#[ignore = "requires PMETAL_NEMOTRON_HEADERS; see the module docs"]
fn released_nemotron_checkpoints_match_the_model_their_config_builds() {
    let root = std::env::var("PMETAL_NEMOTRON_HEADERS")
        .expect("set PMETAL_NEMOTRON_HEADERS to a directory of <slug>/{config.json,header.json}");
    let mut dirs: Vec<_> = std::fs::read_dir(&root)
        .expect("readable header directory")
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("config.json").exists() && p.join("header.json").exists())
        .collect();
    dirs.sort();
    assert!(!dirs.is_empty(), "no checkpoints under {root}");

    let mut failures = Vec::new();
    for dir in &dirs {
        let slug = dir.file_name().unwrap().to_string_lossy().to_string();
        let config = json(&dir.join("config.json"));
        let text = std::fs::read_to_string(dir.join("config.json")).unwrap();
        let model = match DynamicModel::from_config(&text) {
            Ok(m) => m,
            Err(e) => {
                failures.push(format!("{slug}: config does not build: {e}"));
                continue;
            }
        };
        let DynamicModel::NemotronH(mut nemotron) = model else {
            failures.push(format!("{slug}: did not route to NemotronH"));
            continue;
        };
        let layers = nemotron.config().layer_types().len();
        let tensors = checkpoint_tensors(&config, &json(&dir.join("header.json")));
        let count = tensors.len();
        match assign_nemotron_h_weights(&mut nemotron, tensors) {
            Ok(()) => println!("  ok       {slug:58}  {count:>6} tensors, {layers} layers"),
            Err(e) => failures.push(format!("{slug} ({count} tensors): {e}")),
        }
    }
    pmetal_runtime::check_unobserved_error().expect("no bridge op failed");
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
