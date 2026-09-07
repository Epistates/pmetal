//! Holds every dispatcher architecture against its **real released**
//! `config.json`.
//!
//! The synthetic parity fixtures prove the math, but they are generated from
//! configs that pmetal and the dumper agree on by construction. Nothing in the
//! suite ever asked the cheaper question: *given the config file that actually
//! ships with the checkpoint, does pmetal read it the way `transformers` does?*
//!
//! That gap is not hypothetical. Two live examples:
//!
//! * A stock Phi-2 `config.json` says `"hidden_act": "gelu_new"`, a spelling
//!   `PhiActivation` did not accept — so the config failed to deserialize and
//!   the model could not load at all. No test noticed, because no test ever
//!   fed pmetal a real Phi config.
//! * Nearly every config struct here is `#[serde(default)]`. A field the
//!   checkpoint spells differently than pmetal does not error — it silently
//!   takes pmetal's default. `layer_norm_eps` vs `rms_norm_eps` is a one-word
//!   difference that changes which normalizer runs.
//!
//! So this test does two things per checkpoint:
//!
//! 1. **Detection + parse.** `ModelArchitecture::detect` must resolve to the
//!    architecture the manifest expects, and
//!    [`ModelArchitecture::parse_config_json`] must deserialize the config.
//! 2. **Field-by-field agreement.** Every field pmetal parsed is looked up in
//!    the raw JSON; where the checkpoint states a value, pmetal's parsed value
//!    must equal it. This is what catches a silent default: the checkpoint says
//!    `4096` and pmetal quietly holds `131072`.
//!
//! Configs are *not* committed. They carry the upstream model licenses (Gemma,
//! Llama, and friends are not permissive), and vendoring them into the tracked
//! tree would violate the repo's permissive-only rule. So this is `#[ignore]`d
//! and gated on `PMETAL_REAL_CONFIGS`, following the same pattern as the real
//! Mllama checkpoint test. Populate it with:
//!
//! ```bash
//! PYTHONPATH=.strategy/parity .venv-parity/bin/python \
//!     .strategy/parity/download_configs.py \
//!     arch_models.txt /Volumes/AmBa/huggingface/parity-configs
//! PMETAL_REAL_CONFIGS=/Volumes/AmBa/huggingface/parity-configs \
//!     cargo test -p pmetal-models --test real_config_parity -- --ignored --nocapture
//! ```
//!
//! Model artifacts live on `/Volumes/AmBa`, not the boot volume — the config
//! snapshots pull whole tokenizer files alongside the configs and run to
//! hundreds of megabytes, which is enough to matter next to a Rust `target/`.
//!
//! Even so the download is JSON only, never weights, so the sweep covers a
//! 30 GB checkpoint and a 700 GB one at the same price.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use pmetal_models::architectures::utils::resolve_activation;
use pmetal_models::dispatcher::ModelArchitecture;

/// Config keys whose value is an `ACT2FN` name.
///
/// These compare by *resolved function* rather than by spelling: a config
/// saying `"gelu_fast"` and a parsed enum re-serializing as `"gelu_new"` name
/// the same function, and flagging that would be noise. Anything that resolves
/// differently — or does not resolve at all — is still a finding.
const ACTIVATION_KEYS: &[&str] = &["hidden_act", "hidden_activation", "activation_function"];

/// Manifest tag for a checkpoint pmetal must decline to load.
const UNSUPPORTED: &str = "Unsupported";

/// Two JSON values agree, allowing for integer/float spelling and float noise.
///
/// `1e-05` parsed into an `f32` and re-serialized is not bitwise `1e-05` as an
/// `f64`, and `10000` in the config becomes `10000.0` in a float field. Neither
/// is a defect, so numbers compare with a relative tolerance wide enough to
/// absorb the f32 round trip and nothing else.
fn values_agree(raw: &serde_json::Value, parsed: &serde_json::Value) -> bool {
    use serde_json::Value;
    match (raw, parsed) {
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64(), b.as_f64());
            match (a, b) {
                (Some(a), Some(b)) => {
                    let scale = a.abs().max(b.abs()).max(1.0);
                    (a - b).abs() <= 1e-6 * scale
                }
                _ => false,
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| values_agree(x, y))
        }
        // Subset, not equality: the interesting question is whether pmetal
        // contradicts a value the checkpoint states, and filling a default for
        // a key the checkpoint omits is not a contradiction. Requiring equal
        // key sets would flag every nested default (and every HF bookkeeping
        // field like `torch_dtype` that no architecture models).
        (Value::Object(a), Value::Object(b)) => a
            .iter()
            .filter(|(_, v)| !v.is_null())
            .all(|(k, v)| b.get(k).is_none_or(|w| values_agree(v, w))),
        _ => raw == parsed,
    }
}

/// Both names resolve to the same `ACT2FN` function.
fn same_activation(raw: &serde_json::Value, parsed: &serde_json::Value) -> bool {
    let (Some(a), Some(b)) = (raw.as_str(), parsed.as_str()) else {
        return false;
    };
    match (resolve_activation(a), resolve_activation(b)) {
        (Some(f), Some(g)) => std::ptr::fn_addr_eq(f, g),
        _ => false,
    }
}

/// Every place `key` could legitimately be stated in the raw config.
///
/// A multimodal checkpoint states some fields twice — once on the wrapper and
/// once on `text_config` — and which one pmetal holds depends on whether that
/// architecture unwraps. Gemma 4 unwraps, so it parses the text tower's
/// `model_type` of `gemma4_text`; Mllama does not, so it parses the wrapper's
/// `mllama`. Both are correct, and preferring either scope reports the other
/// as a defect. So agreement with *any* stated scope counts, and the loader's
/// unwrap rule stays where it belongs — in the loader.
fn raw_candidates<'a>(raw: &'a serde_json::Value, key: &str) -> Vec<&'a serde_json::Value> {
    [
        raw.get(key),
        raw.get("text_config").and_then(|t| t.get(key)),
    ]
    .into_iter()
    .flatten()
    .filter(|v| !v.is_null())
    .collect()
}

struct Finding {
    slug: String,
    detail: String,
}

fn check_one(
    dir: &Path,
    slug: &str,
    expected_arch: &str,
    findings: &mut Vec<Finding>,
) -> Option<(ModelArchitecture, usize)> {
    let mut fail = |detail: String| {
        findings.push(Finding {
            slug: slug.to_string(),
            detail,
        });
    };

    let config_path = dir.join("config.json");
    let content = match std::fs::read_to_string(&config_path) {
        Ok(c) => c,
        Err(e) => {
            fail(format!("unreadable config.json: {e}"));
            return None;
        }
    };
    // `config_value`, not `serde_json::from_str`: a Nemotron-H config states
    // `"time_step_limit": [0.0, Infinity]`, which is Python's JSON dialect, not
    // JSON. The loader tolerates it and so must the oracle side of this check.
    let raw = match pmetal_models::dispatcher::config_value(&content) {
        Ok(v) => v,
        Err(e) => {
            fail(format!("config.json is not valid JSON: {e}"));
            return None;
        }
    };

    let detected = ModelArchitecture::detect(dir);

    // `UNSUPPORTED` marks a `model_type` pmetal must *refuse*. Rejecting up
    // front is the honest outcome for a family whose config or key layout the
    // implementation does not actually handle; silently routing it somewhere
    // plausible buys a confusing failure much later, which is what RoBERTa and
    // DistilBERT did by landing on the BERT path.
    if expected_arch == UNSUPPORTED {
        if let Ok(arch) = detected {
            fail(format!(
                "detection resolved to {arch:?}, but this family is not supported — \
                 either implement it or keep it rejected"
            ));
        }
        return None;
    }

    let arch = match detected {
        Ok(a) => a,
        Err(e) => {
            fail(format!(
                "architecture detection failed (model_type {:?}): {e}",
                raw.get("model_type").and_then(|v| v.as_str())
            ));
            return None;
        }
    };
    if format!("{arch:?}") != expected_arch {
        fail(format!(
            "detected {arch:?}, manifest expects {expected_arch}"
        ));
    }

    let parsed = match arch.parse_config_json(&content) {
        Ok(v) => v,
        Err(e) => {
            fail(format!("{arch:?} cannot parse the released config: {e}"));
            return None;
        }
    };
    let Some(parsed_obj) = parsed.as_object() else {
        fail(format!("{arch:?} config did not serialize to an object"));
        return None;
    };

    // Every field pmetal models, held against what the checkpoint states.
    let mut compared = 0usize;
    for (key, parsed_value) in parsed_obj {
        let candidates = raw_candidates(&raw, key);
        if candidates.is_empty() {
            continue; // pmetal-only field, or the checkpoint leaves it default
        }
        compared += 1;
        let agrees = candidates.iter().any(|raw_value| {
            values_agree(raw_value, parsed_value)
                || (ACTIVATION_KEYS.contains(&key.as_str())
                    && same_activation(raw_value, parsed_value))
        });
        if agrees {
            continue;
        }
        let stated: Vec<String> = candidates.iter().map(|v| v.to_string()).collect();
        fail(format!(
            "{key}: checkpoint says {}, pmetal parsed {parsed_value}",
            stated.join(" / ")
        ));
    }

    // A checkpoint that shares no field name with pmetal's struct compares
    // nothing and passes for free. That is how a config could be renamed out
    // from under an architecture without this noticing.
    if compared == 0 {
        fail(format!(
            "{arch:?} shares no field name with this config — the comparison was vacuous"
        ));
    }

    Some((arch, compared))
}

#[test]
#[ignore = "requires PMETAL_REAL_CONFIGS; see the module docs"]
fn released_configs_parse_the_way_the_checkpoint_states() {
    let Ok(root) = std::env::var("PMETAL_REAL_CONFIGS") else {
        panic!("set PMETAL_REAL_CONFIGS to a directory built by download_configs.py");
    };
    let root = PathBuf::from(root);
    let manifest_path = root.join("_manifest.json");
    let manifest: BTreeMap<String, String> = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path)
            .unwrap_or_else(|e| panic!("missing {manifest_path:?}: {e}")),
    )
    .expect("_manifest.json is slug -> ModelArchitecture");

    assert!(
        !manifest.is_empty(),
        "manifest is empty — nothing was downloaded"
    );

    let mut findings = Vec::new();
    let mut rows = Vec::new();
    for (slug, expected_arch) in &manifest {
        let dir = root.join(slug);
        let before = findings.len();
        let result = check_one(&dir, slug, expected_arch, &mut findings);
        let status = if findings.len() == before {
            "ok"
        } else {
            "FAIL"
        };
        let (arch, compared) = result
            .map(|(a, c)| (format!("{a:?}"), c.to_string()))
            .unwrap_or_else(|| ("-".to_string(), "-".to_string()));
        rows.push(format!(
            "  {status:4}  {slug:52}  {arch:16}  {compared:>3} fields"
        ));
    }

    println!("\nreal released configs, {} checkpoints:", manifest.len());
    for row in &rows {
        println!("{row}");
    }

    if !findings.is_empty() {
        let mut msg = format!(
            "\n{} disagreement(s) with released configs:\n",
            findings.len()
        );
        for f in &findings {
            msg.push_str(&format!("  {}: {}\n", f.slug, f.detail));
        }
        panic!("{msg}");
    }
}
