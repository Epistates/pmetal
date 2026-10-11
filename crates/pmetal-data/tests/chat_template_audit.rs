//! Cross-model chat-template audit.
//!
//! # Safety note
//!
//! This file contains a single `unsafe` block inside the test body — the
//! env-var setter `std::env::set_var` was marked unsafe in Rust 2024
//! because another thread could read while we write. Our test is
//! single-threaded (no other code runs concurrently) and the env var is
//! read-only from the chat-template renderer, so the usage is sound.
//!
//! This test walks every subdirectory of `$PMETAL_CHAT_AUDIT_DIR` and, for
#![allow(unsafe_code)]
//! each model, verifies that pmetal's chat-template stack matches the real
//! upstream Jinja template. Each subdirectory holds one model's tokenizer and
//! config files as published on the Hugging Face Hub (`tokenizer.json`,
//! `tokenizer_config.json`, `config.json`, `generation_config.json`, any
//! `chat_template.jinja`; no weights) plus an `expected.json` rendered with
//! `transformers.AutoTokenizer.apply_chat_template(...)` on
//! [`CANONICAL_PROMPT`] with `add_generation_prompt=True`, the clock frozen at
//! 2026-04-14, and the ids encoded with `add_special_tokens=False`:
//!
//! ```json
//! {
//!   "model_id": "Qwen/Qwen3-0.6B",
//!   "arch_tag": "Qwen",
//!   "prompt_ids_default": [...],
//!   "prompt_ids_thinking": [...],
//!   "prompt_text_default": "...",
//!   "prompt_text_thinking": "...",
//!   "eos_token_id": [...],
//!   "generation_config": { ... }
//! }
//! ```
//!
//! `arch_tag` is the [`ChatTemplateType`] name the model should be detected
//! as. The `_default` render passes `enable_thinking=False` and the
//! `_thinking` one `enable_thinking=True` (templates without the switch
//! render the same text twice). `eos_token_id` is `generation_config.json`'s,
//! falling back to the tokenizer's, always as a list, and `generation_config`
//! is that file's sampling fields verbatim. The audit needs model downloads,
//! so it is skipped when `PMETAL_CHAT_AUDIT_DIR` is unset:
//!
//! ```bash
//! PMETAL_CHAT_AUDIT_DIR=<dir of model config dirs> \
//!     cargo test -p pmetal-data --test chat_template_audit -- --nocapture
//! ```
//!
//! `PMETAL_CHAT_AUDIT_VERBOSE=1` adds the rendered text and tokens of every
//! mismatch to the report.
//!
//! Pass criteria per model (in priority order):
//!   1. `detect_chat_template` returns a template with the arch_tag we
//!      expected (or a close variant — e.g. Qwen is compatible with
//!      ChatMl since it inherits the same markers).
//!   2. Applying the template to the canonical prompt produces a token
//!      sequence that MATCHES EITHER the reference default or thinking
//!      variant. We don't require bit-exact text match because harmless
//!      whitespace drift (a trailing newline) is common; we compare token
//!      IDs after running both sides through the same tokenizer.
//!   3. `collect_all_stop_tokens` covers every ID in the reference EOS set.
//!   4. `SamplingDefaults::from_dir` picks up any fields defined in
//!      generation_config.json.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use pmetal_data::chat_templates::{ChatTemplateType, Message, detect_chat_template};
use pmetal_data::inference_config::{
    collect_all_stop_tokens, load_sampling_from_generation_config,
};
use pmetal_data::tokenizer::Tokenizer;
use serde_json::Value;

/// The single user turn every `expected.json` is rendered from.
const CANONICAL_PROMPT: &str = "What is the capital of France?";

/// Which `arch_tag` strings should be considered equivalent for the
/// "detected template matches" check. pmetal uses `Qwen` for Qwen-family
/// models (they inherit ChatML tags), so accepting either `Qwen` or
/// `ChatMl` for Qwen-tagged models is correct. Same for GPT-OSS harmony.
fn template_matches(expected_tag: &str, detected: ChatTemplateType) -> bool {
    let detected_str = format!("{:?}", detected);
    if detected_str == expected_tag {
        return true;
    }
    // Accept known equivalents.
    matches!(
        (expected_tag, detected_str.as_str()),
        ("Qwen", "ChatMl")
            | ("ChatMl", "Qwen")
            // DeepSeek R1 Distill Qwen variants re-use Qwen/ChatML markers.
            | ("DeepSeek", "Qwen")
            | ("DeepSeek", "ChatMl")
    )
}

/// The configs directory, or `None` when the audit isn't configured. A path
/// that is set but missing is a mistake, not a reason to skip quietly.
fn audit_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("PMETAL_CHAT_AUDIT_DIR")?);
    assert!(
        dir.is_dir(),
        "PMETAL_CHAT_AUDIT_DIR={} is not a directory",
        dir.display()
    );
    Some(dir)
}

#[derive(Debug)]
struct ModelReport {
    model_id: String,
    expected_tag: String,
    detected_tag: String,
    template_ok: bool,
    ids_match_mode: Option<&'static str>, // "default" | "thinking" | None
    pmetal_len: usize,
    ref_default_len: usize,
    ref_thinking_len: usize,
    stop_missing: Vec<u32>,
    sampling_defaults_set: bool,
    notes: Vec<String>,
}

fn run_audit_on(dir: &Path) -> ModelReport {
    let expected: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("expected.json")).unwrap()).unwrap();
    let model_id = expected["model_id"].as_str().unwrap().to_string();
    let expected_tag = expected["arch_tag"].as_str().unwrap().to_string();

    let mut report = ModelReport {
        model_id: model_id.clone(),
        expected_tag: expected_tag.clone(),
        detected_tag: String::new(),
        template_ok: false,
        ids_match_mode: None,
        pmetal_len: 0,
        ref_default_len: expected["prompt_ids_default"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
        ref_thinking_len: expected["prompt_ids_thinking"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
        stop_missing: Vec::new(),
        sampling_defaults_set: false,
        notes: Vec::new(),
    };

    let template = detect_chat_template(dir, &model_id);
    report.detected_tag = format!("{:?}", template.template_type);
    report.template_ok = template_matches(&expected_tag, template.template_type);

    let tokenizer = match Tokenizer::from_model_dir(dir) {
        Ok(t) => t,
        Err(e) => {
            report.notes.push(format!("tokenizer load failed: {e}"));
            return report;
        }
    };

    // Run pmetal's inference prompt format (default = no-thinking path).
    let messages = vec![Message::user(CANONICAL_PROMPT)];
    let formatted = template.apply_inference(&messages, true, None);
    let pmetal_ids = match tokenizer.encode(&formatted.text) {
        Ok(ids) => ids,
        Err(e) => {
            report.notes.push(format!("tokenize failed: {e}"));
            return report;
        }
    };
    report.pmetal_len = pmetal_ids.len();

    let ref_default: Vec<u32> = expected["prompt_ids_default"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|x| x as u32))
                .collect()
        })
        .unwrap_or_default();
    let ref_thinking: Vec<u32> = expected["prompt_ids_thinking"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|x| x as u32))
                .collect()
        })
        .unwrap_or_default();

    // The reference ids are encoded with `add_special_tokens=False`, and
    // pmetal's `Tokenizer::encode` does the same — both correctly parse
    // `<|begin_of_text|>` etc as real special tokens from the text.
    // Compare the raw sequences verbatim.
    if pmetal_ids == ref_default {
        report.ids_match_mode = Some("default");
    } else if pmetal_ids == ref_thinking {
        report.ids_match_mode = Some("thinking");
    } else {
        let common = pmetal_ids
            .iter()
            .zip(ref_default.iter())
            .take_while(|(a, b)| a == b)
            .count();
        report.notes.push(format!(
            "ids mismatch: pmetal={} ref_default={} ref_thinking={} common_prefix={}",
            pmetal_ids.len(),
            ref_default.len(),
            ref_thinking.len(),
            common
        ));
        if std::env::var("PMETAL_CHAT_AUDIT_VERBOSE").is_ok() {
            let pmetal_toks: Vec<String> = pmetal_ids
                .iter()
                .map(|&id| {
                    tokenizer
                        .inner()
                        .id_to_token(id)
                        .unwrap_or_else(|| format!("<id={id}>"))
                })
                .collect();
            report
                .notes
                .push(format!("pmetal text:   {:?}", formatted.text));
            report
                .notes
                .push(format!("pmetal tokens: {:?}", pmetal_toks));
            if let Some(text) = expected["prompt_text_default"].as_str() {
                report.notes.push(format!("ref default:   {text:?}"));
            }
            if let Some(text) = expected["prompt_text_thinking"].as_str() {
                report.notes.push(format!("ref thinking:  {text:?}"));
            }
        }
    }

    // Stop-token coverage.
    let ref_eos: Vec<u32> = expected["eos_token_id"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64().map(|x| x as u32))
                .collect()
        })
        .unwrap_or_default();
    let pmetal_stops = collect_all_stop_tokens(dir, &tokenizer, Some(template.template_type));
    let stop_set: HashSet<u32> = pmetal_stops.iter().copied().collect();
    for id in &ref_eos {
        if !stop_set.contains(id) {
            report.stop_missing.push(*id);
        }
    }

    // Sampling defaults — verify every sampling field that
    // generation_config.json explicitly specifies is picked up by pmetal's
    // raw loader (`load_sampling_from_generation_config`). We intentionally
    // use the raw loader instead of `load_sampling_defaults` so the audit
    // isn't confused by model-card presets that deliberately override the
    // JSON values. Fields the JSON doesn't set aren't checked.
    let gc_obj = expected["generation_config"].as_object();
    let loaded = load_sampling_from_generation_config(dir);
    let mut sampling_ok = true;
    if let Some(obj) = gc_obj {
        let check_f32 = |want: f32, got: f32, name: &str, rpt: &mut ModelReport| {
            if (want - got).abs() > 1e-6 {
                rpt.notes
                    .push(format!("{name} mismatch: expected={want} got={got}"));
                false
            } else {
                true
            }
        };
        // transformers samples only with `do_sample: true`; otherwise the
        // file's temperature is ignored and generation is greedy.
        let samples = obj.get("do_sample").and_then(|v| v.as_bool()) == Some(true);
        if let Some(v) = obj.get("temperature").and_then(|v| v.as_f64()) {
            let want = if samples { v as f32 } else { 0.0 };
            if !check_f32(want, loaded.temperature, "temperature", &mut report) {
                sampling_ok = false;
            }
        }
        if let Some(v) = obj.get("top_p").and_then(|v| v.as_f64()) {
            if !check_f32(v as f32, loaded.top_p, "top_p", &mut report) {
                sampling_ok = false;
            }
        }
        if let Some(v) = obj.get("top_k").and_then(|v| v.as_u64()) {
            if (v as usize) != loaded.top_k {
                report.notes.push(format!(
                    "top_k mismatch: expected={} got={}",
                    v, loaded.top_k
                ));
                sampling_ok = false;
            }
        }
        if let Some(v) = obj.get("min_p").and_then(|v| v.as_f64()) {
            if !check_f32(v as f32, loaded.min_p, "min_p", &mut report) {
                sampling_ok = false;
            }
        }
        if let Some(v) = obj.get("repetition_penalty").and_then(|v| v.as_f64()) {
            if !check_f32(
                v as f32,
                loaded.repetition_penalty,
                "repetition_penalty",
                &mut report,
            ) {
                sampling_ok = false;
            }
        }
    }
    report.sampling_defaults_set = sampling_ok;

    report
}

fn fmt_row(r: &ModelReport) -> String {
    let template = if r.template_ok {
        "OK".to_string()
    } else {
        format!("FAIL:{}!={}", r.expected_tag, r.detected_tag)
    };
    let ids = match r.ids_match_mode {
        Some(m) => format!("OK:{}", m),
        None => format!(
            "FAIL(p={} d={} t={})",
            r.pmetal_len, r.ref_default_len, r.ref_thinking_len
        ),
    };
    let stops = if r.stop_missing.is_empty() {
        "OK".to_string()
    } else {
        format!("MISS:{:?}", r.stop_missing)
    };
    let sampling = if r.sampling_defaults_set {
        "OK"
    } else {
        "FAIL"
    };
    format!(
        "{:50} | {:10} | {:18} | {:10} | {}",
        r.model_id, template, ids, stops, sampling
    )
}

#[test]
#[allow(unsafe_code)]
fn chat_template_audit() {
    // Freeze the clock that Llama 3 / gpt-oss chat templates see through
    // `strftime_now`. The references are rendered on 2026-04-14 and encode
    // `Today Date: 14 Apr 2026` (or the gpt-oss equivalent) directly into
    // the expected prompt ids. Without this override the test flips from
    // green to red every UTC midnight as the rendered `now()` moves past
    // the fixture's date. The override is a no-op for every code path
    // outside the audit because `PMETAL_CHAT_TEMPLATE_FROZEN_DATE` is
    // normally unset.
    //
    // SAFETY: single-threaded test context; no other code reads or sets
    // this env var concurrently. See the module-level safety note.
    unsafe {
        std::env::set_var("PMETAL_CHAT_TEMPLATE_FROZEN_DATE", "2026-04-14");
    }

    let Some(audit_root) = audit_dir() else {
        eprintln!(
            "chat_template_audit: skipped; set PMETAL_CHAT_AUDIT_DIR to a directory of \
             model config dirs, each with an expected.json (see the module docs)."
        );
        return;
    };

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&audit_root)
        .expect("read audit dir")
        .filter_map(|e| {
            let e = e.ok()?;
            let p = e.path();
            if p.is_dir() && p.join("expected.json").exists() {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    entries.sort();

    let reports: Vec<ModelReport> = entries.iter().map(|d| run_audit_on(d)).collect();

    println!(
        "\n{:50} | {:10} | {:18} | {:10} | sampling",
        "model", "template", "ids", "stops"
    );
    println!("{}", "-".repeat(110));
    for r in &reports {
        println!("{}", fmt_row(r));
        for note in &r.notes {
            println!("    -> {note}");
        }
    }
    println!();

    let failed: Vec<&ModelReport> = reports
        .iter()
        .filter(|r| {
            !r.template_ok
                || r.ids_match_mode.is_none()
                || !r.stop_missing.is_empty()
                || !r.sampling_defaults_set
        })
        .collect();

    println!(
        "summary: {} total | {} passing | {} failing",
        reports.len(),
        reports.len() - failed.len(),
        failed.len()
    );

    if !failed.is_empty() {
        let names: Vec<&str> = failed.iter().map(|r| r.model_id.as_str()).collect();
        panic!(
            "chat_template_audit: {} models failing: {:?}",
            failed.len(),
            names
        );
    }
}
