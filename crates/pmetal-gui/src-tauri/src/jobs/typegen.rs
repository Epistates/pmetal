//! The TypeScript the frontend is checked against, generated from the specs.
//!
//! `src/lib/specs.generated.ts` lists, for every job, the names of the fields
//! its form may set, and gives the one spec the page builds directly
//! (`InferSpec`) in full. The frontend's own code is type-checked against it,
//! so a page that sets a field a spec no longer has fails `svelte-check`, and
//! this test fails if the file is not what the specs produce.
//!
//! After a spec changes, regenerate with
//! `UPDATE_SPEC_TYPES=1 cargo test --manifest-path crates/pmetal-gui/src-tauri/Cargo.toml spec_types`.
//!
//! A generator for the whole command surface was considered (tauri-specta,
//! ts-rs). tauri-specta is still a release candidate and needs `specta` derives
//! on every spec in `pmetal-core`; ts-rs needs the same derives. The specs
//! already describe themselves (`JobFields`), so this reads that instead.

use std::collections::BTreeSet;
use std::fmt::Write;

use pmetal::core::jobs::InferSpec;
use pmetal::core::{FieldKind, FieldShape, JobFields};
use serde_json::Value;

use super::kinds;

const PATH: &str = "../src/lib/specs.generated.ts";

fn quote(name: &str) -> String {
    format!("'{name}'")
}

/// The TypeScript type of a field of `S`, from its descriptor when it has one
/// and from the value of the default spec when it does not.
fn type_of(descriptor: Option<&pmetal::core::FieldDescriptor>, default: Option<&Value>) -> String {
    match descriptor.map(|d| (d.kind, d.shape)) {
        Some((kind, shape)) => {
            let one = match kind {
                FieldKind::Number { .. } | FieldKind::Integer { .. } => "number".to_string(),
                FieldKind::Toggle => "boolean".to_string(),
                FieldKind::Enum { options } => options
                    .iter()
                    .map(|o| quote(o))
                    .collect::<Vec<_>>()
                    .join(" | "),
                _ => "string".to_string(),
            };
            match shape {
                FieldShape::List => "string[]".to_string(),
                FieldShape::Optional => format!("{one} | null"),
                FieldShape::Scalar => one,
            }
        }
        None => match default {
            Some(Value::Bool(_)) => "boolean".to_string(),
            Some(Value::Number(_)) => "number".to_string(),
            Some(Value::String(_)) => "string".to_string(),
            Some(Value::Array(_)) => "string[]".to_string(),
            _ => "unknown".to_string(),
        },
    }
}

/// The whole file.
pub fn render() -> String {
    let mut out = String::new();
    out.push_str(
        "// Generated from the job specs in pmetal-core by the `spec_types` test in\n\
         // crates/pmetal-gui/src-tauri/src/jobs/typegen.rs. Do not edit by hand:\n\
         // change the spec, then run\n\
         //   UPDATE_SPEC_TYPES=1 cargo test --manifest-path crates/pmetal-gui/src-tauri/Cargo.toml spec_types\n\n",
    );

    out.push_str("/** The fields each job's form may set, by the spec's own names. */\n");
    out.push_str("export interface JobFields {\n");
    for kind in kinds::all() {
        let fields: Vec<String> = kind.form().fields.iter().map(|f| quote(&f.name)).collect();
        let _ = writeln!(out, "  {}: {};", kind.key(), fields.join(" | "));
    }
    out.push_str("}\n\n");
    out.push_str("export type JobKey = keyof JobFields;\n\n");

    out.push_str(
        "/** `pmetal infer`'s spec, as the chat sends it. Every field may be left out. */\n",
    );
    out.push_str("export interface InferSpec {\n");
    let default = serde_json::to_value(InferSpec::default()).expect("a spec serializes");
    let descriptors = InferSpec::field_descriptors();
    let mut keys: BTreeSet<&str> = default
        .as_object()
        .map(|o| o.keys().map(String::as_str).collect())
        .unwrap_or_default();
    keys.extend(descriptors.iter().map(|d| d.name));
    for key in keys {
        let descriptor = descriptors.iter().find(|d| d.name == key);
        let value = default.get(key);
        let mut ty = type_of(descriptor, value);
        // An optional field is `null` when absent.
        if value == Some(&Value::Null) && !ty.ends_with("| null") && ty != "unknown" {
            ty.push_str(" | null");
        }
        let _ = writeln!(out, "  {key}?: {ty};");
    }
    out.push_str("}\n");
    out
}

#[test]
fn spec_types_are_what_the_specs_produce() {
    let expected = render();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(PATH);
    if std::env::var_os("UPDATE_SPEC_TYPES").is_some() {
        std::fs::write(&path, &expected).expect("write the generated types");
        return;
    }
    let actual = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        actual == expected,
        "{} is out of date with the job specs. Regenerate it with:\n  \
         UPDATE_SPEC_TYPES=1 cargo test --manifest-path crates/pmetal-gui/src-tauri/Cargo.toml spec_types",
        path.display()
    );
}

#[test]
fn the_generated_file_names_every_job_and_the_chat_spec() {
    let text = render();
    for kind in kinds::all() {
        assert!(
            text.contains(&format!("  {}: ", kind.key())),
            "{}",
            kind.key()
        );
    }
    assert!(text.contains("  prompt?: string;"), "{text}");
    assert!(text.contains("  model?: string;"), "{text}");
}
