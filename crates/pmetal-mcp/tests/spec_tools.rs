//! The tools generated from job specs carry exactly what the specs define:
//! every parameter reaches a spec field, every spec field has a parameter, and
//! every default is the spec's.

mod common;

use common::scratch_dir;
use pmetal_mcp::policy::PathPolicy;
use pmetal_mcp::spec_tools::SpecTool;
use serde_json::{Map, Value, json};

fn definitions(tool: &dyn SpecTool) -> (Map<String, Value>, Vec<String>) {
    let definition = serde_json::to_value(tool.definition()).unwrap();
    let schema = &definition["inputSchema"];
    let properties = schema["properties"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    let required = schema["required"]
        .as_array()
        .map(|r| r.iter().map(|v| v.as_str().unwrap().to_string()).collect())
        .unwrap_or_default();
    (properties, required)
}

/// A value for a parameter that differs from its default and is within its limits.
fn sample(name: &str, schema: &Value, default: Option<&Value>) -> Value {
    let differs = |v: &Value| default.is_none_or(|d| d != v);
    match schema["type"].as_str().unwrap() {
        "boolean" => json!(!default.and_then(Value::as_bool).unwrap_or(false)),
        "array" => json!(["sample-a", "sample-b"]),
        "string" => match schema["enum"].as_array() {
            Some(options) => options
                .iter()
                .find(|o| differs(o))
                .cloned()
                .expect("an option"),
            None => json!(format!("sample-{name}")),
        },
        "integer" => {
            let min = schema["minimum"].as_i64();
            let max = schema["maximum"].as_i64();
            let candidates = [
                min.unwrap_or(1).max(1),
                min.unwrap_or(0),
                min.unwrap_or(0) + 1,
                max.unwrap_or(1000),
                7,
            ];
            candidates
                .into_iter()
                .map(|n| json!(n))
                .find(|v| {
                    differs(v)
                        && min.is_none_or(|m| v.as_i64().unwrap() >= m)
                        && max.is_none_or(|m| v.as_i64().unwrap() <= m)
                })
                // A range of one value (a deprecated, ignored field) has no other.
                .or_else(|| default.cloned().filter(|_| min == max))
                .unwrap_or_else(|| panic!("no sample for {name}: {schema}"))
        }
        "number" => {
            let min = schema["minimum"].as_f64();
            let max = schema["maximum"].as_f64();
            let mid = match (min, max) {
                (Some(a), Some(b)) => (a + b) / 2.0,
                (Some(a), None) => a + 1.0,
                (None, Some(b)) => b - 1.0,
                (None, None) => 0.5,
            };
            [mid, min.unwrap_or(mid), max.unwrap_or(mid), 0.25]
                .into_iter()
                .map(|n| json!(n))
                .find(|v| differs(v))
                .unwrap()
        }
        other => panic!("{name}: type {other}"),
    }
}

fn same(a: &Value, b: &Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) if a.is_number() && b.is_number() => {
            (x - y).abs() <= 1e-6 * x.abs().max(y.abs()).max(1e-12)
        }
        _ => a == b,
    }
}

#[test]
fn every_parameter_reaches_its_spec_field() {
    let tools = pmetal_mcp::spec_tool_list();
    assert!(tools.len() >= 20);
    let mut checked = 0;
    for tool in &tools {
        let defaults = tool.defaults();
        let (properties, _) = definitions(tool.as_ref());
        for (name, schema) in &properties {
            let value = sample(name, schema, defaults.get(name));
            let parsed = tool
                .parsed(json!({ name: value.clone() }))
                .unwrap_or_else(|e| panic!("{}.{name} = {value}: {e}", tool.name()));
            assert!(
                same(&parsed[name], &value),
                "{}.{name}: sent {value}, the spec holds {}",
                tool.name(),
                parsed[name]
            );
            checked += 1;
        }
    }
    assert!(checked > 300, "only {checked} parameters were checked");
}

#[test]
fn a_tool_takes_exactly_the_fields_of_its_spec() {
    for tool in pmetal_mcp::spec_tool_list() {
        let defaults = tool.defaults();
        let mut parameters = tool.parameters();
        parameters.sort();
        let mut fields: Vec<String> = defaults.as_object().unwrap().keys().cloned().collect();
        fields.sort();
        assert_eq!(parameters, fields, "{}", tool.name());
        let (properties, _) = definitions(tool.as_ref());
        for (name, schema) in &properties {
            assert!(
                schema["description"]
                    .as_str()
                    .is_some_and(|d| !d.is_empty()),
                "{}.{name}",
                tool.name()
            );
        }
    }
}

#[test]
fn leaving_a_parameter_out_gives_the_specs_default() {
    let dir = scratch_dir("defaults");
    let policy = PathPolicy::new(dir.to_path_buf(), vec![dir.to_path_buf()]);
    for tool in pmetal_mcp::spec_tool_list() {
        let (properties, required) = definitions(tool.as_ref());
        let mut args = Map::new();
        for name in &required {
            args.insert(name.clone(), sample(name, &properties[name], None));
        }
        // A spec's own rules can need a parameter no descriptor marks required:
        // `eval` takes a dataset (perplexity) or a task, so neither is.
        let needed_by_rules: &[&str] = if tool.name() == "eval_perplexity" {
            &["dataset"]
        } else {
            &[]
        };
        for name in needed_by_rules {
            args.insert((*name).to_string(), sample(name, &properties[*name], None));
        }
        // Required references and paths are given as paths under the allowed root.
        for (name, value) in args.iter_mut() {
            if value.is_string() && properties[name].get("enum").is_none() {
                *value = json!(dir.join(name).display().to_string());
            }
        }
        let prepared = tool
            .prepare(Value::Object(args.clone()), &policy)
            .unwrap_or_else(|e| panic!("{} with only {args:?}: {e}", tool.name()));
        let defaults = tool.defaults();
        for (name, default) in defaults.as_object().unwrap() {
            if args.contains_key(name) {
                continue;
            }
            // A default path is made absolute, and the metrics file is placed.
            let got = &prepared.spec[name];
            let is_relative_path = default.as_str().is_some_and(|d| d.starts_with("./"));
            if is_relative_path || name == "log_metrics" || default.is_null() && got.is_string() {
                continue;
            }
            assert!(
                same(got, default),
                "{}.{name}: the spec defaults to {default}, the tool built {got}",
                tool.name()
            );
        }
        // And the command line is the one the spec itself builds.
        assert!(
            !prepared.argv.is_empty() || tool.name() == "bench_train",
            "{}",
            tool.name()
        );
    }
}

#[test]
fn omitting_a_required_parameter_names_it() {
    let dir = scratch_dir("required");
    let policy = PathPolicy::new(dir.to_path_buf(), vec![dir.to_path_buf()]);
    for tool in pmetal_mcp::spec_tool_list() {
        let (_, required) = definitions(tool.as_ref());
        if required.is_empty() {
            continue;
        }
        let error = tool.prepare(json!({}), &policy).unwrap_err().to_string();
        for name in &required {
            assert!(error.contains(name.as_str()), "{}: {error}", tool.name());
        }
    }
}

#[test]
fn the_tool_defaults_the_cli_defaults_and_not_a_copy() {
    // Values that were once typed into the MCP tools by hand.
    let tools = pmetal_mcp::spec_tool_list();
    let default_of = |tool: &str, field: &str| {
        tools.iter().find(|t| t.name() == tool).unwrap().defaults()[field].clone()
    };
    assert!(same(&default_of("distill", "temperature"), &json!(2.0)));
    assert!(same(
        &default_of("embed_train", "temperature"),
        &json!(0.05)
    ));
    assert!(same(&default_of("dflash", "temperature"), &json!(0.0)));
    assert!(same(&default_of("rlkd", "final_alpha"), &json!(0.05)));
    // The inference server listens on this machine only, unless asked.
    assert_eq!(default_of("start_serve", "host"), json!("127.0.0.1"));
}

#[test]
fn a_removed_flag_is_not_a_control() {
    // GRPO speculative decoding is gone from the pipeline; no GRPO-family tool
    // offers it (DFlash's own speculative tokens are another thing).
    for tool in pmetal_mcp::spec_tool_list() {
        if matches!(tool.name(), "grpo" | "rlkd") {
            for parameter in tool.parameters() {
                assert!(
                    !parameter.contains("speculative"),
                    "{}.{parameter}",
                    tool.name()
                );
            }
        }
    }
}
