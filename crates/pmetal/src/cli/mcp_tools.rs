//! The command lines the MCP server builds are ones this CLI parses.
//!
//! The MCP tools for the dataset and Ollama commands spell their flags in a
//! table, and the tools for everything else add a few of their own; the parser
//! here is the judge of whether those flags exist.

use super::Cli;
use clap::Parser;
use pmetal_mcp::policy::PathPolicy;
use serde_json::{Value, json};

fn parses(label: &str, argv: &[String]) {
    let mut full = vec!["pmetal".to_string()];
    full.extend(argv.iter().cloned());
    if let Err(e) = Cli::try_parse_from(&full) {
        panic!(
            "{label}: `pmetal {}` is not a command line the CLI accepts:\n{e}",
            argv.join(" ")
        );
    }
}

#[test]
fn every_dataset_and_ollama_tool_builds_a_command_the_cli_parses() {
    let root = std::env::temp_dir().canonicalize().unwrap();
    let mut checked = 0;
    for tool in pmetal_mcp::cli_tools::CLI_TOOLS {
        for argv in tool.example_invocations(&root) {
            parses(tool.name, &argv);
            checked += 1;
        }
    }
    assert!(checked >= 28, "only {checked} command lines were checked");
}

#[test]
fn every_spec_tool_builds_a_command_the_cli_parses() {
    let root = std::env::temp_dir().canonicalize().unwrap();
    let policy = PathPolicy::new(root.clone(), vec![root]);
    for tool in pmetal_mcp::spec_tool_list() {
        // `pmetal serve` exists only in a build with the `serve` feature.
        if cfg!(not(feature = "serve")) && tool.name() == "start_serve" {
            continue;
        }
        let definition = serde_json::to_value(tool.definition()).unwrap();
        // Each required parameter is a Hub-style name, which is also a valid
        // relative path for the ones that name files.
        let mut args: serde_json::Map<String, Value> = definition["inputSchema"]["required"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|name| {
                let name = name.as_str().unwrap();
                let list = definition["inputSchema"]["properties"][name]["type"] == "array";
                let value = if list {
                    json!(["owner/name"])
                } else {
                    json!("owner/name")
                };
                (name.to_string(), value)
            })
            .collect();
        // A spec's own rules can need a parameter no descriptor marks required:
        // `eval` takes a dataset (perplexity) or a task, so neither is.
        if tool.name() == "eval_perplexity" {
            args.insert("dataset".to_string(), json!("owner/name"));
        }
        let prepared = tool
            .prepare(Value::Object(args), &policy)
            .unwrap_or_else(|e| panic!("{}: {e}", tool.name()));
        let mut argv = vec![prepared.plan.subcommand];
        argv.extend(prepared.plan.argv);
        parses(tool.name(), &argv);
    }
}
