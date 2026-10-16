//! The MCP docs list exactly the tools the server defines.

mod common;

use std::collections::BTreeSet;

use common::Harness;
use serde_json::json;

/// Names of the tools the server lists.
async fn defined_tools() -> BTreeSet<String> {
    let list = Harness::new("exit 0").result("tools/list", json!({})).await;
    list["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect()
}

/// The tool names in the docs' tool table, and the count its intro states.
fn documented_tools() -> (BTreeSet<String>, usize) {
    let doc = include_str!("../../../docs/cli/mcp.md");
    let start = doc.find("| Group | Tools |").expect("tool table");
    let table = &doc[start..];
    let end = table.find("\n\n").unwrap_or(table.len());
    let mut tools = BTreeSet::new();
    for cell in table[..end].split('`').skip(1).step_by(2) {
        tools.insert(cell.to_string());
    }
    let count = doc[..start]
        .lines()
        .rev()
        .find_map(|l| l.strip_suffix(" tools, grouped by what they do."))
        .and_then(|n| n.trim().parse().ok())
        .expect("the stated tool count");
    (tools, count)
}

#[tokio::test]
async fn the_docs_list_every_tool_and_nothing_else() {
    let defined = defined_tools().await;
    let (documented, stated) = documented_tools();
    assert!(defined.len() > 40, "found only {} tools", defined.len());
    let undocumented: Vec<_> = defined.difference(&documented).collect();
    let phantom: Vec<_> = documented.difference(&defined).collect();
    assert!(
        undocumented.is_empty(),
        "tools missing from docs/cli/mcp.md: {undocumented:?}"
    );
    assert!(
        phantom.is_empty(),
        "docs/cli/mcp.md lists tools that don't exist: {phantom:?}"
    );
    assert_eq!(
        stated,
        defined.len(),
        "docs/cli/mcp.md states the wrong tool count"
    );
}
