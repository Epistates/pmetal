//! The MCP docs list exactly the tools the server defines.

use std::collections::BTreeSet;

/// Names of the `#[tool]` methods in the server source.
fn defined_tools() -> BTreeSet<String> {
    let src = include_str!("../src/lib.rs");
    let mut tools = BTreeSet::new();
    let mut after_attr = false;
    for line in src.lines() {
        let line = line.trim();
        if line.starts_with("#[tool") {
            after_attr = true;
            continue;
        }
        if after_attr {
            if let Some(rest) = line
                .strip_prefix("pub async fn ")
                .or_else(|| line.strip_prefix("async fn "))
            {
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                tools.insert(name);
                after_attr = false;
            } else if !line.starts_with("#[") && !line.starts_with("///") {
                after_attr = false;
            }
        }
    }
    tools
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

#[test]
fn the_docs_list_every_tool_and_nothing_else() {
    let defined = defined_tools();
    let (documented, stated) = documented_tools();
    assert!(defined.len() > 40, "parsed only {} tools", defined.len());
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
