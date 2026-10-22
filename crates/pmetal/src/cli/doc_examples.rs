//! The `pmetal` commands in the CLI docs and the example scripts, parsed by the
//! real argument parser.
//!
//! A flag renamed or removed, or a required argument added, fails here instead
//! of leaving a page or a script that no longer runs. Only the parse is
//! checked: nothing is executed and no model is loaded.
//!
//! What counts as a command: a line of a fenced block (or of a script) that
//! runs `pmetal`, with `\` continuations joined, `&&`, `|` and `;` ending it,
//! and a redirection cut off. A synopsis with `<PLACEHOLDER>` or `[OPTIONS]`
//! in it is not a command and is skipped.

use super::Cli;
use clap::Parser;
use clap::error::ErrorKind;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The repository root, or `None` when this crate is built from a package that
/// does not carry the docs.
fn repo_root() -> Option<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    root.join("docs/cli").is_dir().then_some(root)
}

/// The variables a script has set so far, by name.
type Env = HashMap<String, String>;

/// Split one logical line into words the way a shell would, enough for these
/// files: single and double quotes, backslash escapes, `$VAR` and
/// `${VAR:-default}` expansion (from `env`, else the default, else `x`), and a
/// `#` that starts a word ends the line.
fn words(line: &str, env: &Env) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\'' => {
                in_word = true;
                for q in chars.by_ref() {
                    if q == '\'' {
                        break;
                    }
                    word.push(q);
                }
            }
            '"' => {
                in_word = true;
                while let Some(q) = chars.next() {
                    match q {
                        '"' => break,
                        '\\' => {
                            if let Some(e) = chars.next() {
                                if !matches!(e, '"' | '\\' | '$' | '`') {
                                    word.push('\\');
                                }
                                word.push(e);
                            }
                        }
                        '$' => expand(&mut chars, &mut word, env),
                        _ => word.push(q),
                    }
                }
            }
            '\\' => {
                in_word = true;
                if let Some(e) = chars.next() {
                    word.push(e);
                }
            }
            '$' => {
                in_word = true;
                expand(&mut chars, &mut word, env);
            }
            '#' if !in_word => break,
            c if c.is_whitespace() => {
                if in_word {
                    words.push(std::mem::take(&mut word));
                    in_word = false;
                }
            }
            c => {
                in_word = true;
                word.push(c);
            }
        }
    }
    if in_word {
        words.push(word);
    }
    words
}

/// Append what the variable after a `$` stands for: `pmetal` for the
/// `PMETAL_BIN` the scripts run, the value the script gave it (the default of
/// its own `${VAR:-default}`), the default of this `${VAR:-default}`, else a
/// placeholder.
fn expand(chars: &mut std::iter::Peekable<std::str::Chars<'_>>, word: &mut String, env: &Env) {
    let braced = chars.peek() == Some(&'{');
    if braced {
        chars.next();
    }
    let mut name = String::new();
    while chars
        .peek()
        .is_some_and(|c| c.is_alphanumeric() || *c == '_')
    {
        name.extend(chars.next());
    }
    let mut default = None;
    if braced {
        let mut rest = String::new();
        for c in chars.by_ref() {
            if c == '}' {
                break;
            }
            rest.push(c);
        }
        default = rest.strip_prefix(":-").map(str::to_string);
    }
    match (name.as_str(), env.get(&name), default) {
        ("PMETAL_BIN", ..) => word.push_str("pmetal"),
        (_, Some(value), _) => word.push_str(value),
        (_, None, Some(default)) => word.push_str(&default),
        _ => word.push('x'),
    }
}

/// The `pmetal` argv of each command on one logical line.
fn commands_of(line: &str, env: &Env) -> Vec<Vec<String>> {
    let line = line.trim().trim_start_matches("$ ");
    let words = words(line, env);
    let mut commands = Vec::new();
    let mut segment: Vec<String> = Vec::new();
    for word in words.into_iter().chain(std::iter::once(";".to_string())) {
        if matches!(word.as_str(), "&&" | "||" | "|" | ";") {
            if let Some(command) = pmetal_command(std::mem::take(&mut segment)) {
                commands.push(command);
            }
        } else {
            segment.push(word);
        }
    }
    commands
}

/// `segment` as a `pmetal` argv, if it runs `pmetal` and is a command rather
/// than a synopsis.
fn pmetal_command(segment: Vec<String>) -> Option<Vec<String>> {
    let mut words = segment.into_iter().skip_while(|w| {
        w.split_once('=').is_some_and(|(name, _)| {
            !name.is_empty() && name.chars().all(|c| c.is_ascii_uppercase() || c == '_')
        })
    });
    let first = words.next()?;
    if first != "pmetal" {
        return None;
    }
    let mut argv = vec!["pmetal".to_string()];
    for word in words {
        if word.starts_with('>') || word.starts_with("2>") {
            break;
        }
        argv.push(word);
    }
    let synopsis = argv
        .iter()
        .any(|w| w.contains('<') || w.contains('[') || w.contains('>'));
    (!synopsis && argv.len() > 1).then_some(argv)
}

/// Every `pmetal` command of a markdown file's fenced blocks or of a script,
/// with the line it starts on.
fn commands_in(path: &Path) -> Vec<(usize, Vec<String>)> {
    let text = std::fs::read_to_string(path).expect("a doc file reads");
    let markdown = path.extension().is_some_and(|e| e == "md");
    let mut fenced = !markdown;
    let mut found = Vec::new();
    let mut env = Env::new();
    let mut pending: Option<(usize, String)> = None;
    for (index, line) in text.lines().enumerate() {
        if markdown && line.trim_start().starts_with("```") {
            fenced = !fenced;
            pending = None;
            continue;
        }
        if !fenced {
            continue;
        }
        let (start, mut logical) = pending.take().unwrap_or((index + 1, String::new()));
        let trimmed = line.trim_end();
        if let Some(continued) = trimmed.strip_suffix('\\') {
            logical.push_str(continued);
            logical.push(' ');
            pending = Some((start, logical));
            continue;
        }
        logical.push_str(trimmed);
        // A script's `NAME="${NAME:-default}"` stands for its default.
        if let Some((name, value)) = logical.split_once('=') {
            if !markdown
                && !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
            {
                let value = words(value, &env).into_iter().next().unwrap_or_default();
                env.insert(name.to_string(), value);
                continue;
            }
        }
        for argv in commands_of(&logical, &env) {
            found.push((start, argv));
        }
    }
    found
}

/// The files that hold commands: every CLI page and the examples.
fn doc_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for (dir, extensions) in [("docs/cli", &["md"][..]), ("examples", &["md", "sh"][..])] {
        let Ok(entries) = std::fs::read_dir(root.join(dir)) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| extensions.contains(&e))
            {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// Whether this build has the command a doc names (`serve` and `mcp` are
/// opt-in features).
fn command_is_built(argv: &[String]) -> bool {
    match argv.get(1).map(String::as_str) {
        Some("serve") => cfg!(feature = "serve"),
        Some("mcp") => cfg!(feature = "mcp"),
        _ => true,
    }
}

#[test]
fn the_commands_in_the_docs_and_examples_parse() {
    let Some(root) = repo_root() else {
        return;
    };
    let mut failures = Vec::new();
    let mut parsed = 0;
    for file in doc_files(&root) {
        for (line, argv) in commands_in(&file) {
            if !command_is_built(&argv) {
                continue;
            }
            match Cli::try_parse_from(&argv) {
                Ok(_) => parsed += 1,
                Err(e)
                    if matches!(e.kind(), ErrorKind::DisplayHelp | ErrorKind::DisplayVersion) =>
                {
                    parsed += 1;
                }
                Err(e) => failures.push(format!(
                    "{}:{line}: {}\n{}",
                    file.strip_prefix(&root).unwrap_or(&file).display(),
                    argv.join(" "),
                    e.to_string().lines().next().unwrap_or_default()
                )),
            }
        }
    }
    assert!(
        parsed > 50,
        "found only {parsed} commands; is the extractor broken?"
    );
    assert!(
        failures.is_empty(),
        "{} documented commands do not parse:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn the_extractor_reads_commands_the_way_a_shell_does() {
    let argv = |line: &str| commands_of(line, &Env::new());
    assert_eq!(
        argv("HF_HOME=/x pmetal train --model 'a b' --column-separator \"\\n\\n\" # note"),
        [[
            "pmetal",
            "train",
            "--model",
            "a b",
            "--column-separator",
            "\\n\\n"
        ]]
    );
    assert_eq!(
        argv("pmetal verify ./m && pmetal infer -m ./m -p hi"),
        [
            vec!["pmetal", "verify", "./m"],
            vec!["pmetal", "infer", "-m", "./m", "-p", "hi"]
        ]
    );
    assert_eq!(
        argv("pmetal completions zsh > ~/.zfunc/_pmetal"),
        [["pmetal", "completions", "zsh"]]
    );
    assert_eq!(
        argv("\"$PMETAL_BIN\" tokenize --input \"${INPUT:-corpus.jsonl}\""),
        [["pmetal", "tokenize", "--input", "corpus.jsonl"]]
    );
    // Synopses and other programs are not commands.
    assert!(argv("pmetal train --model <MODEL> [OPTIONS]").is_empty());
    assert!(argv("cargo run -p pmetal -- train").is_empty());
}
