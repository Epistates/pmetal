//! The app's security configuration, checked against the code it governs.
//!
//! A capability file that names a command the app no longer has is harmless; one
//! that is missing a command breaks the window; and one that grants a broad
//! permission quietly widens what a compromised page can do. These tests make
//! each of those a failure here, not a surprise in a release.

use std::collections::BTreeSet;

use serde_json::Value;

fn read(path: &str) -> String {
    let full = format!("{}/{path}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&full).unwrap_or_else(|e| panic!("cannot read {full}: {e}"))
}

fn json(path: &str) -> Value {
    serde_json::from_str(&read(path)).unwrap_or_else(|e| panic!("{path} is not JSON: {e}"))
}

/// The functions under `#[tauri::command]` in `src/commands.rs`.
fn commands() -> BTreeSet<String> {
    let source = read("src/commands.rs");
    let mut names = BTreeSet::new();
    let mut lines = source.lines();
    while let Some(line) = lines.next() {
        if line.trim() == "#[tauri::command]" {
            for next in lines.by_ref() {
                if let Some(rest) = next
                    .trim()
                    .strip_prefix("pub async fn ")
                    .or_else(|| next.trim().strip_prefix("pub fn "))
                {
                    names.insert(
                        rest.chars()
                            .take_while(|c| c.is_alphanumeric() || *c == '_')
                            .collect(),
                    );
                    break;
                }
            }
        }
    }
    names
}

#[test]
fn every_command_is_registered_and_nothing_else() {
    let lib = read("src/lib.rs");
    let start =
        lib.find("generate_handler![").expect("a handler list") + "generate_handler![".len();
    let list = &lib[start..lib[start..].find(']').map(|i| i + start).unwrap()];
    let registered: BTreeSet<String> = list
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    assert_eq!(registered, commands());
}

#[test]
fn the_main_window_may_call_each_command_and_listen_for_events_and_nothing_more() {
    let capability = json("capabilities/default.json");
    assert_eq!(capability["windows"], serde_json::json!(["main"]));
    let granted: BTreeSet<String> = capability["permissions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_str().expect("a plain permission").to_string())
        .collect();
    let mut expected: BTreeSet<String> = commands()
        .iter()
        .map(|c| format!("allow-{}", c.replace('_', "-")))
        .collect();
    expected.insert("core:event:allow-listen".into());
    expected.insert("core:event:allow-unlisten".into());
    assert_eq!(granted, expected);
}

#[test]
fn no_capability_grants_the_page_the_file_system_a_shell_or_the_network() {
    for file in std::fs::read_dir(format!("{}/capabilities", env!("CARGO_MANIFEST_DIR"))).unwrap() {
        let path = file.unwrap().path();
        let text = std::fs::read_to_string(&path).unwrap();
        for forbidden in [
            "fs:",
            "shell:",
            "http:",
            "opener:",
            "process:",
            "updater:",
            "dialog:",
            "os:",
            "core:default",
            "core:window",
            "core:webview",
            "core:path",
            "core:app:",
            "core:resources",
            "core:menu",
            "core:tray",
            "core:image",
            "\"remote\"",
        ] {
            assert!(
                !text.contains(forbidden),
                "{} grants {forbidden}",
                path.display()
            );
        }
        let capability: Value = serde_json::from_str(&text).unwrap();
        assert!(
            capability.get("remote").is_none(),
            "a capability must not extend to remote URLs"
        );
    }
}

#[test]
fn the_content_security_policy_allows_nothing_the_app_does_not_use() {
    let config = json("tauri.conf.json");
    let csp = config["app"]["security"]["csp"].as_str().expect("a CSP");
    let directives: std::collections::BTreeMap<&str, Vec<&str>> = csp
        .split(';')
        .filter_map(|d| {
            let mut words = d.split_whitespace();
            Some((words.next()?, words.collect()))
        })
        .collect();

    assert_eq!(directives["default-src"], ["'none'"]);
    assert_eq!(
        directives["script-src"],
        ["'self'"],
        "no inline or eval scripts"
    );
    assert_eq!(directives["object-src"], ["'none'"]);
    assert_eq!(directives["base-uri"], ["'none'"]);
    assert_eq!(directives["frame-ancestors"], ["'none'"]);
    // The page talks to the backend, and to nothing else.
    assert_eq!(directives["connect-src"], ["ipc:", "http://ipc.localhost"]);
    for sources in directives.values() {
        for source in sources {
            assert!(
                !matches!(
                    *source,
                    "*" | "https:" | "http:" | "'unsafe-eval'" | "'unsafe-hashes'"
                ),
                "{source} is too broad"
            );
        }
    }
    assert_eq!(config["app"]["security"]["freezePrototype"], true);
}

#[test]
fn there_is_one_window_and_the_capability_names_it() {
    let config = json("tauri.conf.json");
    let windows = config["app"]["windows"].as_array().unwrap();
    assert_eq!(windows.len(), 1);
    assert_eq!(windows[0]["label"], "main");
}

#[test]
fn the_updater_is_off_everywhere_or_on_with_a_key() {
    // An updater needs a public key to verify what it downloads. There is none
    // configured, so the plugin, its permission and its screen must all be
    // absent: an updater that checks nothing is worse than none.
    let config = json("tauri.conf.json");
    let pubkey = config["plugins"]["updater"]["pubkey"]
        .as_str()
        .unwrap_or("");
    let cargo = read("Cargo.toml");
    let package = read("../package.json");
    if pubkey.is_empty() {
        assert!(
            !cargo.contains("tauri-plugin-updater"),
            "the plugin is built in without a key"
        );
        assert!(
            !package.contains("plugin-updater"),
            "the page asks for updates it cannot verify"
        );
        assert_eq!(config["bundle"]["createUpdaterArtifacts"], false);
    } else {
        let endpoints = config["plugins"]["updater"]["endpoints"]
            .as_array()
            .unwrap();
        assert!(!endpoints.is_empty());
        assert!(
            endpoints
                .iter()
                .all(|e| e.as_str().unwrap().starts_with("https://"))
        );
    }
}

#[test]
fn nothing_is_shipped_with_a_secret_or_a_personal_path_in_its_configuration() {
    for file in [
        "tauri.conf.json",
        "tauri.bundle.conf.json",
        "capabilities/default.json",
    ] {
        let text = read(file);
        assert!(!text.contains(concat!("/Use", "rs/")), "{file}");
        assert!(!text.to_lowercase().contains("hf_"), "{file}");
    }
}
