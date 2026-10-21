//! Declares the app's commands to Tauri, so the capability file can allow them
//! one by one.
//!
//! Without this every `#[tauri::command]` is callable from every window. With
//! it each command has an `allow-<name>` permission, and a window can call
//! only those its capability lists. The list is read from `src/commands.rs`
//! (every function under `#[tauri::command]`), so a new command is declared by
//! being written, and a test keeps the capability file and the handler list
//! in step with it.

fn main() {
    println!("cargo:rerun-if-changed=src/commands.rs");
    let source = std::fs::read_to_string("src/commands.rs").expect("src/commands.rs is readable");
    // Tauri wants the names for the life of the program; the build script's
    // is a few milliseconds.
    let commands: &'static [&'static str] = Box::leak(
        command_names(&source)
            .into_iter()
            .map(|name| &*Box::leak(name.into_boxed_str()))
            .collect::<Vec<&'static str>>()
            .into_boxed_slice(),
    );
    tauri_build::try_build(
        tauri_build::Attributes::new()
            .app_manifest(tauri_build::AppManifest::new().commands(commands)),
    )
    .expect("failed to run the Tauri build script");
}

/// The names of the functions under `#[tauri::command]`.
fn command_names(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut lines = source.lines();
    while let Some(line) = lines.next() {
        if line.trim() != "#[tauri::command]" {
            continue;
        }
        for next in lines.by_ref() {
            let next = next.trim();
            if let Some(rest) = next
                .strip_prefix("pub async fn ")
                .or_else(|| next.strip_prefix("pub fn "))
            {
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                names.push(name);
                break;
            }
        }
    }
    names
}
