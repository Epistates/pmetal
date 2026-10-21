//! Where the app reads and writes, and which locations it refuses.
//!
//! Every path that reaches the backend from the webview is text a person typed
//! or a dialog returned. A compromised page could send anything, so each
//! command that writes or deletes checks its paths here first, and the
//! frontend gets no filesystem access of its own.

use std::path::{Component, Path, PathBuf};

/// Where the GUI writes what jobs produce: `~/pmetal-output`.
pub fn output_root() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("pmetal-output")
}

/// `~` and `~/x` spelled out; anything else as it was.
pub fn expand_home(text: &str) -> PathBuf {
    let text = text.trim();
    if text == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from("~"));
    }
    if let Some(rest) = text.strip_prefix("~/")
        && let Some(home) = dirs::home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(text)
}

/// Directories that are never a place to write into, scan or delete.
fn protected(path: &Path) -> Option<&'static str> {
    const SYSTEM: &[&str] = &[
        "/",
        "/System",
        "/usr",
        "/bin",
        "/sbin",
        "/etc",
        "/private/etc",
        "/dev",
        "/cores",
        "/Library",
        "/Applications",
        "/opt",
    ];
    for system in SYSTEM {
        if path == Path::new(system) {
            return Some("a system directory");
        }
    }
    for system in &[
        "/System",
        "/usr",
        "/bin",
        "/sbin",
        "/etc",
        "/private/etc",
        "/dev",
    ] {
        if path.starts_with(system) {
            return Some("a system directory");
        }
    }
    if let Some(home) = dirs::home_dir() {
        if path == home {
            return Some("your home directory itself");
        }
        for private in [".ssh", ".gnupg", ".aws", "Library", ".config", ".cargo"] {
            if path.starts_with(home.join(private)) {
                return Some("a directory that holds your settings or credentials");
            }
        }
    }
    None
}

/// The absolute location a job may write to for the path `raw` a person
/// entered. A relative path is placed under `root`; `..` is refused, so a
/// relative path cannot climb out of it.
pub fn resolve_output(raw: &str, root: &Path) -> Result<PathBuf, String> {
    let text = raw.trim();
    if text.is_empty() {
        return Err("needs a location".to_string());
    }
    if text.contains('\0') {
        return Err("is not a valid path".to_string());
    }
    let path = expand_home(text);
    if path.components().any(|c| c == Component::ParentDir) {
        return Err("must not contain '..'. Give the full path instead.".to_string());
    }
    let path = if path.is_absolute() {
        path
    } else {
        root.join(path)
    };
    // `./a/./b` and `a//b` mean `a/b`.
    let path: PathBuf = path.components().collect();
    match protected(&path) {
        Some(why) => Err(format!(
            "is {why} ({}). Choose a folder of your own.",
            path.display()
        )),
        None => Ok(path),
    }
}

/// The deepest ancestor of `path` that exists, canonicalized, with the
/// remainder put back: where `path` really is, even though it is not made yet.
fn really(path: &Path) -> PathBuf {
    let mut tail = Vec::new();
    let mut current = path;
    loop {
        if let Ok(real) = current.canonicalize() {
            return tail.iter().rev().fold(real, |p, name| p.join(name));
        }
        match (current.parent(), current.file_name()) {
            (Some(parent), Some(name)) => {
                tail.push(name.to_os_string());
                current = parent;
            }
            _ => return path.to_path_buf(),
        }
    }
}

/// Whether writing to `output` would write over, inside or around `other`.
pub fn overlaps(output: &Path, other: &Path) -> bool {
    let (output, other) = (really(output), really(other));
    output.starts_with(&other) || other.starts_with(&output)
}

/// A directory the app may scan for models: it exists, is a directory, and
/// is not a system location. Returns it in canonical form.
pub fn scan_dir(raw: &str) -> Result<PathBuf, String> {
    let path = expand_home(raw);
    if !path.is_absolute() {
        return Err(format!(
            "'{raw}' is not a full path. Choose the folder with the Browse button."
        ));
    }
    let real = path
        .canonicalize()
        .map_err(|_| format!("'{raw}' does not exist."))?;
    if !real.is_dir() {
        return Err(format!("'{raw}' is a file, not a folder."));
    }
    match protected(&real) {
        Some(why) => Err(format!(
            "'{raw}' is {why}; it cannot be scanned for models."
        )),
        None => Ok(real),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> PathBuf {
        PathBuf::from("/Users/someone/pmetal-output")
    }

    #[test]
    fn a_relative_output_lands_under_the_output_root() {
        assert_eq!(
            resolve_output("runs/a", &root()).unwrap(),
            root().join("runs/a")
        );
        assert_eq!(
            resolve_output("./a//b/.", &root()).unwrap(),
            root().join("a/b")
        );
    }

    #[test]
    fn an_absolute_output_is_kept() {
        assert_eq!(
            resolve_output("/Volumes/Data/out", &root()).unwrap(),
            PathBuf::from("/Volumes/Data/out")
        );
    }

    #[test]
    fn climbing_out_with_dot_dot_is_refused() {
        for bad in ["../x", "a/../../b", "/tmp/../etc/x"] {
            assert!(
                resolve_output(bad, &root()).unwrap_err().contains(".."),
                "{bad}"
            );
        }
    }

    #[test]
    fn system_and_credential_locations_are_refused() {
        for bad in [
            "/",
            "/System/Library/x",
            "/usr/local",
            "/etc",
            "/bin/sh",
            "~/.ssh",
            "~/.ssh/keys",
            "~",
        ] {
            assert!(resolve_output(bad, &root()).is_err(), "{bad} was accepted");
        }
        assert!(resolve_output("", &root()).is_err());
        assert!(resolve_output("   ", &root()).is_err());
        assert!(resolve_output("a\0b", &root()).is_err());
    }

    #[test]
    fn overlap_catches_writing_over_or_into_a_model() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model");
        std::fs::create_dir_all(&model).unwrap();
        assert!(overlaps(&model, &model));
        assert!(overlaps(&model.join("fused"), &model));
        assert!(overlaps(dir.path(), &model));
        assert!(!overlaps(&dir.path().join("other"), &model));
        // Through a symlink.
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&model, &link).unwrap();
        assert!(overlaps(&link, &model));
        assert!(overlaps(&link.join("not-made-yet"), &model));
    }

    #[test]
    fn only_existing_ordinary_directories_can_be_scanned() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("f");
        std::fs::write(&file, "x").unwrap();
        assert!(scan_dir(dir.path().to_str().unwrap()).is_ok());
        assert!(
            scan_dir(file.to_str().unwrap())
                .unwrap_err()
                .contains("file")
        );
        assert!(scan_dir("relative/dir").unwrap_err().contains("full path"));
        assert!(
            scan_dir("/definitely/not/here")
                .unwrap_err()
                .contains("does not exist")
        );
        assert!(scan_dir("/").is_err());
        assert!(scan_dir("/System").is_err());
    }
}
