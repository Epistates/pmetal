//! Where a command may write: wherever the person running it can.
//!
//! The command line runs with the user's own permissions, so an output path is
//! theirs to choose (another volume, a parent directory, a path under a
//! mounted drive). What a command owes is to make the path absolute, create the
//! directory it goes in, and say what is wrong when it cannot, before hours of
//! work rather than after them. Servers that take paths from a client (the MCP
//! server) apply their own policy before a path gets here.

use std::path::{Component, Path, PathBuf};

use crate::{PMetalError, Result};

/// The absolute, lexically normalised path `path` names (relative to the
/// current directory; `.` and `..` resolved without touching the disk), with
/// its parent directory created. `what` names the option in errors
/// (`"quantization output"`).
///
/// # Errors
///
/// An empty path, the current directory being unreadable, or a parent
/// directory that cannot be created (a file is in the way, no permission).
pub fn output_path(path: &str, what: &str) -> Result<PathBuf> {
    if path.trim().is_empty() {
        return Err(PMetalError::InvalidArgument(format!(
            "{what}: the path is empty"
        )));
    }
    let given = Path::new(path);
    let absolute = if given.is_absolute() {
        given.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| {
                PMetalError::InvalidArgument(format!("{what}: no current directory: {e}"))
            })?
            .join(given)
    };
    let mut clean = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::ParentDir => {
                clean.pop();
            }
            Component::CurDir => {}
            other => clean.push(other.as_os_str()),
        }
    }
    if let Some(parent) = clean.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| {
            PMetalError::InvalidArgument(format!(
                "{what}: could not create '{}': {e}",
                parent.display()
            ))
        })?;
    }
    Ok(clean)
}

/// Where `path` is, when relative paths are relative to `dir`: an absolute
/// `path` as it is, anything else beneath `dir`.
///
/// The rule for every file a command writes into its output directory
/// (`--log-metrics`): the same answer for the command that writes the file and
/// for whatever follows it.
pub fn resolve_in(dir: &Path, path: &str) -> PathBuf {
    let path = expand_home(path);
    if path.is_absolute() {
        path
    } else {
        dir.join(path)
    }
}

/// `~` and `~/x` spelled out against the home directory; anything else (a
/// `~user` path included) as it was.
pub fn expand_home(text: &str) -> PathBuf {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_relative_path_is_beneath_the_directory_and_an_absolute_one_is_not() {
        let dir = Path::new("/runs/a");
        assert_eq!(
            resolve_in(dir, "metrics.jsonl"),
            PathBuf::from("/runs/a/metrics.jsonl")
        );
        assert_eq!(
            resolve_in(dir, "logs/m.jsonl"),
            PathBuf::from("/runs/a/logs/m.jsonl")
        );
        assert_eq!(
            resolve_in(dir, "/elsewhere/m.jsonl"),
            PathBuf::from("/elsewhere/m.jsonl")
        );
    }

    #[test]
    fn a_leading_tilde_is_the_home_directory() {
        let home = dirs::home_dir().expect("a home directory");
        assert_eq!(expand_home("~"), home);
        assert_eq!(expand_home("~/models/a"), home.join("models/a"));
        // Not the current user's home, and not a tilde at the start.
        assert_eq!(expand_home("~other/x"), PathBuf::from("~other/x"));
        assert_eq!(expand_home("a/~/b"), PathBuf::from("a/~/b"));
        assert_eq!(expand_home("/abs"), PathBuf::from("/abs"));
    }

    /// `..` is a path like any other, and a directory that does not exist yet
    /// is created: what `--output ../models/out.gguf` means.
    #[test]
    fn parent_references_resolve_and_the_directory_is_created() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let given = root
            .join("a")
            .join("..")
            .join("b")
            .join("c")
            .join("out.gguf");
        let resolved = output_path(given.to_str().unwrap(), "output").unwrap();
        assert_eq!(resolved, root.join("b").join("c").join("out.gguf"));
        assert!(root.join("b").join("c").is_dir());
        assert!(
            !root.join("a").exists(),
            "a directory the path only passes over is not made"
        );
    }

    #[test]
    fn a_relative_path_is_made_absolute_from_the_current_directory() {
        let resolved = output_path("some-pmetal-test-output-dir/../out.bin", "output").unwrap();
        assert!(resolved.is_absolute());
        assert_eq!(resolved, std::env::current_dir().unwrap().join("out.bin"));
    }

    #[test]
    fn what_cannot_be_made_is_an_error_naming_the_option() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let file = root.join("a-file");
        std::fs::write(&file, b"x").unwrap();
        let err = output_path(
            file.join("out.gguf").to_str().unwrap(),
            "quantization output",
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("quantization output") && err.contains("could not create"),
            "{err}"
        );
        let err = output_path("  ", "config output").unwrap_err().to_string();
        assert!(
            err.contains("config output") && err.contains("empty"),
            "{err}"
        );
    }
}
