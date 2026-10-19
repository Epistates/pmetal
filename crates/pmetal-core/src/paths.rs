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

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("pmetal_paths_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// `..` is a path like any other, and a directory that does not exist yet
    /// is created: what `--output ../models/out.gguf` means.
    #[test]
    fn parent_references_resolve_and_the_directory_is_created() {
        let root = scratch("parent");
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
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn a_relative_path_is_made_absolute_from_the_current_directory() {
        let resolved = output_path("some-pmetal-test-output-dir/../out.bin", "output").unwrap();
        assert!(resolved.is_absolute());
        assert_eq!(resolved, std::env::current_dir().unwrap().join("out.bin"));
    }

    #[test]
    fn what_cannot_be_made_is_an_error_naming_the_option() {
        let root = scratch("blocked");
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
        std::fs::remove_dir_all(root).ok();
    }
}
