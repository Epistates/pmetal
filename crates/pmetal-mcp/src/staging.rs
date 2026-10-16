//! Short-lived files handed to a subprocess.

use std::io::Write;
use std::path::{Path, PathBuf};

/// A file in the temporary directory that only this user can read, removed when
/// dropped. The name is random and the file is created exclusively, so it
/// cannot be a link someone else put there.
pub struct Staged {
    path: PathBuf,
}

impl Staged {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Staged {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Write `contents` to a new private file whose name starts with `prefix`.
pub fn stage_private_file(prefix: &str, contents: &[u8]) -> std::io::Result<Staged> {
    let path = std::env::temp_dir().join(format!(
        "pmetal-mcp-{prefix}-{}.json",
        uuid::Uuid::new_v4().simple()
    ));
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&path)?;
    let staged = Staged { path };
    file.write_all(contents)?;
    file.flush()?;
    Ok(staged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_is_private_unique_and_removed_on_drop() {
        let a = stage_private_file("t", b"one").unwrap();
        let b = stage_private_file("t", b"two").unwrap();
        assert_ne!(a.path(), b.path());
        assert_eq!(std::fs::read(a.path()).unwrap(), b"one");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(a.path()).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let path = a.path().to_path_buf();
        drop(a);
        assert!(!path.exists());
        drop(b);
    }
}
