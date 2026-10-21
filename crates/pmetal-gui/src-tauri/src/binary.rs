//! Finding the `pmetal` command-line tool the jobs run.
//!
//! The search is the one every other surface makes (`PMETAL_BIN`, then a
//! `pmetal` beside the running program), plus the usual install locations,
//! because an app opened from Finder gets launchd's `PATH`, which has neither
//! Homebrew's nor Cargo's `bin`.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// The `pmetal` to run, or the reason there is none.
pub fn locate() -> Result<PathBuf, String> {
    locate_with(
        std::env::var_os("PMETAL_BIN"),
        std::env::current_exe().ok().as_deref(),
        std::env::var_os("CARGO_HOME").map(PathBuf::from),
        dirs::home_dir(),
        std::env::var_os("PATH"),
        &[
            PathBuf::from("/opt/homebrew/bin/pmetal"),
            PathBuf::from("/usr/local/bin/pmetal"),
        ],
    )
}

/// [`locate`] with its inputs given, so the search can be tested.
pub fn locate_with(
    explicit: Option<OsString>,
    exe: Option<&Path>,
    cargo_home: Option<PathBuf>,
    home: Option<PathBuf>,
    path_var: Option<OsString>,
    fixed: &[PathBuf],
) -> Result<PathBuf, String> {
    // An explicit choice is never second-guessed: if it is wrong, say so,
    // rather than quietly running a different pmetal.
    if let Some(explicit) = explicit.filter(|v| !v.is_empty()) {
        let path = PathBuf::from(explicit);
        return if path.is_file() {
            Ok(path)
        } else {
            Err(format!(
                "PMETAL_BIN points to {}, which is not a file. Unset it to use the \
                 installed pmetal, or point it at the pmetal executable.",
                path.display()
            ))
        };
    }

    let mut candidates: Vec<PathBuf> = Vec::new();
    // The bundled app carries it beside its own executable.
    if let Some(dir) = exe.and_then(Path::parent) {
        candidates.push(dir.join("pmetal"));
    }
    if let Some(cargo) = cargo_home.or_else(|| home.as_ref().map(|h| h.join(".cargo"))) {
        candidates.push(cargo.join("bin/pmetal"));
    }
    candidates.extend(fixed.iter().cloned());
    if let Some(path_var) = path_var {
        candidates.extend(std::env::split_paths(&path_var).map(|dir| dir.join("pmetal")));
    }

    candidates.into_iter().find(|c| c.is_file()).ok_or_else(|| {
        "The pmetal command-line tool was not found, and every job needs it. \
             Reinstall PMetal, install the tool (brew install pmetal, or cargo install pmetal), \
             or set PMETAL_BIN to its path."
            .to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn executable(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(&path, "#!/bin/sh\n").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[test]
    fn pmetal_bin_wins_over_everything() {
        let dir = tempfile::tempdir().unwrap();
        let chosen = executable(&dir.path().join("chosen"), "mine");
        let beside = executable(&dir.path().join("app"), "pmetal");
        let found = locate_with(
            Some(chosen.clone().into()),
            Some(&dir.path().join("app/gui")),
            None,
            None,
            None,
            &[],
        )
        .unwrap();
        assert_eq!(found, chosen);
        assert!(beside.exists());
    }

    #[test]
    fn a_wrong_pmetal_bin_is_an_error_not_a_quiet_fallback() {
        let dir = tempfile::tempdir().unwrap();
        executable(&dir.path().join("app"), "pmetal");
        let err = locate_with(
            Some("/no/such/pmetal".into()),
            Some(&dir.path().join("app/gui")),
            None,
            None,
            None,
            &[],
        )
        .unwrap_err();
        assert!(
            err.contains("PMETAL_BIN") && err.contains("/no/such/pmetal"),
            "{err}"
        );
    }

    #[test]
    fn the_bundled_sidecar_is_found_before_an_installed_copy() {
        let dir = tempfile::tempdir().unwrap();
        let sidecar = executable(&dir.path().join("Contents/MacOS"), "pmetal");
        let cargo = dir.path().join("cargo");
        executable(&cargo.join("bin"), "pmetal");
        let found = locate_with(
            None,
            Some(&dir.path().join("Contents/MacOS/PMetal")),
            Some(cargo),
            None,
            None,
            &[],
        )
        .unwrap();
        assert_eq!(found, sidecar);
    }

    #[test]
    fn cargo_install_is_found_without_it_being_on_path() {
        let dir = tempfile::tempdir().unwrap();
        let installed = executable(&dir.path().join(".cargo/bin"), "pmetal");
        let found =
            locate_with(None, None, None, Some(dir.path().to_path_buf()), None, &[]).unwrap();
        assert_eq!(found, installed);
    }

    #[test]
    fn path_is_searched_last() {
        let dir = tempfile::tempdir().unwrap();
        let on_path = executable(&dir.path().join("elsewhere"), "pmetal");
        let found = locate_with(
            None,
            None,
            None,
            None,
            Some(std::env::join_paths([dir.path().join("elsewhere")]).unwrap()),
            &[],
        )
        .unwrap();
        assert_eq!(found, on_path);
    }

    #[test]
    fn nothing_found_says_what_to_do() {
        let dir = tempfile::tempdir().unwrap();
        let err = locate_with(
            None,
            Some(&dir.path().join("x/gui")),
            Some(dir.path().join("cargo")),
            Some(dir.path().to_path_buf()),
            Some(OsString::new()),
            &[],
        )
        .unwrap_err();
        assert!(err.contains("PMETAL_BIN"), "{err}");
    }
}
