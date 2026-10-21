//! Where a tool may read and write, and what it may name as a model.
//!
//! An MCP client hands the server strings that end up as paths on the command
//! line of a process running with the user's privileges. Everything a tool
//! forwards goes through [`PathPolicy`] first: a path is made absolute, its `.`
//! and `..` are resolved, symlinks on the part that exists are followed, and the
//! result must lie under one of the allowed roots. A model or dataset that is
//! not a path must be a well-formed Hub identifier.
//!
//! The roots are the server's working directory (unless that is `/`, which a
//! GUI launcher gives and which allows nothing in particular), the system
//! temporary directory, the model cache, and whatever `PMETAL_MCP_ROOTS` adds
//! (a `:`-separated list).

use std::path::{Component, Path, PathBuf};

/// The environment variable that adds allowed roots.
pub const ROOTS_ENV: &str = "PMETAL_MCP_ROOTS";

/// The set of directories tools may touch.
#[derive(Debug, Clone)]
pub struct PathPolicy {
    roots: Vec<PathBuf>,
    cwd: PathBuf,
}

impl PathPolicy {
    /// The policy of this process: its working directory, the temporary
    /// directory, the model cache, and `PMETAL_MCP_ROOTS`.
    pub fn from_env() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        let mut roots = Vec::new();
        if cwd != Path::new("/") {
            roots.push(cwd.clone());
        }
        roots.push(std::env::temp_dir());
        roots.push(pmetal_hub::cache_dir());
        if let Ok(extra) = std::env::var(ROOTS_ENV) {
            roots.extend(
                extra
                    .split(':')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|s| expand_home(s).unwrap_or_else(|| PathBuf::from(s))),
            );
        }
        Self::new(cwd, roots)
    }

    /// A policy with exactly these roots, relative paths resolved against `cwd`.
    pub fn new(cwd: PathBuf, roots: Vec<PathBuf>) -> Self {
        let roots = roots.into_iter().map(|r| resolve(&r, &cwd)).collect();
        Self { roots, cwd }
    }

    /// The roots, resolved.
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// This policy with `more` roots allowed too.
    #[must_use]
    pub fn with_extra_roots(&self, more: impl IntoIterator<Item = PathBuf>) -> Self {
        let mut roots = self.roots.clone();
        roots.extend(more.into_iter().map(|r| resolve(&r, &self.cwd)));
        Self {
            roots,
            cwd: self.cwd.clone(),
        }
    }

    /// The absolute, normalized form of `raw`, if it lies under an allowed root.
    ///
    /// `field` names the parameter for the error.
    pub fn check_path(&self, field: &str, raw: &str) -> Result<PathBuf, String> {
        if raw.is_empty() {
            return Err(format!("{field}: the path is empty"));
        }
        if raw.contains('\0') {
            return Err(format!("{field}: the path contains a NUL byte"));
        }
        let home = expand_home(raw);
        let resolved = resolve(home.as_deref().unwrap_or(Path::new(raw)), &self.cwd);
        if self.roots.iter().any(|root| resolved.starts_with(root)) {
            Ok(resolved)
        } else {
            Err(format!(
                "{field}: {} is outside the directories this server may use ({}). \
                 Set {ROOTS_ENV} to a `:`-separated list of directories to allow more",
                resolved.display(),
                self.roots
                    .iter()
                    .map(|r| r.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        }
    }

    /// A model or dataset reference: a Hub ID kept as it is, or a path that is
    /// checked and made absolute.
    pub fn check_reference(&self, field: &str, raw: &str) -> Result<String, String> {
        if raw.is_empty() {
            return Err(format!("{field}: the reference is empty"));
        }
        if raw.contains('\0') {
            return Err(format!("{field}: the reference contains a NUL byte"));
        }
        if looks_like_path(raw) || self.cwd.join(raw).exists() {
            return self.check_path(field, raw).map(|p| p.display().to_string());
        }
        if is_hub_id(raw) {
            Ok(raw.to_string())
        } else {
            Err(format!(
                "{field}: `{raw}` is neither a Hugging Face ID (`owner/name`) nor a path that exists"
            ))
        }
    }
}

/// Whether `raw` is spelled as a path rather than a Hub ID.
fn looks_like_path(raw: &str) -> bool {
    raw.starts_with('/')
        || raw.starts_with("./")
        || raw.starts_with("../")
        || raw.starts_with('~')
        || raw == "."
        || raw == ".."
}

/// `owner/name`, with an optional `:QUANT` selector for one quantization of a
/// GGUF repository (`org/model-GGUF:Q4_K_M`), of the characters the Hub allows.
/// `.` and `..` are not names. This is what the `pmetal` commands take as a Hub
/// model: a string without the owner is read as a path.
fn is_hub_id(raw: &str) -> bool {
    let (id, quant) = pmetal_hub::split_quant_selector(raw);
    let valid = |part: &str| {
        !part.is_empty()
            && part != "."
            && part != ".."
            && !part.starts_with('-')
            && part
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    };
    if raw.contains(':') && quant.is_none() {
        return false;
    }
    if quant.is_some_and(|q| {
        !q.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
    }) {
        return false;
    }
    let mut parts = id.split('/');
    matches!(
        (parts.next(), parts.next(), parts.next()),
        (Some(owner), Some(name), None) if valid(owner) && valid(name)
    )
}

/// The directory a client's root names: a `file://` URI (as MCP roots are), on
/// this machine, with its percent-escapes decoded. `None` for anything else.
pub fn root_uri_to_path(uri: &str) -> Option<PathBuf> {
    let rest = uri.strip_prefix("file://")?;
    // `file:///path` has an empty host; `file://localhost/path` names this one.
    let path = match rest.strip_prefix("localhost") {
        Some(path) => path,
        None => rest,
    };
    if !path.starts_with('/') {
        return None;
    }
    let bytes = path.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = path.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    let decoded = String::from_utf8(out).ok()?;
    (!decoded.contains('\0')).then(|| PathBuf::from(decoded))
}

fn expand_home(raw: &str) -> Option<PathBuf> {
    let rest = raw.strip_prefix('~')?;
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    match rest {
        "" => Some(home),
        _ => rest.strip_prefix('/').map(|r| home.join(r)),
    }
}

/// `path` made absolute against `cwd`, with `.` and `..` removed and symlinks
/// followed as far as the path exists.
fn resolve(path: &Path, cwd: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };
    let mut lexical = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::ParentDir => {
                lexical.pop();
            }
            Component::CurDir => {}
            other => lexical.push(other.as_os_str()),
        }
    }
    // Follow symlinks on the deepest part that exists, then put the rest back:
    // a link inside an allowed root that points out of it must not pass.
    let mut existing = lexical.as_path();
    let mut tail: Vec<&std::ffi::OsStr> = Vec::new();
    loop {
        if let Ok(real) = existing.canonicalize() {
            let mut out = real;
            out.extend(tail.iter().rev());
            return out;
        }
        match (existing.parent(), existing.file_name()) {
            (Some(parent), Some(name)) => {
                tail.push(name);
                existing = parent;
            }
            _ => return lexical,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh directory and its canonical path; the directory is removed when
    /// the guard drops, so bind the guard for as long as the path is used.
    fn scratch(name: &str) -> (tempfile::TempDir, PathBuf) {
        let guard = tempfile::Builder::new()
            .prefix(&format!("pmetal-mcp-policy-{name}-"))
            .tempdir()
            .unwrap();
        let path = guard.path().canonicalize().unwrap();
        (guard, path)
    }

    #[test]
    fn a_path_under_a_root_is_made_absolute() {
        let (_guard, root) = scratch("under");
        let policy = PathPolicy::new(root.clone(), vec![root.clone()]);
        assert_eq!(
            policy.check_path("output", "out/run1").unwrap(),
            root.join("out/run1")
        );
        assert_eq!(
            policy.check_path("output", "./a/../b").unwrap(),
            root.join("b")
        );
        assert_eq!(
            policy
                .check_path("output", root.join("x").to_str().unwrap())
                .unwrap(),
            root.join("x")
        );
    }

    #[test]
    fn traversal_out_of_a_root_is_refused() {
        let (_guard, root) = scratch("traversal");
        let policy = PathPolicy::new(root.clone(), vec![root.clone()]);
        for bad in [
            "../escape",
            "a/../../escape",
            "/etc/passwd",
            "/",
            "~/.ssh/id_rsa",
        ] {
            let err = policy.check_path("output", bad).unwrap_err();
            assert!(err.contains("outside the directories"), "{bad}: {err}");
            assert!(err.contains(ROOTS_ENV), "{err}");
        }
        assert!(policy.check_path("output", "a\0b").is_err());
        assert!(policy.check_path("output", "").is_err());
    }

    #[test]
    fn a_sibling_with_the_root_as_a_name_prefix_is_outside() {
        let (_guard, parent) = scratch("prefix");
        let root = parent.join("data");
        std::fs::create_dir_all(&root).unwrap();
        let policy = PathPolicy::new(root.clone(), vec![root.clone()]);
        assert!(
            policy
                .check_path("p", parent.join("data-evil/x").to_str().unwrap())
                .is_err()
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_out_of_a_root_is_followed_and_refused() {
        let (_root_guard, root) = scratch("symlink-root");
        let (_outside_guard, outside) = scratch("symlink-outside");
        std::os::unix::fs::symlink(&outside, root.join("link")).unwrap();
        let policy = PathPolicy::new(root.clone(), vec![root.clone()]);
        // Both the link itself and a path that does not exist yet beneath it.
        assert!(policy.check_path("p", "link").is_err());
        assert!(policy.check_path("p", "link/new/file").is_err());
        assert!(policy.check_path("p", "real/file").is_ok());
    }

    #[test]
    fn hub_ids_pass_and_malformed_ones_do_not() {
        let (_guard, root) = scratch("hub");
        let policy = PathPolicy::new(root.clone(), vec![root]);
        for good in [
            "Qwen/Qwen3-0.6B",
            "meta-llama/Llama-3.2-1B",
            "org/Model-GGUF:Q4_K_M",
        ] {
            assert_eq!(policy.check_reference("model", good).unwrap(), good);
        }
        // `gpt2` has no owner, so the commands read it as a path; none exists.
        for bad in [
            "gpt2",
            "a/b/c",
            "owner/..",
            "../x/y",
            "owner/na me",
            "a//b",
            "owner/",
            "x;rm -rf",
            "owner/name:Q4/../x",
            "owner/name:",
            "-owner/name",
        ] {
            assert!(policy.check_reference("model", bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_local_model_directory_is_checked_as_a_path() {
        let (_root_guard, root) = scratch("local-model");
        std::fs::create_dir_all(root.join("models/qwen")).unwrap();
        let (_outside_guard, outside) = scratch("local-model-outside");
        let policy = PathPolicy::new(root.clone(), vec![root.clone()]);
        // A relative path that exists is a path even though it looks like owner/name.
        assert_eq!(
            policy.check_reference("model", "models/qwen").unwrap(),
            root.join("models/qwen").display().to_string()
        );
        assert!(
            policy
                .check_reference("model", outside.to_str().unwrap())
                .is_err()
        );
    }

    #[test]
    fn a_roots_uri_names_a_local_directory() {
        assert_eq!(
            root_uri_to_path("file:///srv/me/proj"),
            Some(PathBuf::from("/srv/me/proj"))
        );
        assert_eq!(
            root_uri_to_path("file://localhost/tmp/a"),
            Some(PathBuf::from("/tmp/a"))
        );
        assert_eq!(
            root_uri_to_path("file:///srv/me/My%20Models/q%C3%A9"),
            Some(PathBuf::from("/srv/me/My Models/qé"))
        );
        for bad in [
            "https://example.com/x",
            "file://otherhost/tmp",
            "file://relative/path",
            "/srv/me",
            "file:///a%zz",
            "file:///a%00b",
            "file:///a%",
        ] {
            assert_eq!(root_uri_to_path(bad), None, "{bad}");
        }
    }

    #[test]
    fn extra_roots_widen_the_policy() {
        let (_root_guard, root) = scratch("extra-a");
        let (_other_guard, other) = scratch("extra-b");
        let policy = PathPolicy::new(root.clone(), vec![root]);
        let target = other.join("run");
        assert!(policy.check_path("p", target.to_str().unwrap()).is_err());
        let wider = policy.with_extra_roots([other]);
        assert!(wider.check_path("p", target.to_str().unwrap()).is_ok());
    }
}
