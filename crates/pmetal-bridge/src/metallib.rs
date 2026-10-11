//! Which `mlx.metallib` MLX loads, and a cache of it keyed by version.
//!
//! MLX's Metal kernels live in `mlx.metallib`, built with libmlx, and a
//! metallib from another MLX revision can't be trusted to hold the kernels
//! this one asks for. Two things decide which file MLX loads:
//!
//! - **At load, the bridge itself.** It points MLX at
//!   `PMETAL_METALLIB_PATH` if set, else at the metallib its own build
//!   produced ([`BUILT_PATH`], in the MLX prefix slot or the build tree) while
//!   that file exists. Every binary linking the bridge, test binaries
//!   included, gets the matching file without doing anything. MLX's own
//!   fallback, a path compiled into libmlx, names the build tree that first
//!   built a prefix slot and is gone once that tree is.
//! - **Applications that outlive their build tree** (the CLI, the GUI) carry a
//!   copy, [`locate`] it before their first GPU op, and hand it to
//!   [`set_metallib_path`](crate::inline_array::set_metallib_path).
//!
//! The [`Cache`] (`~/.cache/pmetal/lib`) holds one file per MLX revision and
//! content hash, `mlx-<revision>-<sha256>.metallib`, so pmetal binaries built
//! against different MLX revisions keep their own copies side by side. Files
//! are written whole or not at all and checked against their hash first.

use std::io::{self, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::inline_array::validate_metallib;

/// The MLX revision (tag or commit) this bridge builds and links.
pub const MLX_REVISION: &str = env!("PMETAL_BRIDGE_MLX_GIT_TAG");

/// The `mlx.metallib` built with the MLX this bridge links, as the build left
/// it. `None` without the `metal` feature.
pub const BUILT_PATH: Option<&str> = option_env!("PMETAL_BRIDGE_MLX_METALLIB");

/// SHA-256 (lower-case hex) of [`BUILT_PATH`] when it was built.
pub const BUILT_SHA256: Option<&str> = option_env!("PMETAL_BRIDGE_MLX_METALLIB_SHA256");

/// The variable that names a metallib to use ahead of every other location.
pub const OVERRIDE_VAR: &str = "PMETAL_METALLIB_PATH";

/// SHA-256 of `bytes`, lower-case hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    hex(&sha2::Sha256::digest(bytes))
}

/// SHA-256 of the file at `path`, lower-case hex.
pub fn sha256_file(path: &Path) -> io::Result<String> {
    use sha2::Digest as _;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// How long a cached metallib nothing has used stays. Each pmetal binary
/// marks its own entry as used whenever it starts, so this only removes the
/// copies of versions no longer run; one that runs again extracts its copy
/// again.
const STALE_AFTER: Duration = Duration::from_secs(30 * 24 * 60 * 60);

/// How long a partly written file from an interrupted install stays.
const ABANDONED_AFTER: Duration = Duration::from_secs(24 * 60 * 60);

/// A directory of metallibs keyed by MLX revision and content hash.
#[derive(Debug, Clone)]
pub struct Cache {
    dir: PathBuf,
}

impl Cache {
    /// `~/.cache/pmetal/lib`, the cache every pmetal binary shares.
    pub fn user() -> Option<Self> {
        let home = std::env::var_os("HOME").filter(|h| !h.is_empty())?;
        Some(Self::at(PathBuf::from(home).join(".cache/pmetal/lib")))
    }

    /// A cache in `dir`.
    pub fn at(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The cache directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Where the metallib with this SHA-256, built with [`MLX_REVISION`],
    /// lives in the cache.
    pub fn entry(&self, sha256: &str) -> PathBuf {
        let revision: String = MLX_REVISION.chars().take(12).collect();
        let hash: String = sha256.chars().take(16).collect();
        self.dir.join(format!("mlx-{revision}-{hash}.metallib"))
    }

    /// The cached metallib with this SHA-256, if it is there and whole.
    ///
    /// A damaged entry (truncated, or not a metallib) is removed, so the
    /// caller installs it again. Using an entry marks it as used, which keeps
    /// [`Cache::prune`] off it.
    pub fn get(&self, sha256: &str) -> Option<PathBuf> {
        let path = self.entry(sha256);
        match validate_metallib(&path) {
            Ok(()) => {
                let _ = std::fs::File::options()
                    .write(true)
                    .open(&path)
                    .and_then(|f| f.set_modified(SystemTime::now()));
                Some(path)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(_) => {
                let _ = std::fs::remove_file(&path);
                None
            }
        }
    }

    /// Write `bytes` into the cache as the metallib with this SHA-256 and
    /// return its path.
    ///
    /// `bytes` must hash to `sha256` and look like a whole Metal library. They
    /// are written to a temporary file in the cache directory, flushed to disk
    /// and renamed into place, so a reader sees the whole file or none, and
    /// two processes installing at once each rename a whole copy. Entries no
    /// binary has used for 30 days are removed afterwards.
    pub fn install(&self, bytes: &[u8], sha256: &str) -> io::Result<PathBuf> {
        let actual = sha256_hex(bytes);
        if actual != sha256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("its SHA-256 is {actual}, not the expected {sha256}"),
            ));
        }
        std::fs::create_dir_all(&self.dir)?;
        let dest = self.entry(sha256);
        let mut tmp = tempfile::Builder::new()
            .prefix(".mlx-")
            .suffix(".tmp")
            .tempfile_in(&self.dir)?;
        tmp.write_all(bytes)?;
        tmp.as_file().sync_all()?;
        validate_metallib(tmp.path())?;
        tmp.persist(&dest).map_err(|e| e.error)?;
        let _ = self.prune(&dest);
        Ok(dest)
    }

    /// Remove metallibs no binary has used for 30 days, other than `keep`,
    /// and partly written files from interrupted installs. The unversioned
    /// `mlx.metallib` that releases before this cache wrote counts as a
    /// metallib.
    pub fn prune(&self, keep: &Path) -> io::Result<()> {
        let now = SystemTime::now();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if path == keep {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let limit = if name.starts_with(".mlx-") && name.ends_with(".tmp") {
                ABANDONED_AFTER
            } else if (name.starts_with("mlx-") && name.ends_with(".metallib"))
                || name == "mlx.metallib"
            {
                STALE_AFTER
            } else {
                continue;
            };
            let age = entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|modified| now.duration_since(modified).ok());
            if age.is_some_and(|age| age > limit) {
                let _ = std::fs::remove_file(&path);
            }
        }
        Ok(())
    }
}

/// Where [`locate`] found a metallib.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// [`OVERRIDE_VAR`].
    Override,
    /// The [`Cache`] entry for the expected hash.
    Cache,
    /// [`BUILT_PATH`], still where the build left it.
    Build,
    /// One of the caller's candidates.
    Candidate,
}

/// A metallib [`locate`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Located {
    pub path: PathBuf,
    pub source: Source,
}

/// Find a whole metallib for the MLX this bridge links.
///
/// In order: [`OVERRIDE_VAR`]; the [`Cache::user`] entry for `sha256`, the
/// hash of the copy the caller carries; [`BUILT_PATH`]; then `candidates`,
/// locations the caller ships its copy in (next to the executable, an app
/// bundle's resources). Every file is checked with [`validate_metallib`];
/// `on_reject` hears about each one that exists but fails, and the search
/// moves on.
///
/// Locations that hold some other MLX's metallib, such as a package manager's
/// `lib/mlx.metallib`, are deliberately not searched.
pub fn locate(
    sha256: Option<&str>,
    candidates: &[PathBuf],
    mut on_reject: impl FnMut(&Path, &io::Error),
) -> Option<Located> {
    // An override that names nothing is a mistake worth hearing about too.
    if let Some(path) = std::env::var_os(OVERRIDE_VAR).filter(|p| !p.is_empty()) {
        let path = PathBuf::from(path);
        match validate_metallib(&path) {
            Ok(()) => {
                return Some(Located {
                    path,
                    source: Source::Override,
                });
            }
            Err(e) => on_reject(&path, &e),
        }
    }

    let mut usable = |path: &Path| match validate_metallib(path) {
        Ok(()) => true,
        Err(e) if e.kind() == io::ErrorKind::NotFound => false,
        Err(e) => {
            on_reject(path, &e);
            false
        }
    };
    if let Some(path) = sha256.and_then(|sha| Cache::user()?.get(sha)) {
        return Some(Located {
            path,
            source: Source::Cache,
        });
    }
    if let Some(path) = BUILT_PATH.map(PathBuf::from).filter(|p| usable(p)) {
        return Some(Located {
            path,
            source: Source::Build,
        });
    }
    candidates.iter().find(|p| usable(p)).map(|path| Located {
        path: path.clone(),
        source: Source::Candidate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal file `validate_metallib` accepts: the `MTLB` magic and the
    /// total size at offset 16.
    fn fake_metallib(len: usize, fill: u8) -> Vec<u8> {
        let mut bytes = vec![fill; len];
        bytes[..4].copy_from_slice(b"MTLB");
        bytes[16..24].copy_from_slice(&(len as u64).to_le_bytes());
        bytes
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "pmetal-metallib-cache-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// The regression this module exists for: MLX must be pointed at the
    /// metallib this build produced, not left to the path compiled into
    /// libmlx, which names whichever build tree first populated the MLX
    /// prefix slot and stops existing with it.
    #[test]
    fn mlx_is_pointed_at_the_built_metallib() {
        if std::env::var_os(OVERRIDE_VAR).is_some() {
            return;
        }
        let Some(built) = BUILT_PATH else {
            return; // no metal feature, no metallib
        };
        if !Path::new(built).exists() {
            return; // a test binary that outlived its build
        }
        assert_eq!(
            crate::inline_array::metallib_path().as_deref(),
            Some(Path::new(built))
        );
        let sha = BUILT_SHA256.expect("a built metallib has a hash");
        assert_eq!(sha256_file(Path::new(built)).unwrap(), sha);
    }

    #[test]
    fn versions_are_cached_side_by_side() {
        let cache = Cache::at(scratch("side-by-side"));
        let a = fake_metallib(4096, 1);
        let b = fake_metallib(8192, 2);
        let (sha_a, sha_b) = (sha256_hex(&a), sha256_hex(&b));

        let path_a = cache.install(&a, &sha_a).unwrap();
        let path_b = cache.install(&b, &sha_b).unwrap();
        assert_ne!(path_a, path_b);
        assert_eq!(std::fs::read(&path_a).unwrap(), a);
        assert_eq!(std::fs::read(&path_b).unwrap(), b);
        assert_eq!(cache.get(&sha_a), Some(path_a.clone()));
        assert_eq!(cache.get(&sha_b), Some(path_b));
        let name = path_a.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with(&format!("mlx-{}-", &MLX_REVISION[..12])));
        // No temporary file is left behind.
        let leftovers = std::fs::read_dir(cache.dir())
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .ends_with(".tmp")
            })
            .count();
        assert_eq!(leftovers, 0);
        std::fs::remove_dir_all(cache.dir()).unwrap();
    }

    #[test]
    fn install_rejects_bytes_that_are_not_the_expected_file() {
        let cache = Cache::at(scratch("reject"));
        let good = fake_metallib(4096, 3);
        let sha = sha256_hex(&good);

        let mut tampered = good.clone();
        tampered[100] ^= 1;
        assert!(cache.install(&tampered, &sha).is_err());
        assert!(!cache.entry(&sha).exists());

        let truncated = &good[..2048];
        assert!(cache.install(truncated, &sha256_hex(truncated)).is_err());
        let _ = std::fs::remove_dir_all(cache.dir());
    }

    #[test]
    fn a_damaged_entry_is_dropped_so_it_is_installed_again() {
        let cache = Cache::at(scratch("damaged"));
        let good = fake_metallib(4096, 4);
        let sha = sha256_hex(&good);
        let path = cache.install(&good, &sha).unwrap();
        std::fs::write(&path, &good[..1000]).unwrap();

        assert_eq!(cache.get(&sha), None);
        assert!(!path.exists());
        assert_eq!(cache.install(&good, &sha).unwrap(), path);
        std::fs::remove_dir_all(cache.dir()).unwrap();
    }

    #[test]
    fn prune_drops_what_has_gone_unused() {
        let cache = Cache::at(scratch("prune"));
        let kept = cache
            .install(
                &fake_metallib(4096, 5),
                &sha256_hex(&fake_metallib(4096, 5)),
            )
            .unwrap();
        let recent = cache
            .install(
                &fake_metallib(4096, 6),
                &sha256_hex(&fake_metallib(4096, 6)),
            )
            .unwrap();
        let stale = cache
            .dir()
            .join("mlx-0123456789ab-0123456789abcdef.metallib");
        let legacy = cache.dir().join("mlx.metallib");
        let abandoned = cache.dir().join(".mlx-abc.tmp");
        let unrelated = cache.dir().join("notes.txt");
        let long_ago = SystemTime::now() - STALE_AFTER - Duration::from_secs(60);
        for path in [&stale, &legacy, &abandoned, &unrelated] {
            std::fs::write(path, b"x").unwrap();
            std::fs::File::options()
                .write(true)
                .open(path)
                .unwrap()
                .set_modified(long_ago)
                .unwrap();
        }
        // The one in use is kept however old.
        std::fs::File::options()
            .write(true)
            .open(&kept)
            .unwrap()
            .set_modified(long_ago)
            .unwrap();

        cache.prune(&kept).unwrap();
        assert!(kept.exists());
        assert!(recent.exists());
        assert!(!stale.exists());
        assert!(!legacy.exists());
        assert!(!abandoned.exists());
        assert!(unrelated.exists());
        std::fs::remove_dir_all(cache.dir()).unwrap();
    }
}
