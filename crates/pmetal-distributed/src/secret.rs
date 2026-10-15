//! The cluster secret: the one credential that admits a machine to a cluster.
//!
//! Every connection between ranks is authenticated and encrypted with keys
//! derived from this secret (see [`crate::transport`]). It is 256 random bits,
//! generated once and copied to every machine in the cluster, the way an SSH
//! key is. It is deliberately **not** derived from a human-chosen string: the
//! transport handshake would let anyone who recorded it guess a weak secret
//! offline, so only full-entropy secrets are accepted.
//!
//! A machine finds its secret, in order, in
//!
//! 1. the [`PMETAL_CLUSTER_SECRET`](ENV_VAR) environment variable (64 hex
//!    characters),
//! 2. the file `~/.pmetal/cluster_secret` (the same 64 characters; the file
//!    must be readable by its owner only).
//!
//! Create one with [`ClusterSecret::generate`] (the CLI wraps it as
//! `pmetal cluster keygen`). The secret is wiped from memory when dropped and
//! never appears in `Debug` output or logs.

use crate::error::DistributedError;
use secrecy::{ExposeSecret, SecretBox};
use sha3::{Digest, Sha3_256};
use std::path::{Path, PathBuf};

/// Environment variable holding the secret as 64 hex characters.
pub const ENV_VAR: &str = "PMETAL_CLUSTER_SECRET";

/// File (under the pmetal data directory) holding the secret.
const SECRET_FILE: &str = "cluster_secret";
const PMETAL_DIR: &str = ".pmetal";

/// Length of the secret in bytes.
pub const SECRET_BYTES: usize = 32;

/// Domain separation for the key the ring handshake derives from the secret.
const RING_PSK_DOMAIN: &[u8] = b"pmetal-distributed/ring-psk/v1";
/// Domain separation for the proof of membership advertised to peers.
const ATTEST_DOMAIN: &[u8] = b"pmetal-distributed/peer-attestation/v1";

/// A 256-bit pre-shared cluster secret.
pub struct ClusterSecret(SecretBox<[u8; SECRET_BYTES]>);

impl ClusterSecret {
    /// Generate a fresh secret from the operating system's CSPRNG.
    pub fn generate() -> Self {
        let mut bytes = [0u8; SECRET_BYTES];
        rand::fill(&mut bytes);
        Self::from_bytes(bytes)
    }

    /// Wrap raw secret bytes.
    pub fn from_bytes(bytes: [u8; SECRET_BYTES]) -> Self {
        Self(SecretBox::new(Box::new(bytes)))
    }

    /// Parse the 64-hex-character form. Surrounding whitespace is ignored.
    pub fn from_hex(text: &str) -> Result<Self, DistributedError> {
        let text = text.trim();
        let bad = || {
            DistributedError::Config(format!(
                "a cluster secret is {} hex characters (256 random bits); \
                 create one with `pmetal cluster keygen`",
                SECRET_BYTES * 2
            ))
        };
        if text.len() != SECRET_BYTES * 2 || !text.is_ascii() {
            return Err(bad());
        }
        let mut bytes = [0u8; SECRET_BYTES];
        for (slot, pair) in bytes.iter_mut().zip(text.as_bytes().as_chunks::<2>().0) {
            let digits = std::str::from_utf8(pair).map_err(|_| bad())?;
            *slot = u8::from_str_radix(digits, 16).map_err(|_| bad())?;
        }
        Ok(Self::from_bytes(bytes))
    }

    /// The 64-hex-character form, for storing or copying to another machine.
    ///
    /// The returned string is the secret itself: do not log it.
    pub fn to_hex(&self) -> String {
        use std::fmt::Write as _;
        self.0
            .expose_secret()
            .iter()
            .fold(String::with_capacity(SECRET_BYTES * 2), |mut out, b| {
                let _ = write!(out, "{b:02x}");
                out
            })
    }

    /// Where the secret is kept by default: `~/.pmetal/cluster_secret`.
    pub fn default_path() -> Result<PathBuf, DistributedError> {
        let home = dirs::home_dir()
            .ok_or_else(|| DistributedError::Config("Cannot determine home directory".into()))?;
        Ok(home.join(PMETAL_DIR).join(SECRET_FILE))
    }

    /// Read a secret from `path`, refusing a file other users can read.
    pub fn load(path: &Path) -> Result<Self, DistributedError> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(path)
                .map_err(|e| {
                    DistributedError::Config(format!(
                        "cannot read cluster secret {}: {e}",
                        path.display()
                    ))
                })?
                .permissions()
                .mode();
            if mode & 0o077 != 0 {
                return Err(DistributedError::Config(format!(
                    "cluster secret {} is accessible to other users (mode {:o}); \
                     run `chmod 600` on it",
                    path.display(),
                    mode & 0o777
                )));
            }
        }
        let text = std::fs::read_to_string(path).map_err(|e| {
            DistributedError::Config(format!(
                "cannot read cluster secret {}: {e}",
                path.display()
            ))
        })?;
        Self::from_hex(&text)
    }

    /// Write the secret to `path`, owner-only, failing if the file exists.
    pub fn save_new(&self, path: &Path) -> Result<(), DistributedError> {
        use std::io::Write as _;
        let io = |what: &str, e: std::io::Error| {
            DistributedError::Config(format!("{what} {}: {e}", path.display()))
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| io("cannot create directory for", e))?;
        }
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options
            .open(path)
            .map_err(|e| io("cannot create cluster secret", e))?;
        writeln!(file, "{}", self.to_hex()).map_err(|e| io("cannot write cluster secret", e))
    }

    /// Find this machine's secret: the environment, then the default file.
    pub fn resolve() -> Result<Self, DistributedError> {
        if let Ok(value) = std::env::var(ENV_VAR) {
            return Self::from_hex(&value).map_err(|e| {
                DistributedError::Config(format!("{ENV_VAR} is set but unusable: {e}"))
            });
        }
        let path = Self::default_path()?;
        if path.exists() {
            return Self::load(&path);
        }
        Err(DistributedError::Config(format!(
            "no cluster secret: set {ENV_VAR} or create {} \
             (`pmetal cluster keygen`, then copy it to every machine in the cluster)",
            path.display()
        )))
    }

    /// The 32-byte pre-shared key the ring handshake mixes in.
    pub(crate) fn ring_psk(&self) -> [u8; 32] {
        self.derive(RING_PSK_DOMAIN, &[])
    }

    /// A short proof that the holder of this secret vouches for `subject`
    /// (a peer id): a keyed SHA3-256, which unlike SHA-2 needs no HMAC
    /// construction to be a sound MAC.
    pub(crate) fn attest(&self, subject: &[u8]) -> [u8; 16] {
        let full = self.derive(ATTEST_DOMAIN, subject);
        let mut out = [0u8; 16];
        out.copy_from_slice(&full[..16]);
        out
    }

    fn derive(&self, domain: &[u8], data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update((domain.len() as u64).to_le_bytes());
        hasher.update(domain);
        hasher.update(self.0.expose_secret());
        hasher.update(data);
        hasher.finalize().into()
    }

    /// Constant-time equality.
    pub fn matches(&self, other: &Self) -> bool {
        let (a, b) = (self.0.expose_secret(), other.0.expose_secret());
        a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
    }
}

impl Clone for ClusterSecret {
    fn clone(&self) -> Self {
        Self::from_bytes(*self.0.expose_secret())
    }
}

impl std::fmt::Debug for ClusterSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ClusterSecret(..)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips_and_rejects_anything_else() {
        let secret = ClusterSecret::generate();
        let again = ClusterSecret::from_hex(&format!("  {}\n", secret.to_hex())).unwrap();
        assert!(secret.matches(&again));

        for bad in ["", "abc", &"g".repeat(64), &"a".repeat(63), &"a".repeat(66)] {
            assert!(ClusterSecret::from_hex(bad).is_err(), "{bad:?} accepted");
        }
        // Multi-byte characters must not slice mid-character.
        assert!(ClusterSecret::from_hex(&"é".repeat(32)).is_err());
    }

    #[test]
    fn generated_secrets_differ() {
        assert!(!ClusterSecret::generate().matches(&ClusterSecret::generate()));
    }

    #[test]
    fn debug_output_never_contains_the_secret() {
        let secret = ClusterSecret::generate();
        let shown = format!("{secret:?}");
        assert!(!shown.contains(&secret.to_hex()[..8]));
    }

    #[test]
    fn derived_values_are_domain_separated() {
        let secret = ClusterSecret::from_bytes([7; 32]);
        assert_ne!(secret.ring_psk()[..16], secret.attest(b"")[..]);
        assert_ne!(secret.attest(b"a"), secret.attest(b"b"));
        let other = ClusterSecret::from_bytes([8; 32]);
        assert_ne!(secret.ring_psk(), other.ring_psk());
    }

    #[cfg(unix)]
    #[test]
    fn a_secret_file_is_owner_only_and_a_loose_one_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("cluster_secret");
        let secret = ClusterSecret::generate();
        secret.save_new(&path).unwrap();
        assert!(secret.save_new(&path).is_err(), "must not overwrite");
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert!(secret.matches(&ClusterSecret::load(&path).unwrap()));

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let err = ClusterSecret::load(&path).unwrap_err().to_string();
        assert!(err.contains("accessible to other users"), "{err}");
    }
}
