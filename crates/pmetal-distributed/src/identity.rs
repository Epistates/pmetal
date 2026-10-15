//! Persistent node identity management.
//!
//! Each node in the cluster has a unique, persistent identity based on an
//! Ed25519 keypair. The keypair is stored at `~/.pmetal/node_keypair` and
//! loaded on startup (or generated if not present).
//!
//! This design ensures:
//! - Consistent node identification across restarts
//! - Cryptographic verification of peer identity
//! - Compatibility with libp2p's PeerId system

use crate::error::DistributedError;
use anyhow::Result;
use libp2p::PeerId;
use libp2p::identity::{Keypair, ed25519};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use tracing::{debug, info};

/// Default directory name for pmetal data.
const PMETAL_DIR: &str = ".pmetal";

/// Filename for the node keypair.
const KEYPAIR_FILE: &str = "node_keypair";

/// Node identity containing the keypair and derived PeerId.
#[derive(Clone)]
pub struct NodeIdentity {
    /// The Ed25519 keypair for this node.
    keypair: Keypair,
    /// The PeerId derived from the public key.
    peer_id: PeerId,
}

impl NodeIdentity {
    /// Load or generate a persistent node identity.
    ///
    /// The keypair is stored at `~/.pmetal/node_keypair`.
    /// If the file doesn't exist, a new keypair is generated and saved.
    pub fn load_or_generate() -> Result<Self> {
        Self::load_or_generate_at(&Self::keypair_path()?)
    }

    /// [`load_or_generate`](Self::load_or_generate) with the keypair file at
    /// `keypair_path`.
    ///
    /// Safe to call from several processes at once (two ranks starting
    /// together on one machine): the file is created exclusively, and the
    /// loser of the race loads what the winner wrote instead of failing.
    pub fn load_or_generate_at(keypair_path: &PathBuf) -> Result<Self> {
        let keypair = if keypair_path.exists() {
            Self::load_keypair(keypair_path)?
        } else {
            match Self::generate_and_save(keypair_path) {
                Ok(kp) => {
                    info!("Generated new node identity");
                    kp
                }
                Err(e) if keypair_path.exists() => {
                    debug!("Lost the race to create the node identity ({e}); loading it");
                    Self::load_keypair(keypair_path)?
                }
                Err(e) => return Err(e),
            }
        };

        let peer_id = PeerId::from(keypair.public());
        info!("Node identity: {}", peer_id);

        Ok(Self { keypair, peer_id })
    }

    /// Generate a new ephemeral identity (for testing).
    pub fn ephemeral() -> Self {
        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());
        debug!("Generated ephemeral identity: {}", peer_id);
        Self { keypair, peer_id }
    }

    /// Get the keypair.
    pub fn keypair(&self) -> &Keypair {
        &self.keypair
    }

    /// Get the PeerId.
    pub fn peer_id(&self) -> &PeerId {
        &self.peer_id
    }

    /// Get the PeerId as a base58 string.
    pub fn peer_id_string(&self) -> String {
        self.peer_id.to_base58()
    }

    /// Get the path to the keypair file.
    fn keypair_path() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| DistributedError::Config("Cannot determine home directory".into()))?;

        let pmetal_dir = home.join(PMETAL_DIR);
        if !pmetal_dir.exists() {
            fs::create_dir_all(&pmetal_dir).map_err(|e| {
                DistributedError::Config(format!(
                    "Failed to create {}: {}",
                    pmetal_dir.display(),
                    e
                ))
            })?;
        }

        Ok(pmetal_dir.join(KEYPAIR_FILE))
    }

    /// Load a keypair from a file.
    fn load_keypair(path: &PathBuf) -> Result<Keypair> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(path)
                .map_err(|e| DistributedError::Config(format!("Failed to stat keypair file: {e}")))?
                .permissions()
                .mode();
            if mode & 0o077 != 0 {
                return Err(DistributedError::Config(format!(
                    "node keypair {} is accessible to other users (mode {:o}); run `chmod 600` on it",
                    path.display(),
                    mode & 0o777
                ))
                .into());
            }
        }
        let mut file = File::open(path)
            .map_err(|e| DistributedError::Config(format!("Failed to open keypair file: {}", e)))?;

        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| DistributedError::Config(format!("Failed to read keypair file: {}", e)))?;

        // Decode the Ed25519 secret key (32 bytes)
        if bytes.len() != 32 {
            return Err(DistributedError::Config(format!(
                "Invalid keypair file: expected 32 bytes, got {}",
                bytes.len()
            ))
            .into());
        }

        let secret = ed25519::SecretKey::try_from_bytes(&mut bytes)
            .map_err(|e| DistributedError::Config(format!("Invalid Ed25519 secret key: {}", e)))?;

        let keypair = ed25519::Keypair::from(secret);
        debug!("Loaded keypair from {}", path.display());

        Ok(keypair.into())
    }

    /// Generate a new keypair and save it to a file.
    fn generate_and_save(path: &PathBuf) -> Result<Keypair> {
        // Generate a new Ed25519 keypair using libp2p's internal method
        let ed25519_keypair = ed25519::Keypair::generate();
        let keypair: Keypair = ed25519_keypair.clone().into();

        // Get the secret key bytes (32 bytes)
        let secret = ed25519_keypair.secret();
        let secret_bytes = secret.as_ref();

        // Write the key to a private temporary file, then link it into place:
        // `hard_link` fails if the destination exists, so the file appears
        // whole and owner-only or not at all. Creating the destination
        // directly would let a second process starting at the same moment
        // read it before the key was written.
        let temporary = path.with_extension(format!(
            "tmp.{}.{:08x}",
            std::process::id(),
            rand::random::<u32>()
        ));
        let mut options = std::fs::OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let write = || -> std::io::Result<()> {
            let mut file = options.open(&temporary)?;
            file.write_all(secret_bytes)?;
            file.sync_all()?;
            fs::hard_link(&temporary, path)
        };
        let outcome = write();
        let _ = fs::remove_file(&temporary);
        outcome.map_err(|e| {
            DistributedError::Config(format!(
                "Failed to create keypair file {}: {e}",
                path.display()
            ))
        })?;

        debug!("Saved keypair to {}", path.display());
        Ok(keypair)
    }
}

impl std::fmt::Debug for NodeIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeIdentity")
            .field("peer_id", &self.peer_id.to_base58())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two ranks starting together on one machine both find a keypair, and
    /// the same one: neither fails, and neither reads a half-written file.
    #[test]
    fn concurrent_first_runs_agree_on_one_identity() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("node_keypair");
        let ids: Vec<String> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let path = path.clone();
                    scope.spawn(move || {
                        NodeIdentity::load_or_generate_at(&path)
                            .expect("load or generate")
                            .peer_id_string()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(ids.windows(2).all(|w| w[0] == w[1]), "{ids:?}");
        // No temporary files are left behind.
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn the_keypair_is_owner_only_and_a_loose_one_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("node_keypair");
        NodeIdentity::load_or_generate_at(&path).unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let err = NodeIdentity::load_or_generate_at(&path)
            .unwrap_err()
            .to_string();
        assert!(err.contains("accessible to other users"), "{err}");
    }

    #[test]
    fn test_ephemeral_identity() {
        let id1 = NodeIdentity::ephemeral();
        let id2 = NodeIdentity::ephemeral();

        // Each ephemeral identity should be unique
        assert_ne!(id1.peer_id(), id2.peer_id());
    }

    #[test]
    fn test_peer_id_string() {
        let id = NodeIdentity::ephemeral();
        let s = id.peer_id_string();

        // Base58 encoded PeerId should be a reasonable length
        assert!(s.len() > 40);
        assert!(s.len() < 60);
    }
}
