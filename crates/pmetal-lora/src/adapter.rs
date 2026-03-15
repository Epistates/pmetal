//! Adapter management utilities.

use pmetal_core::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Adapter container for managing multiple LoRA adapters.
pub struct AdapterManager {
    adapters: Vec<String>,
    active: Option<String>,
    adapter_paths: HashMap<String, PathBuf>,
}

impl AdapterManager {
    /// Create a new adapter manager.
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
            active: None,
            adapter_paths: HashMap::new(),
        }
    }

    /// Load an adapter from disk.
    pub fn load<P: AsRef<Path>>(&mut self, path: P, name: &str) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(pmetal_core::PMetalError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("adapter path does not exist: {}", path.display()),
            )));
        }

        let resolved_path = if path.is_dir() {
            let supported = [
                path.join("adapter_model.safetensors"),
                path.join("lora_weights.safetensors"),
                path.join("adapter_config.json"),
            ];

            if supported.iter().all(|candidate| !candidate.exists()) {
                return Err(pmetal_core::PMetalError::InvalidArgument(format!(
                    "directory does not contain adapter artifacts: {}",
                    path.display()
                )));
            }

            path.to_path_buf()
        } else {
            let is_safetensors =
                path.extension().and_then(|ext| ext.to_str()) == Some("safetensors");
            if !is_safetensors {
                return Err(pmetal_core::PMetalError::InvalidArgument(format!(
                    "unsupported adapter file: {}",
                    path.display()
                )));
            }

            path.to_path_buf()
        };

        if !self.adapters.contains(&name.to_string()) {
            self.adapters.push(name.to_string());
        }
        self.adapter_paths.insert(name.to_string(), resolved_path);
        Ok(())
    }

    /// Set the active adapter.
    pub fn set_active(&mut self, name: &str) -> Result<()> {
        if self.adapters.contains(&name.to_string()) {
            self.active = Some(name.to_string());
            Ok(())
        } else {
            Err(pmetal_core::PMetalError::InvalidArgument(format!(
                "Adapter '{}' not found",
                name
            )))
        }
    }

    /// Get the active adapter name.
    pub fn active(&self) -> Option<&str> {
        self.active.as_deref()
    }

    /// List all loaded adapters.
    pub fn list(&self) -> &[String] {
        &self.adapters
    }

    /// Get the loaded path for an adapter.
    pub fn path(&self, name: &str) -> Option<&Path> {
        self.adapter_paths.get(name).map(PathBuf::as_path)
    }
}

impl Default for AdapterManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::AdapterManager;

    #[test]
    fn load_rejects_missing_path() {
        let mut manager = AdapterManager::new();
        let err = manager
            .load("/tmp/does-not-exist-pmetal", "missing")
            .unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn load_accepts_adapter_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("adapter_config.json"), "{}").unwrap();

        let mut manager = AdapterManager::new();
        manager.load(dir.path(), "adapter").unwrap();

        assert_eq!(manager.active(), None);
        assert_eq!(manager.list().len(), 1);
        assert_eq!(manager.list()[0], "adapter");
        assert_eq!(manager.path("adapter"), Some(dir.path()));
    }
}
