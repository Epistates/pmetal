//! Adapter management utilities.

use pmetal_core::Result;
use std::path::Path;

/// Adapter container for managing multiple LoRA adapters.
pub struct AdapterManager {
    adapters: Vec<String>,
    active: Option<String>,
}

impl AdapterManager {
    /// Create a new adapter manager.
    pub fn new() -> Self {
        Self {
            adapters: Vec::new(),
            active: None,
        }
    }

    /// Load an adapter from disk.
    pub fn load<P: AsRef<Path>>(&mut self, _path: P, name: &str) -> Result<()> {
        self.adapters.push(name.to_string());
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
}

impl Default for AdapterManager {
    fn default() -> Self {
        Self::new()
    }
}
