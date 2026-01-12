//! Model patching utilities for applying LoRA.

use pmetal_core::LoraConfig;

/// Patches a model to add LoRA layers.
pub struct ModelPatcher {
    config: LoraConfig,
}

impl ModelPatcher {
    /// Create a new model patcher.
    pub fn new(config: LoraConfig) -> Self {
        Self { config }
    }

    /// Get target module names based on configuration.
    pub fn target_modules(&self) -> &[String] {
        &self.config.target_modules
    }

    /// Check if a module name should be patched.
    pub fn should_patch(&self, name: &str) -> bool {
        self.config.target_modules.iter().any(|t| name.contains(t))
    }
}
