//! Model registry for architecture detection and instantiation.

use std::collections::HashMap;

/// Supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    /// Llama family (2, 3, 3.1, 3.2, 3.3, 4).
    Llama,
    /// Llama 3.2 Vision (Mllama).
    Mllama,
    /// Mistral family.
    Mistral,
    /// Qwen family (2, 2.5, 3).
    Qwen,
    /// Gemma family (2, 3).
    Gemma,
    /// Phi family (3, 4).
    Phi,
    /// DeepSeek family.
    DeepSeek,
}

/// Model registry for looking up architectures.
pub struct ModelRegistry {
    patterns: HashMap<String, Architecture>,
}

impl ModelRegistry {
    /// Create a new model registry with default patterns.
    #[must_use]
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        // Llama patterns
        patterns.insert("llama".to_string(), Architecture::Llama);
        patterns.insert("meta-llama".to_string(), Architecture::Llama);

        // Mllama (Vision) patterns
        patterns.insert("mllama".to_string(), Architecture::Mllama);
        patterns.insert("llama-3.2-11b-vision".to_string(), Architecture::Mllama);
        patterns.insert("llama-3.2-90b-vision".to_string(), Architecture::Mllama);

        // Mistral patterns
        patterns.insert("mistral".to_string(), Architecture::Mistral);
        patterns.insert("ministral".to_string(), Architecture::Mistral);

        // Qwen patterns
        patterns.insert("qwen".to_string(), Architecture::Qwen);

        // Gemma patterns
        patterns.insert("gemma".to_string(), Architecture::Gemma);

        // Phi patterns
        patterns.insert("phi".to_string(), Architecture::Phi);
        patterns.insert("microsoft/phi".to_string(), Architecture::Phi);

        // DeepSeek patterns
        patterns.insert("deepseek".to_string(), Architecture::DeepSeek);

        Self { patterns }
    }

    /// Detect architecture from model ID.
    pub fn detect_architecture(&self, model_id: &str) -> Option<Architecture> {
        let lower = model_id.to_lowercase();
        for (pattern, arch) in &self.patterns {
            if lower.contains(pattern) {
                return Some(*arch);
            }
        }
        None
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
