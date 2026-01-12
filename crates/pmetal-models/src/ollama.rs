//! Ollama Modelfile generation.
//!
//! This module provides utilities to generate Ollama Modelfiles for exporting
//! models trained with pmetal to Ollama format.
//!
//! # Modelfile Structure
//!
//! A Modelfile is a declarative format that specifies:
//! - Base model (FROM instruction)
//! - Model parameters (PARAMETER)
//! - Prompt template (TEMPLATE)
//! - System message (SYSTEM)
//! - LoRA adapters (ADAPTER)
//!
//! # Example
//!
//! ```ignore
//! use pmetal_models::ollama::ModelfileBuilder;
//!
//! let modelfile = ModelfileBuilder::new()
//!     .from("./model.gguf")
//!     .system("You are a helpful assistant.")
//!     .parameter("temperature", "0.7")
//!     .parameter("num_ctx", "4096")
//!     .build();
//!
//! std::fs::write("Modelfile", modelfile)?;
//! ```

use std::collections::HashMap;
use std::fmt::Write;
use std::path::Path;

/// Error type for Modelfile operations.
#[derive(Debug, thiserror::Error)]
pub enum ModelfileError {
    /// FROM instruction is required.
    #[error("FROM instruction is required - specify a base model")]
    MissingFrom,
    /// IO error writing file.
    #[error("Failed to write Modelfile: {0}")]
    Io(#[from] std::io::Error),
}

/// Known Ollama parameters with their defaults.
#[derive(Debug, Clone, PartialEq)]
pub enum Parameter {
    /// Controls randomness (0.0 - 2.0, default: 0.8)
    Temperature(f32),
    /// Nucleus sampling probability (0.0 - 1.0, default: 0.9)
    TopP(f32),
    /// Top-K sampling (default: 40)
    TopK(i32),
    /// Context window size (default: 2048)
    NumCtx(i32),
    /// Maximum tokens to predict (default: 128, -1 for infinite)
    NumPredict(i32),
    /// Number of GPU layers to use (-1 for all)
    NumGpu(i32),
    /// Number of threads to use (default: auto)
    NumThread(i32),
    /// Stop sequences
    Stop(Vec<String>),
    /// Mirostat sampling (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)
    Mirostat(i32),
    /// Mirostat target entropy (default: 5.0)
    MirostatEta(f32),
    /// Mirostat learning rate (default: 0.1)
    MirostatTau(f32),
    /// Repetition penalty (1.0 = disabled)
    RepeatPenalty(f32),
    /// Last n tokens to consider for repetition (default: 64)
    RepeatLastN(i32),
    /// Penalize newlines (default: true)
    PenalizeNewline(bool),
    /// Random seed (-1 for random)
    Seed(i32),
    /// Typical P sampling (default: 1.0)
    TypicalP(f32),
    /// Frequency penalty (default: 0.0)
    FrequencyPenalty(f32),
    /// Presence penalty (default: 0.0)
    PresencePenalty(f32),
    /// Custom parameter
    Custom(String, String),
}

impl Parameter {
    /// Get the parameter name.
    pub fn name(&self) -> &str {
        match self {
            Parameter::Temperature(_) => "temperature",
            Parameter::TopP(_) => "top_p",
            Parameter::TopK(_) => "top_k",
            Parameter::NumCtx(_) => "num_ctx",
            Parameter::NumPredict(_) => "num_predict",
            Parameter::NumGpu(_) => "num_gpu",
            Parameter::NumThread(_) => "num_thread",
            Parameter::Stop(_) => "stop",
            Parameter::Mirostat(_) => "mirostat",
            Parameter::MirostatEta(_) => "mirostat_eta",
            Parameter::MirostatTau(_) => "mirostat_tau",
            Parameter::RepeatPenalty(_) => "repeat_penalty",
            Parameter::RepeatLastN(_) => "repeat_last_n",
            Parameter::PenalizeNewline(_) => "penalize_newline",
            Parameter::Seed(_) => "seed",
            Parameter::TypicalP(_) => "typical_p",
            Parameter::FrequencyPenalty(_) => "frequency_penalty",
            Parameter::PresencePenalty(_) => "presence_penalty",
            Parameter::Custom(name, _) => name,
        }
    }

    /// Get the parameter value as a string.
    pub fn value(&self) -> String {
        match self {
            Parameter::Temperature(v) => format!("{}", v),
            Parameter::TopP(v) => format!("{}", v),
            Parameter::TopK(v) => format!("{}", v),
            Parameter::NumCtx(v) => format!("{}", v),
            Parameter::NumPredict(v) => format!("{}", v),
            Parameter::NumGpu(v) => format!("{}", v),
            Parameter::NumThread(v) => format!("{}", v),
            Parameter::Stop(stops) => stops.first().cloned().unwrap_or_default(),
            Parameter::Mirostat(v) => format!("{}", v),
            Parameter::MirostatEta(v) => format!("{}", v),
            Parameter::MirostatTau(v) => format!("{}", v),
            Parameter::RepeatPenalty(v) => format!("{}", v),
            Parameter::RepeatLastN(v) => format!("{}", v),
            Parameter::PenalizeNewline(v) => format!("{}", v),
            Parameter::Seed(v) => format!("{}", v),
            Parameter::TypicalP(v) => format!("{}", v),
            Parameter::FrequencyPenalty(v) => format!("{}", v),
            Parameter::PresencePenalty(v) => format!("{}", v),
            Parameter::Custom(_, v) => v.clone(),
        }
    }
}

/// Message role for conversation history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// System message
    System,
    /// User message
    User,
    /// Assistant message
    Assistant,
}

impl Role {
    /// Get the role name for Modelfile.
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// A message in the conversation history.
#[derive(Debug, Clone)]
pub struct Message {
    /// Message role.
    pub role: Role,
    /// Message content.
    pub content: String,
}

impl Message {
    /// Create a new message.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

/// Builder for creating Ollama Modelfiles.
#[derive(Debug, Clone, Default)]
pub struct ModelfileBuilder {
    from: Option<String>,
    parameters: HashMap<String, Vec<String>>,
    template: Option<String>,
    system: Option<String>,
    adapters: Vec<String>,
    license: Option<String>,
    messages: Vec<Message>,
}

impl ModelfileBuilder {
    /// Create a new Modelfile builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder from a base model.
    pub fn from_model(model: impl Into<String>) -> Self {
        let mut builder = Self::new();
        builder.from = Some(model.into());
        builder
    }

    /// Set the base model (required).
    pub fn from(mut self, model: impl Into<String>) -> Self {
        self.from = Some(model.into());
        self
    }

    /// Set the prompt template.
    pub fn template(mut self, template: impl Into<String>) -> Self {
        self.template = Some(template.into());
        self
    }

    /// Set the system message.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Add a parameter.
    pub fn parameter(mut self, name: &str, value: impl Into<String>) -> Self {
        self.parameters
            .entry(name.to_string())
            .or_default()
            .push(value.into());
        self
    }

    /// Add a typed parameter.
    pub fn with_parameter(mut self, param: Parameter) -> Self {
        match &param {
            Parameter::Stop(stops) => {
                for stop in stops {
                    self.parameters
                        .entry("stop".to_string())
                        .or_default()
                        .push(format!("\"{}\"", stop));
                }
            }
            _ => {
                self.parameters
                    .entry(param.name().to_string())
                    .or_default()
                    .push(param.value());
            }
        }
        self
    }

    /// Set temperature (0.0-2.0).
    pub fn temperature(self, temp: f32) -> Self {
        self.with_parameter(Parameter::Temperature(temp))
    }

    /// Set context window size.
    pub fn num_ctx(self, ctx: i32) -> Self {
        self.with_parameter(Parameter::NumCtx(ctx))
    }

    /// Set top-k sampling.
    pub fn top_k(self, k: i32) -> Self {
        self.with_parameter(Parameter::TopK(k))
    }

    /// Set top-p (nucleus) sampling.
    pub fn top_p(self, p: f32) -> Self {
        self.with_parameter(Parameter::TopP(p))
    }

    /// Add a stop sequence.
    pub fn stop(mut self, stop: impl Into<String>) -> Self {
        self.parameters
            .entry("stop".to_string())
            .or_default()
            .push(format!("\"{}\"", stop.into()));
        self
    }

    /// Add a LoRA adapter path.
    pub fn adapter(mut self, path: impl Into<String>) -> Self {
        self.adapters.push(path.into());
        self
    }

    /// Set the license text.
    pub fn license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }

    /// Add a message to the conversation history.
    pub fn message(mut self, role: Role, content: impl Into<String>) -> Self {
        self.messages.push(Message::new(role, content));
        self
    }

    /// Build the Modelfile content.
    pub fn build(&self) -> Result<String, ModelfileError> {
        let from = self.from.as_ref().ok_or(ModelfileError::MissingFrom)?;

        let mut output = String::new();

        writeln!(output, "# Modelfile generated by pmetal").unwrap();
        writeln!(output).unwrap();

        writeln!(output, "FROM {}", from).unwrap();
        writeln!(output).unwrap();

        for adapter in &self.adapters {
            writeln!(output, "ADAPTER {}", adapter).unwrap();
        }
        if !self.adapters.is_empty() {
            writeln!(output).unwrap();
        }

        for (name, values) in &self.parameters {
            for value in values {
                writeln!(output, "PARAMETER {} {}", name, value).unwrap();
            }
        }
        if !self.parameters.is_empty() {
            writeln!(output).unwrap();
        }

        if let Some(system) = &self.system {
            writeln!(output, "SYSTEM \"\"\"").unwrap();
            writeln!(output, "{}", system).unwrap();
            writeln!(output, "\"\"\"").unwrap();
            writeln!(output).unwrap();
        }

        if let Some(template) = &self.template {
            writeln!(output, "TEMPLATE \"\"\"").unwrap();
            writeln!(output, "{}", template).unwrap();
            writeln!(output, "\"\"\"").unwrap();
            writeln!(output).unwrap();
        }

        for message in &self.messages {
            writeln!(output, "MESSAGE {} \"\"\"", message.role.as_str()).unwrap();
            writeln!(output, "{}", message.content).unwrap();
            writeln!(output, "\"\"\"").unwrap();
        }
        if !self.messages.is_empty() {
            writeln!(output).unwrap();
        }

        if let Some(license) = &self.license {
            writeln!(output, "LICENSE \"\"\"").unwrap();
            writeln!(output, "{}", license).unwrap();
            writeln!(output, "\"\"\"").unwrap();
        }

        Ok(output)
    }

    /// Build and write to a file.
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> Result<(), ModelfileError> {
        let content = self.build()?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// Common prompt templates for different model families.
pub mod templates {
    /// Llama 3 chat template.
    pub const LLAMA3_CHAT: &str = r#"{{- if .System }}<|start_header_id|>system<|end_header_id|>
{{ .System }}<|eot_id|>{{- end }}
<|start_header_id|>user<|end_header_id|>
{{ .Prompt }}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{{ .Response }}<|eot_id|>"#;

    /// Qwen3 chat template.
    pub const QWEN3_CHAT: &str = r#"{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>"#;

    /// Gemma instruction template.
    pub const GEMMA_INSTRUCT: &str = r#"<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>"#;

    /// Mistral instruct template.
    pub const MISTRAL_INSTRUCT: &str = r#"[INST] {{ if .System }}{{ .System }}
{{ end }}{{ .Prompt }} [/INST] {{ .Response }}</s>"#;

    /// Phi-3 instruct template.
    pub const PHI3_INSTRUCT: &str = r#"{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ .Response }}<|end|>"#;

    /// DeepSeek chat template.
    pub const DEEPSEEK_CHAT: &str = r#"{{ if .System }}<|begin_of_sentence|>{{ .System }}{{ end }}User: {{ .Prompt }}
Assistant: {{ .Response }}<|end_of_sentence|>"#;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_basic_modelfile() {
        let modelfile = ModelfileBuilder::new()
            .from("./model.gguf")
            .build()
            .unwrap();

        assert!(modelfile.contains("FROM ./model.gguf"));
    }

    #[test]
    fn test_build_with_parameters() {
        let modelfile = ModelfileBuilder::new()
            .from("llama3.2")
            .temperature(0.7)
            .num_ctx(4096)
            .top_p(0.9)
            .build()
            .unwrap();

        assert!(modelfile.contains("PARAMETER temperature 0.7"));
        assert!(modelfile.contains("PARAMETER num_ctx 4096"));
        assert!(modelfile.contains("PARAMETER top_p 0.9"));
    }

    #[test]
    fn test_build_with_system() {
        let modelfile = ModelfileBuilder::new()
            .from("llama3.2")
            .system("You are a helpful assistant.")
            .build()
            .unwrap();

        assert!(modelfile.contains("SYSTEM"));
        assert!(modelfile.contains("You are a helpful assistant."));
    }

    #[test]
    fn test_build_with_adapter() {
        let modelfile = ModelfileBuilder::new()
            .from("./base.gguf")
            .adapter("./lora.gguf")
            .build()
            .unwrap();

        assert!(modelfile.contains("ADAPTER ./lora.gguf"));
    }

    #[test]
    fn test_missing_from_error() {
        let result = ModelfileBuilder::new().build();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ModelfileError::MissingFrom));
    }

    #[test]
    fn test_build_with_template() {
        let modelfile = ModelfileBuilder::new()
            .from("llama3.2")
            .template(templates::LLAMA3_CHAT)
            .build()
            .unwrap();

        assert!(modelfile.contains("TEMPLATE"));
        assert!(modelfile.contains("<|start_header_id|>"));
    }
}
