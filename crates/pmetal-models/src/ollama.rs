//! Ollama Modelfile generation.
//!
//! This module provides utilities to generate Ollama Modelfiles for exporting
//! models trained with pmetal to Ollama format, and a reader for the same
//! format that mirrors Ollama's own parser (so what is written can be checked
//! to read back as what was meant).
//!
//! # Modelfile structure
//!
//! A Modelfile is a declarative format that specifies:
//! - Base model (FROM instruction)
//! - Model parameters (PARAMETER)
//! - Prompt template (TEMPLATE)
//! - System message (SYSTEM)
//! - LoRA adapters (ADAPTER)
//!
//! # Quoting
//!
//! Ollama's parser reads a value as the rest of its line, unless it opens with
//! `"` or `"""`, in which case it runs on, newlines included, to the line that
//! ends with the matching close. There is **no escaping**: a value containing
//! `"""` cannot be written at all, and one that opens with a single `"` must be
//! quoted or the parser swallows the lines after it. The builder quotes every
//! free-text value (`SYSTEM`, `TEMPLATE`, `MESSAGE`, `LICENSE`) in `"""` with
//! the text starting and ending right at the quotes (any newline inside the
//! quotes is part of the value, so framing the text on lines of its own would
//! put a newline at both ends of the system prompt or template), and refuses a
//! value Ollama's parser would alter instead of writing one that reads back
//! as something else.
//!
//! # Example
//!
//! ```
//! use pmetal_models::ollama::ModelfileBuilder;
//!
//! let modelfile = ModelfileBuilder::new()
//!     .from("./model.gguf")
//!     .system("You are a helpful assistant.")
//!     .parameter("temperature", "0.7")
//!     .parameter("num_ctx", "4096")
//!     .build()?;
//! assert!(modelfile.contains("PARAMETER num_ctx 4096"));
//! # Ok::<(), pmetal_models::ollama::ModelfileError>(())
//! ```

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
    /// A value Ollama's Modelfile format cannot carry unchanged.
    #[error("{what} cannot be written to a Modelfile: {reason}")]
    Unrepresentable {
        /// Which instruction's value.
        what: String,
        /// Why.
        reason: String,
    },
    /// A Modelfile Ollama's parser would reject.
    #[error("Modelfile line {line}: {message}")]
    Parse {
        /// 1-based line the parser had reached.
        line: usize,
        /// What it objected to.
        message: String,
    },
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
    /// Maximum tokens to predict (default: -1, infinite; -2 fills the context)
    NumPredict(i32),
    /// Number of GPU layers to use (-1 for all)
    NumGpu(i32),
    /// Number of threads to use (default: auto)
    NumThread(i32),
    /// Stop sequences, one `PARAMETER stop` line each
    Stop(Vec<String>),
    /// Mirostat sampling (0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0).
    /// Ollama now ignores it, with a warning.
    Mirostat(i32),
    /// Mirostat learning rate (default: 0.1). Ignored by current Ollama.
    MirostatEta(f32),
    /// Mirostat target entropy (default: 5.0). Ignored by current Ollama.
    MirostatTau(f32),
    /// Repetition penalty (1.0 = disabled)
    RepeatPenalty(f32),
    /// Last n tokens to consider for repetition (default: 64)
    RepeatLastN(i32),
    /// Penalize newlines (default: true). Ignored by current Ollama.
    PenalizeNewline(bool),
    /// Random seed (default: 0, a random one is not drawn)
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

    /// The parameter's values as text, unquoted: one per `PARAMETER` line
    /// (several for [`Parameter::Stop`], none for an empty list).
    pub fn values(&self) -> Vec<String> {
        match self {
            Parameter::Temperature(v)
            | Parameter::TopP(v)
            | Parameter::MirostatEta(v)
            | Parameter::MirostatTau(v)
            | Parameter::RepeatPenalty(v)
            | Parameter::TypicalP(v)
            | Parameter::FrequencyPenalty(v)
            | Parameter::PresencePenalty(v) => vec![v.to_string()],
            Parameter::TopK(v)
            | Parameter::NumCtx(v)
            | Parameter::NumPredict(v)
            | Parameter::NumGpu(v)
            | Parameter::NumThread(v)
            | Parameter::Mirostat(v)
            | Parameter::RepeatLastN(v)
            | Parameter::Seed(v) => vec![v.to_string()],
            Parameter::PenalizeNewline(v) => vec![v.to_string()],
            Parameter::Stop(stops) => stops.clone(),
            Parameter::Custom(_, v) => vec![v.clone()],
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
///
/// Values are given as the text they mean, never pre-quoted: the builder
/// quotes what needs it, once.
#[derive(Debug, Clone, Default)]
pub struct ModelfileBuilder {
    from: Option<String>,
    /// `PARAMETER` lines in the order they were added.
    parameters: Vec<(String, String)>,
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

    /// Set the prompt template (Go `text/template` syntax, as Ollama's
    /// `TEMPLATE`).
    pub fn template(mut self, template: impl Into<String>) -> Self {
        self.template = Some(template.into());
        self
    }

    /// Set the system message.
    pub fn system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Add a parameter. `value` is the parameter's text, not quoted.
    pub fn parameter(mut self, name: &str, value: impl Into<String>) -> Self {
        self.parameters.push((name.to_string(), value.into()));
        self
    }

    /// Add a typed parameter.
    pub fn with_parameter(mut self, param: Parameter) -> Self {
        for value in param.values() {
            self.parameters.push((param.name().to_string(), value));
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
    pub fn stop(self, stop: impl Into<String>) -> Self {
        self.with_parameter(Parameter::Stop(vec![stop.into()]))
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
    ///
    /// Fails, naming the value, if one cannot be written so that Ollama reads
    /// it back unchanged (see the [module documentation](self)).
    pub fn build(&self) -> Result<String, ModelfileError> {
        let from = self.from.as_ref().ok_or(ModelfileError::MissingFrom)?;

        let mut output = String::new();

        writeln!(output, "# Modelfile generated by pmetal").unwrap();
        writeln!(output).unwrap();

        writeln!(output, "FROM {}", line_value("FROM", from)?).unwrap();
        writeln!(output).unwrap();

        for adapter in &self.adapters {
            writeln!(output, "ADAPTER {}", line_value("ADAPTER", adapter)?).unwrap();
        }
        if !self.adapters.is_empty() {
            writeln!(output).unwrap();
        }

        for (name, value) in &self.parameters {
            if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                return Err(ModelfileError::Unrepresentable {
                    what: format!("PARAMETER {name:?}"),
                    reason: "a parameter name is letters, digits and underscores".into(),
                });
            }
            let what = format!("PARAMETER {name}");
            // A stop sequence is always quoted, as Ollama's own Modelfiles
            // write it; other values only when they would not read back bare.
            let value = if name == "stop" {
                quoted(&what, value)?
            } else {
                line_value(&what, value)?
            };
            writeln!(output, "PARAMETER {name} {value}").unwrap();
        }
        if !self.parameters.is_empty() {
            writeln!(output).unwrap();
        }

        if let Some(system) = &self.system {
            writeln!(output, "SYSTEM {}", quoted("SYSTEM", system)?).unwrap();
            writeln!(output).unwrap();
        }

        if let Some(template) = &self.template {
            writeln!(output, "TEMPLATE {}", quoted("TEMPLATE", template)?).unwrap();
            writeln!(output).unwrap();
        }

        for message in &self.messages {
            let what = format!("MESSAGE {}", message.role.as_str());
            writeln!(
                output,
                "MESSAGE {} {}",
                message.role.as_str(),
                quoted(&what, &message.content)?
            )
            .unwrap();
        }
        if !self.messages.is_empty() {
            writeln!(output).unwrap();
        }

        if let Some(license) = &self.license {
            writeln!(output, "LICENSE {}", quoted("LICENSE", license)?).unwrap();
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

/// Whether Ollama's parser keeps `c` in a value: it drops every rune Go's
/// `strconv.IsPrint` rejects, except the whitespace it handles itself.
///
/// This is the control characters and the common invisible code points (soft
/// hyphen, zero-width and bidirectional marks, the BOM, no-break and other
/// non-ASCII spaces); an unassigned or private-use code point is also dropped
/// by Ollama and not caught here.
fn ollama_keeps(c: char) -> bool {
    if matches!(c, '\n' | '\r' | '\t' | ' ') {
        return true;
    }
    !(c.is_control()
        || matches!(
            c,
            '\u{a0}'
                | '\u{ad}'
                | '\u{1680}'
                | '\u{180e}'
                | '\u{2000}'..='\u{200f}'
                | '\u{2028}'..='\u{202f}'
                | '\u{205f}'..='\u{2064}'
                | '\u{2066}'..='\u{206f}'
                | '\u{3000}'
                | '\u{feff}'
                | '\u{fff9}'..='\u{fffb}'
        ))
}

fn check_text(what: &str, value: &str) -> Result<(), ModelfileError> {
    if let Some(c) = value.chars().find(|c| !ollama_keeps(*c)) {
        return Err(ModelfileError::Unrepresentable {
            what: what.to_string(),
            reason: format!(
                "it contains U+{:04X}, which Ollama's parser drops from a value",
                c as u32
            ),
        });
    }
    Ok(())
}

/// `value` in `"""`, with the text right against the quotes.
fn quoted(what: &str, value: &str) -> Result<String, ModelfileError> {
    check_text(what, value)?;
    if value.contains("\"\"\"") {
        return Err(ModelfileError::Unrepresentable {
            what: what.to_string(),
            reason: "it contains `\"\"\"`, and a Modelfile has no escape for a quote".into(),
        });
    }
    // A single pair of quotes is enough when the text has no `"`; otherwise
    // the parser would end the value at an inner quote that closes a line.
    if value.contains('"') {
        Ok(format!("\"\"\"{value}\"\"\""))
    } else {
        Ok(format!("\"{value}\""))
    }
}

/// A value for a one-line instruction: bare when it reads back as written,
/// quoted when it has a newline, leading or trailing blanks (trimmed from a
/// bare value), or opens with a quote.
fn line_value(what: &str, value: &str) -> Result<String, ModelfileError> {
    check_text(what, value)?;
    let bare = !value.is_empty()
        && !value.contains(['\n', '\r'])
        && value.trim() == value
        && !value.starts_with('"');
    if bare {
        Ok(value.to_string())
    } else {
        quoted(what, value)
    }
}

/// One instruction of a parsed Modelfile, in Ollama's own representation:
/// `FROM` is `model`, a `MESSAGE` carries `role: content`, and a `PARAMETER` is
/// named by the parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelfileCommand {
    /// Lowercased instruction (`model`, `template`, `system`, `adapter`,
    /// `license`, `message`, ...) or the parameter's name.
    pub name: String,
    /// The value, unquoted.
    pub args: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParseState {
    Nil,
    Name,
    Value,
    Parameter,
    Message,
    Comment,
}

fn is_space(c: char) -> bool {
    c == ' ' || c == '\t'
}

fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

fn is_valid_command(name: &str) -> bool {
    matches!(
        name.to_lowercase().as_str(),
        "from"
            | "license"
            | "template"
            | "system"
            | "adapter"
            | "draft"
            | "renderer"
            | "parser"
            | "parameter"
            | "message"
            | "requires"
            | "capability"
    )
}

/// Ollama's `unquote`: `"""..."""` or `"..."`, else the text as is. `None`
/// for an opening quote with no close.
fn unquote(s: &str) -> Option<String> {
    if let Some(rest) = s.strip_prefix("\"\"\"") {
        return rest.strip_suffix("\"\"\"").map(str::to_string);
    }
    if let Some(rest) = s.strip_prefix('"') {
        return rest.strip_suffix('"').map(str::to_string);
    }
    Some(s.to_string())
}

/// Read a Modelfile the way Ollama's parser does (`parser.ParseFile`): the same
/// state machine, the same quoting rules, the same drop of runes it deems
/// unprintable (approximately; see the module notes). A `FROM` is required.
pub fn parse_modelfile(text: &str) -> Result<Vec<ModelfileCommand>, ModelfileError> {
    let mut commands = Vec::new();
    let mut state = ParseState::Nil;
    let mut line = 1usize;
    let mut buffer = String::new();
    let mut role = String::new();
    let mut command = ModelfileCommand {
        name: String::new(),
        args: String::new(),
    };
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let error = |line: usize, message: &str| ModelfileError::Parse {
        line,
        message: message.to_string(),
    };

    for c in text.chars() {
        if is_newline(c) {
            line += 1;
        }
        // `parseRuneForState`: the next state, and the rune to keep (if any).
        let (mut next, kept) = match state {
            ParseState::Nil => match c {
                '#' => (ParseState::Comment, None),
                c if is_space(c) || is_newline(c) => (ParseState::Nil, None),
                c => (ParseState::Name, Some(c)),
            },
            ParseState::Name => match c {
                c if c.is_ascii_alphabetic() => (ParseState::Name, Some(c)),
                c if is_space(c) => (ParseState::Value, None),
                _ => return Err(error(line, "unknown command")),
            },
            ParseState::Value => match c {
                c if is_newline(c) || is_space(c) => (ParseState::Nil, Some(c)),
                c => (ParseState::Value, Some(c)),
            },
            ParseState::Parameter => match c {
                c if c.is_ascii_alphanumeric() || c == '_' => (ParseState::Parameter, Some(c)),
                c if is_space(c) => (ParseState::Value, None),
                _ => return Err(error(line, "unexpected end of file")),
            },
            ParseState::Message => match c {
                c if c.is_ascii_alphabetic() => (ParseState::Message, Some(c)),
                c if is_space(c) => (ParseState::Value, None),
                _ => return Err(error(line, "unexpected end of file")),
            },
            ParseState::Comment => match c {
                c if is_newline(c) => (ParseState::Nil, None),
                _ => (ParseState::Comment, None),
            },
        };

        if next != state {
            match state {
                ParseState::Name => {
                    if !is_valid_command(&buffer) {
                        return Err(error(line, "unknown command"));
                    }
                    match buffer.to_lowercase().as_str() {
                        "from" => command.name = "model".into(),
                        "parameter" => next = ParseState::Parameter,
                        "message" => {
                            next = ParseState::Message;
                            command.name = "message".into();
                        }
                        other => command.name = other.to_string(),
                    }
                }
                ParseState::Parameter => command.name = buffer.clone(),
                ParseState::Message => {
                    if !matches!(buffer.as_str(), "system" | "user" | "assistant") {
                        return Err(error(
                            line,
                            "message role must be one of \"system\", \"user\", or \"assistant\"",
                        ));
                    }
                    role = buffer.clone();
                }
                ParseState::Comment | ParseState::Nil => {}
                ParseState::Value => {
                    let value = unquote(buffer.trim());
                    // Still inside a quote, or a blank: the value goes on.
                    if value.is_none() || kept.is_some_and(is_space) {
                        if let Some(c) = kept {
                            buffer.push(c);
                        }
                        continue;
                    }
                    let mut value = value.unwrap_or_default();
                    if !role.is_empty() {
                        value = format!("{role}: {value}");
                        role.clear();
                    }
                    command.args = value;
                    commands.push(command.clone());
                }
            }
            buffer.clear();
            state = next;
        }

        // `strconv.IsPrint`: tabs and newlines are not printable (they reach a
        // value through the branch above, inside quotes).
        if let Some(c) = kept
            && (c == ' ' || (ollama_keeps(c) && !matches!(c, '\n' | '\r' | '\t')))
        {
            buffer.push(c);
        }
    }

    match state {
        ParseState::Comment | ParseState::Nil => {}
        ParseState::Value => {
            let Some(mut value) = unquote(buffer.trim()) else {
                return Err(error(line, "unexpected end of file"));
            };
            if !role.is_empty() {
                value = format!("{role}: {value}");
            }
            command.args = value;
            commands.push(command);
        }
        _ => return Err(error(line, "unexpected end of file")),
    }

    if commands.iter().any(|c| c.name == "model") {
        Ok(commands)
    } else {
        Err(ModelfileError::MissingFrom)
    }
}

/// Common prompt templates for different model families, in the Go
/// `text/template` syntax of Ollama's `TEMPLATE`.
///
/// Each renders, for a system message `S` and a prompt `P`, the text its
/// family's published chat template renders for that conversation with the
/// generation prompt added, up to `{{ .Response }}`, where Ollama stops (the
/// beginning-of-sequence token is left out, since Ollama adds the model's own).
/// That is checked against the `transformers` chat templates, with and without
/// a system message, in `tests/ollama_modelfile.rs`.
pub mod templates {
    /// Llama 3 chat template (`<|start_header_id|>` roles, `\n\n` after each
    /// header, no separator between turns).
    pub const LLAMA3_CHAT: &str = r#"{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"#;

    /// Qwen3 chat template (ChatML), thinking left to the model.
    pub const QWEN3_CHAT: &str = r#"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>"#;

    /// Gemma instruction template. Gemma has no system role: a system message
    /// leads the first user turn.
    pub const GEMMA_INSTRUCT: &str = r#"<start_of_turn>user
{{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>"#;

    /// Mistral instruct template (v0.3: no space before `[/INST]`). A system
    /// message leads the user turn.
    pub const MISTRAL_INSTRUCT: &str = r#"[INST] {{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}[/INST]{{ .Response }}</s>"#;

    /// Phi-3 instruct template.
    pub const PHI3_INSTRUCT: &str = r#"{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ .Response }}<|end|>"#;

    /// DeepSeek chat template (V2, V2-Lite, V2.5, Coder-V2: `User:` and
    /// `Assistant:` turns). V3 and R1 use `<｜User｜>` tokens instead.
    pub const DEEPSEEK_CHAT: &str = r#"{{ if .System }}{{ .System }}

{{ end }}User: {{ .Prompt }}

Assistant:{{ .Response }}<｜end▁of▁sentence｜>"#;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn commands(text: &str) -> Vec<(String, String)> {
        parse_modelfile(text)
            .unwrap()
            .into_iter()
            .map(|c| (c.name, c.args))
            .collect()
    }

    fn pair(name: &str, args: &str) -> (String, String) {
        (name.to_string(), args.to_string())
    }

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

    /// Parameters come out in the order they were added: a Modelfile built
    /// twice from the same calls is the same file.
    #[test]
    fn parameters_are_written_in_the_order_they_were_added() {
        let build = || {
            ModelfileBuilder::new()
                .from("m")
                .temperature(0.5)
                .num_ctx(8192)
                .stop("<|a|>")
                .top_k(20)
                .stop("<|b|>")
                .top_p(0.8)
                .build()
                .unwrap()
        };
        let text = build();
        assert_eq!(text, build());
        let names: Vec<_> = commands(&text).into_iter().map(|(n, _)| n).collect();
        assert_eq!(
            names,
            [
                "model",
                "temperature",
                "num_ctx",
                "stop",
                "top_k",
                "stop",
                "top_p"
            ]
        );
    }

    /// A system prompt reads back as itself, not with the newlines the text was
    /// framed by.
    #[test]
    fn free_text_reads_back_exactly() {
        let system = "You are a helpful assistant.\nBe brief.";
        let template = "{{ .System }}\n{{ .Prompt }}";
        let modelfile = ModelfileBuilder::new()
            .from("base")
            .system(system)
            .template(template)
            .message(Role::User, "Hi")
            .message(Role::Assistant, "Hello, \"friend\"")
            .license("MIT\n\nCopyright")
            .build()
            .unwrap();
        assert_eq!(
            commands(&modelfile),
            vec![
                pair("model", "base"),
                pair("system", system),
                pair("template", template),
                pair("message", "user: Hi"),
                pair("message", "assistant: Hello, \"friend\""),
                pair("license", "MIT\n\nCopyright"),
            ]
        );
    }

    #[test]
    fn stop_sequences_and_awkward_values_read_back_exactly() {
        let modelfile = ModelfileBuilder::new()
            .from("base")
            .stop("<|im_end|>")
            .stop("a \"quoted\" stop")
            .stop("two\nlines")
            .stop(" padded ")
            .parameter("num_ctx", "4096")
            .build()
            .unwrap();
        assert_eq!(
            commands(&modelfile),
            vec![
                pair("model", "base"),
                pair("stop", "<|im_end|>"),
                pair("stop", "a \"quoted\" stop"),
                pair("stop", "two\nlines"),
                pair("stop", " padded "),
                pair("num_ctx", "4096"),
            ]
        );
    }

    #[test]
    fn a_value_starting_with_a_quote_is_quoted() {
        let modelfile = ModelfileBuilder::new()
            .from("base")
            .system("\"Quoted\" opening, and a close.\"")
            .build()
            .unwrap();
        assert_eq!(
            commands(&modelfile)[1],
            pair("system", "\"Quoted\" opening, and a close.\"")
        );
    }

    #[test]
    fn a_value_ollama_would_alter_is_refused() {
        let triple = ModelfileBuilder::new()
            .from("base")
            .system("say \"\"\" now")
            .build()
            .unwrap_err();
        assert!(
            matches!(triple, ModelfileError::Unrepresentable { .. }),
            "{triple}"
        );
        assert!(triple.to_string().contains("SYSTEM"));

        // A zero-width space is dropped from a value by Ollama's parser.
        let invisible = ModelfileBuilder::new()
            .from("base")
            .template("a\u{200b}b")
            .build()
            .unwrap_err();
        assert!(invisible.to_string().contains("U+200B"), "{invisible}");

        let name = ModelfileBuilder::new()
            .from("base")
            .parameter("num ctx", "1")
            .build()
            .unwrap_err();
        assert!(name.to_string().contains("num ctx"), "{name}");
    }

    #[test]
    fn the_parser_reads_ollamas_forms() {
        // Comments, case-insensitive instructions, bare and quoted values.
        let text = "# a comment\nfrom ./m.gguf\nPARAMETER stop \"</s>\"\n\
                    SYSTEM \"\"\"\nLine one\nLine two\n\"\"\"\nMESSAGE user \"Hi there\"\n";
        assert_eq!(
            commands(text),
            vec![
                pair("model", "./m.gguf"),
                pair("stop", "</s>"),
                pair("system", "\nLine one\nLine two\n"),
                pair("message", "user: Hi there"),
            ]
        );
        assert!(matches!(
            parse_modelfile("SYSTEM hi\n").unwrap_err(),
            ModelfileError::MissingFrom
        ));
        assert!(parse_modelfile("FROM m\nBOGUS x\n").is_err());
        assert!(parse_modelfile("FROM m\nMESSAGE robot hi\n").is_err());
        assert!(parse_modelfile("FROM m\nSYSTEM \"\"\"never closed\n").is_err());
    }
}
