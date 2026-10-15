//! Chat template system for formatting conversations.
//!
//! Provides model-specific chat templates for SFT training and inference with:
//! - Proper formatting for each model family (Llama, Mistral, Gemma, etc.)
//! - Response masking support (identifying which tokens to train on)
//! - System message customization
//! - Tool/function calling support (Qwen, Llama 3.1+, Mistral v3+, ChatML)

use serde::{Deserialize, Serialize};
use std::ops::Range;

mod builtin;

// ============================================================================
// Tool calling types (OpenAI-compatible)
// ============================================================================

/// A function definition within a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    /// Function name.
    pub name: String,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the function's parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// A tool definition (OpenAI-compatible format).
///
/// ```json
/// {"type": "function", "function": {"name": "get_weather", "description": "...", "parameters": {...}}}
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool type. Always "function" for now.
    #[serde(rename = "type", default = "default_tool_type")]
    pub tool_type: String,
    /// The function definition.
    pub function: FunctionDefinition,
}

fn default_tool_type() -> String {
    "function".to_string()
}

/// A function call made by the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Name of the function to call.
    pub name: String,
    /// Arguments as a JSON string or object.
    pub arguments: serde_json::Value,
}

/// A tool call within an assistant message.
///
/// ```json
/// {"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"SF\"}"}}
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique call ID (optional, used for matching responses).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Tool type. Always "function" for now.
    #[serde(rename = "type", default = "default_tool_type")]
    pub tool_type: String,
    /// The function call details.
    pub function: FunctionCall,
}

/// One item of a message whose content is more than text.
///
/// A chat template renders an image or video item as the model's own
/// placeholder (Qwen 3.5's `<|vision_start|><|image_pad|><|vision_end|>`,
/// Llama 3.2 Vision's `<|image|>`), where the item sits among the text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentPart {
    /// A run of text.
    Text(String),
    /// An image.
    Image,
    /// A video.
    Video,
}

impl ContentPart {
    /// The item as a chat template reads it: `{"type": "text", "text": ...}`,
    /// `{"type": "image"}` or `{"type": "video"}`.
    pub fn to_template_value(&self) -> serde_json::Value {
        match self {
            Self::Text(text) => serde_json::json!({"type": "text", "text": text}),
            Self::Image => serde_json::json!({"type": "image"}),
            Self::Video => serde_json::json!({"type": "video"}),
        }
    }
}

/// A single message in a conversation.
#[derive(Debug, Clone)]
pub struct Message {
    /// Role: "system", "user", "assistant", or "tool"
    pub role: String,
    /// Content of the message.
    pub content: String,
    /// The content as a list of items, when it carries images or videos.
    /// Only the upstream Jinja template places media; `content` holds the
    /// text alone for everything else.
    pub parts: Option<Vec<ContentPart>>,
    /// The thinking behind an assistant turn, apart from its `content`
    /// (the OpenAI-style `reasoning_content` field). Templates that render
    /// thinking themselves read it; see [`ChatTemplate::apply_inference_with_kwargs`]
    /// for how a `<think>…</think>` left inside `content` is handled.
    pub reasoning_content: Option<String>,
    /// Tool calls made by the assistant (role="assistant" only).
    pub tool_calls: Option<Vec<ToolCall>>,
    /// ID of the tool call this message responds to (role="tool" only).
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Create a new message.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            parts: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a message whose content is a list of text, image and video
    /// items. `content` becomes the text items joined.
    pub fn with_parts(role: impl Into<String>, parts: Vec<ContentPart>) -> Self {
        let content = parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        Self {
            parts: Some(parts),
            ..Self::new(role, content)
        }
    }

    /// Whether the content carries an image or a video.
    pub fn has_media(&self) -> bool {
        self.parts
            .as_ref()
            .is_some_and(|parts| parts.iter().any(|p| !matches!(p, ContentPart::Text(_))))
    }

    /// Attach the thinking that preceded this (assistant) message.
    pub fn with_reasoning_content(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning_content = Some(reasoning.into());
        self
    }

    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }

    /// Create an assistant message with tool calls.
    pub fn assistant_tool_calls(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            tool_calls: Some(tool_calls),
            ..Self::new("assistant", content)
        }
    }

    /// Create a tool response message.
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            tool_call_id: Some(tool_call_id.into()),
            ..Self::new("tool", content)
        }
    }

    /// Check if this message contains tool calls.
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty())
    }
}

/// Split a `<think>…</think>` left at the head of an assistant `content`
/// into `(reasoning, answer)`, the way Qwen3's own template does:
/// reasoning is the text before `</think>` (after any `<think>`, newlines
/// trimmed at both ends), the answer is the text after the last `</think>`
/// with leading newlines trimmed. `None` when there is no `</think>`.
pub fn split_reasoning_content(content: &str) -> Option<(String, String)> {
    let (before, _) = content.split_once("</think>")?;
    let (_, after) = content.rsplit_once("</think>")?;
    let before = before.trim_end_matches('\n');
    let reasoning = before
        .rsplit_once("<think>")
        .map_or(before, |(_, r)| r)
        .trim_start_matches('\n');
    Some((
        reasoning.to_string(),
        after.trim_start_matches('\n').to_string(),
    ))
}

/// Tool-call arguments as chat templates expect them: an object.
///
/// OpenAI clients send `arguments` as a JSON-encoded string, while the
/// templates iterate it as a mapping (`tool_call.arguments|items`), which a
/// string cannot do. A string that holds a JSON object is decoded; anything
/// else is passed through unchanged.
fn tool_call_arguments(arguments: &serde_json::Value) -> serde_json::Value {
    if let Some(s) = arguments.as_str() {
        if let Ok(object @ serde_json::Value::Object(_)) = serde_json::from_str(s) {
            return object;
        }
    }
    arguments.clone()
}

/// Keyword arguments for a chat template: the `**kwargs` of transformers'
/// `apply_chat_template`, which a template reads as plain variables.
///
/// The model makers document theirs: `enable_thinking` (Qwen3 onwards,
/// Gemma 4, SmolLM3), `reasoning_effort` (gpt-oss: `low|medium|high`;
/// Qwen3.8: `xhigh|medium|low`), `preserve_thinking` (Qwen3.8), and the
/// rest are whatever a given template happens to read. Serialized as a
/// plain JSON object, insertion-ordered.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ChatTemplateKwargs(serde_json::Map<String, serde_json::Value>);

impl ChatTemplateKwargs {
    /// `enable_thinking`: whether the model thinks before it answers.
    pub const ENABLE_THINKING: &'static str = "enable_thinking";
    /// `reasoning_effort`: how long the model thinks.
    pub const REASONING_EFFORT: &'static str = "reasoning_effort";
    /// `preserve_thinking`: keep the thinking of earlier turns in the prompt.
    pub const PRESERVE_THINKING: &'static str = "preserve_thinking";

    /// No kwargs.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from a JSON object, refusing the names the renderer binds
    /// itself (`messages`, `tools`, `documents`, `add_generation_prompt`).
    pub fn from_map(map: serde_json::Map<String, serde_json::Value>) -> Result<Self, String> {
        let kwargs = Self(map);
        kwargs.check_reserved()?;
        Ok(kwargs)
    }

    /// Parse a JSON object string, e.g. `{"enable_thinking": false}`.
    pub fn from_json(json: &str) -> Result<Self, String> {
        match serde_json::from_str::<serde_json::Value>(json) {
            Ok(serde_json::Value::Object(map)) => Self::from_map(map),
            Ok(_) => Err("chat template kwargs must be a JSON object".into()),
            Err(e) => Err(format!("chat template kwargs are not valid JSON: {e}")),
        }
    }

    /// The kwargs of a `pmetal infer`-style surface: a JSON object (the
    /// `--chat-template-kwargs` value), with the dedicated controls set over
    /// it (`--no-thinking`, `--reasoning-effort`, `--no-preserve-thinking`).
    pub fn from_controls(
        json: Option<&str>,
        no_thinking: bool,
        reasoning_effort: Option<&str>,
        no_preserve_thinking: bool,
    ) -> Result<Self, String> {
        let mut kwargs = match json.filter(|j| !j.trim().is_empty()) {
            Some(json) => Self::from_json(json)?,
            None => Self::new(),
        };
        if no_thinking {
            kwargs.set(Self::ENABLE_THINKING, false);
        }
        if let Some(effort) = reasoning_effort.filter(|e| !e.is_empty()) {
            kwargs.set(Self::REASONING_EFFORT, effort);
        }
        if no_preserve_thinking {
            kwargs.set(Self::PRESERVE_THINKING, false);
        }
        Ok(kwargs)
    }

    /// Refuse a name the renderer binds itself.
    pub fn check_reserved(&self) -> Result<(), String> {
        match self
            .0
            .keys()
            .find(|k| crate::jinja_chat::RESERVED_TEMPLATE_KWARGS.contains(&k.as_str()))
        {
            Some(key) => Err(format!(
                "`{key}` cannot be a chat template kwarg: it is set from the request itself"
            )),
            None => Ok(()),
        }
    }

    /// Set one kwarg, replacing any earlier value.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) {
        self.0.insert(key.into(), value.into());
    }

    /// Set one kwarg (builder form).
    pub fn with(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.set(key, value);
        self
    }

    /// Set a kwarg only when the caller has not set it already, so an
    /// explicit value always beats a default.
    pub fn set_default(&mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) {
        self.0.entry(key.into()).or_insert_with(|| value.into());
    }

    /// Copy every kwarg of `other` over this one.
    pub fn extend(&mut self, other: &ChatTemplateKwargs) {
        for (k, v) in &other.0 {
            self.0.insert(k.clone(), v.clone());
        }
    }

    /// Look up one kwarg.
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.0.get(key)
    }

    /// Remove one kwarg.
    pub fn remove(&mut self, key: &str) -> Option<serde_json::Value> {
        self.0.shift_remove(key)
    }

    /// `enable_thinking` when it is set to a boolean.
    pub fn enable_thinking(&self) -> Option<bool> {
        self.get(Self::ENABLE_THINKING).and_then(|v| v.as_bool())
    }

    /// `reasoning_effort` when it is set to a string.
    pub fn reasoning_effort(&self) -> Option<&str> {
        self.get(Self::REASONING_EFFORT).and_then(|v| v.as_str())
    }

    /// Whether no kwarg is set.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Kwarg names, in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }

    /// The kwargs as a JSON object.
    pub fn as_map(&self) -> &serde_json::Map<String, serde_json::Value> {
        &self.0
    }
}

/// The `reasoning_effort` levels a template family is documented to accept,
/// for templates that do not check the value themselves. gpt-oss's harmony
/// template writes any string into its system message, while OpenAI's model
/// card names three levels. Templates that validate (Qwen3.8 raises on
/// anything but `xhigh|medium|low`) are checked by rendering them.
pub fn documented_reasoning_effort_levels(
    template: ChatTemplateType,
) -> Option<&'static [&'static str]> {
    match template {
        ChatTemplateType::GptOss => Some(&["low", "medium", "high"]),
        _ => None,
    }
}

/// Known chat template types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChatTemplateType {
    /// ChatML format: <|im_start|>role\ncontent<|im_end|>
    ChatMl,
    /// Llama-2 format: `[INST] <<SYS>>...message...[/INST]`
    Llama2,
    /// Llama-3 format: <|start_header_id|>role<|end_header_id|>content<|eot_id|>
    Llama3,
    /// Mistral format: `[INST] message [/INST]`
    Mistral,
    /// Gemma format: <start_of_turn>role\ncontent<end_of_turn>
    Gemma,
    /// Gemma 4 format: `<|turn>role\ncontent<turn|>\n`, role=`model` for the
    /// assistant turn, with an optional `<|think|>` system block for the
    /// built-in reasoning channel. Distinct from Gemma 2/3 — the tokens are
    /// different (`<|turn>`/`<turn|>` vs `<start_of_turn>`/`<end_of_turn>`).
    Gemma4,
    /// Phi-3 format: <|user|>content<|end|><|assistant|>content<|end|>
    Phi3,
    /// Phi-4 format: <|im_start|>role<|im_sep|>content<|im_end|>
    Phi4,
    /// Qwen format: <|im_start|>role\ncontent<|im_end|>
    Qwen,
    /// Alpaca format: ### Instruction:\n...\n### Response:\n...
    Alpaca,
    /// Vicuna format: USER: ... ASSISTANT: ...
    Vicuna,
    /// Zephyr format: <|user|>...<|assistant|>...
    Zephyr,
    /// GPT-OSS Harmony format: <|start|>role<|message|>content<|end|>
    /// OpenAI's first open-weight models with multi-channel support.
    GptOss,
    /// Llama-4 format: <|header_start|>role<|header_end|>content<|eot|>
    Llama4,
    /// DeepSeek format: <｜User｜>content<｜end▁of▁sentence｜>
    DeepSeek,
    /// Cohere Command R format: <|START_OF_TURN_TOKEN|><|USER_TOKEN|>content<|END_OF_TURN_TOKEN|>
    Cohere,
    /// Custom template defined by user
    Custom,
}

impl ChatTemplateType {
    /// Get the EOS token for this template type.
    pub fn eos_token(&self) -> &'static str {
        match self {
            Self::ChatMl | Self::Qwen | Self::Phi4 => "<|im_end|>",
            Self::Llama2 => "</s>",
            Self::Llama3 => "<|eot_id|>",
            Self::Mistral => "</s>",
            Self::Gemma => "<end_of_turn>",
            Self::Gemma4 => "<turn|>",
            Self::Phi3 => "<|end|>",
            Self::Alpaca => "</s>",
            Self::Vicuna => "</s>",
            Self::Zephyr => "</s>",
            Self::GptOss => "<|return|>",
            Self::Llama4 => "<|eot|>",
            Self::DeepSeek => "<｜end▁of▁sentence｜>",
            Self::Cohere => "<|END_OF_TURN_TOKEN|>",
            Self::Custom => "</s>",
        }
    }

    /// Get the BOS token for this template type (if any).
    ///
    /// `None` here does not mean the model has no BOS, it means the
    /// tokenizer's own post-processor prepends it. Gemma 2 and 3 do exactly
    /// that; Gemma 4 ships a pass-through post-processor and emits
    /// `{{ bos_token }}` from its chat template instead, so it has to come
    /// from here or the model receives a conversation with no beginning and
    /// answers as though the prompt were empty.
    pub fn bos_token(&self) -> Option<&'static str> {
        match self {
            Self::Llama2 => Some("<s>"),
            Self::Llama3 => Some("<|begin_of_text|>"),
            Self::Llama4 => Some("<|begin_of_text|>"),
            Self::DeepSeek => Some("<｜begin▁of▁sentence｜>"),
            Self::Cohere => Some("<BOS_TOKEN>"),
            Self::Gemma4 => Some("<bos>"),
            _ => None,
        }
    }
}

/// A conversation rendered by a chat template, with where the model's own
/// output sits in it.
#[derive(Debug, Clone)]
pub struct FormattedChat {
    /// The full formatted text.
    pub text: String,
    /// Byte offset where the final response begins: everything before it is
    /// the prompt. Where the conversation does not end in an assistant turn
    /// the text is a generation prompt and this is its length.
    pub response_start: usize,
    /// The template type used.
    pub template_type: ChatTemplateType,
    /// Byte ranges of `text` the model itself wrote, one per assistant turn
    /// (its reasoning and tool calls included, up to and including the token
    /// that ends the turn). Everything outside the ranges is prompt.
    ///
    /// A turn is listed only if the template renders it the same way
    /// whether or not later turns follow, which is how its range can be
    /// found: Qwen3, for one, drops the thinking of every turn before the
    /// last user message, so only its final turn is listed.
    pub response_spans: Vec<Range<usize>>,
    /// The index in the conversation's messages of the assistant turn each of
    /// [`response_spans`](Self::response_spans) is, in the same order. A turn
    /// the template does not render the same whatever follows has no span, so
    /// this is how to tell which turns have one.
    pub response_turns: Vec<usize>,
}

impl FormattedChat {
    /// Get the prompt portion (before response).
    pub fn prompt(&self) -> &str {
        &self.text[..self.response_start]
    }

    /// Get the response portion.
    pub fn response(&self) -> &str {
        &self.text[self.response_start..]
    }
}

/// Chat template configuration and application.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    /// The template type.
    pub template_type: ChatTemplateType,
    /// System message used when a conversation has none (read by the
    /// built-in templates only; a model's own template has its own default).
    pub default_system_message: Option<String>,
    /// BOS token string, as `{{ bos_token }}` renders it.
    pub bos_token: Option<String>,
    /// EOS token string, as `{{ eos_token }}` renders it.
    pub eos_token: String,
    /// The Jinja source of the model's own chat template, from
    /// `chat_template.jinja` or `tokenizer_config.json`. Everything this type
    /// renders goes through it, with the engine and environment transformers
    /// uses, so what the model is trained on is what it is served. `None` for
    /// a checkpoint with no chat template (a base model): the conventional
    /// format of [`template_type`](Self::template_type) is rendered instead,
    /// from the built-in Jinja templates.
    pub jinja_source: Option<String>,
}

impl ChatTemplate {
    /// Create a new chat template with the given type.
    pub fn new(template_type: ChatTemplateType) -> Self {
        Self {
            template_type,
            default_system_message: None,
            bos_token: template_type.bos_token().map(String::from),
            eos_token: template_type.eos_token().to_string(),
            jinja_source: None,
        }
    }

    /// Create a ChatML template.
    pub fn chatml() -> Self {
        Self::new(ChatTemplateType::ChatMl)
    }

    /// Create a Llama-2 template.
    pub fn llama2() -> Self {
        Self::new(ChatTemplateType::Llama2)
    }

    /// Create a Llama-3 template.
    pub fn llama3() -> Self {
        let mut template = Self::new(ChatTemplateType::Llama3);
        template.bos_token = Some("<|begin_of_text|>".to_string());
        template
    }

    /// Create a Mistral template.
    pub fn mistral() -> Self {
        Self::new(ChatTemplateType::Mistral)
    }

    /// Create a Gemma template.
    pub fn gemma() -> Self {
        Self::new(ChatTemplateType::Gemma)
    }

    /// Create a Gemma 4 template (distinct from Gemma 2/3 — uses `<|turn>`
    /// and `<turn|>` with an asymmetric pipe on each side and role `model`
    /// for the assistant turn).
    pub fn gemma4() -> Self {
        Self::new(ChatTemplateType::Gemma4)
    }

    /// Create a Phi-3 template.
    pub fn phi3() -> Self {
        Self::new(ChatTemplateType::Phi3)
    }

    /// Create a Qwen template.
    pub fn qwen() -> Self {
        Self::new(ChatTemplateType::Qwen)
    }

    /// Create an Alpaca template.
    pub fn alpaca() -> Self {
        Self::new(ChatTemplateType::Alpaca)
    }

    /// Create a GPT-OSS Harmony template.
    ///
    /// GPT-OSS uses OpenAI's Harmony format with multi-channel support:
    /// - `<|start|>role<|message|>content<|end|>` for basic messages
    /// - `<|start|>role<|channel|>channel_name<|message|>content<|end|>` for channels
    /// - Special EOS token: `<|return|>`
    /// - Roles: system, developer, user, assistant
    pub fn gpt_oss() -> Self {
        let mut template = Self::new(ChatTemplateType::GptOss);
        template.eos_token = "<|return|>".to_string();
        template
    }

    /// Create a Llama-4 template.
    pub fn llama4() -> Self {
        let mut template = Self::new(ChatTemplateType::Llama4);
        template.bos_token = Some("<|begin_of_text|>".to_string());
        template
    }

    /// Create a DeepSeek template.
    pub fn deepseek() -> Self {
        Self::new(ChatTemplateType::DeepSeek)
    }

    /// Create a Cohere Command R template.
    pub fn cohere() -> Self {
        let mut template = Self::new(ChatTemplateType::Cohere);
        template.bos_token = Some("<BOS_TOKEN>".to_string());
        template
    }

    /// The chat template a model ships, from its Jinja `source` and the
    /// strings of its BOS and EOS tokens (`{{ bos_token }}`,
    /// `{{ eos_token }}`).
    ///
    /// The source renders every conversation. Its family (from the markers it
    /// writes; `model_name` tells Qwen's ChatML from the rest) sets the
    /// template type: the stop tokens of a model of that family, the
    /// reasoning levels it documents, and the built-in format a conversation
    /// it refuses is rendered in. A template of no family known here is
    /// [`ChatTemplateType::Custom`].
    pub fn from_jinja(
        source: impl Into<String>,
        bos_token: Option<String>,
        eos_token: Option<String>,
        model_name: &str,
    ) -> Self {
        let source = source.into();
        let mut template = detect_template_from_jinja(&source)
            .unwrap_or_else(|| Self::new(ChatTemplateType::Custom));
        if template.template_type == ChatTemplateType::ChatMl
            && model_name.to_lowercase().contains("qwen")
        {
            template = Self::qwen();
        }
        if bos_token.is_some() {
            template.bos_token = bos_token;
        }
        if let Some(eos) = eos_token {
            template.eos_token = eos;
        }
        template.jinja_source = Some(source);
        template
    }

    /// Set the default system message.
    pub fn with_system_message(mut self, message: impl Into<String>) -> Self {
        self.default_system_message = Some(message.into());
        self
    }

    /// Format a conversation for training: the text the model's own chat
    /// template renders it as, and where the model's output sits in it.
    ///
    /// A conversation ending in an assistant turn renders without a
    /// generation prompt, as the model is trained on it;
    /// [`FormattedChat::response_spans`] marks every assistant turn,
    /// [`FormattedChat::response_start`] the last. A conversation that ends
    /// on a user (or tool) turn renders with the generation prompt, as a
    /// prompt to continue, and has no spans.
    ///
    /// A conversation the template refuses (its own `raise_exception`: roles
    /// that must alternate, a system turn Gemma 2 has no place for) is
    /// rendered by the built-in format of the template type instead, with a
    /// warning. [`try_apply`](Self::try_apply) is the strict form that
    /// reports the refusal.
    pub fn apply(&self, messages: &[Message]) -> FormattedChat {
        self.apply_with_tools(messages, None)
    }

    /// [`apply`](Self::apply) with tool definitions, which the template
    /// places as it does for serving.
    pub fn apply_with_tools(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> FormattedChat {
        match self.try_apply(messages, tools, &ChatTemplateKwargs::new()) {
            Ok(formatted) => formatted,
            Err(e) => {
                tracing::warn!(
                    "{e}; rendering with the built-in {:?} format",
                    self.template_type
                );
                let builtin = Self {
                    jinja_source: None,
                    ..self.clone()
                };
                builtin
                    .try_apply(messages, tools, &ChatTemplateKwargs::new())
                    .unwrap_or_else(|e| {
                        tracing::warn!("{e}; rendering the conversation as plain text");
                        plain_conversation(messages, self.template_type)
                    })
            }
        }
    }

    /// Format a conversation for training, or say why the template refuses
    /// it. `kwargs` are the template's own variables (`enable_thinking`,
    /// `reasoning_effort`, …), as `apply_chat_template(**kwargs)` takes them.
    ///
    /// The prompt of the final assistant turn is the generation prompt the
    /// model is served with, and the response is what follows it, so the
    /// model is trained on the text it will see and write. Almost always that
    /// is simply how the template renders the conversation. Where it is not
    /// (Gemma 4 prompts with an empty thought channel its history turns
    /// leave out), the generation prompt is put in front of the turn's
    /// content.
    pub fn try_apply(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        kwargs: &ChatTemplateKwargs,
    ) -> Result<FormattedChat, String> {
        let Some((last, earlier)) = messages
            .split_last()
            .filter(|(last, _)| last.role == "assistant")
        else {
            let text = self.render(messages, tools, kwargs, true)?;
            return Ok(FormattedChat {
                response_start: text.len(),
                text,
                template_type: self.template_type,
                response_spans: Vec::new(),
                response_turns: Vec::new(),
            });
        };

        let TrainingRender {
            mut text,
            response_start,
            variants,
        } = self.render_final_turn(messages, last, earlier, tools, kwargs)?;
        let (mut turns, mut spans): (Vec<usize>, Vec<Range<usize>>) = self
            .earlier_assistant_spans(messages, tools, &variants, &text, response_start)
            .into_iter()
            .unzip();
        turns.push(messages.len() - 1);
        spans.push(response_start..text.len());

        // Harmony ends a conversation's final message with `<|return|>`, and
        // the `<|end|>` its template writes there would teach the model to
        // stop on a token that closes every other message too.
        if self.template_type == ChatTemplateType::GptOss {
            if let Some(body) = text.strip_suffix("<|end|>") {
                text.truncate(body.len());
                text.push_str("<|return|>");
                if let Some(span) = spans.last_mut() {
                    span.end = text.len();
                }
            }
        }

        Ok(FormattedChat {
            text,
            response_start,
            template_type: self.template_type,
            response_spans: spans,
            response_turns: turns,
        })
    }

    /// The text of a conversation ending in an assistant turn, and where that
    /// turn's response begins in it.
    ///
    /// A template with an `enable_thinking` switch the caller left unset is
    /// tried with thinking off, then on (on first for a turn that carries
    /// reasoning): the generation prompt of a non-thinking conversation ends
    /// in an empty `<think>` block serving supplies, which is not the
    /// model's to learn, while one with reasoning needs the prompt that stops
    /// before the block.
    fn render_final_turn(
        &self,
        messages: &[Message],
        last: &Message,
        earlier: &[Message],
        tools: Option<&[ToolDefinition]>,
        kwargs: &ChatTemplateKwargs,
    ) -> Result<TrainingRender, String> {
        let thinks = last
            .reasoning_content
            .as_deref()
            .is_some_and(|r| !r.is_empty())
            || last.content.contains("</think>");
        let variants: Vec<ChatTemplateKwargs> =
            if kwargs.get(ChatTemplateKwargs::ENABLE_THINKING).is_none()
                && self.reads_kwarg(ChatTemplateKwargs::ENABLE_THINKING)
            {
                let order = if thinks { [true, false] } else { [false, true] };
                order
                    .into_iter()
                    .map(|thinking| {
                        let mut k = kwargs.clone();
                        k.set(ChatTemplateKwargs::ENABLE_THINKING, thinking);
                        k
                    })
                    .collect()
            } else {
                vec![kwargs.clone()]
            };

        // The turn as the template renders it, for each variant.
        let mut renders = Vec::with_capacity(variants.len());
        let mut first_error = None;
        for k in &variants {
            match self.render(messages, tools, k, false) {
                Ok(text) => {
                    let prompt = self.render(earlier, tools, k, true);
                    renders.push((k, text, prompt.ok()));
                }
                Err(e) => {
                    first_error.get_or_insert(e);
                }
            }
        }

        // For the first variant that fits: the template's rendering of the
        // conversation, when the generation prompt is the start of it;
        // otherwise the generation prompt followed by the turn from where
        // the two part, provided they part inside the final turn.
        for (k, text, prompt) in &renders {
            let Some(prompt) = prompt else { continue };
            if text.starts_with(prompt.as_str()) {
                return Ok(TrainingRender {
                    response_start: prompt.len(),
                    text: text.clone(),
                    variants: chosen_first(&variants, k),
                });
            }
            let Ok(before) = self.render(earlier, tools, k, false) else {
                continue;
            };
            let shared = common_prefix_len(prompt, text);
            if text.starts_with(&before) && shared >= before.len() {
                return Ok(TrainingRender {
                    response_start: prompt.len(),
                    text: format!("{prompt}{}", &text[shared..]),
                    variants: chosen_first(&variants, k),
                });
            }
        }

        // The template renders the turn in a way its prompt does not lead to
        // at all. Train on the text from the turn's content.
        let Some((k, text, _)) = renders.first() else {
            return Err(first_error.unwrap_or_else(|| "no chat template to render".into()));
        };
        tracing::warn!(
            "the {:?} chat template does not render the final assistant turn as it prompts for \
             it; training on the text from the turn's content",
            self.template_type
        );
        let content = last.content.trim();
        let response_start = (!content.is_empty())
            .then(|| text.rfind(content))
            .flatten()
            .unwrap_or(text.len());
        Ok(TrainingRender {
            response_start,
            text: text.clone(),
            variants: chosen_first(&variants, k),
        })
    }

    /// The byte range of `text` that each assistant turn before the last
    /// wrote.
    ///
    /// A range starts where the generation prompt for that turn ends and ends
    /// where the conversation up to that turn ends. Both are found by
    /// rendering the shorter conversations and requiring them to be prefixes
    /// of `text`, which holds when the template writes a turn the same
    /// whatever follows it. A turn for which it does not (Qwen3 drops the
    /// thinking of every turn before the last user message, so serving would
    /// have written something else) is left out.
    fn earlier_assistant_spans(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        variants: &[ChatTemplateKwargs],
        text: &str,
        final_start: usize,
    ) -> Vec<(usize, Range<usize>)> {
        let mut spans: Vec<(usize, Range<usize>)> = Vec::new();
        for (index, message) in messages[..messages.len() - 1].iter().enumerate() {
            if message.role != "assistant" {
                continue;
            }
            let start = variants.iter().find_map(|k| {
                let prompt = self.render(&messages[..index], tools, k, true).ok()?;
                text.starts_with(&prompt).then_some(prompt.len())
            });
            let end = variants.iter().find_map(|k| {
                let turn = self.render(&messages[..=index], tools, k, false).ok()?;
                text.starts_with(&turn).then_some(turn.len())
            });
            if let (Some(start), Some(end)) = (start, end) {
                if start <= end
                    && end <= final_start
                    && spans.last().is_none_or(|(_, prev)| prev.end <= start)
                {
                    spans.push((index, start..end));
                }
            }
        }
        spans
    }

    /// Format an inference prompt with any template-specific generation prefill.
    ///
    /// Renders the model's own chat template (`tokenizer_config.json` or
    /// `chat_template.jinja`), as `AutoTokenizer.apply_chat_template` does;
    /// the built-in format of the template type when the model ships none, or
    /// when its template refuses the conversation.
    pub fn apply_inference(
        &self,
        messages: &[Message],
        no_thinking: bool,
        tools: Option<&[ToolDefinition]>,
    ) -> FormattedChat {
        let kwargs =
            ChatTemplateKwargs::new().with(ChatTemplateKwargs::ENABLE_THINKING, !no_thinking);
        self.apply_inference_with_kwargs(messages, tools, &kwargs)
            .unwrap_or_else(|e| {
                tracing::warn!("{e}; falling back to the built-in formatter");
                self.apply_builtin_inference(messages, tools, &kwargs)
            })
    }

    /// Format an inference prompt with template kwargs, as transformers'
    /// `apply_chat_template(messages, tools=tools, add_generation_prompt=True,
    /// **kwargs)` renders it.
    ///
    /// `kwargs` reach the upstream Jinja template verbatim
    /// (`enable_thinking`, `reasoning_effort`, `preserve_thinking`, …). A
    /// kwarg left out stays undefined, so the template's own default applies,
    /// as it does in transformers (Qwen3.8 thinks unless told not to; Gemma 4
    /// and Qwen3.5-0.8B do not unless told to).
    ///
    /// An assistant message without `reasoning_content` whose `content`
    /// still carries `<think>…</think>` (the raw text a thinking model
    /// generated) is split into the two, as Qwen3's template does itself,
    /// but only for templates that read `reasoning_content`: those render
    /// the thinking block from it, so leaving it inside `content` would
    /// print a second, nested one.
    ///
    /// Without a Jinja template the built-in format runs. A template that
    /// fails to render falls back the same way when the kwargs are
    /// `enable_thinking` alone; with any other kwarg the error is returned,
    /// because the built-in format cannot honour it and a template's own
    /// `raise_exception` (an unsupported `reasoning_effort` level, say) is
    /// the caller's to see.
    pub fn apply_inference_with_kwargs(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        kwargs: &ChatTemplateKwargs,
    ) -> Result<FormattedChat, String> {
        kwargs.check_reserved()?;
        let formatted = |text: String| FormattedChat {
            response_start: text.len(),
            text,
            template_type: self.template_type,
            response_spans: Vec::new(),
            response_turns: Vec::new(),
        };
        match self.render(messages, tools, kwargs, true) {
            Ok(text) => Ok(formatted(text)),
            Err(e)
                if self.jinja_source.is_some()
                    && kwargs
                        .keys()
                        .all(|k| k == ChatTemplateKwargs::ENABLE_THINKING) =>
            {
                tracing::warn!(
                    "Jinja chat template render failed, falling back to the built-in format: {e}"
                );
                Ok(self.apply_builtin_inference(messages, tools, kwargs))
            }
            Err(e) => Err(format!("chat template: {e}")),
        }
    }

    /// Whether the attached Jinja template reads `name` (a template kwarg
    /// such as `reasoning_effort`). `false` without a Jinja template.
    pub fn reads_kwarg(&self, name: &str) -> bool {
        let Some(src) = self.jinja_source.as_deref() else {
            return false;
        };
        let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
        src.match_indices(name).any(|(at, _)| {
            let before = src[..at].chars().next_back();
            let after = src[at + name.len()..].chars().next();
            !before.is_some_and(is_ident) && !after.is_some_and(is_ident)
        })
    }

    /// Whether the model thinks before it answers, as far as its Jinja chat
    /// template shows: it has a thinking control (`enable_thinking`,
    /// `reasoning_effort`, `thinking`) or writes a thinking block (`<think>`,
    /// `[THINK]`). `false` without a Jinja template.
    pub fn has_thinking(&self) -> bool {
        let Some(src) = self.jinja_source.as_deref() else {
            return false;
        };
        ["enable_thinking", "reasoning_effort", "thinking"]
            .iter()
            .any(|kwarg| self.reads_kwarg(kwarg))
            || src.contains("<think>")
            || src.contains("[THINK]")
    }

    /// Whether the model thinks under `kwargs`. It must [have
    /// thinking](Self::has_thinking); a template with an `enable_thinking`
    /// control thinks when that is `true`, and when it is unset, when the
    /// template's own default is on (Qwen3 and Qwen3.8 think by default,
    /// Gemma 4 and Qwen3.5-0.8B do not). That default is read by
    /// rendering a one-turn probe with and without `enable_thinking: true`.
    pub fn thinks_with(&self, kwargs: &ChatTemplateKwargs) -> bool {
        if !self.has_thinking() {
            return false;
        }
        if !self.reads_kwarg(ChatTemplateKwargs::ENABLE_THINKING) {
            return true;
        }
        if let Some(enabled) = kwargs.enable_thinking() {
            return enabled;
        }
        let probe = [Message::user("hi")];
        let mut on = kwargs.clone();
        on.set(ChatTemplateKwargs::ENABLE_THINKING, true);
        match (
            self.apply_inference_with_kwargs(&probe, None, kwargs),
            self.apply_inference_with_kwargs(&probe, None, &on),
        ) {
            (Ok(default), Ok(on)) => default.text == on.text,
            _ => true,
        }
    }

    /// Check caller-chosen thinking controls against this template, for a
    /// surface that should say so rather than ignore them (the CLI flags).
    ///
    /// Each of `reasoning_effort` / `preserve_thinking` that is set must be
    /// read by the template, and a `reasoning_effort` must be one of the
    /// documented levels where the template does not validate it itself
    /// (see [`documented_reasoning_effort_levels`]).
    pub fn check_thinking_controls(&self, kwargs: &ChatTemplateKwargs) -> Result<(), String> {
        for key in [
            ChatTemplateKwargs::REASONING_EFFORT,
            ChatTemplateKwargs::PRESERVE_THINKING,
        ] {
            if kwargs.get(key).is_some() && !self.reads_kwarg(key) {
                return Err(format!(
                    "this model's chat template has no `{key}` control ({:?} template)",
                    self.template_type
                ));
            }
        }
        self.check_reasoning_effort_level(kwargs)
    }

    /// Refuse a `reasoning_effort` outside the documented levels of a
    /// template that does not check it itself. Other kwargs pass.
    pub fn check_reasoning_effort_level(&self, kwargs: &ChatTemplateKwargs) -> Result<(), String> {
        let Some(effort) = kwargs.get(ChatTemplateKwargs::REASONING_EFFORT) else {
            return Ok(());
        };
        if let Some(levels) = documented_reasoning_effort_levels(self.template_type) {
            if !effort.as_str().is_some_and(|e| levels.contains(&e)) {
                return Err(format!(
                    "unsupported reasoning_effort {effort}: this model accepts {}",
                    levels.join(", ")
                ));
            }
        }
        Ok(())
    }

    /// The generation prompt in the built-in format of the template type,
    /// for models without a chat template of their own.
    fn apply_builtin_inference(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        kwargs: &ChatTemplateKwargs,
    ) -> FormattedChat {
        let builtin = Self {
            jinja_source: None,
            ..self.clone()
        };
        match builtin.render(messages, tools, kwargs, true) {
            Ok(text) => FormattedChat {
                response_start: text.len(),
                text,
                template_type: self.template_type,
                response_spans: Vec::new(),
                response_turns: Vec::new(),
            },
            Err(e) => {
                tracing::warn!("{e}; rendering the conversation as plain text");
                plain_conversation(messages, self.template_type)
            }
        }
    }

    /// [`apply_inference`](Self::apply_inference) through the upstream Jinja
    /// template only, with no fallback: an error when the checkpoint ships no
    /// template or it fails to render.
    ///
    /// Messages with [`parts`](Message::parts) reach the template as a list
    /// of items, so it places each image and video placeholder where the
    /// reference processor's `apply_chat_template` would. The built-in
    /// formats know nothing of media, which is why a prompt carrying media
    /// must come through here.
    pub fn render_inference_jinja(
        &self,
        messages: &[Message],
        no_thinking: bool,
        tools: Option<&[ToolDefinition]>,
    ) -> Result<String, String> {
        let kwargs =
            ChatTemplateKwargs::new().with(ChatTemplateKwargs::ENABLE_THINKING, !no_thinking);
        self.render_inference_jinja_with_kwargs(messages, tools, &kwargs)
    }

    /// [`render_inference_jinja`](Self::render_inference_jinja) with template
    /// kwargs, as [`apply_inference_with_kwargs`](Self::apply_inference_with_kwargs)
    /// passes them, and no fallback.
    pub fn render_inference_jinja_with_kwargs(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        kwargs: &ChatTemplateKwargs,
    ) -> Result<String, String> {
        kwargs.check_reserved()?;
        let src = self
            .jinja_source
            .as_deref()
            .ok_or("the model ships no chat template")?;
        self.render_jinja(src, messages, true, tools, kwargs)
    }

    /// Render a conversation through the chat template, as transformers'
    /// `apply_chat_template(messages, tools=tools,
    /// add_generation_prompt=add_generation_prompt, tokenize=False,
    /// **kwargs)` renders it.
    ///
    /// The template is the model's own when it ships one, else the built-in
    /// format of [`template_type`](Self::template_type) (the formats of `chat_templates/builtin.rs`).
    ///
    /// An `Err` is the template refusing the conversation (its own
    /// `raise_exception`, such as roles that do not alternate), failing to
    /// compile, or a built-in format asked for what it cannot do: tools it
    /// has no place for, or images and videos.
    pub fn render(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        kwargs: &ChatTemplateKwargs,
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        kwargs.check_reserved()?;
        match self.jinja_source.as_deref() {
            Some(src) => self.render_jinja(src, messages, add_generation_prompt, tools, kwargs),
            None => {
                if messages.iter().any(Message::has_media) {
                    return Err(
                        "the model ships no chat template, and the built-in formats \
                                have no place for images or videos"
                            .into(),
                    );
                }
                let mut kwargs = kwargs.clone();
                if let Some(system) = &self.default_system_message {
                    kwargs.set_default("default_system_message", system.as_str());
                }
                self.render_jinja(
                    builtin::source(self.template_type),
                    messages,
                    add_generation_prompt,
                    tools,
                    &kwargs,
                )
            }
        }
    }

    /// Render `src`, a Jinja chat template, over the conversation.
    fn render_jinja(
        &self,
        src: &str,
        messages: &[Message],
        add_generation_prompt: bool,
        tools: Option<&[ToolDefinition]>,
        kwargs: &ChatTemplateKwargs,
    ) -> Result<String, String> {
        // Raw `<think>…</think>` text in an assistant message moves to
        // `reasoning_content` only for a template that renders that field as
        // a `<think>` block. Gemma 4's reads the field too, but writes its
        // thinking as a thought channel, so a `<think>` in the content is
        // just content there.
        let split_thinking = self.reads_kwarg("reasoning_content") && src.contains("</think>");

        // Map pmetal's internal Message into the shape HF Jinja templates
        // expect. We carry tool_calls / tool_call_id through verbatim.
        let jinja_messages: Vec<crate::jinja_chat::JinjaMessage> = messages
            .iter()
            .map(|m| {
                let (reasoning_content, content) = match (&m.reasoning_content, split_thinking) {
                    (Some(r), _) => (Some(r.clone()), m.content.clone()),
                    (None, true) if m.role == "assistant" && m.parts.is_none() => {
                        match split_reasoning_content(&m.content) {
                            Some((r, c)) => (Some(r), c),
                            None => (None, m.content.clone()),
                        }
                    }
                    (None, _) => (None, m.content.clone()),
                };
                crate::jinja_chat::JinjaMessage {
                    role: m.role.clone(),
                    content: match &m.parts {
                        Some(parts) => serde_json::Value::Array(
                            parts.iter().map(ContentPart::to_template_value).collect(),
                        ),
                        None => serde_json::Value::String(content),
                    },
                    reasoning_content,
                    tool_calls: m.tool_calls.as_ref().map(|calls| {
                        calls
                            .iter()
                            .map(|c| {
                                serde_json::json!({
                                    "id": c.id,
                                    "type": "function",
                                    "function": {
                                        "name": c.function.name,
                                        "arguments": tool_call_arguments(&c.function.arguments),
                                    },
                                })
                            })
                            .collect()
                    }),
                    tool_call_id: m.tool_call_id.clone(),
                }
            })
            .collect();

        // Tools go in as the caller wrote them: `{"type": "function",
        // "function": {"name", "description"?, "parameters"?}}`.
        let tools_json: Option<Vec<serde_json::Value>> = tools.map(|tool_defs| {
            tool_defs
                .iter()
                .map(|td| serde_json::to_value(td).unwrap_or(serde_json::Value::Null))
                .collect()
        });

        let options = crate::jinja_chat::JinjaRenderOptions {
            add_generation_prompt,
            enable_thinking: None,
            tools: tools_json,
            bos_token: self.bos_token.clone(),
            eos_token: Some(self.eos_token.clone()),
            template_kwargs: kwargs.as_map().clone(),
        };

        crate::jinja_chat::render_chat_template(src, &jinja_messages, &options)
    }
}

/// A conversation rendered for training: the text, where the final response
/// begins in it, and the template kwargs (the thinking switch among them)
/// the rendering was made with, the one that fits first.
struct TrainingRender {
    text: String,
    response_start: usize,
    variants: Vec<ChatTemplateKwargs>,
}

/// `variants` with `chosen` moved to the front.
fn chosen_first(
    variants: &[ChatTemplateKwargs],
    chosen: &ChatTemplateKwargs,
) -> Vec<ChatTemplateKwargs> {
    let mut ordered = vec![chosen.clone()];
    ordered.extend(variants.iter().filter(|v| *v != chosen).cloned());
    ordered
}

/// Length in bytes of the longest common prefix of `a` and `b` that ends on a
/// character boundary of both.
fn common_prefix_len(a: &str, b: &str) -> usize {
    a.char_indices()
        .zip(b.chars())
        .find(|((_, x), y)| x != y)
        .map_or_else(|| a.len().min(b.len()), |((at, _), _)| at)
}

/// A conversation as `role: content` lines, for a template that refuses it
/// and a built-in format that cannot take it either. The last assistant turn
/// is the response.
fn plain_conversation(messages: &[Message], template_type: ChatTemplateType) -> FormattedChat {
    let mut text = String::new();
    let mut response_spans = Vec::new();
    let mut response_turns = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        text.push_str(&message.role);
        text.push_str(": ");
        let start = text.len();
        text.push_str(&message.content);
        if message.role == "assistant" {
            response_spans.push(start..text.len());
            response_turns.push(index);
        }
        text.push_str("\n\n");
    }
    let ends_with_assistant = messages.last().is_some_and(|m| m.role == "assistant");
    if !ends_with_assistant {
        text.push_str("assistant: ");
        response_spans.clear();
        response_turns.clear();
    }
    let response_start = match (ends_with_assistant, response_spans.last()) {
        (true, Some(span)) => span.start,
        _ => text.len(),
    };
    FormattedChat {
        text,
        response_start,
        template_type,
        response_spans,
        response_turns,
    }
}

/// Detect the appropriate chat template by inspecting the model's `tokenizer_config.json` Jinja
/// string first, then falling back to model-name heuristics.
///
/// This is the preferred entry point — it gives consistent results across training, inference,
/// and distillation because the same Jinja patterns are checked everywhere.
pub fn detect_chat_template(model_path: &std::path::Path, model_name: &str) -> ChatTemplate {
    // 1. The Jinja template: a standalone `chat_template.jinja`, which
    //    transformers writes since v4.43 and reads ahead of the config (the
    //    gpt-oss release ships its template only there), else the
    //    `chat_template` in tokenizer_config.json.
    let config_path = model_path.join("tokenizer_config.json");
    let json = std::fs::read_to_string(&config_path)
        .ok()
        .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok())
        .unwrap_or(serde_json::Value::Null);
    let standalone = std::fs::read_to_string(model_path.join("chat_template.jinja")).ok();
    if let Some(jinja) = standalone.or_else(|| extract_jinja_template(&json)) {
        // Pull bos / eos token strings from the tokenizer config (else the
        // special-tokens map) so the Jinja renderer can substitute
        // `{{ bos_token }}` and `{{ eos_token }}` exactly like HF does.
        let special_tokens_map =
            std::fs::read_to_string(model_path.join("special_tokens_map.json"))
                .ok()
                .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok())
                .unwrap_or(serde_json::Value::Null);
        let token_string = |key| {
            extract_token_string_field(&json, key)
                .or_else(|| extract_token_string_field(&special_tokens_map, key))
        };
        let model_type = json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let template = ChatTemplate::from_jinja(
            jinja,
            token_string("bos_token"),
            token_string("eos_token"),
            &format!("{model_name} {model_type}"),
        );
        tracing::info!(
            "Chat template: {:?} (jinja attached)",
            template.template_type
        );
        return template;
    }

    // 1b. A GGUF carries its chat template in `tokenizer.chat_template`.
    if let Some(gguf) = crate::tokenizer::gguf_chat_template(model_path) {
        let template =
            ChatTemplate::from_jinja(gguf.template, gguf.bos_token, gguf.eos_token, model_name);
        tracing::info!(
            "Chat template: {:?} (jinja from GGUF metadata)",
            template.template_type
        );
        return template;
    }

    // 2. Fall back to model-name heuristics
    let template = detect_template_from_model(model_name);
    tracing::info!(
        "Chat template: {:?} (from model name, no jinja)",
        template.template_type
    );
    template
}

/// Extract the string form of a special token field (`bos_token`,
/// `eos_token`, …) from a parsed `tokenizer_config.json`. Handles both
/// the plain-string shape (`"bos_token": "<s>"`) and the object shape
/// (`"bos_token": {"content": "<s>", ...}`). Returns `None` for null /
/// missing / unrecognised shapes.
fn extract_token_string_field(json: &serde_json::Value, key: &str) -> Option<String> {
    let field = json.get(key)?;
    if let Some(s) = field.as_str() {
        return Some(s.to_string());
    }
    if let Some(obj) = field.as_object() {
        if let Some(content) = obj.get("content").and_then(|v| v.as_str()) {
            return Some(content.to_string());
        }
    }
    None
}

/// Extract the Jinja template string from the `chat_template` field in tokenizer_config.json.
///
/// Handles two formats:
/// - **String**: `"chat_template": "<jinja string>"` — returned directly.
/// - **Array**: `"chat_template": [{"name": "default", "template": "..."}, ...]` —
///   returns the `"default"` entry if present, otherwise the first entry.
fn extract_jinja_template(json: &serde_json::Value) -> Option<String> {
    let field = json.get("chat_template")?;

    // Simple string
    if let Some(s) = field.as_str() {
        return Some(s.to_string());
    }

    // Array of {name, template} objects (HuggingFace Transformers ≥ v4.39)
    if let Some(arr) = field.as_array() {
        // Prefer "default", fall back to first entry
        let default = arr.iter().find(|obj| {
            obj.get("name")
                .and_then(|n| n.as_str())
                .is_some_and(|n| n == "default")
        });
        let chosen = default.or_else(|| arr.first())?;
        return chosen
            .get("template")
            .and_then(|t| t.as_str())
            .map(String::from);
    }

    None
}

/// Detect a chat template from a Jinja template string found in `tokenizer_config.json`.
///
/// Order matters — more specific patterns (Phi4's `<|im_sep|>`) are checked before
/// generic ones (ChatML's `<|im_start|>`).
fn detect_template_from_jinja(jinja: &str) -> Option<ChatTemplate> {
    // Phi4: uses <|im_start|> + <|im_sep|> (distinct from plain ChatML)
    if jinja.contains("<|im_start|>") && jinja.contains("<|im_sep|>") {
        return Some(ChatTemplate::new(ChatTemplateType::Phi4));
    }
    // DeepSeek: full-width unicode pipe character ｜ (U+FF5C)
    if jinja.contains("｜") || jinja.contains("<｜") {
        return Some(ChatTemplate::deepseek());
    }
    // Cohere Command R: <|START_OF_TURN_TOKEN|>
    if jinja.contains("<|START_OF_TURN_TOKEN|>") || jinja.contains("<|CHATBOT_TOKEN|>") {
        return Some(ChatTemplate::cohere());
    }
    // ChatML / Qwen: <|im_start|> without <|im_sep|>
    if jinja.contains("<|im_start|>") {
        return Some(ChatTemplate::chatml());
    }
    // Llama-4: <|header_start|> (must check before Llama-3's <|start_header_id|>)
    if jinja.contains("<|header_start|>") {
        return Some(ChatTemplate::llama4());
    }
    // Llama-3: <|begin_of_text|> or <|start_header_id|>
    if jinja.contains("<|begin_of_text|>") || jinja.contains("<|start_header_id|>") {
        return Some(ChatTemplate::llama3());
    }
    // Llama-2: <<SYS>> (must come before Mistral which also uses [INST])
    if jinja.contains("<<SYS>>") {
        return Some(ChatTemplate::llama2());
    }
    // Gemma 4: <|turn> / <turn|> (asymmetric pipes, role=model). Must be
    // checked before the Gemma 2/3 `<start_of_turn>` match because Gemma 4's
    // template does NOT contain `<start_of_turn>`.
    if jinja.contains("<|turn>") && jinja.contains("<turn|>") {
        return Some(ChatTemplate::gemma4());
    }
    // Gemma 2/3: <start_of_turn>
    if jinja.contains("<start_of_turn>") {
        return Some(ChatTemplate::gemma());
    }
    // Mistral: [INST] without <<SYS>>
    if jinja.contains("[INST]") {
        return Some(ChatTemplate::mistral());
    }
    // Phi-3: `<|user|>` + `<|end|>`. We also accept the concatenation
    // variant `'<|' + message['role'] + '|>'` that Phi-4-mini uses — it
    // still renders Phi-3-style markers at runtime, and the `<|end|>` +
    // `<|assistant|>` literals are present in the jinja source.
    if (jinja.contains("<|user|>") || jinja.contains("'<|' + message['role']"))
        && jinja.contains("<|end|>")
        && jinja.contains("<|assistant|>")
    {
        return Some(ChatTemplate::phi3());
    }
    // GPT-OSS Harmony: <|start|> + <|message|>
    if jinja.contains("<|start|>") && jinja.contains("<|message|>") {
        return Some(ChatTemplate::gpt_oss());
    }
    // Zephyr: `<|user|>` / `<|assistant|>` turns closed by the EOS token
    // (Phi-3's close with `<|end|>`, which was checked above).
    if jinja.contains("<|user|>") && jinja.contains("<|assistant|>") {
        return Some(ChatTemplate::new(ChatTemplateType::Zephyr));
    }
    // Alpaca: ### Instruction
    if jinja.contains("### Instruction") {
        return Some(ChatTemplate::alpaca());
    }
    // Vicuna: USER: + ASSISTANT:
    if jinja.contains("USER:") && jinja.contains("ASSISTANT:") {
        return Some(ChatTemplate::new(ChatTemplateType::Vicuna));
    }
    None
}

/// Detect the appropriate chat template from a model name.
pub fn detect_template_from_model(model_name: &str) -> ChatTemplate {
    let name_lower = model_name.to_lowercase();

    // GPT-OSS / Harmony format (OpenAI's first open-weight models)
    if name_lower.contains("gpt-oss")
        || name_lower.contains("gptoss")
        || name_lower.contains("gpt_oss")
        || name_lower.contains("openai/gpt")
        || name_lower.contains("harmony")
    {
        ChatTemplate::gpt_oss()
    } else if name_lower.contains("deepseek") {
        ChatTemplate::deepseek()
    } else if name_lower.contains("cohere")
        || name_lower.contains("command-r")
        || name_lower.contains("command_r")
    {
        ChatTemplate::cohere()
    } else if name_lower.contains("llama-4") || name_lower.contains("llama4") {
        ChatTemplate::llama4()
    } else if name_lower.contains("llama-3") || name_lower.contains("llama3") {
        ChatTemplate::llama3()
    } else if name_lower.contains("llama-2") || name_lower.contains("llama2") {
        ChatTemplate::llama2()
    } else if name_lower.contains("llama") {
        // Default Llama to Llama-3 format
        ChatTemplate::llama3()
    } else if name_lower.contains("mistral") || name_lower.contains("mixtral") {
        ChatTemplate::mistral()
    } else if name_lower.contains("gemma-4")
        || name_lower.contains("gemma_4")
        || name_lower.contains("gemma4")
    {
        // Gemma 4 must come before the generic "gemma" match: its chat
        // template uses `<|turn>` / `<turn|>`, NOT the Gemma 2/3 markers.
        ChatTemplate::gemma4()
    } else if name_lower.contains("gemma") {
        ChatTemplate::gemma()
    } else if name_lower.contains("phi-4") || name_lower.contains("phi4") {
        ChatTemplate::new(ChatTemplateType::Phi4)
    } else if name_lower.contains("phi-3")
        || name_lower.contains("phi3")
        || name_lower.contains("phi")
    {
        ChatTemplate::phi3()
    } else if name_lower.contains("qwen") {
        ChatTemplate::qwen()
    } else if name_lower.contains("vicuna") {
        ChatTemplate::new(ChatTemplateType::Vicuna)
    } else if name_lower.contains("zephyr") {
        ChatTemplate::new(ChatTemplateType::Zephyr)
    } else if name_lower.contains("alpaca") {
        ChatTemplate::alpaca()
    } else {
        // Default to ChatML
        ChatTemplate::chatml()
    }
}

impl FormattedChat {
    /// Which tokens the model wrote, given the byte span of the text each
    /// token covers ([`TokenizedText::offsets`](crate::TokenizedText::offsets)).
    ///
    /// A token counts only if it lies wholly inside one of the
    /// [`response_spans`](Self::response_spans) and is not empty (a BOS the
    /// tokenizer added covers nothing); a token that straddles the edge of a
    /// span began in the prompt and is masked.
    pub fn trainable_tokens(&self, offsets: &[(usize, usize)]) -> Vec<bool> {
        tokens_inside(&self.response_spans, offsets)
    }
}

/// Which tokens lie wholly inside one of `spans` (sorted, disjoint byte
/// ranges), given each token's byte span of the text.
///
/// A token that covers nothing (a BOS the tokenizer added) is outside every
/// span, and one that straddles the edge of a span is outside it. This is
/// the one rule that turns a loss mask in bytes into a loss mask over tokens,
/// for conversations ([`FormattedChat::trainable_tokens`]) and for the
/// prompt/completion and document samples of dataset loading alike.
pub(crate) fn tokens_inside(spans: &[Range<usize>], offsets: &[(usize, usize)]) -> Vec<bool> {
    let mut spans = spans.iter().peekable();
    offsets
        .iter()
        .map(|&(start, end)| {
            // Tokens come in text order, so spans already passed are done.
            while spans.next_if(|span| span.end <= start).is_some() {}
            end > start
                && spans
                    .peek()
                    .is_some_and(|span| span.start <= start && end <= span.end)
        })
        .collect()
}

/// Builder for creating training samples with response masking.
#[derive(Debug, Clone)]
pub struct TrainingSampleBuilder {
    template: ChatTemplate,
}

impl TrainingSampleBuilder {
    /// Create a new builder with the given template.
    pub fn new(template: ChatTemplate) -> Self {
        Self { template }
    }

    /// Build a training sample from messages.
    ///
    /// Returns the formatted text and the byte offset where the final
    /// response begins.
    pub fn build(&self, messages: &[Message]) -> (String, usize) {
        let formatted = self.template.apply(messages);
        (formatted.text, formatted.response_start)
    }

    /// Build a training sample and tokenize it with label masking.
    ///
    /// Returns `(input_ids, labels)`; a label is `-100` for every token the
    /// model did not write (the system and user turns, the role headers) and
    /// the token id for each token of an assistant turn, through the token
    /// that ends it. All the assistant turns of the conversation that the
    /// template renders the same way with or without later turns are
    /// trained, not just the last.
    ///
    /// The text is tokenized once, without the tokenizer's own BOS (the
    /// template wrote the model's), and the masks come from each token's span
    /// of that text, so a token cannot be counted into the wrong side of the
    /// prompt/response boundary by tokenizing the prompt separately.
    pub fn build_tokenized(
        &self,
        messages: &[Message],
        tokenizer: &super::Tokenizer,
        max_length: usize,
    ) -> pmetal_core::Result<(Vec<u32>, Vec<i64>)> {
        let formatted = self
            .template
            .try_apply(messages, None, &ChatTemplateKwargs::new())
            .map_err(pmetal_core::PMetalError::Tokenizer)?;
        Self::tokenize(&formatted, tokenizer, max_length)
    }

    /// Tokenize a formatted conversation with label masking, as
    /// [`build_tokenized`](Self::build_tokenized) does, truncating to
    /// `max_length` tokens.
    pub fn tokenize(
        formatted: &FormattedChat,
        tokenizer: &super::Tokenizer,
        max_length: usize,
    ) -> pmetal_core::Result<(Vec<u32>, Vec<i64>)> {
        let mut encoded = tokenizer.encode_with_offsets(&formatted.text, false)?;
        encoded.ids.truncate(max_length);
        encoded.offsets.truncate(max_length);
        let trainable = formatted.trainable_tokens(&encoded.offsets);
        let labels = encoded
            .ids
            .iter()
            .zip(&trainable)
            .map(|(&id, &train)| if train { i64::from(id) } else { -100 })
            .collect();
        Ok((encoded.ids, labels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The gpt-oss release keeps its template in a standalone
    /// `chat_template.jinja` (tokenizer_config.json has none), which
    /// transformers reads ahead of the config. It was ignored, so gpt-oss
    /// prompts were built by a hand-written fallback.
    #[test]
    fn a_standalone_chat_template_jinja_is_read_first() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"eos_token": "<|return|>", "chat_template": "{{ 'from the config' }}"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{{ '<|start|>assistant' }}",
        )
        .unwrap();
        let template = detect_chat_template(dir.path(), "gpt-oss-20b");
        assert_eq!(
            template.jinja_source.as_deref(),
            Some("{{ '<|start|>assistant' }}")
        );
        assert_eq!(template.eos_token, "<|return|>");

        std::fs::remove_file(dir.path().join("chat_template.jinja")).unwrap();
        let template = detect_chat_template(dir.path(), "gpt-oss-20b");
        assert_eq!(
            template.jinja_source.as_deref(),
            Some("{{ 'from the config' }}")
        );
    }

    /// A template of no family this knows is `Custom` (it used to be called
    /// ChatML, and Granite's `<|start_of_role|>` template was detected as
    /// one), and the special tokens come from the tokenizer files.
    #[test]
    fn an_unrecognised_template_is_custom_with_its_own_special_tokens() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template": "<|start_of_role|>{{ messages[0].role }}<|end_of_role|>"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("special_tokens_map.json"),
            r#"{"bos_token": "<bos>", "eos_token": {"content": "<|end_of_text|>"}}"#,
        )
        .unwrap();
        let template = detect_chat_template(dir.path(), "granite-3.1-2b-instruct");
        assert_eq!(template.template_type, ChatTemplateType::Custom);
        assert_eq!(template.eos_token, "<|end_of_text|>");
        assert_eq!(template.bos_token.as_deref(), Some("<bos>"));
    }

    /// A message's parts reach the template as the list of items the
    /// reference processor gives it; Qwen 3.5's `render_content` (verbatim
    /// below, minus the vision-id and system checks) puts each placeholder
    /// where its item is. String content still renders as a string.
    #[test]
    fn test_media_parts_render_where_the_template_places_them() {
        let mut template = ChatTemplate::chatml();
        template.jinja_source = Some(
            r#"{%- macro render_content(content) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
{%- endmacro %}
{%- for message in messages %}<|im_start|>{{ message.role }}
{{ render_content(message.content) }}<|im_end|>
{% endfor %}<|im_start|>assistant
"#
            .into(),
        );
        let messages = [
            Message::system("Be brief."),
            Message::with_parts(
                "user",
                vec![
                    ContentPart::Text("Compare ".into()),
                    ContentPart::Image,
                    ContentPart::Text(" with ".into()),
                    ContentPart::Video,
                ],
            ),
        ];
        assert!(!messages[0].has_media());
        assert!(messages[1].has_media());
        assert_eq!(messages[1].content, "Compare  with ");
        let rendered = template
            .render_inference_jinja(&messages, false, None)
            .unwrap();
        assert_eq!(
            rendered,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nCompare \
             <|vision_start|><|image_pad|><|vision_end|> with \
             <|vision_start|><|video_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant"
        );
        // Without the model's own template, a prompt with media has nowhere
        // to go: the built-in formats do not guess at placeholders.
        assert!(
            ChatTemplate::chatml()
                .render_inference_jinja(&messages, false, None)
                .is_err()
        );
        assert!(
            ChatTemplate::chatml()
                .render(&messages, None, &ChatTemplateKwargs::new(), true)
                .is_err()
        );
    }

    #[test]
    fn test_detect_chat_template_upgrades_qwen_chatml_from_model_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template":"<|im_start|>{{ messages }}<|im_end|>"}"#,
        )
        .unwrap();

        let detected = detect_chat_template(dir.path(), "Qwen/Qwen3.5-0.8B");
        assert_eq!(detected.template_type, ChatTemplateType::Qwen);
    }

    #[test]
    fn test_qwen_inference_prompt_adds_empty_think_block_when_disabled() {
        let template = ChatTemplate::qwen();
        let messages = vec![Message::user("write fizzbuzz")];
        let formatted = template.apply_inference(&messages, true, None);
        assert!(
            formatted
                .text
                .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        );
        assert_eq!(formatted.response_start, formatted.text.len());
    }

    #[test]
    fn test_qwen_inference_prompt_adds_open_think_block_when_enabled() {
        let template = ChatTemplate::qwen();
        let messages = vec![Message::user("write fizzbuzz")];
        let formatted = template.apply_inference(&messages, false, None);
        assert!(formatted.text.ends_with("<|im_start|>assistant\n<think>\n"));
        assert_eq!(formatted.response_start, formatted.text.len());
    }

    #[test]
    fn test_qwen_inference_prompt_does_not_inject_default_system_message() {
        let template = ChatTemplate::qwen();
        let messages = vec![Message::user("write fizzbuzz")];
        let formatted = template.apply_inference(&messages, true, None);
        assert_eq!(
            formatted.text,
            "<|im_start|>user\nwrite fizzbuzz<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    /// Qwen 3's template, reduced to the parts a single exchange reaches:
    /// the last assistant turn carries a `<think>` block, empty without
    /// reasoning, and the generation prompt is the empty block in
    /// non-thinking mode.
    const QWEN3_REASONING_TEMPLATE: &str = r#"{%- for message in messages %}
    {%- set content = message.content %}
    {%- if message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- elif '</think>' in content %}
            {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
            {%- set content = content.split('</think>')[-1].lstrip('\n') %}
        {%- endif %}
        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
        {{- '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"#;

    /// A training conversation renders the way the template writes it, so
    /// the answer follows the empty `<think>` block serving prompts with
    /// rather than the bare assistant header.
    #[test]
    fn a_reasoning_template_trains_the_answer_after_its_think_block() {
        let mut template = ChatTemplate::qwen();
        template.jinja_source = Some(QWEN3_REASONING_TEMPLATE.to_string());
        let messages = vec![
            Message::user("What is 39 + 28?"),
            Message::assistant("Answer: 67."),
        ];
        let formatted = template.apply(&messages);
        assert_eq!(
            formatted.text,
            "<|im_start|>user\nWhat is 39 + 28?<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: 67.<|im_end|>\n"
        );
        assert!(formatted.prompt().ends_with("<think>\n\n</think>\n\n"));
        assert_eq!(formatted.response(), "Answer: 67.<|im_end|>\n");

        // Reasoning in the answer is trained, after the opening tag.
        let messages = vec![
            Message::user("What is 39 + 28?"),
            Message::assistant("<think>\n39 + 28 = 67\n</think>\n\nAnswer: 67."),
        ];
        let formatted = template.apply(&messages);
        assert!(formatted.prompt().ends_with("<|im_start|>assistant\n"));
        assert_eq!(
            formatted.response(),
            "<think>\n39 + 28 = 67\n</think>\n\nAnswer: 67.<|im_end|>\n"
        );
    }

    /// Every assistant turn a template writes the same whatever follows it is
    /// trained, and nothing else is: the system and user turns, the role
    /// headers, and a turn whose rendering changes once later turns follow.
    #[test]
    fn every_assistant_turn_the_template_writes_stably_is_a_response_span() {
        let template = ChatTemplate::chatml();
        let messages = vec![
            Message::system("Be brief."),
            Message::user("Hi"),
            Message::assistant("Hello"),
            Message::user("2+2?"),
            Message::assistant("4"),
        ];
        let formatted = template.apply(&messages);
        let spans: Vec<&str> = formatted
            .response_spans
            .iter()
            .map(|s| &formatted.text[s.clone()])
            .collect();
        assert_eq!(spans, ["Hello<|im_end|>\n", "4<|im_end|>\n"]);
        // The messages the spans are the assistant turns of.
        assert_eq!(formatted.response_turns, [2, 4]);
        assert_eq!(formatted.response(), "4<|im_end|>\n");

        // A reasoning template writes the empty think block of every turn the
        // same way, and that block is part of the prompt serving supplies.
        let mut qwen3 = ChatTemplate::qwen();
        qwen3.jinja_source = Some(QWEN3_REASONING_TEMPLATE.to_string());
        let formatted = qwen3.apply(&messages);
        let spans: Vec<&str> = formatted
            .response_spans
            .iter()
            .map(|s| &formatted.text[s.clone()])
            .collect();
        assert_eq!(spans, ["Hello<|im_end|>\n", "4<|im_end|>\n"]);
    }

    /// The labels of a tokenized training conversation are the ids of the
    /// assistant's tokens and nothing else, through the end-of-turn token; and
    /// the template's BOS is the only one (the tokenizer's post-processor adds
    /// another to whatever it is asked to encode with specials, which is how
    /// every Llama 3 sample began with two).
    #[test]
    fn labels_cover_the_assistant_turns_and_the_text_has_one_bos() {
        use crate::tokenizer::testing::byte_level_bpe;
        let tokenizer = byte_level_bpe(
            &[],
            &[
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ],
            Some("<|begin_of_text|>"),
        );
        let bos = tokenizer.inner().token_to_id("<|begin_of_text|>").unwrap();
        let eot = tokenizer.inner().token_to_id("<|eot_id|>").unwrap();
        let messages = [
            Message::system("Be brief."),
            Message::user("Hi"),
            Message::assistant("Hello"),
            Message::user("2+2?"),
            Message::assistant("4"),
        ];
        let (ids, labels) = TrainingSampleBuilder::new(ChatTemplate::llama3())
            .build_tokenized(&messages, &tokenizer, usize::MAX)
            .unwrap();
        assert_eq!(ids.iter().filter(|&&id| id == bos).count(), 1);
        assert_eq!(ids[0], bos);

        let trained: Vec<u32> = ids
            .iter()
            .zip(&labels)
            .filter(|&(_, &label)| label != -100)
            .map(|(&id, _)| id)
            .collect();
        // "Hello" and "4", each closed by <|eot_id|>: bytes are one token each.
        let bytes = |text: &str| -> Vec<u32> { tokenizer.encode(text).unwrap() };
        let expected: Vec<u32> = [bytes("Hello"), vec![eot], bytes("4"), vec![eot]].concat();
        assert_eq!(trained, expected);
        assert!(
            labels
                .iter()
                .all(|&l| l == -100 || ids.contains(&(l as u32)))
        );
        // Truncation keeps the first tokens, labels included.
        let (short_ids, short_labels) = TrainingSampleBuilder::new(ChatTemplate::llama3())
            .build_tokenized(&messages, &tokenizer, 10)
            .unwrap();
        assert_eq!(short_ids, ids[..10]);
        assert_eq!(short_labels, labels[..10]);
    }

    #[test]
    fn a_conversation_ending_on_the_user_is_a_prompt() {
        let formatted = ChatTemplate::chatml().apply(&[Message::user("Hi")]);
        assert_eq!(
            formatted.text,
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(formatted.response_spans.is_empty());
        assert_eq!(formatted.response_start, formatted.text.len());
        assert_eq!(formatted.response(), "");
    }

    #[test]
    fn harmony_trains_the_final_message_to_end_in_return() {
        let formatted =
            ChatTemplate::gpt_oss().apply(&[Message::user("Hi"), Message::assistant("Hello")]);
        assert!(
            formatted.text.ends_with("Hello<|return|>"),
            "{}",
            formatted.text
        );
        assert_eq!(formatted.response(), "Hello<|return|>");
        assert!(!formatted.text.contains("<|end|><|return|>"));
    }

    #[test]
    fn test_detect_template() {
        assert_eq!(
            detect_template_from_model("meta-llama/Llama-3.1-8B").template_type,
            ChatTemplateType::Llama3
        );
        assert_eq!(
            detect_template_from_model("mistralai/Mistral-7B-v0.1").template_type,
            ChatTemplateType::Mistral
        );
        assert_eq!(
            detect_template_from_model("google/gemma-2-9b").template_type,
            ChatTemplateType::Gemma
        );
        assert_eq!(
            detect_template_from_model("Qwen/Qwen2-7B").template_type,
            ChatTemplateType::Qwen
        );
    }

    #[test]
    fn test_gpt_oss_eos_token() {
        assert_eq!(ChatTemplateType::GptOss.eos_token(), "<|return|>");
    }

    #[test]
    fn test_detect_gpt_oss_template() {
        assert_eq!(
            detect_template_from_model("openai/gpt-oss-20b").template_type,
            ChatTemplateType::GptOss
        );
        assert_eq!(
            detect_template_from_model("gptoss-120b-instruct").template_type,
            ChatTemplateType::GptOss
        );
        assert_eq!(
            detect_template_from_model("harmony-chat-model").template_type,
            ChatTemplateType::GptOss
        );
    }
}
