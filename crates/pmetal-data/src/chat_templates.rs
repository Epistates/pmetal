//! Chat template system for formatting conversations.
//!
//! Provides model-specific chat templates for SFT training and inference with:
//! - Proper formatting for each model family (Llama, Mistral, Gemma, etc.)
//! - Response masking support (identifying which tokens to train on)
//! - System message customization
//! - Tool/function calling support (Qwen, Llama 3.1+, Mistral v3+, ChatML)

use serde::{Deserialize, Serialize};

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

/// A single message in a conversation.
#[derive(Debug, Clone)]
pub struct Message {
    /// Role: "system", "user", "assistant", or "tool"
    pub role: String,
    /// Content of the message.
    pub content: String,
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
            tool_calls: None,
            tool_call_id: None,
        }
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
            role: "assistant".into(),
            content: content.into(),
            tool_calls: Some(tool_calls),
            tool_call_id: None,
        }
    }

    /// Create a tool response message.
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    /// Check if this message contains tool calls.
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty())
    }
}

// ============================================================================
// Tool formatting helpers (shared across templates)
// ============================================================================

/// Format tool definitions as a JSON list for injection into system prompts.
///
/// Used by Qwen and ChatML templates. Produces:
/// ```text
/// # Tools
///
/// You may call one or more functions to assist with the user query.
///
/// You are provided with function signatures within <tools></tools> XML tags:
/// <tools>
/// {"type": "function", "function": {"name": "...", ...}}
/// </tools>
///
/// For each function call, return a json object with function name and arguments
/// within <tool_call></tool_call> XML tags:
/// <tool_call>
/// {"name": <function-name>, "arguments": <args-json-object>}
/// </tool_call>
/// ```
fn format_tools_qwen(tools: &[ToolDefinition]) -> String {
    let mut s = String::from(
        "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>",
    );
    for tool in tools {
        s.push('\n');
        s.push_str(&serde_json::to_string(tool).unwrap_or_default());
    }
    s.push_str("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
    s
}

/// Format tool definitions for Llama 3.1+ system prompt.
///
/// Produces:
/// ```text
/// You have access to the following functions. To call a function, please respond
/// with JSON for a function call.
/// Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.
/// Do not use variables.
///
/// {tool_json}
/// ```
fn format_tools_llama3(tools: &[ToolDefinition]) -> String {
    let mut s = String::from(
        "You have access to the following functions. To call a function, please respond with JSON for a function call.\nRespond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.\nDo not use variables.\n\n",
    );
    for tool in tools {
        if let Ok(json) = serde_json::to_string_pretty(tool) {
            s.push_str(&json);
            s.push_str("\n\n");
        }
    }
    s
}

/// Format tool definitions for Mistral v3+ templates.
///
/// Produces: `[AVAILABLE_TOOLS] [{"type":"function","function":{...}}, ...] [/AVAILABLE_TOOLS]`
fn format_tools_mistral(tools: &[ToolDefinition]) -> String {
    let json = serde_json::to_string(tools).unwrap_or_else(|_| "[]".to_string());
    format!("[AVAILABLE_TOOLS] {json} [/AVAILABLE_TOOLS]")
}

/// Format a single tool call in Qwen `<tool_call>` format.
fn format_tool_call_qwen(tc: &ToolCall) -> String {
    let args = if tc.function.arguments.is_string() {
        tc.function.arguments.as_str().unwrap_or("{}").to_string()
    } else {
        serde_json::to_string(&tc.function.arguments).unwrap_or_else(|_| "{}".to_string())
    };
    format!(
        "\n<tool_call>\n{{\"name\": \"{}\", \"arguments\": {}}}\n</tool_call>",
        tc.function.name, args
    )
}

/// Format a single tool call in Llama 3.1 JSON format.
fn format_tool_call_llama3(tc: &ToolCall) -> String {
    let args = if tc.function.arguments.is_string() {
        tc.function.arguments.as_str().unwrap_or("{}").to_string()
    } else {
        serde_json::to_string(&tc.function.arguments).unwrap_or_else(|_| "{}".to_string())
    };
    format!(
        "{{\"name\": \"{}\", \"parameters\": {}}}",
        tc.function.name, args
    )
}

/// Format tool calls for Mistral v3+ templates.
///
/// Produces: `[TOOL_CALLS] [{"name":"...","arguments":{...}}]`
fn format_tool_calls_mistral(tool_calls: &[ToolCall]) -> String {
    let calls: Vec<serde_json::Value> = tool_calls
        .iter()
        .map(|tc| {
            serde_json::json!({
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            })
        })
        .collect();
    let json = serde_json::to_string(&calls).unwrap_or_else(|_| "[]".to_string());
    format!("[TOOL_CALLS] {json}")
}

/// Known chat template types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChatTemplateType {
    /// ChatML format: <|im_start|>role\ncontent<|im_end|>
    ChatMl,
    /// Llama-2 format: [INST] <<SYS>>...message...[/INST]
    Llama2,
    /// Llama-3 format: <|start_header_id|>role<|end_header_id|>content<|eot_id|>
    Llama3,
    /// Mistral format: [INST] message [/INST]
    Mistral,
    /// Gemma format: <start_of_turn>role\ncontent<end_of_turn>
    Gemma,
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
    pub fn bos_token(&self) -> Option<&'static str> {
        match self {
            Self::Llama2 => Some("<s>"),
            Self::Llama3 => Some("<|begin_of_text|>"),
            Self::Llama4 => Some("<|begin_of_text|>"),
            Self::DeepSeek => Some("<｜begin▁of▁sentence｜>"),
            Self::Cohere => Some("<BOS_TOKEN>"),
            _ => None,
        }
    }
}

/// Result of applying a chat template.
#[derive(Debug, Clone)]
pub struct FormattedChat {
    /// The full formatted text.
    pub text: String,
    /// Byte offset where the response begins (for loss masking).
    /// All tokens before this offset should be masked with -100.
    pub response_start: usize,
    /// The template type used.
    pub template_type: ChatTemplateType,
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
    /// Optional default system message.
    pub default_system_message: Option<String>,
    /// BOS token (optional).
    pub bos_token: Option<String>,
    /// EOS token.
    pub eos_token: String,
    /// Whether to add BOS at the start.
    pub add_bos: bool,
    /// Whether to add EOS at the end.
    pub add_eos: bool,
}

impl ChatTemplate {
    /// Create a new chat template with the given type.
    pub fn new(template_type: ChatTemplateType) -> Self {
        Self {
            template_type,
            default_system_message: None,
            bos_token: template_type.bos_token().map(String::from),
            eos_token: template_type.eos_token().to_string(),
            add_bos: matches!(
                template_type,
                ChatTemplateType::Llama2
                    | ChatTemplateType::Llama3
                    | ChatTemplateType::Llama4
                    | ChatTemplateType::DeepSeek
                    | ChatTemplateType::Cohere
            ),
            add_eos: true,
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

    /// Create a Phi-3 template.
    pub fn phi3() -> Self {
        Self::new(ChatTemplateType::Phi3)
    }

    /// Create a Qwen template.
    pub fn qwen() -> Self {
        let mut template = Self::new(ChatTemplateType::Qwen);
        template.default_system_message = Some(
            "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.".to_string(),
        );
        template
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

    /// Set the default system message.
    pub fn with_system_message(mut self, message: impl Into<String>) -> Self {
        self.default_system_message = Some(message.into());
        self
    }

    /// Set whether to add BOS token.
    pub fn with_add_bos(mut self, add_bos: bool) -> Self {
        self.add_bos = add_bos;
        self
    }

    /// Set whether to add EOS token.
    pub fn with_add_eos(mut self, add_eos: bool) -> Self {
        self.add_eos = add_eos;
        self
    }

    /// Format a conversation using this template.
    ///
    /// Returns the formatted text and the byte offset where the final
    /// assistant response begins (for loss masking in SFT training).
    pub fn apply(&self, messages: &[Message]) -> FormattedChat {
        self.apply_with_tools(messages, None)
    }

    /// Format a conversation with optional tool definitions.
    ///
    /// When `tools` is `Some`, tool definitions are injected into the system
    /// prompt using the model-specific format, and messages with role="tool"
    /// or `tool_calls` are formatted accordingly.
    pub fn apply_with_tools(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> FormattedChat {
        match self.template_type {
            ChatTemplateType::ChatMl | ChatTemplateType::Qwen => {
                self.format_chatml(messages, tools)
            }
            ChatTemplateType::Llama3 => self.format_llama3(messages, tools),
            ChatTemplateType::Llama4 => self.format_llama4(messages, tools),
            ChatTemplateType::Mistral => self.format_mistral(messages, tools),
            ChatTemplateType::DeepSeek => self.format_deepseek(messages, tools),
            // Templates without native tool support — inject tools into system prompt via ChatML style
            ChatTemplateType::Llama2 => self.format_llama2(messages),
            ChatTemplateType::Gemma => self.format_gemma(messages),
            ChatTemplateType::Phi3 => self.format_phi3(messages),
            ChatTemplateType::Phi4 => self.format_phi4(messages),
            ChatTemplateType::Alpaca => self.format_alpaca(messages),
            ChatTemplateType::Vicuna => self.format_vicuna(messages),
            ChatTemplateType::Zephyr => self.format_zephyr(messages),
            ChatTemplateType::GptOss => self.format_gpt_oss(messages),
            ChatTemplateType::Cohere => self.format_cohere(messages),
            ChatTemplateType::Custom => self.format_chatml(messages, tools),
        }
    }

    /// Format using ChatML / Qwen format with tool support.
    ///
    /// Tool format (Qwen 2.5 style):
    /// - System message includes `# Tools` section with JSON schemas in `<tools>` tags
    /// - Assistant tool calls: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
    /// - Tool responses: role="tool" rendered as `<|im_start|>user\n<tool_response>...</tool_response><|im_end|>`
    fn format_chatml(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> FormattedChat {
        let mut text = String::new();
        let mut response_start = 0;

        // Build system message content
        let has_system = messages.iter().any(|m| m.role == "system");
        let system_content = if has_system {
            let sys_msg = messages.iter().find(|m| m.role == "system").unwrap();
            let mut content = sys_msg.content.clone();
            if let Some(tool_defs) = tools {
                if !tool_defs.is_empty() {
                    content.push_str(&format_tools_qwen(tool_defs));
                }
            }
            Some(content)
        } else {
            let base = self.default_system_message.clone().unwrap_or_default();
            if let Some(tool_defs) = tools {
                if !tool_defs.is_empty() {
                    Some(format!("{}{}", base, format_tools_qwen(tool_defs)))
                } else if !base.is_empty() {
                    Some(base)
                } else {
                    None
                }
            } else if !base.is_empty() {
                Some(base)
            } else {
                None
            }
        };

        // Emit system message
        if let Some(ref sys) = system_content {
            text.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys));
        }

        // Track consecutive tool messages to merge them (Qwen style)
        let non_system: Vec<&Message> = messages.iter().filter(|m| m.role != "system").collect();

        for (i, msg) in non_system.iter().enumerate() {
            let is_last = i == non_system.len() - 1;

            match msg.role.as_str() {
                "assistant" if msg.has_tool_calls() => {
                    // Assistant message with tool calls
                    let header = "<|im_start|>assistant".to_string();
                    if is_last {
                        response_start = text.len() + header.len() + 1; // +1 for \n
                    }
                    text.push_str(&header);
                    if !msg.content.is_empty() {
                        text.push('\n');
                        text.push_str(&msg.content);
                    }
                    for tc in msg.tool_calls.as_ref().unwrap() {
                        text.push_str(&format_tool_call_qwen(tc));
                    }
                    text.push_str("<|im_end|>\n");
                }
                "tool" => {
                    // Qwen merges consecutive tool responses into one user turn
                    let is_first_tool = i == 0
                        || non_system
                            .get(i - 1)
                            .is_none_or(|prev| prev.role != "tool");
                    let is_last_tool = non_system
                        .get(i + 1)
                        .is_none_or(|next| next.role != "tool");

                    if is_first_tool {
                        text.push_str("<|im_start|>user");
                    }
                    text.push_str("\n<tool_response>\n");
                    text.push_str(&msg.content);
                    text.push_str("\n</tool_response>");
                    if is_last_tool {
                        text.push_str("<|im_end|>\n");
                    }
                }
                role => {
                    // Regular user/assistant message
                    let header = format!("<|im_start|>{}\n", role);
                    if role == "assistant" && is_last {
                        response_start = text.len() + header.len();
                    }
                    text.push_str(&header);
                    text.push_str(&msg.content);
                    text.push_str("<|im_end|>\n");
                }
            }
        }

        // For training/inference, add assistant header if last message is user or tool
        if let Some(last) = non_system.last() {
            if last.role == "user" || last.role == "tool" {
                text.push_str("<|im_start|>assistant\n");
                response_start = text.len();
            }
        }

        FormattedChat {
            text,
            response_start,
            template_type: self.template_type,
        }
    }

    /// Format using Llama-2 format.
    fn format_llama2(&self, messages: &[Message]) -> FormattedChat {
        let mut text = String::new();
        let mut response_start = 0;

        if self.add_bos {
            if let Some(ref bos) = self.bos_token {
                text.push_str(bos);
            }
        }

        let mut system_message = self.default_system_message.clone();
        let mut user_messages: Vec<&Message> = Vec::new();
        let mut assistant_messages: Vec<&Message> = Vec::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" => system_message = Some(msg.content.clone()),
                "user" => user_messages.push(msg),
                "assistant" => assistant_messages.push(msg),
                _ => {}
            }
        }

        // Format first turn with system message
        if let Some(first_user) = user_messages.first() {
            text.push_str("[INST] ");
            if let Some(ref sys) = system_message {
                text.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", sys));
            }
            text.push_str(&first_user.content);
            text.push_str(" [/INST]");

            // First assistant response
            if let Some(first_assistant) = assistant_messages.first() {
                response_start = text.len() + 1;
                text.push(' ');
                text.push_str(&first_assistant.content);
            } else {
                response_start = text.len() + 1;
                text.push(' ');
            }
        }

        // Additional turns
        for i in 1..user_messages.len().max(assistant_messages.len()) {
            if i < user_messages.len() {
                text.push_str(&format!(" [INST] {} [/INST]", user_messages[i].content));
            }
            if i < assistant_messages.len() {
                response_start = text.len() + 1;
                text.push(' ');
                text.push_str(&assistant_messages[i].content);
            } else {
                response_start = text.len() + 1;
                text.push(' ');
            }
        }

        if self.add_eos && !text.ends_with("</s>") {
            text.push_str("</s>");
        }

        FormattedChat {
            text,
            response_start,
            template_type: self.template_type,
        }
    }

    /// Format using Llama-3 format.
    /// Format using Llama 3 / 3.1+ format with tool support.
    ///
    /// Tool format (Llama 3.1 style):
    /// - System message includes "Environment: ipython" and tool schemas when tools present
    /// - Assistant tool calls: `{"name": "...", "parameters": {...}}`
    /// - Tool responses: role="ipython" with content
    fn format_llama3(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> FormattedChat {
        let mut text = String::new();
        let mut response_start = 0;
        let has_tools = tools.is_some_and(|t| !t.is_empty());

        if self.add_bos {
            text.push_str("<|begin_of_text|>");
        }

        // Build system message
        let has_system = messages.iter().any(|m| m.role == "system");
        let system_content = if has_system {
            let sys = messages.iter().find(|m| m.role == "system").unwrap();
            let mut content = String::new();
            if has_tools {
                content.push_str("Environment: ipython\n");
            }
            content.push_str(&sys.content);
            if let Some(tool_defs) = tools {
                if !tool_defs.is_empty() {
                    content.push('\n');
                    content.push_str(&format_tools_llama3(tool_defs));
                }
            }
            content
        } else {
            let mut content = String::new();
            if has_tools {
                content.push_str("Environment: ipython\n");
            }
            if let Some(ref default) = self.default_system_message {
                content.push_str(default);
            }
            if let Some(tool_defs) = tools {
                if !tool_defs.is_empty() {
                    content.push('\n');
                    content.push_str(&format_tools_llama3(tool_defs));
                }
            }
            content
        };

        // Always emit system header for Llama 3
        text.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
        text.push_str(system_content.trim());
        text.push_str("<|eot_id|>");

        // Process non-system messages
        let non_system: Vec<&Message> = messages.iter().filter(|m| m.role != "system").collect();

        for (i, msg) in non_system.iter().enumerate() {
            let is_last = i == non_system.len() - 1;

            match msg.role.as_str() {
                "assistant" if msg.has_tool_calls() => {
                    let header = "<|start_header_id|>assistant<|end_header_id|>\n\n";
                    if is_last {
                        response_start = text.len() + header.len();
                    }
                    text.push_str(header);
                    if !msg.content.is_empty() {
                        text.push_str(msg.content.trim());
                    }
                    // Format each tool call
                    for tc in msg.tool_calls.as_ref().unwrap() {
                        text.push_str(&format_tool_call_llama3(tc));
                    }
                    text.push_str("<|eot_id|>");
                }
                "tool" | "ipython" => {
                    // Tool responses use ipython header in Llama 3.1
                    text.push_str("<|start_header_id|>ipython<|end_header_id|>\n\n");
                    text.push_str(msg.content.trim());
                    text.push_str("<|eot_id|>");
                }
                role => {
                    let header = format!("<|start_header_id|>{}<|end_header_id|>\n\n", role);
                    if role == "assistant" && is_last {
                        response_start = text.len() + header.len();
                    }
                    text.push_str(&header);
                    text.push_str(msg.content.trim());
                    text.push_str("<|eot_id|>");
                }
            }
        }

        // Add assistant generation prompt if last is user or tool
        if let Some(last) = non_system.last() {
            if last.role == "user" || last.role == "tool" || last.role == "ipython" {
                text.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
                response_start = text.len();
            }
        }

        FormattedChat {
            text,
            response_start,
            template_type: self.template_type,
        }
    }

    /// Format using Mistral format.
    /// Format using Mistral v3+ format with tool support.
    ///
    /// Tool format:
    /// - `[AVAILABLE_TOOLS] [{...}, ...] [/AVAILABLE_TOOLS]` before first user message
    /// - `[TOOL_CALLS] [{"name": "...", "arguments": {...}}]` for assistant tool calls
    /// - `[TOOL_RESULTS] {"content": "..."} [/TOOL_RESULTS]` for tool responses
    fn format_mistral(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> FormattedChat {
        let mut text = String::new();
        let mut response_start = 0;

        if self.add_bos {
            text.push_str("<s>");
        }

        let mut system_content: Option<String> = None;
        let mut is_first_user = true;

        for (i, msg) in messages.iter().enumerate() {
            match msg.role.as_str() {
                "system" => {
                    system_content = Some(msg.content.clone());
                }
                "user" => {
                    // Inject tool definitions before first user message
                    if is_first_user {
                        if let Some(tool_defs) = tools {
                            if !tool_defs.is_empty() {
                                text.push_str(&format_tools_mistral(tool_defs));
                            }
                        }
                    }
                    text.push_str("[INST] ");
                    if is_first_user {
                        if let Some(ref sys) = system_content {
                            text.push_str(sys);
                            text.push_str("\n\n");
                        }
                        is_first_user = false;
                    }
                    text.push_str(&msg.content);
                    text.push_str(" [/INST]");
                }
                "assistant" if msg.has_tool_calls() => {
                    response_start = text.len();
                    text.push_str(&format_tool_calls_mistral(msg.tool_calls.as_ref().unwrap()));
                    text.push_str("</s>");
                }
                "assistant" => {
                    response_start = text.len();
                    text.push_str(&msg.content);
                    if i < messages.len() - 1 {
                        text.push_str("</s>");
                    }
                }
                "tool" => {
                    text.push_str(&format!(
                        "[TOOL_RESULTS] {{\"content\": {}}} [/TOOL_RESULTS]",
                        serde_json::to_string(&msg.content)
                            .unwrap_or_else(|_| format!("\"{}\"", msg.content))
                    ));
                }
                _ => {}
            }
        }

        if let Some(last) = messages.last() {
            if last.role == "user" || last.role == "tool" {
                response_start = text.len();
            }
        }

        if self.add_eos && !text.ends_with("</s>") {
            text.push_str("</s>");
        }

        FormattedChat {
            text,
            response_start,
            template_type: self.template_type,
        }
    }

    /// Format using Gemma format.
    fn format_gemma(&self, messages: &[Message]) -> FormattedChat {
        let mut text = String::new();
        let mut response_start = 0;

        for (i, msg) in messages.iter().enumerate() {
            let role = match msg.role.as_str() {
                "user" | "human" => "user",
                "assistant" | "gpt" => "model",
                "system" => "user", // Gemma treats system as user turn
                _ => &msg.role,
            };

            let formatted = format!("<start_of_turn>{}\n{}<end_of_turn>\n", role, msg.content);

            // Track where the last model response begins
            if (msg.role == "assistant" || msg.role == "model") && i == messages.len() - 1 {
                let header = format!("<start_of_turn>{}\n", role);
                response_start = text.len() + header.len();
            }

            text.push_str(&formatted);
        }

        // For training, add model turn header if last is user
        if let Some(last) = messages.last() {
            if last.role == "user" || last.role == "human" {
                text.push_str("<start_of_turn>model\n");
                response_start = text.len();
            }
        }

        FormattedChat {
            text,
            response_start,
            template_type: self.template_type,
        }
    }

    /// Format using Phi-3 format.
    fn format_phi3(&self, messages: &[Message]) -> FormattedChat {
        let mut text = String::new();
        let mut response_start = 0;

        for (i, msg) in messages.iter().enumerate() {
            let tag = match msg.role.as_str() {
                "system" => "<|system|>",
                "user" => "<|user|>",
                "assistant" => "<|assistant|>",
                _ => "<|user|>",
            };

            text.push_str(tag);
            text.push('\n');
            text.push_str(&msg.content);
            text.push_str("<|end|>\n");

            if msg.role == "assistant" && i == messages.len() - 1 {
                // Response started after the tag and newline
                response_start = text.len() - msg.content.len() - "<|end|>\n".len();
            }
        }

        // For training, add assistant header if last is user
        if let Some(last) = messages.last() {
            if last.role == "user" {
                text.push_str("<|assistant|>\n");
                response_start = text.len();
            }
        }

        FormattedChat {
            text,
            response_start,
            template_type: self.template_type,
        }
    }

    /// Format using Phi-4 format.
    fn format_phi4(&self, messages: &[Message]) -> FormattedChat {
        let mut text = String::new();
        let mut response_start = 0;

        for (i, msg) in messages.iter().enumerate() {
            let formatted = format!(
                "<|im_start|>{}<|im_sep|>{}<|im_end|>",
                msg.role, msg.content
            );

            if msg.role == "assistant" && i == messages.len() - 1 {
                let header = format!("<|im_start|>{}<|im_sep|>", msg.role);
                response_start = text.len() + header.len();
            }

            text.push_str(&formatted);
        }

        // For training, add assistant header if last is user
        if let Some(last) = messages.last() {
            if last.role == "user" {
                text.push_str("<|im_start|>assistant<|im_sep|>");
                response_start = text.len();
            }
        }

        FormattedChat {
            text,
            response_start,
            template_type: self.template_type,
        }
    }

    /// Format using Alpaca format.
    fn format_alpaca(&self, messages: &[Message]) -> FormattedChat {
        let mut text = String::new();

        let mut instruction = String::new();
        let mut input = String::new();
        let mut output = String::new();

        for msg in messages {
            match msg.role.as_str() {
                "system" | "instruction" => instruction = msg.content.clone(),
                "user" | "input" => input = msg.content.clone(),
                "assistant" | "output" => output = msg.content.clone(),
                _ => {}
            }
        }

        if !instruction.is_empty() {
            text.push_str("### Instruction:\n");
            text.push_str(&instruction);
            text.push_str("\n\n");
        }

        if !input.is_empty() {
            text.push_str("### Input:\n");
            text.push_str(&input);
            text.push_str("\n\n");
        }

        text.push_str("### Response:\n");
        let response_start = text.len();

        if !output.is_empty() {
            text.push_str(&output);
        }

        FormattedChat {
            text,
            response_start,
            template_type: self.template_type,
        }
    }

    /// Format using Vicuna format.
    fn format_vicuna(&self, messages: &[Message]) -> FormattedChat {
        let mut text = String::new();
        let mut response_start = 0;

        // System message at the start
        if let Some(ref sys) = self.default_system_message {
            text.push_str(sys);
            text.push_str("\n\n");
        }

        for msg in messages {
            match msg.role.as_str() {
                "system" => {
                    // Already handled or override
                    if text.is_empty() {
                        text.push_str(&msg.content);
                        text.push_str("\n\n");
                    }
                }
                "user" => {
                    text.push_str("USER: ");
                    text.push_str(&msg.content);
                    text.push('\n');
                }
                "assistant" => {
                    text.push_str("ASSISTANT: ");
                    response_start = text.len();
                    text.push_str(&msg.content);
                    text.push('\n');
                }
                _ => {}
            }
        }

        // For training, add assistant prefix if last is user
        if let Some(last) = messages.last() {
            if last.role == "user" {
                text.push_str("ASSISTANT: ");
                response_start = text.len();
            }
        }

        FormattedChat {
            text,
            response_start,
            template_type: self.template_type,
        }
    }

    /// Format using Zephyr format.
    fn format_zephyr(&self, messages: &[Message]) -> FormattedChat {
        let mut text = String::new();
        let mut response_start = 0;

        for (i, msg) in messages.iter().enumerate() {
            let tag = match msg.role.as_str() {
                "system" => "<|system|>",
                "user" => "<|user|>",
                "assistant" => "<|assistant|>",
                _ => "<|user|>",
            };

            text.push_str(tag);
            text.push('\n');
            text.push_str(&msg.content);
            text.push_str("</s>\n");

            if msg.role == "assistant" && i == messages.len() - 1 {
                response_start = text.len() - msg.content.len() - "</s>\n".len();
            }
        }

        // For training, add assistant header if last is user
        if let Some(last) = messages.last() {
            if last.role == "user" {
                text.push_str("<|assistant|>\n");
                response_start = text.len();
            }
        }

        FormattedChat {
            text,
            response_start,
            template_type: self.template_type,
        }
    }

    /// Format using GPT-OSS Harmony format.
    ///
    /// Harmony format structure:
    /// - `<|start|>role<|message|>content<|end|>` for basic messages
    /// - `<|start|>assistant<|channel|>final<|message|>content<|end|>` for assistant
    ///
    /// Roles: system, developer, user, assistant
    /// Channels: analysis (reasoning), commentary, final (response)
    /// EOS: `<|return|>` marks end of generation
    fn format_gpt_oss(&self, messages: &[Message]) -> FormattedChat {
        let mut text = String::new();
        let mut response_start = 0;

        // Add default system message if no system message and we have a default
        let has_system = messages.iter().any(|m| m.role == "system");
        let all_messages: Vec<Message> = if !has_system {
            if let Some(ref system_msg) = self.default_system_message {
                let mut msgs = vec![Message::system(system_msg.clone())];
                msgs.extend(messages.iter().cloned());
                msgs
            } else {
                messages.to_vec()
            }
        } else {
            messages.to_vec()
        };

        for (i, msg) in all_messages.iter().enumerate() {
            match msg.role.as_str() {
                "system" => {
                    // System messages use simple format
                    text.push_str("<|start|>system<|message|>");
                    text.push_str(&msg.content);
                    text.push_str("<|end|>");
                }
                "developer" => {
                    // Developer messages for system-level instructions
                    text.push_str("<|start|>developer<|message|>");
                    text.push_str(&msg.content);
                    text.push_str("<|end|>");
                }
                "user" => {
                    // User messages use simple format
                    text.push_str("<|start|>user<|message|>");
                    text.push_str(&msg.content);
                    text.push_str("<|end|>");
                }
                "assistant" => {
                    // Assistant uses final channel for responses
                    let header = "<|start|>assistant<|channel|>final<|message|>";
                    text.push_str(header);

                    // Track where the last assistant response begins
                    if i == all_messages.len() - 1 {
                        response_start = text.len();
                    }

                    text.push_str(&msg.content);
                    text.push_str("<|end|>");
                }
                _ => {
                    // Unknown role: treat as user
                    text.push_str("<|start|>user<|message|>");
                    text.push_str(&msg.content);
                    text.push_str("<|end|>");
                }
            }
        }

        // For training, add assistant header if last message is user
        if let Some(last) = all_messages.last() {
            if last.role == "user" {
                text.push_str("<|start|>assistant<|channel|>final<|message|>");
                response_start = text.len();
            }
        }

        // Add return token at the end for training
        if self.add_eos {
            text.push_str("<|return|>");
        }

        FormattedChat {
            text,
            response_start,
            template_type: self.template_type,
        }
    }

    /// Format using Llama-4 format with tool support.
    ///
    /// Llama 4 uses different header tokens than Llama 3:
    /// - `<|header_start|>` / `<|header_end|>` (not `<|start_header_id|>` / `<|end_header_id|>`)
    /// - `<|eot|>` (not `<|eot_id|>`)
    ///
    /// Tool format follows Llama 3.1 conventions but with Llama 4 tokens.
    fn format_llama4(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> FormattedChat {
        let mut text = String::new();
        let mut response_start = 0;
        let has_tools = tools.is_some_and(|t| !t.is_empty());

        if self.add_bos {
            text.push_str("<|begin_of_text|>");
        }

        // Build system message with tools
        let has_system = messages.iter().any(|m| m.role == "system");
        let system_content = {
            let mut content = String::new();
            if has_tools {
                content.push_str("Environment: ipython\n");
            }
            if has_system {
                let sys = messages.iter().find(|m| m.role == "system").unwrap();
                content.push_str(&sys.content);
            } else if let Some(ref default) = self.default_system_message {
                content.push_str(default);
            }
            if let Some(tool_defs) = tools {
                if !tool_defs.is_empty() {
                    content.push('\n');
                    content.push_str(&format_tools_llama3(tool_defs));
                }
            }
            content
        };

        if !system_content.is_empty() {
            text.push_str("<|header_start|>system<|header_end|>\n\n");
            text.push_str(system_content.trim());
            text.push_str("<|eot|>");
        }

        let non_system: Vec<&Message> = messages.iter().filter(|m| m.role != "system").collect();

        for (i, msg) in non_system.iter().enumerate() {
            let is_last = i == non_system.len() - 1;

            match msg.role.as_str() {
                "assistant" if msg.has_tool_calls() => {
                    let header = "<|header_start|>assistant<|header_end|>\n\n";
                    if is_last {
                        response_start = text.len() + header.len();
                    }
                    text.push_str(header);
                    if !msg.content.is_empty() {
                        text.push_str(msg.content.trim());
                    }
                    for tc in msg.tool_calls.as_ref().unwrap() {
                        text.push_str(&format_tool_call_llama3(tc));
                    }
                    text.push_str("<|eot|>");
                }
                "tool" | "ipython" => {
                    text.push_str("<|header_start|>ipython<|header_end|>\n\n");
                    text.push_str(msg.content.trim());
                    text.push_str("<|eot|>");
                }
                role => {
                    let header = format!("<|header_start|>{}<|header_end|>\n\n", role);
                    if role == "assistant" && is_last {
                        response_start = text.len() + header.len();
                    }
                    text.push_str(&header);
                    text.push_str(msg.content.trim());
                    text.push_str("<|eot|>");
                }
            }
        }

        if let Some(last) = non_system.last() {
            if last.role == "user" || last.role == "tool" || last.role == "ipython" {
                text.push_str("<|header_start|>assistant<|header_end|>\n\n");
                response_start = text.len();
            }
        }

        FormattedChat {
            text,
            response_start,
            template_type: self.template_type,
        }
    }

    /// Format using DeepSeek format with tool support.
    ///
    /// DeepSeek uses full-width unicode pipe characters (U+FF5C) and
    /// lower-one-eighth-block (U+2581) for underscores in token names.
    /// Tool format: tools injected into system prompt using Qwen-style tags.
    fn format_deepseek(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> FormattedChat {
        let mut text = String::new();
        let mut response_start = 0;

        if self.add_bos {
            text.push_str("<｜begin▁of▁sentence｜>");
        }

        for (i, msg) in messages.iter().enumerate() {
            match msg.role.as_str() {
                "system" => {
                    text.push_str(&msg.content);
                    if let Some(tool_defs) = tools {
                        if !tool_defs.is_empty() {
                            text.push_str(&format_tools_qwen(tool_defs));
                        }
                    }
                }
                "user" => {
                    text.push_str("<｜User｜>");
                    text.push_str(&msg.content);
                }
                "assistant" if msg.has_tool_calls() => {
                    text.push_str("<｜Assistant｜>");
                    if i == messages.len() - 1 {
                        response_start = text.len();
                    }
                    if !msg.content.is_empty() {
                        text.push_str(&msg.content);
                    }
                    for tc in msg.tool_calls.as_ref().unwrap() {
                        text.push_str(&format_tool_call_qwen(tc));
                    }
                    if i < messages.len() - 1 {
                        text.push_str("<｜end▁of▁sentence｜>");
                    }
                }
                "assistant" => {
                    text.push_str("<｜Assistant｜>");
                    if i == messages.len() - 1 {
                        response_start = text.len();
                    }
                    text.push_str(&msg.content);
                    if i < messages.len() - 1 {
                        text.push_str("<｜end▁of▁sentence｜>");
                    }
                }
                "tool" => {
                    text.push_str("<｜User｜>\n<tool_response>\n");
                    text.push_str(&msg.content);
                    text.push_str("\n</tool_response>");
                }
                _ => {
                    text.push_str("<｜User｜>");
                    text.push_str(&msg.content);
                }
            }
        }

        if let Some(last) = messages.last() {
            if last.role == "user" || last.role == "tool" {
                text.push_str("<｜Assistant｜>");
                response_start = text.len();
            }
        }

        if self.add_eos && !text.ends_with("<｜end▁of▁sentence｜>") {
            text.push_str("<｜end▁of▁sentence｜>");
        }

        FormattedChat {
            text,
            response_start,
            template_type: self.template_type,
        }
    }

    /// Format using Cohere Command R format.
    ///
    /// Uses `<|START_OF_TURN_TOKEN|>`, `<|USER_TOKEN|>`, `<|CHATBOT_TOKEN|>`,
    /// and `<|END_OF_TURN_TOKEN|>` tokens.
    fn format_cohere(&self, messages: &[Message]) -> FormattedChat {
        let mut text = String::new();
        let mut response_start = 0;

        if self.add_bos {
            if let Some(ref bos) = self.bos_token {
                text.push_str(bos);
            }
        }

        for (i, msg) in messages.iter().enumerate() {
            let role_token = match msg.role.as_str() {
                "system" => "<|SYSTEM_TOKEN|>",
                "user" => "<|USER_TOKEN|>",
                "assistant" => "<|CHATBOT_TOKEN|>",
                _ => "<|USER_TOKEN|>",
            };

            text.push_str("<|START_OF_TURN_TOKEN|>");
            text.push_str(role_token);
            text.push_str(&msg.content);
            text.push_str("<|END_OF_TURN_TOKEN|>");

            if msg.role == "assistant" && i == messages.len() - 1 {
                response_start = text.len() - msg.content.len() - "<|END_OF_TURN_TOKEN|>".len();
            }
        }

        if let Some(last) = messages.last() {
            if last.role == "user" {
                text.push_str("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>");
                response_start = text.len();
            }
        }

        FormattedChat {
            text,
            response_start,
            template_type: self.template_type,
        }
    }
}

/// Detect the appropriate chat template by inspecting the model's `tokenizer_config.json` Jinja
/// string first, then falling back to model-name heuristics.
///
/// This is the preferred entry point — it gives consistent results across training, inference,
/// and distillation because the same Jinja patterns are checked everywhere.
pub fn detect_chat_template(model_path: &std::path::Path, model_name: &str) -> ChatTemplate {
    // 1. Try tokenizer_config.json Jinja string
    let config_path = model_path.join("tokenizer_config.json");
    if let Ok(content) = std::fs::read_to_string(&config_path) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(jinja) = extract_jinja_template(&json) {
                if let Some(mut template) = detect_template_from_jinja(&jinja) {
                    // Refine: if the Jinja matched generic ChatML but model_type is "qwen2",
                    // upgrade to Qwen variant (adds default system message for training).
                    if template.template_type == ChatTemplateType::ChatMl {
                        let is_qwen = json
                            .get("model_type")
                            .and_then(|v| v.as_str())
                            .is_some_and(|t| t.to_lowercase().contains("qwen"));
                        if is_qwen {
                            template = ChatTemplate::qwen();
                        }
                    }
                    tracing::info!(
                        "Chat template: {:?} (from tokenizer_config.json)",
                        template.template_type
                    );
                    return template;
                }
            }
        }
    }

    // 2. Fall back to model-name heuristics
    let template = detect_template_from_model(model_name);
    tracing::info!(
        "Chat template: {:?} (from model name)",
        template.template_type
    );
    template
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
    // Gemma: <start_of_turn>
    if jinja.contains("<start_of_turn>") {
        return Some(ChatTemplate::gemma());
    }
    // Mistral: [INST] without <<SYS>>
    if jinja.contains("[INST]") {
        return Some(ChatTemplate::mistral());
    }
    // Phi-3: <|user|> + <|end|>
    if jinja.contains("<|user|>") && jinja.contains("<|end|>") {
        return Some(ChatTemplate::phi3());
    }
    // GPT-OSS Harmony: <|start|> + <|message|>
    if jinja.contains("<|start|>") && jinja.contains("<|message|>") {
        return Some(ChatTemplate::gpt_oss());
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
    /// Returns the formatted text and the prompt length in characters.
    /// The prompt length can be used to mask prompt tokens in the loss.
    pub fn build(&self, messages: &[Message]) -> (String, usize) {
        let formatted = self.template.apply(messages);
        (formatted.text, formatted.response_start)
    }

    /// Build a training sample and tokenize it with label masking.
    ///
    /// Returns (input_ids, labels) where labels have -100 for prompt tokens.
    pub fn build_tokenized(
        &self,
        messages: &[Message],
        tokenizer: &super::Tokenizer,
        max_length: usize,
    ) -> pmetal_core::Result<(Vec<u32>, Vec<i64>)> {
        let formatted = self.template.apply(messages);

        // Tokenize full text
        let mut input_ids = tokenizer.encode_with_special_tokens(&formatted.text)?;

        // Truncate if needed
        if input_ids.len() > max_length {
            input_ids.truncate(max_length);
        }

        // Tokenize just the prompt to find the split point
        let prompt_ids = tokenizer.encode_with_special_tokens(formatted.prompt())?;
        let prompt_len = prompt_ids.len().min(input_ids.len());

        // Create labels with prompt masked
        let mut labels: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
        for label in labels.iter_mut().take(prompt_len) {
            *label = -100;
        }

        Ok((input_ids, labels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatml_format() {
        let template = ChatTemplate::chatml();
        let messages = vec![
            Message::user("Hello, how are you?"),
            Message::assistant("I'm doing well, thank you!"),
        ];

        let formatted = template.apply(&messages);
        assert!(formatted.text.contains("<|im_start|>user"));
        assert!(formatted.text.contains("<|im_start|>assistant"));
        assert!(formatted.text.contains("<|im_end|>"));
    }

    #[test]
    fn test_llama3_format() {
        let template = ChatTemplate::llama3();
        let messages = vec![
            Message::user("What is 2+2?"),
            Message::assistant("2+2 equals 4."),
        ];

        let formatted = template.apply(&messages);
        assert!(formatted.text.contains("<|begin_of_text|>"));
        assert!(
            formatted
                .text
                .contains("<|start_header_id|>user<|end_header_id|>")
        );
        assert!(formatted.text.contains("<|eot_id|>"));
    }

    #[test]
    fn test_response_masking() {
        let template = ChatTemplate::chatml();
        let messages = vec![Message::user("Hello"), Message::assistant("Hi there!")];

        let formatted = template.apply(&messages);

        // Response start should be after the user turn
        assert!(formatted.response_start > 0);
        assert!(formatted.response_start < formatted.text.len());

        // The prompt should end before the response content
        let prompt = formatted.prompt();
        assert!(prompt.contains("Hello"));
        assert!(!prompt.contains("Hi there"));

        // The response should contain the assistant's message
        let response = formatted.response();
        assert!(response.contains("Hi there"));
    }

    #[test]
    fn test_gemma_format() {
        let template = ChatTemplate::gemma();
        let messages = vec![
            Message::user("Tell me a joke"),
            Message::assistant("Why did the chicken cross the road?"),
        ];

        let formatted = template.apply(&messages);
        assert!(formatted.text.contains("<start_of_turn>user"));
        assert!(formatted.text.contains("<start_of_turn>model"));
        assert!(formatted.text.contains("<end_of_turn>"));
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
    fn test_alpaca_format() {
        let template = ChatTemplate::alpaca();
        let messages = vec![
            Message::new("instruction", "Summarize the following text"),
            Message::new("input", "The quick brown fox jumps over the lazy dog."),
            Message::new("output", "A fox jumps over a dog."),
        ];

        let formatted = template.apply(&messages);
        assert!(formatted.text.contains("### Instruction:"));
        assert!(formatted.text.contains("### Input:"));
        assert!(formatted.text.contains("### Response:"));
    }

    #[test]
    fn test_training_for_response_only() {
        let template = ChatTemplate::llama3();
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("What is the capital of France?"),
            Message::assistant("The capital of France is Paris."),
        ];

        let formatted = template.apply(&messages);

        // Verify prompt contains system and user but not the answer
        let prompt = formatted.prompt();
        assert!(prompt.contains("helpful assistant"));
        assert!(prompt.contains("capital of France"));

        // Verify response is just the assistant content
        let response = formatted.response();
        assert!(response.contains("Paris"));
    }

    #[test]
    fn test_gpt_oss_format() {
        let template = ChatTemplate::gpt_oss();
        let messages = vec![
            Message::user("Hello, how are you?"),
            Message::assistant("I'm doing well, thank you!"),
        ];

        let formatted = template.apply(&messages);

        // Check Harmony format tokens
        assert!(formatted.text.contains("<|start|>user<|message|>"));
        assert!(
            formatted
                .text
                .contains("<|start|>assistant<|channel|>final<|message|>")
        );
        assert!(formatted.text.contains("<|end|>"));
        assert!(formatted.text.contains("<|return|>"));

        // Verify content is preserved
        assert!(formatted.text.contains("Hello, how are you?"));
        assert!(formatted.text.contains("I'm doing well, thank you!"));
    }

    #[test]
    fn test_gpt_oss_response_masking() {
        let template = ChatTemplate::gpt_oss();
        let messages = vec![
            Message::system("You are a helpful AI assistant."),
            Message::user("What is 2+2?"),
            Message::assistant("2+2 equals 4."),
        ];

        let formatted = template.apply(&messages);

        // Response start should point to after the assistant header
        assert!(formatted.response_start > 0);
        assert!(formatted.response_start < formatted.text.len());

        // The prompt should contain system and user content
        let prompt = formatted.prompt();
        assert!(prompt.contains("helpful AI assistant"));
        assert!(prompt.contains("What is 2+2?"));

        // The response should be the assistant's answer
        let response = formatted.response();
        assert!(response.contains("2+2 equals 4"));
    }

    #[test]
    fn test_gpt_oss_developer_role() {
        let template = ChatTemplate::gpt_oss();
        let messages = vec![
            Message::new("developer", "Use formal language."),
            Message::user("Hello"),
            Message::assistant("Greetings."),
        ];

        let formatted = template.apply(&messages);

        // Developer role should be formatted
        assert!(formatted.text.contains("<|start|>developer<|message|>"));
        assert!(formatted.text.contains("Use formal language."));
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

    #[test]
    fn test_gpt_oss_training_prompt_generation() {
        let template = ChatTemplate::gpt_oss();
        let messages = vec![Message::user("Explain quantum computing.")];

        let formatted = template.apply(&messages);

        // Last message is user, so should add assistant header for training
        assert!(
            formatted
                .text
                .contains("<|start|>assistant<|channel|>final<|message|>")
        );

        // Response start should be at the end, ready for model to generate
        assert_eq!(
            formatted.response_start,
            formatted.text.len() - "<|return|>".len()
        );
    }
}
