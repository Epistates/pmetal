//! Chat template system for formatting conversations.
//!
//! Provides model-specific chat templates for SFT training with:
//! - Proper formatting for each model family (Llama, Mistral, Gemma, etc.)
//! - Response masking support (identifying which tokens to train on)
//! - System message customization


/// A single message in a conversation.
#[derive(Debug, Clone)]
pub struct Message {
    /// Role: "system", "user", or "assistant"
    pub role: String,
    /// Content of the message
    pub content: String,
}

impl Message {
    /// Create a new message.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
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
            Self::Custom => "</s>",
        }
    }

    /// Get the BOS token for this template type (if any).
    pub fn bos_token(&self) -> Option<&'static str> {
        match self {
            Self::Llama2 => Some("<s>"),
            Self::Llama3 => Some("<|begin_of_text|>"),
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
            add_bos: matches!(template_type, ChatTemplateType::Llama2 | ChatTemplateType::Llama3),
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
        template.default_system_message =
            Some("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.".to_string());
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
        match self.template_type {
            ChatTemplateType::ChatMl | ChatTemplateType::Qwen => self.format_chatml(messages),
            ChatTemplateType::Llama2 => self.format_llama2(messages),
            ChatTemplateType::Llama3 => self.format_llama3(messages),
            ChatTemplateType::Mistral => self.format_mistral(messages),
            ChatTemplateType::Gemma => self.format_gemma(messages),
            ChatTemplateType::Phi3 => self.format_phi3(messages),
            ChatTemplateType::Phi4 => self.format_phi4(messages),
            ChatTemplateType::Alpaca => self.format_alpaca(messages),
            ChatTemplateType::Vicuna => self.format_vicuna(messages),
            ChatTemplateType::Zephyr => self.format_zephyr(messages),
            ChatTemplateType::GptOss => self.format_gpt_oss(messages),
            ChatTemplateType::Custom => self.format_chatml(messages), // Fallback
        }
    }

    /// Format using ChatML format.
    fn format_chatml(&self, messages: &[Message]) -> FormattedChat {
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
            let formatted = format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content);

            // Track where the last assistant response begins
            if msg.role == "assistant" && i == all_messages.len() - 1 {
                response_start = text.len() + format!("<|im_start|>{}\n", msg.role).len();
            }

            text.push_str(&formatted);
        }

        // For training, add assistant header if last message is user
        if let Some(last) = all_messages.last() {
            if last.role == "user" {
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
    fn format_llama3(&self, messages: &[Message]) -> FormattedChat {
        let mut text = String::new();
        let mut response_start = 0;

        if self.add_bos {
            text.push_str("<|begin_of_text|>");
        }

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
            let formatted = format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                msg.role,
                msg.content.trim()
            );

            // Track where the last assistant response begins
            if msg.role == "assistant" && i == all_messages.len() - 1 {
                let header = format!("<|start_header_id|>{}<|end_header_id|>\n\n", msg.role);
                response_start = text.len() + header.len();
            }

            text.push_str(&formatted);
        }

        // For training, add assistant header if last message is user
        if let Some(last) = all_messages.last() {
            if last.role == "user" {
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
    fn format_mistral(&self, messages: &[Message]) -> FormattedChat {
        let mut text = String::new();
        let mut response_start = 0;

        if self.add_bos {
            text.push_str("<s>");
        }

        // Mistral prepends system message to first user message
        let mut system_content: Option<String> = None;
        let mut is_first_user = true;

        for (i, msg) in messages.iter().enumerate() {
            match msg.role.as_str() {
                "system" => {
                    system_content = Some(msg.content.clone());
                }
                "user" => {
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
                "assistant" => {
                    response_start = text.len();
                    text.push_str(&msg.content);
                    if i < messages.len() - 1 {
                        text.push_str("</s>");
                    }
                }
                _ => {}
            }
        }

        // For training, prepare for assistant response
        if let Some(last) = messages.last() {
            if last.role == "user" {
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
            let formatted = format!("<|im_start|>{}<|im_sep|>{}<|im_end|>", msg.role, msg.content);

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
    } else if name_lower.contains("phi-3") || name_lower.contains("phi3") || name_lower.contains("phi") {
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
        assert!(formatted.text.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(formatted.text.contains("<|eot_id|>"));
    }

    #[test]
    fn test_response_masking() {
        let template = ChatTemplate::chatml();
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ];

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
        assert!(formatted.text.contains("<|start|>assistant<|channel|>final<|message|>"));
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
        assert!(formatted.text.contains("<|start|>assistant<|channel|>final<|message|>"));

        // Response start should be at the end, ready for model to generate
        assert_eq!(formatted.response_start, formatted.text.len() - "<|return|>".len());
    }
}
