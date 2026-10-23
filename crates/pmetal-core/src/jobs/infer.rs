//! `pmetal infer` — one-shot or chat inference.

use crate::{FieldError, JobFields};
use pmetal_core_derive::JobSpec;
use serde::{Deserialize, Serialize};

/// Spec for `pmetal infer`.
#[derive(Debug, Clone, Serialize, Deserialize, JobSpec)]
#[spec(kind = "Infer", subcommand = "infer")]
#[serde(rename_all = "snake_case")]
pub struct InferSpec {
    #[job(
        label = "Model",
        group = "Model",
        argv = "--model",
        kind = "model_picker",
        required
    )]
    #[serde(default)]
    pub model: String,

    #[job(
        label = "LoRA Adapter",
        group = "Model",
        argv = "--lora",
        kind = "path"
    )]
    #[serde(default)]
    pub lora: Option<String>,

    #[job(
        label = "Prompt",
        group = "Input",
        argv = "--prompt",
        help = "The text to answer; give this or a messages file"
    )]
    #[serde(default)]
    pub prompt: String,

    #[job(
        label = "Messages File",
        group = "Input",
        argv = "--messages-file",
        kind = "path",
        help = "A JSON array of chat messages ({\"role\": \"user\", \"content\": \"...\"}) to continue instead of a prompt; the chat template, tools and images apply"
    )]
    #[serde(default)]
    pub messages_file: Option<String>,

    #[job(
        label = "Max Tokens",
        group = "Sampling",
        argv = "--max-tokens",
        min = 1,
        max = 1_048_576,
        help = "Default: the model's generation_config.json, else 32768 for a thinking model, else 256"
    )]
    #[serde(default)]
    pub max_tokens: Option<usize>,

    #[job(
        label = "Temperature",
        group = "Sampling",
        argv = "--temperature",
        min = 0.0,
        max = 5.0
    )]
    #[serde(default)]
    pub temperature: Option<f32>,

    #[job(
        label = "Top-k",
        group = "Sampling",
        argv = "--top-k",
        min = 0,
        max = 1_000_000
    )]
    #[serde(default)]
    pub top_k: Option<usize>,

    #[job(
        label = "Top-p",
        group = "Sampling",
        argv = "--top-p",
        min = 0.0,
        max = 1.0
    )]
    #[serde(default)]
    pub top_p: Option<f32>,

    #[job(
        label = "Min-p",
        group = "Sampling",
        argv = "--min-p",
        min = 0.0,
        max = 1.0
    )]
    #[serde(default)]
    pub min_p: Option<f32>,

    #[job(
        label = "Repetition Penalty",
        group = "Sampling",
        argv = "--repetition-penalty",
        min = 0.0,
        max = 10.0
    )]
    #[serde(default)]
    pub repetition_penalty: Option<f32>,

    #[job(label = "Frequency Penalty", group = "Sampling", argv = "--frequency-penalty", min = -10.0, max = 10.0)]
    #[serde(default)]
    pub frequency_penalty: Option<f32>,

    #[job(label = "Presence Penalty", group = "Sampling", argv = "--presence-penalty", min = -10.0, max = 10.0)]
    #[serde(default)]
    pub presence_penalty: Option<f32>,

    #[job(label = "Seed", group = "Sampling", argv = "--seed")]
    #[serde(default)]
    pub seed: Option<u64>,

    #[job(
        label = "Chat Mode",
        group = "Input",
        argv = "--chat",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub chat: bool,

    #[job(label = "System Message", group = "Input", argv = "--system")]
    #[serde(default)]
    pub system: Option<String>,

    #[job(
        label = "No Thinking",
        group = "Input",
        argv = "--no-thinking",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub no_thinking: bool,

    #[job(
        label = "Reasoning Effort",
        group = "Input",
        argv = "--reasoning-effort",
        help = "For chat templates with a reasoning_effort control: xhigh | medium | low on Qwen3.8, low | medium | high on gpt-oss"
    )]
    #[serde(default)]
    pub reasoning_effort: Option<String>,

    #[job(
        label = "No Preserved Thinking",
        group = "Input",
        argv = "--no-preserve-thinking",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub no_preserve_thinking: bool,

    #[job(
        label = "Chat Template Kwargs",
        group = "Input",
        argv = "--chat-template-kwargs",
        help = "JSON object of extra chat template keyword arguments"
    )]
    #[serde(default)]
    pub chat_template_kwargs: Option<String>,

    #[job(
        label = "Hide Thinking",
        group = "Output",
        argv = "--hide-thinking",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub hide_thinking: bool,

    #[job(
        label = "Sampling Mode",
        group = "Sampling",
        argv = "--mode",
        kind = "enum",
        enum_options = ["auto", "thinking-general", "thinking-coding", "instruct-general", "instruct-reasoning"],
        default = "auto"
    )]
    #[serde(default = "default_mode")]
    pub mode: String,

    #[job(
        label = "Backend",
        group = "Compute",
        argv = "--backend",
        kind = "enum",
        enum_options = ["auto", "standard", "ane"],
        default = "auto"
    )]
    #[serde(default = "default_backend")]
    pub backend: String,

    #[job(
        label = "Draft Model",
        group = "Compute",
        argv = "--draft-model",
        kind = "model_picker"
    )]
    #[serde(default)]
    pub draft_model: Option<String>,

    #[job(
        label = "Images",
        group = "Input",
        argv = "--image",
        repeat,
        help = "Image files shown before the prompt text, in order (Qwen3.5-family vision models)"
    )]
    #[serde(default)]
    pub image: Vec<String>,

    #[job(
        label = "Videos",
        group = "Input",
        argv = "--video",
        repeat,
        help = "Directories of extracted video frames, shown after the images, in order"
    )]
    #[serde(default)]
    pub video: Vec<String>,

    #[job(
        label = "Video FPS",
        group = "Input",
        argv = "--video-fps",
        min = 0.001,
        max = 1000.0,
        help = "Frame rate the video frames were extracted at (24 when not given)"
    )]
    #[serde(default)]
    pub video_fps: Option<f64>,

    #[job(
        label = "JSON Output",
        group = "Output",
        argv = "--json",
        flag,
        default_bool = false,
        help = "Print one JSON object (the text, token counts, timing, why it stopped) on stdout instead of streaming the text; the banner and statistics go to stderr"
    )]
    #[serde(default)]
    pub json: bool,

    #[job(
        label = "Qwen MTP",
        group = "Speculative decoding",
        argv = "--mtp",
        flag,
        default_bool = false,
        help = "Bundled Qwen3Next/Qwen3.6 multi-token-prediction speculative decoding"
    )]
    #[serde(default)]
    pub mtp: bool,

    #[job(
        label = "MTP Checkpoint",
        group = "Speculative decoding",
        argv = "--mtp-model",
        kind = "path",
        help = "Qwen MTP checkpoint directory; defaults to the mtp.* weights bundled in the model"
    )]
    #[serde(default)]
    pub mtp_model: Option<String>,

    #[job(
        label = "MTP Draft Tokens",
        group = "Speculative decoding",
        argv = "--mtp-draft-tokens",
        min = 1,
        max = 16,
        default_int = 3,
        help = "Draft tokens verified per speculative step"
    )]
    #[serde(default = "default_mtp_draft_tokens")]
    pub mtp_draft_tokens: usize,

    #[job(label = "Tools JSON", group = "Input", argv = "--tools", kind = "path")]
    #[serde(default)]
    pub tools: Option<String>,

    #[job(
        label = "FP8 Weights",
        group = "Quantization",
        argv = "--fp8",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub fp8: bool,

    #[job(
        label = "Experts Dir",
        group = "Model",
        argv = "--experts-dir",
        kind = "path"
    )]
    #[serde(default)]
    pub experts_dir: Option<String>,

    #[job(
        label = "Use ANE",
        group = "Compute",
        argv = "--ane",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub ane: bool,

    #[job(
        label = "ANE Max Context",
        group = "Compute",
        argv = "--ane-max-seq-len",
        default_int = 4096
    )]
    #[serde(default = "default_ane_max_seq_len")]
    pub ane_max_seq_len: usize,

    #[job(
        label = "Benchmark",
        group = "Benchmark",
        argv = "--benchmark",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub benchmark: bool,

    #[job(
        label = "Benchmark Iters",
        group = "Benchmark",
        argv = "--benchmark-iters",
        min = 1,
        max = 1_000_000,
        default_int = 5
    )]
    #[serde(default = "default_benchmark_iters")]
    pub benchmark_iters: usize,

    #[job(
        label = "Benchmark Prompt Tokens",
        group = "Benchmark",
        argv = "--benchmark-prompt-tokens"
    )]
    #[serde(default)]
    pub benchmark_prompt_tokens: Option<usize>,

    #[job(
        label = "Profile Layers",
        group = "Benchmark",
        argv = "--profile-layers",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub profile_layers: bool,

    #[job(
        label = "Profile Output",
        group = "Benchmark",
        argv = "--profile-output",
        kind = "path"
    )]
    #[serde(default)]
    pub profile_output: Option<String>,

    #[job(label = "KV Quant Bits", group = "Quantization", argv = "--kv-quant")]
    #[serde(default)]
    pub kv_quant: Option<u8>,

    #[job(label = "KV K Bits", group = "Quantization", argv = "--kv-k-bits")]
    #[serde(default)]
    pub kv_k_bits: Option<u8>,

    #[job(label = "KV V Bits", group = "Quantization", argv = "--kv-v-bits")]
    #[serde(default)]
    pub kv_v_bits: Option<u8>,

    #[job(
        label = "KV Group Size",
        group = "Quantization",
        argv = "--kv-group-size",
        default_int = 64
    )]
    #[serde(default = "default_kv_group_size")]
    pub kv_group_size: usize,

    #[job(
        label = "KV TurboQuant",
        group = "Quantization",
        argv = "--kv-turboquant",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub kv_turboquant: bool,

    #[job(label = "TurboQuant Preset", group = "Quantization", argv = "--kv-turboquant-preset", kind = "enum", enum_options = ["q2_5", "q3_5"])]
    #[serde(default)]
    pub kv_turboquant_preset: Option<String>,

    #[job(label = "KV Quant Preset", group = "Quantization", argv = "--kv-quant-preset", kind = "enum", enum_options = ["q2_5", "q3_5"])]
    #[serde(default)]
    pub kv_quant_preset: Option<String>,

    #[job(
        label = "Disable KV Quant",
        group = "Quantization",
        argv = "--no-kv-quant",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub no_kv_quant: bool,

    #[job(
        label = "KV QJL Correction",
        group = "Quantization",
        argv = "--kv-qjl",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub kv_qjl: bool,

    #[job(
        label = "Detect Repetition",
        group = "Sampling",
        argv = "--detect-repetition",
        flag,
        default_bool = false
    )]
    #[serde(default)]
    pub detect_repetition: bool,
}

impl Default for InferSpec {
    fn default() -> Self {
        Self {
            model: String::new(),
            lora: None,
            prompt: String::new(),
            messages_file: None,
            max_tokens: None,
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            chat: false,
            system: None,
            no_thinking: false,
            reasoning_effort: None,
            no_preserve_thinking: false,
            chat_template_kwargs: None,
            hide_thinking: false,
            mode: default_mode(),
            backend: default_backend(),
            draft_model: None,
            image: Vec::new(),
            video: Vec::new(),
            video_fps: None,
            json: false,
            mtp: false,
            mtp_model: None,
            mtp_draft_tokens: default_mtp_draft_tokens(),
            tools: None,
            fp8: false,
            experts_dir: None,
            ane: false,
            ane_max_seq_len: default_ane_max_seq_len(),
            benchmark: false,
            benchmark_iters: default_benchmark_iters(),
            benchmark_prompt_tokens: None,
            profile_layers: false,
            profile_output: None,
            kv_quant: None,
            kv_k_bits: None,
            kv_v_bits: None,
            kv_group_size: default_kv_group_size(),
            kv_turboquant: false,
            kv_turboquant_preset: None,
            kv_quant_preset: None,
            no_kv_quant: false,
            kv_qjl: false,
            detect_repetition: false,
        }
    }
}

impl InferSpec {
    /// Run descriptor + cross-field validation.
    pub fn normalize(&mut self) -> Result<(), Vec<FieldError>> {
        let mut errs = self.validate_descriptors();
        // What to answer is a prompt or a conversation, not both and not neither
        // (a benchmark makes its own).
        let has_prompt = !self.prompt.trim().is_empty();
        let has_messages = self
            .messages_file
            .as_deref()
            .is_some_and(|f| !f.trim().is_empty());
        if has_prompt && has_messages {
            errs.push(FieldError::new(
                "messages_file",
                "use either a prompt or a messages file, not both",
            ));
        } else if !has_prompt && !has_messages && !self.benchmark {
            errs.push(FieldError::new(
                "prompt",
                "needs a prompt or a messages file",
            ));
        }
        if self.video_fps.is_some() && self.video.is_empty() {
            errs.push(FieldError::new("video_fps", "needs at least one --video"));
        }
        if self.mtp_model.is_some() && !self.mtp {
            errs.push(FieldError::new("mtp_model", "needs --mtp"));
        }
        if self.mtp && self.draft_model.is_some() {
            errs.push(FieldError::new(
                "draft_model",
                "use either --mtp (bundled Qwen MTP) or --draft-model (Gemma 4 MTP), not both",
            ));
        }
        if errs.is_empty() { Ok(()) } else { Err(errs) }
    }
}

/// One message of a conversation as `--messages-file` holds it (the OpenAI chat
/// shape): the file is a JSON array of these.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    /// `system`, `user`, `assistant` or `tool`.
    pub role: String,
    /// The text.
    #[serde(default)]
    pub content: String,
    /// The thinking behind an assistant turn, apart from its content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// The tool calls an assistant turn made (OpenAI `tool_calls`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// The call a `tool` message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// A message of `role` with `content`.
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            ..Self::default()
        }
    }
}

/// The most a messages file may hold.
pub const MAX_MESSAGES_BYTES: usize = 32 * 1024 * 1024;

/// The conversation in `text`, a JSON array of [`ChatMessage`], checked: it is
/// not empty, every role is known, a system message is only the first, nothing
/// but a tool call may be empty, and the last message is one a model can
/// answer (a user's, or a tool's result). The error names the message.
pub fn parse_messages(text: &str) -> Result<Vec<ChatMessage>, String> {
    if text.len() > MAX_MESSAGES_BYTES {
        return Err(format!("is larger than {} MiB", MAX_MESSAGES_BYTES >> 20));
    }
    let messages: Vec<ChatMessage> = serde_json::from_str(text)
        .map_err(|e| format!("is not a JSON array of chat messages: {e}"))?;
    if messages.is_empty() {
        return Err("has no messages".to_string());
    }
    for (i, message) in messages.iter().enumerate() {
        let n = i + 1;
        match message.role.as_str() {
            "system" if i > 0 => {
                return Err(format!("message {n}: a system message can only come first"));
            }
            "system" | "user" | "assistant" | "tool" => {}
            other => {
                return Err(format!(
                    "message {n}: '{other}' is not a role (use system, user, assistant or tool)"
                ));
            }
        }
        let empty = message.content.trim().is_empty();
        let calls = message.tool_calls.as_ref().is_some_and(|c| !c.is_empty());
        if empty && !(message.role == "assistant" && calls) {
            return Err(format!(
                "message {n}: the {} message is empty",
                message.role
            ));
        }
    }
    let last = &messages[messages.len() - 1];
    if !matches!(last.role.as_str(), "user" | "tool") {
        let article = if last.role == "user" { "a" } else { "an" };
        return Err(format!(
            "ends with {article} {} message; the last message must be a user's (or a tool's result) for the model to answer",
            last.role
        ));
    }
    Ok(messages)
}

fn default_mode() -> String {
    "auto".to_string()
}
fn default_backend() -> String {
    "auto".to_string()
}
fn default_ane_max_seq_len() -> usize {
    4096
}
fn default_benchmark_iters() -> usize {
    5
}
fn default_mtp_draft_tokens() -> usize {
    3
}
fn default_kv_group_size() -> usize {
    64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prompt_or_a_messages_file_but_not_both_or_neither() {
        let ask = |prompt: &str, file: Option<&str>| {
            InferSpec {
                model: "m".into(),
                prompt: prompt.into(),
                messages_file: file.map(Into::into),
                ..Default::default()
            }
            .normalize()
        };
        assert!(ask("hi", None).is_ok());
        assert!(ask("", Some("chat.json")).is_ok());
        let both = ask("hi", Some("chat.json")).unwrap_err();
        assert!(both.iter().any(|e| e.field == "messages_file"), "{both:?}");
        let neither = ask("", None).unwrap_err();
        assert!(neither.iter().any(|e| e.field == "prompt"), "{neither:?}");
        // A benchmark makes its own prompt.
        let mut bench = InferSpec {
            model: "m".into(),
            benchmark: true,
            ..Default::default()
        };
        assert!(bench.normalize().is_ok());
    }

    #[test]
    fn a_conversation_is_checked_message_by_message() {
        let ok = parse_messages(
            r#"[{"role":"system","content":"be brief"},{"role":"user","content":"hi"},
                {"role":"assistant","content":"","tool_calls":[{"id":"c","function":{"name":"f","arguments":"{}"}}]},
                {"role":"tool","content":"42","tool_call_id":"c"}]"#,
        )
        .unwrap();
        assert_eq!(ok.len(), 4);
        for (text, needle) in [
            ("{}", "not a JSON array"),
            ("[]", "no messages"),
            (
                r#"[{"role":"user","content":"a"},{"role":"system","content":"s"},{"role":"user","content":"b"}]"#,
                "message 2: a system message",
            ),
            (
                r#"[{"role":"robot","content":"a"}]"#,
                "'robot' is not a role",
            ),
            (
                r#"[{"role":"user","content":" "}]"#,
                "message 1: the user message is empty",
            ),
            (r#"[{"role":"user","content":"a","tone":"x"}]"#, "tone"),
            (
                r#"[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]"#,
                "ends with an assistant message",
            ),
        ] {
            let err = parse_messages(text).unwrap_err();
            assert!(err.contains(needle), "{text}: {err}");
        }
        let one = vec![ChatMessage::new("user", "hi")];
        assert_eq!(
            parse_messages(&serde_json::to_string(&one).unwrap()).unwrap(),
            one
        );
    }

    #[test]
    fn argv_round_trip() {
        let spec = InferSpec {
            model: "m".into(),
            prompt: "hi".into(),
            ..Default::default()
        };
        let argv = spec.to_argv();
        assert!(argv.contains(&"--model".to_string()));
        assert!(argv.contains(&"--prompt".to_string()));
    }

    #[test]
    fn bundled_mtp_and_a_draft_model_are_one_or_the_other() {
        let mut spec = InferSpec {
            model: "m".into(),
            prompt: "hi".into(),
            mtp: true,
            draft_model: Some("d".into()),
            ..Default::default()
        };
        let errors = spec.normalize().unwrap_err();
        assert!(
            errors
                .iter()
                .any(|e| e.field == "draft_model" && e.message.contains("not both")),
            "{errors:?}"
        );
        spec.draft_model = None;
        assert!(spec.normalize().is_ok());
    }

    #[test]
    fn unknown_backend_is_rejected() {
        // DFlash runs as `pmetal dflash`, not an `infer` backend, and the
        // other three were generation paths that are gone.
        for backend in ["dflash", "compiled", "metal-sampler", "minimal"] {
            let mut spec = InferSpec {
                model: "m".into(),
                prompt: "hi".into(),
                backend: backend.into(),
                ..Default::default()
            };
            assert!(spec.normalize().is_err(), "{backend} was accepted");
        }
    }
}
