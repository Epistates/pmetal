//! Execute upstream Jinja chat templates from `tokenizer_config.json`.
//!
//! HuggingFace models ship a Jinja template string that fully describes how
//! `messages` map to the tokenized prompt, including model-specific default
//! system messages, tool formatting, thinking-mode toggles, and dynamic
//! fields like the current date (Llama 3). Re-implementing each template
//! in Rust by hand is brittle — upstream Gemma 4 alone uses ~200 lines of
//! Jinja with macros, namespaces, and nested control flow.
//!
//! This module wraps [`minijinja`] + [`minijinja-contrib`] to execute those
//! templates directly. The result is a bit-exact match against
//! `transformers.AutoTokenizer.apply_chat_template(...)` for every model in
//! the parity audit suite.
//!
//! Design notes:
//!
//! * **Python-compatible attribute access.** HF templates use `message.role`,
//!   `message['role']`, and `messages[0]`. We enable the `pycompat` syntax
//!   support from `minijinja-contrib` so dict / attr access just works.
//! * **`strftime_now`** is installed from `minijinja-contrib::datetime` —
//!   Llama 3.1/3.2 uses it to inject `Today Date: <today>` into its default
//!   system block.
//! * **`raise_exception`** is a no-op that returns an error message — HF
//!   templates use it to assert invariants that we should surface as a
//!   render failure rather than panic.
//! * **bos/eos tokens** are passed in as globals because templates often
//!   concatenate `bos_token + '<|turn>user…'` rather than having the
//!   tokenizer add them post-hoc.
//! * **Custom prefill (`add_generation_prompt`)** and `enable_thinking` are
//!   the two main knobs callers flip.

use minijinja::{Environment, Error, ErrorKind, Value, context};
use serde::Serialize;

/// A single chat message in the shape HF Jinja templates expect.
#[derive(Debug, Clone, Serialize)]
pub struct JinjaMessage {
    /// `"user"` / `"assistant"` / `"system"` / `"tool"`.
    pub role: String,
    /// Raw message text. Tools-only messages can set this to an empty string.
    pub content: String,
    /// Structured tool calls (serialized as a list of objects). Kept
    /// optional so the default case stays zero-overhead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// Tool call response payload (role="tool").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl JinjaMessage {
    /// Build a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }
    /// Build a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }
    /// Build an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// Runtime options passed to the Jinja renderer.
#[derive(Debug, Clone, Default)]
pub struct JinjaRenderOptions {
    /// `add_generation_prompt` — when true, emits the assistant generation
    /// prefill (e.g. `<|turn>model\n` for Gemma 4). Callers generating from
    /// a user turn almost always want this true.
    pub add_generation_prompt: bool,
    /// `enable_thinking` — controls whether reasoning-mode templates
    /// (Gemma 4, Qwen 3, DeepSeek R1) insert their `<|think|>` / `<think>`
    /// prefill. When set to `Some(v)` the value is exposed verbatim; when
    /// `None` it's absent from the context (so templates that probe
    /// `enable_thinking is defined` see it as undefined).
    pub enable_thinking: Option<bool>,
    /// Optional tool definitions. Passed verbatim as `tools` in the
    /// template context. Each entry should already be in the shape HF
    /// expects (an object with `type`/`function` fields).
    pub tools: Option<Vec<serde_json::Value>>,
    /// Tokenizer's bos_token string, e.g. `<bos>`. Exposed as `bos_token`.
    pub bos_token: Option<String>,
    /// Tokenizer's eos_token string. Exposed as `eos_token`.
    pub eos_token: Option<String>,
}

/// Render a HuggingFace chat-template Jinja string into a prompt.
///
/// Returns the rendered string on success, or an error containing the
/// Jinja error chain on failure. The caller is responsible for tokenizing
/// the result (pmetal's `Tokenizer::encode` handles the embedded special
/// tokens correctly — see the Gemma 4 investigation).
pub fn render_chat_template(
    jinja_src: &str,
    messages: &[JinjaMessage],
    options: &JinjaRenderOptions,
) -> Result<String, String> {
    let mut env = Environment::new();
    // `pycompat` lets templates use Python-style attribute access on dicts
    // (e.g. `message.role` and `message['role']` interchangeably), which
    // HF templates rely on heavily.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

    // `strftime_now` / `now` — used by Llama 3.1/3.2 `Today Date: ...`.
    minijinja_contrib::add_to_environment(&mut env);

    // HF templates call `raise_exception(...)` to signal invalid input.
    // minijinja's built-in `raise_exception` isn't available, so install a
    // tiny shim that turns it into a template error.
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        },
    );

    // Some templates call `strftime_now(fmt)` without any date argument.
    // minijinja-contrib exposes it as `now().strftime(fmt)`, so add the
    // upstream-HF shim that matches `transformers`' default handler.
    env.add_function(
        "strftime_now",
        |fmt: String| -> Result<String, Error> {
            let now = chrono_now_strftime(&fmt)
                .map_err(|e| Error::new(ErrorKind::InvalidOperation, e))?;
            Ok(now)
        },
    );

    // Register the template under a stable name. minijinja requires the
    // source to live for the lifetime of the environment, so clone it.
    let template_name = "chat_template";
    env.add_template_owned(template_name, jinja_src.to_owned())
        .map_err(|e| format!("compile: {e:#}"))?;

    let tmpl = env
        .get_template(template_name)
        .map_err(|e| format!("lookup: {e:#}"))?;

    // Build the context. `messages` serialises via serde; `enable_thinking`
    // is omitted from the context when the caller passes `None` so
    // `enable_thinking is defined` in the template evaluates to false.
    let messages_value = Value::from_serialize(messages);
    let tools_value = match options.tools.as_ref() {
        Some(t) => Value::from_serialize(t),
        None => Value::from(None::<bool>),
    };

    let ctx = if let Some(thinking) = options.enable_thinking {
        context! {
            messages => messages_value,
            add_generation_prompt => options.add_generation_prompt,
            enable_thinking => thinking,
            tools => tools_value,
            bos_token => options.bos_token.clone().unwrap_or_default(),
            eos_token => options.eos_token.clone().unwrap_or_default(),
        }
    } else {
        context! {
            messages => messages_value,
            add_generation_prompt => options.add_generation_prompt,
            tools => tools_value,
            bos_token => options.bos_token.clone().unwrap_or_default(),
            eos_token => options.eos_token.clone().unwrap_or_default(),
        }
    };

    tmpl.render(ctx).map_err(|e| format!("render: {e:#}"))
}

/// Format the current time via `chrono` in the caller's requested strftime
/// pattern. Used as the fallback implementation of HF's `strftime_now`.
///
/// The `PMETAL_CHAT_TEMPLATE_FROZEN_DATE` env var overrides the "current"
/// date with a fixed `YYYY-MM-DD` value (midnight UTC). This lets parity
/// tests that bake Llama-3-style `Today Date: DD Mon YYYY` strings into
/// fixtures stay reproducible — without this, `chat_template_audit` flips
/// from green to red at UTC midnight because the rendered date no longer
/// matches the dumped fixture. Any non-empty value that fails to parse is
/// ignored silently (falls through to real `now_utc()`), so the override
/// is strictly additive.
fn chrono_now_strftime(fmt: &str) -> Result<String, String> {
    // We avoid pulling in `chrono` by delegating to `time`, which is
    // already in the dep tree via minijinja-contrib. Format follows the
    // same `%d %b %Y`-style directives.
    use time::OffsetDateTime;
    use time::format_description;
    use time::macros::format_description as fd_macro;

    let fmt_owned = translate_strftime(fmt);
    let desc = format_description::parse_borrowed::<2>(&fmt_owned)
        .map_err(|e| format!("strftime_now: bad format {fmt:?}: {e}"))?;
    let now = match std::env::var("PMETAL_CHAT_TEMPLATE_FROZEN_DATE") {
        Ok(v) if !v.is_empty() => {
            let iso = fd_macro!("[year]-[month]-[day]");
            match time::Date::parse(&v, &iso) {
                Ok(d) => d.midnight().assume_utc(),
                Err(_) => OffsetDateTime::now_utc(),
            }
        }
        _ => OffsetDateTime::now_utc(),
    };
    now.format(&desc)
        .map_err(|e| format!("strftime_now: format error: {e}"))
}

/// Translate a subset of C-style `strftime` directives into the
/// `time` crate's format description syntax. Only the directives actually
/// used by Llama 3 / Mistral / upstream HF templates are supported —
/// anything else passes through verbatim.
fn translate_strftime(src: &str) -> String {
    let mut out = String::with_capacity(src.len() + 16);
    let mut chars = src.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str("[year]"),
            Some('y') => out.push_str("[year repr:last_two]"),
            Some('m') => out.push_str("[month]"),
            Some('B') => out.push_str("[month repr:long]"),
            Some('b') | Some('h') => out.push_str("[month repr:short]"),
            Some('d') => out.push_str("[day]"),
            Some('e') => out.push_str("[day padding:space]"),
            Some('H') => out.push_str("[hour]"),
            Some('I') => out.push_str("[hour repr:12]"),
            Some('M') => out.push_str("[minute]"),
            Some('S') => out.push_str("[second]"),
            Some('p') => out.push_str("[period]"),
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_simple_chatml() {
        let src = r#"{% for m in messages %}<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#;
        let msgs = vec![JinjaMessage::user("hi")];
        let out = render_chat_template(
            src,
            &msgs,
            &JinjaRenderOptions {
                add_generation_prompt: true,
                ..Default::default()
            },
        )
        .expect("render");
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn render_with_bos() {
        let src = "{{ bos_token }}[INST] {{ messages[0].content }} [/INST]";
        let msgs = vec![JinjaMessage::user("ping")];
        let out = render_chat_template(
            src,
            &msgs,
            &JinjaRenderOptions {
                add_generation_prompt: true,
                bos_token: Some("<s>".into()),
                ..Default::default()
            },
        )
        .expect("render");
        assert_eq!(out, "<s>[INST] ping [/INST]");
    }

    #[test]
    fn render_enable_thinking_defined() {
        let src = r#"{% if enable_thinking is defined and enable_thinking %}THINK
{% endif %}{{ messages[0].content }}"#;
        let msgs = vec![JinjaMessage::user("hi")];

        let out_on = render_chat_template(
            src,
            &msgs,
            &JinjaRenderOptions {
                enable_thinking: Some(true),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(out_on, "THINK\nhi");

        let out_off = render_chat_template(src, &msgs, &JinjaRenderOptions::default()).unwrap();
        assert_eq!(out_off, "hi");
    }

    #[test]
    fn render_strftime_now() {
        // Assert the directive at least emits a 4-digit year.
        let src = r#"{{ strftime_now('%Y') }}"#;
        let out = render_chat_template(src, &[], &JinjaRenderOptions::default()).unwrap();
        assert_eq!(out.len(), 4);
        assert!(out.chars().all(|c| c.is_ascii_digit()));
    }
}
