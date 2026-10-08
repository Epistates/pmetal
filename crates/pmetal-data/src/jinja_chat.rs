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
//! * **Template kwargs** ([`JinjaRenderOptions::template_kwargs`]) land in
//!   the render context exactly as `apply_chat_template(..., **kwargs)`
//!   passes them: next to the tokenizer's special tokens, which they
//!   override. That is how a template reads `reasoning_effort`,
//!   `preserve_thinking`, `model_identity` and the like.
//! * **Same environment as transformers**: `trim_blocks` and `lstrip_blocks`
//!   on, `{% break %}` / `{% continue %}` available, and a `tojson` that is
//!   Python's `json.dumps` (`", "` / `": "` separators, no HTML escaping)
//!   rather than minijinja's compact, HTML-safe one.

use minijinja::value::{Kwargs, ValueKind};
use minijinja::{Environment, Error, ErrorKind, Value};
use serde::Serialize;

/// Context names `apply_chat_template` binds itself. A template kwarg of the
/// same name is a `TypeError` there ("got multiple values for keyword
/// argument"), so it is refused here too.
pub const RESERVED_TEMPLATE_KWARGS: &[&str] =
    &["messages", "tools", "documents", "add_generation_prompt"];

/// A single chat message in the shape HF Jinja templates expect.
#[derive(Debug, Clone, Serialize)]
pub struct JinjaMessage {
    /// `"user"` / `"assistant"` / `"system"` / `"tool"`.
    pub role: String,
    /// The message's content: a string, or a list of `{"type": "text" |
    /// "image" | "video", ...}` items for a message carrying media. Tools-only
    /// messages can set this to an empty string.
    pub content: serde_json::Value,
    /// The thinking that preceded an assistant turn's `content`, kept apart
    /// from it. Templates that render thinking themselves (Qwen3, Qwen3.5
    /// and later) read `message.reasoning_content`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Structured tool calls (serialized as a list of objects). Kept
    /// optional so the default case stays zero-overhead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// Tool call response payload (role="tool").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl JinjaMessage {
    fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: serde_json::Value::String(content.into()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
    /// Build a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }
    /// Build a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new("system", content)
    }
    /// Build an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
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
    /// Extra keyword arguments for the template, the `**kwargs` of
    /// `apply_chat_template`: `reasoning_effort`, `preserve_thinking`, or
    /// anything else a template reads. They override `bos_token` /
    /// `eos_token` and `enable_thinking` of the same name; the names in
    /// [`RESERVED_TEMPLATE_KWARGS`] are refused.
    pub template_kwargs: serde_json::Map<String, serde_json::Value>,
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
    if let Some(key) = options
        .template_kwargs
        .keys()
        .find(|key| RESERVED_TEMPLATE_KWARGS.contains(&key.as_str()))
    {
        return Err(format!(
            "template kwarg `{key}` is not allowed: the renderer sets it itself"
        ));
    }

    let mut env = Environment::new();
    // transformers compiles chat templates with `trim_blocks=True,
    // lstrip_blocks=True`: the newline after a block tag and the whitespace
    // before one are dropped. Templates that write every tag with `-` render
    // the same either way; Llama 3.2 Vision's does not, and without these its
    // prompt gained a newline after `<|begin_of_text|>`.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    // ...and with its own `tojson`, Python's `json.dumps`.
    env.add_filter("tojson", py_tojson);
    // `pycompat` lets templates use Python-style attribute access on dicts
    // (e.g. `message.role` and `message['role']` interchangeably), which
    // HF templates rely on heavily.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

    // `strftime_now` / `now` — used by Llama 3.1/3.2 `Today Date: ...`.
    minijinja_contrib::add_to_environment(&mut env);

    // HF templates call `raise_exception(...)` to signal invalid input.
    // minijinja's built-in `raise_exception` isn't available, so install a
    // tiny shim that turns it into a template error.
    env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
        Err(Error::new(ErrorKind::InvalidOperation, msg))
    });

    // Some templates call `strftime_now(fmt)` without any date argument.
    // minijinja-contrib exposes it as `now().strftime(fmt)`, so add the
    // upstream-HF shim that matches `transformers`' default handler.
    env.add_function("strftime_now", |fmt: String| -> Result<String, Error> {
        let now =
            chrono_now_strftime(&fmt).map_err(|e| Error::new(ErrorKind::InvalidOperation, e))?;
        Ok(now)
    });

    // Register the template under a stable name. minijinja requires the
    // source to live for the lifetime of the environment, so clone it.
    let template_name = "chat_template";
    env.add_template_owned(template_name, jinja_src.to_owned())
        .map_err(|e| format!("compile: {e}"))?;

    let tmpl = env
        .get_template(template_name)
        .map_err(|e| format!("lookup: {e}"))?;

    // Build the context the way `apply_chat_template` does: the special
    // tokens, then the caller's kwargs over them, then the four names it
    // binds itself. `enable_thinking` is omitted when the caller passes
    // `None`, so `enable_thinking is defined` evaluates to false.
    let mut ctx: Vec<(String, Value)> = vec![
        (
            "bos_token".into(),
            Value::from(options.bos_token.clone().unwrap_or_default()),
        ),
        (
            "eos_token".into(),
            Value::from(options.eos_token.clone().unwrap_or_default()),
        ),
    ];
    if let Some(thinking) = options.enable_thinking {
        ctx.push(("enable_thinking".into(), Value::from(thinking)));
    }
    for (key, value) in &options.template_kwargs {
        ctx.retain(|(existing, _)| existing != key);
        ctx.push((key.clone(), Value::from_serialize(value)));
    }
    let tools_value = match options.tools.as_ref() {
        Some(t) => Value::from_serialize(t),
        None => Value::from(()),
    };
    ctx.extend([
        ("messages".into(), Value::from_serialize(messages)),
        ("tools".into(), tools_value),
        ("documents".into(), Value::from(())),
        (
            "add_generation_prompt".into(),
            Value::from(options.add_generation_prompt),
        ),
    ]);

    tmpl.render(Value::from_iter(ctx))
        .map_err(|e| format!("render: {e}"))
}

/// `tojson` as transformers installs it: Python's
/// `json.dumps(x, ensure_ascii=False, indent=None, separators=None,
/// sort_keys=False)`, with each of those four as a keyword argument.
///
/// minijinja's own filter writes compact JSON with `<`, `>`, `&` and `'`
/// escaped for HTML, so every tool schema a template embedded came out
/// different from what the model was trained on.
fn py_tojson(value: &Value, kwargs: Kwargs) -> Result<Value, Error> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<Value> = kwargs.get("indent")?;
    let separators: Option<Value> = kwargs.get("separators")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    kwargs.assert_all_used()?;

    let indent = match indent {
        None => None,
        Some(v) if v.is_none() => None,
        Some(v) => match v.as_str() {
            Some(s) => Some(s.to_string()),
            None => Some(" ".repeat(usize::try_from(v)?)),
        },
    };
    let (item_sep, key_sep) = match separators {
        Some(v) if !v.is_none() => {
            let parts: Vec<String> = v
                .try_iter()?
                .map(|p| p.as_str().map(str::to_string))
                .collect::<Option<_>>()
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::InvalidOperation,
                        "tojson: separators must be two strings",
                    )
                })?;
            match <[String; 2]>::try_from(parts) {
                Ok([item, key]) => (item, key),
                Err(_) => {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        "tojson: separators must be two strings",
                    ));
                }
            }
        }
        // json.dumps: `(', ', ': ')`, or `(',', ': ')` when indenting.
        _ if indent.is_some() => (",".to_string(), ": ".to_string()),
        _ => (", ".to_string(), ": ".to_string()),
    };
    let dumper = PyJsonDumper {
        ensure_ascii: ensure_ascii.unwrap_or(false),
        indent,
        item_sep,
        key_sep,
        sort_keys: sort_keys.unwrap_or(false),
    };
    let mut out = String::new();
    dumper.write(value, 0, &mut out)?;
    Ok(Value::from_safe_string(out))
}

/// The encoder behind [`py_tojson`]: `json.dumps` byte for byte.
struct PyJsonDumper {
    ensure_ascii: bool,
    indent: Option<String>,
    item_sep: String,
    key_sep: String,
    sort_keys: bool,
}

impl PyJsonDumper {
    fn newline(&self, level: usize, out: &mut String) {
        if let Some(indent) = &self.indent {
            out.push('\n');
            for _ in 0..level {
                out.push_str(indent);
            }
        }
    }

    fn write(&self, value: &Value, level: usize, out: &mut String) -> Result<(), Error> {
        match value.kind() {
            ValueKind::None => out.push_str("null"),
            ValueKind::Undefined => {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "tojson: Undefined is not JSON serializable",
                ));
            }
            ValueKind::Bool => out.push_str(if value.is_true() { "true" } else { "false" }),
            ValueKind::Number if value.is_integer() => out.push_str(&value.to_string()),
            ValueKind::Number => out.push_str(&py_float_repr(f64::try_from(value.clone())?)),
            ValueKind::String => self.write_str(value.as_str().unwrap_or_default(), out),
            ValueKind::Seq | ValueKind::Iterable => {
                let items: Vec<Value> = value.try_iter()?.collect();
                if items.is_empty() {
                    out.push_str("[]");
                    return Ok(());
                }
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(&self.item_sep);
                    }
                    self.newline(level + 1, out);
                    self.write(item, level + 1, out)?;
                }
                self.newline(level, out);
                out.push(']');
            }
            ValueKind::Map => {
                let mut keys: Vec<Value> = value.try_iter()?.collect();
                if keys.is_empty() {
                    out.push_str("{}");
                    return Ok(());
                }
                if self.sort_keys {
                    keys.sort();
                }
                out.push('{');
                for (i, key) in keys.iter().enumerate() {
                    if i > 0 {
                        out.push_str(&self.item_sep);
                    }
                    self.newline(level + 1, out);
                    // json.dumps coerces None / bool / number keys to strings.
                    let key_str = match key.kind() {
                        ValueKind::String => key.as_str().unwrap_or_default().to_string(),
                        ValueKind::None => "null".into(),
                        ValueKind::Bool => (if key.is_true() { "true" } else { "false" }).into(),
                        _ => key.to_string(),
                    };
                    self.write_str(&key_str, out);
                    out.push_str(&self.key_sep);
                    self.write(&value.get_item(key)?, level + 1, out)?;
                }
                self.newline(level, out);
                out.push('}');
            }
            _ => {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("tojson: {} is not JSON serializable", value.kind()),
                ));
            }
        }
        Ok(())
    }

    fn write_str(&self, s: &str, out: &mut String) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                '\u{08}' => out.push_str("\\b"),
                '\u{0c}' => out.push_str("\\f"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c if self.ensure_ascii && !c.is_ascii() => {
                    let mut units = [0u16; 2];
                    for unit in c.encode_utf16(&mut units) {
                        out.push_str(&format!("\\u{unit:04x}"));
                    }
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }
}

/// Python's `repr(float)` (what `json.dumps` writes): the shortest digits that
/// round-trip, positional for exponents in `[-4, 16)`, otherwise `1e-05` /
/// `1e+16` style; `Infinity` / `NaN` for the non-finite values.
fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    // `{:e}` gives the shortest round-trip digits: `-1.2345e-5`.
    let sci = format!("{f:e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp) {
        let point = exp + 1; // digits before the decimal point
        let body = if point <= 0 {
            format!("0.{}{}", "0".repeat((-point) as usize), digits)
        } else if point as usize >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(point as usize - digits.len()))
        } else {
            let (int, frac) = digits.split_at(point as usize);
            format!("{int}.{frac}")
        };
        format!("{sign}{body}")
    } else {
        let mantissa = if digits.len() == 1 {
            digits
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let exp_sign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{mantissa}e{exp_sign}{:02}", exp.abs())
    }
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

    fn render_with_kwargs(src: &str, kwargs: serde_json::Value) -> Result<String, String> {
        render_chat_template(
            src,
            &[],
            &JinjaRenderOptions {
                template_kwargs: kwargs.as_object().unwrap().clone(),
                ..Default::default()
            },
        )
    }

    #[test]
    fn tojson_matches_python_json_dumps() {
        let value = serde_json::json!({
            "b": [1, 2.5, null, true], "a": "x<y> & 'z' \"q\" é\n", "n": {}, "e": [],
        });
        // json.dumps(value, ensure_ascii=False)
        assert_eq!(
            render_with_kwargs("{{ v|tojson }}", serde_json::json!({ "v": value })).unwrap(),
            r#"{"b": [1, 2.5, null, true], "a": "x<y> & 'z' \"q\" é\n", "n": {}, "e": []}"#
        );
        // json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
        assert_eq!(
            render_with_kwargs(
                "{{ v|tojson(indent=2, sort_keys=true, ensure_ascii=true) }}",
                serde_json::json!({ "v": value }),
            )
            .unwrap(),
            "{\n  \"a\": \"x<y> & 'z' \\\"q\\\" \\u00e9\\n\",\n  \"b\": [\n    1,\n    2.5,\n    null,\n    true\n  ],\n  \"e\": [],\n  \"n\": {}\n}"
        );
        // json.dumps(value, separators=(',', ':'))
        assert_eq!(
            render_with_kwargs(
                "{{ v|tojson(separators=[',', ':']) }}",
                serde_json::json!({ "v": {"k": [1, 2]} }),
            )
            .unwrap(),
            r#"{"k":[1,2]}"#
        );
    }

    #[test]
    fn float_repr_matches_python() {
        for (f, py) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (-0.5, "-0.5"),
            (1e-5, "1e-05"),
            (0.00015, "0.00015"),
            (1e16, "1e+16"),
            (1234567890123456.0, "1234567890123456.0"),
            (2.5e-7, "2.5e-07"),
            (123.456, "123.456"),
            (f64::INFINITY, "Infinity"),
        ] {
            assert_eq!(py_float_repr(f), py, "{f}");
        }
    }

    #[test]
    fn template_kwargs_override_special_tokens_and_refuse_reserved() {
        let src = "{{ bos_token }}|{{ reasoning_effort }}|{{ enable_thinking }}";
        let out = render_chat_template(
            src,
            &[],
            &JinjaRenderOptions {
                bos_token: Some("<s>".into()),
                enable_thinking: Some(true),
                template_kwargs: serde_json::json!({
                    "bos_token": "<B>", "reasoning_effort": "low", "enable_thinking": false,
                })
                .as_object()
                .unwrap()
                .clone(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(out, "<B>|low|False");
        assert!(render_with_kwargs("x", serde_json::json!({ "messages": [] })).is_err());
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
