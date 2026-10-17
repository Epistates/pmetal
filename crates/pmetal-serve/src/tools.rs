//! Tool calls in a model's output.
//!
//! Each model family writes a tool call its own way, as its chat template
//! teaches it by rendering earlier calls back into the prompt:
//!
//! | family | the call |
//! |--------|----------|
//! | Hermes / Qwen 2.5, 3 | `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` |
//! | Qwen 3.5, Qwen3-Coder | `<tool_call><function=NAME><parameter=P>value</parameter></function></tool_call>` |
//! | Llama 3.1+ | `<\|python_tag\|>{"name": ..., "parameters": {...}}`, or the bare JSON object as the whole reply |
//! | Mistral | `[TOOL_CALLS][{"name": ..., "arguments": {...}}]`, or `[TOOL_CALLS]NAME[ARGS]{...}` |
//! | Gemma 4 | `<\|tool_call>call:NAME{key:<\|"\|>value<\|"\|>}<tool_call\|>` |
//! | gpt-oss (Harmony) | `<\|channel\|>commentary to=functions.NAME <\|constrain\|>json<\|message\|>{...}<\|call\|>` |
//!
//! [`ToolFormat::detect`] picks the family from the chat template, and a
//! [`ToolStream`] cuts a reply's text into plain text and parsed calls as the
//! text arrives, so the markup never reaches `content` and a call's name and
//! arguments can be streamed as OpenAI's `tool_calls` deltas.

use pmetal_data::chat_templates::{ChatTemplate, ChatTemplateType, ToolDefinition};
use serde_json::{Map, Value};

/// How a model family writes its tool calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// `<tool_call>` ... `</tool_call>` around a JSON object.
    Hermes,
    /// `<tool_call>` ... `</tool_call>` around the `<function=NAME>` XML of
    /// Qwen 3.5 and Qwen3-Coder (the parser reads either body).
    QwenXml,
    /// `<|python_tag|>` before JSON objects, or a reply that is one.
    Llama3,
    /// `[TOOL_CALLS]` before a JSON array, or `NAME[ARGS]{...}`.
    Mistral,
    /// `<|tool_call>call:NAME{...}<tool_call|>`.
    Gemma4,
    /// Harmony messages addressed to `functions.NAME`.
    Harmony,
}

impl ToolFormat {
    /// The family whose calls this chat template renders and trains on, read
    /// from the template's markers (the model's own Jinja is the authority
    /// on what it writes), else its type. `None` for a template that knows
    /// no tools.
    pub fn detect(template: &ChatTemplate) -> Option<Self> {
        if let Some(source) = template.jinja_source.as_deref() {
            if source.contains("<|tool_call>") {
                return Some(Self::Gemma4);
            }
            if source.contains("<|channel|>") && source.contains("functions.") {
                return Some(Self::Harmony);
            }
            if source.contains("[TOOL_CALLS]") {
                return Some(Self::Mistral);
            }
            if source.contains("<|python_tag|>") {
                return Some(Self::Llama3);
            }
            if source.contains("<tool_call>") && source.contains("<function=") {
                return Some(Self::QwenXml);
            }
            if source.contains("<tool_call>") {
                return Some(Self::Hermes);
            }
        }
        match template.template_type {
            ChatTemplateType::Gemma4 => Some(Self::Gemma4),
            ChatTemplateType::GptOss => Some(Self::Harmony),
            ChatTemplateType::Mistral => Some(Self::Mistral),
            ChatTemplateType::Llama3 | ChatTemplateType::Llama4 => Some(Self::Llama3),
            ChatTemplateType::ChatMl | ChatTemplateType::Qwen => Some(Self::Hermes),
            _ => None,
        }
    }

    /// The text that opens a call, when there is one.
    fn start(self) -> Option<&'static str> {
        match self {
            Self::Hermes | Self::QwenXml => Some("<tool_call>"),
            Self::Llama3 => Some("<|python_tag|>"),
            Self::Mistral => Some("[TOOL_CALLS]"),
            Self::Gemma4 => Some("<|tool_call>"),
            Self::Harmony => Some("<|channel|>commentary to=functions."),
        }
    }

    /// The text that closes a call; `None` when the call runs to the end of
    /// the reply.
    fn end(self) -> Option<&'static str> {
        match self {
            Self::Hermes | Self::QwenXml => Some("</tool_call>"),
            Self::Gemma4 => Some("<tool_call|>"),
            Self::Harmony => Some("<|call|>"),
            Self::Llama3 | Self::Mistral => None,
        }
    }

    /// Whether the whole reply being one JSON call object also counts: Llama
    /// 3.1 is trained to answer a tool request that way, without its
    /// `<|python_tag|>` marker. The call must name an offered tool; a JSON
    /// answer that merely has a `name` key is not a call.
    fn bare_json(self) -> bool {
        matches!(self, Self::Llama3)
    }
}

/// What a request's `tool_choice` asks of the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// The model's own call: tools are offered and it may use them.
    Auto,
    /// The model is not told about the tools.
    None,
    /// The model must call some tool.
    Required,
    /// The model must call this tool.
    Named(String),
}

/// What a request does with the tools it offers: what the prompt lists, how
/// the reply's calls are read, and what the model must do with them.
#[derive(Debug, Clone)]
pub struct ToolPlan {
    /// Tools to render into the prompt.
    pub prompt: Option<Vec<ToolDefinition>>,
    /// Parse calls out of the reply as this family writes them.
    pub parse: Option<(ToolFormat, ToolSet)>,
    /// Keep only the first call of a reply (`parallel_tool_calls: false`).
    pub single_call: bool,
    /// What the model must call, if anything.
    pub choice: ToolChoice,
}

impl ToolPlan {
    /// No tools in play.
    pub fn none() -> Self {
        Self {
            prompt: None,
            parse: None,
            single_call: false,
            choice: ToolChoice::None,
        }
    }

    /// Offer `tools` under `choice`, in the calling convention `format`.
    ///
    /// # Errors
    ///
    /// A named tool that is not among `tools`, or a requirement with no
    /// tools to meet it.
    pub fn new(
        tools: Vec<ToolDefinition>,
        choice: ToolChoice,
        single_call: bool,
        format: ToolFormat,
    ) -> Result<Self, String> {
        match (&choice, tools.is_empty()) {
            (ToolChoice::Required | ToolChoice::Named(_), true) => {
                return Err("`tool_choice` requires a tool, but no `tools` were given".into());
            }
            (ToolChoice::Named(name), false) if !tools.iter().any(|t| &t.function.name == name) => {
                return Err(format!(
                    "`tool_choice` names `{name}`, which is not among `tools`"
                ));
            }
            _ => {}
        }
        if tools.is_empty() || choice == ToolChoice::None {
            return Ok(Self {
                single_call,
                choice,
                ..Self::none()
            });
        }
        Ok(Self {
            parse: Some((format, ToolSet::new(&tools))),
            prompt: Some(tools),
            single_call,
            choice,
        })
    }
}

/// A call parsed from the reply.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedCall {
    pub name: String,
    /// The arguments as JSON text, an object.
    pub arguments: String,
}

/// What a [`ToolStream`] makes of the text so far.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolEvent {
    /// Reply text for `content`.
    Text(String),
    /// A complete call.
    Call(ParsedCall),
}

/// The tools a request offers: what a call's arguments are coerced against.
#[derive(Debug, Clone, Default)]
pub struct ToolSet {
    tools: Vec<ToolDefinition>,
}

impl ToolSet {
    pub fn new(tools: &[ToolDefinition]) -> Self {
        Self {
            tools: tools.to_vec(),
        }
    }

    /// Whether `name` is an offered tool.
    fn offers(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.function.name == name)
    }

    /// The `type` JSON Schema gives parameter `param` of tool `name`.
    fn parameter_type(&self, name: &str, param: &str) -> Option<&str> {
        let tool = self.tools.iter().find(|t| t.function.name == name)?;
        tool.function
            .parameters
            .as_ref()?
            .get("properties")?
            .get(param)?
            .get("type")?
            .as_str()
    }
}

/// Splits a reply into text and tool calls as its text arrives.
#[derive(Debug)]
pub struct ToolStream {
    format: ToolFormat,
    tools: ToolSet,
    /// Text received and not yet decided.
    buf: String,
    in_call: bool,
    /// Whitespace between a call and what follows it, held back: a model
    /// writes a newline between calls that is no part of the reply.
    pending_space: String,
    calls: usize,
    /// Whether anything but whitespace has been seen outside a call.
    started: bool,
    /// Text of the reply from its start, kept while it could still be one
    /// bare JSON call.
    maybe_bare: bool,
}

impl ToolStream {
    pub fn new(format: ToolFormat, tools: ToolSet) -> Self {
        Self {
            format,
            tools,
            buf: String::new(),
            in_call: false,
            pending_space: String::new(),
            calls: 0,
            started: false,
            maybe_bare: format.bare_json(),
        }
    }

    /// How many calls have been parsed.
    pub fn calls(&self) -> usize {
        self.calls
    }

    /// Take in `text`; returns what is now known.
    pub fn push(&mut self, text: &str) -> Vec<ToolEvent> {
        self.buf.push_str(text);
        let mut events = Vec::new();
        self.drain(false, &mut events);
        events
    }

    /// The reply is over: whatever is left is text, or a call that ran to
    /// the end of the reply.
    pub fn finish(&mut self) -> Vec<ToolEvent> {
        let mut events = Vec::new();
        self.drain(true, &mut events);
        if self.calls == 0 && !self.pending_space.is_empty() {
            // A reply that called nothing keeps its trailing whitespace.
            events.push(ToolEvent::Text(std::mem::take(&mut self.pending_space)));
        }
        events
    }

    fn drain(&mut self, last: bool, events: &mut Vec<ToolEvent>) {
        loop {
            if self.in_call {
                if !self.drain_call(last, events) {
                    return;
                }
                continue;
            }
            // The reply may be a bare JSON call: decided at its first
            // non-space character, parsed at its end.
            if self.maybe_bare && !self.started {
                let trimmed = self.buf.trim_start();
                match trimmed.chars().next() {
                    None => return,
                    Some('{' | '[') => {
                        if last {
                            let body = std::mem::take(&mut self.buf);
                            match self.parse_body(&body, true) {
                                Some(calls) => self.push_calls(calls, events),
                                None => events.push(ToolEvent::Text(body)),
                            }
                        }
                        return;
                    }
                    Some(_) => self.maybe_bare = false,
                }
            }
            let Some(start) = self.format.start() else {
                self.flush_text(last, events);
                return;
            };
            match self.buf.find(start) {
                Some(at) => {
                    let before: String = self.buf.drain(..at).collect();
                    self.buf.drain(..start.len());
                    self.emit_text(before, events);
                    self.in_call = true;
                }
                None => {
                    // Hold back a tail that could still become the marker.
                    let keep = if last {
                        0
                    } else {
                        partial_suffix(&self.buf, start)
                    };
                    let ready: String = self.buf.drain(..self.buf.len() - keep).collect();
                    self.emit_text(ready, events);
                    if last {
                        let rest = std::mem::take(&mut self.buf);
                        self.emit_text(rest, events);
                    }
                    return;
                }
            }
        }
    }

    /// Reply text outside any call, for formats with no call marker.
    fn flush_text(&mut self, last: bool, events: &mut Vec<ToolEvent>) {
        let text = std::mem::take(&mut self.buf);
        self.emit_text(text, events);
        let _ = last;
    }

    /// Returns whether to go round again.
    fn drain_call(&mut self, last: bool, events: &mut Vec<ToolEvent>) -> bool {
        match self.format.end() {
            Some(end) => match self.buf.find(end) {
                Some(at) => {
                    let body: String = self.buf.drain(..at).collect();
                    self.buf.drain(..end.len());
                    self.in_call = false;
                    self.finish_call(&body, events);
                    true
                }
                None if last => {
                    // Unterminated (the reply ended inside the call): parse
                    // what there is; if it is no call, it was text.
                    let body = std::mem::take(&mut self.buf);
                    self.in_call = false;
                    self.finish_call(&body, events);
                    false
                }
                None => false,
            },
            None => {
                if !last {
                    return false;
                }
                let body = std::mem::take(&mut self.buf);
                self.in_call = false;
                self.finish_call(&body, events);
                false
            }
        }
    }

    fn finish_call(&mut self, body: &str, events: &mut Vec<ToolEvent>) {
        match self.parse_body(body, false) {
            Some(calls) => self.push_calls(calls, events),
            // Not a call after all: the marker and body were the model's text.
            None => {
                let start = self.format.start().unwrap_or("");
                self.emit_text(format!("{start}{body}"), events);
            }
        }
    }

    fn push_calls(&mut self, calls: Vec<ParsedCall>, events: &mut Vec<ToolEvent>) {
        self.started = true;
        // Whitespace between calls, and before the first, is no content.
        self.pending_space.clear();
        for call in calls {
            self.calls += 1;
            events.push(ToolEvent::Call(call));
        }
    }

    fn emit_text(&mut self, text: String, events: &mut Vec<ToolEvent>) {
        if text.is_empty() {
            return;
        }
        if text.trim().is_empty() {
            self.pending_space.push_str(&text);
            return;
        }
        self.started = true;
        // The whitespace a text ends in is held until more text follows: it
        // belongs to the reply only if the reply goes on, not before a call
        // or at the end of a reply that made one.
        let body = text.trim_end();
        let trailing = text[body.len()..].to_owned();
        let mut out = std::mem::take(&mut self.pending_space);
        out.push_str(body);
        self.pending_space = trailing;
        events.push(ToolEvent::Text(out));
    }

    fn parse_body(&self, body: &str, bare: bool) -> Option<Vec<ParsedCall>> {
        let calls = self.parse_format(body, bare)?;
        (!bare || calls.iter().all(|c| self.tools.offers(&c.name))).then_some(calls)
    }

    fn parse_format(&self, body: &str, bare: bool) -> Option<Vec<ParsedCall>> {
        match self.format {
            ToolFormat::Hermes | ToolFormat::QwenXml => {
                let body = body.trim();
                if body.starts_with("<function=") {
                    parse_qwen_xml(body, &self.tools).map(|c| vec![c])
                } else {
                    parse_json_calls(body)
                }
            }
            ToolFormat::Llama3 => parse_llama3(body, bare),
            ToolFormat::Mistral => parse_mistral(body, bare),
            ToolFormat::Gemma4 => parse_gemma4_call(body).map(|c| vec![c]),
            ToolFormat::Harmony => parse_harmony(body),
        }
    }
}

/// The length of the longest suffix of `text` that is a proper prefix of
/// `marker`.
fn partial_suffix(text: &str, marker: &str) -> usize {
    let max = marker.len().saturating_sub(1).min(text.len());
    (1..=max)
        .rev()
        .find(|&n| {
            text.is_char_boundary(text.len() - n)
                && marker.is_char_boundary(n)
                && text.ends_with(&marker[..n])
        })
        .unwrap_or(0)
}

/// Compact JSON text of `value`, which must be an object: a string holding
/// one (a model that double-encodes) is unwrapped.
fn arguments_text(value: Value) -> Option<String> {
    match value {
        Value::Object(_) => serde_json::to_string(&value).ok(),
        Value::Null => Some("{}".into()),
        Value::String(s) => match serde_json::from_str::<Value>(&s).ok()? {
            object @ Value::Object(_) => serde_json::to_string(&object).ok(),
            _ => None,
        },
        _ => None,
    }
}

/// A call object: `{"name": ..., "arguments": {...}}`, also with
/// `parameters`, or nested under `function` (OpenAI's shape).
fn call_from_object(value: &Value) -> Option<ParsedCall> {
    let object = value.as_object()?;
    let inner = object
        .get("function")
        .and_then(Value::as_object)
        .unwrap_or(object);
    let name = inner.get("name")?.as_str()?.to_owned();
    if name.is_empty() {
        return None;
    }
    let arguments = inner
        .get("arguments")
        .or_else(|| inner.get("parameters"))
        .cloned()
        .unwrap_or(Value::Null);
    Some(ParsedCall {
        name,
        arguments: arguments_text(arguments)?,
    })
}

/// One call object, an array of them, or `{"tool_calls": [...]}`.
pub(crate) fn parse_json_calls(text: &str) -> Option<Vec<ParsedCall>> {
    let value: Value = serde_json::from_str(text.trim()).ok()?;
    let calls: Vec<ParsedCall> = match &value {
        Value::Array(items) => items.iter().map(call_from_object).collect::<Option<_>>()?,
        Value::Object(object) => match object.get("tool_calls").and_then(Value::as_array) {
            Some(items) => items.iter().map(call_from_object).collect::<Option<_>>()?,
            None => vec![call_from_object(&value)?],
        },
        _ => return None,
    };
    (!calls.is_empty()).then_some(calls)
}

/// Qwen 3.5 / Qwen3-Coder: `<function=NAME>` and `<parameter=P>` blocks.
fn parse_qwen_xml(body: &str, tools: &ToolSet) -> Option<ParsedCall> {
    let rest = body.strip_prefix("<function=")?;
    let (name, mut rest) = rest.split_once('>')?;
    let name = name.trim().to_owned();
    if name.is_empty() {
        return None;
    }
    let mut arguments = Map::new();
    loop {
        rest = rest.trim_start();
        let Some(after) = rest.strip_prefix("<parameter=") else {
            break;
        };
        let (param, after) = after.split_once('>')?;
        let (value, after) = match after.find("</parameter>") {
            Some(at) => (&after[..at], &after[at + "</parameter>".len()..]),
            // An unterminated parameter runs to the end of the function.
            None => {
                let end = after.find("</function>").unwrap_or(after.len());
                (&after[..end], &after[end..])
            }
        };
        // The value sits on its own lines; exactly the one newline each side
        // is the format's.
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        let param = param.trim().to_owned();
        let coerced = coerce_parameter(value, tools.parameter_type(&name, &param));
        arguments.insert(param, coerced);
        rest = after;
    }
    let tail = rest.trim();
    if !(tail.is_empty() || tail == "</function>") {
        return None;
    }
    Some(ParsedCall {
        name,
        arguments: serde_json::to_string(&Value::Object(arguments)).ok()?,
    })
}

/// A parameter's text as the JSON value its schema type asks for.
fn coerce_parameter(text: &str, kind: Option<&str>) -> Value {
    let json = || serde_json::from_str::<Value>(text.trim()).ok();
    match kind {
        Some("string") => Value::String(text.to_owned()),
        Some("integer" | "number" | "boolean" | "object" | "array" | "null") => {
            json().unwrap_or_else(|| Value::String(text.to_owned()))
        }
        _ => match json() {
            Some(value @ (Value::Object(_) | Value::Array(_) | Value::Bool(_) | Value::Null)) => {
                value
            }
            Some(number @ Value::Number(_)) => number,
            _ => Value::String(text.to_owned()),
        },
    }
}

/// Llama 3.1: JSON objects after `<|python_tag|>`, separated by `;` or
/// newlines, or the whole reply as one object.
fn parse_llama3(body: &str, bare: bool) -> Option<Vec<ParsedCall>> {
    let body = body.trim();
    if bare {
        return parse_json_calls(body);
    }
    let mut calls = Vec::new();
    for part in split_json_values(body) {
        calls.extend(parse_json_calls(part)?);
    }
    (!calls.is_empty()).then_some(calls)
}

/// The top-level JSON values of `text`, adjacent or separated by `;`/space.
fn split_json_values(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut rest = text.trim_start_matches(|c: char| c.is_whitespace() || c == ';');
    while !rest.is_empty() {
        let mut stream = serde_json::Deserializer::from_str(rest).into_iter::<Value>();
        match stream.next() {
            Some(Ok(_)) => {
                let end = stream.byte_offset();
                parts.push(&rest[..end]);
                rest = rest[end..].trim_start_matches(|c: char| c.is_whitespace() || c == ';');
            }
            _ => {
                parts.push(rest);
                break;
            }
        }
    }
    parts
}

/// Mistral: `[{"name", "arguments"}, ...]`, or `NAME[ARGS]{...}` (the newer
/// tokenizers), after `[TOOL_CALLS]`.
fn parse_mistral(body: &str, bare: bool) -> Option<Vec<ParsedCall>> {
    let body = body.trim();
    if bare {
        return parse_json_calls(body);
    }
    if body.starts_with('[') || body.starts_with('{') {
        return parse_json_calls(body);
    }
    // NAME[ARGS]{...}, possibly several, each opened by its own [TOOL_CALLS]
    // (which the stream cut at, so only the first is here).
    let mut calls = Vec::new();
    for segment in body.split("[TOOL_CALLS]") {
        let (name, args) = segment.split_once("[ARGS]")?;
        let name = name.trim();
        if name.is_empty() {
            return None;
        }
        let value: Value = serde_json::from_str(args.trim()).ok()?;
        calls.push(ParsedCall {
            name: name.to_owned(),
            arguments: arguments_text(value)?,
        });
    }
    (!calls.is_empty()).then_some(calls)
}

/// Harmony, after `<|channel|>commentary to=functions.`: `NAME <|constrain|>json<|message|>{...}`.
fn parse_harmony(body: &str) -> Option<Vec<ParsedCall>> {
    let (header, args) = body.split_once("<|message|>")?;
    let name = header
        .split(|c: char| c.is_whitespace() || c == '<')
        .next()?
        .trim();
    if name.is_empty() {
        return None;
    }
    let value: Value = if args.trim().is_empty() {
        Value::Null
    } else {
        serde_json::from_str(args.trim()).ok()?
    };
    Some(vec![ParsedCall {
        name: name.to_owned(),
        arguments: arguments_text(value)?,
    }])
}

// ────────────────────────────────────────────────────────────────────────────
// Gemma 4: call:NAME{key:<|"|>value<|"|>,...}
// ────────────────────────────────────────────────────────────────────────────

const GEMMA_QUOTE: &str = "<|\"|>";

/// The body between `<|tool_call>` and `<tool_call|>`: `call:NAME{...}`.
fn parse_gemma4_call(body: &str) -> Option<ParsedCall> {
    let body = body.trim();
    let after = body.strip_prefix("call:")?;
    let brace = after.find('{')?;
    let name = after[..brace].trim();
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
    {
        return None;
    }
    let args_start = brace + 1;
    let args_end = find_gemma4_matching_brace(&after[args_start..])?;
    let arguments = parse_gemma4_arguments(&after[args_start..args_start + args_end]);
    Some(ParsedCall {
        name: name.to_owned(),
        arguments: serde_json::to_string(&arguments).ok()?,
    })
}

fn find_gemma4_matching_brace(input: &str) -> Option<usize> {
    let mut idx = 0usize;
    let mut depth = 0i32;
    let mut in_string = false;
    while idx < input.len() {
        let rest = &input[idx..];
        if rest.starts_with(GEMMA_QUOTE) {
            in_string = !in_string;
            idx += GEMMA_QUOTE.len();
            continue;
        }
        let byte = input.as_bytes()[idx];
        if !in_string {
            match byte {
                b'{' => depth += 1,
                b'}' if depth == 0 => return Some(idx),
                b'}' => depth -= 1,
                _ => {}
            }
        }
        idx += 1;
    }
    None
}

fn parse_gemma4_arguments(input: &str) -> Value {
    let mut object = Map::new();
    for part in split_gemma4_top_level(input, b',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(colon) = find_gemma4_top_level(trimmed, b':') {
            let key = trimmed[..colon].trim();
            if key.is_empty() {
                continue;
            }
            object.insert(
                key.to_owned(),
                parse_gemma4_value(trimmed[colon + 1..].trim()),
            );
        }
    }
    Value::Object(object)
}

fn parse_gemma4_value(input: &str) -> Value {
    let trimmed = input.trim();
    if let Some(inner) = trimmed
        .strip_prefix(GEMMA_QUOTE)
        .and_then(|s| s.strip_suffix(GEMMA_QUOTE))
    {
        return Value::String(inner.to_owned());
    }
    if let Some(inner) = trimmed.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
        return parse_gemma4_arguments(inner);
    }
    if let Some(inner) = trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        let values = split_gemma4_top_level(inner, b',')
            .into_iter()
            .filter(|part| !part.trim().is_empty())
            .map(parse_gemma4_value)
            .collect();
        return Value::Array(values);
    }
    match trimmed {
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        "null" => Value::Null,
        _ => {
            if let Ok(value) = trimmed.parse::<i64>() {
                Value::Number(value.into())
            } else if let Ok(value) = trimmed.parse::<f64>() {
                serde_json::Number::from_f64(value)
                    .map(Value::Number)
                    .unwrap_or_else(|| Value::String(trimmed.to_owned()))
            } else {
                Value::String(trimmed.to_owned())
            }
        }
    }
}

fn split_gemma4_top_level(input: &str, delimiter: u8) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut idx = 0usize;
    while idx < input.len() {
        let rest = &input[idx..];
        if rest.starts_with(GEMMA_QUOTE) {
            in_string = !in_string;
            idx += GEMMA_QUOTE.len();
            continue;
        }
        let byte = input.as_bytes()[idx];
        if !in_string {
            match byte {
                b'{' | b'[' => depth += 1,
                b'}' | b']' => depth -= 1,
                b if b == delimiter && depth == 0 => {
                    parts.push(&input[start..idx]);
                    start = idx + 1;
                }
                _ => {}
            }
        }
        idx += 1;
    }
    parts.push(&input[start..]);
    parts
}

fn find_gemma4_top_level(input: &str, needle: u8) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut idx = 0usize;
    while idx < input.len() {
        let rest = &input[idx..];
        if rest.starts_with(GEMMA_QUOTE) {
            in_string = !in_string;
            idx += GEMMA_QUOTE.len();
            continue;
        }
        let byte = input.as_bytes()[idx];
        if !in_string {
            match byte {
                b'{' | b'[' => depth += 1,
                b'}' | b']' => depth -= 1,
                b if b == needle && depth == 0 => return Some(idx),
                _ => {}
            }
        }
        idx += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(format: ToolFormat, pieces: &[&str]) -> Vec<ToolEvent> {
        run_with(format, ToolSet::default(), pieces)
    }

    fn run_with(format: ToolFormat, tools: ToolSet, pieces: &[&str]) -> Vec<ToolEvent> {
        let mut stream = ToolStream::new(format, tools);
        let mut events = Vec::new();
        for piece in pieces {
            events.extend(stream.push(piece));
        }
        events.extend(stream.finish());
        // Adjacent text events are one run of text.
        let mut merged: Vec<ToolEvent> = Vec::new();
        for event in events {
            match (merged.last_mut(), event) {
                (Some(ToolEvent::Text(last)), ToolEvent::Text(next)) => last.push_str(&next),
                (_, event) => merged.push(event),
            }
        }
        merged
    }

    fn call(name: &str, arguments: &str) -> ToolEvent {
        ToolEvent::Call(ParsedCall {
            name: name.into(),
            arguments: arguments.into(),
        })
    }

    fn tool(name: &str, params: Value) -> ToolDefinition {
        serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object", "properties": params}}
        }))
        .unwrap()
    }

    #[test]
    fn hermes_json_call_between_text() {
        let events = run(
            ToolFormat::Hermes,
            &[
                "Let me check. <tool",
                "_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"SF\"}}\n</tool_",
                "call>",
            ],
        );
        assert_eq!(
            events,
            vec![
                ToolEvent::Text("Let me check.".into()),
                call("get_weather", r#"{"city":"SF"}"#),
            ]
        );
    }

    #[test]
    fn parallel_calls_and_the_whitespace_between_them_is_no_content() {
        let events = run(
            ToolFormat::Hermes,
            &[
                "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call>\n<tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>\n",
            ],
        );
        assert_eq!(events, vec![call("a", "{}"), call("b", r#"{"x":1}"#)]);
    }

    #[test]
    fn text_that_only_looks_like_a_marker_is_text() {
        let events = run(ToolFormat::Hermes, &["a < b and <tool and <tool_cal"]);
        assert_eq!(
            events,
            vec![ToolEvent::Text("a < b and <tool and <tool_cal".into())]
        );
    }

    #[test]
    fn a_call_body_that_is_not_a_call_is_returned_as_text() {
        let events = run(ToolFormat::Hermes, &["<tool_call>not json</tool_call>"]);
        assert_eq!(events, vec![ToolEvent::Text("<tool_call>not json".into())]);
    }

    #[test]
    fn an_unterminated_call_still_parses() {
        let events = run(
            ToolFormat::Hermes,
            &["<tool_call>{\"name\":\"a\",\"arguments\":{\"k\":\"v\"}}"],
        );
        assert_eq!(events, vec![call("a", r#"{"k":"v"}"#)]);
    }

    #[test]
    fn qwen_xml_parameters_take_the_type_the_schema_gives() {
        let tools = ToolSet::new(&[tool(
            "book",
            serde_json::json!({
                "city": {"type": "string"},
                "nights": {"type": "integer"},
                "flexible": {"type": "boolean"},
                "tags": {"type": "array"},
                "note": {"type": "string"}
            }),
        )]);
        let events = run_with(
            ToolFormat::Hermes,
            tools,
            &[
                "<tool_call>\n<function=book>\n<parameter=city>\nParis\n</parameter>\n",
                "<parameter=nights>\n3\n</parameter>\n<parameter=flexible>\ntrue\n</parameter>\n",
                "<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n<parameter=note>\n42\n</parameter>\n",
                "</function>\n</tool_call>",
            ],
        );
        assert_eq!(
            events,
            vec![call(
                "book",
                r#"{"city":"Paris","nights":3,"flexible":true,"tags":["a","b"],"note":"42"}"#
            )]
        );
    }

    #[test]
    fn qwen_xml_multiline_value_keeps_inner_newlines() {
        let events = run(
            ToolFormat::Hermes,
            &[
                "<tool_call>\n<function=write>\n<parameter=text>\nline one\nline two\n</parameter>\n</function>\n</tool_call>",
            ],
        );
        assert_eq!(
            events,
            vec![call("write", r#"{"text":"line one\nline two"}"#)]
        );
    }

    #[test]
    fn llama3_python_tag_and_bare_json() {
        let events = run_with(
            ToolFormat::Llama3,
            offered(),
            &[
                "<|python_tag|>{\"name\": \"f\", \"parameters\": {\"a\": 1}}; {\"name\": \"g\", \"parameters\": {}}",
            ],
        );
        assert_eq!(events, vec![call("f", r#"{"a":1}"#), call("g", "{}")]);

        let events = run_with(
            ToolFormat::Llama3,
            offered(),
            &["{\"name\": \"f\", ", "\"parameters\": {\"a\": 1}}"],
        );
        assert_eq!(events, vec![call("f", r#"{"a":1}"#)]);

        // JSON that is not a call is the reply.
        let events = run_with(ToolFormat::Llama3, offered(), &["{\"answer\": 42}"]);
        assert_eq!(events, vec![ToolEvent::Text("{\"answer\": 42}".into())]);
        // So is a call to a tool that was not offered.
        let events = run_with(
            ToolFormat::Llama3,
            offered(),
            &["{\"name\": \"zzz\", \"parameters\": {}}"],
        );
        assert_eq!(
            events,
            vec![ToolEvent::Text(
                "{\"name\": \"zzz\", \"parameters\": {}}".into()
            )]
        );
    }

    fn offered() -> ToolSet {
        ToolSet::new(&[
            tool("f", serde_json::json!({})),
            tool("g", serde_json::json!({})),
        ])
    }

    #[test]
    fn mistral_array_and_args_forms() {
        let events = run(
            ToolFormat::Mistral,
            &["[TOOL_CALLS][{\"name\": \"f\", \"arguments\": {\"a\": 1}, \"id\": \"abc123def\"}]"],
        );
        assert_eq!(events, vec![call("f", r#"{"a":1}"#)]);
        let events = run(ToolFormat::Mistral, &["[TOOL_CALLS]f[ARGS]{\"a\": 1}"]);
        assert_eq!(events, vec![call("f", r#"{"a":1}"#)]);
    }

    #[test]
    fn gemma4_call() {
        let events = run(
            ToolFormat::Gemma4,
            &[
                "<|tool_call>call:web.search{query:<|\"|>qwen mtp<|\"|>,limit:5,meta:{ok:true},scores:[1,2.5]}<tool_call|>",
            ],
        );
        assert_eq!(
            events,
            vec![call(
                "web.search",
                r#"{"query":"qwen mtp","limit":5,"meta":{"ok":true},"scores":[1,2.5]}"#
            )]
        );
    }

    #[test]
    fn harmony_function_call() {
        let events = run(
            ToolFormat::Harmony,
            &[
                "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"SF\"}<|call|>",
            ],
        );
        assert_eq!(events, vec![call("get_weather", r#"{"city":"SF"}"#)]);
    }

    #[test]
    fn splitting_the_stream_anywhere_changes_nothing() {
        let text = "I will look. <tool_call>\n{\"name\": \"f\", \"arguments\": {\"q\": \"x < y\"}}\n</tool_call> done";
        let whole = run(ToolFormat::Hermes, &[text]);
        assert_eq!(
            whole,
            vec![
                ToolEvent::Text("I will look.".into()),
                call("f", r#"{"q":"x < y"}"#),
                ToolEvent::Text(" done".into()),
            ]
        );
        for size in 1..text.len() {
            let pieces: Vec<&str> = text
                .as_bytes()
                .chunks(size)
                .map(|c| std::str::from_utf8(c).unwrap())
                .collect();
            assert_eq!(run(ToolFormat::Hermes, &pieces), whole, "chunks of {size}");
        }
    }

    #[test]
    fn detect_reads_the_template() {
        let mut template = ChatTemplate::new(ChatTemplateType::ChatMl);
        template.jinja_source = Some("... <tool_call>\n{ ...".into());
        assert_eq!(ToolFormat::detect(&template), Some(ToolFormat::Hermes));
        template.jinja_source = Some("... <tool_call>\n<function= ...".into());
        assert_eq!(ToolFormat::detect(&template), Some(ToolFormat::QwenXml));
        template.jinja_source = Some("[TOOL_CALLS] ...".into());
        assert_eq!(ToolFormat::detect(&template), Some(ToolFormat::Mistral));
        template.jinja_source = Some("... <|tool_call> ...".into());
        assert_eq!(ToolFormat::detect(&template), Some(ToolFormat::Gemma4));
        template.jinja_source = Some("<|channel|> to=functions.x".into());
        assert_eq!(ToolFormat::detect(&template), Some(ToolFormat::Harmony));
        assert_eq!(
            ToolFormat::detect(&ChatTemplate::new(ChatTemplateType::Llama3)),
            Some(ToolFormat::Llama3)
        );
        assert_eq!(
            ToolFormat::detect(&ChatTemplate::new(ChatTemplateType::Alpaca)),
            None
        );
    }
}
