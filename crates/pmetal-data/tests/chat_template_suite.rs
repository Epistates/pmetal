//! Chat templates against transformers, over a conversation suite.
//!
//! `fixtures/chat_template_suite_reference.json` holds the upstream chat
//! templates of ten model families under permissive licences, and what
//! `AutoTokenizer.apply_chat_template(messages, tools=tools,
//! add_generation_prompt=..., tokenize=False, **kwargs)` rendered for
//! seventeen conversations: single and multi-turn, with and without a
//! generation prompt, with tool definitions, a tool call and its result, with
//! reasoning content, and under the thinking switch. Every case must render
//! byte for byte, and a case the template refused must be refused here too.
//!
//! The same cases drive the loss-mask checks: wherever the template renders a
//! conversation ending in an assistant turn, the trainable span must be that
//! turn as the template writes it, and nothing before it.
//!
//! `PMETAL_CHAT_SUITE_FILE=<suite.json>` runs a suite of the same shape
//! beyond the committed one (families whose licence keeps their template out
//! of the repository).
#![allow(unsafe_code)]

use pmetal_data::chat_templates::{
    ChatTemplate, ChatTemplateKwargs, ChatTemplateType, FunctionCall, Message, ToolCall,
    ToolDefinition,
};
use serde_json::Value;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/chat_template_suite_reference.json"
);

fn load(path: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("read suite")).expect("parse suite")
}

/// The committed suite, then the external one when asked for.
fn suites() -> Vec<Value> {
    let mut suites = vec![load(FIXTURE)];
    if let Some(path) = std::env::var_os("PMETAL_CHAT_SUITE_FILE") {
        suites.push(load(path.to_str().unwrap()));
    }
    suites
}

fn template_for(suite: &Value, name: &str) -> ChatTemplate {
    let entry = &suite["templates"][name];
    let mut template = ChatTemplate::new(ChatTemplateType::Custom);
    template.jinja_source = Some(entry["source"].as_str().unwrap().to_string());
    let special = &entry["special_tokens"];
    template.bos_token = special["bos_token"].as_str().map(str::to_string);
    template.eos_token = special["eos_token"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    template
}

fn messages_from(value: &Value) -> Vec<Message> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|m| {
            let mut msg = Message::new(
                m["role"].as_str().unwrap(),
                m["content"].as_str().unwrap_or_default(),
            );
            msg.reasoning_content = m["reasoning_content"].as_str().map(str::to_string);
            msg.tool_call_id = m["tool_call_id"].as_str().map(str::to_string);
            msg.tool_calls = m["tool_calls"].as_array().map(|calls| {
                calls
                    .iter()
                    .map(|c| ToolCall {
                        id: c["id"].as_str().map(str::to_string),
                        tool_type: "function".into(),
                        function: FunctionCall {
                            name: c["function"]["name"].as_str().unwrap().to_string(),
                            arguments: c["function"]["arguments"].clone(),
                        },
                    })
                    .collect()
            });
            msg
        })
        .collect()
}

fn tools_from(case: &Value) -> Option<Vec<ToolDefinition>> {
    case["tools"]
        .as_array()
        .map(|t| serde_json::from_value(Value::Array(t.clone())).expect("tool definitions"))
}

fn render_case(suite: &Value, case: &Value) -> Result<String, String> {
    let template = template_for(suite, case["template"].as_str().unwrap());
    let messages = messages_from(&case["messages"]);
    let tools = tools_from(case);
    let kwargs = ChatTemplateKwargs::from_map(case["kwargs"].as_object().unwrap().clone())?;
    template.render(
        &messages,
        tools.as_deref(),
        &kwargs,
        case["add_generation_prompt"].as_bool().unwrap(),
    )
}

fn set_frozen_date(suite: &Value) {
    // SAFETY: set before any rendering, and only this test binary reads it.
    unsafe {
        std::env::set_var(
            "PMETAL_CHAT_TEMPLATE_FROZEN_DATE",
            suite["frozen_date"].as_str().unwrap(),
        );
    }
}

fn first_difference(expected: &str, got: &str) -> String {
    let at = expected
        .bytes()
        .zip(got.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or(expected.len().min(got.len()));
    let mut lo = at.saturating_sub(60);
    while !expected.is_char_boundary(lo) || !got.is_char_boundary(lo) {
        lo -= 1;
    }
    let cut = |s: &str| {
        let mut hi = (at + 60).min(s.len());
        while !s.is_char_boundary(hi) {
            hi += 1;
        }
        s[lo..hi].to_string()
    };
    format!(
        "diverges at byte {at}\n  expected …{:?}\n  pmetal   …{:?}",
        cut(expected),
        cut(got)
    )
}

#[test]
fn templates_render_like_transformers() {
    for suite in suites() {
        set_frozen_date(&suite);
        let cases = suite["cases"].as_array().unwrap();
        assert!(cases.len() >= 100, "suite has {} cases", cases.len());
        let mut failures = Vec::new();
        for case in cases {
            let name = case["name"].as_str().unwrap();
            // Raw thinking in content is moved to `reasoning_content` on
            // purpose; it has its own test below.
            if name.ends_with("/inline_think_train") {
                continue;
            }
            match (case["expected"].as_str(), render_case(&suite, case)) {
                (Some(expected), Ok(text)) if text == expected => {}
                (Some(expected), Ok(text)) => {
                    failures.push(format!("{name}: {}", first_difference(expected, &text)));
                }
                (Some(_), Err(e)) => failures.push(format!("{name}: render failed: {e}")),
                (None, Ok(_)) => failures.push(format!(
                    "{name}: transformers refused ({}), pmetal rendered",
                    case["error"]
                )),
                (None, Err(_)) => {}
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} cases\n{}",
            failures.len(),
            cases.len(),
            failures.join("\n")
        );
    }
}

/// Raw generated text left in an assistant message (`<think>…</think>answer`)
/// renders as the same conversation with the thinking passed as
/// `reasoning_content`, wherever the template writes a `<think>` block from
/// that field; a template that does not treat `<think>` specially renders
/// the text as it is, as transformers does.
#[test]
fn raw_thinking_in_content_is_the_same_conversation_as_reasoning_content() {
    for suite in suites() {
        set_frozen_date(&suite);
        let cases = suite["cases"].as_array().unwrap();
        let expected = |template: &str, case: &str| -> Option<String> {
            cases
                .iter()
                .find(|c| c["name"] == format!("{template}/{case}"))
                .and_then(|c| c["expected"].as_str().map(str::to_string))
        };
        for (name, entry) in suite["templates"].as_object().unwrap() {
            let inline = cases
                .iter()
                .find(|c| c["name"] == format!("{name}/inline_think_train"))
                .unwrap();
            let rendered = render_case(&suite, inline);
            let source = entry["source"].as_str().unwrap();
            let want = if source.contains("</think>") && source.contains("reasoning_content") {
                expected(name, "reasoning_train")
            } else {
                expected(name, "inline_think_train")
            };
            assert_eq!(rendered.ok(), want, "{name}");
        }
    }
}
