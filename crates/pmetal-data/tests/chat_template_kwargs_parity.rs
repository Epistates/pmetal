//! Chat-template kwargs parity against transformers.
//!
//! `fixtures/chat_template_kwargs_reference.json` holds real upstream chat
//! templates (Qwen3.8, gpt-oss) and what
//! `AutoTokenizer.apply_chat_template(messages, tools=tools,
//! add_generation_prompt=True, tokenize=False, **kwargs)` rendered for a
//! multi-turn history with thinking, under every documented kwarg:
//! `reasoning_effort` at each level, `preserve_thinking` on and off,
//! `enable_thinking`, gpt-oss's `model_identity`, and tool schemas (which go
//! through `tojson`). Each case must render byte for byte, and a case the
//! template refused (an unsupported `reasoning_effort`) must be refused here
//! too. Regenerate with `.strategy/parity/dump_chat_template_kwargs_reference.py`.
#![allow(unsafe_code)]

use pmetal_data::chat_templates::{
    ChatTemplate, ChatTemplateKwargs, ChatTemplateType, FunctionCall, Message, ToolCall,
    ToolDefinition,
};
use serde_json::Value;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/chat_template_kwargs_reference.json"
);

fn load_fixture() -> Value {
    serde_json::from_str(&std::fs::read_to_string(FIXTURE).expect("read fixture"))
        .expect("parse fixture")
}

fn template_for(fixture: &Value, name: &str) -> ChatTemplate {
    let entry = &fixture["templates"][name];
    let template_type = match name {
        "qwen3_8" => ChatTemplateType::Qwen,
        "gpt_oss" => ChatTemplateType::GptOss,
        other => panic!("unknown template {other}"),
    };
    let mut template = ChatTemplate::new(template_type);
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

fn render_case(fixture: &Value, case: &Value) -> Result<String, String> {
    let template = template_for(fixture, case["template"].as_str().unwrap());
    let messages = messages_from(&case["messages"]);
    let tools: Option<Vec<ToolDefinition>> = case["tools"]
        .as_array()
        .map(|t| serde_json::from_value(Value::Array(t.clone())).expect("tool definitions"));
    let kwargs = ChatTemplateKwargs::from_map(case["kwargs"].as_object().unwrap().clone())?;
    template.check_reasoning_effort_level(&kwargs)?;
    template
        .apply_inference_with_kwargs(&messages, tools.as_deref(), &kwargs)
        .map(|f| f.text)
}

fn set_frozen_date(fixture: &Value) {
    // SAFETY: set before any rendering, and only this test binary reads it.
    unsafe {
        std::env::set_var(
            "PMETAL_CHAT_TEMPLATE_FROZEN_DATE",
            fixture["frozen_date"].as_str().unwrap(),
        );
    }
}

#[test]
fn template_kwargs_render_like_transformers() {
    let fixture = load_fixture();
    set_frozen_date(&fixture);
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() >= 28, "fixture has {} cases", cases.len());

    let mut failures = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let rendered = render_case(&fixture, case);
        match (case["expected"].as_str(), rendered) {
            (Some(expected), Ok(text)) if text == expected => {}
            (Some(expected), Ok(text)) => {
                let at = expected
                    .bytes()
                    .zip(text.bytes())
                    .position(|(a, b)| a != b)
                    .unwrap_or(expected.len().min(text.len()));
                let lo = at.saturating_sub(60);
                failures.push(format!(
                    "{name}: diverges at byte {at}\n  expected …{:?}\n  pmetal   …{:?}",
                    &expected[lo..(at + 60).min(expected.len())],
                    &text[lo..(at + 60).min(text.len())],
                ));
            }
            (Some(_), Err(e)) => failures.push(format!("{name}: render failed: {e}")),
            (None, Ok(_)) => failures.push(format!(
                "{name}: transformers refused ({}), pmetal rendered",
                case["error"]
            )),
            (None, Err(_)) => {}
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The rendering must depend on every kwarg the fixture varies: a renderer
/// that dropped `reasoning_effort` or `preserve_thinking` would otherwise
/// pass on the cases where the template's default happens to apply.
#[test]
fn fixture_cases_are_distinct_per_kwarg() {
    let fixture = load_fixture();
    let expected = |name: &str| -> String {
        fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap_or_else(|| panic!("no case {name}"))["expected"]
            .as_str()
            .unwrap()
            .to_string()
    };
    let low_keep = expected("qwen3_8/history/effort=low/preserve=True");
    let low_drop = expected("qwen3_8/history/effort=low/preserve=False");
    let xhigh_keep = expected("qwen3_8/history/effort=xhigh/preserve=True");
    let medium_keep = expected("qwen3_8/history/effort=medium/preserve=True");
    assert_ne!(low_keep, low_drop);
    assert_ne!(low_keep, xhigh_keep);
    assert_ne!(low_keep, medium_keep);
    assert_ne!(xhigh_keep, medium_keep);
    // Earlier turns' thinking is in the prompt only when preserved.
    assert!(low_keep.contains("17 * 20 + 17 * 3"));
    assert!(!low_drop.contains("17 * 20 + 17 * 3"));
    assert_ne!(
        expected("gpt_oss/effort=low"),
        expected("gpt_oss/effort=high")
    );
}

/// Raw generated text left in `content` (`<think>…</think>answer`) renders
/// as if the thinking had been passed as `reasoning_content`.
#[test]
fn raw_thinking_in_content_renders_like_reasoning_content() {
    let fixture = load_fixture();
    let template = template_for(&fixture, "qwen3_8");
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "qwen3_8/history/effort=low/preserve=True")
        .unwrap();
    let messages: Vec<Message> = messages_from(&case["messages"])
        .into_iter()
        .map(|mut m| {
            if let Some(reasoning) = m.reasoning_content.take() {
                m.content = format!("<think>\n{reasoning}\n</think>\n\n{}", m.content);
            }
            m
        })
        .collect();
    let kwargs = ChatTemplateKwargs::from_map(case["kwargs"].as_object().unwrap().clone()).unwrap();
    let text = template
        .apply_inference_with_kwargs(&messages, None, &kwargs)
        .unwrap()
        .text;
    assert_eq!(text, case["expected"].as_str().unwrap());
}

#[test]
fn unsupported_controls_are_reported() {
    let fixture = load_fixture();
    let qwen = template_for(&fixture, "qwen3_8");
    let gpt_oss = template_for(&fixture, "gpt_oss");
    let effort = |level: &str| ChatTemplateKwargs::new().with("reasoning_effort", level);

    // Qwen3.8 validates the level itself; the render refuses it.
    let err = qwen
        .apply_inference_with_kwargs(&[Message::user("hi")], None, &effort("high"))
        .unwrap_err();
    assert!(err.contains("Unexpected reasoning effort high"), "{err}");
    // gpt-oss writes any string, so the documented levels are checked.
    assert!(gpt_oss.check_thinking_controls(&effort("high")).is_ok());
    assert!(gpt_oss.check_thinking_controls(&effort("xhigh")).is_err());
    // gpt-oss has no preserve_thinking control.
    let preserve = ChatTemplateKwargs::new().with("preserve_thinking", false);
    assert!(gpt_oss.check_thinking_controls(&preserve).is_err());
    assert!(qwen.check_thinking_controls(&preserve).is_ok());
    // A reserved name is refused rather than silently shadowed.
    assert!(ChatTemplateKwargs::from_json(r#"{"messages": []}"#).is_err());
}
