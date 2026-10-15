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
    let special = &entry["special_tokens"];
    ChatTemplate::from_jinja(
        entry["source"].as_str().unwrap(),
        special["bos_token"].as_str().map(str::to_string),
        special["eos_token"].as_str().map(str::to_string),
        entry["model_id"].as_str().unwrap_or(name),
    )
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

/// A conversation that ends in an assistant turn is trained on exactly the
/// text transformers renders for it. (A Harmony conversation ends in
/// `<|return|>` rather than the `<|end|>` its template writes, and raw
/// thinking in content is moved on purpose.)
#[test]
fn training_text_is_what_the_template_renders() {
    for suite in suites() {
        set_frozen_date(&suite);
        let mut checked = 0;
        for case in suite["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let messages = messages_from(&case["messages"]);
            if case["add_generation_prompt"] == true
                || messages.last().is_none_or(|m| m.role != "assistant")
                || name.ends_with("/inline_think_train")
            {
                continue;
            }
            let template = template_for(&suite, case["template"].as_str().unwrap());
            let kwargs =
                ChatTemplateKwargs::from_map(case["kwargs"].as_object().unwrap().clone()).unwrap();
            let formatted = template.try_apply(&messages, tools_from(case).as_deref(), &kwargs);
            match (case["expected"].as_str(), formatted) {
                (Some(expected), Ok(formatted)) => {
                    checked += 1;
                    let source = suite["templates"][case["template"].as_str().unwrap()]["source"]
                        .as_str()
                        .unwrap();
                    let mut expected = expected.to_string();
                    // Harmony ends a conversation in `<|return|>`.
                    if source.contains("<|channel|>") {
                        if let Some(body) = expected.strip_suffix("<|end|>") {
                            expected = format!("{body}<|return|>");
                        }
                    }
                    // A template with a thinking switch is trained in the
                    // mode the turn is in: a turn with reasoning in thinking
                    // mode, which Qwen3.8 and Gemma 4 mark in their system
                    // turns, a turn without in non-thinking mode.
                    let head = &messages[..messages.len() - 1];
                    let mut candidates = vec![expected.clone()];
                    if source.contains("enable_thinking") {
                        for thinking in [false, true] {
                            let mut k = kwargs.clone();
                            k.set_default("enable_thinking", thinking);
                            candidates.push(
                                template
                                    .render(&messages, tools_from(case).as_deref(), &k, false)
                                    .unwrap(),
                            );
                        }
                    }
                    // Gemma 4 prompts with an empty thought channel that its
                    // history turns leave out: the text is the prompt, then
                    // the turn from where the two part.
                    if source.contains("<|channel>") {
                        for candidate in candidates.clone() {
                            for thinking in [false, true] {
                                let mut k = kwargs.clone();
                                k.set("enable_thinking", thinking);
                                let prompt = template
                                    .render(head, tools_from(case).as_deref(), &k, true)
                                    .unwrap();
                                let shared = prompt
                                    .bytes()
                                    .zip(candidate.bytes())
                                    .take_while(|(a, b)| a == b)
                                    .count();
                                candidates.push(format!("{prompt}{}", &candidate[shared..]));
                            }
                        }
                    }
                    assert!(
                        candidates.contains(&formatted.text),
                        "{name}: {}",
                        first_difference(&expected, &formatted.text)
                    );
                }
                (None, Err(_)) => {}
                (expected, got) => panic!("{name}: transformers {expected:?}, pmetal {got:?}"),
            }
        }
        assert!(checked >= 60, "only {checked} training cases");
    }
}

/// The response spans hold the assistant's words and none of the prompt's:
/// the final turn's content is inside the last span, the system and user
/// content is in no span, and the prompt before the response is what
/// serving prompts with (the generation prompt of the conversation so far).
#[test]
fn response_spans_cover_the_assistant_and_nothing_else() {
    for suite in suites() {
        set_frozen_date(&suite);
        let cases = suite["cases"].as_array().unwrap();
        let mut checked = 0;
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let messages = messages_from(&case["messages"]);
            if case["add_generation_prompt"] == true
                || messages.last().is_none_or(|m| m.role != "assistant")
                || name.ends_with("/inline_think_train")
                || case["expected"].is_null()
            {
                continue;
            }
            let template = template_for(&suite, case["template"].as_str().unwrap());
            let kwargs =
                ChatTemplateKwargs::from_map(case["kwargs"].as_object().unwrap().clone()).unwrap();
            let tools = tools_from(case);
            let formatted = template
                .try_apply(&messages, tools.as_deref(), &kwargs)
                .unwrap();
            let in_span = |needle: &str| -> bool {
                formatted.text.match_indices(needle).any(|(at, _)| {
                    formatted
                        .response_spans
                        .iter()
                        .any(|s| s.start <= at && at + needle.len() <= s.end)
                })
            };
            let spans: Vec<&str> = formatted
                .response_spans
                .iter()
                .map(|s| &formatted.text[s.clone()])
                .collect();
            let last = messages.last().unwrap();
            assert!(!last.content.is_empty());
            assert!(in_span(&last.content), "{name}: {spans:?}");
            for m in &messages {
                if m.role == "user" || m.role == "system" {
                    assert!(
                        !formatted
                            .response_spans
                            .iter()
                            .any(|s| formatted.text[s.clone()].contains(m.content.as_str())),
                        "{name}: the {} turn {:?} is in a response span: {spans:?}",
                        m.role,
                        m.content
                    );
                }
            }
            // The prompt before the last response is the generation prompt of
            // the conversation so far, with thinking off or on. (Mistral v0.3
            // writes a system message only into the last user turn of a
            // conversation, so a training conversation that ends on an
            // assistant turn has none, and no prompt leads to it.)
            let drops_system =
                case["template"] == "mistral_v03" && messages.iter().any(|m| m.role == "system");
            let prompt = formatted.prompt();
            let (head, _) = messages.split_at(messages.len() - 1);
            let serves = [false, true].into_iter().any(|thinking| {
                let mut k = kwargs.clone();
                k.set_default("enable_thinking", thinking);
                template
                    .render(head, tools.as_deref(), &k, true)
                    .is_ok_and(|p| p == prompt)
            });
            assert!(serves || drops_system, "{name}: prompt {prompt:?}");
            checked += 1;
        }
        assert!(checked >= 60, "only {checked} cases");
    }
}

/// The built-in formats, for checkpoints with no template, agree with the
/// published template of the family wherever the family's template adds
/// nothing of its own to a plain conversation.
#[test]
fn builtin_formats_match_the_families_published_templates() {
    let suite = load(FIXTURE);
    set_frozen_date(&suite);
    // (built-in type, published template, conversations to compare)
    let plain = [
        "user_gen",
        "system_user_gen",
        "multiturn_gen",
        "multiturn_nosystem",
    ];
    let comparisons: [(ChatTemplateType, &str, &[&str]); 3] = [
        (ChatTemplateType::Gemma4, "gemma4", &plain),
        (ChatTemplateType::DeepSeek, "deepseek_v3", &plain),
        (
            ChatTemplateType::Phi3,
            "phi3",
            // The published Phi-3 template ends a finished conversation with
            // the EOS token; a generation prompt it does not.
            &["user_gen", "system_user_gen", "multiturn_gen"],
        ),
    ];
    for (template_type, name, case_names) in comparisons {
        let builtin = ChatTemplate::new(template_type);
        for case_name in case_names {
            let case = suite["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["name"] == format!("{name}/{case_name}"))
                .unwrap();
            let rendered = builtin.render(
                &messages_from(&case["messages"]),
                None,
                &ChatTemplateKwargs::new(),
                case["add_generation_prompt"].as_bool().unwrap(),
            );
            assert_eq!(
                rendered.as_deref(),
                Ok(case["expected"].as_str().unwrap()),
                "{template_type:?} vs {name}/{case_name}"
            );
        }
    }
}
