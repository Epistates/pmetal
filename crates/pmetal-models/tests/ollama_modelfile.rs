//! The Ollama Modelfile writer and reader against Ollama's own parser, and the
//! shipped `TEMPLATE`s against the models' chat templates.
//!
//! `ollama_modelfile_reference.json` holds, for each case the builder writes
//! below, the Modelfile text and the instructions Ollama's `parser.ParseFile`
//! (run unchanged, from Ollama's source) reads out of it, plus a corpus of
//! hand-written Modelfiles with what that parser reads from each or the error
//! it gives. `ollama_templates_reference.json` holds each shipped template
//! rendered by Go's `text/template` for a conversation, next to what the
//! model's `transformers` chat template renders for the same one.

use pmetal_models::ollama::{ModelfileBuilder, Role, parse_modelfile, templates};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("fixtures/ollama_modelfile_reference.json"))
        .expect("fixture parses")
}

/// The builder cases, by name. The reference holds the text each wrote.
fn built(name: &str) -> ModelfileBuilder {
    match name {
        "minimal" => ModelfileBuilder::new().from("./model.gguf"),
        "parameters" => ModelfileBuilder::new()
            .from("llama3.2")
            .temperature(0.7)
            .num_ctx(4096)
            .top_p(0.9)
            .top_k(40),
        "stops_in_order" => ModelfileBuilder::new()
            .from("base")
            .stop("<|im_end|>")
            .temperature(0.2)
            .stop("<|im_start|>")
            .stop("a \"quoted\" stop")
            .stop("two\nlines")
            .stop(" padded "),
        "system_one_line" => ModelfileBuilder::new()
            .from("base")
            .system("You are a helpful assistant."),
        "system_multi_line" => ModelfileBuilder::new()
            .from("base")
            .system("You are a helpful assistant.\nBe brief.\n\nNever guess."),
        "system_with_quotes" => ModelfileBuilder::new()
            .from("base")
            .system("\"Quoted\" opening, \"quoted\" close.\""),
        "system_with_edges" => ModelfileBuilder::new()
            .from("base")
            .system("  leading and trailing  "),
        "template_llama3" => ModelfileBuilder::new()
            .from("base")
            .template(templates::LLAMA3_CHAT),
        "template_deepseek" => ModelfileBuilder::new()
            .from("base")
            .template(templates::DEEPSEEK_CHAT),
        "messages" => ModelfileBuilder::new()
            .from("base")
            .message(Role::System, "Answer in French.")
            .message(Role::User, "Hi")
            .message(Role::Assistant, "Bonjour, \"mon ami\"."),
        "adapter_and_license" => ModelfileBuilder::new()
            .from("./base.gguf")
            .adapter("./lora.gguf")
            .license("MIT License\n\nCopyright (c) 2026"),
        "path_with_spaces" => ModelfileBuilder::new()
            .from("/models/my model/base.gguf")
            .adapter("/adapters/my adapter.gguf"),
        "everything" => ModelfileBuilder::new()
            .from("base")
            .adapter("./lora.gguf")
            .temperature(0.5)
            .stop("</s>")
            .num_ctx(8192)
            .system("Be useful.")
            .template(templates::QWEN3_CHAT)
            .message(Role::User, "Hello")
            .license("Apache-2.0"),
        other => panic!("no builder case {other}"),
    }
}

#[test]
fn the_builder_writes_the_reference_text() {
    let reference = reference();
    for case in reference["built"].as_array().expect("built cases") {
        let name = case["name"].as_str().unwrap();
        assert_eq!(
            built(name).build().unwrap(),
            case["modelfile"].as_str().unwrap(),
            "{name}"
        );
    }
}

/// What the builder writes, Ollama reads back as what was meant, and the
/// reader here agrees with Ollama's parser on it.
#[test]
fn ollama_reads_what_the_builder_wrote() {
    let reference = reference();
    for case in reference["built"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let want = &case["commands"];
        let got = parse_modelfile(case["modelfile"].as_str().unwrap())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let got: Vec<(String, String)> = got.into_iter().map(|c| (c.name, c.args)).collect();
        let want: Vec<(String, String)> = want
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                (
                    c["name"].as_str().unwrap().to_string(),
                    c["args"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        assert_eq!(got, want, "{name}");
    }
}

/// The reader agrees with Ollama's parser on hand-written Modelfiles, errors
/// included.
#[test]
fn the_reader_agrees_with_ollamas_parser() {
    let reference = reference();
    for case in reference["corpus"].as_array().unwrap() {
        let text = case["modelfile"].as_str().unwrap();
        let ours = parse_modelfile(text);
        match case.get("error") {
            Some(_) => assert!(
                ours.is_err(),
                "{text:?} should be refused, read as {ours:?}"
            ),
            None => {
                let got: Vec<(String, String)> = ours
                    .unwrap_or_else(|e| panic!("{text:?}: {e}"))
                    .into_iter()
                    .map(|c| (c.name, c.args))
                    .collect();
                let want: Vec<(String, String)> = case["commands"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| {
                        (
                            c["name"].as_str().unwrap().to_string(),
                            c["args"].as_str().unwrap().to_string(),
                        )
                    })
                    .collect();
                assert_eq!(got, want, "{text:?}");
            }
        }
    }
}

#[test]
fn the_shipped_templates_are_the_ones_checked_against_the_chat_templates() {
    let reference: Value =
        serde_json::from_str(include_str!("fixtures/ollama_templates_reference.json"))
            .expect("fixture parses");
    let shipped = [
        ("LLAMA3_CHAT", templates::LLAMA3_CHAT),
        ("QWEN3_CHAT", templates::QWEN3_CHAT),
        ("GEMMA_INSTRUCT", templates::GEMMA_INSTRUCT),
        ("MISTRAL_INSTRUCT", templates::MISTRAL_INSTRUCT),
        ("PHI3_INSTRUCT", templates::PHI3_INSTRUCT),
        ("DEEPSEEK_CHAT", templates::DEEPSEEK_CHAT),
    ];
    let checked = reference["templates"].as_object().unwrap();
    assert_eq!(
        checked.len(),
        shipped.len(),
        "a template was added unchecked"
    );
    for (name, template) in shipped {
        let entry = &checked[name];
        assert_eq!(
            template,
            entry["template"].as_str().unwrap(),
            "{name} changed"
        );
        for case in entry["cases"].as_array().unwrap() {
            assert_eq!(
                case["go"], case["transformers"],
                "{name}: Go renders {} where the chat template renders {}",
                case["go"], case["transformers"]
            );
        }
        assert!(
            entry["cases"].as_array().unwrap().len() >= 2,
            "{name}: with and without a system message"
        );
    }
}
