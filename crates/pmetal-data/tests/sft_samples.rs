//! What a supervised fine-tuning sample trains on: which tokens carry a label,
//! for every dataset format and container the loader reads.
//!
//! The tokenizer is character level, so a label mask is readable as the text
//! it covers: `trained_text` decodes exactly the tokens the loss scores.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use pmetal_data::chat_templates::ChatTemplate;
use pmetal_data::{DatasetFormat, Sample, Tokenizer, TrainingDataset};
use serde_json::{Value, json};

/// Ids of the special tokens the test tokenizers know.
const SPECIALS: &[(&str, u32)] = &[
    ("<s>", 100),
    ("</s>", 101),
    ("<|im_start|>", 102),
    ("<|im_end|>", 103),
    ("<|begin_of_text|>", 104),
    ("<|start_header_id|>", 105),
    ("<|end_header_id|>", 106),
    ("<|eot_id|>", 107),
];

/// A tokenizer with one token per ASCII character and newline, plus the
/// special tokens above, which the pre-tokenizer leaves whole. `post` is a
/// post-processor template (`"<s> $A </s>"`) or `None`.
fn char_tokenizer(post: Option<&str>) -> Tokenizer {
    let mut vocab = serde_json::Map::new();
    vocab.insert("[UNK]".into(), 0.into());
    for (i, c) in (32u8..127).enumerate() {
        vocab.insert((c as char).to_string(), (1 + i as u32).into());
    }
    vocab.insert("\n".into(), 96.into());
    let added: Vec<Value> = SPECIALS
        .iter()
        .map(|(content, id)| {
            vocab.insert((*content).into(), (*id).into());
            json!({"id": id, "content": content, "single_word": false, "lstrip": false,
                   "rstrip": false, "normalized": false, "special": true})
        })
        .collect();
    let post_processor = post.map_or(Value::Null, |template| {
        let mut single = Vec::new();
        for piece in template.split_whitespace() {
            single.push(if piece == "$A" {
                json!({"Sequence": {"id": "A", "type_id": 0}})
            } else {
                json!({"SpecialToken": {"id": piece, "type_id": 0}})
            });
        }
        let special_tokens: serde_json::Map<String, Value> = SPECIALS
            .iter()
            .map(|(content, id)| {
                (
                    (*content).to_string(),
                    json!({"id": content, "ids": [id], "tokens": [content]}),
                )
            })
            .collect();
        json!({"type": "TemplateProcessing", "single": single, "pair": [],
               "special_tokens": special_tokens})
    });
    let spec = json!({
        "version": "1.0",
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "(?m:.)"},
                          "behavior": "Isolated", "invert": false},
        "post_processor": post_processor,
        "decoder": {"type": "Fuse"},
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    });
    Tokenizer::from_bytes(spec.to_string().as_bytes()).expect("a character tokenizer")
}

fn write_jsonl(path: &Path, rows: &[Value]) {
    let text: String = rows.iter().map(|r| format!("{r}\n")).collect();
    fs::write(path, text).unwrap();
}

/// Load `rows` through a temporary JSONL file.
fn load(
    rows: &[Value],
    tokenizer: &Tokenizer,
    format: DatasetFormat,
    template: Option<&ChatTemplate>,
) -> TrainingDataset {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.jsonl");
    write_jsonl(&path, rows);
    TrainingDataset::from_jsonl_tokenized(&path, tokenizer, format, 4096, template, None).unwrap()
}

/// The text of the tokens whose label is not `-100`.
fn trained_text(sample: &Sample, tokenizer: &Tokenizer) -> String {
    let labels = sample.labels.as_ref().expect("labels");
    assert_eq!(labels.len(), sample.input_ids.len());
    let ids: Vec<u32> = labels
        .iter()
        .zip(&sample.input_ids)
        .filter(|&(&label, _)| label != -100)
        .map(|(&label, &id)| {
            assert_eq!(label, i64::from(id), "a trained label is its own token");
            id
        })
        .collect();
    tokenizer.decode_with_special_tokens(&ids).unwrap()
}

fn full_text(sample: &Sample, tokenizer: &Tokenizer) -> String {
    tokenizer
        .decode_with_special_tokens(&sample.input_ids)
        .unwrap()
}

fn alpaca(instruction: &str, output: &str) -> Value {
    json!({"instruction": instruction, "input": "", "output": output})
}

// -- the prompt/completion split ---------------------------------------------

/// A tokenizer whose post-processor appends `</s>` encodes the prompt on its
/// own with that `</s>` too. Counting its tokens to find where the response
/// starts then lands one token late and masks the first token of the answer.
#[test]
fn the_split_is_not_thrown_off_by_tokens_the_post_processor_adds() {
    let tokenizer = char_tokenizer(Some("$A </s>"));
    let ds = load(
        &[alpaca("Say hi", "Hello")],
        &tokenizer,
        DatasetFormat::Alpaca,
        None,
    );
    let s = ds.get(0).unwrap();
    assert_eq!(trained_text(s, &tokenizer), "Hello</s>");
    assert!(full_text(s, &tokenizer).starts_with("### Instruction:"));
}

/// A chat template that writes its own BOS must not get a second one from the
/// tokenizer's post-processor.
#[test]
fn templated_text_gets_no_second_bos() {
    let tokenizer = char_tokenizer(Some("<|begin_of_text|> $A"));
    let template = ChatTemplate::llama3();
    let ds = load(
        &[json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"}]})],
        &tokenizer,
        DatasetFormat::OpenAi,
        Some(&template),
    );
    let s = ds.get(0).unwrap();
    let bos = 104;
    assert_eq!(s.input_ids.iter().filter(|&&id| id == bos).count(), 1);
    assert_eq!(s.input_ids[0], bos);
}

/// Instruction-style completions end in EOS, or the model never learns to
/// stop. A tokenizer that already appends it does not get it twice.
#[test]
fn a_completion_ends_in_one_eos() {
    let plain = char_tokenizer(None);
    let ds = load(
        &[alpaca("Say hi", "Hello")],
        &plain,
        DatasetFormat::Alpaca,
        None,
    );
    assert_eq!(trained_text(ds.get(0).unwrap(), &plain), "Hello</s>");

    let appends = char_tokenizer(Some("$A </s>"));
    let ds = load(
        &[alpaca("Say hi", "Hello")],
        &appends,
        DatasetFormat::Alpaca,
        None,
    );
    let ids = &ds.get(0).unwrap().input_ids;
    assert_eq!(ids.iter().filter(|&&id| id == 101).count(), 1);
}

#[test]
fn plain_text_is_trained_whole_and_ends_in_eos() {
    let tokenizer = char_tokenizer(None);
    let ds = load(
        &[json!({"text": "just text"})],
        &tokenizer,
        DatasetFormat::Simple,
        None,
    );
    assert_eq!(
        trained_text(ds.get(0).unwrap(), &tokenizer),
        "just text</s>"
    );
}

// -- multi-turn chat -----------------------------------------------------------

fn two_turns() -> Value {
    json!({"messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"}]})
}

/// Every assistant turn trains, nothing else does.
#[test]
fn every_assistant_turn_of_a_conversation_is_trained() {
    let tokenizer = char_tokenizer(None);
    let template = ChatTemplate::chatml();
    let ds = load(
        &[two_turns()],
        &tokenizer,
        DatasetFormat::OpenAi,
        Some(&template),
    );
    assert_eq!(
        trained_text(ds.get(0).unwrap(), &tokenizer),
        "a1<|im_end|>\na2<|im_end|>\n"
    );
}

/// A template that writes the last assistant turn differently from earlier
/// ones (an empty reasoning block only on the turn being answered, as
/// reasoning models' templates do) cannot have its earlier turns located by
/// rendering them alone. Each becomes a sample of its own, the conversation up
/// to it rendered as the template renders it for generation, rather than
/// being trained at the wrong bytes or lost; the final turn trains in the
/// first sample, and `LastTurn` makes no extra ones.
#[test]
fn a_template_that_rewrites_earlier_turns_renders_each_as_its_own_sample() {
    let tokenizer = char_tokenizer(None);
    let mut template = ChatTemplate::chatml();
    template.jinja_source = Some(
        "{% for m in messages %}<|im_start|>{{ m.role }}\n\
         {% if m.role == 'assistant' and loop.last %}<think>\n\n</think>\n\n{% endif %}\
         {{ m.content }}<|im_end|>\n{% endfor %}\
         {% if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
            .to_string(),
    );
    let ds = load(
        &[two_turns()],
        &tokenizer,
        DatasetFormat::OpenAi,
        Some(&template),
    );
    assert_eq!(ds.len(), 2);
    assert_eq!(
        trained_text(ds.get(0).unwrap(), &tokenizer),
        "a2<|im_end|>\n"
    );
    // The first turn, as the template renders it when asked to answer it (its
    // empty reasoning block is part of the generation prompt, so it is
    // context): the conversation stops there.
    assert_eq!(
        trained_text(ds.get(1).unwrap(), &tokenizer),
        "a1<|im_end|>\n"
    );
    let second = full_text(ds.get(1).unwrap(), &tokenizer);
    assert!(second.contains("<think>\n\n</think>\n\na1"), "{second}");
    assert!(!second.contains("a2"), "{second}");

    use pmetal_data::{AssistantLoss, DatasetLoadOptions};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.jsonl");
    write_jsonl(&path, &[two_turns()]);
    let last = TrainingDataset::load(
        &path,
        &tokenizer,
        &DatasetLoadOptions::new(4096)
            .template(Some(&template))
            .assistant_loss(AssistantLoss::LastTurn),
    )
    .unwrap();
    assert_eq!(last.len(), 1);
}

#[test]
fn sharegpt_turns_train_like_openai_turns() {
    let tokenizer = char_tokenizer(None);
    let template = ChatTemplate::chatml();
    let ds = load(
        &[json!({"conversations": [
            {"from": "system", "value": "be brief"},
            {"from": "human", "value": "q1"},
            {"from": "gpt", "value": "a1"},
            {"from": "human", "value": "q2"},
            {"from": "gpt", "value": "a2"}]})],
        &tokenizer,
        DatasetFormat::ShareGpt,
        Some(&template),
    );
    assert_eq!(
        trained_text(ds.get(0).unwrap(), &tokenizer),
        "a1<|im_end|>\na2<|im_end|>\n"
    );
}

/// Without a template the assistant's words train and the `Assistant: `
/// header (the prompt) does not.
#[test]
fn without_a_template_only_the_assistant_words_train() {
    let tokenizer = char_tokenizer(None);
    let ds = load(&[two_turns()], &tokenizer, DatasetFormat::OpenAi, None);
    assert_eq!(
        trained_text(ds.get(0).unwrap(), &tokenizer),
        "a1\n\na2\n\n</s>"
    );
}

/// A message's `weight` of 0 keeps that assistant turn out of the loss, as
/// OpenAI's chat fine-tuning format defines it.
#[test]
fn an_assistant_turn_of_weight_zero_is_not_trained() {
    let tokenizer = char_tokenizer(None);
    let template = ChatTemplate::chatml();
    let ds = load(
        &[json!({"messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1", "weight": 0},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2", "weight": 1}]})],
        &tokenizer,
        DatasetFormat::OpenAi,
        Some(&template),
    );
    assert_eq!(
        trained_text(ds.get(0).unwrap(), &tokenizer),
        "a2<|im_end|>\n"
    );
}

/// Tool calls are the assistant's words and train; what the tool answered
/// does not, and neither does the user turn that carries it.
#[test]
fn tool_calls_train_and_tool_results_do_not() {
    let tokenizer = char_tokenizer(None);
    let template = ChatTemplate::chatml();
    let ds = load(
        &[json!({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\": \"SF\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "SECRET-RESULT"},
            {"role": "assistant", "content": "It is fine."}]})],
        &tokenizer,
        DatasetFormat::OpenAi,
        Some(&template),
    );
    let trained = trained_text(ds.get(0).unwrap(), &tokenizer);
    assert!(trained.contains("get_weather"), "{trained}");
    assert!(trained.contains("It is fine."), "{trained}");
    assert!(!trained.contains("SECRET-RESULT"), "{trained}");
    assert!(!trained.contains("weather?"), "{trained}");
    assert!(full_text(ds.get(0).unwrap(), &tokenizer).contains("SECRET-RESULT"));
}

// -- format detection and parsing --------------------------------------------

/// `tatsu-lab/alpaca` has a pre-rendered `text` column next to the
/// instruction fields; the structured columns are the ones that mask.
#[test]
fn a_text_column_does_not_hide_the_structured_columns() {
    let tokenizer = char_tokenizer(None);
    let row = json!({"instruction": "Say hi", "input": "", "output": "Hello",
                     "text": "### Instruction:\nSay hi\n\n### Response:\nHello"});
    let ds = load(&[row], &tokenizer, DatasetFormat::Auto, None);
    assert_eq!(trained_text(ds.get(0).unwrap(), &tokenizer), "Hello</s>");
}

#[test]
fn an_alpaca_row_with_a_null_input_loads() {
    let tokenizer = char_tokenizer(None);
    let row = json!({"instruction": "Say hi", "input": null, "output": "Hello"});
    let ds = load(&[row], &tokenizer, DatasetFormat::Auto, None);
    assert_eq!(trained_text(ds.get(0).unwrap(), &tokenizer), "Hello</s>");
    let with_input = json!({"instruction": "Say hi", "input": "to Bob", "output": "Hello Bob"});
    let ds = load(&[with_input], &tokenizer, DatasetFormat::Auto, None);
    let text = full_text(ds.get(0).unwrap(), &tokenizer);
    assert!(text.contains("### Input:\nto Bob"), "{text}");
}

#[test]
fn prompt_completion_rows_are_detected_and_masked() {
    let tokenizer = char_tokenizer(None);
    let ds = load(
        &[json!({"prompt": "2+2=", "completion": " 4"})],
        &tokenizer,
        DatasetFormat::Auto,
        None,
    );
    let s = ds.get(0).unwrap();
    assert_eq!(trained_text(s, &tokenizer), " 4</s>");
    assert!(full_text(s, &tokenizer).starts_with("2+2= 4"));
}

/// The conversational prompt/completion form: only the completion trains,
/// even when the prompt holds assistant turns.
#[test]
fn conversational_prompt_completion_trains_the_completion_only() {
    let tokenizer = char_tokenizer(None);
    let template = ChatTemplate::chatml();
    let ds = load(
        &[json!({
            "prompt": [{"role": "user", "content": "q1"},
                       {"role": "assistant", "content": "shot"},
                       {"role": "user", "content": "q2"}],
            "completion": [{"role": "assistant", "content": "a2"}]})],
        &tokenizer,
        DatasetFormat::Auto,
        Some(&template),
    );
    assert_eq!(
        trained_text(ds.get(0).unwrap(), &tokenizer),
        "a2<|im_end|>\n"
    );
}

/// An explicit format wins over detection.
#[test]
fn an_explicit_format_overrides_detection() {
    let tokenizer = char_tokenizer(None);
    let row = json!({"instruction": "Say hi", "input": "", "output": "Hello",
                     "text": "whole text"});
    let ds = load(&[row], &tokenizer, DatasetFormat::Simple, None);
    assert_eq!(
        trained_text(ds.get(0).unwrap(), &tokenizer),
        "whole text</s>"
    );
}

/// A row nothing recognises is an error naming the columns it has.
#[test]
fn an_unrecognised_row_names_its_columns() {
    let tokenizer = char_tokenizer(None);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.jsonl");
    write_jsonl(&path, &[json!({"foo": 1, "bar": 2})]);
    let err = TrainingDataset::from_jsonl_tokenized(
        &path,
        &tokenizer,
        DatasetFormat::Auto,
        64,
        None,
        None,
    )
    .err()
    .expect("an error")
    .to_string();
    assert!(err.contains("foo") && err.contains("bar"), "{err}");
}

// -- containers and layouts --------------------------------------------------

/// A Hugging Face dataset directory: the `train` shards, all of them, in
/// order; `test` and `validation` are other splits.
#[test]
fn a_directory_of_train_shards_loads_every_shard() {
    let tokenizer = char_tokenizer(None);
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("data");
    fs::create_dir_all(&data).unwrap();
    write_jsonl(
        &data.join("train-00000-of-00002.jsonl"),
        &[json!({"text": "one"}), json!({"text": "two"})],
    );
    write_jsonl(
        &data.join("train-00001-of-00002.jsonl"),
        &[json!({"text": "three"})],
    );
    write_jsonl(
        &data.join("test-00000-of-00001.jsonl"),
        &[json!({"text": "TEST"})],
    );
    write_jsonl(
        &data.join("validation-00000-of-00001.jsonl"),
        &[json!({"text": "VALID"})],
    );
    let ds = TrainingDataset::from_jsonl_tokenized(
        dir.path(),
        &tokenizer,
        DatasetFormat::Auto,
        64,
        None,
        None,
    )
    .unwrap();
    let texts: Vec<String> = ds
        .samples()
        .iter()
        .map(|s| trained_text(s, &tokenizer))
        .collect();
    assert_eq!(texts, ["one</s>", "two</s>", "three</s>"]);
}

#[test]
fn a_json_array_file_is_read_as_rows() {
    let tokenizer = char_tokenizer(None);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.json");
    fs::write(
        &path,
        json!([alpaca("a", "b"), alpaca("c", "d")]).to_string(),
    )
    .unwrap();
    let ds = TrainingDataset::from_jsonl_tokenized(
        &path,
        &tokenizer,
        DatasetFormat::Auto,
        256,
        None,
        None,
    )
    .unwrap();
    assert_eq!(ds.len(), 2);
    assert_eq!(trained_text(ds.get(1).unwrap(), &tokenizer), "d</s>");
}

/// A chat dataset in Parquet (nested `messages` column) trains exactly as the
/// same rows in JSONL.
#[test]
fn a_parquet_chat_dataset_trains_like_its_jsonl() {
    use arrow::json::ReaderBuilder;
    use arrow::json::reader::infer_json_schema_from_seekable;
    use parquet::arrow::ArrowWriter;

    let tokenizer = char_tokenizer(None);
    let template = ChatTemplate::chatml();
    let rows = [two_turns(), two_turns()];
    let dir = tempfile::tempdir().unwrap();
    let jsonl = dir.path().join("data.jsonl");
    write_jsonl(&jsonl, &rows);

    let mut file = std::io::BufReader::new(fs::File::open(&jsonl).unwrap());
    let (schema, _) = infer_json_schema_from_seekable(&mut file, None).unwrap();
    let schema = Arc::new(schema);
    let batch = ReaderBuilder::new(schema.clone())
        .build(std::io::BufReader::new(fs::File::open(&jsonl).unwrap()))
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let parquet_path = dir.path().join("data.parquet");
    let mut writer =
        ArrowWriter::try_new(fs::File::create(&parquet_path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let from_jsonl = TrainingDataset::from_jsonl_tokenized(
        &jsonl,
        &tokenizer,
        DatasetFormat::Auto,
        4096,
        Some(&template),
        None,
    )
    .unwrap();
    let from_parquet = TrainingDataset::from_jsonl_tokenized(
        &parquet_path,
        &tokenizer,
        DatasetFormat::Auto,
        4096,
        Some(&template),
        None,
    )
    .unwrap();
    assert_eq!(from_parquet.len(), 2);
    for (a, b) in from_jsonl.samples().iter().zip(from_parquet.samples()) {
        assert_eq!(a.input_ids, b.input_ids);
        assert_eq!(a.labels, b.labels);
    }
}

/// A prompt as long as the context leaves nothing to train on; the sample is
/// dropped rather than fed to the loss as a batch of `-100`s.
#[test]
fn a_sample_truncated_to_its_prompt_is_dropped() {
    let tokenizer = char_tokenizer(None);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.jsonl");
    write_jsonl(
        &path,
        &[
            alpaca(&"x".repeat(200), "never seen"),
            alpaca("short", "kept"),
        ],
    );
    let ds = TrainingDataset::from_jsonl_tokenized(
        &path,
        &tokenizer,
        DatasetFormat::Alpaca,
        64,
        None,
        None,
    )
    .unwrap();
    assert_eq!(ds.len(), 1);
    assert!(trained_text(ds.get(0).unwrap(), &tokenizer).starts_with("kept"));
}

// -- a real byte-level BPE tokenizer ---------------------------------------------

/// The masks above, read off a real byte-level BPE tokenizer (offsets trimmed,
/// digits split, multi-byte characters across token boundaries). Set
/// `PMETAL_SFT_TOKENIZER_DIR` to a model directory with a `tokenizer.json`
/// that has the ChatML markers.
#[test]
#[ignore = "needs a real tokenizer"]
fn a_real_bpe_tokenizer_masks_exactly_the_assistant_turns() {
    let dir = std::env::var_os("PMETAL_SFT_TOKENIZER_DIR").expect("PMETAL_SFT_TOKENIZER_DIR");
    let tokenizer = Tokenizer::from_model_dir(dir).unwrap();
    let template = ChatTemplate::chatml();
    let first = |rows: &[Value]| {
        let ds = load(rows, &tokenizer, DatasetFormat::Auto, Some(&template));
        trained_text(ds.get(0).unwrap(), &tokenizer)
    };
    let trained = [
        first(&[json!({"messages": [
            {"role": "user", "content": "Wie geht's? 日本語 ✓ 12345"},
            {"role": "assistant", "content": "  Gut, danke!\n\nUnd dir? 🙂 x²"},
            {"role": "user", "content": "Gut."},
            {"role": "assistant", "content": "Prima."}]})]),
        first(&[alpaca("Say hi \u{e9}", " Hello  wörld")]),
        first(&[json!({"prompt": "2+2=", "completion": " 4"})]),
    ];
    println!("{trained:#?}");
    // The answer starts with two spaces, and the tokenizer joins the first to
    // the header's newline ("\n " is one token, which starts in the prompt), so
    // that token is prompt and one space is not trained: the model is shown
    // the header alone, and could not produce that joined token anyway.
    assert_eq!(
        trained[0],
        " Gut, danke!\n\nUnd dir? 🙂 x²<|im_end|>\nPrima.<|im_end|>\n"
    );
    assert_eq!(trained[1], " Hello  wörld<|endoftext|>");
    assert_eq!(trained[2], " 4<|endoftext|>");
}

// -- offsets, truncation and options -------------------------------------------

/// The mask is cut in bytes of the text: a multi-byte character in the prompt
/// must not move it.
#[test]
fn a_multibyte_prompt_masks_exactly_the_prompt() {
    let tokenizer = char_tokenizer(None);
    let ds = load(
        &[alpaca("héllo wörld ✓ 日本", "Hello")],
        &tokenizer,
        DatasetFormat::Alpaca,
        None,
    );
    assert_eq!(trained_text(ds.get(0).unwrap(), &tokenizer), "Hello</s>");
}

/// Truncation cuts tokens and labels together, from the end.
#[test]
fn truncation_cuts_the_end_and_keeps_the_mask_aligned() {
    let tokenizer = char_tokenizer(None);
    let prompt_len = "### Instruction:\nhi\n\n### Response:\n".len();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.jsonl");
    write_jsonl(&path, &[alpaca("hi", "abcdefgh")]);
    let ds = TrainingDataset::from_jsonl_tokenized(
        &path,
        &tokenizer,
        DatasetFormat::Alpaca,
        prompt_len + 3,
        None,
        None,
    )
    .unwrap();
    let s = ds.get(0).unwrap();
    assert_eq!(s.input_ids.len(), prompt_len + 3);
    assert_eq!(trained_text(s, &tokenizer), "abc");
    let stats = ds.compute_statistics(prompt_len + 3);
    assert_eq!(stats.truncated_count, 1, "the original length is recorded");
}

/// `LastTurn` trains the final assistant turn only.
#[test]
fn last_turn_loss_trains_only_the_final_assistant_turn() {
    use pmetal_data::{AssistantLoss, DatasetLoadOptions};
    let tokenizer = char_tokenizer(None);
    let template = ChatTemplate::chatml();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.jsonl");
    write_jsonl(&path, &[two_turns()]);
    let ds = TrainingDataset::load(
        &path,
        &tokenizer,
        &DatasetLoadOptions::new(4096)
            .template(Some(&template))
            .assistant_loss(AssistantLoss::LastTurn),
    )
    .unwrap();
    assert_eq!(
        trained_text(ds.get(0).unwrap(), &tokenizer),
        "a2<|im_end|>\n"
    );
}

/// The builder the preference trainers use tokenizes a conversation as dataset
/// loading does: one BOS, the final assistant turn trained.
#[test]
fn the_training_sample_builder_agrees_with_dataset_loading() {
    use pmetal_data::chat_templates::{Message, TrainingSampleBuilder};
    let tokenizer = char_tokenizer(Some("<|begin_of_text|> $A"));
    let template = ChatTemplate::llama3();
    let (ids, labels) = TrainingSampleBuilder::new(template.clone())
        .build_tokenized(
            &[Message::user("hi"), Message::assistant("yo")],
            &tokenizer,
            4096,
        )
        .unwrap();
    assert_eq!(ids.iter().filter(|&&id| id == 104).count(), 1);
    let sample = Sample::with_labels(ids, labels);
    assert!(trained_text(&sample, &tokenizer).starts_with("yo"));

    let ds = load(
        &[json!({"messages": [
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]})],
        &tokenizer,
        DatasetFormat::OpenAi,
        Some(&template),
    );
    assert_eq!(ds.get(0).unwrap().input_ids, sample.input_ids);
    assert_eq!(ds.get(0).unwrap().labels, sample.labels);
}

/// Loading is chunked; rows past the first chunk are read the same way.
#[test]
fn a_dataset_larger_than_a_chunk_loads_in_order() {
    let tokenizer = char_tokenizer(None);
    let rows: Vec<Value> = (0..5000)
        .map(|i| json!({"text": format!("row {i}")}))
        .collect();
    let ds = load(&rows, &tokenizer, DatasetFormat::Simple, None);
    assert_eq!(ds.len(), 5000);
    assert_eq!(
        trained_text(ds.get(4999).unwrap(), &tokenizer),
        "row 4999</s>"
    );
    assert_eq!(
        trained_text(ds.get(2048).unwrap(), &tokenizer),
        "row 2048</s>"
    );
}

/// A bad row fails the load and says which.
#[test]
fn a_bad_row_names_its_position() {
    let tokenizer = char_tokenizer(None);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.jsonl");
    write_jsonl(
        &path,
        &[alpaca("a", "b"), json!({"instruction": "no answer"})],
    );
    let err = TrainingDataset::from_jsonl_tokenized(
        &path,
        &tokenizer,
        DatasetFormat::Alpaca,
        64,
        None,
        None,
    )
    .err()
    .expect("an error")
    .to_string();
    assert!(err.contains("row 2"), "{err}");
}
