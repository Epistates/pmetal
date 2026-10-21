//! `pmetal dataset template` and `prepare`: putting a dataset in a model's
//! chat format, and the whole way from a Hub dataset to train and validation
//! files.

use std::collections::HashSet;
use std::io::Write;

use pmetal_core::Result as DataResult;
use pmetal_data::chat_templates::ChatTemplate;
use pmetal_data::rows::Row;
use pmetal_data::{DatasetFormat, TextSample};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde_json::Value;

use super::common::{
    JsonlWriter, ModelText, TemplateChoice, banner, choose_template, open_dataset, open_rows,
    parse_column_map, pct, percentile, rename_columns, summary, text_row,
};
use crate::cli::dataset::TemplateName;

/// How a conversation is written out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Output {
    /// `{"text": ...}`: the whole conversation.
    Text,
    /// `{"text": ...}` up to the generation prompt of the last assistant turn.
    GenerationPrompt,
    /// `{"prompt": ..., "completion": ...}`: the last assistant turn is the
    /// completion, the only part a model trains on.
    PromptCompletion,
}

/// What a templating run did.
#[derive(Debug, Default)]
pub(super) struct TemplateReport {
    pub(super) total: usize,
    pub(super) written: usize,
    /// Rows of a layout no format reads.
    pub(super) skipped: usize,
    pub(super) first_problem: Option<String>,
}

/// Where the model's answer begins in a sample: the end of what a model is
/// shown before it answers its last turn.
fn answer_start(sample: &TextSample) -> Option<usize> {
    sample
        .trained
        .as_ref()
        .and_then(|spans| spans.last())
        .map(|span| span.start)
        .or_else(|| sample.prompt.as_ref().map(String::len))
        .filter(|&at| at > 0 && at <= sample.text.len())
}

/// `row` with the system message a conversation lacks. Only conversations
/// take one; any other row is returned as it is.
fn with_system(mut row: Row, system: Option<&str>) -> Row {
    let is_chat = ["messages", "conversations"]
        .iter()
        .any(|key| row.get(*key).is_some_and(Value::is_array));
    if let Some(system) = system {
        if is_chat && !row.contains_key("system") {
            row.insert("system".into(), system.into());
        }
    }
    row
}

/// Write each row of `rows` as the text (or prompt and completion) the chat
/// template makes of it. Rows that are an instruction and its answer (Alpaca,
/// reasoning, prompt and completion) are formatted as the conversation they
/// stand for; a document is passed through.
pub(super) fn template_rows<W: Write>(
    rows: impl IntoIterator<Item = DataResult<Row>>,
    template: Option<&ChatTemplate>,
    system: Option<&str>,
    output: Output,
    model: Option<&ModelText>,
    out: &mut JsonlWriter<W>,
) -> anyhow::Result<TemplateReport> {
    let stored = |text: &str| match (model, template) {
        (Some(model), Some(template)) => model.for_training(text, template),
        _ => text.to_string(),
    };
    let mut report = TemplateReport::default();
    for item in rows {
        report.total += 1;
        let row = with_system(item?, system);
        let samples = DatasetFormat::detect(&row)
            .and_then(|format| format.render_row_as_chat(&row, template));
        let samples = match samples {
            Ok(samples) => samples,
            Err(e) => {
                report.skipped += 1;
                report
                    .first_problem
                    .get_or_insert(format!("row {}: {e}", report.total));
                continue;
            }
        };
        // A generation prompt is one per row: the whole conversation's.
        let take = if output == Output::GenerationPrompt {
            1
        } else {
            samples.len()
        };
        for sample in samples.iter().take(take) {
            let split = answer_start(sample);
            let written = match (output, split) {
                (Output::GenerationPrompt, Some(at)) => text_row(stored(&sample.text[..at])),
                (Output::PromptCompletion, Some(at)) => {
                    let mut row = Row::new();
                    row.insert("prompt".into(), stored(&sample.text[..at]).into());
                    row.insert("completion".into(), sample.text[at..].into());
                    row
                }
                _ => text_row(stored(&sample.text)),
            };
            out.write(&written)?;
            report.written += 1;
        }
    }
    Ok(report)
}

pub(super) async fn template(
    input: String,
    output: String,
    name: Option<TemplateName>,
    system: Option<String>,
    model: Option<String>,
    add_generation_prompt: bool,
    mask_prompt: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        !(add_generation_prompt && mask_prompt),
        "--add-generation-prompt writes the prompt alone and --mask-prompt writes the prompt \
         and its completion; give one"
    );
    let model = match &model {
        Some(model) => Some(ModelText::load(model).await?),
        None => None,
    };
    let TemplateChoice { template, label } = choose_template(name, model.as_ref())?;
    let mut fields = vec![
        ("Input", input.clone()),
        ("Output", output.clone()),
        ("Template", label),
    ];
    if let Some(system) = &system {
        fields.push(("System", system.clone()));
    }
    if add_generation_prompt {
        fields.push(("Generation", "up to the generation prompt".into()));
    }
    if mask_prompt {
        fields.push(("Layout", "prompt and completion".into()));
    }
    banner("PMetal Chat Template", &fields);

    let layout = if add_generation_prompt {
        Output::GenerationPrompt
    } else if mask_prompt {
        Output::PromptCompletion
    } else {
        Output::Text
    };
    let mut out = JsonlWriter::create(&output, "dataset template output")?;
    let report = template_rows(
        open_rows(&input)?,
        template.as_ref(),
        system.as_deref(),
        layout,
        model.as_ref(),
        &mut out,
    )?;
    out.finish()?;

    let mut lines = vec![
        ("Total samples", report.total.to_string()),
        ("Templated", report.written.to_string()),
    ];
    if report.skipped > 0 {
        lines.push(("Skipped", report.skipped.to_string()));
        if let Some(problem) = report.first_problem {
            lines.push(("First skipped", problem));
        }
    }
    lines.push(("Output", output));
    summary("Template Complete!", &lines);
    Ok(())
}

// -- prepare ---------------------------------------------------------------------

/// What preparing a dataset did.
#[derive(Debug, Default)]
pub(super) struct PrepareReport {
    pub(super) total: usize,
    pub(super) kept: usize,
    pub(super) too_long: usize,
    pub(super) duplicate: usize,
    pub(super) skipped: usize,
    pub(super) first_problem: Option<String>,
}

/// What a prepared row says to training: a document, trained whole, or a
/// prompt (context, not trained) and the completion that follows it (the
/// only part trained).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum PreparedRow {
    Document(String),
    Completion { prompt: String, completion: String },
}

impl PreparedRow {
    /// The row as it is written: the columns training reads back.
    pub(super) fn to_row(&self) -> Row {
        match self {
            Self::Document(text) => text_row(text.clone()),
            Self::Completion { prompt, completion } => {
                let mut row = Row::new();
                row.insert("prompt".into(), prompt.clone().into());
                row.insert("completion".into(), completion.clone().into());
                row
            }
        }
    }

    /// The sample training makes of this row.
    fn sample(&self) -> TextSample {
        match self {
            Self::Document(text) => TextSample::document(text.clone()),
            Self::Completion { prompt, completion } => {
                TextSample::completion(prompt.clone(), completion)
            }
        }
    }
}

/// The prepared samples: each row and the tokens it has in training.
pub(super) struct Prepared {
    pub(super) samples: Vec<(PreparedRow, usize)>,
    pub(super) report: PrepareReport,
}

/// The rows that train what `sample` says to train.
///
/// A conversation trains on its assistant turns and nothing else; written as
/// one `{"text"}` row it would be read back as a document and trained whole,
/// the user's words and the system prompt included. So each assistant turn
/// becomes a prompt (everything before it) and a completion (the turn). The
/// turns of one conversation are separate rows because a prompt and completion
/// are one turn; the gradient is the same as the conversation's, since a turn
/// only reads the tokens before it.
///
/// Text with no trained span of its own is a document.
fn training_rows(
    sample: TextSample,
    model: &ModelText,
    template: Option<&ChatTemplate>,
) -> Vec<PreparedRow> {
    let stored = |text: &str| match template {
        Some(template) => model.for_training(text, template),
        None => text.to_string(),
    };
    let spans = match (&sample.trained, &sample.prompt) {
        (Some(spans), _) => spans.clone(),
        (None, Some(prompt)) if sample.text.starts_with(prompt.as_str()) => {
            std::iter::once(prompt.len()..sample.text.len()).collect()
        }
        (None, _) => return vec![PreparedRow::Document(stored(&sample.text))],
    };
    spans
        .into_iter()
        .filter(|span| span.start < span.end && span.end <= sample.text.len())
        .map(|span| PreparedRow::Completion {
            prompt: stored(&sample.text[..span.start]),
            completion: sample.text[span.start..span.end].to_string(),
        })
        .collect()
}

/// Format every row for the model, and keep what fits `max_seq_len` tokens
/// and, unless `dedup` is off, what is not a repeat.
///
/// Length is measured as training will see each prepared row: a document or a
/// prompt and completion, which the tokenizer brackets with its special
/// tokens.
pub(super) fn prepare_rows(
    rows: impl IntoIterator<Item = DataResult<Row>>,
    columns: Option<&std::collections::HashMap<String, String>>,
    model: &ModelText,
    template: Option<&ChatTemplate>,
    max_seq_len: usize,
    dedup: bool,
) -> anyhow::Result<Prepared> {
    let mut seen: HashSet<PreparedRow> = HashSet::new();
    let mut report = PrepareReport::default();
    let mut samples = Vec::new();
    for item in rows {
        report.total += 1;
        let row = item?;
        let row = match columns {
            Some(map) => rename_columns(row, map),
            None => row,
        };
        let rendered = DatasetFormat::detect(&row)
            .and_then(|format| format.render_row_as_chat(&row, template));
        let rendered = match rendered {
            Ok(rendered) => rendered,
            Err(e) => {
                report.skipped += 1;
                report
                    .first_problem
                    .get_or_insert(format!("row {}: {e}", report.total));
                continue;
            }
        };
        let mut trains_something = false;
        for sample in rendered {
            for prepared in training_rows(sample, model, template) {
                trains_something = true;
                let tokens = prepared.sample().token_len(&model.tokenizer)?;
                if tokens > max_seq_len {
                    report.too_long += 1;
                    continue;
                }
                if dedup && !seen.insert(prepared.clone()) {
                    report.duplicate += 1;
                    continue;
                }
                samples.push((prepared, tokens));
                report.kept += 1;
            }
        }
        if !trains_something {
            // A conversation with no assistant turn has nothing to learn from.
            report.skipped += 1;
            report.first_problem.get_or_insert(format!(
                "row {}: no assistant turn to train on",
                report.total
            ));
        }
    }
    Ok(Prepared { samples, report })
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn prepare(
    dataset: String,
    output_dir: String,
    model: String,
    name: Option<TemplateName>,
    max_seq_len: usize,
    val_ratio: f64,
    seed: u64,
    no_dedup: bool,
    columns: Option<String>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        (0.0..1.0).contains(&val_ratio),
        "--val-ratio must be at least 0 and below 1 (got {val_ratio})"
    );
    let columns = columns.as_deref().map(parse_column_map).transpose()?;
    let mut fields = vec![
        ("Dataset", dataset.clone()),
        ("Output", output_dir.clone()),
        ("Model", model.clone()),
        ("Max seq len", max_seq_len.to_string()),
        ("Val ratio", format!("{val_ratio:.2}")),
        ("Seed", seed.to_string()),
    ];
    if let Some(columns) = &columns {
        let mut renames: Vec<String> = columns.iter().map(|(s, t)| format!("{t}={s}")).collect();
        renames.sort();
        fields.push(("Columns", renames.join(",")));
    }
    banner("PMetal Dataset Prepare", &fields);

    // The outputs are checked before anything is fetched.
    let train_path = format!("{}/train.jsonl", output_dir.trim_end_matches('/'));
    let val_path = format!("{}/val.jsonl", output_dir.trim_end_matches('/'));
    let mut train_out = JsonlWriter::create(&train_path, "dataset prepare output")?;
    let mut val_out = JsonlWriter::create(&val_path, "dataset prepare output")?;

    println!("[1/4] Tokenizer and chat template");
    let model_text = ModelText::load(&model).await?;
    let TemplateChoice { template, label } = choose_template(name, Some(&model_text))?;
    println!("  Template: {label}");

    println!("[2/4] Loading the dataset...");
    let rows = open_dataset(&dataset, "train", None).await?;

    println!("[3/4] Formatting and filtering...");
    let Prepared {
        mut samples,
        report,
    } = prepare_rows(
        rows,
        columns.as_ref(),
        &model_text,
        template.as_ref(),
        max_seq_len,
        !no_dedup,
    )?;
    println!(
        "  Total: {}, Kept: {}, Filtered (long): {}, Filtered (dup): {}, Skipped: {}",
        report.total, report.kept, report.too_long, report.duplicate, report.skipped
    );
    if let Some(problem) = &report.first_problem {
        println!("  First skipped row: {problem}");
    }
    if samples.is_empty() {
        anyhow::bail!(
            "no sample is left to train on; the dataset's layouts are conversations \
             (messages, conversations), instruction/output, problem/solution, prompt/completion \
             and text; use --columns to rename custom columns"
        );
    }

    println!("[4/4] Splitting and writing...");
    samples.shuffle(&mut StdRng::seed_from_u64(seed));
    let val_count = (samples.len() as f64 * val_ratio).round() as usize;
    let train_count = samples.len() - val_count;
    let mut lengths: Vec<usize> = Vec::with_capacity(train_count);
    for (i, (prepared, tokens)) in samples.iter().enumerate() {
        if i < train_count {
            train_out.write(&prepared.to_row())?;
            lengths.push(*tokens);
        } else {
            val_out.write(&prepared.to_row())?;
        }
    }
    train_out.finish()?;
    val_out.finish()?;
    lengths.sort_unstable();

    summary(
        "Prepare Complete!",
        &[
            ("Train samples", train_count.to_string()),
            ("Val samples", val_count.to_string()),
            ("Token P50", percentile(&lengths, 0.5).to_string()),
            (
                "Token P95",
                format!(
                    "{} ({:.1}% of samples at most {max_seq_len})",
                    percentile(&lengths, 0.95),
                    pct(report.kept, report.kept + report.too_long)
                ),
            ),
            (
                "Token Max",
                lengths.last().copied().unwrap_or(0).to_string(),
            ),
            ("Train file", train_path),
            ("Val file", val_path),
        ],
    );
    Ok(())
}

/// `pmetal dataset formats`.
pub(super) fn formats() {
    println!("{FORMATS}");
}

const FORMATS: &str = r#"========================================
  PMetal Supported Formats & Templates
========================================

FILES:
------
JSONL, JSON (an array, or an object holding one), Parquet, Arrow, CSV and TSV,
a file or a directory (its train split). Hub datasets are fetched as Parquet.

ROW LAYOUTS (detected from the columns of a row):
-------------------------------------------------
1. OpenAI messages (also tool calls, content parts, per-message "weight"):
   {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

2. ShareGPT:
   {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}

3. Alpaca:
   {"instruction": "...", "input": "...", "output": "..."}

4. Reasoning:
   {"problem": "...", "thinking": "...", "solution": "..."}

5. Prompt and completion:
   {"prompt": "...", "completion": "..."}

6. Text:
   {"text": "The full text for training"}

CHAT TEMPLATES:
---------------
By default the template of --model is used: the Jinja chat template the model
ships (chat_template.jinja or tokenizer_config.json), rendered as the model's
own tokenizer library does, so what a model is prepared on is what it is
served. A model that ships none gets the built-in template of its family.

--template picks a built-in one instead:
  chatml, qwen   <|im_start|>role\ncontent<|im_end|>
  llama3         <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
  llama2         [INST] user [/INST] assistant </s>
  mistral        <s>[INST] user [/INST] assistant</s>
  phi            <|user|>\ncontent<|end|>\n<|assistant|>\ncontent<|end|>
  gemma          <start_of_turn>role\ncontent<end_of_turn>
  raw            no chat template: "Role: content" blocks
  auto           the model's own (needs --model)

EXAMPLE WORKFLOW:
-----------------
# Download and preview
pmetal dataset preview tatsu-lab/alpaca --num 3

# Full preparation pipeline
pmetal dataset prepare tatsu-lab/alpaca \
  --output-dir ./alpaca_prepared \
  --model Qwen/Qwen3-0.6B \
  --max-seq-len 2048 \
  --val-ratio 0.05

# Or step by step:
pmetal dataset download tatsu-lab/alpaca -o raw.jsonl
pmetal dataset filter -i raw.jsonl -o filtered.jsonl --model Qwen/Qwen3-0.6B --max-tokens 2048 --dedup
pmetal dataset template -i filtered.jsonl -o templated.jsonl --model Qwen/Qwen3-0.6B
pmetal dataset split -i templated.jsonl -o ./splits --val-ratio 0.1

# Analyze your data
pmetal dataset analyze -p train.jsonl --model Qwen/Qwen3-0.6B"#;

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_data::testing::word_level;
    use serde_json::json;

    fn row(value: serde_json::Value) -> Row {
        value.as_object().unwrap().clone()
    }

    fn ok(rows: &[Row]) -> Vec<DataResult<Row>> {
        rows.iter().cloned().map(Ok).collect()
    }

    fn texts(
        rows: &[Row],
        template: Option<&ChatTemplate>,
        system: Option<&str>,
        output: Output,
    ) -> (Vec<Row>, TemplateReport) {
        let mut out = JsonlWriter::new(Vec::new());
        let report = template_rows(ok(rows), template, system, output, None, &mut out).unwrap();
        let written = String::from_utf8(out.into_inner().unwrap())
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        (written, report)
    }

    /// Every instruction layout is a conversation in the model's format, and
    /// a conversation is written with the template's own words.
    #[test]
    fn every_layout_is_formatted_with_the_template() {
        let template = ChatTemplate::chatml();
        let rows = vec![
            row(json!({"instruction": "Say hi", "input": "politely", "output": "Hello!"})),
            row(json!({"problem": "2+2?", "thinking": "add", "solution": "4"})),
            row(
                json!({"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}),
            ),
            row(json!({"text": "plain text"})),
            row(json!({"unknown": 1})),
        ];
        let (written, report) = texts(&rows, Some(&template), None, Output::Text);
        assert_eq!((report.total, report.written, report.skipped), (5, 4, 1));
        assert!(report.first_problem.unwrap().contains("row 5"));
        let text = |i: usize| written[i]["text"].as_str().unwrap().to_string();
        assert!(
            text(0).contains("<|im_start|>user\nSay hi\n\npolitely<|im_end|>"),
            "{}",
            text(0)
        );
        assert!(text(0).contains("<|im_start|>assistant\nHello!<|im_end|>"));
        assert!(text(1).contains("<|im_start|>assistant\n<think>\nadd\n</think>\n\n4"));
        assert!(text(2).contains("<|im_start|>user\nq<|im_end|>"));
        assert_eq!(text(3), "plain text");
    }

    #[test]
    fn a_system_message_goes_to_conversations_that_have_none() {
        let template = ChatTemplate::chatml();
        let rows = vec![
            row(
                json!({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}),
            ),
            row(
                json!({"messages": [{"role": "system", "content": "mine"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}),
            ),
        ];
        let (written, _) = texts(&rows, Some(&template), Some("Be brief."), Output::Text);
        let first = written[0]["text"].as_str().unwrap();
        let second = written[1]["text"].as_str().unwrap();
        assert!(
            first.contains("<|im_start|>system\nBe brief.<|im_end|>"),
            "{first}"
        );
        assert!(
            second.contains("system\nmine<|im_end|>") && !second.contains("Be brief."),
            "{second}"
        );
    }

    #[test]
    fn a_generation_prompt_ends_where_the_model_starts_to_answer() {
        let template = ChatTemplate::chatml();
        let rows = vec![row(
            json!({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}),
        )];
        let (written, _) = texts(&rows, Some(&template), None, Output::GenerationPrompt);
        let text = written[0]["text"].as_str().unwrap();
        assert!(text.ends_with("<|im_start|>assistant\n"), "{text:?}");
        assert!(!text.contains("\na<|im_end|>"));
    }

    /// `--mask-prompt` was accepted and did nothing.
    #[test]
    fn mask_prompt_writes_the_prompt_and_the_completion_apart() {
        let template = ChatTemplate::chatml();
        let rows = vec![row(json!({"instruction": "Say hi", "output": "Hello!"}))];
        let (written, _) = texts(&rows, Some(&template), None, Output::PromptCompletion);
        let prompt = written[0]["prompt"].as_str().unwrap();
        let completion = written[0]["completion"].as_str().unwrap();
        assert!(prompt.ends_with("<|im_start|>assistant\n"), "{prompt:?}");
        assert!(completion.starts_with("Hello!"), "{completion:?}");
        // Together they are the conversation.
        let (whole, _) = texts(&rows, Some(&template), None, Output::Text);
        assert_eq!(
            format!("{prompt}{completion}"),
            whole[0]["text"].as_str().unwrap()
        );
    }

    fn model() -> ModelText {
        ModelText {
            tokenizer: word_level(&["<unk>", "a", "b", "c"], &[("<s>", true), ("</s>", true)]),
            template: ChatTemplate::chatml(),
            adds_bos: false,
        }
    }

    #[test]
    fn preparing_drops_what_is_too_long_and_what_repeats() {
        let model = model();
        let rows = vec![
            row(json!({"text": "a b"})),
            row(json!({"text": "a b"})),
            row(json!({"text": "a b c a b c a b c a b c"})),
            row(json!({"text": "c"})),
            row(json!({"nope": 1})),
        ];
        let prepared =
            prepare_rows(ok(&rows), None, &model, Some(&model.template), 6, true).unwrap();
        let r = &prepared.report;
        assert_eq!(
            (r.total, r.kept, r.too_long, r.duplicate, r.skipped),
            (5, 2, 1, 1, 1)
        );
        assert_eq!(prepared.samples.len(), 2);
        assert!(prepared.samples.iter().all(|(_, tokens)| *tokens <= 6));

        let with_dups =
            prepare_rows(ok(&rows), None, &model, Some(&model.template), 6, false).unwrap();
        assert_eq!(with_dups.report.kept, 3);
    }

    #[test]
    fn preparing_renames_columns_before_it_reads_them() {
        let model = model();
        let rows = vec![row(json!({"question": "a", "reply": "b"}))];
        let map = parse_column_map("instruction=question,output=reply").unwrap();
        let prepared = prepare_rows(
            ok(&rows),
            Some(&map),
            &model,
            Some(&model.template),
            99,
            true,
        )
        .unwrap();
        assert_eq!(prepared.report.kept, 1);
        // The question is context; the reply is what is trained.
        let PreparedRow::Completion { prompt, completion } = &prepared.samples[0].0 else {
            panic!("an instruction and its reply are a prompt and a completion");
        };
        assert!(
            prompt.contains("<|im_start|>user\na<|im_end|>"),
            "{prompt:?}"
        );
        assert!(completion.starts_with('b'), "{completion:?}");
    }

    fn byte_model() -> ModelText {
        ModelText {
            tokenizer: pmetal_data::tokenizer::testing::byte_level_bpe(
                &[],
                &["<|im_start|>", "<|im_end|>"],
                None,
            ),
            template: ChatTemplate::chatml(),
            adds_bos: false,
        }
    }

    /// What training takes a prepared file for: the text of the tokens that
    /// carry loss in each sample.
    fn trained_text_of(rows: &[Row], model: &ModelText) -> Vec<String> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("train.jsonl");
        let mut out = JsonlWriter::new(Vec::new());
        let prepared = prepare_rows(ok(rows), None, model, Some(&model.template), 4096, true)
            .unwrap()
            .samples;
        for (prepared, _) in &prepared {
            out.write(&prepared.to_row()).unwrap();
        }
        std::fs::write(&path, out.into_inner().unwrap()).unwrap();

        let dataset = pmetal_data::TrainingDataset::from_jsonl_tokenized(
            &path,
            &model.tokenizer,
            DatasetFormat::Auto,
            4096,
            Some(&model.template),
            None,
        )
        .unwrap();
        dataset
            .samples()
            .iter()
            .map(|sample| {
                let labels = sample
                    .labels
                    .as_ref()
                    .expect("a prepared sample is labeled");
                let trained: Vec<u32> = sample
                    .input_ids
                    .iter()
                    .zip(labels)
                    .filter(|&(_, &label)| label != pmetal_data::dataset::IGNORE_INDEX)
                    .map(|(&id, _)| id)
                    .collect();
                model.tokenizer.decode(&trained).unwrap()
            })
            .collect()
    }

    /// A prepared conversation was written as `{"text": ...}`, which training
    /// reads as a document and trains whole: the system prompt, the user's
    /// words and the template's own markers were all learned as if the model
    /// had written them.
    #[test]
    fn a_prepared_conversation_trains_the_assistants_words_and_nothing_else() {
        let model = byte_model();
        let rows = vec![
            row(json!({"messages": [
                {"role": "system", "content": "SYSTEMPROMPT"},
                {"role": "user", "content": "FIRSTQUESTION"},
                {"role": "assistant", "content": "FIRSTANSWER"},
                {"role": "user", "content": "SECONDQUESTION"},
                {"role": "assistant", "content": "SECONDANSWER"},
            ]})),
            row(json!({"instruction": "INSTRUCTION", "input": "INPUT", "output": "OUTPUT"})),
        ];
        let trained = trained_text_of(&rows, &model);

        for text in &trained {
            for context in [
                "SYSTEMPROMPT",
                "FIRSTQUESTION",
                "SECONDQUESTION",
                "INSTRUCTION",
                "INPUT",
                "<|im_start|>",
                "user",
            ] {
                assert!(
                    !text.contains(context),
                    "{context:?} is context, not something to learn: {text:?}"
                );
            }
        }
        // Every assistant turn is trained, each exactly once, and nothing
        // beyond it but the end of the turn (a special token the decoder
        // drops, which is how the model learns where it stops).
        assert_eq!(trained, ["FIRSTANSWER\n", "SECONDANSWER\n", "OUTPUT\n"]);
    }

    /// A document has no one to mask: it stays a document.
    #[test]
    fn plain_text_stays_a_document_and_is_trained_whole() {
        let model = byte_model();
        let trained = trained_text_of(&[row(json!({"text": "just some prose"}))], &model);
        assert_eq!(trained, ["just some prose"]);
    }

    #[test]
    fn a_conversation_with_no_assistant_turn_is_skipped_and_said_so() {
        let model = byte_model();
        let rows = vec![row(
            json!({"messages": [{"role": "user", "content": "anyone there?"}]}),
        )];
        let prepared =
            prepare_rows(ok(&rows), None, &model, Some(&model.template), 4096, true).unwrap();
        assert!(prepared.samples.is_empty());
        assert_eq!(prepared.report.skipped, 1);
        assert!(
            prepared
                .report
                .first_problem
                .unwrap()
                .contains("no assistant turn")
        );
    }

    /// A tokenizer that adds its own BOS must not get a second one from the
    /// text the template wrote.
    #[test]
    fn the_bos_a_tokenizer_adds_is_not_written_into_the_text_as_well() {
        let tokenizer = pmetal_data::testing::byte_level_bpe(
            &[],
            &["<|begin_of_text|>"],
            Some("<|begin_of_text|>"),
        );
        let mut template = ChatTemplate::llama3();
        template.bos_token = Some("<|begin_of_text|>".into());
        let model = ModelText {
            adds_bos: tokenizer
                .encode_with_special_tokens("x")
                .unwrap()
                .first()
                .copied()
                == tokenizer.bos_token_id(),
            tokenizer,
            template: template.clone(),
        };
        assert!(model.adds_bos);
        let rendered = format!("<|begin_of_text|>{}", "hello");
        assert_eq!(model.for_training(&rendered, &template), "hello");
        // Text that does not start with it is as it was.
        assert_eq!(model.for_training("hello", &template), "hello");
        // What a trainer then tokenizes has the one BOS.
        let ids = model.tokenizer.encode_with_special_tokens("hello").unwrap();
        let bos = model.tokenizer.bos_token_id().unwrap();
        assert_eq!(ids.iter().filter(|&&id| id == bos).count(), 1);
    }
}
