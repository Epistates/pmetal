//! `pmetal dataset analyze`, `validate`, `preview` and `download`: looking at
//! a dataset, and getting one.

use std::collections::BTreeMap;

use pmetal_core::Result as DataResult;
use pmetal_data::DatasetFormat;
use pmetal_data::rows::Row;

use super::common::{
    JsonlWriter, ModelText, banner, format_name, open_dataset, open_hub_dataset, pct, percentile,
    summary,
};

/// What a dataset's rows are like.
#[derive(Debug, Default)]
pub(super) struct Analysis {
    /// Rows read, and rows that could not be.
    pub(super) total: usize,
    pub(super) unreadable: usize,
    /// Rows of a layout no format reads.
    pub(super) unrecognised: usize,
    /// Rows with no text in them.
    pub(super) empty: usize,
    pub(super) formats: BTreeMap<&'static str, usize>,
    pub(super) char_lengths: Vec<usize>,
    /// What each row is in training, in tokens: with `model` only.
    pub(super) token_lengths: Vec<usize>,
    /// The first thing that went wrong with a row, as the dataset says it.
    pub(super) first_problem: Option<String>,
}

impl Analysis {
    fn note(&mut self, problem: String) {
        self.first_problem.get_or_insert(problem);
    }
}

/// Measure the rows: layouts, lengths in characters and, with a model, in the
/// tokens training would see.
pub(super) fn analyze_rows(
    rows: impl IntoIterator<Item = DataResult<Row>>,
    model: Option<&ModelText>,
) -> anyhow::Result<Analysis> {
    let template = model.map(|m| &m.template);
    let mut stats = Analysis::default();
    for item in rows {
        stats.total += 1;
        let row = match item {
            Ok(row) => row,
            Err(e) => {
                stats.unreadable += 1;
                stats.note(e.to_string());
                continue;
            }
        };
        let samples = DatasetFormat::detect(&row)
            .and_then(|format| format.render_row(&row, template).map(|s| (format, s)));
        let (format, samples) = match samples {
            Ok(read) => read,
            Err(e) => {
                stats.unrecognised += 1;
                *stats.formats.entry("unknown").or_default() += 1;
                stats.note(format!("row {}: {e}", stats.total));
                continue;
            }
        };
        *stats.formats.entry(format_name(&format)).or_default() += 1;
        let Some(sample) = samples.first().filter(|s| !s.text.trim().is_empty()) else {
            stats.empty += 1;
            continue;
        };
        stats.char_lengths.push(sample.text.chars().count());
        if let Some(model) = model {
            stats
                .token_lengths
                .push(sample.token_len(&model.tokenizer)?);
        }
    }
    Ok(stats)
}

fn print_lengths(unit: &str, sorted: &[usize], recommend: bool) {
    let mean = sorted.iter().sum::<usize>() as f64 / sorted.len() as f64;
    println!("{unit} lengths:");
    println!("  Min:    {}", sorted[0]);
    println!("  Max:    {}", sorted[sorted.len() - 1]);
    println!("  Mean:   {mean:.0}");
    println!("  Median: {}", percentile(sorted, 0.5));
    println!("  P90:    {}", percentile(sorted, 0.9));
    if recommend {
        println!(
            "  P95:    {} (recommended max_seq_len)",
            percentile(sorted, 0.95)
        );
    } else {
        println!("  P95:    {}", percentile(sorted, 0.95));
    }
    if recommend {
        println!("  P99:    {}", percentile(sorted, 0.99));
    }
    println!();
}

fn print_analysis(stats: &mut Analysis, detailed: bool) {
    println!("=== Dataset Statistics ===\n");
    println!("Total samples:    {}", stats.total);
    println!("Empty samples:    {}", stats.empty);
    println!(
        "Valid samples:    {}",
        stats.total - stats.empty - stats.unreadable - stats.unrecognised
    );
    if stats.unreadable > 0 {
        println!("Unreadable rows:  {}", stats.unreadable);
    }
    if stats.unrecognised > 0 {
        println!("Unrecognised:     {}", stats.unrecognised);
    }
    if let Some(problem) = &stats.first_problem {
        println!("First problem:    {problem}");
    }
    println!();

    println!("Detected formats:");
    for (format, count) in &stats.formats {
        println!("  {format}: {count} ({:.1}%)", pct(*count, stats.total));
    }
    println!();

    if !stats.char_lengths.is_empty() {
        stats.char_lengths.sort_unstable();
        print_lengths("Character", &stats.char_lengths, false);
    }
    if stats.token_lengths.is_empty() {
        return;
    }
    stats.token_lengths.sort_unstable();
    let tokens = &stats.token_lengths;
    print_lengths("Token", tokens, true);

    let buckets = [512, 1024, 2048, 4096, 8192, 16384, 32768];
    println!("Token length distribution:");
    for (i, &bucket) in buckets.iter().enumerate() {
        let lower = if i == 0 { 0 } else { buckets[i - 1] };
        let count = tokens.iter().filter(|&&l| l > lower && l <= bucket).count();
        let share = pct(count, tokens.len());
        println!(
            "  {lower:>5}-{bucket:<5}: {count:>5} ({share:5.1}%) {}",
            "█".repeat((share / 2.0) as usize)
        );
    }
    let last = buckets[buckets.len() - 1];
    let over = tokens.iter().filter(|&&l| l > last).count();
    if over > 0 {
        println!(
            "  >{last:5}:     {over:>5} ({:5.1}%)",
            pct(over, tokens.len())
        );
    }
    if detailed {
        println!("\n=== Sample Details ===");
        for (i, len) in tokens.iter().take(10).enumerate() {
            println!("  Sample {}: {len} tokens", i + 1);
        }
        if tokens.len() > 10 {
            println!("  ... and {} more samples", tokens.len() - 10);
        }
    }
}

pub(super) async fn analyze(
    path: String,
    model: Option<String>,
    detailed: bool,
) -> anyhow::Result<()> {
    banner("PMetal Dataset Analysis", &[("Path", path.clone())]);
    let model = match &model {
        Some(model) => Some(ModelText::load(model).await?),
        None => None,
    };
    let mut stats = analyze_rows(super::common::open_rows(&path)?, model.as_ref())?;
    print_analysis(&mut stats, detailed);
    Ok(())
}

/// What a check of a dataset found.
#[derive(Debug, Default)]
pub(super) struct Validation {
    pub(super) total: usize,
    pub(super) valid: usize,
    pub(super) too_long: usize,
    pub(super) empty: usize,
    pub(super) unrecognised: usize,
    pub(super) unreadable: usize,
    pub(super) issues: Vec<String>,
}

impl Validation {
    /// Whether the dataset can be trained on as it is (samples longer than
    /// the context are truncated, so they are a note, not a fault).
    pub(super) fn is_valid(&self) -> bool {
        self.total > 0 && self.empty == 0 && self.unrecognised == 0 && self.unreadable == 0
    }
}

/// Check every row: it reads, it has text, and, with a model, it fits
/// `max_seq_len` tokens.
pub(super) fn validate_rows(
    rows: impl IntoIterator<Item = DataResult<Row>>,
    model: Option<&ModelText>,
    max_seq_len: usize,
) -> anyhow::Result<Validation> {
    let template = model.map(|m| &m.template);
    let mut v = Validation::default();
    let mut number = 0usize;
    for item in rows {
        v.total += 1;
        let row = match item {
            Ok(row) => {
                number += 1;
                row
            }
            Err(e) => {
                v.unreadable += 1;
                v.issues.push(e.to_string());
                continue;
            }
        };
        let samples =
            DatasetFormat::detect(&row).and_then(|format| format.render_row(&row, template));
        let samples = match samples {
            Ok(samples) => samples,
            Err(e) => {
                v.unrecognised += 1;
                v.issues.push(format!("Row {number}: {e}"));
                continue;
            }
        };
        let Some(sample) = samples.first().filter(|s| !s.text.trim().is_empty()) else {
            v.empty += 1;
            v.issues.push(format!("Row {number}: no text"));
            continue;
        };
        if let Some(model) = model {
            let tokens = sample.token_len(&model.tokenizer)?;
            if tokens > max_seq_len {
                v.too_long += 1;
                v.issues.push(format!(
                    "Row {number}: {tokens} tokens, more than max_seq_len {max_seq_len}"
                ));
                continue;
            }
        }
        v.valid += 1;
    }
    Ok(v)
}

pub(super) async fn validate(
    path: String,
    model: Option<String>,
    max_seq_len: usize,
) -> anyhow::Result<()> {
    banner(
        "PMetal Dataset Validation",
        &[
            ("Path", path.clone()),
            ("Max Seq Len", max_seq_len.to_string()),
        ],
    );
    let model = match &model {
        Some(model) => Some(ModelText::load(model).await?),
        None => None,
    };
    let v = validate_rows(
        super::common::open_rows(&path)?,
        model.as_ref(),
        max_seq_len,
    )?;

    println!("=== Validation Results ===\n");
    println!("Total samples:   {}", v.total);
    println!(
        "Valid samples:   {} ({:.1}%)",
        v.valid,
        pct(v.valid, v.total)
    );
    println!("Unreadable:      {}", v.unreadable);
    println!("Unrecognised:    {}", v.unrecognised);
    println!("Empty samples:   {}", v.empty);
    if model.is_some() {
        println!("Too long:        {} (>{} tokens)", v.too_long, max_seq_len);
    } else {
        println!("Too long:        not checked (pass --model to measure tokens)");
    }
    println!();
    if !v.issues.is_empty() {
        println!("Issues (first 10):");
        for issue in v.issues.iter().take(10) {
            println!("  - {issue}");
        }
        if v.issues.len() > 10 {
            println!("  ... and {} more issues", v.issues.len() - 10);
        }
    }
    if !v.is_valid() {
        anyhow::bail!(
            "the dataset has issues that need attention ({} of {} rows are not usable)",
            v.total - v.valid - v.too_long,
            v.total
        );
    }
    println!("\nDataset is valid for training");
    if v.too_long > 0 {
        println!(
            "  Note: {} samples exceed max_seq_len and will be truncated",
            v.too_long
        );
    }
    Ok(())
}

pub(super) async fn preview(dataset_id: String, split: String, num: usize) -> anyhow::Result<()> {
    banner(
        "PMetal Dataset Preview",
        &[
            ("Dataset", dataset_id.clone()),
            ("Split", split.clone()),
            ("Samples", num.to_string()),
        ],
    );
    let rows = open_dataset(&dataset_id, &split, None).await?;
    let mut shown = 0usize;
    for row in rows.take(num) {
        let row = row?;
        if shown == 0 {
            let columns: Vec<&str> = row.keys().map(String::as_str).collect();
            println!("Columns: {columns:?}\n");
        }
        println!("--- Sample {} ---", shown + 1);
        println!("{}", serde_json::to_string_pretty(&row)?);
        println!();
        shown += 1;
    }
    println!("Showed {shown} sample(s)");
    Ok(())
}

pub(super) async fn download(
    dataset_id: String,
    split: String,
    output: Option<String>,
    revision: Option<String>,
) -> anyhow::Result<()> {
    banner(
        "PMetal Dataset Download",
        &[("Dataset", dataset_id.clone()), ("Split", split.clone())],
    );
    let output = output.unwrap_or_else(|| format!("{}.jsonl", dataset_id.replace('/', "_")));
    // The output is checked before anything is fetched.
    let mut out = JsonlWriter::create(&output, "dataset download output")?;
    let rows = open_hub_dataset(&dataset_id, &split, revision.as_deref()).await?;
    println!("Converting to JSONL: {output}");
    for row in rows {
        out.write(&row?)?;
    }
    let total = out.finish()?;
    summary(
        "Download Complete!",
        &[("Total samples", total.to_string()), ("Output", output)],
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_data::chat_templates::ChatTemplate;
    use pmetal_data::testing::word_level;
    use serde_json::json;

    fn rows(values: Vec<serde_json::Value>) -> Vec<DataResult<Row>> {
        values
            .into_iter()
            .map(|v| Ok(v.as_object().unwrap().clone()))
            .collect()
    }

    fn model() -> ModelText {
        let tokenizer = word_level(&["<unk>", "a", "b", "c"], &[("<s>", true), ("</s>", true)]);
        ModelText {
            tokenizer,
            template: ChatTemplate::chatml(),
            adds_bos: false,
        }
    }

    #[test]
    fn analysis_counts_every_layout_and_names_what_it_cannot_read() {
        let stats = analyze_rows(
            rows(vec![
                json!({"text": "a b c"}),
                json!({"instruction": "a", "output": "b"}),
                json!({"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]}),
                json!({"conversations": [{"from": "human", "value": "a"}, {"from": "gpt", "value": "b"}]}),
                json!({"text": "  "}),
                json!({"weird": 1}),
            ]),
            None,
        )
        .unwrap();
        assert_eq!(stats.total, 6);
        assert_eq!(stats.empty, 1);
        assert_eq!(stats.unrecognised, 1);
        assert_eq!(stats.formats["text"], 2);
        assert_eq!(stats.formats["alpaca"], 1);
        assert_eq!(stats.formats["messages"], 1);
        assert_eq!(stats.formats["sharegpt"], 1);
        assert_eq!(stats.formats["unknown"], 1);
        assert!(stats.first_problem.unwrap().contains("weird"));
        assert_eq!(stats.char_lengths.len(), 4);
        assert!(stats.token_lengths.is_empty());
    }

    /// A conversation is measured as the model's chat template writes it, the
    /// length it has in training, not as the bare words of its turns.
    #[test]
    fn a_conversation_is_measured_through_the_chat_template() {
        let model = model();
        let row = json!({"messages": [{"role": "user", "content": "a b"}, {"role": "assistant", "content": "c"}]});
        let stats = analyze_rows(rows(vec![row]), Some(&model)).unwrap();
        // The words of the turns are three tokens; the template's markers
        // (<|im_start|>, roles and <|im_end|> split into words) make it more.
        assert_eq!(stats.token_lengths.len(), 1);
        assert!(stats.token_lengths[0] > 3, "{:?}", stats.token_lengths);
    }

    #[test]
    fn validation_separates_rows_that_cannot_be_trained_from_those_that_are_long() {
        let model = model();
        let v = validate_rows(
            rows(vec![
                json!({"text": "a b"}),
                json!({"text": "a b c a b c a b c a b c"}),
                json!({"text": ""}),
                json!({"nothing": true}),
            ]),
            Some(&model),
            6,
        )
        .unwrap();
        assert_eq!(
            (v.total, v.valid, v.too_long, v.empty, v.unrecognised),
            (4, 1, 1, 1, 1)
        );
        assert!(!v.is_valid());
        assert!(v.issues.iter().any(|i| i.contains("more than max_seq_len")));

        // Long rows alone do not fail a dataset: they are truncated.
        let long_only = validate_rows(
            rows(vec![
                json!({"text": "a b"}),
                json!({"text": "a b c a b c a b c a b c"}),
            ]),
            Some(&model),
            6,
        )
        .unwrap();
        assert!(long_only.is_valid());
        assert!(!validate_rows(Vec::new(), None, 6).unwrap().is_valid());
    }

    /// A row that cannot be read is a finding, not the end of the check.
    #[test]
    fn an_unreadable_row_does_not_stop_the_check() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("d.jsonl");
        std::fs::write(
            &path,
            "{\"text\": \"ok\"}\nnot json\n{\"text\": \"fine\"}\n",
        )
        .unwrap();
        let reader = pmetal_data::rows::RowReader::open(&path).unwrap();
        let v = validate_rows(reader, None, 8).unwrap();
        assert_eq!((v.total, v.valid, v.unreadable), (3, 2, 1));
        assert!(v.issues[0].contains("d.jsonl:2"), "{:?}", v.issues);
        assert!(!v.is_valid());
    }
}
