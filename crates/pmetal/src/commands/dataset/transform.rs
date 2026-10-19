//! `pmetal dataset convert`, `filter`, `split`, `merge` and `sample`: moving
//! rows between datasets. Rows are carried as they are read, whatever file
//! they came from, and written as JSON Lines.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::io::Write;

use anyhow::Context;
use pmetal_core::Result as DataResult;
use pmetal_data::DatasetFormat;
use pmetal_data::rows::Row;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

use super::common::{
    JsonlWriter, ModelText, banner, open_rows, parse_column_map, pct, read_all, rename_columns,
    summary,
};

// -- convert ---------------------------------------------------------------------

/// Write `rows` to `out`, with the columns of `columns` (source to target)
/// renamed, shuffled with the seed when there is one.
pub(super) fn convert_rows<W: Write>(
    rows: impl IntoIterator<Item = DataResult<Row>>,
    columns: Option<&HashMap<String, String>>,
    shuffle_seed: Option<u64>,
    out: &mut JsonlWriter<W>,
) -> anyhow::Result<()> {
    let rename = |row: Row| match columns {
        Some(map) => rename_columns(row, map),
        None => row,
    };
    match shuffle_seed {
        None => {
            for row in rows {
                out.write(&rename(row?))?;
            }
        }
        Some(seed) => {
            let mut all = rows
                .into_iter()
                .map(|row| Ok(rename(row?)))
                .collect::<anyhow::Result<Vec<Row>>>()?;
            all.shuffle(&mut StdRng::seed_from_u64(seed));
            for row in &all {
                out.write(row)?;
            }
        }
    }
    Ok(())
}

pub(super) fn convert(
    input: String,
    output: String,
    columns: Option<String>,
    shuffle: bool,
    seed: u64,
) -> anyhow::Result<()> {
    banner(
        "PMetal Dataset Conversion",
        &[
            ("Input", input.clone()),
            ("Output", output.clone()),
            ("Shuffle", shuffle.to_string()),
        ],
    );
    let columns = columns.as_deref().map(parse_column_map).transpose()?;
    // The output is checked before the input is read.
    let mut out = JsonlWriter::create(&output, "dataset convert output")?;
    convert_rows(
        open_rows(&input)?,
        columns.as_ref(),
        shuffle.then_some(seed),
        &mut out,
    )?;
    let total = out.finish()?;
    summary(
        "Conversion Complete!",
        &[("Output", output), ("Samples", total.to_string())],
    );
    Ok(())
}

// -- filter ----------------------------------------------------------------------

/// Which rows a filter keeps.
#[derive(Default)]
pub(super) struct FilterOptions {
    pub(super) min_tokens: Option<usize>,
    pub(super) max_tokens: Option<usize>,
    pub(super) dedup: bool,
    pub(super) pattern: Option<regex::Regex>,
    pub(super) invert: bool,
    pub(super) complete_only: bool,
}

/// What a filter did.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct FilterReport {
    pub(super) total: usize,
    pub(super) kept: usize,
    pub(super) tokens: usize,
    pub(super) pattern: usize,
    pub(super) duplicate: usize,
    pub(super) incomplete: usize,
}

/// Whether a conversation row has both a user and an assistant turn. A row
/// that is not a conversation is complete.
fn is_complete(row: &Row) -> bool {
    let turns = ["messages", "conversations"]
        .iter()
        .find_map(|key| row.get(*key).and_then(|v| v.as_array()));
    let Some(turns) = turns else {
        return true;
    };
    let has = |roles: &[&str]| {
        turns.iter().any(|turn| {
            ["role", "from"]
                .iter()
                .find_map(|key| turn.get(*key).and_then(|v| v.as_str()))
                .is_some_and(|role| roles.contains(&role.to_ascii_lowercase().as_str()))
        })
    };
    has(&["user", "human"]) && has(&["assistant", "gpt", "model"])
}

/// Keep the rows that pass every filter in `options`, in order, and write them
/// to `out`. A row that no format reads is an error, as it is for training.
///
/// Token counts are what the row has in training (through the model's chat
/// template); the pattern and the duplicate check read the row's text, a
/// conversation as `Role: content` blocks.
pub(super) fn filter_rows<W: Write>(
    rows: impl IntoIterator<Item = DataResult<Row>>,
    options: &FilterOptions,
    model: Option<&ModelText>,
    out: &mut JsonlWriter<W>,
) -> anyhow::Result<FilterReport> {
    let wants_tokens = options.min_tokens.is_some() || options.max_tokens.is_some();
    let wants_text = options.pattern.is_some() || options.dedup;
    let mut seen: HashSet<u64> = HashSet::new();
    let mut report = FilterReport::default();

    for (index, row) in rows.into_iter().enumerate() {
        let row = row?;
        report.total += 1;
        let at = |e: pmetal_core::PMetalError| anyhow::anyhow!("row {}: {e}", index + 1);

        if options.complete_only && !is_complete(&row) {
            report.incomplete += 1;
            continue;
        }
        if wants_tokens {
            let model = model.context("token filters need a model")?;
            let format = DatasetFormat::detect(&row).map_err(at)?;
            let samples = format.render_row(&row, Some(&model.template)).map_err(at)?;
            let tokens = samples
                .first()
                .map_or(Ok(0), |s| s.token_len(&model.tokenizer))?;
            if options.min_tokens.is_some_and(|min| tokens < min)
                || options.max_tokens.is_some_and(|max| tokens > max)
            {
                report.tokens += 1;
                continue;
            }
        }
        if wants_text {
            let format = DatasetFormat::detect(&row).map_err(at)?;
            let samples = format.render_row(&row, None).map_err(at)?;
            let text = samples.first().map_or("", |s| s.text.as_str());
            if let Some(pattern) = &options.pattern {
                if pattern.is_match(text) == options.invert {
                    report.pattern += 1;
                    continue;
                }
            }
            if options.dedup {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                text.hash(&mut hasher);
                if !seen.insert(hasher.finish()) {
                    report.duplicate += 1;
                    continue;
                }
            }
        }
        out.write(&row)?;
        report.kept += 1;
    }
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn filter(
    input: String,
    output: String,
    model: Option<String>,
    min_tokens: Option<usize>,
    max_tokens: Option<usize>,
    dedup: bool,
    pattern: Option<String>,
    invert: bool,
    complete_only: bool,
) -> anyhow::Result<()> {
    let mut fields = vec![("Input", input.clone()), ("Output", output.clone())];
    if let Some(min) = min_tokens {
        fields.push(("Min tokens", min.to_string()));
    }
    if let Some(max) = max_tokens {
        fields.push(("Max tokens", max.to_string()));
    }
    if dedup {
        fields.push(("Deduplication", "enabled".into()));
    }
    if let Some(p) = &pattern {
        fields.push(("Pattern", format!("{p} (invert: {invert})")));
    }
    if complete_only {
        fields.push(("Complete only", "enabled".into()));
    }
    banner("PMetal Dataset Filter", &fields);

    let wants_tokens = min_tokens.is_some() || max_tokens.is_some();
    anyhow::ensure!(
        !wants_tokens || model.is_some(),
        "--model is required for --min-tokens and --max-tokens"
    );
    let pattern = pattern
        .as_deref()
        .map(regex::Regex::new)
        .transpose()
        .context("--pattern is not a valid regular expression")?;
    let model = match (&model, wants_tokens) {
        (Some(model), true) => Some(ModelText::load(model).await?),
        _ => None,
    };
    let options = FilterOptions {
        min_tokens,
        max_tokens,
        dedup,
        pattern,
        invert,
        complete_only,
    };
    let mut out = JsonlWriter::create(&output, "dataset filter output")?;
    let report = filter_rows(open_rows(&input)?, &options, model.as_ref(), &mut out)?;
    out.finish()?;

    let mut lines = vec![
        ("Total samples", report.total.to_string()),
        (
            "Kept samples",
            format!("{} ({:.1}%)", report.kept, pct(report.kept, report.total)),
        ),
    ];
    for (name, count) in [
        ("Filtered (tokens)", report.tokens),
        ("Filtered (pattern)", report.pattern),
        ("Filtered (dup)", report.duplicate),
        ("Filtered (incomplete)", report.incomplete),
    ] {
        if count > 0 {
            lines.push((name, count.to_string()));
        }
    }
    lines.push(("Output", output));
    summary("Filter Complete!", &lines);
    Ok(())
}

// -- split -----------------------------------------------------------------------

/// A dataset divided into its parts.
#[derive(Debug, Default)]
pub(super) struct Splits {
    pub(super) train: Vec<Row>,
    pub(super) val: Vec<Row>,
    pub(super) test: Vec<Row>,
}

/// Divide `rows` into train, validation and test sets of the given shares,
/// shuffled with `seed`. With `stratify`, each value of that column is divided
/// on its own, so every set holds the column's values in the proportions the
/// whole does (a value too rare to reach a share is rounded away from the
/// smaller sets).
pub(super) fn split_rows(
    rows: Vec<Row>,
    val_ratio: f64,
    test_ratio: f64,
    seed: u64,
    stratify: Option<&str>,
) -> anyhow::Result<Splits> {
    anyhow::ensure!(
        val_ratio >= 0.0 && test_ratio >= 0.0 && val_ratio + test_ratio < 1.0,
        "--val-ratio and --test-ratio must not be negative and must add up to less than 1.0 \
         (got {val_ratio} and {test_ratio})"
    );
    let groups: Vec<Vec<Row>> = match stratify {
        None => vec![rows],
        Some(column) => {
            if !rows.is_empty() && rows.iter().all(|r| !r.contains_key(column)) {
                let columns: Vec<&str> = rows[0].keys().map(String::as_str).collect();
                anyhow::bail!(
                    "--stratify: no row has a column '{column}'; the rows have {columns:?}"
                );
            }
            let mut by_value: BTreeMap<String, Vec<Row>> = BTreeMap::new();
            for row in rows {
                let key = row.get(column).map(|v| v.to_string()).unwrap_or_default();
                by_value.entry(key).or_default().push(row);
            }
            by_value.into_values().collect()
        }
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut splits = Splits::default();
    for mut group in groups {
        group.shuffle(&mut rng);
        let n = group.len();
        let test = (n as f64 * test_ratio).round() as usize;
        let val = (n as f64 * val_ratio).round() as usize;
        let mut rows = group.into_iter();
        splits.train.extend(rows.by_ref().take(n - test - val));
        splits.val.extend(rows.by_ref().take(val));
        splits.test.extend(rows);
    }
    if stratify.is_some() {
        // The values were divided one after another: mix them.
        splits.train.shuffle(&mut rng);
        splits.val.shuffle(&mut rng);
        splits.test.shuffle(&mut rng);
    }
    Ok(splits)
}

pub(super) fn split(
    input: String,
    output_dir: String,
    val_ratio: f64,
    test_ratio: f64,
    seed: u64,
    stratify: Option<String>,
) -> anyhow::Result<()> {
    let mut fields = vec![
        ("Input", input.clone()),
        ("Output", output_dir.clone()),
        ("Val ratio", format!("{val_ratio:.2}")),
        ("Test ratio", format!("{test_ratio:.2}")),
        ("Seed", seed.to_string()),
    ];
    if let Some(column) = &stratify {
        fields.push(("Stratify", column.clone()));
    }
    banner("PMetal Dataset Split", &fields);

    let rows = read_all(&input)?;
    println!("Loaded {} samples", rows.len());
    let splits = split_rows(rows, val_ratio, test_ratio, seed, stratify.as_deref())?;

    for (label, name, rows) in [
        ("Train", "train", &splits.train),
        ("Val", "val", &splits.val),
        ("Test", "test", &splits.test),
    ] {
        // No test file unless a test set was asked for.
        if name == "test" && rows.is_empty() {
            continue;
        }
        let path = format!("{}/{name}.jsonl", output_dir.trim_end_matches('/'));
        let mut out = JsonlWriter::create(&path, "dataset split output")?;
        for row in rows {
            out.write(row)?;
        }
        out.finish()?;
        println!("{:<7}{} samples -> {path}", format!("{label}:"), rows.len());
    }
    summary("Split Complete!", &[]);
    Ok(())
}

// -- merge -----------------------------------------------------------------------

/// `--weights`: one non-negative number per input, 1.0 each without it.
pub(super) fn parse_weights(spec: Option<&str>, inputs: usize) -> anyhow::Result<Vec<f64>> {
    let Some(spec) = spec else {
        return Ok(vec![1.0; inputs]);
    };
    let weights = spec
        .split(',')
        .map(|w| {
            let value: f64 = w
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("--weights: '{}' is not a number", w.trim()))?;
            anyhow::ensure!(
                value.is_finite() && value >= 0.0,
                "--weights: '{}' is not a non-negative number",
                w.trim()
            );
            Ok(value)
        })
        .collect::<anyhow::Result<Vec<f64>>>()?;
    anyhow::ensure!(
        weights.len() == inputs,
        "--weights gives {} weight(s) for {inputs} input(s); give one per input",
        weights.len()
    );
    Ok(weights)
}

/// `rows` taken `weight` times: the whole number of times in full, the
/// fraction of the rows (a random choice) once more.
fn weighted(rows: Vec<Row>, weight: f64, rng: &mut StdRng) -> Vec<Row> {
    let copies = weight.floor() as usize;
    let fraction = weight - copies as f64;
    let mut out = Vec::with_capacity(rows.len() * copies);
    for _ in 0..copies {
        out.extend(rows.iter().cloned());
    }
    if fraction > 0.0 {
        out.extend(
            rows.iter()
                .filter(|_| rng.random::<f64>() < fraction)
                .cloned(),
        );
    }
    out
}

/// Put the datasets together, each taken as often as its weight says (see
/// [`weighted`]): one after another, or, with `interleave`, a row of each in
/// turn. The same seed makes the same dataset.
pub(super) fn merge_datasets(
    datasets: Vec<Vec<Row>>,
    weights: &[f64],
    interleave: bool,
    shuffle: bool,
    seed: u64,
) -> Vec<Row> {
    let lists: Vec<Vec<Row>> = datasets
        .into_iter()
        .zip(weights)
        .enumerate()
        .map(|(i, (rows, &weight))| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
            weighted(rows, weight, &mut rng)
        })
        .collect();

    let mut merged: Vec<Row> = Vec::new();
    if interleave {
        let mut turns: Vec<_> = lists.into_iter().map(Vec::into_iter).collect();
        loop {
            let before = merged.len();
            for list in &mut turns {
                merged.extend(list.next());
            }
            if merged.len() == before {
                break;
            }
        }
    } else {
        merged.extend(lists.into_iter().flatten());
    }
    if shuffle {
        merged.shuffle(&mut StdRng::seed_from_u64(seed));
    }
    merged
}

pub(super) fn merge(
    inputs: Vec<String>,
    output: String,
    shuffle: bool,
    seed: u64,
    interleave: bool,
    weights: Option<String>,
) -> anyhow::Result<()> {
    let mut fields: Vec<(&str, String)> = Vec::new();
    for input in &inputs {
        fields.push(("Input", input.clone()));
    }
    fields.push(("Output", output.clone()));
    fields.push(("Shuffle", shuffle.to_string()));
    fields.push(("Interleave", interleave.to_string()));
    banner("PMetal Dataset Merge", &fields);

    let weights = parse_weights(weights.as_deref(), inputs.len())?;
    let mut out = JsonlWriter::create(&output, "dataset merge output")?;
    let mut datasets = Vec::with_capacity(inputs.len());
    for input in &inputs {
        let rows = read_all(input)?;
        println!("Loaded {} samples from {input}", rows.len());
        datasets.push(rows);
    }
    for row in merge_datasets(datasets, &weights, interleave, shuffle, seed) {
        out.write(&row)?;
    }
    let total = out.finish()?;
    summary(
        "Merge Complete!",
        &[("Total samples", total.to_string()), ("Output", output)],
    );
    Ok(())
}

// -- sample ----------------------------------------------------------------------

/// `num` rows drawn at random from `rows` (all of them if there are fewer),
/// in random order, and how many rows there were. Holds only the rows drawn.
pub(super) fn sample_rows(
    rows: impl IntoIterator<Item = DataResult<Row>>,
    num: usize,
    seed: u64,
) -> anyhow::Result<(Vec<Row>, usize)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut reservoir: Vec<Row> = Vec::with_capacity(num.min(1 << 16));
    let mut seen = 0usize;
    for row in rows {
        let row = row?;
        if reservoir.len() < num {
            reservoir.push(row);
        } else {
            let slot = rng.random_range(0..=seen);
            if slot < num {
                reservoir[slot] = row;
            }
        }
        seen += 1;
    }
    reservoir.shuffle(&mut rng);
    Ok((reservoir, seen))
}

pub(super) fn sample(input: String, output: String, num: usize, seed: u64) -> anyhow::Result<()> {
    banner(
        "PMetal Dataset Sample",
        &[
            ("Input", input.clone()),
            ("Output", output.clone()),
            ("Samples", num.to_string()),
            ("Seed", seed.to_string()),
        ],
    );
    let mut out = JsonlWriter::create(&output, "dataset sample output")?;
    let (rows, seen) = sample_rows(open_rows(&input)?, num, seed)?;
    for row in &rows {
        out.write(row)?;
    }
    out.finish()?;
    summary(
        "Sample Complete!",
        &[
            ("Sampled", format!("{} of {seen} samples", rows.len())),
            ("Output", output),
        ],
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn row(value: serde_json::Value) -> Row {
        value.as_object().unwrap().clone()
    }

    fn ok(rows: &[Row]) -> Vec<DataResult<Row>> {
        rows.iter().cloned().map(Ok).collect()
    }

    fn numbered(n: usize) -> Vec<Row> {
        (0..n)
            .map(|i| row(json!({"text": format!("row {i}"), "i": i})))
            .collect()
    }

    fn written(f: impl FnOnce(&mut JsonlWriter<Vec<u8>>)) -> Vec<Row> {
        let mut out = JsonlWriter::new(Vec::new());
        f(&mut out);
        String::from_utf8(out.into_inner().unwrap())
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn convert_renames_and_shuffles_reproducibly() {
        let rows = numbered(20);
        let map = parse_column_map("body=text").unwrap();
        let once = written(|out| {
            convert_rows(ok(&rows), Some(&map), Some(7), out).unwrap();
        });
        let again = written(|out| {
            convert_rows(ok(&rows), Some(&map), Some(7), out).unwrap();
        });
        assert_eq!(once, again);
        assert_eq!(once.len(), 20);
        assert!(
            once.iter()
                .all(|r| r.contains_key("body") && !r.contains_key("text"))
        );
        assert_ne!(
            once.iter().map(|r| r["i"].clone()).collect::<Vec<_>>(),
            (0..20).map(|i| json!(i)).collect::<Vec<_>>()
        );
        // Without a seed the order is the input's.
        let plain = written(|out| {
            convert_rows(ok(&rows), None, None, out).unwrap();
        });
        assert_eq!(plain, rows);
    }

    #[test]
    fn filter_keeps_the_rows_every_filter_keeps() {
        let rows = vec![
            row(json!({"text": "keep this one"})),
            row(json!({"text": "keep this one"})),
            row(json!({"text": "drop SPAM here"})),
            row(json!({"messages": [{"role": "user", "content": "hi"}]})),
            row(
                json!({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}),
            ),
        ];
        let run = |options: FilterOptions| {
            let mut out = JsonlWriter::new(Vec::new());
            let report = filter_rows(ok(&rows), &options, None, &mut out).unwrap();
            (report, out.finish().unwrap())
        };
        let (report, kept) = run(FilterOptions {
            dedup: true,
            ..Default::default()
        });
        assert_eq!((report.duplicate, kept), (1, 4));

        let (report, kept) = run(FilterOptions {
            pattern: Some(regex::Regex::new("SPAM").unwrap()),
            invert: true,
            ..Default::default()
        });
        assert_eq!((report.pattern, kept), (1, 4));

        let (report, kept) = run(FilterOptions {
            complete_only: true,
            ..Default::default()
        });
        assert_eq!((report.incomplete, kept), (1, 4));
        assert_eq!(report.total, 5);
    }

    #[test]
    fn filter_counts_tokens_as_training_does() {
        use pmetal_data::chat_templates::ChatTemplate;
        let model = ModelText {
            tokenizer: pmetal_data::testing::word_level(
                &["<unk>", "a", "b"],
                &[("<s>", true), ("</s>", true)],
            ),
            template: ChatTemplate::chatml(),
            adds_bos: false,
        };
        let rows = vec![
            row(json!({"text": "a"})),
            row(json!({"text": "a b a b a b a b"})),
        ];
        let mut out = JsonlWriter::new(Vec::new());
        let report = filter_rows(
            ok(&rows),
            &FilterOptions {
                max_tokens: Some(4),
                ..Default::default()
            },
            Some(&model),
            &mut out,
        )
        .unwrap();
        assert_eq!((report.kept, report.tokens), (1, 1));
    }

    #[test]
    fn a_row_no_format_reads_stops_a_filter_and_names_itself() {
        let rows = vec![row(json!({"text": "fine"})), row(json!({"weird": 1}))];
        let mut out = JsonlWriter::new(Vec::new());
        let err = filter_rows(
            ok(&rows),
            &FilterOptions {
                dedup: true,
                ..Default::default()
            },
            None,
            &mut out,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("row 2") && err.contains("weird"), "{err}");
    }

    #[test]
    fn a_split_follows_its_ratios_and_its_seed() {
        let a = split_rows(numbered(100), 0.1, 0.2, 3, None).unwrap();
        let b = split_rows(numbered(100), 0.1, 0.2, 3, None).unwrap();
        assert_eq!((a.train.len(), a.val.len(), a.test.len()), (70, 10, 20));
        assert_eq!(a.train, b.train);
        let c = split_rows(numbered(100), 0.1, 0.2, 4, None).unwrap();
        assert_ne!(a.train, c.train);
        // Nothing is lost or repeated.
        let mut all: Vec<_> = [&a.train, &a.val, &a.test]
            .iter()
            .flat_map(|s| s.iter().map(|r| r["i"].as_u64().unwrap()))
            .collect();
        all.sort_unstable();
        assert_eq!(all, (0..100).collect::<Vec<_>>());

        assert!(split_rows(numbered(10), 0.6, 0.4, 1, None).is_err());
        assert!(split_rows(numbered(10), -0.1, 0.0, 1, None).is_err());
    }

    /// `--stratify` was accepted and ignored.
    #[test]
    fn a_stratified_split_keeps_the_proportions_of_a_column() {
        let rows: Vec<Row> = (0..100)
            .map(|i| row(json!({"text": "x", "kind": if i < 80 { "common" } else { "rare" }})))
            .collect();
        let splits = split_rows(rows, 0.25, 0.0, 9, Some("kind")).unwrap();
        let count = |set: &[Row], kind: &str| set.iter().filter(|r| r["kind"] == kind).count();
        assert_eq!(
            (count(&splits.val, "common"), count(&splits.val, "rare")),
            (20, 5)
        );
        assert_eq!(
            (count(&splits.train, "common"), count(&splits.train, "rare")),
            (60, 15)
        );
        assert!(splits.test.is_empty());

        let err = split_rows(numbered(4), 0.5, 0.0, 1, Some("nope"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no row has a column 'nope'"), "{err}");
    }

    #[test]
    fn weights_are_one_number_per_input() {
        assert_eq!(parse_weights(None, 3).unwrap(), [1.0, 1.0, 1.0]);
        assert_eq!(parse_weights(Some("1, 2.5"), 2).unwrap(), [1.0, 2.5]);
        for bad in ["1,x", "1", "1,-2", "1,inf"] {
            assert!(parse_weights(Some(bad), 2).is_err(), "{bad}");
        }
    }

    #[test]
    fn a_weight_repeats_or_thins_a_dataset_and_a_seed_makes_it_repeatable() {
        let a = numbered(10);
        let b: Vec<Row> = (0..10)
            .map(|i| row(json!({"text": "b", "i": 100 + i})))
            .collect();
        let merged = merge_datasets(vec![a.clone(), b.clone()], &[2.0, 0.5], false, false, 1);
        let from_a = merged
            .iter()
            .filter(|r| r["i"].as_u64().unwrap() < 100)
            .count();
        let from_b = merged.len() - from_a;
        assert_eq!(from_a, 20);
        assert!(from_b > 0 && from_b < 10, "{from_b}");
        // The same seed, the same dataset: in both modes.
        for interleave in [false, true] {
            let once = merge_datasets(vec![a.clone(), b.clone()], &[1.0, 0.5], interleave, true, 5);
            let again =
                merge_datasets(vec![a.clone(), b.clone()], &[1.0, 0.5], interleave, true, 5);
            assert_eq!(once, again);
        }
        // Interleaved, the datasets alternate.
        let woven = merge_datasets(vec![a, b], &[1.0, 1.0], true, false, 1);
        let origin: Vec<bool> = woven
            .iter()
            .map(|r| r["i"].as_u64().unwrap() < 100)
            .collect();
        assert_eq!(&origin[..4], [true, false, true, false]);
        assert_eq!(woven.len(), 20);
    }

    #[test]
    fn a_sample_is_the_size_asked_for_and_repeatable() {
        let rows = numbered(1000);
        let (first, seen) = sample_rows(ok(&rows), 25, 11).unwrap();
        let (again, _) = sample_rows(ok(&rows), 25, 11).unwrap();
        assert_eq!((first.len(), seen), (25, 1000));
        assert_eq!(first, again);
        let distinct: HashSet<u64> = first.iter().map(|r| r["i"].as_u64().unwrap()).collect();
        assert_eq!(distinct.len(), 25);
        let (other, _) = sample_rows(ok(&rows), 25, 12).unwrap();
        assert_ne!(first, other);
        // Asking for more than there is gives what there is.
        let (all, _) = sample_rows(ok(&numbered(5)), 25, 1).unwrap();
        assert_eq!(all.len(), 5);
    }

    /// A Parquet column that is not a string was written as `null`, so a
    /// converted dataset lost its labels, scores and message lists.
    #[test]
    fn converting_parquet_keeps_columns_that_are_not_strings() {
        use arrow::array::{
            ArrayRef, Float64Array, Int64Array, ListBuilder, StringArray, StringBuilder,
        };
        use arrow::datatypes::{Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::arrow_writer::ArrowWriter;
        use std::sync::Arc;

        let mut tags = ListBuilder::new(StringBuilder::new());
        for row_tags in [["x", "y"].as_slice(), [].as_slice()] {
            for t in row_tags {
                tags.values().append_value(t);
            }
            tags.append(true);
        }
        let tags: ArrayRef = Arc::new(tags.finish());
        let columns: Vec<(&str, ArrayRef)> = vec![
            ("text", Arc::new(StringArray::from(vec!["one", "two"]))),
            ("label", Arc::new(Int64Array::from(vec![3, 4]))),
            ("score", Arc::new(Float64Array::from(vec![0.5, 1.5]))),
            ("tags", tags),
        ];
        let schema = Arc::new(Schema::new(
            columns
                .iter()
                .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
                .collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(
            schema.clone(),
            columns.into_iter().map(|(_, a)| a).collect(),
        )
        .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.parquet");
        let mut writer =
            ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let rows = written(|out| {
            convert_rows(
                pmetal_data::rows::RowReader::open(&path).unwrap(),
                None,
                None,
                out,
            )
            .unwrap();
        });
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["text"], "one");
        assert_eq!(rows[0]["label"], 3);
        assert_eq!(rows[0]["score"], 0.5);
        assert_eq!(rows[0]["tags"], json!(["x", "y"]));
        assert_eq!(rows[1]["tags"], json!([]));
    }
}
