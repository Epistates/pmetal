//! Running a task: ask the model, score what it says.

use std::collections::BTreeMap;

use pmetal_data::Tokenizer;
use serde_json::{Value, json};

use super::gsm8k;
use super::model::{EvalModel, LlRequest};
use super::stats::Metric;
use super::tasks::{self, GenItem, GenScorer, Items, McItem, McMetric};
use super::{EvalError, GroupReport, Task, TaskData, TaskOptions, TaskReport};

/// Items whose requests go to the model together: what bounds the scoring
/// requests held at once, and how often progress is reported.
const ITEMS_PER_CALL: usize = 8;
/// Tokens of the tail of a generation checked for a stop string.
const STOP_WINDOW: usize = 32;

fn tokenizer_error(e: impl std::fmt::Display) -> EvalError {
    EvalError::Model(format!("the tokenizer failed: {e}"))
}

/// A context and a continuation as token ids.
///
/// Spaces at the end of the context move to the front of the continuation (a
/// token's leading space belongs to it), then the continuation's tokens are
/// what the whole text has beyond the context's, so a tokenizer that merges
/// across the boundary is still scored on the text as it is tokenized whole.
/// The tokenizer's own start-of-text token is part of the context.
///
/// # Errors
///
/// [`EvalError::Model`] if the tokenizer fails, or the continuation has no
/// token of its own.
pub fn encode_pair(
    tokenizer: &Tokenizer,
    context: &str,
    continuation: &str,
) -> Result<(Vec<u32>, Vec<u32>), EvalError> {
    let trimmed = context.trim_end_matches(' ');
    let moved = &context[trimmed.len()..];
    let continuation = format!("{moved}{continuation}");
    let whole = tokenizer
        .encode_with_special_tokens(&format!("{trimmed}{continuation}"))
        .map_err(tokenizer_error)?;
    let context = tokenizer
        .encode_with_special_tokens(trimmed)
        .map_err(tokenizer_error)?;
    if context.len() >= whole.len() {
        return Err(EvalError::Model(format!(
            "the continuation {continuation:?} has no tokens of its own after the context"
        )));
    }
    let continuation = whole[context.len()..].to_vec();
    Ok((context, continuation))
}

/// The index of the largest value; the first among equals.
fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

/// What one multiple-choice item scored.
struct Scored {
    lls: Vec<f64>,
    greedy: Vec<bool>,
    acc: bool,
    acc_norm: bool,
}

fn score_multiple_choice(
    model: &mut dyn EvalModel,
    tokenizer: &Tokenizer,
    items: &[McItem],
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<Scored>, EvalError> {
    let mut scored = Vec::with_capacity(items.len());
    for (call, chunk) in items.chunks(ITEMS_PER_CALL).enumerate() {
        let mut requests = Vec::new();
        for item in chunk {
            for choice in &item.choices {
                let (context, continuation) =
                    encode_pair(tokenizer, &item.context, &format!(" {choice}"))?;
                requests.push(LlRequest {
                    context,
                    continuation,
                });
            }
        }
        let results = model.loglikelihood(&requests)?;
        if results.len() != requests.len() {
            return Err(EvalError::Model(format!(
                "the model scored {} of {} requests",
                results.len(),
                requests.len()
            )));
        }
        let mut next = results.into_iter();
        for item in chunk {
            let outputs: Vec<_> = next.by_ref().take(item.choices.len()).collect();
            let lls: Vec<f64> = outputs.iter().map(|o| o.logprob).collect();
            // Normalised by the choice's length in characters, not tokens, so the
            // measure does not depend on the tokenizer.
            let normalized: Vec<f64> = lls
                .iter()
                .zip(&item.choices)
                .map(|(ll, choice)| ll / choice.chars().count().max(1) as f64)
                .collect();
            scored.push(Scored {
                acc: argmax(&lls) == item.gold,
                acc_norm: argmax(&normalized) == item.gold,
                greedy: outputs.iter().map(|o| o.greedy).collect(),
                lls,
            });
        }
        progress((call + 1) * ITEMS_PER_CALL.min(items.len()), items.len());
    }
    Ok(scored)
}

/// The metrics of the multiple-choice items at `indices`.
fn mc_metrics(
    scored: &[Scored],
    indices: &[usize],
    metrics: &[McMetric],
) -> BTreeMap<String, Metric> {
    let values = |pick: fn(&Scored) -> bool| -> Vec<f64> {
        indices
            .iter()
            .map(|&i| f64::from(u8::from(pick(&scored[i]))))
            .collect()
    };
    let mut out = BTreeMap::new();
    for metric in metrics {
        let (name, pick): (&str, fn(&Scored) -> bool) = match metric {
            McMetric::Acc => ("acc", |s| s.acc),
            McMetric::AccNorm => ("acc_norm", |s| s.acc_norm),
        };
        out.insert(name.to_string(), Metric::of(&values(pick)));
    }
    out
}

/// What one generation item scored: per-metric values, and the instructions'
/// own verdicts for IFEval.
struct Generated {
    response: String,
    values: BTreeMap<&'static str, f64>,
    /// Per-instruction verdicts (strict, loose), for a task scored by them.
    instructions: Option<(Vec<bool>, Vec<bool>)>,
    skipped: bool,
}

/// Whether the text generated so far ends in a stop string.
fn stopped(tokenizer: &Tokenizer, generated: &[u32], stops: &[&str]) -> bool {
    if stops.is_empty() {
        return false;
    }
    let tail = &generated[generated.len().saturating_sub(STOP_WINDOW)..];
    tokenizer
        .decode(tail)
        .is_ok_and(|text| stops.iter().any(|stop| text.contains(stop)))
}

/// A custom answer compared without the differences that are not the
/// answer's: case, surrounding space and quotes, runs of whitespace, and
/// closing punctuation.
fn normalize_answer(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed
        .trim_matches(|c: char| matches!(c, '"' | '\'' | '`'))
        .trim_end_matches(['.', ',', ';', ':', '!', '?'])
        .trim()
        .to_lowercase()
}

fn score_generation(
    model: &mut dyn EvalModel,
    tokenizer: &Tokenizer,
    items: &[GenItem],
    options: &TaskOptions,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Vec<Generated>, EvalError> {
    let mut out = Vec::with_capacity(items.len());
    for (done, item) in items.iter().enumerate() {
        if let GenScorer::IfEval(prompt) = &item.scorer {
            if !prompt.supported() {
                out.push(Generated {
                    response: String::new(),
                    values: BTreeMap::new(),
                    instructions: None,
                    skipped: true,
                });
                progress(done + 1, items.len());
                continue;
            }
        }
        let prompt_ids = tokenizer
            .encode_with_special_tokens(&item.prompt)
            .map_err(tokenizer_error)?;
        let max_tokens = item.max_tokens.unwrap_or(options.max_gen_tokens);
        let stops = item.stop.clone();
        let generated = model.generate(&prompt_ids, max_tokens, &mut |so_far| {
            stopped(tokenizer, so_far, &stops)
        })?;
        let text = tokenizer.decode(&generated).map_err(tokenizer_error)?;
        let response = gsm8k::truncate_at(&text, &item.stop).to_string();

        let mut values = BTreeMap::new();
        let mut instructions = None;
        match &item.scorer {
            GenScorer::Gsm8k { reference } => {
                let strict = gsm8k::exact_match(&gsm8k::strict_answer(&response), reference);
                let flexible = gsm8k::exact_match(&gsm8k::flexible_answer(&response), reference);
                values.insert("exact_match,strict-match", f64::from(u8::from(strict)));
                values.insert(
                    "exact_match,flexible-extract",
                    f64::from(u8::from(flexible)),
                );
            }
            GenScorer::IfEval(prompt) => {
                let verdict = prompt.score(&item.prompt, &response);
                let all = |flags: &[bool]| f64::from(u8::from(flags.iter().all(|&f| f)));
                values.insert("prompt_level_strict_acc", all(&verdict.strict));
                values.insert("prompt_level_loose_acc", all(&verdict.loose));
                instructions = Some((verdict.strict, verdict.loose));
            }
            GenScorer::Exact { answers } => {
                let given = normalize_answer(&response);
                let right = answers.iter().any(|a| normalize_answer(a) == given);
                values.insert("exact_match", f64::from(u8::from(right)));
            }
        }
        out.push(Generated {
            response,
            values,
            instructions,
            skipped: false,
        });
        progress(done + 1, items.len());
    }
    Ok(out)
}

/// The metrics of the generated items at `indices`.
fn gen_metrics(generated: &[Generated], indices: &[usize]) -> BTreeMap<String, Metric> {
    let scored: Vec<&Generated> = indices
        .iter()
        .map(|&i| &generated[i])
        .filter(|g| !g.skipped)
        .collect();
    let mut names: Vec<&'static str> = scored
        .iter()
        .flat_map(|g| g.values.keys().copied())
        .collect();
    names.sort_unstable();
    names.dedup();
    let mut out = BTreeMap::new();
    for name in names {
        let values: Vec<f64> = scored
            .iter()
            .filter_map(|g| g.values.get(name).copied())
            .collect();
        out.insert(name.to_string(), Metric::of(&values));
    }
    // IFEval's instruction-level scores count instructions, not prompts.
    type Flags = (Vec<bool>, Vec<bool>);
    let flatten = |pick: fn(&Flags) -> &Vec<bool>| -> Vec<f64> {
        scored
            .iter()
            .filter_map(|g| g.instructions.as_ref())
            .flat_map(|v| pick(v).iter().map(|&f| f64::from(u8::from(f))))
            .collect()
    };
    if scored.iter().any(|g| g.instructions.is_some()) {
        out.insert(
            "inst_level_strict_acc".into(),
            Metric::of(&flatten(|v| &v.0)),
        );
        out.insert(
            "inst_level_loose_acc".into(),
            Metric::of(&flatten(|v| &v.1)),
        );
    }
    out
}

/// The indices of the items of each subject, for the tasks that have them.
fn groups_of(subjects: &[Option<&str>]) -> BTreeMap<String, Vec<usize>> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, subject) in subjects.iter().enumerate() {
        if let Some(subject) = subject {
            groups.entry((*subject).to_string()).or_default().push(i);
        }
    }
    groups
}

/// Run `task` over `data` and report what the model scores.
///
/// `progress` is called with `(items done, items in all)` as the run goes.
///
/// # Errors
///
/// [`EvalError::Data`] for rows that are not the task's layout,
/// [`EvalError::Config`] for options the task cannot use, and
/// [`EvalError::Model`] if the model or the tokenizer fails.
pub fn evaluate(
    model: &mut dyn EvalModel,
    tokenizer: &Tokenizer,
    task: Task,
    data: &TaskData,
    options: &TaskOptions,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<TaskReport, EvalError> {
    let fewshot = options.fewshot.unwrap_or_else(|| task.default_fewshot());
    let items = tasks::build(task, data, options, fewshot)?;
    let mut report = TaskReport {
        task: task.name().to_string(),
        fewshot,
        samples: 0,
        skipped: 0,
        metrics: BTreeMap::new(),
        groups: BTreeMap::new(),
        samples_log: Vec::new(),
    };
    match items {
        Items::MultipleChoice(items, metrics) => {
            if items.is_empty() {
                return Err(EvalError::Data("the task has no items to score".into()));
            }
            let scored = score_multiple_choice(model, tokenizer, &items, progress)?;
            let all: Vec<usize> = (0..items.len()).collect();
            report.samples = items.len();
            report.metrics = mc_metrics(&scored, &all, metrics);
            let subjects: Vec<Option<&str>> = items.iter().map(|i| i.subject.as_deref()).collect();
            for (subject, indices) in groups_of(&subjects) {
                report.groups.insert(
                    subject,
                    GroupReport {
                        samples: indices.len(),
                        metrics: mc_metrics(&scored, &indices, metrics),
                    },
                );
            }
            if options.log_samples {
                report.samples_log = items
                    .iter()
                    .zip(&scored)
                    .enumerate()
                    .map(|(i, (item, s))| {
                        json!({
                            "index": i,
                            "subject": item.subject,
                            "prompt": item.context,
                            "choices": item.choices,
                            "gold": item.gold,
                            "loglikelihoods": s.lls,
                            "is_greedy": s.greedy,
                            "prediction": argmax(&s.lls),
                            "acc": s.acc,
                            "acc_norm": s.acc_norm,
                        })
                    })
                    .collect();
            }
        }
        Items::Generate(items) => {
            if items.is_empty() {
                return Err(EvalError::Data("the task has no items to score".into()));
            }
            let generated = score_generation(model, tokenizer, &items, options, progress)?;
            let all: Vec<usize> = (0..items.len()).collect();
            report.skipped = generated.iter().filter(|g| g.skipped).count();
            report.samples = items.len() - report.skipped;
            if report.samples == 0 {
                return Err(EvalError::Data(format!(
                    "every item of {task} uses an instruction this build cannot check"
                )));
            }
            report.metrics = gen_metrics(&generated, &all);
            let subjects: Vec<Option<&str>> = items.iter().map(|i| i.subject.as_deref()).collect();
            for (subject, indices) in groups_of(&subjects) {
                report.groups.insert(
                    subject,
                    GroupReport {
                        samples: indices.len(),
                        metrics: gen_metrics(&generated, &indices),
                    },
                );
            }
            if options.log_samples {
                report.samples_log = items
                    .iter()
                    .zip(&generated)
                    .enumerate()
                    .map(|(i, (item, g))| {
                        let mut entry = json!({
                            "index": i,
                            "subject": item.subject,
                            "prompt": item.prompt,
                            "response": g.response,
                            "skipped": g.skipped,
                        });
                        for (name, value) in &g.values {
                            entry[*name] = Value::from(*value);
                        }
                        if let Some((strict, loose)) = &g.instructions {
                            entry["instructions_strict"] = json!(strict);
                            entry["instructions_loose"] = json!(loose);
                        }
                        if let GenScorer::Gsm8k { reference } = &item.scorer {
                            entry["reference"] = json!(reference);
                            entry["strict_answer"] = json!(gsm8k::strict_answer(&g.response));
                            entry["flexible_answer"] = json!(gsm8k::flexible_answer(&g.response));
                        }
                        if let GenScorer::Exact { answers } = &item.scorer {
                            entry["answers"] = json!(answers);
                        }
                        entry
                    })
                    .collect();
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::super::model::LlResult;
    use super::*;
    use pmetal_data::tokenizer::testing::word_level;
    use serde_json::json;
    use std::collections::VecDeque;

    /// A model that answers from a script: a table of log-likelihoods by the
    /// continuation's first token, and generations in order.
    struct Scripted {
        scores: Box<dyn FnMut(&LlRequest) -> f64>,
        generations: VecDeque<Vec<u32>>,
        asked: Vec<LlRequest>,
        generated_for: Vec<Vec<u32>>,
    }

    impl Scripted {
        fn new(scores: impl FnMut(&LlRequest) -> f64 + 'static) -> Self {
            Self {
                scores: Box::new(scores),
                generations: VecDeque::new(),
                asked: Vec::new(),
                generated_for: Vec::new(),
            }
        }
    }

    impl EvalModel for Scripted {
        fn loglikelihood(&mut self, requests: &[LlRequest]) -> Result<Vec<LlResult>, EvalError> {
            self.asked.extend(requests.iter().cloned());
            Ok(requests
                .iter()
                .map(|r| LlResult {
                    logprob: (self.scores)(r),
                    greedy: false,
                })
                .collect())
        }

        fn generate(
            &mut self,
            prompt: &[u32],
            max_new_tokens: usize,
            stop: &mut dyn FnMut(&[u32]) -> bool,
        ) -> Result<Vec<u32>, EvalError> {
            self.generated_for.push(prompt.to_vec());
            let script = self.generations.pop_front().expect("a scripted generation");
            let mut out = Vec::new();
            for token in script.into_iter().take(max_new_tokens) {
                out.push(token);
                if stop(&out) {
                    break;
                }
            }
            Ok(out)
        }
    }

    const WORDS: [&str; 14] = [
        "[UNK]",
        "ab",
        "abcdefgh",
        "Question:",
        "Answer:",
        "5",
        "6",
        "####",
        "apples.",
        "so",
        "the",
        "total",
        "is",
        "x",
    ];

    fn tokenizer() -> Tokenizer {
        word_level(&WORDS, &[])
    }

    fn id(word: &str) -> u32 {
        WORDS.iter().position(|w| *w == word).unwrap() as u32
    }

    fn options() -> TaskOptions {
        TaskOptions {
            log_samples: true,
            ..TaskOptions::default()
        }
    }

    fn custom_choices(rows: Vec<serde_json::Value>) -> TaskData {
        TaskData {
            test: rows,
            fewshot: Vec::new(),
        }
    }

    /// Whichever of two choices the model likes, by their lengths in characters.
    #[test]
    fn acc_norm_divides_by_the_choices_length_in_characters() {
        // Gold is the long choice. The model rates the short one higher in total
        // (-4 against -6) and the long one higher per character (-6/8 > -4/2).
        let data = custom_choices(vec![json!({
            "question": "so the total", "choices": ["ab", "abcdefgh"], "answer": 1
        })]);
        let ab = id("ab");
        let mut model = Scripted::new(move |r| if r.continuation[0] == ab { -4.0 } else { -6.0 });
        let report = evaluate(
            &mut model,
            &tokenizer(),
            Task::Custom,
            &data,
            &options(),
            &mut |_, _| {},
        )
        .unwrap();
        assert_eq!(report.value("acc"), Some(0.0));
        assert_eq!(report.value("acc_norm"), Some(1.0));
        let sample = &report.samples_log[0];
        assert_eq!(sample["prediction"], 0);
        assert_eq!(sample["loglikelihoods"], json!([-4.0, -6.0]));
        // Each choice was scored after the question, with a space before it.
        assert_eq!(model.asked.len(), 2);
        let (context, continuation) =
            encode_pair(&tokenizer(), "Question: so the total\nAnswer:", " ab").unwrap();
        assert_eq!(model.asked[0].context, context);
        assert_eq!(model.asked[0].continuation, continuation);
        assert_eq!(continuation, [id("ab")]);
    }

    #[test]
    fn accuracy_counts_each_item_and_reports_its_standard_error() {
        // Four items, the model right on three: it prefers the choice named in the question.
        let rows: Vec<_> = (0..4)
            .map(|i| json!({"question": "so", "choices": ["ab", "abcdefgh"], "answer": usize::from(i == 3)}))
            .collect();
        let ab = id("ab");
        let mut model = Scripted::new(move |r| if r.continuation[0] == ab { -1.0 } else { -2.0 });
        let report = evaluate(
            &mut model,
            &tokenizer(),
            Task::Custom,
            &custom_choices(rows),
            &options(),
            &mut |_, _| {},
        )
        .unwrap();
        let acc = report.metrics["acc"];
        assert_eq!(acc.value, 0.75);
        assert!((acc.stderr.unwrap() - 0.25).abs() < 1e-12);
        assert_eq!(report.samples, 4);
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["metrics"]["acc"]["value"], 0.75);
        assert_eq!(json["samples"], 4);
    }

    #[test]
    fn subjects_are_reported_apart_and_a_limit_stops_the_run_early() {
        let row = |subject: &str| json!({"subject": subject, "question": "so", "choices": ["ab", "abcdefgh"], "answer": 0});
        let rows = vec![row("a"), row("a"), row("b"), row("b"), row("b")];
        let ab = id("ab");
        // Right (the first choice) for subject a's items, wrong for b's: by call order.
        let mut calls = 0;
        let mut model = Scripted::new(move |r| {
            let first = r.continuation[0] == ab;
            if first {
                calls += 1;
            }
            let right = calls <= 2;
            if first == right { -1.0 } else { -2.0 }
        });
        let report = evaluate(
            &mut model,
            &tokenizer(),
            Task::Custom,
            &custom_choices(rows.clone()),
            &options(),
            &mut |_, _| {},
        )
        .unwrap();
        assert_eq!(report.groups["a"].metrics["acc"].value, 1.0);
        assert_eq!(report.groups["b"].metrics["acc"].value, 0.0);
        assert_eq!(report.groups["b"].samples, 3);
        assert_eq!(report.value("acc"), Some(0.4));

        // A limit scores the first items only.
        let mut model = Scripted::new(|_| -1.0);
        let limited = TaskOptions {
            limit: 2,
            ..options()
        };
        let report = evaluate(
            &mut model,
            &tokenizer(),
            Task::Custom,
            &custom_choices(rows),
            &limited,
            &mut |_, _| {},
        )
        .unwrap();
        assert_eq!(report.samples, 2);
        assert_eq!(model.asked.len(), 4, "two items, two choices each");
    }

    #[test]
    fn gsm8k_stops_at_the_next_question_and_scores_strict_and_flexible() {
        let data = TaskData {
            test: vec![
                json!({"question": "x", "answer": "so\n#### 5"}),
                json!({"question": "x", "answer": "so\n#### 6"}),
            ],
            fewshot: Vec::new(),
        };
        let mut model = Scripted::new(|_| 0.0);
        // Item 1: the format, then a runaway second question the stop string must cut.
        model.generations.push_back(
            [
                "so",
                "the",
                "total",
                "is",
                "5",
                "apples.",
                "####",
                "5",
                "Question:",
                "6",
            ]
            .map(id)
            .to_vec(),
        );
        // Item 2: the right number, not in the format.
        model
            .generations
            .push_back(["the", "total", "is", "6", "apples."].map(id).to_vec());
        let report = evaluate(
            &mut model,
            &tokenizer(),
            Task::Gsm8k,
            &data,
            &TaskOptions {
                fewshot: Some(0),
                ..options()
            },
            &mut |_, _| {},
        )
        .unwrap();
        // Strict: only the first item wrote `#### N`. Flexible: both end on their number.
        assert_eq!(report.value("exact_match,strict-match"), Some(0.5));
        assert_eq!(report.value("exact_match,flexible-extract"), Some(1.0));
        // The first generation was cut at `Question:`, so the `6` after it did not count.
        let first = &report.samples_log[0];
        assert!(!first["response"].as_str().unwrap().contains("Question:"));
        assert_eq!(first["strict_answer"], "5");
        // The prompt the model saw is the question and the cue.
        let shown = tokenizer().decode(&model.generated_for[0]).unwrap();
        assert!(
            shown.contains("Question:") && shown.ends_with("Answer:"),
            "{shown}"
        );
        assert_eq!(report.fewshot, 0);
    }

    #[test]
    fn a_custom_question_is_matched_by_its_answer_not_its_punctuation() {
        let data = custom_choices(vec![
            json!({"question": "x", "answer": ["Five apples", "5"]}),
            json!({"question": "x", "answer": "6"}),
        ]);
        let mut model = Scripted::new(|_| 0.0);
        model.generations.push_back(vec![id("5"), id("apples.")]);
        model.generations.push_back(vec![id("5")]);
        let report = evaluate(
            &mut model,
            &tokenizer(),
            Task::Custom,
            &data,
            &options(),
            &mut |_, _| {},
        )
        .unwrap();
        // "5 apples." is not "five apples" or "5"; the second answer is wrong too.
        assert_eq!(report.value("exact_match"), Some(0.0));
        assert_eq!(normalize_answer("  \"Five  Apples!\" "), "five apples");
        assert_eq!(normalize_answer("5."), "5");
    }

    #[test]
    fn ifeval_scores_prompts_and_instructions_and_skips_what_it_cannot_check() {
        let row = |ids: &[&str], kwargs: serde_json::Value| json!({"prompt": "so", "instruction_id_list": ids, "kwargs": kwargs});
        let data = TaskData {
            test: vec![
                // Two instructions, one followed.
                row(
                    &["punctuation:no_comma", "keywords:existence"],
                    json!([{}, {"keywords": ["apples"]}]),
                ),
                // An instruction with no checker: skipped.
                row(&["language:response_language"], json!([{"language": "de"}])),
            ],
            fewshot: Vec::new(),
        };
        let mut model = Scripted::new(|_| 0.0);
        model.generations.push_back(vec![id("the"), id("total")]);
        let report = evaluate(
            &mut model,
            &tokenizer(),
            Task::IfevalLite,
            &data,
            &options(),
            &mut |_, _| {},
        )
        .unwrap();
        assert_eq!((report.samples, report.skipped), (1, 1));
        assert_eq!(report.value("prompt_level_strict_acc"), Some(0.0));
        assert_eq!(report.value("inst_level_strict_acc"), Some(0.5));
        assert_eq!(report.metrics.len(), 4);
        assert_eq!(
            model.generated_for.len(),
            1,
            "the skipped prompt was not generated"
        );
    }

    #[test]
    fn a_pair_is_encoded_with_the_contexts_trailing_space_moved() {
        let tokenizer = tokenizer();
        let (context, continuation) = encode_pair(&tokenizer, "so the Answer: ", "ab").unwrap();
        // The space left the context for the continuation's front, which the
        // word-level tokenizer drops: the same tokens as the bare pair.
        let (bare, bare_continuation) = encode_pair(&tokenizer, "so the Answer:", " ab").unwrap();
        assert_eq!((context, continuation), (bare, bare_continuation));
        // A continuation with no tokens of its own is an error, not an empty score.
        assert!(encode_pair(&tokenizer, "so the", " ").is_err());
    }

    /// Gemma 4's tokenizer adds no BOS and its chat template writes one: a
    /// benchmark's prompts must start with it all the same, once, in the
    /// context and not in the continuation.
    #[test]
    fn a_model_that_gets_its_bos_from_its_template_scores_prompts_after_one() {
        let bare =
            pmetal_data::tokenizer::testing::byte_level_bpe(&["a b"], &["<bos>", "<|turn>"], None);
        let dir = tempfile::tempdir().unwrap();
        bare.inner()
            .save(dir.path().join("tokenizer.json"), false)
            .unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"bos_token": "<bos>"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join("chat_template.jinja"),
            "{{- bos_token -}}{%- for m in messages -%}{{ m['content'] }}{%- endfor -%}",
        )
        .unwrap();
        let tokenizer = Tokenizer::from_model_dir(dir.path()).unwrap();
        let bos = tokenizer.inner().token_to_id("<bos>").unwrap();

        let (context, continuation) = encode_pair(&tokenizer, "ab ab:", " ab").unwrap();
        assert_eq!(context.iter().filter(|&&t| t == bos).count(), 1);
        assert_eq!(context[0], bos);
        assert!(!continuation.contains(&bos));
        // A prompt that is generated from starts the same way.
        let prompt = tokenizer.encode_with_special_tokens("ab ab:").unwrap();
        assert_eq!(prompt, context);
    }

    #[test]
    fn an_empty_task_is_an_error() {
        let mut model = Scripted::new(|_| 0.0);
        let data = custom_choices(Vec::new());
        assert!(
            evaluate(
                &mut model,
                &tokenizer(),
                Task::Custom,
                &data,
                &options(),
                &mut |_, _| {},
            )
            .is_err()
        );
    }
}
