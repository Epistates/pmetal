//! The tasks' prompts: how each dataset row becomes something to ask a model.
//!
//! A multiple-choice item is a context and the continuations to score after
//! it; a generation item is a prompt, where to stop, and how its answer is
//! scored. The few-shot examples of an item are part of its context or prompt,
//! each followed by its answer and separated by a blank line.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use regex::Regex;
use serde_json::Value;

use super::datasets::TaskData;
use super::ifeval;
use super::{EvalError, Task, TaskOptions, gsm8k};

/// How a multiple-choice item is scored, beyond whether the likeliest choice
/// is the right one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum McMetric {
    /// The likeliest choice is the gold one.
    Acc,
    /// The same with each choice's log-likelihood divided by its length in
    /// characters, so a longer ending is not penalised for being longer.
    AccNorm,
}

/// A multiple-choice item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct McItem {
    /// The subject, for a task that has them.
    pub subject: Option<String>,
    /// What the model is shown (the few-shot examples included).
    pub context: String,
    /// The choices, as the text that continues the context after a space.
    pub choices: Vec<String>,
    /// Which choice is right.
    pub gold: usize,
}

/// How a generation item is scored.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GenScorer {
    /// GSM8K: the reference answer's full text, `#### N` at the end.
    Gsm8k { reference: String },
    /// IFEval: the instructions the response must follow.
    IfEval(ifeval::Prompt),
    /// A custom question: any of these answers, after normalisation.
    Exact { answers: Vec<String> },
}

/// A generation item.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GenItem {
    /// The subject, for a task that has them.
    pub subject: Option<String>,
    /// What the model continues.
    pub prompt: String,
    /// Strings that end the generation (the text from the first is dropped).
    pub stop: Vec<&'static str>,
    /// The most tokens to generate, when the task wants other than the default.
    pub max_tokens: Option<usize>,
    /// How the output is scored.
    pub scorer: GenScorer,
}

/// What a task asks.
#[derive(Debug, Clone)]
pub(crate) enum Items {
    /// Score each choice's log-likelihood.
    MultipleChoice(Vec<McItem>, &'static [McMetric]),
    /// Generate.
    Generate(Vec<GenItem>),
}

const FEWSHOT_DELIMITER: &str = "\n\n";
const LETTERS: [&str; 4] = ["A", "B", "C", "D"];

fn text<'a>(row: &'a Value, key: &str) -> Result<&'a str, EvalError> {
    row.get(key).and_then(Value::as_str).ok_or_else(|| {
        EvalError::Data(format!(
            "a row has no text `{key}` (the row is {})",
            truncated(row)
        ))
    })
}

fn truncated(row: &Value) -> String {
    let text = row.to_string();
    if text.chars().count() > 160 {
        format!("{}...", text.chars().take(160).collect::<String>())
    } else {
        text
    }
}

/// The text of the first of `keys` the row has as a string.
fn first_text<'a>(row: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| row.get(*key).and_then(Value::as_str))
}

fn strings(row: &Value, key: &str) -> Result<Vec<String>, EvalError> {
    row.get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|v| v.as_str().map(str::to_string))
                .collect::<Option<Vec<_>>>()
        })
        .and_then(|v| v)
        .ok_or_else(|| {
            EvalError::Data(format!(
                "a row has no list of strings `{key}` (the row is {})",
                truncated(row)
            ))
        })
}

/// Which of the rows to score: all, the first `limit`, or for a task with
/// subjects `limit` of them drawn at random with `seed`, so that every subject
/// is in the sample and not only the first.
pub(crate) fn select(rows: usize, limit: usize, by_subject: bool, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..rows).collect();
    if limit == 0 || limit >= rows {
        return indices;
    }
    if by_subject {
        indices.shuffle(&mut StdRng::seed_from_u64(seed));
    }
    indices.truncate(limit);
    indices.sort_unstable();
    indices
}

/// The few-shot examples for the item at `index`: `n` of `pool`, the same ones
/// for the same `seed` and `index`, never `exclude`.
fn draw(
    pool: &[Value],
    n: usize,
    seed: u64,
    index: usize,
    exclude: Option<usize>,
) -> Result<Vec<&Value>, EvalError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut candidates: Vec<usize> = (0..pool.len()).filter(|&i| Some(i) != exclude).collect();
    if candidates.len() < n {
        return Err(EvalError::Data(format!(
            "{n} few-shot examples are asked for and the data has {}",
            candidates.len()
        )));
    }
    let mixed = seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    candidates.shuffle(&mut StdRng::seed_from_u64(mixed));
    Ok(candidates[..n].iter().map(|&i| &pool[i]).collect())
}

/// The context: the few-shot examples, each ending in its answer, then the
/// item's own question.
fn with_examples(examples: &[String], question: &str) -> String {
    let mut out = String::new();
    for example in examples {
        out.push_str(example);
        out.push_str(FEWSHOT_DELIMITER);
    }
    out.push_str(question);
    out
}

// -- GSM8K ---------------------------------------------------------------------------------

fn gsm8k(data: &TaskData, options: &TaskOptions, fewshot: usize) -> Result<Items, EvalError> {
    let selected = select(data.test.len(), options.limit, false, options.seed);
    let mut items = Vec::with_capacity(selected.len());
    for index in selected {
        let row = &data.test[index];
        let question = text(row, "question")?;
        let reference = text(row, "answer")?;
        let mut examples = Vec::new();
        for shot in draw(&data.fewshot, fewshot, options.seed, index, None)? {
            examples.push(format!(
                "Question: {}\nAnswer: {}",
                text(shot, "question")?,
                text(shot, "answer")?
            ));
        }
        items.push(GenItem {
            subject: None,
            prompt: with_examples(&examples, &format!("Question: {question}\nAnswer:")),
            stop: gsm8k::STOP.to_vec(),
            max_tokens: None,
            scorer: GenScorer::Gsm8k {
                reference: reference.to_string(),
            },
        });
    }
    Ok(Items::Generate(items))
}

// -- MMLU ----------------------------------------------------------------------------------

fn mmlu_question(row: &Value) -> Result<(String, Vec<String>, usize), EvalError> {
    let choices = strings(row, "choices")?;
    if choices.len() != LETTERS.len() {
        return Err(EvalError::Data(format!(
            "an MMLU question has {} choices, not 4",
            choices.len()
        )));
    }
    let gold = row
        .get("answer")
        .and_then(Value::as_u64)
        .filter(|&a| (a as usize) < choices.len())
        .ok_or_else(|| {
            EvalError::Data(format!(
                "a row has no valid `answer` index ({})",
                truncated(row)
            ))
        })? as usize;
    let question = format!(
        "{}\nA. {}\nB. {}\nC. {}\nD. {}\nAnswer:",
        text(row, "question")?.trim(),
        choices[0],
        choices[1],
        choices[2],
        choices[3]
    );
    Ok((question, choices, gold))
}

fn mmlu(data: &TaskData, options: &TaskOptions, fewshot: usize) -> Result<Items, EvalError> {
    let selected = select(data.test.len(), options.limit, true, options.seed);
    let mut items = Vec::with_capacity(selected.len());
    for index in selected {
        let row = &data.test[index];
        let subject = text(row, "subject")?;
        let (question, _, gold) = mmlu_question(row)?;
        // The subject's own `dev` questions, in the order the data has them.
        let mut examples = Vec::new();
        for shot in data
            .fewshot
            .iter()
            .filter(|r| r.get("subject").and_then(Value::as_str) == Some(subject))
            .take(fewshot)
        {
            let (shot_question, _, shot_gold) = mmlu_question(shot)?;
            examples.push(format!("{shot_question} {}", LETTERS[shot_gold]));
        }
        if examples.len() < fewshot {
            return Err(EvalError::Data(format!(
                "{fewshot} few-shot examples are asked for and `{subject}` has {}",
                examples.len()
            )));
        }
        let description = format!(
            "The following are multiple choice questions (with answers) about {}.\n\n",
            subject.replace('_', " ")
        );
        items.push(McItem {
            subject: Some(subject.to_string()),
            context: format!("{description}{}", with_examples(&examples, &question)),
            choices: LETTERS.iter().map(|l| (*l).to_string()).collect(),
            gold,
        });
    }
    Ok(Items::MultipleChoice(items, &[McMetric::Acc]))
}

// -- ARC -----------------------------------------------------------------------------------

fn arc_question(row: &Value) -> Result<(String, Vec<String>, usize), EvalError> {
    let choices = row
        .get("choices")
        .ok_or_else(|| EvalError::Data(format!("a row has no `choices` ({})", truncated(row))))?;
    let texts = strings(choices, "text")?;
    let labels = strings(choices, "label")?;
    let key = text(row, "answerKey")?;
    let gold = labels
        .iter()
        .position(|l| l == key)
        .ok_or_else(|| EvalError::Data(format!("answer key {key} is not one of {labels:?}")))?;
    Ok((
        format!("Question: {}\nAnswer:", text(row, "question")?),
        texts,
        gold,
    ))
}

fn arc(data: &TaskData, options: &TaskOptions, fewshot: usize) -> Result<Items, EvalError> {
    let selected = select(data.test.len(), options.limit, false, options.seed);
    let mut items = Vec::with_capacity(selected.len());
    for index in selected {
        let (question, choices, gold) = arc_question(&data.test[index])?;
        let mut examples = Vec::new();
        for shot in draw(&data.fewshot, fewshot, options.seed, index, None)? {
            let (shot_question, shot_choices, shot_gold) = arc_question(shot)?;
            examples.push(format!("{shot_question} {}", shot_choices[shot_gold]));
        }
        items.push(McItem {
            subject: None,
            context: with_examples(&examples, &question),
            choices,
            gold,
        });
    }
    Ok(Items::MultipleChoice(
        items,
        &[McMetric::Acc, McMetric::AccNorm],
    ))
}

// -- HellaSwag -----------------------------------------------------------------------------

/// The dataset's clean-up: WikiHow's `[title]` and bracketed artifacts out.
fn hellaswag_preprocess(text: &str) -> String {
    static BRACKETS: std::sync::LazyLock<Regex> =
        std::sync::LazyLock::new(|| Regex::new(r"\[.*?\]").expect("a valid pattern"));
    let text = text.trim().replace(" [title]", ". ");
    BRACKETS.replace_all(&text, "").replace("  ", " ")
}

/// Python's `str.capitalize`: the first character upper case, the rest lower.
fn capitalize(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first
            .to_uppercase()
            .chain(chars.flat_map(char::to_lowercase))
            .collect(),
        None => String::new(),
    }
}

fn hellaswag_question(row: &Value) -> Result<(String, Vec<String>, usize), EvalError> {
    let context = format!(
        "{} {}",
        text(row, "ctx_a")?,
        capitalize(text(row, "ctx_b").unwrap_or(""))
    );
    let query = hellaswag_preprocess(&format!("{}: {context}", text(row, "activity_label")?));
    let endings: Vec<String> = strings(row, "endings")?
        .iter()
        .map(|e| hellaswag_preprocess(e))
        .collect();
    let label = row.get("label");
    let gold = label
        .and_then(|l| {
            l.as_u64()
                .or_else(|| l.as_str().and_then(|s| s.parse().ok()))
        })
        .map(|g| g as usize)
        .filter(|&g| g < endings.len())
        .ok_or_else(|| {
            EvalError::Data(format!("a row has no valid `label` ({})", truncated(row)))
        })?;
    Ok((query, endings, gold))
}

fn hellaswag(data: &TaskData, options: &TaskOptions, fewshot: usize) -> Result<Items, EvalError> {
    let selected = select(data.test.len(), options.limit, false, options.seed);
    let mut items = Vec::with_capacity(selected.len());
    for index in selected {
        let (query, choices, gold) = hellaswag_question(&data.test[index])?;
        let mut examples = Vec::new();
        for shot in draw(&data.fewshot, fewshot, options.seed, index, None)? {
            let (shot_query, shot_endings, shot_gold) = hellaswag_question(shot)?;
            examples.push(format!("{shot_query} {}", shot_endings[shot_gold]));
        }
        items.push(McItem {
            subject: None,
            context: with_examples(&examples, &query),
            choices,
            gold,
        });
    }
    Ok(Items::MultipleChoice(
        items,
        &[McMetric::Acc, McMetric::AccNorm],
    ))
}

// -- IFEval --------------------------------------------------------------------------------

/// IFEval's answers are long, and its prompts are asked as they are.
const IFEVAL_MAX_TOKENS: usize = 1280;

fn ifeval(data: &TaskData, options: &TaskOptions) -> Result<Items, EvalError> {
    let selected = select(data.test.len(), options.limit, false, options.seed);
    let mut items = Vec::with_capacity(selected.len());
    for index in selected {
        let row = &data.test[index];
        let prompt = ifeval::Prompt::from_row(row)?;
        items.push(GenItem {
            subject: None,
            prompt: text(row, "prompt")?.to_string(),
            stop: Vec::new(),
            max_tokens: Some(IFEVAL_MAX_TOKENS),
            scorer: GenScorer::IfEval(prompt),
        });
    }
    Ok(Items::Generate(items))
}

// -- Custom --------------------------------------------------------------------------------

const QUESTION_KEYS: [&str; 5] = ["question", "prompt", "input", "query", "context"];
const ANSWER_KEYS: [&str; 5] = ["answer", "answers", "target", "output", "response"];

fn custom_question(row: &Value) -> Result<&str, EvalError> {
    first_text(row, &QUESTION_KEYS).ok_or_else(|| {
        EvalError::Data(format!(
            "a custom row needs a `question` (or prompt, input, query, context): {}",
            truncated(row)
        ))
    })
}

/// The index of the right choice, from an answer given as an index, a letter,
/// or the choice's text.
fn custom_gold(row: &Value, choices: &[String]) -> Result<usize, EvalError> {
    let answer = ANSWER_KEYS
        .iter()
        .find_map(|k| row.get(*k))
        .ok_or_else(|| {
            EvalError::Data(format!("a custom row has no `answer`: {}", truncated(row)))
        })?;
    let by_index = answer.as_u64().map(|a| a as usize);
    let by_text = answer.as_str().and_then(|a| {
        choices
            .iter()
            .position(|c| c.trim() == a.trim())
            .or_else(|| {
                // A letter names the choice at its place in the alphabet.
                let mut chars = a.trim().chars();
                match (chars.next(), chars.next()) {
                    (Some(c), None) if c.is_ascii_uppercase() => Some((c as u8 - b'A') as usize),
                    _ => None,
                }
            })
    });
    by_index
        .or(by_text)
        .filter(|&g| g < choices.len())
        .ok_or_else(|| {
            EvalError::Data(format!(
                "the answer of a custom row is not one of its {} choices: {}",
                choices.len(),
                truncated(row)
            ))
        })
}

fn custom_answers(row: &Value) -> Result<Vec<String>, EvalError> {
    let answer = ANSWER_KEYS
        .iter()
        .find_map(|k| row.get(*k))
        .ok_or_else(|| {
            EvalError::Data(format!("a custom row has no `answer`: {}", truncated(row)))
        })?;
    match answer {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Number(n) => Ok(vec![n.to_string()]),
        Value::Array(items) => items
            .iter()
            .map(|v| match v {
                Value::String(s) => Ok(s.clone()),
                Value::Number(n) => Ok(n.to_string()),
                _ => Err(EvalError::Data(format!(
                    "a custom answer list holds only strings and numbers: {}",
                    truncated(row)
                ))),
            })
            .collect(),
        _ => Err(EvalError::Data(format!(
            "the answer of a custom row is text, a number or a list of them: {}",
            truncated(row)
        ))),
    }
}

fn custom(data: &TaskData, options: &TaskOptions, fewshot: usize) -> Result<Items, EvalError> {
    let multiple_choice = data.test.iter().all(|row| row.get("choices").is_some());
    let any_choices = data.test.iter().any(|row| row.get("choices").is_some());
    if any_choices && !multiple_choice {
        return Err(EvalError::Data(
            "a custom dataset is either all questions with `choices` or all without".into(),
        ));
    }
    let selected = select(data.test.len(), options.limit, false, options.seed);
    // A custom dataset's few-shot examples are its other rows.
    let pool = if data.fewshot.is_empty() {
        &data.test
    } else {
        &data.fewshot
    };
    let own_pool = data.fewshot.is_empty();
    if multiple_choice {
        let mut items = Vec::with_capacity(selected.len());
        for index in selected {
            let row = &data.test[index];
            let choices = strings(row, "choices")?;
            if choices.len() < 2 {
                return Err(EvalError::Data(format!(
                    "a custom question needs at least two choices: {}",
                    truncated(row)
                )));
            }
            let gold = custom_gold(row, &choices)?;
            let mut examples = Vec::new();
            for shot in draw(
                pool,
                fewshot,
                options.seed,
                index,
                own_pool.then_some(index),
            )? {
                let shot_choices = strings(shot, "choices")?;
                let shot_gold = custom_gold(shot, &shot_choices)?;
                examples.push(format!(
                    "Question: {}\nAnswer: {}",
                    custom_question(shot)?,
                    shot_choices[shot_gold]
                ));
            }
            items.push(McItem {
                subject: row
                    .get("subject")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                context: with_examples(
                    &examples,
                    &format!("Question: {}\nAnswer:", custom_question(row)?),
                ),
                choices,
                gold,
            });
        }
        return Ok(Items::MultipleChoice(
            items,
            &[McMetric::Acc, McMetric::AccNorm],
        ));
    }
    let mut items = Vec::with_capacity(selected.len());
    for index in selected {
        let row = &data.test[index];
        let mut examples = Vec::new();
        for shot in draw(
            pool,
            fewshot,
            options.seed,
            index,
            own_pool.then_some(index),
        )? {
            examples.push(format!(
                "Question: {}\nAnswer: {}",
                custom_question(shot)?,
                custom_answers(shot)?[0]
            ));
        }
        items.push(GenItem {
            subject: row
                .get("subject")
                .and_then(Value::as_str)
                .map(str::to_string),
            prompt: with_examples(
                &examples,
                &format!("Question: {}\nAnswer:", custom_question(row)?),
            ),
            stop: vec!["\n\n", "Question:", "</s>", "<|im_end|>"],
            max_tokens: None,
            scorer: GenScorer::Exact {
                answers: custom_answers(row)?,
            },
        });
    }
    Ok(Items::Generate(items))
}

/// The items of `task` over `data`.
///
/// # Errors
///
/// [`EvalError::Data`] for a row that is not the task's layout, or too few
/// rows to draw the few-shot examples from.
pub(crate) fn build(
    task: Task,
    data: &TaskData,
    options: &TaskOptions,
    fewshot: usize,
) -> Result<Items, EvalError> {
    if task == Task::IfevalLite && fewshot > 0 {
        // IFEval's prompts are asked as they are.
        return Err(EvalError::Config(format!(
            "{task} takes no few-shot examples"
        )));
    }
    match task {
        Task::Gsm8k => gsm8k(data, options, fewshot),
        Task::Mmlu => mmlu(data, options, fewshot),
        Task::ArcChallenge => arc(data, options, fewshot),
        Task::Hellaswag => hellaswag(data, options, fewshot),
        Task::IfevalLite => ifeval(data, options),
        Task::Custom => custom(data, options, fewshot),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn options() -> TaskOptions {
        TaskOptions::default()
    }

    fn mc(items: Items) -> Vec<McItem> {
        match items {
            Items::MultipleChoice(items, _) => items,
            Items::Generate(_) => panic!("expected multiple choice"),
        }
    }

    fn generated(items: Items) -> Vec<GenItem> {
        match items {
            Items::Generate(items) => items,
            Items::MultipleChoice(..) => panic!("expected generation"),
        }
    }

    #[test]
    fn gsm8k_asks_the_question_after_the_few_shot_answers() {
        let data = TaskData {
            test: vec![json!({"question": "What is 2+3?", "answer": "2+3=<<2+3=5>>5\n#### 5"})],
            fewshot: vec![
                json!({"question": "What is 1+1?", "answer": "1+1=2\n#### 2"}),
                json!({"question": "What is 4+4?", "answer": "4+4=8\n#### 8"}),
            ],
        };
        let item = generated(build(Task::Gsm8k, &data, &options(), 2).unwrap()).remove(0);
        // Two examples, each with its chain-of-thought answer, then the question.
        assert!(item.prompt.ends_with("\n\nQuestion: What is 2+3?\nAnswer:"));
        assert!(
            item.prompt
                .contains("Question: What is 1+1?\nAnswer: 1+1=2\n#### 2\n\n")
        );
        assert!(
            item.prompt
                .contains("Question: What is 4+4?\nAnswer: 4+4=8\n#### 8\n\n")
        );
        assert_eq!(item.stop, ["Question:", "</s>", "<|im_end|>"]);
        let GenScorer::Gsm8k { reference } = item.scorer else {
            panic!("a GSM8K scorer");
        };
        assert!(reference.ends_with("#### 5"));
        // Without examples the prompt is the question alone.
        let bare = generated(build(Task::Gsm8k, &data, &options(), 0).unwrap()).remove(0);
        assert_eq!(bare.prompt, "Question: What is 2+3?\nAnswer:");
    }

    #[test]
    fn the_same_seed_draws_the_same_examples_and_another_draws_others() {
        let pool: Vec<Value> = (0..50).map(|i| json!({"i": i})).collect();
        let picks = |seed, index| -> Vec<i64> {
            draw(&pool, 5, seed, index, None)
                .unwrap()
                .iter()
                .map(|v| v["i"].as_i64().unwrap())
                .collect()
        };
        assert_eq!(picks(1, 3), picks(1, 3));
        assert_ne!(picks(1, 3), picks(2, 3));
        assert_ne!(picks(1, 3), picks(1, 4));
        // Five distinct examples, none the excluded one.
        let mut distinct = picks(1, 3);
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 5);
        let without = draw(&pool, 49, 1, 0, Some(7)).unwrap();
        assert!(without.iter().all(|v| v["i"] != 7));
        assert!(draw(&pool, 51, 1, 0, None).is_err());
    }

    fn mmlu_row(subject: &str, question: &str, answer: u64) -> Value {
        json!({"subject": subject, "question": question, "choices": ["w", "x", "y", "z"], "answer": answer})
    }

    #[test]
    fn mmlu_describes_the_subject_then_shows_its_dev_questions_then_asks() {
        let data = TaskData {
            test: vec![mmlu_row("abstract_algebra", "  Which is a group? ", 2)],
            fewshot: vec![
                mmlu_row("anatomy", "Not this subject", 0),
                mmlu_row("abstract_algebra", "First example", 1),
                mmlu_row("abstract_algebra", "Second example", 3),
            ],
        };
        let item = mc(build(Task::Mmlu, &data, &options(), 2).unwrap()).remove(0);
        assert_eq!(
            item.context,
            "The following are multiple choice questions (with answers) about abstract algebra.\n\n\
             First example\nA. w\nB. x\nC. y\nD. z\nAnswer: B\n\n\
             Second example\nA. w\nB. x\nC. y\nD. z\nAnswer: D\n\n\
             Which is a group?\nA. w\nB. x\nC. y\nD. z\nAnswer:"
        );
        assert_eq!(item.choices, ["A", "B", "C", "D"]);
        assert_eq!(item.gold, 2);
        assert_eq!(item.subject.as_deref(), Some("abstract_algebra"));
        // More examples than the subject has is an error, not a shorter prompt.
        assert!(build(Task::Mmlu, &data, &options(), 3).is_err());
    }

    #[test]
    fn arc_asks_the_question_and_scores_the_choices_texts() {
        let row = |question: &str, key: &str| {
            json!({"question": question, "answerKey": key,
                   "choices": {"text": ["a seed", "a rock", "a tree"], "label": ["A", "B", "C"]}})
        };
        let data = TaskData {
            test: vec![row("What grows?", "C")],
            fewshot: vec![row("Which is hard?", "B")],
        };
        let item = mc(build(Task::ArcChallenge, &data, &options(), 1).unwrap()).remove(0);
        assert_eq!(
            item.context,
            "Question: Which is hard?\nAnswer: a rock\n\nQuestion: What grows?\nAnswer:"
        );
        assert_eq!(item.choices, ["a seed", "a rock", "a tree"]);
        assert_eq!(item.gold, 2);
        // A numeric key indexes the labels, which need not be letters.
        let numeric = json!({"question": "q", "answerKey": "2",
                             "choices": {"text": ["x", "y"], "label": ["1", "2"]}});
        assert_eq!(arc_question(&numeric).unwrap().2, 1);
        let missing = json!({"question": "q", "answerKey": "Z",
                             "choices": {"text": ["x"], "label": ["A"]}});
        assert!(arc_question(&missing).is_err());
    }

    #[test]
    fn hellaswag_cleans_the_markup_and_capitalizes_the_second_context() {
        assert_eq!(
            hellaswag_preprocess(" Grow tomatoes [title] Pick a spot [step] now.  Done. "),
            "Grow tomatoes. Pick a spot now. Done."
        );
        assert_eq!(capitalize("the dog RAN"), "The dog ran");
        assert_eq!(capitalize(""), "");
        let row = json!({
            "activity_label": "Baking", "ctx_a": "A man mixes dough.", "ctx_b": "then he SHAPES it",
            "endings": ["He bakes it.", "He eats [bracket] it."], "label": "1"
        });
        let (query, endings, gold) = hellaswag_question(&row).unwrap();
        assert_eq!(query, "Baking: A man mixes dough. Then he shapes it");
        assert_eq!(endings, ["He bakes it.", "He eats it."]);
        assert_eq!(gold, 1);
        // The label is a string in the dataset and a number in a copy of it.
        let numeric = json!({"activity_label": "x", "ctx_a": "a", "ctx_b": "b",
                             "endings": ["p", "q"], "label": 0});
        assert_eq!(hellaswag_question(&numeric).unwrap().2, 0);
    }

    #[test]
    fn a_limit_takes_the_first_items_or_an_even_draw_across_subjects() {
        assert_eq!(select(5, 0, false, 1), [0, 1, 2, 3, 4]);
        assert_eq!(select(5, 9, true, 1), [0, 1, 2, 3, 4]);
        assert_eq!(select(10, 3, false, 1), [0, 1, 2]);
        let drawn = select(1000, 5, true, 7);
        assert_eq!(drawn.len(), 5);
        assert!(drawn.windows(2).all(|w| w[0] < w[1]), "in the data's order");
        assert_eq!(drawn, select(1000, 5, true, 7));
        assert_ne!(drawn, select(1000, 5, true, 8));
        assert!(drawn.iter().any(|&i| i >= 5), "not just the first items");
    }

    #[test]
    fn custom_rows_are_choices_or_questions_with_answers() {
        let choices = TaskData {
            test: vec![
                json!({"question": "Pick red", "choices": ["blue", "red"], "answer": 1}),
                json!({"question": "Pick blue", "choices": ["blue", "red"], "answer": "blue"}),
                json!({"question": "Pick red again", "choices": ["blue", "red"], "answer": "B"}),
            ],
            fewshot: Vec::new(),
        };
        let items = mc(build(Task::Custom, &choices, &options(), 0).unwrap());
        assert_eq!(items.iter().map(|i| i.gold).collect::<Vec<_>>(), [1, 0, 1]);
        assert_eq!(items[0].context, "Question: Pick red\nAnswer:");
        // The few-shot examples are the other rows, never the item itself.
        let shown = mc(build(Task::Custom, &choices, &options(), 2).unwrap());
        assert!(!shown[0].context.contains("Pick red\nAnswer: blue"));
        assert_eq!(shown[0].context.matches("Question:").count(), 3);

        let qa = TaskData {
            test: vec![json!({"question": "2+2?", "answer": ["4", "four"]})],
            fewshot: Vec::new(),
        };
        let item = generated(build(Task::Custom, &qa, &options(), 0).unwrap()).remove(0);
        assert_eq!(item.prompt, "Question: 2+2?\nAnswer:");
        assert_eq!(
            item.scorer,
            GenScorer::Exact {
                answers: vec!["4".into(), "four".into()]
            }
        );

        // Mixed layouts and unanswerable rows are refused, not guessed at.
        let mixed = TaskData {
            test: vec![
                json!({"question": "a", "answer": "x"}),
                json!({"question": "b", "choices": ["p", "q"], "answer": 0}),
            ],
            fewshot: Vec::new(),
        };
        assert!(build(Task::Custom, &mixed, &options(), 0).is_err());
        let bad_answer = TaskData {
            test: vec![json!({"question": "a", "choices": ["p", "q"], "answer": "zebra"})],
            fewshot: Vec::new(),
        };
        assert!(build(Task::Custom, &bad_answer, &options(), 0).is_err());
        let no_question = TaskData {
            test: vec![json!({"answer": "x"})],
            fewshot: Vec::new(),
        };
        assert!(build(Task::Custom, &no_question, &options(), 0).is_err());
    }
}
