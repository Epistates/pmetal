//! A subset of IFEval: verifiable instructions, checked by code.
//!
//! Each prompt carries instructions a program can verify ("answer in at least
//! 300 words", "do not use commas", "end with the phrase ..."). A response is
//! scored two ways: *strict*, the response as it is, and *loose*, where it
//! passes if any of eight variants of it does: the response with its first
//! line, its last line or both removed (models open with "Sure, here is ..."
//! and close with a sign-off), each with and without markdown asterisks. A
//! prompt follows its instructions when every one of them is followed.
//!
//! "Lite" is what needs no language tooling. Every instruction type is
//! checked except `language:response_language`, which needs a language
//! detector; a prompt that has one is skipped and counted as skipped. Two
//! checks are approximations: `english_capital` and `english_lowercase`
//! check the case and not that the text is English, and the word tokens of
//! `capital_word_frequency` are runs of word characters, where the reference
//! uses a Penn Treebank tokenizer. Sentences are split by the reference's own
//! rules.

use std::sync::LazyLock;

use regex::{Regex, RegexBuilder};
use serde_json::Value;

use super::EvalError;

/// An instruction no checker is written for.
const UNSUPPORTED: [&str; 1] = ["language:response_language"];

/// One instruction of a prompt, with its parameters.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Instruction {
    pub id: String,
    pub kwargs: Value,
}

/// A prompt's instructions.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Prompt {
    pub instructions: Vec<Instruction>,
}

/// Which instructions a response followed, strictly and loosely, in the
/// prompt's order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Verdict {
    pub strict: Vec<bool>,
    pub loose: Vec<bool>,
}

impl Prompt {
    /// The instructions of a dataset row: `instruction_id_list` and `kwargs`,
    /// one entry each.
    pub(crate) fn from_row(row: &Value) -> Result<Self, EvalError> {
        let ids = row
            .get("instruction_id_list")
            .and_then(Value::as_array)
            .ok_or_else(|| EvalError::Data("an IFEval row has no `instruction_id_list`".into()))?;
        let kwargs = row.get("kwargs").and_then(Value::as_array);
        let instructions = ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let id = id
                    .as_str()
                    .ok_or_else(|| EvalError::Data("an instruction id is not text".into()))?;
                // The dataset has a null for every parameter an instruction does not take.
                let params = kwargs
                    .and_then(|k| k.get(i))
                    .and_then(Value::as_object)
                    .map(|map| {
                        map.iter()
                            .filter(|(_, v)| !v.is_null())
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect::<serde_json::Map<_, _>>()
                    })
                    .unwrap_or_default();
                Ok(Instruction {
                    id: id.to_string(),
                    kwargs: Value::Object(params),
                })
            })
            .collect::<Result<Vec<_>, EvalError>>()?;
        Ok(Self { instructions })
    }

    /// Whether every instruction has a checker.
    pub(crate) fn supported(&self) -> bool {
        self.instructions
            .iter()
            .all(|i| !UNSUPPORTED.contains(&i.id.as_str()) && checker(&i.id).is_some())
    }

    /// Score `response` to `prompt` (the text the model was shown).
    pub(crate) fn score(&self, prompt: &str, response: &str) -> Verdict {
        let variants = loose_variants(response);
        let (mut strict, mut loose) = (Vec::new(), Vec::new());
        for instruction in &self.instructions {
            let check = checker(&instruction.id);
            let follows = |text: &str| {
                !text.trim().is_empty()
                    && check.is_some_and(|check| check(&instruction.kwargs, prompt, text))
            };
            strict.push(follows(response));
            loose.push(variants.iter().any(|v| follows(v)));
        }
        Verdict { strict, loose }
    }
}

/// The response, and the seven variants the loose score also tries.
fn loose_variants(response: &str) -> Vec<String> {
    let lines: Vec<&str> = response.split('\n').collect();
    let join = |part: &[&str]| part.join("\n").trim().to_string();
    let n = lines.len();
    let bases = [
        response.to_string(),
        join(lines.get(1..).unwrap_or_default()),
        join(&lines[..n.saturating_sub(1)]),
        join(lines.get(1..n.saturating_sub(1)).unwrap_or_default()),
    ];
    let mut all: Vec<String> = bases.iter().map(|b| b.replace('*', "")).collect();
    // The response itself, then the rest, as the reference lists them.
    all.splice(0..0, bases);
    all
}

type Check = fn(&Value, &str, &str) -> bool;

fn checker(id: &str) -> Option<Check> {
    Some(match id {
        "keywords:existence" => keywords_existence,
        "keywords:frequency" => keywords_frequency,
        "keywords:forbidden_words" => forbidden_words,
        "keywords:letter_frequency" => letter_frequency,
        "length_constraints:number_sentences" => number_sentences,
        "length_constraints:number_paragraphs" => number_paragraphs,
        "length_constraints:number_words" => number_words,
        "length_constraints:nth_paragraph_first_word" => nth_paragraph_first_word,
        "detectable_content:number_placeholders" => number_placeholders,
        "detectable_content:postscript" => postscript,
        "detectable_format:number_bullet_lists" => number_bullet_lists,
        "detectable_format:constrained_response" => constrained_response,
        "detectable_format:number_highlighted_sections" => number_highlighted_sections,
        "detectable_format:multiple_sections" => multiple_sections,
        "detectable_format:json_format" => json_format,
        "detectable_format:title" => title,
        "combination:two_responses" => two_responses,
        "combination:repeat_prompt" => repeat_prompt,
        "startend:end_checker" => end_checker,
        "startend:quotation" => quotation,
        "change_case:capital_word_frequency" => capital_word_frequency,
        "change_case:english_capital" => english_capital,
        "change_case:english_lowercase" => english_lowercase,
        "punctuation:no_comma" => no_comma,
        _ => return None,
    })
}

// -- parameters ---------------------------------------------------------------------------

fn int(kwargs: &Value, key: &str) -> Option<i64> {
    let v = kwargs.get(key)?;
    v.as_i64().or_else(|| v.as_f64().map(|f| f as i64))
}

fn string<'a>(kwargs: &'a Value, key: &str) -> Option<&'a str> {
    kwargs.get(key)?.as_str()
}

/// `count` against `target` as the relation says: "less than" is strict,
/// "at least" is not.
fn relates(relation: Option<&str>, count: i64, target: i64) -> bool {
    match relation {
        Some("less than") => count < target,
        Some("at least") => count >= target,
        _ => false,
    }
}

fn ignore_case(pattern: &str) -> Option<Regex> {
    RegexBuilder::new(pattern)
        .case_insensitive(true)
        .build()
        .ok()
}

fn multi_line(pattern: &str) -> Regex {
    RegexBuilder::new(pattern)
        .multi_line(true)
        .build()
        .expect("a valid pattern")
}

// -- the checks ---------------------------------------------------------------------------

fn keywords_existence(kwargs: &Value, _: &str, text: &str) -> bool {
    let Some(keywords) = kwargs.get("keywords").and_then(Value::as_array) else {
        return false;
    };
    keywords
        .iter()
        .filter_map(Value::as_str)
        .all(|k| ignore_case(k.trim()).is_some_and(|re| re.is_match(text)))
}

fn keywords_frequency(kwargs: &Value, _: &str, text: &str) -> bool {
    let (Some(keyword), Some(target)) = (string(kwargs, "keyword"), int(kwargs, "frequency"))
    else {
        return false;
    };
    let Some(re) = ignore_case(keyword) else {
        return false;
    };
    relates(
        string(kwargs, "relation"),
        re.find_iter(text).count() as i64,
        target,
    )
}

fn forbidden_words(kwargs: &Value, _: &str, text: &str) -> bool {
    let Some(words) = kwargs.get("forbidden_words").and_then(Value::as_array) else {
        return false;
    };
    words
        .iter()
        .filter_map(Value::as_str)
        .all(|w| ignore_case(&format!(r"\b{w}\b")).is_some_and(|re| !re.is_match(text)))
}

fn letter_frequency(kwargs: &Value, _: &str, text: &str) -> bool {
    let (Some(letter), Some(target)) = (string(kwargs, "letter"), int(kwargs, "let_frequency"))
    else {
        return false;
    };
    let letter = letter.to_lowercase();
    let count = text
        .to_lowercase()
        .chars()
        .filter(|c| letter.starts_with(*c))
        .count() as i64;
    relates(string(kwargs, "let_relation"), count, target)
}

fn word_count(text: &str) -> i64 {
    static WORDS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\w+").expect("a valid pattern"));
    WORDS.find_iter(text).count() as i64
}

fn number_words(kwargs: &Value, _: &str, text: &str) -> bool {
    let Some(target) = int(kwargs, "num_words") else {
        return false;
    };
    relates(string(kwargs, "relation"), word_count(text), target)
}

fn number_sentences(kwargs: &Value, _: &str, text: &str) -> bool {
    let Some(target) = int(kwargs, "num_sentences") else {
        return false;
    };
    relates(
        string(kwargs, "relation"),
        split_into_sentences(text).len() as i64,
        target,
    )
}

fn number_paragraphs(kwargs: &Value, _: &str, text: &str) -> bool {
    static SPLIT: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\s?\*\*\*\s?").expect("a valid pattern"));
    let Some(target) = int(kwargs, "num_paragraphs") else {
        return false;
    };
    let paragraphs: Vec<&str> = SPLIT.split(text).collect();
    let mut count = paragraphs.len() as i64;
    for (i, paragraph) in paragraphs.iter().enumerate() {
        if paragraph.trim().is_empty() {
            // An empty paragraph at either end is the separator's own; one in the middle is a miss.
            if i == 0 || i == paragraphs.len() - 1 {
                count -= 1;
            } else {
                return false;
            }
        }
    }
    count == target
}

fn nth_paragraph_first_word(kwargs: &Value, _: &str, text: &str) -> bool {
    let (Some(num), Some(nth), Some(first)) = (
        int(kwargs, "num_paragraphs"),
        int(kwargs, "nth_paragraph"),
        string(kwargs, "first_word"),
    ) else {
        return false;
    };
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let count = paragraphs.iter().filter(|p| !p.trim().is_empty()).count() as i64;
    if nth < 1 || nth > count {
        return false;
    }
    let Some(paragraph) = paragraphs.get(nth as usize - 1).map(|p| p.trim()) else {
        return false;
    };
    let Some(word) = paragraph.split_whitespace().next() else {
        return false;
    };
    let word = word.trim_start_matches('\'').trim_start_matches('"');
    let mut found = String::new();
    for c in word.chars() {
        if matches!(c, '.' | ',' | '?' | '!' | '\'' | '"') {
            break;
        }
        found.extend(c.to_lowercase());
    }
    count == num && found == first
}

fn number_placeholders(kwargs: &Value, _: &str, text: &str) -> bool {
    static PLACEHOLDER: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\[.*?\]").expect("a valid pattern"));
    int(kwargs, "num_placeholders").is_some_and(|n| PLACEHOLDER.find_iter(text).count() as i64 >= n)
}

fn postscript(kwargs: &Value, _: &str, text: &str) -> bool {
    let Some(marker) = string(kwargs, "postscript_marker") else {
        return false;
    };
    let pattern = match marker {
        "P.P.S" => r"\s*p\.\s?p\.\s?s.*$".to_string(),
        "P.S." => r"\s*p\.\s?s\..*$".to_string(),
        other => format!(r"\s*{}.*$", regex::escape(&other.to_lowercase())),
    };
    multi_line(&pattern).is_match(&text.to_lowercase())
}

fn number_bullet_lists(kwargs: &Value, _: &str, text: &str) -> bool {
    static STAR: LazyLock<Regex> = LazyLock::new(|| multi_line(r"^\s*\*[^\*].*$"));
    static DASH: LazyLock<Regex> = LazyLock::new(|| multi_line(r"^\s*-.*$"));
    int(kwargs, "num_bullets")
        .is_some_and(|n| (STAR.find_iter(text).count() + DASH.find_iter(text).count()) as i64 == n)
}

fn constrained_response(_: &Value, _: &str, text: &str) -> bool {
    [
        "My answer is yes.",
        "My answer is no.",
        "My answer is maybe.",
    ]
    .iter()
    .any(|option| text.contains(option))
}

fn number_highlighted_sections(kwargs: &Value, _: &str, text: &str) -> bool {
    static SINGLE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\*[^\n\*]*\*").expect("a valid pattern"));
    static DOUBLE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\*\*[^\n\*]*\*\*").expect("a valid pattern"));
    let Some(target) = int(kwargs, "num_highlights") else {
        return false;
    };
    let mut count = SINGLE
        .find_iter(text)
        .filter(|m| !m.as_str().trim_matches('*').trim().is_empty())
        .count();
    count += DOUBLE
        .find_iter(text)
        .filter(|m| {
            let inner = m.as_str();
            let inner = inner.strip_prefix("**").unwrap_or(inner);
            !inner.strip_suffix("**").unwrap_or(inner).trim().is_empty()
        })
        .count();
    count as i64 >= target
}

fn multiple_sections(kwargs: &Value, _: &str, text: &str) -> bool {
    let (Some(splitter), Some(target)) = (
        string(kwargs, "section_spliter"),
        int(kwargs, "num_sections"),
    ) else {
        return false;
    };
    let Ok(re) = Regex::new(&format!(r"\s?{}\s?\d+\s?", regex::escape(splitter.trim()))) else {
        return false;
    };
    // The reference does not escape the splitter; a plain word is the same either way.
    re.split(text).count() as i64 > target
}

fn json_format(_: &Value, _: &str, text: &str) -> bool {
    let mut value = text.trim();
    for prefix in ["```json", "```Json", "```JSON", "```"] {
        if let Some(rest) = value.strip_prefix(prefix) {
            value = rest;
            break;
        }
    }
    let value = value.strip_suffix("```").unwrap_or(value).trim();
    serde_json::from_str::<Value>(value).is_ok()
}

fn title(_: &Value, _: &str, text: &str) -> bool {
    static TITLE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"<<[^\n]+>>").expect("a valid pattern"));
    TITLE.find_iter(text).any(|m| {
        !m.as_str()
            .trim_start_matches('<')
            .trim_end_matches('>')
            .trim()
            .is_empty()
    })
}

fn two_responses(_: &Value, _: &str, text: &str) -> bool {
    let responses: Vec<&str> = text.split("******").collect();
    let mut valid = Vec::new();
    for (i, response) in responses.iter().enumerate() {
        if response.trim().is_empty() {
            if i != 0 && i != responses.len() - 1 {
                return false;
            }
        } else {
            valid.push(*response);
        }
    }
    valid.len() == 2 && valid[0].trim() != valid[1].trim()
}

fn repeat_prompt(kwargs: &Value, _: &str, text: &str) -> bool {
    string(kwargs, "prompt_to_repeat").is_some_and(|p| {
        text.trim()
            .to_lowercase()
            .starts_with(&p.trim().to_lowercase())
    })
}

fn end_checker(kwargs: &Value, _: &str, text: &str) -> bool {
    string(kwargs, "end_phrase").is_some_and(|phrase| {
        text.trim()
            .trim_matches('"')
            .to_lowercase()
            .ends_with(&phrase.trim().to_lowercase())
    })
}

fn quotation(_: &Value, _: &str, text: &str) -> bool {
    let value = text.trim();
    value.chars().count() > 1 && value.starts_with('"') && value.ends_with('"')
}

/// Python's `str.isupper`: some character is cased and none is lower case.
fn is_upper(text: &str) -> bool {
    text.chars().any(char::is_uppercase) && !text.chars().any(char::is_lowercase)
}

/// Python's `str.islower`.
fn is_lower(text: &str) -> bool {
    text.chars().any(char::is_lowercase) && !text.chars().any(char::is_uppercase)
}

fn capital_word_frequency(kwargs: &Value, _: &str, text: &str) -> bool {
    static TOKEN: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"[\w'’-]+").expect("a valid pattern"));
    let Some(target) = int(kwargs, "capital_frequency") else {
        return false;
    };
    let count = TOKEN
        .find_iter(text)
        .filter(|m| is_upper(m.as_str()))
        .count() as i64;
    relates(string(kwargs, "capital_relation"), count, target)
}

fn english_capital(_: &Value, _: &str, text: &str) -> bool {
    is_upper(text)
}

fn english_lowercase(_: &Value, _: &str, text: &str) -> bool {
    is_lower(text)
}

fn no_comma(_: &Value, _: &str, text: &str) -> bool {
    !text.contains(',')
}

// -- sentences ----------------------------------------------------------------------------

/// The sentences of `text`, by the reference's rules: periods that are not
/// sentence ends (titles, initials, acronyms, decimals, websites, `Ph.D`,
/// ellipses) are set aside before the text is split after `.`, `?` and `!`.
pub(crate) fn split_into_sentences(text: &str) -> Vec<String> {
    const STARTERS: &str = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)";
    let sub = |text: String, pattern: &str, with: &str| -> String {
        Regex::new(pattern)
            .expect("a valid pattern")
            .replace_all(&text, with)
            .into_owned()
    };
    let mut text = format!(" {text}  ").replace('\n', " ");
    text = sub(text, r"(Mr|St|Mrs|Ms|Dr)[.]", "${1}<prd>");
    text = sub(text, r"[.](com|net|org|io|gov|edu|me)", "<prd>${1}");
    text = sub(text, r"([0-9])[.]([0-9])", "${1}<prd>${2}");
    text = Regex::new(r"\.{2,}")
        .expect("a valid pattern")
        .replace_all(&text, |caps: &regex::Captures<'_>| {
            format!("{}<stop>", "<prd>".repeat(caps[0].len()))
        })
        .into_owned();
    if text.contains("Ph.D") {
        text = text.replace("Ph.D.", "Ph<prd>D<prd>");
    }
    text = sub(text, r"\s([A-Za-z])[.] ", " ${1}<prd> ");
    text = sub(
        text,
        &format!(r"([A-Z][.][A-Z][.](?:[A-Z][.])?) {STARTERS}"),
        "${1}<stop> ${2}",
    );
    text = sub(
        text,
        r"([A-Za-z])[.]([A-Za-z])[.]([A-Za-z])[.]",
        "${1}<prd>${2}<prd>${3}<prd>",
    );
    text = sub(text, r"([A-Za-z])[.]([A-Za-z])[.]", "${1}<prd>${2}<prd>");
    text = sub(
        text,
        &format!(r" (Inc|Ltd|Jr|Sr|Co)[.] {STARTERS}"),
        " ${1}<stop> ${2}",
    );
    text = sub(text, r" (Inc|Ltd|Jr|Sr|Co)[.]", " ${1}<prd>");
    text = sub(text, r" ([A-Za-z])[.]", " ${1}<prd>");
    if text.contains('”') {
        text = text.replace(".”", "”.");
    }
    if text.contains('"') {
        text = text.replace(".\"", "\".");
    }
    if text.contains('!') {
        text = text.replace("!\"", "\"!");
    }
    if text.contains('?') {
        text = text.replace("?\"", "\"?");
    }
    text = text.replace('.', ".<stop>");
    text = text.replace('?', "?<stop>");
    text = text.replace('!', "!<stop>");
    text = text.replace("<prd>", ".");
    let mut sentences: Vec<String> = text.split("<stop>").map(|s| s.trim().to_string()).collect();
    if sentences.last().is_some_and(String::is_empty) {
        sentences.pop();
    }
    sentences
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn row(ids: &[&str], kwargs: Value) -> Prompt {
        Prompt::from_row(&json!({"instruction_id_list": ids, "kwargs": kwargs})).unwrap()
    }

    /// Whether `response` follows the one instruction `id` with `kwargs`, strictly.
    fn strict(id: &str, kwargs: Value, response: &str) -> bool {
        row(&[id], json!([kwargs]))
            .score("the prompt", response)
            .strict[0]
    }

    #[test]
    fn sentences_are_split_by_the_references_rules() {
        let count = |text: &str| split_into_sentences(text).len();
        assert_eq!(count("One. Two! Three?"), 3);
        assert_eq!(count("Dr. Smith went to the U.S.A. today. He left."), 2);
        assert_eq!(count("It costs 3.50 dollars. Visit example.com now."), 2);
        assert_eq!(count("Wait... what? Yes."), 3);
        assert_eq!(count("No terminal punctuation"), 1);
        assert_eq!(count("She said \"stop.\" Then left."), 2);
        assert_eq!(
            split_into_sentences("Hello there. How are you?"),
            ["Hello there.", "How are you?"]
        );
    }

    #[test]
    fn keywords_and_counts_follow_their_relation() {
        assert!(strict(
            "keywords:existence",
            json!({"keywords": ["Sun", "moon"]}),
            "the SUN and the Moon"
        ));
        assert!(!strict(
            "keywords:existence",
            json!({"keywords": ["sun", "star"]}),
            "the sun"
        ));
        let freq = |relation: &str, n: i64| json!({"keyword": "cat", "frequency": n, "relation": relation});
        assert!(strict(
            "keywords:frequency",
            freq("at least", 2),
            "Cat cat dog"
        ));
        assert!(!strict(
            "keywords:frequency",
            freq("less than", 2),
            "Cat cat dog"
        ));
        assert!(strict(
            "keywords:frequency",
            freq("less than", 3),
            "Cat cat dog"
        ));
        assert!(strict(
            "keywords:forbidden_words",
            json!({"forbidden_words": ["bad"]}),
            "all good here"
        ));
        assert!(!strict(
            "keywords:forbidden_words",
            json!({"forbidden_words": ["bad"]}),
            "a BAD day"
        ));
        // A forbidden word is a whole word.
        assert!(strict(
            "keywords:forbidden_words",
            json!({"forbidden_words": ["bad"]}),
            "badge"
        ));
        let letters = json!({"letter": "e", "let_frequency": 3, "let_relation": "at least"});
        assert!(strict(
            "keywords:letter_frequency",
            letters.clone(),
            "Eerie sleep"
        ));
        assert!(!strict(
            "keywords:letter_frequency",
            letters,
            "no vowels hrd"
        ));
    }

    #[test]
    fn length_instructions_count_words_sentences_and_paragraphs() {
        let words = |relation: &str, n: i64| json!({"num_words": n, "relation": relation});
        assert!(strict(
            "length_constraints:number_words",
            words("at least", 4),
            "one two three four"
        ));
        assert!(!strict(
            "length_constraints:number_words",
            words("less than", 4),
            "one two three four"
        ));
        let sentences = json!({"num_sentences": 3, "relation": "at least"});
        assert!(strict(
            "length_constraints:number_sentences",
            sentences,
            "One. Two. Three."
        ));
        // Paragraphs are separated by ***; an empty end is the separator's.
        let paragraphs = json!({"num_paragraphs": 2});
        assert!(strict(
            "length_constraints:number_paragraphs",
            paragraphs.clone(),
            "one *** two"
        ));
        assert!(!strict(
            "length_constraints:number_paragraphs",
            paragraphs.clone(),
            "one *** *** two"
        ));
        assert!(!strict(
            "length_constraints:number_paragraphs",
            paragraphs,
            "just one"
        ));
        let nth = json!({"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "second"});
        assert!(strict(
            "length_constraints:nth_paragraph_first_word",
            nth.clone(),
            "First one.\n\n\"Second, really.\""
        ));
        assert!(!strict(
            "length_constraints:nth_paragraph_first_word",
            nth,
            "First one.\n\nThird two."
        ));
    }

    #[test]
    fn format_instructions_check_the_markup() {
        assert!(strict(
            "detectable_content:number_placeholders",
            json!({"num_placeholders": 2}),
            "Dear [name], at [address]"
        ));
        assert!(!strict(
            "detectable_content:number_placeholders",
            json!({"num_placeholders": 2}),
            "Dear [name]"
        ));
        assert!(strict(
            "detectable_content:postscript",
            json!({"postscript_marker": "P.S."}),
            "Body.\nP.S. remember"
        ));
        assert!(!strict(
            "detectable_content:postscript",
            json!({"postscript_marker": "P.S."}),
            "Body."
        ));
        let bullets = json!({"num_bullets": 2});
        assert!(strict(
            "detectable_format:number_bullet_lists",
            bullets.clone(),
            "* one\n- two"
        ));
        assert!(!strict(
            "detectable_format:number_bullet_lists",
            bullets,
            "* one\n* two\n* three"
        ));
        assert!(strict(
            "detectable_format:constrained_response",
            json!({}),
            "Hmm. My answer is maybe."
        ));
        assert!(strict(
            "detectable_format:number_highlighted_sections",
            json!({"num_highlights": 2}),
            "*one* and **two** here"
        ));
        assert!(!strict(
            "detectable_format:number_highlighted_sections",
            json!({"num_highlights": 2}),
            "*one* only ** **"
        ));
        let sections = json!({"section_spliter": "SECTION", "num_sections": 2});
        assert!(strict(
            "detectable_format:multiple_sections",
            sections.clone(),
            "SECTION 1 a SECTION 2 b SECTION 3 c"
        ));
        assert!(!strict(
            "detectable_format:multiple_sections",
            sections,
            "SECTION 1 a"
        ));
        assert!(strict(
            "detectable_format:json_format",
            json!({}),
            "```json\n{\"a\": [1, 2]}\n```"
        ));
        assert!(!strict(
            "detectable_format:json_format",
            json!({}),
            "Here: {\"a\": 1}"
        ));
        assert!(strict(
            "detectable_format:title",
            json!({}),
            "<<A Title>>\ntext"
        ));
        assert!(!strict("detectable_format:title", json!({}), "<< >>\ntext"));
    }

    #[test]
    fn combination_and_boundary_instructions() {
        assert!(strict(
            "combination:two_responses",
            json!({}),
            "first ****** second"
        ));
        assert!(!strict(
            "combination:two_responses",
            json!({}),
            "same ****** same"
        ));
        assert!(!strict("combination:two_responses", json!({}), "only one"));
        assert!(strict(
            "combination:repeat_prompt",
            json!({"prompt_to_repeat": "Say hi"}),
            "say HI. Then more"
        ));
        assert!(!strict(
            "combination:repeat_prompt",
            json!({"prompt_to_repeat": "Say hi"}),
            "Then say hi"
        ));
        assert!(strict(
            "startend:end_checker",
            json!({"end_phrase": "Is there more?"}),
            "Fine. \"Is there more?\""
        ));
        assert!(!strict(
            "startend:end_checker",
            json!({"end_phrase": "Is there more?"}),
            "Is there more? Yes"
        ));
        assert!(strict("startend:quotation", json!({}), "  \"wrapped\" "));
        assert!(!strict("startend:quotation", json!({}), "\"only opens"));
        assert!(strict("punctuation:no_comma", json!({}), "no commas here"));
        assert!(!strict("punctuation:no_comma", json!({}), "one, two"));
    }

    #[test]
    fn case_instructions() {
        assert!(strict(
            "change_case:english_capital",
            json!({}),
            "ALL CAPS, 1 2."
        ));
        assert!(!strict("change_case:english_capital", json!({}), "Not ALL"));
        assert!(strict(
            "change_case:english_lowercase",
            json!({}),
            "all lower 123"
        ));
        assert!(!strict(
            "change_case:english_lowercase",
            json!({}),
            "Not lower"
        ));
        // Digits alone have no case.
        assert!(!strict("change_case:english_lowercase", json!({}), "123"));
        let caps = json!({"capital_frequency": 2, "capital_relation": "at least"});
        assert!(strict(
            "change_case:capital_word_frequency",
            caps.clone(),
            "THE cat and A DOG"
        ));
        assert!(!strict(
            "change_case:capital_word_frequency",
            caps,
            "The cat and A dog"
        ));
    }

    #[test]
    fn loose_forgives_a_preamble_a_sign_off_and_markdown() {
        let prompt = row(
            &["punctuation:no_comma", "change_case:english_lowercase"],
            json!([{}, {}]),
        );
        // A "Sure," opener breaks both strictly; without its line both hold.
        let verdict = prompt.score("p", "Sure, here you go:\nthis has no commas\nbut ends here");
        assert_eq!(verdict.strict, [false, false]);
        assert_eq!(verdict.loose, [true, true]);
        // Asterisks around words are markdown, not the response's case.
        let bold = row(&["change_case:english_lowercase"], json!([{}]));
        let verdict = bold.score("p", "**some** bold text");
        assert_eq!((verdict.strict[0], verdict.loose[0]), (true, true));
        // An empty response follows nothing.
        let verdict = prompt.score("p", "  ");
        assert_eq!(
            (verdict.strict.clone(), verdict.loose.clone()),
            (vec![false, false], vec![false, false])
        );
    }

    #[test]
    fn rows_carry_instructions_with_their_non_null_parameters() {
        let row = json!({
            "key": 1, "prompt": "p",
            "instruction_id_list": ["keywords:existence", "language:response_language"],
            "kwargs": [{"keywords": ["a"], "frequency": null}, {"language": "de"}]
        });
        let prompt = Prompt::from_row(&row).unwrap();
        assert_eq!(prompt.instructions[0].kwargs, json!({"keywords": ["a"]}));
        // One instruction with no checker leaves the prompt unscored.
        assert!(!prompt.supported());
        assert!(Prompt::from_row(&json!({"instruction_id_list": ["keywords:existence"], "kwargs": [{"keywords": ["a"]}]}))
            .unwrap()
            .supported());
        assert!(Prompt::from_row(&json!({"prompt": "no list"})).is_err());
        assert!(
            !Prompt::from_row(&json!({"instruction_id_list": ["made:up"], "kwargs": [{}]}))
                .unwrap()
                .supported()
        );
    }
}
