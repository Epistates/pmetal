//! GSM8K's answer extraction and exact match.
//!
//! The standard 5-shot definition generates greedily until the next
//! `Question:` and reads the answer out of the text two ways:
//!
//! - **strict**: the number after `#### `, the format the few-shot answers end
//!   in (`#### (\-?[0-9\.\,]+)`, the first match);
//! - **flexible**: the last number anywhere in the text
//!   (`(-?[$0-9.,]{2,})|(-?[0-9]+)`, the last match), which is how a model that
//!   writes its answer in its own words is still read.
//!
//! An answer that matches nothing is `[invalid]`. Prediction and reference
//! are then compared by exact match after removing commas, dollar signs,
//! everything up to a `#### `, and a trailing full stop, ignoring case.

use std::sync::LazyLock;

use regex::Regex;

/// What an answer that matches no pattern is.
pub(crate) const INVALID: &str = "[invalid]";

/// The strings a generation stops at.
pub(crate) const STOP: [&str; 3] = ["Question:", "</s>", "<|im_end|>"];

static STRICT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"#### (-?[0-9.,]+)").expect("a valid pattern"));
static FLEXIBLE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(-?[$0-9.,]{2,})|(-?[0-9]+)").expect("a valid pattern"));

/// What the exact match removes from prediction and reference alike, in order.
static IGNORED: LazyLock<[Regex; 4]> = LazyLock::new(|| {
    [
        Regex::new(",").expect("a valid pattern"),
        Regex::new(r"\$").expect("a valid pattern"),
        Regex::new(r"(?s).*#### ").expect("a valid pattern"),
        Regex::new(r"\.$").expect("a valid pattern"),
    ]
});

/// The answer after `#### `: the first match, stripped.
pub(crate) fn strict_answer(response: &str) -> String {
    STRICT
        .captures(response)
        .and_then(|c| c.get(1))
        .map_or_else(|| INVALID.to_string(), |m| m.as_str().trim().to_string())
}

/// The last number in the response.
pub(crate) fn flexible_answer(response: &str) -> String {
    FLEXIBLE
        .captures_iter(response)
        .last()
        .and_then(|c| c.get(1).or_else(|| c.get(2)))
        .map_or_else(|| INVALID.to_string(), |m| m.as_str().trim().to_string())
}

fn normalize(text: &str) -> String {
    let mut text = text.to_string();
    for pattern in IGNORED.iter() {
        text = pattern.replace_all(&text, "").into_owned();
    }
    text.to_lowercase()
}

/// Whether `prediction` is `reference` by the task's exact match.
pub(crate) fn exact_match(prediction: &str, reference: &str) -> bool {
    normalize(prediction) == normalize(reference)
}

/// The text of a generation up to the first of `stops`.
pub(crate) fn truncate_at<'a>(text: &'a str, stops: &[&str]) -> &'a str {
    let end = stops
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min()
        .unwrap_or(text.len());
    &text[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFERENCE: &str = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72";

    #[test]
    fn strict_reads_the_number_after_the_marker_and_nothing_else() {
        assert_eq!(strict_answer("so it is\n#### 72"), "72");
        assert_eq!(strict_answer("#### -1,200.50 apples"), "-1,200.50");
        // The first marker counts.
        assert_eq!(strict_answer("#### 5\n#### 6"), "5");
        // No marker, no answer, however clear the text is.
        assert_eq!(strict_answer("The answer is 72."), INVALID);
        assert_eq!(strict_answer("#### many"), INVALID);
    }

    #[test]
    fn flexible_reads_the_last_number_in_the_text() {
        assert_eq!(flexible_answer("She sold 48 then 24 so 72 clips"), "72");
        assert_eq!(flexible_answer("The answer is $18."), "$18.");
        assert_eq!(flexible_answer("it costs -3 and then 7"), "7");
        // A one-digit number is a number; a lone punctuation mark is not.
        assert_eq!(flexible_answer("total: 7"), "7");
        assert_eq!(flexible_answer("no digits here, only words."), INVALID);
        assert_eq!(flexible_answer(""), INVALID);
    }

    #[test]
    fn exact_match_ignores_commas_dollars_the_marker_a_final_stop_and_case() {
        assert!(exact_match("72", REFERENCE));
        assert!(exact_match("$1,200", "work...\n#### 1200"));
        assert!(exact_match("1200.", "#### 1200"));
        assert!(exact_match("1200", "#### 1,200"));
        assert!(!exact_match("1201", "#### 1200"));
        assert!(!exact_match(INVALID, "#### 1200"));
        // Only a final stop goes: a decimal point in the middle stays.
        assert!(!exact_match("12.5", "#### 125"));
        assert!(exact_match("Yes", "#### yes"));
    }

    #[test]
    fn a_generation_is_cut_at_the_first_stop_string() {
        let text = "#### 72\n\nQuestion: next one\nAnswer: x";
        assert_eq!(truncate_at(text, &STOP), "#### 72\n\n");
        assert_eq!(truncate_at("no stop here", &STOP), "no stop here");
        assert_eq!(truncate_at("a</s>b<|im_end|>c", &STOP), "a");
        assert_eq!(truncate_at("Question:", &STOP), "");
    }

    #[test]
    fn the_standard_example_is_scored_the_standard_way() {
        // A response that follows the format.
        let response = " She sold 24 in May, so 48 + 24 = 72.\n#### 72\n\nQuestion: ...";
        let response = truncate_at(response, &STOP);
        assert!(exact_match(&strict_answer(response), REFERENCE));
        assert!(exact_match(&flexible_answer(response), REFERENCE));
        // One that only states it: strict misses, flexible finds it.
        let plain = " The total is 72 clips.";
        assert!(!exact_match(&strict_answer(plain), REFERENCE));
        assert!(exact_match(&flexible_answer(plain), REFERENCE));
    }
}
