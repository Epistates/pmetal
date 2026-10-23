//! Reward functions for GRPO and how they compose.
//!
//! A [`RewardFunction`] scores the completions of one prompt, a group at a
//! time. Verifiable rewards ([`AccuracyReward`]) compare against the prompt's
//! reference answer, which reaches them through
//! [`RewardFunction::compute_with_references`]: the dataset row's answer is
//! carried next to the prompt, so a reward never has to guess which prompt it
//! is scoring. [`CombinedReward`] is their weighted sum.
//!
//! Rewards need not be built in. [`ExternalCommandReward`] scores with a
//! long-lived process of the user's, [`RegexReward`] and [`JsonReward`] are
//! verifiers over the completion's text, and a binding can implement
//! [`RewardFunction`] over its own callables. All of them see the same
//! [`RewardBatch`]: the prompts, the completions and each completion's
//! dataset row.

mod external;
mod verifiers;

pub use external::{ExternalCommandReward, ExternalRewardConfig};
pub use verifiers::{JsonReward, RegexReward};

use std::sync::Arc;

use pmetal_runtime::compat::Array;

use super::{GrpoError, GrpoResult};

/// A dataset row's columns, as the row had them.
pub type DatasetRow = Arc<serde_json::Map<String, serde_json::Value>>;

/// One batch of completions to score, with everything a reward may look at.
///
/// Every slice has one entry per completion, except `columns`, which is empty
/// when the caller has no dataset rows to give (a reward that needs a column
/// then fails on its own terms, naming it).
#[derive(Clone, Copy)]
pub struct RewardBatch<'a> {
    /// The prompt each completion answers, as the model saw it (chat
    /// template applied).
    pub prompts: &'a [String],
    /// The completions' text.
    pub completions: &'a [String],
    /// Each completion's reference answer (`None` where the dataset has none).
    pub references: &'a [Option<String>],
    /// Each completion's dataset row, minus the prompt columns and images.
    pub columns: &'a [DatasetRow],
    /// Images the prompts refer to (vision-language models).
    pub images: Option<&'a [Vec<Array>]>,
}

impl RewardBatch<'_> {
    /// Check the dataset rows, when there are any, are one per completion.
    pub(crate) fn validate(&self) -> GrpoResult<()> {
        if !self.columns.is_empty() && self.columns.len() != self.completions.len() {
            return Err(GrpoError::Reward(format!(
                "a reward batch has {} completions but {} dataset rows",
                self.completions.len(),
                self.columns.len()
            )));
        }
        Ok(())
    }
}

/// Trait for GRPO reward functions.
///
/// `prompts`, `completions` and `references` have one entry per completion.
pub trait RewardFunction: Send + Sync {
    /// Score `completions`, given the `prompts` they answer.
    fn compute(
        &self,
        prompts: &[String],
        completions: &[String],
        images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>>;

    /// Score `completions` given each one's reference answer (`None` where
    /// the dataset has none). Rewards that need no reference ignore them, as
    /// this default does.
    fn compute_with_references(
        &self,
        prompts: &[String],
        completions: &[String],
        _references: &[Option<String>],
        images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        self.compute(prompts, completions, images)
    }

    /// Score a whole [`RewardBatch`]. Rewards that read the dataset's columns
    /// override this; the default is
    /// [`compute_with_references`](Self::compute_with_references).
    fn score(&self, batch: &RewardBatch<'_>) -> GrpoResult<Vec<f64>> {
        self.compute_with_references(
            batch.prompts,
            batch.completions,
            batch.references,
            batch.images,
        )
    }

    /// A short name, for logs.
    fn name(&self) -> &str;
}

/// Reward function that checks for proper XML tags (e.g., `<think>` and
/// `<answer>`): half a point for every tag pair present and in order.
pub struct XmlFormatReward {
    /// `(opening tag, closing tag)` pairs.
    pub tags: Vec<(String, String)>,
}

impl XmlFormatReward {
    /// A format reward over `tags`.
    pub fn new(tags: Vec<(String, String)>) -> Self {
        Self { tags }
    }

    /// `<think>…</think><answer>…</answer>`, the layout reasoning models
    /// (DeepSeek-R1, Qwen3) are prompted for.
    pub fn default_reasoning() -> Self {
        Self::new(vec![
            ("<think>".into(), "</think>".into()),
            ("<answer>".into(), "</answer>".into()),
        ])
    }
}

impl RewardFunction for XmlFormatReward {
    fn compute(
        &self,
        _: &[String],
        completions: &[String],
        _: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        Ok(completions
            .iter()
            .map(|completion| {
                self.tags
                    .iter()
                    .filter(|(start_tag, end_tag)| {
                        matches!(
                            (completion.find(start_tag), completion.find(end_tag)),
                            (Some(start), Some(end)) if start < end
                        )
                    })
                    .count() as f64
                    * 0.5
            })
            .collect())
    }

    fn name(&self) -> &str {
        "xml_format"
    }
}

/// A small shaping reward on a completion's length in characters: `-0.5` under
/// ten, `-0.1` over `max_chars`, `0.1` in between. It keeps a format reward
/// from being satisfied by an empty or runaway completion.
pub struct LengthReward {
    max_chars: usize,
}

impl LengthReward {
    /// A length reward whose upper bound is `max_chars` characters.
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars }
    }
}

impl RewardFunction for LengthReward {
    fn compute(
        &self,
        _: &[String],
        completions: &[String],
        _: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        Ok(completions
            .iter()
            .map(|c| {
                let len = c.chars().count();
                if len > self.max_chars {
                    -0.1
                } else if len < 10 {
                    -0.5
                } else {
                    0.1
                }
            })
            .collect())
    }

    fn name(&self) -> &str {
        "length"
    }
}

/// Reward function that checks a completion's final answer against the
/// prompt's reference answer: `1.0` when they match, else `0.0`.
///
/// The answer is extracted with the first of these that applies:
/// 1. the last `<answer>...</answer>` pair (a retry inside a chain of thought
///    leaves the last one standing);
/// 2. the last `\boxed{...}` expression;
/// 3. the last non-empty line.
///
/// Two answers match when they are equal after collapsing whitespace, or when
/// both are numbers that are equal (`1,000` and `1000.0`, `$18` and `18`).
/// The reference answers come with the prompts
/// ([`RewardFunction::compute_with_references`]); scoring without one is an
/// error, since a reward of zero for every completion would train nothing and
/// look like a model that never gets it right.
#[derive(Debug, Clone, Default)]
pub struct AccuracyReward;

impl AccuracyReward {
    /// An accuracy reward.
    pub fn new() -> Self {
        Self
    }
}

/// Normalize whitespace for answer comparison: trim + collapse internal runs to single space.
fn normalize_answer(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A number from `s`, tolerating `$`, thousands separators, a `%` and a
/// trailing full stop.
fn parse_number(s: &str) -> Option<f64> {
    let cleaned: String = s
        .trim()
        .trim_end_matches('.')
        .chars()
        .filter(|c| !matches!(c, '$' | ',' | '%' | ' '))
        .collect();
    cleaned.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Whether `answer` is `reference`, as [`AccuracyReward`] compares them.
pub(crate) fn answers_match(answer: &str, reference: &str) -> bool {
    let (a, r) = (normalize_answer(answer), normalize_answer(reference));
    if a == r {
        return true;
    }
    match (parse_number(&a), parse_number(&r)) {
        (Some(x), Some(y)) => (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1.0),
        _ => false,
    }
}

/// Extract the model's answer from a completion string.
///
/// Tries (in order):
/// 1. Last `<answer>...</answer>` tag pair
/// 2. Last `\boxed{...}` expression (with brace-depth tracking)
/// 3. Last non-empty line
pub(crate) fn extract_answer(completion: &str) -> &str {
    // Strategy 1: Last <answer>...</answer> pair
    if let Some(end_pos) = completion.rfind("</answer>") {
        let search_region = &completion[..end_pos];
        if let Some(start_pos) = search_region.rfind("<answer>") {
            let content_start = start_pos + "<answer>".len();
            if content_start <= end_pos {
                return completion[content_start..end_pos].trim();
            }
        }
    }

    // Strategy 2: Last \boxed{...} with brace-depth tracking
    if let Some(boxed_pos) = completion.rfind("\\boxed{") {
        let brace_start = boxed_pos + "\\boxed{".len();
        let mut depth = 1i32;
        let mut end = brace_start;
        for (i, ch) in completion[brace_start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = brace_start + i;
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth == 0 {
            return completion[brace_start..end].trim();
        }
    }

    // Strategy 3: Last non-empty line
    for line in completion.lines().rev() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            return trimmed;
        }
    }

    completion.trim()
}

impl RewardFunction for AccuracyReward {
    fn compute(
        &self,
        _: &[String],
        _: &[String],
        _: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        Err(GrpoError::Reward(
            "the accuracy reward compares against each prompt's reference answer, and none was \
             supplied; give the dataset an `answer` column"
                .into(),
        ))
    }

    fn compute_with_references(
        &self,
        _: &[String],
        completions: &[String],
        references: &[Option<String>],
        _: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        if references.len() != completions.len() {
            return Err(GrpoError::Reward(format!(
                "{} reference answers for {} completions",
                references.len(),
                completions.len()
            )));
        }
        completions
            .iter()
            .zip(references)
            .map(|(completion, reference)| match reference {
                Some(reference) => Ok(f64::from(answers_match(
                    extract_answer(completion),
                    reference,
                ))),
                None => Err(GrpoError::Reward(
                    "a prompt has no reference answer for the accuracy reward; give every \
                     dataset row an `answer`"
                        .into(),
                )),
            })
            .collect()
    }

    fn name(&self) -> &str {
        "accuracy"
    }
}

/// Reward function that evaluates VLM completions for image-understanding quality.
///
/// Scores each completion by checking how many of the expected answer patterns
/// appear in the model's response.  The score is normalized to `[0.0, 1.0]`:
/// `0.0` means no expected pattern was found; `1.0` means all of them were.
///
/// This is designed for visual QA tasks where the dataset supplies a reference
/// answer list.  Images are accepted via the `images` parameter but are not
/// directly inspected by this reward — they were already used during generation.
///
/// # Example
/// ```no_run
/// use pmetal_trainer::{VlmAccuracyReward, RewardFunction};
/// let reward = VlmAccuracyReward::new(vec!["cat".into(), "orange".into()]);
/// let scores = reward.compute(&[], &["I see an orange cat.".into()], None).unwrap();
/// assert!((scores[0] - 1.0).abs() < 1e-6); // both patterns found
/// ```
pub struct VlmAccuracyReward {
    /// Expected answer patterns (case-insensitive substring matches).
    pub expected_answers: Vec<String>,
}

impl VlmAccuracyReward {
    /// Create a new `VlmAccuracyReward` from a list of expected answer strings.
    pub fn new(expected_answers: Vec<String>) -> Self {
        Self { expected_answers }
    }
}

impl RewardFunction for VlmAccuracyReward {
    /// Score completions against expected answer patterns.
    ///
    /// For each completion, counts how many `expected_answers` appear as
    /// case-insensitive substrings and divides by the total count.
    ///
    /// The `images` parameter is accepted for API compatibility but is not
    /// inspected here — visual context is already baked into the completions
    /// via the model's multimodal forward pass.
    fn compute(
        &self,
        _prompts: &[String],
        completions: &[String],
        _images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        let n = self.expected_answers.len();
        completions
            .iter()
            .map(|completion| {
                if n == 0 {
                    return Ok(0.0);
                }
                let lower = completion.to_lowercase();
                let hits = self
                    .expected_answers
                    .iter()
                    .filter(|ans| lower.contains(ans.to_lowercase().as_str()))
                    .count();
                Ok(hits as f64 / n as f64)
            })
            .collect()
    }

    fn name(&self) -> &str {
        "vlm_accuracy"
    }
}

/// A weighted sum of reward functions.
#[derive(Default)]
pub struct CombinedReward {
    /// The functions and their weights.
    pub functions: Vec<(Box<dyn RewardFunction>, f64)>,
}

impl CombinedReward {
    /// An empty combination; scoring with it is an error until a function is
    /// added.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add `function` with `weight`.
    pub fn add(mut self, function: Box<dyn RewardFunction>, weight: f64) -> Self {
        self.functions.push((function, weight));
        self
    }

    /// The heuristic rewards for a reasoning model: the `<think>`/`<answer>`
    /// format (weight 1), a length bound of `max_chars` characters (0.2), and,
    /// when the prompts carry reference answers, the accuracy of the final
    /// answer (1).
    pub fn reasoning(max_chars: usize, with_accuracy: bool) -> Self {
        let combined = Self::new()
            .add(Box::new(XmlFormatReward::default_reasoning()), 1.0)
            .add(Box::new(LengthReward::new(max_chars)), 0.2);
        if with_accuracy {
            combined.add(Box::new(AccuracyReward::new()), 1.0)
        } else {
            combined
        }
    }

    /// Whether no function is configured.
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }

    /// The weighted sum of the functions' rewards. Every function must
    /// return one finite reward per completion.
    pub fn compute(
        &self,
        prompts: &[String],
        completions: &[String],
        images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        let references = vec![None; completions.len()];
        self.compute_with_references(prompts, completions, &references, images)
    }

    /// [`compute`](Self::compute) with the prompts' reference answers.
    pub fn compute_with_references(
        &self,
        prompts: &[String],
        completions: &[String],
        references: &[Option<String>],
        images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        self.score(&RewardBatch {
            prompts,
            completions,
            references,
            columns: &[],
            images,
        })
    }

    /// The weighted sum of the functions' rewards for `batch`. Every function
    /// must return one finite reward per completion.
    pub fn score(&self, batch: &RewardBatch<'_>) -> GrpoResult<Vec<f64>> {
        if self.functions.is_empty() {
            return Err(GrpoError::Reward("No reward functions configured".into()));
        }
        batch.validate()?;
        let mut total = vec![0.0; batch.completions.len()];
        for (function, weight) in &self.functions {
            let rewards = function.score(batch)?;
            if rewards.len() != batch.completions.len() {
                return Err(GrpoError::Reward(format!(
                    "reward function '{}' returned {} rewards for {} completions",
                    function.name(),
                    rewards.len(),
                    batch.completions.len()
                )));
            }
            for (sum, reward) in total.iter_mut().zip(rewards) {
                if !reward.is_finite() {
                    return Err(GrpoError::Reward(format!(
                        "reward function '{}' returned {reward}",
                        function.name()
                    )));
                }
                *sum += reward * weight;
            }
        }
        Ok(total)
    }
}

impl RewardFunction for CombinedReward {
    fn compute(
        &self,
        prompts: &[String],
        completions: &[String],
        images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        CombinedReward::compute(self, prompts, completions, images)
    }

    fn compute_with_references(
        &self,
        prompts: &[String],
        completions: &[String],
        references: &[Option<String>],
        images: Option<&[Vec<Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        CombinedReward::compute_with_references(self, prompts, completions, references, images)
    }

    fn score(&self, batch: &RewardBatch<'_>) -> GrpoResult<Vec<f64>> {
        CombinedReward::score(self, batch)
    }

    fn name(&self) -> &str {
        "combined_reward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(text: &str) -> String {
        text.to_string()
    }

    fn refs(answers: &[&str]) -> Vec<Option<String>> {
        answers.iter().map(|a| Some(s(a))).collect()
    }

    #[test]
    fn the_xml_format_reward_scores_each_ordered_tag_pair() {
        let reward = XmlFormatReward::default_reasoning();
        let score = |text: &str| reward.compute(&[s("p")], &[s(text)], None).unwrap()[0];

        // Both pairs, in order.
        assert_eq!(
            score("<think>I need to think.</think><answer>42</answer>"),
            1.0
        );
        // Only the answer pair.
        assert_eq!(score("<answer>42</answer>"), 0.5);
        // Opening tags without closing ones.
        assert_eq!(score("<think>no close tag <answer>42"), 0.0);
        // A closing tag before its opening one does not count.
        assert_eq!(score("</think>content<think><answer>42</answer>"), 0.5);
        assert_eq!(score(""), 0.0);
    }

    #[test]
    fn the_length_reward_bounds_a_completion_from_both_sides() {
        let reward = LengthReward::new(20);
        let long = "x".repeat(21);
        let got = reward
            .compute(&[], &[s("short"), s("a fine length"), long], None)
            .unwrap();
        assert_eq!(got, vec![-0.5, 0.1, -0.1]);
    }

    #[test]
    fn the_accuracy_reward_compares_against_each_prompts_reference() {
        let reward = AccuracyReward::new();
        // Two prompts, two completions each: every completion is scored
        // against the answer of its own prompt.
        let completions = [
            s("<think>x</think><answer>42</answer>"),
            s("<answer>99</answer>"),
            s("so it is \\boxed{ 7 }"),
            s("the last line\n  hello  "),
        ];
        let references = refs(&["42", "42", "7", "hello"]);
        let got = reward
            .compute_with_references(&[], &completions, &references, None)
            .unwrap();
        assert_eq!(got, vec![1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn answers_match_as_text_or_as_numbers() {
        assert!(answers_match("1,000", "1000.0"));
        assert!(answers_match("$18", "18"));
        assert!(answers_match("18.", "18"));
        assert!(answers_match("a   b", "a b"));
        assert!(!answers_match("18", "19"));
        assert!(!answers_match("abc", "abd"));
        assert!(!answers_match("NaN", "nan"));
    }

    #[test]
    fn the_accuracy_reward_refuses_to_score_without_a_reference() {
        let reward = AccuracyReward::new();
        assert!(matches!(
            reward.compute(&[], &[s("42")], None),
            Err(GrpoError::Reward(_))
        ));
        assert!(matches!(
            reward.compute_with_references(&[], &[s("42")], &[None], None),
            Err(GrpoError::Reward(_))
        ));
        assert!(matches!(
            reward.compute_with_references(&[], &[s("42")], &[], None),
            Err(GrpoError::Reward(_))
        ));
    }

    #[test]
    fn a_combined_reward_is_the_weighted_sum_and_carries_the_references() {
        let combined = CombinedReward::new()
            .add(Box::new(XmlFormatReward::default_reasoning()), 0.5)
            .add(Box::new(AccuracyReward::new()), 0.5);
        let completions = [
            s("<think>some reasoning</think><answer>42</answer>"),
            s("<think>some reasoning</think><answer>99</answer>"),
            s("the answer is probably 7"),
        ];
        let references = refs(&["42", "42", "42"]);
        let rewards = combined
            .compute_with_references(&[], &completions, &references, None)
            .unwrap();
        // 0.5 * format + 0.5 * accuracy.
        assert_eq!(rewards, vec![1.0, 0.5, 0.0]);

        let empty = CombinedReward::new();
        assert!(matches!(
            empty.compute(&[], &[], None),
            Err(GrpoError::Reward(_))
        ));
    }

    /// A reward function that misreports its length or returns NaN is an
    /// error naming it, not an index panic or a NaN advantage.
    #[test]
    fn a_combined_reward_checks_what_its_functions_return() {
        struct Fixed(Vec<f64>);
        impl RewardFunction for Fixed {
            fn compute(
                &self,
                _: &[String],
                _: &[String],
                _: Option<&[Vec<Array>]>,
            ) -> GrpoResult<Vec<f64>> {
                Ok(self.0.clone())
            }
            fn name(&self) -> &str {
                "fixed"
            }
        }
        let short = CombinedReward::new().add(Box::new(Fixed(vec![1.0])), 1.0);
        assert!(short.compute(&[], &[s("a"), s("b")], None).is_err());
        let nan = CombinedReward::new().add(Box::new(Fixed(vec![f64::NAN])), 1.0);
        assert!(nan.compute(&[], &[s("a")], None).is_err());
    }
}
