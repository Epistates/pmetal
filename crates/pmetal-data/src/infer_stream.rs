//! Reading what `pmetal infer` prints while it generates.
//!
//! The command prints a banner, a line that says generation is starting
//! (`Generating...`, or `Loading model...` for the engines that load their
//! weights inside generation), the text as it is generated, and then a
//! `---` footer with statistics (commands/infer.rs). The chat tab shows the
//! text as it arrives and the statistics when they do. The reasoning of a
//! thinking model is in the text between the labels `pmetal_data`'s stream
//! formatter writes.
//!
//! The terminal UI's chat and the desktop app's both read a reply this way.

use crate::stream_format::{ANSWER_LABEL, THINKING_LABEL};

/// The lines that end the banner and start the text.
const START_LINES: &[&str] = &["Generating...", "Loading model..."];
/// What separates the text from the statistics.
const FOOTER: &str = "\n---\n";

/// What a finished generation reports.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GenerationStats {
    /// Tokens generated.
    pub tokens: usize,
    /// The measured decode rate, else tokens over the time shown.
    pub tokens_per_second: f64,
    /// Seconds spent generating after the first token. With
    /// [`first_token_seconds`](Self::first_token_seconds), the whole run.
    pub generating_seconds: f64,
    /// Seconds to the first token, model load included, when the footer says.
    pub first_token_seconds: Option<f64>,
}

/// Where the reader is in the output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Before the line that starts the text.
    Banner,
    /// In the generated text.
    Text,
    /// In the statistics after it.
    Footer,
}

/// Incremental reader of one generation's standard output.
#[derive(Debug)]
pub struct InferStream {
    phase: Phase,
    /// Text received and not yet decided: the banner's current line, or the
    /// tail of the text that might be the start of the footer.
    pending: String,
    /// The text of the generation, with what is pending still out of it.
    text: String,
    footer: String,
}

impl Default for InferStream {
    fn default() -> Self {
        Self::new()
    }
}

impl InferStream {
    /// A reader at the start of the output.
    pub fn new() -> Self {
        Self {
            phase: Phase::Banner,
            pending: String::new(),
            text: String::new(),
            footer: String::new(),
        }
    }

    /// Take the next piece of standard output. Returns the text to show after
    /// it (everything so far that is certainly the generation's).
    pub fn push(&mut self, chunk: &str) -> &str {
        match self.phase {
            Phase::Banner => {
                self.pending.push_str(chunk);
                // Banner lines end in newlines; the start line is whole when
                // its newline arrives.
                while let Some(end) = self.pending.find('\n') {
                    let line: String = self.pending.drain(..=end).collect();
                    if START_LINES.contains(&line.trim_end()) {
                        self.phase = Phase::Text;
                        let rest = std::mem::take(&mut self.pending);
                        self.push_text(&rest);
                        return &self.text;
                    }
                }
            }
            Phase::Text => self.push_text(chunk),
            Phase::Footer => self.footer.push_str(chunk),
        }
        &self.text
    }

    fn push_text(&mut self, chunk: &str) {
        self.pending.push_str(chunk);
        // The footer may be the very first thing (a generation of nothing).
        let search_from = if self.text.is_empty() {
            // Treat the start of the text as following a newline.
            let probe = format!("\n{}", self.pending);
            probe.find(FOOTER).map(|i| i.saturating_sub(1))
        } else {
            self.pending.find(FOOTER)
        };
        if let Some(at) = search_from {
            // Everything before the footer is the text; the newlines the
            // command prints after it (it ends the line, then a blank one)
            // are not.
            let (before, after) = self.pending.split_at(at.min(self.pending.len()));
            let after = after.trim_start_matches('\n');
            let after = after.strip_prefix("---\n").unwrap_or(after);
            self.text.push_str(before.trim_end_matches('\n'));
            self.footer = after.to_string();
            self.pending.clear();
            self.phase = Phase::Footer;
            return;
        }
        // Hold back what could still turn out to be the start of the footer.
        let hold = footer_prefix_len(&self.pending);
        let keep_from = self.pending.len() - hold;
        // Trailing newlines are held too: they are the text's end or its
        // body, and only the next character says which.
        let trailing_newlines = self.pending[..keep_from]
            .bytes()
            .rev()
            .take_while(|b| *b == b'\n')
            .count();
        let release = keep_from - trailing_newlines;
        self.text.push_str(&self.pending[..release]);
        self.pending.drain(..release);
    }

    /// The text of the generation so far.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The generation has ended (standard output closed): whatever is held
    /// back that is not a footer was text after all.
    pub fn finish(&mut self) -> GenerationStats {
        if self.phase == Phase::Text {
            let rest = std::mem::take(&mut self.pending);
            self.text.push_str(rest.trim_end_matches('\n'));
        }
        self.stats()
    }

    /// Whether the statistics footer has been reached.
    pub fn is_complete(&self) -> bool {
        self.phase == Phase::Footer
    }

    /// The statistics, as far as the footer has given them.
    pub fn stats(&self) -> GenerationStats {
        let mut stats = GenerationStats::default();
        let mut seconds = 0.0;
        for line in self.footer.lines() {
            if let Some(rest) = line.strip_prefix("Generated ") {
                // `Generated 42 tokens in 1.23s (...)`
                let mut words = rest.split_whitespace();
                stats.tokens = words.next().and_then(|w| w.parse().ok()).unwrap_or(0);
                seconds = rest
                    .split(" in ")
                    .nth(1)
                    .and_then(|r| r.split('s').next())
                    .and_then(|s| s.trim().parse::<f64>().ok())
                    .unwrap_or(0.0);
                stats.generating_seconds = seconds;
                // `(0.20s to the first token, model load included)`
                stats.first_token_seconds = rest
                    .split_once('(')
                    .and_then(|(_, r)| r.split_once("s to the first token"))
                    .and_then(|(s, _)| s.trim().parse::<f64>().ok());
            } else if let Some(rest) = line.strip_prefix("Decode: ") {
                // `Decode: 310 tok/s (...)`: the measured decode rate.
                if let Some(rate) = rest
                    .split_whitespace()
                    .next()
                    .and_then(|w| w.parse::<f64>().ok())
                {
                    stats.tokens_per_second = rate;
                }
            }
        }
        if stats.tokens_per_second == 0.0 && seconds > 0.0 {
            stats.tokens_per_second = stats.tokens as f64 / seconds;
        }
        stats
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of `marker`.
fn partial_marker_len(text: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .find(|&n| {
            text.len() >= n
                && text.is_char_boundary(text.len() - n)
                && marker.starts_with(&text[text.len() - n..])
        })
        .unwrap_or(0)
}

/// Length of the longest suffix of `text` that is a proper prefix of the
/// footer marker.
fn footer_prefix_len(text: &str) -> usize {
    partial_marker_len(text, FOOTER)
}

/// A generation's text split into the model's reasoning and its answer.
///
/// Text with no reasoning label is all answer. While the model is still
/// reasoning there is no answer yet.
pub fn split_reasoning(text: &str) -> (Option<&str>, &str) {
    let Some(body) = text.strip_prefix(THINKING_LABEL) else {
        return (None, text);
    };
    match body.split_once(ANSWER_LABEL) {
        Some((thinking, answer)) => (Some(thinking), answer),
        None => (Some(body), ""),
    }
}

/// [`split_reasoning`] for text that is still arriving: what could yet turn
/// out to be the start of a label is held back, so the reasoning never shows
/// half of the `[answer]` label that ends it, and a reply that opens with
/// half of the `[thinking]` label shows nothing yet.
pub fn split_reasoning_streaming(text: &str) -> (Option<&str>, &str) {
    if !text.is_empty() && text.len() < THINKING_LABEL.len() && THINKING_LABEL.starts_with(text) {
        return (None, "");
    }
    let Some(body) = text.strip_prefix(THINKING_LABEL) else {
        return (None, text);
    };
    match body.split_once(ANSWER_LABEL) {
        Some((thinking, answer)) => (Some(thinking), answer),
        None => {
            let settled = body.len() - partial_marker_len(body, ANSWER_LABEL);
            (Some(&body[..settled]), "")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the command prints for one reply, as commands/infer.rs prints it.
    fn output(start: &str, text: &str, footer: &str) -> String {
        format!(
            "========================================\n  PMetal Inference\n  Max tokens:  64\n========================================\n\n{start}\n{text}\n\n---\n{footer}"
        )
    }

    const FOOTER_TEXT: &str = "Generated 42 tokens in 1.50s (0.20s to the first token, model load included)\nStopped by: EOS token\nDecode: 310 tok/s (avg=3.20ms p50=3.10ms over 41 steps)\n";

    fn read_in_pieces(out: &str, piece: usize) -> InferStream {
        let mut stream = InferStream::new();
        let bytes: Vec<char> = out.chars().collect();
        for chunk in bytes.chunks(piece) {
            let chunk: String = chunk.iter().collect();
            stream.push(&chunk);
        }
        stream.finish();
        stream
    }

    #[test]
    fn the_text_is_what_lies_between_the_start_line_and_the_footer() {
        let out = output("Generating...", "The capital is Paris.", FOOTER_TEXT);
        let stream = read_in_pieces(&out, usize::MAX);
        assert_eq!(stream.text(), "The capital is Paris.");
        assert!(stream.is_complete());
        // The same however the output is cut into reads.
        for piece in [1, 2, 3, 5, 7, 64] {
            let stream = read_in_pieces(&out, piece);
            assert_eq!(stream.text(), "The capital is Paris.", "piece {piece}");
            assert_eq!(stream.stats().tokens, 42, "piece {piece}");
        }
    }

    #[test]
    fn engines_that_load_inside_generation_start_the_text_the_same_way() {
        // These print `Loading model...` and never `Generating...`: the old
        // reader waited for the second line forever and showed nothing.
        let out = output(
            "Loading model...",
            "Hello there",
            "Generated 3 tokens in 0.50s\n",
        );
        let stream = read_in_pieces(&out, 4);
        assert_eq!(stream.text(), "Hello there");
        assert_eq!(stream.stats().tokens, 3);
    }

    #[test]
    fn statistics_come_from_the_footer_the_command_really_prints() {
        let stream = read_in_pieces(&output("Generating...", "x", FOOTER_TEXT), 9);
        let stats = stream.stats();
        assert_eq!(stats.tokens, 42);
        // The measured decode rate, not tokens over wall time.
        assert_eq!(stats.tokens_per_second, 310.0);
        // Without the decode line the rate is worked out from the time shown.
        let stream = read_in_pieces(
            &output("Generating...", "x", "Generated 30 tokens in 2.00s\n"),
            9,
        );
        assert_eq!(stream.stats().tokens_per_second, 15.0);
    }

    #[test]
    fn text_that_contains_dashes_and_blank_lines_is_not_cut_short() {
        let text = "Item one\n\n---\nnot a footer? It is: the marker is a line of three dashes";
        // A line of three dashes IS the footer marker; a reply that contains
        // one in the middle is cut there (the command offers no escape).
        let stream = read_in_pieces(&output("Generating...", text, FOOTER_TEXT), 11);
        assert_eq!(stream.text(), "Item one");
        // Dashes that are not on a line of their own are text.
        let text = "a --- b\n----\nc";
        let stream = read_in_pieces(&output("Generating...", text, FOOTER_TEXT), 3);
        assert_eq!(stream.text(), text);
        // Interior blank lines stay.
        let text = "first\n\nsecond";
        let stream = read_in_pieces(&output("Generating...", text, FOOTER_TEXT), 2);
        assert_eq!(stream.text(), text);
    }

    #[test]
    fn a_reply_of_nothing_is_still_a_complete_generation() {
        let out = "Generating...\n\n\n---\nGenerated 0 tokens in 0.10s\n";
        let stream = read_in_pieces(out, 3);
        assert_eq!(stream.text(), "");
        assert!(stream.is_complete());
    }

    #[test]
    fn output_cut_short_keeps_the_text_that_arrived() {
        // The process was stopped part-way: no footer, text still shown.
        let mut stream = InferStream::new();
        stream.push("banner\nGenerating...\nPartial answ");
        assert_eq!(stream.text(), "Partial answ");
        let stats = stream.finish();
        assert_eq!(stream.text(), "Partial answ");
        assert!(!stream.is_complete());
        assert_eq!(stats, GenerationStats::default());
        // Text held back as a possible footer start is not lost.
        let mut stream = InferStream::new();
        stream.push("Generating...\nanswer\n-");
        stream.finish();
        assert_eq!(stream.text(), "answer\n-");
    }

    #[test]
    fn the_text_streams_while_the_footer_is_still_a_possibility() {
        let mut stream = InferStream::new();
        stream.push("Generating...\n");
        assert_eq!(stream.push("Hel"), "Hel");
        assert_eq!(
            stream.push("lo\n"),
            "Hel".to_owned() + "lo".to_string().as_str()
        );
        assert_eq!(stream.push("wor"), "Hello\nwor");
    }

    #[test]
    fn reasoning_is_split_on_the_formatters_own_labels() {
        let text = format!("{THINKING_LABEL}plan the answer{ANSWER_LABEL}Paris");
        assert_eq!(split_reasoning(&text), (Some("plan the answer"), "Paris"));
        // Still reasoning: no answer yet.
        let text = format!("{THINKING_LABEL}hmm");
        assert_eq!(split_reasoning(&text), (Some("hmm"), ""));
        assert_eq!(split_reasoning("just an answer"), (None, "just an answer"));
    }

    #[test]
    fn streaming_reasoning_never_shows_half_a_label() {
        // The reply opens with part of the thinking label: nothing is known yet.
        for cut in 1..THINKING_LABEL.len() {
            assert_eq!(
                split_reasoning_streaming(&THINKING_LABEL[..cut]),
                (None, "")
            );
        }
        assert_eq!(
            split_reasoning_streaming("[th is text"),
            (None, "[th is text")
        );
        // The answer label arriving: the reasoning stops short of it.
        let whole = format!("{THINKING_LABEL}plan{ANSWER_LABEL}Paris");
        let at = THINKING_LABEL.len() + "plan".len();
        for cut in at..whole.len() {
            let (thinking, answer) = split_reasoning_streaming(&whole[..cut]);
            assert_eq!(thinking, Some("plan"), "cut {cut}");
            assert!("Paris".starts_with(answer), "cut {cut}: {answer:?}");
        }
        assert_eq!(split_reasoning_streaming(&whole), (Some("plan"), "Paris"));
        // Text that is not reasoning is all answer, as it arrives.
        assert_eq!(split_reasoning_streaming("Par"), (None, "Par"));
    }

    #[test]
    fn the_footer_gives_the_time_to_the_first_token() {
        let stream = read_in_pieces(&output("Generating...", "x", FOOTER_TEXT), 9);
        let stats = stream.stats();
        assert_eq!(stats.first_token_seconds, Some(0.20));
        assert_eq!(stats.generating_seconds, 1.50);
        let stream = read_in_pieces(
            &output("Generating...", "x", "Generated 30 tokens in 2.00s\n"),
            9,
        );
        assert_eq!(stream.stats().first_token_seconds, None);
    }
}
