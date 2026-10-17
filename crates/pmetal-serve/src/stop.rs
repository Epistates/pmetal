//! Stop sequences, matched in the text.
//!
//! A stop sequence is a string, not a run of tokens: `END` can arrive as
//! ` E` then `ND`, or inside a token that goes on past it (` ENDING`). The
//! text before the match belongs to the reply however the tokens fell, and the
//! match and everything after it do not. [`StopFilter`] sees the reply's text
//! as it is decoded and passes on what is safe to show: the text up to the
//! first stop sequence, holding back the tail that could still grow into one.

/// Cuts a stream of text at the first of a set of stop sequences.
#[derive(Debug, Clone)]
pub struct StopFilter {
    stops: Vec<String>,
    /// The longest stop sequence's length in bytes.
    longest: usize,
    /// Text received and not yet released: it could be the start of a stop
    /// sequence.
    held: String,
    /// The stop sequence that ended the text, once one has.
    matched: Option<String>,
}

impl StopFilter {
    /// A filter for `stops`; empty strings are ignored.
    pub fn new(stops: &[String]) -> Self {
        let stops: Vec<String> = stops.iter().filter(|s| !s.is_empty()).cloned().collect();
        let longest = stops.iter().map(String::len).max().unwrap_or(0);
        Self {
            stops,
            longest,
            held: String::new(),
            matched: None,
        }
    }

    /// Whether a stop sequence has been seen. Text pushed after that is
    /// dropped.
    pub fn stopped(&self) -> bool {
        self.matched.is_some()
    }

    /// The stop sequence that was seen (the longest, when several begin at
    /// the same place).
    pub fn matched(&self) -> Option<&str> {
        self.matched.as_deref()
    }

    /// Take in `text`; returns the part of the reply that can be shown.
    pub fn push(&mut self, text: &str) -> String {
        if self.stopped() {
            return String::new();
        }
        if self.stops.is_empty() {
            return text.to_owned();
        }
        self.held.push_str(text);
        if let Some((at, stop)) = self
            .stops
            .iter()
            .filter_map(|stop| self.held.find(stop.as_str()).map(|at| (at, stop)))
            .min_by_key(|(at, stop)| (*at, std::cmp::Reverse(stop.len())))
        {
            self.matched = Some(stop.clone());
            self.held.truncate(at);
            return std::mem::take(&mut self.held);
        }
        // Hold back the longest tail that is the beginning of a stop
        // sequence; a tail at most `longest - 1` bytes long can be.
        let from = self
            .held
            .len()
            .saturating_sub(self.longest.saturating_sub(1));
        let keep_from = (from..self.held.len())
            .filter(|&i| self.held.is_char_boundary(i))
            .find(|&i| {
                let tail = &self.held[i..];
                self.stops.iter().any(|stop| stop.starts_with(tail))
            })
            .unwrap_or(self.held.len());
        let kept = self.held.split_off(keep_from);
        std::mem::replace(&mut self.held, kept)
    }

    /// The reply ended without completing a stop sequence: release what was
    /// held.
    pub fn finish(&mut self) -> String {
        if self.stopped() {
            return String::new();
        }
        std::mem::take(&mut self.held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let mut filter = StopFilter::new(&stops);
        let mut out = String::new();
        for piece in pieces {
            out.push_str(&filter.push(piece));
        }
        out.push_str(&filter.finish());
        (out, filter.stopped())
    }

    #[test]
    fn text_before_a_stop_is_kept_whole() {
        // The stop starts in the middle of a piece: the pre-stop text of that
        // piece is still the reply's.
        assert_eq!(run(&["END"], &["hello ENDING"]), ("hello ".into(), true));
        assert_eq!(
            run(&["END"], &["hello", " E", "ND", " more"]),
            ("hello ".into(), true)
        );
        assert_eq!(run(&["\n\n"], &["a\n", "\nb"]), ("a".into(), true));
    }

    #[test]
    fn a_partial_stop_is_held_and_released_when_it_does_not_complete() {
        let stops = vec!["END".to_string()];
        let mut filter = StopFilter::new(&stops);
        assert_eq!(filter.push("an E"), "an ");
        assert_eq!(filter.push("N"), "");
        assert_eq!(filter.push("X"), "ENX");
        assert_eq!(filter.finish(), "");
        assert!(!filter.stopped());
        // Held at the end of the reply, released by finish.
        let mut filter = StopFilter::new(&stops);
        assert_eq!(filter.push("an EN"), "an ");
        assert_eq!(filter.finish(), "EN");
    }

    #[test]
    fn the_earliest_of_several_stops_wins() {
        assert_eq!(
            run(&["two", "one"], &["zero one two"]),
            ("zero ".into(), true)
        );
        assert_eq!(run(&["aaa", "aab"], &["xaab"]), ("x".into(), true));
    }

    #[test]
    fn multi_byte_text_is_never_split() {
        assert_eq!(
            run(&["\u{4e16}\u{754c}"], &["\u{4f60}\u{4e16}", "\u{754c}!"]),
            ("\u{4f60}".into(), true)
        );
        let (out, stopped) = run(&["stop"], &["caf\u{e9} \u{1f980} s", "top"]);
        assert_eq!((out.as_str(), stopped), ("caf\u{e9} \u{1f980} ", true));
    }

    #[test]
    fn no_stops_pass_everything() {
        assert_eq!(run(&[], &["a", "b"]), ("ab".into(), false));
        assert_eq!(run(&[""], &["a", "b"]), ("ab".into(), false));
    }

    #[test]
    fn text_after_the_stop_is_dropped() {
        let mut filter = StopFilter::new(&["x".to_string()]);
        assert_eq!(filter.push("axb"), "a");
        assert_eq!(filter.push("more"), "");
        assert_eq!(filter.finish(), "");
    }

    /// Whatever the pieces, the output is the full text cut at the first
    /// stop: checked against the cut of the concatenation.
    #[test]
    fn splitting_does_not_change_the_cut() {
        let text = "the quick STOP brown fox ST jumped OVER";
        let stops = ["STOP".to_string(), "OVER".to_string()];
        let want = "the quick ";
        for size in 1..=text.len() {
            let mut filter = StopFilter::new(&stops);
            let mut out = String::new();
            for chunk in text.as_bytes().chunks(size) {
                out.push_str(&filter.push(std::str::from_utf8(chunk).unwrap()));
            }
            out.push_str(&filter.finish());
            assert_eq!(out, want, "chunks of {size}");
        }
    }
}
