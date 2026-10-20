//! A job's output: ordered, bounded by lines, bytes and line length.

use std::collections::VecDeque;

use serde::Serialize;

/// Most lines kept per job (both streams together).
pub const MAX_LOG_LINES: usize = 10_000;
/// Most bytes kept per job (both streams together).
pub const MAX_LOG_BYTES: usize = 4 * 1024 * 1024;
/// Longest line kept whole; the rest of a longer line is dropped.
pub const MAX_LINE_BYTES: usize = 16 * 1024;

/// Which stream a line of output came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stream {
    /// Standard output.
    Stdout,
    /// Standard error.
    Stderr,
}

/// One line of a job's output.
#[derive(Debug, Clone)]
pub struct LogLine {
    /// Position in the job's output, counted from 0 over every line it has
    /// printed, including ones no longer kept.
    pub seq: u64,
    /// The stream it was printed on.
    pub stream: Stream,
    /// The line, without its newline, redacted and safe to show.
    pub text: String,
}

/// Lines of a job's output in arrival order, bounded by count and bytes.
#[derive(Debug, Default)]
pub(crate) struct LogBuffer {
    lines: VecDeque<LogLine>,
    bytes: usize,
    next_seq: u64,
}

impl LogBuffer {
    /// Add a line, dropping the oldest while the buffer is over its bounds.
    /// Returns the sequence number the line was given.
    pub(crate) fn push(&mut self, stream: Stream, text: String) -> u64 {
        let seq = self.next_seq;
        self.bytes += text.len();
        self.lines.push_back(LogLine { seq, stream, text });
        self.next_seq += 1;
        while self.lines.len() > MAX_LOG_LINES || self.bytes > MAX_LOG_BYTES {
            match self.lines.pop_front() {
                Some(dropped) => self.bytes -= dropped.text.len(),
                None => break,
            }
        }
        seq
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lines.len()
    }

    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    #[cfg(test)]
    pub(crate) fn next_seq(&self) -> u64 {
        self.next_seq
    }

    pub(crate) fn first_seq(&self) -> Option<u64> {
        self.lines.front().map(|l| l.seq)
    }

    /// The last line `stream` printed that has something on it.
    pub(crate) fn last_nonblank(&self, stream: Stream) -> Option<&LogLine> {
        self.lines
            .iter()
            .rev()
            .find(|l| l.stream == stream && !l.text.trim().is_empty())
    }

    /// The last `tail` lines at or after sequence `since`.
    pub(crate) fn page(&self, tail: usize, since: u64) -> LogPage {
        let first_kept = self.first_seq().unwrap_or(self.next_seq);
        let wanted: Vec<&LogLine> = self.lines.iter().filter(|l| l.seq >= since).collect();
        let start = wanted.len().saturating_sub(tail);
        LogPage {
            lines: wanted[start..]
                .iter()
                .map(|l| match l.stream {
                    Stream::Stdout => l.text.clone(),
                    Stream::Stderr => format!("[stderr] {}", l.text),
                })
                .collect(),
            entries: wanted[start..].iter().map(|l| (*l).clone()).collect(),
            total_lines: self.next_seq,
            next_since: self.next_seq,
            dropped_lines: first_kept.saturating_sub(since),
        }
    }
}

/// A page of a job's output.
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct LogPage {
    /// The lines, oldest first; stderr lines are prefixed `[stderr] `.
    pub lines: Vec<String>,
    /// The same lines with their stream and position, for a reader that
    /// shows the two streams differently.
    #[serde(skip)]
    pub entries: Vec<LogLine>,
    /// Lines the job has printed in all, including ones no longer kept.
    pub total_lines: u64,
    /// Pass as `since` to get only what comes after this page.
    pub next_since: u64,
    /// Lines that were printed after `since` but have already been dropped.
    pub dropped_lines: u64,
}

/// Cuts a byte stream into lines of at most `max` bytes. What a longer line has
/// beyond `max` is dropped as it arrives, so a stream with no newline cannot grow
/// memory. Bytes that are not UTF-8 become U+FFFD.
#[derive(Debug)]
pub(crate) struct LineAssembler {
    line: Vec<u8>,
    dropped: usize,
    max: usize,
}

impl LineAssembler {
    pub(crate) fn new(max: usize) -> Self {
        Self {
            line: Vec::new(),
            dropped: 0,
            max,
        }
    }

    /// Take the next bytes of the stream; `on_line` gets each line they complete.
    pub(crate) fn feed(&mut self, mut bytes: &[u8], mut on_line: impl FnMut(String)) {
        while !bytes.is_empty() {
            let (chunk, found_newline) = match bytes.iter().position(|&b| b == b'\n') {
                Some(i) => (&bytes[..i], true),
                None => (bytes, false),
            };
            let room = self.max.saturating_sub(self.line.len());
            let keep = chunk.len().min(room);
            self.line.extend_from_slice(&chunk[..keep]);
            self.dropped += chunk.len() - keep;
            bytes = &bytes[chunk.len() + usize::from(found_newline)..];
            if found_newline {
                on_line(self.take());
            }
        }
    }

    /// The end of the stream: a last line with no newline is still a line.
    pub(crate) fn finish(&mut self, mut on_line: impl FnMut(String)) {
        if !self.line.is_empty() || self.dropped > 0 {
            on_line(self.take());
        }
    }

    fn take(&mut self) -> String {
        if self.line.last() == Some(&b'\r') {
            self.line.pop();
        }
        let mut text = String::from_utf8_lossy(&self.line).into_owned();
        if self.dropped > 0 {
            text.push_str(&format!(" … [{} more bytes dropped]", self.dropped));
        }
        self.line.clear();
        self.dropped = 0;
        text
    }
}

/// Decodes a byte stream to text a piece at a time. A character split across
/// two reads is held until its last byte arrives; bytes that are not UTF-8
/// become U+FFFD.
#[derive(Debug, Default)]
pub(crate) struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        let mut rest: &[u8] = &self.pending;
        loop {
            match std::str::from_utf8(rest) {
                Ok(valid) => {
                    out.push_str(valid);
                    rest = &[];
                    break;
                }
                Err(e) => {
                    let (valid, after) = rest.split_at(e.valid_up_to());
                    out.push_str(std::str::from_utf8(valid).unwrap_or_default());
                    match e.error_len() {
                        // The bytes so far are the start of a character.
                        None => {
                            rest = after;
                            break;
                        }
                        Some(bad) => {
                            out.push('\u{fffd}');
                            rest = &after[bad..];
                        }
                    }
                }
            }
        }
        self.pending = rest.to_vec();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_log_ring_is_bounded_by_lines_and_by_bytes() {
        let mut buf = LogBuffer::default();
        for i in 0..(MAX_LOG_LINES + 50) {
            buf.push(Stream::Stdout, format!("line {i}"));
        }
        assert_eq!(buf.len(), MAX_LOG_LINES);
        assert_eq!(buf.first_seq(), Some(50));
        assert_eq!(buf.next_seq(), (MAX_LOG_LINES + 50) as u64);

        let mut buf = LogBuffer::default();
        let big = "x".repeat(MAX_LINE_BYTES);
        for _ in 0..(MAX_LOG_BYTES / MAX_LINE_BYTES + 10) {
            buf.push(Stream::Stderr, big.clone());
        }
        assert!(buf.bytes() <= MAX_LOG_BYTES, "{}", buf.bytes());
    }

    #[test]
    fn a_page_reports_what_it_skipped() {
        let mut buf = LogBuffer::default();
        for i in 0..(MAX_LOG_LINES + 5) {
            buf.push(Stream::Stdout, format!("line {i}"));
        }
        let page = buf.page(3, 0);
        assert_eq!(page.lines, ["line 10002", "line 10003", "line 10004"]);
        assert_eq!(page.dropped_lines, 5);
        let later = buf.page(10, page.next_since);
        assert!(later.lines.is_empty());
        assert_eq!(later.dropped_lines, 0);
    }

    fn lines_of(max: usize, pieces: &[&[u8]]) -> Vec<String> {
        let mut assembler = LineAssembler::new(max);
        let mut got = Vec::new();
        for piece in pieces {
            assembler.feed(piece, |l| got.push(l));
        }
        assembler.finish(|l| got.push(l));
        got
    }

    #[test]
    fn a_line_without_a_newline_is_cut_not_buffered() {
        let data = vec![b'a'; 1_000_000];
        let got = lines_of(1024, &[&data]);
        assert_eq!(got.len(), 1);
        assert!(got[0].len() < 1024 + 64, "{}", got[0].len());
        assert!(got[0].contains("more bytes dropped"), "{}", got[0]);
    }

    #[test]
    fn lines_split_on_newlines_and_crlf_wherever_the_reads_end() {
        let whole = lines_of(1024, &[b"one\r\ntwo\n\nlast"]);
        assert_eq!(whole, ["one", "two", "", "last"]);
        // The same bytes in awkward pieces.
        let pieces = lines_of(1024, &[b"o", b"ne\r", b"\ntw", b"o\n", b"\nla", b"st"]);
        assert_eq!(pieces, whole);
        assert!(lines_of(1024, &[b""]).is_empty());
    }

    #[test]
    fn bytes_that_are_not_utf8_become_replacement_characters() {
        let got = lines_of(1024, &[b"ok \xff\xfe bytes\nnext"]);
        assert_eq!(got, ["ok \u{fffd}\u{fffd} bytes", "next"]);
    }

    #[test]
    fn a_character_split_across_reads_is_whole_in_the_text() {
        let mut stream = Utf8Stream::default();
        let euro = "€".as_bytes();
        assert_eq!(stream.push(&euro[..1]), "");
        assert_eq!(stream.push(&euro[1..2]), "");
        assert_eq!(stream.push(&euro[2..]), "€");
        assert_eq!(stream.push(b"a\xffb"), "a\u{fffd}b");
        // A character cut short by the end of what arrived is held back.
        assert_eq!(stream.push(b"x\xe2\x82"), "x");
        assert_eq!(stream.push(b"\xacy"), "€y");
    }
}
