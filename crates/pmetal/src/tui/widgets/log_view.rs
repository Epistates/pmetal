//! A job's log: bounded, scrollable, wrapped to the panel it is drawn in.

use std::collections::VecDeque;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget};

use crate::tui::theme::THEME;

/// Lines kept when no other cap is given.
pub const DEFAULT_CAP: usize = 5_000;

/// One line of a log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogEntry {
    pub text: String,
    /// It came from standard error.
    pub stderr: bool,
}

/// A capped log with a scroll position.
///
/// The newest lines are shown by default and follow the log as it grows. Once
/// the reader scrolls back the view stays on what they are reading until they
/// return to the end, however much is printed in the meantime. Lines longer
/// than the panel are wrapped, not cut.
#[derive(Debug)]
pub struct LogView {
    lines: VecDeque<LogEntry>,
    cap: usize,
    /// Rows scrolled back from the newest; 0 follows the log.
    back: usize,
    /// Lines dropped from the front to stay under the cap.
    dropped: u64,
    /// The width the log was last drawn at, to know how many rows a new line
    /// adds while the reader is scrolled back.
    width: usize,
}

impl LogView {
    /// A log that keeps at most `cap` lines.
    pub fn new(cap: usize) -> Self {
        Self {
            lines: VecDeque::new(),
            cap: cap.max(1),
            back: 0,
            dropped: 0,
            width: 80,
        }
    }

    pub fn push(&mut self, text: impl Into<String>, stderr: bool) {
        let text = text.into();
        if self.back > 0 {
            // Hold the reader's place: what they are looking at moves up by
            // the rows the new line adds.
            self.back += wrap(&text, self.width).len();
        }
        self.lines.push_back(LogEntry { text, stderr });
        while self.lines.len() > self.cap {
            self.lines.pop_front();
            self.dropped += 1;
        }
    }

    pub fn clear(&mut self) {
        self.lines.clear();
        self.back = 0;
        self.dropped = 0;
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// The lines, oldest first.
    pub fn lines(&self) -> impl Iterator<Item = &LogEntry> {
        self.lines.iter()
    }

    /// The last `n` lines' text, oldest first.
    pub fn tail_text(&self, n: usize) -> Vec<&str> {
        let skip = self.lines.len().saturating_sub(n);
        self.lines
            .iter()
            .skip(skip)
            .map(|l| l.text.as_str())
            .collect()
    }

    pub fn scroll_up(&mut self, rows: usize) {
        self.back = self.back.saturating_add(rows);
    }

    pub fn scroll_down(&mut self, rows: usize) {
        self.back = self.back.saturating_sub(rows);
    }

    /// Back to the newest line, following the log again.
    pub fn follow(&mut self) {
        self.back = 0;
    }

    pub fn is_following(&self) -> bool {
        self.back == 0
    }

    /// Draw the log in a bordered panel titled `title`.
    pub fn render(&mut self, area: Rect, buf: &mut Buffer, title: &str) {
        let title = if self.back > 0 {
            format!(" {title} (scrolled back; End: latest) ")
        } else if self.dropped > 0 {
            format!(" {title} ({} older lines dropped) ", self.dropped)
        } else {
            format!(" {title} ")
        };
        let block = Block::default()
            .title(title)
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);
        if inner.width < 3 || inner.height == 0 {
            return;
        }
        self.width = inner.width.saturating_sub(1) as usize;
        let rows = self.rows_for(self.width, inner.height as usize);
        let lines: Vec<Line<'_>> = rows
            .into_iter()
            .map(|(text, stderr)| {
                let style = if stderr { THEME.text_dim } else { THEME.text };
                Line::from(Span::styled(format!(" {text}"), style))
            })
            .collect();
        Paragraph::new(lines).render(inner, buf);
    }

    /// The `height` rows to show, each line wrapped to `width` columns, with
    /// the scroll position applied and clamped to what exists.
    fn rows_for(&mut self, width: usize, height: usize) -> Vec<(String, bool)> {
        // Walk back from the newest line until there are enough rows to fill
        // the panel at this scroll position.
        let want = height + self.back;
        let mut rows_newest_first: Vec<(String, bool)> = Vec::new();
        for entry in self.lines.iter().rev() {
            let mut wrapped = wrap(&entry.text, width);
            wrapped.reverse();
            rows_newest_first.extend(wrapped.into_iter().map(|row| (row, entry.stderr)));
            if rows_newest_first.len() >= want {
                break;
            }
        }
        let available = rows_newest_first.len();
        // Nothing earlier to scroll to: the scroll stops where the log starts.
        self.back = self.back.min(available.saturating_sub(height));
        let start = self.back;
        let mut shown: Vec<(String, bool)> = rows_newest_first
            .into_iter()
            .skip(start)
            .take(height)
            .collect();
        shown.reverse();
        shown
    }
}

/// `text` cut into rows of at most `width` columns (always at least one row).
fn wrap(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    let mut rows = Vec::new();
    let mut row = String::new();
    let mut used = 0;
    for c in text.chars() {
        let w = Span::raw(c.to_string()).width();
        if used + w > width && !row.is_empty() {
            rows.push(std::mem::take(&mut row));
            used = 0;
        }
        row.push(c);
        used += w;
    }
    rows.push(row);
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log_of(lines: &[&str]) -> LogView {
        let mut log = LogView::new(100);
        for line in lines {
            log.push(*line, false);
        }
        log
    }

    fn text(rows: Vec<(String, bool)>) -> Vec<String> {
        rows.into_iter().map(|(t, _)| t).collect()
    }

    #[test]
    fn the_newest_lines_fill_the_panel_and_the_log_is_capped() {
        let mut log = LogView::new(3);
        for i in 0..5 {
            log.push(format!("l{i}"), false);
        }
        assert_eq!(log.len(), 3);
        assert_eq!(log.tail_text(10), ["l2", "l3", "l4"]);
        assert_eq!(text(log.rows_for(20, 2)), ["l3", "l4"]);
    }

    #[test]
    fn long_lines_wrap_instead_of_disappearing() {
        let mut log = log_of(&["abcdefghij", "tail"]);
        assert_eq!(text(log.rows_for(4, 10)), ["abcd", "efgh", "ij", "tail"]);
        // Only the bottom of a tall line shows when the panel is short.
        assert_eq!(text(log.rows_for(4, 2)), ["ij", "tail"]);
    }

    #[test]
    fn wide_characters_count_for_their_width() {
        let rows = wrap("日本語日本語", 6);
        assert_eq!(rows, ["日本語", "日本語"]);
        assert_eq!(wrap("", 5), [""]);
    }

    #[test]
    fn scrolling_back_holds_the_view_while_the_log_grows() {
        let mut log = log_of(&["a", "b", "c", "d", "e"]);
        log.scroll_up(2);
        assert_eq!(text(log.rows_for(10, 2)), ["b", "c"]);
        // New output arrives; the reader's place does not move.
        log.push("f", false);
        assert_eq!(text(log.rows_for(10, 2)), ["b", "c"]);
        log.follow();
        assert_eq!(text(log.rows_for(10, 2)), ["e", "f"]);
        assert!(log.is_following());
    }

    #[test]
    fn scrolling_stops_at_the_start_of_the_log() {
        let mut log = log_of(&["a", "b", "c"]);
        log.scroll_up(1000);
        assert_eq!(text(log.rows_for(10, 2)), ["a", "b"]);
        // The stored position was clamped, so one step down moves.
        log.scroll_down(1);
        assert_eq!(text(log.rows_for(10, 2)), ["b", "c"]);
    }

    #[test]
    fn a_log_longer_than_the_panel_is_not_wrapped_in_full_to_draw_it() {
        // 20k lines of 1k characters: drawing a 10-row panel must not wrap all.
        let mut log = LogView::new(25_000);
        for _ in 0..20_000 {
            log.push("x".repeat(1000), false);
        }
        let started = std::time::Instant::now();
        let rows = log.rows_for(40, 10);
        assert_eq!(rows.len(), 10);
        assert!(started.elapsed() < std::time::Duration::from_millis(250));
    }

    #[test]
    fn rendering_never_panics_in_a_tiny_area() {
        let mut log = log_of(&["hello"]);
        for (w, h) in [(0, 0), (1, 1), (2, 2), (3, 3), (10, 1)] {
            let area = Rect::new(0, 0, w, h);
            let mut buf = Buffer::empty(area);
            log.render(area, &mut buf, "Log");
        }
    }

    #[test]
    fn stderr_lines_are_marked_and_the_title_says_when_the_view_is_behind() {
        let mut log = LogView::new(10);
        log.push("out", false);
        log.push("err", true);
        let rows = log.rows_for(10, 2);
        assert_eq!(
            rows,
            [("out".to_string(), false), ("err".to_string(), true)]
        );
        log.scroll_up(1);
        let area = Rect::new(0, 0, 50, 5);
        let mut buf = Buffer::empty(area);
        log.render(area, &mut buf, "Log");
        let top: String = (0..50).map(|x| buf[(x, 0)].symbol()).collect();
        assert!(top.contains("scrolled back"), "{top}");
    }
}
