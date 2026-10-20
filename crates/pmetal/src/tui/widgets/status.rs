//! Status rendering shared by every tab that runs a job.

use ratatui::style::Style;
use ratatui::text::{Line, Span};

use crate::tui::theme::THEME;

/// Visual tone for a status line. The badge renderer maps each tone to a
/// distinct glyph and style, so every tab reports status the same way and a
/// status never depends on its color alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusTone {
    Idle,
    Running,
    Completed,
    Failed,
}

impl StatusTone {
    pub fn glyph(self) -> &'static str {
        match self {
            Self::Idle => "·",
            Self::Running => "●",
            Self::Completed => "✓",
            Self::Failed => "✗",
        }
    }

    pub fn style(self) -> Style {
        match self {
            Self::Idle => THEME.status_idle,
            Self::Running => THEME.status_running,
            Self::Completed => THEME.status_success,
            Self::Failed => THEME.status_error,
        }
    }
}

/// Build a single-line status badge: `  ● Running — binding to port 8080`.
///
/// The glyph and style are driven by `tone`; `label` is shown in the same
/// accent, and the optional `detail` trails after an em-dash in muted text.
/// Used as the first line of every tab's status panel.
pub fn status_line(tone: StatusTone, label: &str, detail: Option<&str>) -> Line<'static> {
    let style = tone.style();
    let mut spans = vec![
        Span::raw("  "),
        Span::styled(tone.glyph(), style),
        Span::raw(" "),
        Span::styled(label.to_string(), style),
    ];
    if let Some(detail) = detail {
        spans.push(Span::styled(" — ", THEME.text_muted));
        spans.push(Span::styled(detail.to_string(), THEME.text_muted));
    }
    Line::from(spans)
}

/// Seconds as `5s`, `3m 07s`, `2h 05m`.
pub fn format_duration(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{}h {:02}m", secs / 3600, (secs % 3600) / 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn durations_read_naturally() {
        assert_eq!(format_duration(Duration::from_secs(5)), "5s");
        assert_eq!(format_duration(Duration::from_secs(187)), "3m 07s");
        assert_eq!(format_duration(Duration::from_secs(7500)), "2h 05m");
    }

    #[test]
    fn every_tone_has_its_own_glyph() {
        let glyphs: Vec<&str> = [
            StatusTone::Idle,
            StatusTone::Running,
            StatusTone::Completed,
            StatusTone::Failed,
        ]
        .into_iter()
        .map(StatusTone::glyph)
        .collect();
        for (i, a) in glyphs.iter().enumerate() {
            assert!(glyphs[i + 1..].iter().all(|b| a != b), "{glyphs:?}");
        }
    }
}
