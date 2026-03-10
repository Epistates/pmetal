//! Context-sensitive footer/help bar.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::Widget;

use crate::tui::tabs::Tab;
use crate::tui::theme::THEME;

/// Footer widget showing context-sensitive keybindings.
pub struct Footer {
    pub tab: Tab,
}

impl Widget for Footer {
    fn render(self, area: Rect, buf: &mut Buffer) {
        buf.set_style(area, THEME.footer);

        let mut spans = vec![
            Span::raw(" "),
            Span::styled("Tab", THEME.footer_key),
            Span::styled(" switch  ", THEME.footer_desc),
        ];

        // Tab-specific bindings — only advertise implemented features
        match self.tab {
            Tab::Dashboard => {
                spans.extend([
                    Span::styled("r", THEME.footer_key),
                    Span::styled(" reset  ", THEME.footer_desc),
                    Span::styled("p", THEME.footer_key),
                    Span::styled(" pause  ", THEME.footer_desc),
                ]);
            }
            Tab::Models => {
                spans.extend([
                    Span::styled("jk", THEME.footer_key),
                    Span::styled(" navigate  ", THEME.footer_desc),
                    Span::styled("/", THEME.footer_key),
                    Span::styled(" search  ", THEME.footer_desc),
                    Span::styled("R", THEME.footer_key),
                    Span::styled(" refresh  ", THEME.footer_desc),
                ]);
            }
            Tab::Datasets | Tab::Jobs => {
                spans.extend([
                    Span::styled("jk", THEME.footer_key),
                    Span::styled(" navigate  ", THEME.footer_desc),
                    Span::styled("R", THEME.footer_key),
                    Span::styled(" refresh  ", THEME.footer_desc),
                ]);
            }
            Tab::Inference => {
                spans.extend([
                    Span::styled("Enter", THEME.footer_key),
                    Span::styled(" send  ", THEME.footer_desc),
                    Span::styled("Esc", THEME.footer_key),
                    Span::styled(" stop  ", THEME.footer_desc),
                    Span::styled("C-q", THEME.footer_key),
                    Span::styled(" quit  ", THEME.footer_desc),
                ]);
            }
            Tab::Training => {
                spans.extend([
                    Span::styled("jk", THEME.footer_key),
                    Span::styled(" navigate  ", THEME.footer_desc),
                ]);
            }
            Tab::Device => {}
        }

        // Global quit (not shown for Inference since it has its own)
        if self.tab != Tab::Inference {
            spans.extend([
                Span::styled("q", THEME.footer_key),
                Span::styled(" quit", THEME.footer_desc),
            ]);
        }

        Line::from(spans).render(area, buf);
    }
}
