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

        // Tab-specific bindings
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
                    Span::styled(" nav  ", THEME.footer_desc),
                    Span::styled("t", THEME.footer_key),
                    Span::styled(" train  ", THEME.footer_desc),
                    Span::styled("s", THEME.footer_key),
                    Span::styled(" distill  ", THEME.footer_desc),
                    Span::styled("i", THEME.footer_key),
                    Span::styled(" infer  ", THEME.footer_desc),
                    Span::styled("f", THEME.footer_key),
                    Span::styled(" fuse  ", THEME.footer_desc),
                    Span::styled("d", THEME.footer_key),
                    Span::styled(" download  ", THEME.footer_desc),
                ]);
            }
            Tab::Datasets => {
                spans.extend([
                    Span::styled("jk", THEME.footer_key),
                    Span::styled(" navigate  ", THEME.footer_desc),
                    Span::styled("c", THEME.footer_key),
                    Span::styled(" convert  ", THEME.footer_desc),
                    Span::styled("a", THEME.footer_key),
                    Span::styled(" add dir  ", THEME.footer_desc),
                    Span::styled("R", THEME.footer_key),
                    Span::styled(" refresh  ", THEME.footer_desc),
                ]);
            }
            Tab::Training | Tab::Distillation | Tab::Grpo => {
                spans.extend([
                    Span::styled("jk", THEME.footer_key),
                    Span::styled(" navigate  ", THEME.footer_desc),
                    Span::styled("Enter", THEME.footer_key),
                    Span::styled(" edit/pick  ", THEME.footer_desc),
                    Span::styled("S", THEME.footer_key),
                    Span::styled(" start  ", THEME.footer_desc),
                    Span::styled("x", THEME.footer_key),
                    Span::styled(" stop  ", THEME.footer_desc),
                ]);
            }
            Tab::Inference => {
                spans.extend([
                    Span::styled("Enter", THEME.footer_key),
                    Span::styled(" send  ", THEME.footer_desc),
                    Span::styled("Ctrl+P", THEME.footer_key),
                    Span::styled(" model  ", THEME.footer_desc),
                    Span::styled("Ctrl+S", THEME.footer_key),
                    Span::styled(" settings  ", THEME.footer_desc),
                    Span::styled("Ctrl+H", THEME.footer_key),
                    Span::styled(" sidebar  ", THEME.footer_desc),
                    Span::styled("F2", THEME.footer_key),
                    Span::styled(" browse/yank  ", THEME.footer_desc),
                    Span::styled("Esc", THEME.footer_key),
                    Span::styled(" stop  ", THEME.footer_desc),
                    Span::styled("Ctrl+Q", THEME.footer_key),
                    Span::styled(" quit  ", THEME.footer_desc),
                ]);
            }
            Tab::Jobs => {
                spans.extend([
                    Span::styled("jk", THEME.footer_key),
                    Span::styled(" navigate  ", THEME.footer_desc),
                    Span::styled("JK", THEME.footer_key),
                    Span::styled(" scroll log  ", THEME.footer_desc),
                    Span::styled("R", THEME.footer_key),
                    Span::styled(" refresh  ", THEME.footer_desc),
                ]);
            }
            Tab::Device => {
                // Device tab is read-only; no tab-specific bindings
            }
        }

        // Global keybindings (not shown for Inference since it has its own quit)
        if self.tab != Tab::Inference {
            spans.extend([
                Span::styled("q", THEME.footer_key),
                Span::styled(" quit  ", THEME.footer_desc),
            ]);
        }
        spans.extend([
            Span::styled("F2", THEME.footer_key),
            Span::styled(" mouse/select", THEME.footer_desc),
        ]);

        Line::from(spans).render(area, buf);
    }
}
