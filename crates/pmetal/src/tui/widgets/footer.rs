//! Context-sensitive footer/help bar.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::Widget;

use crate::tui::keys::footer_items;
use crate::tui::tabs::Tab;
use crate::tui::theme::THEME;

/// Footer widget showing the key bindings of the active tab.
pub struct Footer {
    pub tab: Tab,
    /// Number of background jobs currently running (training, serve,
    /// quantize, ...). Surfaces a compact "N running" badge on the right
    /// edge so the user never loses track of active work.
    pub active_jobs: usize,
    /// A note to show in place of the hints (why a key did nothing).
    pub notice: Option<String>,
}

impl Widget for Footer {
    fn render(self, area: Rect, buf: &mut Buffer) {
        buf.set_style(area, THEME.footer);
        let width = area.width as usize;

        // The badge is always shown; the key hints get what is left.
        let badge = (self.active_jobs > 0).then(|| format!(" ● {} running ", self.active_jobs));
        let badge_width = badge.as_ref().map_or(0, |b| b.chars().count());
        let room = width.saturating_sub(badge_width);

        let mut spans: Vec<Span<'_>> = vec![Span::raw(" ")];
        let mut used = 1;
        if let Some(notice) = &self.notice {
            let text: String = notice.chars().take(room.saturating_sub(2)).collect();
            spans.push(Span::styled(text, THEME.text_warning));
        } else {
            for (key, label) in footer_items(self.tab) {
                let need = key.chars().count() + label.chars().count() + 3;
                if used + need > room {
                    break;
                }
                used += need;
                spans.push(Span::styled(key, THEME.footer_key));
                spans.push(Span::styled(format!(" {label}  "), THEME.footer_desc));
            }
        }

        Line::from(spans).render(area, buf);
        if let Some(badge) = badge {
            let x = area.right().saturating_sub(badge_width as u16);
            if x >= area.x {
                Line::from(Span::styled(badge, THEME.status_running))
                    .render(Rect::new(x, area.y, badge_width as u16, 1), buf);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(footer: Footer, width: u16) -> String {
        let area = Rect::new(0, 0, width, 1);
        let mut buf = Buffer::empty(area);
        footer.render(area, &mut buf);
        (0..width).map(|x| buf[(x, 0)].symbol()).collect()
    }

    #[test]
    fn the_footer_lists_the_tabs_bindings_that_fit() {
        let wide = row(
            Footer {
                tab: Tab::Training,
                active_jobs: 0,
                notice: None,
            },
            200,
        );
        for hint in ["navigate", "start", "stop", "set LR", "help", "quit"] {
            assert!(wide.contains(hint), "{hint} in {wide:?}");
        }
        let narrow = row(
            Footer {
                tab: Tab::Training,
                active_jobs: 0,
                notice: None,
            },
            40,
        );
        assert!(narrow.contains("navigate"));
        assert!(!narrow.contains("quit"), "{narrow:?}");
        assert_eq!(narrow.chars().count(), 40);
    }

    #[test]
    fn the_running_badge_keeps_its_place_on_a_crowded_footer() {
        let line = row(
            Footer {
                tab: Tab::Models,
                active_jobs: 3,
                notice: None,
            },
            60,
        );
        assert!(line.trim_end().ends_with("● 3 running"), "{line:?}");
    }

    #[test]
    fn a_notice_replaces_the_hints() {
        let line = row(
            Footer {
                tab: Tab::Serve,
                active_jobs: 0,
                notice: Some("Model is required".into()),
            },
            80,
        );
        assert!(line.contains("Model is required"));
        assert!(!line.contains("navigate"));
    }

    #[test]
    fn the_inference_tab_does_not_offer_q_to_quit() {
        let line = row(
            Footer {
                tab: Tab::Inference,
                active_jobs: 0,
                notice: None,
            },
            200,
        );
        assert!(!line.contains(" quit"), "{line:?}");
    }

    #[test]
    fn a_zero_width_footer_does_not_panic() {
        for width in [0, 1, 2] {
            row(
                Footer {
                    tab: Tab::Jobs,
                    active_jobs: 1,
                    notice: Some("x".into()),
                },
                width,
            );
        }
    }
}
