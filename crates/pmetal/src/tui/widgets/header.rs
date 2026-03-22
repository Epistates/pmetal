//! Application header widget with logo and tab bar.

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Tabs, Widget};

use crate::tui::tabs::Tab;
use crate::tui::theme::THEME;

/// Header widget displaying the PMetal logo and tab navigation.
pub struct Header<'a> {
    pub active_tab: Tab,
    pub tabs: &'a [Tab],
}

impl Widget for Header<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Fill background
        buf.set_style(area, THEME.header);

        let [logo_area, tabs_area] =
            Layout::horizontal([Constraint::Length(20), Constraint::Fill(1)]).areas(area);

        // Logo
        let logo = Line::from(vec![
            Span::styled(" P", THEME.logo),
            Span::styled("Metal", THEME.header_title),
            Span::raw(" "),
        ]);
        logo.render(logo_area, buf);

        // Tab bar
        let tab_titles: Vec<Line> = self
            .tabs
            .iter()
            .map(|t| {
                let icon = t.icon();
                Line::from(format!(" {icon} {t} "))
            })
            .collect();

        let active_idx = self
            .tabs
            .iter()
            .position(|t| *t == self.active_tab)
            .unwrap_or(0);

        Tabs::new(tab_titles)
            .select(active_idx)
            .style(THEME.tab_inactive)
            .highlight_style(THEME.tab_active)
            .divider(" ")
            .render(tabs_area, buf);
    }
}
