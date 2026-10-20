//! Application header widget with logo and tab bar.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::Widget;

use crate::tui::tabs::Tab;
use crate::tui::theme::THEME;

/// Columns the logo takes before the tab bar starts.
pub const LOGO_WIDTH: u16 = 12;

/// Where each visible tab sits in the header, as column ranges relative to the
/// start of the tab bar.
///
/// Twenty-one tabs do not fit a terminal. The bar shows a window of them that
/// always contains the active one, with `‹` / `›` on the edges when there is
/// more, and this is the one place that decides which: the drawing and the
/// mouse hit-testing both use it, so a click lands on the tab that was drawn
/// under it.
pub fn tab_slots(bar_width: u16, tabs: &[Tab], active: Tab) -> Vec<(Tab, u16, u16)> {
    let width = |tab: Tab| tab.to_string().chars().count() as u16 + 4;
    let bar = bar_width as usize;
    let active_idx = tabs.iter().position(|t| *t == active).unwrap_or(0);

    // Grow a window around the active tab, one tab at a time, alternating
    // sides, until the next one would not fit. Each tab costs its width and a
    // one-column divider; two columns are kept for the `‹` and `›` markers.
    let usable = bar.saturating_sub(2);
    let (mut lo, mut hi) = (active_idx, active_idx + 1);
    let mut used = (width(tabs[active_idx]) as usize + 1).min(usable.max(1));
    let mut grow_right = true;
    loop {
        let candidate = if grow_right && hi < tabs.len() {
            Some((hi, true))
        } else if lo > 0 {
            Some((lo - 1, false))
        } else if hi < tabs.len() {
            Some((hi, true))
        } else {
            None
        };
        let Some((idx, right)) = candidate else { break };
        let cost = width(tabs[idx]) as usize + 1;
        if used + cost > usable {
            break;
        }
        used += cost;
        if right {
            hi += 1;
        } else {
            lo -= 1;
        }
        grow_right = !grow_right;
    }

    // With everything shown there are no markers to make room for.
    let all_shown = lo == 0 && hi == tabs.len();
    let mut x: u16 = if all_shown || lo == 0 { 0 } else { 1 };
    let mut slots = Vec::new();
    for &tab in &tabs[lo..hi] {
        let w = width(tab).min(bar_width.saturating_sub(x));
        if w == 0 {
            break;
        }
        slots.push((tab, x, x + w));
        x += w + 1;
    }
    slots
}

/// Header widget displaying the PMetal logo and tab navigation.
pub struct Header<'a> {
    pub active_tab: Tab,
    pub tabs: &'a [Tab],
}

impl Widget for Header<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Fill background
        buf.set_style(area, THEME.header);
        if area.width <= LOGO_WIDTH {
            return;
        }

        Line::from(vec![
            Span::styled(" P", THEME.logo),
            Span::styled("Metal", THEME.header_title),
        ])
        .render(Rect::new(area.x, area.y, LOGO_WIDTH, 1), buf);

        let bar = Rect::new(
            area.x + LOGO_WIDTH,
            area.y,
            area.width - LOGO_WIDTH,
            area.height,
        );
        let slots = tab_slots(bar.width, self.tabs, self.active_tab);
        for &(tab, from, to) in &slots {
            let label = format!(" {} {} ", tab.icon(), tab);
            let style = if tab == self.active_tab {
                THEME.tab_active
            } else {
                THEME.tab_inactive
            };
            let width = to - from;
            let text: String = label.chars().take(width as usize).collect();
            Line::from(Span::styled(text, style))
                .render(Rect::new(bar.x + from, bar.y, width, 1), buf);
        }
        // Markers for the tabs out of view.
        if let (Some(&(first, from, _)), Some(&(last, _, to))) = (slots.first(), slots.last()) {
            if self.tabs.first() != Some(&first) && from > 0 {
                Line::from(Span::styled("‹", THEME.footer_key))
                    .render(Rect::new(bar.x, bar.y, 1, 1), buf);
            }
            if self.tabs.last() != Some(&last) && to < bar.width {
                Line::from(Span::styled("›", THEME.footer_key))
                    .render(Rect::new(bar.x + bar.width - 1, bar.y, 1, 1), buf);
            }
        }
    }
}

/// The tab drawn under column `x` of a header `width` columns wide, if any.
pub fn tab_at(x: u16, width: u16, tabs: &[Tab], active: Tab) -> Option<Tab> {
    let x = x.checked_sub(LOGO_WIDTH)?;
    let bar = width.checked_sub(LOGO_WIDTH)?;
    tab_slots(bar, tabs, active)
        .into_iter()
        .find(|&(_, from, to)| x >= from && x < to)
        .map(|(tab, _, _)| tab)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drawn(width: u16, active: Tab) -> String {
        let area = Rect::new(0, 0, width, 2);
        let mut buf = Buffer::empty(area);
        Header {
            active_tab: active,
            tabs: Tab::ALL,
        }
        .render(area, &mut buf);
        (0..width).map(|x| buf[(x, 0)].symbol()).collect()
    }

    #[test]
    fn the_active_tab_is_always_on_screen() {
        for width in [20, 40, 60, 80, 120, 200, 400] {
            for &active in Tab::ALL {
                let slots = tab_slots(width - LOGO_WIDTH.min(width), Tab::ALL, active);
                if width >= 40 {
                    assert!(
                        slots.iter().any(|(t, _, _)| *t == active),
                        "{active} missing at width {width}"
                    );
                }
            }
        }
    }

    #[test]
    fn what_was_drawn_is_what_a_click_hits() {
        for width in [60u16, 100, 140] {
            for &active in Tab::ALL {
                let slots = tab_slots(width - LOGO_WIDTH, Tab::ALL, active);
                for (tab, from, to) in slots {
                    for x in from..to {
                        assert_eq!(
                            tab_at(x + LOGO_WIDTH, width, Tab::ALL, active),
                            Some(tab),
                            "x={x} width={width}"
                        );
                    }
                }
                // The logo is not a tab.
                assert_eq!(tab_at(3, width, Tab::ALL, active), None);
            }
        }
    }

    #[test]
    fn slots_are_in_order_inside_the_bar_and_do_not_overlap() {
        for width in [14u16, 30, 60, 100, 300] {
            for &active in Tab::ALL {
                let slots = tab_slots(width, Tab::ALL, active);
                let mut end = 0;
                for (_, from, to) in slots {
                    assert!(
                        from >= end && to > from && to <= width,
                        "{from}..{to} in {width}"
                    );
                    end = to;
                }
            }
        }
    }

    #[test]
    fn a_narrow_terminal_shows_the_active_tab_with_markers() {
        let line = drawn(50, Tab::Merge);
        assert!(line.contains("Merge"), "{line}");
        assert!(line.contains('‹') && line.contains('›'), "{line}");
        // The first tab needs no left marker.
        let first = drawn(50, Tab::ALL[0]);
        assert!(!first.contains('‹'), "{first}");
        assert!(first.contains('›'));
        // Plenty of room shows them all and no markers.
        let wide = drawn(400, Tab::Merge);
        assert!(!wide.contains('‹') && !wide.contains('›'));
        assert!(wide.contains("Jobs"));
    }

    #[test]
    fn a_header_too_narrow_for_the_logo_does_not_panic() {
        for width in 0..LOGO_WIDTH + 3 {
            drawn(width, Tab::Jobs);
        }
    }
}
