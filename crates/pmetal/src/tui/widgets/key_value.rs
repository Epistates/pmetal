//! Key-value pair display widget.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::Widget;

use crate::tui::theme::THEME;

/// A key-value pair for display.
pub struct KvPair {
    pub key: String,
    pub value: String,
    pub unit: Option<String>,
}

impl KvPair {
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
            unit: None,
        }
    }

    pub fn with_unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }
}

/// Renders a list of key-value pairs with aligned columns.
pub struct KeyValueList<'a> {
    pub items: &'a [KvPair],
    pub key_width: u16,
}

impl<'a> KeyValueList<'a> {
    pub fn new(items: &'a [KvPair]) -> Self {
        let key_width = items
            .iter()
            .map(|kv| kv.key.len() as u16)
            .max()
            .unwrap_or(10)
            + 2; // padding
        Self { items, key_width }
    }
}

impl Widget for KeyValueList<'_> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        for (i, item) in self.items.iter().enumerate() {
            let y = area.y + i as u16;
            if y >= area.y + area.height {
                break;
            }
            let row_area = Rect::new(area.x, y, area.width, 1);

            let padded_key = format!("{:>width$}: ", item.key, width = self.key_width as usize);

            let mut spans = vec![
                Span::styled(padded_key, THEME.kv_key),
                Span::styled(&item.value, THEME.kv_value),
            ];

            if let Some(unit) = &item.unit {
                spans.push(Span::styled(format!(" {unit}"), THEME.kv_unit));
            }

            Line::from(spans).render(row_area, buf);
        }
    }
}
