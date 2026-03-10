//! Datasets tab — browse and analyze local datasets.

use std::path::PathBuf;

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, HighlightSpacing, Paragraph, Row, Scrollbar, ScrollbarOrientation,
    ScrollbarState, StatefulWidget, Table, TableState, Widget, Wrap,
};

use crate::tui::theme::THEME;

/// A dataset entry.
#[derive(Debug, Clone)]
pub struct DatasetEntry {
    pub name: String,
    pub path: PathBuf,
    pub format: String,
    pub size_bytes: u64,
    pub sample_count: Option<usize>,
    pub preview: Option<String>,
}

impl DatasetEntry {
    pub fn size_display(&self) -> String {
        let mb = self.size_bytes as f64 / (1024.0 * 1024.0);
        if mb >= 1024.0 {
            format!("{:.1} GB", mb / 1024.0)
        } else if mb >= 1.0 {
            format!("{:.1} MB", mb)
        } else {
            format!("{:.0} KB", self.size_bytes as f64 / 1024.0)
        }
    }
}

/// Datasets tab state.
pub struct DatasetsTab {
    pub datasets: Vec<DatasetEntry>,
    pub table_state: TableState,
    pub scrollbar_state: ScrollbarState,
    pub scan_dirs: Vec<PathBuf>,
}

impl DatasetsTab {
    pub fn new() -> Self {
        let scan_dirs = vec![
            PathBuf::from("./"),
            PathBuf::from("./data"),
            PathBuf::from("./datasets"),
        ];
        let mut tab = Self {
            datasets: Vec::new(),
            table_state: TableState::default(),
            scrollbar_state: ScrollbarState::default(),
            scan_dirs,
        };
        tab.scan_datasets();
        tab
    }

    /// Scan configured directories for dataset files.
    pub fn scan_datasets(&mut self) {
        self.datasets.clear();

        for dir in &self.scan_dirs {
            if !dir.exists() {
                continue;
            }

            let entries = match std::fs::read_dir(dir) {
                Ok(e) => e,
                Err(_) => continue,
            };

            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }

                let ext = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();

                let format = match ext.as_str() {
                    "jsonl" => "JSONL",
                    "json" => "JSON",
                    "parquet" => "Parquet",
                    "csv" => "CSV",
                    _ => continue,
                };

                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                let size_bytes = path.metadata().map(|m| m.len()).unwrap_or(0);

                // Count lines for JSONL (quick scan)
                let sample_count = if ext == "jsonl" {
                    count_lines(&path)
                } else {
                    None
                };

                // Read first line as preview
                let preview = read_first_line(&path);

                self.datasets.push(DatasetEntry {
                    name,
                    path,
                    format: format.to_string(),
                    size_bytes,
                    sample_count,
                    preview,
                });
            }
        }

        self.datasets.sort_by(|a, b| a.name.cmp(&b.name));
        if !self.datasets.is_empty() {
            self.table_state.select(Some(0));
        }
        self.scrollbar_state = ScrollbarState::new(self.datasets.len());
    }

    pub fn next_row(&mut self) {
        let count = self.datasets.len();
        if count == 0 {
            return;
        }
        let i = self.table_state.selected().map_or(0, |i| (i + 1) % count);
        self.table_state.select(Some(i));
        self.scrollbar_state = self.scrollbar_state.position(i);
    }

    pub fn prev_row(&mut self) {
        let count = self.datasets.len();
        if count == 0 {
            return;
        }
        let i = self.table_state.selected().map_or(0, |i| (i + count - 1) % count);
        self.table_state.select(Some(i));
        self.scrollbar_state = self.scrollbar_state.position(i);
    }

    pub fn selected_dataset(&self) -> Option<&DatasetEntry> {
        self.table_state.selected().and_then(|i| self.datasets.get(i))
    }
}

impl Widget for &mut DatasetsTab {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let [table_area, detail_area] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(area);

        self.render_table(table_area, buf);
        self.render_detail(detail_area, buf);
    }
}

impl DatasetsTab {
    fn render_table(&mut self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(format!(" Datasets ({}) ", self.datasets.len()))
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);

        if self.datasets.is_empty() {
            Paragraph::new("\n  No datasets found in ./, ./data/, ./datasets/")
                .style(THEME.text_muted)
                .block(block)
                .render(area, buf);
            return;
        }

        let header = Row::new(vec!["Name", "Format", "Size", "Samples"])
            .style(THEME.table_header)
            .height(1);

        let rows: Vec<Row> = self
            .datasets
            .iter()
            .enumerate()
            .map(|(i, ds)| {
                let style = if i % 2 == 0 {
                    THEME.table_row
                } else {
                    THEME.table_row_alt
                };
                Row::new(vec![
                    Cell::new(ds.name.clone()),
                    Cell::new(ds.format.clone()),
                    Cell::new(ds.size_display()),
                    Cell::new(
                        ds.sample_count
                            .map(|c| c.to_string())
                            .unwrap_or_else(|| "-".to_string()),
                    ),
                ])
                .style(style)
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Fill(1),
                Constraint::Length(8),
                Constraint::Length(10),
                Constraint::Length(8),
            ],
        )
        .header(header)
        .block(block)
        .row_highlight_style(THEME.table_selected)
        .highlight_spacing(HighlightSpacing::Always);

        StatefulWidget::render(table, area, buf, &mut self.table_state);

        Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .render(area, buf, &mut self.scrollbar_state);
    }

    fn render_detail(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Preview ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let Some(ds) = self.selected_dataset() else {
            Paragraph::new("Select a dataset to preview")
                .style(THEME.text_muted)
                .render(inner, buf);
            return;
        };

        let mut lines = vec![
            Line::from(vec![
                Span::styled("File:    ", THEME.kv_key),
                Span::styled(&ds.name, THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Format:  ", THEME.kv_key),
                Span::styled(&ds.format, THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Size:    ", THEME.kv_key),
                Span::styled(ds.size_display(), THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Path:    ", THEME.kv_key),
                Span::styled(ds.path.display().to_string(), THEME.text_dim),
            ]),
        ];

        if let Some(count) = ds.sample_count {
            lines.push(Line::from(vec![
                Span::styled("Samples: ", THEME.kv_key),
                Span::styled(count.to_string(), THEME.kv_value),
            ]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "First sample:",
            THEME.text_dim,
        )));

        if let Some(preview) = &ds.preview {
            // Pretty-print JSON if possible
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(preview) {
                if let Ok(pretty) = serde_json::to_string_pretty(&json) {
                    for line in pretty.lines().take(20) {
                        lines.push(Line::from(Span::styled(line.to_string(), THEME.text_dim)));
                    }
                } else {
                    lines.push(Line::from(Span::styled(preview.clone(), THEME.text_dim)));
                }
            } else {
                lines.push(Line::from(Span::styled(preview.clone(), THEME.text_dim)));
            }
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

fn count_lines(path: &std::path::Path) -> Option<usize> {
    use std::io::Read;
    // Read in chunks to avoid loading entire file into memory.
    // Cap at 64 MB to avoid blocking the UI thread on huge files.
    const MAX_SCAN_BYTES: u64 = 64 * 1024 * 1024;
    let file = std::fs::File::open(path).ok()?;
    let file_size = file.metadata().ok()?.len();
    let scan_size = file_size.min(MAX_SCAN_BYTES);

    let mut reader = std::io::BufReader::with_capacity(64 * 1024, file.take(scan_size));
    let mut buf = [0u8; 64 * 1024];
    let mut count = 0usize;
    loop {
        let n = reader.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        count += buf[..n].iter().filter(|&&b| b == b'\n').count();
    }
    // Account for final line without trailing newline
    if file_size > 0 && count == 0 {
        count = 1;
    }
    if file_size > scan_size {
        // Extrapolate from the scanned portion
        let ratio = file_size as f64 / scan_size as f64;
        count = (count as f64 * ratio) as usize;
    }
    Some(count)
}

fn read_first_line(path: &std::path::Path) -> Option<String> {
    use std::io::BufRead;
    let file = std::fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    if line.len() > 500 {
        line.truncate(500);
        line.push_str("...");
    }
    Some(line.trim().to_string())
}
