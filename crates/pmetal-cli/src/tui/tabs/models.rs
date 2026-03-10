//! Models tab — browse cached HuggingFace models.

use std::path::PathBuf;

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, HighlightSpacing, Paragraph, Row, Scrollbar,
    ScrollbarOrientation, ScrollbarState, StatefulWidget, Table, TableState, Widget, Wrap,
};

use crate::tui::theme::THEME;

/// A cached model entry.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub id: String,
    pub path: PathBuf,
    pub size_bytes: u64,
    pub architecture: Option<String>,
    pub params: Option<String>,
    pub quantization: Option<String>,
    pub modified: String,
}

impl ModelEntry {
    pub fn size_display(&self) -> String {
        let gb = self.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        if gb >= 1.0 {
            format!("{:.1} GB", gb)
        } else {
            let mb = self.size_bytes as f64 / (1024.0 * 1024.0);
            format!("{:.0} MB", mb)
        }
    }
}

/// Models tab state.
pub struct ModelsTab {
    pub models: Vec<ModelEntry>,
    pub table_state: TableState,
    pub scrollbar_state: ScrollbarState,
    pub loading: bool,
    pub search_query: String,
    pub searching: bool,
}

impl ModelsTab {
    pub fn new() -> Self {
        let mut tab = Self {
            models: Vec::new(),
            table_state: TableState::default(),
            scrollbar_state: ScrollbarState::default(),
            loading: true,
            search_query: String::new(),
            searching: false,
        };
        tab.scan_models();
        tab
    }

    /// Scan the HuggingFace cache for downloaded models.
    pub fn scan_models(&mut self) {
        self.loading = true;
        self.models.clear();

        let cache_dir = pmetal_hub::cache_dir();
        if !cache_dir.exists() {
            self.loading = false;
            return;
        }

        // HF cache structure: ~/.cache/pmetal/hub/models--{org}--{name}/snapshots/{hash}/
        let hub_dir = cache_dir.join("hub");
        if !hub_dir.exists() {
            self.loading = false;
            return;
        }

        if let Ok(entries) = std::fs::read_dir(&hub_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if !name.starts_with("models--") {
                    continue;
                }

                // Parse model ID from directory name
                let model_id = name
                    .strip_prefix("models--")
                    .unwrap_or(&name)
                    .replace("--", "/");

                // Find the latest snapshot (by modification time, not FS order)
                let snapshots_dir = path.join("snapshots");
                let snapshot_path = if snapshots_dir.exists() {
                    std::fs::read_dir(&snapshots_dir)
                        .ok()
                        .and_then(|entries| {
                            entries
                                .flatten()
                                .filter(|e| e.path().is_dir())
                                .max_by_key(|e| {
                                    e.metadata()
                                        .and_then(|m| m.modified())
                                        .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                                })
                                .map(|e| e.path())
                        })
                } else {
                    None
                };

                let snapshot = snapshot_path.unwrap_or(path.clone());

                // Calculate total size
                let size = dir_size(&snapshot);

                // Try to read config for architecture info
                let config_path = snapshot.join("config.json");
                let (architecture, params) = if config_path.exists() {
                    read_model_config(&config_path)
                } else {
                    (None, None)
                };

                // Check for quantization
                let quantization = if snapshot.join("quantization_config.json").exists() {
                    Some("Quantized".to_string())
                } else {
                    detect_weight_format(&snapshot)
                };

                // Modified time
                let modified = std::fs::metadata(&snapshot)
                    .and_then(|m| m.modified())
                    .ok()
                    .map(|t| {
                        let duration = t.elapsed().unwrap_or_default();
                        let hours = duration.as_secs() / 3600;
                        if hours < 24 {
                            format!("{}h ago", hours)
                        } else {
                            format!("{}d ago", hours / 24)
                        }
                    })
                    .unwrap_or_else(|| "unknown".to_string());

                self.models.push(ModelEntry {
                    id: model_id,
                    path: snapshot,
                    size_bytes: size,
                    architecture,
                    params,
                    quantization,
                    modified,
                });
            }
        }

        // Sort by name
        self.models.sort_by(|a, b| a.id.cmp(&b.id));

        if !self.models.is_empty() {
            self.table_state.select(Some(0));
        }
        self.scrollbar_state = ScrollbarState::new(self.models.len());
        self.loading = false;
    }

    pub fn filtered_models(&self) -> Vec<&ModelEntry> {
        if self.search_query.is_empty() {
            self.models.iter().collect()
        } else {
            let q = self.search_query.to_lowercase();
            self.models
                .iter()
                .filter(|m| m.id.to_lowercase().contains(&q))
                .collect()
        }
    }

    pub fn next_row(&mut self) {
        let count = self.filtered_models().len();
        if count == 0 {
            return;
        }
        let i = self.table_state.selected().map_or(0, |i| (i + 1) % count);
        self.table_state.select(Some(i));
        self.scrollbar_state = ScrollbarState::new(count).position(i);
    }

    pub fn prev_row(&mut self) {
        let count = self.filtered_models().len();
        if count == 0 {
            return;
        }
        let i = self
            .table_state
            .selected()
            .map_or(0, |i| (i + count - 1) % count);
        self.table_state.select(Some(i));
        self.scrollbar_state = ScrollbarState::new(count).position(i);
    }

    pub fn selected_model(&self) -> Option<&ModelEntry> {
        let filtered = self.filtered_models();
        self.table_state
            .selected()
            .and_then(|i| filtered.get(i).copied())
    }
}

impl Widget for &mut ModelsTab {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let [table_area, detail_area] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(area);

        self.render_table(table_area, buf);
        self.render_detail(detail_area, buf);
    }
}

impl ModelsTab {
    fn render_table(&mut self, area: Rect, buf: &mut Buffer) {
        let title = if self.searching {
            format!(" Models [/{}] ", self.search_query)
        } else {
            format!(" Models ({}) ", self.models.len())
        };
        let block = Block::default()
            .title(title)
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);

        if self.loading {
            Paragraph::new("\n  Scanning cache...")
                .style(THEME.text_muted)
                .block(block)
                .render(area, buf);
            return;
        }

        let filtered = self.filtered_models();
        if filtered.is_empty() {
            Paragraph::new(if self.search_query.is_empty() {
                "\n  No cached models found.\n  Use: pmetal download <model-id>"
            } else {
                "\n  No models match search."
            })
            .style(THEME.text_muted)
            .block(block)
            .render(area, buf);
            return;
        }

        let header = Row::new(vec!["Model", "Size", "Arch", "Modified"])
            .style(THEME.table_header)
            .height(1);

        let rows: Vec<Row> = filtered
            .iter()
            .enumerate()
            .map(|(i, model)| {
                let style = if i % 2 == 0 {
                    THEME.table_row
                } else {
                    THEME.table_row_alt
                };
                Row::new(vec![
                    Cell::new(model.id.clone()),
                    Cell::new(model.size_display()),
                    Cell::new(
                        model
                            .architecture
                            .as_deref()
                            .unwrap_or("-")
                            .to_string(),
                    ),
                    Cell::new(model.modified.clone()),
                ])
                .style(style)
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Fill(1),
                Constraint::Length(10),
                Constraint::Length(12),
                Constraint::Length(10),
            ],
        )
        .header(header)
        .block(block)
        .row_highlight_style(THEME.table_selected)
        .highlight_spacing(HighlightSpacing::Always);

        StatefulWidget::render(table, area, buf, &mut self.table_state);

        // Scrollbar
        Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .render(area, buf, &mut self.scrollbar_state);
    }

    fn render_detail(&self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Details ")
            .title_style(THEME.block_title)
            .borders(Borders::ALL)
            .border_style(THEME.block);
        let inner = block.inner(area);
        block.render(area, buf);

        let Some(model) = self.selected_model() else {
            Paragraph::new("Select a model to view details")
                .style(THEME.text_muted)
                .render(inner, buf);
            return;
        };

        let mut lines = vec![
            Line::from(vec![
                Span::styled("ID:    ", THEME.kv_key),
                Span::styled(&model.id, THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Size:  ", THEME.kv_key),
                Span::styled(model.size_display(), THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Path:  ", THEME.kv_key),
                Span::styled(model.path.display().to_string(), THEME.text_dim),
            ]),
        ];

        if let Some(arch) = &model.architecture {
            lines.push(Line::from(vec![
                Span::styled("Arch:  ", THEME.kv_key),
                Span::styled(arch, THEME.kv_value),
            ]));
        }

        if let Some(params) = &model.params {
            lines.push(Line::from(vec![
                Span::styled("Params:", THEME.kv_key),
                Span::styled(format!(" {params}"), THEME.kv_value),
            ]));
        }

        if let Some(quant) = &model.quantization {
            lines.push(Line::from(vec![
                Span::styled("Format:", THEME.kv_key),
                Span::styled(format!(" {quant}"), THEME.kv_value),
            ]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("Modified: ", THEME.kv_key),
            Span::styled(&model.modified, THEME.text_dim),
        ]));

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

// --- Helpers ---

fn dir_size(path: &std::path::Path) -> u64 {
    // Cap the walk to avoid blocking UI on huge model directories.
    // Use symlink_metadata to avoid double-counting HF blob symlinks.
    const MAX_ENTRIES: usize = 2000;
    walkdir::WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .take(MAX_ENTRIES)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
        .sum()
}

fn read_model_config(path: &std::path::Path) -> (Option<String>, Option<String>) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return (None, None);
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) else {
        return (None, None);
    };

    let arch = json["model_type"]
        .as_str()
        .or_else(|| json["architectures"].as_array()?.first()?.as_str())
        .map(String::from);

    // Estimate parameter count from config
    let params = estimate_params(&json);

    (arch, params)
}

fn estimate_params(config: &serde_json::Value) -> Option<String> {
    let hidden = config["hidden_size"].as_u64()?;
    let layers = config["num_hidden_layers"].as_u64()?;
    let vocab = config["vocab_size"].as_u64().unwrap_or(32000);
    let intermediate = config["intermediate_size"].as_u64().unwrap_or(hidden * 4);

    // Rough estimate: embeddings + layers * (attention + MLP)
    let embed = vocab * hidden;
    let attn = 4 * hidden * hidden; // Q, K, V, O
    let mlp = 3 * hidden * intermediate; // gate, up, down
    let total = embed + layers * (attn + mlp);

    let billions = total as f64 / 1e9;
    if billions >= 1.0 {
        Some(format!("{:.1}B", billions))
    } else {
        Some(format!("{:.0}M", billions * 1000.0))
    }
}

fn detect_weight_format(path: &std::path::Path) -> Option<String> {
    if path.join("model.safetensors").exists()
        || path.join("model.safetensors.index.json").exists()
    {
        Some("SafeTensors".to_string())
    } else if path.join("pytorch_model.bin").exists() {
        Some("PyTorch".to_string())
    } else {
        // Check for sharded safetensors
        let has_shards = std::fs::read_dir(path)
            .ok()?
            .flatten()
            .any(|e| {
                e.file_name()
                    .to_string_lossy()
                    .ends_with(".safetensors")
            });
        if has_shards {
            Some("SafeTensors (sharded)".to_string())
        } else {
            None
        }
    }
}
