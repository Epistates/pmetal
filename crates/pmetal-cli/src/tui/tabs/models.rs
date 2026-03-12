//! Models tab — browse cached HuggingFace models and custom model directories.

use std::collections::HashSet;
use std::path::PathBuf;

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, HighlightSpacing, Paragraph, Row, Scrollbar, ScrollbarOrientation,
    ScrollbarState, StatefulWidget, Table, TableState, Widget, Wrap,
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
    /// Where this model was found.
    pub source: ModelSource,
    /// For LoRA adapters: the base model ID or path needed to load the adapter.
    pub base_model: Option<String>,
    /// Memory fit estimation for this device.
    pub fit_estimate: Option<pmetal_hub::FitEstimate>,
}

/// Where a model was discovered from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// Standard HuggingFace hub cache.
    HfCache,
    /// A user-added custom directory.
    Custom(PathBuf),
    /// A trained/fine-tuned model from an output directory.
    Trained,
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
    /// User-configured custom directories to scan for models.
    pub custom_dirs: Vec<PathBuf>,
    /// Device spec for fit estimation (queried once at startup).
    pub device_spec: Option<pmetal_hub::DeviceSpec>,
}

impl ModelsTab {
    pub fn new() -> Self {
        let custom_dirs = load_custom_model_dirs();

        // Query device spec for fit estimation
        let device_spec = pmetal_metal::context::MetalContext::global()
            .ok()
            .map(|ctx| {
                let props = ctx.properties();
                pmetal_hub::DeviceSpec {
                    memory_gb: props.recommended_working_set_size as f64
                        / (1024.0 * 1024.0 * 1024.0),
                    bandwidth_gbps: props.memory_bandwidth_gbps,
                    unified_memory: props.has_unified_memory,
                }
            });

        let mut tab = Self {
            models: Vec::new(),
            table_state: TableState::default(),
            scrollbar_state: ScrollbarState::default(),
            loading: true,
            search_query: String::new(),
            searching: false,
            custom_dirs,
            device_spec,
        };
        tab.scan_models();
        tab
    }

    /// Add a custom directory to scan for models.
    pub fn add_custom_dir(&mut self, dir: PathBuf) {
        if !self.custom_dirs.contains(&dir) {
            self.custom_dirs.push(dir);
            save_custom_model_dirs(&self.custom_dirs);
            self.scan_models();
        }
    }

    /// Remove a custom directory.
    pub fn remove_custom_dir(&mut self, idx: usize) {
        if idx < self.custom_dirs.len() {
            self.custom_dirs.remove(idx);
            save_custom_model_dirs(&self.custom_dirs);
            self.scan_models();
        }
    }

    /// Scan HuggingFace cache, output directories, and custom directories for models.
    pub fn scan_models(&mut self) {
        self.loading = true;
        self.models.clear();

        // 1. Scan trained/fine-tuned model output directories first
        self.scan_trained_outputs();

        // 2. Scan HuggingFace hub cache (cross-platform, respects HF_HOME/HF_HUB_CACHE)
        let hf_cache = pmetal_hub::cache_dir();
        self.scan_hf_cache(&hf_cache);

        // 3. Scan custom user directories
        let custom_dirs: Vec<PathBuf> = self.custom_dirs.clone();
        for dir in &custom_dirs {
            self.scan_custom_dir(dir);
        }

        // Sort: trained models first, then alphabetically within each group
        self.models.sort_by(|a, b| {
            let a_trained = matches!(a.source, ModelSource::Trained);
            let b_trained = matches!(b.source, ModelSource::Trained);
            b_trained.cmp(&a_trained).then(a.id.cmp(&b.id))
        });

        // Compute fit estimates for all models
        self.compute_fit_estimates();

        if !self.models.is_empty() {
            self.table_state.select(Some(0));
        }
        self.scrollbar_state = ScrollbarState::new(self.models.len());
        self.loading = false;
    }

    /// Compute fit estimates for all local models.
    fn compute_fit_estimates(&mut self) {
        let Some(device) = &self.device_spec else {
            return;
        };
        for model in &mut self.models {
            let config_path = model.path.join("config.json");
            if !config_path.exists() {
                continue;
            }
            let Ok(content) = std::fs::read_to_string(&config_path) else {
                continue;
            };
            let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) else {
                continue;
            };
            let model_spec = pmetal_hub::model_spec_from_config(&config, Some(model.size_bytes));
            model.fit_estimate = Some(pmetal_hub::estimate_fit(&model_spec, device));
        }
    }

    /// Scan common output directories for trained/fine-tuned models.
    /// Looks in ./output/ and all its subdirectories for lora_weights.safetensors,
    /// adapter_config.json, or other model artifacts.
    fn scan_trained_outputs(&mut self) {
        let output_root = PathBuf::from("./output");
        if !output_root.exists() {
            return;
        }

        let mut seen: HashSet<String> = self.models.iter().map(|m| m.id.clone()).collect();

        // Recursively scan output/ for model dirs (up to 2 levels deep)
        self.scan_trained_dir_recursive(&output_root, &mut seen, 0, 2);
    }

    /// Recursively scan a directory for trained model outputs.
    fn scan_trained_dir_recursive(
        &mut self,
        dir: &std::path::Path,
        seen: &mut HashSet<String>,
        depth: usize,
        max_depth: usize,
    ) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let dir_name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            // Skip hidden dirs and checkpoints
            if dir_name.starts_with('.') || dir_name == "checkpoints" {
                continue;
            }

            if is_model_dir(&path) {
                let model_id = format!("trained/{dir_name}");

                if !seen.contains(&model_id) {
                    if let Some(mut entry) =
                        build_model_entry(&model_id, &path, ModelSource::Trained)
                    {
                        if is_trained_dir(&path) {
                            entry.quantization = Some("LoRA Adapter".to_string());
                        }
                        seen.insert(model_id);
                        self.models.push(entry);
                    }
                }
            } else if depth < max_depth {
                // Recurse into subdirectories (e.g., output/distilled/, output/grpo/)
                self.scan_trained_dir_recursive(&path, seen, depth + 1, max_depth);
            }
        }
    }

    /// Scan the HuggingFace hub cache directory.
    /// Structure: {cache}/models--{org}--{name}/snapshots/{hash}/
    fn scan_hf_cache(&mut self, hub_dir: &std::path::Path) {
        if !hub_dir.exists() {
            return;
        }

        let entries = match std::fs::read_dir(hub_dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if !name.starts_with("models--") {
                continue;
            }

            // Parse model ID from directory name: models--org--name → org/name
            let model_id = name
                .strip_prefix("models--")
                .unwrap_or(&name)
                .replace("--", "/");

            // Find the latest snapshot
            let snapshots_dir = path.join("snapshots");
            let snapshot_path = if snapshots_dir.exists() {
                std::fs::read_dir(&snapshots_dir).ok().and_then(|entries| {
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

            let snapshot = snapshot_path.unwrap_or_else(|| path.clone());
            if let Some(entry) = build_model_entry(&model_id, &snapshot, ModelSource::HfCache) {
                self.models.push(entry);
            }
        }
    }

    /// Scan a custom directory for model directories.
    /// Looks for directories containing config.json or safetensors files.
    /// Directories with adapter_config.json are treated as trained models.
    fn scan_custom_dir(&mut self, dir: &std::path::Path) {
        if !dir.exists() {
            return;
        }

        let seen: HashSet<String> = self.models.iter().map(|m| m.id.clone()).collect();

        // Check if the directory itself is a model
        if is_model_dir(dir) {
            let is_trained = is_trained_dir(dir);
            let model_id = if is_trained {
                format!(
                    "trained/{}",
                    dir.file_name().unwrap_or_default().to_string_lossy()
                )
            } else {
                dir.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            };
            if !seen.contains(&model_id) {
                let source = if is_trained {
                    ModelSource::Trained
                } else {
                    ModelSource::Custom(dir.to_path_buf())
                };
                if let Some(mut entry) = build_model_entry(&model_id, dir, source) {
                    if is_trained {
                        entry.quantization = Some("LoRA Adapter".to_string());
                    }
                    self.models.push(entry);
                }
            }
            return;
        }

        // Otherwise scan subdirectories
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() || !is_model_dir(&path) {
                continue;
            }

            let is_trained = is_trained_dir(&path);
            let dir_name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let model_id = if is_trained {
                format!("trained/{dir_name}")
            } else {
                dir_name
            };

            if seen.contains(&model_id) {
                continue;
            }

            let source = if is_trained {
                ModelSource::Trained
            } else {
                ModelSource::Custom(dir.to_path_buf())
            };
            if let Some(mut entry) = build_model_entry(&model_id, &path, source) {
                if is_trained {
                    entry.quantization = Some("LoRA Adapter".to_string());
                }
                self.models.push(entry);
            }
        }
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
                "\n  No models found.\n\n  Download:  pmetal download <model-id>\n  Add dir:   press 'a' to add a custom model directory"
            } else {
                "\n  No models match search."
            })
            .style(THEME.text_muted)
            .block(block)
            .render(area, buf);
            return;
        }

        let header = Row::new(vec!["Model", "Size", "Arch / Base", "Fit", "Source"])
            .style(THEME.table_header)
            .height(1);

        let rows: Vec<Row> = filtered
            .iter()
            .enumerate()
            .map(|(i, model)| {
                use ratatui::style::{Color, Style};

                let style = if i % 2 == 0 {
                    THEME.table_row
                } else {
                    THEME.table_row_alt
                };
                let source_cell = match &model.source {
                    ModelSource::Trained => Cell::new("Trained").style(THEME.text_success),
                    ModelSource::HfCache => Cell::new("HF Cache"),
                    ModelSource::Custom(_) => Cell::new("Custom"),
                };
                let arch_display = model
                    .architecture
                    .as_deref()
                    .map(String::from)
                    .or_else(|| model.base_model.as_ref().map(|b| format!("< {b}")))
                    .unwrap_or_else(|| "-".to_string());
                let fit_cell = if let Some(fit) = &model.fit_estimate {
                    let fit_style = match fit.fit_level {
                        pmetal_hub::FitLevel::Fits => Style::default().fg(Color::Green),
                        pmetal_hub::FitLevel::Tight => Style::default().fg(Color::Yellow),
                        pmetal_hub::FitLevel::TooLarge => Style::default().fg(Color::Red),
                    };
                    Cell::new(format!("{:.1}GB", fit.total_required_gb)).style(fit_style)
                } else {
                    Cell::new("-")
                };
                Row::new(vec![
                    Cell::new(model.id.clone()),
                    Cell::new(model.size_display()),
                    Cell::new(arch_display),
                    fit_cell,
                    source_cell,
                ])
                .style(style)
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Fill(1),
                Constraint::Length(10),
                Constraint::Length(18),
                Constraint::Length(9),
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

        let source_label = match &model.source {
            ModelSource::HfCache => "HF Cache".to_string(),
            ModelSource::Custom(p) => format!("Custom ({})", p.display()),
            ModelSource::Trained => "Trained (local)".to_string(),
        };

        let mut lines = vec![
            Line::from(vec![
                Span::styled("ID:     ", THEME.kv_key),
                Span::styled(&model.id, THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Size:   ", THEME.kv_key),
                Span::styled(model.size_display(), THEME.kv_value),
            ]),
            Line::from(vec![
                Span::styled("Source: ", THEME.kv_key),
                Span::styled(source_label, THEME.text_dim),
            ]),
            Line::from(vec![
                Span::styled("Path:   ", THEME.kv_key),
                Span::styled(model.path.display().to_string(), THEME.text_dim),
            ]),
        ];

        if let Some(arch) = &model.architecture {
            lines.push(Line::from(vec![
                Span::styled("Arch:   ", THEME.kv_key),
                Span::styled(arch, THEME.kv_value),
            ]));
        }

        if let Some(params) = &model.params {
            lines.push(Line::from(vec![
                Span::styled("Params: ", THEME.kv_key),
                Span::styled(params.as_str(), THEME.kv_value),
            ]));
        }

        if let Some(quant) = &model.quantization {
            lines.push(Line::from(vec![
                Span::styled("Format: ", THEME.kv_key),
                Span::styled(quant.as_str(), THEME.kv_value),
            ]));
        }

        if let Some(base) = &model.base_model {
            lines.push(Line::from(vec![
                Span::styled("Base:   ", THEME.kv_key),
                Span::styled(base.as_str(), THEME.kv_value),
            ]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("Modified: ", THEME.kv_key),
            Span::styled(&model.modified, THEME.text_dim),
        ]));

        // Fit estimation section
        if let Some(fit) = &model.fit_estimate {
            use ratatui::style::{Color, Style};

            lines.push(Line::from(""));
            let fit_style = match fit.fit_level {
                pmetal_hub::FitLevel::Fits => Style::default().fg(Color::Green),
                pmetal_hub::FitLevel::Tight => Style::default().fg(Color::Yellow),
                pmetal_hub::FitLevel::TooLarge => Style::default().fg(Color::Red),
            };
            lines.push(Line::from(vec![Span::styled(
                "--- Memory Fit ---",
                THEME.kv_key,
            )]));
            lines.push(Line::from(vec![
                Span::styled("Status: ", THEME.kv_key),
                Span::styled(fit.fit_level.label(), fit_style),
                Span::styled(
                    format!(
                        " ({:.0}% of {:.0}GB)",
                        fit.utilization_pct, fit.available_gb
                    ),
                    THEME.text_dim,
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled("Weights: ", THEME.kv_key),
                Span::styled(format!("{:.1} GB", fit.weights_gb), THEME.kv_value),
                Span::styled("  KV: ", THEME.kv_key),
                Span::styled(format!("{:.1} GB", fit.kv_cache_gb), THEME.kv_value),
            ]));
            lines.push(Line::from(vec![
                Span::styled("Decode:  ", THEME.kv_key),
                Span::styled(format!("~{:.0} tok/s", fit.estimated_tps), THEME.kv_value),
            ]));

            // Training info
            let train_style = match fit.training_fit {
                pmetal_hub::FitLevel::Fits => Style::default().fg(Color::Green),
                pmetal_hub::FitLevel::Tight => Style::default().fg(Color::Yellow),
                pmetal_hub::FitLevel::TooLarge => Style::default().fg(Color::Red),
            };
            lines.push(Line::from(vec![
                Span::styled("Train:   ", THEME.kv_key),
                Span::styled(format!("{:.1} GB", fit.training_memory_gb), train_style),
                Span::styled(
                    format!("  batch {}", fit.recommended_batch_size),
                    THEME.text_dim,
                ),
            ]));

            for note in &fit.notes {
                lines.push(Line::from(Span::styled(note.as_str(), THEME.text_muted)));
            }
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

// --- Helpers ---

/// Check if a directory looks like a model directory.
fn is_model_dir(path: &std::path::Path) -> bool {
    path.join("config.json").exists()
        || path.join("adapter_config.json").exists()
        || path.join("model.safetensors").exists()
        || path.join("model.safetensors.index.json").exists()
        || path.join("lora_weights.safetensors").exists()
}

/// Check if a directory looks like a trained/fine-tuned output directory.
fn is_trained_dir(path: &std::path::Path) -> bool {
    path.join("adapter_config.json").exists() || path.join("lora_weights.safetensors").exists()
}

/// Build a ModelEntry from a model directory path.
fn build_model_entry(
    model_id: &str,
    path: &std::path::Path,
    source: ModelSource,
) -> Option<ModelEntry> {
    let size = dir_size(path);

    let config_path = path.join("config.json");
    let (architecture, params) = if config_path.exists() {
        read_model_config(&config_path)
    } else {
        (None, None)
    };

    let quantization = if path.join("quantization_config.json").exists() {
        Some("Quantized".to_string())
    } else {
        detect_weight_format(path)
    };

    let modified = std::fs::metadata(path)
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

    // For trained models, read training_info.json or parse from dir name ({base}--{suffix})
    let base_model = read_training_info(path).or_else(|| {
        let dir_name = path.file_name()?.to_string_lossy().to_string();
        // Parse "{base}--{suffix}" convention
        dir_name.split_once("--").map(|(base, _)| base.to_string())
    });

    Some(ModelEntry {
        id: model_id.to_string(),
        path: path.to_path_buf(),
        size_bytes: size,
        architecture,
        params,
        quantization,
        modified,
        source,
        base_model,
        fit_estimate: None, // computed after scan in compute_fit_estimates()
    })
}

fn dir_size(path: &std::path::Path) -> u64 {
    const MAX_ENTRIES: usize = 2000;
    walkdir::WalkDir::new(path)
        .follow_links(true)
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

    let params = estimate_params(&json);
    (arch, params)
}

fn estimate_params(config: &serde_json::Value) -> Option<String> {
    let hidden = config["hidden_size"].as_u64()?;
    let layers = config["num_hidden_layers"].as_u64()?;
    let vocab = config["vocab_size"].as_u64().unwrap_or(32000);
    let intermediate = config["intermediate_size"].as_u64().unwrap_or(hidden * 4);

    let embed = vocab * hidden;
    let attn = 4 * hidden * hidden;
    let mlp = 3 * hidden * intermediate;
    let total = embed + layers * (attn + mlp);

    let billions = total as f64 / 1e9;
    if billions >= 1.0 {
        Some(format!("{:.1}B", billions))
    } else {
        Some(format!("{:.0}M", billions * 1000.0))
    }
}

fn detect_weight_format(path: &std::path::Path) -> Option<String> {
    if path.join("model.safetensors").exists() || path.join("model.safetensors.index.json").exists()
    {
        Some("SafeTensors".to_string())
    } else if path.join("pytorch_model.bin").exists() {
        Some("PyTorch".to_string())
    } else {
        let has_shards = std::fs::read_dir(path)
            .ok()?
            .flatten()
            .any(|e| e.file_name().to_string_lossy().ends_with(".safetensors"));
        if has_shards {
            Some("SafeTensors (sharded)".to_string())
        } else {
            None
        }
    }
}

/// Read training metadata from a trained model output directory.
/// Returns the base model ID/path if available.
fn read_training_info(path: &std::path::Path) -> Option<String> {
    let info_path = path.join("training_info.json");
    let content = std::fs::read_to_string(&info_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    json.get("base_model")
        .and_then(|v| v.as_str())
        .map(String::from)
}

/// Write training metadata to an output directory.
pub fn write_training_info(output_dir: &std::path::Path, base_model: &str, base_model_path: &str) {
    let info = serde_json::json!({
        "base_model": base_model,
        "base_model_path": base_model_path,
        "created": chrono::Local::now().to_rfc3339(),
    });
    let _ = std::fs::create_dir_all(output_dir);
    let _ = std::fs::write(
        output_dir.join("training_info.json"),
        serde_json::to_string_pretty(&info).unwrap_or_default(),
    );
}

/// Config file path for persisting custom model directories.
fn custom_dirs_config_path() -> PathBuf {
    pmetal_hub::pmetal_cache_dir().join("custom_model_dirs.json")
}

/// Load custom model directories from config.
fn load_custom_model_dirs() -> Vec<PathBuf> {
    let path = custom_dirs_config_path();
    if !path.exists() {
        return Vec::new();
    }
    let Ok(content) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    serde_json::from_str::<Vec<PathBuf>>(&content).unwrap_or_default()
}

/// Persist custom model directories to config.
fn save_custom_model_dirs(dirs: &[PathBuf]) {
    let path = custom_dirs_config_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(dirs) {
        let _ = std::fs::write(&path, json);
    }
}
