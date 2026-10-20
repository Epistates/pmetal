//! Models tab: browse cached Hugging Face models, trained outputs and custom
//! model directories.
//!
//! The scan reads directories, config files and weight headers, so it runs off
//! the interface's thread ([`scan`] is a plain function the app hands to a
//! blocking task) and the tab shows "Scanning..." until it reports. Parameter
//! counts and memory fit come from `pmetal_hub` (the model's own config and
//! the weights' measured size against what the GPU reports it can address),
//! never from guesses about system RAM.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, HighlightSpacing, Paragraph, Row, Scrollbar, ScrollbarOrientation,
    ScrollbarState, StatefulWidget, Table, TableState, Widget, Wrap,
};

use crate::tui::tabs::custom_dirs::CustomDirs;
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

    /// Whether this is a LoRA adapter (not a full model).
    pub fn is_adapter(&self) -> bool {
        is_lora_adapter(&self.path)
    }
}

/// Where a scan looks.
#[derive(Debug, Clone)]
pub struct ScanPlan {
    /// The directory trained models are written under (`./output`).
    pub output_root: PathBuf,
    /// The Hugging Face hub cache.
    pub hub_cache: PathBuf,
    /// Directories the user added.
    pub custom_dirs: Vec<PathBuf>,
    /// What the GPU can address, for the fit estimate; `None` without a GPU.
    pub device: Option<pmetal_hub::DeviceSpec>,
}

/// The device the fit estimate is made for, from the GPU's own report of how
/// much it can address and how fast: never derived from the system's RAM.
pub fn query_device_spec() -> Option<pmetal_hub::DeviceSpec> {
    let ctx = pmetal_metal::context::MetalContext::global().ok()?;
    let props = ctx.properties();
    Some(pmetal_hub::DeviceSpec::from_device(
        props.recommended_working_set_size,
        props.memory_bandwidth_gbps,
        props.has_unified_memory,
    ))
}

/// Find every model the plan reaches: trained outputs first, then the hub
/// cache, then the custom directories. Blocking.
pub fn scan(plan: &ScanPlan) -> Vec<ModelEntry> {
    let mut models: Vec<ModelEntry> = Vec::new();

    // 1. Trained outputs
    let mut seen: HashSet<String> = HashSet::new();
    scan_trained_dir(&plan.output_root, &mut models, &mut seen, 0, 2);

    // 2. The hub cache, with the snapshot each repository's `refs/main` names
    for cached in pmetal_hub::list_cached_models_in(&plan.hub_cache) {
        if let Some(entry) =
            build_model_entry(&cached.repo_id, &cached.snapshot, ModelSource::HfCache)
        {
            models.push(entry);
        }
    }

    // 3. Custom directories
    for dir in &plan.custom_dirs {
        scan_custom_dir(dir, &mut models);
    }

    // Trained models first, then alphabetically within each group
    models.sort_by(|a, b| {
        let a_trained = matches!(a.source, ModelSource::Trained);
        let b_trained = matches!(b.source, ModelSource::Trained);
        b_trained.cmp(&a_trained).then(a.id.cmp(&b.id))
    });

    if let Some(device) = &plan.device {
        for model in &mut models {
            // The config's architecture and the weights' measured size.
            if let Some(spec) = pmetal_hub::model_spec_from_dir(&model.path) {
                model.fit_estimate = Some(pmetal_hub::estimate_fit(&spec, device));
            }
        }
    }
    models
}

/// Scan a directory for trained/fine-tuned models, a couple of levels deep
/// (`output/`, `output/distilled/`, ...).
fn scan_trained_dir(
    dir: &Path,
    models: &mut Vec<ModelEntry>,
    seen: &mut HashSet<String>,
    depth: usize,
    max_depth: usize,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
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
            if seen.insert(model_id.clone()) {
                if let Some(mut entry) = build_model_entry(&model_id, &path, ModelSource::Trained) {
                    if is_lora_adapter(&path) {
                        entry.quantization = Some("LoRA Adapter".to_string());
                    }
                    models.push(entry);
                }
            }
        } else if depth < max_depth {
            scan_trained_dir(&path, models, seen, depth + 1, max_depth);
        }
    }
}

/// Scan a custom directory: the directory itself if it is a model, else the
/// models directly inside it.
fn scan_custom_dir(dir: &Path, models: &mut Vec<ModelEntry>) {
    let candidates: Vec<PathBuf> = if is_model_dir(dir) {
        vec![dir.to_path_buf()]
    } else {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir() && is_model_dir(p))
            .collect()
    };
    for path in candidates {
        let adapter = is_lora_adapter(&path);
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let model_id = if adapter {
            format!("trained/{name}")
        } else {
            name
        };
        if models.iter().any(|m| m.id == model_id) {
            continue;
        }
        let source = if adapter {
            ModelSource::Trained
        } else {
            ModelSource::Custom(dir.to_path_buf())
        };
        if let Some(mut entry) = build_model_entry(&model_id, &path, source) {
            if adapter {
                entry.quantization = Some("LoRA Adapter".to_string());
            }
            models.push(entry);
        }
    }
}

/// Models tab state.
pub struct ModelsTab {
    pub models: Vec<ModelEntry>,
    pub table_state: TableState,
    pub scrollbar_state: ScrollbarState,
    /// A scan is running.
    pub loading: bool,
    pub search_query: String,
    pub searching: bool,
    /// User-configured custom directories to scan for models.
    pub custom_dirs: CustomDirs,
    /// What the GPU reports it can address (queried once at startup).
    pub device_spec: Option<pmetal_hub::DeviceSpec>,
}

impl ModelsTab {
    pub fn new() -> Self {
        Self::with_device(query_device_spec())
    }

    /// A tab whose fit estimates are made for `device`.
    pub fn with_device(device_spec: Option<pmetal_hub::DeviceSpec>) -> Self {
        Self {
            models: Vec::new(),
            table_state: TableState::default(),
            scrollbar_state: ScrollbarState::default(),
            loading: true,
            search_query: String::new(),
            searching: false,
            custom_dirs: CustomDirs::models(),
            device_spec,
        }
    }

    /// What a scan should look at now.
    pub fn scan_plan(&self) -> ScanPlan {
        ScanPlan {
            output_root: PathBuf::from("./output"),
            hub_cache: pmetal_hub::cache_dir(),
            custom_dirs: self.custom_dirs.dirs().to_vec(),
            device: self.device_spec.clone(),
        }
    }

    /// A scan is about to run.
    pub fn begin_scan(&mut self) {
        self.loading = true;
    }

    /// A scan finished. The selection stays on the same model if it is still
    /// there.
    pub fn set_models(&mut self, models: Vec<ModelEntry>) {
        let selected = self.selected_model().map(|m| m.id.clone());
        self.models = models;
        self.loading = false;
        let shown = self.filtered_models();
        let index = selected
            .and_then(|id| shown.iter().position(|m| m.id == id))
            .or(if shown.is_empty() { None } else { Some(0) });
        let count = shown.len();
        self.table_state.select(index);
        self.scrollbar_state = ScrollbarState::new(count).position(index.unwrap_or(0));
    }

    /// Add a directory to scan. Returns whether it was new.
    pub fn add_custom_dir(&mut self, dir: PathBuf) -> bool {
        self.custom_dirs.add(dir)
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

    /// The search text changed: keep the selection inside the shorter list.
    pub fn search_changed(&mut self) {
        let count = self.filtered_models().len();
        self.table_state
            .select(if count == 0 { None } else { Some(0) });
        self.scrollbar_state = ScrollbarState::new(count);
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

    /// The model with this id, if the tab knows it.
    pub fn find(&self, id: &str) -> Option<&ModelEntry> {
        self.models.iter().find(|m| m.id == id)
    }

    /// The path a model id stands for: its directory if the tab knows it, else
    /// the id itself (a Hugging Face id the trainer will resolve).
    pub fn resolve_path(&self, id: &str) -> String {
        self.find(id)
            .map(|m| m.path.display().to_string())
            .unwrap_or_else(|| id.to_string())
    }
}

impl Default for ModelsTab {
    fn default() -> Self {
        Self::new()
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

/// How a fit level is drawn: green fits, yellow is tight, red is too large,
/// and each is also spelled out.
pub fn fit_style(level: pmetal_hub::FitLevel) -> Style {
    match level {
        pmetal_hub::FitLevel::Fits => THEME.text_success,
        pmetal_hub::FitLevel::Tight => THEME.text_warning,
        pmetal_hub::FitLevel::TooLarge => THEME.text_error,
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

        if self.loading && self.models.is_empty() {
            Paragraph::new("\n  Scanning the model directories...")
                .style(THEME.text_muted)
                .block(block)
                .render(area, buf);
            return;
        }

        let filtered = self.filtered_models();
        if filtered.is_empty() {
            Paragraph::new(if self.search_query.is_empty() {
                "\n  No models found.\n\n  Download:  d (or pmetal download <model-id>)\n  Add dir:   a"
            } else {
                "\n  No models match the filter."
            })
            .style(THEME.text_muted)
            .block(block)
            .render(area, buf);
            return;
        }

        let header = Row::new(vec!["Model", "Size", "Arch / Base", "Fit", "Source"])
            .style(THEME.table_header)
            .height(1);

        let rows: Vec<Row<'_>> = filtered
            .iter()
            .enumerate()
            .map(|(i, model)| {
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
                let fit_cell = match &model.fit_estimate {
                    Some(fit) => Cell::new(format!(
                        "{:.1}GB {}",
                        fit.total_required_gb,
                        fit.fit_level.label()
                    ))
                    .style(fit_style(fit.fit_level)),
                    None => Cell::new("-"),
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
                Constraint::Length(16),
                Constraint::Length(10),
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

        let kv = |key: &'static str, value: String, style: Style| {
            Line::from(vec![
                Span::styled(format!("{key:<9}"), THEME.kv_key),
                Span::styled(value, style),
            ])
        };
        let mut lines = vec![
            kv("ID:", model.id.clone(), THEME.kv_value),
            kv("Size:", model.size_display(), THEME.kv_value),
            kv("Source:", source_label, THEME.text_dim),
            kv("Path:", model.path.display().to_string(), THEME.text_dim),
        ];
        if let Some(arch) = &model.architecture {
            lines.push(kv("Arch:", arch.clone(), THEME.kv_value));
        }
        if let Some(params) = &model.params {
            lines.push(kv("Params:", params.clone(), THEME.kv_value));
        }
        if let Some(quant) = &model.quantization {
            lines.push(kv("Format:", quant.clone(), THEME.kv_value));
        }
        if let Some(base) = &model.base_model {
            lines.push(kv("Base:", base.clone(), THEME.kv_value));
        }
        lines.push(Line::from(""));
        lines.push(kv("Modified:", model.modified.clone(), THEME.text_dim));

        // Fit estimation section
        if let Some(fit) = &model.fit_estimate {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("--- Memory Fit ---", THEME.kv_key)));
            lines.push(Line::from(vec![
                Span::styled(format!("{:<9}", "Status:"), THEME.kv_key),
                Span::styled(fit.fit_level.label(), fit_style(fit.fit_level)),
                Span::styled(
                    format!(
                        " ({:.0}% of {:.0}GB)",
                        fit.utilization_pct, fit.available_gb
                    ),
                    THEME.text_dim,
                ),
            ]));
            lines.push(Line::from(vec![
                Span::styled(format!("{:<9}", "Weights:"), THEME.kv_key),
                Span::styled(format!("{:.1} GB", fit.weights_gb), THEME.kv_value),
                Span::styled("  KV: ", THEME.kv_key),
                Span::styled(format!("{:.1} GB", fit.kv_cache_gb), THEME.kv_value),
            ]));
            lines.push(kv(
                "Decode:",
                format!("~{:.0} tok/s", fit.estimated_tps),
                THEME.kv_value,
            ));
            lines.push(Line::from(vec![
                Span::styled(format!("{:<9}", "Train:"), THEME.kv_key),
                Span::styled(
                    format!("{:.1} GB", fit.training_memory_gb),
                    fit_style(fit.training_fit),
                ),
                Span::styled(
                    format!(
                        "  {}  batch {}",
                        fit.training_fit.label(),
                        fit.recommended_batch_size
                    ),
                    THEME.text_dim,
                ),
            ]));
            for note in &fit.notes {
                lines.push(Line::from(Span::styled(note.as_str(), THEME.text_muted)));
            }
        } else if self.device_spec.is_none() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "No GPU was found, so memory fit is not estimated.",
                THEME.text_muted,
            )));
        }

        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .render(inner, buf);
    }
}

// --- Helpers ---

/// Whether a directory looks like a model directory.
pub fn is_model_dir(path: &Path) -> bool {
    path.join("config.json").exists()
        || path.join("adapter_config.json").exists()
        || path.join("model.safetensors").exists()
        || path.join("model.safetensors.index.json").exists()
        || path.join("lora_weights.safetensors").exists()
        || path.join("adapter_model.safetensors").exists()
        || has_gguf_files(path)
}

fn has_gguf_files(path: &Path) -> bool {
    std::fs::read_dir(path)
        .ok()
        .map(|entries| {
            entries
                .flatten()
                .any(|e| e.path().extension().is_some_and(|ext| ext == "gguf"))
        })
        .unwrap_or(false)
}

/// Whether a directory is a LoRA adapter (has adapter weights or config).
pub fn is_lora_adapter(path: &Path) -> bool {
    path.join("adapter_config.json").exists()
        || path.join("adapter_model.safetensors").exists()
        || path.join("lora_weights.safetensors").exists()
}

/// Build a ModelEntry from a model directory path.
fn build_model_entry(model_id: &str, path: &Path, source: ModelSource) -> Option<ModelEntry> {
    let size = dir_size(path);

    // Architecture from the config; parameter count from the hub's own spec
    // (the same one the fit estimate is made from).
    let architecture = read_architecture(&path.join("config.json"));
    let params = pmetal_hub::model_spec_from_dir(path)
        .filter(|spec| spec.params_b > 0.0)
        .map(|spec| pmetal_hub::format_params(spec.params_b));

    let quantization = if path.join("quantization_config.json").exists() {
        Some("Quantized".to_string())
    } else {
        detect_weight_format(path)
    };

    let modified = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .map(|t| {
            let hours = t.elapsed().unwrap_or_default().as_secs() / 3600;
            if hours < 24 {
                format!("{hours}h ago")
            } else {
                format!("{}d ago", hours / 24)
            }
        })
        .unwrap_or_else(|| "unknown".to_string());

    // For trained models, read training_info.json or parse from dir name ({base}--{suffix})
    let base_model = read_training_info(path).or_else(|| {
        if !is_lora_adapter(path) {
            return None;
        }
        let dir_name = path.file_name()?.to_string_lossy().to_string();
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
        fit_estimate: None,
    })
}

fn dir_size(path: &Path) -> u64 {
    const MAX_ENTRIES: usize = 4000;
    walkdir::WalkDir::new(path)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
        .take(MAX_ENTRIES)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
        .sum()
}

fn read_architecture(config: &Path) -> Option<String> {
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(config).ok()?).ok()?;
    json["model_type"]
        .as_str()
        .or_else(|| json["architectures"].as_array()?.first()?.as_str())
        .map(String::from)
}

fn detect_weight_format(path: &Path) -> Option<String> {
    if path.join("model.safetensors").exists() || path.join("model.safetensors.index.json").exists()
    {
        Some("SafeTensors".to_string())
    } else if path.join("pytorch_model.bin").exists() {
        Some("PyTorch".to_string())
    } else {
        let mut has_shards = false;
        let mut has_gguf = false;
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.ends_with(".safetensors") {
                    has_shards = true;
                } else if name.ends_with(".gguf") {
                    has_gguf = true;
                }
            }
        }
        if has_shards {
            Some("SafeTensors (sharded)".to_string())
        } else if has_gguf {
            Some("GGUF".to_string())
        } else {
            None
        }
    }
}

/// Read training metadata from a trained model output directory.
/// Returns the base model ID/path if available.
fn read_training_info(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path.join("training_info.json")).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    json.get("base_model")
        .and_then(|v| v.as_str())
        .map(String::from)
}

/// Write training metadata to an output directory.
pub fn write_training_info(output_dir: &Path, base_model: &str, base_model_path: &str) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    const CONFIG: &str = r#"{"model_type":"llama","hidden_size":64,"num_hidden_layers":2,
        "num_attention_heads":4,"num_key_value_heads":4,"intermediate_size":128,"vocab_size":1000}"#;

    fn plan(root: &Path) -> ScanPlan {
        ScanPlan {
            output_root: root.join("output"),
            hub_cache: root.join("hub"),
            custom_dirs: vec![root.join("mine")],
            device: Some(pmetal_hub::DeviceSpec::from_device(16 << 30, 200.0, true)),
        }
    }

    #[test]
    fn the_scan_finds_trained_cached_and_custom_models_in_that_order() {
        let root = tempfile::tempdir().unwrap();
        let r = root.path();
        // A trained adapter two levels down in the output directory.
        write(&r.join("output/grpo/run--lora/adapter_config.json"), "{}");
        write(
            &r.join("output/grpo/run--lora/training_info.json"),
            r#"{"base_model":"org/base"}"#,
        );
        // A cached model with two snapshots; main points at the older one.
        let sha_a = "a".repeat(40);
        let sha_b = "b".repeat(40);
        write(
            &r.join(format!(
                "hub/models--org--base/snapshots/{sha_a}/config.json"
            )),
            CONFIG,
        );
        write(
            &r.join(format!(
                "hub/models--org--base/snapshots/{sha_b}/config.json"
            )),
            "{}",
        );
        write(&r.join("hub/models--org--base/refs/main"), &sha_a);
        // A directory the user added, holding one model.
        write(&r.join("mine/local-model/config.json"), CONFIG);

        let models = scan(&plan(r));
        let ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["trained/run--lora", "local-model", "org/base"]);

        let trained = &models[0];
        assert_eq!(trained.source, ModelSource::Trained);
        assert_eq!(trained.base_model.as_deref(), Some("org/base"));
        assert!(trained.is_adapter());
        assert_eq!(trained.quantization.as_deref(), Some("LoRA Adapter"));

        let cached = &models[2];
        assert!(
            cached.path.ends_with(&sha_a),
            "the snapshot refs/main names, not the newest: {}",
            cached.path.display()
        );
        assert_eq!(cached.architecture.as_deref(), Some("llama"));
        // Parameters and fit come from the hub's own model spec, not a formula here.
        assert!(cached.params.is_some());
        assert!(cached.fit_estimate.is_some());
    }

    #[test]
    fn without_a_device_the_models_are_listed_but_not_fitted() {
        let root = tempfile::tempdir().unwrap();
        write(&root.path().join("mine/m/config.json"), CONFIG);
        let mut plan = plan(root.path());
        plan.device = None;
        let models = scan(&plan);
        assert_eq!(models.len(), 1);
        assert!(models[0].fit_estimate.is_none());
    }

    #[test]
    fn a_rescan_keeps_the_selection_on_the_same_model() {
        let entry = |id: &str| ModelEntry {
            id: id.into(),
            path: PathBuf::from(id),
            size_bytes: 1 << 20,
            architecture: None,
            params: None,
            quantization: None,
            modified: "now".into(),
            source: ModelSource::HfCache,
            base_model: None,
            fit_estimate: None,
        };
        let mut tab = ModelsTab::with_device(None);
        tab.set_models(vec![entry("a"), entry("b"), entry("c")]);
        tab.next_row();
        tab.next_row();
        assert_eq!(tab.selected_model().unwrap().id, "c");
        // A new model sorts in front; the selection follows "c".
        tab.begin_scan();
        assert!(tab.loading);
        tab.set_models(vec![entry("0"), entry("a"), entry("b"), entry("c")]);
        assert_eq!(tab.selected_model().unwrap().id, "c");
        assert!(!tab.loading);
        // The model is gone: the selection falls back to the first.
        tab.set_models(vec![entry("a")]);
        assert_eq!(tab.selected_model().unwrap().id, "a");
        tab.set_models(Vec::new());
        assert!(tab.selected_model().is_none());
    }

    #[test]
    fn filtering_narrows_the_list_and_resets_the_selection() {
        let mut tab = ModelsTab::with_device(None);
        let entry = |id: &str| ModelEntry {
            id: id.into(),
            path: PathBuf::from(id),
            size_bytes: 0,
            architecture: None,
            params: None,
            quantization: None,
            modified: String::new(),
            source: ModelSource::HfCache,
            base_model: None,
            fit_estimate: None,
        };
        tab.set_models(vec![entry("qwen/a"), entry("llama/b"), entry("qwen/c")]);
        tab.next_row();
        tab.search_query = "QWEN".into();
        tab.search_changed();
        assert_eq!(tab.filtered_models().len(), 2);
        assert_eq!(tab.selected_model().unwrap().id, "qwen/a");
        tab.search_query = "nothing".into();
        tab.search_changed();
        assert!(tab.selected_model().is_none());
    }

    #[test]
    fn a_model_id_resolves_to_its_directory_or_stays_a_hub_id() {
        let mut tab = ModelsTab::with_device(None);
        tab.set_models(vec![ModelEntry {
            id: "org/m".into(),
            path: PathBuf::from("/cache/snap"),
            size_bytes: 0,
            architecture: None,
            params: None,
            quantization: None,
            modified: String::new(),
            source: ModelSource::HfCache,
            base_model: None,
            fit_estimate: None,
        }]);
        assert_eq!(tab.resolve_path("org/m"), "/cache/snap");
        assert_eq!(tab.resolve_path("other/model"), "other/model");
    }

    #[test]
    fn training_info_is_written_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        write_training_info(dir.path(), "org/base", "/cache/base");
        assert_eq!(read_training_info(dir.path()).as_deref(), Some("org/base"));
    }

    #[test]
    fn the_tab_draws_in_every_state_without_panicking() {
        let mut tab = ModelsTab::with_device(None);
        let sizes = [(0, 0), (1, 1), (20, 5), (60, 12), (120, 30)];
        for &(w, h) in &sizes {
            let area = Rect::new(0, 0, w, h);
            let mut buf = Buffer::empty(area);
            (&mut tab).render(area, &mut buf);
        }
        tab.set_models(Vec::new());
        for &(w, h) in &sizes {
            let area = Rect::new(0, 0, w, h);
            let mut buf = Buffer::empty(area);
            (&mut tab).render(area, &mut buf);
        }
    }
}
