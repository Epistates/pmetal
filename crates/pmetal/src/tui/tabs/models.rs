//! Models tab: browse cached Hugging Face models, trained outputs and custom
//! model directories.
//!
//! The scan reads directories, config files and weight headers, so it runs off
//! the interface's thread ([`scan`] is a plain function the app hands to a
//! blocking task) and the tab shows "Scanning..." until it reports. Parameter
//! counts and memory fit come from `pmetal_hub` (the model's own config and
//! the weights' measured size against what the GPU reports it can address),
//! never from guesses about system RAM.

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
        pmetal_core::format::human_size(self.size_bytes)
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

/// Find every model the plan reaches (`pmetal_hub::local::scan`, the scan every
/// surface shares) and present it for the table: trained outputs first, then
/// the hub cache, then the custom directories, alphabetically within each.
/// Blocking.
pub fn scan(plan: &ScanPlan) -> Vec<ModelEntry> {
    let found = pmetal_hub::local::scan(&pmetal_hub::local::ScanPlan {
        hub_cache: plan.hub_cache.clone(),
        output_root: plan.output_root.clone(),
        custom_dirs: plan.custom_dirs.clone(),
        well_known: Vec::new(),
    });

    let mut models: Vec<ModelEntry> = Vec::new();
    for model in found {
        let entry = entry_of(&model, plan);
        // Two folders of the same name are one row, the first found.
        if models.iter().all(|m| m.id != entry.id) {
            models.push(entry);
        }
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

/// A found model as the table shows it.
fn entry_of(model: &pmetal_hub::local::LocalModel, plan: &ScanPlan) -> ModelEntry {
    use pmetal_hub::local::ModelSource as Found;

    // Adapters are listed as trained models wherever they were found.
    let trained = model.source == Found::Trained || model.is_adapter;
    let id = if trained {
        format!("trained/{}", model.folder)
    } else if model.source == Found::HfCache {
        model.id.clone()
    } else {
        model.folder.clone()
    };
    let source = if trained {
        ModelSource::Trained
    } else if model.source == Found::HfCache {
        ModelSource::HfCache
    } else {
        let root = plan
            .custom_dirs
            .iter()
            .find(|dir| model.path.starts_with(dir))
            .cloned()
            .or_else(|| model.path.parent().map(Path::to_path_buf))
            .unwrap_or_default();
        ModelSource::Custom(root)
    };
    let quantization = if model.is_adapter && trained {
        Some("LoRA Adapter".to_string())
    } else if model.quantized {
        Some("Quantized".to_string())
    } else {
        model.weights.map(str::to_string)
    };
    // Parameter count from the hub's own spec (the same one the fit estimate
    // is made from).
    let params = pmetal_hub::model_spec_from_dir(&model.path)
        .filter(|spec| spec.params_b > 0.0)
        .map(|spec| pmetal_hub::format_params(spec.params_b));
    let modified = model
        .modified
        .map(|t| {
            let hours = t.elapsed().unwrap_or_default().as_secs() / 3600;
            if hours < 24 {
                format!("{hours}h ago")
            } else {
                format!("{}d ago", hours / 24)
            }
        })
        .unwrap_or_else(|| "unknown".to_string());
    ModelEntry {
        id,
        path: model.path.clone(),
        size_bytes: model.size_bytes,
        architecture: model.architecture.clone(),
        params,
        quantization,
        modified,
        source,
        base_model: model.base_model.clone(),
        fit_estimate: None,
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

/// Whether a directory is a LoRA adapter (has adapter weights or config).
pub fn is_lora_adapter(path: &Path) -> bool {
    pmetal_hub::local::is_lora_adapter(path)
}

/// Record, for an adapter that does not say, the base model it was trained from.
/// What the folder already records (a dataset, a start time) is kept.
pub fn record_base_model(output_dir: &Path, base_model: &str, base_model_path: &str) {
    let mut info = pmetal_core::TrainingInfo::read(output_dir).unwrap_or_default();
    info.base_model = Some(base_model.to_string());
    info.base_model_path = Some(base_model_path.to_string());
    info.write_or_warn(output_dir);
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
    fn a_base_model_is_recorded_and_read_back_without_losing_the_rest() {
        let dir = tempfile::tempdir().unwrap();
        pmetal_core::TrainingInfo {
            dataset: Some("acme/data".into()),
            ..Default::default()
        }
        .write(dir.path())
        .unwrap();
        record_base_model(dir.path(), "org/base", "/cache/base");
        let info = pmetal_core::TrainingInfo::read(dir.path()).unwrap();
        assert_eq!(info.base_model.as_deref(), Some("org/base"));
        assert_eq!(info.dataset.as_deref(), Some("acme/data"));
        assert_eq!(info.base_model_path.as_deref(), Some("/cache/base"));
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
