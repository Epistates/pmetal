//! What the dialogs answer, and the dialogs the app opens for itself.

use super::{App, Pending};
use crate::tui::event::AppMsg;
use crate::tui::modal::{HfSearchEntry, Modal, ModalAction, PickerEntry};
use crate::tui::tabs::{ModelEntry, ModelSource, Pick, Tab};

/// The most a training spec allows a learning rate to be.
fn learning_rate_max() -> f64 {
    use pmetal_core::JobFields;
    pmetal_core::jobs::TrainSpec::field_descriptors()
        .iter()
        .find(|d| d.name == "learning_rate")
        .and_then(|d| match d.kind {
            pmetal_core::FieldKind::Number { max, .. } => Some(max),
            _ => None,
        })
        .unwrap_or(1.0)
}

impl App {
    // ── Quitting ────────────────────────────────────────────────────────

    /// `q` / Ctrl+C with nothing to stop: leave, asking first if jobs are
    /// running, since leaving stops them.
    pub(crate) fn request_quit(&mut self) {
        let running = self.running_jobs();
        if running == 0 {
            self.should_quit = true;
            return;
        }
        self.pending = Some(Pending::Quit);
        let what = if running == 1 {
            "1 job is running.".to_string()
        } else {
            format!("{running} jobs are running.")
        };
        self.modal_stack.push(Modal::confirm_no(
            "Quit?",
            vec![
                what,
                "Quitting stops them (training in progress is lost unless it has".to_string(),
                "saved a checkpoint).".to_string(),
                String::new(),
                "Quit and stop them?".to_string(),
            ],
        ));
    }

    /// Leave without asking: a termination signal has already decided.
    pub(crate) fn begin_quit(&mut self) {
        self.should_quit = true;
    }

    // ── Pickers ─────────────────────────────────────────────────────────

    pub(crate) fn model_picker_entries(
        &self,
        keep: impl Fn(&PickerEntry) -> bool,
    ) -> Vec<PickerEntry> {
        self.models
            .models
            .iter()
            .map(|m| {
                let tag = if matches!(m.source, ModelSource::Trained) {
                    " [trained]"
                } else {
                    ""
                };
                let base = m
                    .base_model
                    .as_deref()
                    .map(|b| format!(" (base: {b})"))
                    .unwrap_or_default();
                PickerEntry {
                    id: m.id.clone(),
                    detail: format!(
                        "{} | {}{tag}{base}",
                        m.size_display(),
                        m.architecture.as_deref().unwrap_or("-"),
                    ),
                    path: m.path.display().to_string(),
                }
            })
            .filter(|entry| keep(entry))
            .collect()
    }

    pub(crate) fn dataset_picker_entries(&self) -> Vec<PickerEntry> {
        self.datasets
            .datasets
            .iter()
            .map(|d| PickerEntry {
                id: d.name.clone(),
                detail: format!(
                    "{} | {}",
                    d.format,
                    d.sample_count
                        .map(|c| format!(
                            "{}{c} samples",
                            if d.sample_count_estimated { "~" } else { "" }
                        ))
                        .unwrap_or_else(|| d.size_display())
                ),
                path: d.path.display().to_string(),
            })
            .collect()
    }

    pub(crate) fn confirm_fuse(&mut self, model: &ModelEntry, base: String) {
        let name = model.path.file_name().unwrap_or_default().to_string_lossy();
        self.pending = Some(Pending::FuseStart);
        self.modal_stack.push(Modal::confirm(
            "Fuse LoRA Adapter",
            vec![
                format!("Adapter:  {}", model.id),
                format!("Base:     {base}"),
                format!("Output:   ./output/{name}--fused"),
                String::new(),
                "This merges the LoRA weights into the base model".to_string(),
                "and saves a complete fused model.".to_string(),
                String::new(),
                "Proceed?".to_string(),
            ],
        ));
    }

    /// Remember the base model of an adapter, here and on disk.
    fn set_base_model(&mut self, adapter_id: &str, base: &str) {
        let base_path = self.models.resolve_path(base);
        if let Some(entry) = self.models.models.iter_mut().find(|m| m.id == adapter_id) {
            entry.base_model = Some(base.to_string());
            crate::tui::tabs::record_base_model(&entry.path, base, &base_path);
        }
    }

    // ── Answers ─────────────────────────────────────────────────────────

    /// Handle a dialog's answer.
    pub(crate) fn handle_modal_action(&mut self, action: ModalAction) {
        let target = self.pending.take();
        match action {
            ModalAction::None => {}
            ModalAction::Confirmed => match target {
                Some(Pending::Start(tab)) => self.start(tab),
                Some(Pending::FuseStart) => self.start_fuse(),
                Some(Pending::Quit) => self.should_quit = true,
                Some(Pending::TestInference(dir)) => self.chat_with_output(&dir),
                _ => {}
            },
            ModalAction::SelectModel(id) => self.model_chosen(target, id),
            ModalAction::SelectDataset(path) => {
                if let Some(Pending::PickDataset { tab, field }) = target {
                    let wants_peek = self
                        .forms
                        .get_mut(tab)
                        .map(|form| {
                            form.on_pick(&field, Pick::Dataset(&path));
                            form.wants_dataset_peek() && field == "Dataset"
                        })
                        .unwrap_or(false);
                    if wants_peek {
                        self.peek_dataset(path);
                    }
                }
            }
            ModalAction::SelectTab(tab) => self.switch_tab(tab),
            ModalAction::HfDownload(model_id) => self.download_model(&model_id),
            ModalAction::TextSubmitted(text) => self.text_submitted(target, text),
        }
    }

    fn model_chosen(&mut self, target: Option<Pending>, id: String) {
        match target {
            Some(Pending::PickModel { tab, field }) => {
                let architecture = self.models.find(&id).and_then(|m| m.architecture.clone());
                if let Some(form) = self.forms.get_mut(tab) {
                    form.on_pick(
                        &field,
                        Pick::Model {
                            id: &id,
                            architecture: architecture.as_deref(),
                        },
                    );
                }
            }
            Some(Pending::InferenceModel) => {
                let model = self.models.find(&id).cloned();
                if let Some(m) = &model {
                    self.inference.load_model_settings(&m.path);
                }
                let needs_base = model
                    .as_ref()
                    .is_some_and(|m| m.is_adapter() && m.base_model.is_none());
                if needs_base {
                    self.pending = Some(Pending::InferenceBase(id));
                    let entries = self.model_picker_entries(|e| !e.id.starts_with("trained/"));
                    self.modal_stack.push(Modal::model_picker(entries));
                } else {
                    self.inference.model_id = Some(id);
                }
            }
            Some(Pending::InferenceBase(adapter)) => {
                self.set_base_model(&adapter, &id);
                self.inference.model_id = Some(adapter);
            }
            Some(Pending::FuseBase(adapter)) => {
                self.set_base_model(&adapter, &id);
                if let Some(model) = self.models.find(&adapter).cloned() {
                    self.confirm_fuse(&model, id);
                }
            }
            _ => {}
        }
    }

    fn text_submitted(&mut self, target: Option<Pending>, text: String) {
        match target {
            Some(Pending::HfSearch) => self.search_hf(&text),
            Some(Pending::DownloadModel) => self.download_model(text.trim()),
            Some(Pending::AddModelDir) => {
                let path = pmetal_core::paths::expand_home(text.trim());
                if path.is_dir() {
                    if self.models.add_custom_dir(path) {
                        self.rescan_models();
                    }
                } else {
                    self.modal_stack.push(Modal::error(
                        "Invalid Path",
                        format!("'{}' is not a directory.", text.trim()),
                    ));
                }
            }
            Some(Pending::AddDatasetDir) => {
                let path = pmetal_core::paths::expand_home(text.trim());
                if path.is_dir() {
                    if self.datasets.add_custom_dir(path) {
                        self.rescan_datasets();
                    }
                } else {
                    self.modal_stack.push(Modal::error(
                        "Invalid Path",
                        format!("'{}' is not a directory.", text.trim()),
                    ));
                }
            }
            Some(Pending::ConvertDataset) => self.convert_dataset(text.trim()),
            Some(Pending::SetLearningRate(dir)) => {
                let max = learning_rate_max();
                match text.trim().parse::<f64>() {
                    Ok(lr) if lr > 0.0 && lr <= max => self.write_learning_rate(&dir, lr),
                    _ => self.modal_stack.push(Modal::error(
                        "Invalid Learning Rate",
                        format!(
                            "'{}' is not a learning rate: it must be above 0 and at most {max}. Use e.g. 1e-5 or 0.0001.",
                            text.trim(),
                        ),
                    )),
                }
            }
            _ => {}
        }
    }

    /// Chat with what a finished run wrote to `dir`.
    fn chat_with_output(&mut self, dir: &std::path::Path) {
        let id = self
            .models
            .models
            .iter()
            .find(|m| m.path == dir)
            .map(|m| m.id.clone())
            .unwrap_or_else(|| {
                format!(
                    "trained/{}",
                    dir.file_name().unwrap_or_default().to_string_lossy()
                )
            });
        self.inference.load_model_settings(dir);
        self.inference.model_id = Some(id);
        self.switch_tab(Tab::Inference);
    }

    // ── Hugging Face search ─────────────────────────────────────────────

    pub(crate) fn search_hf(&mut self, query: &str) {
        let Some(tx) = self.tx.clone() else { return };
        let query = query.trim().to_string();
        // A newer search replaces the one in flight.
        self.cancel_hf_search();
        let cancel = tokio_util::sync::CancellationToken::new();
        self.hf_search_cancel = Some(cancel.clone());
        self.modal_stack.push(Modal::progress(
            "Searching",
            format!("Searching HuggingFace for '{query}'..."),
        ));
        tokio::spawn(async move {
            tokio::select! {
                result = pmetal_hub::search_models(&query, 20, None) => {
                    let msg = match result {
                        Ok(results) => AppMsg::HfSearchResults { results },
                        Err(e) => AppMsg::HfSearchError { message: e.to_string() },
                    };
                    let _ = tx.send(msg).await;
                }
                // Esc closed the dialog: the results are no longer wanted.
                _ = cancel.cancelled() => {}
            }
        });
    }

    pub(crate) fn cancel_hf_search(&mut self) {
        if let Some(cancel) = self.hf_search_cancel.take() {
            cancel.cancel();
        }
    }

    pub(crate) fn show_hf_results(&mut self, results: Vec<pmetal_hub::HfSearchResult>) {
        if matches!(self.modal_stack.last(), Some(Modal::Progress { .. })) {
            self.modal_stack.pop();
        }
        // Fit comes from the device's own report, never from a guess at RAM.
        let device = self.models.device_spec.as_ref();
        let entries: Vec<HfSearchEntry> = results
            .iter()
            .map(|r| {
                let params = pmetal_hub::fit::format_params(r.estimated_params_b.unwrap_or(0.0));
                let fit = device.and_then(|dev| r.fit(dev, pmetal_hub::fit::DEFAULT_FIT_CONTEXT));
                let (memory, fit_level, tps) = match fit {
                    Some(fit) => (
                        format!("{:.1}GB", fit.total_required_gb),
                        fit.fit_level,
                        format!("{:.0}", fit.estimated_tps),
                    ),
                    None => (
                        r.safetensors_bytes
                            .map(pmetal_core::format::human_size)
                            .unwrap_or_else(|| "?".to_string()),
                        pmetal_hub::FitLevel::Fits,
                        "-".to_string(),
                    ),
                };
                HfSearchEntry {
                    model_id: r.model_id.clone(),
                    params,
                    downloads: pmetal_hub::search::format_downloads(r.downloads),
                    memory,
                    fit_level,
                    estimated_tps: tps,
                }
            })
            .collect();
        if entries.is_empty() {
            self.modal_stack
                .push(Modal::error("No Results", "No models found."));
        } else {
            self.modal_stack.push(Modal::hf_search(entries));
        }
    }
}
