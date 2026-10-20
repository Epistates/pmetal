//! Starting jobs, following them, and settling what they leave behind.

use std::path::{Path, PathBuf};
use std::time::Duration;

use pmetal_core::events::{JobEvent, MetricPayload};
use pmetal_supervisor::{EventKind, JobId, ProcessStatus, Stream};

use super::{App, Pending, datasets_seq, history_seq, models_seq};
use crate::tui::event::AppMsg;
use crate::tui::jobs::{LaunchPlan, Owner};
use crate::tui::modal::Modal;
use crate::tui::tabs::{self, Applied, DatasetPeek, ModelEntry, Tab, write_training_info};

/// Tabs whose finished run leaves a model to chat with.
fn leaves_a_chat_model(tab: Tab) -> bool {
    matches!(
        tab,
        Tab::Training | Tab::Distillation | Tab::Grpo | Tab::Rlkd | Tab::Preference
    )
}

impl App {
    // ── Counting ────────────────────────────────────────────────────────

    /// Jobs whose process is alive, for the footer and the quit prompt.
    pub(crate) fn running_jobs(&self) -> usize {
        self.hub.as_ref().map_or(0, |hub| hub.running())
    }

    // ── Background scans ────────────────────────────────────────────────

    pub(crate) fn rescan_models(&mut self) {
        let Some(tx) = self.tx.clone() else { return };
        let seq = self.next_scan(models_seq);
        self.models.begin_scan();
        let plan = self.models.scan_plan();
        tokio::task::spawn_blocking(move || {
            let models = tabs::scan_models(&plan);
            let _ = tx.blocking_send(AppMsg::ModelsScanned { seq, models });
        });
    }

    pub(crate) fn rescan_datasets(&mut self) {
        let Some(tx) = self.tx.clone() else { return };
        let seq = self.next_scan(datasets_seq);
        self.datasets.begin_scan();
        let plan = self.datasets.scan_plan();
        tokio::task::spawn_blocking(move || {
            let datasets = tabs::scan_datasets(&plan);
            let _ = tx.blocking_send(AppMsg::DatasetsScanned { seq, datasets });
        });
    }

    pub(crate) fn rescan_history(&mut self) {
        let Some(tx) = self.tx.clone() else { return };
        let seq = self.next_scan(history_seq);
        self.jobs.begin_scan();
        tokio::task::spawn_blocking(move || {
            let runs = tabs::scan_history(&tabs::history_roots());
            let _ = tx.blocking_send(AppMsg::JobHistoryScanned { seq, runs });
        });
    }

    pub(crate) fn peek_dataset(&mut self, path: String) {
        let Some(tx) = self.tx.clone() else { return };
        tokio::task::spawn_blocking(move || {
            let peek = DatasetPeek::sample(Path::new(&path));
            let _ = tx.blocking_send(AppMsg::DatasetPeeked { path, peek });
        });
    }

    pub(crate) fn sample_memory(&mut self) {
        let Some(tx) = self.tx.clone() else { return };
        let installed = self.device.memory.total_bytes;
        tokio::task::spawn_blocking(move || {
            let used = tabs::read_memory_in_use(installed);
            let _ = tx.blocking_send(AppMsg::MemorySampled(used));
        });
    }

    /// Show the steps of the metrics file given on the command line.
    pub(crate) fn follow_metrics_file(&mut self) {
        let (Some(path), Some(tx)) = (self.metrics_file.clone(), self.tx.clone()) else {
            return;
        };
        self.dashboard
            .follow_file(format!("file {}", path.display()));
        tokio::spawn(async move {
            let mut tail = pmetal_supervisor::Tail::new(path);
            loop {
                let steps: Vec<_> = tail
                    .poll(false)
                    .await
                    .iter()
                    .filter_map(|line| match pmetal_supervisor::parse_record(line, "") {
                        Some(JobEvent::Metric {
                            payload: MetricPayload::Step(step),
                            ..
                        }) => Some(step),
                        _ => None,
                    })
                    .collect();
                if !steps.is_empty() && tx.send(AppMsg::MetricsFile(steps)).await.is_err() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
        });
    }

    // ── Messages ────────────────────────────────────────────────────────

    /// Handle a message from background work.
    pub(crate) fn handle_app_msg(&mut self, msg: AppMsg) {
        match msg {
            AppMsg::Job(event) => self.on_job_event(event),
            AppMsg::JobsLagged => self.resync_jobs(),
            AppMsg::ModelsScanned { seq, models } => {
                if self.scan_is_current(models_seq, seq) {
                    self.models.set_models(models);
                }
            }
            AppMsg::DatasetsScanned { seq, datasets } => {
                if self.scan_is_current(datasets_seq, seq) {
                    self.datasets.set_datasets(datasets);
                }
            }
            AppMsg::JobHistoryScanned { seq, runs } => {
                if self.scan_is_current(history_seq, seq) {
                    self.jobs.set_history(runs);
                }
            }
            AppMsg::DatasetPeeked { path, peek } => {
                for tab in Tab::ALL {
                    if let Some(form) = self.forms.get_mut(*tab) {
                        if form.wants_dataset_peek() {
                            form.set_peek(&path, peek.clone());
                        }
                    }
                }
            }
            AppMsg::MemorySampled(used) => self.device.set_used(used),
            AppMsg::MetricsFile(steps) => {
                for step in steps {
                    self.dashboard.push_step(step);
                }
            }
            AppMsg::Reachable { job, reachable } => {
                let alive = self.hub.as_ref().is_some_and(|hub| hub.is_active(&job));
                if reachable && alive {
                    if let Some(Owner::Tab(tab)) =
                        self.hub.as_ref().and_then(|h| h.owner(&job)).cloned()
                    {
                        if let Some(form) = self.forms.get_mut(tab) {
                            form.on_reachable();
                        }
                    }
                }
            }
            AppMsg::HfSearchResults { results } => self.show_hf_results(results),
            AppMsg::HfSearchError { message } => {
                if matches!(self.modal_stack.last(), Some(Modal::Progress { .. })) {
                    self.modal_stack.pop();
                }
                self.modal_stack
                    .push(Modal::error("Search Failed", message));
            }
        }
    }

    // ── Job events ──────────────────────────────────────────────────────

    pub(crate) fn on_job_event(&mut self, event: pmetal_supervisor::Event) {
        let pmetal_supervisor::Event { job, kind } = event;
        let Some(owner) = self.hub.as_ref().and_then(|hub| hub.owner(&job)).cloned() else {
            return;
        };
        self.jobs.apply(&job, &kind);
        match owner {
            Owner::Tab(Tab::Inference) => self.on_chat_event(&job, &kind),
            Owner::Tab(tab) => self.on_tab_event(tab, &job, &kind),
            Owner::Download(_) | Owner::Convert | Owner::Fuse => {
                if let EventKind::Exited(status) = &kind {
                    self.on_background_exit(&owner, status);
                }
            }
        }
    }

    fn on_tab_event(&mut self, tab: Tab, job: &str, kind: &EventKind) {
        let following = self.dashboard.following() == Some(job);
        if following {
            if let EventKind::Job(event) = kind {
                match event {
                    JobEvent::Metric {
                        payload: MetricPayload::Step(step),
                        ..
                    } => self.dashboard.push_step(step.clone()),
                    JobEvent::Phase { phase, .. } => {
                        self.dashboard.phase = Some(phase.message());
                    }
                    _ => {}
                }
            }
        }
        let tx = self.tx.clone();
        let Some(form) = self.forms.get_mut(tab) else {
            return;
        };
        let applied = form.panel_mut().apply(kind);
        if let EventKind::Output(line) = kind {
            form.on_line(&line.text);
            if let (Some(address), Some(tx)) = (form.take_announcement(), tx) {
                probe_address(tx, job.to_string(), address);
            }
        }
        if let Applied::Ended { success } = applied {
            self.on_tab_job_ended(tab, job, success);
        }
    }

    /// A tab's job has ended: let go of the monitor, and see what it left.
    fn on_tab_job_ended(&mut self, tab: Tab, job: &str, success: bool) {
        self.dashboard.release(job);
        let title = self.forms.get(tab).map_or("The job", |f| f.panel().title);
        let output = self
            .hub
            .as_ref()
            .and_then(|hub| hub.supervisor().summary(job).ok())
            .and_then(|summary| summary.output);
        if !success {
            let failed = matches!(
                self.forms.get(tab).map(|f| f.panel().run.clone()),
                Some(crate::tui::tabs::RunState::Failed(_))
            );
            if failed {
                self.notify(format!("{title} failed (see the tab for why)"));
            }
            return;
        }
        match tab {
            Tab::Quantize | Tab::Merge => self.rescan_models(),
            tab if leaves_a_chat_model(tab) => {
                if let Some(dir) = output.map(PathBuf::from) {
                    // Keep the output where the Models tab will find it next time.
                    let canonical = dir.canonicalize().unwrap_or(dir);
                    self.models.add_custom_dir(canonical.clone());
                    self.rescan_models();
                    self.rescan_history();
                    let (loss, steps) = self
                        .forms
                        .get(tab)
                        .map(|f| (f.panel().metrics.last_loss, f.panel().metrics.last_step))
                        .unwrap_or((None, None));
                    let mut lines = vec![match (loss, steps) {
                        (Some(loss), Some(steps)) => {
                            format!("Final loss: {loss:.4} over {steps} steps")
                        }
                        _ => format!("{title} finished."),
                    }];
                    lines.push(String::new());
                    lines.push("Chat with the result?".to_string());
                    self.pending = Some(Pending::TestInference(canonical));
                    self.modal_stack
                        .push(Modal::confirm(format!("{title} Complete"), lines));
                }
            }
            _ => {}
        }
    }

    fn on_background_exit(&mut self, owner: &Owner, status: &ProcessStatus) {
        if matches!(self.modal_stack.last(), Some(Modal::Progress { .. })) {
            self.modal_stack.pop();
        }
        match status {
            ProcessStatus::Completed { .. } => match owner {
                Owner::Download(id) => {
                    self.notify(format!("Downloaded {id}"));
                    self.rescan_models();
                }
                Owner::Fuse => {
                    self.notify("Fused");
                    self.rescan_models();
                }
                Owner::Convert => {
                    self.notify("Converted");
                    self.rescan_datasets();
                }
                Owner::Tab(_) => {}
            },
            ProcessStatus::Failed { error, .. } => {
                self.modal_stack
                    .push(Modal::error("Job Failed", error.clone()));
            }
            ProcessStatus::Stopped | ProcessStatus::Running | ProcessStatus::Stopping => {}
        }
    }

    // ── Settling what events missed ─────────────────────────────────────

    /// A job's last event may have been lost; its state has not. Settle every
    /// tab on what the supervisor says.
    pub(crate) fn reconcile_jobs(&mut self) {
        let Some(hub) = &self.hub else { return };
        let mut ended: Vec<(Tab, JobId, ProcessStatus)> = Vec::new();
        for (tab, form) in self.forms.each() {
            if let Some(job) = &form.panel().job {
                if let Some(status) = hub.outcome(job) {
                    ended.push((tab, job.clone(), status));
                }
            }
        }
        let chat = self
            .inference
            .job
            .clone()
            .and_then(|job| hub.outcome(&job).map(|status| (job, status)));
        for (tab, job, status) in ended {
            let applied = self
                .forms
                .get_mut(tab)
                .map(|form| form.panel_mut().settle(&status));
            if let Some(Applied::Ended { success }) = applied {
                self.on_tab_job_ended(tab, &job, success);
            }
        }
        if let Some((job, status)) = chat {
            self.finish_chat(&job, &status);
        }
    }

    /// The event channel overflowed: events were lost. Rebuild what depends on
    /// a complete stream from the supervisor's own record.
    pub(crate) fn resync_jobs(&mut self) {
        let Some(hub) = &self.hub else { return };
        let supervisor = hub.supervisor().clone();
        self.forms.for_each_mut(|_, form| {
            let panel = form.panel_mut();
            let Some(job) = panel.job.clone() else { return };
            if let Ok(page) = supervisor.logs(&job, usize::MAX, None) {
                panel.log.clear();
                if let Some(command) = &panel.command {
                    panel.log.push(format!("$ {command}"), false);
                }
                for entry in page.entries {
                    panel.log.push(entry.text, entry.stream == Stream::Stderr);
                }
            }
            if let Ok(summary) = supervisor.summary(&job) {
                panel.metrics = summary.metrics;
            }
        });
        if let Some(job) = self.dashboard.following().map(str::to_string) {
            if let Ok(page) = supervisor.steps(&job, 0) {
                self.dashboard.follow(job, "Restored");
                for step in page.steps {
                    self.dashboard.push_step(step);
                }
            }
        }
        self.reconcile_jobs();
    }

    // ── Starting ────────────────────────────────────────────────────────

    /// `S` on a form tab: check the form, then ask.
    pub(crate) fn start_prompt(&mut self, tab: Tab) {
        let Some(form) = self.forms.get_mut(tab) else {
            return;
        };
        if form.panel().is_running() {
            let title = form.panel().title;
            self.modal_stack.push(Modal::error(
                "Already Running",
                format!("{title} is already running. Stop it first (x)."),
            ));
            return;
        }
        match form.plan() {
            Err(issues) => {
                if let Some(first) = issues.first() {
                    form.panel_mut().form.focus_field(&first.field);
                }
                let lines = issues
                    .iter()
                    .map(|issue| format!("{}: {}", issue.field, issue.message))
                    .collect::<Vec<_>>()
                    .join("\n");
                self.modal_stack
                    .push(Modal::error("Fix the form first", lines));
            }
            Ok(plan) if plan.missing_command().is_some() => {
                let why = plan.missing_command().unwrap_or_default();
                self.modal_stack
                    .push(Modal::error("Not in this build", why));
            }
            Ok(_) => {
                let mut lines = form.summary();
                lines.push(String::new());
                lines.push("Proceed?".to_string());
                let title = form.start_title();
                self.pending = Some(Pending::Start(tab));
                self.modal_stack.push(Modal::confirm(title, lines));
            }
        }
    }

    /// Start a form tab's job, the form having been confirmed.
    pub(crate) fn start(&mut self, tab: Tab) {
        let Some(form) = self.forms.get_mut(tab) else {
            return;
        };
        let plan = match form.plan() {
            Ok(plan) => plan,
            Err(issues) => {
                let lines = issues
                    .iter()
                    .map(|i| format!("{}: {}", i.field, i.message))
                    .collect::<Vec<_>>()
                    .join("\n");
                self.modal_stack
                    .push(Modal::error("Fix the form first", lines));
                return;
            }
        };
        if let Err(message) = self.prepare(&plan) {
            if let Some(form) = self.forms.get_mut(tab) {
                form.panel_mut().failed_to_start(message.clone());
            }
            self.modal_stack.push(Modal::error("Cannot Start", message));
            return;
        }
        let command = plan.command_line();
        let launched = match self.hub.as_mut() {
            Some(hub) => hub.launch(&plan, Owner::Tab(tab)),
            None => Err("jobs are not available".to_string()),
        };
        let Some(form) = self.forms.get_mut(tab) else {
            return;
        };
        match launched {
            Err(message) => {
                form.panel_mut().failed_to_start(message.clone());
                self.modal_stack.push(Modal::error("Cannot Start", message));
            }
            Ok(job) => {
                let title = form.panel().title;
                form.panel_mut().begin(job.clone(), command.clone());
                form.begin();
                self.jobs.started(job.clone(), title, command);
                if plan.dashboard {
                    self.dashboard.follow(job, title);
                    self.switch_tab(Tab::Dashboard);
                }
            }
        }
    }

    /// What must exist before the command runs: its output directory, and a
    /// note of the model it starts from so a later chat can find it.
    fn prepare(&self, plan: &LaunchPlan) -> Result<(), String> {
        let Some(dir) = &plan.output_dir else {
            return Ok(());
        };
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("Cannot create the output directory {}: {e}", dir.display()))?;
        if let Some(base) = &plan.base_model {
            write_training_info(dir, base, &self.models.resolve_path(base));
        }
        Ok(())
    }

    // ── Stopping ────────────────────────────────────────────────────────

    pub(crate) fn stop_job(&mut self, id: &str) {
        if let Some(hub) = &self.hub {
            hub.stop(id);
        }
    }

    pub(crate) fn stop_tab_job(&mut self, tab: Tab) {
        let Some(form) = self.forms.get_mut(tab) else {
            return;
        };
        if let Some(job) = form.panel().job.clone() {
            form.panel_mut().note_stopping();
            self.stop_job(&job);
        }
    }

    // ── Chat ────────────────────────────────────────────────────────────

    fn on_chat_event(&mut self, job: &str, kind: &EventKind) {
        if self.inference.job.as_deref() != Some(job) {
            return;
        }
        match kind {
            EventKind::Delta(text) => self.inference.receive(text),
            EventKind::Exited(status) => self.finish_chat(job, status),
            _ => {}
        }
    }

    /// The chat job ended: settle the reply from everything it printed.
    pub(crate) fn finish_chat(&mut self, job: &str, status: &ProcessStatus) {
        if self.inference.job.as_deref() != Some(job) {
            return;
        }
        let stdout = self.stdout_of(job);
        let stats = self.inference.finish_generation(&stdout);
        match status {
            ProcessStatus::Failed { error, .. } => {
                self.inference.discard_empty_reply();
                self.modal_stack
                    .push(Modal::error("Inference Error", error.clone()));
            }
            ProcessStatus::Completed { .. } if stats.is_none() => {
                self.inference.discard_empty_reply();
                self.modal_stack.push(Modal::error(
                    "Inference Error",
                    "The model finished without printing a reply.",
                ));
            }
            _ => {}
        }
    }

    /// Everything a job printed on standard output, as the supervisor kept it.
    fn stdout_of(&self, job: &str) -> String {
        let Some(hub) = &self.hub else {
            return String::new();
        };
        let Ok(page) = hub.supervisor().logs(job, usize::MAX, None) else {
            return String::new();
        };
        let mut text = String::new();
        for entry in page.entries.iter().filter(|e| e.stream == Stream::Stdout) {
            text.push_str(&entry.text);
            text.push('\n');
        }
        text
    }

    pub(crate) fn stop_generation(&mut self) {
        if let Some(job) = self.inference.job.clone() {
            self.stop_job(&job);
        }
    }

    /// Send the message in the input box.
    pub(crate) fn submit_inference(&mut self) {
        if self.inference.generating {
            return;
        }
        let prompt = self.inference.take_input();
        if prompt.trim().is_empty() {
            return;
        }
        let Some(model_id) = self.inference.model_id.clone() else {
            self.modal_stack
                .push(Modal::error("No Model", "Select a model first (Ctrl+P)."));
            self.inference.set_input(&prompt);
            return;
        };

        // A LoRA adapter runs on its base model with the adapter beside it.
        let entry: Option<ModelEntry> = self.models.find(&model_id).cloned();
        let (model, lora) = match &entry {
            Some(entry) if tabs::is_lora_adapter(&entry.path) => match &entry.base_model {
                Some(base) => (
                    self.models.resolve_path(base),
                    Some(entry.path.display().to_string()),
                ),
                None => (entry.path.display().to_string(), None),
            },
            Some(entry) => (entry.path.display().to_string(), None),
            None => (model_id.clone(), None),
        };

        let spec = self.inference.build_infer_spec(model, prompt.clone(), lora);
        let plan = LaunchPlan {
            subcommand: "infer".to_string(),
            argv: pmetal_core::JobFields::to_argv(&spec),
            deltas: true,
            ..LaunchPlan::default()
        };
        let command = plan.command_line();
        let launched = match self.hub.as_mut() {
            Some(hub) => hub.launch(&plan, Owner::Tab(Tab::Inference)),
            None => Err("jobs are not available".to_string()),
        };
        match launched {
            Ok(job) => {
                self.inference.add_user_message(&prompt);
                self.inference.start_generation(job.clone());
                self.jobs.started(job, "Chat", command);
            }
            Err(message) => {
                self.inference.set_input(&prompt);
                self.modal_stack.push(Modal::error("Cannot Start", message));
            }
        }
    }

    pub(crate) fn chat_with_selected_model(&mut self) {
        let Some(model) = self.models.selected_model().cloned() else {
            return;
        };
        self.inference.load_model_settings(&model.path);
        if model.is_adapter() && model.base_model.is_none() {
            // An adapter needs the model it was trained from.
            self.pending = Some(Pending::InferenceBase(model.id.clone()));
            let entries = self.model_picker_entries(|e| !e.id.starts_with("trained/"));
            self.modal_stack.push(Modal::model_picker(entries));
        } else {
            self.inference.model_id = Some(model.id.clone());
        }
        self.switch_tab(Tab::Inference);
    }

    // ── Learning rate of a running job ──────────────────────────────────

    /// `L`: ask for a new learning rate for the tab's running job.
    pub(crate) fn open_learning_rate_modal(&mut self, tab: Tab) {
        let running = self
            .forms
            .get(tab)
            .and_then(|f| f.panel().job.clone())
            .and_then(|job| {
                self.hub
                    .as_ref()
                    .and_then(|hub| hub.supervisor().running_output(&job).ok())
            });
        match running {
            Some((_, dir)) if dir.exists() => {
                self.pending = Some(Pending::SetLearningRate(dir));
                self.modal_stack.push(Modal::text_input(
                    "Set Learning Rate",
                    "Enter new LR (e.g. 1e-5, 0.0001):",
                ));
            }
            _ => self.modal_stack.push(Modal::error(
                "No Active Job",
                "Start a training job first. The learning rate can only be adjusted while a job is running.",
            )),
        }
    }

    pub(crate) fn write_learning_rate(&mut self, dir: &Path, lr: f64) {
        let command = pmetal_core::LrControlCommand::SetLr { value: lr };
        if let Err(e) = pmetal_core::write_lr_control(dir, &command) {
            let message = if e.kind() == std::io::ErrorKind::AlreadyExists {
                "The job has not read the previous command yet; try again in a moment.".to_string()
            } else {
                format!("Could not write the control file: {e}")
            };
            self.modal_stack
                .push(Modal::error("Learning Rate Not Changed", message));
        } else {
            self.notify(format!("Learning rate set to {lr:e}"));
        }
    }

    // ── Models and datasets: jobs the Models and Datasets tabs start ────

    pub(crate) fn download_model(&mut self, model_id: &str) {
        let plan = LaunchPlan {
            subcommand: "download".to_string(),
            argv: vec![model_id.to_string()],
            ..LaunchPlan::default()
        };
        self.launch_background(
            plan,
            Owner::Download(model_id.to_string()),
            ("Download", "Downloading"),
            format!("Downloading {model_id}..."),
        );
    }

    pub(crate) fn convert_dataset(&mut self, output: &str) {
        let Some(ds) = self.datasets.selected_dataset() else {
            return;
        };
        let input = ds.path.display().to_string();
        let output = expand_tilde(output);
        if let Some(parent) = Path::new(&output).parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                self.modal_stack.push(Modal::error(
                    "Cannot Convert",
                    format!("Cannot create {}: {e}", parent.display()),
                ));
                return;
            }
        }
        let plan = LaunchPlan {
            subcommand: "dataset".to_string(),
            argv: vec![
                "convert".to_string(),
                "--input".to_string(),
                input.clone(),
                "--output".to_string(),
                output.clone(),
            ],
            ..LaunchPlan::default()
        };
        self.launch_background(
            plan,
            Owner::Convert,
            ("Convert", "Converting"),
            format!("Converting {input} -> {output}..."),
        );
    }

    pub(crate) fn fuse_selected_model(&mut self) {
        let Some(model) = self.models.selected_model().cloned() else {
            return;
        };
        if !model.is_adapter() {
            self.modal_stack.push(Modal::error(
                "Not a LoRA Adapter",
                "Fuse is only available for LoRA adapters.",
            ));
            return;
        }
        match &model.base_model {
            Some(base) => self.confirm_fuse(&model, base.clone()),
            None => {
                self.pending = Some(Pending::FuseBase(model.id.clone()));
                let entries = self.model_picker_entries(|e| !e.id.starts_with("trained/"));
                self.modal_stack.push(Modal::model_picker(entries));
            }
        }
    }

    pub(crate) fn convert_selected_dataset(&mut self) {
        let Some(ds) = self.datasets.selected_dataset() else {
            self.modal_stack
                .push(Modal::error("No Dataset", "Select a dataset first."));
            return;
        };
        let input = ds.path.display().to_string();
        let stem = ds
            .path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let mut modal = Modal::text_input(
            "Convert Dataset",
            format!("Input: {input}\nOutput path (JSONL):"),
        );
        modal.paste(&format!("./data/{stem}_converted.jsonl"));
        self.pending = Some(Pending::ConvertDataset);
        self.modal_stack.push(modal);
    }

    /// Run a command the interface starts for itself, with a dialog that stays
    /// until it ends.
    fn launch_background(
        &mut self,
        plan: LaunchPlan,
        owner: Owner,
        (label, title): (&str, &str),
        message: String,
    ) {
        let command = plan.command_line();
        let launched = match self.hub.as_mut() {
            Some(hub) => hub.launch(&plan, owner),
            None => Err("jobs are not available".to_string()),
        };
        match launched {
            Ok(job) => {
                self.jobs.started(job, label, command);
                self.modal_stack.push(Modal::progress(title, message));
            }
            Err(message) => self.modal_stack.push(Modal::error("Cannot Start", message)),
        }
    }

    pub(crate) fn start_fuse(&mut self) {
        let Some(model) = self.models.selected_model().cloned() else {
            return;
        };
        let Some(base) = model.base_model.clone() else {
            self.modal_stack.push(Modal::error(
                "No Base Model",
                "Select a base model first (the model this adapter was trained from).",
            ));
            return;
        };
        let name = model
            .path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let spec = pmetal_core::jobs::FuseSpec {
            model: self.models.resolve_path(&base),
            lora: model.path.display().to_string(),
            output: format!("./output/{name}--fused"),
            ..pmetal_core::jobs::FuseSpec::default()
        };
        let plan = LaunchPlan {
            subcommand: "fuse".to_string(),
            argv: pmetal_core::JobFields::to_argv(&spec),
            ..LaunchPlan::default()
        };
        self.launch_background(
            plan,
            Owner::Fuse,
            ("Fuse", "Fusing Model"),
            format!("Fusing {} into {base}...", model.id),
        );
    }
}

/// Connect to `address` until something answers or a minute has gone, and say
/// so. The first try waits a moment: a server that cannot bind exits at once,
/// and must not be mistaken for one that answers.
fn probe_address(tx: tokio::sync::mpsc::Sender<AppMsg>, job: JobId, address: String) {
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(300)).await;
        for _ in 0..240 {
            let attempt = tokio::time::timeout(
                Duration::from_secs(1),
                tokio::net::TcpStream::connect(address.as_str()),
            )
            .await;
            if matches!(attempt, Ok(Ok(_))) {
                let _ = tx
                    .send(AppMsg::Reachable {
                        job,
                        reachable: true,
                    })
                    .await;
                return;
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        let _ = tx
            .send(AppMsg::Reachable {
                job,
                reachable: false,
            })
            .await;
    });
}

/// Expand a leading `~` to the user's home directory.
pub(crate) fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest).display().to_string();
        }
    } else if path == "~" {
        if let Some(home) = dirs::home_dir() {
            return home.display().to_string();
        }
    }
    path.to_string()
}
