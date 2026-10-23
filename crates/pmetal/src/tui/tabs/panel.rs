//! What every tab that runs a job from a form has in common.
//!
//! A [`JobPanel`] is the form, the job's run state, its log and its progress;
//! a tab owns one and adds only what is its own (a field that follows another,
//! a table parsed from the output, a different status box). The app drives all
//! of them through the [`FormTab`] trait: one key handler, one start flow, one
//! place where job events land.

use std::time::{Duration, Instant};

use pmetal_core::events::JobEvent;
use pmetal_core::jobs::Spec;
use pmetal_supervisor::{EventKind, JobId, JobMetrics, LaunchOptions, ProcessStatus};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

use crate::tui::jobs::LaunchPlan;
use crate::tui::tabs::dashboard::Telemetry;
use crate::tui::widgets::{FormIssue, FormTabState, LogView};

/// How a tab's job is going.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum RunState {
    #[default]
    Idle,
    Running {
        since: Instant,
        /// What the job says it is doing ("Loading model…").
        phase: Option<String>,
    },
    Completed {
        took: Duration,
    },
    Failed(String),
    /// Stopped on request.
    Stopped,
}

impl RunState {
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running { .. })
    }
}

/// What a tab does with a job's event beyond the common handling.
#[derive(Debug, PartialEq, Eq)]
pub enum Applied {
    /// Nothing more for the tab to do.
    Nothing,
    /// The job ended; `true` if it succeeded. The app reacts (rescans the
    /// models, offers to try the result).
    Ended { success: bool },
}

/// The form, state, log and progress of one tab's job.
pub struct JobPanel {
    /// Names the job in dialogs ("Training", "Quantize").
    pub title: &'static str,
    pub form: FormTabState,
    pub run: RunState,
    pub log: LogView,
    /// What the job has reported (steps, loss, ...), for the status box.
    pub metrics: JobMetrics,
    /// The running job, if any.
    pub job: Option<JobId>,
    /// The command the job was started with, for the log's first line.
    pub command: Option<String>,
    /// Values the tab filled in from another field, by label.
    derived: Vec<(String, String)>,
}

impl JobPanel {
    /// A panel whose form is `S`'s.
    pub fn new<S: Spec>(title: &'static str) -> Self {
        Self::with_form(title, FormTabState::for_spec::<S>())
    }

    pub fn with_form(title: &'static str, form: FormTabState) -> Self {
        Self {
            title,
            form,
            run: RunState::Idle,
            log: LogView::new(crate::tui::widgets::DEFAULT_LOG_CAP),
            metrics: JobMetrics::default(),
            job: None,
            command: None,
            derived: Vec::new(),
        }
    }

    /// Fill the field labelled `label` with a value worked out from another
    /// field (an output directory named after the model), unless the person
    /// has typed their own: it is replaced only while it still holds the
    /// form's default or the last value derived here.
    pub fn derive(&mut self, label: &str, value: String) {
        let current = self.form.value(label);
        let derived = self.derived.iter().position(|(l, _)| l == label);
        let untouched = current.is_empty()
            || self.form.initial_of(label) == Some(current.as_str())
            || derived.is_some_and(|i| self.derived[i].1 == current);
        if !untouched {
            return;
        }
        self.form.set_value(label, value.clone());
        match derived {
            Some(i) => self.derived[i].1 = value,
            None => self.derived.push((label.to_string(), value)),
        }
    }

    pub fn is_running(&self) -> bool {
        self.run.is_running()
    }

    /// The panel's command for the form as it stands, for the common case: the
    /// spec `S`, run as `pmetal <S::subcommand> <S::to_argv>`, asking for the
    /// event log and metrics file the spec's command writes (`LaunchPlan`).
    pub fn plan_for<S: Spec>(&self) -> Result<LaunchPlan, Vec<FormIssue>> {
        let spec = self.form.to_spec::<S>()?;
        LaunchPlan::for_spec(&spec, &LaunchOptions::default()).map_err(|e| {
            vec![FormIssue {
                field: String::new(),
                message: e.to_string(),
            }]
        })
    }

    /// The job was started.
    pub fn begin(&mut self, job: JobId, command: String) {
        self.log.clear();
        self.log.push(format!("$ {command}"), false);
        self.metrics = JobMetrics::default();
        self.job = Some(job);
        self.command = Some(command);
        self.run = RunState::Running {
            since: Instant::now(),
            phase: None,
        };
    }

    /// A stop was asked for; the job is still going until it is gone.
    pub fn note_stopping(&mut self) {
        if let RunState::Running { phase, .. } = &mut self.run {
            *phase = Some("Stopping...".to_string());
        }
    }

    /// The job could not be started.
    pub fn failed_to_start(&mut self, message: String) {
        self.job = None;
        self.run = RunState::Failed(message);
    }

    /// Fold a job event into the panel.
    pub fn apply(&mut self, kind: &EventKind) -> Applied {
        match kind {
            EventKind::Started | EventKind::Delta(_) => {}
            EventKind::Output(line) => {
                self.log.push(
                    line.text.clone(),
                    line.stream == pmetal_supervisor::Stream::Stderr,
                );
            }
            EventKind::Job(event) => {
                self.metrics.apply(event);
                match event {
                    JobEvent::Phase { phase, .. } => {
                        if let RunState::Running { phase: shown, .. } = &mut self.run {
                            *shown = Some(phase.message());
                        }
                    }
                    JobEvent::Metric { .. } => {
                        if let RunState::Running { phase, .. } = &mut self.run {
                            *phase = None;
                        }
                    }
                    _ => {}
                }
            }
            EventKind::Exited(status) => {
                let took = match &self.run {
                    RunState::Running { since, .. } => since.elapsed(),
                    _ => Duration::ZERO,
                };
                self.job = None;
                match status {
                    ProcessStatus::Completed { .. } => {
                        self.run = RunState::Completed { took };
                        return Applied::Ended { success: true };
                    }
                    ProcessStatus::Failed { error, .. } => {
                        self.run = RunState::Failed(error.clone());
                    }
                    ProcessStatus::Stopped | ProcessStatus::Running | ProcessStatus::Stopping => {
                        self.run = RunState::Stopped;
                    }
                }
                return Applied::Ended { success: false };
            }
        }
        Applied::Nothing
    }

    /// The job ended without the panel hearing of it (its event was lost):
    /// settle on the supervisor's account.
    pub fn settle(&mut self, status: &ProcessStatus) -> Applied {
        if self.run.is_running() && !status.is_active() {
            return self.apply(&EventKind::Exited(status.clone()));
        }
        Applied::Nothing
    }
}

/// What a picker chose.
#[derive(Debug, Clone, Copy)]
pub enum Pick<'a> {
    /// A model, with the architecture the Models tab read from its config.
    Model {
        id: &'a str,
        architecture: Option<&'a str>,
    },
    /// A dataset file.
    Dataset(&'a str),
}

impl Pick<'_> {
    /// The text that goes in the field.
    pub fn text(&self) -> &str {
        match self {
            Pick::Model { id, .. } => id,
            Pick::Dataset(path) => path,
        }
    }
}

/// A tab that runs a job from a form.
pub trait FormTab {
    fn panel(&self) -> &JobPanel;
    fn panel_mut(&mut self) -> &mut JobPanel;

    /// The command this form runs, or what is wrong with the form.
    fn plan(&self) -> Result<LaunchPlan, Vec<FormIssue>>;

    /// A picker chose `picked` for the field labelled `field`.
    fn on_pick(&mut self, field: &str, picked: Pick<'_>) {
        self.panel_mut().form.set_value(field, picked.text());
    }

    /// The tab sizes its context from a sample of the dataset it is given.
    fn wants_dataset_peek(&self) -> bool {
        false
    }

    /// A sample of the dataset the tab asked for arrived.
    fn set_peek(&mut self, _path: &str, _peek: crate::tui::tabs::DatasetPeek) {}

    /// The field labelled `label` was changed by the person.
    fn field_changed(&mut self, _label: &str) {}

    /// A new job is about to start: forget what the last one said.
    fn begin(&mut self) {}

    /// A line of the job's output arrived.
    fn on_line(&mut self, _line: &str) {}

    /// The job announced an address it will listen on (`host:port`), once.
    /// The app checks that something answers there and then calls
    /// [`on_reachable`](Self::on_reachable).
    fn take_announcement(&mut self) -> Option<String> {
        None
    }

    /// Something answers at the announced address.
    fn on_reachable(&mut self) {}

    /// Lines for the "start?" prompt: what the job will do.
    fn summary(&self) -> Vec<String> {
        self.panel().form.summary()
    }

    /// The prompt's title.
    fn start_title(&self) -> String {
        format!("Start {}?", self.panel().title)
    }

    /// Draw the tab.
    fn render(&mut self, area: Rect, buf: &mut Buffer, telemetry: &Telemetry<'_>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_core::StepMetrics;
    use pmetal_core::events::{MetricPayload, Phase};
    use pmetal_core::jobs::QuantizeSpec;
    use pmetal_supervisor::{LogLine, Stream};

    fn panel() -> JobPanel {
        JobPanel::new::<QuantizeSpec>("Quantize")
    }

    fn line(text: &str, stream: Stream) -> EventKind {
        EventKind::Output(LogLine {
            seq: 0,
            stream,
            text: text.into(),
        })
    }

    #[test]
    fn a_job_runs_from_start_to_a_finished_state() {
        let mut p = panel();
        p.begin("j1".into(), "pmetal quantize --model m".into());
        assert!(p.is_running());
        assert_eq!(p.log.tail_text(5), ["$ pmetal quantize --model m"]);

        p.apply(&line("loading", Stream::Stdout));
        p.apply(&line("warn", Stream::Stderr));
        assert_eq!(p.log.len(), 3);
        assert!(p.log.lines().last().unwrap().stderr);

        let ended = p.apply(&EventKind::Exited(ProcessStatus::Completed {
            exit_code: 0,
        }));
        assert_eq!(ended, Applied::Ended { success: true });
        assert!(matches!(p.run, RunState::Completed { .. }));
        assert!(p.job.is_none());
    }

    #[test]
    fn a_failure_carries_its_reason_and_a_stop_is_not_a_failure() {
        let mut p = panel();
        p.begin("j".into(), "c".into());
        let ended = p.apply(&EventKind::Exited(ProcessStatus::Failed {
            exit_code: 3,
            error: "model not found".into(),
        }));
        assert_eq!(ended, Applied::Ended { success: false });
        assert_eq!(p.run, RunState::Failed("model not found".into()));

        p.begin("j2".into(), "c".into());
        p.apply(&EventKind::Exited(ProcessStatus::Stopped));
        assert_eq!(p.run, RunState::Stopped);
    }

    #[test]
    fn the_phase_and_steps_a_job_reports_reach_the_panel() {
        let mut p = panel();
        p.begin("j".into(), "c".into());
        p.apply(&EventKind::Job(JobEvent::Phase {
            job_id: "j".into(),
            phase: Phase::LoadingModel,
        }));
        assert!(
            matches!(&p.run, RunState::Running { phase: Some(p), .. } if p == "Loading model…")
        );
        p.apply(&EventKind::Job(JobEvent::Metric {
            job_id: "j".into(),
            payload: MetricPayload::Step(StepMetrics {
                step: 5,
                total_steps: 50,
                loss: 1.5,
                ..StepMetrics::default()
            }),
        }));
        assert_eq!(p.metrics.last_step, Some(5));
        assert_eq!(p.metrics.total_steps, Some(50));
        // Set-up is over once a step lands.
        assert!(matches!(&p.run, RunState::Running { phase: None, .. }));
    }

    #[test]
    fn a_lost_exit_event_is_made_up_from_the_supervisors_account() {
        let mut p = panel();
        p.begin("j".into(), "c".into());
        assert_eq!(p.settle(&ProcessStatus::Running), Applied::Nothing);
        assert!(p.is_running());
        let ended = p.settle(&ProcessStatus::Completed { exit_code: 0 });
        assert_eq!(ended, Applied::Ended { success: true });
        // Settled once: the next look finds nothing to do.
        assert_eq!(
            p.settle(&ProcessStatus::Completed { exit_code: 0 }),
            Applied::Nothing
        );
    }

    #[test]
    fn a_start_that_fails_is_shown_as_a_failure() {
        let mut p = panel();
        p.failed_to_start("3 jobs are already running".into());
        assert!(matches!(p.run, RunState::Failed(_)));
        assert!(!p.is_running());
    }

    #[test]
    fn the_plan_is_the_specs_command() {
        let mut p = panel();
        p.form.set_value("Model", "Qwen/Qwen3-0.6B");
        p.form.set_value("Output Path", "./out.gguf");
        let plan = p.plan_for::<QuantizeSpec>().unwrap();
        assert_eq!(plan.subcommand, "quantize");
        assert_eq!(plan.argv.first().map(String::as_str), Some("--model"));
        // A command with no event log is not asked for one.
        assert_eq!(plan.events_file, None);
    }
}
