//! What a job reports about its progress, from the three places it says it:
//! the `--log-events` file (`JobEvent` lines), the `--log-metrics` file (the
//! flat records the trainer's JSONL logger writes) and its own standard output.

use pmetal_core::events::{JobEvent, MetricPayload};
use pmetal_core::{EvalMetrics, StepMetrics};
use serde::Serialize;

/// Training metrics extracted from a job's output.
#[derive(Debug, Clone, Default, Serialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct JobMetrics {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_step: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_steps: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_lr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grad_norm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f64>,
    /// Current epoch, counted from 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epoch: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_epochs: Option<u64>,
}

impl JobMetrics {
    /// Fold one event into the summary. Events that carry no step or
    /// evaluation (phases, logs, lifecycle) leave it as it was.
    pub fn apply(&mut self, event: &JobEvent) {
        let JobEvent::Metric { payload, .. } = event else {
            return;
        };
        match payload {
            MetricPayload::Step(step) => {
                self.last_step = Some(step.step as u64);
                if step.total_steps > 0 {
                    self.total_steps = Some(step.total_steps as u64);
                }
                self.last_loss = Some(step.loss);
                self.last_lr = Some(step.lr);
                self.grad_norm = step.grad_norm.or(self.grad_norm);
                if step.tok_sec > 0.0 {
                    self.tokens_per_second = Some(step.tok_sec);
                }
                // The trainer counts epochs from 0.
                self.epoch = Some((step.epoch + 1) as f64);
                if step.total_epochs > 0 {
                    self.total_epochs = Some(step.total_epochs as u64);
                }
            }
            MetricPayload::Eval(eval) => self.eval_loss = Some(eval.loss),
            _ => {}
        }
    }
}

/// The event one line of a job's JSONL output stands for, if it stands for one.
///
/// A line is either a `JobEvent` (what `--log-events` writes) or a flat record
/// (what `--log-metrics` writes): a step carries `step` and `loss`, and an
/// epoch's evaluation is `{"event":"epoch_end","loss":…}`. Lifecycle records
/// (`train_start`, `checkpoint_saved`, …) and lines that are not JSON stand for
/// nothing. The event's job id is `job_id`, whatever the line said.
pub fn parse_record(line: &str, job_id: &str) -> Option<JobEvent> {
    let line = line.trim();
    if !line.starts_with('{') {
        return None;
    }
    if let Ok(mut event) = pmetal_core::events::parse_event(line) {
        set_job_id(&mut event, job_id);
        return Some(event);
    }
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    let record = value.as_object()?;
    if let Some(kind) = record.get("event").and_then(|e| e.as_str()) {
        // `epoch_end` carries the evaluation; the other records carry no step.
        if kind != "epoch_end" {
            return None;
        }
        let eval: EvalMetrics = serde_json::from_value(value.clone()).ok()?;
        record.get("loss")?.as_f64()?;
        return Some(JobEvent::Metric {
            job_id: job_id.to_string(),
            payload: MetricPayload::Eval(eval),
        });
    }
    record.get("step")?.as_u64()?;
    let step: StepMetrics = serde_json::from_value(value.clone()).ok()?;
    Some(JobEvent::Metric {
        job_id: job_id.to_string(),
        payload: MetricPayload::Step(step),
    })
}

fn set_job_id(event: &mut JobEvent, id: &str) {
    match event {
        JobEvent::Started { job_id, .. }
        | JobEvent::Phase { job_id, .. }
        | JobEvent::Metric { job_id, .. }
        | JobEvent::Token { job_id, .. }
        | JobEvent::Log { job_id, .. }
        | JobEvent::Progress { job_id, .. }
        | JobEvent::Completed { job_id, .. }
        | JobEvent::Failed { job_id, .. }
        | JobEvent::Cancelled { job_id, .. } => *job_id = id.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics_of(lines: &[&str]) -> JobMetrics {
        let mut m = JobMetrics::default();
        for line in lines {
            if let Some(event) = parse_record(line, "j") {
                m.apply(&event);
            }
        }
        m
    }

    #[test]
    fn a_flat_step_record_updates_the_metrics() {
        let m = metrics_of(&[
            r#"{"step":7,"epoch":0,"total_epochs":3,"total_steps":90,"loss":2.5,"lr":0.0002,"tok_sec":1234.5,"grad_norm":0.8,"timestamp":"t"}"#,
        ]);
        assert_eq!(m.last_step, Some(7));
        assert_eq!(m.total_steps, Some(90));
        assert_eq!(m.last_loss, Some(2.5));
        assert_eq!(m.last_lr, Some(0.0002));
        assert_eq!(m.tokens_per_second, Some(1234.5));
        assert_eq!(m.grad_norm, Some(0.8));
        // Counted from 1, as the JobEvent path counts it.
        assert_eq!(m.epoch, Some(1.0));
        assert_eq!(m.total_epochs, Some(3));
    }

    #[test]
    fn an_epoch_record_is_the_eval_loss_not_the_training_loss() {
        let m = metrics_of(&[
            r#"{"step":3,"loss":2.0}"#,
            r#"{"event":"epoch_end","epoch":0,"loss":1.5,"perplexity":4.4}"#,
            // Lifecycle records are not steps.
            r#"{"event":"train_start","timestamp":"t"}"#,
        ]);
        assert_eq!(m.last_loss, Some(2.0));
        assert_eq!(m.eval_loss, Some(1.5));
        assert_eq!(m.last_step, Some(3));
    }

    #[test]
    fn text_that_is_not_a_record_is_ignored() {
        let m = metrics_of(&["Epoch 1/3 loss=2.0", "{not json", "[1,2,3]", ""]);
        assert!(m.last_step.is_none() && m.last_loss.is_none());
    }

    #[test]
    fn a_job_event_line_takes_the_supervisors_id() {
        let line =
            r#"{"event":"metric","job_id":"inner","payload":{"kind":"step","step":4,"loss":1.25}}"#;
        let Some(JobEvent::Metric { job_id, payload }) = parse_record(line, "outer") else {
            panic!("a metric event");
        };
        assert_eq!(job_id, "outer");
        let MetricPayload::Step(step) = payload else {
            panic!("a step");
        };
        assert_eq!((step.step, step.loss), (4, 1.25));
    }

    #[test]
    fn a_non_finite_loss_the_trainer_wrote_as_text_still_parses() {
        // The logger serialises NaN as null or a string; the record is kept
        // (as a zero) rather than lost, so the step count still advances.
        let m = metrics_of(&[r#"{"step":9,"loss":null,"lr":0.1}"#]);
        assert_eq!(m.last_step, Some(9));
    }
}
