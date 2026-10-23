//! Chat with a model inside the app.
//!
//! A reply is `pmetal infer --messages-file`, run by the supervisor like every
//! other job the app starts: the conversation is written to a private file, the
//! command answers it, and what it prints is read as it arrives. Nothing here
//! loads a model or samples a token; the command is the one place that does,
//! so a chat answers exactly as `pmetal infer` does with the same settings.
//!
//! Chats have a supervisor of their own, apart from the jobs the person
//! starts: a reply is not a job to list, and it must not take one of their
//! slots or be stopped with them.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pmetal::core::SecretString;
use pmetal::core::jobs::{ChatMessage, InferSpec, parse_messages};
use pmetal::core::sync::locked;
use pmetal::data::infer_stream::{InferStream, split_reasoning, split_reasoning_streaming};
use pmetal_supervisor::{
    Error as SupervisorError, EventKind, JobId, LaunchOptions, LaunchPlan, Limits, ProcessStatus,
    Supervisor,
};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast::error::RecvError;

use crate::error::{AppError, Result};

/// Replies that may be generated at once. A model each, so a few.
const MAX_REPLIES: usize = 4;
/// The most of a reply that is kept: a model that never stops is not allowed to
/// grow the app without bound.
const MAX_REPLY_BYTES: usize = 16 * 1024 * 1024;

/// One turn of the conversation so far.
#[derive(Debug, Deserialize, Clone)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

/// What a finished generation measured.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Metrics {
    pub session_id: String,
    pub generated_tokens: usize,
    pub total_ms: f64,
    pub ttft_ms: Option<f64>,
    pub tok_per_sec: Option<f64>,
    pub response_text: String,
    pub thinking: Option<String>,
    pub truncated_thinking: bool,
    /// Ended by Stop, not by the model.
    pub stopped: bool,
}

/// New text of a reply: the model's reasoning and its answer, apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Piece<'a> {
    pub thinking: &'a str,
    pub text: &'a str,
}

/// Where a generation reports.
pub trait Sink: Send + Sync + 'static {
    fn token(&self, session_id: &str, piece: Piece<'_>);
    fn done(&self, metrics: &Metrics);
    fn failed(&self, session_id: &str, message: &str);
}

struct Shared {
    supervisor: Supervisor,
    /// `Err` when no `pmetal` could be found: the reason, shown on start.
    binary: std::result::Result<PathBuf, String>,
    /// The replies being generated: session to job. A session is registered
    /// before its job exists, so a second start for it is refused at once.
    live: Mutex<HashMap<String, Option<JobId>>>,
}

/// The generations in progress.
#[derive(Clone)]
pub struct Chat {
    shared: Arc<Shared>,
}

impl Chat {
    /// A chat running `binary` (or refusing every start with the reason it is
    /// missing). Must be created inside a tokio runtime.
    pub fn new(binary: std::result::Result<PathBuf, String>) -> Self {
        let supervisor = Supervisor::new(
            binary.clone().unwrap_or_else(|_| PathBuf::from("pmetal")),
            Limits {
                max_running: MAX_REPLIES,
                max_finished: 2,
                ..Limits::default()
            },
        );
        Self {
            shared: Arc::new(Shared {
                supervisor,
                binary,
                live: Mutex::new(HashMap::new()),
            }),
        }
    }

    fn begin(&self, id: &str) -> Result<()> {
        let valid = !id.is_empty()
            && id.len() <= 64
            && id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        if !valid {
            return Err(AppError::new(
                "The chat session id is not valid. Reload the window and try again.",
            ));
        }
        let mut live = locked(&self.shared.live);
        if live.contains_key(id) {
            return Err(AppError::new("That chat session is already generating."));
        }
        live.insert(id.to_string(), None);
        Ok(())
    }

    fn end(&self, id: &str) {
        locked(&self.shared.live).remove(id);
    }

    /// Ask one generation, or with `None` every generation, to stop. The reply
    /// ends after the token in flight and keeps what was written.
    pub fn cancel(&self, id: Option<&str>) {
        let jobs: Vec<JobId> = locked(&self.shared.live)
            .iter()
            .filter(|(key, _)| id.is_none_or(|id| id == key.as_str()))
            .filter_map(|(_, job)| job.clone())
            .collect();
        for job in jobs {
            // A reply that has just finished cannot be stopped; that is fine.
            let _ = self.shared.supervisor.stop(&job);
        }
    }

    /// Stop every generation and wait for the commands to leave.
    pub async fn shutdown(&self) {
        self.shared.supervisor.shutdown().await;
    }

    #[cfg(test)]
    pub fn active(&self) -> usize {
        locked(&self.shared.live).len()
    }

    /// Start generating a reply for session `session_id`, which the caller
    /// chose so it can recognise the events before this returns. Tokens, the
    /// end and any failure arrive through `sink`, tagged with it.
    ///
    /// `spec` says what to run and how to sample; its prompt and system
    /// message are replaced by the conversation, `system` first.
    pub async fn start(
        &self,
        mut spec: InferSpec,
        history: Vec<ChatTurn>,
        token: Option<SecretString>,
        sink: Arc<dyn Sink>,
        session_id: String,
    ) -> Result<()> {
        // Everything that can be wrong with the request is said before a
        // process is started and a model is loaded.
        if spec.model.trim().is_empty() {
            return Err(AppError::new("Choose a model to chat with."));
        }
        if let Err(reason) = &self.shared.binary {
            return Err(AppError::new(reason.clone()));
        }
        let mut conversation = Vec::with_capacity(history.len() + 2);
        if let Some(system) = spec.system.take().filter(|s| !s.trim().is_empty()) {
            conversation.push(ChatMessage::new("system", system));
        }
        let prompt = std::mem::take(&mut spec.prompt);
        conversation.extend(
            history
                .into_iter()
                .map(|t| ChatMessage::new(t.role, t.content)),
        );
        if !prompt.trim().is_empty() {
            conversation.push(ChatMessage::new("user", prompt));
        }
        let text = serde_json::to_string(&conversation)
            .map_err(|e| AppError::new(format!("The conversation could not be written: {e}")))?;
        // The command checks it again; saying what is wrong here puts it
        // before the model is loaded and in words about the conversation.
        parse_messages(&text)
            .map_err(|e| AppError::new(format!("The conversation cannot be sent: {e}")))?;

        // The conversation is private: a file of its own that only this person
        // can read, removed when the reply ends.
        let file = tempfile::Builder::new()
            .prefix("pmetal-chat-")
            .suffix(".json")
            .tempfile()
            .and_then(|mut file| {
                use std::io::Write;
                file.write_all(text.as_bytes())?;
                file.flush()?;
                Ok(file)
            })
            .map_err(|e| AppError::new(format!("The conversation could not be saved: {e}")))?;
        spec.messages_file = Some(file.path().display().to_string());
        spec.normalize().map_err(|errors| {
            AppError::new(
                errors
                    .iter()
                    .map(|e| format!("{}: {}", e.field, e.message))
                    .collect::<Vec<_>>()
                    .join("; "),
            )
        })?;
        let plan = LaunchPlan::for_spec(
            &spec,
            &LaunchOptions {
                deltas: true,
                ..LaunchOptions::default()
            },
        )
        .map_err(|e| AppError::new(e.to_string()))?;

        self.begin(&session_id)?;
        // Listening before starting, so no piece of the reply is missed.
        let events = self.shared.supervisor.subscribe();
        let env = token
            .map(|t| vec![("HF_TOKEN".to_string(), t.expose_secret().to_string())])
            .unwrap_or_default();
        let job = match self
            .shared
            .supervisor
            .spawn(plan.into_request_with_env(env))
        {
            Ok(job) => job,
            Err(e) => {
                self.end(&session_id);
                return Err(AppError::new(match e {
                    SupervisorError::TooManyRunning { .. } => {
                        "Replies are already being generated. Stop one, or wait for it to finish."
                            .to_string()
                    }
                    other => format!("Could not start `pmetal infer`: {other}"),
                }));
            }
        };
        locked(&self.shared.live).insert(session_id.clone(), Some(job.clone()));

        let chat = self.clone();
        tokio::spawn(async move {
            // The file lives until the command has read it and gone.
            let _file = file;
            let outcome = follow(&chat.shared.supervisor, events, &job, &session_id, &*sink).await;
            match outcome {
                Ok(metrics) => sink.done(&metrics),
                Err(message) => sink.failed(&session_id, &message),
            }
            chat.end(&session_id);
        });
        Ok(())
    }
}

/// What has been sent on of a reply, so each piece goes once.
#[derive(Default)]
struct Sent {
    thinking: usize,
    text: usize,
}

impl Sent {
    /// Send what `text` (the reply so far, labels and all) has that was not
    /// sent before.
    fn advance(&mut self, text: &str, session: &str, sink: &dyn Sink) {
        let (thinking, answer) = split_reasoning_streaming(text);
        let thinking = thinking.unwrap_or("");
        let piece = Piece {
            thinking: thinking.get(self.thinking..).unwrap_or(""),
            text: answer.get(self.text..).unwrap_or(""),
        };
        if piece == Piece::default() {
            return;
        }
        self.thinking = self.thinking.max(thinking.len());
        self.text = self.text.max(answer.len());
        sink.token(session, piece);
    }
}

/// Read the reply of `job` as it is generated, until the command ends.
async fn follow(
    supervisor: &Supervisor,
    mut events: tokio::sync::broadcast::Receiver<pmetal_supervisor::Event>,
    job: &str,
    session: &str,
    sink: &dyn Sink,
) -> std::result::Result<Metrics, String> {
    let mut stream = InferStream::new();
    let mut sent = Sent::default();
    // Everything the command printed, for when a piece was missed.
    let mut printed = String::new();
    let mut missed = false;
    let status = loop {
        match events.recv().await {
            Ok(event) if event.job == job => match event.kind {
                EventKind::Delta(piece) => {
                    if printed.len() < MAX_REPLY_BYTES {
                        printed.push_str(&piece);
                        let text = stream.push(&piece).to_string();
                        sent.advance(&text, session, sink);
                    }
                }
                EventKind::Exited(status) => break status,
                _ => {}
            },
            Ok(_) => {}
            // The webview is slower than the command: the pieces that were
            // dropped are in the command's output.
            Err(RecvError::Lagged(_)) => missed = true,
            Err(RecvError::Closed) => return Err("The chat ended unexpectedly.".to_string()),
        }
    };

    if missed {
        // Whole lines of what it printed; the text between them is what was
        // streamed, so this is as good as the stream was.
        if let Ok(page) = supervisor.logs(job, usize::MAX, None) {
            printed = page.lines.join("\n");
        }
    }
    // The whole output is read afresh: it is authoritative, where the pieces
    // that streamed in may have missed some.
    let mut whole = InferStream::new();
    whole.push(&printed);
    let stats = whole.finish();
    let complete = whole.is_complete();
    let text = whole.text().to_string();

    let stopped = match status {
        ProcessStatus::Completed { .. } => false,
        ProcessStatus::Stopped => true,
        ProcessStatus::Failed { error, .. } => {
            return Err(if error.trim().is_empty() {
                "The model stopped without an answer.".to_string()
            } else {
                error
            });
        }
        ProcessStatus::Running | ProcessStatus::Stopping => {
            return Err("The chat did not end.".to_string());
        }
    };
    let (thinking, answer) = split_reasoning(&text);
    let (thinking, answer) = (
        thinking.map(|t| t.trim().to_string()),
        answer.trim().to_string(),
    );
    Ok(Metrics {
        session_id: session.to_string(),
        generated_tokens: if complete { stats.tokens } else { 0 },
        total_ms: stats
            .first_token_seconds
            .map_or(0.0, |ttft| (ttft + stats.generating_seconds) * 1000.0),
        ttft_ms: stats.first_token_seconds.map(|s| s * 1000.0),
        tok_per_sec: (stats.tokens_per_second > 0.0).then_some(stats.tokens_per_second),
        truncated_thinking: !stopped && thinking.is_some() && answer.is_empty(),
        response_text: answer,
        thinking,
        stopped,
    })
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::time::Duration;

    use super::*;

    /// A stand-in for `pmetal`: prints what `pmetal infer` prints, so the chat
    /// can be tested without a model. `body` is a shell script that runs with
    /// `$MESSAGES` set to the conversation file it was given.
    fn fake_pmetal(dir: &std::path::Path, body: &str) -> PathBuf {
        let path = dir.join("pmetal");
        let script = format!(
            "#!/bin/sh\nwhile [ $# -gt 0 ]; do\n  [ \"$1\" = --messages-file ] && MESSAGES=\"$2\"\n  shift\ndone\n{body}\n"
        );
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[derive(Default)]
    struct Recorder {
        pieces: Mutex<Vec<(String, String)>>,
        done: Mutex<Option<Metrics>>,
        failed: Mutex<Option<String>>,
        ended: tokio::sync::Notify,
    }

    impl Sink for Arc<Recorder> {
        fn token(&self, _: &str, piece: Piece<'_>) {
            locked(&self.pieces).push((piece.thinking.to_string(), piece.text.to_string()));
        }
        fn done(&self, metrics: &Metrics) {
            *locked(&self.done) = Some(metrics.clone());
            self.ended.notify_one();
        }
        fn failed(&self, _: &str, message: &str) {
            *locked(&self.failed) = Some(message.to_string());
            self.ended.notify_one();
        }
    }

    fn spec() -> InferSpec {
        InferSpec {
            model: "org/model".into(),
            prompt: "Capital of France?".into(),
            system: Some("Be brief.".into()),
            ..InferSpec::default()
        }
    }

    async fn ask(
        chat: &Chat,
        spec: InferSpec,
        history: Vec<ChatTurn>,
    ) -> (Arc<Recorder>, Result<()>) {
        let recorder = Arc::new(Recorder::default());
        let started = chat
            .start(
                spec,
                history,
                None,
                Arc::new(recorder.clone()),
                "s1".to_string(),
            )
            .await;
        (recorder, started)
    }

    async fn ended(recorder: &Recorder) {
        tokio::time::timeout(Duration::from_secs(20), recorder.ended.notified())
            .await
            .expect("the reply did not end");
    }

    const FOOTER: &str = "printf '\\n\\n---\\nGenerated 6 tokens in 0.50s (0.20s to the first token, model load included)\\nStopped by: EOS token\\nDecode: 200 tok/s (avg=5ms p50=5ms over 5 steps)\\n'";

    #[tokio::test]
    async fn a_reply_streams_in_and_ends_with_its_statistics() {
        let dir = tempfile::tempdir().unwrap();
        // The conversation the command is given is kept, to be read back.
        let kept = dir.path().join("conversation.json");
        let binary = fake_pmetal(
            dir.path(),
            &format!(
                "cp \"$MESSAGES\" {kept}\nprintf 'banner\\nGenerating...\\nThe capital '\nsleep 0.2\nprintf 'is Paris.'\n{FOOTER}",
                kept = kept.display()
            ),
        );
        let chat = Chat::new(Ok(binary));
        let history = vec![
            ChatTurn {
                role: "user".into(),
                content: "Hi".into(),
            },
            ChatTurn {
                role: "assistant".into(),
                content: "Hello!".into(),
            },
        ];
        let (recorder, started) = ask(&chat, spec(), history).await;
        started.unwrap();
        ended(&recorder).await;

        assert_eq!(locked(&recorder.failed).clone(), None);
        let metrics = locked(&recorder.done).clone().unwrap();
        assert_eq!(metrics.session_id, "s1");
        assert_eq!(metrics.response_text, "The capital is Paris.");
        assert_eq!(metrics.thinking, None);
        assert_eq!(metrics.generated_tokens, 6);
        assert_eq!(metrics.tok_per_sec, Some(200.0));
        assert_eq!(metrics.ttft_ms, Some(200.0));
        assert!(!metrics.stopped && !metrics.truncated_thinking);
        // What streamed adds up to the reply, and the banner is not in it.
        let streamed: String = locked(&recorder.pieces)
            .iter()
            .map(|(_, t)| t.as_str())
            .collect();
        assert_eq!(streamed, "The capital is Paris.");

        // The command was given the whole conversation, system message first
        // and the model's own prompt replaced by it.
        let given: Vec<ChatMessage> =
            serde_json::from_str(&std::fs::read_to_string(&kept).unwrap()).unwrap();
        let roles: Vec<_> = given.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "user"]);
        assert_eq!(given[0].content, "Be brief.");
        assert_eq!(given[3].content, "Capital of France?");
        assert_eq!(chat.active(), 0);
        chat.shutdown().await;
    }

    #[tokio::test]
    async fn reasoning_streams_apart_from_the_answer() {
        let dir = tempfile::tempdir().unwrap();
        let binary = fake_pmetal(
            dir.path(),
            &format!(
                "printf 'Generating...\\n[thinking]\\nCount the r'\nsleep 0.1\nprintf 'ight letters\\n\\n[ans'\nsleep 0.1\nprintf 'wer]\\nThree'\n{FOOTER}"
            ),
        );
        let chat = Chat::new(Ok(binary));
        let (recorder, started) = ask(&chat, spec(), Vec::new()).await;
        started.unwrap();
        ended(&recorder).await;

        let metrics = locked(&recorder.done).clone().unwrap();
        assert_eq!(metrics.thinking.as_deref(), Some("Count the right letters"));
        assert_eq!(metrics.response_text, "Three");
        let pieces = locked(&recorder.pieces).clone();
        let thinking: String = pieces.iter().map(|(t, _)| t.as_str()).collect();
        let text: String = pieces.iter().map(|(_, t)| t.as_str()).collect();
        assert_eq!(thinking.trim_end(), "Count the right letters");
        assert_eq!(text, "Three");
        // Half of the label that ends the reasoning was never shown as reasoning.
        assert!(!thinking.contains("[ans"), "{thinking:?}");
        chat.shutdown().await;
    }

    #[tokio::test]
    async fn a_model_still_thinking_when_it_ran_out_is_said_to_be() {
        let dir = tempfile::tempdir().unwrap();
        let binary = fake_pmetal(
            dir.path(),
            &format!("printf 'Generating...\\n[thinking]\\nHmm, let me'\n{FOOTER}"),
        );
        let chat = Chat::new(Ok(binary));
        let (recorder, started) = ask(&chat, spec(), Vec::new()).await;
        started.unwrap();
        ended(&recorder).await;
        let metrics = locked(&recorder.done).clone().unwrap();
        assert!(metrics.truncated_thinking);
        assert_eq!(metrics.response_text, "");
        chat.shutdown().await;
    }

    #[tokio::test]
    async fn a_command_that_fails_says_why() {
        let dir = tempfile::tempdir().unwrap();
        let binary = fake_pmetal(
            dir.path(),
            "echo 'error: model org/model not found' >&2\nexit 1",
        );
        let chat = Chat::new(Ok(binary));
        let (recorder, started) = ask(&chat, spec(), Vec::new()).await;
        started.unwrap();
        ended(&recorder).await;
        let failure = locked(&recorder.failed).clone().unwrap();
        assert!(failure.contains("not found"), "{failure}");
        assert!(locked(&recorder.done).is_none());
        assert_eq!(chat.active(), 0);
        chat.shutdown().await;
    }

    #[tokio::test]
    async fn stop_ends_the_reply_and_keeps_what_was_written() {
        let dir = tempfile::tempdir().unwrap();
        let binary = fake_pmetal(
            dir.path(),
            "trap 'exit 130' TERM\nprintf 'Generating...\\nSo far'\nwhile :; do sleep 0.05; done",
        );
        let chat = Chat::new(Ok(binary));
        let (recorder, started) = ask(&chat, spec(), Vec::new()).await;
        started.unwrap();
        // Until the first words have arrived, a stop would end nothing worth keeping.
        for _ in 0..200 {
            if !locked(&recorder.pieces).is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        chat.cancel(Some("s1"));
        ended(&recorder).await;
        let metrics = locked(&recorder.done).clone().unwrap();
        assert!(metrics.stopped);
        assert_eq!(metrics.response_text, "So far");
        assert_eq!(chat.active(), 0);
        chat.shutdown().await;
    }

    #[tokio::test]
    async fn a_session_id_must_be_plain_and_unused() {
        let chat = Chat::new(Err("none".into()));
        chat.begin("chat-1_a").unwrap();
        assert!(
            chat.begin("chat-1_a").is_err(),
            "a session cannot start twice"
        );
        for bad in ["", "a b", "x/../y", &"z".repeat(65)] {
            assert!(chat.begin(bad).is_err(), "{bad:?}");
        }
        chat.end("chat-1_a");
        assert_eq!(chat.active(), 0);
    }

    struct Nobody;
    impl Sink for Nobody {
        fn token(&self, _: &str, _: Piece<'_>) {}
        fn done(&self, _: &Metrics) {}
        fn failed(&self, _: &str, _: &str) {}
    }

    #[tokio::test]
    async fn a_request_that_cannot_work_is_refused_before_anything_starts() {
        let dir = tempfile::tempdir().unwrap();
        let chat = Chat::new(Ok(fake_pmetal(dir.path(), "exit 0")));
        let sink: Arc<dyn Sink> = Arc::new(Nobody);
        let start = |spec: InferSpec, history: Vec<ChatTurn>, id: &str| {
            chat.start(spec, history, None, sink.clone(), id.to_string())
        };

        let no_model = start(InferSpec::default(), Vec::new(), "s1")
            .await
            .unwrap_err();
        assert!(no_model.message().contains("Choose a model"));

        let bad_role = start(
            spec(),
            vec![ChatTurn {
                role: "robot".into(),
                content: "x".into(),
            }],
            "s2",
        )
        .await
        .unwrap_err();
        assert!(
            bad_role.message().contains("robot"),
            "{}",
            bad_role.message()
        );

        // A conversation that ends on the model's own turn has nothing to answer.
        let nothing_to_answer = start(
            InferSpec {
                prompt: String::new(),
                ..spec()
            },
            vec![ChatTurn {
                role: "assistant".into(),
                content: "x".into(),
            }],
            "s3",
        )
        .await
        .unwrap_err();
        assert!(
            nothing_to_answer.message().contains("must be a user"),
            "{}",
            nothing_to_answer.message()
        );

        let missing = Chat::new(Err("PMetal's command-line tool was not found".into()));
        let no_binary = missing
            .start(spec(), Vec::new(), None, sink.clone(), "s4".into())
            .await
            .unwrap_err();
        assert!(no_binary.message().contains("not found"));
        assert_eq!(chat.active(), 0);
        assert_eq!(missing.active(), 0);
    }
}
