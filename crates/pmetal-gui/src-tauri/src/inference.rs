//! Chat with a model inside the app.
//!
//! A conversation needs the whole history to reach the chat template, which
//! `pmetal infer` does not take, so this is the one thing the app runs itself
//! rather than as a supervised job. It runs on a blocking thread (a
//! generation holds the thread for as long as it takes), reports into a sink,
//! and can be stopped between tokens.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use pmetal::core::SecretString;
use pmetal::core::jobs::InferSpec;
use pmetal::data::chat_templates::{ChatTemplateKwargs, Message};
use serde::{Deserialize, Serialize};

use crate::error::{AppError, Result};

/// One turn of the conversation so far.
#[derive(Debug, Deserialize, Clone)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

fn message_of(turn: &ChatTurn) -> std::result::Result<Message, String> {
    match turn.role.as_str() {
        "user" => Ok(Message::user(turn.content.clone())),
        "assistant" => Ok(Message::assistant(turn.content.clone())),
        "system" => Ok(Message::system(turn.content.clone())),
        other => Err(format!(
            "The conversation has a message from '{other}', which a chat model does not take. \
             Only user, assistant and system messages can be sent."
        )),
    }
}

/// What a finished generation measured.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Metrics {
    pub session_id: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_ms: f64,
    pub ttft_ms: Option<f64>,
    pub tok_per_sec: Option<f64>,
    pub avg_step_ms: Option<f64>,
    pub p50_step_ms: Option<f64>,
    pub response_text: String,
    pub thinking: Option<String>,
    pub truncated_thinking: bool,
    /// Ended by Stop, not by the model.
    pub stopped: bool,
}

/// Where a generation reports.
pub trait Sink: Send + Sync + 'static {
    fn token(&self, session_id: &str, text: &str);
    fn done(&self, metrics: &Metrics);
    fn failed(&self, session_id: &str, message: &str);
}

fn locked<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The generations in progress.
#[derive(Default)]
pub struct Sessions {
    flags: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

impl Sessions {
    fn begin(&self, id: &str) -> Result<Arc<AtomicBool>> {
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
        let mut flags = locked(&self.flags);
        if flags.contains_key(id) {
            return Err(AppError::new("That chat session is already generating."));
        }
        let flag = Arc::new(AtomicBool::new(false));
        flags.insert(id.to_string(), flag.clone());
        Ok(flag)
    }

    fn end(&self, id: &str) {
        locked(&self.flags).remove(id);
    }

    /// Ask one generation, or with `None` every generation, to stop.
    pub fn cancel(&self, id: Option<&str>) {
        for (key, flag) in locked(&self.flags).iter() {
            if id.is_none_or(|id| id == key) {
                flag.store(true, Ordering::SeqCst);
            }
        }
    }

    #[cfg(test)]
    pub fn active(&self) -> usize {
        locked(&self.flags).len()
    }
}

/// Start generating a reply for session `session_id`, which the caller chose so
/// it can recognise the events before this returns. Tokens, the end and any
/// failure arrive through `sink`, tagged with it.
pub async fn start(
    sessions: Arc<Sessions>,
    spec: InferSpec,
    history: Option<Vec<ChatTurn>>,
    token: Option<SecretString>,
    sink: Arc<dyn Sink>,
    session_id: String,
) -> Result<()> {
    // Everything that can be wrong with the request is said before a thread is
    // taken and a model is loaded.
    let messages = history
        .as_deref()
        .map(|turns| {
            turns
                .iter()
                .map(message_of)
                .collect::<std::result::Result<Vec<_>, _>>()
        })
        .transpose()
        .map_err(AppError::new)?;
    if spec.model.trim().is_empty() {
        return Err(AppError::new("Choose a model to chat with."));
    }
    let kv_turboquant_preset = match spec.kv_turboquant_preset.as_deref() {
        None => None,
        Some("q2_5") => Some(pmetal::inference_runner::TurboQuantPreset::Q2_5),
        Some("q3_5") => Some(pmetal::inference_runner::TurboQuantPreset::Q3_5),
        Some(other) => {
            return Err(AppError::new(format!(
                "Unknown TurboQuant preset '{other}'; the choices are q2_5 and q3_5."
            )));
        }
    };
    let chat_template_kwargs = ChatTemplateKwargs::from_controls(
        spec.chat_template_kwargs.as_deref(),
        spec.no_thinking,
        spec.reasoning_effort.as_deref(),
        spec.no_preserve_thinking,
    )
    .map_err(AppError::new)?;
    let mode = spec.mode.parse().map_err(AppError::new)?;

    let model_path = pmetal::hub::resolve_model_path(&spec.model, None, token.as_ref())
        .await
        .map_err(|e| {
            AppError::new(format!(
                "Could not get the model '{}': {e}. Check the name, the internet connection, \
                 and for a gated model that an access token is saved in Settings.",
                spec.model
            ))
        })?;

    let config = pmetal::inference_runner::InferenceRunnerConfig {
        model_path,
        lora_path: spec.lora.clone(),
        experts_dir: spec.experts_dir.clone(),
        fp8: spec.fp8,
        prompt: spec.prompt.clone(),
        chat_messages: messages,
        system_message: spec.system.clone(),
        no_thinking: spec.no_thinking,
        chat_template_kwargs,
        temperature: spec.temperature,
        top_k: spec.top_k,
        top_p: spec.top_p,
        min_p: spec.min_p,
        max_tokens: spec.max_tokens,
        repetition_penalty: spec.repetition_penalty,
        frequency_penalty: spec.frequency_penalty,
        presence_penalty: spec.presence_penalty,
        seed: spec.seed,
        kv_quant: spec.kv_quant,
        kv_k_bits: spec.kv_k_bits,
        kv_v_bits: spec.kv_v_bits,
        kv_group_size: spec.kv_group_size,
        kv_turboquant: spec.kv_turboquant,
        kv_turboquant_preset,
        kv_quant_preset: spec.kv_quant_preset.clone(),
        no_kv_quant: spec.no_kv_quant,
        kv_qjl: spec.kv_qjl,
        mode,
        detect_repetition: spec.detect_repetition,
        ..Default::default()
    };

    let cancel = sessions.begin(&session_id)?;
    let session = session_id;
    tokio::task::spawn_blocking(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            generate(&session, config, &cancel, &*sink)
        }));
        match outcome {
            Ok(Ok(metrics)) => sink.done(&metrics),
            Ok(Err(message)) => sink.failed(&session, &message),
            Err(_) => sink.failed(
                &session,
                "The model crashed while generating. Try again, or restart PMetal if it keeps happening.",
            ),
        }
        sessions.end(&session);
    });
    Ok(())
}

fn generate(
    session: &str,
    config: pmetal::inference_runner::InferenceRunnerConfig,
    cancel: &AtomicBool,
    sink: &dyn Sink,
) -> std::result::Result<Metrics, String> {
    use pmetal::inference_runner::InferenceRunner;

    let prompt_tokens = config.prompt.len();
    let mut runner = InferenceRunner::prepare(config).map_err(|e| e.to_string())?;

    let mut token_buf: Vec<u32> = Vec::new();
    let mut streamed = String::new();
    let start = std::time::Instant::now();
    let mut first_token: Option<std::time::Instant> = None;
    let mut generated = 0usize;
    let mut stopped = false;

    let tokenizer = &runner.tokenizer;
    runner
        .state
        .generate_streaming(|token_id| {
            if cancel.load(Ordering::SeqCst) {
                stopped = true;
                return false;
            }
            generated += 1;
            first_token.get_or_insert_with(std::time::Instant::now);
            token_buf.push(token_id);
            // A token can end part-way through a character; send only what has
            // become whole text since the last send.
            if let Ok(text) = tokenizer.decode(&token_buf)
                && text.len() > streamed.len()
            {
                let from = text.ceil_char_boundary(streamed.len());
                if from < text.len() {
                    sink.token(session, &text[from..]);
                }
                streamed = text;
            }
            true
        })
        .map_err(|e| e.to_string())?;

    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = first_token.map(|t| t.duration_since(start).as_secs_f64() * 1000.0);
    // The runtime's own decode metrics skip warm-up and stop-token overhead;
    // without them, time the tokens after the first.
    let (tok_per_sec, avg_step_ms, p50_step_ms) = match runner.state.last_decode_metrics {
        Some(m) => (
            Some(m.tok_per_sec),
            Some(m.avg_step_ms),
            Some(m.p50_step_ms),
        ),
        None => {
            let decode_ms = ttft_ms.map(|ttft| total_ms - ttft);
            let tps = decode_ms
                .filter(|ms| *ms > 0.0 && generated > 1)
                .map(|ms| (generated - 1) as f64 / (ms / 1000.0));
            (tps, None, None)
        }
    };
    let parsed = pmetal::response_parser::parse_assistant_response(&streamed);
    Ok(Metrics {
        session_id: session.to_string(),
        prompt_tokens,
        generated_tokens: generated,
        total_ms,
        ttft_ms,
        tok_per_sec,
        avg_step_ms,
        p50_step_ms,
        response_text: parsed.response,
        thinking: parsed.thinking,
        truncated_thinking: parsed.truncated_thinking,
        stopped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_assistant_and_system_messages_are_carried_over() {
        for role in ["user", "assistant", "system"] {
            let m = message_of(&ChatTurn {
                role: role.into(),
                content: "hi".into(),
            })
            .unwrap();
            assert_eq!(m.role, role);
            assert_eq!(m.content, "hi");
        }
    }

    #[test]
    fn a_message_from_nobody_the_model_knows_is_refused_by_name() {
        let err = message_of(&ChatTurn {
            role: "tool".into(),
            content: "{}".into(),
        })
        .unwrap_err();
        assert!(err.contains("'tool'"), "{err}");
    }

    #[test]
    fn a_stop_reaches_one_session_or_all_of_them() {
        let sessions = Sessions::default();
        let flag_a = sessions.begin("a").unwrap();
        let flag_b = sessions.begin("b").unwrap();
        sessions.cancel(Some("a"));
        assert!(flag_a.load(Ordering::SeqCst));
        assert!(!flag_b.load(Ordering::SeqCst));
        sessions.cancel(None);
        assert!(flag_b.load(Ordering::SeqCst));
        sessions.end("a");
        assert_eq!(sessions.active(), 1);
    }

    #[test]
    fn a_session_id_must_be_plain_and_unused() {
        let sessions = Sessions::default();
        sessions.begin("chat-1_a").unwrap();
        assert!(
            sessions.begin("chat-1_a").is_err(),
            "a session cannot start twice"
        );
        for bad in ["", "a b", "x/../y", &"z".repeat(65)] {
            assert!(sessions.begin(bad).is_err(), "{bad:?}");
        }
    }

    struct Nobody;
    impl Sink for Nobody {
        fn token(&self, _: &str, _: &str) {}
        fn done(&self, _: &Metrics) {}
        fn failed(&self, _: &str, _: &str) {}
    }

    #[tokio::test]
    async fn a_request_that_cannot_work_is_refused_before_a_model_is_loaded() {
        let sessions = Arc::new(Sessions::default());
        let sink: Arc<dyn Sink> = Arc::new(Nobody);

        let no_model = start(
            sessions.clone(),
            InferSpec::default(),
            None,
            None,
            sink.clone(),
            "s1".into(),
        )
        .await
        .unwrap_err();
        assert!(no_model.message().contains("Choose a model"));

        let spec = InferSpec {
            model: "/some/model".into(),
            ..InferSpec::default()
        };
        let bad_role = start(
            sessions.clone(),
            spec.clone(),
            Some(vec![ChatTurn {
                role: "tool".into(),
                content: "x".into(),
            }]),
            None,
            sink.clone(),
            "s2".into(),
        )
        .await
        .unwrap_err();
        assert!(bad_role.message().contains("tool"));

        let bad_preset = start(
            sessions.clone(),
            InferSpec {
                kv_turboquant_preset: Some("q9".into()),
                ..spec
            },
            None,
            None,
            sink,
            "s3".into(),
        )
        .await
        .unwrap_err();
        assert!(bad_preset.message().contains("q9"));
        assert_eq!(sessions.active(), 0);
    }
}
