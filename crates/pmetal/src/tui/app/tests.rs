//! State-machine tests of the app: keys in, jobs run (a shell script stands in
//! for `pmetal`), screens out.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use tokio::sync::mpsc;

use super::*;
use crate::tui::jobs::LaunchPlan;
use crate::tui::modal::ModalAction;
use crate::tui::tabs::{FormTab, Pick, RunState};

// ── Harness ─────────────────────────────────────────────────────────────

fn fake_pmetal(dir: &Path, body: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = dir.join("fake-pmetal");
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

fn environment(binary: PathBuf) -> Environment {
    Environment {
        device: None,
        device_spec: None,
        installed_memory: 16 << 30,
        binary,
        metrics_file: None,
        start_tab: Tab::Device,
    }
}

/// An app whose `pmetal` is the script `body`, attached to a channel the test
/// drains with [`pump`].
fn app_running(dir: &Path, body: &str) -> (App, mpsc::Receiver<AppMsg>) {
    let mut app = App::with_environment(environment(fake_pmetal(dir, body)));
    let (tx, rx) = mpsc::channel(4096);
    app.attach(tx);
    (app, rx)
}

fn key(app: &mut App, code: KeyCode) {
    app.handle_event(Event::Key(KeyEvent::new(code, KeyModifiers::NONE)));
}

fn chord(app: &mut App, code: KeyCode, modifiers: KeyModifiers) {
    app.handle_event(Event::Key(KeyEvent::new(code, modifiers)));
}

fn ctrl(app: &mut App, c: char) {
    chord(app, KeyCode::Char(c), KeyModifiers::CONTROL);
}

fn typed(app: &mut App, text: &str) {
    for c in text.chars() {
        key(app, KeyCode::Char(c));
    }
}

/// Handle background messages until `done` holds.
async fn pump(app: &mut App, rx: &mut mpsc::Receiver<AppMsg>, mut done: impl FnMut(&App) -> bool) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    while !done(app) {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let msg = tokio::time::timeout(remaining.max(Duration::from_millis(1)), rx.recv())
            .await
            .expect("timed out waiting for the app to reach the expected state")
            .expect("the message channel closed");
        app.handle_event(Event::App(msg));
    }
}

fn screen(app: &mut App, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|frame| app.draw(frame)).unwrap();
    let buffer = terminal.backend().buffer().clone();
    (0..height)
        .map(|y| {
            (0..width)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn top_modal_text(app: &mut App) -> String {
    screen(app, 140, 40)
}

/// Fill the Training form so it can start, writing its output under `out`.
fn ready_training(app: &mut App, out: &Path) {
    let training = &mut app.forms.training;
    training.on_pick(
        "Model",
        Pick::Model {
            id: "org/Tiny",
            architecture: Some("llama"),
        },
    );
    training.on_pick("Dataset", Pick::Dataset("data.jsonl"));
    training
        .panel_mut()
        .form
        .set_value("Output Dir", out.display().to_string());
}

/// A training run that reports a phase, two steps and exits cleanly.
const TRAINING_SCRIPT: &str = r#"events=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--log-events" ]; then events="$2"; fi
  shift
done
echo "loading the model"
printf '%s\n' '{"event":"phase","job_id":"x","phase":{"phase":"loading_model"}}' >> "$events"
sleep 0.2
printf '%s\n' '{"event":"metric","job_id":"x","payload":{"kind":"step","step":1,"total_steps":2,"total_epochs":1,"loss":2.5,"lr":0.0001,"tok_sec":900.0,"total_ms":120.0,"gpu_fwd_bwd_ms":100.0}}' >> "$events"
printf '%s\n' '{"event":"metric","job_id":"x","payload":{"kind":"step","step":2,"total_steps":2,"total_epochs":1,"loss":1.5,"lr":0.0001,"tok_sec":950.0,"total_ms":110.0,"gpu_fwd_bwd_ms":90.0}}' >> "$events"
sleep 0.2
echo "done"
"#;

// ── Keys ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn q_quits_when_nothing_is_running() {
    let mut app = App::with_environment(environment("pmetal".into()));
    key(&mut app, KeyCode::Char('q'));
    assert!(app.should_quit);
}

#[tokio::test]
async fn typing_q_or_a_question_mark_into_a_field_types_it() {
    // The global shortcuts used to run before the field editor, so a `q` in a
    // path closed the whole interface and a `?` opened help.
    let mut app = App::with_environment(environment("pmetal".into()));
    app.switch_tab(Tab::Training);
    app.forms
        .training
        .panel_mut()
        .form
        .focus_field("Output Dir");
    key(&mut app, KeyCode::Enter);
    assert!(app.forms.training.panel().form.is_editing());
    typed(&mut app, "q?p");
    assert!(
        !app.should_quit,
        "q quit while a field was being typed into"
    );
    assert!(app.modal_stack.is_empty(), "? opened help over the field");
    key(&mut app, KeyCode::Enter);
    assert!(
        app.forms
            .training
            .panel()
            .form
            .value("Output Dir")
            .ends_with("q?p")
    );
}

#[tokio::test]
async fn the_help_overlay_opens_and_closes_outside_a_field() {
    let mut app = App::with_environment(environment("pmetal".into()));
    app.switch_tab(Tab::Serve);
    key(&mut app, KeyCode::Char('?'));
    assert_eq!(app.modal_stack.len(), 1);
    let text = top_modal_text(&mut app);
    assert!(
        text.contains("Help") && text.contains("Review and start the job"),
        "{text}"
    );
    key(&mut app, KeyCode::Esc);
    assert!(app.modal_stack.is_empty());
}

#[tokio::test]
async fn tab_keys_move_between_tabs_and_leave_an_open_edit_behind() {
    let mut app = App::with_environment(environment("pmetal".into()));
    app.switch_tab(Tab::Training);
    app.forms.training.panel_mut().form.focus_field("Epochs");
    key(&mut app, KeyCode::Enter);
    typed(&mut app, "9");
    assert!(app.forms.training.panel().form.is_editing());
    key(&mut app, KeyCode::Tab);
    assert_ne!(app.active_tab, Tab::Training);
    assert!(
        !app.forms.training.panel().form.is_editing(),
        "an edit stayed open on a tab nobody is looking at"
    );
    assert_ne!(app.forms.training.panel().form.value("Epochs"), "19");
    chord(&mut app, KeyCode::Char('3'), KeyModifiers::ALT);
    assert_eq!(app.active_tab, Tab::ALL[2]);
    chord(&mut app, KeyCode::BackTab, KeyModifiers::SHIFT);
    assert_eq!(app.active_tab, Tab::ALL[1]);
}

#[tokio::test]
async fn the_tab_palette_jumps_to_any_tab() {
    let mut app = App::with_environment(environment("pmetal".into()));
    ctrl(&mut app, 'p');
    typed(&mut app, "quant");
    key(&mut app, KeyCode::Enter);
    assert_eq!(app.active_tab, Tab::Quantize);
}

#[tokio::test]
async fn a_click_on_the_tab_bar_switches_tabs() {
    let mut app = App::with_environment(environment("pmetal".into()));
    screen(&mut app, 200, 40); // draws once: the header's width is learned
    let slots = crate::tui::widgets::tab_slots(
        200 - crate::tui::widgets::LOGO_WIDTH,
        Tab::ALL,
        Tab::Device,
    );
    let (tab, from, _) = *slots.iter().find(|(t, _, _)| *t == Tab::Serve).unwrap();
    app.handle_event(Event::Mouse(MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: crate::tui::widgets::LOGO_WIDTH + from + 1,
        row: 0,
        modifiers: KeyModifiers::NONE,
    }));
    assert_eq!(app.active_tab, tab);
}

// ── Ctrl+C ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn ctrl_c_ends_an_edit_before_it_does_anything_else() {
    let mut app = App::with_environment(environment("pmetal".into()));
    app.switch_tab(Tab::Training);
    key(&mut app, KeyCode::Enter); // the first field is a picker: open and close it
    app.modal_stack.clear();
    app.pending = None;
    app.forms.training.panel_mut().form.focus_field("Epochs");
    key(&mut app, KeyCode::Enter);
    assert!(app.forms.training.panel().form.is_editing());
    ctrl(&mut app, 'c');
    assert!(!app.forms.training.panel().form.is_editing());
    assert!(!app.should_quit);
}

#[tokio::test]
async fn ctrl_c_stops_the_tabs_job_instead_of_quitting() {
    let dir = tempfile::tempdir().unwrap();
    let (mut app, mut rx) = app_running(dir.path(), "sleep 300");
    ready_training(&mut app, &dir.path().join("out"));
    app.switch_tab(Tab::Training);
    app.start(Tab::Training);
    assert!(app.forms.training.panel().is_running());
    ctrl(&mut app, 'c');
    assert!(
        !app.should_quit,
        "Ctrl+C quit with a job running on the tab"
    );
    pump(&mut app, &mut rx, |a| {
        matches!(a.forms.training.panel().run, RunState::Stopped)
    })
    .await;
    // With nothing left on the tab, the next Ctrl+C quits.
    ctrl(&mut app, 'c');
    assert!(app.should_quit);
}

#[tokio::test]
async fn quitting_with_jobs_running_asks_and_defaults_to_no() {
    let dir = tempfile::tempdir().unwrap();
    let (mut app, _rx) = app_running(dir.path(), "sleep 300");
    ready_training(&mut app, &dir.path().join("out"));
    app.start(Tab::Training);
    app.switch_tab(Tab::Device);
    key(&mut app, KeyCode::Char('q'));
    assert!(!app.should_quit);
    assert_eq!(app.modal_stack.len(), 1);
    assert!(top_modal_text(&mut app).contains("1 job is running"));
    // Enter takes the default: stay.
    key(&mut app, KeyCode::Enter);
    assert!(!app.should_quit && app.modal_stack.is_empty());
    key(&mut app, KeyCode::Char('q'));
    key(&mut app, KeyCode::Char('y'));
    assert!(app.should_quit);
    // The jobs are stopped on the way out.
    app.hub.as_ref().unwrap().supervisor().shutdown().await;
    assert_eq!(app.running_jobs(), 0);
}

#[tokio::test]
async fn a_termination_signal_quits_without_asking() {
    let dir = tempfile::tempdir().unwrap();
    let (mut app, _rx) = app_running(dir.path(), "sleep 300");
    ready_training(&mut app, &dir.path().join("out"));
    app.start(Tab::Training);
    app.handle_event(Event::Terminate);
    assert!(app.should_quit && app.modal_stack.is_empty());
    app.hub.as_ref().unwrap().supervisor().shutdown().await;
}

// ── Starting and following a job ────────────────────────────────────────

#[tokio::test]
async fn a_training_run_goes_from_the_form_to_a_finished_result() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("run");
    let (mut app, mut rx) = app_running(dir.path(), TRAINING_SCRIPT);
    ready_training(&mut app, &out);
    app.switch_tab(Tab::Training);

    key(&mut app, KeyCode::Char('S'));
    let prompt = top_modal_text(&mut app);
    assert!(
        prompt.contains("Start Training?") && prompt.contains("org/Tiny"),
        "{prompt}"
    );
    key(&mut app, KeyCode::Char('y'));

    // The run is followed on the monitor.
    assert!(app.forms.training.panel().is_running());
    assert_eq!(app.active_tab, Tab::Dashboard);
    assert!(app.dashboard.following().is_some());
    assert_eq!(app.jobs.live.len(), 1);

    pump(&mut app, &mut rx, |a| {
        a.forms.training.panel().run != RunState::Idle && !a.forms.training.panel().is_running()
    })
    .await;
    let panel = app.forms.training.panel();
    assert!(
        matches!(panel.run, RunState::Completed { .. }),
        "{:?}",
        panel.run
    );
    assert_eq!(panel.metrics.last_step, Some(2));
    assert_eq!(panel.metrics.last_loss, Some(1.5));
    assert_eq!(panel.metrics.total_steps, Some(2));
    // Both the run's own words and its output reached the log.
    let log = panel.log.tail_text(20).join("\n");
    assert!(
        log.contains("$ pmetal train") && log.contains("loading the model"),
        "{log}"
    );
    // The monitor charted both steps and let go of the run.
    assert_eq!(app.dashboard.step_count(), 2);
    assert_eq!(app.dashboard.following(), None);
    // The event log was made for the run to write to.
    assert!(out.join("events.jsonl").exists());
    // The run offers to be tried.
    let text = top_modal_text(&mut app);
    assert!(
        text.contains("Training Complete") && text.contains("Final loss: 1.5000 over 2 steps"),
        "{text}"
    );
    key(&mut app, KeyCode::Char('y'));
    assert_eq!(app.active_tab, Tab::Inference);
    assert!(
        app.inference
            .model_id
            .as_deref()
            .unwrap()
            .starts_with("trained/")
    );
}

#[tokio::test]
async fn a_failing_run_shows_why_on_its_tab() {
    let dir = tempfile::tempdir().unwrap();
    let (mut app, mut rx) = app_running(
        dir.path(),
        "echo starting; echo 'out of memory' >&2; exit 3",
    );
    ready_training(&mut app, &dir.path().join("out"));
    app.start(Tab::Training);
    pump(&mut app, &mut rx, |a| {
        matches!(a.forms.training.panel().run, RunState::Failed(_))
    })
    .await;
    assert_eq!(
        app.forms.training.panel().run,
        RunState::Failed("out of memory".into())
    );
    assert_eq!(app.dashboard.following(), None);
    // A failure offers nothing to try.
    assert!(app.modal_stack.is_empty());
    app.switch_tab(Tab::Training);
    assert!(screen(&mut app, 140, 40).contains("out of memory"));
}

#[tokio::test]
async fn a_form_with_a_problem_says_which_field_and_does_not_run() {
    let dir = tempfile::tempdir().unwrap();
    let (mut app, _rx) = app_running(dir.path(), "echo should not run");
    app.switch_tab(Tab::Training);
    key(&mut app, KeyCode::Char('S'));
    let text = top_modal_text(&mut app);
    assert!(text.contains("Fix the form first"), "{text}");
    assert!(
        text.contains("Model: required") && text.contains("Dataset: required"),
        "{text}"
    );
    assert_eq!(app.running_jobs(), 0);
    // The cursor is on the first thing to fix.
    key(&mut app, KeyCode::Esc);
    assert_eq!(
        app.forms.training.panel().form.current().unwrap().label,
        "Model"
    );
}

#[tokio::test]
async fn a_second_start_while_running_is_refused_not_queued() {
    let dir = tempfile::tempdir().unwrap();
    let (mut app, _rx) = app_running(dir.path(), "sleep 300");
    ready_training(&mut app, &dir.path().join("out"));
    app.start(Tab::Training);
    app.switch_tab(Tab::Training);
    key(&mut app, KeyCode::Char('S'));
    assert!(top_modal_text(&mut app).contains("already running"));
    assert_eq!(app.running_jobs(), 1);
    app.hub.as_ref().unwrap().supervisor().shutdown().await;
}

#[tokio::test]
async fn x_stops_the_job_and_nothing_survives_it() {
    let dir = tempfile::tempdir().unwrap();
    let pidfile = dir.path().join("helper.pid");
    let (mut app, mut rx) = app_running(
        dir.path(),
        &format!("sleep 300 & echo $! > {}; wait", pidfile.display()),
    );
    ready_training(&mut app, &dir.path().join("out"));
    app.start(Tab::Training);
    app.switch_tab(Tab::Training);
    let helper = loop {
        if let Ok(text) = std::fs::read_to_string(&pidfile) {
            if !text.trim().is_empty() {
                break text.trim().to_string();
            }
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    };
    key(&mut app, KeyCode::Char('x'));
    pump(&mut app, &mut rx, |a| {
        matches!(a.forms.training.panel().run, RunState::Stopped)
    })
    .await;
    // The helper the job spawned went with it.
    let mut gone = false;
    for _ in 0..80 {
        let alive = std::process::Command::new("kill")
            .args(["-0", &helper])
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        if !alive {
            gone = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(gone, "the job's helper outlived the stop");
    assert!(screen(&mut app, 140, 40).contains("Stopped"));
}

#[tokio::test]
async fn a_job_whose_last_event_was_lost_is_settled_on_the_next_tick() {
    let dir = tempfile::tempdir().unwrap();
    let (mut app, mut rx) = app_running(dir.path(), "echo hi");
    ready_training(&mut app, &dir.path().join("out"));
    app.start(Tab::Training);
    // The process ends, but the interface never hears of it.
    let supervisor = app.hub.as_ref().unwrap().supervisor().clone();
    let job = app.forms.training.panel().job.clone().unwrap();
    supervisor
        .wait(&job, Duration::from_secs(10))
        .await
        .unwrap();
    while rx.try_recv().is_ok() {}
    assert!(
        app.forms.training.panel().is_running(),
        "it was told nothing"
    );
    app.handle_event(Event::Tick);
    assert!(matches!(
        app.forms.training.panel().run,
        RunState::Completed { .. }
    ));
    assert_eq!(app.dashboard.following(), None);
}

#[tokio::test]
async fn a_flood_of_output_is_bounded_and_the_interface_keeps_up() {
    let dir = tempfile::tempdir().unwrap();
    let (mut app, mut rx) = app_running(
        dir.path(),
        "i=0; while [ $i -lt 30000 ]; do echo \"line $i\"; i=$((i+1)); done",
    );
    ready_training(&mut app, &dir.path().join("out"));
    app.start(Tab::Training);
    pump(&mut app, &mut rx, |a| {
        matches!(a.forms.training.panel().run, RunState::Completed { .. })
    })
    .await;
    let panel = app.forms.training.panel();
    assert!(panel.log.len() <= crate::tui::widgets::DEFAULT_LOG_CAP);
    assert_eq!(panel.log.tail_text(1), ["line 29999"]);
    // The Jobs tab keeps the same bound.
    assert!(app.jobs.live[0].log.len() <= crate::tui::widgets::DEFAULT_LOG_CAP);
}

#[tokio::test]
async fn output_with_terminal_escapes_cannot_corrupt_the_screen() {
    let dir = tempfile::tempdir().unwrap();
    let (mut app, mut rx) = app_running(
        dir.path(),
        r#"printf '\033[2J\033[31mred\033[0m \033]0;pwned\007text\n'; printf 'bad \377 byte\n'"#,
    );
    ready_training(&mut app, &dir.path().join("out"));
    app.start(Tab::Training);
    pump(&mut app, &mut rx, |a| {
        matches!(a.forms.training.panel().run, RunState::Completed { .. })
    })
    .await;
    let lines = app.forms.training.panel().log.tail_text(5);
    assert!(lines.contains(&"red text"), "{lines:?}");
    assert!(lines.iter().all(|l| !l.contains('\u{1b}')), "{lines:?}");
}

// ── Pickers and routing ─────────────────────────────────────────────────

#[tokio::test]
async fn a_dataset_picked_for_the_eval_field_does_not_replace_the_training_one() {
    let mut app = App::with_environment(environment("pmetal".into()));
    app.switch_tab(Tab::Training);
    app.forms
        .training
        .panel_mut()
        .form
        .set_value("Dataset", "train.jsonl");
    app.forms
        .training
        .panel_mut()
        .form
        .focus_field("Eval Dataset");
    key(&mut app, KeyCode::Enter);
    assert_eq!(
        app.pending,
        Some(Pending::PickDataset {
            tab: Tab::Training,
            field: "Eval Dataset".into()
        })
    );
    app.handle_modal_action(ModalAction::SelectDataset("eval.jsonl".into()));
    assert_eq!(
        app.forms.training.panel().form.value("Dataset"),
        "train.jsonl"
    );
    assert_eq!(
        app.forms.training.panel().form.value("Eval Dataset"),
        "eval.jsonl"
    );
}

#[tokio::test]
async fn the_teacher_and_the_student_are_picked_independently() {
    let mut app = App::with_environment(environment("pmetal".into()));
    app.switch_tab(Tab::Distillation);
    app.forms
        .distillation
        .panel_mut()
        .form
        .focus_field("Teacher");
    key(&mut app, KeyCode::Enter);
    app.modal_stack.pop();
    app.handle_modal_action(ModalAction::SelectModel("big/Teacher".into()));
    app.forms
        .distillation
        .panel_mut()
        .form
        .focus_field("Student");
    key(&mut app, KeyCode::Enter);
    app.handle_modal_action(ModalAction::SelectModel("small/Student".into()));
    let form = &app.forms.distillation.panel().form;
    assert_eq!(form.value("Teacher"), "big/Teacher");
    assert_eq!(form.value("Student"), "small/Student");
}

#[tokio::test]
async fn a_dialog_answer_with_nothing_pending_is_ignored() {
    let mut app = App::with_environment(environment("pmetal".into()));
    app.handle_modal_action(ModalAction::SelectModel("x".into()));
    app.handle_modal_action(ModalAction::Confirmed);
    app.handle_modal_action(ModalAction::TextSubmitted("x".into()));
    assert!(app.modal_stack.is_empty() && !app.should_quit);
}

// ── Scans ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_slow_scan_that_finishes_after_a_newer_one_is_ignored() {
    let mut app = App::with_environment(environment("pmetal".into()));
    let (tx, _rx) = mpsc::channel(8);
    app.attach(tx);
    app.rescan_models();
    let first = app.scan_seq.models;
    app.rescan_models();
    let second = app.scan_seq.models;
    assert!(second > first);
    let entry = |id: &str| crate::tui::tabs::ModelEntry {
        id: id.into(),
        path: PathBuf::from(id),
        size_bytes: 0,
        architecture: None,
        params: None,
        quantization: None,
        modified: String::new(),
        source: crate::tui::tabs::ModelSource::HfCache,
        base_model: None,
        fit_estimate: None,
    };
    app.handle_app_msg(AppMsg::ModelsScanned {
        seq: second,
        models: vec![entry("new")],
    });
    app.handle_app_msg(AppMsg::ModelsScanned {
        seq: first,
        models: vec![entry("stale")],
    });
    assert_eq!(app.models.models.len(), 1);
    assert_eq!(app.models.models[0].id, "new");
}

// ── Chat ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_chat_message_streams_a_reply_and_reports_its_speed() {
    let dir = tempfile::tempdir().unwrap();
    let (mut app, mut rx) = app_running(
        dir.path(),
        "echo '== banner =='; echo 'Generating...'; printf 'The capital'; sleep 0.2; printf ' is Paris.\\n'; echo; echo '---'; echo 'Generated 5 tokens in 0.40s'; echo 'Decode: 210 tok/s (x)'",
    );
    app.switch_tab(Tab::Inference);
    app.inference.model_id = Some("org/Chat".into());
    typed(&mut app, "capital of France?");
    key(&mut app, KeyCode::Enter);
    assert!(app.inference.generating);
    assert_eq!(app.inference.messages.len(), 2);
    pump(&mut app, &mut rx, |a| !a.inference.generating).await;
    let reply = app.inference.messages.last().unwrap();
    assert_eq!(reply.content, "The capital is Paris.");
    assert_eq!((reply.tokens, reply.tok_sec), (Some(5), Some(210.0)));
    assert!(app.modal_stack.is_empty());
    assert_eq!(app.running_jobs(), 0);
}

#[tokio::test]
async fn a_chat_that_fails_says_why() {
    let dir = tempfile::tempdir().unwrap();
    let (mut app, mut rx) = app_running(dir.path(), "echo 'model not found' >&2; exit 2");
    app.switch_tab(Tab::Inference);
    app.inference.model_id = Some("org/Missing".into());
    typed(&mut app, "hello");
    key(&mut app, KeyCode::Enter);
    pump(&mut app, &mut rx, |a| !a.inference.generating).await;
    assert_eq!(
        app.inference.messages.len(),
        1,
        "no empty reply was left behind"
    );
    assert!(top_modal_text(&mut app).contains("model not found"));
}

#[tokio::test]
async fn esc_stops_a_generation_and_keeps_what_was_said() {
    let dir = tempfile::tempdir().unwrap();
    let (mut app, mut rx) = app_running(
        dir.path(),
        "echo 'Generating...'; printf 'Partial'; sleep 300",
    );
    app.switch_tab(Tab::Inference);
    app.inference.model_id = Some("m".into());
    typed(&mut app, "go");
    key(&mut app, KeyCode::Enter);
    pump(&mut app, &mut rx, |a| {
        a.inference
            .messages
            .last()
            .is_some_and(|m| m.content == "Partial")
    })
    .await;
    key(&mut app, KeyCode::Esc);
    pump(&mut app, &mut rx, |a| !a.inference.generating).await;
    assert_eq!(app.inference.messages.last().unwrap().content, "Partial");
    assert!(app.modal_stack.is_empty());
}

#[tokio::test]
async fn sending_without_a_model_keeps_what_was_typed() {
    let mut app = App::with_environment(environment("pmetal".into()));
    app.switch_tab(Tab::Inference);
    typed(&mut app, "hello there");
    key(&mut app, KeyCode::Enter);
    assert!(top_modal_text(&mut app).contains("Select a model first"));
    assert_eq!(app.inference.input(), "hello there");
}

// ── Serve ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn a_server_is_serving_once_something_answers_on_its_address() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let dir = tempfile::tempdir().unwrap();
    let (mut app, mut rx) = app_running(
        dir.path(),
        &format!(
            "echo 'INFO pmetal_serve: OpenAI-compatible API available at http://127.0.0.1:{port}/v1'; sleep 300"
        ),
    );
    app.switch_tab(Tab::Serve);
    app.forms.serve.on_pick(
        "Model",
        Pick::Model {
            id: "/models/m",
            architecture: None,
        },
    );
    app.forms
        .serve
        .panel_mut()
        .form
        .set_value("Port", port.to_string());
    app.start(Tab::Serve);
    assert!(!screen(&mut app, 140, 40).contains("✓ Serving"));
    pump(&mut app, &mut rx, |a| a.forms.serve.is_serving()).await;
    assert!(screen(&mut app, 140, 40).contains("✓ Serving"));
    drop(listener);
    app.hub.as_ref().unwrap().supervisor().shutdown().await;
}

// ── Drawing ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn every_tab_draws_at_every_size_without_panicking() {
    let mut app = App::with_environment(environment("pmetal".into()));
    for &tab in Tab::ALL {
        app.switch_tab(tab);
        for (w, h) in [
            (0, 0),
            (1, 1),
            (59, 30),
            (60, 13),
            (60, 14),
            (80, 24),
            (120, 40),
            (300, 80),
        ] {
            screen(&mut app, w, h);
        }
    }
}

#[tokio::test]
async fn a_terminal_that_is_too_small_is_told_so() {
    let mut app = App::with_environment(environment("pmetal".into()));
    let text = screen(&mut app, 40, 10);
    assert!(
        text.contains("Terminal too small: 40x10 (need 60x14)"),
        "{text}"
    );
    // Making it bigger brings the interface back.
    assert!(screen(&mut app, 100, 30).contains("PMetal"));
}

#[tokio::test]
async fn the_active_tab_stays_visible_on_a_narrow_terminal() {
    let mut app = App::with_environment(environment("pmetal".into()));
    app.switch_tab(Tab::Eval);
    let text = screen(&mut app, 70, 20);
    assert!(text.lines().next().unwrap().contains("Eval"), "{text}");
}

#[tokio::test]
async fn each_form_tab_shows_its_form_and_what_it_will_do() {
    let mut app = App::with_environment(environment("pmetal".into()));
    for &tab in Tab::ALL {
        if !tab.is_form() {
            continue;
        }
        app.switch_tab(tab);
        let text = screen(&mut app, 140, 45);
        let title = app.forms.get(tab).unwrap().panel().title;
        assert!(
            text.contains(&format!("{title} Configuration")) || text.contains("Configuration"),
            "{tab}: {text}"
        );
        assert!(
            text.contains("S start") || text.contains(" start"),
            "{tab}'s footer: {text}"
        );
    }
}

#[tokio::test]
async fn too_many_jobs_is_a_message_not_a_hang() {
    let dir = tempfile::tempdir().unwrap();
    let (mut app, _rx) = app_running(dir.path(), "sleep 300");
    // Fill all but one slot with downloads.
    let filler = LaunchPlan {
        subcommand: "download".into(),
        argv: vec!["org/model".into()],
        ..LaunchPlan::default()
    };
    for _ in 0..crate::tui::jobs::MAX_RUNNING - 1 {
        app.hub
            .as_mut()
            .unwrap()
            .launch(&filler, crate::tui::jobs::Owner::Convert)
            .unwrap();
    }
    ready_training(&mut app, &dir.path().join("one"));
    app.start(Tab::Training);
    assert!(app.forms.training.panel().is_running());

    app.forms.grpo.on_pick(
        "Model",
        Pick::Model {
            id: "m",
            architecture: None,
        },
    );
    app.forms.grpo.on_pick("Dataset", Pick::Dataset("d.jsonl"));
    app.start(Tab::Grpo);
    assert!(matches!(app.forms.grpo.panel().run, RunState::Failed(_)));
    let text = top_modal_text(&mut app);
    assert!(
        text.contains("Cannot Start") && text.contains("already running"),
        "{text}"
    );
    app.hub.as_ref().unwrap().supervisor().shutdown().await;
}

// ── The key tables against the handlers ─────────────────────────────────

/// A key as the key tables write it, as the terminal sends it.
fn events_for(binding_keys: &str, on: Tab) -> Vec<KeyEvent> {
    binding_keys
        .split(" / ")
        .map(|part| {
            let (modifiers, name) = if let Some(rest) = part.strip_prefix("Ctrl+") {
                (KeyModifiers::CONTROL, rest)
            } else if let Some(rest) = part.strip_prefix("Alt+") {
                (KeyModifiers::ALT, rest)
            } else if part == "Shift+Tab" {
                (KeyModifiers::SHIFT, "BackTab")
            } else {
                (KeyModifiers::NONE, part)
            };
            let code = match name {
                "Tab" => KeyCode::Tab,
                "BackTab" => KeyCode::BackTab,
                "Enter" => KeyCode::Enter,
                "Esc" => KeyCode::Esc,
                "PgUp" => KeyCode::PageUp,
                "PgDn" => KeyCode::PageDown,
                "End" => KeyCode::End,
                "Left" => KeyCode::Left,
                "Right" => KeyCode::Right,
                "F2" => KeyCode::F(2),
                // `Alt+1..9`: one that is not this tab.
                "1..9" => KeyCode::Char(if on == Tab::ALL[0] { '2' } else { '1' }),
                single if single.chars().count() == 1 => {
                    let c = single.chars().next().unwrap();
                    // `Ctrl+P` is the key `p` with Control; a bare `S` is a capital.
                    KeyCode::Char(if modifiers.is_empty() {
                        c
                    } else {
                        c.to_ascii_lowercase()
                    })
                }
                other => panic!("the key table has a key this test cannot press: {other}"),
            };
            KeyEvent::new(code, modifiers)
        })
        .collect()
}

/// Everything a key can change that a person would see.
fn fingerprint(app: &App) -> String {
    let form = app
        .forms
        .get(app.active_tab)
        .map(|f| {
            let panel = f.panel();
            format!(
                "{:?}/{}/{}/{}",
                panel
                    .form
                    .current()
                    .map(|c| (&c.label, &c.value, c.editing)),
                panel.log.is_following(),
                panel.is_running(),
                panel
                    .form
                    .fields
                    .iter()
                    .map(|f| f.value.len())
                    .sum::<usize>(),
            )
        })
        .unwrap_or_default();
    format!(
        "{:?}|{}|{:?}|{}|{:?}|{}|{}|{:?}|{}|{:?}|{}|{}|{:?}|{:?}|{}|{}|{:?}|{:?}|{}|{}|{:?}|{form}",
        app.active_tab,
        app.modal_stack.len(),
        app.notice.as_ref().map(|(t, _)| t),
        app.should_quit,
        app.models.table_state.selected(),
        app.models.searching,
        app.models.loading,
        app.datasets.table_state.selected(),
        app.datasets.loading,
        app.jobs.table_state.selected(),
        app.jobs.scanning,
        app.dashboard.paused,
        app.inference.focus,
        app.inference.selected_message,
        app.inference.settings_selected,
        app.inference.sidebar_visible,
        app.inference.spec,
        app.jobs.live.first().map(|j| j.log.is_following()),
        app.dashboard.step_count(),
        app.inference.scroll_position(),
        app.scan_seq,
    )
}

/// An app in which every documented key has something to act on.
fn armed(dir: &Path) -> (App, mpsc::Receiver<AppMsg>) {
    let (mut app, rx) = app_running(dir, "sleep 300");
    let model = |id: &str| crate::tui::tabs::ModelEntry {
        id: id.into(),
        path: dir.join(id),
        size_bytes: 1 << 20,
        architecture: Some("llama".into()),
        params: None,
        quantization: None,
        modified: "now".into(),
        source: crate::tui::tabs::ModelSource::HfCache,
        base_model: None,
        fit_estimate: None,
    };
    app.models.set_models(vec![model("org/a"), model("org/b")]);
    let dataset = |name: &str| crate::tui::tabs::DatasetEntry {
        name: name.into(),
        path: dir.join(name),
        format: "JSONL".into(),
        size_bytes: 10,
        sample_count: Some(1),
        sample_count_estimated: false,
        preview: None,
        source: crate::tui::tabs::DatasetSource::Local,
        columns: vec![],
        readiness: crate::tui::tabs::DatasetReadiness::Unreadable,
    };
    app.datasets
        .set_datasets(vec![dataset("a.jsonl"), dataset("b.jsonl")]);
    app.jobs.started("j1".into(), "Eval", "pmetal eval".into());
    app.jobs.started("j2".into(), "Eval", "pmetal eval".into());
    for n in 0..100 {
        app.jobs.apply(
            "j1",
            &pmetal_supervisor::EventKind::Output(pmetal_supervisor::LogLine {
                seq: n,
                stream: pmetal_supervisor::Stream::Stdout,
                text: format!("line {n}"),
            }),
        );
    }
    app.jobs.next_row();
    app.dashboard.push_step(pmetal_core::StepMetrics {
        step: 1,
        loss: 1.0,
        ..Default::default()
    });
    app.inference.model_id = Some("org/a".into());
    app.inference.add_user_message("hi");
    (app, rx)
}

/// Keys whose effect is checked by a test of its own, not by a change of
/// anything the fingerprint sees: `(tab, key as the table writes it, where)`.
const COVERED_ELSEWHERE: &[(Tab, &str, &str)] = &[
    (
        Tab::Inference,
        "Esc",
        "esc_stops_a_generation_and_keeps_what_was_said",
    ),
    (
        Tab::Inference,
        "Ctrl+Q",
        "q_quits_when_nothing_is_running (the same request)",
    ),
    (
        Tab::Inference,
        "Enter",
        "sending_without_a_model_keeps_what_was_typed",
    ),
];

#[tokio::test]
async fn every_key_the_tables_document_does_something() {
    let dir = tempfile::tempdir().unwrap();
    let mut checked = 0;
    for &tab in Tab::ALL {
        let (mut app, _rx) = armed(dir.path());
        app.switch_tab(tab);
        let bindings = crate::tui::keys::tab_bindings(tab)
            .iter()
            .chain(crate::tui::keys::GLOBAL.iter())
            // Quitting and stopping are checked where they have something to stop.
            .filter(|b| !matches!(b.keys, "Ctrl+C" | "q" | "Esc"))
            .filter(|b| !(tab == Tab::Inference && b.keys == "?"))
            .collect::<Vec<_>>();
        for binding in bindings {
            if COVERED_ELSEWHERE
                .iter()
                .any(|(t, k, _)| *t == tab && *k == binding.keys)
            {
                continue;
            }
            // Cycling works on an option or a toggle; put the cursor on one,
            // and skip a form that has none (Tokenize has only text fields).
            if binding.keys == "Left / Right" {
                let Some(form) = app.forms.get_mut(tab) else {
                    continue;
                };
                let Some(label) = form
                    .panel()
                    .form
                    .fields
                    .iter()
                    .find(|f| f.is_cycleable())
                    .map(|f| f.label.clone())
                else {
                    assert_eq!(tab, Tab::Tokenize, "{tab} has no option to cycle");
                    continue;
                };
                form.panel_mut().form.focus_field(&label);
            }
            // Following the log again needs a log that was scrolled back.
            if binding.keys == "End" {
                app.handle_event(Event::Key(KeyEvent::new(
                    KeyCode::PageUp,
                    KeyModifiers::NONE,
                )));
            }
            for event in events_for(binding.keys, tab) {
                let before = fingerprint(&app);
                let from = app.active_tab;
                app.modal_stack.clear();
                app.pending = None;
                app.handle_event(Event::Key(event));
                let after = fingerprint(&app);
                assert_ne!(
                    before, after,
                    "{tab}: the table says {:?} ({}), and pressing it changed nothing",
                    binding.keys, binding.help
                );
                checked += 1;
                // Back where it was, with nothing open.
                app.modal_stack.clear();
                app.pending = None;
                app.switch_tab(from);
                app.should_quit = false;
                app.models.searching = false;
                if let Some(form) = app.forms.get_mut(tab) {
                    form.panel_mut().form.cancel_edit();
                }
            }
        }
    }
    assert!(checked > 80, "only {checked} keys were pressed");
}

#[tokio::test]
async fn the_footer_only_offers_keys_that_are_in_the_tables() {
    for &tab in Tab::ALL {
        for (key, label) in crate::tui::keys::footer_items(tab) {
            let in_tables = crate::tui::keys::tab_bindings(tab)
                .iter()
                .chain(crate::tui::keys::GLOBAL.iter())
                .any(|b| b.footer == Some(label));
            assert!(in_tables, "{tab}: footer offers {key} {label}");
        }
    }
}
