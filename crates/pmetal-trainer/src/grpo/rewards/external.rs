//! A reward scored by a process of the user's.
//!
//! The trainer starts the command once, when the run begins, and keeps it for
//! the whole run, so whatever it loads (a verifier's sandbox, a judge model, a
//! test suite) is paid for once. Every batch of completions is one JSON line
//! on its standard input, and the command answers with one JSON line on its
//! standard output:
//!
//! ```text
//! -> {"prompts": ["..."], "completions": ["..."], "answer": ["42"], ...}
//! <- {"rewards": [1.0]}
//! ```
//!
//! `prompts` and `completions` hold one entry per completion; every other key
//! is a column of the dataset (the prompt's own columns excepted), also one
//! entry per completion, so the command scores a completion against the row it
//! answers. The answer is `{"rewards": [...]}` with one finite number per
//! completion, or `{"error": "why"}` for a batch it cannot score. Anything on
//! standard error is the command's own log: it is passed on, and the last
//! lines of it go in the error when the command fails.
//!
//! A command that cannot score a completion fails the run; it never scores it
//! `0`, which would train the policy away from a completion that was simply
//! never judged. It fails the run when it exits, when it does not answer in
//! time, when it answers with anything but the protocol, and when its answer
//! has the wrong number of rewards or a reward that is not a finite number.
//! The command's whole process group is killed when the reward is dropped, so
//! nothing it started outlives the run. A command should exit when its
//! standard input ends, which is the polite way the trainer tells it the run is
//! over.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Map, Value};

use super::{GrpoError, GrpoResult, RewardBatch, RewardFunction};

/// Lines of the command's standard error kept for an error message.
const STDERR_TAIL_LINES: usize = 20;
/// Characters of a bad answer quoted in an error message.
const QUOTE_CHARS: usize = 200;
/// The keys the request itself uses; a dataset column of the same name is not sent.
const RESERVED_KEYS: [&str; 2] = ["prompts", "completions"];

/// How to run an external reward.
#[derive(Debug, Clone)]
pub struct ExternalRewardConfig {
    /// The program and its arguments.
    pub command: Vec<String>,
    /// How long one batch may take to score, the first included (so a command
    /// that loads a model on start has this long to do it).
    pub timeout: Duration,
    /// How long the command has to exit after its input ends, before its
    /// process group is killed.
    pub shutdown_grace: Duration,
}

impl ExternalRewardConfig {
    /// A config for `command` with the default timeout (two minutes) and
    /// shutdown grace (two seconds).
    pub fn new(command: Vec<String>) -> Self {
        Self {
            command,
            timeout: Duration::from_secs(120),
            shutdown_grace: Duration::from_secs(2),
        }
    }

    /// A config for a command line, split as a shell would split it (quotes
    /// and backslashes), without running it through a shell.
    pub fn from_command_line(line: &str) -> GrpoResult<Self> {
        let command = shlex::split(line).ok_or_else(|| {
            GrpoError::Config(format!("the reward command `{line}` has an unclosed quote"))
        })?;
        if command.is_empty() {
            return Err(GrpoError::Config("the reward command is empty".into()));
        }
        Ok(Self::new(command))
    }

    /// With `timeout` per batch.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// With `grace` to exit once its input ends.
    pub fn with_shutdown_grace(mut self, grace: Duration) -> Self {
        self.shutdown_grace = grace;
        self
    }
}

/// A reward scored by a long-lived process that reads one JSON line per batch
/// and answers with the rewards (the protocol is in this module's docs).
pub struct ExternalCommandReward {
    name: String,
    timeout: Duration,
    process: Mutex<Process>,
}

/// The running command and the threads that carry its pipes.
struct Process {
    child: Child,
    /// Lines for the command's stdin; `None` once it has been closed.
    requests: Option<mpsc::Sender<Vec<u8>>>,
    /// Lines from its stdout; disconnected when it closes the pipe.
    responses: mpsc::Receiver<String>,
    stderr: Arc<Mutex<VecDeque<String>>>,
    shutdown_grace: Duration,
    /// Why the command can no longer be used, once it has failed.
    failed: Option<String>,
}

impl ExternalCommandReward {
    /// Start the command. It runs as the leader of a process group of its own
    /// (a Ctrl-C at the terminal does not reach it; the trainer stops it), and
    /// is killed with its group when the reward is dropped.
    ///
    /// # Errors
    ///
    /// [`GrpoError::Config`] if the command is empty or cannot be started.
    pub fn spawn(config: ExternalRewardConfig) -> GrpoResult<Self> {
        let Some((program, args)) = config.command.split_first() else {
            return Err(GrpoError::Config("the reward command is empty".into()));
        };
        let name = std::path::Path::new(program)
            .file_name()
            .map_or_else(|| program.clone(), |n| n.to_string_lossy().into_owned());
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .spawn()
            .map_err(|e| {
                GrpoError::Config(format!("cannot start the reward command `{program}`: {e}"))
            })?;
        let (Some(stdin), Some(stdout), Some(stderr)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            return Err(GrpoError::Config(format!(
                "the reward command `{program}` has no standard streams"
            )));
        };

        // The pipes are carried by threads so that a command that neither
        // reads nor writes cannot block the trainer past the timeout.
        let (request_tx, request_rx) = mpsc::channel::<Vec<u8>>();
        thread::Builder::new()
            .name(format!("reward-{name}-stdin"))
            .spawn(move || {
                let mut stdin = stdin;
                while let Ok(line) = request_rx.recv() {
                    if stdin.write_all(&line).and_then(|()| stdin.flush()).is_err() {
                        break; // The command is gone; the reader reports it.
                    }
                }
            })?;
        let (response_tx, response_rx) = mpsc::channel::<String>();
        thread::Builder::new()
            .name(format!("reward-{name}-stdout"))
            .spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let Ok(line) = line else { break };
                    if response_tx.send(line).is_err() {
                        break;
                    }
                }
            })?;
        let tail = Arc::new(Mutex::new(VecDeque::new()));
        let logged = Arc::clone(&tail);
        let log_name = name.clone();
        thread::Builder::new()
            .name(format!("reward-{name}-stderr"))
            .spawn(move || {
                for line in BufReader::new(stderr).lines() {
                    let Ok(line) = line else { break };
                    eprintln!("[reward {log_name}] {line}");
                    let mut tail = logged.lock().unwrap_or_else(PoisonError::into_inner);
                    if tail.len() == STDERR_TAIL_LINES {
                        tail.pop_front();
                    }
                    tail.push_back(line);
                }
            })?;

        Ok(Self {
            name,
            timeout: config.timeout,
            process: Mutex::new(Process {
                child,
                requests: Some(request_tx),
                responses: response_rx,
                stderr: tail,
                shutdown_grace: config.shutdown_grace,
                failed: None,
            }),
        })
    }

    /// The request line for `batch`: `prompts`, `completions` and one list per
    /// dataset column.
    fn request_line(batch: &RewardBatch<'_>) -> GrpoResult<Vec<u8>> {
        let mut request = Map::new();
        request.insert("prompts".into(), batch.prompts.into());
        request.insert("completions".into(), batch.completions.into());
        let mut keys: Vec<&String> = batch
            .columns
            .iter()
            .flat_map(|row| row.keys())
            .filter(|key| !RESERVED_KEYS.contains(&key.as_str()))
            .collect();
        keys.sort_unstable();
        keys.dedup();
        for key in keys {
            let column = batch
                .columns
                .iter()
                .map(|row| row.get(key).cloned().unwrap_or(Value::Null))
                .collect();
            request.insert(key.clone(), Value::Array(column));
        }
        let mut line = serde_json::to_vec(&Value::Object(request))
            .map_err(|e| GrpoError::Reward(format!("cannot encode the reward request: {e}")))?;
        line.push(b'\n');
        Ok(line)
    }

    fn fail(&self, process: &mut Process, why: String) -> GrpoError {
        let tail = process.stderr_tail();
        let message = if tail.is_empty() {
            why
        } else {
            format!("{why}; its last output on stderr:\n{tail}")
        };
        process.kill();
        process.failed = Some(message.clone());
        GrpoError::Reward(message)
    }
}

impl Process {
    fn stderr_tail(&self) -> String {
        // A moment for the reader thread to take what the command wrote last.
        thread::sleep(Duration::from_millis(50));
        let tail = self.stderr.lock().unwrap_or_else(PoisonError::into_inner);
        tail.iter().cloned().collect::<Vec<_>>().join("\n")
    }

    /// End the command and everything it started.
    fn kill(&mut self) {
        self.requests = None;
        pmetal_supervisor::kill_group(self.child.id());
        let _ = self.child.wait();
    }

    /// How the command ended, once it has (it closed its output): its exit
    /// status, or a note that it is still running.
    fn exit_status(&mut self) -> String {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return format!("exited with {status}"),
                Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
                _ => return "closed its output".into(),
            }
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // End of input is the polite stop; the group is killed after the
        // grace, and always, so a helper the command started cannot outlive it.
        self.requests = None;
        let deadline = Instant::now() + self.shutdown_grace;
        while Instant::now() < deadline && matches!(self.child.try_wait(), Ok(None)) {
            thread::sleep(Duration::from_millis(10));
        }
        pmetal_supervisor::kill_group(self.child.id());
        let _ = self.child.wait();
    }
}

fn quote(text: &str) -> String {
    let mut quoted: String = text.chars().take(QUOTE_CHARS).collect();
    if text.chars().count() > QUOTE_CHARS {
        quoted.push_str("...");
    }
    quoted
}

/// The rewards in a command's answer line.
fn parse_answer(name: &str, line: &str, expected: usize) -> GrpoResult<Vec<f64>> {
    let wrong = |what: String| {
        GrpoError::Reward(format!(
            "the reward command `{name}` answered {what}; it must answer each request with one \
             line `{{\"rewards\": [...]}}`"
        ))
    };
    let answer: Value = serde_json::from_str(line).map_err(|e| {
        wrong(format!(
            "with a line that is not JSON ({e}): {}",
            quote(line)
        ))
    })?;
    if let Some(error) = answer.get("error") {
        let reason = error
            .as_str()
            .map_or_else(|| error.to_string(), str::to_string);
        return Err(GrpoError::Reward(format!(
            "the reward command `{name}` could not score the batch: {reason}"
        )));
    }
    let rewards = answer
        .get("rewards")
        .and_then(Value::as_array)
        .ok_or_else(|| wrong(format!("without a `rewards` list: {}", quote(line))))?;
    if rewards.len() != expected {
        return Err(wrong(format!(
            "with {} rewards for {expected} completions",
            rewards.len()
        )));
    }
    rewards
        .iter()
        .enumerate()
        .map(|(i, reward)| match reward.as_f64() {
            Some(value) if value.is_finite() => Ok(value),
            _ => Err(GrpoError::Reward(format!(
                "the reward command `{name}` gave completion {i} the reward {reward}; a reward \
                 is a finite number, and a completion it cannot score is an error, not a 0"
            ))),
        })
        .collect()
}

impl RewardFunction for ExternalCommandReward {
    fn compute(
        &self,
        prompts: &[String],
        completions: &[String],
        images: Option<&[Vec<pmetal_runtime::compat::Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        let references = vec![None; completions.len()];
        self.compute_with_references(prompts, completions, &references, images)
    }

    fn compute_with_references(
        &self,
        prompts: &[String],
        completions: &[String],
        references: &[Option<String>],
        images: Option<&[Vec<pmetal_runtime::compat::Array>]>,
    ) -> GrpoResult<Vec<f64>> {
        self.score(&RewardBatch {
            prompts,
            completions,
            references,
            columns: &[],
            images,
        })
    }

    fn score(&self, batch: &RewardBatch<'_>) -> GrpoResult<Vec<f64>> {
        batch.validate()?;
        let line = Self::request_line(batch)?;
        let mut process = self.process.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(failed) = &process.failed {
            return Err(GrpoError::Reward(failed.clone()));
        }
        let sent = process
            .requests
            .as_ref()
            .is_some_and(|requests| requests.send(line).is_ok());
        if !sent {
            let status = process.exit_status();
            return Err(self.fail(
                &mut process,
                format!("the reward command `{}` {status}", self.name),
            ));
        }
        match process.responses.recv_timeout(self.timeout) {
            Ok(line) => parse_answer(&self.name, &line, batch.completions.len()),
            Err(RecvTimeoutError::Timeout) => {
                let why = format!(
                    "the reward command `{}` did not answer within {:?} (killed)",
                    self.name, self.timeout
                );
                Err(self.fail(&mut process, why))
            }
            Err(RecvTimeoutError::Disconnected) => {
                let status = process.exit_status();
                let why = format!(
                    "the reward command `{}` {status} before it answered",
                    self.name
                );
                Err(self.fail(&mut process, why))
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grpo::{CombinedReward, DatasetRow};
    use serde_json::json;

    const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/fake_reward.py");

    fn fake_with(timeout: Duration, args: &[&str]) -> ExternalCommandReward {
        let mut command = vec!["python3".to_string(), FIXTURE.to_string()];
        command.extend(args.iter().map(|a| a.to_string()));
        ExternalCommandReward::spawn(
            ExternalRewardConfig::new(command)
                .with_timeout(timeout)
                .with_shutdown_grace(Duration::from_millis(300)),
        )
        .expect("the fake reward starts")
    }

    fn fake(args: &[&str]) -> ExternalCommandReward {
        fake_with(Duration::from_secs(20), args)
    }

    fn strings(texts: &[&str]) -> Vec<String> {
        texts.iter().map(|t| t.to_string()).collect()
    }

    /// Score `completions`, each answering a prompt `p` and carrying `rows`.
    fn score(
        reward: &dyn RewardFunction,
        completions: &[&str],
        rows: &[serde_json::Value],
    ) -> GrpoResult<Vec<f64>> {
        let prompts = vec!["p".to_string(); completions.len()];
        let completions = strings(completions);
        let references = vec![None; completions.len()];
        let columns: Vec<DatasetRow> = rows
            .iter()
            .map(|row| Arc::new(row.as_object().cloned().unwrap_or_default()))
            .collect();
        reward.score(&RewardBatch {
            prompts: &prompts,
            completions: &completions,
            references: &references,
            columns: &columns,
            images: None,
        })
    }

    fn reward_error(result: GrpoResult<Vec<f64>>) -> String {
        match result {
            Err(GrpoError::Reward(message)) => message,
            other => panic!("expected a reward error, got {other:?}"),
        }
    }

    #[test]
    fn one_process_scores_batch_after_batch() {
        let reward = fake(&["length"]);
        assert_eq!(score(&reward, &["a", "abc"], &[]).unwrap(), [1.0, 3.0]);
        assert_eq!(score(&reward, &["abcd"], &[]).unwrap(), [4.0]);
        // The trait's own entry points reach the process too.
        let got = reward
            .compute(&strings(&["p"]), &strings(&["xy"]), None)
            .unwrap();
        assert_eq!(got, [2.0]);
        assert_eq!(reward.name(), "python3");
    }

    #[test]
    fn dataset_columns_reach_the_command_one_entry_per_completion() {
        let reward = fake(&["column", "score"]);
        let rows = [json!({"score": 0.25}), json!({"score": 7})];
        assert_eq!(score(&reward, &["a", "b"], &rows).unwrap(), [0.25, 7.0]);
        // A row without the column is a null for that completion, which the
        // command (here, `float(None)`) has to handle: the run fails.
        let reward = fake(&["column", "score"]);
        let rows = [json!({"score": 1}), json!({})];
        assert!(score(&reward, &["a", "b"], &rows).is_err());
    }

    #[test]
    fn the_request_carries_prompts_completions_and_every_column_once() {
        let reward = fake(&["keys"]);
        // prompts, completions, plus `answer` and `id`; a column that shares
        // a name with a request key is not sent twice.
        let rows = [
            json!({"answer": "1", "id": 1, "completions": "shadowed"}),
            json!({"answer": "2", "id": 2}),
        ];
        assert_eq!(score(&reward, &["a", "b"], &rows).unwrap(), [4.0, 4.0]);
        assert_eq!(score(&reward, &["a"], &[]).unwrap(), [2.0]);
    }

    #[test]
    fn external_rewards_combine_with_weights_and_the_built_ins() {
        let combined = CombinedReward::new()
            .add(Box::new(fake(&["constant", "1"])), 0.5)
            .add(Box::new(fake(&["length"])), 2.0)
            .add(Box::new(crate::grpo::LengthReward::new(100)), 1.0);
        let completions = strings(&["a fine length", "x"]);
        let prompts = strings(&["p", "p"]);
        let got = combined.compute(&prompts, &completions, None).unwrap();
        // 0.5 * 1 + 2 * len + the built-in length reward (0.1, then -0.5).
        assert_eq!(got, [0.5 + 26.0 + 0.1, 0.5 + 2.0 - 0.5]);
    }

    #[test]
    fn a_command_that_exits_fails_the_run_with_what_it_said() {
        let reward = fake(&["crash-after", "1"]);
        assert_eq!(score(&reward, &["a"], &[]).unwrap(), [1.0]);
        let message = reward_error(score(&reward, &["a"], &[]));
        assert!(message.contains("exited with"), "{message}");
        assert!(message.contains("status: 3"), "{message}");
        assert!(
            message.contains("scoring service lost its database"),
            "{message}"
        );
        // It stays failed: there is no silent restart, and no zero.
        let again = reward_error(score(&reward, &["a"], &[]));
        assert!(again.contains("exited with"), "{again}");
    }

    #[test]
    fn a_command_that_does_not_answer_in_time_is_killed() {
        let reward = fake_with(Duration::from_millis(400), &["hang"]);
        let started = Instant::now();
        let message = reward_error(score(&reward, &["a"], &[]));
        assert!(message.contains("did not answer within"), "{message}");
        assert!(started.elapsed() < Duration::from_secs(10));
        assert!(reward_error(score(&reward, &["a"], &[])).contains("did not answer"));
    }

    #[test]
    fn an_answer_that_is_not_a_reward_per_completion_is_an_error() {
        let cases = [
            ("wrong-length", "3 rewards for 2 completions"),
            ("null", "completion 0 the reward null"),
            ("nan", "completion 0 the reward \"NaN\""),
            ("garbage", "not JSON"),
        ];
        for (mode, expected) in cases {
            let reward = fake(&[mode]);
            let message = reward_error(score(&reward, &["a", "b"], &[]));
            assert!(message.contains(expected), "{mode}: {message}");
        }
        let reward = fake(&["error", "the sandbox is down"]);
        let message = reward_error(score(&reward, &["a"], &[]));
        assert!(message.contains("the sandbox is down"), "{message}");
    }

    #[test]
    fn a_command_that_cannot_start_is_a_config_error() {
        let config = ExternalRewardConfig::new(vec!["/no/such/reward-command".into()]);
        let Err(GrpoError::Config(message)) = ExternalCommandReward::spawn(config) else {
            panic!("expected a config error");
        };
        assert!(message.contains("/no/such/reward-command"), "{message}");
        assert!(ExternalRewardConfig::from_command_line("  ").is_err());
        assert!(ExternalRewardConfig::from_command_line("python3 'unclosed").is_err());
    }

    #[test]
    fn a_command_line_is_split_like_a_shell_would_split_it() {
        let config =
            ExternalRewardConfig::from_command_line(r#"python3 "my reward.py" --mode 'a b'"#)
                .unwrap();
        assert_eq!(config.command, ["python3", "my reward.py", "--mode", "a b"]);
    }

    /// Whether a process with `pid` is still running.
    fn alive(pid: &str) -> bool {
        Command::new("kill")
            .args(["-0", pid])
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    #[test]
    fn dropping_the_reward_kills_the_commands_whole_process_group() {
        let dir = tempfile::tempdir().unwrap();
        let pidfile = dir.path().join("child.pid");
        let reward = fake(&["spawn-child", pidfile.to_str().unwrap()]);
        assert_eq!(score(&reward, &["ab"], &[]).unwrap(), [2.0]);
        let pid = std::fs::read_to_string(&pidfile).unwrap();
        assert!(alive(&pid), "the helper the command started is running");

        drop(reward);

        let deadline = Instant::now() + Duration::from_secs(5);
        while alive(&pid) && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(50));
        }
        assert!(!alive(&pid), "the helper outlived the reward");
    }
}
