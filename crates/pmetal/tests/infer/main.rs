//! `pmetal infer` on a tiny model, on both engines a checkpoint can run on:
//! the native engine its architecture selects (Qwen3's here), and the shared
//! model, chosen with `PMETAL_DISABLE_NATIVE_BRIDGE` as a person comparing the
//! two would. Each run starts the real binary.

#[path = "../tiny_model/mod.rs"]
mod tiny_model;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

/// The engines a checkpoint can run on.
#[derive(Clone, Copy, Debug)]
enum Engine {
    Native,
    Shared,
}

impl Engine {
    /// What the run logs when it picks this engine.
    fn log_line(self) -> &'static str {
        match self {
            Self::Native => "engine=Native",
            Self::Shared => "engine=Shared",
        }
    }
}

/// The tiny model, in a fresh directory.
fn workspace() -> (TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model");
    tiny_model::write_tiny_model(&model, 1);
    (dir, model)
}

/// `pmetal infer --model <model> <args>` on `engine`.
fn infer(model: &Path, engine: Engine, args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pmetal"));
    command
        .arg("infer")
        .arg("--model")
        .arg(model)
        .args(args)
        .env("RUST_LOG", "info");
    match engine {
        Engine::Native => command.env_remove("PMETAL_DISABLE_NATIVE_BRIDGE"),
        Engine::Shared => command.env("PMETAL_DISABLE_NATIVE_BRIDGE", "1"),
    };
    command.output().expect("start the pmetal binary")
}

/// A successful run's standard output and error, or a failure showing both.
fn succeeded(engine: Engine, output: &Output) -> (String, String) {
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "{engine:?} engine: {}\n{stdout}\n{stderr}",
        output.status
    );
    assert!(
        stderr.contains(engine.log_line()),
        "the run did not use the {engine:?} engine:\n{stderr}"
    );
    (stdout, stderr)
}

/// The `name=value` figures of a benchmark line, such as
/// `Averages: prompt_tps=812.441, generation_tps=95.120, peak_memory=0.012`.
fn figures(line: &str) -> Vec<(&str, f64)> {
    let (_, figures) = line.split_once(": ").unwrap_or(("", line));
    figures
        .split(", ")
        .map(|pair| {
            let (name, value) = pair.split_once('=').unwrap();
            (name, value.trim().parse().unwrap())
        })
        .collect()
}

#[test]
fn benchmark_times_the_model_on_either_engine() {
    let (_dir, model) = workspace();
    for engine in [Engine::Native, Engine::Shared] {
        let output = infer(
            &model,
            engine,
            &[
                "--benchmark",
                "--benchmark-prompt-tokens",
                "16",
                "--max-tokens",
                "8",
                "--benchmark-iters",
                "2",
            ],
        );
        let (stdout, _) = succeeded(engine, &output);
        let trials = stdout.lines().filter(|l| l.starts_with("Trial ")).count();
        assert_eq!(trials, 2, "{engine:?} engine:\n{stdout}");
        let averages = stdout
            .lines()
            .find(|l| l.starts_with("Averages: "))
            .unwrap_or_else(|| panic!("{engine:?} engine printed no averages:\n{stdout}"));
        let figures = figures(averages);
        let names: Vec<&str> = figures.iter().map(|(name, _)| *name).collect();
        assert_eq!(names, ["prompt_tps", "generation_tps", "peak_memory"]);
        // The rates; the peak memory, in GB to three places, rounds to 0
        // on a model this small.
        for (name, value) in &figures[..2] {
            assert!(*value > 0.0, "{engine:?} engine: {name} is {value}");
        }
    }
}

#[test]
fn benchmark_refuses_what_it_cannot_time_before_the_model_loads() {
    let (_dir, model) = workspace();
    let draft = model.to_str().unwrap();
    let timed = ["--benchmark", "--benchmark-prompt-tokens", "16"];
    let cases: [(Vec<&str>, &str); 5] = [
        (vec!["--benchmark"], "--benchmark-prompt-tokens"),
        ([&timed[..], &["--ane"]].concat(), "--ane"),
        ([&timed[..], &["--mtp"]].concat(), "--mtp"),
        (
            [&timed[..], &["--draft-model", draft]].concat(),
            "--draft-model",
        ),
        (
            [&timed[..], &["--profile-layers"]].concat(),
            "--profile-layers",
        ),
    ];
    for (args, names) in cases {
        let output = infer(&model, Engine::Shared, &args);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{args:?} ran:\n{stderr}");
        assert!(
            stderr.contains(names),
            "{args:?} failed without naming {names}:\n{stderr}"
        );
        // The shared model loads in the runner's preparation.
        assert!(
            !stderr.contains("Loading dynamic model"),
            "{args:?} loaded the model before refusing:\n{stderr}"
        );
    }
}
