"""run / start: any spec as a supervised ``pmetal`` subprocess, against a stand-in command line."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import warnings

import pmetal
import pytest
from pmetal import _jobs

SPEC = pmetal.TrainSpec(model="m", dataset="d", epochs=2)
MERGE = pmetal.MergeSpec(model_a="a", model_b="b", output="out")
QUANT = pmetal.QuantizeSpec(model="m", output="o")

ECHO = """
    print(json.dumps(sys.argv[1:]))
"""

#: A job that writes an event a moment after it starts, then runs for a minute.
SLOW = """
    import signal
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
    path = sys.argv[sys.argv.index("--log-events") + 1]
    for i in range(1000):
        open(path, "a").write(json.dumps({"event": "log", "job_id": "j", "level": "info", "line": str(i)}) + "\\n")
        time.sleep(0.05)
"""


def test_the_command_is_the_subcommand_then_the_specs_arguments(fake_cli):
    cli = fake_cli(ECHO)
    result = pmetal.run(MERGE, cli=cli)
    assert result.json() == ["merge", *MERGE.to_argv()]
    assert result.status == "completed" and result.returncode == 0 and result.command[0] == str(cli)


def test_training_commands_get_an_event_log_and_others_do_not(fake_cli):
    cli = fake_cli(ECHO)
    train = pmetal.run(SPEC, cli=cli).json()
    assert train[0] == "train" and train[-2] == "--log-events" and train[-1].endswith("events.jsonl")
    for spec in (MERGE, QUANT, pmetal.InferSpec(model="m", prompt="p")):
        assert "--log-events" not in pmetal.run(spec, cli=cli).json()


def test_the_specs_that_stream_events_are_the_trainers_the_command_line_accepts_the_flag_for():
    # The specs state it themselves (`#[spec(events)]`); the command line refuses
    # `--log-events` for every other command (Commands::writes_event_log).
    names = {
        cls.subcommand
        for cls in vars(pmetal).values()
        if isinstance(cls, type) and hasattr(cls, "reports_events") and cls.reports_events
    }
    assert names == {"train", "distill", "grpo", "rlkd", "preference"}


def test_events_reach_on_event_in_order_and_the_result(fake_cli):
    cli = fake_cli(
        """
        path = sys.argv[sys.argv.index("--log-events") + 1]
        with open(path, "a") as log:
            for i in range(3):
                log.write(json.dumps({"event": "metric", "job_id": "j", "payload": {"kind": "step", "step": i,
                          "epoch": 0, "total_epochs": 1, "total_steps": 3, "loss": 1.0 / (i + 1), "lr": 0.1,
                          "tok_sec": 10.0, "tokens": 5}}) + "\\n")
                log.flush()
                time.sleep(0.15)
            log.write(json.dumps({"event": "completed", "job_id": "j", "summary": {"final_loss": 1.5}}) + "\\n")
        """
    )
    seen: list[dict] = []
    result = pmetal.run(SPEC, cli=cli, on_event=seen.append)
    assert [e["event"] for e in seen] == ["metric", "metric", "metric", "completed"]
    assert [e["payload"]["step"] for e in seen[:3]] == [0, 1, 2]
    assert list(result.events) == seen
    assert result.summary is not None and result.summary["final_loss"] == 1.5
    assert result.metrics["last_step"] == 2 and result.metrics["last_loss"] == pytest.approx(1 / 3)


def test_a_partial_last_line_is_not_lost_or_mangled(fake_cli):
    cli = fake_cli(
        """
        path = sys.argv[sys.argv.index("--log-events") + 1]
        with open(path, "a") as log:
            log.write('{"event": "started", "job_id": "j", "kind": "train"}\\n{"event": "comp')
            log.flush(); time.sleep(0.4)
            log.write('leted", "job_id": "j", "summary": {}}\\n')
        """
    )
    result = pmetal.run(SPEC, cli=cli)
    assert [e["event"] for e in result.events] == ["started", "completed"]


def test_a_failing_command_raises_with_its_status_and_the_end_of_stderr(fake_cli):
    cli = fake_cli(
        """
        print("partial output")
        print("loading...", file=sys.stderr)
        print("Error: out of memory", file=sys.stderr)
        sys.exit(3)
        """
    )
    with pytest.raises(pmetal.JobFailedError, match="out of memory") as caught:
        pmetal.run(SPEC, cli=cli)
    error = caught.value
    assert error.returncode == 3 and "Error: out of memory" in error.stderr
    assert error.command[1] == "train" and isinstance(error, pmetal.PMetalError)
    # check=False hands back the result instead.
    result = pmetal.run(SPEC, cli=cli, check=False)
    assert result.status == "failed" and result.returncode == 3 and "partial output" in result.stdout


def test_an_interrupted_command_is_cancelled_not_failed(fake_cli):
    cli = fake_cli("sys.exit(130)")
    with pytest.raises(pmetal.CancelledError):
        pmetal.run(SPEC, cli=cli)


def test_a_missing_command_line_says_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.delenv("PMETAL_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(_jobs, "_fallback_directories", list)
    with pytest.raises(pmetal.CliNotFoundError, match="PMETAL_BIN") as caught:
        pmetal.run(SPEC, cli=tmp_path / "nope")
    assert isinstance(caught.value, FileNotFoundError) and isinstance(caught.value, pmetal.JobError)
    with pytest.raises(pmetal.CliNotFoundError):
        pmetal.find_cli()


def test_the_cli_is_found_through_the_environment_and_the_path(fake_cli, monkeypatch):
    cli = fake_cli(ECHO)
    monkeypatch.setenv("PMETAL_BIN", str(cli))
    assert pmetal.find_cli() == cli
    assert pmetal.run(MERGE).returncode == 0
    monkeypatch.delenv("PMETAL_BIN")
    monkeypatch.setenv("PATH", f"{cli.parent}{os.pathsep}{os.environ['PATH']}")
    assert pmetal.find_cli() == cli


def test_an_invalid_spec_never_starts_a_process(fake_cli, tmp_path):
    marker = tmp_path / "ran"
    cli = fake_cli(f"open({str(marker)!r}, 'w').close()")
    with pytest.raises(pmetal.SpecValidationError):
        pmetal.run(pmetal.TrainSpec(), cli=cli)
    assert not marker.exists()


def test_the_spec_is_not_changed_by_running_it(fake_cli):
    cli = fake_cli(ECHO)
    spec = MERGE.copy()
    before = spec.to_dict()
    pmetal.run(spec, cli=cli)
    assert spec.to_dict() == before


def test_a_command_line_of_another_release_is_warned_about(fake_cli):
    cli = fake_cli(ECHO, version="0.0.1")
    _jobs._checked.discard(cli)
    with pytest.warns(RuntimeWarning, match="0.0.1"):
        pmetal.find_cli(cli)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pmetal.find_cli(cli)  # said once


def test_the_job_inherits_the_environment_and_the_working_directory(fake_cli, tmp_path, monkeypatch):
    cli = fake_cli('print(json.dumps([os.environ.get("PMETAL_TEST_VAR"), os.getcwd()]))')
    monkeypatch.setenv("PMETAL_TEST_VAR", "yes")
    monkeypatch.chdir(tmp_path)
    assert pmetal.run(MERGE, cli=cli).json() == ["yes", str(tmp_path.resolve())]


def test_a_timeout_stops_the_job(fake_cli):
    cli = fake_cli("time.sleep(60)")
    started = time.perf_counter()
    with pytest.raises(subprocess.TimeoutExpired):
        pmetal.run(MERGE, cli=cli, timeout=0.3, stop_grace=0.5)
    assert time.perf_counter() - started < 20


def test_an_exception_in_on_event_is_raised_by_wait(fake_cli):
    cli = fake_cli(
        """
        path = sys.argv[sys.argv.index("--log-events") + 1]
        open(path, "a").write(json.dumps({"event": "started", "job_id": "j", "kind": "train"}) + "\\n")
        """
    )

    def boom(event: dict) -> None:
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        pmetal.run(SPEC, cli=cli, on_event=boom)


# -- Job ------------------------------------------------------------------------------------------


def test_a_started_job_streams_events_and_can_be_cancelled(fake_cli):
    cli = fake_cli(SLOW)
    with pmetal.start(SPEC, cli=cli) as job:
        assert job.status == "running" and job.poll() is None and job.id
        first = next(iter(job))
        assert first["event"] == "log"
        job.cancel()
        assert job.status == "stopped" and job.poll() is None
        result = job.wait(check=False)
    assert result.status == "stopped" and result.returncode is None and len(result.events) >= 1
    with pytest.raises(pmetal.CancelledError):
        job.wait()
    assert "stopped" in repr(job)


def test_leaving_the_with_block_stops_a_running_job(fake_cli):
    cli = fake_cli(SLOW)
    with pmetal.start(SPEC, cli=cli) as job:
        pass
    assert job.status == "stopped"


def test_cancel_kills_a_job_that_ignores_the_signal(fake_cli, tmp_path):
    ready = tmp_path / "ready"
    cli = fake_cli(
        f"""
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        open({str(ready)!r}, "w").close()
        time.sleep(60)
        """
    )
    job = pmetal.start(MERGE, cli=cli, stop_grace=0.4)
    while not ready.exists():  # the signal is ignored from here on
        time.sleep(0.01)
    started = time.perf_counter()
    job.cancel()
    assert job.status == "stopped"
    assert time.perf_counter() - started < 30


def test_a_stop_reaches_the_processes_the_job_started(fake_cli, tmp_path):
    """The job leads a process group, so what it spawned goes with it."""
    pids = tmp_path / "grandchild.pid"
    cli = fake_cli(
        f"""
        child = __import__("subprocess").Popen(["sleep", "60"])
        open({str(pids)!r}, "w").write(str(child.pid))
        time.sleep(60)
        """
    )
    job = pmetal.start(MERGE, cli=cli, stop_grace=2)
    while not pids.exists():
        time.sleep(0.01)
    grandchild = int(pids.read_text())
    os.kill(grandchild, 0)  # it is alive
    job.cancel()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild, 0)


def test_ctrl_c_while_waiting_stops_the_job_and_raises(fake_cli):
    cli = fake_cli(SLOW)
    timer = threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT))
    started = time.perf_counter()
    job = pmetal.start(SPEC, cli=cli)
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            job.wait()
    finally:
        timer.cancel()
    assert job.status == "stopped" and time.perf_counter() - started < 30


def test_credentials_are_hidden_in_the_output_and_the_command(fake_cli):
    cli = fake_cli(
        """
        print("token hf_abcdefghijklmnop1234 in a line")
        print("Authorization: Bearer abcdefghijklmnop1234", file=sys.stderr)
        sys.exit(1)
        """
    )

    class Secretive:
        """A spec whose command line carries a credential (the real specs keep keys out of argv)."""

        subcommand = "serve"

        def normalize(self) -> None:
            pass

        def to_argv(self) -> list[str]:
            return ["--api-key", "super-secret-value", "--hf-token=hf_abcdefghijklmnop1234", "--port", "8080"]

    spec = Secretive()
    with pytest.raises(pmetal.JobFailedError) as caught:
        pmetal.run(spec, cli=cli)
    error = caught.value
    assert "super-secret-value" not in " ".join(error.command)
    assert "hf_abcdefghijklmnop1234" not in " ".join(error.command) and "8080" in error.command
    assert "abcdefghijklmnop1234" not in error.stderr and "[REDACTED]" in error.stderr
    result = pmetal.run(spec, cli=cli, check=False)
    assert "abcdefghijklmnop1234" not in result.stdout and "[REDACTED]" in result.stdout


def test_a_result_larger_than_a_log_line_arrives_whole(fake_cli):
    """`pmetal infer --json` prints one line; the log keeps 16 KiB of a line, the result keeps all of it."""
    cli = fake_cli('print(json.dumps({"text": "x" * 500_000}))')
    result = pmetal.run(pmetal.InferSpec(model="m", prompt="p", json=True), cli=cli)
    assert len(result.json()["text"]) == 500_000
    job = pmetal.start(MERGE, cli=cli)
    job.wait()
    assert 0 < max(len(text) for _, text in job.logs()) <= 17_000


def test_the_log_is_bounded_and_the_last_lines_are_kept(fake_cli):
    cli = fake_cli('for i in range(30000): print("line", i)')
    job = pmetal.start(MERGE, cli=cli)
    job.wait()
    lines = job.logs(tail=50)
    assert len(lines) == 50 and lines[-1] == ("stdout", "line 29999")
    assert len(job.logs(tail=10**6)) <= 10_000


def test_job_spec_protocol_is_every_spec():
    for name in dir(pmetal):
        if name.endswith("Spec") and name != "JobSpec":
            assert isinstance(getattr(pmetal, name)(), pmetal.JobSpec), name
    assert not isinstance(object(), pmetal.JobSpec)


def test_python_dash_m_pmetal_runs_the_command_line(fake_cli, monkeypatch):
    cli = fake_cli('print("args:", " ".join(sys.argv[1:]))')
    monkeypatch.setenv("PMETAL_BIN", str(cli))
    result = subprocess.run(
        [sys.executable, "-m", "pmetal", "info", "--json"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0 and result.stdout.strip() == "args: info --json", result.stderr
    monkeypatch.setenv("PMETAL_BIN", "/nonexistent/pmetal")
    missing = subprocess.run([sys.executable, "-m", "pmetal"], capture_output=True, text=True, check=False)
    assert missing.returncode != 0 and "no pmetal command line" in missing.stderr


def test_the_places_a_gui_launched_program_would_miss_are_searched(tmp_path, monkeypatch):
    monkeypatch.delenv("PMETAL_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    cargo_bin = tmp_path / ".cargo" / "bin"
    cargo_bin.mkdir(parents=True)
    cli = cargo_bin / "pmetal"
    cli.write_text(f"#!{sys.executable}\nprint('pmetal {pmetal.__version__}')\n")
    cli.chmod(0o755)
    monkeypatch.setattr(_jobs, "_fallback_directories", lambda: [tmp_path / ".cargo" / "bin"])
    assert pmetal.find_cli() == cli


def test_a_job_still_running_at_exit_is_stopped(fake_cli, tmp_path):
    """No job outlives the program that started it."""
    pid_file = tmp_path / "pid"
    cli = fake_cli(f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(120)")
    script = (
        "import pmetal, sys, time\n"
        "spec = pmetal.MergeSpec(model_a='a', model_b='b', output='o')\n"
        f"job = pmetal.start(spec, cli={str(cli)!r}, stop_grace=1)\n"
        f"while not __import__('pathlib').Path({str(pid_file)!r}).exists(): time.sleep(0.01)\n"
        "print('started', flush=True)\n"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0 and "started" in done.stdout, done.stderr
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
    pytest.fail("the job outlived the program that started it")
