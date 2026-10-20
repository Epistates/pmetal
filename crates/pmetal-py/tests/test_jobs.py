"""run / start: any spec as a ``pmetal`` subprocess, against a stand-in command line."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import warnings

import pytest

import pmetal
from pmetal import _jobs

SPEC = pmetal.TrainSpec(model="m", dataset="d", epochs=2)
MERGE = pmetal.MergeSpec(model_a="a", model_b="b", output="out")
QUANT = pmetal.QuantizeSpec(model="m", output="o")


def event_lines(*events: dict) -> str:
    return "".join(json.dumps(e) + "\n" for e in events)


ECHO = """
    print(json.dumps(sys.argv[1:]))
"""


def test_the_command_is_the_subcommand_then_the_specs_arguments(fake_cli):
    cli = fake_cli(ECHO)
    result = pmetal.run(MERGE, cli=cli)
    assert result.json() == ["merge", *MERGE.to_argv()]
    assert result.returncode == 0 and result.command[0] == str(cli)


def test_training_commands_get_an_event_log_and_others_do_not(fake_cli):
    cli = fake_cli(ECHO)
    train = pmetal.run(SPEC, cli=cli).json()
    assert train[0] == "train" and train[-2] == "--log-events" and train[-1].endswith("events.jsonl")
    for spec in (MERGE, QUANT, pmetal.InferSpec(model="m", prompt="p")):
        assert "--log-events" not in pmetal.run(spec, cli=cli).json()


def test_the_event_log_subcommands_are_the_specs_that_stream_events():
    # The command line's own list (main.rs: Distill, Grpo, Rlkd, Preference, and Train).
    names = {cls.subcommand for cls in (pmetal.TrainSpec, pmetal.DistillSpec, pmetal.GrpoSpec,
                                        pmetal.RlkdSpec, pmetal.PreferenceSpec)}
    assert names == set(_jobs.EVENT_LOG_SUBCOMMANDS)


def test_events_reach_on_event_in_order_and_the_result(fake_cli):
    cli = fake_cli(
        """
        path = sys.argv[sys.argv.index("--log-events") + 1]
        with open(path, "a") as log:
            for i in range(3):
                log.write(json.dumps({"event": "metric", "job_id": "j", "payload": {"step": i}}) + "\\n")
                log.flush()
                time.sleep(0.05)
            log.write(json.dumps({"event": "completed", "job_id": "j", "summary": {"final_loss": 1.5}}) + "\\n")
        """
    )
    seen: list[dict] = []
    result = pmetal.run(SPEC, cli=cli, on_event=seen.append)
    assert [e["event"] for e in seen] == ["metric", "metric", "metric", "completed"]
    assert [e["payload"]["step"] for e in seen[:3]] == [0, 1, 2]
    assert list(result.events) == seen
    assert result.summary == {"final_loss": 1.5}


def test_a_partial_last_line_is_not_lost_or_mangled(fake_cli):
    cli = fake_cli(
        """
        path = sys.argv[sys.argv.index("--log-events") + 1]
        with open(path, "a") as log:
            log.write('{"event": "started", "job_id": "j", "kind": "train"}\\n{"event": "comp')
            log.flush(); time.sleep(0.2)
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
        sys.exit(1)
        """
    )
    with pytest.raises(pmetal.JobFailedError, match="out of memory") as caught:
        pmetal.run(SPEC, cli=cli)
    error = caught.value
    assert error.returncode == 1 and "Error: out of memory" in error.stderr
    assert error.command[1] == "train" and isinstance(error, pmetal.PMetalError)
    # check=False hands back the result instead.
    result = pmetal.run(SPEC, cli=cli, check=False)
    assert result.returncode == 1 and "partial output" in result.stdout


def test_an_interrupted_command_is_cancelled_not_failed(fake_cli):
    cli = fake_cli("sys.exit(130)")
    with pytest.raises(pmetal.CancelledError):
        pmetal.run(SPEC, cli=cli)


def test_a_missing_command_line_says_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.delenv("PMETAL_CLI", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(pmetal.CliNotFoundError, match="PMETAL_CLI") as caught:
        pmetal.run(SPEC, cli=tmp_path / "nope")
    assert isinstance(caught.value, FileNotFoundError) and isinstance(caught.value, pmetal.JobError)
    with pytest.raises(pmetal.CliNotFoundError):
        pmetal.find_cli()


def test_the_cli_is_found_through_the_environment_and_the_path(fake_cli, monkeypatch, tmp_path):
    cli = fake_cli(ECHO)
    monkeypatch.setenv("PMETAL_CLI", str(cli))
    assert pmetal.find_cli() == cli
    assert pmetal.run(MERGE).returncode == 0
    monkeypatch.delenv("PMETAL_CLI")
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


def test_environment_and_working_directory_reach_the_job(fake_cli, tmp_path):
    cli = fake_cli('print(json.dumps([os.environ.get("PMETAL_TEST_VAR"), os.getcwd()]))')
    workdir = tmp_path / "work"
    workdir.mkdir()
    result = pmetal.run(MERGE, cli=cli, env={"PMETAL_TEST_VAR": "yes"}, cwd=workdir)
    assert result.json() == ["yes", str(workdir.resolve())] or result.json()[0] == "yes"


def test_a_timeout_cancels_the_job(fake_cli):
    cli = fake_cli("time.sleep(60)")
    started = time.perf_counter()
    with pytest.raises(subprocess.TimeoutExpired):
        pmetal.run(MERGE, cli=cli, timeout=0.3)
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


SLOW = """
    import signal
    signal.signal(signal.SIGINT, lambda *a: sys.exit(130))
    path = sys.argv[sys.argv.index("--log-events") + 1]
    for i in range(1000):
        open(path, "a").write(json.dumps({"event": "log", "job_id": "j", "level": "info", "line": str(i)}) + "\\n")
        time.sleep(0.05)
"""


def test_a_started_job_streams_events_and_can_be_cancelled(fake_cli):
    cli = fake_cli(SLOW)
    with pmetal.start(SPEC, cli=cli) as job:
        assert job.poll() is None and job.pid > 0
        first = next(iter(job))
        assert first["event"] == "log"
        job.cancel(grace=10)
        assert job.poll() == 130
        result = job.wait(check=False)
    assert result.returncode == 130 and len(result.events) >= 1
    assert "running" not in repr(job) and "returncode=130" in repr(job)


def test_leaving_the_with_block_stops_a_running_job(fake_cli):
    cli = fake_cli(SLOW)
    with pmetal.start(SPEC, cli=cli) as job:
        pid = job.pid
    assert job.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_cancel_kills_a_job_that_ignores_the_interrupt(fake_cli, tmp_path):
    ready = tmp_path / "ready"
    cli = fake_cli(
        f"""
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        open({str(ready)!r}, "w").close()
        time.sleep(60)
        """
    )
    job = pmetal.start(MERGE, cli=cli)
    while not ready.exists():  # the signals are ignored from here on
        time.sleep(0.01)
    started = time.perf_counter()
    job.cancel(grace=0.3)
    assert job.poll() == -signal.SIGKILL
    assert time.perf_counter() - started < 30


def test_ctrl_c_while_waiting_cancels_the_job_and_raises(fake_cli):
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
    assert job.poll() is not None and time.perf_counter() - started < 30


def test_inherited_output_goes_to_the_terminal_not_the_result(fake_cli, capfd):
    cli = fake_cli('print("visible")')
    result = pmetal.run(MERGE, cli=cli, output="inherit")
    assert result.stdout == "" and "visible" in capfd.readouterr().out


def test_output_beyond_the_cap_is_dropped_not_buffered_forever(fake_cli, monkeypatch):
    monkeypatch.setattr(_jobs, "_MAX_CAPTURE", 1000)
    cli = fake_cli('sys.stdout.write("x" * 100000)')
    assert len(pmetal.run(MERGE, cli=cli).stdout) == 1000


def test_job_spec_protocol_is_every_spec():
    for name in dir(pmetal):
        if name.endswith("Spec") and name != "JobSpec":
            assert isinstance(getattr(pmetal, name)(), pmetal.JobSpec), name
    assert not isinstance(object(), pmetal.JobSpec)
