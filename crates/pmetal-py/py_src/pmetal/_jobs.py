"""Run any ``pmetal`` job as a subprocess: the whole command line from Python.

Every job spec (``TrainSpec``, ``GrpoSpec``, ``MergeSpec``, ...) knows the
arguments of the ``pmetal`` command it stands for (``spec.to_argv()``). ``run``
and ``start`` run that command, so anything the command line does is reachable
from Python with the command line's own behaviour, defaults and validation, and
with process isolation: a job that exhausts memory or crashes the GPU driver
does not take the interpreter with it.

Training commands report their progress as a stream of events (a loss and
learning rate per step); ``on_event`` receives them as dicts.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from .pmetal import CancelledError, PMetalError, __version__

__all__ = [
    "CliNotFoundError",
    "Job",
    "JobError",
    "JobFailedError",
    "JobResult",
    "JobSpec",
    "find_cli",
    "run",
    "start",
]

#: The subcommands that write a ``--log-events`` stream.
EVENT_LOG_SUBCOMMANDS = frozenset({"train", "distill", "grpo", "rlkd", "preference"})

#: Exit status of a ``pmetal`` run that was interrupted (128 + SIGINT).
_INTERRUPTED = 130

#: The most stdout a job's result keeps, so a chatty job cannot fill memory.
_MAX_CAPTURE = 64 * 1024 * 1024


class JobError(PMetalError):
    """A ``pmetal`` subprocess job could not run, or did not succeed."""


JobError.__module__ = "pmetal"


class CliNotFoundError(JobError, FileNotFoundError):
    """No ``pmetal`` command line was found to run the job."""


CliNotFoundError.__module__ = "pmetal"


class JobFailedError(JobError):
    """The command exited with a failure status.

    Attributes:
        returncode: The exit status.
        command: The command that ran.
        stderr: The end of the job's standard error, where the reason is.
        events: The events the job wrote before it failed.
    """

    def __init__(
        self,
        message: str,
        *,
        returncode: int,
        command: tuple[str, ...],
        stderr: str,
        events: tuple[dict[str, Any], ...],
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.command = command
        self.stderr = stderr
        self.events = events


JobFailedError.__module__ = "pmetal"


@runtime_checkable
class JobSpec(Protocol):
    """What ``run`` and ``start`` need of a spec: every ``pmetal.*Spec`` is one."""

    subcommand: ClassVar[str]

    def normalize(self) -> None: ...

    def to_argv(self) -> list[str]: ...


@dataclass(frozen=True)
class JobResult:
    """A finished job.

    Attributes:
        returncode: The exit status (0 on success).
        command: The command that ran.
        stdout: What the job printed on standard output (empty with ``output="inherit"``).
        stderr: What it printed on standard error (the last 64 KiB).
        events: The events of a training job, in order.
    """

    returncode: int
    command: tuple[str, ...]
    stdout: str
    stderr: str
    events: tuple[dict[str, Any], ...]

    def json(self) -> Any:
        """Standard output parsed as JSON (for a spec that asks for ``json=True``)."""
        return json.loads(self.stdout)

    @property
    def summary(self) -> dict[str, Any] | None:
        """The ``completed`` event's summary, when the job wrote one."""
        for event in reversed(self.events):
            if event.get("event") == "completed":
                summary = event.get("summary")
                return summary if isinstance(summary, dict) else None
        return None


def _bundled_cli() -> Path:
    return Path(__file__).resolve().parent / "_bin" / "pmetal"


_checked: set[Path] = set()


def _fallback_directories() -> list[Path]:
    """Where ``pmetal`` is installed to that a program with a short ``PATH`` does not look."""
    return [Path.home() / ".cargo" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin")]


def find_cli(cli: str | os.PathLike[str] | None = None) -> Path:
    """The ``pmetal`` command line to run jobs with.

    In order: the ``cli`` argument; the ``PMETAL_CLI`` environment variable; the
    command line shipped inside this package; ``pmetal`` on ``PATH``; then
    ``~/.cargo/bin``, ``/opt/homebrew/bin`` and ``/usr/local/bin``, which a program
    started from an IDE or Finder (with launchd's short ``PATH``) does not search.
    The first of the arguments or the variable that is set is the only one tried.

    Raises:
        CliNotFoundError: None of them exists.
    """
    candidates: list[Path] = []
    if cli is not None:
        candidates.append(Path(cli))
    elif env := os.environ.get("PMETAL_CLI"):
        candidates.append(Path(env))
    else:
        candidates.append(_bundled_cli())
        if on_path := shutil.which("pmetal"):
            candidates.append(Path(on_path))
        for directory in _fallback_directories():
            if (fallback := directory / "pmetal") not in candidates:
                candidates.append(fallback)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            _warn_on_version_mismatch(candidate)
            return candidate
    raise CliNotFoundError(
        "no pmetal command line to run the job with: pass cli=..., set PMETAL_CLI, "
        "or install the `pmetal` command (looked for " + ", ".join(str(c) for c in candidates) + ")"
    )


def _warn_on_version_mismatch(cli: Path) -> None:
    """A command line from another release may not take the spec's arguments."""
    if cli in _checked:
        return
    _checked.add(cli)
    try:
        out = subprocess.run(
            [str(cli), "--version"], capture_output=True, text=True, timeout=30, check=False
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return
    version = out[-1] if out else ""
    if version and version != __version__:
        warnings.warn(
            f"{cli} is pmetal {version} but this package is {__version__}; "
            "a spec may carry arguments the command line does not know",
            RuntimeWarning,
            stacklevel=3,
        )


class _Tail(threading.Thread):
    """Reads a JSONL event file as the job writes it."""

    def __init__(self, path: Path, on_event: Callable[[dict[str, Any]], None] | None) -> None:
        super().__init__(name="pmetal-events", daemon=True)
        self.path = path
        self.on_event = on_event
        self.events: list[dict[str, Any]] = []
        self.error: BaseException | None = None
        self.finished = threading.Event()

    def run(self) -> None:
        pending = ""
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                while True:
                    chunk = stream.read()
                    if chunk:
                        pending += chunk
                        *lines, pending = pending.split("\n")
                        for line in lines:
                            self._deliver(line)
                    elif self.finished.is_set():
                        break
                    else:
                        time.sleep(0.05)
            if pending.strip():
                self._deliver(pending)
        except BaseException as error:  # noqa: BLE001 - re-raised by Job.wait
            self.error = error

    def _deliver(self, line: str) -> None:
        if not line.strip():
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return
        self.events.append(event)
        if self.on_event is not None and self.error is None:
            try:
                self.on_event(event)
            except BaseException as error:  # noqa: BLE001 - re-raised by Job.wait
                self.error = error


class _Drain(threading.Thread):
    """Reads a pipe to its end, keeping the first or last ``limit`` bytes."""

    def __init__(self, stream: Any, limit: int, keep: Literal["head", "tail"]) -> None:
        super().__init__(name="pmetal-output", daemon=True)
        self.stream = stream
        self.limit = limit
        self.keep = keep
        self.data = bytearray()

    def run(self) -> None:
        for chunk in iter(lambda: self.stream.read1(65536), b""):
            self.data += chunk
            if len(self.data) > self.limit:
                if self.keep == "tail":
                    del self.data[: len(self.data) - self.limit]
                else:
                    del self.data[self.limit :]

    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")


class Job:
    """A running ``pmetal`` job. Make one with :func:`start`.

    A ``Job`` is a context manager: leaving the block cancels a job that is
    still running.
    """

    def __init__(
        self,
        command: tuple[str, ...],
        process: subprocess.Popen[bytes],
        tail: _Tail | None,
        stdout: _Drain | None,
        stderr: _Drain | None,
        events_dir: tempfile.TemporaryDirectory[str] | None,
    ) -> None:
        self.command = command
        self._process = process
        self._tail = tail
        self._stdout = stdout
        self._stderr = stderr
        self._events_dir = events_dir
        self._result: JobResult | None = None

    @property
    def pid(self) -> int:
        """The job's process id."""
        return self._process.pid

    @property
    def events(self) -> list[dict[str, Any]]:
        """The events received so far."""
        return list(self._tail.events) if self._tail is not None else []

    def poll(self) -> int | None:
        """The exit status, or None while the job runs."""
        return self._process.poll()

    def wait(self, timeout: float | None = None, *, check: bool = True) -> JobResult:
        """Wait for the job to end and return its result.

        Args:
            timeout: Seconds to wait; ``subprocess.TimeoutExpired`` if it runs longer
                (the job keeps running).
            check: Raise ``JobFailedError`` for a failure status. With False, the
                result carries the status.

        Raises:
            KeyboardInterrupt: Ctrl-C while waiting. The job is cancelled first.
            CancelledError: The job was interrupted from outside.
            JobFailedError: The command failed.
            BaseException: Whatever an ``on_event`` callback raised.
        """
        if self._result is None:
            try:
                self._process.wait(timeout=timeout)
            except KeyboardInterrupt:
                self.cancel()
                raise
            self._result = self._finish()
        result = self._result
        if self._tail is not None and self._tail.error is not None:
            raise self._tail.error
        if check and result.returncode != 0:
            if result.returncode == _INTERRUPTED:
                raise CancelledError("the job was interrupted")
            reason = result.stderr.strip().splitlines()[-1:] or ["no error output"]
            raise JobFailedError(
                f"`{' '.join(result.command[:2])}` exited with status {result.returncode}: {reason[0]}",
                returncode=result.returncode,
                command=result.command,
                stderr=result.stderr,
                events=result.events,
            )
        return result

    def _finish(self) -> JobResult:
        for drain in (self._stdout, self._stderr):
            if drain is not None:
                drain.join()
        if self._tail is not None:
            self._tail.finished.set()
            self._tail.join()
        if self._events_dir is not None:
            self._events_dir.cleanup()
        return JobResult(
            returncode=self._process.returncode,
            command=self.command,
            stdout=self._stdout.text() if self._stdout else "",
            stderr=self._stderr.text() if self._stderr else "",
            events=tuple(self._tail.events) if self._tail else (),
        )

    def cancel(self, grace: float = 30.0) -> None:
        """Stop the job: interrupt it as Ctrl-C would (a training run finishes its
        step and saves), and kill it if it has not ended after ``grace`` seconds."""
        if self._process.poll() is not None:
            return
        for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
            try:
                self._process.send_signal(sig)
                self._process.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
            except ProcessLookupError:
                break
        if self._result is None:
            self._result = self._finish()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Events as they arrive, until the job ends."""
        seen = 0
        while True:
            events = self.events
            yield from events[seen:]
            seen = len(events)
            if self._process.poll() is not None:
                final = self.events
                yield from final[seen:]
                return
            time.sleep(0.05)

    def __enter__(self) -> Job:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.cancel()

    def __repr__(self) -> str:
        status = self.poll()
        return (
            f"Job(pid={self.pid}, command={' '.join(self.command[1:3])!r}, "
            f"{'running' if status is None else f'returncode={status}'})"
        )


def start(
    spec: JobSpec,
    *,
    cli: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    output: Literal["capture", "inherit"] = "capture",
) -> Job:
    """Start the ``pmetal`` command a spec stands for and return at once.

    Args:
        spec: Any ``pmetal.*Spec``. It is validated first (``SpecValidationError``).
        cli: The command line to run; see :func:`find_cli`.
        env: Environment variables to add to this process's for the job.
        cwd: The job's working directory (relative paths in the spec are relative to it).
        on_event: Called with each event a training job writes, on a helper thread.
            An exception it raises is raised by ``Job.wait``.
        output: ``"capture"`` keeps the job's output in the result; ``"inherit"``
            lets it print to this process's terminal.

    Raises:
        SpecValidationError: The spec is invalid.
        CliNotFoundError: There is no command line to run.
    """
    spec = copy.copy(spec)
    spec.normalize()
    program = find_cli(cli)
    argv = [str(program), spec.subcommand, *spec.to_argv()]

    events_dir: tempfile.TemporaryDirectory[str] | None = None
    tail: _Tail | None = None
    if spec.subcommand in EVENT_LOG_SUBCOMMANDS:
        events_dir = tempfile.TemporaryDirectory(prefix="pmetal-events-")
        events_path = Path(events_dir.name) / "events.jsonl"
        events_path.touch()
        argv += ["--log-events", str(events_path)]
        tail = _Tail(events_path, on_event)

    capture = output == "capture"
    process = subprocess.Popen(  # noqa: S603 - the command is the pmetal CLI and a spec's arguments
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env={**os.environ, **(env or {})},
        cwd=cwd,
    )
    stdout = _Drain(process.stdout, _MAX_CAPTURE, "head") if capture and process.stdout else None
    stderr = _Drain(process.stderr, 64 * 1024, "tail") if capture and process.stderr else None
    for thread in (stdout, stderr, tail):
        if thread is not None:
            thread.start()
    return Job(tuple(argv), process, tail, stdout, stderr, events_dir)


def run(
    spec: JobSpec,
    *,
    cli: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    output: Literal["capture", "inherit"] = "capture",
    timeout: float | None = None,
    check: bool = True,
) -> JobResult:
    """Run the ``pmetal`` command a spec stands for and wait for it.

    Takes ``start``'s arguments and ``wait``'s ``timeout`` and ``check``.
    Ctrl-C cancels the job (a training run finishes its step and saves) and
    raises ``KeyboardInterrupt``.

    >>> import pmetal
    >>> spec = pmetal.MergeSpec(...)  # doctest: +SKIP
    >>> result = pmetal.run(spec)  # doctest: +SKIP
    """
    job = start(spec, cli=cli, env=env, cwd=cwd, on_event=on_event, output=output)
    try:
        return job.wait(timeout=timeout, check=check)
    except subprocess.TimeoutExpired:
        job.cancel()
        raise
    finally:
        if job.poll() is None:
            job.cancel()
