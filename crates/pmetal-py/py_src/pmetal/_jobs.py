"""Run any ``pmetal`` job as a supervised subprocess: the whole command line from Python.

Every job spec (``TrainSpec``, ``GrpoSpec``, ``MergeSpec``, ...) knows the
arguments of the ``pmetal`` command it stands for (``spec.to_argv()``). ``run``
and ``start`` run that command, so anything the command line does is reachable
from Python with the command line's own behaviour, defaults and validation, and
with process isolation: a job that exhausts memory or crashes the GPU driver
does not take the interpreter with it.

The job runs under ``pmetal-supervisor``, the runner the terminal UI, the MCP
server and the desktop app use: the job leads a process group of its own (a
stop reaches what it spawned, and none outlives this program), its output is
redacted of credentials and bounded, and a stop is SIGTERM, then SIGKILL after a
grace period. Training commands report their progress as a stream of events (a
loss and learning rate per step); ``on_event`` receives them as dicts.
"""

from __future__ import annotations

import atexit
import copy
import json
import os
import shutil
import subprocess
import tempfile
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from .pmetal import (
    CancelledError,
    PMetalError,
    __version__,
    _Job as _NativeJob,
    _start_job as _start_job,
    _stop_all_jobs as _stop_all_jobs,
)

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

#: Exit status of a ``pmetal`` run that was interrupted (128 + SIGINT).
_INTERRUPTED = 130

# A job still running when the interpreter exits is stopped: none outlives it.
atexit.register(_stop_all_jobs)


class JobError(PMetalError):
    """A ``pmetal`` subprocess job could not run, or did not succeed."""


JobError.__module__ = "pmetal"


class CliNotFoundError(JobError, FileNotFoundError):
    """No ``pmetal`` command line was found to run the job."""


CliNotFoundError.__module__ = "pmetal"


class JobFailedError(JobError):
    """The command exited with a failure status.

    Attributes:
        returncode: The exit status (-1 when a signal ended it).
        command: The command that ran, with credentials hidden.
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
    reports_events: ClassVar[bool]

    def normalize(self) -> None: ...

    def to_argv(self) -> list[str]: ...

    def _launch(self, events_dir: str | os.PathLike[str] | None = None) -> tuple[str, list[str], str | None]: ...


@dataclass(frozen=True)
class JobResult:
    """A finished job.

    Attributes:
        status: ``"completed"`` (exit status 0), ``"failed"`` or ``"stopped"``
            (``Job.cancel`` ended it).
        returncode: The exit status: 0 on success, -1 when a signal ended the
            process, None for a job that was stopped.
        command: The command that ran, with credentials hidden.
        stdout: What the job printed on standard output, with credentials hidden.
        stderr: What it printed on standard error (the last 64 KiB), likewise.
        events: The events of a training job, in order.
        metrics: What the job reported of its progress (``last_step``,
            ``last_loss``, ``last_lr``, ``tokens_per_second``, ...).
        completion: What the job said when it finished, when it said anything.
    """

    status: Literal["completed", "failed", "stopped"]
    returncode: int | None
    command: tuple[str, ...]
    stdout: str
    stderr: str
    events: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    completion: dict[str, Any] | None

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
        return self.completion


def _bundled_cli() -> Path:
    return Path(__file__).resolve().parent / "_bin" / "pmetal"


_checked: set[Path] = set()


def _fallback_directories() -> list[Path]:
    """Where ``pmetal`` is installed to that a program with a short ``PATH`` does not look."""
    return [Path.home() / ".cargo" / "bin", Path("/opt/homebrew/bin"), Path("/usr/local/bin")]


def find_cli(cli: str | os.PathLike[str] | None = None) -> Path:
    """The ``pmetal`` command line to run jobs with.

    In order: the ``cli`` argument; the ``PMETAL_BIN`` environment variable (as
    for the terminal UI and the MCP server); the command line shipped inside this
    package; ``pmetal`` on ``PATH``; then ``~/.cargo/bin``, ``/opt/homebrew/bin``
    and ``/usr/local/bin``, which a program started from an IDE or Finder (with
    launchd's short ``PATH``) does not search. The first of the argument or the
    variable that is set is the only one tried.

    Raises:
        CliNotFoundError: None of them exists.
    """
    candidates: list[Path] = []
    if cli is not None:
        candidates.append(Path(cli))
    elif env := os.environ.get("PMETAL_BIN"):
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
        "no pmetal command line to run the job with: pass cli=..., set PMETAL_BIN, "
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


class Job:
    """A running ``pmetal`` job. Make one with :func:`start`.

    A ``Job`` is a context manager: leaving the block stops a job that is still
    running. Letting go of it does not stop the job; the end of the program does.
    """

    def __init__(self, native: _NativeJob) -> None:
        self._native = native

    @property
    def id(self) -> str:
        """The supervisor's identifier for the job."""
        return self._native.id

    @property
    def command(self) -> tuple[str, ...]:
        """The command that runs, with credentials hidden."""
        return tuple(self._native.command)

    @property
    def status(self) -> Literal["running", "stopping", "completed", "failed", "stopped"]:
        """Where the job stands."""
        return self._native.status()[0]

    @property
    def events(self) -> list[dict[str, Any]]:
        """The events received so far."""
        return self._native.events()

    @property
    def metrics(self) -> dict[str, Any]:
        """What the job has reported of its progress so far."""
        return self._native.metrics()

    def poll(self) -> int | None:
        """The exit status; None while the job runs, and for one that was stopped."""
        state, code, _ = self._native.status()
        return None if state in ("running", "stopping") else code

    def logs(self, tail: int = 200) -> list[tuple[Literal["stdout", "stderr"], str]]:
        """The last ``tail`` lines the job printed, as ``(stream, text)``.

        Lines are redacted of credentials and bounded; a long line is cut.
        """
        return self._native.logs(tail)

    def wait(self, timeout: float | None = None, *, check: bool = True) -> JobResult:
        """Wait for the job to end and return its result.

        Args:
            timeout: Seconds to wait; ``subprocess.TimeoutExpired`` if it runs longer
                (the job keeps running).
            check: Raise for a job that failed or was stopped. With False, the
                result carries the status.

        Raises:
            KeyboardInterrupt: Ctrl-C while waiting. The job is stopped first.
            CancelledError: The job was stopped, or interrupted from outside.
            JobFailedError: The command failed.
            BaseException: Whatever an ``on_event`` callback raised.
        """
        try:
            done = self._native.wait(timeout)
        except KeyboardInterrupt:
            self.cancel()
            raise
        if not done:
            raise subprocess.TimeoutExpired(list(self.command), timeout if timeout is not None else 0.0)
        self._native.raise_callback_error()
        result = self._result()
        if check:
            if result.status == "stopped" or result.returncode == _INTERRUPTED:
                raise CancelledError("the job was stopped")
            if result.status == "failed":
                reason = self._native.status()[2] or "no error output"
                raise JobFailedError(
                    f"`{' '.join(result.command[:2])}` failed: {reason}",
                    returncode=-1 if result.returncode is None else result.returncode,
                    command=result.command,
                    stderr=result.stderr,
                    events=result.events,
                )
        return result

    def _result(self) -> JobResult:
        state, code, _ = self._native.status()
        if state == "running" or state == "stopping":
            raise RuntimeError("the job has not ended")  # wait() returns only when it has
        return JobResult(
            status=state,
            returncode=code,
            command=self.command,
            stdout=self._native.stdout(),
            stderr=self._native.stderr(),
            events=tuple(self._native.events()),
            metrics=self._native.metrics(),
            completion=self._native.completion(),
        )

    def cancel(self) -> None:
        """Stop the job and wait for it to go: SIGTERM to its process group (a
        training run finishes its step and saves), SIGKILL after the grace period
        ``start`` was given."""
        if self._native.finished():
            return
        self._native.stop()
        self._native.wait(self._native.stop_deadline())

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Events as they arrive, until the job ends."""
        seen = 0
        while True:
            done = self._native.wait(0.05)
            events = self._native.events(seen)
            yield from events
            seen += len(events)
            if done:
                return

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
        return f"Job(id={self.id!r}, command={' '.join(self.command[1:3])!r}, {self.status})"


def start(
    spec: JobSpec,
    *,
    cli: str | os.PathLike[str] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    stop_grace: float = 30.0,
) -> Job:
    """Start the ``pmetal`` command a spec stands for and return at once.

    The job inherits this process's environment and working directory (relative
    paths in the spec are relative to it).

    Args:
        spec: Any ``pmetal.*Spec``. It is validated first (``SpecValidationError``).
        cli: The command line to run; see :func:`find_cli`.
        on_event: Called with each event a training job writes, on a helper thread.
            An exception it raises is raised by ``Job.wait``.
        stop_grace: Seconds a stopped job has to exit after SIGTERM before it is killed.

    Raises:
        SpecValidationError: The spec is invalid.
        CliNotFoundError: There is no command line to run.
    """
    spec = copy.copy(spec)
    spec.normalize()
    program = find_cli(cli)
    # A command that streams events is given a scratch directory for the file;
    # the job removes it when it is gone.
    events_dir = tempfile.mkdtemp(prefix="pmetal-events-") if spec.reports_events else None
    try:
        subcommand, argv, events_file = spec._launch(events_dir)
        native = _start_job(str(program), subcommand, argv, events_file, events_dir, stop_grace, on_event)
    except BaseException:
        if events_dir is not None:
            shutil.rmtree(events_dir, ignore_errors=True)
        raise
    return Job(native)


def run(
    spec: JobSpec,
    *,
    cli: str | os.PathLike[str] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    stop_grace: float = 30.0,
    timeout: float | None = None,
    check: bool = True,
) -> JobResult:
    """Run the ``pmetal`` command a spec stands for and wait for it.

    Takes ``start``'s arguments and ``wait``'s ``timeout`` and ``check``.
    Ctrl-C stops the job (a training run finishes its step and saves) and
    raises ``KeyboardInterrupt``.

    >>> import pmetal
    >>> spec = pmetal.MergeSpec(...)  # doctest: +SKIP
    >>> result = pmetal.run(spec)  # doctest: +SKIP
    """
    job = start(spec, cli=cli, on_event=on_event, stop_grace=stop_grace)
    try:
        return job.wait(timeout=timeout, check=check)
    finally:
        job.cancel()  # nothing, if it ended
