"""Trainer: a TrainSpec run in this process, on a tiny random checkpoint.

The run is the one ``pmetal train`` makes, so what is tested here is what the
bindings add: spec validation, callbacks, errors and Ctrl-C.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pmetal
import pytest


def spec_for(tiny_model: Path, dataset: Path, output: Path, **fields: object) -> pmetal.TrainSpec:
    base: dict[str, object] = {
        "model": str(tiny_model),
        "dataset": str(dataset),
        "output_dir": str(output),
        "epochs": 1,
        "max_steps": 3,
        "logging_steps": 1,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_seq_len": 32,
        "lora_r": 4,
        "lora_alpha": 8.0,
        "learning_rate": 1e-3,
        "seed": 1,
        "no_adaptive_lr": True,
    }
    base.update(fields)
    return pmetal.TrainSpec(**base)  # type: ignore[arg-type]


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def __getattr__(self, name: str):
        if not name.startswith("on_"):
            raise AttributeError(name)

        def record(*args: object) -> None:
            self.calls.append((name, args))

        return record

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


def test_a_run_trains_saves_an_adapter_and_reports_a_result(tiny_model, tiny_dataset, tmp_path):
    recorder = Recorder()
    trainer = pmetal.Trainer(spec_for(tiny_model, tiny_dataset, tmp_path / "out"), [recorder])
    result = trainer.train()
    assert isinstance(result, pmetal.TrainResult)
    assert result.total_steps == 3 and result.total_tokens > 0
    assert 0.0 < result.final_loss < 100.0
    assert Path(result.output_dir) == tmp_path / "out"
    weights = Path(result.weights_path)
    assert weights.exists(), result.weights_path
    assert any(p.name == "adapter_config.json" for p in (tmp_path / "out").rglob("*")), "a PEFT adapter"
    assert result.to_dict()["total_steps"] == 3
    assert "TrainResult(" in repr(result)

    names = recorder.names()
    assert names[0] == "on_train_start" and names[-1] == "on_train_end"
    assert names.count("on_step_end") == 3 and names.count("on_step_metrics") == 3
    assert names.count("on_train_start") == names.count("on_train_end") == 1
    steps = [args for name, args in recorder.calls if name == "on_step_end"]
    assert [s[0] for s in steps] == [1, 2, 3]
    assert all(0.0 < s[1] < 100.0 for s in steps)
    metrics = next(args[0] for name, args in recorder.calls if name == "on_step_metrics")
    assert {"step", "loss", "lr", "tok_sec", "tokens", "grad_norm", "total_ms"} <= set(metrics)


def test_builtin_callbacks_run_natively(tiny_model, tiny_dataset, tmp_path):
    log = tmp_path / "metrics.jsonl"
    pmetal.Trainer(
        spec_for(tiny_model, tiny_dataset, tmp_path / "out", max_steps=2),
        [pmetal.MetricsJsonCallback(log), pmetal.LoggingCallback(1), pmetal.ProgressCallback(2)],
    ).train()
    lines = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    steps = [line for line in lines if "step" in line]
    assert len(steps) == 2 and all("loss" in line for line in steps)


def test_the_function_form_is_the_same_run(tiny_model, tiny_dataset, tmp_path):
    result = pmetal.finetune(
        str(tiny_model),
        str(tiny_dataset),
        output_dir=str(tmp_path / "out"),
        max_steps=2,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_seq_len=32,
        lora_r=4,
        no_adaptive_lr=True,
    )
    assert result.total_steps == 2


def test_the_same_seed_gives_the_same_losses(tiny_model, tiny_dataset, tmp_path):
    def losses(out: str) -> list[float]:
        rec = Recorder()
        pmetal.Trainer(spec_for(tiny_model, tiny_dataset, tmp_path / out), [rec]).train()
        return [args[1] for name, args in rec.calls if name == "on_step_end"]

    first, second = losses("a"), losses("b")
    assert first == pytest.approx(second, rel=1e-4)


def test_an_invalid_spec_is_rejected_before_anything_loads(tmp_path):
    with pytest.raises(pmetal.SpecValidationError) as caught:
        pmetal.Trainer(pmetal.TrainSpec(lora_r=0))
    assert {"model", "dataset", "lora_r"} <= {f for f, _ in caught.value.errors}
    with pytest.raises(pmetal.ConfigError):
        pmetal.Trainer(pmetal.TrainSpec(model="m", dataset="d", optimizer="nope"))


def test_a_missing_dataset_or_model_is_the_errors_own_type(tiny_model, tiny_dataset, tmp_path):
    with pytest.raises(FileNotFoundError):
        pmetal.Trainer(spec_for(tiny_model, tmp_path / "missing.jsonl", tmp_path / "o")).train()
    with pytest.raises(pmetal.PMetalError):
        pmetal.Trainer(spec_for(tmp_path / "no-model", tiny_dataset, tmp_path / "o")).train()


def test_the_trainer_exposes_a_copy_of_its_spec(tiny_model, tiny_dataset, tmp_path):
    spec = spec_for(tiny_model, tiny_dataset, tmp_path)
    trainer = pmetal.Trainer(spec)
    spec.epochs = 9
    assert trainer.spec.epochs == 1 and trainer.spec is not trainer.spec
    assert trainer.spec == spec.copy(epochs=1)
    assert "Trainer(" in repr(trainer)


# -- callbacks end a run cleanly -------------------------------------------------------------------


class Boom(Exception):
    pass


def test_an_exception_in_a_callback_ends_the_run_and_is_raised(tiny_model, tiny_dataset, tmp_path):
    class Raises:
        def on_step_end(self, step: int, loss: float) -> None:
            raise Boom(f"at step {step}")

    recorder = Recorder()
    with pytest.raises(Boom, match="at step"):
        pmetal.Trainer(spec_for(tiny_model, tiny_dataset, tmp_path / "out", max_steps=50), [Raises(), recorder]).train()
    # The run ended after the step that raised: it did not go on to 50.
    assert recorder.names().count("on_step_end") <= 2


def test_should_stop_ends_the_run_early_and_cleanly(tiny_model, tiny_dataset, tmp_path):
    class StopAfterTwo:
        steps = 0

        def on_step_end(self, step: int, loss: float) -> None:
            self.steps += 1

        def should_stop(self) -> bool:
            return self.steps >= 2

    result = pmetal.Trainer(
        spec_for(tiny_model, tiny_dataset, tmp_path / "out", max_steps=50), [StopAfterTwo()]
    ).train()
    assert 2 <= result.total_steps < 50
    assert Path(result.weights_path).exists()


def test_ctrl_c_ends_the_run_after_the_step_and_keeps_the_weights(tiny_model, tiny_dataset, tmp_path):
    class Interrupts:
        def on_step_end(self, step: int, loss: float) -> None:
            if step >= 1:
                os.kill(os.getpid(), signal.SIGINT)

    out = tmp_path / "out"
    started = time.perf_counter()
    with pytest.raises(KeyboardInterrupt):
        pmetal.Trainer(spec_for(tiny_model, tiny_dataset, out, max_steps=200), [Interrupts()]).train()
    assert time.perf_counter() - started < 60
    assert any(out.rglob("adapter_model.safetensors")) or any(out.rglob("*.safetensors")), "weights are saved"


#: Serves a dataset through a named pipe, and only once `gate` exists. Argv: the pipe, the files `seen`
#: (made when the run is reading the pipe), `gate` and `stop`, and the dataset to serve. After a
#: minute, or with no gate, it closes the pipe unwritten, so the run reads an empty dataset and fails.
PIPE_SERVER = """\
import os, sys, time
pipe, seen, gate, stop, data = sys.argv[1:6]
deadline = time.monotonic() + 60
while not os.path.exists(stop) and time.monotonic() < deadline:
    try:
        fd = os.open(pipe, os.O_WRONLY | os.O_NONBLOCK)  # fails until something is reading the pipe
    except OSError:
        time.sleep(0.005)
        continue
    open(seen, "w").close()
    while not os.path.exists(gate) and time.monotonic() < deadline:
        time.sleep(0.005)
    os.set_blocking(fd, True)
    if os.path.exists(gate):
        try:
            os.write(fd, open(data, "rb").read())
        except BrokenPipeError:  # the reader looked at the start and closed the pipe
            pass
    os.close(fd)
"""


def test_the_gil_is_released_during_training(tiny_model, tiny_dataset, tmp_path):
    """The run waits for another Python thread, so it can only finish if the GIL is free.

    The dataset is a named pipe that a helper process fills only once a file exists. A ticker
    thread takes the GIL ``wanted`` times after the run is reading the pipe, and then makes the
    file. That takes no time the run can lose by being short. A run that kept the GIL while it
    waited would never let the ticker run, and the helper would give up and close the pipe empty.
    """
    wanted = 100
    pipe, seen, gate, stop = (tmp_path / name for name in ("pipe.jsonl", "seen", "gate", "stop"))
    os.mkfifo(pipe)
    server = subprocess.Popen([sys.executable, "-c", PIPE_SERVER, *map(str, (pipe, seen, gate, stop, tiny_dataset))])
    ticks = 0

    def tick() -> None:
        nonlocal ticks
        while not seen.exists() and not stop.exists():
            time.sleep(0.001)
        while ticks < wanted and not stop.exists():
            ticks += 1
            time.sleep(0)  # let go of the GIL and take it back
        gate.write_text("open")

    thread = threading.Thread(target=tick)
    thread.start()
    try:
        result = pmetal.Trainer(spec_for(tiny_model, pipe, tmp_path / "out", max_steps=3)).train()
    finally:
        stop.write_text("stop")
        thread.join()
        server.wait(timeout=60)
    assert result.total_steps == 3 and ticks == wanted, "another Python thread ran while the run read its dataset"


def test_callbacks_may_be_called_from_the_training_thread_only_with_the_gil(tiny_model, tiny_dataset, tmp_path):
    """A callback that touches shared Python state while another thread does too."""
    shared: list[int] = []
    stop = threading.Event()

    def churn() -> None:
        while not stop.is_set():
            shared.append(len(shared))
            if len(shared) > 10_000:
                shared.clear()

    thread = threading.Thread(target=churn)
    thread.start()

    class Reads:
        def on_step_end(self, step: int, loss: float) -> None:
            _ = len(shared)

    try:
        pmetal.Trainer(spec_for(tiny_model, tiny_dataset, tmp_path / "out", max_steps=3), [Reads()]).train()
    finally:
        stop.set()
        thread.join()


@pytest.mark.real_model
def test_a_real_model_trains_two_steps(tmp_path, tiny_dataset):
    model = os.environ["PMETAL_TEST_MODEL_DIR"]
    result = pmetal.finetune(
        model,
        str(tiny_dataset),
        output_dir=str(tmp_path / "out"),
        max_steps=2,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_seq_len=64,
        lora_r=8,
        no_adaptive_lr=True,
    )
    assert result.total_steps == 2 and result.final_loss > 0
    assert Path(result.weights_path).exists()
