"""The Python surface against the real ``pmetal`` command line.

Set ``PMETAL_CLI`` to a built ``pmetal``. These check that a spec and the
command it stands for agree: what ``run`` does is what the in-process calls do.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import pmetal
import pytest
from conftest import make_tiny_model

pytestmark = pytest.mark.needs_cli


def train_spec(model: Path, dataset: Path, output: Path, **fields: object) -> pmetal.TrainSpec:
    base: dict[str, object] = {
        "model": str(model),
        "dataset": str(dataset),
        "output_dir": str(output),
        "epochs": 1,
        "max_steps": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "max_seq_len": 32,
        "lora_r": 4,
        "logging_steps": 1,
        "seed": 1,
        "no_adaptive_lr": True,
    }
    base.update(fields)
    return pmetal.TrainSpec(**base)  # type: ignore[arg-type]


def test_the_command_line_trains_what_the_trainer_trains(tiny_model, tiny_dataset, tmp_path):
    events: list[dict] = []
    ran = pmetal.run(train_spec(tiny_model, tiny_dataset, tmp_path / "cli"), on_event=events.append)
    assert ran.returncode == 0
    steps = [e["payload"] for e in events if e["event"] == "metric" and e["payload"].get("kind") == "step"]
    assert [s["step"] for s in steps] == [1, 2, 3]
    assert (tmp_path / "cli" / "adapter_model.safetensors").exists()

    in_process = []

    class Collect:
        def on_step_end(self, step, loss):
            in_process.append(loss)

    pmetal.Trainer(train_spec(tiny_model, tiny_dataset, tmp_path / "py"), [Collect()]).train()
    assert in_process == pytest.approx([s["loss"] for s in steps], rel=1e-4), "one implementation, one result"


def test_the_command_line_and_infer_generate_the_same_text(tiny_model):
    spec = pmetal.InferSpec(model=str(tiny_model), prompt="the cat", max_tokens=12, temperature=0.0, json=True)
    text = pmetal.run(spec).json()["text"]
    assert text == pmetal.infer(tiny_model, "the cat", max_tokens=12, temperature=0.0)


def test_a_merge_runs_and_its_output_loads(tiny_model, tmp_path):
    other = make_tiny_model(tmp_path / "other", seed=5)
    merged = tmp_path / "merged"
    pmetal.run(pmetal.MergeSpec(model_a=str(tiny_model), model_b=str(other), output=str(merged)))
    assert (merged / "model.safetensors").exists()
    assert pmetal.verify_model(merged).is_complete
    assert pmetal.Model.load(merged).generate("the cat", max_tokens=4, temperature=0.0)


def test_a_failing_command_is_a_job_failed_error_with_its_message(tmp_path):
    with pytest.raises(pmetal.JobFailedError) as caught:
        pmetal.run(
            pmetal.MergeSpec(model_a=str(tmp_path / "no"), model_b=str(tmp_path / "nope"), output=str(tmp_path / "o"))
        )
    assert caught.value.returncode != 0 and caught.value.stderr.strip()


def test_cancelling_a_training_job_stops_it_after_the_step_and_keeps_the_adapter(tiny_model, tiny_dataset, tmp_path):
    spec = train_spec(tiny_model, tiny_dataset, tmp_path / "out", max_steps=None, epochs=1000)
    with pmetal.start(spec) as job:
        first = next(iter(job))
        assert first["event"] == "started"
        time.sleep(0.5)
        started = time.perf_counter()
        job.cancel(grace=60)
        result = job.wait(check=False)
    assert result.returncode == 130, result.stderr[-500:]
    assert time.perf_counter() - started < 30
    assert (tmp_path / "out" / "adapter_model.safetensors").exists(), "the adapter was saved"
    with pytest.raises(pmetal.CancelledError):
        job.wait()


def test_the_command_line_is_the_packages_version():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a version mismatch is a RuntimeWarning
        assert pmetal.find_cli() == Path(os.environ["PMETAL_CLI"])
