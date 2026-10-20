"""The exception hierarchy, and Rust panics as exceptions."""

from __future__ import annotations

import pickle

import pmetal
import pytest
from pmetal import pmetal as native

LEAVES = [
    "ConfigError",
    "SpecValidationError",
    "SerializationError",
    "HubError",
    "ModelLoadError",
    "UnsupportedArchitectureError",
    "TokenizerError",
    "TrainingError",
    "QuantizationError",
    "BackendError",
    "OutOfMemoryError",
    "CancelledError",
]


def test_every_error_is_a_pmetal_error():
    for name in LEAVES:
        cls = getattr(pmetal, name)
        assert issubclass(cls, pmetal.PMetalError) and issubclass(cls, Exception), name
        assert cls.__module__ == "pmetal" and (cls.__doc__ or "").strip(), name
    assert issubclass(pmetal.PMetalError, Exception)


def test_errors_that_stand_for_a_builtin_are_that_builtin_too():
    assert issubclass(pmetal.ConfigError, ValueError)
    assert issubclass(pmetal.SpecValidationError, pmetal.ConfigError)
    assert issubclass(pmetal.SerializationError, ValueError)
    assert issubclass(pmetal.OutOfMemoryError, MemoryError)
    assert issubclass(pmetal.UnsupportedArchitectureError, pmetal.ModelLoadError)
    assert issubclass(pmetal.CliNotFoundError, FileNotFoundError)
    assert issubclass(pmetal.JobFailedError, pmetal.JobError) and issubclass(pmetal.JobError, pmetal.PMetalError)


def test_errors_are_not_runtime_errors():
    # The old bindings raised RuntimeError for everything.
    for name in LEAVES:
        assert not issubclass(getattr(pmetal, name), RuntimeError), name


def test_errors_survive_pickling_and_carry_their_message():
    for name in LEAVES:
        cls = getattr(pmetal, name)
        restored = pickle.loads(pickle.dumps(cls("it broke")))
        assert type(restored) is cls and str(restored) == "it broke"


def test_failures_from_rust_arrive_as_their_own_type(tiny_model, tmp_path):
    with pytest.raises(pmetal.SpecValidationError):
        pmetal.TrainSpec().normalize()
    with pytest.raises(pmetal.HubError):
        pmetal.download_model("org/never-cached", offline=True, cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        pmetal.Model.load(tmp_path / "missing")
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "tokenizer.json").write_text("{ not json")
    with pytest.raises(pmetal.TokenizerError):
        pmetal.Tokenizer.from_pretrained(broken)
    with pytest.raises(FileNotFoundError, match="no tokenizer.json"):
        pmetal.Model.load(tiny_model.parent)  # a directory holding the model, not the model
    no_weights = tmp_path / "no-weights"
    no_weights.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (no_weights / name).write_text((tiny_model / name).read_text())
    with pytest.raises(pmetal.ModelLoadError):
        pmetal.Model.load(no_weights)


def test_an_error_message_names_what_went_wrong(tmp_path):
    with pytest.raises(pmetal.HubError) as caught:
        pmetal.download_model("org/never-cached", offline=True, cache_dir=tmp_path)
    assert "org/never-cached" in str(caught.value)


def test_the_build_unwinds_on_a_panic():
    """A `--release` build aborts the interpreter on a panic; the wheel profile does not."""
    assert pmetal.UNWINDS_ON_PANIC is True


# -- panics -------------------------------------------------------------------------------------------

needs_hooks = pytest.mark.skipif(not hasattr(native, "_panic"), reason="built without the test-hooks feature")


@needs_hooks
def test_this_build_unwinds_on_a_panic():
    """The wheel profile unwinds; the workspace's release profile aborts, which would end the interpreter."""
    assert native._panic_unwinds() is True


@needs_hooks
def test_a_rust_panic_is_a_python_exception_and_the_interpreter_lives_on():
    with pytest.raises(BaseException, match="boom") as caught:
        native._panic("boom")
    assert type(caught.value).__name__ == "PanicException"
    assert not isinstance(caught.value, Exception), "a panic is a BaseException, like SystemExit"
    assert pmetal.TrainSpec(model="m", dataset="d").epochs  # still working


@needs_hooks
def test_a_panic_in_an_async_task_is_a_python_exception_too():
    with pytest.raises(BaseException, match="task boom") as caught:
        native._panic_in_task("task boom")
    assert type(caught.value).__name__ == "PanicException"
    native._panic_in_task("")  # and the runtime is fine afterwards
