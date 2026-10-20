"""infer / infer_with_metrics: ``pmetal infer``'s runner, with InferSpec's fields."""

from __future__ import annotations

import os
import signal
import threading
import time

import pytest

import pmetal
from pmetal import pmetal as native


def test_infer_returns_the_text_and_agrees_with_a_loaded_model(tiny_model):
    text = pmetal.infer(tiny_model, "the cat", max_tokens=12, temperature=0.0)
    assert text
    model = pmetal.Model.load(tiny_model)
    assert text == model.generate("the cat", max_tokens=12, temperature=0.0)


def test_infer_with_metrics_describes_the_generation(tiny_model):
    result = pmetal.infer_with_metrics(tiny_model, "the cat", max_tokens=12, temperature=0.0)
    assert isinstance(result, pmetal.InferResult)
    assert result.text == pmetal.infer(tiny_model, "the cat", max_tokens=12, temperature=0.0)
    assert result.num_generated == 12 and result.stopped_by_length and not result.stopped_by_token
    assert result.speculative is None and result.generation_seconds > 0
    assert result.to_dict()["num_generated"] == 12
    assert "InferResult(" in repr(result)


def test_a_seed_makes_sampling_repeatable(tiny_model):
    kwargs = {"max_tokens": 24, "temperature": 1.0, "top_k": 0, "top_p": 1.0}
    a = pmetal.infer(tiny_model, "the cat", seed=3, **kwargs)
    assert pmetal.infer(tiny_model, "the cat", seed=3, **kwargs) == a
    assert pmetal.infer(tiny_model, "the cat", seed=4, **kwargs) != a


def test_options_are_the_specs_fields_with_its_validation(tiny_model):
    with pytest.raises(TypeError, match="unexpected keyword argument 'temprature'.*Did you mean 'temperature'"):
        pmetal.infer(tiny_model, "p", temprature=0.0)
    with pytest.raises(pmetal.SpecValidationError):
        pmetal.infer(tiny_model, "p", top_p=7.5)
    with pytest.raises(pmetal.SpecValidationError):
        pmetal.infer(tiny_model, "p", mtp_model="m")  # --mtp-model needs --mtp


@pytest.mark.parametrize("field,value", [("image", ["a.png"]), ("video", ["frames"]), ("ane", True), ("benchmark", True),
                                         ("json", True), ("tools", "tools.json"), ("backend", "standard"),
                                         ("profile_layers", True), ("hide_thinking", True)])
def test_a_field_only_the_command_line_honours_is_refused_by_name(tiny_model, field, value):
    with pytest.raises(pmetal.ConfigError, match=field) as caught:
        pmetal.infer(tiny_model, "p", **{field: value})
    assert "pmetal.run" in str(caught.value)


def test_every_refused_field_is_a_real_spec_field():
    names = {f.name for f in pmetal.InferSpec.fields()}
    assert set(native.INFER_COMMAND_LINE_ONLY) <= names


def test_a_conflicting_pair_of_speculative_options_is_refused(tiny_model):
    with pytest.raises(pmetal.ConfigError, match="not both"):
        pmetal.infer(tiny_model, "p", mtp=True, draft_model=str(tiny_model))


def test_a_bad_preset_is_a_config_error(tiny_model):
    with pytest.raises(pmetal.ConfigError, match="preset"):
        pmetal.infer(tiny_model, "p", kv_turboquant_preset="q9_9")


def test_a_missing_model_is_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        pmetal.infer(tmp_path / "nope", "p")


def test_ctrl_c_stops_an_inference(tiny_model):
    timer = threading.Timer(0.6, lambda: os.kill(os.getpid(), signal.SIGINT))
    timer.start()
    started = time.perf_counter()
    try:
        with pytest.raises(KeyboardInterrupt):
            pmetal.infer(tiny_model, "the cat", max_tokens=1_000_000, temperature=0.0)
    finally:
        timer.cancel()
    assert time.perf_counter() - started < 30


@pytest.mark.real_model
def test_a_real_model_answers_with_a_chat_template():
    text = pmetal.infer(os.environ["PMETAL_TEST_MODEL_DIR"], "What is 2+2? Answer with one digit.",
                        chat=True, no_thinking=True, max_tokens=24, temperature=0.0)
    assert "4" in text
