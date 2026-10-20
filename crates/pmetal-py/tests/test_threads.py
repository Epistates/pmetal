"""Use from several threads, with the GIL or without it (free-threaded builds).

Run under ``python3.14t -X gil=0`` as well: the module declares itself safe
without the GIL, and these tests are what holds it to that.
"""

from __future__ import annotations

import sys
import threading

import pmetal
import pytest
from conftest import make_tiny_model


def run_threads(target, count: int) -> list[BaseException]:
    errors: list[BaseException] = []

    def guarded(index: int) -> None:
        try:
            target(index)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=guarded, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


def test_the_module_does_not_turn_the_gil_back_on():
    if hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled():
        # A free-threaded interpreter that started without the GIL still has none.
        assert pmetal.TrainSpec(model="m").model == "m"
        assert not sys._is_gil_enabled()


def test_threads_with_their_own_specs_do_not_interfere():
    def work(index: int) -> None:
        for step in range(300):
            spec = pmetal.TrainSpec(model=f"m{index}", dataset="d", epochs=1 + step % 5)
            spec.learning_rate = 1e-5 * (index + 1)
            assert spec.model == f"m{index}" and spec.epochs == 1 + step % 5
            assert pmetal.TrainSpec.from_dict(spec.to_dict()) == spec

    assert run_threads(work, 8) == []


def test_threads_may_read_one_spec_at_once():
    spec = pmetal.TrainSpec(model="m", dataset="d", epochs=3, lora_targets=["q_proj", "v_proj"])

    def work(_: int) -> None:
        for _ in range(500):
            assert spec.epochs == 3 and spec.lora_targets == ["q_proj", "v_proj"]
            assert spec.to_argv()[:2] == ["--model", "m"]

    assert run_threads(work, 8) == []


def test_threads_writing_one_spec_see_a_clean_error_or_succeed():
    """PyO3 guards a mutable object per call: a clash is a RuntimeError, never a torn spec or a crash."""
    spec = pmetal.TrainSpec(model="m", dataset="d")
    clashes = []

    def work(index: int) -> None:
        for step in range(400):
            try:
                spec.epochs = 1 + (index + step) % 7
                assert 1 <= spec.epochs <= 7
            except RuntimeError as clash:
                clashes.append(clash)

    assert run_threads(work, 8) == []
    assert all("borrowed" in str(clash).lower() for clash in clashes)
    assert 1 <= spec.epochs <= 7


def test_tokenizers_and_errors_work_from_many_threads(tiny_model):
    tokenizer = pmetal.Tokenizer.from_pretrained(tiny_model)
    expected = tokenizer.encode("the cat sat")

    def work(_: int) -> None:
        for _ in range(200):
            assert tokenizer.encode("the cat sat") == expected
            assert tokenizer.decode(expected) == "the cat sat"
            with pytest.raises(pmetal.SpecValidationError) as caught:
                pmetal.TrainSpec(model="m", lora_r=0).normalize()
            assert {"dataset", "lora_r"} <= {field for field, _ in caught.value.errors}

    assert run_threads(work, 8) == []


def test_two_models_generate_at_the_same_time(tmp_path):
    """Each model has a thread of its own, so they really do run at once."""
    first = pmetal.Model.load(make_tiny_model(tmp_path / "a", seed=1))
    second = pmetal.Model.load(make_tiny_model(tmp_path / "b", seed=2))
    models = [first, second]
    expected = [m.generate("the cat", max_tokens=48, temperature=0.0) for m in models]
    assert expected[0] != expected[1], "different weights give different text"

    def work(index: int) -> None:
        for _ in range(15):
            assert models[index % 2].generate("the cat", max_tokens=48, temperature=0.0) == expected[index % 2]

    assert run_threads(work, 4) == []
