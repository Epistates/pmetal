"""DFlashGenerator on a tiny target and a tiny random draft.

A random draft is almost never right, which is the point of two of these tests:
whatever the draft guesses, the tokens are the target's own.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time

import pytest

import pmetal
from conftest import make_tiny_dflash_draft


@pytest.fixture(scope="module")
def generator(tiny_model, tiny_draft):
    return pmetal.DFlashGenerator(tiny_model, tiny_draft)


@pytest.fixture(scope="module")
def prompt_ids(tiny_model):
    return pmetal.Tokenizer.from_pretrained(tiny_model).encode("the cat")


def test_it_reports_the_drafts_block_size(generator, tiny_model, tiny_draft):
    assert generator.block_size == 4
    assert "block_size=4" in repr(generator) and str(tiny_draft) in repr(generator)


def test_greedy_speculative_decoding_is_the_targets_own_greedy_decoding(generator, prompt_ids, tiny_model):
    tokens, metrics = generator.generate(prompt_ids, max_new_tokens=24)
    assert tokens[: len(prompt_ids)] == prompt_ids and len(tokens) == len(prompt_ids) + 24
    model = pmetal.Model.load(tiny_model)
    plain = model.generate("the cat", max_tokens=24, temperature=0.0)
    assert model.tokenizer.decode(tokens[len(prompt_ids):]) == plain
    assert metrics["num_generated"] == 24 and len(metrics["acceptance_lengths"]) >= 1
    assert 0.0 <= metrics["acceptance_rate"] <= 1.0
    assert set(metrics) == {"num_generated", "total_drafted", "total_accepted", "avg_acceptance_length",
                            "acceptance_rate", "acceptance_lengths"}


def test_a_seed_makes_a_sampled_run_repeatable(generator, prompt_ids):
    kwargs = {"max_new_tokens": 32, "temperature": 1.0}
    a, _ = generator.generate(prompt_ids, seed=5, **kwargs)
    b, _ = generator.generate(prompt_ids, seed=5, **kwargs)
    c, _ = generator.generate(prompt_ids, seed=6, **kwargs)
    assert a == b and a != c


def test_truncation_arguments_reach_the_sampler(generator, prompt_ids):
    base = {"max_new_tokens": 32, "temperature": 1.0, "seed": 3}
    open_run, _ = generator.generate(prompt_ids, **base)
    narrow, _ = generator.generate(prompt_ids, top_k=1, **base)
    greedy, _ = generator.generate(prompt_ids, max_new_tokens=32)
    assert narrow == greedy, "top_k=1 is argmax"
    assert open_run != narrow
    assert generator.generate(prompt_ids, top_p=0.01, **base)[0] == greedy
    assert generator.generate(prompt_ids, min_p=0.99, **base)[0] == greedy


def test_unset_arguments_are_the_specs_defaults(generator, prompt_ids):
    spec = pmetal.DflashSpec()
    explicit, _ = generator.generate(prompt_ids, max_new_tokens=spec.max_new_tokens, temperature=spec.temperature)
    default, _ = generator.generate(prompt_ids)
    assert explicit == default and len(default) == len(prompt_ids) + spec.max_new_tokens


def test_stop_tokens_end_the_generation_and_are_not_emitted(generator, prompt_ids):
    tokens, _ = generator.generate(prompt_ids, max_new_tokens=16)
    stop = tokens[len(prompt_ids) + 2]
    stopped, _ = generator.generate(prompt_ids, max_new_tokens=16, stop_tokens=[stop])
    generated = stopped[len(prompt_ids):]
    assert stop not in generated and len(generated) <= 2


def test_a_block_size_override_changes_speed_not_tokens(generator, prompt_ids):
    a, _ = generator.generate(prompt_ids, max_new_tokens=20)
    b, _ = generator.generate(prompt_ids, max_new_tokens=20, speculative_tokens=2)
    c, _ = generator.generate(prompt_ids, max_new_tokens=20, speculative_tokens=1)
    assert a == b == c


def test_an_empty_prompt_is_a_value_error(generator):
    with pytest.raises(ValueError, match="at least one token"):
        generator.generate([])


def test_a_draft_for_another_vocabulary_is_a_value_error(tiny_model, tmp_path):
    draft = make_tiny_dflash_draft(tmp_path / "draft")
    config = json.loads((draft / "config.json").read_text())
    config["vocab_size"] = 99
    (draft / "config.json").write_text(json.dumps(config))
    with pytest.raises((ValueError, pmetal.ModelLoadError)):
        pmetal.DFlashGenerator(tiny_model, draft)


def test_a_missing_draft_is_file_not_found(tiny_model, tmp_path):
    with pytest.raises(FileNotFoundError):
        pmetal.DFlashGenerator(tiny_model, tmp_path / "nope")


def test_it_is_shared_between_threads_and_releases_the_gil(generator, prompt_ids):
    expected, _ = generator.generate(prompt_ids, max_new_tokens=16)
    results: list[list[int]] = []
    threads = [threading.Thread(target=lambda: results.append(generator.generate(prompt_ids, max_new_tokens=16)[0]))
               for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == [expected] * 4

    ticks = 0
    stop = threading.Event()

    def tick() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            time.sleep(0.001)

    thread = threading.Thread(target=tick)
    thread.start()
    try:
        started = time.perf_counter()
        generator.generate(prompt_ids, max_new_tokens=1500)
        elapsed = time.perf_counter() - started
    finally:
        stop.set()
        thread.join()
    assert ticks >= elapsed * 100 * 0.2, f"{ticks} ticks in {elapsed:.2f}s"


def test_ctrl_c_stops_a_long_run_and_the_generator_is_usable_after(generator, prompt_ids):
    timer = threading.Timer(0.4, lambda: os.kill(os.getpid(), signal.SIGINT))
    timer.start()
    started = time.perf_counter()
    try:
        with pytest.raises(KeyboardInterrupt):
            generator.generate(prompt_ids, max_new_tokens=50_000_000)
    finally:
        timer.cancel()
    assert time.perf_counter() - started < 20
    assert len(generator.generate(prompt_ids, max_new_tokens=8)[0]) == len(prompt_ids) + 8
