"""Model, Tokenizer and generation, on a tiny random checkpoint.

The checkpoint's text is gibberish; the tests check what does not depend on
quality: determinism, that every sampling argument reaches the sampler, that the
GIL is released, that Ctrl-C stops a generation, and the errors.
"""

from __future__ import annotations

import os
import signal
import threading
import time

import pytest

import pmetal
from conftest import ALPHABET, UNK, VOCAB


@pytest.fixture(scope="module")
def model(tiny_model):
    return pmetal.Model.load(tiny_model)


def test_load_reports_the_architecture_and_path(model, tiny_model):
    assert model.architecture == pmetal.ModelArchitecture.Qwen3
    assert model.path == str(tiny_model)
    assert "Qwen3" in repr(model)


def test_model_load_takes_a_path_object(tiny_model):
    assert pmetal.Model.load(tiny_model).path == str(tiny_model)


def test_greedy_generation_is_deterministic(model):
    first = model.generate("the cat", max_tokens=24, temperature=0.0)
    assert first and model.generate("the cat", max_tokens=24, temperature=0.0) == first
    # temperature 0 or less is greedy, however it is asked for.
    assert model.generate("the cat", max_tokens=24, temperature=-1.0) == first
    assert model.generate("the cat", pmetal.GenerationConfig.greedy(24)) == first


def test_a_seed_gives_the_same_text_and_another_seed_another(model):
    kwargs = {"max_tokens": 32, "temperature": 1.0, "top_k": 0, "top_p": 1.0}
    a = model.generate("the cat", seed=3, **kwargs)
    assert model.generate("the cat", seed=3, **kwargs) == a
    assert model.generate("the cat", seed=4, **kwargs) != a


def test_max_tokens_bounds_the_output(model):
    tokenizer = model.tokenizer
    for n in (1, 5, 17):
        text = model.generate("the cat", max_tokens=n, temperature=0.0)
        assert len(tokenizer.encode(text)) <= n


def test_logit_bias_steers_the_sampler(model):
    target = VOCAB["d"]
    text = model.generate("the cat", max_tokens=12, temperature=0.0, logit_bias={target: 100.0})
    assert text == "d" * len(text) and len(text) == 12
    banned = model.generate("the cat", max_tokens=40, temperature=0.0, logit_bias={VOCAB["e"]: -100.0})
    assert "e" not in banned


def test_extra_stop_tokens_end_the_generation(model):
    plain = model.generate("the cat", max_tokens=20, temperature=0.0)
    stop = model.tokenizer.encode(plain)[3]
    stopped = model.generate("the cat", max_tokens=20, temperature=0.0, stop_tokens=[stop])
    assert len(model.tokenizer.encode(stopped)) <= 3


def test_penalties_reach_the_sampler(model):
    plain = model.generate("the cat", max_tokens=24, temperature=0.0, repetition_penalty=1.0)
    penalised = model.generate("the cat", max_tokens=24, temperature=0.0, repetition_penalty=3.0)
    assert plain != penalised
    freq = model.generate("the cat", max_tokens=24, temperature=0.0, frequency_penalty=5.0)
    assert freq != plain


def test_a_config_and_keywords_combine_with_keywords_winning(model):
    config = pmetal.GenerationConfig(max_tokens=30, temperature=0.0)
    assert len(model.generate("the cat", config)) == len(model.generate("the cat", max_tokens=30, temperature=0.0))
    short = model.generate("the cat", config, max_tokens=4)
    assert len(model.tokenizer.encode(short)) <= 4


def test_generation_config_fields_and_repr():
    config = pmetal.GenerationConfig(max_tokens=9, seed=2, logit_bias={5: 1.5}, stop_tokens=[7])
    assert (config.max_tokens, config.seed, config.temperature) == (9, 2, None)
    assert config.logit_bias == {5: 1.5} and config.stop_tokens == [7]
    assert repr(config) == "GenerationConfig(max_tokens=9, seed=2, stop_tokens=[7], logit_bias=[(5, 1.5)])"
    assert pmetal.GenerationConfig.greedy().temperature == 0.0
    with pytest.raises(TypeError):
        pmetal.GenerationConfig(maxtokens=3)


def test_unset_sampling_fields_take_the_models_own_defaults(tiny_model, model):
    """The tiny model's generation_config.json says top_k=8: with it, text differs from top_k=0."""
    seeded = {"max_tokens": 40, "seed": 11, "temperature": 1.0}
    default = model.generate("the cat", **seeded)
    explicit = model.generate("the cat", top_k=8, top_p=0.95, **seeded)
    assert default == explicit
    assert default != model.generate("the cat", top_k=0, top_p=1.0, **seeded)


def test_stream_yields_the_text_generate_returns(model):
    kwargs = {"max_tokens": 40, "seed": 5, "temperature": 1.0}
    whole = model.generate("the cat", **kwargs)
    pieces = list(model.stream("the cat", **kwargs))
    assert "".join(pieces) == whole
    assert len(pieces) > 1, "text arrives in pieces"


def test_closing_a_stream_ends_the_generation_and_the_model_stays_usable(model):
    stream = model.stream("the cat", max_tokens=100_000, temperature=0.0)
    assert next(stream)
    stream.close()
    assert list(stream) == []
    assert model.generate("the cat", max_tokens=4, temperature=0.0)
    with model.stream("the cat", max_tokens=100_000, temperature=0.0) as again:
        assert next(again)
    assert model.generate("the cat", max_tokens=4, temperature=0.0)


def test_the_model_can_be_used_from_several_threads(model):
    expected = model.generate("the cat", max_tokens=16, temperature=0.0)
    results: list[str] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            results.append(model.generate("the cat", max_tokens=16, temperature=0.0))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and results == [expected] * 6


def test_the_gil_is_released_while_generating(model):
    """Another thread keeps running through a long generation."""
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
        model.generate("the cat", max_tokens=3000, temperature=0.0)
        elapsed = time.perf_counter() - started
    finally:
        stop.set()
        thread.join()
    # A held GIL would leave the ticker with only the moments between calls.
    assert ticks >= elapsed * 100 * 0.2, f"{ticks} ticks in {elapsed:.2f}s"


def interrupt_after(seconds: float) -> threading.Timer:
    timer = threading.Timer(seconds, lambda: os.kill(os.getpid(), signal.SIGINT))
    timer.start()
    return timer


def test_ctrl_c_stops_a_generation_and_the_model_is_usable_after(model):
    timer = interrupt_after(0.3)
    started = time.perf_counter()
    try:
        with pytest.raises(KeyboardInterrupt):
            model.generate("the cat", max_tokens=5_000_000, temperature=0.0)
    finally:
        timer.cancel()
    assert time.perf_counter() - started < 10
    assert model.generate("the cat", max_tokens=4, temperature=0.0)


def test_ctrl_c_stops_a_stream(model):
    timer = interrupt_after(0.3)
    try:
        with pytest.raises(KeyboardInterrupt):
            for _ in model.stream("the cat", max_tokens=5_000_000, temperature=0.0):
                pass
    finally:
        timer.cancel()
    assert model.generate("the cat", max_tokens=4, temperature=0.0)


# -- tokenizer ---------------------------------------------------------------------------------------


def test_tokenizer_round_trips_and_reports_its_special_tokens(tiny_model):
    tokenizer = pmetal.Tokenizer.from_pretrained(tiny_model)
    ids = tokenizer.encode("the cat.")
    assert tokenizer.decode(ids) == "the cat."
    assert tokenizer.vocab_size == len(VOCAB) == len(tokenizer)
    assert tokenizer.eos_token_id == VOCAB["<eos>"]
    assert tokenizer.unk_token_id == VOCAB[UNK]
    assert tokenizer.bos_token_id is None
    assert tokenizer.encode("A") == [VOCAB[UNK]], "characters outside the vocabulary are unknown"
    assert all(c in ALPHABET for c in tokenizer.decode(ids))
    assert pmetal.Tokenizer.from_file(tiny_model / "tokenizer.json").encode("the") == tokenizer.encode("the")
    assert repr(tokenizer) == f"Tokenizer(vocab_size={len(VOCAB)})"


def test_tokenizer_is_shared_between_threads(tiny_model):
    tokenizer = pmetal.Tokenizer.from_pretrained(tiny_model)
    out: list[list[int]] = []
    threads = [threading.Thread(target=lambda: out.append(tokenizer.encode("the cat"))) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(out) == 8 and all(o == out[0] for o in out)


# -- errors ------------------------------------------------------------------------------------------


def test_a_missing_model_directory_is_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a model directory"):
        pmetal.Model.load(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        pmetal.Tokenizer.from_pretrained(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        pmetal.Tokenizer.from_file(tmp_path / "nope.json")


def test_a_directory_without_weights_is_a_model_load_error(tmp_path, tiny_model):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text((tiny_model / "config.json").read_text())
    (broken / "tokenizer.json").write_text((tiny_model / "tokenizer.json").read_text())
    with pytest.raises(pmetal.ModelLoadError) as caught:
        pmetal.Model.load(broken)
    assert isinstance(caught.value, pmetal.PMetalError)


def test_an_unsupported_architecture_is_reported_as_such(tmp_path, tiny_model):
    odd = tmp_path / "odd"
    odd.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (odd / name).write_text((tiny_model / name).read_text())
    (odd / "config.json").write_text('{"model_type": "not-a-real-arch", "architectures": ["Nope"]}')
    with pytest.raises(pmetal.ModelLoadError):
        pmetal.Model.load(odd)


def test_an_empty_prompt_is_a_value_error(model):
    with pytest.raises(ValueError, match="no tokens"):
        model.generate("")


@pytest.mark.real_model
def test_a_real_model_generates_sensible_text():
    model = pmetal.Model.load(os.environ["PMETAL_TEST_MODEL_DIR"])
    text = model.generate("The capital of France is", max_tokens=12, temperature=0.0)
    assert "Paris" in text
