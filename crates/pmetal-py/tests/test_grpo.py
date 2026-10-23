"""GrpoTrainer: a GrpoSpec run in this process with Python reward functions.

The run is the one ``pmetal grpo`` makes, on a tiny random checkpoint, so what is
tested here is what the bindings add: the reward callables, their
weights, the GIL around them, and what a failing one raises.
"""

from __future__ import annotations

import json
import shlex
import sys
import textwrap
import threading
from pathlib import Path

import pmetal
import pytest


@pytest.fixture
def prompts(tmp_path: Path) -> Path:
    """Three prompts, each with a reference answer and another column."""
    path = tmp_path / "prompts.jsonl"
    rows = [
        {"prompt": "the cat sat on", "answer": "mat", "difficulty": 1},
        {"prompt": "the dog ran to", "answer": "park", "difficulty": 2},
        {"prompt": "a cat and a", "answer": "dog", "difficulty": 3},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def spec_for(tiny_model: Path, dataset: Path, output: Path, **fields: object) -> pmetal.GrpoSpec:
    base: dict[str, object] = {
        "model": str(tiny_model),
        "dataset": str(dataset),
        "output_dir": str(output),
        "num_generations": 2,
        "max_completion_length": 4,
        "max_seq_len": 128,
        "epochs": 1,
        "lora_r": 4,
        "lora_alpha": 8.0,
        "learning_rate": 1e-4,
        "checkpoint_every": 0,
        "seed": 1,
    }
    base.update(fields)
    return pmetal.GrpoSpec(**base)  # type: ignore[arg-type]


class Recorder:
    """A reward function that records what it was called with."""

    def __init__(self, reward: float = 1.0) -> None:
        self.reward = reward
        self.calls: list[dict[str, object]] = []
        self.threads: set[int] = set()

    def __call__(self, prompts: list[str], completions: list[str], **columns: list[object]) -> list[float]:
        self.threads.add(threading.get_ident())
        self.calls.append({"prompts": prompts, "completions": completions, **columns})
        # Distinct rewards, so a group has an advantage to learn from.
        return [self.reward + 0.1 * i for i in range(len(completions))]


def test_a_reward_function_gets_prompts_completions_and_the_datasets_columns(tiny_model, prompts, tmp_path):
    reward = Recorder()
    result = pmetal.GrpoTrainer(spec_for(tiny_model, prompts, tmp_path / "out"), [reward]).train()

    assert isinstance(result, pmetal.GrpoResult)
    assert result.total_steps == 3 and not result.stopped
    assert (tmp_path / "out" / "adapter_model.safetensors").exists()
    assert result.to_dict()["total_steps"] == 3 and "GrpoResult(" in repr(result)

    assert len(reward.calls) == 3, "one batch per prompt"
    answers = []
    for call in reward.calls:
        # One entry per completion (a group of num_generations) in every argument.
        assert len(call["prompts"]) == len(call["completions"]) == 2  # type: ignore[arg-type]
        assert all(isinstance(p, str) for p in call["prompts"])  # type: ignore[attr-defined]
        assert all(isinstance(c, str) for c in call["completions"])  # type: ignore[attr-defined]
        assert len(call["answer"]) == len(call["difficulty"]) == 2  # type: ignore[arg-type]
        assert call["answer"][0] == call["answer"][1]  # type: ignore[index]
        answers.append(call["answer"][0])  # type: ignore[index]
        assert "prompt" not in call, "the prompt column is `prompts`, not a column"
    assert answers == ["mat", "park", "dog"]
    assert [call["difficulty"][0] for call in reward.calls] == [1, 2, 3]  # type: ignore[index]


def test_functions_are_weighted_and_summed_with_the_specs_rewards(tiny_model, prompts, tmp_path):
    first, second = Recorder(), Recorder(2.0)
    pmetal.GrpoTrainer(
        spec_for(tiny_model, prompts, tmp_path / "out", reward_regex="."),
        [first, second],
        reward_weights=[0.5, 2.0],
    ).train()
    assert len(first.calls) == len(second.calls) == 3


def test_the_function_form_is_the_same_run(tiny_model, prompts, tmp_path):
    reward = Recorder()
    result = pmetal.grpo(
        str(tiny_model),
        str(prompts),
        reward_funcs=[reward],
        output_dir=str(tmp_path / "out"),
        num_generations=2,
        max_completion_length=4,
        max_seq_len=128,
        lora_r=4,
        checkpoint_every=0,
    )
    assert result.total_steps == 3 and len(reward.calls) == 3


def test_the_trainer_checks_its_arguments_before_loading_anything(tiny_model, prompts, tmp_path):
    spec = spec_for(tiny_model, prompts, tmp_path)
    with pytest.raises(TypeError, match="callable"):
        pmetal.GrpoTrainer(spec, ["not a function"])  # type: ignore[list-item]
    with pytest.raises(pmetal.ConfigError, match="one weight per function"):
        pmetal.GrpoTrainer(spec, [Recorder()], reward_weights=[1.0, 2.0])
    with pytest.raises(pmetal.ConfigError, match="not finite"):
        pmetal.GrpoTrainer(spec, [Recorder()], reward_weights=[float("nan")])
    with pytest.raises(pmetal.SpecValidationError):
        pmetal.GrpoTrainer(pmetal.GrpoSpec(), [Recorder()])
    trainer = pmetal.GrpoTrainer(spec, [Recorder()])
    assert trainer.spec == spec and "GrpoTrainer(" in repr(trainer)


# -- a function that fails fails the run with a typed error ---------------------------------------


def test_an_exception_in_a_reward_function_is_a_reward_error_with_it_as_the_cause(tiny_model, prompts, tmp_path):
    calls = 0

    def reward(prompts: list[str], completions: list[str], **columns: object) -> list[float]:
        nonlocal calls
        calls += 1
        raise ZeroDivisionError("the verifier divided by zero")

    with pytest.raises(pmetal.RewardError, match="reward") as caught:
        pmetal.GrpoTrainer(spec_for(tiny_model, prompts, tmp_path / "out"), [reward]).train()
    assert isinstance(caught.value, pmetal.TrainingError) and isinstance(caught.value, pmetal.PMetalError)
    assert isinstance(caught.value.__cause__, ZeroDivisionError)
    assert "ZeroDivisionError" in str(caught.value) and "divided by zero" in str(caught.value)
    assert calls == 1, "the run ends at the first failure"


@pytest.mark.parametrize(
    ("returns", "message"),
    [
        (lambda n: [1.0, None][:n], "None for completion 1"),
        (lambda n: [1.0] * (n + 1), "3 rewards for 2 completions"),
        (lambda n: ["high"] * n, "not a number"),
        (lambda n: 1.0, "not iterable|not a number|object is not iterable"),
    ],
)
def test_a_result_that_is_not_one_number_per_completion_is_refused(tiny_model, prompts, tmp_path, returns, message):
    def reward(prompts: list[str], completions: list[str], **columns: object) -> object:
        return returns(len(completions))

    with pytest.raises(pmetal.RewardError, match=message) as caught:
        pmetal.GrpoTrainer(spec_for(tiny_model, prompts, tmp_path / "out"), [reward]).train()
    assert caught.value.__cause__ is not None


def test_ctrl_c_inside_a_reward_function_is_a_keyboard_interrupt(tiny_model, prompts, tmp_path):
    def reward(prompts: list[str], completions: list[str], **columns: object) -> list[float]:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        pmetal.GrpoTrainer(spec_for(tiny_model, prompts, tmp_path / "out"), [reward]).train()


def test_a_run_with_nothing_to_score_by_says_so(tiny_model, tmp_path):
    bare = tmp_path / "bare.jsonl"
    bare.write_text(json.dumps({"prompt": "the cat sat on"}) + "\n")
    with pytest.raises(pmetal.TrainingError, match="needs a reward"):
        pmetal.GrpoTrainer(spec_for(tiny_model, bare, tmp_path / "out")).train()


# -- threads and the GIL --------------------------------------------------------------------------


def test_the_gil_is_free_while_the_run_trains_and_taken_for_each_call(tiny_model, prompts, tmp_path):
    ticks = 0
    stop = threading.Event()

    def tick() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            stop.wait(0.001)

    reward = Recorder()
    thread = threading.Thread(target=tick)
    thread.start()
    try:
        pmetal.GrpoTrainer(spec_for(tiny_model, prompts, tmp_path / "out"), [reward]).train()
    finally:
        stop.set()
        thread.join()
    assert len(reward.calls) == 3
    assert ticks > 5, "another Python thread made progress during the run"


def test_with_async_rewards_a_function_runs_on_the_scorer_thread(tiny_model, prompts, tmp_path):
    reward = Recorder()
    result = pmetal.GrpoTrainer(spec_for(tiny_model, prompts, tmp_path / "out", async_rewards=True), [reward]).train()
    assert result.total_steps >= 2 and len(reward.calls) >= 2
    assert threading.get_ident() not in reward.threads, "scored off the calling thread, GIL reacquired there"


# -- the spec's own rewards ------------------------------------------------------------------------


def test_the_spec_carries_reward_commands_as_repeated_flags(tmp_path):
    spec = pmetal.GrpoSpec(
        model="m",
        dataset="d",
        reward_cmd=["python3 judge.py --strict", "./tests.sh"],
        reward_weight=["0.5", "2"],
        reward_regex=r"^\d+$",
        reward_json_schema="schema.json",
    )
    argv = spec.to_argv()
    assert argv.count("--reward-cmd") == 2 and argv.count("--reward-weight") == 2
    assert argv[argv.index("--reward-cmd") + 1] == "python3 judge.py --strict"
    spec.normalize()
    bad = pmetal.GrpoSpec(model="m", dataset="d", reward_cmd=["a"], reward_weight=["heavy"])
    with pytest.raises(pmetal.SpecValidationError) as caught:
        bad.normalize()
    assert ("reward_weight", "`heavy` is not a number") in caught.value.errors


def test_a_reward_command_and_a_python_function_score_the_same_run(tiny_model, prompts, tmp_path):
    script = tmp_path / "reward.py"
    log = tmp_path / "requests.jsonl"
    script.write_text(
        textwrap.dedent(
            f"""\
            import json, sys
            for line in sys.stdin:
                request = json.loads(line)
                with open({str(log)!r}, "a") as f:
                    f.write(json.dumps(sorted(request)) + "\\n")
                print(json.dumps({{"rewards": [float(i) for i in range(len(request["completions"]))]}}), flush=True)
            """
        )
    )
    reward = Recorder()
    result = pmetal.GrpoTrainer(
        spec_for(
            tiny_model,
            prompts,
            tmp_path / "out",
            reward_cmd=[shlex.join([sys.executable, str(script)])],
            reward_weight=["0.5"],
            reward_regex=".",
        ),
        [reward],
    ).train()
    assert result.total_steps == 3 and len(reward.calls) == 3
    keys = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(keys) == 3
    assert all(k == ["answer", "completions", "difficulty", "prompts"] for k in keys), keys


def test_a_reward_command_that_fails_fails_the_run_with_what_it_said(tiny_model, prompts, tmp_path):
    script = tmp_path / "broken.py"
    script.write_text("import sys\nsys.stdin.readline()\nprint('the judge is down', file=sys.stderr)\nsys.exit(4)\n")
    spec = spec_for(tiny_model, prompts, tmp_path / "out", reward_cmd=[shlex.join([sys.executable, str(script)])])
    with pytest.raises(pmetal.TrainingError, match="the judge is down"):
        pmetal.GrpoTrainer(spec).train()
