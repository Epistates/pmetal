# Full Control

Advanced Python SDK usage: job specs, callbacks, running any `pmetal` command, and the Hub.

## The job specs

Every `pmetal` subcommand has a spec class (`TrainSpec`, `GrpoSpec`, `PreferenceSpec`, `DistillSpec`,
`RlkdSpec`, `EmbedTrainSpec`, `QuantizeSpec`, `MergeSpec`, `FuseSpec`, `ServeSpec`, `InferSpec`, and
the rest). They wrap the job specs of `pmetal-core`, the one definition of each command's fields,
defaults, bounds and checks that the command line, TUI, GUI, MCP server and Python share.

```python
import pmetal

spec = pmetal.TrainSpec(
    model="Qwen/Qwen3-0.6B",
    dataset="train.jsonl",
    epochs=2,
    lora_r=16,
    optimizer="muon",
    lr_schedule="wsd",
    wsd_decay_style="cosine",
)
spec.learning_rate = 1e-4

spec.validation_errors()   # [] when valid
spec.normalize()           # raises SpecValidationError listing every problem
spec.to_argv()             # the arguments of `pmetal train`
spec.lora_config()         # the adapter config the spec asks for, as a dict
spec.training_config()     # the optimisation settings, as a dict
pmetal.TrainSpec.fields()  # every field: type, default, bounds, flag, help
```

Misspelled fields and wrongly typed values are errors that name the field, and type checkers
reject them (`TrainSpec(epochz=3)`, `spec.optimizer = "sgdd"`).

## Training with callbacks

```python
class Collect:
    def __init__(self):
        self.losses = []

    def on_step_end(self, step, loss):
        self.losses.append(loss)

    def should_stop(self):
        return len(self.losses) > 1 and self.losses[-1] < 0.05

trainer = pmetal.Trainer(spec, callbacks=[Collect(), pmetal.LoggingCallback(10)])
result = trainer.train()
```

| Method | Called |
|--------|--------|
| `on_train_start()` | when the model and data are ready |
| `on_step_end(step, loss)`, `on_step_metrics(metrics)` | after each optimizer step the run logs: the first, then every `logging_steps` |
| `on_train_end()` | when the run is over, however it ended |
| `should_stop() -> bool` | after every optimizer step; true ends the run cleanly |

`pmetal.ProgressCallback(total_steps)`, `pmetal.LoggingCallback(log_every)` and
`pmetal.MetricsJsonCallback(path)` are the trainer's own. An exception in a callback, like Ctrl-C,
ends the run after the current step with the weights saved, and is raised from `train()`.

## GRPO with Python reward functions

`pmetal.grpo(model, dataset, reward_funcs=[...], reward_weights=[...], **GrpoSpec fields)` (or
`pmetal.GrpoTrainer(spec, reward_funcs, reward_weights, callbacks).train()`) runs `pmetal grpo` in
this process with Python callables as rewards:

```python
import json
import pmetal

def valid_json(prompts, completions, **columns):
    # prompts and completions are lists of strings, one entry per completion; every other
    # column of the dataset's row arrives as a keyword argument (a list, one entry per completion).
    def ok(text):
        try:
            json.loads(text)
            return 1.0
        except ValueError:
            return 0.0
    return [ok(c) for c in completions]

def matches_answer(prompts, completions, answer, **columns):
    return [float(a in c) for c, a in zip(completions, answer)]

result = pmetal.grpo(
    "Qwen/Qwen3-0.6B",
    "prompts.jsonl",
    reward_funcs=[valid_json, matches_answer],
    reward_weights=[0.5, 1.0],
    num_generations=8,
)
print(result.total_steps, result.output_dir)
```

The weights multiply each function's rewards before they are summed with each other and with the
spec's own (`reasoning_rewards`, `reward_cmd`, `reward_regex`, `reward_json_schema`, ...). The GIL
is released while the run trains and taken for each call (on the scorer's thread with
`async_rewards=True`). A completion a function cannot score is an error, not a zero: returning
`None`, the wrong number of rewards or a non-number raises `pmetal.RewardError` (a `TrainingError`),
as does an exception in the function, which is the error's `__cause__`. See
[`pmetal grpo`](/cli/grpo/#rewards) for the reward commands and verifiers the spec adds.

## Running any command

`pmetal.run(spec)` runs the `pmetal` command a spec stands for under the job supervisor the TUI, MCP
server and GUI share (its own process group, credentials redacted from its output, a stop that reaches
what the job spawned), so everything the command line does is reachable:

```python
pmetal.run(pmetal.MergeSpec(model_a="a", model_b="b", output="merged"))

job = pmetal.start(
    pmetal.GrpoSpec(model="Qwen/Qwen3-0.6B", dataset="prompts.jsonl"),
    on_event=lambda event: print(event["event"]),
)
for event in job:          # training commands stream their events
    ...
result = job.wait()        # JobFailedError if the command fails; job.cancel() to stop it
```

`job.cancel()` sends SIGTERM to the job's process group and SIGKILL after `stop_grace` seconds;
`pmetal train` finishes its step and saves the adapter first. The command line is found from `cli=`,
then `PMETAL_BIN`, then the copy shipped in the package, then `PATH`.

## Model loading and generation

```python
model = pmetal.Model.load("Qwen/Qwen3-0.6B")
output = model.generate(
    "Hello world",
    temperature=0.7,
    seed=42,
    logit_bias={tokenizer_id: -100.0},
)
```

`Model` keeps its weights on a thread of its own: it is safe to share between threads, generation
releases the GIL, and Ctrl-C stops it at the next token. `pmetal.GenerationConfig` bundles sampling
settings; what a config leaves unset takes the model's own default.

`pmetal.DFlashGenerator(target, draft).generate(prompt_ids, seed=...)` does block-diffusion
speculative decoding; its tokens are the target's own.

## Hub operations

```python
path = pmetal.download_model("Qwen/Qwen3-0.6B", revision="main", verify="size", progress=print)
pmetal.verify_model(path).is_complete
pmetal.plan_download("Qwen/Qwen3-0.6B").total_bytes
pmetal.search_models("qwen3 0.6b", limit=5)            # each result carries its fit on this Mac
pmetal.estimate_fit(path, context_length=8192, kv_cache_bits=8)
```

## Tokenizer

```python
tokenizer = pmetal.Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
ids = tokenizer.encode("Hello world")
text = tokenizer.decode(ids)
```

## Errors

Everything pmetal raises is a `pmetal.PMetalError`: `ConfigError` (a `ValueError`),
`SpecValidationError`, `HubError`, `ModelLoadError`, `TokenizerError`, `TrainingError`,
`BackendError`, `OutOfMemoryError` (a `MemoryError`), `CancelledError`, and `JobError`. I/O failures
are the matching `OSError` subclass.

## See Also

- [Quick Start](/python/quick-start/): the short form
- [Rust SDK](/sdk/advanced/): Rust equivalent
