# pmetal (Python)

Fine-tuning and inference for Apple Silicon, from Python. The extension module
wraps the same code `pmetal` the command line runs; a job you can do there, you
can do here with the same fields, defaults and checks.

- macOS 14 or later, Apple Silicon (arm64) only. Python 3.11 or later.
- One abi3 wheel covers CPython 3.11 and later.
- Fully typed (`py.typed`, stubs checked with mypy and pyright in strict mode).
- Long calls (training, generation, downloads) release the GIL, stop cleanly on
  Ctrl-C, and report failures as typed exceptions.

## Install

```bash
pip install pmetal
```

From a checkout, with [maturin](https://www.maturin.rs):

```bash
cd crates/pmetal-py
pip install maturin
maturin develop --profile wheel
```

## The job specs

Every `pmetal` subcommand has a spec class: `TrainSpec`, `GrpoSpec`,
`PreferenceSpec`, `DistillSpec`, `RlkdSpec`, `EmbedTrainSpec`, `PretrainSpec`,
`QuantizeSpec`, `MergeSpec`, `FuseSpec`, `ServeSpec`, `InferSpec`, `EvalSpec`,
`DflashSpec`, `TrainMtpSpec`, `TrainDraftSpec`, `TrainDiffusionSpec`, `DecideSpec`,
`TokenizeSpec`, `VocodeSpec`, `PackExpertsSpec` and `BenchSpec`. They wrap the
job specs of `pmetal-core`, which define every field, default, bound and check
for the command line, the TUI, the GUI, the MCP server and Python alike: there
is no second list of options here to drift.

```python
import pmetal

spec = pmetal.TrainSpec(model="Qwen/Qwen3-0.6B", dataset="train.jsonl", epochs=2, lora_r=16)
spec.learning_rate = 1e-4
spec.optimizer = "muon"  # a name outside the spec's options fails type checking

spec.validation_errors()  # [] when valid, else [(field, message), ...]
spec.normalize()  # raises SpecValidationError listing every problem
spec.to_argv()  # the arguments of `pmetal train`
pmetal.TrainSpec.fields()  # every field: type, default, bounds, flag, help
spec.copy(epochs=3)  # a changed copy
```

A spec is a mutable value: it compares by fields, round-trips through `to_dict`
and `to_json`, copies and pickles.

## Training

`Trainer` runs a `TrainSpec` in this process, the run `pmetal train` makes:

```python
trainer = pmetal.Trainer(spec, callbacks=[pmetal.LoggingCallback(10)])
result = trainer.train()  # releases the GIL; Ctrl-C stops after the step
print(result.final_loss, result.weights_path)
```

`pmetal.finetune("Qwen/Qwen3-0.6B", "train.jsonl", epochs=1)` is the one-line
form. Callbacks are any object with some of these methods:

| Method | Called |
|---|---|
| `on_train_start()` | when the model and data are ready |
| `on_step_end(step, loss)`, `on_step_metrics(metrics)` | after each optimizer step the run logs: the first, then every `logging_steps` |
| `on_train_end()` | when the run is over, however it ended |
| `should_stop() -> bool` | after every optimizer step; true ends the run cleanly |

An exception in a callback, like Ctrl-C, ends the run after the current step
with the weights saved, and is raised from `train()`.

## Every other command

`run` and `start` run the `pmetal` command a spec stands for, as a subprocess:
the command line's own behaviour, with a process of its own to fail in.

```python
pmetal.run(pmetal.MergeSpec(model_a="a", model_b="b", output="merged"))

job = pmetal.start(
    pmetal.GrpoSpec(model="Qwen/Qwen3-0.6B", dataset="prompts.jsonl"), on_event=lambda e: print(e["event"])
)
for event in job:  # events as the job writes them
    ...
job.wait()  # or job.cancel()
```

The command line is found from `cli=`, `PMETAL_CLI`, the one shipped in the
package, then `PATH`. Failures are `JobFailedError`.

## Inference

```python
model = pmetal.Model.load("Qwen/Qwen3-0.6B")  # lives on its own thread; share it freely
model.generate("The capital of France is", max_tokens=16, temperature=0.0)
for piece in model.stream("Once upon a time", max_tokens=64):
    print(piece, end="", flush=True)

pmetal.infer("Qwen/Qwen3-0.6B", "What is 2+2?", chat=True, no_thinking=True)  # InferSpec fields
gen = pmetal.DFlashGenerator("mlx-community/Qwen3-4B-bf16", "z-lab/Qwen3-4B-DFlash-b16")
tokens, metrics = gen.generate(prompt_ids, max_new_tokens=128, temperature=0.7, seed=1)
```

Sampling starts from the model's own `generation_config.json`; what you pass
overrides it. Temperature 0 or less is greedy.

## The Hub

`download_model` (revision, token, GGUF quantization, verification, offline mode,
progress), `download_file`, `plan_download`, `verify_model`, `search_models`
(with how each result fits this Mac), `estimate_fit` and `upload_model`.

```python
path = pmetal.download_model("Qwen/Qwen3-0.6B", revision="main", progress=print)
pmetal.estimate_fit(path, context_length=8192, kv_cache_bits=8)
```

## Errors

Everything pmetal raises is a `pmetal.PMetalError`: `ConfigError` (a
`ValueError`), `SpecValidationError`, `HubError`, `ModelLoadError`,
`TokenizerError`, `TrainingError`, `QuantizationError`, `BackendError`,
`OutOfMemoryError` (a `MemoryError`), `CancelledError`, and `JobError`
(`JobFailedError`, `CliNotFoundError`). I/O failures are the matching `OSError`
subclass. A Rust panic is `pyo3_runtime.PanicException`, not a crash.

## Threads

The module declares itself safe without the GIL. `Model` and `DFlashGenerator`
keep their weights on a thread of their own, so any Python thread may call them
(one generation runs at a time). Callbacks run on the thread that called
`train()`.

## Development

```bash
python stub.py            # regenerate py_src/pmetal/pmetal.pyi after a spec changes
python -m pytest tests    # about 30 seconds; no download needed
```

The suite writes a tiny random checkpoint for generation and training tests and
serves a fake Hub on localhost. Tests that need a real model or the network are
skipped unless `PMETAL_TEST_MODEL_DIR` (a local model directory) or
`PMETAL_TEST_HUB=1` is set. `python -m pytest tests/test_stubs.py` runs mypy,
pyright and `mypy.stubtest` against the stubs.

The `test-hooks` cargo feature adds functions that panic on purpose, for the
tests that check a panic is an exception; it is never in a published wheel.

### Building a wheel

```bash
just build-wheel       # what CI runs; the wheel lands in target/wheels
```

or, by hand, `PMETAL_MLX_STATIC=1 MACOSX_DEPLOYMENT_TARGET=14.0 maturin build`.

Do not pass `--release` to maturin, here or to `maturin develop`: it overrides
the `wheel` profile in `pyproject.toml` with the workspace's `release` one, which
aborts the interpreter on a Rust panic instead of raising `PanicException`
(`import pmetal` warns, and `pmetal.UNWINDS_ON_PANIC` says which you have). The
`wheel` profile is release optimization with `panic = "unwind"`. MLX is linked
statically (`PMETAL_MLX_STATIC=1`) so the wheel runs where it was not built, the
deployment target gives the wheel its `macosx_14_0` tag, and the `mlx.metallib`
built with that MLX and the `pmetal` command line go in `py_src/pmetal/` before
the build, which `just build-wheel` does.

## License

MIT OR Apache-2.0
