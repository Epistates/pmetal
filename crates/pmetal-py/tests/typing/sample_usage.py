"""Sample usage, type-checked in strict mode by ``tests/test_stubs.py``.

It is never run (nothing here loads a model): it exists so the type checker
reads every part of the public API the way a user writes it.
"""

# pyright: strict
from __future__ import annotations

from pathlib import Path

import pmetal


def specs() -> None:
    spec = pmetal.TrainSpec(model="Qwen/Qwen3-0.6B", dataset="train.jsonl", epochs=2, lora_r=8)
    epochs: int = spec.epochs
    lr: float = spec.learning_rate
    eval_set: str | None = spec.eval_dataset
    targets: list[str] = spec.lora_targets
    spec.optimizer = "muon"
    spec.warmup_ratio = None
    spec.output_dir = str(Path("out"))
    wsd: str | None = spec.wsd_decay_style
    other: pmetal.TrainSpec = spec.copy(epochs=3, seed=1)
    problems: list[tuple[str, str]] = other.validation_errors()
    spec.normalize()
    argv: list[str] = spec.to_argv()
    as_dict: dict[str, object] = spec.to_dict(exclude_defaults=True)
    again: pmetal.TrainSpec = pmetal.TrainSpec.from_dict(as_dict)
    from_json: pmetal.TrainSpec = pmetal.TrainSpec.from_json(spec.to_json())
    fields: list[pmetal.FieldInfo] = pmetal.TrainSpec.fields()
    command: str = pmetal.TrainSpec.subcommand
    lora: dict[str, object] = spec.lora_config()
    training: dict[str, object] = spec.training_config()
    print(epochs, lr, eval_set, targets, wsd, problems, argv, again, from_json, fields, command, lora, training)

    pmetal.GrpoSpec(model="m", dataset="d")
    pmetal.MergeSpec()
    pmetal.ServeSpec()
    pmetal.QuantizeSpec()
    pmetal.DflashSpec(max_new_tokens=64)


def training() -> None:
    class Collect:
        def __init__(self) -> None:
            self.losses: list[float] = []

        def on_step_end(self, step: int, loss: float) -> None:
            self.losses.append(loss)

    spec = pmetal.TrainSpec(model="m", dataset="d")
    trainer = pmetal.Trainer(spec, [pmetal.LoggingCallback(5), pmetal.ProgressCallback(100)])
    trainer.add_callback(Collect())
    trainer.add_callback(pmetal.MetricsJsonCallback(Path("metrics.jsonl")))
    result: pmetal.TrainResult = trainer.train()
    loss: float = result.final_loss
    weights: str = result.weights_path
    print(loss, weights, result.to_dict(), trainer.spec)

    result2 = pmetal.finetune("m", "d", epochs=1, lora_r=4, optimizer="adamw")
    print(result2.total_steps)


def inference() -> None:
    model = pmetal.Model.load("Qwen/Qwen3-0.6B", fp8=False)
    text: str = model.generate("Hello", max_tokens=16, temperature=0.0, seed=1, logit_bias={1: -100.0})
    config = pmetal.GenerationConfig(temperature=0.7, top_p=0.9)
    model.generate("Hello", config, max_tokens=8)
    for piece in model.stream("Once upon a time", max_tokens=32):
        print(piece, end="")
    with model.stream("x") as stream:
        first: str = next(stream)
        print(first)
    arch: pmetal.ModelArchitecture = model.architecture
    tokenizer: pmetal.Tokenizer = model.tokenizer
    ids: list[int] = tokenizer.encode(text)
    back: str = tokenizer.decode(ids)
    eos: int | None = tokenizer.eos_token_id
    print(arch == pmetal.ModelArchitecture.Qwen3, back, eos, len(tokenizer), model.path)

    gen = pmetal.DFlashGenerator("target", "draft")
    tokens, metrics = gen.generate(ids, max_new_tokens=32, temperature=0.7, seed=3)
    n: int = len(tokens)
    rate: object = metrics["acceptance_rate"]
    print(n, rate, gen.block_size)

    quick: str = pmetal.infer("m", "p", temperature=0.0, max_tokens=8, mtp=True)
    detailed: pmetal.InferResult = pmetal.infer_with_metrics("m", "p", seed=1)
    print(quick, detailed.text, detailed.num_generated, detailed.speculative)


def hub() -> None:
    def show(progress: pmetal.pmetal.TransferProgress) -> None:
        print(progress["bytes_done"], progress["total_bytes"])

    path: str = pmetal.download_model("org/name", revision="main", weights="gguf", gguf_quant="Q4_K_M", verify="sha256", progress=show)
    file: str = pmetal.download_file("org/name", "config.json", offline=True, cache_dir=Path("cache"))
    plan: pmetal.DownloadPlan = pmetal.plan_download("org/name")
    report: pmetal.ModelReport = pmetal.verify_model(path)
    if report and report.is_complete:
        print(report.weight_bytes, report.format, plan.files, file)
    found = pmetal.search_models("qwen3", limit=5, tags=["mlx"], sort="likes")
    print(found[0]["model_id"], found[0]["fit"])
    fit = pmetal.estimate_fit("org/name", context_length=8192, kv_cache_bits=8)
    print(fit["fit_level"], fit["weights_gb"])
    uploaded: pmetal.UploadReport = pmetal.upload_model("dir", "me/repo", private=True, exclude=["*.tmp"])
    print(uploaded.repo_url)


def jobs() -> None:
    spec = pmetal.MergeSpec()

    def on_event(event: dict[str, object]) -> None:
        print(event.get("event"))

    result: pmetal.JobResult = pmetal.run(spec, on_event=on_event, timeout=60.0, check=False)
    code: int = result.returncode
    summary = result.summary
    print(code, summary, result.json())
    with pmetal.start(spec, output="inherit") as job:
        for event in job:
            print(event)
        done = job.wait()
        print(done.stdout, job.pid, job.poll())
    cli: Path = pmetal.find_cli()
    print(cli)
    assert isinstance(spec, pmetal.JobSpec)


def errors() -> None:
    try:
        pmetal.TrainSpec().normalize()
    except pmetal.SpecValidationError as e:
        pairs: list[tuple[str, str]] = e.errors
        print(pairs)
    except pmetal.ConfigError as e:
        print(e)
    except pmetal.JobFailedError as e:
        code: int = e.returncode
        print(code, e.stderr)
    except pmetal.PMetalError as e:
        print(e)
