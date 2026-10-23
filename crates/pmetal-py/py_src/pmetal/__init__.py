"""PMetal: fine-tuning and inference for Apple Silicon.

Every job the ``pmetal`` command line does is a *spec* class here
(``TrainSpec``, ``GrpoSpec``, ``MergeSpec``, ...): the same fields, defaults and
checks as the command's flags. A spec is data. Run it:

* ``Trainer(TrainSpec(...)).train()`` fine-tunes in this process, with Python
  callbacks and Ctrl-C;
* ``grpo(model, dataset, reward_funcs=[...])`` runs GRPO in this process with
  Python reward functions;
* ``run(spec)`` runs the ``pmetal`` command for any spec as a subprocess;
* ``Model.load(...)``, ``infer(...)`` and ``DFlashGenerator`` generate.

>>> import pmetal
>>> spec = pmetal.TrainSpec(model="Qwen/Qwen3-0.6B", dataset="train.jsonl", epochs=1)  # doctest: +SKIP
>>> result = pmetal.Trainer(spec).train()  # doctest: +SKIP
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from ._jobs import (
    CliNotFoundError as CliNotFoundError,
    Job as Job,
    JobError as JobError,
    JobFailedError as JobFailedError,
    JobResult as JobResult,
    JobSpec as JobSpec,
    find_cli as find_cli,
    run as run,
    start as start,
)
from .pmetal import (
    INFER_COMMAND_LINE_ONLY as INFER_COMMAND_LINE_ONLY,
    UNWINDS_ON_PANIC as UNWINDS_ON_PANIC,
    BackendError as BackendError,
    BenchSpec as BenchSpec,
    CancelledError as CancelledError,
    ConfigError as ConfigError,
    DecideSpec as DecideSpec,
    DFlashGenerator as DFlashGenerator,
    DflashSpec as DflashSpec,
    DistillSpec as DistillSpec,
    DownloadPlan as DownloadPlan,
    EmbedTrainSpec as EmbedTrainSpec,
    EvalSpec as EvalSpec,
    FieldInfo as FieldInfo,
    FuseSpec as FuseSpec,
    GenerationConfig as GenerationConfig,
    GrpoResult as GrpoResult,
    GrpoSpec as GrpoSpec,
    GrpoTrainer as GrpoTrainer,
    HubError as HubError,
    ImatrixSpec as ImatrixSpec,
    InferResult as InferResult,
    InferSpec as InferSpec,
    LoggingCallback as LoggingCallback,
    MergeSpec as MergeSpec,
    MetricsJsonCallback as MetricsJsonCallback,
    Model as Model,
    ModelArchitecture as ModelArchitecture,
    ModelLoadError as ModelLoadError,
    ModelReport as ModelReport,
    OutOfMemoryError as OutOfMemoryError,
    PackExpertsSpec as PackExpertsSpec,
    PMetalError as PMetalError,
    PreferenceSpec as PreferenceSpec,
    PretrainSpec as PretrainSpec,
    ProgressCallback as ProgressCallback,
    QuantizationError as QuantizationError,
    QuantizeSpec as QuantizeSpec,
    RewardError as RewardError,
    RlkdSpec as RlkdSpec,
    SerializationError as SerializationError,
    ServeSpec as ServeSpec,
    SpecValidationError as SpecValidationError,
    TextStream as TextStream,
    Tokenizer as Tokenizer,
    TokenizerError as TokenizerError,
    TokenizeSpec as TokenizeSpec,
    TrainDiffusionSpec as TrainDiffusionSpec,
    TrainDraftSpec as TrainDraftSpec,
    Trainer as Trainer,
    TrainingError as TrainingError,
    TrainMtpSpec as TrainMtpSpec,
    TrainResult as TrainResult,
    TrainSpec as TrainSpec,
    UnsupportedArchitectureError as UnsupportedArchitectureError,
    UploadReport as UploadReport,
    VocodeSpec as VocodeSpec,
    __version__ as __version__,
    configure_metallib as _configure_metallib,
    download_file as download_file,
    download_model as download_model,
    estimate_fit as estimate_fit,
    infer as infer,
    infer_with_metrics as infer_with_metrics,
    plan_download as plan_download,
    search_models as search_models,
    upload_model as upload_model,
    verify_model as verify_model,
)

# MLX loads its Metal kernels from the metallib built with it: point it at the
# one this package carries before anything touches the GPU.
_configure_metallib(Path(__file__).resolve().parent)

if not UNWINDS_ON_PANIC:
    warnings.warn(
        "this pmetal build aborts the whole process on a Rust panic instead of raising an exception "
        "(it was built with the release profile; `maturin build` and `maturin develop --profile wheel` "
        "build the unwinding one)",
        RuntimeWarning,
        stacklevel=2,
    )

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Unpack

    from .pmetal import GrpoCallOptions, TrainCallOptions


def finetune(model: str, dataset: str, **options: Unpack[TrainCallOptions]) -> TrainResult:
    """Fine-tune ``model`` on ``dataset`` in this process and return the result.

    ``options`` are the fields of :class:`TrainSpec` with its defaults
    (``TrainSpec.fields()`` lists them). This is
    ``Trainer(TrainSpec(model=model, dataset=dataset, **options)).train()``.

    >>> import pmetal
    >>> pmetal.finetune("Qwen/Qwen3-0.6B", "train.jsonl", epochs=1, lora_r=8)  # doctest: +SKIP
    """
    return Trainer(TrainSpec(model=model, dataset=dataset, **options)).train()


def grpo(
    model: str,
    dataset: str,
    *,
    reward_funcs: Sequence[Callable[..., Sequence[float]]] | None = None,
    reward_weights: Sequence[float] | None = None,
    callbacks: Sequence[object] | None = None,
    **options: Unpack[GrpoCallOptions],
) -> GrpoResult:
    """Train ``model`` with GRPO on the prompts of ``dataset``, in this process.

    ``reward_funcs`` are Python callables scoring completions,
    ``fn(prompts, completions, **columns) -> list[float]``: ``prompts`` and
    ``completions`` are lists of strings, one entry per completion, and every
    other column of the dataset's row arrives as a keyword argument, a list with
    one entry per completion. Return one number per completion; a completion a
    function cannot score is an error, so ``None`` is refused. ``reward_weights``
    gives each function's weight (1 each by default). They are summed with the
    rewards ``options`` ask for (``reward_cmd``, ``reward_regex``,
    ``reasoning_rewards``, ...). This is
    ``GrpoTrainer(GrpoSpec(model=model, dataset=dataset, **options), reward_funcs,
    reward_weights, callbacks).train()``.

    >>> import pmetal
    >>> def exact(prompts, completions, answer, **kwargs):
    ...     return [float(c.strip() == a) for c, a in zip(completions, answer)]
    >>> pmetal.grpo("Qwen/Qwen3-0.6B", "prompts.jsonl", reward_funcs=[exact])  # doctest: +SKIP
    """
    spec = GrpoSpec(model=model, dataset=dataset, **options)
    return GrpoTrainer(spec, reward_funcs, reward_weights, callbacks).train()
