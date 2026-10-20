"""PMetal: fine-tuning and inference for Apple Silicon.

Every job the ``pmetal`` command line does is a *spec* class here
(``TrainSpec``, ``GrpoSpec``, ``MergeSpec``, ...): the same fields, defaults and
checks as the command's flags. A spec is data. Run it:

* ``Trainer(TrainSpec(...)).train()`` fine-tunes in this process, with Python
  callbacks and Ctrl-C;
* ``run(spec)`` runs the ``pmetal`` command for any spec as a subprocess;
* ``Model.load(...)``, ``infer(...)`` and ``DFlashGenerator`` generate.

>>> import pmetal
>>> spec = pmetal.TrainSpec(model="Qwen/Qwen3-0.6B", dataset="train.jsonl", epochs=1)  # doctest: +SKIP
>>> result = pmetal.Trainer(spec).train()  # doctest: +SKIP
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from . import pmetal as _native

# MLX loads its Metal kernels from the metallib built with it: point it at the
# one this package carries before anything touches the GPU.
_native.configure_metallib(Path(__file__).resolve().parent)

from ._jobs import (  # noqa: E402
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
from .pmetal import (  # noqa: E402
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
    GrpoSpec as GrpoSpec,
    HubError as HubError,
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
    RlkdSpec as RlkdSpec,
    SerializationError as SerializationError,
    ServeSpec as ServeSpec,
    SpecValidationError as SpecValidationError,
    TextStream as TextStream,
    Tokenizer as Tokenizer,
    TokenizeSpec as TokenizeSpec,
    TokenizerError as TokenizerError,
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

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from .pmetal import TrainCallOptions


def finetune(model: str, dataset: str, **options: Unpack[TrainCallOptions]) -> TrainResult:
    """Fine-tune ``model`` on ``dataset`` in this process and return the result.

    ``options`` are the fields of :class:`TrainSpec` with its defaults
    (``TrainSpec.fields()`` lists them). This is
    ``Trainer(TrainSpec(model=model, dataset=dataset, **options)).train()``.

    >>> import pmetal
    >>> pmetal.finetune("Qwen/Qwen3-0.6B", "train.jsonl", epochs=1, lora_r=8)  # doctest: +SKIP
    """
    return Trainer(TrainSpec(model=model, dataset=dataset, **options)).train()
