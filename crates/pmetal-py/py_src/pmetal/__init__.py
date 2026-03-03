"""PMetal: High-performance LLM fine-tuning for Apple Silicon.

>>> import pmetal
>>> print(pmetal.__version__)
"""

from .pmetal import *  # noqa: F401,F403

__all__ = [
    # Version
    "__version__",
    # Config types
    "LoraConfig",
    "TrainingConfig",
    "GenerationConfig",
    "DataLoaderConfig",
    # Enums
    "Dtype",
    "Quantization",
    "LoraBias",
    "LrSchedulerType",
    "OptimizerType",
    "DatasetFormat",
    "ModelArchitecture",
    # Hub
    "download_model",
    "download_file",
    # Model
    "Model",
    # Tokenizer
    "Tokenizer",
    # Trainer
    "Trainer",
    # Callbacks
    "ProgressCallback",
    "LoggingCallback",
    "MetricsJsonCallback",
    # Easy API
    "finetune",
    "infer",
]
