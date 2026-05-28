from __future__ import annotations

from typing import Any, ClassVar, Sequence

__version__: str


class LoraConfig:
    def __init__(
        self,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        use_rslora: bool = False,
        use_dora: bool = False,
    ) -> None: ...
    @property
    def r(self) -> int: ...
    @property
    def alpha(self) -> float: ...
    @property
    def dropout(self) -> float: ...
    @property
    def use_rslora(self) -> bool: ...
    @property
    def use_dora(self) -> bool: ...
    @property
    def scaling(self) -> float: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> LoraConfig: ...


class TrainingConfig:
    def __init__(
        self,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        num_epochs: int = 3,
        max_seq_len: int = 2048,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        use_packing: bool = True,
        output_dir: str = "./output",
    ) -> None: ...
    @property
    def learning_rate(self) -> float: ...
    @property
    def batch_size(self) -> int: ...
    @property
    def num_epochs(self) -> int: ...
    @property
    def max_seq_len(self) -> int: ...
    @property
    def warmup_steps(self) -> int: ...
    @property
    def weight_decay(self) -> float: ...
    @property
    def max_grad_norm(self) -> float: ...
    @property
    def use_packing(self) -> bool: ...
    @property
    def output_dir(self) -> str: ...
    def to_json(self) -> str: ...
    @staticmethod
    def from_json(json: str) -> TrainingConfig: ...


class GenerationConfig:
    def __init__(
        self,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        min_p: float = 0.05,
        seed: int | None = None,
    ) -> None: ...
    @staticmethod
    def greedy(max_tokens: int) -> GenerationConfig: ...
    @staticmethod
    def sampling(max_tokens: int = 256, temperature: float = 0.7) -> GenerationConfig: ...
    @property
    def max_tokens(self) -> int: ...
    @property
    def temperature(self) -> float: ...
    @property
    def top_k(self) -> int: ...
    @property
    def top_p(self) -> float: ...
    @property
    def min_p(self) -> float: ...
    @property
    def seed(self) -> int | None: ...


class DataLoaderConfig:
    def __init__(
        self,
        batch_size: int = 4,
        max_seq_len: int = 2048,
        shuffle: bool = True,
        seed: int = 42,
        pad_token_id: int = 0,
        drop_last: bool = False,
    ) -> None: ...
    @property
    def batch_size(self) -> int: ...
    @property
    def max_seq_len(self) -> int: ...
    @property
    def shuffle(self) -> bool: ...
    @property
    def seed(self) -> int: ...
    @property
    def pad_token_id(self) -> int: ...
    @property
    def drop_last(self) -> bool: ...


class Dtype:
    Float32: ClassVar[Dtype]
    Float16: ClassVar[Dtype]
    BFloat16: ClassVar[Dtype]
    Float8E4M3: ClassVar[Dtype]
    Float8E5M2: ClassVar[Dtype]
    Int32: ClassVar[Dtype]
    Int64: ClassVar[Dtype]
    UInt8: ClassVar[Dtype]
    Bool: ClassVar[Dtype]


class Quantization:
    NF4: ClassVar[Quantization]
    FP4: ClassVar[Quantization]
    Int8: ClassVar[Quantization]
    FP8: ClassVar[Quantization]


class LoraBias:
    All: ClassVar[LoraBias]
    LoraOnly: ClassVar[LoraBias]


class LrSchedulerType:
    Constant: ClassVar[LrSchedulerType]
    Linear: ClassVar[LrSchedulerType]
    Cosine: ClassVar[LrSchedulerType]
    CosineWithRestarts: ClassVar[LrSchedulerType]
    Polynomial: ClassVar[LrSchedulerType]
    Wsd: ClassVar[LrSchedulerType]


class OptimizerType:
    AdamW: ClassVar[OptimizerType]
    Sgd: ClassVar[OptimizerType]
    Adafactor: ClassVar[OptimizerType]
    Lion: ClassVar[OptimizerType]


class DatasetFormat:
    Simple: ClassVar[DatasetFormat]
    Alpaca: ClassVar[DatasetFormat]
    ShareGpt: ClassVar[DatasetFormat]
    OpenAi: ClassVar[DatasetFormat]
    Auto: ClassVar[DatasetFormat]


class ModelArchitecture:
    Llama: ClassVar[ModelArchitecture]
    Llama4: ClassVar[ModelArchitecture]
    Qwen2: ClassVar[ModelArchitecture]
    Qwen3: ClassVar[ModelArchitecture]
    Qwen3MoE: ClassVar[ModelArchitecture]
    Gemma: ClassVar[ModelArchitecture]
    Mistral: ClassVar[ModelArchitecture]
    Phi: ClassVar[ModelArchitecture]
    Phi4: ClassVar[ModelArchitecture]
    DeepSeek: ClassVar[ModelArchitecture]
    Cohere: ClassVar[ModelArchitecture]
    Granite: ClassVar[ModelArchitecture]
    NemotronH: ClassVar[ModelArchitecture]
    Qwen3Next: ClassVar[ModelArchitecture]
    GptOss: ClassVar[ModelArchitecture]
    Gemma4: ClassVar[ModelArchitecture]
    Bert: ClassVar[ModelArchitecture]
    Flux: ClassVar[ModelArchitecture]


class Model:
    @staticmethod
    def load(path_or_id: str, fp8: bool = False) -> Model: ...
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        seed: int | None = None,
    ) -> str: ...
    def architecture(self) -> ModelArchitecture: ...


class DFlashGenerator:
    def __init__(self, target_model: str, draft_model: str) -> None: ...
    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        stop_tokens: Sequence[int] | None = None,
        speculative_tokens: int | None = None,
    ) -> tuple[list[int], dict[str, Any]]: ...


class Tokenizer:
    @staticmethod
    def from_file(path: str) -> Tokenizer: ...
    @staticmethod
    def from_pretrained(model_id: str) -> Tokenizer: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def pad_token_id(self) -> int | None: ...
    @property
    def eos_token_id(self) -> int | None: ...
    @property
    def bos_token_id(self) -> int | None: ...
    @property
    def unk_token_id(self) -> int | None: ...


class Trainer:
    def __init__(
        self,
        model_id: str,
        lora_config: LoraConfig,
        training_config: TrainingConfig,
        dataset_path: str,
        eval_dataset_path: str | None = None,
    ) -> None: ...
    def add_callback(self, callback: Any) -> None: ...
    def set_sequence_packing(self, enabled: bool) -> None: ...
    def set_gradient_checkpointing(self, enabled: bool) -> None: ...
    def set_metal_fused_optimizer(self, enabled: bool) -> None: ...
    def set_embedding_lr(self, lr: float) -> None: ...
    def train(self) -> dict[str, Any]: ...


class ProgressCallback:
    def __init__(self, total_steps: int) -> None: ...


class LoggingCallback:
    def __init__(self, log_every: int = 10) -> None: ...


class MetricsJsonCallback:
    def __init__(self, path: str) -> None: ...


def download_model(model_id: str, revision: str | None = None) -> str: ...
def download_file(model_id: str, filename: str, revision: str | None = None) -> str: ...


def finetune(
    model_id: str,
    dataset_path: str,
    lora_r: int = 16,
    lora_alpha: float = 32.0,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    max_seq_len: int = 2048,
    output: str = "./output",
) -> dict[str, Any]: ...


def infer(
    model_id: str,
    prompt: str,
    lora: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
    seed: int | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    repetition_penalty: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    fp8: bool = False,
    experts_dir: str | None = None,
    draft_model: str | None = None,
    mtp: bool = False,
    mtp_model: str | None = None,
    mtp_draft_tokens: int = 3,
    chat: bool = False,
    system_message: str | None = None,
    no_thinking: bool = False,
    kv_quant: int | None = None,
    no_kv_quant: bool = False,
    detect_repetition: bool = False,
) -> str: ...


def infer_with_metrics(
    model_id: str,
    prompt: str,
    lora: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
    seed: int | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    repetition_penalty: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    fp8: bool = False,
    experts_dir: str | None = None,
    draft_model: str | None = None,
    mtp: bool = False,
    mtp_model: str | None = None,
    mtp_draft_tokens: int = 3,
    chat: bool = False,
    system_message: str | None = None,
    no_thinking: bool = False,
    kv_quant: int | None = None,
    no_kv_quant: bool = False,
    detect_repetition: bool = False,
) -> dict[str, Any]: ...


def train_mtp(
    model: str,
    shards: Sequence[str],
    output: str = "./mtp-output",
    family: str = "auto",
    init_mtp: str | None = None,
    seq_len: int = 512,
    batch_size: int = 1,
    steps: int = 1000,
    learning_rate: float = 2e-4,
    min_lr: float = 1e-5,
    warmup_steps: int = 100,
    lr_schedule: str = "cosine",
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    checkpoint_every: int = 500,
    log_every: int = 10,
    eos_token_id: int = 0,
    mtp_layers: int = 1,
    assistant_layers: int = 2,
    num_assistant_tokens: int = 6,
    seed: int = 42,
) -> dict[str, Any]: ...


def train_draft(
    target: str,
    shards: Sequence[str],
    output: str = "./draft-output",
    draft: str | None = None,
    draft_config: str | None = None,
    seq_len: int = 512,
    batch_size: int = 1,
    steps: int = 1000,
    learning_rate: float = 2e-4,
    min_lr: float = 1e-5,
    warmup_steps: int = 100,
    lr_schedule: str = "cosine",
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    checkpoint_every: int = 500,
    log_every: int = 10,
    eos_token_id: int = 0,
    seed: int = 42,
) -> dict[str, Any]: ...
