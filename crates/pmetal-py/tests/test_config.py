"""Tests for PMetal config type wrappers."""

import json

import pmetal


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

def test_version():
    assert pmetal.__version__
    assert isinstance(pmetal.__version__, str)


# ---------------------------------------------------------------------------
# LoRA Configuration
# ---------------------------------------------------------------------------

def test_lora_config_defaults():
    config = pmetal.LoraConfig()
    assert config.r == 16
    assert config.alpha == 32.0
    assert config.dropout == 0.0
    assert config.use_rslora is False
    assert config.use_dora is False
    assert config.scaling == 2.0  # 32 / 16


def test_lora_config_custom():
    config = pmetal.LoraConfig(r=32, alpha=64.0, dropout=0.1, use_rslora=True)
    assert config.r == 32
    assert config.alpha == 64.0
    assert config.dropout == 0.1
    assert config.use_rslora is True
    assert config.use_dora is False


def test_lora_config_repr():
    config = pmetal.LoraConfig()
    r = repr(config)
    assert "LoraConfig" in r
    assert "r=16" in r
    assert "alpha=32" in r


def test_lora_config_json_roundtrip():
    config = pmetal.LoraConfig(r=8, alpha=16.0, dropout=0.05, use_rslora=True, use_dora=True)
    json_str = config.to_json()
    parsed = json.loads(json_str)
    assert parsed["r"] == 8
    assert parsed["alpha"] == 16.0

    restored = pmetal.LoraConfig.from_json(json_str)
    assert restored.r == 8
    assert restored.alpha == 16.0
    assert restored.dropout == 0.05
    assert restored.use_rslora is True
    assert restored.use_dora is True


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------

def test_training_config_defaults():
    config = pmetal.TrainingConfig()
    assert config.learning_rate == 2e-4
    assert config.batch_size == 4
    assert config.num_epochs == 3
    assert config.max_seq_len == 2048
    assert config.warmup_steps == 100
    assert config.weight_decay == 0.01
    assert config.max_grad_norm == 1.0
    assert config.use_packing is True
    assert config.output_dir == "./output"


def test_training_config_custom():
    config = pmetal.TrainingConfig(
        learning_rate=1e-5,
        batch_size=8,
        num_epochs=5,
        max_seq_len=4096,
        warmup_steps=200,
        weight_decay=0.1,
        max_grad_norm=0.5,
        output_dir="/tmp/test",
    )
    assert config.learning_rate == 1e-5
    assert config.batch_size == 8
    assert config.num_epochs == 5
    assert config.max_seq_len == 4096
    assert config.warmup_steps == 200
    assert config.weight_decay == 0.1
    assert config.max_grad_norm == 0.5
    assert config.output_dir == "/tmp/test"


def test_training_config_repr():
    config = pmetal.TrainingConfig()
    r = repr(config)
    assert "TrainingConfig" in r
    assert "warmup=" in r
    assert "output_dir=" in r


def test_training_config_json_roundtrip():
    config = pmetal.TrainingConfig(learning_rate=1e-3, batch_size=16)
    json_str = config.to_json()
    restored = pmetal.TrainingConfig.from_json(json_str)
    assert restored.learning_rate == 1e-3
    assert restored.batch_size == 16


# ---------------------------------------------------------------------------
# Generation Configuration
# ---------------------------------------------------------------------------

def test_generation_config_defaults():
    config = pmetal.GenerationConfig()
    assert config.max_tokens == 256
    assert config.temperature == 0.7
    assert config.top_k == 50
    assert config.top_p == 0.9
    assert config.min_p == 0.05


def test_generation_config_custom():
    config = pmetal.GenerationConfig(max_tokens=100, temperature=0.5, seed=42)
    assert config.max_tokens == 100
    assert config.temperature == 0.5
    assert config.seed == 42
    assert config.top_k == 50
    assert config.top_p == 0.9


def test_generation_config_greedy():
    config = pmetal.GenerationConfig.greedy(512)
    assert config.temperature == 0.0
    assert config.max_tokens == 512


def test_generation_config_sampling():
    config = pmetal.GenerationConfig.sampling(256, 0.8)
    assert config.temperature == 0.8
    assert config.max_tokens == 256


def test_generation_config_repr():
    config = pmetal.GenerationConfig(max_tokens=100)
    r = repr(config)
    assert "GenerationConfig" in r
    assert "max_tokens=100" in r
    assert "min_p=" in r


# ---------------------------------------------------------------------------
# DataLoader Configuration
# ---------------------------------------------------------------------------

def test_dataloader_config_defaults():
    config = pmetal.DataLoaderConfig()
    assert config.batch_size == 4
    assert config.max_seq_len == 2048
    assert config.shuffle is True
    assert config.seed == 42
    assert config.pad_token_id == 0
    assert config.drop_last is False


def test_dataloader_config_custom():
    config = pmetal.DataLoaderConfig(batch_size=16, max_seq_len=4096, pad_token_id=151643)
    assert config.batch_size == 16
    assert config.max_seq_len == 4096
    assert config.pad_token_id == 151643


def test_dataloader_config_repr():
    config = pmetal.DataLoaderConfig(pad_token_id=100)
    r = repr(config)
    assert "DataLoaderConfig" in r
    assert "pad_token_id=100" in r


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

def test_dtype_enum():
    # Check distinct variants exist and are not equal to each other
    assert pmetal.Dtype.Float32 != pmetal.Dtype.Float16
    assert pmetal.Dtype.BFloat16 != pmetal.Dtype.Float32
    # Verify all variants are accessible
    for name in ("Float32", "Float16", "BFloat16", "Float8E4M3", "Float8E5M2",
                 "Int32", "Int64", "UInt8", "Bool"):
        assert hasattr(pmetal.Dtype, name), f"Dtype.{name} missing"


def test_quantization_enum():
    # "None" is a reserved word in Python — verify PyO3 handles it
    assert hasattr(pmetal.Quantization, "None") or hasattr(pmetal.Quantization, "None_")
    for name in ("NF4", "FP4", "Int8", "FP8"):
        assert hasattr(pmetal.Quantization, name), f"Quantization.{name} missing"


def test_lora_bias_enum():
    for name in ("None", "All", "LoraOnly"):
        assert hasattr(pmetal.LoraBias, name) or hasattr(pmetal.LoraBias, f"{name}_"), \
            f"LoraBias.{name} missing"


def test_lr_scheduler_type_enum():
    for name in ("Constant", "Linear", "Cosine", "CosineWithRestarts", "Polynomial"):
        assert hasattr(pmetal.LrSchedulerType, name), f"LrSchedulerType.{name} missing"


def test_optimizer_type_enum():
    for name in ("AdamW", "Sgd", "Adafactor", "Lion"):
        assert hasattr(pmetal.OptimizerType, name), f"OptimizerType.{name} missing"


def test_dataset_format_enum():
    for name in ("Simple", "Alpaca", "ShareGpt", "OpenAi", "Auto"):
        assert hasattr(pmetal.DatasetFormat, name), f"DatasetFormat.{name} missing"


def test_model_architecture_enum():
    for name in ("Llama", "Llama4", "Qwen2", "Qwen3", "Qwen3MoE", "Gemma",
                 "Mistral", "Phi", "Phi4", "DeepSeek", "Cohere", "Granite",
                 "NemotronH", "StarCoder2", "RecurrentGemma", "Jamba", "Flux"):
        assert hasattr(pmetal.ModelArchitecture, name), f"ModelArchitecture.{name} missing"


# ---------------------------------------------------------------------------
# Callbacks (construction only — no model needed)
# ---------------------------------------------------------------------------

def test_progress_callback():
    cb = pmetal.ProgressCallback(100)
    r = repr(cb)
    assert "ProgressCallback" in r
    assert "100" in r


def test_logging_callback_default():
    cb = pmetal.LoggingCallback()
    r = repr(cb)
    assert "LoggingCallback" in r
    assert "10" in r  # default log_every=10


def test_logging_callback_custom():
    cb = pmetal.LoggingCallback(log_every=50)
    r = repr(cb)
    assert "50" in r


def test_metrics_json_callback():
    cb = pmetal.MetricsJsonCallback("/tmp/metrics.jsonl")
    r = repr(cb)
    assert "MetricsJsonCallback" in r
    assert "/tmp/metrics.jsonl" in r
