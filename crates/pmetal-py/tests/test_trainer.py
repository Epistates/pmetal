"""Tests for PMetal trainer.

These tests require a model and dataset.
"""

import os
import json
import tempfile

import pytest

import pmetal

SKIP_TRAINING = os.environ.get("PMETAL_TEST_TRAINING", "0") != "1"
TEST_MODEL = os.environ.get("PMETAL_MODEL_PATH", "Qwen/Qwen3-0.6B")


def create_sample_dataset(path: str, n_samples: int = 10):
    """Create a minimal JSONL dataset for testing."""
    samples = [
        {"messages": [
            {"role": "user", "content": f"What is {i}+{i}?"},
            {"role": "assistant", "content": f"The answer is {i+i}."},
        ]}
        for i in range(n_samples)
    ]
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


@pytest.mark.skipif(SKIP_TRAINING, reason="Training tests require model (set PMETAL_TEST_TRAINING=1)")
def test_trainer_creation():
    """Test creating a trainer instance."""
    lora_config = pmetal.LoraConfig(r=8, alpha=16.0)
    training_config = pmetal.TrainingConfig(
        learning_rate=1e-4,
        batch_size=1,
        num_epochs=1,
        max_seq_len=128,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "data.jsonl")
        create_sample_dataset(dataset_path)
        trainer = pmetal.Trainer(TEST_MODEL, lora_config, training_config, dataset_path)
        assert repr(trainer).startswith("Trainer(")


@pytest.mark.skipif(SKIP_TRAINING, reason="Training tests require model (set PMETAL_TEST_TRAINING=1)")
def test_trainer_train_1_step():
    """Test running 1 step of training."""
    lora_config = pmetal.LoraConfig(r=8, alpha=16.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "data.jsonl")
        create_sample_dataset(dataset_path)

        training_config = pmetal.TrainingConfig(
            learning_rate=1e-4,
            batch_size=1,
            num_epochs=1,
            max_seq_len=128,
            output_dir=os.path.join(tmpdir, "output"),
        )

        trainer = pmetal.Trainer(TEST_MODEL, lora_config, training_config, dataset_path)
        result = trainer.train()

        assert "final_loss" in result
        assert "total_steps" in result
        assert "total_tokens" in result
        assert "output_dir" in result
        assert "lora_weights_path" in result
        assert result["total_steps"] > 0
