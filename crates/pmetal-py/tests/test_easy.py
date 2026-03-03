"""Tests for the PMetal easy API (top-level functions)."""

import os
import json
import tempfile

import pytest

import pmetal

SKIP_EASY = os.environ.get("PMETAL_TEST_EASY", "0") != "1"
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


@pytest.mark.skipif(SKIP_EASY, reason="Easy API tests require model (set PMETAL_TEST_EASY=1)")
def test_easy_finetune():
    """Test one-liner fine-tuning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "data.jsonl")
        create_sample_dataset(dataset_path)

        result = pmetal.finetune(
            TEST_MODEL,
            dataset_path,
            lora_r=8,
            lora_alpha=16.0,
            epochs=1,
            batch_size=1,
            max_seq_len=128,
            output=os.path.join(tmpdir, "output"),
        )

        assert "final_loss" in result
        assert "lora_weights_path" in result
        assert os.path.exists(result["lora_weights_path"])


@pytest.mark.skipif(SKIP_EASY, reason="Easy API tests require model (set PMETAL_TEST_EASY=1)")
def test_easy_infer():
    """Test one-liner inference."""
    text = pmetal.infer(TEST_MODEL, "What is 2+2?", max_tokens=20, temperature=0.0)
    assert isinstance(text, str)
    assert len(text) > 0
