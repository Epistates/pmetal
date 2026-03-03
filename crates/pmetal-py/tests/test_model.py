"""Tests for PMetal model loading and inference.

These tests require a model to be available locally or downloadable.
"""

import os

import pytest

import pmetal

SKIP_MODEL = os.environ.get("PMETAL_TEST_MODEL", "0") != "1"
TEST_MODEL = os.environ.get("PMETAL_MODEL_PATH", "Qwen/Qwen3-0.6B")


@pytest.mark.skipif(SKIP_MODEL, reason="Model tests require model (set PMETAL_TEST_MODEL=1)")
def test_model_load():
    """Test loading a model."""
    model = pmetal.Model.load(TEST_MODEL)
    arch = model.architecture()
    assert arch is not None
    assert repr(model).startswith("Model(")


@pytest.mark.skipif(SKIP_MODEL, reason="Model tests require model (set PMETAL_TEST_MODEL=1)")
def test_model_generate():
    """Test basic text generation."""
    model = pmetal.Model.load(TEST_MODEL)
    text = model.generate("Hello", max_tokens=10, temperature=0.0)
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.skipif(SKIP_MODEL, reason="Model tests require model (set PMETAL_TEST_MODEL=1)")
def test_model_generate_with_sampling():
    """Test text generation with sampling."""
    model = pmetal.Model.load(TEST_MODEL)
    text = model.generate(
        "The capital of France is",
        max_tokens=20,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        seed=42,
    )
    assert isinstance(text, str)
