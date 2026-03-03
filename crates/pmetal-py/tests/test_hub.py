"""Tests for PMetal hub operations.

These are integration tests that require network access.
Mark with skip if running in offline CI.
"""

import os

import pytest

import pmetal

# Skip hub tests in CI unless explicitly enabled
SKIP_HUB = os.environ.get("PMETAL_TEST_HUB", "0") != "1"


@pytest.mark.skipif(SKIP_HUB, reason="Hub tests require network (set PMETAL_TEST_HUB=1)")
def test_download_model():
    """Test downloading a small model from HuggingFace Hub."""
    path = pmetal.download_model("Qwen/Qwen3-0.6B")
    assert path
    assert os.path.isdir(path)


@pytest.mark.skipif(SKIP_HUB, reason="Hub tests require network (set PMETAL_TEST_HUB=1)")
def test_download_file():
    """Test downloading a specific file from HuggingFace Hub."""
    path = pmetal.download_file("Qwen/Qwen3-0.6B", "config.json")
    assert path
    assert os.path.isfile(path)
