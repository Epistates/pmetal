"""Shared fixtures.

``tiny_model`` writes a complete, tiny, randomly initialised Qwen3 checkpoint
(config, weights, tokenizer) so generation and training run end to end in
seconds without a download. Its outputs are gibberish, which is fine: the tests
check determinism, plumbing and error handling, not language quality.

Tests that need a real model are marked ``real_model`` and run only when
``PMETAL_TEST_MODEL_DIR`` names a local model directory (set it, and run the
suite with nothing else on the GPU).
"""

from __future__ import annotations

import json
import os
import random
import stat
import struct
import sys
import textwrap
from pathlib import Path

import pytest

#: The characters of the tiny tokenizer: one token each.
ALPHABET = "abcdefghijklmnopqrstuvwxyz .,\n"
UNK, EOS = "<unk>", "<eos>"
VOCAB = {UNK: 0, EOS: 1, **{c: i + 2 for i, c in enumerate(ALPHABET)}}

HIDDEN, LAYERS, HEADS, KV_HEADS, HEAD_DIM, INTERMEDIATE = 32, 2, 4, 2, 8, 64


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-real-model", action="store_true", help="run tests that load a real model")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "real_model: needs a real model (PMETAL_TEST_MODEL_DIR)")
    config.addinivalue_line("markers", "needs_hub: needs network access to the Hugging Face Hub")
    config.addinivalue_line("markers", "needs_cli: needs the real pmetal command line (PMETAL_CLI)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    real = os.environ.get("PMETAL_TEST_MODEL_DIR")
    hub = os.environ.get("PMETAL_TEST_HUB") == "1"
    cli = os.environ.get("PMETAL_CLI")
    for item in items:
        if "real_model" in item.keywords and not (real and Path(real).is_dir()):
            item.add_marker(pytest.mark.skip(reason="set PMETAL_TEST_MODEL_DIR to a model directory"))
        if "needs_cli" in item.keywords and not (cli and Path(cli).is_file()):
            item.add_marker(pytest.mark.skip(reason="set PMETAL_CLI to the pmetal command line"))
        if "needs_hub" in item.keywords and not hub:
            item.add_marker(pytest.mark.skip(reason="set PMETAL_TEST_HUB=1 to use the network"))


def _tensor(rng: random.Random, shape: tuple[int, ...], scale: float = 0.05) -> bytes:
    count = 1
    for dim in shape:
        count *= dim
    return struct.pack(f"<{count}f", *(rng.gauss(0.0, scale) for _ in range(count)))


def _ones(shape: tuple[int, ...]) -> bytes:
    count = 1
    for dim in shape:
        count *= dim
    return struct.pack(f"<{count}f", *([1.0] * count))


def write_safetensors(path: Path, tensors: dict[str, tuple[tuple[int, ...], bytes]]) -> None:
    """A safetensors file of float32 tensors."""
    header: dict[str, object] = {}
    offset = 0
    for name, (shape, data) in tensors.items():
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, data in tensors.values():
            f.write(data)


def make_tiny_model(directory: Path, seed: int = 0) -> Path:
    """Write a tiny Qwen3 checkpoint into ``directory`` and return it."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    vocab = len(VOCAB)
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "vocab_size": vocab,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 256,
        "tie_word_embeddings": True,
        "hidden_act": "silu",
        "attention_bias": False,
        "eos_token_id": VOCAB[EOS],
        "bos_token_id": VOCAB[EOS],
        "torch_dtype": "float32",
    }
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "generation_config.json").write_text(
        json.dumps({"do_sample": True, "temperature": 0.8, "top_k": 8, "top_p": 0.95, "eos_token_id": VOCAB[EOS]})
    )

    tensors: dict[str, tuple[tuple[int, ...], bytes]] = {
        "model.embed_tokens.weight": ((vocab, HIDDEN), _tensor(rng, (vocab, HIDDEN))),
        "model.norm.weight": ((HIDDEN,), _ones((HIDDEN,))),
    }
    for i in range(LAYERS):
        p = f"model.layers.{i}"
        shapes = {
            f"{p}.self_attn.q_proj.weight": (HEADS * HEAD_DIM, HIDDEN),
            f"{p}.self_attn.k_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
            f"{p}.self_attn.v_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
            f"{p}.self_attn.o_proj.weight": (HIDDEN, HEADS * HEAD_DIM),
            f"{p}.mlp.gate_proj.weight": (INTERMEDIATE, HIDDEN),
            f"{p}.mlp.up_proj.weight": (INTERMEDIATE, HIDDEN),
            f"{p}.mlp.down_proj.weight": (HIDDEN, INTERMEDIATE),
        }
        for name, shape in shapes.items():
            tensors[name] = (shape, _tensor(rng, shape))
        for name, shape in {
            f"{p}.self_attn.q_norm.weight": (HEAD_DIM,),
            f"{p}.self_attn.k_norm.weight": (HEAD_DIM,),
            f"{p}.input_layernorm.weight": (HIDDEN,),
            f"{p}.post_attention_layernorm.weight": (HIDDEN,),
        }.items():
            tensors[name] = (shape, _ones(shape))
    write_safetensors(directory / "model.safetensors", tensors)

    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": VOCAB[UNK],
                "content": UNK,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": VOCAB[EOS],
                "content": EOS,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "."}, "behavior": "Isolated", "invert": False},
        "post_processor": None,
        "decoder": {"type": "Fuse"},
        "model": {"type": "WordLevel", "vocab": VOCAB, "unk_token": UNK},
    }
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer))
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"eos_token": EOS, "unk_token": UNK, "bos_token": None, "pad_token": EOS})
    )
    return directory


def make_tiny_dflash_draft(directory: Path, seed: int = 1, block_size: int = 4) -> Path:
    """Write a tiny DFlash draft for the tiny model into ``directory`` and return it."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    vocab = len(VOCAB)
    taps = [0, LAYERS - 1]
    config = {
        "architectures": ["DFlashDraftModel"],
        "model_type": "dflash_qwen3",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 2,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-6,
        "vocab_size": vocab,
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "block_size": block_size,
        "dflash_config": {"target_layer_ids": taps, "mask_token_id": VOCAB[UNK]},
    }
    (directory / "config.json").write_text(json.dumps(config))
    tensors: dict[str, tuple[tuple[int, ...], bytes]] = {
        "fc.weight": ((HIDDEN, len(taps) * HIDDEN), _tensor(rng, (HIDDEN, len(taps) * HIDDEN))),
        "hidden_norm.weight": ((HIDDEN,), _ones((HIDDEN,))),
        "norm.weight": ((HIDDEN,), _ones((HIDDEN,))),
    }
    for i in range(2):
        p = f"layers.{i}"
        for name, shape in {
            f"{p}.self_attn.q_proj.weight": (HEADS * HEAD_DIM, HIDDEN),
            f"{p}.self_attn.k_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
            f"{p}.self_attn.v_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
            f"{p}.self_attn.o_proj.weight": (HIDDEN, HEADS * HEAD_DIM),
            f"{p}.mlp.gate_proj.weight": (INTERMEDIATE, HIDDEN),
            f"{p}.mlp.up_proj.weight": (INTERMEDIATE, HIDDEN),
            f"{p}.mlp.down_proj.weight": (HIDDEN, INTERMEDIATE),
        }.items():
            tensors[name] = (shape, _tensor(rng, shape))
        for name, shape in {
            f"{p}.self_attn.q_norm.weight": (HEAD_DIM,),
            f"{p}.self_attn.k_norm.weight": (HEAD_DIM,),
            f"{p}.input_layernorm.weight": (HIDDEN,),
            f"{p}.post_attention_layernorm.weight": (HIDDEN,),
        }.items():
            tensors[name] = (shape, _ones(shape))
    write_safetensors(directory / "model.safetensors", tensors)
    return directory


@pytest.fixture(scope="session")
def tiny_draft(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny random DFlash draft for ``tiny_model`` (block of four)."""
    return make_tiny_dflash_draft(tmp_path_factory.mktemp("tiny-draft") / "draft")


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny random Qwen3 checkpoint (see the module docstring)."""
    return make_tiny_model(tmp_path_factory.mktemp("tiny-model") / "model")


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """A small plain-text JSONL dataset the tiny tokenizer can encode."""
    rng = random.Random(0)
    words = ["the", "cat", "sat", "on", "a", "mat", "and", "the", "dog", "ran"]
    path = tmp_path / "train.jsonl"
    with path.open("w") as f:
        for _ in range(16):
            f.write(json.dumps({"text": " ".join(rng.choice(words) for _ in range(12)) + "."}) + "\n")
    return path


@pytest.fixture
def fake_cli(tmp_path: Path):
    """Write an executable stand-in for the ``pmetal`` command and return its path.

    ``body`` is the Python source of its behaviour; it can read ``sys.argv``.
    """

    import pmetal

    def make(body: str, name: str = "pmetal", version: str = pmetal.__version__) -> Path:
        script = tmp_path / name
        script.write_text(
            f"#!{sys.executable}\nimport json, os, sys, time\n"
            f"if sys.argv[1:] == ['--version']:\n    print('pmetal {version}'); sys.exit(0)\n" + textwrap.dedent(body)
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        return script

    return make
