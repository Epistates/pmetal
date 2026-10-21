"""The Hub calls, against a fake Hub on localhost (see ``fake_hub.py``)."""

from __future__ import annotations

import os
import signal
import threading
import time
from pathlib import Path

import pmetal
import pytest
from conftest import make_tiny_model
from fake_hub import FakeHub


@pytest.fixture(autouse=True)
def isolated_hub_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No test sends a real credential anywhere or touches the real cache."""
    for name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_ENDPOINT",
        "HF_HUB_OFFLINE",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HOME",
        "HF_TOKEN_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))


@pytest.fixture
def hub(tmp_path: Path):
    source = make_tiny_model(tmp_path / "source")
    with FakeHub() as fake:
        fake.source = source  # type: ignore[attr-defined]
        fake.commit = fake.add("org/tiny", source)  # type: ignore[attr-defined]
        yield fake


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "hub-cache"


def download(hub: FakeHub, cache: Path, **kwargs):
    return pmetal.download_model("org/tiny", endpoint=hub.endpoint, cache_dir=cache, **kwargs)


def test_a_download_lands_a_complete_model_that_loads_and_generates(hub, cache):
    path = Path(download(hub, cache))
    assert path.parent.name == "snapshots" and path.name == hub.commit
    assert (path / "model.safetensors").is_file() and (path / "tokenizer.json").is_file()
    report = pmetal.verify_model(path)
    assert report.is_complete and report.format == "safetensors" and report and report.weight_bytes > 0
    assert report.is_loadable_language_model and report.problems == [] and report.warnings == []
    # The same model, from the Hub id and the cache, generates what the source does.
    from_hub = pmetal.Model.load(path).generate("the cat", max_tokens=8, temperature=0.0)
    assert from_hub == pmetal.Model.load(hub.source).generate("the cat", max_tokens=8, temperature=0.0)


def test_progress_is_reported_as_dicts_and_ends_finished(hub, cache):
    seen: list[dict] = []
    download(hub, cache, progress=seen.append)
    assert seen and all(
        set(p)
        == {
            "files_done",
            "total_files",
            "bytes_done",
            "total_bytes",
            "bytes_per_sec",
            "eta_seconds",
            "current_file",
            "finished",
        }
        for p in seen
    )
    assert seen[-1]["finished"] is True and seen[-1]["bytes_done"] == seen[-1]["total_bytes"] > 0
    assert seen[-1]["files_done"] == seen[-1]["total_files"] >= 4
    done = [p["bytes_done"] for p in seen]
    assert done == sorted(done), "progress never goes backwards"


def test_an_exception_in_the_progress_callback_stops_the_download_and_is_raised(hub, cache):
    class Boom(Exception):
        pass

    def progress(_: dict) -> None:
        raise Boom("progress failed")

    with pytest.raises(Boom, match="progress failed"):
        download(hub, cache, progress=progress)


def test_a_pinned_commit_is_downloaded_and_served_from_the_cache_offline(hub, cache):
    first = download(hub, cache, revision=hub.commit)
    requests_after_first = len(hub.requests)
    # By commit, a complete cached copy answers without a request; offline never asks.
    assert download(hub, cache, revision=hub.commit) == first
    assert download(hub, cache, offline=True) == first
    assert len(hub.requests) == requests_after_first


def test_offline_with_nothing_cached_is_a_hub_error(hub, cache):
    with pytest.raises(pmetal.HubError, match="(?i)offline"):
        download(hub, cache, offline=True)
    assert hub.requests == []


def test_an_unknown_repository_is_a_hub_error_that_says_so(hub, cache):
    with pytest.raises(pmetal.HubError, match="not found"):
        pmetal.download_model("org/missing", endpoint=hub.endpoint, cache_dir=cache)


def test_a_private_repository_needs_the_token_and_the_token_is_sent(hub, cache, tmp_path):
    hub.add("org/private", hub.source, token="secret-token")
    with pytest.raises(pmetal.HubError):
        pmetal.download_model("org/private", endpoint=hub.endpoint, cache_dir=cache)
    with pytest.raises(pmetal.HubError):
        pmetal.download_model("org/private", endpoint=hub.endpoint, cache_dir=cache, token="wrong")
    path = pmetal.download_model("org/private", endpoint=hub.endpoint, cache_dir=cache, token="secret-token")
    assert Path(path).is_dir()
    assert any(auth == "Bearer secret-token" for _, _, auth in hub.requests)


def test_the_token_never_appears_in_an_error(hub, cache):
    hub.add("org/private", hub.source, token="secret-token-value")
    with pytest.raises(pmetal.HubError) as caught:
        pmetal.download_model("org/private", endpoint=hub.endpoint, cache_dir=cache, token="wrong-token-value")
    assert "wrong-token-value" not in str(caught.value) and "secret-token-value" not in str(caught.value)


def test_a_transient_server_error_is_retried(hub, cache):
    hub.fail_once.add(f"/org/tiny/resolve/{hub.commit}/config.json")
    path = download(hub, cache)
    assert (Path(path) / "config.json").is_file()


def test_a_size_check_passes_a_same_length_corruption(hub, cache):
    hub.corrupt.add("model.safetensors")  # same length, different bytes
    assert Path(download(hub, cache, verify="size")).is_dir()


def test_sha256_verification_catches_a_same_length_corruption(hub, cache):
    hub.corrupt.add("model.safetensors")
    with pytest.raises(pmetal.HubError, match="(?i)sha|hash|digest|checksum"):
        download(hub, cache, verify="sha256")


def test_a_bad_verification_or_weights_name_is_an_error_before_anything_is_fetched(hub, cache):
    with pytest.raises(ValueError, match="verify must be"):
        download(hub, cache, verify="md5")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weights must be"):
        download(hub, cache, weights="onnx")  # type: ignore[arg-type]
    assert hub.requests == []


def test_plan_download_lists_the_files_without_fetching_them(hub, cache):
    plan = pmetal.plan_download("org/tiny", endpoint=hub.endpoint, cache_dir=cache)
    assert isinstance(plan, pmetal.DownloadPlan)
    assert plan.repo_id == "org/tiny" and plan.commit == hub.commit
    names = {name for name, _ in plan.files}
    assert {"config.json", "model.safetensors", "tokenizer.json"} <= names
    assert plan.total_bytes == sum(size for _, size in plan.files) > 0
    assert not any("/resolve/" in path and method == "GET" for method, path, _ in hub.requests)
    assert not (cache / "models--org--tiny").exists()


def test_download_file_fetches_one_file_and_serves_it_offline(hub, cache):
    path = pmetal.download_file("org/tiny", "config.json", endpoint=hub.endpoint, cache_dir=cache)
    assert Path(path).name == "config.json" and "Qwen3" in Path(path).read_text()
    again = pmetal.download_file("org/tiny", "config.json", offline=True, cache_dir=cache)
    assert again == path
    with pytest.raises(pmetal.HubError):
        pmetal.download_file("org/tiny", "nope.bin", endpoint=hub.endpoint, cache_dir=cache)


def test_verify_model_finds_a_hub_id_in_the_cache_and_reports_damage(hub, cache, tmp_path):
    path = Path(download(hub, cache))
    assert pmetal.verify_model("org/tiny", cache_dir=cache).is_complete
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "config.json").write_text((path / "config.json").read_text())
    data = (path / "model.safetensors").read_bytes()
    (broken / "model.safetensors").write_bytes(data[: len(data) // 2])
    report = pmetal.verify_model(broken)
    assert not report and not report.is_complete and report.problems
    assert pmetal.verify_model(tmp_path / "empty-nowhere").problems, (
        "a missing directory is a problem, not an exception"
    )
    empty = tmp_path / "empty"
    empty.mkdir()
    assert pmetal.verify_model(empty).format is None


def test_search_results_carry_the_hubs_facts_and_a_fit(hub):
    hub.search_results = [
        {
            "id": "org/tiny",
            "modelId": "org/tiny",
            "downloads": 12,
            "likes": 3,
            "tags": ["safetensors", "qwen3"],
            "pipeline_tag": "text-generation",
            "lastModified": "2026-01-01T00:00:00.000Z",
            "safetensors": {"parameters": {"F32": 20_000}, "total": 20_000},
        },
    ]
    results = pmetal.search_models("tiny", endpoint=hub.endpoint, limit=3, tags=["qwen3"], sort="likes")
    assert [r["model_id"] for r in results] == ["org/tiny"]
    result = results[0]
    assert result["downloads"] == 12 and result["likes"] == 3 and "qwen3" in result["tags"]
    assert result["safetensors_bytes"] == 80_000
    fit = result["fit"]
    assert fit is not None and fit["fit_level"] == "Fits" and fit["weights_gb"] == pytest.approx(80_000 / 1e9, rel=0.01)
    assert fit["context_length"] == 4096
    with pytest.raises(ValueError, match="sort must be"):
        pmetal.search_models("x", sort="oldest")  # type: ignore[arg-type]


def test_estimate_fit_for_a_local_model(tiny_model):
    fit = pmetal.estimate_fit(tiny_model)
    assert fit["fit_level"] in ("Fits", "Tight", "TooLarge") and fit["context_length"] == 4096
    assert fit["total_required_gb"] > fit["weights_gb"] > 0 and fit["available_gb"] > 0
    assert fit["weights_measured"] is True
    longer = pmetal.estimate_fit(tiny_model, context_length=65536)
    assert longer["kv_cache_gb"] > fit["kv_cache_gb"] and longer["context_length"] == 65536
    quantized = pmetal.estimate_fit(tiny_model, context_length=65536, kv_cache_bits=4)
    assert quantized["kv_cache_gb"] < longer["kv_cache_gb"]


def test_estimate_fit_needs_a_config(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(pmetal.ConfigError, match="config.json"):
        pmetal.estimate_fit(empty)


def test_upload_rejects_what_is_not_a_directory(tmp_path):
    with pytest.raises(pmetal.ConfigError, match="not a directory"):
        pmetal.upload_model(tmp_path / "nope", "me/repo")


# -- the GIL and Ctrl-C ------------------------------------------------------------------------------


def test_the_gil_is_released_during_a_download(hub, cache):
    hub.delay = 0.3
    ticks = 0
    stop = threading.Event()

    def tick() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            time.sleep(0.001)

    thread = threading.Thread(target=tick)
    thread.start()
    try:
        started = time.perf_counter()
        download(hub, cache)
        elapsed = time.perf_counter() - started
    finally:
        stop.set()
        thread.join()
    assert elapsed > 0.25 and ticks >= elapsed * 100 * 0.2, f"{ticks} ticks in {elapsed:.2f}s"


def test_ctrl_c_abandons_a_download(hub, cache):
    hub.delay = 30.0  # every file body takes half a minute
    timer = threading.Timer(0.6, lambda: os.kill(os.getpid(), signal.SIGINT))
    timer.start()
    started = time.perf_counter()
    try:
        with pytest.raises(KeyboardInterrupt):
            download(hub, cache)
    finally:
        timer.cancel()
    assert time.perf_counter() - started < 15
    hub.delay = 0.0
    assert Path(download(hub, cache)).is_dir(), "a later call completes the download"


@pytest.mark.needs_hub
def test_the_real_hub_serves_a_small_file(tmp_path):
    path = pmetal.download_file("Qwen/Qwen3-0.6B", "config.json", cache_dir=tmp_path)
    assert Path(path).is_file()
