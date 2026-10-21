"""Pointing MLX at the metallib built with it."""

from __future__ import annotations

from pathlib import Path

import pytest
from pmetal import pmetal as native


def test_a_metallib_is_found_and_is_a_whole_metal_library(tmp_path):
    path = native.configure_metallib(tmp_path)
    assert path is not None, "the build's own metallib is found while its tree exists"
    data = Path(path).read_bytes()
    assert data[:4] == b"MTLB"
    # The file's own header records its total size.
    assert int.from_bytes(data[16:24], "little") == len(data)


def test_an_unusable_override_is_skipped_with_a_warning(tmp_path, monkeypatch):
    bad = tmp_path / "mlx.metallib"
    bad.write_bytes(b"not a metallib")
    monkeypatch.setenv("PMETAL_METALLIB_PATH", str(bad))
    with pytest.warns(UserWarning, match="unusable mlx.metallib"):
        path = native.configure_metallib(tmp_path)
    assert path != str(bad)


def test_a_damaged_copy_in_the_package_is_not_used(tmp_path):
    (tmp_path / "mlx.metallib").write_bytes(b"MTLB" + b"\0" * 100)
    path = native.configure_metallib(tmp_path)
    assert path is None or path != str(tmp_path / "mlx.metallib")
