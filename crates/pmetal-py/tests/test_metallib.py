"""Pointing MLX at the metallib built with it."""

from __future__ import annotations

from pathlib import Path

import pmetal
import pytest
from pmetal import pmetal as native

PACKAGE = Path(pmetal.__file__).resolve().parent


def assert_a_whole_metal_library(path: str | Path) -> None:
    data = Path(path).read_bytes()
    assert data[:4] == b"MTLB"
    # The file's own header records its total size.
    assert int.from_bytes(data[16:24], "little") == len(data)


def test_a_metallib_is_found_and_is_a_whole_metal_library():
    # Where the package's `__init__` looks: the directory the module is installed in. A
    # wheel carries the file there and has no build tree; a development tree has the build's own.
    path = native.configure_metallib(PACKAGE)
    assert path is not None, "the metallib built with this MLX is found from the package directory"
    assert_a_whole_metal_library(path)


def test_the_metallib_in_the_package_is_a_whole_metal_library():
    # A wheel carries the command line and the metallib together; a development tree neither.
    if not (PACKAGE / "_bin").exists():
        pytest.skip("not a wheel install: the package carries no command line, so no metallib either")
    packaged = PACKAGE / "mlx.metallib"
    assert packaged.exists(), "the wheel carries the metallib built with its MLX"
    assert_a_whole_metal_library(packaged)


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
