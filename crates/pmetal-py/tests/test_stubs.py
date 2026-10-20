"""The type stubs: current, accurate, checked by type checkers, documented.

* the committed stub is what ``stub.py`` writes from the compiled module;
* ``mypy --strict`` and ``pyright --strict`` accept a sample that uses the whole
  API, and mypy rejects each of the mistakes in ``negative_usage.py``;
* ``mypy.stubtest`` finds no disagreement between the stub and the module;
* every public symbol has a docstring, at runtime and in the stub.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import shutil
import subprocess
import sys
from pathlib import Path

import pmetal
import pytest
from pmetal import pmetal as native

CRATE = Path(__file__).resolve().parent.parent
STUB = CRATE / "py_src" / "pmetal" / "pmetal.pyi"
TYPING = CRATE / "tests" / "typing"


def run(*args: str, cwd: Path = CRATE) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=600, check=False)


needs_mypy = pytest.mark.skipif(importlib.util.find_spec("mypy") is None, reason="mypy not installed")


def test_the_committed_stub_is_what_the_generator_writes():
    result = run(sys.executable, "stub.py", "--check")
    assert result.returncode == 0, result.stderr


@needs_mypy
def test_mypy_strict_accepts_the_sample_and_the_package():
    result = run(
        sys.executable,
        "-m",
        "mypy",
        "--strict",
        "--warn-unused-ignores",
        str(TYPING / "sample_usage.py"),
        str(CRATE / "py_src" / "pmetal"),
    )
    assert result.returncode == 0, result.stdout + result.stderr


@needs_mypy
def test_mypy_rejects_each_mistake_in_the_negative_sample():
    # --warn-unused-ignores: a mistake the stubs let through leaves its ignore unused.
    result = run(sys.executable, "-m", "mypy", "--strict", "--warn-unused-ignores", str(TYPING / "negative_usage.py"))
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("pyright") is None, reason="pyright not installed")
def test_pyright_strict_accepts_the_sample():
    result = run("pyright", "--pythonpath", sys.executable, str(TYPING / "sample_usage.py"))
    assert result.returncode == 0, result.stdout + result.stderr


@needs_mypy
def test_stubtest_finds_no_disagreement_between_stub_and_module():
    result = run(
        sys.executable,
        "-m",
        "mypy.stubtest",
        "pmetal.pmetal",
        "--allowlist",
        str(CRATE / "stubtest_allowlist.txt"),
        "--ignore-unused-allowlist",
        cwd=CRATE.parent,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# -- documentation ----------------------------------------------------------------------------------


def public_members():
    for name, obj in vars(native).items():
        if name.startswith("_") or inspect.ismodule(obj):
            continue
        yield name, obj


def test_every_public_symbol_has_a_runtime_docstring():
    missing = []
    for name, obj in public_members():
        if inspect.isclass(obj) or inspect.isroutine(obj):
            if not (obj.__doc__ or "").strip():
                missing.append(name)
        if inspect.isclass(obj) and obj.__module__ == "pmetal":
            for attr, member in vars(obj).items():
                if attr.startswith("_") or attr in {"errors"}:
                    continue
                if (
                    inspect.isroutine(member)
                    or isinstance(member, (property, staticmethod, classmethod))
                    or type(member).__name__ in {"getset_descriptor", "method_descriptor"}
                ):
                    doc = getattr(member, "__doc__", None) or getattr(
                        getattr(member, "__func__", None), "__doc__", None
                    )
                    if not (doc or "").strip():
                        missing.append(f"{name}.{attr}")
    # Enum members are documented by their class.
    missing = [m for m in missing if not m.startswith("ModelArchitecture.")]
    assert not missing, f"no docstring: {missing}"


def test_every_public_symbol_has_a_stub_docstring():
    tree = ast.parse(STUB.read_text())
    missing: list[str] = []

    def has_doc(body: list[ast.stmt]) -> bool:
        return (
            bool(body)
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        )

    def visit(body: list[ast.stmt], prefix: str, in_class: bool) -> None:
        for i, node in enumerate(body):
            if isinstance(node, ast.ClassDef):
                if not node.name.startswith("_") and not has_doc(node.body):
                    missing.append(prefix + node.name)
                # A TypedDict's keys are documented by the class that has them as attributes.
                is_typed_dict = any(getattr(b, "id", None) == "TypedDict" for b in node.bases)
                if not is_typed_dict:
                    visit(node.body, f"{prefix}{node.name}.", True)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                public = not node.name.startswith("_")
                if public and not has_doc(node.body):
                    missing.append(prefix + node.name)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and in_class:
                # An attribute is documented by the string literal after it.
                nxt = body[i + 1] if i + 1 < len(body) else None
                documented = isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant)
                if not node.target.id.startswith("_") and not documented:
                    missing.append(prefix + node.target.id)

    visit(tree.body, "", False)
    # Enum members and the ModelArchitecture variants are documented by their class.
    missing = [m for m in missing if not m.startswith("ModelArchitecture.")]
    assert not missing, f"no stub docstring: {missing}"


def test_the_package_re_exports_every_public_name_of_the_module():
    """Nothing the module defines is unreachable from ``pmetal`` (except internals)."""
    internal = {"configure_metallib"}
    for name, _ in public_members():
        if name in internal:
            continue
        assert hasattr(pmetal, name), f"pmetal.{name} is not exported by the package"
