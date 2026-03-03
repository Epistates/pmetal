"""Generate .pyi type stubs for the pmetal Python module.

Run after `maturin develop`:
    python stub.py

Based on candle-pyo3's introspection pattern.
"""

import inspect
import os
from pathlib import Path

INDENT = "    "
GENERATED_COMMENT = "# Generated content DO NOT EDIT\n"
TYPING_HEADER = """from typing import Any, Dict, List, Optional, Union\n"""


def do_indent(text: str | None, indent: str) -> str:
    if text is None:
        return ""
    return text.replace("\n", f"\n{indent}")


def format_function(obj, indent: str, text_signature: str | None = None) -> str:
    if text_signature is None:
        text_signature = getattr(obj, "__text_signature__", None)
    if text_signature is None:
        text_signature = "(*args, **kwargs)"

    text_signature = text_signature.replace("$self", "self").strip()
    doc_string = obj.__doc__ or ""

    result = f"{indent}def {obj.__name__}{text_signature}:\n"
    inner = indent + INDENT
    if doc_string:
        result += f'{inner}"""\n'
        result += f"{inner}{do_indent(doc_string, inner)}\n"
        result += f'{inner}"""\n'
    result += f"{inner}...\n\n"
    return result


def format_property(name: str, indent: str) -> str:
    return f"{indent}@property\n{indent}def {name}(self): ...\n\n"


def format_class(name: str, obj, indent: str = "") -> str:
    result = f"{indent}class {name}:\n"
    inner = indent + INDENT

    # Document class docstring
    if obj.__doc__:
        result += f'{inner}"""{obj.__doc__}"""\n\n'

    has_members = False

    # Methods and descriptors
    for attr_name, attr in sorted(inspect.getmembers(obj)):
        if attr_name.startswith("_") and attr_name not in ("__init__", "__repr__", "__new__"):
            continue

        if inspect.ismethoddescriptor(attr) or inspect.isbuiltin(attr):
            sig = getattr(attr, "__text_signature__", None)
            if sig:
                result += format_function(attr, inner, sig)
                has_members = True
        elif inspect.isgetsetdescriptor(attr):
            result += format_property(attr_name, inner)
            has_members = True

    if not has_members:
        result += f"{inner}...\n"

    result += "\n"
    return result


def format_enum(name: str, obj, indent: str = "") -> str:
    result = f"{indent}class {name}:\n"
    inner = indent + INDENT

    for attr_name in sorted(dir(obj)):
        if attr_name.startswith("_"):
            continue
        val = getattr(obj, attr_name, None)
        if isinstance(val, obj):
            result += f"{inner}{attr_name}: {name}\n"

    result += "\n"
    return result


def generate_stubs() -> str:
    import pmetal

    output = GENERATED_COMMENT
    output += TYPING_HEADER
    output += "\n"
    output += f'__version__: str = "{pmetal.__version__}"\n\n'

    members = []
    for name in sorted(dir(pmetal)):
        if name.startswith("_"):
            continue
        obj = getattr(pmetal, name)
        members.append((name, obj))

    # Functions first
    for name, obj in members:
        if inspect.isbuiltin(obj) or inspect.isfunction(obj):
            sig = getattr(obj, "__text_signature__", None)
            if sig:
                output += format_function(obj, "")

    # Classes
    for name, obj in members:
        if inspect.isclass(obj):
            # Check if it's an enum-like class (pyo3 enums)
            variants = [a for a in dir(obj) if not a.startswith("_") and isinstance(getattr(obj, a, None), obj)]
            if variants:
                output += format_enum(name, obj)
            else:
                output += format_class(name, obj)

    return output


def main():
    stubs = generate_stubs()
    stub_path = Path(__file__).parent / "py_src" / "pmetal" / "__init__.pyi"
    stub_path.write_text(stubs)
    print(f"Generated stubs at {stub_path}")


if __name__ == "__main__":
    main()
