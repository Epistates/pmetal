"""Generate ``py_src/pmetal/pmetal.pyi``, the type stubs of the extension module.

The job specs are the one definition of their fields, so their stubs are not
written by hand: this script reads each spec class's ``fields()`` from the
compiled module (name, type, default, flag, help) and writes the class, with
a ``TypedDict`` of its options for ``**kwargs`` typing. Everything else is
``stub_template.pyi``, which is hand-written.

    python stub.py            # rewrite the stub from the installed build
    python stub.py --check    # exit 1 if the stub on disk is not what it would write

Run after ``maturin develop``. ``tests/test_stubs.py`` runs the check.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "stub_template.pyi"
OUTPUT = HERE / "py_src" / "pmetal" / "pmetal.pyi"

#: Spec classes in stub order: the training specs first, then the rest.
SPEC_ORDER = [
    "TrainSpec",
    "InferSpec",
    "ServeSpec",
    "GrpoSpec",
    "PreferenceSpec",
    "DistillSpec",
    "RlkdSpec",
    "EmbedTrainSpec",
    "PretrainSpec",
    "QuantizeSpec",
    "MergeSpec",
    "FuseSpec",
    "EvalSpec",
    "BenchSpec",
    "DflashSpec",
    "TrainMtpSpec",
    "TrainDraftSpec",
    "TrainDiffusionSpec",
    "DecideSpec",
    "TokenizeSpec",
    "VocodeSpec",
    "PackExpertsSpec",
]

#: Methods a spec class has beyond the common ones.
EXTRA_METHODS = {
    "TrainSpec": '''\
    def lora_config(self) -> dict[str, Any]:
        """The adapter configuration the spec asks for (``LoraConfig``), as a dict."""
    def training_config(self) -> dict[str, Any]:
        """The optimisation settings the spec asks for (``TrainingConfig``), as a dict.

        Raises:
            ConfigError: An optimizer or schedule name the trainer does not know.
        """
''',
}


def options_name(spec: str) -> str:
    """``TrainSpec`` -> ``TrainOptions``."""
    return spec.removesuffix("Spec") + "Options"


def doc(text: str, indent: str) -> str:
    """A docstring, indented."""
    first, *rest = textwrap.dedent(text).strip().splitlines()
    if not rest:
        return f'{indent}"""{first}"""\n'
    body = "\n".join((indent + line).rstrip() for line in rest)
    return f'{indent}"""{first}\n{body}\n{indent}"""\n'


def field_doc(info: object) -> str:
    """The attribute docstring of a field."""
    parts = [info.label.rstrip(".") + "."]  # type: ignore[attr-defined]
    if info.help:  # type: ignore[attr-defined]
        parts.append(info.help.rstrip(".") + ".")  # type: ignore[attr-defined]
    if info.flag:  # type: ignore[attr-defined]
        parts.append(f"Flag: ``{info.flag}``.")  # type: ignore[attr-defined]
    if info.options:  # type: ignore[attr-defined]
        parts.append("One of: " + ", ".join(f"``{o}``" for o in info.options) + ".")  # type: ignore[attr-defined]
    if info.minimum is not None and info.maximum is not None:  # type: ignore[attr-defined]
        parts.append(f"From {info.minimum:g} to {info.maximum:g}.")  # type: ignore[attr-defined]
    parts.append(f"Default: ``{info.default!r}``.")  # type: ignore[attr-defined]
    return " ".join(parts).replace('"""', "'''")


def spec_stub(name: str, cls: type, only: set[str] | None = None) -> str:
    """The stub of one spec class, with its options ``TypedDict``."""
    fields = cls.fields()  # type: ignore[attr-defined]
    options = options_name(name)
    out = ["@type_check_only\n", f"class {options}(TypedDict, total=False):\n"]
    out.append(f'    """The fields of ``{name}``, as keyword arguments."""\n\n')
    for info in fields:
        out.append(f"    {info.name}: {info.annotation}\n")
    out.append("\n")
    out.append(f"@final\nclass {name}(_Spec):\n")
    out.append(doc(cls.__doc__ or name, "    "))
    out.append(f"    def __new__(cls, **options: Unpack[{options}]) -> Self:\n")
    out.append(
        doc(
            """\
            Args:
                **options: Any field of the spec; the rest keep their defaults.

            Raises:
                TypeError: An unknown field, or a value of the wrong type.
            """,
            "        ",
        )
    )
    out.append(f"    def copy(self, **changes: Unpack[{options}]) -> Self:\n")
    out.append('        """A copy with the given fields replaced."""\n')
    out.append(EXTRA_METHODS.get(name, ""))
    out.append("\n")
    for info in fields:
        out.append(f"    {info.name}: {info.annotation}\n")
        out.append(doc(field_doc(info), "    "))
    out.append("\n")
    return "".join(out)


def call_options(name: str, cls: type, positional: tuple[str, ...], used_by: str) -> str:
    """A ``TypedDict`` of a spec's fields minus the ones a function takes positionally."""
    out = ["@type_check_only\n", f"class {name}(TypedDict, total=False):\n"]
    out.append(f'    """The fields of ``{cls.__name__}`` that ``{used_by}`` takes as keyword arguments."""\n\n')
    for info in cls.fields():  # type: ignore[attr-defined]
        if info.name not in positional:
            out.append(f"    {info.name}: {info.annotation}\n")
    out.append("\n")
    return "".join(out)


def infer_stub(spec_cls: type, unsupported: tuple[str, ...]) -> str:
    """``infer`` and ``infer_with_metrics``, typed by ``InferSpec``'s fields."""
    out = [call_options("InferCallOptions", spec_cls, ("model", "prompt", *unsupported), "infer")]
    out.append("INFER_COMMAND_LINE_ONLY: tuple[str, ...]\n")
    out.append(
        doc(
            "The ``InferSpec`` fields only the command line honours: ``infer`` raises "
            "``ConfigError`` for one that is set. Run the spec with ``pmetal.run``.",
            "",
        )
    )
    common = textwrap.dedent(
        """\
    Options are the fields of ``InferSpec`` (``InferSpec.fields()``), with the
    same defaults: a sampling option left out takes the model's own value. The
    model is loaded for the call; for many prompts against one model, keep a
    ``Model``.

    Raises:
        ConfigError: An option is invalid, or is one only the command line honours.
        KeyboardInterrupt: On Ctrl-C, after the token in flight.
    """
    )
    out.append("\n")
    out.append("def infer(model: str | os.PathLike[str], prompt: str, **options: Unpack[InferCallOptions]) -> str:\n")
    out.append(doc("Generate text for one prompt and return it.\n\n" + common, "    "))
    out.append("\n")
    out.append(
        "def infer_with_metrics(\n    model: str | os.PathLike[str], prompt: str, **options: Unpack[InferCallOptions]\n) -> InferResult:\n"
    )
    out.append(
        doc(
            "Like ``infer``, with the generation's statistics: token counts, why it "
            "stopped, its time, and the acceptance statistics of speculative decoding.\n\n" + common,
            "    ",
        )
    )
    return "".join(out)


def generate() -> str:
    """The stub text for the installed build."""
    from pmetal import pmetal as native

    specs = "".join(spec_stub(name, getattr(native, name)) for name in SPEC_ORDER)
    specs += call_options("TrainCallOptions", native.TrainSpec, ("model", "dataset"), "finetune")
    specs += call_options("GrpoCallOptions", native.GrpoSpec, ("model", "dataset"), "grpo")
    unsupported = tuple(native.INFER_COMMAND_LINE_ONLY)
    infer = infer_stub(native.InferSpec, unsupported)
    text = TEMPLATE.read_text()
    text = text.replace("# @@SPECS@@\n", specs)
    text = text.replace("# @@INFER@@\n", infer)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the stub is out of date")
    args = parser.parse_args()
    text = generate()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            print(f"{OUTPUT} is out of date: run `python stub.py`", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
