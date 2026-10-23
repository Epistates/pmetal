"""The job specs: Python classes over ``pmetal-core``'s specs.

The specs define every field, default and check; these tests hold the Python
classes to them. Nothing here restates a default: the expected value is always
read from the spec itself (``fields()``) or checked against what the spec
produces (``to_argv``, ``training_config``).
"""

from __future__ import annotations

import copy
import json
import pickle
from typing import Any

import pmetal
import pytest

SPEC_NAMES = sorted(n for n in dir(pmetal) if n.endswith("Spec") and n != "JobSpec")
SPECS = [getattr(pmetal, n) for n in SPEC_NAMES]


def different_value(info: pmetal.FieldInfo) -> Any:
    """A value of the field's type that is not its default."""
    default = info.default
    if info.options:
        return next(o for o in info.options if o != default)
    if info.annotation.startswith("bool"):
        return not default
    if info.annotation.startswith("list"):
        return ["x"]
    lo, hi = info.minimum, info.maximum
    if info.annotation.startswith("int"):
        base = int(lo) if lo is not None else 0
        value = base if default != base else base + 1
        return value
    if info.annotation.startswith("float"):
        base = lo if lo is not None else 0.0
        top = hi if hi is not None else base + 1.0
        mid = base + (top - base) / 4
        return mid if mid != default else base + (top - base) / 3
    return "other-value"


def test_there_is_a_class_for_every_subcommand_spec():
    # One class per spec in pmetal_core::jobs (23), each with a distinct subcommand.
    assert len(SPECS) == 23
    assert len({cls.subcommand for cls in SPECS}) == 23


@pytest.mark.parametrize("cls", SPECS, ids=SPEC_NAMES)
def test_fields_are_the_specs_and_their_defaults_are_used(cls):
    spec = cls()
    fields = cls.fields()
    assert fields, "a spec has fields"
    assert [f.name for f in fields] == sorted(spec.to_dict()) or {f.name for f in fields} == set(spec.to_dict())
    for info in fields:
        assert getattr(spec, info.name) == info.default, f"{cls.__name__}.{info.name}"


@pytest.mark.parametrize("cls", SPECS, ids=SPEC_NAMES)
def test_no_field_hides_a_method(cls):
    # Fields are read through __getattr__, so a field named like a method would be unreachable.
    for info in cls.fields():
        assert info.name not in {
            "fields",
            "normalize",
            "to_argv",
            "to_dict",
            "from_dict",
            "to_json",
            "from_json",
            "copy",
            "validation_errors",
            "subcommand",
        }


@pytest.mark.parametrize("cls", SPECS, ids=SPEC_NAMES)
def test_every_flagged_field_reaches_the_command_line(cls):
    """Set each field away from its default: the spec's own flag is in the argv."""
    argv_default = cls().to_argv()
    for info in cls.fields():
        if info.flag is None:
            continue
        spec = cls()
        value = different_value(info)
        setattr(spec, info.name, value)
        argv = spec.to_argv()
        assert info.flag in argv, f"{cls.__name__}.{info.name}={value!r} did not produce {info.flag}: {argv}"
        assert argv != argv_default


@pytest.mark.parametrize("cls", SPECS, ids=SPEC_NAMES)
def test_round_trips(cls):
    spec = cls()
    for info in cls.fields()[:6]:
        setattr(spec, info.name, different_value(info))
    assert cls.from_dict(spec.to_dict()) == spec
    assert cls.from_json(spec.to_json()) == spec
    assert copy.deepcopy(spec) == spec
    assert copy.copy(spec) == spec
    assert pickle.loads(pickle.dumps(spec)) == spec
    assert json.loads(spec.to_json()) == spec.to_dict()
    assert eval(repr(spec), {cls.__name__: cls}) == spec  # noqa: S307 - repr is constructor syntax


def test_f32_fields_read_back_as_written():
    """A single-precision field holds 0.05, not the double nearest the float32 0.05."""
    spec = pmetal.TrainSpec(model="m", dataset="d", lora_dropout=0.05, lora_alpha=0.1)
    assert spec.lora_dropout == 0.05 and spec.lora_alpha == 0.1
    assert spec.to_dict()["lora_dropout"] == 0.05
    assert spec.lora_config()["dropout"] == 0.05
    assert spec.lora_config()["alpha"] == 0.1
    assert json.loads(spec.to_json())["lora_dropout"] == 0.05
    # An f32 default reads back as its shortest decimal too.
    assert (
        pmetal.DistillSpec().to_dict()["learning_rate"] == json.loads(pmetal.DistillSpec().to_json())["learning_rate"]
    )


def test_a_spec_is_a_value_not_a_dict():
    spec = pmetal.TrainSpec(model="m", dataset="d")
    other = spec.copy(epochs=3)
    assert other.epochs == 3 and spec.epochs != 3, "copy() must not change the original"
    assert spec == pmetal.TrainSpec(model="m", dataset="d")
    assert spec != other
    assert spec != "not a spec"
    with pytest.raises(TypeError):
        hash(spec)  # mutable, so not hashable
    assert repr(spec) == "TrainSpec(model='m', dataset='d')"


def test_unknown_fields_are_caught_with_a_suggestion():
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'modle'.*Did you mean 'model'"):
        pmetal.TrainSpec(modle="x")
    spec = pmetal.TrainSpec()
    with pytest.raises(AttributeError, match=r"no field 'epochz'.*Did you mean 'epochs'"):
        spec.epochz = 3
    with pytest.raises(AttributeError, match="no attribute 'epochz'"):
        _ = spec.epochz
    with pytest.raises(TypeError, match="unexpected keyword"):
        spec.copy(nope=1)
    with pytest.raises(KeyError, match="no field"):
        pmetal.TrainSpec.from_json('{"nope": 1}')


def test_wrongly_typed_values_name_the_field():
    with pytest.raises(TypeError, match=r"TrainSpec\.epochs"):
        pmetal.TrainSpec(epochs="three")
    with pytest.raises(TypeError, match=r"TrainSpec\.learning_rate"):
        pmetal.TrainSpec(learning_rate="fast")
    spec = pmetal.TrainSpec()
    with pytest.raises(TypeError, match=r"TrainSpec\.epochs"):
        spec.epochs = -1
    assert spec.epochs == pmetal.TrainSpec().epochs, "a rejected assignment leaves the field alone"
    with pytest.raises(OverflowError):
        pmetal.TrainSpec(seed=2**70)
    with pytest.raises(ValueError, match="cannot be represented"):
        pmetal.TrainSpec(learning_rate=float("nan"))


def test_fields_cannot_be_deleted():
    with pytest.raises(AttributeError, match="cannot delete"):
        del pmetal.TrainSpec().epochs


def test_paths_are_accepted_wherever_a_string_is(tmp_path):
    spec = pmetal.TrainSpec(model=tmp_path / "m", dataset=tmp_path / "d.jsonl", output_dir=tmp_path)
    assert spec.model == str(tmp_path / "m")
    assert spec.output_dir == str(tmp_path)


def test_validation_lists_every_problem():
    spec = pmetal.TrainSpec(lora_r=0)
    assert ("lora_r", "must be ≥ 1") in spec.validation_errors()
    with pytest.raises(pmetal.SpecValidationError) as caught:
        spec.normalize()
    fields = {field for field, _ in caught.value.errors}
    assert {"model", "dataset", "lora_r"} <= fields
    assert isinstance(caught.value, ValueError) and isinstance(caught.value, pmetal.PMetalError)
    for field in fields:
        assert field in str(caught.value)
    assert pmetal.TrainSpec(model="m", dataset="d").validation_errors() == []


def test_enum_fields_are_checked_against_the_specs_options():
    info = next(f for f in pmetal.TrainSpec.fields() if f.name == "optimizer")
    assert info.options and info.default in info.options
    for option in info.options:
        assert pmetal.TrainSpec(model="m", dataset="d", optimizer=option).validation_errors() == []
    bad = pmetal.TrainSpec(model="m", dataset="d", optimizer="adamw-typo")
    assert [f for f, _ in bad.validation_errors()] == ["optimizer"]


def test_field_info_describes_the_field():
    info = {f.name: f for f in pmetal.TrainSpec.fields()}
    lr = info["learning_rate"]
    assert lr.annotation == "float" and lr.kind == "number" and lr.flag == "--learning-rate"
    assert lr.minimum is not None and lr.maximum is not None and lr.minimum < lr.default < lr.maximum
    assert info["eval_dataset"].annotation == "str | None" and info["eval_dataset"].default is None
    assert info["lora_targets"].annotation == "list[str]"
    assert info["model"].required and info["model"].kind == "model"
    assert info["resume"].flag == "--resume" or info["resume"].flag is None
    assert info["wsd_decay_style"].annotation.startswith("Literal[") and info["wsd_decay_style"].annotation.endswith(
        "| None"
    )
    assert "label" in repr(info["learning_rate"]) or "learning_rate" in repr(info["learning_rate"])


# -- TrainSpec: the trainer's configs come from the spec -------------------------------------------


def test_train_spec_builds_the_trainers_configs():
    spec = pmetal.TrainSpec(
        model="m",
        dataset="d",
        learning_rate=1e-4,
        epochs=2,
        batch_size=3,
        lora_r=8,
        lora_alpha=16.0,
        dora=True,
        rslora=True,
        lora_dropout=0.05,
        lora_targets=["q_proj", "v_proj"],
        optimizer="muon",
        lr_schedule="wsd",
        wsd_decay_style="cosine",
        wsd_stable_ratio=0.6,
        label_smoothing=0.1,
        z_loss=1e-4,
        neftune_alpha=5.0,
        warmup_ratio=0.1,
        min_lr=1e-6,
        embedding_lr=5e-5,
        gradient_accumulation_steps=2,
        seed=7,
    )
    training = spec.training_config()
    assert training["learning_rate"] == 1e-4
    assert training["num_epochs"] == 2 and training["batch_size"] == 3
    assert training["optimizer"].lower() == "muon"
    assert training["wsd_decay_style"] in ("Cosine", "cosine")
    assert training["label_smoothing"] == pytest.approx(0.1)
    assert training["z_loss"] == pytest.approx(1e-4)
    assert training["neftune_noise_alpha"] == 5.0
    assert training["warmup_ratio"] == pytest.approx(0.1)
    assert training["embedding_learning_rate"] == pytest.approx(5e-5)
    assert training["gradient_accumulation_steps"] == 2 and training["seed"] == 7
    lora = spec.lora_config()
    assert lora["r"] == 8 and lora["alpha"] == 16.0 and lora["use_dora"] is True
    assert lora["use_rslora"] is True and lora["dropout"] == pytest.approx(0.05)
    assert lora["target_modules"] == ["q_proj", "v_proj"]


def test_train_spec_defaults_are_the_command_lines():
    """No default is restated in Python: the config a bare spec yields is the spec's."""
    spec = pmetal.TrainSpec(model="m", dataset="d")
    training = spec.training_config()
    for name, key in [
        ("learning_rate", "learning_rate"),
        ("batch_size", "batch_size"),
        ("epochs", "num_epochs"),
        ("weight_decay", "weight_decay"),
        ("max_grad_norm", "max_grad_norm"),
        ("warmup_steps", "warmup_steps"),
        ("gradient_accumulation_steps", "gradient_accumulation_steps"),
    ]:
        assert training[key] == getattr(spec, name), name
    lora = spec.lora_config()
    assert lora["r"] == spec.lora_r and lora["alpha"] == spec.lora_alpha


@pytest.mark.parametrize(
    "name,field", [("optimizer", "optimizer"), ("lr_schedule", "lr_schedule"), ("wsd_decay_style", "wsd_decay_style")]
)
def test_every_enum_option_of_the_spec_is_one_the_trainer_parses(name, field):
    info = next(f for f in pmetal.TrainSpec.fields() if f.name == name)
    assert info.options
    for option in info.options:
        spec = pmetal.TrainSpec(model="m", dataset="d", **{field: option})
        assert isinstance(spec.training_config(), dict), f"{field}={option}"


def test_an_unknown_optimizer_is_an_error_not_a_default():
    spec = pmetal.TrainSpec(model="m", dataset="d", optimizer="sgdd")
    with pytest.raises(pmetal.ConfigError):
        spec.training_config()
    with pytest.raises(pmetal.SpecValidationError):
        pmetal.Trainer(spec)


def test_the_cli_and_python_agree_on_the_subcommand_and_argv():
    spec = pmetal.TrainSpec(model="m", dataset="d", epochs=2)
    argv = spec.to_argv()
    assert argv[argv.index("--model") + 1] == "m"
    assert argv[argv.index("--epochs") + 1] == "2"
    assert pmetal.TrainSpec.subcommand == "train"
    assert pmetal.DflashSpec.subcommand == "dflash"


# -- removed options stay removed --------------------------------------------------------------------


def test_removed_options_are_gone():
    every_field = {f.name for cls in SPECS for f in cls.fields()}
    assert not {n for n in every_field if "flash_attention" in n or "flash_attn" in n}
    assert not [f.name for f in pmetal.GrpoSpec.fields() if "specul" in f.name]
    for name in ("use_flash_attention", "flash_attention", "speculative"):
        with pytest.raises(TypeError):
            pmetal.GrpoSpec(**{name: True})
        with pytest.raises(TypeError):
            pmetal.TrainSpec(**{name: True})


def test_python_exports_no_standalone_config_classes():
    """The old LoraConfig/TrainingConfig/DataLoaderConfig re-declared the specs' defaults."""
    for name in (
        "LoraConfig",
        "TrainingConfig",
        "DataLoaderConfig",
        "Dtype",
        "Quantization",
        "LoraBias",
        "LrSchedulerType",
        "OptimizerType",
        "DatasetFormat",
    ):
        assert not hasattr(pmetal, name), name
