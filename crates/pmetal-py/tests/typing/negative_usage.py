"""Mistakes a type checker must catch.

Each line carries ``# type: ignore[code]`` for the error the checker has to
report. ``tests/test_stubs.py`` runs mypy with ``--warn-unused-ignores``, so a
line that stops being an error (a stub that went loose) fails the test.
"""

from __future__ import annotations

import pmetal

spec = pmetal.TrainSpec(model="m", dataset="d")

pmetal.TrainSpec(epochz=3)  # type: ignore[call-arg]
pmetal.TrainSpec(epochs="three")  # type: ignore[arg-type]
pmetal.TrainSpec(optimizer="not-an-optimizer")  # type: ignore[arg-type]
spec.epochs = "three"  # type: ignore[assignment]
spec.no_such_field = 1  # type: ignore[attr-defined]
lr: str = spec.learning_rate  # type: ignore[assignment]
pmetal.Trainer("not a spec")  # type: ignore[arg-type]
pmetal.Trainer(pmetal.MergeSpec())  # type: ignore[arg-type]
pmetal.finetune("m", "d", epochs="x")  # type: ignore[arg-type]
pmetal.finetune("m", "d", dataset="again")  # type: ignore[misc]
pmetal.infer("m", "p", image=["x.png"])  # type: ignore[call-arg]
pmetal.infer("m", "p", temperature="hot")  # type: ignore[arg-type]
pmetal.Model.load("m").generate(3)  # type: ignore[arg-type]
pmetal.Model.load("m").generate("p", temperature="hot")  # type: ignore[arg-type]
pmetal.download_model("m", verify="md5")  # type: ignore[arg-type]
pmetal.download_model("m", weights="onnx")  # type: ignore[arg-type]
pmetal.search_models("q", sort="oldest")  # type: ignore[arg-type]
pmetal.run("not a spec")  # type: ignore[arg-type]
