import hashlib
import json
from pathlib import Path

from sketchsense.data.dataset import load_classes

REPOSITORY_ROOT = Path(__file__).parents[2]


def test_v3_release_quality_and_resolution_decision() -> None:
    report = json.loads(
        (REPOSITORY_ROOT / "artifacts/evaluation/evaluation-summary.v3.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["metrics"]["macro_f1"] >= 0.94
    assert report["worst_class_recall"] >= 0.79
    assert report["per_class"]["cat"]["recall"] >= 0.89
    assert report["input_size"] == 28
    assert "memory safety budget" in report["resolution_decision"]


def test_v3_public_model_matches_manifest() -> None:
    public = REPOSITORY_ROOT / "apps/web/public/models"
    manifest = json.loads((public / "model-manifest.v3.json").read_text(encoding="utf-8"))
    model = public / "sketch-cnn.v3.onnx"
    assert manifest["model_version"] == "3.0.0"
    assert manifest["input"]["shape"] == [1, 1, 28, 28]
    assert manifest["onnx"]["bytes"] == model.stat().st_size
    assert manifest["onnx"]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()


def test_every_v3_example_is_model_validated() -> None:
    evidence = json.loads(
        (REPOSITORY_ROOT / "artifacts/evaluation/example-validation.v3.json").read_text(
            encoding="utf-8"
        )
    )
    assert {item["category"] for item in evidence} == set(load_classes())
    assert all(item["predicted"] == item["category"] for item in evidence)
    assert all(item["confidence"] >= 0.55 for item in evidence)
    for item in evidence:
        assert (REPOSITORY_ROOT / f"apps/web/public/examples/v3/{item['category']}.png").is_file()
