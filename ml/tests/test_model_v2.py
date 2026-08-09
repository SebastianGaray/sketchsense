import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from sketchsense.models.cnn import DepthwiseSketchCNN, WidenedBatchNormCNN, parameter_count
from sketchsense.training.v2 import V2_SPLITS, augment_canvas_batch, validate_v2_dataset


def test_v2_candidates_respect_output_and_parameter_contracts() -> None:
    inputs = torch.zeros((2, 1, 28, 28))
    for model in (WidenedBatchNormCNN(), DepthwiseSketchCNN()):
        assert model(inputs).shape == (2, 16)
        assert parameter_count(model) < 1_000_000


def test_canvas_augmentation_is_deterministic_and_bounded() -> None:
    inputs = torch.zeros((4, 1, 28, 28))
    inputs[:, :, 8:20, 13:15] = 1
    first = augment_canvas_batch(inputs, 42)
    second = augment_canvas_batch(inputs, 42)
    torch.testing.assert_close(first, second)
    assert first.shape == inputs.shape
    assert float(first.min()) >= 0
    assert float(first.max()) <= 1
    assert not torch.equal(first, inputs)


def test_v2_split_contract_and_local_dataset() -> None:
    assert V2_SPLITS == {"train": 800, "validation": 100, "test": 100}
    root = Path(__file__).parents[2]
    dataset = root / "ml" / "data" / "medium-v2"
    if dataset.exists():
        result = validate_v2_dataset(dataset, root / "ml/data/small-v1/manifest.json")
        assert result == {"train": 12_800, "validation": 1_600, "test": 1_600}


def test_augmented_batch_keeps_blank_background() -> None:
    blank = torch.zeros((3, 1, 28, 28), dtype=torch.float32)
    np.testing.assert_array_equal(augment_canvas_batch(blank, 7).numpy(), blank.numpy())


def test_released_v2_artifact_and_evidence_pass_contracts() -> None:
    root = Path(__file__).parents[2]
    manifest = json.loads(
        (root / "apps/web/public/models/model-manifest.v2.json").read_text(encoding="utf-8")
    )
    model = root / "apps/web/public/models/compact-cnn.v2.onnx"
    assert model.stat().st_size == manifest["onnx"]["bytes"] == 441_021
    assert hashlib.sha256(model.read_bytes()).hexdigest() == manifest["onnx"]["sha256"]
    evaluation = json.loads(
        (root / "artifacts/evaluation/evaluation-summary.v2.json").read_text(encoding="utf-8")
    )
    parity = json.loads((root / "artifacts/models/onnx-parity.v2.json").read_text(encoding="utf-8"))
    assert evaluation["quality_gates_passed"] is True
    assert evaluation["test_evaluations"] == 1
    assert evaluation["worst_class_recall"] >= 0.60
    assert parity["latency_gate_passed"] is True


def test_authored_browser_regression_suite_covers_every_class() -> None:
    root = Path(__file__).parents[2]
    payload = json.loads((root / "fixtures/authored-sketches.v2.json").read_text(encoding="utf-8"))
    cases = payload["cases"]
    assert {case["category"] for case in cases} == {
        "apple",
        "bicycle",
        "bird",
        "book",
        "car",
        "cat",
        "chair",
        "cloud",
        "cup",
        "dog",
        "fish",
        "flower",
        "house",
        "key",
        "star",
        "tree",
    }
    assert all(8 <= case["width"] <= 16 and len(case["points"]) >= 4 for case in cases)
