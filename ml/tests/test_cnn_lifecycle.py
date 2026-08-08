import json
from pathlib import Path

import numpy as np
import torch

from sketchsense.contracts.model_artifacts import MODEL_SIZE_TARGET, validate_model_manifest
from sketchsense.export.onnx import ABSOLUTE_TOLERANCE, validate_onnx_parity
from sketchsense.models.cnn import CompactSketchCNN, parameter_count
from sketchsense.training.config import TrainingConfig
from sketchsense.training.pipeline import load_checkpoint, split_training_validation

REPOSITORY_ROOT = Path(__file__).parents[2]
MODEL_DIR = REPOSITORY_ROOT / "artifacts" / "models"
EVALUATION_DIR = REPOSITORY_ROOT / "artifacts" / "evaluation"


def test_cnn_output_shape_and_parameter_budget() -> None:
    model = CompactSketchCNN()
    assert model(torch.zeros((2, 1, 28, 28))).shape == (2, 16)
    assert parameter_count(model) == 106_256


def test_training_configuration_is_versioned() -> None:
    config = TrainingConfig()
    assert config.architecture_version == "compact-cnn-v1"
    assert config.dataset_version == "small-v1"
    assert config.class_manifest_version == "1.0.0"
    assert config.seed == 20260808


def test_checkpoint_loads_and_matches_metadata() -> None:
    metadata = json.loads((MODEL_DIR / "training-metadata.v1.json").read_text(encoding="utf-8"))
    model = load_checkpoint(MODEL_DIR / "compact-cnn.v1.checkpoint.npz")
    assert model(torch.zeros((1, 1, 28, 28))).shape == (1, 16)
    assert metadata["architecture"]["parameters"] == parameter_count(model)
    assert metadata["dataset"]["test_samples_accessed"] == 0
    assert metadata["selection"]["best_epoch"] <= metadata["selection"]["epochs_completed"]


def test_training_boundary_excludes_test_samples() -> None:
    images = np.zeros((6, 28, 28), dtype=np.uint8)
    labels = np.arange(6, dtype=np.uint8)
    splits = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.uint8)
    (train_images, train_labels), (validation_images, validation_labels) = (
        split_training_validation(images, labels, splits)
    )
    assert train_labels.tolist() == [0, 1]
    assert validation_labels.tolist() == [2, 3]
    assert len(train_images) + len(validation_images) == 4


def test_evaluation_artifacts_have_complete_dimensions() -> None:
    summary = json.loads(
        (EVALUATION_DIR / "evaluation-summary.v1.json").read_text(encoding="utf-8")
    )
    confusion = json.loads(
        (EVALUATION_DIR / "confusion-matrix.v1.json").read_text(encoding="utf-8")
    )
    per_class = json.loads(
        (EVALUATION_DIR / "per-class-metrics.v1.json").read_text(encoding="utf-8")
    )
    assert summary["test_evaluations"] == 1
    assert summary["test_samples"] == 480
    assert len(confusion["matrix"]) == 16
    assert all(len(row) == 16 for row in confusion["matrix"])
    assert len(per_class["classes"]) == 16
    assert sum(sum(row) for row in confusion["matrix"]) == 480


def test_onnx_runtime_parity_and_size_budget() -> None:
    report_path = validate_onnx_parity(MODEL_DIR)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report["cases"]) == 3
    assert max(case["max_absolute_difference"] for case in report["cases"]) <= ABSOLUTE_TOLERANCE
    assert report["malformed_input_rejected"] is True
    assert (MODEL_DIR / "compact-cnn.v1.onnx").stat().st_size < MODEL_SIZE_TARGET


def test_model_manifest_checksums_and_sizes() -> None:
    result = validate_model_manifest(REPOSITORY_ROOT)
    assert result["artifacts"] == 10
    assert result["onnx_bytes"] == 441_021
