"""One-time held-out evaluation for the validation-selected CNN checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from sketchsense.data.dataset import load_classes, validate_dataset
from sketchsense.evaluation.metrics import detailed_classification_report
from sketchsense.training.pipeline import load_checkpoint


def evaluate_selected_model(
    dataset_dir: Path, model_dir: Path, evaluation_dir: Path, baseline_report: Path
) -> Path:
    """Evaluate a checkpoint once; reuse the report when its checksum is unchanged."""
    validate_dataset(dataset_dir)
    checkpoint = model_dir / "compact-cnn.v1.checkpoint.npz"
    checkpoint_hash = _sha256(checkpoint)
    summary_path = evaluation_dir / "evaluation-summary.v1.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("checkpoint_sha256") == checkpoint_hash:
            return summary_path

    with np.load(dataset_dir / "small-v1.npz") as payload:
        mask = payload["splits"] == 2
        images = payload["images"][mask]
        labels = payload["labels"][mask]
    inputs = torch.from_numpy(images.astype(np.float32) / np.float32(255.0)).unsqueeze(1)
    model = load_checkpoint(checkpoint)
    with torch.inference_mode():
        probabilities = torch.softmax(model(inputs), dim=1).numpy()
    class_names = load_classes()
    report = detailed_classification_report(labels, probabilities, class_names)
    confusion = cast(list[list[int]], report.pop("confusion_matrix"))
    per_class = cast(dict[str, dict[str, float | int]], report.pop("per_class"))
    baseline = json.loads(baseline_report.read_text(encoding="utf-8"))["test"]["metrics"]
    metrics = cast(dict[str, float], report["metrics"])
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        evaluation_dir / "per-class-metrics.v1.json",
        {"schema_version": "1.0.0", "model_version": "1.0.0", "classes": per_class},
    )
    _write_json(
        evaluation_dir / "confusion-matrix.v1.json",
        {
            "schema_version": "1.0.0",
            "model_version": "1.0.0",
            "class_order": list(class_names),
            "matrix": confusion,
        },
    )
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "model_version": "1.0.0",
        "dataset_version": "small-v1",
        "checkpoint_sha256": checkpoint_hash,
        "test_samples": len(labels),
        "test_evaluations": 1,
        "metrics": metrics,
        "baseline_metrics": baseline,
        "absolute_delta_from_baseline": {
            key: float(metrics[key]) - float(baseline[key]) for key in metrics
        },
        "most_common_confusions": _common_confusions(np.asarray(confusion), class_names, limit=8),
        "evaluated_at": _timestamp(),
        "selection_note": (
            "The checkpoint was selected by validation loss before this single held-out "
            "test evaluation."
        ),
    }
    _write_json(summary_path, summary)
    return summary_path


def _common_confusions(
    matrix: np.ndarray, classes: tuple[str, ...], limit: int
) -> list[dict[str, object]]:
    candidates: list[tuple[int, int, int]] = []
    for actual in range(len(classes)):
        for predicted in range(len(classes)):
            if actual != predicted and int(matrix[actual, predicted]) > 0:
                candidates.append((int(matrix[actual, predicted]), actual, predicted))
    candidates.sort(key=lambda item: (-item[0], classes[item[1]], classes[item[2]]))
    return [
        {"actual": classes[actual], "predicted": classes[predicted], "count": count}
        for count, actual, predicted in candidates[:limit]
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    moment = datetime.fromtimestamp(int(epoch), UTC) if epoch else datetime.now(UTC)
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")
