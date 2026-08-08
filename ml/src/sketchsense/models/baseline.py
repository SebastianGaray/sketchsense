"""Reproducible multinomial logistic-regression baseline."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from sketchsense.contracts.artifacts import save_npz_deterministic
from sketchsense.data.dataset import load_classes, validate_dataset
from sketchsense.evaluation.metrics import detailed_classification_report


def fit_model(
    features: NDArray[np.float32], labels: NDArray[np.uint8], seed: int
) -> LogisticRegression:
    """Fit the fixed baseline configuration."""
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=seed, tol=1e-4)
    model.fit(features, labels)
    return model


def train_baseline(dataset_dir: Path, artifact_dir: Path, seed: int = 20260808) -> Path:
    validate_dataset(dataset_dir)
    images, labels, splits = _load(dataset_dir)
    train_mask, validation_mask, test_mask = splits == 0, splits == 1, splits == 2
    model = fit_model(_features(images[train_mask]), labels[train_mask], seed)
    validation_probabilities = model.predict_proba(_features(images[validation_mask]))
    test_probabilities = model.predict_proba(_features(images[test_mask]))
    class_names = load_classes()

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "logistic-regression-baseline.v1.npz"
    save_npz_deterministic(
        model_path,
        {
            "coef": model.coef_.astype(np.float32),
            "intercept": np.asarray(model.intercept_, dtype=np.float32),
            "classes": model.classes_.astype(np.int16),
        },
    )
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "kind": "baseline",
        "name": "multinomial-logistic-regression",
        "intended_use": "Development comparison only; not a production model or browser artifact.",
        "seed": seed,
        "features": 784,
        "classes": list(class_names),
        "training": {
            "samples": int(train_mask.sum()),
            "solver": "lbfgs",
            "regularization": "L2",
            "C": 1.0,
            "max_iter": 300,
            "iterations": [int(value) for value in model.n_iter_],
            "converged": bool(np.all(model.n_iter_ < model.max_iter)),
        },
        "validation": {
            "samples": int(validation_mask.sum()),
            **detailed_classification_report(
                labels[validation_mask], validation_probabilities, class_names
            ),
        },
        "test": {
            "samples": int(test_mask.sum()),
            **detailed_classification_report(labels[test_mask], test_probabilities, class_names),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "model_artifact": {
            "file": model_path.name,
            "bytes": model_path.stat().st_size,
            "sha256": _sha256(model_path),
        },
    }
    report_path = artifact_dir / "baseline-summary.v1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def validate_baseline(artifact_dir: Path) -> dict[str, float]:
    report = json.loads((artifact_dir / "baseline-summary.v1.json").read_text(encoding="utf-8"))
    model = artifact_dir / report["model_artifact"]["file"]
    if (
        _sha256(model) != report["model_artifact"]["sha256"]
        or model.stat().st_size != report["model_artifact"]["bytes"]
    ):
        raise ValueError("Baseline artifact checksum or size is invalid")
    with np.load(model) as payload:
        if payload["coef"].shape != (16, 784) or payload["intercept"].shape != (16,):
            raise ValueError("Baseline parameter shapes are invalid")
    return {key: float(value) for key, value in report["test"]["metrics"].items()}


def _load(dataset_dir: Path) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]:
    with np.load(dataset_dir / "small-v1.npz") as payload:
        return payload["images"].copy(), payload["labels"].copy(), payload["splits"].copy()


def _features(images: NDArray[np.uint8]) -> NDArray[np.float32]:
    return images.reshape(len(images), -1).astype(np.float32) / np.float32(255.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
