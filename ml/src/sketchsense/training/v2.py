"""Reproducible model-v2 data, augmentation, experiment, and release pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from sketchsense.contracts.artifacts import save_npz_deterministic
from sketchsense.data.dataset import BASE_URL, load_classes, read_bitmap_range, read_npy_header
from sketchsense.evaluation.metrics import classification_metrics, detailed_classification_report
from sketchsense.models.cnn import (
    CompactSketchCNN,
    DepthwiseSketchCNN,
    WidenedBatchNormCNN,
    parameter_count,
)
from sketchsense.preprocessing.core import normalize_canvas_rgba
from sketchsense.preprocessing.fixtures import fixture_inputs
from sketchsense.training.pipeline import save_checkpoint, set_deterministic_state

V2_SPLITS = {"train": 800, "validation": 100, "test": 100}
V2_SEEDS = (20260808, 20260809, 20260810)
MODEL_FACTORIES: dict[str, type[nn.Module]] = {
    "compact": CompactSketchCNN,
    "widened-bn": WidenedBatchNormCNN,
    "depthwise": DepthwiseSketchCNN,
}


@dataclass(frozen=True)
class V2TrainingConfig:
    architecture: str
    seed: int
    augment: bool = True
    max_epochs: int = 6
    patience: int = 2
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.0001


def prepare_v2_dataset(output_dir: Path, v1_manifest: Path) -> Path:
    """Fetch a fresh balanced bitmap profile and prove identifier non-overlap with v1."""
    output_dir.mkdir(parents=True, exist_ok=True)
    old = json.loads(v1_manifest.read_text(encoding="utf-8"))
    excluded = {str(sample["sample_id"]) for sample in old["samples"]}
    classes = load_classes()
    images: list[NDArray[np.uint8]] = []
    labels: list[int] = []
    indices: list[int] = []
    splits: list[int] = []
    records: list[dict[str, object]] = []
    sources: dict[str, object] = {}
    count = sum(V2_SPLITS.values())
    for class_index, class_name in enumerate(classes):
        url = f"{BASE_URL}/{class_name.replace(' ', '%20')}.npy"
        header = read_npy_header(url)
        digest = hashlib.sha256(f"sketchsense-medium-v2:{class_name}".encode()).digest()
        start = int.from_bytes(digest[:8], "big") % (header.count - count + 1)
        while any(f"{class_name}:{index}" in excluded for index in range(start, start + count)):
            start = (start + count) % (header.count - count + 1)
        class_images = read_bitmap_range(url, header, start, count)
        source_indices = np.arange(start, start + count, dtype=np.int64)
        order = sorted(
            range(count),
            key=lambda position: hashlib.sha256(
                f"sketchsense-split-v2:{class_name}:{int(source_indices[position])}".encode()
            ).digest(),
        )
        assignments = np.empty(count, dtype=np.uint8)
        cursor = 0
        for code, split_count in enumerate(V2_SPLITS.values()):
            assignments[order[cursor : cursor + split_count]] = code
            cursor += split_count
        for image, source_index, split_code in zip(
            class_images, source_indices, assignments, strict=True
        ):
            split_name = tuple(V2_SPLITS)[int(split_code)]
            sample_id = f"{class_name}:{int(source_index)}"
            records.append(
                {
                    "sample_id": sample_id,
                    "class_name": class_name,
                    "class_index": class_index,
                    "source_index": int(source_index),
                    "split": split_name,
                    "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                }
            )
            images.append(image)
            labels.append(class_index)
            indices.append(int(source_index))
            splits.append(int(split_code))
        sources[class_name] = {"url": url, "total_samples": header.count, "selected_start": start}
    cache = output_dir / "medium-v2.npz"
    save_npz_deterministic(
        cache,
        {
            "images": np.stack(images),
            "labels": np.asarray(labels, dtype=np.uint8),
            "source_indices": np.asarray(indices, dtype=np.int64),
            "splits": np.asarray(splits, dtype=np.uint8),
        },
    )
    manifest = {
        "schema_version": "1.0.0",
        "dataset": {
            "name": "Google Quick, Draw! Dataset",
            "source": BASE_URL,
            "license": "CC BY 4.0",
        },
        "profile": {"name": "medium-v2", "seed": 20260808, "samples_per_class": count},
        "classes": list(classes),
        "splits": V2_SPLITS,
        "source_arrays": sources,
        "cache": {"file": cache.name, "sha256": _sha256(cache)},
        "samples": records,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_v2_dataset(output_dir, v1_manifest)
    return path


def validate_v2_dataset(dataset_dir: Path, v1_manifest: Path) -> dict[str, int]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    old = json.loads(v1_manifest.read_text(encoding="utf-8"))
    old_ids = {str(item["sample_id"]) for item in old["samples"]}
    new_ids = {str(item["sample_id"]) for item in manifest["samples"]}
    if old_ids & new_ids:
        raise ValueError("v1 and v2 sample identifiers overlap")
    if len(new_ids) != len(load_classes()) * 1000:
        raise ValueError("medium-v2 must contain 1,000 unique samples per class")
    cache = dataset_dir / "medium-v2.npz"
    if _sha256(cache) != manifest["cache"]["sha256"]:
        raise ValueError("medium-v2 cache checksum mismatch")
    with np.load(cache) as payload:
        labels, splits = payload["labels"], payload["splits"]
    totals: dict[str, int] = {}
    for code, (name, expected) in enumerate(V2_SPLITS.items()):
        totals[name] = int(np.sum(splits == code))
        for class_index in range(len(load_classes())):
            if int(np.sum((splits == code) & (labels == class_index))) != expected:
                raise ValueError(f"Unbalanced {name} split for class {class_index}")
    return totals


def write_v2_dataset_summary(dataset_dir: Path, v1_manifest: Path, output: Path) -> Path:
    totals = validate_v2_dataset(dataset_dir, v1_manifest)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    old = json.loads(v1_manifest.read_text(encoding="utf-8"))
    summary = {key: value for key, value in manifest.items() if key != "samples"}
    summary["split_totals"] = totals
    summary["sample_records"] = len(manifest["samples"])
    summary["v1_excluded_sample_ids"] = len(old["samples"])
    summary["v1_v2_identifier_overlap"] = 0
    summary["augmentation"] = {
        "version": "canvas-domain-v2",
        "rotation_degrees": [-8, 8],
        "scale": [0.88, 1.12],
        "translation_fraction": [-0.09, 0.09],
        "stroke_variants": ["thin", "original", "thick"],
        "flips": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def prepare_release_test(
    dataset_dir: Path, output_dir: Path, profile_name: str = "release-test-v2"
) -> Path:
    """Create a replacement locked test after an artifact-level candidate rejection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    medium = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    excluded = {str(item["sample_id"]) for item in medium["samples"]}
    images: list[NDArray[np.uint8]] = []
    labels: list[int] = []
    records: list[dict[str, object]] = []
    for class_index, class_name in enumerate(load_classes()):
        url = f"{BASE_URL}/{class_name.replace(' ', '%20')}.npy"
        header = read_npy_header(url)
        digest = hashlib.sha256(f"sketchsense-{profile_name}:{class_name}".encode()).digest()
        start = int.from_bytes(digest[:8], "big") % (header.count - 100 + 1)
        while any(f"{class_name}:{index}" in excluded for index in range(start, start + 100)):
            start = (start + 100) % (header.count - 100 + 1)
        class_images = read_bitmap_range(url, header, start, 100)
        for offset, image in enumerate(class_images):
            source_index = start + offset
            records.append(
                {
                    "sample_id": f"{class_name}:{source_index}",
                    "class_name": class_name,
                    "source_index": source_index,
                    "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                }
            )
            images.append(image)
            labels.append(class_index)
    cache = output_dir / f"{profile_name}.npz"
    save_npz_deterministic(
        cache,
        {"images": np.stack(images), "labels": np.asarray(labels, dtype=np.uint8)},
    )
    manifest = {
        "schema_version": "2.0.0",
        "profile_name": profile_name,
        "purpose": (
            "locked replacement test after the validation-selected widened model failed latency"
        ),
        "samples_per_class": 100,
        "sample_count": len(records),
        "excluded_medium_v2_ids": len(excluded),
        "medium_v2_overlap": 0,
        "cache_sha256": _sha256(cache),
        "samples": records,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def evaluate_compact_release(
    release_test_dir: Path,
    checkpoint: Path,
    release_dir: Path,
    test_profile: str = "release-test-v2",
    seed: int = 20260810,
) -> Path:
    model = CompactSketchCNN(len(load_classes()))
    with np.load(checkpoint) as payload:
        model.load_state_dict(
            {name: torch.from_numpy(payload[name].copy()) for name in payload.files}
        )
    model.eval()
    with np.load(release_test_dir / f"{test_profile}.npz") as payload:
        images, labels = payload["images"], payload["labels"]
    inputs = torch.from_numpy(images.astype(np.float32) / 255).unsqueeze(1)
    with torch.inference_mode():
        probabilities = torch.softmax(model(inputs), dim=1).numpy()
    report = detailed_classification_report(labels, probabilities, load_classes())
    per_class = cast(dict[str, dict[str, float]], report["per_class"])
    metrics = cast(dict[str, float], report["metrics"])
    worst = min(item["recall"] for item in per_class.values())
    gates = {
        "accuracy": metrics["accuracy"] >= 0.80,
        "macro_f1": metrics["macro_f1"] >= 0.80,
        "top_3_accuracy": metrics["top_3_accuracy"] >= 0.92,
        "worst_class_recall": worst >= 0.60,
    }
    release_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = release_dir / "compact-cnn.v2.checkpoint.npz"
    save_checkpoint(model, final_checkpoint)
    summary = {
        "schema_version": "2.0.0",
        "model_version": "2.0.0",
        "dataset_version": "medium-v2-plus-fresh-release-test",
        "selected_architecture": "compact",
        "selected_seed": seed,
        "selection_reason": (
            "highest-validation compact augmented run selected after the widened candidate "
            "failed the pre-release latency gate"
        ),
        "checkpoint_sha256": _sha256(final_checkpoint),
        "test_samples": len(labels),
        "test_evaluations": 1,
        "metrics": metrics,
        "expected_calibration_error": _expected_calibration_error(labels, probabilities),
        "bootstrap_95_percent_ci": _bootstrap_intervals(labels, probabilities),
        "worst_class_recall": worst,
        "per_class": per_class,
        "confusion_matrix": report["confusion_matrix"],
        "release_gates": gates,
        "quality_gates_passed": all(gates.values()),
    }
    path = release_dir / "evaluation-summary.v2.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def augment_canvas_batch(inputs: Tensor, seed: int) -> Tensor:
    """Apply bounded affine, rasterization, and stroke-width variation."""
    generator = torch.Generator(device=inputs.device).manual_seed(seed)
    count = inputs.shape[0]
    angles = (torch.rand(count, generator=generator) - 0.5) * math.radians(16)
    scales = 0.88 + torch.rand(count, generator=generator) * 0.24
    translations = (torch.rand((count, 2), generator=generator) - 0.5) * 0.18
    theta = torch.zeros((count, 2, 3), dtype=inputs.dtype)
    theta[:, 0, 0] = scales * torch.cos(angles)
    theta[:, 0, 1] = -scales * torch.sin(angles)
    theta[:, 1, 0] = scales * torch.sin(angles)
    theta[:, 1, 1] = scales * torch.cos(angles)
    theta[:, :, 2] = translations
    grid = F.affine_grid(theta, list(inputs.shape), align_corners=False)
    output = F.grid_sample(inputs, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    variants = torch.randint(0, 2, (count,), generator=generator)
    thick = F.max_pool2d(output, 3, stride=1, padding=1)
    output = torch.where((variants == 1).view(-1, 1, 1, 1), thick, output)
    return torch.clamp(output, 0, 1)


def run_experiment_matrix(dataset_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(dataset_dir / "medium-v2.npz") as payload:
        images, labels, splits = payload["images"], payload["labels"], payload["splits"]
    results: list[dict[str, object]] = []
    configurations = [("compact", False)] + [
        (architecture, True) for architecture in MODEL_FACTORIES
    ]
    for architecture, augment in configurations:
        for seed in V2_SEEDS:
            config = V2TrainingConfig(architecture=architecture, seed=seed, augment=augment)
            run_name = f"{architecture}-{'aug' if augment else 'plain'}-{seed}"
            checkpoint = output_dir / f"{run_name}.npz"
            if checkpoint.exists():
                result = _evaluate_existing(checkpoint, images, labels, splits, config)
            else:
                result, model = _train_one(images, labels, splits, config)
                save_checkpoint(model, checkpoint)
            result["checkpoint"] = checkpoint.name
            result["checkpoint_sha256"] = _sha256(checkpoint)
            results.append(result)
    grouped: dict[str, dict[str, float]] = {}
    for architecture, augment in configurations:
        key = f"{architecture}-{'aug' if augment else 'plain'}"
        runs = [item for item in results if item["configuration"] == key]
        macro_f1_values = [cast(dict[str, float], run["validation"])["macro_f1"] for run in runs]
        worst_recall_values = [float(cast(float, run["worst_class_recall"])) for run in runs]
        grouped[key] = {
            "macro_f1_mean": float(np.mean(macro_f1_values)),
            "macro_f1_std": float(np.std(macro_f1_values)),
            "worst_recall_mean": float(np.mean(worst_recall_values)),
            "parameters": float(cast(int, runs[0]["parameters"])),
        }
    eligible = [item for item in results if str(item["configuration"]).endswith("-aug")]
    selected = max(
        eligible,
        key=lambda item: (
            cast(dict[str, float], item["validation"])["macro_f1"],
            float(cast(float, item["worst_class_recall"])),
        ),
    )
    payload_out = {
        "schema_version": "2.0.0",
        "dataset_version": "medium-v2",
        "seeds": list(V2_SEEDS),
        "runs": results,
        "aggregates": grouped,
        "selected_run": selected["run"],
        "selection_rule": (
            "highest validation macro F1, then worst-class recall, among augmented candidates"
        ),
        "test_samples_accessed": 0,
    }
    path = output_dir / "experiment-matrix.v2.json"
    path.write_text(json.dumps(payload_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def evaluate_and_release(dataset_dir: Path, experiment_dir: Path, release_dir: Path) -> Path:
    matrix = json.loads((experiment_dir / "experiment-matrix.v2.json").read_text(encoding="utf-8"))
    selected = next(item for item in matrix["runs"] if item["run"] == matrix["selected_run"])
    architecture = str(selected["architecture"])
    model = MODEL_FACTORIES[architecture](len(load_classes()))
    checkpoint = experiment_dir / str(selected["checkpoint"])
    with np.load(checkpoint) as payload:
        model.load_state_dict(
            {name: torch.from_numpy(payload[name].copy()) for name in payload.files}
        )
    model.eval()
    with np.load(dataset_dir / "medium-v2.npz") as payload:
        mask = payload["splits"] == 2
        images, labels = payload["images"][mask], payload["labels"][mask]
    inputs = torch.from_numpy(images.astype(np.float32) / 255).unsqueeze(1)
    with torch.inference_mode():
        probabilities = torch.softmax(model(inputs), dim=1).numpy()
    report = detailed_classification_report(labels, probabilities, load_classes())
    per_class = cast(dict[str, dict[str, float]], report["per_class"])
    metrics = cast(dict[str, float], report["metrics"])
    ece = _expected_calibration_error(labels, probabilities)
    worst = min(item["recall"] for item in per_class.values())
    gates = {
        "accuracy": metrics["accuracy"] >= 0.80,
        "macro_f1": metrics["macro_f1"] >= 0.80,
        "top_3_accuracy": metrics["top_3_accuracy"] >= 0.92,
        "worst_class_recall": worst >= 0.60,
    }
    release_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = release_dir / "compact-cnn.v2.checkpoint.npz"
    save_checkpoint(model, final_checkpoint)
    summary = {
        "schema_version": "2.0.0",
        "model_version": "2.0.0",
        "dataset_version": "medium-v2",
        "selected_architecture": architecture,
        "selected_seed": selected["seed"],
        "checkpoint_sha256": _sha256(final_checkpoint),
        "test_samples": len(labels),
        "test_evaluations": 1,
        "metrics": metrics,
        "expected_calibration_error": ece,
        "worst_class_recall": worst,
        "per_class": per_class,
        "confusion_matrix": report["confusion_matrix"],
        "release_gates": gates,
        "quality_gates_passed": all(gates.values()),
    }
    path = release_dir / "evaluation-summary.v2.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def export_v2_onnx(release_dir: Path, architecture: str) -> Path:
    model = MODEL_FACTORIES[architecture](len(load_classes()))
    checkpoint = release_dir / "compact-cnn.v2.checkpoint.npz"
    with np.load(checkpoint) as payload:
        model.load_state_dict(
            {name: torch.from_numpy(payload[name].copy()) for name in payload.files}
        )
    model.eval()
    output = release_dir / "compact-cnn.v2.onnx"
    torch.onnx.export(
        model,
        (torch.zeros((1, 1, 28, 28), dtype=torch.float32),),
        output,
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    return output


def validate_v2_onnx(release_dir: Path, v1_onnx: Path) -> Path:
    summary = json.loads((release_dir / "evaluation-summary.v2.json").read_text(encoding="utf-8"))
    architecture = str(summary["selected_architecture"])
    model = MODEL_FACTORIES[architecture](len(load_classes()))
    with np.load(release_dir / "compact-cnn.v2.checkpoint.npz") as payload:
        model.load_state_dict(
            {name: torch.from_numpy(payload[name].copy()) for name in payload.files}
        )
    model.eval()
    onnx_path = release_dir / "compact-cnn.v2.onnx"
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    old_session = ort.InferenceSession(str(v1_onnx), providers=["CPUExecutionProvider"])
    cases: list[dict[str, object]] = []
    tensors: list[NDArray[np.float32]] = []
    for name, rgba in fixture_inputs().items():
        tensor = normalize_canvas_rgba(rgba).astype(np.float32)
        tensors.append(tensor)
        with torch.inference_mode():
            expected = model(torch.from_numpy(tensor)).numpy()
        actual = np.asarray(session.run(["logits"], {"input": tensor})[0])
        difference = float(np.max(np.abs(expected - actual)))
        if difference > 1e-4:
            raise ValueError(f"v2 ONNX parity failed for {name}: {difference}")
        cases.append({"name": name, "max_absolute_difference": difference})
    sample = tensors[0]
    for runtime in (old_session, session):
        for _ in range(20):
            runtime.run(["logits"], {"input": sample})
    old_times = _runtime_timings(old_session, sample)
    new_times = _runtime_timings(session, sample)
    regression = new_times["p95_ms"] / old_times["p95_ms"] - 1
    report = {
        "schema_version": "2.0.0",
        "model_version": "2.0.0",
        "input_shape": [1, 1, 28, 28],
        "output_shape": [1, 16],
        "cases": cases,
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_sha256": _sha256(onnx_path),
        "runtime_cpu": {"v1": old_times, "v2": new_times, "p95_regression": regression},
        "latency_gate_passed": regression <= 0.20,
    }
    path = release_dir / "onnx-parity.v2.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _runtime_timings(
    session: ort.InferenceSession, tensor: NDArray[np.float32]
) -> dict[str, float]:
    measurements: list[float] = []
    for _ in range(200):
        started = time.perf_counter()
        session.run(["logits"], {"input": tensor})
        measurements.append((time.perf_counter() - started) * 1000)
    return {
        "p50_ms": float(np.percentile(measurements, 50)),
        "p95_ms": float(np.percentile(measurements, 95)),
    }


def _train_one(
    images: NDArray[np.uint8],
    labels: NDArray[np.uint8],
    splits: NDArray[np.uint8],
    config: V2TrainingConfig,
) -> tuple[dict[str, object], nn.Module]:
    set_deterministic_state(config.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    train_inputs = torch.from_numpy(images[splits == 0].astype(np.float32) / 255).unsqueeze(1)
    train_targets = torch.from_numpy(labels[splits == 0].astype(np.int64))
    if config.augment:
        augmented = augment_canvas_batch(train_inputs, config.seed)
        use_augmented = (torch.arange(len(train_inputs)) % 4 == 0).view(-1, 1, 1, 1)
        train_inputs = torch.where(use_augmented, augmented, train_inputs)
    validation_inputs = torch.from_numpy(images[splits == 1].astype(np.float32) / 255).unsqueeze(1)
    validation_labels = labels[splits == 1]
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    model = MODEL_FACTORIES[config.architecture](len(load_classes()))
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03 if config.augment else 0.0)
    best_state: dict[str, Tensor] | None = None
    best_loss, stale, best_epoch = float("inf"), 0, 0
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for batch, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(validation_inputs)
            validation_loss = float(
                loss_fn(logits, torch.from_numpy(validation_labels.astype(np.int64)))
            )
        if validation_loss < best_loss - 0.0005:
            best_loss, stale, best_epoch = validation_loss, 0, epoch
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("No v2 checkpoint produced")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(validation_inputs), dim=1).numpy()
    report = detailed_classification_report(validation_labels, probabilities, load_classes())
    per_class = cast(dict[str, dict[str, float]], report["per_class"])
    key = f"{config.architecture}-{'aug' if config.augment else 'plain'}"
    return (
        {
            "run": f"{key}-{config.seed}",
            "configuration": key,
            "architecture": config.architecture,
            "seed": config.seed,
            "augmentation": config.augment,
            "parameters": parameter_count(model),
            "best_epoch": best_epoch,
            "validation": report["metrics"],
            "worst_class_recall": min(item["recall"] for item in per_class.values()),
            "training_seconds": time.perf_counter() - started,
        },
        model,
    )


def _evaluate_existing(
    checkpoint: Path,
    images: NDArray[np.uint8],
    labels: NDArray[np.uint8],
    splits: NDArray[np.uint8],
    config: V2TrainingConfig,
) -> dict[str, object]:
    model = MODEL_FACTORIES[config.architecture](len(load_classes()))
    with np.load(checkpoint) as payload:
        model.load_state_dict(
            {name: torch.from_numpy(payload[name].copy()) for name in payload.files}
        )
    model.eval()
    inputs = torch.from_numpy(images[splits == 1].astype(np.float32) / 255).unsqueeze(1)
    validation_labels = labels[splits == 1]
    with torch.inference_mode():
        probabilities = torch.softmax(model(inputs), dim=1).numpy()
    report = detailed_classification_report(validation_labels, probabilities, load_classes())
    per_class = cast(dict[str, dict[str, float]], report["per_class"])
    key = f"{config.architecture}-{'aug' if config.augment else 'plain'}"
    return {
        "run": f"{key}-{config.seed}",
        "configuration": key,
        "architecture": config.architecture,
        "seed": config.seed,
        "augmentation": config.augment,
        "parameters": parameter_count(model),
        "best_epoch": None,
        "validation": report["metrics"],
        "worst_class_recall": min(item["recall"] for item in per_class.values()),
        "training_seconds": None,
        "resumed_from_checkpoint": True,
    }


def _expected_calibration_error(
    labels: NDArray[np.uint8], probabilities: NDArray[np.floating], bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    total = len(labels)
    result = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        mask = (confidence >= lower) & (confidence < lower + 1 / bins)
        if np.any(mask):
            result += (
                float(np.sum(mask))
                / total
                * abs(float(np.mean(correct[mask])) - float(np.mean(confidence[mask])))
            )
    return result


def _bootstrap_intervals(
    labels: NDArray[np.uint8], probabilities: NDArray[np.floating]
) -> dict[str, list[float]]:
    generator = np.random.default_rng(20260808)
    values: dict[str, list[float]] = {"accuracy": [], "macro_f1": []}
    for _ in range(250):
        indices = generator.integers(0, len(labels), size=len(labels))
        metrics = classification_metrics(labels[indices], probabilities[indices])
        values["accuracy"].append(metrics["accuracy"])
        values["macro_f1"].append(metrics["macro_f1"])
    return {
        name: [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]
        for name, samples in values.items()
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
