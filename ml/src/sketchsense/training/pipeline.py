"""Deterministic training, validation selection, and checkpoint lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from sketchsense.contracts.artifacts import save_npz_deterministic
from sketchsense.data.dataset import load_classes, validate_dataset
from sketchsense.models.cnn import CompactSketchCNN, parameter_count
from sketchsense.training.config import TrainingConfig


def set_deterministic_state(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def split_training_validation(
    images: NDArray[np.uint8], labels: NDArray[np.uint8], splits: NDArray[np.uint8]
) -> tuple[
    tuple[NDArray[np.uint8], NDArray[np.uint8]], tuple[NDArray[np.uint8], NDArray[np.uint8]]
]:
    """Return train/validation arrays while making test access impossible to do accidentally."""
    return (images[splits == 0], labels[splits == 0]), (images[splits == 1], labels[splits == 1])


def train_compact_cnn(dataset_dir: Path, artifact_dir: Path, config: TrainingConfig) -> Path:
    validate_dataset(dataset_dir)
    set_deterministic_state(config.seed)
    with np.load(dataset_dir / "small-v1.npz") as payload:
        images, labels, splits = payload["images"], payload["labels"], payload["splits"]
        source_indices = payload["source_indices"]
    (train_images, train_labels), (validation_images, validation_labels) = (
        split_training_validation(images, labels, splits)
    )
    train_loader = _loader(train_images, train_labels, config.batch_size, True, config.seed)
    validation_loader = _loader(
        validation_images, validation_labels, config.batch_size, False, config.seed
    )

    model = CompactSketchCNN(len(load_classes()))
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_function = nn.CrossEntropyLoss()
    best_state: dict[str, Tensor] | None = None
    best_epoch, best_validation_loss, stale_epochs = 0, float("inf"), 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss, train_accuracy = _run_epoch(model, train_loader, loss_function, optimizer)
        validation_loss, validation_accuracy = _run_epoch(
            model, validation_loader, loss_function, None
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_loss < best_validation_loss - config.early_stopping_min_delta:
            best_validation_loss, best_epoch, stale_epochs = validation_loss, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "compact-cnn.v1.checkpoint.npz"
    save_checkpoint(model, checkpoint_path)
    metadata: dict[str, Any] = {
        "schema_version": "1.0.0",
        "model_version": config.model_version,
        "configuration": config.to_dict(),
        "architecture": {
            "name": "CompactSketchCNN",
            "layers": [
                "Conv2d(1,16,3,pad=1)",
                "ReLU",
                "MaxPool2d(2)",
                "Conv2d(16,32,3,pad=1)",
                "ReLU",
                "MaxPool2d(2)",
                "Flatten",
                "Linear(1568,64)",
                "ReLU",
                "Linear(64,16)",
            ],
            "parameters": parameter_count(model),
            "output": "logits",
        },
        "dataset": {
            "version": config.dataset_version,
            "split_checksum": split_checksum(labels, source_indices, splits),
            "train_samples": len(train_images),
            "validation_samples": len(validation_images),
            "test_samples_accessed": 0,
        },
        "selection": {
            "criterion": "minimum validation cross-entropy",
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "epochs_completed": len(history),
            "stopped_early": len(history) < config.max_epochs,
        },
        "history": history,
        "checkpoint": {
            "file": checkpoint_path.name,
            "sha256": _sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pytorch": torch.__version__,
            "device": "cpu",
            "platform": platform.platform(),
        },
        "determinism": {
            "algorithms_enforced": True,
            "data_loader_workers": 0,
            "limitations": (
                "CPU operations are deterministic for this environment; exact floating-point "
                "results can differ across PyTorch, BLAS, CPU, or operating-system versions."
            ),
        },
        "trained_at": _timestamp(),
    }
    metadata_path = artifact_dir / "training-metadata.v1.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata_path


def save_checkpoint(model: nn.Module, path: Path) -> None:
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    save_npz_deterministic(path, arrays)


def load_checkpoint(path: Path) -> CompactSketchCNN:
    model = CompactSketchCNN(len(load_classes()))
    with np.load(path) as payload:
        state = {name: torch.from_numpy(payload[name].copy()) for name in payload.files}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def split_checksum(
    labels: NDArray[np.uint8], source_indices: NDArray[np.int64], splits: NDArray[np.uint8]
) -> str:
    digest = hashlib.sha256(b"sketchsense-split-checksum-v1")
    for values in (labels, source_indices, splits):
        digest.update(values.tobytes())
    return digest.hexdigest()


def _loader(
    images: NDArray[np.uint8], labels: NDArray[np.uint8], batch_size: int, shuffle: bool, seed: int
) -> DataLoader[tuple[Tensor, ...]]:
    inputs = torch.from_numpy(images.astype(np.float32) / np.float32(255.0)).unsqueeze(1)
    targets = torch.from_numpy(labels.astype(np.int64))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(inputs, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, ...]],
    loss_function: nn.Module,
    optimizer: Adam | None,
) -> tuple[float, float]:
    model.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for inputs, targets in loader:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = loss_function(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            correct += int((logits.argmax(dim=1) == targets).sum())
            count += len(targets)
    return total_loss / count, correct / count


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _timestamp() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    moment = datetime.fromtimestamp(int(epoch), UTC) if epoch else datetime.now(UTC)
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")
