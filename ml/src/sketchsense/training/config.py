"""Centralized compact-CNN training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260808
    dataset_version: str = "small-v1"
    class_manifest_version: str = "1.0.0"
    architecture_version: str = "compact-cnn-v1"
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 64
    max_epochs: int = 30
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 0.0005
    model_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
