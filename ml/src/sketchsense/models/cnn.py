"""Compact convolutional classifier for 28x28 sketches."""

from __future__ import annotations

from torch import Tensor, nn


class CompactSketchCNN(nn.Module):
    """Two convolution blocks followed by a compact classifier."""

    def __init__(self, class_count: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, class_count),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs))


class WidenedBatchNormCNN(nn.Module):
    """Higher-capacity candidate that remains comfortably browser-sized."""

    def __init__(self, class_count: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, class_count),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs))


class DepthwiseSketchCNN(nn.Module):
    """Small depthwise-separable candidate with global average pooling."""

    def __init__(self, class_count: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1, groups=32),
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1, groups=64),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs).flatten(1))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
