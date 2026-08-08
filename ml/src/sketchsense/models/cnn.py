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
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, class_count),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.features(inputs))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
