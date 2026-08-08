"""Small deterministic dataset contact sheet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sketchsense.data.dataset import load_classes, validate_dataset


def create_sample_grid(dataset_dir: Path, output: Path, samples_per_class: int = 5) -> Path:
    validate_dataset(dataset_dir)
    with np.load(dataset_dir / "small-v1.npz") as payload:
        images, labels, splits = payload["images"], payload["labels"], payload["splits"]
    classes = load_classes()
    scale, label_width, row_height = 3, 116, 28 * 3 + 12
    grid = Image.new(
        "L", (label_width + samples_per_class * 28 * scale, len(classes) * row_height), 255
    )
    draw = ImageDraw.Draw(grid)
    for class_index, class_name in enumerate(classes):
        y = class_index * row_height
        draw.text((6, y + row_height // 2 - 6), class_name, fill=0)
        candidates = images[(labels == class_index) & (splits == 0)][:samples_per_class]
        for column, image in enumerate(candidates):
            tile = Image.fromarray(255 - image, mode="L").resize(
                (28 * scale, 28 * scale), Image.Resampling.NEAREST
            )
            grid.paste(tile, (label_width + column * 28 * scale, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output, optimize=True)
    return output
