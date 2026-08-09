"""Pure preprocessing shared by training and future browser parity fixtures."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatTensor = NDArray[np.float32]
Uint8Image = NDArray[np.uint8]


class EmptySketchError(ValueError):
    """Raised when canvas input contains no foreground pixels."""


def normalize_dataset_bitmap(image: Uint8Image) -> FloatTensor:
    """Normalize an official Quick, Draw! 28x28 white-on-black bitmap."""
    if image.shape != (28, 28):
        raise ValueError(f"Expected a 28x28 bitmap, received {image.shape}")
    return (image.astype(np.float32) / np.float32(255.0)).reshape(1, 1, 28, 28)


def normalize_canvas_rgba(
    rgba: Uint8Image, output_size: int = 28, content_size: int | None = None
) -> FloatTensor:
    """Apply the version 1 canvas crop, resize, center, and normalization contract."""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Expected an HxWx4 RGBA image, received {rgba.shape}")
    if rgba.dtype != np.uint8:
        raise ValueError("Canvas input must use uint8 channels")

    rgb = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / np.float32(255.0)
    composited = np.rint(rgb * alpha + np.float32(255.0) * (np.float32(1.0) - alpha))
    luminance = np.rint(
        np.float32(0.2126) * composited[..., 0]
        + np.float32(0.7152) * composited[..., 1]
        + np.float32(0.0722) * composited[..., 2]
    ).astype(np.uint8)

    foreground_y, foreground_x = np.nonzero(luminance < 250)
    if foreground_x.size == 0:
        raise EmptySketchError("The sketch does not contain foreground pixels")

    x0, x1 = int(foreground_x.min()), int(foreground_x.max())
    y0, y1 = int(foreground_y.min()), int(foreground_y.max())
    width, height = x1 - x0 + 1, y1 - y0 + 1
    padding = max(2, math.ceil(max(width, height) * 0.1))
    x0, x1 = max(0, x0 - padding), min(luminance.shape[1] - 1, x1 + padding)
    y0, y1 = max(0, y0 - padding), min(luminance.shape[0] - 1, y1 + padding)
    crop = luminance[y0 : y1 + 1, x0 : x1 + 1]

    if output_size < 8:
        raise ValueError("Output size must be at least 8")
    content_size = content_size or round(output_size * 20 / 28)
    if not 1 <= content_size <= output_size:
        raise ValueError("Content size must fit inside the output")
    scale = min(content_size / crop.shape[1], content_size / crop.shape[0])
    target_width = max(1, min(content_size, round(crop.shape[1] * scale)))
    target_height = max(1, min(content_size, round(crop.shape[0] * scale)))
    resized = _resize_bilinear_half_pixel(crop, target_height, target_width)

    canvas = np.full((output_size, output_size), 255, dtype=np.uint8)
    left = (output_size - target_width) // 2
    top = (output_size - target_height) // 2
    canvas[top : top + target_height, left : left + target_width] = resized
    normalized = (np.float32(255.0) - canvas.astype(np.float32)) / np.float32(255.0)
    return normalized.reshape(1, 1, output_size, output_size)


def _resize_bilinear_half_pixel(image: Uint8Image, height: int, width: int) -> Uint8Image:
    source_height, source_width = image.shape
    output = np.empty((height, width), dtype=np.uint8)
    for target_y in range(height):
        source_y = (target_y + 0.5) * source_height / height - 0.5
        y0 = max(0, min(source_height - 1, math.floor(source_y)))
        y1 = max(0, min(source_height - 1, y0 + 1))
        wy = max(0.0, source_y - math.floor(source_y))
        for target_x in range(width):
            source_x = (target_x + 0.5) * source_width / width - 0.5
            x0 = max(0, min(source_width - 1, math.floor(source_x)))
            x1 = max(0, min(source_width - 1, x0 + 1))
            wx = max(0.0, source_x - math.floor(source_x))
            top = float(image[y0, x0]) * (1.0 - wx) + float(image[y0, x1]) * wx
            bottom = float(image[y1, x0]) * (1.0 - wx) + float(image[y1, x1]) * wx
            output[target_y, target_x] = np.uint8(round(top * (1.0 - wy) + bottom * wy))
    return output
