import hashlib
from pathlib import Path

import numpy as np
import pytest

from sketchsense.preprocessing import (
    EmptySketchError,
    normalize_canvas_rgba,
    normalize_dataset_bitmap,
)
from sketchsense.preprocessing.fixtures import validate_fixtures


def white_canvas(height: int = 32, width: int = 32) -> np.ndarray:
    return np.full((height, width, 4), 255, dtype=np.uint8)


def test_dataset_bitmap_shape_and_range() -> None:
    image = np.zeros((28, 28), dtype=np.uint8)
    image[10:18, 12:16] = 255
    tensor = normalize_dataset_bitmap(image)
    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) == 0.0
    assert float(tensor.max()) == 1.0


def test_canvas_preprocessing_centers_square() -> None:
    image = white_canvas()
    image[8:24, 10:22, :3] = 0
    tensor = normalize_canvas_rgba(image)[0, 0]
    ys, xs = np.nonzero(tensor > 0.1)
    assert abs(float(xs.mean()) - 13.5) <= 0.5
    assert abs(float(ys.mean()) - 13.5) <= 0.5


def test_canvas_preprocessing_preserves_tall_aspect_ratio() -> None:
    image = white_canvas(40, 20)
    image[4:36, 8:12, :3] = 0
    tensor = normalize_canvas_rgba(image)[0, 0]
    ys, xs = np.nonzero(tensor > 0.1)
    assert np.ptp(ys) > np.ptp(xs) * 2


def test_empty_canvas_is_explicit() -> None:
    with pytest.raises(EmptySketchError):
        normalize_canvas_rgba(white_canvas())


def test_canvas_preprocessing_supports_vector_native_56_input() -> None:
    image = white_canvas(64, 64)
    image[12:52, 28:36, :3] = 0
    tensor = normalize_canvas_rgba(image, output_size=56, content_size=40)
    assert tensor.shape == (1, 1, 56, 56)
    assert float(tensor.max()) == 1.0


def test_committed_fixture_is_stable() -> None:
    fixture = Path(__file__).parents[2] / "fixtures" / "preprocessing.v1.json"
    assert validate_fixtures(fixture) == 3
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()
