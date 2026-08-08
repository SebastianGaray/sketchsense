"""Deterministic cross-language preprocessing fixture generation and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from sketchsense.preprocessing.core import normalize_canvas_rgba


def fixture_inputs() -> dict[str, np.ndarray]:
    square = np.full((16, 20, 4), 255, dtype=np.uint8)
    square[4:12, 7:13, :3] = 0
    vertical = np.full((24, 12, 4), 255, dtype=np.uint8)
    vertical[2:22, 5:7, :3] = 0
    translucent = np.full((12, 18, 4), 255, dtype=np.uint8)
    translucent[3:9, 2:16, :3] = 0
    translucent[3:9, 2:16, 3] = 200
    return {
        "centered-rectangle": square,
        "vertical-stroke": vertical,
        "translucent-bar": translucent,
    }


def write_fixtures(path: Path) -> Path:
    cases = []
    for name, rgba in fixture_inputs().items():
        tensor = normalize_canvas_rgba(rgba)
        cases.append(
            {
                "name": name,
                "input_shape": list(rgba.shape),
                "rgba": rgba.reshape(-1).tolist(),
                "expected_shape": list(tensor.shape),
                "expected": [round(float(value), 8) for value in tensor.reshape(-1)],
                "expected_sha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
            }
        )
    payload = {"schema_version": "1.0.0", "tolerance": 1e-5, "cases": cases}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def validate_fixtures(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tolerance = float(payload["tolerance"])
    for case in payload["cases"]:
        rgba = np.asarray(case["rgba"], dtype=np.uint8).reshape(case["input_shape"])
        expected = np.asarray(case["expected"], dtype=np.float32).reshape(case["expected_shape"])
        actual = normalize_canvas_rgba(rgba)
        if not np.allclose(actual, expected, rtol=0.0, atol=tolerance):
            raise ValueError(f"Preprocessing fixture failed: {case['name']}")
        if hashlib.sha256(actual.tobytes()).hexdigest() != case["expected_sha256"]:
            raise ValueError(f"Preprocessing fixture checksum failed: {case['name']}")
    return len(payload["cases"])
