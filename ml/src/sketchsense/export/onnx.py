"""Modern PyTorch ONNX export and multi-fixture runtime parity validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument

from sketchsense.preprocessing.core import normalize_canvas_rgba
from sketchsense.preprocessing.fixtures import fixture_inputs
from sketchsense.training.pipeline import load_checkpoint

ABSOLUTE_TOLERANCE = 1e-5
RELATIVE_TOLERANCE = 1e-4


def export_onnx(model_dir: Path) -> Path:
    model = load_checkpoint(model_dir / "compact-cnn.v1.checkpoint.npz")
    output = model_dir / "compact-cnn.v1.onnx"
    example = torch.zeros((1, 1, 28, 28), dtype=torch.float32)
    torch.onnx.export(
        model,
        (example,),
        output,
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    onnx.checker.check_model(onnx.load(output))
    return output


def validate_onnx_parity(model_dir: Path) -> Path:
    checkpoint = model_dir / "compact-cnn.v1.checkpoint.npz"
    onnx_path = model_dir / "compact-cnn.v1.onnx"
    model = load_checkpoint(checkpoint)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].name != "input" or session.get_outputs()[0].name != "logits":
        raise ValueError("ONNX input/output names do not match the contract")
    if session.get_inputs()[0].shape != [1, 1, 28, 28]:
        raise ValueError(f"Unexpected ONNX input shape: {session.get_inputs()[0].shape}")

    cases: list[dict[str, object]] = []
    for name, rgba in fixture_inputs().items():
        tensor = normalize_canvas_rgba(rgba).astype(np.float32)
        with torch.inference_mode():
            torch_logits = model(torch.from_numpy(tensor)).numpy()
        onnx_logits = np.asarray(session.run(["logits"], {"input": tensor})[0], dtype=np.float32)
        difference = np.abs(torch_logits - onnx_logits)
        if torch_logits.shape != (1, 16) or onnx_logits.shape != (1, 16):
            raise ValueError(f"Unexpected output shape for {name}")
        if not np.allclose(
            torch_logits, onnx_logits, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
        ):
            raise ValueError(f"ONNX parity failed for {name}: {float(difference.max())}")
        cases.append(
            {
                "name": name,
                "max_absolute_difference": float(difference.max()),
                "output_shape": [1, 16],
            }
        )

    malformed_rejected = False
    try:
        session.run(["logits"], {"input": np.zeros((1, 1, 27, 28), dtype=np.float32)})
    except (InvalidArgument, ValueError):
        malformed_rejected = True
    if not malformed_rejected:
        raise ValueError("ONNX Runtime accepted malformed input")
    report = {
        "schema_version": "1.0.0",
        "model_version": "1.0.0",
        "input_name": "input",
        "input_shape": [1, 1, 28, 28],
        "input_dtype": "float32",
        "output_name": "logits",
        "output_shape": [1, 16],
        "softmax_location": "application-boundary",
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "cases": cases,
        "malformed_input_rejected": malformed_rejected,
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
    }
    report_path = model_dir / "onnx-parity.v1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path
