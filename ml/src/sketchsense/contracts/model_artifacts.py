"""Generate and validate the versioned model artifact manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

MODEL_SIZE_TARGET = 5 * 1024 * 1024


def create_model_manifest(repository_root: Path) -> Path:
    files = {
        "onnx_model": repository_root / "artifacts" / "models" / "compact-cnn.v1.onnx",
        "checkpoint": repository_root / "artifacts" / "models" / "compact-cnn.v1.checkpoint.npz",
        "training_metadata": repository_root / "artifacts" / "models" / "training-metadata.v1.json",
        "onnx_parity": repository_root / "artifacts" / "models" / "onnx-parity.v1.json",
        "evaluation_summary": repository_root
        / "artifacts"
        / "evaluation"
        / "evaluation-summary.v1.json",
        "per_class_metrics": repository_root
        / "artifacts"
        / "evaluation"
        / "per-class-metrics.v1.json",
        "confusion_matrix": repository_root
        / "artifacts"
        / "evaluation"
        / "confusion-matrix.v1.json",
        "class_manifest": repository_root
        / "ml"
        / "src"
        / "sketchsense"
        / "contracts"
        / "classes.v1.json",
        "preprocessing_contract": repository_root
        / "ml"
        / "src"
        / "sketchsense"
        / "contracts"
        / "preprocessing.v1.json",
        "model_card": repository_root / "docs" / "model-card.md",
    }
    artifacts = {
        name: {
            "path": path.relative_to(repository_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for name, path in files.items()
    }
    payload = {
        "schema_version": "1.0.0",
        "model_version": "1.0.0",
        "dataset_version": "small-v1",
        "class_manifest_version": "1.0.0",
        "preprocessing_version": "1.0.0",
        "artifacts": artifacts,
    }
    output = repository_root / "artifacts" / "model-manifest.v1.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_model_manifest(repository_root, output)
    return output


def validate_model_manifest(
    repository_root: Path, manifest_path: Path | None = None
) -> dict[str, int]:
    path = manifest_path or repository_root / "artifacts" / "model-manifest.v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_path = (
        repository_root
        / "ml"
        / "src"
        / "sketchsense"
        / "contracts"
        / "model-artifact-manifest.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(payload)
    for artifact in payload["artifacts"].values():
        artifact_path = repository_root / artifact["path"]
        if (
            artifact_path.stat().st_size != artifact["bytes"]
            or _sha256(artifact_path) != artifact["sha256"]
        ):
            raise ValueError(f"Artifact integrity failed: {artifact['path']}")
    model_size = int(payload["artifacts"]["onnx_model"]["bytes"])
    if model_size >= MODEL_SIZE_TARGET:
        raise ValueError(f"ONNX model exceeds the 5 MB target: {model_size} bytes")
    return {"artifacts": len(payload["artifacts"]), "onnx_bytes": model_size}


def _sha256(path: Path) -> str:
    if path.suffix in {".json", ".md"}:
        canonical = path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()
