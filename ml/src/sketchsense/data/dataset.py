"""Bounded byte-range retrieval and deterministic dataset splitting."""

from __future__ import annotations

import ast
import hashlib
import json
import struct
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from jsonschema import Draft202012Validator
from numpy.typing import NDArray

from sketchsense.contracts.artifacts import save_npz_deterministic

SplitName = Literal["train", "validation", "test"]
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
SPLIT_COUNTS = {"train": 140, "validation": 30, "test": 30}


@dataclass(frozen=True)
class DatasetProfile:
    name: str = "small-v1"
    seed: int = 20260808
    samples_per_class: int = 200


@dataclass(frozen=True)
class NpyHeader:
    count: int
    data_offset: int


def load_classes() -> tuple[str, ...]:
    path = Path(__file__).parents[1] / "contracts" / "classes.v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(cast(list[str], payload["classes"]))


def stable_start_index(seed: int, class_name: str, total: int, count: int) -> int:
    available = total - count
    if available < 0:
        raise ValueError("The source array contains fewer samples than requested")
    digest = hashlib.sha256(f"sketchsense-sample-v1:{seed}:{class_name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (available + 1)


def deterministic_split(
    source_indices: NDArray[np.int64], class_name: str, seed: int
) -> NDArray[np.uint8]:
    expected = sum(SPLIT_COUNTS.values())
    if len(source_indices) != expected:
        raise ValueError(f"Expected {expected} source indices, received {len(source_indices)}")
    order = sorted(
        range(len(source_indices)),
        key=lambda position: hashlib.sha256(
            f"sketchsense-split-v1:{seed}:{class_name}:{int(source_indices[position])}".encode()
        ).digest(),
    )
    assignments = np.empty(expected, dtype=np.uint8)
    cursor = 0
    for split_code, count in enumerate(SPLIT_COUNTS.values()):
        for position in order[cursor : cursor + count]:
            assignments[position] = split_code
        cursor += count
    return assignments


def prepare_dataset(output_dir: Path, profile: DatasetProfile | None = None) -> Path:
    profile = profile or DatasetProfile()
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = load_classes()
    all_images: list[NDArray[np.uint8]] = []
    all_labels: list[int] = []
    all_indices: list[int] = []
    all_splits: list[int] = []
    records: list[dict[str, object]] = []
    source_arrays: dict[str, object] = {}

    for class_index, class_name in enumerate(classes):
        url = f"{BASE_URL}/{urllib.parse.quote(class_name)}.npy"
        header = read_npy_header(url)
        start = stable_start_index(
            profile.seed, class_name, header.count, profile.samples_per_class
        )
        images = read_bitmap_range(url, header, start, profile.samples_per_class)
        source_indices = np.arange(start, start + profile.samples_per_class, dtype=np.int64)
        splits = deterministic_split(source_indices, class_name, profile.seed)
        source_arrays[class_name] = {
            "url": url,
            "total_samples": header.count,
            "data_offset": header.data_offset,
            "selected_start": start,
        }
        for image, source_index, split_code in zip(images, source_indices, splits, strict=True):
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            split = tuple(SPLIT_COUNTS)[int(split_code)]
            sample_id = f"{class_name}:{int(source_index)}"
            records.append(
                {
                    "sample_id": sample_id,
                    "class_name": class_name,
                    "class_index": class_index,
                    "source_index": int(source_index),
                    "split": split,
                    "sha256": digest,
                }
            )
            all_images.append(image)
            all_labels.append(class_index)
            all_indices.append(int(source_index))
            all_splits.append(int(split_code))

    data_path = output_dir / "small-v1.npz"
    save_npz_deterministic(
        data_path,
        {
            "images": np.stack(all_images),
            "labels": np.asarray(all_labels, dtype=np.uint8),
            "source_indices": np.asarray(all_indices, dtype=np.int64),
            "splits": np.asarray(all_splits, dtype=np.uint8),
        },
    )
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "dataset": {
            "name": "Google Quick, Draw! Dataset",
            "source": BASE_URL,
            "license": "CC BY 4.0",
        },
        "profile": {
            "name": profile.name,
            "seed": profile.seed,
            "samples_per_class": profile.samples_per_class,
        },
        "classes": list(classes),
        "splits": SPLIT_COUNTS,
        "source_arrays": source_arrays,
        "cache": {"file": data_path.name, "sha256": _file_sha256(data_path)},
        "samples": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_dataset(output_dir)
    return manifest_path


def read_npy_header(url: str) -> NpyHeader:
    prefix = _request_range(url, 0, 511)
    if prefix[:6] != b"\x93NUMPY":
        raise ValueError(f"Unexpected NPY magic for {url}")
    major = prefix[6]
    if major == 1:
        header_length, length_size = struct.unpack("<H", prefix[8:10])[0], 2
    elif major in {2, 3}:
        header_length, length_size = struct.unpack("<I", prefix[8:12])[0], 4
    else:
        raise ValueError(f"Unsupported NPY version {major}")
    header_start = 8 + length_size
    required = header_start + header_length
    raw = prefix if len(prefix) >= required else _request_range(url, 0, required - 1)
    metadata = cast(
        dict[str, Any], ast.literal_eval(raw[header_start:required].decode("latin1").strip())
    )
    if metadata.get("descr") != "|u1" or metadata.get("fortran_order") is not False:
        raise ValueError(f"Unsupported NPY layout: {metadata}")
    shape = metadata.get("shape")
    if not isinstance(shape, tuple) or shape[1:] not in {(784,), (28, 28)}:
        raise ValueError(f"Expected Nx784 or Nx28x28 source array, received {shape}")
    return NpyHeader(count=int(shape[0]), data_offset=required)


def read_bitmap_range(url: str, header: NpyHeader, start: int, count: int) -> NDArray[np.uint8]:
    bytes_per_image = 28 * 28
    first = header.data_offset + start * bytes_per_image
    raw = _request_range(url, first, first + count * bytes_per_image - 1)
    if len(raw) != count * bytes_per_image:
        raise ValueError(f"Expected {count * bytes_per_image} bytes, received {len(raw)}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(count, 28, 28).copy()


def validate_dataset(output_dir: Path) -> dict[str, int]:
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    schema_path = Path(__file__).parents[1] / "contracts" / "dataset-manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(manifest)
    classes = load_classes()
    if tuple(manifest["classes"]) != classes:
        raise ValueError("Dataset class order does not match classes.v1.json")
    data_path = output_dir / str(manifest["cache"]["file"])
    if _file_sha256(data_path) != manifest["cache"]["sha256"]:
        raise ValueError("Cached dataset checksum does not match its manifest")
    with np.load(data_path) as data:
        images, labels, indices, splits = (
            data["images"],
            data["labels"],
            data["source_indices"],
            data["splits"],
        )
    expected_total = len(classes) * sum(SPLIT_COUNTS.values())
    if images.shape != (expected_total, 28, 28) or labels.shape != (expected_total,):
        raise ValueError("Cached dataset shapes do not match the profile")
    ids = {(int(label), int(index)) for label, index in zip(labels, indices, strict=True)}
    if len(ids) != expected_total:
        raise ValueError("A source sample appears more than once")
    summary: dict[str, int] = {}
    for split_code, (split_name, per_class_count) in enumerate(SPLIT_COUNTS.items()):
        mask = splits == split_code
        summary[split_name] = int(mask.sum())
        for class_index in range(len(classes)):
            if int(np.sum(mask & (labels == class_index))) != per_class_count:
                raise ValueError(f"Class {class_index} is not balanced in {split_name}")
    return summary


def write_dataset_summary(dataset_dir: Path, output: Path) -> Path:
    """Write bounded, reviewable metadata without the per-sample cache manifest."""
    split_summary = validate_dataset(dataset_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = {key: value for key, value in manifest.items() if key != "samples"}
    summary["split_totals"] = split_summary
    summary["sample_records"] = len(manifest["samples"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _request_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "SketchSense/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed official source
        if response.status != 206:
            raise OSError(f"Server ignored bounded range request for {url}: HTTP {response.status}")
        return response.read()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
