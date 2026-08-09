# ruff: noqa: E501
"""Vector-native model-v3 dataset, comparison, and release pipeline."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
import torch
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from sketchsense.data.dataset import load_classes
from sketchsense.evaluation.metrics import detailed_classification_report
from sketchsense.models.cnn import CompactSketchCNN, WidenedBatchNormCNN, parameter_count
from sketchsense.training.pipeline import save_checkpoint, set_deterministic_state

SPLITS = {"train": 10_000, "validation": 500, "test": 500}
SEED = 20260811
VECTOR_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson"


def prepare_vector_dataset(output_dir: Path) -> Path:
    """Fetch recognized vector drawings and rasterize through the browser contract."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "vector-v3.npz"
    manifest_path = output_dir / "manifest.json"
    if cache.exists() and manifest_path.exists():
        return manifest_path
    images_28: list[NDArray[np.uint8]] = []
    images_56: list[NDArray[np.uint8]] = []
    labels: list[int] = []
    splits: list[int] = []
    records: list[dict[str, object]] = []
    needed = sum(SPLITS.values())
    for class_index, category in enumerate(load_classes()):
        encoded = urllib.parse.quote(category)
        accepted = 0
        with urllib.request.urlopen(VECTOR_URL.format(category=encoded), timeout=120) as response:
            for source_index, raw_line in enumerate(response):
                drawing = json.loads(raw_line)
                if not drawing.get("recognized", False):
                    continue
                split_code = _split_code(accepted)
                rgba = _render_strokes(drawing["drawing"], source_index)
                for size, target in ((28, images_28), (56, images_56)):
                    target.append(_fast_contract_raster(rgba, size))
                labels.append(class_index)
                splits.append(split_code)
                records.append(
                    {
                        "category": category,
                        "source_index": source_index,
                        "key_id": str(drawing.get("key_id", "")),
                        "split": tuple(SPLITS)[split_code],
                    }
                )
                accepted += 1
                if accepted == needed:
                    break
        if accepted != needed:
            raise RuntimeError(f"Only found {accepted} recognized {category} drawings")
    np.savez_compressed(
        cache,
        images_28=np.stack(images_28),
        images_56=np.stack(images_56),
        labels=np.asarray(labels, dtype=np.uint8),
        splits=np.asarray(splits, dtype=np.uint8),
    )
    manifest = {
        "schema_version": "3.0.0",
        "dataset": "Google Quick, Draw! simplified vectors",
        "license": "CC BY 4.0",
        "classes": list(load_classes()),
        "splits_per_class": SPLITS,
        "preprocessing": "browser crop/pad/resize/center contract",
        "resolutions": [28, 56],
        "records": records,
        "cache_sha256": _sha256(cache),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def train_release_28(dataset_dir: Path, release_dir: Path) -> Path:
    """Train the evidence-backed 28px candidate without retaining 56px data in memory."""
    with np.load(dataset_dir / "vector-v3.npz") as payload:
        images = payload["images_28"].copy()
        labels = payload["labels"].copy()
        splits = payload["splits"].copy()
    result, model = _train_candidate(images, labels, splits, "widened-bn", 28)
    baseline = json.loads(
        (Path(__file__).parents[4] / "artifacts/evaluation/evaluation-summary.v2.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_f1 = float(baseline["metrics"]["macro_f1"])
    candidate_f1 = float(cast(dict[str, float], result["test"])["macro_f1"])
    if candidate_f1 <= baseline_f1:
        raise RuntimeError(
            f"Candidate macro F1 {candidate_f1:.4f} did not beat v2 {baseline_f1:.4f}"
        )
    model.eval()
    release_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = release_dir / "sketch-cnn.v3.checkpoint.npz"
    onnx_path = release_dir / "sketch-cnn.v3.onnx"
    save_checkpoint(model, checkpoint)
    torch.onnx.export(
        model,
        (torch.zeros((1, 1, 28, 28), dtype=torch.float32),),
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample = images[splits == 2][:1].astype(np.float32)[:, None] / 255
    with torch.inference_mode():
        expected = model(torch.from_numpy(sample)).numpy()
    actual = session.run(["logits"], {"input": sample})[0]
    parity = float(np.max(np.abs(expected - actual)))
    if parity > 1e-4:
        raise RuntimeError(f"ONNX parity failed: {parity}")
    report = {
        "schema_version": "3.0.0",
        "model_version": "3.0.0",
        "dataset_version": "vector-v3",
        "candidate": result,
        "baseline_v2_macro_f1": baseline_f1,
        "input_size": 28,
        "resolution_decision": (
            "Keep 28 x 28: the vector-native 28px candidate clears quality gates; "
            "56px training exceeded the 8GB development-device memory safety budget."
        ),
        "parameters": parameter_count(model),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_sha256": _sha256(onnx_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "onnx_max_absolute_difference": parity,
    }
    output = release_dir / "evaluation-summary.v3.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_training_chart(cast(list[dict[str, float]], result["history"]), release_dir)
    _write_comparison_chart(baseline_f1, candidate_f1, release_dir)
    return output


def release_checkpoint_28(dataset_dir: Path, release_dir: Path) -> Path:
    """Resume evaluation and export from a completed v3 checkpoint."""
    with np.load(dataset_dir / "vector-v3.npz") as payload:
        images = payload["images_28"].copy()
        labels = payload["labels"].copy()
        splits = payload["splits"].copy()
    model = WidenedBatchNormCNN(len(load_classes()))
    checkpoint = release_dir / "sketch-cnn.v3.checkpoint.npz"
    with np.load(checkpoint) as payload:
        model.load_state_dict(
            {name: torch.from_numpy(payload[name].copy()) for name in payload.files}
        )
    model.eval()
    test_mask = splits == 2
    test_x = torch.from_numpy(images[test_mask].astype(np.float32) / 255).unsqueeze(1)
    test_labels = labels[test_mask]
    with torch.inference_mode():
        probabilities = torch.softmax(model(test_x), dim=1).numpy()
    details = detailed_classification_report(test_labels, probabilities, load_classes())
    per_class = cast(dict[str, dict[str, float]], details["per_class"])
    metrics = cast(dict[str, float], details["metrics"])
    baseline = json.loads(
        (Path(__file__).parents[4] / "artifacts/evaluation/evaluation-summary.v2.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_f1 = float(baseline["metrics"]["macro_f1"])
    if metrics["macro_f1"] <= baseline_f1:
        raise RuntimeError("Saved v3 checkpoint does not improve macro F1")
    onnx_path = release_dir / "sketch-cnn.v3.onnx"
    torch.onnx.export(
        model,
        (torch.zeros((1, 1, 28, 28), dtype=torch.float32),),
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample = test_x[:1].numpy()
    with torch.inference_mode():
        expected = model(test_x[:1]).numpy()
    actual = session.run(["logits"], {"input": sample})[0]
    parity = float(np.max(np.abs(expected - actual)))
    if parity > 1e-4:
        raise RuntimeError(f"ONNX parity failed: {parity}")
    report = {
        "schema_version": "3.0.0",
        "model_version": "3.0.0",
        "dataset_version": "vector-v3",
        "architecture": "widened-bn",
        "input_size": 28,
        "training_samples": int(np.sum(splits == 0)),
        "validation_samples": int(np.sum(splits == 1)),
        "test_samples": int(np.sum(test_mask)),
        "metrics": metrics,
        "per_class": per_class,
        "worst_class_recall": min(item["recall"] for item in per_class.values()),
        "baseline_v2_macro_f1": baseline_f1,
        "resolution_decision": (
            "Keep 28 x 28: v3 clears the quality comparison; the 56px candidate exceeded "
            "the 8GB development-device memory safety budget."
        ),
        "parameters": parameter_count(model),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_sha256": _sha256(onnx_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "onnx_max_absolute_difference": parity,
    }
    output = release_dir / "evaluation-summary.v3.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_comparison_chart(baseline_f1, metrics["macro_f1"], release_dir)
    _write_recall_chart(per_class, release_dir)
    return output


def publish_v3(dataset_dir: Path, release_dir: Path, web_public: Path) -> Path:
    """Publish the selected model, evidence charts, and model-validated example prompts."""
    report = json.loads((release_dir / "evaluation-summary.v3.json").read_text(encoding="utf-8"))
    model_dir = web_public / "models"
    evidence_dir = web_public / "evidence"
    examples_dir = web_public / "examples" / "v3"
    for directory in (model_dir, evidence_dir, examples_dir):
        directory.mkdir(parents=True, exist_ok=True)
    source_model = release_dir / "sketch-cnn.v3.onnx"
    target_model = model_dir / "sketch-cnn.v3.onnx"
    shutil.copyfile(source_model, target_model)
    for name in ("model-comparison.v3.svg", "per-class-recall.v3.svg"):
        shutil.copyfile(release_dir / name, evidence_dir / name)
    manifest = {
        "schema_version": "1.0.0",
        "model_version": "3.0.0",
        "preprocessing_version": "1.0.0",
        "class_manifest_version": "1.0.0",
        "input": {"name": "input", "shape": [1, 1, 28, 28], "dtype": "float32"},
        "output": {"name": "logits", "shape": [1, 16]},
        "onnx": {
            "bytes": report["onnx_bytes"],
            "sha256": report["onnx_sha256"],
        },
        "uncertainty": {"minimum_top_score": 0.55, "minimum_margin": 0.15},
    }
    manifest_path = model_dir / "model-manifest.v3.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with np.load(dataset_dir / "vector-v3.npz") as payload:
        images = payload["images_28"][payload["splits"] == 2].copy()
        labels = payload["labels"][payload["splits"] == 2].copy()
    session = ort.InferenceSession(str(source_model), providers=["CPUExecutionProvider"])
    output_batches: list[NDArray[np.float32]] = []
    for image in images:
        output_batches.append(
            np.asarray(
                session.run(
                    ["logits"],
                    {"input": image.astype(np.float32)[None, None] / 255},
                )[0],
                dtype=np.float32,
            )
        )
    logits = np.concatenate(output_batches)
    probabilities = _softmax(logits)
    predictions = probabilities.argmax(axis=1)
    validation: list[dict[str, object]] = []
    for class_index, category in enumerate(load_classes()):
        eligible = np.flatnonzero((labels == class_index) & (predictions == class_index))
        if not len(eligible):
            raise RuntimeError(f"No correctly classified example for {category}")
        scores = probabilities[eligible, class_index]
        selected = int(eligible[int(np.argmax(scores))])
        grayscale = Image.fromarray(255 - images[selected], "L").resize(
            (112, 112), Image.Resampling.LANCZOS
        )
        grayscale.save(examples_dir / f"{category}.png", optimize=True)
        validation.append(
            {
                "category": category,
                "predicted": category,
                "confidence": float(probabilities[selected, class_index]),
                "source": "held-out vector-v3 test",
            }
        )
    validation_path = release_dir / "example-validation.v3.json"
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _train_candidate(
    images: NDArray[np.uint8],
    labels: NDArray[np.uint8],
    splits: NDArray[np.uint8],
    architecture: str,
    input_size: int,
) -> tuple[dict[str, object], nn.Module]:
    set_deterministic_state(SEED)
    torch.set_num_threads(6)
    factory = CompactSketchCNN if architecture == "compact" else WidenedBatchNormCNN
    model = factory(len(load_classes()))
    train_mask, validation_mask, test_mask = splits == 0, splits == 1, splits == 2
    train_x = torch.from_numpy(images[train_mask].astype(np.float32) / 255).unsqueeze(1)
    train_y = torch.from_numpy(labels[train_mask].astype(np.int64))
    validation_x = torch.from_numpy(images[validation_mask].astype(np.float32) / 255).unsqueeze(1)
    validation_y = torch.from_numpy(labels[validation_mask].astype(np.int64))
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    stale = 0
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    for epoch in range(1, 7):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss)
            batches += 1
        model.eval()
        with torch.inference_mode():
            validation_loss = float(loss_fn(model(validation_x), validation_y))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / batches,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 0.0005:
            best_loss = validation_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale == 2:
                break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    test_x = torch.from_numpy(images[test_mask].astype(np.float32) / 255).unsqueeze(1)
    test_labels = labels[test_mask]
    with torch.inference_mode():
        probabilities = torch.softmax(model(test_x), dim=1).numpy()
    report = detailed_classification_report(test_labels, probabilities, load_classes())
    per_class = cast(dict[str, dict[str, float]], report["per_class"])
    sample = test_x[:1]
    for _ in range(20):
        model(sample)
    timings = []
    with torch.inference_mode():
        for _ in range(200):
            tick = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - tick) * 1000)
    return (
        {
            "name": f"{architecture}-{input_size}",
            "architecture": architecture,
            "input_size": input_size,
            "parameters": parameter_count(model),
            "test": report["metrics"],
            "per_class": per_class,
            "worst_class_recall": min(item["recall"] for item in per_class.values()),
            "p95_ms": float(np.percentile(timings, 95)),
            "training_seconds": time.perf_counter() - started,
            "history": history,
        },
        model,
    )


def _split_code(index: int) -> int:
    if index < SPLITS["train"]:
        return 0
    if index < SPLITS["train"] + SPLITS["validation"]:
        return 1
    return 2


def _render_strokes(strokes: list[list[list[int]]], seed: int) -> NDArray[np.uint8]:
    image = Image.new("RGBA", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    width = 6 + seed % 7
    for xs, ys in strokes:
        points = list(zip(xs, ys, strict=True))
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse(
                (x - width // 2, y - width // 2, x + width // 2, y + width // 2), fill="black"
            )
        elif points:
            draw.line(points, fill="black", width=width, joint="curve")
    return np.asarray(image, dtype=np.uint8)


def _fast_contract_raster(rgba: NDArray[np.uint8], output_size: int) -> NDArray[np.uint8]:
    """Native bilinear implementation of the browser crop/center geometry for bulk data."""
    luminance = np.asarray(Image.fromarray(rgba, "RGBA").convert("L"), dtype=np.uint8)
    foreground_y, foreground_x = np.nonzero(luminance < 250)
    if not foreground_x.size:
        return np.zeros((output_size, output_size), dtype=np.uint8)
    x0, x1 = int(foreground_x.min()), int(foreground_x.max())
    y0, y1 = int(foreground_y.min()), int(foreground_y.max())
    padding = max(2, int(np.ceil(max(x1 - x0 + 1, y1 - y0 + 1) * 0.1)))
    x0, x1 = max(0, x0 - padding), min(luminance.shape[1] - 1, x1 + padding)
    y0, y1 = max(0, y0 - padding), min(luminance.shape[0] - 1, y1 + padding)
    crop = Image.fromarray(luminance[y0 : y1 + 1, x0 : x1 + 1])
    content_size = round(output_size * 20 / 28)
    scale = min(content_size / crop.width, content_size / crop.height)
    target = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    resized = crop.resize(target, Image.Resampling.BILINEAR)
    canvas = Image.new("L", (output_size, output_size), 255)
    canvas.paste(resized, ((output_size - target[0]) // 2, (output_size - target[1]) // 2))
    return 255 - np.asarray(canvas, dtype=np.uint8)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _softmax(logits: NDArray[np.floating]) -> NDArray[np.floating]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _write_training_chart(history: list[dict[str, float]], output_dir: Path) -> None:
    width, height = 760, 420
    values = [point[key] for point in history for key in ("train_loss", "validation_loss")]
    maximum, minimum = max(values), min(values)
    spread = maximum - minimum or 1.0

    def points(key: str) -> str:
        return " ".join(
            f"{80 + index * 600 / max(1, len(history) - 1):.1f},"
            f"{340 - (point[key] - minimum) * 260 / spread:.1f}"
            for index, point in enumerate(history)
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">Model v3 training loss</title><desc id="desc">Training and validation loss by epoch.</desc>
<rect width="760" height="420" fill="#fffaf8"/><path d="M80 70V340H700" fill="none" stroke="#182230"/>
<polyline points="{points("train_loss")}" fill="none" stroke="#165dff" stroke-width="4"/>
<polyline points="{points("validation_loss")}" fill="none" stroke="#d1495b" stroke-width="4"/>
<text x="80" y="45" font-family="system-ui" font-size="24" font-weight="700">Model v3 training</text>
<text x="80" y="385" font-family="system-ui" font-size="16">Epoch</text>
<text x="475" y="45" font-family="system-ui" font-size="14" fill="#165dff">Training loss</text>
<text x="590" y="45" font-family="system-ui" font-size="14" fill="#d1495b">Validation loss</text>
</svg>'''
    (output_dir / "training-curves.v3.svg").write_text(svg, encoding="utf-8")


def _write_comparison_chart(baseline: float, candidate: float, output_dir: Path) -> None:
    bars = [("Model v2", baseline), ("Model v3", candidate)]
    rows = "".join(
        f'<text x="70" y="{135 + index * 100}" font-family="system-ui" font-size="18">{name}</text>'
        f'<rect x="190" y="{110 + index * 100}" width="{score * 500:.1f}" height="38" fill="{color}"/>'
        f'<text x="{205 + score * 500:.1f}" y="{137 + index * 100}" font-family="system-ui" font-size="16">{score * 100:.2f}%</text>'
        for index, ((name, score), color) in enumerate(
            zip(bars, ("#667085", "#165dff"), strict=True)
        )
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 340" role="img" aria-labelledby="title desc">
<title id="title">Model quality comparison</title><desc id="desc">Macro F1 comparison between released models.</desc>
<rect width="760" height="340" fill="#fffaf8"/><text x="70" y="60" font-family="system-ui" font-size="24" font-weight="700">Held-out macro F1</text>{rows}</svg>"""
    (output_dir / "model-comparison.v3.svg").write_text(svg, encoding="utf-8")


def _write_recall_chart(per_class: dict[str, dict[str, float]], output_dir: Path) -> None:
    rows = []
    for index, (name, metrics) in enumerate(per_class.items()):
        y = 75 + index * 34
        recall = metrics["recall"]
        rows.append(
            f'<text x="30" y="{y + 15}" font-family="system-ui" font-size="13">{name}</text>'
            f'<rect x="115" y="{y}" width="{recall * 500:.1f}" height="20" fill="#165dff"/>'
            f'<text x="{125 + recall * 500:.1f}" y="{y + 15}" font-family="system-ui" '
            f'font-size="13">{recall * 100:.1f}%</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 650" role="img" aria-labelledby="title desc">
<title id="title">Model v3 recall by category</title><desc id="desc">Held-out recall for every supported drawing category.</desc>
<rect width="760" height="650" fill="#fffaf8"/><text x="30" y="42" font-family="system-ui" font-size="24" font-weight="700">Held-out recall by category</text>{"".join(rows)}</svg>"""
    (output_dir / "per-class-recall.v3.svg").write_text(svg, encoding="utf-8")
