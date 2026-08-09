# SketchSense model v2 specification

Status: implemented. Model v2 met the release gates on a fresh non-overlapping final test.

## Objective

Improve recognition quality across all 16 categories, especially ambiguous and low-recall classes, while preserving private browser inference, deterministic provenance, a compact artifact, and responsive CPU/WASM execution.

The v1 reference is 72.71% accuracy, 73.19% macro F1, 86.88% top-3 accuracy, and a 441,021-byte ONNX artifact. Per-class recall ranges from 46.67% for dog to 96.67% for apple; cat recall is 66.67%. Each v1 class has only 140 training, 30 validation, and 30 test examples.

## Requirements

- **MV2-DATA-001 Fresh profile:** build a deterministic profile from official Quick, Draw! sources with at least 1,000 unused examples per class. No v1 sample identifier may enter the v2 test partition.
- **MV2-DATA-002 Fixed splits:** assign exactly 800 train, 100 validation, and 100 fresh test examples per class before training. Publish identifiers, hashes, provenance, and overlap checks without committing raw source data.
- **MV2-DATA-003 Canvas domain:** training must include deterministic augmentations that approximate browser drawings: translation, scale, small rotation, line-width variation, and mild rasterization variation. Horizontal or vertical flips require category-specific justification.
- **MV2-MODEL-001 Candidates:** compare the released CNN against at least two compact candidates under one million parameters. Candidate choice must use validation evidence only.
- **MV2-MODEL-002 Repeated evidence:** train each serious candidate with three fixed seeds and report mean plus dispersion. A lucky single run cannot select the model.
- **MV2-EVAL-001 Fresh test:** open the v2 test split once after architecture, augmentation, and hyperparameter selection are locked.
- **MV2-EVAL-002 Complete metrics:** report accuracy, macro precision/recall/F1, top-3 accuracy, per-class recall and support, confusion matrix, expected calibration error, artifact size, and measured browser latency.
- **MV2-EVAL-003 Regression gates:** no category may regress by more than five recall points versus v1 without written justification. Track the worst-class recall explicitly rather than optimizing only aggregate accuracy.
- **MV2-ART-001 Browser contract:** retain float32 `[1,1,28,28]` input, 16 ordered logits, fixed batch one, schema validation, checksums, PyTorch/ONNX parity, and Python/TypeScript preprocessing parity.
- **MV2-ART-002 Budget:** target an ONNX artifact below 5 MB and no more than 20% p95 inference-latency regression on the documented browser test device.
- **MV2-PRIV-001 Privacy:** do not collect drawings or telemetry. Any additional qualitative evaluation set must be authored for the project, versioned, licensed, and contain no visitor data.

## Acceptance criteria

The v2 model may replace v1 only when the locked fresh test demonstrates all of the following:

- accuracy and macro F1 are each at least 0.80;
- top-3 accuracy is at least 0.92;
- worst-class recall is at least 0.60 and improves over v1's 0.4667;
- no unexplained material class regression;
- calibration is measured and uncertainty copy remains honest;
- the model stays below 5 MB and passes production browser inference, parity, integrity, accessibility, and privacy checks;
- v1 remains reproducible and available for an explicit rollback.

These thresholds are release gates, not guaranteed outcomes. If experiments do not meet them, publish the negative result and keep v1.

## Non-goals

Model v2 does not add classes, upload drawings, train in the browser, use visitor telemetry, require WebGPU, or optimize specifically for one anecdotal cat drawing.
