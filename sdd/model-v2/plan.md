# SketchSense model v2 implementation plan

## 1. Establish trustworthy data

Create `medium-v2` from unused official sample indices with 16,000 total examples: 12,800 train, 1,600 validation, and 1,600 fresh test. Extend the manifest with a parent/source profile, exclusion hashes for every v1 identifier, rasterization configuration, and augmentation version. Run duplicate identifier and perceptual/image-hash checks across all partitions.

Where practical, use official simplified vector strokes for training rasterization rather than only pre-rendered bitmaps. Rasterize them deterministically with line-width and placement variation matching the browser canvas. Keep the canonical application preprocessing unchanged unless parity evidence justifies a versioned contract update.

## 2. Diagnose v1 before changing it

Use only existing v1 evidence and v2 train/validation data to analyze:

- class imbalance in visual complexity despite equal sample counts;
- directed confusions such as dog/bird/key and visually adjacent categories;
- sensitivity to stroke width, translation, crop size, rotation, and partial drawings;
- confidence calibration and low-margin predictions;
- errors on a small authored, non-user qualitative sketch suite.

This stage produces hypotheses and augmentation bounds. It must not inspect v2 test outcomes.

## 3. Run controlled experiments

Keep a versioned experiment matrix rather than changing several variables at once:

1. Retrain the released architecture on `medium-v2` without augmentation to isolate the data-volume effect.
2. Add canvas-domain augmentation to the same architecture to measure the domain-gap effect.
3. Compare compact candidates:
   - widened two-block CNN with batch normalization;
   - three-block depthwise-separable CNN with global average pooling;
   - small residual CNN only if it remains within the parameter and latency budgets.
4. Evaluate class-weighted loss or label smoothing only after the data and augmentation baselines exist.
5. Measure post-training static quantization only as an optional artifact experiment; reject it if per-class recall or parity degrades materially.

Every serious run uses the same three fixed seeds, deterministic settings where supported, early stopping, and a recorded lock/environment fingerprint. Select using validation macro F1, worst-class recall, calibration, size, and latency as a multi-objective decision. Do not select on accuracy alone.

## 4. Lock and evaluate once

Freeze architecture, seed policy, augmentation, preprocessing, class order, checkpoint rule, and export settings. Evaluate the selected model once on the fresh 1,600-example test split. Compare v1 and v2 on aggregate and per-class metrics with bootstrap confidence intervals where practical.

Treat the existing v1 test metrics as historical evidence, not a tuning set. Report all regressions and the full confusion matrix. A v2 candidate that misses release gates stays experimental.

## 5. Validate browser behavior

Export a single-file ONNX artifact and verify PyTorch/ONNX logits on shared fixtures plus a bounded held-out sample. Run real ONNX Runtime Web inference at the GitHub Pages base path and measure cold load, warm inference p50/p95, memory where observable, and mobile layout behavior.

Add deterministic authored sketches covering each category, thin/medium/thick strokes, small translations, and partial-versus-complete shapes. These are regression fixtures, not claims of real-world accuracy and never originate from visitors.

## 6. Release safely

Publish a versioned model card, data card update, experiment summary, artifact manifest, and migration note. Keep the v1 artifact available until v2 deployment and smoke tests pass. Application compatibility must fail closed on version or checksum mismatch. Rollback is a manifest/model-path change through the protected PR workflow.

## Risks and mitigations

- **More data but same domain gap:** measure the no-augmentation and augmented baselines separately.
- **Aggregate gains hiding weak classes:** gate on worst-class recall and per-class regressions.
- **Overfitting the authored sketch suite:** use it only for qualitative regression, never model selection.
- **Larger/slower model:** enforce artifact and browser-latency budgets during candidate selection.
- **Test leakage:** exclude v1 identifiers, freeze v2 test IDs early, and record a one-time access boundary.
- **False confidence:** measure calibration and retain uncertainty guidance in the interface.
