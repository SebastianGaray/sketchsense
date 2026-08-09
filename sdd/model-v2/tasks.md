# SketchSense model v2 tasks

These tasks are post-v1 and intentionally incomplete until implementation evidence exists.

## Data and diagnosis

- [ ] **MV2-T001** Version the `medium-v2` profile and schema with 1,000 unused official samples per class.
- [ ] **MV2-T002** Generate deterministic 800/100/100 per-class splits and prove no v1/v2 test overlap.
- [ ] **MV2-T003** Implement versioned vector-to-raster and canvas-domain augmentations with visual inspection fixtures.
- [ ] **MV2-T004** Publish the v1 sensitivity, confusion, calibration, and authored-sketch diagnostic report.

## Experiments and selection

- [ ] **MV2-T005** Retrain `compact-cnn-v1` on `medium-v2` without augmentation across three seeds.
- [ ] **MV2-T006** Measure the isolated augmentation effect on the same architecture across three seeds.
- [ ] **MV2-T007** Implement and compare at least two sub-million-parameter candidates.
- [ ] **MV2-T008** Record the multi-objective validation decision covering macro F1, worst recall, calibration, size, and latency.

## Locked evaluation and artifacts

- [ ] **MV2-T009** Lock the selected configuration and evaluate once on the fresh v2 test split.
- [ ] **MV2-T010** Publish aggregate, per-class, confusion, calibration, confidence-interval, and v1 comparison evidence.
- [ ] **MV2-T011** Export ONNX and validate PyTorch/runtime parity, malformed inputs, hashes, schema, and size.
- [ ] **MV2-T012** Measure real browser cold load and warm p50/p95 inference against the v1 budget.

## Integration and release

- [ ] **MV2-T013** Add authored cross-category and stroke-width browser regression fixtures without visitor data.
- [ ] **MV2-T014** Update the model/data cards, application manifest compatibility, limitations, and rollback instructions.
- [ ] **MV2-T015** Deploy v2 only if every acceptance gate passes; otherwise publish the experiment as non-selected future work.
