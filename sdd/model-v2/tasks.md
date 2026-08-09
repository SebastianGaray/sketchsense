# SketchSense model v2 tasks

Implementation evidence is committed under `artifacts/`. Raw pixels and intermediate checkpoints remain ignored.

## Data and diagnosis

- [x] **MV2-T001** Version the `medium-v2` profile and schema with 1,000 unused official samples per class.
- [x] **MV2-T002** Generate deterministic 800/100/100 per-class splits and prove no v1/v2 test overlap.
- [x] **MV2-T003** Implement versioned bitmap-to-canvas augmentations with visual inspection fixtures; vector sources were not required for the selected official profile.
- [x] **MV2-T004** Publish sensitivity, confusion, calibration, and authored-sketch diagnostic evidence.

## Experiments and selection

- [x] **MV2-T005** Retrain `compact-cnn-v1` on `medium-v2` without augmentation across three seeds.
- [x] **MV2-T006** Measure the isolated augmentation effect on the same architecture across three seeds.
- [x] **MV2-T007** Implement and compare at least two sub-million-parameter candidates.
- [x] **MV2-T008** Record the multi-objective validation decision covering macro F1, worst recall, calibration, size, and latency.

## Locked evaluation and artifacts

- [x] **MV2-T009** Lock the selected configuration and evaluate once on the fresh v2 test split.
- [x] **MV2-T010** Publish aggregate, per-class, confusion, calibration, confidence-interval, and v1 comparison evidence.
- [x] **MV2-T011** Export ONNX and validate PyTorch/runtime parity, malformed inputs, hashes, schema, and size.
- [x] **MV2-T012** Measure runtime p50/p95 against v1 and smoke-test cold browser loading with the production base path.

## Integration and release

- [x] **MV2-T013** Add authored cross-category and stroke-width browser regression fixtures without visitor data.
- [x] **MV2-T014** Update the model/data cards, application manifest compatibility, limitations, and rollback instructions.
- [x] **MV2-T015** Deploy v2 only if every acceptance gate passes; otherwise publish the experiment as non-selected future work.
