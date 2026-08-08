# SketchSense Tasks

Completed tasks are checked only when repository evidence exists.

## M1 — Foundation and SDD

- [x] **T001** Define traceable product requirements and acceptance criteria in `spec.md`. Complete when required product, ML, performance, privacy, accessibility, localization, and deployment scopes are explicit. (all requirements)
- [x] **T002** Record architecture, contracts, verification, risks, and milestones in `plan.md`. Complete when implementation choices are unambiguous.
- [x] **T003** Adapt the canonical portfolio design into `DESIGN.md`. Complete when tokens, themes, components, localization, and SketchSense-specific patterns are mapped.
- [x] **T004** Establish locked Python and web environments plus root commands. Complete when clean locked installs and foundation quality commands pass.
- [x] **T005** Implement the honest bilingual, responsive, three-theme application shell and portfolio return link. Complete when EN/ES production routes and theme controls are verified. (SS-I18N-001, SS-I18N-002, SS-DEP-001, SS-FR-009)
- [x] **T006** Add baseline tests, pre-commit, CI validation, and GitHub Pages workflow. Complete when local validation and production build pass. (SS-DEP-001, SS-DEP-002)

## M2 — Dataset and preprocessing

- [x] **T007** Version the class-order and dataset-manifest schemas. Complete when schemas validate the selected 16-class contract. (SS-ML-001, SS-ML-002, SS-ML-006)
- [x] **T008** Implement bounded deterministic category download and subset generation. Complete when reruns produce identical manifests without split overlap. (SS-ML-001)
- [x] **T009** Implement Python preprocessing and fixtures. Complete when contract edge cases pass. (SS-PERF-004)
- [x] **T010** Implement TypeScript preprocessing parity. Complete when shared fixtures match within `1e-5`. (SS-ML-005, SS-PERF-004)

## M3 — Baseline and CNN

- [x] **T011** Train and evaluate the deterministic logistic baseline. Complete when its reproducible report is versioned. (SS-ML-003, SS-ML-004)
- [x] **T012** Implement and train the compact CNN. Complete when checkpoints and training metadata reproduce the selected model. (SS-ML-003)
- [x] **T013** Compare candidates on validation evidence and budgets. Complete when the selection rationale is documented. (SS-ML-004, SS-ML-007)

## M4 — Evaluation and ONNX

- [x] **T014** Evaluate the locked model on the held-out test split. Complete when aggregate/per-class metrics and confusion evidence are published. (SS-ML-004)
- [x] **T015** Export ONNX and validate PyTorch parity. Complete when tolerance and runtime checks pass. (SS-ML-005)
- [x] **T016** Publish schema-valid, hashed, bounded model artifacts. Complete when the manifest and size checks pass. (SS-ML-006, SS-ML-007)

## M5 — Browser inference

- [x] **T017** Implement responsive pointer-capture Canvas behavior and clear state. Complete when mouse/pen/touch browser tests pass. (SS-FR-001, SS-FR-002, SS-PERF-001)
- [x] **T018** Load and validate ONNX artifacts locally. Complete when loading, retry, unsupported-runtime, and mismatch states pass. (SS-FR-003, SS-FR-008)
- [x] **T019** Run local inference and render ranked results, preview, and timings. Complete when no network drawing request occurs and result tests pass. (SS-FR-003–SS-FR-006, SS-PRIV-001)

## M6 — Product experience

- [x] **T020** Complete localized interaction/error copy and route parity. Complete when keys and flows match in EN/ES. (SS-I18N-001)
- [x] **T021** Complete theme, responsive, accessibility, and reduced-motion audits. Complete when automated and documented manual checks pass. (SS-A11Y-001, SS-A11Y-002, SS-I18N-002)

## M7 — Technical transparency

- [x] **T022** Publish model metadata and evaluation views. Complete when model and evidence fields are accessible and secondary to drawing. (SS-FR-007)
- [x] **T023** Publish system card, preprocessing contract, licenses, and dataset attribution. Complete when limitations and CC BY 4.0 obligations are explicit. (SS-ML-001, SS-ML-004)

## M8 — Deployment and hardening

- [x] **T024** Measure and enforce production performance budgets. Complete when browser/model/build measurements and justified limits are recorded. (SS-PERF-001–SS-PERF-003, SS-ML-007)
- [x] **T025** Run end-to-end production-base-path validation and deploy Pages. Complete when the public URL passes EN/ES and interaction smoke tests. (SS-DEP-001, SS-DEP-002)
- [x] **T026** Configure repository security and branch protection. Complete when required checks, Dependabot, CodeQL, secret scanning where available, and private vulnerability reporting are documented and enabled. (SS-DEP-002)
