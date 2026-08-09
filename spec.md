# SketchSense Product Specification

Status: Foundation approved for implementation. Version 1 inference remains planned.

## Problem and goals

SketchSense must demonstrate browser-based neural-network inference through a simple, understandable drawing experience. It is for engineering managers, technical recruiters, software and ML engineers, data professionals, and portfolio visitors.

Version 1 must let a visitor draw with mouse, pen, or touch, clear the canvas, normalize the drawing consistently, run inference locally, see three ranked predictions with understandable confidence, inspect the normalized input, inspect model metadata, and see secondary evaluation evidence. It must support English and Spanish, System/Light/Dark themes, responsive layouts, and static hosting. No drawing may leave the browser.

## Non-goals

Version 1 excludes accounts, cloud inference, saved drawings, social features, hundreds of classes, browser training, generative AI, mandatory WebGPU, and production-readiness claims.

## Functional requirements

- **SS-FR-001 Drawing:** a visible canvas accepts mouse, pen, and touch strokes without scrolling during an active stroke; pointer capture prevents broken strokes. The canvas remains responsive and scales across supported viewports.
- **SS-FR-002 Clear:** an accessible Clear action removes all strokes, resets derived input and predictions, and is disabled when there is nothing to clear.
- **SS-FR-003 Prediction:** after a non-empty drawing and a ready model, the user can request local inference. Empty input, loading, unsupported runtime, and inference failures have actionable, non-technical messages.
- **SS-FR-004 Results:** the three highest-scoring classes appear in deterministic descending order with localized labels, percentages, and a non-color-only confidence visualization.
- **SS-FR-005 Input inspection:** the visitor can inspect the exact 28 x 28 grayscale image supplied to the model, with a textual explanation.
- **SS-FR-006 Latency:** measured preprocessing and inference durations are shown separately and labeled as measurements for the current device, never as guarantees.
- **SS-FR-007 Transparency:** model version, architecture, class count/order, input shape, artifact size, training timestamp, dataset subset, and evaluation summary are visible or linked from the experience.
- **SS-FR-008 Loading and errors:** model loading exposes an accessible status update; failures do not create fake results and offer a retry where recovery is possible.
- **SS-FR-009 Navigation:** a localized, clearly visible link returns to `https://sebastiangaray.github.io/`.
- **SS-FR-010 Progressive prediction:** after the model is ready, predictions update on a bounded debounce while drawing and immediately after a completed stroke; manual prediction remains available and inference never runs for every pointer event.
- **SS-FR-011 Drawing guidance:** visitors can adjust stroke width and open localized guidance listing every supported category with non-dataset illustrative prompts and practical tips.
- **SS-FR-012 Persistent demo navigation:** Canvas, Examples, Model, and About are distinct localized pages that share one persistent menu. Every secondary page provides a direct menu route back to Canvas and indicates the current page.

## Machine-learning requirements

- **SS-ML-001 Dataset:** use a versioned, deterministic subset of Google Quick, Draw!, with deterministic train/validation/test splits, stable sample identifiers, no overlap, and no raw dataset committed or shipped to the browser.
- **SS-ML-002 Classes:** version a deterministic order of 16 verified official categories: apple, bicycle, bird, book, car, cat, chair, cloud, cup, dog, fish, flower, house, key, star, and tree.
- **SS-ML-003 Models:** compare a simple documented baseline with an intentionally compact CNN; record seeds, environment, hyperparameters, dataset manifest, checkpoints, and training timestamps.
- **SS-ML-004 Evaluation:** report per-class and aggregate held-out metrics, confusion evidence, limitations, and the baseline comparison without data leakage.
- **SS-ML-005 Parity:** validate PyTorch-to-ONNX outputs within a documented numeric tolerance and Python-to-TypeScript preprocessing against deterministic shared fixtures.
- **SS-ML-006 Artifacts:** publish a schema-versioned manifest containing class order, tensor contract, hashes, sizes, provenance, metrics references, and compatible application/model versions.
- **SS-ML-007 Size:** target an ONNX artifact below 5 MB. A result from 5 MB through the hard ceiling of 20 MB requires measured justification. Artifacts above 20 MB fail acceptance.

## Preprocessing and performance

- **SS-PERF-001 Interaction:** pointer rendering must stay responsive; expensive preprocessing and inference must not run for every pointer event.
- **SS-PERF-002 Runtime:** inference is designed for low-latency CPU/WASM. Latency claims are added only after measurements on documented devices.
- **SS-PERF-003 Footprint:** ship no raw training data and only necessary fonts, scripts, model artifacts, fixtures, and evaluation summaries. Production asset sizes are measured and recorded before completion.
- **SS-PERF-004 Contract:** a non-empty drawing is cropped to padded bounds, aspect-fit, centered, converted to grayscale, normalized, and emitted as `[1, 1, 28, 28]`; Python and browser results must match the contract in `plan.md`.

## Privacy, accessibility, localization, and themes

- **SS-PRIV-001:** strokes, raster images, tensors, and predictions stay on-device. No upload, remote inference, drawing telemetry, visitor tracking, backend, database, or authentication is allowed.
- **SS-A11Y-001:** actions are keyboard accessible, focus uses a visible two-pixel offset ring, targets are at least 44 px where practical, status changes use an accessible live region, and contrast is sufficient in every theme.
- **SS-A11Y-002:** confidence has text and numeric alternatives, reduced-motion is honored, and the page has semantic landmarks, headings, language, and a skip link. Keyboard-only drawing equivalence is not promised; this limitation is stated honestly while all surrounding actions remain accessible.
- **SS-I18N-001:** `/en/` and `/es/` provide equivalent natural-language content with no flag icons; `/` redirects to English and language switching retains the corresponding experience.
- **SS-I18N-002:** System is the initial theme, follows OS changes, and Light/Dark overrides persist locally without a flash of the wrong theme.

## Deployment

- **SS-DEP-001:** produce a backend-free static build for GitHub Pages at `https://sebastiangaray.github.io/sketchsense/`, with base-path-safe assets and routes.
- **SS-DEP-002:** pull requests validate without deployment; reviewed `main` commits deploy with least-privilege GitHub Pages permissions and explicit concurrency.

## Project-level acceptance

Version 1 is complete only when the functional experience uses a validated model rather than placeholders; all requirements above have traceable completed tasks and tests; EN/ES and three themes work at desktop, tablet, mobile, and 20 rem width without horizontal overflow; accessibility checks and documented manual review pass; model, preprocessing, evaluation, privacy, licensing, system-card, and limitations evidence is public; production budgets are measured; CI and Pages deployment pass; and no acceptance criterion is represented as complete before its evidence exists.
