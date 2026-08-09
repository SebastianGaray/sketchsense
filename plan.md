# SketchSense Implementation Plan

## Architecture and environments

The repository is a small monorepo: `apps/web` contains a static Astro 7 application with strict TypeScript and browser-owned Canvas/ONNX behavior; `ml` is a Python 3.12+ uv project with `src/sketchsense` and pytest tests; `artifacts` holds only bounded, versioned model/evaluation outputs; `docs` holds system, data, and licensing notes. Root commands orchestrate both environments.

Python uses uv, NumPy, PyTorch, ONNX, ONNX Runtime, pytest, Ruff, and Pyright. Heavy ML dependencies are grouped as an optional `ml` dependency group so foundation validation stays quick; the lock includes them for reproducibility. Web uses Astro, strict TypeScript, ONNX Runtime Web when inference is implemented, Vitest, and Playwright for important flows. No React or Tailwind is required.

## Dataset and class contract

The official Quick, Draw! documentation and `categories.txt` are the source of truth. The initial v1 class-order target is 16 categories: `apple`, `bicycle`, `bird`, `book`, `car`, `cat`, `chair`, `cloud`, `cup`, `dog`, `fish`, `flower`, `house`, `key`, `star`, `tree`. This mixes silhouette-friendly classes with useful confusions such as cat/dog, bird/fish, and flower/tree.

The small profile reads only named official 28 x 28 NPY bitmap objects. It first reads each NPY header, derives a stable SHA-256 start index from the profile seed and class, then retrieves one contiguous, wrapping byte range containing 200 samples. This transfers about 2.5 MB of pixels rather than sixteen complete category files. Source indices are stable sample identifiers. Each class is independently ordered by SHA-256 of the split seed, class, and source index, then assigned exactly 140 train, 30 validation, and 30 test samples. Duplicate source identifiers or image hashes across splits fail validation. The local `ml/data` cache is ignored; only bounded metadata, inspection, and evaluation artifacts are committed. Class order lives in a versioned JSON contract and is never inferred from filesystem order. Quick, Draw! data is attributed to Google under CC BY 4.0.

## Preprocessing contract

The authoritative input is an RGBA raster of the drawing surface. Background is opaque white (`255`); strokes are opaque black (`0`). Alpha is composited over white. A pixel is foreground when luminance is below `250`. Empty inputs return an explicit empty result.

For non-empty input, compute the inclusive foreground bounds, expand by 10% of the larger bound dimension with a minimum two-source-pixel pad, clamp to the raster, and crop. Preserve aspect ratio and fit into a `20 x 20` content box. Resize using bilinear interpolation with half-pixel coordinate mapping, place it on a centered `28 x 28` white canvas, and resolve an odd leftover pixel to the right/bottom. Convert using sRGB luminance `round(0.2126R + 0.7152G + 0.0722B)`. Invert and normalize once as `(255 - luminance) / 255`, yielding float32 background `0`, foreground `1`, row-major NCHW tensor `[1, 1, 28, 28]`. No mean/std standardization or thresholding follows resize.

Official NPY pixels are already centered 28 x 28 grayscale bitmaps with white strokes (`255`) on black background (`0`). Dataset-source normalization therefore converts directly to float32 foreground `1` by dividing by 255 and does not re-crop or resize. Canvas normalization remains the separate canonical RGBA path above. Python and TypeScript share small deterministic RGBA fixtures plus expected float tensors. Tests compare outputs within `1e-5`; bounds, empty handling, shape, range, centering, and interpolation edge cases receive unit tests. Any contract change increments its schema version and invalidates incompatible artifacts.

## Models, training, evaluation, and export

The baseline is scikit-learn logistic regression on flattened normalized pixels, using multinomial softmax behavior, L2 regularization, fixed seed, `lbfgs`, and recorded convergence metadata. It is fitted on the deterministic train split, inspected on validation, and reported once on test. A compact coefficient/intercept NPZ supports reproducibility but is not a browser artifact.

The primary `compact-cnn-v1` model has 106,256 parameters. Its feature extractor is `Conv2d(1,16,3,padding=1) → ReLU → MaxPool2d(2) → Conv2d(16,32,3,padding=1) → ReLU → MaxPool2d(2)`. The classifier is `Flatten → Linear(1568,64) → ReLU → Linear(64,16)`. Adam uses learning rate `0.001`, weight decay `0.0001`, batch size `64`, and at most 30 epochs. Checkpoint selection minimizes validation cross-entropy with minimum improvement `0.0005` and patience six. The fixed seed is `20260808`; deterministic PyTorch algorithms and a single-threaded, zero-worker CPU loader are enforced.

Training fixes Python/NumPy/PyTorch seeds, records deterministic-mode limitations, config, dependency lock, data manifest, class order, epoch metrics, best-checkpoint rule, and hardware. Test data is opened only for final evaluation. Reports include accuracy, macro precision/recall/F1, per-class support and recall, confusion matrix, baseline delta, and measured limitations.

The measured `small-v1` logistic baseline trains on 2,240 samples and evaluates on 480 validation and 480 test samples. With seed `20260808`, scikit-learn 1.9.0, L2 `C=1.0`, and 172 converged LBFGS iterations, test accuracy is `0.55625`, macro precision `0.55951`, macro recall `0.55625`, macro F1 `0.55508`, and top-3 accuracy `0.74375`. These are baseline evidence only. They confirm the classes are learnable at small scale while leaving material room for the CNN; they are not performance targets or production claims.

Export uses a named float32 input of shape `[1,1,28,28]`, fixed batch one unless measurements justify dynamic batch, a supported opset, and logits output. ONNX Runtime Python is compared with PyTorch on shared fixtures and a bounded held-out sample using documented absolute/relative tolerances. The artifact manifest is JSON Schema validated and includes SHA-256, bytes, model/version, preprocessing schema, class-order hash, ONNX opset/runtime compatibility, data/training provenance, metrics, and timestamps.

The final checkpoint is epoch 13; early stopping completed at epoch 19. One held-out test evaluation measures accuracy `0.72708`, macro precision `0.74699`, macro recall `0.72708`, macro F1 `0.73186`, and top-3 accuracy `0.86875`, improving on every recorded baseline metric. The single-file ONNX artifact uses opset 18, fixed batch one, `input` and `logits` names, and applies no softmax internally. Its measured size is 441,021 bytes. Three shared fixtures match PyTorch within absolute tolerance `1e-5` and relative tolerance `1e-4`; the observed maximum absolute difference is `1.19e-6`.

## Browser application

The Canvas controller owns pointer capture, coordinate scaling for device pixel ratio, stroke history, redraw, clear state, and resize preservation. It does not infer on pointer movement. A preprocessing module creates the exact 28 x 28 tensor and preview. An inference adapter loads a base-path-safe ONNX artifact once, selects WASM execution by default, validates manifest compatibility, warms up when suitable, and reports preprocessing and inference timings separately.

UI state is explicit: `idle`, `model-loading`, `ready-empty`, `ready-drawn`, `preprocessing`, `inferencing`, `result`, and recoverable/fatal `error`. Controls derive their enabled state from this model. The foundation shell intentionally shows development status without fake canvas controls or predictions.

English and Spanish content uses typed dictionaries with matching keys and route helpers. `/en/` and `/es/` are generated pages; `/` redirects to `/en/`. A pre-paint inline script applies stored `system|light|dark` preference, System follows media changes, and the UI communicates selection with text and `aria-pressed` rather than color. Visual tokens reproduce the canonical warm palette, typography roles, one-pixel borders, four-pixel radii, restrained shadows, focus ring, and attribution, while the drawing/input/prediction composition provides project personality.

## Accessibility and quality

Use landmarks, one h1, skip navigation, live loading/result status, native buttons, visible focus, 44 px targets, textual confidence, reduced motion, and an adjacent honest canvas limitation. Responsive verification covers 320, 768, and 1280 CSS pixels, both languages and themes, zoom, keyboard navigation, and no horizontal overflow.

Python validation is Ruff format/check, Pyright, and pytest. Web validation is Astro check, Vitest, production build, and later Playwright for drawing and navigation flows. Pre-commit mirrors cheap checks. CI pins actions to immutable SHAs, gives validation read-only permissions, caches through official setup actions, and never deploys pull requests. Pages deployment builds the validated static artifact from `main`, uses explicit concurrency, and receives only Pages/id-token permissions.

## Documentation, budgets, and risks

`docs/system-card.md` will explain intended use, data, preprocessing, architecture, metrics, parity, privacy, limitations, and responsible interpretation. Dataset attribution, CC BY 4.0 obligations, source code license, model artifact terms, and third-party notices must be explicit before release.

The ONNX target is under 5 MB and ceiling is 20 MB with written justification. Raw data is zero browser bytes. Build output will record total JS, largest chunk, fonts, model, and evaluation assets before release; budgets are then set from evidence. Risks include dataset bias/noise, ambiguous sketches, preprocessing drift, WASM/browser support, cold model download, misleading confidence, mobile canvas ergonomics, font/network failure, and base-path mistakes. Mitigations are bounded data manifests, parity fixtures, calibrated copy, error states, fallbacks, responsive tests, and production-like builds.

The released browser state machine loads the fixed contract from a base-path-safe URL, exposes retry on recoverable failures, bounds progressive inference with a trailing timer instead of running for every pointer event, and clears drawing, preview, timings, and results together. The measured v1 build ships a 441,021-byte model, an approximately 398 KB application entry, and the 21,872,216-byte ONNX Runtime SIMD WebAssembly runtime. No raw dataset is shipped. The runtime is the largest cold-load cost and is documented separately from model size.

Post-v1 interaction refinement uses one trailing 450 ms prediction timer during an active stroke and a near-immediate prediction after pointer release. Each new request supersedes older results, preserving responsive drawing and avoiding inference per pointer event. Stroke width is adjustable from 8 through 32 canvas pixels with a thinner default of 14. Localized category guidance uses authored symbolic prompts and never publishes source dataset sketches.

## Milestones

M1 establishes SDD, design, environments, CI, bilingual/theme shell, and static deployment. M2 produces deterministic subset and preprocessing parity. M3 trains and compares baseline/CNN. M4 evaluates, exports, validates, and manifests artifacts. M5 integrates canvas and local inference. M6 completes responsive localized product behavior. M7 publishes evidence/system card. M8 measures, audits, deploys, and hardens repository settings.
