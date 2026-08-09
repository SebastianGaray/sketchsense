# SketchSense

SketchSense is a bilingual browser drawing classifier with transparent, on-device ONNX inference. Try the [public demo](https://sebastiangaray.github.io/sketchsense/).

Visitors draw with mouse, pen, or touch, inspect the exact normalized 28 × 28 model input, and receive three ranked predictions with device-measured preprocessing and inference times. The static application has no backend, analytics, accounts, drawing uploads, or remote inference.

## Evidence

The deterministic `small-v1` profile uses 3,200 bounded samples from 16 official Google Quick, Draw! categories: 2,240 train, 480 validation, and 480 test. Raw data is neither committed nor shipped. Quick, Draw! is attributed to Google under CC BY 4.0.

The selected 106,256-parameter compact CNN reaches 72.71% held-out accuracy, 73.19% macro F1, and 86.88% top-3 accuracy. Its fixed-batch ONNX opset 18 artifact is 441,021 bytes. See [the model card](docs/model-card.md) and [data and licensing notes](docs/data-and-licensing.md).

## Architecture

Deterministic Python data preparation and PyTorch training produce schema-validated, checksummed artifacts. Astro and strict TypeScript own the Canvas API, contract-matched preprocessing, and ONNX Runtime Web inference. Shared fixtures enforce Python/TypeScript preprocessing parity within `1e-5`; PyTorch/ONNX logits are validated within documented tolerances.

## Local development

Requirements: Python 3.12+, uv, Node.js 22+, and npm.

```sh
uv sync --project ml --frozen --group dev --group baseline --group ml
npm ci
npm --prefix apps/web ci
make check
make test
make build
npm run test:e2e
make pre-commit
```

The first Playwright run also needs `npm --prefix apps/web exec playwright install chromium`. On Windows, use `npm.cmd` or run the equivalent commands from `Makefile`.

Dataset preparation and training are explicit, comparatively expensive workflows. They are not part of CI. CI validates the committed fixtures, artifact schemas/checksums, model size, preprocessing parity, ONNX parity tests, Python and web quality gates, Playwright flows, and the production build without retraining.

## Limitations

The model uses a small subset with only 200 examples per class. Ambiguous, faint, unusual, or out-of-distribution drawings may be wrong. Confidence is not calibrated probability. The WebAssembly runtime is the largest deployed asset and cold loading depends on the visitor's connection and browser. Freehand drawing has no keyboard equivalent, though every surrounding action is keyboard accessible.

## Contributing

Future changes follow `branch → push → pull request → CI → review → merge to main → deployment`. Create a focused `agent/<short-description>` branch, use English Conventional Commits, keep documentation and SDD aligned, and do not bypass required checks or review.

Requirements live in [spec.md](spec.md), architecture in [plan.md](plan.md), progress in [tasks.md](tasks.md), and visual decisions in [DESIGN.md](DESIGN.md).

Future model-quality work is scoped separately in [the model v2 SDD](sdd/model-v2/spec.md). It requires a fresh test partition and measurable cross-class improvement before replacing the released model.
