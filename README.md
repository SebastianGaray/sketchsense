# SketchSense

SketchSense is a bilingual browser drawing classifier with transparent, on-device ONNX inference. Try the [public demo](https://sebastiangaray.github.io/sketchsense/).

Visitors draw with mouse, pen, or touch, inspect the exact normalized 28 × 28 model input, and receive three ranked predictions with device-measured preprocessing and inference times. The static application has no backend, analytics, accounts, drawing uploads, or remote inference.

## Evidence

The deterministic `vector-v3` profile uses 176,000 recognized vector drawings from 16 official Google Quick, Draw! categories: 160,000 train, 8,000 validation, and 8,000 held-out test examples. Strokes are rasterized through the same crop, padding, centering, and resize geometry used by the browser. Raw data is neither committed nor shipped. Quick, Draw! is attributed to Google under CC BY 4.0.

The selected 422,608-parameter CNN reaches 94.63% held-out accuracy, 94.61% macro F1, 98.76% top-3 accuracy, and 79.8% worst-class recall. Its fixed-batch ONNX opset 18 artifact is 1,707,724 bytes. The interface abstains on weak or closely matched scores. See [the model card](docs/model-card.md) and [data and licensing notes](docs/data-and-licensing.md).

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
npm run audit
npm run test:e2e
make pre-commit
```

The first Playwright run also needs `npm --prefix apps/web exec playwright install chromium`. On Windows, use `npm.cmd` or run the equivalent commands from `Makefile`.

Dataset preparation and training are explicit, comparatively expensive workflows. They are not part of CI. CI validates the committed fixtures, artifact schemas/checksums, model size, preprocessing parity, ONNX parity tests, Python and web quality gates, Playwright flows, and the production build without retraining.

## Limitations

The model still covers only 16 classes. Ambiguous, faint, unusual, or out-of-distribution drawings may be wrong, and bird, dog, and cat remain among the harder categories. Confidence is not calibrated probability. The WebAssembly runtime is the largest deployed asset and cold loading depends on the visitor's connection and browser. Freehand drawing has no keyboard equivalent, though every surrounding action is keyboard accessible.

## Contributing

Future changes follow `branch → push → pull request → CI → review → merge to main → deployment`. Create a focused `agent/<short-description>` branch, use English Conventional Commits, keep documentation and SDD aligned, and do not bypass required checks or review.

Specification-driven development records requirements and acceptance criteria before meaningful implementation. AI-assisted tools may support exploration, implementation, review, documentation, and test generation, but their output is treated as a proposal. Product intent, model tradeoffs, validation thresholds, and final approval remain human decisions. The public demo exposes this process and links directly to its versioned evidence.

Requirements live in [spec.md](spec.md), architecture in [plan.md](plan.md), progress in [tasks.md](tasks.md), and visual decisions in [DESIGN.md](DESIGN.md).

The current quality upgrade is recorded in [the model v3 SDD](sdd/model-v3/spec.md), including vector-native preprocessing, resolution evidence, uncertainty behavior, and release criteria.

Release history is recorded in [CHANGELOG.md](CHANGELOG.md).

## License

SketchSense source code is available under the [MIT License](LICENSE). The Quick, Draw! source data retains its separate CC BY 4.0 license and attribution described in [the data and licensing notes](docs/data-and-licensing.md).
