# Version 3 release validation

## Model v3 evidence

- ONNX model: 1,707,724 bytes, below the 5 MB target.
- Held-out vector test: 94.63% accuracy, 94.61% macro F1, 98.76% top-3 accuracy, and 79.8% worst-class recall across 8,000 drawings.
- Weak-class improvement: cat 89.2% recall, bird 92.0%, and dog 79.8%.
- Training profile: 160,000 train, 8,000 validation, and 8,000 held-out official simplified-vector drawings.
- Input remains `[1, 1, 28, 28]`. A vector-native 56 x 56 cache was generated, but training was stopped when free memory fell to 1.59 GB on the 8 GB development device. The 28px candidate already improved macro F1 by 12.41 percentage points.
- The examples page contains correctly classified held-out prompts. Confidence UX abstains below a 0.55 leading score or a 0.15 top-two margin.

The model comparison and per-class recall charts are deterministic SVG artifacts derived from the evaluation summary. Exact training-loss curves were not published because the first completed training run reached checkpoint creation before a Windows console encoding failure interrupted history serialization. The checkpoint was preserved and its held-out evaluation and ONNX parity were rerun successfully.

## Previous model v2

## Production measurements

Measured from the GitHub Pages production build on 2026-08-08:

- ONNX model: 441,021 bytes, below the 5 MB target.
- ONNX Runtime SIMD/WebAssembly runtime: 21,872,216 bytes, the largest required asset.
- Application JavaScript entry: approximately 398 KB uncompressed.
- Raw dataset shipped: 0 bytes.
- Model release-test metrics: 82.31% accuracy, 82.37% macro precision, 82.31% macro recall, 82.20% macro F1, 92.63% top-3 accuracy, and 60% worst-class recall.

The application uses one runtime worker thread, lazy browser caching, a 441 KB model, no external application data, and honest loading status. Model v2 preserves the v1 architecture after a higher-capacity candidate failed the 20% p95 latency-regression budget. The recorded CPU ONNX benchmark measured 0.0421 ms v2 p95 against 0.0428 ms v1 p95 on the development device.

## Automated evidence

Vitest covers bilingual dictionary parity and all shared Python preprocessing fixtures within `1e-5`. Pytest covers dataset contracts, model architecture, held-out evidence dimensions, ONNX Runtime parity, malformed input rejection, and artifact integrity. Playwright covers production-base-path loading, real model initialization, pointer drawing, inference, three results, clear, language navigation, theme selection, privacy-oriented request inspection, and 320 px overflow.

## Manual review

The application was reviewed at 320, 768, and 1280 CSS pixels in English and Spanish. System, Light, and Dark controls, visible focus, reduced-motion rules, semantic landmarks, live model status, textual confidence, portfolio/GitHub links, favicon, canonical metadata, sitemap, robots, and 404 output were inspected. Freehand drawing remains pointer-oriented and this limitation is stated beside the canvas.
