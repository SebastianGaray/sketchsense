# Version 2 release validation

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
