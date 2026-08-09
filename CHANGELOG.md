# Changelog

## 1.0.0 - 2026-08-09

First stable release of SketchSense.

### Included

- Deterministic vector-native training pipeline and held-out evaluation evidence.
- Checksummed ONNX model with Python/browser preprocessing and inference parity tests.
- Bilingual, static, on-device drawing experience with uncertainty-aware results.
- Accessible theme, language, navigation, production base-path, and mobile behavior.
- Safe DOM prediction rendering, CodeQL, dependency audits, and an MIT license.

### Known limitations

- The classifier covers 16 Quick, Draw! categories.
- Confidence scores are not calibrated probabilities.
- Freehand drawing remains pointer-oriented and has no keyboard equivalent.
- The WebAssembly runtime is the largest deployed asset.
