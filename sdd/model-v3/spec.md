# Model v3 requirements

## Goal

Improve recognition of freehand canvas drawings without presenting weak closed-set scores as reliable answers.

## Requirements

- Training input MUST pass through the same crop, padding, resize, centering, and normalization contract as browser input.
- Model selection MUST compare 28 x 28 and 56 x 56 vector-rasterized candidates when both are practical to run.
- Training MUST use at least 10,000 official Quick, Draw! vector drawings per category.
- Evaluation MUST keep source drawings disjoint and include a canvas-style benchmark that is not used for optimization.
- The examples page MUST use model-validated prompts and state what was validated.
- The UI MUST abstain when the leading score and separation do not provide useful evidence.
- Published artifacts, application contracts, tests, model card, and release evidence MUST agree.

## Acceptance criteria

- The selected candidate improves the canvas benchmark over model v2 and does not regress official-test macro F1.
- Cat, dog, and bird canvas cases are represented in the benchmark.
- Example prompts are classified correctly by the selected artifact.
- Browser preprocessing parity is covered by shared fixtures.
- Formatting, linting, type checking, unit tests, production build, artifact validation, and required CI checks pass.

