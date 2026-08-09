# Canvas input v4 requirements

## Goal

Preserve freehand structure when producing the fixed 28 × 28 model tensor and offer an optional direct pixel-grid drawing mode for inspection and comparison.

## Requirements

- Freehand input MUST be retained as ordered vector strokes until prediction.
- Vector conversion MUST crop occupied stroke bounds, preserve aspect ratio, center the result, use bounded final stroke thickness, and antialias the 28 × 28 tensor.
- Users MUST be able to switch between freehand and direct 28 × 28 pixel drawing without leaving the canvas page.
- Pixel mode MUST render an enlarged, visible grid while supplying its exact 28 × 28 values to the model without a resize step.
- Switching modes MUST clear incompatible drawing state and explain that behavior.
- Both modes MUST support mouse, pen, and touch, live prediction, clearing, and the existing browser-local privacy contract.

## Acceptance criteria

- Unit tests cover vector bounds, centering, thickness, antialiasing, empty input, and exact pixel passthrough.
- Browser tests draw and predict in both modes on the production preview.
- The 320 px viewport has no horizontal overflow.
- Formatting, linting, type checking, unit tests, browser tests, and the production build pass.
