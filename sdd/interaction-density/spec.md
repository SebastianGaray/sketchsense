# Interaction density specification

## Requirements

- Remove instructional copy that repeats visible controls or content available on explanatory pages.
- Keep the canvas and its essential controls above secondary evidence.
- Show the leading live prediction before the canvas so it remains visible while drawing on mobile.
- Show stroke-width controls only in freehand mode.
- Preserve detailed ranked predictions, tensor evidence, timings, accessibility semantics, and bilingual behavior.

## Acceptance criteria

- The interaction page no longer renders the drawing hint, browser-privacy banner, mode explanation, or pointer/live-prediction note.
- The leading prediction updates with the ranked output and resets when the drawing clears.
- Switching to pixel mode hides the stroke-width control and switching back restores it.
- Desktop and 320-pixel layouts have no horizontal overflow and keep the leading prediction above the canvas.
