# Canvas input v4 plan

1. Add a DOM-independent stroke-to-tensor rasterizer with supersampled coverage and bounded output thickness.
2. Keep freehand strokes as canvas-coordinate point sequences and use them as the prediction source.
3. Add a direct pixel tensor with enlarged grid rendering and continuous pointer interpolation.
4. Expose an accessible bilingual mode selector and mode-specific guidance.
5. Validate pure conversion contracts and complete browser interactions for both modes.
