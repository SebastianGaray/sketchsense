# Model v3 plan

1. Add parameterized canonical preprocessing and vector-to-canvas rasterization.
2. Fetch a deterministic, disjoint Quick, Draw! vector profile with bounded local caching.
3. Train compact and widened candidates at 28 x 28, then run a 56 x 56 candidate only if the vector-native comparison is warranted.
4. Evaluate official holdout quality, canvas benchmark quality, latency, size, and abstention coverage.
5. Publish the strongest browser-appropriate candidate and generated evidence.
6. Validate every example prompt with the published ONNX artifact and expose uncertainty in the UI.

The release decision uses real browser latency as an absolute interaction budget. A tiny absolute increase is not rejected only because its relative percentage is large.

