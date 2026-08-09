# Model v2 experiment report

Model v2 improves general recognition while preserving browser-local inference and the v1 runtime footprint. The released compact model reaches 82.31% accuracy, 82.20% macro F1, 92.63% top-3 accuracy, and 60% worst-class recall on a fresh 1,600-example test.

The experiment matrix compared the original compact architecture without augmentation, the same architecture with canvas-domain augmentation, a widened batch-normalized CNN, and a depthwise-separable CNN across three fixed seeds. The widened model produced the strongest validation and initial test quality but was rejected after its p95 runtime regression exceeded 20%. Strong augmentation harmed the compact model's weakest category on a replacement test. A moderate mix without thinning was selected using validation evidence and passed every gate on a third non-overlapping locked test.

The final per-class recall is: apple 91%, bicycle 89%, bird 60%, book 92%, car 84%, cat 66%, chair 90%, cloud 84%, cup 82%, dog 62%, fish 82%, flower 80%, house 93%, key 84%, star 92%, and tree 86%. Bird, dog, and cat remain the most important quality limitations. Confidence is not a calibrated probability, although expected calibration error is measured at 3.30%.

The canvas remains 640 x 640 internally. It already provides far more pointer resolution than the 28 x 28 model input, and preprocessing crops and recenters occupied bounds. Increasing the backing buffer would add memory and preprocessing work without adding model information. Scale, translation, rasterization, and stroke-width robustness are addressed during training instead.

Raw Quick, Draw! pixels, visitor drawings, and intermediate checkpoints are not committed. The v1 ONNX model and manifest remain available for rollback.
