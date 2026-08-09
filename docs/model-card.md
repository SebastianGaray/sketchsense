# SketchSense Compact CNN Model Card

## Released model v3

Model v3 is a 422,608-parameter widened batch-normalized CNN trained from 160,000 official Quick, Draw! simplified vector sketches. Another 8,000 examples select training progress and 8,000 disjoint drawings form the held-out test. Every vector is rendered to a high-resolution white canvas and passed through the same crop, 10% padding, centered-content, and bilinear resize geometry used by browser input.

On the held-out vector test, v3 reaches 0.94625 accuracy, 0.94609 macro F1, and 0.98763 top-3 accuracy. Worst-class recall is 0.798 for dog. Cat reaches 0.892 recall and bird reaches 0.920, compared with 0.66 and 0.60 in v2. The 1,707,724-byte ONNX opset 18 artifact matches PyTorch within `1e-4` and remains below the 5 MB browser target.

The 28 x 28 contract remains deliberate. A vector-native 56 x 56 dataset was generated, but concurrent candidate training left only 1.59 GB free on the 8 GB development device. The run was stopped without publishing a partial result. The vector-native 28 x 28 candidate already improves macro F1 by 12.41 percentage points, so increasing browser memory and compute was not justified.

SketchSense remains a closed-set classifier: every input receives scores for sixteen categories. The interface therefore abstains when the leading score is below 0.55 or its margin over the second score is below 0.15. Abstention communicates insufficient evidence; it is not an additional learned class.

The examples page uses correctly classified held-out v3 drawings selected after training. They are validated prompts, not hand-authored icons and not additional training evidence.

## Previous model v2

Model v2 retains the 106,256-parameter architecture and float32 `[1, 1, 28, 28]` browser contract while replacing its training evidence. It uses 12,800 training and 1,600 validation examples from `medium-v2`, deterministic moderate canvas-domain augmentation, and seed 20260809. Two larger or smaller candidates were rejected on latency or quality evidence.

On the final fresh 1,600-example release test, v2 reaches 0.82313 accuracy, 0.82372 macro precision, 0.82313 macro recall, 0.82199 macro F1, and 0.92625 top-3 accuracy. Worst-class recall is 0.60 for bird, followed by dog at 0.62 and cat at 0.66. Expected calibration error is 0.03296. The 441,021-byte ONNX model matches PyTorch within `1e-5` and showed no p95 CPU-runtime regression against v1 in the recorded local benchmark.

The initially selected widened candidate reached stronger aggregate quality but was rejected because its measured p95 runtime regression exceeded the 20% release budget. A compact candidate with stronger augmentation then missed the worst-class gate on a fresh replacement test. Those outcomes remain recorded. The final moderate-augmentation candidate was selected by validation and evaluated on a third non-overlapping locked test.

The 640 x 640 drawing canvas and 28 x 28 tensor are intentionally unchanged. Quick, Draw! bitmap training data is natively 28 x 28, so enlarging those bitmaps would interpolate existing pixels rather than add information. A meaningful 56 x 56 experiment would need deterministic rasterization from the original vector strokes and a fresh accuracy, latency, and artifact-size comparison. The interface displays the exact 28 x 28 tensor at 168 CSS pixels so its contents remain easy to inspect.

## Historical model v1

## Intended use

Compact CNN v1 classifies a single normalized 28 x 28 sketch among sixteen versioned Quick, Draw! categories for an educational, browser-local portfolio demonstration. It returns logits; the application boundary will apply softmax and present the three highest scores.

## Non-intended use

The model is not suitable for production decisions, safety-critical use, biometric or ability assessment, content moderation, general image recognition, drawings outside the listed classes, or claims about people or populations. Its scores are not calibrated guarantees.

## Data

Training uses the deterministic `small-v1` profile of the Google Quick, Draw! Dataset: 200 samples per class, split into 140 train, 30 validation, and 30 held-out test examples per class. Google provides the source under CC BY 4.0. The selected classes are apple, bicycle, bird, book, car, cat, chair, cloud, cup, dog, fish, flower, house, key, star, and tree.

## Architecture and training

The 106,256-parameter network applies 3 x 3 convolutions from 1 to 16 and 16 to 32 channels, each followed by ReLU and 2 x 2 max pooling. A 1,568-to-64 ReLU classifier produces 16 logits. Adam uses learning rate 0.001, weight decay 0.0001, and batch size 64. Training is capped at 30 epochs with six-epoch early stopping on validation cross-entropy. The selected checkpoint is epoch 13; training stopped after epoch 19 without reading test data.

## Preprocessing and export

Official dataset bitmaps use direct `pixel / 255` float32 normalization. Browser canvas input will use the separate crop, pad, bilinear resize, center, grayscale, and inversion contract in `preprocessing.v1.json`. Both yield `[1, 1, 28, 28]`. ONNX v1 has a fixed batch-one float32 input named `input` and a `[1, 16]` logits output named `logits`. Softmax is intentionally outside the model for a clear portable boundary.

## Evaluation

The checkpoint was selected using validation loss, then evaluated once on 480 held-out examples. Test accuracy is 0.72708, macro precision 0.74699, macro recall 0.72708, macro F1 0.73186, and top-3 accuracy 0.86875. The logistic baseline scores 0.55625 accuracy, 0.55951 macro precision, 0.55625 macro recall, 0.55508 macro F1, and 0.74375 top-3 accuracy.

Dog has the lowest measured recall at 0.46667, followed by key at 0.56667 and bird at 0.63333. The largest directed test confusions are dog → bird (6), bicycle → dog (5), bird → dog (5), and key → fish (5). These observations describe only 30 test examples per class and do not establish general error causes.

## Privacy and limitations

The browser deployment runs locally and does not upload drawings. The training subset is small, culturally and geographically biased, and contains ambiguous or noisy sketches. Test estimates have high uncertainty because each class has only 30 examples. Shared fixtures validate Canvas preprocessing parity and Playwright exercises real browser inference. Exact retraining floats may vary across PyTorch, BLAS, CPU, and operating-system versions despite deterministic settings.
