# SketchSense Compact CNN Model Card

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

The intended browser deployment runs locally and does not upload drawings. The training subset is small, culturally and geographically biased, and contains ambiguous or noisy sketches. Test estimates have high uncertainty because each class has only 30 examples. Canvas preprocessing parity and the full browser inference flow remain separate tasks. Exact retraining floats may vary across PyTorch, BLAS, CPU, and operating-system versions despite deterministic settings.
