# Data and licensing

SketchSense uses a bounded subset of the [Google Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset). Google describes the source as 50 million drawings across 345 categories and provides per-category 28 x 28 NumPy bitmap files.

The source dataset is made available by Google, Inc. under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/). SketchSense does not own the source drawings. Any redistributed derived metadata and model documentation must preserve this attribution and link to the license.

The `small-v1` development profile selects only the sixteen classes in `classes.v1.json`. It retrieves 200 contiguous samples per class from a deterministic hash-derived source offset using bounded HTTP byte ranges. Local pixels live under ignored `ml/data/`; raw category arrays and the cached subset are not committed or shipped with the web application. The committed manifest summary, inspection grid, and baseline report are derived documentation artifacts.

The released `medium-v2` profile retrieves 1,000 unused examples per class, split into 800 train, 100 validation, and 100 initially locked test samples. Identifier checks prove no overlap with v1. Candidate artifact rejection required replacement tests, so the final metrics use another 100 official examples per class with no overlap with v1, medium-v2, or the earlier replacement test. Only bounded summaries and identifiers are versioned; pixels remain ignored.

Training augmentation approximates canvas input with bounded translation, scale, rotation, rasterization, and occasional thick-stroke variation. It does not flip drawings. The selected run mixes one augmented example for every three canonical examples because stronger thinning and augmentation reduced validation stability.

Quick, Draw! contributions can contain cultural, geographic, recognition, and moderation biases. The small subset is not representative of all drawing styles and must not support production or human-ability claims.
