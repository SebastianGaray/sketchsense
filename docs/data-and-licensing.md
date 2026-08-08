# Data and licensing

SketchSense uses a bounded subset of the [Google Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset). Google describes the source as 50 million drawings across 345 categories and provides per-category 28 x 28 NumPy bitmap files.

The source dataset is made available by Google, Inc. under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/). SketchSense does not own the source drawings. Any redistributed derived metadata and model documentation must preserve this attribution and link to the license.

The `small-v1` development profile selects only the sixteen classes in `classes.v1.json`. It retrieves 200 contiguous samples per class from a deterministic hash-derived source offset using bounded HTTP byte ranges. Local pixels live under ignored `ml/data/`; raw category arrays and the cached subset are not committed or shipped with the web application. The committed manifest summary, inspection grid, and baseline report are derived documentation artifacts.

Quick, Draw! contributions can contain cultural, geographic, recognition, and moderation biases. The small subset is not representative of all drawing styles and must not support production or human-ability claims.
