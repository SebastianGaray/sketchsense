# SketchSense

SketchSense is an interactive machine-learning portfolio demo in development. Its target experience turns a browser canvas drawing into three predictions using deterministic preprocessing and a compact ONNX neural network running entirely on the visitor's device.

Current status: the deterministic 16-class dataset pipeline, canonical Python canvas preprocessing, shared parity fixtures, measured logistic-regression baseline, bilingual application shell, quality automation, and GitHub Pages deployment are present. The CNN, ONNX export, Canvas interaction, and browser inference are not implemented yet.

## Architecture

The planned pipeline is a curated 16-class subset of Google Quick, Draw! → deterministic Python preprocessing → baseline and compact PyTorch CNN → held-out evaluation → ONNX export/parity validation → Astro with strict TypeScript → Canvas API and ONNX Runtime Web. There is no runtime Python, backend, database, authentication, remote inference, analytics, or paid infrastructure.

Requirements live in `spec.md`, implementation decisions in `plan.md`, and executable progress in `tasks.md`. `DESIGN.md` maps the shared portfolio identity to this product.

## Local setup

Requirements are Python 3.12+, uv, Node.js 22+, and npm.

```sh
make install
make check
make test
make build
make pre-commit
make dataset-prepare
make dataset-validate
make preprocessing-validate
make baseline-train
make baseline-evaluate
```

On Windows without Make, run the commands listed in `Makefile` directly and use `npm.cmd`. Start the web shell with `npm.cmd run dev`; Astro serves the project under `/sketchsense/`.

The static deployment target is [sebastiangaray.github.io/sketchsense](https://sebastiangaray.github.io/sketchsense/). The drawing privacy contract requires that strokes, images, tensors, and predictions remain in the browser.

The data source is Google's Quick, Draw! Dataset, licensed under CC BY 4.0. The small profile retrieves about 2.5 MB of pixel ranges across selected categories; raw category arrays and the local cache are not committed or shipped to the browser. See `docs/data-and-licensing.md`.

The first measured logistic baseline reaches 55.63% test accuracy, 55.51% macro F1, and 74.38% top-3 accuracy on the deterministic small profile. It is comparison evidence, not a production claim or the future browser model.

SketchSense is part of [Sebastián Garay's portfolio](https://sebastiangaray.github.io/).
