.PHONY: install check test build audit pre-commit
install:
	uv sync --project ml --group dev --group baseline --group ml
	npm.cmd install
	npm.cmd --prefix apps/web install
check:
	uv run --project ml --group baseline --group ml ruff check ml
	uv run --project ml --group baseline --group ml ruff format --check ml
	uv run --project ml --group baseline --group ml pyright
	npm.cmd run lint
	npm.cmd run format:check
	npm.cmd run check
test:
	uv run --project ml --group baseline --group ml pytest
	npm.cmd test
build:
	npm.cmd run build
audit:
	uv run --project ml --group baseline --group ml pip-audit
	npm.cmd run audit
pre-commit:
	uvx pre-commit run --all-files
dataset-prepare:
	uv run --project ml --group baseline sketchsense dataset-prepare
dataset-validate:
	uv run --project ml --group baseline sketchsense dataset-validate
preprocessing-validate:
	uv run --project ml --group baseline sketchsense preprocessing-validate
baseline-train:
	uv run --project ml --group baseline sketchsense baseline-train
baseline-evaluate:
	uv run --project ml --group baseline sketchsense baseline-evaluate
cnn-train:
	uv run --project ml --group baseline --group ml sketchsense cnn-train
cnn-evaluate:
	uv run --project ml --group baseline --group ml sketchsense cnn-evaluate
onnx-export:
	uv run --project ml --group baseline --group ml sketchsense onnx-export
onnx-validate:
	uv run --project ml --group baseline --group ml sketchsense onnx-validate
artifacts-validate:
	uv run --project ml --group baseline --group ml sketchsense artifacts-validate
