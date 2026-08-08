.PHONY: install check test build pre-commit
install:
	uv sync --project ml --group dev --group baseline
	npm.cmd install
	npm.cmd --prefix apps/web install
check:
	uv run --project ml --group baseline ruff check ml
	uv run --project ml --group baseline ruff format --check ml
	uv run --project ml --group baseline pyright
	npm.cmd run lint
	npm.cmd run format:check
	npm.cmd run check
test:
	uv run --project ml --group baseline pytest
	npm.cmd test
build:
	npm.cmd run build
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
