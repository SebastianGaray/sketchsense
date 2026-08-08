.PHONY: install check test build pre-commit
install:
	uv sync --project ml --group dev
	npm.cmd install
	npm.cmd --prefix apps/web install
check:
	uv run --project ml ruff check ml
	uv run --project ml ruff format --check ml
	uv run --project ml pyright
	npm.cmd run lint
	npm.cmd run format:check
	npm.cmd run check
test:
	uv run --project ml pytest
	npm.cmd test
build:
	npm.cmd run build
pre-commit:
	uvx pre-commit run --all-files
