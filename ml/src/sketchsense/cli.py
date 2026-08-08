"""Command-line interface for deterministic data and baseline workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from sketchsense.data.dataset import (
    DatasetProfile,
    prepare_dataset,
    validate_dataset,
    write_dataset_summary,
)
from sketchsense.data.inspection import create_sample_grid
from sketchsense.models.baseline import train_baseline, validate_baseline
from sketchsense.preprocessing.fixtures import validate_fixtures, write_fixtures

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = REPOSITORY_ROOT / "ml" / "data" / "small-v1"
DEFAULT_ARTIFACTS = REPOSITORY_ROOT / "artifacts" / "baseline"
DEFAULT_FIXTURES = REPOSITORY_ROOT / "fixtures" / "preprocessing.v1.json"


def parser() -> argparse.ArgumentParser:
    command_parser = argparse.ArgumentParser(prog="sketchsense")
    subcommands = command_parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser(
        "dataset-prepare", help="Download the bounded official bitmap subset"
    )
    prepare.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    validate = subcommands.add_parser("dataset-validate", help="Validate cached dataset contracts")
    validate.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    fixtures = subcommands.add_parser(
        "preprocessing-fixtures", help="Write deterministic shared fixtures"
    )
    fixtures.add_argument("--output", type=Path, default=DEFAULT_FIXTURES)
    fixture_validation = subcommands.add_parser(
        "preprocessing-validate", help="Validate shared preprocessing fixtures"
    )
    fixture_validation.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    inspect = subcommands.add_parser(
        "dataset-inspect", help="Create a bounded representative sample grid"
    )
    inspect.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    inspect.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "dataset" / "sample-grid.v1.png",
    )
    train = subcommands.add_parser(
        "baseline-train", help="Train and evaluate the logistic baseline"
    )
    train.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    train.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS)
    evaluate = subcommands.add_parser(
        "baseline-evaluate", help="Validate and print baseline test metrics"
    )
    evaluate.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    return command_parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.command == "dataset-prepare":
        prepare_dataset(arguments.output, DatasetProfile())
        result = write_dataset_summary(
            arguments.output, REPOSITORY_ROOT / "artifacts" / "dataset" / "dataset-summary.v1.json"
        )
    elif arguments.command == "dataset-validate":
        result = validate_dataset(arguments.dataset)
    elif arguments.command == "preprocessing-fixtures":
        result = write_fixtures(arguments.output)
    elif arguments.command == "preprocessing-validate":
        result = {"cases": validate_fixtures(arguments.fixtures)}
    elif arguments.command == "dataset-inspect":
        result = create_sample_grid(arguments.dataset, arguments.output)
    elif arguments.command == "baseline-train":
        result = train_baseline(arguments.dataset, arguments.output)
    elif arguments.command == "baseline-evaluate":
        result = validate_baseline(arguments.artifacts)
    else:  # pragma: no cover - argparse enforces available commands
        raise AssertionError(arguments.command)
    print(json.dumps(result, sort_keys=True) if isinstance(result, dict) else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
