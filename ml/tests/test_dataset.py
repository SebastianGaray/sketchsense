import json
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator

from sketchsense.data.dataset import (
    SPLIT_COUNTS,
    deterministic_split,
    load_classes,
    stable_start_index,
)


def test_class_order_is_versioned_and_stable() -> None:
    assert load_classes() == (
        "apple",
        "bicycle",
        "bird",
        "book",
        "car",
        "cat",
        "chair",
        "cloud",
        "cup",
        "dog",
        "fish",
        "flower",
        "house",
        "key",
        "star",
        "tree",
    )


def test_sampling_start_is_deterministic_and_bounded() -> None:
    first = stable_start_index(20260808, "apple", 144_722, 200)
    assert first == stable_start_index(20260808, "apple", 144_722, 200)
    assert 0 <= first <= 144_522
    assert first != stable_start_index(20260808, "tree", 144_722, 200)


def test_split_is_deterministic_balanced_and_disjoint() -> None:
    indices = np.arange(1000, 1200, dtype=np.int64)
    first = deterministic_split(indices, "cat", 20260808)
    second = deterministic_split(indices, "cat", 20260808)
    np.testing.assert_array_equal(first, second)
    assert {code: int(np.sum(first == code)) for code in range(3)} == {0: 140, 1: 30, 2: 30}
    split_ids = [set(indices[first == code].tolist()) for code in range(3)]
    assert split_ids[0].isdisjoint(split_ids[1])
    assert split_ids[0].isdisjoint(split_ids[2])
    assert split_ids[1].isdisjoint(split_ids[2])
    assert sum(SPLIT_COUNTS.values()) == len(indices)


def test_dataset_manifest_schema_is_valid() -> None:
    contracts = Path(__file__).parents[1] / "src" / "sketchsense" / "contracts"
    schema = json.loads((contracts / "dataset-manifest.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
