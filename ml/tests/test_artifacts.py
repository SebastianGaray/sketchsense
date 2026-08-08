import hashlib
from pathlib import Path

import numpy as np

from sketchsense.contracts.artifacts import save_npz_deterministic


def test_npz_serialization_is_byte_stable(tmp_path: Path) -> None:
    arrays = {"values": np.arange(12, dtype=np.float32).reshape(3, 4)}
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    save_npz_deterministic(first, arrays)
    save_npz_deterministic(second, arrays)
    assert (
        hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    )
