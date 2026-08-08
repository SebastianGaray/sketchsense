"""Deterministic serialization helpers for bounded NumPy artifacts."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def save_npz_deterministic(path: Path, arrays: Mapping[str, NDArray[np.generic]]) -> None:
    """Write NPZ entries with stable ordering, metadata, and compression."""
    with zipfile.ZipFile(path, mode="w") as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, buffer.getvalue(), compresslevel=9)
