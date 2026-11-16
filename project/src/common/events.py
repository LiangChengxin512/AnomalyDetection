"""Helpers for contiguous-event bookkeeping shared across the pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd


def binary_to_ranges(flags: np.ndarray, min_gap: int = 0) -> list[tuple[int, int]]:
    """Return inclusive [start, end] index ranges of 1-runs in ``flags``.

    Runs separated by fewer than ``min_gap`` zeros are merged.
    """
    flags = np.asarray(flags).astype(bool)
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > max(1, min_gap))[0]
    groups = np.split(idx, splits + 1)
    return [(int(g[0]), int(g[-1])) for g in groups]


def ranges_to_flags(ranges: list[tuple[int, int]], length: int) -> np.ndarray:
    flags = np.zeros(length, dtype=np.int64)
    for s, e in ranges:
        flags[max(0, s): min(length, e + 1)] = 1
    return flags


def event_id_for(file_name: str, ordinal: int) -> str:
    stem = file_name.replace(".csv", "").replace("_", "-")
    return f"{stem}-evt-{ordinal:03d}"
