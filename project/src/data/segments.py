"""Stable time keys; no operation may infer continuity from concatenation."""
import numpy as np
import pandas as pd
from src.common.provenance import digest

KEYS = ['sample_id', 'source_file', 'segment_id', 'row_in_source', 'timestamp']


def identify(df, dataset_id, source_file, interval=1.0):
    df = df.copy()
    parsed = pd.to_datetime(df['timestamp'], errors='coerce')
    bad = np.flatnonzero(parsed.isna())
    if len(bad):
        raise ValueError(f'{source_file}: invalid timestamp at CSV row {int(bad[0]) + 2}')
    delta = parsed.diff().dt.total_seconds()
    breaks = delta.ne(interval)
    if 'restart' in df:
        breaks |= df['restart'].fillna(False).astype(bool)
    seg = breaks.cumsum().to_numpy() - 1
    df['timestamp'] = parsed
    df['source_file'] = source_file
    df['segment_id'] = seg
    df['row_in_source'] = np.arange(len(df))
    prefix = digest([dataset_id, source_file])[:16]
    df['sample_id'] = [f'{prefix}:{int(s)}:{r}' for r, s in enumerate(seg)]
    return df


def split_segments(df):
    return [g.reset_index(drop=True) for _, g in df.groupby(['source_file', 'segment_id'], sort=False)]


def transition_pairs(frames, horizon=1):
    """Construct adjacent pairs BEFORE any budget sampling."""
    valid = [f for f in frames if len(f) > horizon]
    if not valid:
        raise ValueError('No contiguous transitions available')
    return np.concatenate([f[:-horizon] for f in valid]), np.concatenate([f[horizon:] for f in valid])
