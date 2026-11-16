"""Sliding-window indexing without materializing N x W x D.

A window is identified by ``(segment, end_row)`` and covers rows
``[end-W+1, end]`` of one prepared file. Rows ``end < W-1`` are left
padded by repeating row 0 (``pad_mode="replicate"``, same convention as
upstream TranAD); ``pad_mode="skip"`` drops them instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindowIndex:
    """Maps a flat window id -> (frame, end_row)."""
    frames: list          # list of DataFrames (one per file, feature cols only)
    ends: np.ndarray      # [n_windows] end row inside each frame
    seg: np.ndarray       # [n_windows] frame index
    W: int
    pad_mode: str = "replicate"

    def __len__(self):
        return len(self.ends)

    def gather(self, idx: np.ndarray) -> np.ndarray:
        """Return windows for flat ids ``idx`` as [B, W, D] float32."""
        W = self.W
        out = np.empty((len(idx), W, self.frames[0].shape[1]), dtype=np.float32)
        for k, flat in enumerate(idx):
            seg, end = self.seg[flat], self.ends[flat]
            frame = self.frames[seg]
            lo = end - W + 1
            if lo >= 0:
                out[k] = frame[lo:end + 1]
            else:  # replicate row 0 at segment start
                body = frame[0:end + 1]
                pad = np.repeat(frame[0:1], -lo, axis=0)
                out[k] = np.concatenate([pad, body], axis=0)
        return out


def build_index(frames: list[np.ndarray], W: int, stride: int = 1,
                pad_mode: str = "replicate", max_windows_per_segment: int = 0) -> WindowIndex:
    """Create window ids for a list of per-file feature arrays."""
    ends, seg = [], []
    for si, frame in enumerate(frames):
        n = len(frame)
        start = 0 if pad_mode == "replicate" else W - 1
        e = np.arange(start, n, stride, dtype=np.int64)
        if max_windows_per_segment:
            e = e[:max_windows_per_segment]
        ends.append(e)
        seg.append(np.full(len(e), si, dtype=np.int64))
    return WindowIndex(frames=[np.asarray(f) for f in frames],
                       ends=np.concatenate(ends) if ends else np.array([], dtype=np.int64),
                       seg=np.concatenate(seg) if seg else np.array([], dtype=np.int64),
                       W=W, pad_mode=pad_mode)


def frames_from_parquets(paths: list[str], tags: list[str]) -> list[np.ndarray]:
    frames = []
    for p in paths:
        df = pd.read_parquet(p, columns=tags)
        frames.append(df[tags].to_numpy(dtype=np.float32))
    return frames
