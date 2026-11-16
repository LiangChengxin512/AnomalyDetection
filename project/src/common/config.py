"""Experiment config loading and path resolution.

All modules resolve files relative to the project root (the directory that
contains ``config/experiment.yaml``), so the pipeline can be run from
anywhere via ``python -m`` or the shell entry points.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load experiment.yaml. ``path`` defaults to config/experiment.yaml."""
    cfg_path = Path(path) if path else project_path("config", "experiment.yaml")
    if not cfg_path.is_absolute():
        cfg_path = project_path(str(cfg_path))
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve(cfg: dict[str, Any], key: str, *extra: str) -> Path:
    """Resolve a path stored under cfg['paths'][key] relative to project root."""
    base = Path(cfg["paths"][key])
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    return base.joinpath(*extra) if extra else base


def load_yaml(path: str | os.PathLike) -> Any:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
