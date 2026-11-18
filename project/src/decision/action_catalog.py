"""Load and validate the advisory action catalog.

Actions are advisory suggestions for a human operator — never commands
written to a PLC. Each action records bounds, applicability and cost so
the shadow policy can enumerate only legal suggestions.
"""
from __future__ import annotations

import yaml
from pathlib import Path

from src.common.config import project_path


def load_catalog(path: str | None = None) -> dict:
    p = Path(path) if path else project_path("config", "action_catalog.yaml")
    with open(p, "r", encoding="utf-8") as f:
        cat = yaml.safe_load(f)
    _validate(cat)
    return cat


def _validate(cat: dict) -> None:
    ids = set()
    for a in cat.get("actions", []):
        for req in ("id", "kind"):
            if req not in a:
                raise ValueError(f"action missing '{req}': {a}")
        if a["id"] in ids:
            raise ValueError(f"duplicate action id: {a['id']}")
        ids.add(a["id"])
        if a["kind"] == "adjust":
            if not a.get("affects"):
                raise ValueError(f"adjust action {a['id']} needs 'affects'")
            if a.get("max_delta", 0) <= 0:
                raise ValueError(f"adjust action {a['id']} needs max_delta > 0")


def action_applies(action: dict, candidate_tags: list[str]) -> bool:
    applies = action.get("applies_to") or ["*"]
    if "*" in applies:
        return True
    return any(t in applies for t in candidate_tags)


def enumerate_legal_actions(cat: dict, candidate_tags: list[str],
                            tag_values: dict | None = None) -> list[dict]:
    """Return catalog entries applicable to the root-cause candidates."""
    out = []
    for a in cat["actions"]:
        if not action_applies(a, candidate_tags):
            continue
        ok = a.get("review_status") == "approved"
        for tag, rng in (a.get("requires") or {}).items():
            if tag_values is None or tag not in tag_values:
                ok = False
                continue
            if tag_values is not None and tag in tag_values:
                v = tag_values[tag]
                if v < rng.get("min", -1e18) or v > rng.get("max", 1e18):
                    ok = False
        if ok:
            out.append(a)
    return out
