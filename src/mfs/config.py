"""Loads the YAML files in config/. Each module reads only the files it owns (see AGENTS.md)."""

from __future__ import annotations

import copy
import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("MFS_ROOT") or Path(__file__).resolve().parents[2])
CONFIG = ROOT / "config"
REPORTS = ROOT / "reports"


@cache
def _load(name: str) -> dict[str, Any]:
    data = yaml.safe_load((CONFIG / f"{name}.yml").read_text(encoding="utf-8"))
    assert isinstance(data, dict), name
    return data


def merged(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """`over` replaces `base` key by key; nested dicts merge, lists and scalars replace."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merged(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def platform() -> dict[str, Any]:
    return _load("platform")


def designs() -> dict[str, Any]:
    return _load("designs")


def seeds() -> dict[str, Any]:
    return _load("seeds")


def gates() -> dict[str, Any]:
    return _load("gates")
