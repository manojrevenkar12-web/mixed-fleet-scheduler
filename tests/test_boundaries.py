"""The invariants in AGENTS.md that code can check: the platform never sees the world's truth, and the world
never reads the platform's settings."""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "mfs"


def _imports(folder: str) -> dict[str, set[str]]:
    out = {}
    for p in (SRC / folder).rglob("*.py"):
        mods: set[str] = set()
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.add(("." * node.level) + node.module)
            elif isinstance(node, ast.Import):
                mods.update(a.name for a in node.names)
        out[p.name] = mods
    return out


def test_platform_never_imports_the_world_or_the_lab() -> None:
    for name, mods in _imports("platform").items():
        bad = [m for m in mods if any(x in m for x in ("world", "sim", "lab"))]
        assert not bad, f"platform/{name} imports {bad}"


def test_service_never_imports_the_world_or_the_lab() -> None:
    tree = ast.parse((SRC / "service.py").read_text(encoding="utf-8"))
    mods = {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not [m for m in mods if any(x in m for x in ("world", "sim", "lab"))]


def test_platform_does_not_read_the_world_file() -> None:
    for p in (SRC / "platform").rglob("*.py"):
        assert "world.yml" not in p.read_text(encoding="utf-8"), p.name


def test_world_does_not_read_platform_settings() -> None:
    for p in (SRC / "world").rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        for call in ("config.platform(", "config.designs(", "config.gates(", "platform.yml"):
            assert call not in text, f"{p.name} reads {call}"
