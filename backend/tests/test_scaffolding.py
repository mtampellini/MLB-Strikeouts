"""Phase 1 scaffold gates.

Two checks:
1. All package modules import cleanly.
2. The projection layer has zero imports from the picks layer. The decoupling
   principle is load-bearing for the whole repo (see backend/README.md), and
   the only way to keep it honest as the code grows is a structural test.
"""
from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

EXPECTED_MODULES = [
    "src",
    "src.data",
    "src.features",
    "src.projection",
    "src.picks",
    "src.pipeline",
    "src.results",
    "src.backtest",
]


@pytest.mark.parametrize("module_name", EXPECTED_MODULES)
def test_module_importable(module_name: str) -> None:
    importlib.import_module(module_name)


def test_projection_does_not_import_picks() -> None:
    projection_dir = (
        pathlib.Path(__file__).resolve().parents[1] / "src" / "projection"
    )
    offending: list[tuple[pathlib.Path, str]] = []
    for py in projection_dir.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "src.picks" or alias.name.startswith(
                        "src.picks."
                    ):
                        offending.append((py, alias.name))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "src.picks" or mod.startswith("src.picks."):
                    offending.append((py, mod))
    assert not offending, (
        f"projection layer imports picks layer (violates decoupling): {offending}"
    )
