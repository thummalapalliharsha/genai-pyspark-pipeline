"""Pytest configuration helpers.

This module ensures the repository root is on `sys.path` so tests can import
the `src` package when pytest changes sys.path during collection.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _add_project_root_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_add_project_root_to_path()
