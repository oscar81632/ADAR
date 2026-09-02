"""Repo-local loader for the modified CLIP package used by ADAR/VSTD.

This module prepends the vendored CLIP package and the minimal vendored Vim
model definitions to `sys.path` before importing `clip`.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPO_ROOT / "external"
CLIP_VENDOR_ROOT = EXTERNAL_ROOT / "clip_vstd"
VIM_VENDOR_ROOT = EXTERNAL_ROOT / "vim_vstd"


def _prepend(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def import_clip():
    """Import the vendored modified CLIP package."""

    if not CLIP_VENDOR_ROOT.exists():
        raise RuntimeError(f"Missing vendored CLIP package: {CLIP_VENDOR_ROOT}")

    _prepend(VIM_VENDOR_ROOT)
    _prepend(CLIP_VENDOR_ROOT)
    return importlib.import_module("clip")
