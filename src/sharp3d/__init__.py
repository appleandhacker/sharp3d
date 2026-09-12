"""sharp3d - Convert 2D images/videos to stereoscopic 3D using SHARP 3DGS."""

from __future__ import annotations

import os
import sys
from pathlib import Path

__version__ = "2.0.0-beta"


def resolve_cache_dir() -> Path:
    """Single source of truth for the model/engine cache directory.

    Frozen (PyInstaller) builds MUST NOT use a path derived from ``__file__``:
    that resolves into the install directory (``<exe dir>/.cache``) or, for
    ``ort_engine``, even into its *parent* (``C:\\Program Files\\.cache`` when
    installed there). Both are typically not writable, and every failed write
    is silently swallowed by the surrounding try/except — so TensorRT engines
    and ONNX exports get rebuilt from scratch on every launch, which looks
    exactly like "stuck at loading weights".

    Frozen builds therefore use a per-user, always-writable location under
    %LOCALAPPDATA%. Source checkouts keep using <project root>/.cache.
    """
    if getattr(sys, "frozen", False):
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return Path(base) / "sharp3d" / ".cache"
    # src/sharp3d/__init__.py -> parents[0]=src/sharp3d, [1]=src, [2]=root
    return Path(__file__).resolve().parents[2] / ".cache"
