"""Image save helper: OpenCV PNG encode with Pillow fallback.

Measured (2026-09-10, 7680×2160 SBS PNG): Pillow 1623ms vs cv2 459ms — PNG
encoding dominates still-image conversion latency. JPEG gains are marginal
(1.4x) and video frames never pass through here (ffmpeg owns that path), so
only PNG output is routed through cv2.

Reading stays on Pillow everywhere: it handles EXIF orientation for phone
photos, which cv2 does not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_rgb(path: Path | str, arr: np.ndarray) -> None:
    """Save an H×W×3 RGB uint8 array, using cv2 for PNG when available."""
    path = Path(path)
    if path.suffix.lower() == ".png":
        try:
            import cv2
            ok, buf = cv2.imencode(".png",
                                   np.ascontiguousarray(arr[:, :, ::-1]))
            if ok:
                buf.tofile(str(path))
                return
        except ImportError:
            pass
    from PIL import Image
    Image.fromarray(arr).save(str(path))
