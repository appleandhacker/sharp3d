"""Render one real frame through the FULL-TRT path and compare vs reference.

The old-path reference image (outputs/_q_ref_fp32.png) comes from
bench_quality.py run with SHARP3D_NO_FULL_TRT=1. This script renders the same
frame via SharpPredictor's full-model TRT engine and reports PSNR.

Usage:
    python tests/run_full_quality.py fp16    # int8=False
    python tests/run_full_quality.py int8    # int8=True (needs QDQ model)
"""
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

mode = sys.argv[1] if len(sys.argv) > 1 else "fp16"

OUT = ROOT.parent / "outputs"
IMG_IN = OUT / "_bench_img.png"

from sharp3d.gui.worker import _PipelineWorker  # noqa: E402

cancel = mp.Event()
worker = _PipelineWorker(lambda n, a: None, cancel)

out = OUT / f"_q_full_{mode}.png"
if out.exists():
    out.unlink()
worker.convert({
    "input": str(IMG_IN), "output": str(out),
    "ipd_mm": 63.0, "strength": 1.0, "convergence": 0.0,
    "decompose": "analytical", "format": "full_sbs",
    "renderer": "higs", "out_scale": 1.0,
    "depth": False, "ply": False,
    "perf_mode": "quality",
    "int8": (mode == "int8"),
})
assert out.exists(), "整模型渲染失败"

ref = np.asarray(Image.open(OUT / "_q_ref_fp32.png"), dtype=np.float32)
img = np.asarray(Image.open(out), dtype=np.float32)
mad = float(np.abs(img - ref).mean())
mse = float(((img - ref) ** 2).mean())
psnr = 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")
print(f"[full_{mode}] vs fp32参考: 平均像素差={mad:.3f}/255  PSNR={psnr:.2f}dB")
