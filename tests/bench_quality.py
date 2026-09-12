"""Quality comparison: FP32 / FP16 / INT8 encoders and HiGS vs standard.

Converts one still frame through the production image path for each variant
and compares every result against an FP32 + standard-rasterization reference:
mean absolute pixel diff (0-255) and PSNR.

    sharp3d-env\\Scripts\\python.exe tests\\bench_quality.py
"""
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

OUT = ROOT.parent / "outputs"
IMG_IN = OUT / "_bench_img.png"

VARIANTS = [
    # (label, renderer, fp16, int8)
    ("fp32_standard", "standard", False, False),
    ("fp16_standard", "standard", True, False),
    ("fp16_higs",     "higs",     True, False),
    ("fp16_int8_std", "standard", True, True),
    ("fp16_int8_higs", "higs",    True, True),
]

from sharp3d.gui.worker import _PipelineWorker  # noqa: E402

cancel = mp.Event()
worker = _PipelineWorker(lambda name, a: None, cancel)

for label, renderer, fp16, int8 in VARIANTS:
    out = OUT / f"_q_{label}.png"
    if out.exists():
        out.unlink()
    worker.convert({
        "input": str(IMG_IN), "output": str(out),
        "ipd_mm": 63.0, "strength": 1.0, "convergence": 0.0,
        "decompose": "analytical", "format": "full_sbs",
        "renderer": renderer, "out_scale": 1.0,
        "depth": False, "ply": False,
        "perf_mode": "quality", "fp16": fp16, "int8": int8,
    })
    print(f"  rendered {label} -> exists={out.exists()}", flush=True)

ref = np.asarray(Image.open(OUT / "_q_fp32_standard.png"), dtype=np.float32)
results = []
for label, *_ in VARIANTS:
    p = OUT / f"_q_{label}.png"
    if not p.exists():
        continue
    img = np.asarray(Image.open(p), dtype=np.float32)
    mad = float(np.abs(img - ref).mean())
    mse = float(((img - ref) ** 2).mean())
    psnr = 10 * math.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")
    results.append({"label": label, "mad": round(mad, 3), "psnr": round(psnr, 2)})

print("\n=== 质量对比（参考: FP32 + 标准光栅化） ===")
print(f"{'变体':<18}{'平均像素差(0-255)':>18}{'PSNR(dB)':>12}")
for r in results:
    print(f"{r['label']:<18}{r['mad']:>18.3f}{r['psnr']:>12.2f}")
(OUT / "_bench_quality.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
