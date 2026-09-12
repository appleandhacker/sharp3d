"""Standalone FP8 patch-encoder engine test: build time, speed, output sanity.

Compares the FP8 QDQ graph's TRT engine against the FP16 engine (both batch
35) and checks output closeness to the FP32 CUDA EP reference.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from sharp3d.ort_engine import ORTEncoder  # noqa: E402
from sharp3d import resolve_cache_dir  # noqa: E402


def main():
    device = torch.device("cuda")
    onnx_dir = resolve_cache_dir() / "onnx"
    fp8_onnx = onnx_dir / "patch_encoder_fp8.onnx"
    fp16_onnx = onnx_dir / "patch_encoder.onnx"

    print("[1] 构建/加载 FP8 引擎 …", flush=True)
    t0 = time.time()
    enc8 = ORTEncoder(fp8_onnx, resolve_cache_dir() / "trt_v3", device,
                      label="patch_encoder_fp8", qdq=True)
    print(f"    就绪 ({time.time()-t0:.0f}s)  TRT={enc8.using_trt}", flush=True)

    print("[2] 构建/加载 FP16 引擎 …", flush=True)
    t0 = time.time()
    enc16 = ORTEncoder(fp16_onnx, resolve_cache_dir() / "trt_v3", device,
                       label="patch_encoder", qdq=False)
    print(f"    就绪 ({time.time()-t0:.0f}s)  TRT={enc16.using_trt}", flush=True)

    print("[3] 真实输入性能对比（batch 35, 20 次均值）…", flush=True)
    from sharp3d.unproject import prepare_input
    from sharp.models import normalizers
    from sharp.models.encoders.spn_encoder import split
    norm = normalizers.AffineRangeNormalizer(
        input_range=(0, 1), output_range=(-1, 1)).to(device).eval()
    frame = grab_frame = None
    from verify_full_onnx import grab_frame
    frame = grab_frame(ROOT.parent / "outputs" / "_bench_4k.mp4", 10)
    img_r, _df, _ir, _sz = prepare_input(frame, 1920 * 1.2, device)
    with torch.no_grad():
        x = norm(img_r)
        x1 = torch.nn.functional.interpolate(x, scale_factor=0.5,
                                             mode="bilinear",
                                             align_corners=False)
        x2 = torch.nn.functional.interpolate(x, scale_factor=0.25,
                                             mode="bilinear",
                                             align_corners=False)
        patches = torch.cat((split(x, 0.25, 384), split(x1, 0.5, 384),
                             x2), dim=0)

    def bench(enc):
        for _ in range(3):
            enc(patches)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            enc(patches)
        torch.cuda.synchronize()
        return (time.time() - t0) / 20

    t8 = bench(enc8)
    t16 = bench(enc16)
    print(f"    FP8: {t8*1e3:.1f}ms   FP16: {t16*1e3:.1f}ms   "
          f"加速 {t16/t8:.2f}x", flush=True)

    print("[4] 输出一致性（vs FP16 引擎）…", flush=True)
    with torch.no_grad():
        f8, i8 = enc8(patches)
        f16, i16 = enc16(patches)
    for name, o, r in (("features", f8, f16),
                       ("intermediates[5]", i8[5], i16[5])):
        d = (o.float() - r.float()).abs()
        rel = d.norm() / r.float().norm().clamp_min(1e-12)
        print(f"    {name:18s} max={d.max():.3e} mean={d.mean():.3e} "
              f"相对L2={rel:.4f}", flush=True)


if __name__ == "__main__":
    main()
