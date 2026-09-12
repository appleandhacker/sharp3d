"""FP8 / low-precision feasibility probe for the sharp3d pipeline.

Answers three questions with measurements on this machine:
  1. Does torch expose FP8 GEMM here, and is it actually faster than FP16
     for the DINOv2 ViT-L matmul shapes that dominate encoder time?
  2. Does the ORT TensorRT EP expose an FP8 switch (i.e. is FP8 reachable
     without building TRT engines through the native API)?
  3. Does gsplat accept FP16 inputs (feasibility of an FP16 render path)?

    sharp3d-env\\Scripts\\python.exe tests\\bench_fp8.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

dev = torch.device("cuda")
major, minor = torch.cuda.get_device_capability(dev)
print(f"GPU: {torch.cuda.get_device_name(0)} (sm{major}{minor})")
print(f"torch: {torch.__version__}")

print("\n=== 1. torch FP8 支持 ===")
has_fp8 = hasattr(torch, "float8_e4m3fn")
print(f"  torch.float8_e4m3fn 存在: {has_fp8}")
print(f"  torch._scaled_mm 存在: {hasattr(torch, '_scaled_mm')}")
print(f"  SM{major}{minor} 硬件 FP8 支持: {'是 (Ada+)' if major >= 9 else '否'}")

if has_fp8:
    print("\n=== 2. FP8 vs FP16 GEMM（DINOv2 ViT-L 主导形状） ===")
    # patch encoder: 35 patches × 577 tokens, model dim 1024
    # image encoder: 1 × 577 tokens
    for rows, kdim, ndim, label in [
        (35 * 577, 1024, 3072, "qkv (35 patches)"),
        (35 * 577, 1024, 4096, "mlp.fc1 (35 patches)"),
        (35 * 577, 4096, 1024, "mlp.fc2 (35 patches)"),
        (1 * 577, 1024, 3072, "qkv (image encoder)"),
    ]:
        x16 = torch.randn(rows, kdim, device=dev, dtype=torch.float16)
        w16 = torch.randn(ndim, kdim, device=dev, dtype=torch.float16) / kdim

        for _ in range(3):
            torch.mm(x16, w16.t())
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            torch.mm(x16, w16.t())
        torch.cuda.synchronize()
        t16 = (time.time() - t0) / 20

        flops = 2 * rows * kdim * ndim
        try:
            x8 = x16.to(torch.float8_e4m3fn)
            w8 = w16.to(torch.float8_e4m3fn)
            sx = torch.ones((), device=dev)
            sw = torch.ones((), device=dev)
            torch._scaled_mm(x8, w8.t(), scale_a=sx, scale_b=sw,
                             out_dtype=torch.float16)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(20):
                torch._scaled_mm(x8, w8.t(), scale_a=sx, scale_b=sw,
                                 out_dtype=torch.float16)
            torch.cuda.synchronize()
            t8 = (time.time() - t0) / 20
            print(f"  {label:24s} fp16 {t16*1e3:6.2f}ms ({flops/t16/1e12:6.1f} TFLOPS)"
                  f"  |  fp8 {t8*1e3:6.2f}ms ({flops/t8/1e12:6.1f} TFLOPS)"
                  f"  |  加速 {t16/t8:4.2f}x")
        except Exception as e:
            print(f"  {label:24s} fp16 {t16*1e3:6.2f}ms  |  fp8 不可用: "
                  f"{type(e).__name__}: {str(e)[:90]}")

print("\n=== 3. ORT TensorRT EP 的 FP8 可达性 ===")
try:
    import onnxruntime as ort
    print(f"  onnxruntime: {ort.__version__}")
    prov = ort.get_build_info()
    import re
    m = re.search(r"TensorRT\s*v?([\d.]+)", str(prov))
    print(f"  TRT 版本: {m.group(1) if m else '未知'}")
    # 官方 ORT TRT EP 只暴露 trt_fp16_enable / trt_int8_enable；
    # FP8 需要原生 TRT API（IInt8 之外的 fp8 量化 + 引擎构建）。
    print("  TRT EP FP8 开关: 无（ORT 仅暴露 trt_fp16_enable / trt_int8_enable）")
    print("  结论: FP8 需经原生 TensorRT API 构建引擎并自行挂接到 sharp3d，")
    print("        无法通过现有 ORT EP 配置启用。")
except ImportError:
    print("  onnxruntime 不可用")

print("\n=== 4. gsplat 渲染 FP16 输入可行性 ===")
from sharp.utils.gaussians import Gaussians3D  # noqa: E402

N = 100_000
g16 = Gaussians3D(
    mean_vectors=torch.randn(1, N, 3, device=dev, dtype=torch.float16),
    singular_values=torch.rand(1, N, 3, device=dev, dtype=torch.float16) * 0.02,
    quaternions=torch.nn.functional.normalize(
        torch.randn(1, N, 4, device=dev, dtype=torch.float16), dim=-1),
    colors=torch.rand(1, N, 3, device=dev, dtype=torch.float16),
    opacities=torch.rand(1, N, device=dev, dtype=torch.float16) * 0.5 + 0.5,
)
from sharp3d.render import render_sbs  # noqa: E402

try:
    sbs, _ = render_sbs(g16, f_px=1000.0, orig_w=640, orig_h=360, ipd=0.063)
    print(f"  gsplat 接受 FP16 输入: 是 -> {tuple(sbs.shape)}")
except Exception as e:
    print(f"  gsplat 接受 FP16 输入: 否 ({type(e).__name__}: {str(e)[:100]})")
    print("  -> 渲染端 FP16 需改 gsplat 内核或保持 fp32 输入、仅传输端 FP16")

print("\n=== 5. 渲染输入数据搬运量（现状 vs FP16 传输） ===")
for n, name in [(1_179_648, "SHARP 单帧高斯")]:
    fp32 = n * (3 + 3 + 4 + 3 + 1) * 4
    fp16 = n * (3 + 3 + 4 + 3 + 1) * 2
    print(f"  {name}: 属性张量 FP32={fp32/2**20:.0f}MB  FP16={fp16/2**20:.0f}MB")
