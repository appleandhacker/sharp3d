"""GPU smoke test: render both backends on synthetic gaussians (no model).

    sharp3d-env\\Scripts\\python.exe tests\\smoke_gpu.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from sharp.utils.gaussians import Gaussians3D
from sharp3d.render import render_sbs
from sharp3d.render_vr import render_vr_stereo

dev = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)} "
      f"(sm{torch.cuda.get_device_capability(dev)[0]}{torch.cuda.get_device_capability(dev)[1]})")

torch.manual_seed(0)
N = 200_000
g = Gaussians3D(
    mean_vectors=torch.randn(1, N, 3, device=dev) * torch.tensor([1.0, 1.0, 2.0], device=dev),
    singular_values=torch.rand(1, N, 3, device=dev) * 0.02 + 0.002,
    quaternions=torch.nn.functional.normalize(torch.randn(1, N, 4, device=dev), dim=-1),
    colors=torch.rand(1, N, 3, device=dev),
    opacities=torch.rand(1, N, device=dev) * 0.5 + 0.5,
)

fails = []

# ── 1. render_sbs: standard vs higs ─────────────────────────────────────
for renderer in ("standard", "higs"):
    torch.cuda.synchronize(); t0 = time.time()
    sbs, (sw, sh) = render_sbs(g, f_px=1000.0, orig_w=960, orig_h=540,
                               ipd=0.063, renderer=renderer)
    torch.cuda.synchronize(); dt = time.time() - t0
    ok = (sbs.dtype == torch.uint8 and sbs.shape[0] == sh
          and sbs.shape[1] == 2 * sw and sbs.shape[2] == 3
          and torch.isfinite(sbs.float()).all())
    nz = (sbs > 0).float().mean().item()
    print(f"  render_sbs[{renderer:8s}] {tuple(sbs.shape)}  {dt*1000:6.1f}ms  "
          f"非零像素占比 {nz:.2%}  -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        fails.append(f"render_sbs[{renderer}]")
    if renderer == "standard":
        ref = sbs

# HiGS 与 standard 同场景对比（容差放宽：不同光栅化器实现）
torch.cuda.synchronize(); t0 = time.time()
sbs_h, _ = render_sbs(g, 1000.0, 960, 540, ipd=0.063, renderer="higs")
torch.cuda.synchronize(); dt = time.time() - t0
diff = (sbs_h.float() - ref.float()).abs().mean().item()
print(f"  HiGS vs standard 平均像素差: {diff:.2f}/255  (higs {dt*1000:.1f}ms)")
if diff > 12:
    fails.append(f"higs/standard 视觉偏差过大: {diff:.2f}")
else:
    print("  PASS  HiGS 输出与标准光栅化一致（容差内）")

# ── 2. NDC fold 路径 + higs（SBS 快路径组合）────────────────────────────
from sharp.utils.gaussians import get_unprojection_matrix
U = get_unprojection_matrix(torch.eye(4, device=dev),
                            torch.tensor([[1000., 0, 480, 0],
                                          [0, 1000., 270, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], device=dev),
                            (960, 540))
try:
    sbs_ndc, _ = render_sbs(g, 1000.0, 960, 540, ipd=0.063,
                            ndc_transform=U, renderer="higs")
    print(f"  NDC fold + higs: {tuple(sbs_ndc.shape)} -> PASS")
except Exception as e:
    print(f"  FAIL  NDC fold + higs: {type(e).__name__}: {e}")
    fails.append("ndc_fold+higs")

# ── 3. VR 立体渲染（standard 后端；HiGS 若扩展可用则走 HiGS）──────────────
torch.cuda.synchronize(); t0 = time.time()
vr = render_vr_stereo(g, ipd=0.063, face_size=256, out_w=512, out_h=512,
                      output_projection="equirect180", stereo_layout="sbs",
                      renderer="higs", device=dev)
torch.cuda.synchronize(); dt = time.time() - t0
ok = vr.dtype == torch.uint8 and vr.shape == (512, 1024, 3)
print(f"  render_vr_stereo: {tuple(vr.shape)}  {dt*1000:.1f}ms  -> "
      f"{'PASS' if ok else 'FAIL'}")
if not ok:
    fails.append("render_vr_stereo")

# 360° 输出
vr360 = render_vr_stereo(g, ipd=0.063, face_size=256, out_w=512, out_h=256,
                         output_projection="equirect360", stereo_layout="tb",
                         renderer="standard", device=dev)
print(f"  render_vr_stereo[360/tb]: {tuple(vr360.shape)} -> "
      f"{'PASS' if tuple(vr360.shape) == (512, 512, 3) else 'FAIL'}")
if tuple(vr360.shape) != (512, 512, 3):
    fails.append("vr360 shape")

# ── 4. colorize_depth 真实光栅化数据 ────────────────────────────────────
from sharp3d.render import render_depth_map
d = render_depth_map(g, 1000.0, 960, 540)
nz = (d > 0).float().mean().item()
print(f"  render_depth_map: {tuple(d.shape)}  非零占比 {nz:.2%} -> "
      f"{'PASS' if 0.05 < nz < 1.0 else 'FAIL'}")
if not (0.05 < nz < 1.0):
    fails.append(f"depth map 非零占比异常: {nz:.2%}")

print()
print("=" * 60)
print("GPU 冒烟测试: " + ("全部通过" if not fails else f"失败 {fails}"))
print("=" * 60)
sys.exit(1 if fails else 0)
