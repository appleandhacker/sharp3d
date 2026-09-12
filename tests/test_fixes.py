"""Regression tests for the 2026-09-09 bug-fix pass.

Run from the project root:
    sharp3d-env\\Scripts\\python.exe tests\\test_fixes.py

Pure-CPU tests (no model download, no gsplat CUDA kernels).
"""
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  <- {detail}" if detail and not cond else ""))


print("=" * 70)
print("1. formats.output_size 与 pack() 严格一致（half_sbs/half_tb 修复）")
print("=" * 70)
from sharp3d.formats import output_size, pack, FORMAT_KEYS

bad = []
test_ws = list(range(2, 130)) + [384, 540, 682, 1000, 1366, 1920, 3840, 7680]
test_hs = list(range(2, 130)) + [216, 270, 384, 540, 1080, 2160, 4320]
for fmt in FORMAT_KEYS:
    for w in test_ws:
        for h in (8, 60, 216, 1080):
            src = torch.randint(0, 255, (h, 2 * w, 3), dtype=torch.uint8)
            out = pack(fmt, src)
            ow, oh = output_size(fmt, w, h)
            if tuple(out.shape[:2]) != (oh, ow):
                bad.append((fmt, w, h, tuple(out.shape[:2]), (oh, ow)))
check("output_size() == pack().shape（全部格式 × 全部尺寸）", not bad,
      f"first mismatch: {bad[:3]}")

# 具体触发场景：sw ≡ 2 (mod 4)
src = torch.randint(0, 255, (1080, 2 * 682, 3), dtype=torch.uint8)
out = pack("half_sbs", src)
ow, oh = output_size("half_sbs", 682, 1080)
check("half_sbs @ w=682（旧代码差 2px 的场景）", tuple(out.shape[:2]) == (oh, ow),
      f"pack={tuple(out.shape[:2])} declared={(oh, ow)}")

print()
print("=" * 70)
print("2. resolve_cache_dir（打包路径统一修复）")
print("=" * 70)
from sharp3d import resolve_cache_dir

src_dir = resolve_cache_dir()
check("源码模式 -> <项目根>/.cache",
      src_dir == ROOT / ".cache", f"got {src_dir}")

import sharp3d
real_frozen = getattr(sys, "frozen", False)
try:
    sys.frozen = True
    import importlib
    frozen_dir = sharp3d.resolve_cache_dir()
finally:
    if not real_frozen:
        del sys.frozen
check("frozen 模式 -> %LOCALAPPDATA%\\sharp3d\\.cache",
      str(frozen_dir).lower().endswith("sharp3d\\.cache")
      and "localappdata" in str(frozen_dir).lower().replace("%localappdata%", "localappdata")
      or "appdata\\local" in str(frozen_dir).lower(),
      f"got {frozen_dir}")

print()
print("=" * 70)
print("3. quaternion（Shepperd）vs scipy 参考")
print("=" * 70)
from sharp3d.quaternion import quat_from_rotmat_gpu

torch.manual_seed(0)
Rrand = torch.linalg.qr(torch.randn(200, 3, 3))[0]
dets = torch.linalg.det(Rrand)
Rrand[dets < 0, :, 2] *= -1  # force proper rotations

try:
    from scipy.spatial.transform import Rotation as _R
    q_ours = quat_from_rotmat_gpu(Rrand)          # (w, x, y, z)
    q_ref = _R.from_matrix(Rrand.numpy()).as_quat()  # (x, y, z, w)
    q_ref = torch.from_numpy(q_ref[:, [3, 0, 1, 2]])
    q_ref = torch.where(q_ref[:, :1] < 0, -q_ref, q_ref)
    err = (q_ours - q_ref).abs().max().item()
    check("与 scipy 一致 (max err < 1e-5)", err < 1e-5, f"err={err:.2e}")
except ImportError:
    print("  SKIP  scipy 不可用")

print()
print("=" * 70)
print("4. eigendecompose：解析法 vs SVD（重建 / det=+1 / 降序 / 简并）")
print("=" * 70)
from sharp3d.eigendecompose import decompose_covariance
from sharp.utils.gaussians import compose_covariance_matrices

torch.manual_seed(1)


def rebuild_R(q):
    w_, x_, y_, z_ = q.unbind(-1)
    R = torch.zeros(q.shape[0], 3, 3)
    R[:, 0, 0] = 1 - 2 * (y_ * y_ + z_ * z_)
    R[:, 0, 1] = 2 * (x_ * y_ - z_ * w_)
    R[:, 0, 2] = 2 * (x_ * z_ + y_ * w_)
    R[:, 1, 0] = 2 * (x_ * y_ + z_ * w_)
    R[:, 1, 1] = 1 - 2 * (x_ * x_ + z_ * z_)
    R[:, 1, 2] = 2 * (y_ * z_ - x_ * w_)
    R[:, 2, 0] = 2 * (x_ * z_ - y_ * w_)
    R[:, 2, 1] = 2 * (y_ * z_ + x_ * w_)
    R[:, 2, 2] = 1 - 2 * (x_ * x_ + y_ * y_)
    return R


def rel_recon(cov, q, sv):
    R = rebuild_R(q)
    recon = R @ torch.diag_embed(sv ** 2) @ R.transpose(-1, -2)
    return ((recon - cov).norm(dim=(-2, -1)) / cov.norm(dim=(-2, -1))).max().item()


for label, sv_in in [
    ("随机尺度", torch.rand(512, 3) * 0.05 + 0.001),
    # SHARP 扁平高斯的真实场景：两个特征值几乎相等
    ("近简并 λ1≈λ2", torch.stack([
        torch.full((512,), 0.04) + torch.rand(512) * 1e-4,
        torch.full((512,), 0.04) + torch.rand(512) * 1e-4,
        torch.full((512,), 0.005)], dim=-1)),
    # 完全各向同性
    ("各向同性 λ1=λ2=λ3", torch.full((512, 3), 0.02)),
]:
    q_in = torch.nn.functional.normalize(torch.randn(sv_in.shape[0], 4), dim=-1)
    cov = compose_covariance_matrices(q_in, sv_in)
    for method in ("analytical", "svd"):
        q_out, sv_out = decompose_covariance(cov, method=method)
        err = rel_recon(cov, q_out, sv_out)
        det_min = torch.linalg.det(rebuild_R(q_out)).min().item()
        desc = bool((sv_out[:, :-1] >= sv_out[:, 1:] - 1e-6).all())
        check(f"[{method}/{label}] 相对重建误差 < 5e-3", err < 5e-3, f"{err:.3e}")
        check(f"[{method}/{label}] det=+1", det_min > 0.99, f"{det_min:.4f}")
        check(f"[{method}/{label}] 奇异值降序", desc)

print()
print("=" * 70)
print("5. temporal：_gaussian_grid / _infer_shape / KalmanScalar")
print("=" * 70)
from sharp3d.conversion import _gaussian_grid
from sharp3d.temporal import TemporalStabilizer, KalmanScalar

check("_gaussian_grid(2*768*768) == (2, 768)", _gaussian_grid(2 * 768 * 768) == (2, 768))
check("_gaussian_grid(768*768) == (1, 768)", _gaussian_grid(768 * 768) == (1, 768))
check("_gaussian_grid(素数) 返回 None", _gaussian_grid(7919) is None)

st = TemporalStabilizer(mode="adaptive", device=torch.device("cpu"))
st._infer_shape(2 * 768 * 768)
check("_infer_shape(2*768²) -> (2, 768, 768)", st._shape == (2, 768, 768),
      f"got {st._shape}")

kf = KalmanScalar(q_pos=0.05, q_vel=0.02, r=0.15)
out = [kf.update(2.0 + 0.01 * math.sin(i)) for i in range(200)]
check("Kalman 收敛到信号附近 (err<0.05)", abs(out[-1] - 2.0) < 0.05, f"got {out[-1]:.4f}")
check("Kalman 输出有限且稳定", all(math.isfinite(v) for v in out))
kf.reset()
check("reset 后重新初始化", kf.update(5.0) == 5.0)

print()
print("=" * 70)
print("6. projection：cubemap 采样几何 / 球面覆盖 / 鱼眼单调性")
print("=" * 70)
from sharp3d.projection import (
    _build_equirect_plan, _cubemap_face_rays, _hemisphere_face_rays,
    equirect_to_cubemap, cubemap_to_equirect, _fisheye_theta_to_r,
    get_cubemap_cameras, OVERLAP_FOV_SCALE,
)

# 6a. 每个输出像素恰好被分配一次（否则拼图有洞/重叠）
plan = _build_equirect_plan(256, 128, half_sphere=False, device=torch.device("cpu"))
total = sum(idx.numel() for _, idx, _ in plan)
check("360° plan 覆盖全部像素（无洞/无重叠）", total == 256 * 128, f"{total} != {256*128}")
plan180 = _build_equirect_plan(256, 256, half_sphere=True, device=torch.device("cpu"))
total180 = sum(idx.numel() for _, idx, _ in plan180)
faces_used = {fi for fi, _, _ in plan180}
check("180° plan 覆盖全部像素", total180 == 256 * 256, f"{total180} != {256*256}")
check("180° plan 不使用 -Z 面（与 skip_back 一致）", 5 not in faces_used,
      f"faces={sorted(faces_used)}")

# 6b. 光线方向单位化 + FOV_SCALE 语义
rays = _cubemap_face_rays(64, torch.device("cpu"), fov_scale=OVERLAP_FOV_SCALE)
norms = rays.norm(dim=-1)
check("cubemap 光线已归一化", float((norms - 1).abs().max()) < 1e-5)
half_angle = math.degrees(math.atan(OVERLAP_FOV_SCALE))
check(f"FOV_SCALE={OVERLAP_FOV_SCALE} 对应 ~{half_angle:.1f}° 半角",
      abs(half_angle - 56.3) < 0.1)

hrays = _hemisphere_face_rays(64, torch.device("cpu"), fov_scale=1.0)
check("半球光线已归一化", float((hrays.norm(dim=-1) - 1).abs().max()) < 1e-5)

# 6c. 等距柱状 -> cubemap -> 等距柱状 往返
H, W = 128, 256
gy, gx = torch.meshgrid(
    torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
img = torch.stack([gx, torch.zeros_like(gx), 1 - gy], dim=-1)  # 合成渐变图
faces = equirect_to_cubemap(img, face_size=64, fov_scale=1.0)
check("equirect_to_cubemap 输出形状", tuple(faces.shape) == (6, 3, 64, 64),
      f"got {tuple(faces.shape)}")
back = cubemap_to_equirect(faces, W, H, half_sphere=False)
center = img[32:96, 64:192]      # 远离极区/左右边界的中心区
back_c = back[32:96, 64:192]
rt_err = float((back_c - center).abs().mean())
check("equirect→cubemap→equirect 中心区往返误差 < 0.02", rt_err < 0.02, f"{rt_err:.4f}")

# 6d. 鱼眼模型 r(θ) 单调递增
# orthographic r=sinθ 只在 θ≤π/2 内单调（物理 FOV 上限 180°），其余模型全区间单调
th_full = torch.linspace(1e-3, math.pi * 0.9, 100)
th_hemi = torch.linspace(1e-3, math.pi / 2 - 1e-3, 100)
for model in ("equidistant", "equisolid", "stereographic"):
    r = _fisheye_theta_to_r(th_full, model, None)
    check(f"fisheye[{model}] r(θ) 单调递增", bool((r[1:] >= r[:-1] - 1e-7).all()))
r = _fisheye_theta_to_r(th_hemi, "orthographic", None)
check("fisheye[orthographic] r(θ) 在有效区间单调", bool((r[1:] >= r[:-1] - 1e-7).all()))

# 6e. cubemap 相机：平移 = -R @ eye
vm, ks = get_cubemap_cameras(64, torch.device("cpu"), torch.tensor([-0.0315, 0.0, 0.0]))
eye = torch.tensor([-0.0315, 0.0, 0.0])
t_expect = -(vm[:, :3, :3] @ eye.unsqueeze(-1)).squeeze(-1)
check("cubemap viewmat 平移 = -R·eye", float((vm[:, :3, 3] - t_expect).abs().max()) < 1e-6)
check("cubemap 内参 90° FOV (f = size/2)", float((ks[0, 0, 0] - 32.0).abs()) < 1e-6)

print()
print("=" * 70)
print("7. colorize_depth（背景异常值修复）")
print("=" * 70)
from sharp3d.render import colorize_depth

# 背景像素 alpha=0，深度商爆炸到 1e8 —— 修复前会吞掉整条色带
depth = torch.tensor([[1.0, 2.0],
                      [3.0, 1e8]])
alpha = torch.tensor([[0.95, 0.9],
                      [0.85, 0.0]])
vis = colorize_depth(depth, alpha)
check("输出 uint8 且形状正确", vis.dtype == torch.uint8 and tuple(vis.shape) == (2, 2, 3))
# 修复前: d_max=1e8 → 所有有效像素 depth_norm≈0 → 全部 (255,0,0) 纯红
# 修复后: 背景被掩掉, d_max≈3 → 远处像素 (depth=3) 应落在色带冷端
b_far = vis[1, 0, 2].item()   # depth=3 的蓝通道
check("远处像素到达色带冷端 (b>100)——证明 1e8 背景不再吞掉色带",
      b_far > 100, f"b={b_far}")
check("背景 (alpha=0) 为黑色", vis[1, 1].tolist() == [0, 0, 0], f"got {vis[1,1].tolist()}")

vis2 = colorize_depth(torch.tensor([[1.0, 5.0]]), None)
check("无 alpha 输入也能着色", tuple(vis2.shape) == (1, 2, 3))
vis3 = colorize_depth(torch.zeros(4, 4), torch.zeros(4, 4))
check("全空输入 -> 全黑", bool((vis3 == 0).all()))

# 近→暖(红)，远→冷(蓝)
d = torch.tensor([[1.0, 10.0]])
a = torch.ones(1, 2)
v = colorize_depth(d, a)
check("近处偏红 / 远处偏蓝", v[0, 0, 0].item() > v[0, 1, 0].item()
      and v[0, 1, 2].item() > v[0, 0, 2].item(),
      f"near={v[0,0].tolist()} far={v[0,1].tolist()}")

print()
print("=" * 70)
print("8. render_sbs 签名（renderer 参数已接入）")
print("=" * 70)
import inspect
from sharp3d.render import render_sbs
sig = inspect.signature(render_sbs)
check("render_sbs 接受 renderer 参数", "renderer" in sig.parameters)
check("默认 standard（不改变现有行为）",
      sig.parameters["renderer"].default == "standard")
from sharp3d.conversion import VideoConversionEngine
sig2 = inspect.signature(VideoConversionEngine.__init__)
check("VideoConversionEngine 接受 renderer", "renderer" in sig2.parameters)

print()
print("=" * 70)
print("9. 全模块导入（含 GUI，无显示环境）")
print("=" * 70)
import importlib
mods = [
    "sharp3d", "sharp3d.options", "sharp3d.formats", "sharp3d.quaternion",
    "sharp3d.eigendecompose", "sharp3d.projection", "sharp3d.unproject",
    "sharp3d.render", "sharp3d.render_vr", "sharp3d.temporal",
    "sharp3d.conversion", "sharp3d.video", "sharp3d.hdr", "sharp3d.profiling",
    "sharp3d.pipeline", "sharp3d.predict", "sharp3d.ort_engine",
    "sharp3d.cli",
    "sharp3d.gui", "sharp3d.gui.widgets", "sharp3d.gui.theme",
    "sharp3d.gui.worker", "sharp3d.gui.main_window",
    "sharp3d.gui.sbs_tab", "sharp3d.gui.vr_tab", "sharp3d.gui.anim_tab",
    "sharp3d.gui.gaussian_tab",
]
import_errors = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        import_errors.append((m, f"{type(e).__name__}: {e}"))
check(f"导入 {len(mods)} 个模块", not import_errors,
      "; ".join(f"{m}: {e}" for m, e in import_errors[:5]))

# 残留引用检查
import re
leftover = []
for f, pat in [("src/sharp3d/pipeline.py", r"_ensure_warmup|_warmed_up"),
               ("src/sharp3d/gui/vr_tab.py", r"_stabilize"),
               ("src/sharp3d/gui/worker.py", r"parents\[3\] \*")]:
    txt = (ROOT / f).read_text(encoding="utf-8")
    if re.search(pat, txt):
        leftover.append(f)
check("死代码/失效引用已清除", not leftover, f"residual in {leftover}")

print()
print("=" * 70)
print(f"结果: {len(PASS)} 通过, {len(FAIL)} 失败")
print("=" * 70)
sys.exit(1 if FAIL else 0)
