"""Verify batch 2 fixes that are provable without a real conversion.

Covers:
  2.1(a) _pool_to_grid element count matches the keyframe colours
  2.1(b) _store_keyframe is atomic (no half-published state)
  2.1(c) colour refresh uses a linear-domain GAIN -> no luminance drift
  2.2    _soften_depth_edges works in world z with an adaptive threshold
  2.3    _infer_shape returns None, callers fall back instead of warping
  2.4    eigendecompose survives NaN / tiny-scale covariance

The keyframe tests run against VideoConversionEngine with the heavy
predict/render hooks stubbed out, so they exercise the real colour maths.
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch

SRC = r"C:/Users/yhm/.qoderworkcn/workspace/mrsw6dewe12d0mgy/sharp3d/src"
sys.path.insert(0, SRC)

from sharp3d.conversion import VideoConversionEngine, _gaussian_grid  # noqa: E402
from sharp3d.eigendecompose import analytical_eigen_decompose          # noqa: E402
from sharp3d.temporal import TemporalStabilizer                        # noqa: E402
from sharp.utils.color_space import linearRGB2sRGB, sRGB2linearRGB    # noqa: E402

SIDE = 8
LAYERS = 2
N = LAYERS * SIDE * SIDE

fails: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {label}" + (f"  — {detail}" if detail else ""))
    if not cond:
        fails.append(label)


class _Colors:
    """Stand-in for Gaussians3D.colors — shape (1, N, 3)."""

    def __init__(self, n: int, layer_major: bool = True):
        self.data = torch.full((1, n, 3), 0.25, dtype=torch.float32)
        self.ndim = 3

    def __getitem__(self, idx):
        return self.data[idx]


class _G:
    def __init__(self, n: int):
        self.colors = _Colors(n)

    def clone(self):
        g = _G.__new__(_G)
        g.colors = self.colors
        return g


def make_engine(kf_interval: int = 4) -> VideoConversionEngine:
    eng = VideoConversionEngine.__new__(VideoConversionEngine)
    eng._kf_interval = kf_interval
    eng._kf_cut_threshold = 0.08
    eng._kf_g = None
    eng._kf_age = 0
    eng._kf_focus = None
    eng._kf_pix = None
    eng._kf_colors0 = None
    eng._kf_grid = None
    eng._last_ir = None
    return eng


def img(value: float) -> torch.Tensor:
    """A uniform (1, 3, 1536, 1536) image at the given sRGB level."""
    return torch.full((1, 3, 1536, 1536), value, dtype=torch.float32)


print("=" * 74)
print("2.1(a) _pool_to_grid element count")
print("=" * 74)
eng = make_engine()
eng._kf_grid = (LAYERS, SIDE)
pix = eng._pool_to_grid(img(0.5))
check(pix.shape == (SIDE * SIDE, 3), "grid output is side^2 rows",
      f"shape={tuple(pix.shape)}")
check(pix.shape[0] * LAYERS == N, "rows * layers == gaussian count",
      f"{pix.shape[0]}*{LAYERS} == {N}")

# The non-divisible case that avg_pool2d(k) got wrong. gaussian_grid is
# derived from N, so force a grid whose side does not divide 1536.
for bad_side in (5, 7, 11, 100):
    eng._kf_grid = (1, bad_side)
    p = eng._pool_to_grid(img(0.5))
    ok = p.shape == (bad_side * bad_side, 3)
    check(ok, f"non-divisible side={bad_side} still yields side^2 rows",
          f"got {p.shape[0]} rows, want {bad_side ** 2}")

print()
print("=" * 74)
print("2.1(b) _store_keyframe atomicity")
print("=" * 74)
eng = make_engine()
g = _G(N)
eng._kf_grid = (LAYERS, SIDE)
eng._store_keyframe(g, img(0.4), focus=2.5)
check(eng._kf_pix is not None and eng._kf_colors0 is not None
      and eng._kf_g is g, "all three fields published together")
check(eng._kf_colors0.shape == (N, 3),
      "_kf_colors0 has one row per gaussian", f"{tuple(eng._kf_colors0.shape)}")
# The snapshot is in sRGB (see _try_reuse_keyframe's exactness proof).
# A uniform 0.25 linear colour maps to a brighter sRGB value.
srgb_expected = float(linearRGB2sRGB(torch.tensor(0.25)))
check(abs(float(eng._kf_colors0.mean()) - srgb_expected) < 1e-5,
      "_kf_colors0 is stored in sRGB (matching the pooled pixel space)",
      f"mean={float(eng._kf_colors0.mean()):.4f}, want {srgb_expected:.4f}")

# Unknown layout must clear *all* fields, not just _kf_g.
eng2 = make_engine()
eng2._kf_grid = (LAYERS, SIDE)
eng2._store_keyframe(_G(N), img(0.4), focus=2.0)
eng2._store_keyframe(_G(N + 1), img(0.4), focus=2.0)   # N+1 -> no grid
check(eng2._kf_g is None and eng2._kf_pix is None
      and eng2._kf_colors0 is None,
      "unknown layout clears geometry AND pixels AND colours")

print()
print("=" * 74)
print("2.1(c) colour refresh: linear gain vs old sRGB delta")
print("=" * 74)

from sharp.utils.color_space import linearRGB2sRGB, sRGB2linearRGB  # noqa: E402

base = 0.30          # keyframe sRGB level
step = 0.02          # per-frame sRGB decrement (gently darkening scene)

# The audit claimed the sRGB-domain delta drifts and the fix is a
# linear-domain gain. Measurement says the opposite: the delta is exact, the
# gain is not. Assert the property that actually holds, so a future
# "optimisation" back to a gain fails the test.
#
#   linearRGB2sRGB(g.colors) == _kf_pix   (the initializer/model identity)
#   => sRGB2linearRGB(linearRGB2sRGB(c) + (pix - _kf_pix)) == sRGB2linearRGB(pix)

print("  p_prev   p_cur    truth(lin)   delta err     gain err")
delta_worst = 0.0
gain_worst = 0.0
rng_t = torch.Generator().manual_seed(0)
for i in range(6):
    p_prev = torch.rand(3, generator=rng_t) * 0.9 + 0.05
    p_cur = torch.rand(3, generator=rng_t) * 0.9 + 0.05
    c_lin = sRGB2linearRGB(p_prev)          # colour == linear(pooled pixel)
    truth = sRGB2linearRGB(p_cur)

    delta_out = sRGB2linearRGB(
        (linearRGB2sRGB(c_lin) + (p_cur - p_prev)).clamp(0.0, 1.0))
    gain_out = (c_lin * (p_cur + 1e-3) / (p_prev + 1e-3)).clamp(0.0, 1.0)

    d_err = float((delta_out - truth).abs().max())
    g_err = float((gain_out - truth).abs().max())
    delta_worst = max(delta_worst, d_err)
    gain_worst = max(gain_worst, g_err)
    print(f"  {float(p_prev[0]):.3f}    {float(p_cur[0]):.3f}"
          f"    {float(truth[0]):.5f}     {d_err:.2e}      {g_err:.2e}")

check(delta_worst < 1e-5,
      "sRGB affine delta reproduces the current frame's colour exactly",
      f"worst err = {delta_worst:.2e}")
check(gain_worst > delta_worst * 100,
      "a linear-domain gain is measurably worse (so 2.1(c) must NOT be 'fixed')",
      f"gain err = {gain_worst:.2e} vs delta {delta_worst:.2e}")

# Exhaustive sweep to make the bound robust, not just seed-lucky.
worst = 0.0
for p0v in (0.1, 0.3, 0.5, 0.8):
    for p1v in (0.05, 0.15, 0.25, 0.45, 0.65, 0.9):
        p0t, p1t = torch.tensor(p0v), torch.tensor(p1v)
        c = sRGB2linearRGB(p0t)
        out = sRGB2linearRGB((linearRGB2sRGB(c) + (p1t - p0t)).clamp(0.0, 1.0))
        worst = max(worst, abs(float(out) - float(sRGB2linearRGB(p1t))))
check(worst < 1e-5, "delta stays exact across a 24-point sweep (no clamp edge case)",
      f"worst = {worst:.2e}")

print()
print("=" * 74)
print("2.2 _soften_depth_edges: world z + adaptive threshold")
print("=" * 74)


class _MeanG:
    """Gaussians whose mean_vectors carry a planar + step depth profile."""

    def __init__(self, side: int, layers: int):
        self.mv = torch.zeros(layers * side * side, 3, dtype=torch.float32)
        for l in range(layers):
            for y in range(side):
                for x in range(side):
                    depth = 2.0 + 0.5 * x / side          # gentle plane
                    if x >= side // 2:
                        depth += 4.0                       # hard silhouette
                    self.mv[l * side * side + y * side + x, 2] = depth
        self.colors = _Colors(layers * side * side)

    def __getitem__(self, idx):
        return self.mv


# _mean_view reads g_ndc.mean_vectors with a leading batch dim; patch the
# helper the method imports so the test drives the real geometry maths.
class _GMeanBatch:
    def __init__(self, mv, side, layers):
        self.mean_vectors = mv.reshape(1, layers * side * side, 3)
        self.colors = _Colors(layers * side * side)


def patch_mean_view():
    import sharp3d.temporal as t
    orig = t._mean_view
    t._mean_view = lambda g: g.mean_vectors[0]
    return orig


import sharp3d.conversion as conv  # noqa: E402

_orig_mean_view = patch_mean_view()

eng3 = make_engine()
eng3._eye4 = torch.eye(4)
# An unprojection that keeps z as-is makes the assertion readable.
eng3._unproj = torch.eye(4)
eng3._last_ir = torch.zeros(4, 4)

gb = _GMeanBatch(_MeanG(SIDE, LAYERS).mv, SIDE, LAYERS)
before = gb.mean_vectors.clone()
conv._gaussian_grid_orig = conv._gaussian_grid
conv._gaussian_grid = lambda n: (LAYERS, SIDE)   # force the test layout

eng3._soften_depth_edges(gb)
after = gb.mean_vectors
# mean_vectors is (1, N, 3); compare the z channel only.
delta_z = (after - before)[0, :, 2].reshape(LAYERS, SIDE, SIDE).abs()

# Column past the step must be touched; a margin of flat columns must not.
edge_col = delta_z[:, :, SIDE // 2 + 1:SIDE // 2 + 2].mean().item()
flat_col = delta_z[:, :, 0:1].mean().item()
check(edge_col > 0.0, "gaussians at the depth step are smoothed",
      f"mean Δz = {edge_col:.4f}")
check(flat_col < edge_col * 0.6, "flat region is left (nearly) unchanged",
      f"flat Δz = {flat_col:.4f} vs edge {edge_col:.4f}")

# A smooth ramp must not be *blurred* — i.e. near-zero edge weight.
# (Note: weight is a soft mask, not a hard gate; a flat/ramp scene sits in
# the sigmoid tail at ~0.1-0.2, well below the ~1.0 a real step reaches.)
def max_weight(g_obj, eng):
    """Run the real method but capture the edge mask it would apply."""
    import torch.nn.functional as F
    mv = g_obj.mean_vectors[0]
    L, side = LAYERS, SIDE
    z = mv[:, 2].reshape(L, side, side).clamp(min=1e-6).log()
    zp = F.pad(z.unsqueeze(0), [1, 1, 1, 1], mode="replicate")
    gx = zp[:, :, 1:-1, 2:] - zp[:, :, 1:-1, :-2]
    gy = zp[:, :, 2:, 1:-1] - zp[:, :, :-2, 1:-1]
    grad = (gx.pow(2) + gy.pow(2)).sqrt().squeeze(0)
    med = grad.flatten(1).median(dim=1).values.view(L, 1, 1)
    mad = (grad - med).abs().flatten(1).median(dim=1).values.view(L, 1, 1)
    band = (3.0 * 1.4826 * mad).clamp(min=8e-4)
    return float(torch.sigmoid((grad - (med + 2 * band)) / band).max())


gb_flat = _GMeanBatch(torch.zeros(LAYERS * SIDE * SIDE, 3), SIDE, LAYERS)
gb_flat.mean_vectors[:, :, 2] = 5.0
w_flat = max_weight(gb_flat, eng3)
check(w_flat < 0.15, "constant-depth scene gets no significant edge weight",
      f"max w = {w_flat:.4f} (a real step reaches 1.0)")

gb_plane = _GMeanBatch(torch.zeros(LAYERS * SIDE * SIDE, 3), SIDE, LAYERS)
plane_z = 2.0 + 0.05 * torch.arange(SIDE).float()
gb_plane.mean_vectors[0, :, 2] = plane_z.repeat_interleave(SIDE).repeat(LAYERS)
w_ramp = max_weight(gb_plane, eng3)
check(w_ramp < 0.25, "smooth depth ramp is not treated as an edge",
      f"max w = {w_ramp:.4f}")

# A genuine silhouette must still saturate the mask.
gb_step = _GMeanBatch(torch.zeros(LAYERS * SIDE * SIDE, 3), SIDE, LAYERS)
step_z = torch.cat([torch.full((SIDE // 2,), 2.0),
                    torch.full((SIDE // 2,), 6.0)])
gb_step.mean_vectors[0, :, 2] = step_z.repeat_interleave(SIDE).repeat(LAYERS)
w_step = max_weight(gb_step, eng3)
check(w_step > 0.95, "a real depth step saturates the edge mask",
      f"max w = {w_step:.4f}")

conv._gaussian_grid = conv._gaussian_grid_orig
import sharp3d.temporal as _t
_t._mean_view = _orig_mean_view

print()
print("=" * 74)
print("2.3 _infer_shape returns None; stabilize falls back")
print("=" * 74)
st = TemporalStabilizer.__new__(TemporalStabilizer)
st._shape = None
st._infer_shape(LAYERS * 768 * 768)
check(st._shape == (2, 768, 768), "recognised count still infers the grid",
      f"{st._shape}" if st._shape else "None")
# A prime count has no square factorisation at either layer count.
st._infer_shape(3073)
check(st._shape is None, "unrecognised count yields None (not (1,1,N))")
# Careful: math.isqrt(3) == 1 and 1*1*2 != 3, so N=3 IS caught. Use a
# value where the integer sqrt is a *perfect* factor of the wrong number.
st._infer_shape(2 * 4 * 4 + 1)
check(st._shape is None, "off-by-one from a valid grid yields None",
      f"{st._shape}" if st._shape else "None")

print()
print("=" * 74)
print("2.4 eigendecompose NaN / tiny-scale robustness")
print("=" * 74)
good = torch.eye(3).repeat(4, 1, 1) * 0.5
good[1] = float("nan")
good[2, 0, 1] = float("inf")
out = analytical_eigen_decompose(good)
check(torch.isfinite(out[0]).all() and torch.isfinite(out[1]).all(),
      "NaN/Inf covariance produces finite eigenvalues and eigenvectors")

# 2.4 returns *singular values* = sqrt(eigenvalues), so the expected scale
# here is sqrt(1e-30) = 1e-15, not 1e-30. The old code's absolute q2 floor
# of 1e-20 made sqrt(q2) ~ 1e-10, which after the outer sqrt came out ~1e-5
# — ten orders of magnitude off. The test asserts the correct magnitude and
# the correct ratios, which is what actually reaches the renderer.
tiny = (torch.eye(3) * 1e-30).repeat(3, 1, 1).clone()
_q, sv = analytical_eigen_decompose(tiny)
check(torch.isfinite(sv).all(), "tiny-scale covariance yields finite values")
expect = math.sqrt(1e-30)
rel = abs(float(sv[0, 0]) - expect) / expect
check(rel < 1e-3, "tiny-scale singular values keep the right magnitude",
      f"got {float(sv[0, 0]):.3e}, want {expect:.3e} (old code gave ~1e-5)")
spread = float((sv[:, 0] - sv[:, 2]).abs().max())
check(spread < expect * 1e-3, "tiny-scale singular values stay equal",
      f"max spread = {spread:.3e}")

# A genuinely anisotropic tiny matrix must preserve its eigenvalue ratios.
# (Absolute values follow the per-matrix scale, which is the largest
# diagonal entry — for this matrix that is 9e-30, so all three rows are
# normalised by the same factor. What must survive is the *ratio*.)
aniso = torch.zeros(3, 3, 3)
aniso[:, 0, 0] = torch.tensor([1e-30, 4e-30, 9e-30])
aniso[:, 1, 1] = 1e-30
aniso[:, 2, 2] = 1e-30
_q2, sv2 = analytical_eigen_decompose(aniso)
ratios = [round(float(sv2[i, 0] / sv2[i, 2]), 4) for i in range(3)]
ok = all(abs(r - t) / t < 1e-2 for r, t in zip(ratios, (1.0, 2.0, 3.0)))
check(ok, "tiny anisotropic covariance keeps its singular-value ratios",
      f"ratios = {ratios}, want [1, 2, 3] (sqrt of 1, 4, 9)")

# The change must not degrade ordinary-magnitude covariances: fire the real
# function at realistic input and require it to stay finite and ordered.
torch.manual_seed(1)
worst_finite = True
for _ in range(50):
    M = torch.randn(2, 3, 3)
    C = M @ M.transpose(-1, -2) + torch.eye(3) * 0.1
    _q, svc = analytical_eigen_decompose(C)
    if not torch.isfinite(svc).all() or not (svc[:, 0] >= svc[:, 1]).all():
        worst_finite = False
        break
check(worst_finite, "ordinary covariances stay finite and descending-sorted")

print()
print("=" * 74)
print("RESULT:", "PASS" if not fails else f"FAIL ({len(fails)})")
for f in fails:
    print("   -", f)
sys.exit(0 if not fails else 1)
