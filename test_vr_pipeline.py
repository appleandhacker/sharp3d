"""Smoke test for sharp3d VR pipeline components."""
import sys
sys.path.insert(0, r"C:\Users\yhm\.qoderworkcn\workspace\mrsw6dewe12d0mgy\sharp3d\src")

import torch
import math

def test_projection_imports():
    """Test all projection functions import correctly."""
    from sharp3d.projection import (
        equirect_to_cubemap, equirect_to_hemisphere,
        fisheye_to_cubemap, fisheye_to_hemisphere,
        cubemap_to_equirect, cubemap_to_equirect180,
        get_cubemap_cameras, get_hemisphere_cameras,
        filter_gaussians_by_angle,
        OVERLAP_FOV_SCALE, OVERLAP_KEEP_ANGLE_DEG,
        _HEMISPHERE_AXES, _N_HEMI_FACES, _FACE_DEFS,
    )
    assert _N_HEMI_FACES == 4, f"Expected 4 hemisphere faces, got {_N_HEMI_FACES}"
    assert len(_HEMISPHERE_AXES) == 4
    assert len(_FACE_DEFS) == 6
    assert abs(OVERLAP_FOV_SCALE - 1.5) < 0.01
    assert abs(OVERLAP_KEEP_ANGLE_DEG - 56.0) < 0.01
    print("  [PASS] projection imports")

def test_hemisphere_axes_coverage():
    """Verify 4-axis arrangement covers hemisphere (covering radius < 56°)."""
    from sharp3d.projection import _HEMISPHERE_AXES
    import torch.nn.functional as F

    # Check tilt angle is arctan(sqrt(2)) ≈ 54.74°
    for fwd, up in _HEMISPHERE_AXES:
        fwd_n = F.normalize(fwd, dim=0)
        # Angle from +Z pole
        cos_tilt = fwd_n[2].item()
        tilt_deg = math.degrees(math.acos(cos_tilt))
        assert abs(tilt_deg - 54.74) < 0.5, f"Tilt {tilt_deg}° != 54.74°"

    # Verify covering radius: sample hemisphere points, check max min-angle
    device = torch.device("cpu")
    n_test = 1000
    # Random points on hemisphere (z >= 0)
    theta = torch.rand(n_test) * (math.pi / 2)  # [0, 90°]
    phi = torch.rand(n_test) * (2 * math.pi)
    pts = torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta),
    ], dim=-1)  # [N, 3]

    axes = torch.stack([F.normalize(ax[0], dim=0) for ax in _HEMISPHERE_AXES])  # [4, 3]
    # Cosine of angle between each point and each axis
    cos_angles = pts @ axes.T  # [N, 4]
    # For each point, find nearest axis (max cosine = min angle)
    max_cos = cos_angles.max(dim=1).values  # [N]
    min_angle = torch.acos(max_cos.clamp(max=1.0))
    covering_radius_deg = math.degrees(min_angle.max().item())

    assert covering_radius_deg < 56.0, \
        f"Covering radius {covering_radius_deg:.2f}° >= 56° (half-angle)"
    print(f"  [PASS] hemisphere coverage: radius={covering_radius_deg:.2f}° < 56°")

def test_equirect_roundtrip():
    """Test equirect→cubemap→equirect preserves image (PSNR check)."""
    from sharp3d.projection import equirect_to_cubemap, cubemap_to_equirect
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Smooth gradient image (realistic content, not random noise)
    H, W = 512, 1024
    y = torch.linspace(0, 1, H, device=device).unsqueeze(1).expand(H, W)
    x = torch.linspace(0, 1, W, device=device).unsqueeze(0).expand(H, W)
    img = torch.stack([x, y, (x + y) / 2], dim=-1)  # [H, W, 3] smooth

    faces = equirect_to_cubemap(img, 256)
    assert faces.shape == (6, 3, 256, 256), f"Bad shape: {faces.shape}"

    # Reconstruct
    recon = cubemap_to_equirect(faces, W, H)
    assert recon.shape == (H, W, 3), f"Bad recon shape: {recon.shape}"

    # PSNR (smooth content should reconstruct well)
    mse = ((img - recon) ** 2).mean()
    psnr = -10 * math.log10(mse.item() + 1e-10)
    assert psnr > 20, f"PSNR too low: {psnr:.1f}dB"
    print(f"  [PASS] equirect roundtrip PSNR={psnr:.1f}dB")

def test_equirect180_wrapper():
    """Test cubemap_to_equirect180 is a proper wrapper."""
    from sharp3d.projection import cubemap_to_equirect, cubemap_to_equirect180
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    faces = torch.rand(6, 3, 128, 128, device=device)
    r1 = cubemap_to_equirect180(faces, 256, 256)
    r2 = cubemap_to_equirect(faces, 256, 256, half_sphere=True)
    assert torch.allclose(r1, r2), "equirect180 wrapper mismatch"
    print("  [PASS] equirect180 wrapper consistent")

def test_prepare_input_gpu():
    """Test GPU-direct prepare_input."""
    from sharp3d.unproject import prepare_input_gpu, INTERNAL_SHAPE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Simulate a face already on GPU
    face = torch.rand(3, 1536, 1536, device=device)
    f_px = 1536 / (2.0 * 1.5)

    img_r, df, ir, (w, h) = prepare_input_gpu(face, f_px, device)
    assert img_r.shape == (1, 3, *INTERNAL_SHAPE), f"Bad shape: {img_r.shape}"
    assert df.dtype == torch.float32
    assert ir.shape == (4, 4)
    assert (w, h) == (1536, 1536)
    print("  [PASS] prepare_input_gpu")

def test_fisheye_stereographic_no_nan():
    """Test stereographic model doesn't produce NaN/inf."""
    from sharp3d.projection import _fisheye_theta_to_r

    theta = torch.linspace(0, math.pi, 1000)  # includes π
    r = _fisheye_theta_to_r(theta, "stereographic", None)
    assert not torch.isnan(r).any(), "NaN in stereographic"
    assert not torch.isinf(r).any(), "Inf in stereographic"
    print("  [PASS] stereographic no NaN/inf")

def test_filter_gaussians_by_angle():
    """Test angular filter keeps correct subset."""
    from sharp3d.projection import filter_gaussians_by_angle

    # Points at various angles from +Z
    means = torch.tensor([
        [0, 0, 1.0],    # 0° from +Z
        [1, 0, 1.0],    # 45° from +Z
        [2, 0, 1.0],    # 63.4° from +Z
        [10, 0, 1.0],   # ~84° from +Z
    ])
    fwd = torch.tensor([0.0, 0.0, 1.0])
    mask = filter_gaussians_by_angle(means, fwd, 56.0)
    # Should keep 0° and 45°, reject 63.4° and 84°
    assert mask[0] == True
    assert mask[1] == True
    assert mask[2] == False
    assert mask[3] == False
    print("  [PASS] filter_gaussians_by_angle")

def test_render_face_size():
    """Test adaptive render face size computation."""
    sys.path.insert(0, r"C:\Users\yhm\.qoderworkcn\workspace\mrsw6dewe12d0mgy\sharp3d\src\sharp3d\gui")
    from worker import _compute_render_face_size

    # 180° output: eye_w, min 2048, round up 256
    assert _compute_render_face_size(4096, "equirect180") == 4096
    assert _compute_render_face_size(1024, "equirect180") == 2048  # min
    assert _compute_render_face_size(3000, "equirect180") == 3072  # round up

    # 360° output: eye_w/2, min 2048, round up 256
    assert _compute_render_face_size(4096, "equirect360") == 2048
    assert _compute_render_face_size(8192, "equirect360") == 4096
    assert _compute_render_face_size(2048, "equirect360") == 2048  # min
    print("  [PASS] _compute_render_face_size")


if __name__ == "__main__":
    print("=" * 50)
    print("sharp3d VR pipeline smoke tests")
    print("=" * 50)

    tests = [
        test_projection_imports,
        test_hemisphere_axes_coverage,
        test_fisheye_stereographic_no_nan,
        test_filter_gaussians_by_angle,
        test_render_face_size,
        test_prepare_input_gpu,
        test_equirect_roundtrip,
        test_equirect180_wrapper,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
