"""Projection utilities for panoramic VR conversion.

Handles:
- Input: equirectangular / fisheye → 6 cubemap pinhole faces (for SHARP prediction)
- Output: 6 rendered cubemap faces → equirectangular image (for VR display)

All operations are GPU-accelerated via PyTorch (bilinear grid_sample).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


# ─── Cubemap geometry ────────────────────────────────────────────────────────

# Face directions: +X, -X, +Y, -Y, +Z, -Z
# Each face defined by (forward, up) vectors in world space.
_FACE_DEFS = [
    # (forward, up)
    (torch.tensor([1.0, 0.0, 0.0]),  torch.tensor([0.0, -1.0, 0.0])),   # +X
    (torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0])),   # -X
    (torch.tensor([0.0, 1.0, 0.0]),  torch.tensor([0.0, 0.0, 1.0])),    # +Y
    (torch.tensor([0.0, -1.0, 0.0]), torch.tensor([0.0, 0.0, -1.0])),   # -Y
    (torch.tensor([0.0, 0.0, 1.0]),  torch.tensor([0.0, -1.0, 0.0])),   # +Z
    (torch.tensor([0.0, 0.0, -1.0]), torch.tensor([0.0, -1.0, 0.0])),   # -Z
]


def _look_at_rotation(forward: Tensor, up: Tensor) -> Tensor:
    """Build a 3x3 rotation matrix (world-to-camera) from forward/up.

    Uses OpenCV convention: X-right, Y-down, Z-forward.
    Camera looks along +Z in its local frame.
    """
    f = F.normalize(forward, dim=0)
    r = F.normalize(torch.cross(f, up, dim=0), dim=0)
    d = torch.cross(f, r, dim=0)  # down = fwd × right
    # R rows = [right, down, forward] — OpenCV convention
    return torch.stack([r, d, f], dim=0)  # [3, 3]


def get_cubemap_cameras(
    face_size: int,
    device: torch.device,
    eye_offset: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate 6 cubemap camera matrices.

    Args:
        face_size: pixel size of each square cubemap face.
        device: CUDA device.
        eye_offset: [3] translation for stereo eye (e.g. [-ipd/2, 0, 0]).

    Returns:
        viewmats: [6, 4, 4] world-to-camera extrinsics.
        Ks: [6, 3, 3] pinhole intrinsics (90° FOV).
    """
    f = face_size / 2.0  # 90° FOV: focal = size/2
    K = torch.tensor([
        [f, 0, (face_size - 1) / 2.0],
        [0, f, (face_size - 1) / 2.0],
        [0, 0, 1],
    ], device=device, dtype=torch.float32)
    Ks = K.unsqueeze(0).expand(6, -1, -1).contiguous()

    viewmats = torch.zeros(6, 4, 4, device=device, dtype=torch.float32)
    for i, (fwd, up) in enumerate(_FACE_DEFS):
        fwd = fwd.to(device)
        up = up.to(device)
        R = _look_at_rotation(fwd, up)
        viewmats[i, :3, :3] = R
        if eye_offset is not None:
            # t = -R @ eye_position
            viewmats[i, :3, 3] = -R @ eye_offset
        else:
            viewmats[i, :3, 3] = 0
        viewmats[i, 3, 3] = 1.0

    return viewmats, Ks


# ─── Overlap filtering ───────────────────────────────────────────────────────

# FOV scale for overlapping prediction: tan(55°)/tan(45°) ≈ 1.43 → 110° FOV
OVERLAP_FOV_SCALE = 1.43
# Angular keep radius: 50° from face center (5° overlap beyond standard 45°)
OVERLAP_KEEP_ANGLE_DEG = 50.0


def filter_gaussians_by_angle(
    means: Tensor,
    face_forward: Tensor,
    max_angle_deg: float = OVERLAP_KEEP_ANGLE_DEG,
) -> Tensor:
    """Return boolean mask: True for Gaussians within angular radius of face center.

    Args:
        means: [N, 3] world-space Gaussian positions.
        face_forward: [3] unit vector — face's forward direction in world space.
        max_angle_deg: maximum angular distance from face center to keep.

    Returns:
        mask: [N] boolean tensor.
    """
    # Direction from origin to each Gaussian
    dirs = F.normalize(means, dim=-1)  # [N, 3]
    fwd = F.normalize(face_forward, dim=0)  # [3]
    # Cosine of angle between direction and face forward
    cos_angle = (dirs * fwd.unsqueeze(0)).sum(dim=-1)  # [N]
    cos_thresh = math.cos(math.radians(max_angle_deg))
    return cos_angle >= cos_thresh


# ─── Input: Equirectangular → Cubemap faces ──────────────────────────────────

def _cubemap_face_rays(face_size: int, device: torch.device,
                       fov_scale: float = 1.0) -> Tensor:
    """Generate unit ray directions for each pixel of a cubemap face.

    Args:
        face_size: pixel resolution of each square face.
        device: CUDA device.
        fov_scale: FOV multiplier. 1.0 = standard 90° FOV.
                   >1.0 widens FOV for overlapping prediction
                   (e.g. 1.43 ≈ 110° FOV).

    Returns: [6, face_size, face_size, 3] world-space ray directions.
    """
    # Pixel grid in [-fov_scale, fov_scale]
    coords = torch.linspace(-fov_scale, fov_scale, face_size, device=device)
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    # Local ray: camera looks along +Z (OpenCV convention)
    # X right, Y down, Z forward
    local = torch.stack([gx, gy, torch.ones_like(gx)], dim=-1)  # [H, W, 3]
    local = F.normalize(local, dim=-1)

    all_rays = []
    for fwd, up in _FACE_DEFS:
        fwd = fwd.to(device)
        up = up.to(device)
        R = _look_at_rotation(fwd, up)  # world-to-camera
        R_inv = R.T  # camera-to-world
        # Transform local rays to world space
        rays_world = (R_inv @ local.reshape(-1, 3).T).T  # [H*W, 3]
        rays_world = rays_world.reshape(face_size, face_size, 3)
        all_rays.append(rays_world)

    return torch.stack(all_rays, dim=0)  # [6, H, W, 3]


def equirect_to_cubemap(
    image: Tensor,
    face_size: int,
    fov_scale: float = 1.0,
) -> Tensor:
    """Sample an equirectangular image into 6 cubemap faces.

    Args:
        image: [H, W, 3] or [1, 3, H, W] equirectangular image (float, any range).
        face_size: output cubemap face resolution.
        fov_scale: FOV multiplier (1.0 = 90°, >1.0 = wider for overlap).

    Returns:
        faces: [6, 3, face_size, face_size] cubemap face images.
    """
    if image.dim() == 3:
        image = image.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    device = image.device
    _, C, H, W = image.shape

    rays = _cubemap_face_rays(face_size, device, fov_scale)  # [6, fh, fw, 3]
    rays_flat = rays.reshape(6, -1, 3)  # [6, N, 3]

    # Convert ray directions to equirectangular UV
    x, y, z = rays_flat[..., 0], rays_flat[..., 1], rays_flat[..., 2]
    # longitude: atan2(x, z) → [-π, π]
    lon = torch.atan2(x, z)
    # latitude: asin(y / norm) → [-π/2, π/2]
    lat = torch.asin(y.clamp(-1, 1))

    # Normalize to [-1, 1] for grid_sample
    u = lon / math.pi        # [-1, 1]
    v = -lat / (math.pi / 2) # [-1, 1] (flip: top = +90°)

    grid = torch.stack([u, v], dim=-1)  # [6, N, 2]
    grid = grid.reshape(6, face_size, face_size, 2)

    # Sample each face
    img_expanded = image.expand(6, -1, -1, -1)  # [6, 3, H, W]
    faces = F.grid_sample(
        img_expanded, grid, mode="bilinear",
        padding_mode="border", align_corners=True,
    )  # [6, 3, face_size, face_size]

    return faces


# ─── Input: Fisheye → Cubemap faces ─────────────────────────────────────────

def _fisheye_theta_to_r(theta: Tensor, model: str, coeffs: list[float] | None) -> Tensor:
    """Convert incidence angle θ to radial distance r (normalized by f).

    Models:
        equidistant:     r = θ
        equisolid:       r = 2·sin(θ/2)
        orthographic:    r = sin(θ)
        stereographic:   r = 2·tan(θ/2)
        ftheta:          r = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷
    """
    if model == "equidistant":
        return theta
    elif model == "equisolid":
        return 2.0 * torch.sin(theta / 2.0)
    elif model == "orthographic":
        return torch.sin(theta)
    elif model == "stereographic":
        return 2.0 * torch.tan(theta / 2.0)
    elif model == "ftheta":
        k1, k2, k3 = (coeffs or [0, 0, 0])
        return theta + k1 * theta**3 + k2 * theta**5 + k3 * theta**7
    else:
        return theta  # fallback to equidistant


def fisheye_to_cubemap(
    image: Tensor,
    face_size: int,
    model: str = "equidistant",
    coeffs: list[float] | None = None,
    fisheye_fov: float = 180.0,
    fov_scale: float = 1.0,
) -> Tensor:
    """Sample a circular fisheye image into 6 cubemap faces.

    Args:
        image: [H, W, 3] or [1, 3, H, W] fisheye image.
        face_size: output cubemap face resolution.
        model: fisheye projection model name.
        coeffs: FTheta polynomial coefficients [k1, k2, k3].
        fisheye_fov: full field-of-view of the fisheye in degrees.
        fov_scale: FOV multiplier (1.0 = 90°, >1.0 = wider for overlap).

    Returns:
        faces: [6, 3, face_size, face_size] cubemap faces.
        Pixels outside the fisheye circle are black (zero).
    """
    if image.dim() == 3:
        image = image.permute(2, 0, 1).unsqueeze(0)
    device = image.device
    _, C, H, W = image.shape

    rays = _cubemap_face_rays(face_size, device, fov_scale)  # [6, fh, fw, 3]
    rays_flat = rays.reshape(6, -1, 3)

    # Fisheye camera looks along +Z, image plane is XY
    # Incidence angle from optical axis
    x, y, z = rays_flat[..., 0], rays_flat[..., 1], rays_flat[..., 2]
    theta = torch.acos(z.clamp(-1, 1))  # angle from +Z axis

    # Radial distance in image plane (normalized)
    r_norm = _fisheye_theta_to_r(theta, model, coeffs)

    # Max theta for the fisheye FOV
    max_theta = math.radians(fisheye_fov / 2.0)
    max_r = _fisheye_theta_to_r(
        torch.tensor(max_theta, device=device), model, coeffs
    )

    # Azimuthal angle in image plane
    phi = torch.atan2(y, x)

    # Image coordinates (normalized to [-1, 1] for grid_sample)
    r_px = r_norm / max_r  # normalize to [0, 1] at edge of fisheye
    u = r_px * torch.cos(phi)
    v = r_px * torch.sin(phi)

    # Mask: pixels beyond fisheye FOV are invalid
    valid = theta <= max_theta

    grid = torch.stack([u, v], dim=-1)  # [6, N, 2]
    grid = grid.reshape(6, face_size, face_size, 2)

    img_expanded = image.expand(6, -1, -1, -1)
    faces = F.grid_sample(
        img_expanded, grid, mode="bilinear",
        padding_mode="zeros", align_corners=True,
    )

    # Zero out invalid pixels
    valid_mask = valid.reshape(6, 1, face_size, face_size).float()
    faces = faces * valid_mask

    return faces


# ─── Output: Cubemap faces → Equirectangular ─────────────────────────────────

def cubemap_to_equirect(
    faces: Tensor,
    out_w: int,
    out_h: int,
) -> Tensor:
    """Assemble 6 cubemap faces into an equirectangular image.

    Args:
        faces: [6, 3, face_size, face_size] rendered cubemap faces.
        out_w: output equirectangular width.
        out_h: output equirectangular height.

    Returns:
        equirect: [out_h, out_w, 3] equirectangular image.
    """
    device = faces.device
    face_size = faces.shape[2]

    # Generate equirectangular pixel grid
    # longitude: [-π, π], latitude: [π/2, -π/2] (top to bottom)
    lon = torch.linspace(-math.pi, math.pi, out_w, device=device)
    lat = torch.linspace(math.pi / 2, -math.pi / 2, out_h, device=device)
    grid_lat, grid_lon = torch.meshgrid(lat, lon, indexing="ij")

    # Convert to 3D ray directions
    x = torch.cos(grid_lat) * torch.sin(grid_lon)
    y = torch.sin(grid_lat)
    z = torch.cos(grid_lat) * torch.cos(grid_lon)
    dirs = torch.stack([x, y, z], dim=-1)  # [out_h, out_w, 3]

    # For each pixel, determine which cubemap face it belongs to
    # and compute the UV coordinate on that face
    abs_x = x.abs()
    abs_y = y.abs()
    abs_z = z.abs()

    # Determine dominant axis (face selection)
    # +X: x > 0 and abs_x >= abs_y and abs_x >= abs_z
    # -X: x < 0 and abs_x >= abs_y and abs_x >= abs_z
    # +Y: y > 0 and abs_y > abs_x and abs_y >= abs_z
    # -Y: y < 0 and abs_y > abs_x and abs_y >= abs_z
    # +Z: z > 0 and abs_z > abs_x and abs_z > abs_y
    # -Z: z < 0 and abs_z > abs_x and abs_z > abs_y
    face_idx = torch.zeros(out_h, out_w, dtype=torch.long, device=device)
    face_idx[(x > 0) & (abs_x >= abs_y) & (abs_x >= abs_z)] = 0   # +X
    face_idx[(x < 0) & (abs_x >= abs_y) & (abs_x >= abs_z)] = 1   # -X
    face_idx[(y > 0) & (abs_y > abs_x) & (abs_y >= abs_z)] = 2    # +Y
    face_idx[(y < 0) & (abs_y > abs_x) & (abs_y >= abs_z)] = 3    # -Y
    face_idx[(z > 0) & (abs_z > abs_x) & (abs_z > abs_y)] = 4     # +Z
    face_idx[(z < 0) & (abs_z > abs_x) & (abs_z > abs_y)] = 5     # -Z

    # Compute UV on each face using the face's projection
    # For each face, project the 3D direction onto the face plane
    u = torch.zeros(out_h, out_w, device=device)
    v = torch.zeros(out_h, out_w, device=device)

    # +X face: u = -z/x, v = y/x (OpenCV: Y-down)
    mask = face_idx == 0
    u[mask] = -z[mask] / abs_x[mask]
    v[mask] = y[mask] / abs_x[mask]

    # -X face: u = z/|x|, v = y/|x|
    mask = face_idx == 1
    u[mask] = z[mask] / abs_x[mask]
    v[mask] = y[mask] / abs_x[mask]

    # +Y face: u = x/y, v = -z/y
    mask = face_idx == 2
    u[mask] = x[mask] / abs_y[mask]
    v[mask] = -z[mask] / abs_y[mask]

    # -Y face: u = x/|y|, v = z/|y|
    mask = face_idx == 3
    u[mask] = x[mask] / abs_y[mask]
    v[mask] = z[mask] / abs_y[mask]

    # +Z face: u = x/z, v = y/z
    mask = face_idx == 4
    u[mask] = x[mask] / abs_z[mask]
    v[mask] = y[mask] / abs_z[mask]

    # -Z face: u = -x/|z|, v = y/|z|
    mask = face_idx == 5
    u[mask] = -x[mask] / abs_z[mask]
    v[mask] = y[mask] / abs_z[mask]

    # Clamp to [-1, 1]
    u = u.clamp(-1, 1)
    v = v.clamp(-1, 1)

    # Sample from the appropriate face using grid_sample per face
    equirect = torch.zeros(out_h, out_w, 3, device=device)

    for fi in range(6):
        mask = (face_idx == fi)  # [out_h, out_w]
        if not mask.any():
            continue

        # Build grid for this face's pixels
        face_grid = torch.stack([u[mask], v[mask]], dim=-1)  # [N, 2]
        n_pix = face_grid.shape[0]

        # Reshape for grid_sample: need [1, 3, H, W] input and [1, H_out, W_out, 2] grid
        # We'll use a 1×N grid
        face_grid_4d = face_grid.reshape(1, 1, n_pix, 2)
        face_img = faces[fi:fi+1]  # [1, 3, face_size, face_size]

        sampled = F.grid_sample(
            face_img, face_grid_4d, mode="bilinear",
            padding_mode="border", align_corners=True,
        )  # [1, 3, 1, N]

        equirect[mask] = sampled[0, :, 0, :].T  # [N, 3]

    return equirect


# ─── Utility: 180° equirectangular output ────────────────────────────────────

def cubemap_to_equirect180(
    faces: Tensor,
    out_w: int,
    out_h: int,
) -> Tensor:
    """Assemble cubemap faces into a 180° equirectangular (front hemisphere).

    Only longitude [-90°, +90°] is kept (front-facing).

    Args:
        faces: [6, 3, face_size, face_size] rendered cubemap faces.
        out_w: output width (covers 180° horizontal).
        out_h: output height (covers 180° vertical).

    Returns:
        equirect180: [out_h, out_w, 3] image.
    """
    device = faces.device

    # longitude: [-π/2, π/2], latitude: [π/2, -π/2]
    lon = torch.linspace(-math.pi / 2, math.pi / 2, out_w, device=device)
    lat = torch.linspace(math.pi / 2, -math.pi / 2, out_h, device=device)
    grid_lat, grid_lon = torch.meshgrid(lat, lon, indexing="ij")

    x = torch.cos(grid_lat) * torch.sin(grid_lon)
    y = torch.sin(grid_lat)
    z = torch.cos(grid_lat) * torch.cos(grid_lon)

    abs_x = x.abs()
    abs_y = y.abs()
    abs_z = z.abs()

    face_idx = torch.zeros(out_h, out_w, dtype=torch.long, device=device)
    face_idx[(x > 0) & (abs_x >= abs_y) & (abs_x >= abs_z)] = 0
    face_idx[(x < 0) & (abs_x >= abs_y) & (abs_x >= abs_z)] = 1
    face_idx[(y > 0) & (abs_y > abs_x) & (abs_y >= abs_z)] = 2
    face_idx[(y < 0) & (abs_y > abs_x) & (abs_y >= abs_z)] = 3
    face_idx[(z > 0) & (abs_z > abs_x) & (abs_z > abs_y)] = 4
    face_idx[(z < 0) & (abs_z > abs_x) & (abs_z > abs_y)] = 5

    u = torch.zeros(out_h, out_w, device=device)
    v = torch.zeros(out_h, out_w, device=device)

    mask = face_idx == 0
    u[mask] = -z[mask] / abs_x[mask]
    v[mask] = y[mask] / abs_x[mask]

    mask = face_idx == 1
    u[mask] = z[mask] / abs_x[mask]
    v[mask] = y[mask] / abs_x[mask]

    mask = face_idx == 2
    u[mask] = x[mask] / abs_y[mask]
    v[mask] = -z[mask] / abs_y[mask]

    mask = face_idx == 3
    u[mask] = x[mask] / abs_y[mask]
    v[mask] = z[mask] / abs_y[mask]

    mask = face_idx == 4
    u[mask] = x[mask] / abs_z[mask]
    v[mask] = y[mask] / abs_z[mask]

    mask = face_idx == 5
    u[mask] = -x[mask] / abs_z[mask]
    v[mask] = y[mask] / abs_z[mask]

    u = u.clamp(-1, 1)
    v = v.clamp(-1, 1)

    equirect = torch.zeros(out_h, out_w, 3, device=device)

    for fi in range(6):
        mask = (face_idx == fi)
        if not mask.any():
            continue
        face_grid = torch.stack([u[mask], v[mask]], dim=-1)
        n_pix = face_grid.shape[0]
        face_grid_4d = face_grid.reshape(1, 1, n_pix, 2)
        face_img = faces[fi:fi+1]
        sampled = F.grid_sample(
            face_img, face_grid_4d, mode="bilinear",
            padding_mode="border", align_corners=True,
        )
        equirect[mask] = sampled[0, :, 0, :].T

    return equirect
