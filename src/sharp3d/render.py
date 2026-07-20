"""Batched SBS stereoscopic rendering using gsplat.

BUG#2 FIX: Extrinsics must be world-to-camera (R.T), not camera-to-world (R).
    We use SHARP's camera_model.compute() which correctly handles this via
    create_camera_matrix(inverse=True) → rotation_matrix.transpose(-1, -2).

BUG#3 FIX: SHARP Gaussians store colors in linearRGB. Rendered output must
    be gamma-corrected via linearRGB2sRGB() before display/saving.

BUG#6 FIX: camera_model.compute() creates look_at/world_up on CPU internally.
    eye_pos must be a CPU tensor (no device= argument).

Optimization: Replace GSplatRenderer's Python for-loop (renders L/R separately)
with a single gsplat.rendering.rasterization() call using stacked viewmats [2, 4, 4].
"""

import torch
import gsplat

from sharp.utils import camera as sharp_camera
from sharp.utils.gaussians import Gaussians3D


def linearRGB2sRGB(linearRGB: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB to sRGB (gamma correction).

    Reference: Apple Metal Shading Language Specification, Section 7.7.7.
    BUG#3: Without this, rendered images appear too dark/washed out.
    """
    THRESHOLD = 0.0031308
    low = linearRGB * 12.92
    high = 1.055 * linearRGB.clamp(min=THRESHOLD).pow(1.0 / 2.4) - 0.055
    return torch.where(linearRGB <= THRESHOLD, low, high)


def render_sbs(
    gaussians: Gaussians3D,
    f_px: float,
    orig_w: int,
    orig_h: int,
    ipd: float = 0.063,
    convergence: float | None = None,
    render_width: int | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Render stereoscopic SBS pair using batched gsplat rasterization.

    Args:
        gaussians: World-space Gaussians (unprojected).
        f_px: Focal length in pixels (original image space).
        orig_w: Original image width.
        orig_h: Original image height.
        ipd: Inter-pupillary distance in scene units.
        convergence: Convergence distance in scene units (None = auto focus).
        render_width: If set, render each eye at this width (preview mode,
                      faster). None = render at original resolution.

    Returns:
        sbs_image: (H, W*2, 3) uint8 tensor (left | right concatenated).
        (single_width, height): Dimensions of each eye's image.
    """
    device = gaussians.mean_vectors.device

    # Preview resolution scaling (keeps focal length consistent)
    if render_width is not None and render_width != orig_w:
        scale = render_width / orig_w
        f_px_r = f_px * scale
        w_r = render_width
        h_r = round(orig_h * scale)
        h_r += h_r % 2  # ensure even for video codecs
        w_r += w_r % 2
    else:
        f_px_r, w_r, h_r = f_px, orig_w, orig_h

    # Build intrinsics for output resolution
    intrinsics = torch.tensor([
        [f_px_r, 0, (w_r - 1) / 2.0, 0],
        [0, f_px_r, (h_r - 1) / 2.0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device)

    # Use SHARP's camera model for correct extrinsics computation
    # BUG#2 FIX: camera_model.compute() uses create_camera_matrix(inverse=True)
    # which correctly transposes rotation (world-to-camera).
    camera_model = sharp_camera.create_camera_model(
        gaussians, intrinsics, resolution_px=(w_r, h_r)
    )

    # Convergence control: override the auto focus distance.
    # lookat_point makes both eyes converge at z = convergence.
    if convergence is not None:
        camera_model.lookat_point = (0.0, 0.0, float(convergence))

    # BUG#6 FIX: eye_pos must be CPU tensor (camera_model.compute uses CPU internally)
    left_info = camera_model.compute(torch.tensor([-ipd / 2, 0.0, 0.0]))
    right_info = camera_model.compute(torch.tensor([ipd / 2, 0.0, 0.0]))

    # Stack viewmats for batched rendering: (2, 4, 4)
    viewmats = torch.stack([
        left_info.extrinsics.to(device),
        right_info.extrinsics.to(device),
    ], dim=0)  # (2, 4, 4)

    # Intrinsics for gsplat: (2, 3, 3)
    K = left_info.intrinsics[:3, :3].to(device)
    Ks = K.unsqueeze(0).expand(2, -1, -1)  # (2, 3, 3)

    render_w = left_info.width
    render_h = left_info.height

    # Remove batch dim from gaussians (gsplat expects unbatched)
    means = gaussians.mean_vectors
    quats = gaussians.quaternions
    scales = gaussians.singular_values
    opacities = gaussians.opacities
    colors = gaussians.colors

    if means.ndim == 3:
        means = means[0]
        quats = quats[0]
        scales = scales[0]
        opacities = opacities[0]
        colors = colors[0]

    # Single batched rasterization call for both eyes
    rendered_colors, rendered_alphas, meta = gsplat.rendering.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=render_w,
        height=render_h,
        render_mode="RGB",
        rasterize_mode="classic",
        absgrad=False,
        packed=False,
    )
    # rendered_colors: (2, H, W, 3) in linearRGB
    # rendered_alphas: (2, H, W, 1)

    # BUG#3 FIX: linearRGB → sRGB gamma correction
    rendered_colors = linearRGB2sRGB(rendered_colors)

    # Convert to uint8 images
    left_img = (rendered_colors[0] * 255).clamp(0, 255).to(torch.uint8)   # (H, W, 3)
    right_img = (rendered_colors[1] * 255).clamp(0, 255).to(torch.uint8)  # (H, W, 3)

    # Concatenate horizontally for SBS
    sbs = torch.cat([left_img, right_img], dim=1)  # (H, W*2, 3)

    return sbs, (render_w, render_h)


def render_single(
    gaussians: Gaussians3D,
    f_px: float,
    orig_w: int,
    orig_h: int,
    eye_pos: torch.Tensor,
    render_width: int | None = None,
) -> torch.Tensor:
    """Render a single view from an arbitrary eye position (2.5D animation).

    Args:
        gaussians: World-space Gaussians.
        f_px: Focal length in pixels (original image space).
        orig_w: Original image width.
        orig_h: Original image height.
        eye_pos: (3,) CPU tensor — camera position in scene units.
        render_width: If set, render at this width (preview mode).

    Returns:
        image: (H, W, 3) uint8 tensor.
    """
    device = gaussians.mean_vectors.device

    if render_width is not None and render_width != orig_w:
        scale = render_width / orig_w
        f_px_r = f_px * scale
        w_r = render_width
        h_r = round(orig_h * scale)
        h_r += h_r % 2
        w_r += w_r % 2
    else:
        f_px_r, w_r, h_r = f_px, orig_w, orig_h

    intrinsics = torch.tensor([
        [f_px_r, 0, (w_r - 1) / 2.0, 0],
        [0, f_px_r, (h_r - 1) / 2.0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device)

    camera_model = sharp_camera.create_camera_model(
        gaussians, intrinsics, resolution_px=(w_r, h_r)
    )
    # BUG#6: eye_pos must be a CPU tensor
    info = camera_model.compute(eye_pos.detach().cpu())

    viewmats = info.extrinsics[None].to(device)  # (1, 4, 4)
    Ks = info.intrinsics[:3, :3][None].to(device)  # (1, 3, 3)

    means = gaussians.mean_vectors
    quats = gaussians.quaternions
    scales = gaussians.singular_values
    opacities = gaussians.opacities
    colors = gaussians.colors
    if means.ndim == 3:
        means = means[0]
        quats = quats[0]
        scales = scales[0]
        opacities = opacities[0]
        colors = colors[0]

    rendered_colors, rendered_alphas, meta = gsplat.rendering.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=info.width,
        height=info.height,
        render_mode="RGB",
        rasterize_mode="classic",
        absgrad=False,
        packed=False,
    )

    # BUG#3 FIX: linearRGB → sRGB gamma correction
    rendered_colors = linearRGB2sRGB(rendered_colors)
    return (rendered_colors[0] * 255).clamp(0, 255).to(torch.uint8)


def render_depth_map(
    gaussians: Gaussians3D,
    f_px: float,
    orig_w: int,
    orig_h: int,
) -> torch.Tensor:
    """Render depth map visualization (center viewpoint).

    Returns:
        depth_img: (H, W, 3) uint8 tensor with colorized depth.
    """
    device = gaussians.mean_vectors.device

    intrinsics = torch.tensor([
        [f_px, 0, (orig_w - 1) / 2.0, 0],
        [0, f_px, (orig_h - 1) / 2.0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device)

    camera_model = sharp_camera.create_camera_model(
        gaussians, intrinsics, resolution_px=(orig_w, orig_h)
    )
    center_info = camera_model.compute(torch.tensor([0.0, 0.0, 0.0]))

    viewmats = center_info.extrinsics[None].to(device)  # (1, 4, 4)
    Ks = center_info.intrinsics[:3, :3][None].to(device)  # (1, 3, 3)

    means = gaussians.mean_vectors
    quats = gaussians.quaternions
    scales = gaussians.singular_values
    opacities = gaussians.opacities
    colors = gaussians.colors

    if means.ndim == 3:
        means = means[0]
        quats = quats[0]
        scales = scales[0]
        opacities = opacities[0]
        colors = colors[0]

    rendered_colors, rendered_alphas, meta = gsplat.rendering.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=center_info.width,
        height=center_info.height,
        render_mode="RGB+D",
        rasterize_mode="classic",
        absgrad=False,
        packed=False,
    )

    # Depth is in channel 3
    depth = rendered_colors[0, :, :, 3]  # (H, W)
    alpha = rendered_alphas[0, :, :, 0]  # (H, W)
    depth = depth / alpha.clamp(min=1e-8)

    # Colorize depth (near=warm, far=cool)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_norm = depth_norm.clamp(0, 1)

    # Simple turbo-like colormap via interpolation
    r = (1.0 - depth_norm).clamp(0, 1)
    g = (1.0 - (depth_norm - 0.5).abs() * 2).clamp(0, 1)
    b = depth_norm.clamp(0, 1)
    depth_rgb = torch.stack([r, g, b], dim=-1)

    return (depth_rgb * 255).to(torch.uint8)
