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


def _compute_focus_depth_gpu(means: torch.Tensor, min_depth_focus: float = 2.0,
                             q_focus: float = 0.1) -> float:
    """Compute focus depth entirely on GPU (replaces CPU-bound _compute_depth_quantiles).

    Since screen_extrinsics is always identity, depth = z-coordinate.
    torch.quantile on GPU avoids the 14MB GPU→CPU transfer + CPU sort.
    """
    depth_values = means[:, 2]  # z-coordinate = depth (identity extrinsics)
    depth_values = depth_values[depth_values > 0]
    if depth_values.numel() == 0:
        return min_depth_focus
    focus = float(torch.quantile(depth_values, q_focus))
    return max(min_depth_focus, focus)


def _look_at_extrinsics_gpu(eye_pos: torch.Tensor, look_at: torch.Tensor,
                            world_up: torch.Tensor) -> torch.Tensor:
    """Compute world-to-camera extrinsics (inverse look-at) on GPU.

    Equivalent to create_camera_matrix(position, look_at, world_up, inverse=True)
    but operates entirely on GPU tensors.
    """
    front = look_at - eye_pos
    front = front / front.norm()
    right = torch.linalg.cross(front, world_up)
    right = right / right.norm()
    down = torch.linalg.cross(front, right)

    # rotation_matrix columns = [right, down, front]
    # inverse: R = rotation.T, t = -R @ position
    R = torch.stack([right, down, front], dim=-1)  # (3, 3) columns
    R_inv = R.T  # (3, 3)
    t_inv = -R_inv @ eye_pos

    extrinsics = torch.eye(4, device=eye_pos.device, dtype=eye_pos.dtype)
    extrinsics[:3, :3] = R_inv
    extrinsics[:3, 3] = t_inv
    return extrinsics


def _get_screen_resolution(width: int, height: int) -> tuple[int, int]:
    """Match SHARP's get_screen_resolution_px_from_input logic."""
    w, h = width, height
    if h > 3000:
        w, h = w // 2, h // 2
    if w % 2 != 0:
        w += 1
    if h % 2 != 0:
        h += 1
    return w, h


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

    GPU-native camera setup: bypasses SHARP's CPU-bound create_camera_model
    (which transfers 1.18M points to CPU + sorts for quantiles = ~240ms).
    Instead computes focus depth and look-at matrices entirely on GPU (~1ms).

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

    # Apply SHARP's screen resolution logic (halve if >3000px height, enforce even)
    screen_w, screen_h = _get_screen_resolution(w_r, h_r)
    # Rescale intrinsics to match screen resolution
    f_px_screen = f_px_r * (screen_w / w_r)

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

    # --- GPU-native camera setup (replaces 241ms CPU-bound create_camera_model) ---
    if convergence is not None:
        depth_focus = float(convergence)
    else:
        depth_focus = _compute_focus_depth_gpu(means)

    # Look-at target: origin + [0, 0, depth_focus] (matches SHARP's "point" mode)
    look_at = torch.tensor([0.0, 0.0, depth_focus], device=device)
    world_up = torch.tensor([0.0, -1.0, 0.0], device=device)

    # Eye positions
    left_eye = torch.tensor([-ipd / 2, 0.0, 0.0], device=device)
    right_eye = torch.tensor([ipd / 2, 0.0, 0.0], device=device)

    # Compute extrinsics on GPU (world-to-camera)
    left_ext = _look_at_extrinsics_gpu(left_eye, look_at, world_up)
    right_ext = _look_at_extrinsics_gpu(right_eye, look_at, world_up)
    viewmats = torch.stack([left_ext, right_ext], dim=0)  # (2, 4, 4)

    # Intrinsics for gsplat: (2, 3, 3)
    K = torch.tensor([
        [f_px_screen, 0, (screen_w - 1) / 2.0],
        [0, f_px_screen, (screen_h - 1) / 2.0],
        [0, 0, 1],
    ], dtype=torch.float32, device=device)
    Ks = K.unsqueeze(0).expand(2, -1, -1)  # (2, 3, 3)

    # Single batched rasterization call for both eyes
    rendered_colors, rendered_alphas, meta = gsplat.rendering.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=screen_w,
        height=screen_h,
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

    return sbs, (screen_w, screen_h)


def render_single(
    gaussians: Gaussians3D,
    f_px: float,
    orig_w: int,
    orig_h: int,
    eye_pos: torch.Tensor,
    render_width: int | None = None,
) -> torch.Tensor:
    """Render a single view from an arbitrary eye position (2.5D animation).

    GPU-native camera setup (no CPU-bound create_camera_model).

    Args:
        gaussians: World-space Gaussians.
        f_px: Focal length in pixels (original image space).
        orig_w: Original image width.
        orig_h: Original image height.
        eye_pos: (3,) tensor — camera position in scene units (any device).
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

    screen_w, screen_h = _get_screen_resolution(w_r, h_r)
    f_px_screen = f_px_r * (screen_w / w_r)

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

    # GPU-native camera setup
    depth_focus = _compute_focus_depth_gpu(means)
    look_at = torch.tensor([0.0, 0.0, depth_focus], device=device)
    world_up = torch.tensor([0.0, -1.0, 0.0], device=device)
    eye = eye_pos.to(device)

    ext = _look_at_extrinsics_gpu(eye, look_at, world_up)
    viewmats = ext[None]  # (1, 4, 4)

    K = torch.tensor([
        [f_px_screen, 0, (screen_w - 1) / 2.0],
        [0, f_px_screen, (screen_h - 1) / 2.0],
        [0, 0, 1],
    ], dtype=torch.float32, device=device)
    Ks = K[None]  # (1, 3, 3)

    rendered_colors, rendered_alphas, meta = gsplat.rendering.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=screen_w,
        height=screen_h,
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

    screen_w, screen_h = _get_screen_resolution(orig_w, orig_h)
    f_px_screen = f_px * (screen_w / orig_w)

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

    # GPU-native camera setup (center viewpoint)
    depth_focus = _compute_focus_depth_gpu(means)
    look_at = torch.tensor([0.0, 0.0, depth_focus], device=device)
    world_up = torch.tensor([0.0, -1.0, 0.0], device=device)
    eye = torch.tensor([0.0, 0.0, 0.0], device=device)

    ext = _look_at_extrinsics_gpu(eye, look_at, world_up)
    viewmats = ext[None]  # (1, 4, 4)

    K = torch.tensor([
        [f_px_screen, 0, (screen_w - 1) / 2.0],
        [0, f_px_screen, (screen_h - 1) / 2.0],
        [0, 0, 1],
    ], dtype=torch.float32, device=device)
    Ks = K[None]  # (1, 3, 3)

    rendered_colors, rendered_alphas, meta = gsplat.rendering.rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=screen_w,
        height=screen_h,
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
