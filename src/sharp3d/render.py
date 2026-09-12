"""Batched SBS stereoscopic rendering using gsplat.

BUG#2 FIX: Extrinsics must be world-to-camera (R.T), not camera-to-world (R).
    We use SHARP's camera_model.compute() which correctly handles this via
    create_camera_matrix(inverse=True) → rotation_matrix.transpose(-1, -2).

BUG#3 FIX: SHARP Gaussians store colors in linearRGB. Rendered output must
    be gamma-corrected via linearRGB2sRGB() before display/saving.

BUG#6 FIX: camera_model.compute() creates look_at/world_up on CPU internally.
    eye_pos must be a CPU tensor (no device= argument).

Optimization: Replace GSplatRenderer's Python for-loop (renders L/R separately)
with a single rasterization() call using stacked viewmats [2, 4, 4].
"""

import logging

import torch
from gsplat.rendering import rasterization

from sharp.utils.gaussians import Gaussians3D

logger = logging.getLogger(__name__)

# Sticky failure flag for the HiGS backend. gsplat loads its experimental
# HiGS kernels through a lazy backend that re-raises ONE cached module-level
# exception on every call, and each re-raise pins the caller's stack frames
# (i.e. the rendered tensors). Probe once, remember, stop retrying.
_HIGS_BROKEN = False


def _rasterize_higs(means, quats, scales, opacities, colors,
                    viewmats, Ks, width, height):
    """Render one image per camera with the HiGS inference renderer.

    Returns (C, H, W, 3) linearRGB — same layout as gsplat's rasterization().
    Self-contained under torch.no_grad(): the HiGS entry point refuses to run
    outside inference mode, and relying on every caller to wrap it silently
    tripped the sticky fallback flag instead (observed in testing).
    """
    from gsplat.scene import GaussianInferenceScene
    from gsplat.experimental.render import rasterize_gaussian_inference_scene

    import torch.nn.functional as _F

    if opacities.dim() == 2:
        opacities = (opacities.squeeze(0) if opacities.shape[0] == 1
                     else opacities.squeeze(-1))

    scene = GaussianInferenceScene.from_gaussian_tensors(
        means=means,
        quats=_F.normalize(quats, dim=-1),
        scales=scales,
        opacities=opacities,
        colors=colors,
        sh_degree=None,
        sh_compression="none",
        id="sbs_render",
    )
    frames = []
    with torch.no_grad():
        for i in range(viewmats.shape[0]):
            res = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmats[i], K=Ks[i],
                width=width, height=height,
            )
            # res.frame: [1, H, W, 3]
            frames.append(res.frame[0])
    return torch.stack(frames, dim=0)


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
                             q_focus: float = 0.50) -> float:
    """Compute focus depth entirely on GPU (replaces CPU-bound _compute_depth_quantiles).

    Since screen_extrinsics is always identity, depth = z-coordinate.
    torch.quantile on GPU avoids the 14MB GPU→CPU transfer + CPU sort.

    q_focus: quantile of scene depths to place the convergence (screen) plane.
        Larger value → screen plane further back → more foreground pops out.
        0.50 = 50% of geometry is in front of the screen (pops out),
        50% is behind. Default 0.50 gives strong foreground pop with
        balanced depth distribution.

    BUG#9 FIX: callers pass the raw Gaussians3D.mean_vectors with the batch
    dim intact ([1, N, 3]); means[:, 2] on that tensor selects gaussian #2's
    xyz (shape [1, 3]) instead of every gaussian's z (shape [N]) — the focus
    depth degenerated to ~min_depth_focus. Squeeze the batch dim first.
    """
    if means.ndim == 3:
        means = means[0]
    # torch.quantile rejects half precision — the render path may hand this
    # fp16 gaussians once the FP16 pipeline lands.
    depth_values = means[:, 2].float()  # z-coordinate = depth (identity extr.)
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


# Pinhole intrinsics only depend on (focal, size, device) — constant across
# every frame of a conversion. Without the cache, render_sbs rebuilt and
# re-uploaded this tensor on every frame.
_K_CACHE: dict = {}


def _pinhole_K(f_px: float, w: int, h: int, device: torch.device) -> torch.Tensor:
    key = (round(float(f_px), 3), w, h, device.index)
    K = _K_CACHE.get(key)
    if K is None:
        K = torch.tensor([
            [f_px, 0, (w - 1) / 2.0],
            [0, f_px, (h - 1) / 2.0],
            [0, 0, 1],
        ], dtype=torch.float32, device=device)
        _K_CACHE[key] = K
    return K


def render_sbs(
    gaussians: Gaussians3D,
    f_px: float,
    orig_w: int,
    orig_h: int,
    ipd: float = 0.063,
    convergence: float | None = None,
    render_width: int | None = None,
    ndc_transform: torch.Tensor | None = None,
    renderer: str = "standard",
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Render stereoscopic SBS pair using batched gsplat rasterization.

    GPU-native camera setup: bypasses SHARP's CPU-bound create_camera_model
    (which transfers 1.18M points to CPU + sorts for quantiles = ~240ms).
    Instead computes focus depth and look-at matrices entirely on GPU (~1ms).

    Args:
        gaussians: World-space Gaussians (unprojected). If `ndc_transform` is
            given, these are *NDC-space* Gaussians straight from the predictor.
        f_px: Focal length in pixels (original image space).
        orig_w: Original image width.
        orig_h: Original image height.
        ipd: Inter-pupillary distance in scene units.
        convergence: Convergence distance in scene units (None = auto focus).
        render_width: If set, render each eye at this width (preview mode,
                      faster). None = render at original resolution.
        ndc_transform: Optional (4, 4) NDC→world unprojection matrix. When
            provided, it is folded into the view matrices (viewmat' = V @ U)
            instead of transforming the gaussians — mathematically identical
            (Σ_cam = (V·U) Σ_ndc (V·U)ᵀ = V Σ_world Vᵀ; matrix multiplication
            is associative), but skips the per-frame
            compose-covariance → transform → eigendecompose round-trip over
            1.18M gaussians (tens of ms per frame).
        renderer: "standard" (gsplat batched rasterization) or "higs"
            (fp16 packed inference renderer). HiGS is faster but has no
            alpha/depth channel, so it is skipped for depth output and falls
            back to standard on any backend failure.

    Returns:
        sbs_image: (H, W*2, 3) uint8 tensor (left | right concatenated).
        (single_width, height): Dimensions of each eye's image.
    """
    global _HIGS_BROKEN
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
    elif ndc_transform is not None:
        # World-space z is a linear readout of the NDC means (row 2 of U):
        # z_world = U[2,:3] @ μ_ndc + U[2,3]. Cheaper than unprojecting.
        U_f = ndc_transform.to(device=device, dtype=torch.float32)
        z_world = means @ U_f[2, :3] + U_f[2, 3]
        z_pos = z_world[z_world > 0]
        if z_pos.numel() > 262_144:
            z_pos = z_pos[::16]  # quantile sorts everything — sample instead
        depth_focus = (max(2.0, float(torch.quantile(z_pos, 0.50)))
                       if z_pos.numel() else 2.0)
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

    if ndc_transform is not None:
        # Fold NDC→world into the view matrices; gaussians stay in NDC space.
        U_f = ndc_transform.to(device=device, dtype=torch.float32)
        viewmats = viewmats @ U_f  # (2, 4, 4)

    # Intrinsics for gsplat: (2, 3, 3)
    # Cached: identical every frame of a conversion (f_px_screen, screen size
    # and device are fixed), and rebuilding the tensor costs a host->device
    # transfer plus allocation churn on every frame.
    K = _pinhole_K(f_px_screen, screen_w, screen_h, device)
    Ks = K.unsqueeze(0).expand(2, -1, -1)  # (2, 3, 3)

    # Single batched rasterization call for both eyes (or two HiGS calls).
    rendered_colors = None
    if renderer == "higs" and not _HIGS_BROKEN:
        try:
            rendered_colors = _rasterize_higs(
                means, quats, scales, opacities, colors,
                viewmats, Ks, screen_w, screen_h)
        except Exception as e:  # noqa: BLE001
            _HIGS_BROKEN = True
            logger.warning("HiGS 渲染不可用，回退标准光栅化: %s: %s",
                           type(e).__name__, e)
            # Stop the cached gsplat exception from pinning this frame's
            # stack (and the tensors it references) for the rest of the run.
            e.__traceback__ = None
            rendered_colors = None
    if rendered_colors is None:
        rendered_colors, rendered_alphas, meta = rasterization(
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

    rendered_colors, rendered_alphas, meta = rasterization(
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

    rendered_colors, rendered_alphas, meta = rasterization(
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
    return colorize_depth(depth, alpha)


def colorize_depth(depth: torch.Tensor, alpha: torch.Tensor | None = None,
                   hi_q: float = 0.99) -> torch.Tensor:
    """Colorize a depth map (near=warm, far=cool) with log normalization.

    gsplat's ``RGB+D`` mode returns *alpha-weighted* depth, so it must be
    divided by alpha to recover real depth — but only where something was
    actually rasterized. Where alpha≈0 the quotient explodes to ~1e8, and
    normalizing with a plain ``max()`` then compresses every real depth to the
    near end of the ramp (the whole map turns red). So: mask by alpha, and
    take a robust high quantile instead of the max.

    Args:
        depth: (H, W) raw depth (already divided by alpha if alpha is None).
        alpha: (H, W) accumulated alpha, or None if depth is already normalized.
        hi_q: Upper quantile used as the "far" end of the color ramp.

    Returns:
        (H, W, 3) uint8 tensor; empty background is black.
    """
    if alpha is not None:
        valid = alpha > 0.1
        depth = torch.where(valid, depth / alpha.clamp(min=1e-6),
                            torch.zeros_like(depth))
    else:
        valid = torch.ones_like(depth, dtype=torch.bool)

    d_pos = depth[valid & (depth > 0)]
    if d_pos.numel() == 0:
        return torch.zeros((*depth.shape, 3), dtype=torch.uint8,
                           device=depth.device)

    d_min = d_pos.min()
    # Robust upper bound from a subsample — sorting 16M elements every frame
    # just to pick a color range is not worth it.
    step = max(1, d_pos.numel() // 200_000)
    d_max = torch.quantile(d_pos[::step].float(), hi_q)
    if d_max <= d_min * (1.0 + 1e-6):
        d_max = d_pos.max()

    if d_max > d_min:
        log_max = torch.log(d_max / d_min + 1e-6)
        depth_log = torch.log(depth.clamp(min=d_min) / d_min + 1e-6)
        depth_norm = (depth_log / log_max).clamp(0, 1)
    else:
        depth_norm = torch.zeros_like(depth)

    # Simple turbo-like colormap via interpolation
    r = (1.0 - depth_norm).clamp(0, 1)
    g = (1.0 - (depth_norm - 0.5).abs() * 2).clamp(0, 1)
    b = depth_norm.clamp(0, 1)
    depth_rgb = torch.stack([r, g, b], dim=-1)

    # Empty background carries no depth — paint it black instead of letting it
    # fall to the warm/near end of the ramp.
    depth_rgb = depth_rgb * valid.unsqueeze(-1)
    return (depth_rgb * 255).to(torch.uint8)
