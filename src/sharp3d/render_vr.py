"""VR stereo rendering — cubemap 6-face × 2-eye + equirectangular assembly.

Supports two render backends:
- HiGS: fp16 packed scene, sequential single-camera calls (faster, less VRAM)
- Standard: gsplat rasterization() batched (supports depth output)
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from .projection import (
    cubemap_to_equirect,
    cubemap_to_equirect180,
    get_cubemap_cameras,
)
from .render import linearRGB2sRGB

logger = logging.getLogger(__name__)


# Camera matrices depend only on (face_size, eye offset, device) — cache them
# across frames so the video loop doesn't rebuild 6 look-at matrices per eye
# per frame. Bounded: per-frame floating offsets used to mint a new entry per
# frame (6 look-at matrices each) and grew without limit on long videos.
_CAM_CACHE: dict = {}
_CAM_CACHE_MAX = 8

# Sticky HiGS failure flag. When the gsplat_scene_cuda extension is not
# importable (e.g. no MSVC for the JIT build), gsplat's lazy backend re-raises
# ONE cached module-level ImportError on every call. Each re-raise appends the
# current stack frames to that immortal exception's __traceback__ chain, and
# those frames pin this module's locals — including the freshly rendered
# 2×~300MB cubemap faces — leaking ~750MB of VRAM per video frame. Probe once,
# remember the failure, and never re-enter the raising path.
_HIGS_BROKEN = False


def _cameras_cached(face_size: int, device, eye_offset: Tensor,
                    offset_key: float):
    # Quantize the offset before it becomes a key: raw float keys made every
    # per-frame offset variation a new cache entry (6 look-at matrices each,
    # retained forever), and fp representation noise alone could mint
    # duplicates. 1e-3 scene units is ~1.6% of the default IPD (0.063) — far
    # below any visible parallax change.
    offset_q = round(float(offset_key), 3)
    key = (face_size, device.index, offset_q)
    cams = _CAM_CACHE.get(key)
    if cams is None:
        # Rebuild the eye offset at the quantized value so a cache hit and a
        # fresh build produce identical geometry. Every caller passes
        # eye_offset = [offset_x, 0, 0] (left/right eye at ±ipd/2), so only
        # the x component carries information; keep the others verbatim.
        offset_vec = eye_offset.detach().clone()
        offset_vec[0] = offset_q
        cams = get_cubemap_cameras(face_size, device, offset_vec)
        if len(_CAM_CACHE) >= _CAM_CACHE_MAX:
            _CAM_CACHE.pop(next(iter(_CAM_CACHE)))  # FIFO eviction
        _CAM_CACHE[key] = cams
    return cams


def render_vr_stereo(
    gaussians,  # Gaussians3D
    ipd: float = 0.063,
    face_size: int = 1024,
    out_w: int = 4096,
    out_h: int = 4096,
    output_projection: str = "equirect180",
    stereo_layout: str = "sbs",
    renderer: str = "higs",
    device: torch.device | None = None,
    progress_cb=None,  # callable(step, total)
) -> Tensor:
    """Render stereo VR equirectangular image from world-space Gaussians.

    Args:
        gaussians: Gaussians3D in world space (means, quats, scales, opacities, colors).
        ipd: inter-pupillary distance in scene units.
        face_size: cubemap face resolution for rendering.
        out_w: output equirectangular width per eye.
        out_h: output equirectangular height per eye.
        output_projection: "equirect180" or "equirect360".
        stereo_layout: "sbs" (side-by-side) or "tb" (top-bottom).
        renderer: "higs" or "standard".
        device: CUDA device (defaults to gaussians device).

    Returns:
        output: uint8 tensor.
            SBS: [out_h, out_w*2, 3]
            TB:  [out_h*2, out_w, 3]
    """
    if device is None:
        device = gaussians.mean_vectors.device
    global _HIGS_BROKEN

    # For 180° output, skip back face (-Z, index 5) — never sampled
    skip_back = (output_projection == "equirect180")

    # Eye offsets
    left_offset = torch.tensor([-ipd / 2, 0.0, 0.0], device=device)
    right_offset = torch.tensor([ipd / 2, 0.0, 0.0], device=device)

    left_faces = None
    right_faces = None
    if renderer == "higs" and not _HIGS_BROKEN:
        try:
            # Build scene once, reuse for both eyes (avoid redundant fp16 packing)
            from gsplat.scene import GaussianInferenceScene
            means = gaussians.mean_vectors
            quats = gaussians.quaternions
            scales = gaussians.singular_values
            opacities = gaussians.opacities
            colors = gaussians.colors
            if means.dim() == 3:
                means = means.squeeze(0)
                quats = quats.squeeze(0)
                scales = scales.squeeze(0)
                colors = colors.squeeze(0)
            if opacities.dim() == 2:
                opacities = opacities.squeeze(0) if opacities.shape[0] == 1 else opacities.squeeze(-1)
            import torch.nn.functional as _F
            higs_scene = GaussianInferenceScene.from_gaussian_tensors(
                means=means, quats=_F.normalize(quats, dim=-1),
                scales=scales, opacities=opacities, colors=colors,
                sh_degree=None, sh_compression="none", id="vr_render",
            )
            left_faces = _render_cubemap_higs(gaussians, left_offset, face_size,
                                              device, skip_back=skip_back,
                                              scene=higs_scene,
                                              offset_x=-ipd / 2)
            if progress_cb:
                progress_cb(1, 6)
            right_faces = _render_cubemap_higs(gaussians, right_offset, face_size,
                                               device, skip_back=skip_back,
                                               scene=higs_scene,
                                               offset_x=ipd / 2)
            if progress_cb:
                progress_cb(2, 6)
        except Exception as e:  # noqa: BLE001
            # HiGS unavailable (JIT build failure / missing library) — but
            # also any shape/arg mismatch from gsplat's *experimental* HiGS
            # API. It has to be a broad catch: a TypeError or AttributeError
            # escaping here used to abort the whole conversion instead of
            # falling back to the standard rasterizer, and the failure is
            # environment-dependent (which is why tools/build_higs_prebuilt.py
            # exists at all).
            _HIGS_BROKEN = True  # don't re-enter the raising path every frame
            logger.warning("HiGS 渲染不可用，回退标准光栅化: %s: %s",
                           type(e).__name__, e)
            # gsplat's lazy backend re-raises a cached module-level exception;
            # detach its traceback so it stops pinning our stack frames (and
            # the tensors they reference) forever.
            e.__traceback__ = None
            left_faces = None
            right_faces = None

    if left_faces is None:
        left_faces = _render_cubemap_standard(gaussians, left_offset, face_size,
                                              device, skip_back=skip_back,
                                              offset_x=-ipd / 2)
        if progress_cb:
            progress_cb(1, 6)
        right_faces = _render_cubemap_standard(gaussians, right_offset, face_size,
                                               device, skip_back=skip_back,
                                               offset_x=ipd / 2)
        if progress_cb:
            progress_cb(2, 6)

    # Assemble equirectangular
    map_fn = cubemap_to_equirect180 if output_projection == "equirect180" else cubemap_to_equirect
    left_equirect = map_fn(left_faces, out_w, out_h)   # [H, W, 3] linearRGB
    if progress_cb:
        progress_cb(3, 6)
    right_equirect = map_fn(right_faces, out_w, out_h)
    if progress_cb:
        progress_cb(4, 6)

    # Gamma correction (clamp negatives from rasterizer numerical error)
    left_srgb = linearRGB2sRGB(left_equirect.clamp(min=0))
    right_srgb = linearRGB2sRGB(right_equirect.clamp(min=0))

    # Quantize to uint8
    left_u8 = (left_srgb * 255).clamp(0, 255).to(torch.uint8)
    right_u8 = (right_srgb * 255).clamp(0, 255).to(torch.uint8)
    if progress_cb:
        progress_cb(5, 6)

    # Pack stereo layout
    if stereo_layout == "sbs":
        result = torch.cat([left_u8, right_u8], dim=1)  # [H, W*2, 3]
    else:  # tb
        result = torch.cat([left_u8, right_u8], dim=0)  # [H*2, W, 3]
    if progress_cb:
        progress_cb(6, 6)
    return result


def _render_cubemap_higs(
    gaussians,
    eye_offset: Tensor,
    face_size: int,
    device: torch.device,
    skip_back: bool = False,
    scene=None,
    offset_x: float = 0.0,
) -> Tensor:
    """Render 6 cubemap faces using HiGS inference renderer.

    Args:
        scene: Pre-built GaussianInferenceScene (skip packing if provided).

    Returns: [6, 3, face_size, face_size] linearRGB.
    """
    from gsplat.scene import GaussianInferenceScene
    from gsplat.experimental.render import rasterize_gaussian_inference_scene

    if scene is None:
        # Pack scene (fp16)
        means = gaussians.mean_vectors
        quats = gaussians.quaternions
        scales = gaussians.singular_values
        opacities = gaussians.opacities
        colors = gaussians.colors  # [N, 3] linearRGB

        # Squeeze batch dim if present ([1, N, ...] → [N, ...])
        if means.dim() == 3:
            means = means.squeeze(0)
            quats = quats.squeeze(0)
            scales = scales.squeeze(0)
            colors = colors.squeeze(0)
        if opacities.dim() == 2:
            opacities = opacities.squeeze(0) if opacities.shape[0] == 1 else opacities.squeeze(-1)

        scene = GaussianInferenceScene.from_gaussian_tensors(
        means=means,
        quats=torch.nn.functional.normalize(quats, dim=-1),
        scales=scales,
        opacities=opacities,
        colors=colors,
        sh_degree=None,
        sh_compression="none",
        id="vr_render",
    )

    viewmats, Ks = _cameras_cached(face_size, device, eye_offset, offset_x)

    faces = torch.zeros(6, 3, face_size, face_size, device=device)
    n_render = 5 if skip_back else 6
    with torch.no_grad():
        for i in range(n_render):
            result = rasterize_gaussian_inference_scene(
                scene,
                viewmat=viewmats[i],
                K=Ks[i],
                width=face_size,
                height=face_size,
            )
            # result.frame: [1, H, W, 3]
            faces[i] = result.frame[0].permute(2, 0, 1)  # [3, H, W]

    return faces


def _render_cubemap_standard(
    gaussians,
    eye_offset: Tensor,
    face_size: int,
    device: torch.device,
    skip_back: bool = False,
    offset_x: float = 0.0,
) -> Tensor:
    """Render 6 cubemap faces using standard gsplat batched rasterization.

    Returns: [6, 3, face_size, face_size] linearRGB.
    """
    from gsplat.rendering import rasterization

    means = gaussians.mean_vectors
    quats = gaussians.quaternions
    scales = gaussians.singular_values
    opacities = gaussians.opacities
    colors = gaussians.colors

    # Squeeze batch dim if present ([1, N, ...] → [N, ...])
    if means.dim() == 3:
        means = means.squeeze(0)
        quats = quats.squeeze(0)
        scales = scales.squeeze(0)
        colors = colors.squeeze(0)
    if opacities.dim() == 2:
        opacities = opacities.squeeze(0) if opacities.shape[0] == 1 else opacities.squeeze(-1)

    viewmats, Ks = _cameras_cached(face_size, device, eye_offset, offset_x)

    # Skip back face (-Z) for 180° output
    n_render = 5 if skip_back else 6
    with torch.no_grad():
        rendered, alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats[:n_render],
            Ks=Ks[:n_render],
            width=face_size,
            height=face_size,
            render_mode="RGB",
            rasterize_mode="classic",
        )
    # rendered: [n_render, H, W, 3]
    faces = rendered.permute(0, 3, 1, 2)  # [n_render, 3, H, W]
    if skip_back:
        # Pad with zero back face
        zero_face = torch.zeros(1, 3, face_size, face_size, device=device)
        faces = torch.cat([faces, zero_face], dim=0)  # [6, 3, H, W]
    return faces
