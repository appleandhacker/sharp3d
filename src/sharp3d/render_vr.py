"""VR stereo rendering — cubemap 6-face × 2-eye + equirectangular assembly.

Supports two render backends:
- HiGS: fp16 packed scene, sequential single-camera calls (faster, less VRAM)
- Standard: gsplat rasterization() batched (supports depth output)
"""

from __future__ import annotations

import torch
from torch import Tensor

from .projection import (
    cubemap_to_equirect,
    cubemap_to_equirect180,
    get_cubemap_cameras,
)
from .render import linearRGB2sRGB


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

    # Eye offsets
    left_offset = torch.tensor([-ipd / 2, 0.0, 0.0], device=device)
    right_offset = torch.tensor([ipd / 2, 0.0, 0.0], device=device)

    if renderer == "higs":
        try:
            left_faces = _render_cubemap_higs(gaussians, left_offset, face_size, device)
            right_faces = _render_cubemap_higs(gaussians, right_offset, face_size, device)
        except Exception:
            # HiGS unavailable (JIT build failure) — fallback to standard
            left_faces = _render_cubemap_standard(gaussians, left_offset, face_size, device)
            right_faces = _render_cubemap_standard(gaussians, right_offset, face_size, device)
    else:
        left_faces = _render_cubemap_standard(gaussians, left_offset, face_size, device)
        right_faces = _render_cubemap_standard(gaussians, right_offset, face_size, device)

    # Assemble equirectangular
    map_fn = cubemap_to_equirect180 if output_projection == "equirect180" else cubemap_to_equirect
    left_equirect = map_fn(left_faces, out_w, out_h)   # [H, W, 3] linearRGB
    right_equirect = map_fn(right_faces, out_w, out_h)

    # Gamma correction
    left_srgb = linearRGB2sRGB(left_equirect)
    right_srgb = linearRGB2sRGB(right_equirect)

    # Quantize to uint8
    left_u8 = (left_srgb * 255).clamp(0, 255).to(torch.uint8)
    right_u8 = (right_srgb * 255).clamp(0, 255).to(torch.uint8)

    # Pack stereo layout
    if stereo_layout == "sbs":
        return torch.cat([left_u8, right_u8], dim=1)  # [H, W*2, 3]
    else:  # tb
        return torch.cat([left_u8, right_u8], dim=0)  # [H*2, W, 3]


def _render_cubemap_higs(
    gaussians,
    eye_offset: Tensor,
    face_size: int,
    device: torch.device,
) -> Tensor:
    """Render 6 cubemap faces using HiGS inference renderer.

    Returns: [6, 3, face_size, face_size] linearRGB.
    """
    from gsplat.scene import GaussianInferenceScene
    from gsplat.experimental.render import rasterize_gaussian_inference_scene

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

    viewmats, Ks = get_cubemap_cameras(face_size, device, eye_offset)

    faces = torch.zeros(6, 3, face_size, face_size, device=device)
    with torch.no_grad():
        for i in range(6):
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

    viewmats, Ks = get_cubemap_cameras(face_size, device, eye_offset)

    with torch.no_grad():
        rendered, alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=face_size,
            height=face_size,
            render_mode="RGB",
            rasterize_mode="classic",
        )
    # rendered: [6, H, W, 3]
    return rendered.permute(0, 3, 1, 2)  # [6, 3, H, W]
