"""Unprojection: transform Gaussians from NDC to world space.

BUG#1 (CRITICAL): Must transform covariance matrices to world space via
    cov_world = T_linear @ cov_ndc @ T_linear.T
before decomposing. Skipping this step leaves Gaussian orientations/scales
in NDC space → blurry rendering (not completely wrong, just very blurry).

BUG#5: disparity_factor must be dtype=float32. Python float creates f64
tensor → "Input type (double) and bias type (Half)" error under FP16 autocast.
"""

import torch
import torch.nn.functional as F

from sharp.utils.gaussians import (
    Gaussians3D,
    get_unprojection_matrix,
    compose_covariance_matrices,
)

from .eigendecompose import decompose_covariance

# SHARP architectural constraint: SPN 3-level pyramid requires x2=384²
# internal_resolution() = patch_size * 4 = 384 * 4 = 1536
INTERNAL_SHAPE = (1536, 1536)


def prepare_input(image_np, f_px: float, device: torch.device):
    """Prepare image for SHARP predictor.

    Args:
        image_np: (H, W, 3) uint8 numpy array.
        f_px: Focal length in pixels.
        device: Target device.

    Returns:
        img_resized: (1, 3, 1536, 1536) float tensor [0, 1].
        disparity_factor: (1,) float32 tensor.  # BUG#5: must be float32!
        intrinsics_resized: (4, 4) intrinsics scaled to 1536x1536.
        orig_size: (W, H) original image size.
    """
    img = torch.from_numpy(image_np.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, h, w = img.shape

    # BUG#5 FIX: explicit dtype=float32 (Python float → f64 → FP16 autocast error)
    disparity_factor = torch.tensor([f_px / w], device=device, dtype=torch.float32)

    img_resized = F.interpolate(
        img[None], size=INTERNAL_SHAPE, mode="bilinear", align_corners=True
    )

    intrinsics = torch.tensor([
        [f_px, 0, w / 2, 0],
        [0, f_px, h / 2, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32, device=device)

    # Scale intrinsics to internal resolution
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= INTERNAL_SHAPE[0] / w
    intrinsics_resized[1] *= INTERNAL_SHAPE[1] / h

    return img_resized, disparity_factor, intrinsics_resized, (w, h)


def fast_unproject(
    g_ndc: Gaussians3D,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
    decompose_method: str = "analytical",
) -> Gaussians3D:
    """Unproject Gaussians from NDC to world space.

    Args:
        g_ndc: Gaussians in NDC space (from predictor).
        extrinsics: (4, 4) camera extrinsics.
        intrinsics: (4, 4) camera intrinsics (scaled to internal resolution).
        image_shape: (W, H) of internal resolution.
        decompose_method: "analytical" or "svd".

    Returns:
        Gaussians3D in world space.
    """
    unproj = get_unprojection_matrix(extrinsics, intrinsics, image_shape)
    tl = unproj[:3, :3]
    to = unproj[:3, 3]

    # Transform means
    mean_vectors = g_ndc.mean_vectors @ tl.T + to

    # BUG#1 FIX: Transform covariance to world space before decomposing!
    # Without this: Gaussian orientations/scales stay in NDC → blurry output.
    # Reference: SHARP apply_transform() in utils/gaussians.py line 120-122.
    cov = compose_covariance_matrices(g_ndc.quaternions, g_ndc.singular_values)
    cov = tl @ cov @ tl.transpose(-1, -2)

    # Decompose back to quaternion + singular values
    q, sv = decompose_covariance(cov, method=decompose_method)

    return Gaussians3D(
        mean_vectors=mean_vectors,
        singular_values=sv,
        quaternions=q,
        colors=g_ndc.colors,
        opacities=g_ndc.opacities,
    )
