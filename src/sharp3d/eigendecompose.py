"""3x3 symmetric matrix eigendecomposition.

Two methods:
1. Analytical (trigonometric) — ~0.07s for 1.18M gaussians, pure GPU
2. SVD fallback — ~0.47s, guaranteed correct

BUG#4 FIX: The analytical method's eigenvectors from cross products have
arbitrary sign (v and -v are both valid eigenvectors). Without sign
enforcement, the rotation matrix may have det=-1 (improper rotation),
which produces garbage quaternions → completely garbled rendering.

Fix: (1) Each eigenvector's largest-absolute-component is forced positive.
     (2) Final det(R) check: if det=-1, flip the smallest-eigenvalue
         eigenvector to ensure a proper rotation (det=+1).
"""

import torch
from .quaternion import quat_from_rotmat_gpu


def analytical_eigen_decompose(cov_matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytical eigendecomposition for 3x3 symmetric PSD matrices.

    Args:
        cov_matrices: (..., 3, 3) symmetric positive semi-definite matrices.
                      Supports arbitrary leading batch dimensions.

    Returns:
        quaternions: (..., 4) in (w, x, y, z) convention.
        singular_values: (..., 3) sqrt of eigenvalues, descending order.
    """
    cov = cov_matrices.detach().to(torch.float32)
    batch_shape = cov.shape[:-2]
    cov = cov.reshape(-1, 3, 3)  # Flatten to (N, 3, 3)
    N = cov.shape[0]

    # Extract symmetric matrix elements
    a = cov[:, 0, 0]
    b = cov[:, 0, 1]
    c = cov[:, 0, 2]
    d = cov[:, 1, 1]
    e = cov[:, 1, 2]
    f = cov[:, 2, 2]

    # === Eigenvalues via trigonometric method for 3x3 symmetric ===
    # Characteristic equation: λ³ - p*λ² + q*λ - r = 0
    p = a + d + f  # trace
    q = a * d + a * f + d * f - b * b - c * c - e * e  # sum of 2x2 minors
    r = (a * d * f + 2 * b * c * e
         - a * e * e - d * c * c - f * b * b)  # determinant

    # Depressed cubic: t³ + pt + q = 0 where λ = t + p/3
    p3 = p / 3.0
    q2 = (p * p - 3.0 * q) / 9.0
    r2 = (2.0 * p * p * p - 9.0 * p * q + 27.0 * r) / 54.0

    # Clamp for numerical safety (acos domain [-1, 1])
    q2 = q2.clamp(min=1e-20)
    cos_arg = (r2 / q2.sqrt()**3).clamp(-1.0, 1.0)
    theta = torch.acos(cos_arg)

    sqrt_q2 = q2.sqrt()
    # Three eigenvalues (descending order after sort)
    lam1 = p3 + 2.0 * sqrt_q2 * torch.cos(theta / 3.0)
    lam2 = p3 + 2.0 * sqrt_q2 * torch.cos((theta - 2.0 * torch.pi) / 3.0)
    lam3 = p3 + 2.0 * sqrt_q2 * torch.cos((theta + 2.0 * torch.pi) / 3.0)

    # Sort descending (required: aligns with SVD convention)
    eigenvalues = torch.stack([lam1, lam2, lam3], dim=-1)
    eigenvalues, sort_idx = eigenvalues.sort(dim=-1, descending=True)
    eigenvalues = eigenvalues.clamp(min=0.0)  # PSD guarantee

    # === Eigenvectors via cross product method ===
    # For each eigenvalue λ, (A - λI) has rank ≤ 2.
    # Cross product of two independent rows gives the null vector (eigenvector).
    eigvecs = torch.zeros(N, 3, 3, device=cov.device, dtype=torch.float32)

    for k in range(3):
        lam = eigenvalues[:, k]
        # A - λI
        m00 = a - lam; m01 = b;     m02 = c
        m10 = b;       m11 = d - lam; m12 = e
        m20 = c;       m21 = e;     m22 = f - lam

        # Cross product of row0 × row1
        v0 = m01 * m12 - m02 * m11
        v1 = m02 * m10 - m00 * m12
        v2 = m00 * m11 - m01 * m10
        norm_sq = v0 * v0 + v1 * v1 + v2 * v2

        # Fallback: row0 × row2 when row0 ∥ row1
        alt0 = m01 * m22 - m02 * m21
        alt1 = m02 * m20 - m00 * m22
        alt2 = m00 * m21 - m01 * m20
        alt_norm_sq = alt0 * alt0 + alt1 * alt1 + alt2 * alt2

        # Fallback: row1 × row2
        alt2_0 = m11 * m22 - m12 * m21
        alt2_1 = m12 * m20 - m10 * m22
        alt2_2 = m10 * m21 - m11 * m20
        alt2_norm_sq = alt2_0 * alt2_0 + alt2_1 * alt2_1 + alt2_2 * alt2_2

        # Pick the cross product with largest norm (most numerically stable)
        use_alt1 = (alt_norm_sq > norm_sq)
        use_alt2 = (alt2_norm_sq > norm_sq) & (alt2_norm_sq > alt_norm_sq)

        v0 = torch.where(use_alt2, alt2_0, torch.where(use_alt1, alt0, v0))
        v1 = torch.where(use_alt2, alt2_1, torch.where(use_alt1, alt1, v1))
        v2 = torch.where(use_alt2, alt2_2, torch.where(use_alt1, alt2, v2))
        norm_sq = torch.where(use_alt2, alt2_norm_sq,
                    torch.where(use_alt1, alt_norm_sq, norm_sq))

        # Normalize (guard against zero norm for degenerate eigenvalues)
        norm = norm_sq.clamp(min=1e-30).sqrt()
        v0 = v0 / norm; v1 = v1 / norm; v2 = v2 / norm

        # BUG#4 FIX Part 1: Sign convention — largest |component| is positive
        max_comp = torch.stack([v0.abs(), v1.abs(), v2.abs()], dim=-1)
        max_idx = max_comp.argmax(dim=-1)
        sign = torch.where(max_idx == 0, v0.sign(),
                 torch.where(max_idx == 1, v1.sign(), v2.sign()))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        v0 = v0 * sign; v1 = v1 * sign; v2 = v2 * sign

        eigvecs[:, 0, k] = v0
        eigvecs[:, 1, k] = v1
        eigvecs[:, 2, k] = v2

    # BUG#4 FIX Part 2: Ensure proper rotation (det = +1)
    # det < 0 means reflection → quaternion would be garbage
    dets = torch.linalg.det(eigvecs)
    flip_mask = dets < 0
    if flip_mask.any():
        eigvecs = eigvecs.clone()
        # Flip the last column (smallest eigenvalue's eigenvector)
        eigvecs[flip_mask, :, 2] *= -1

    # Convert rotation matrices to quaternions
    quaternions = quat_from_rotmat_gpu(eigvecs)
    singular_values = eigenvalues.sqrt()

    # Reshape back to original batch shape
    quaternions = quaternions.reshape(*batch_shape, 4)
    singular_values = singular_values.reshape(*batch_shape, 3)

    return quaternions, singular_values


def svd_decompose(cov_matrices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU SVD decomposition (verified correct, used as reference/fallback).

    Args:
        cov_matrices: (..., 3, 3) symmetric PSD matrices.
                      Supports arbitrary leading batch dimensions.

    Returns:
        quaternions: (..., 4) in (w, x, y, z) convention.
        singular_values: (..., 3) descending order.
    """
    cov = cov_matrices.detach().to(torch.float32)
    batch_shape = cov.shape[:-2]
    cov = cov.reshape(-1, 3, 3)

    rotations, sv_sq, _ = torch.linalg.svd(cov)

    # Ensure proper rotation (det = +1)
    dets = torch.linalg.det(rotations)
    neg_mask = dets < 0
    if neg_mask.any():
        rotations = rotations.clone()
        rotations[neg_mask, :, -1] *= -1

    sv = sv_sq.clamp(min=0).sqrt()
    q = quat_from_rotmat_gpu(rotations)

    # Reshape back to original batch shape
    q = q.reshape(*batch_shape, 4).to(torch.float32)
    sv = sv.reshape(*batch_shape, 3).to(torch.float32)
    return q, sv


def decompose_covariance(cov_matrices: torch.Tensor,
                         method: str = "analytical") -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose covariance matrices into quaternions + singular values.

    Args:
        cov_matrices: (..., 3, 3) symmetric PSD matrices.
        method: "analytical" (fast, 0.07s) or "svd" (reference, 0.47s).

    Returns:
        quaternions: (..., 4), singular_values: (..., 3).
    """
    if method == "analytical":
        return analytical_eigen_decompose(cov_matrices)
    elif method == "svd":
        return svd_decompose(cov_matrices)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'analytical' or 'svd'.")
