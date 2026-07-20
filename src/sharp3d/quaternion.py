"""GPU quaternion conversion using Shepperd's method.

Replaces scipy Rotation.from_matrix() which runs on CPU (2.77s for 1.18M gaussians).
GPU Shepperd: 0.003s — 884x speedup.
"""

import torch


def quat_from_rotmat_gpu(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions (w, x, y, z) on GPU.

    Uses Shepperd's method for numerical stability.
    Input shape: (..., 3, 3). Output shape: (..., 4).
    """
    r00 = R[..., 0, 0]; r01 = R[..., 0, 1]; r02 = R[..., 0, 2]
    r10 = R[..., 1, 0]; r11 = R[..., 1, 1]; r12 = R[..., 1, 2]
    r20 = R[..., 2, 0]; r21 = R[..., 2, 1]; r22 = R[..., 2, 2]

    t = r00 + r11 + r22

    s1 = (t + 1.0).clamp(min=1e-10).sqrt() * 2
    s2 = (1.0 + r00 - r11 - r22).clamp(min=1e-10).sqrt() * 2
    s3 = (1.0 + r11 - r00 - r22).clamp(min=1e-10).sqrt() * 2
    s4 = (1.0 + r22 - r00 - r11).clamp(min=1e-10).sqrt() * 2

    c1 = t > 0
    c2 = (r00 > r11) & (r00 > r22) & ~c1
    c3 = (r11 > r22) & ~c1 & ~c2

    w = torch.where(c1, 0.25 * s1,
        torch.where(c2, (r21 - r12) / s2,
        torch.where(c3, (r02 - r20) / s3, (r10 - r01) / s4)))
    x = torch.where(c1, (r21 - r12) / s1,
        torch.where(c2, 0.25 * s2,
        torch.where(c3, (r01 + r10) / s3, (r02 + r20) / s4)))
    y = torch.where(c1, (r02 - r20) / s1,
        torch.where(c2, (r01 + r10) / s2,
        torch.where(c3, 0.25 * s3, (r12 + r21) / s4)))
    z = torch.where(c1, (r10 - r01) / s1,
        torch.where(c2, (r02 + r20) / s2,
        torch.where(c3, (r12 + r21) / s3, 0.25 * s4)))

    q = torch.stack([w, x, y, z], dim=-1)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    # Canonical form: w >= 0
    return torch.where(q[..., :1] < 0, -q, q)
