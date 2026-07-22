"""Temporal depth stabilization for video conversion.

Eliminates inter-frame flickering caused by SHARP's per-frame monodepth
scale/shift drift. Operates on the z-component of NDC Gaussians between
predict and unproject steps.

Modes:
    off      – passthrough, no stabilization.
    global   – global scale-shift alignment + EMA blend.
    adaptive – per-pixel confidence-weighted EMA (protects moving objects).
"""

import torch


class TemporalStabilizer:
    """Frame-to-frame depth stabilizer for the video conversion loop.

    Usage:
        stab = TemporalStabilizer(mode="adaptive", alpha=0.35, device=device)
        for each frame:
            g_ndc = predictor(img)
            stab.stabilize(g_ndc)   # modifies g_ndc.mean_vectors in-place
            g = fast_unproject(g_ndc, ...)
    """

    def __init__(self, mode: str = "off", alpha: float = 0.35,
                 sigma: float = 0.02, cut_threshold: float = 0.08,
                 device: torch.device | None = None):
        """
        Args:
            mode: "off", "global", or "adaptive".
            alpha: EMA blend factor for the *aligned current* frame.
                   Lower = more smoothing (more temporal coherence, more ghosting).
                   Higher = less smoothing (less ghosting, more residual flicker).
            sigma: Confidence decay for adaptive mode. Controls how quickly
                   per-pixel weight drops off with alignment residual.
                   Smaller = stricter (only very stable pixels get smoothed).
            cut_threshold: Mean absolute residual above which a scene cut is
                           declared and temporal state is reset.
            device: CUDA device for state tensors.
        """
        self.mode = mode
        self.alpha = alpha
        self.sigma = sigma
        self.cut_threshold = cut_threshold
        self._device = device

        # State: previous frame's stabilized z (flattened) and spatial shape.
        self._prev_z: torch.Tensor | None = None
        self._shape: tuple[int, ...] | None = None  # (H, W, L)

    def reset(self) -> None:
        """Clear temporal state (call at video start or after scene cut)."""
        self._prev_z = None
        self._shape = None

    @torch.no_grad()
    def stabilize(self, g_ndc) -> None:
        """Stabilize the z-component of g_ndc.mean_vectors in-place.

        Args:
            g_ndc: Gaussians3D with mean_vectors (N, 3) in NDC space.
        """
        if self.mode == "off":
            return

        z = g_ndc.mean_vectors[:, 2]  # (N,) depth in NDC
        N = z.numel()

        # First frame: just store and return.
        if self._prev_z is None:
            self._prev_z = z.clone()
            self._infer_shape(N)
            return

        # ── Scale-shift alignment ────────────────────────────────────────
        # Solve: s * z_curr + t ≈ z_prev  (least squares, 2x2 normal eq.)
        prev = self._prev_z
        x = z
        y = prev

        sx = x.sum()
        sxx = (x * x).sum()
        sxy = (x * y).sum()
        sy = y.sum()
        n = torch.tensor(float(N), device=z.device, dtype=z.dtype)

        # Normal equations: [[sxx, sx], [sx, n]] @ [s, t] = [sxy, sy]
        det = sxx * n - sx * sx
        if det.abs() < 1e-12:
            # Degenerate (constant depth) — skip alignment.
            self._prev_z = z.clone()
            return

        s = (sxy * n - sx * sy) / det
        t = (sxx * sy - sx * sxy) / det

        z_aligned = s * z + t

        # ── Scene cut detection ──────────────────────────────────────────
        residual = (z_aligned - prev).abs()
        mean_residual = residual.mean()

        if mean_residual > self.cut_threshold:
            # Scene cut: reset state, use raw prediction.
            self._prev_z = z.clone()
            return

        # ── Blend ────────────────────────────────────────────────────────
        if self.mode == "global":
            z_out = self.alpha * z_aligned + (1.0 - self.alpha) * prev
        else:
            # Adaptive: per-pixel confidence weighting.
            # Pixels with low residual (static background) → strong smoothing.
            # Pixels with high residual (moving objects) → weak smoothing.
            confidence = torch.exp(-residual / max(self.sigma, 1e-6))
            # Blend: confidence gates between smoothed and raw.
            z_smooth = self.alpha * z_aligned + (1.0 - self.alpha) * prev
            z_out = confidence * z_smooth + (1.0 - confidence) * z

        # Write back in-place.
        g_ndc.mean_vectors[:, 2] = z_out
        self._prev_z = z_out.clone()

    def _infer_shape(self, N: int) -> None:
        """Infer spatial shape (H, W, L) from total element count."""
        # SHARP: 1536x1536 with num_layers (typically 2).
        # N = H * W * L
        if N == 1536 * 1536 * 2:
            self._shape = (1536, 1536, 2)
        elif N == 1536 * 1536:
            self._shape = (1536, 1536, 1)
        else:
            # Fallback: treat as flat.
            self._shape = (N,)
