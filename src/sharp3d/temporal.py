"""Temporal depth stabilization for video conversion.

Eliminates inter-frame flickering caused by SHARP's per-frame monodepth
scale/shift drift. Operates on the z-component of NDC Gaussians between
predict and unproject steps.

Modes:
    off      – passthrough, no stabilization.
    global   – global scale-shift alignment + EMA blend.
    adaptive – per-pixel confidence-weighted EMA (protects moving objects).
    flow     – optical flow warp + occlusion-aware blend (best quality).
"""

import torch
import torch.nn.functional as F


class KalmanScalar:
    """Constant-velocity Kalman filter for a scalar signal.

    Smooths a noisy per-frame measurement (e.g., convergence depth) while
    tracking genuine trends (camera dolly) with minimal lag.

    State: [x, v] — position and velocity.
    Measurement: z = x + noise.

    Usage:
        kf = KalmanScalar(q_pos=0.01, q_vel=0.005, r=0.02)
        for each frame:
            smoothed = kf.update(measured_value)
    """

    def __init__(self, q_pos: float = 0.01, q_vel: float = 0.005,
                 r: float = 0.02):
        """
        Args:
            q_pos: Process noise std for position (how fast true value can change).
            q_vel: Process noise std for velocity (how fast trend can accelerate).
            r: Measurement noise std (per-frame quantile jitter).
        """
        self._q_pos = q_pos ** 2
        self._q_vel = q_vel ** 2
        self._r = r ** 2
        # State: [x, v], covariance 2x2
        self._x: float | None = None
        self._v: float = 0.0
        self._p = [[1.0, 0.0], [0.0, 1.0]]  # initial uncertainty

    def reset(self) -> None:
        self._x = None
        self._v = 0.0
        self._p = [[1.0, 0.0], [0.0, 1.0]]

    def update(self, z: float) -> float:
        """Feed a new measurement, return the filtered estimate."""
        if self._x is None:
            # First measurement: initialize state directly.
            self._x = z
            self._v = 0.0
            self._p = [[self._r, 0.0], [0.0, 1.0]]
            return z

        # ── Predict ──────────────────────────────────────────────────────
        x_pred = self._x + self._v
        v_pred = self._v
        # P_pred = F @ P @ F.T + Q, where F = [[1,1],[0,1]]
        p = self._p
        p00 = p[0][0] + p[0][1] + p[1][0] + p[1][1] + self._q_pos
        p01 = p[0][1] + p[1][1]
        p10 = p[1][0] + p[1][1]
        p11 = p[1][1] + self._q_vel

        # ── Update ───────────────────────────────────────────────────────
        # Innovation: y = z - H @ x_pred, H = [1, 0]
        innov = z - x_pred
        # S = H @ P_pred @ H.T + R = p00 + R
        s = p00 + self._r
        # K = P_pred @ H.T / S = [p00, p10] / s
        k0 = p00 / s
        k1 = p10 / s

        self._x = x_pred + k0 * innov
        self._v = v_pred + k1 * innov

        # P = (I - K @ H) @ P_pred
        self._p = [
            [(1 - k0) * p00, (1 - k0) * p01],
            [p10 - k1 * p00, p11 - k1 * p01],
        ]

        return self._x


class TemporalStabilizer:
    """Frame-to-frame depth stabilizer for the video conversion loop.

    Usage:
        stab = TemporalStabilizer(mode="adaptive", device=device)
        for each frame:
            g_ndc = predictor(img)
            stab.stabilize(g_ndc, img=img_tensor)  # img needed for flow mode
            g = fast_unproject(g_ndc, ...)
    """

    def __init__(self, mode: str = "off", alpha: float = 0.35,
                 sigma: float = 0.02, cut_threshold: float = 0.08,
                 flow_resolution: int = 384,
                 device: torch.device | None = None):
        """
        Args:
            mode: "off", "global", "adaptive", or "flow".
            alpha: EMA blend factor for the *aligned current* frame.
                   Lower = more smoothing (more temporal coherence, more ghosting).
                   Higher = less smoothing (less ghosting, more residual flicker).
            sigma: Confidence decay for adaptive mode. Controls how quickly
                   per-pixel weight drops off with alignment residual.
                   Smaller = stricter (only very stable pixels get smoothed).
            cut_threshold: Mean absolute residual above which a scene cut is
                           declared and temporal state is reset.
            flow_resolution: Internal resolution for RAFT flow estimation.
                             Lower = faster but less accurate. 384 is a good
                             balance (~15ms on RTX 5070 Ti).
            device: CUDA device for state tensors.
        """
        self.mode = mode
        self.alpha = alpha
        self.sigma = sigma
        self.cut_threshold = cut_threshold
        self.flow_resolution = flow_resolution
        self._device = device

        # State: previous frame's stabilized z (flattened) and spatial shape.
        self._prev_z: torch.Tensor | None = None
        self._shape: tuple[int, ...] | None = None  # (L, H, W)

        # Attribute smoothing state (opacity + scale).
        self._prev_opacities: torch.Tensor | None = None
        self._prev_scales: torch.Tensor | None = None
        self._attr_alpha = 0.4  # EMA factor for attributes (higher = less smoothing)

        # Flow mode state.
        self._prev_img: torch.Tensor | None = None  # (1, 3, flow_res, flow_res)
        self._flow_model = None

    def reset(self) -> None:
        """Clear temporal state (call at video start or after scene cut)."""
        self._prev_z = None
        self._shape = None
        self._prev_img = None
        self._prev_opacities = None
        self._prev_scales = None

    def _ensure_flow_model(self) -> None:
        """Lazy-load RAFT model on first use."""
        if self._flow_model is not None:
            return
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        weights = Raft_Large_Weights.DEFAULT
        self._flow_model = raft_large(weights=weights).to(self._device).eval()
        self._flow_transforms = weights.transforms()

    @torch.no_grad()
    def stabilize(self, g_ndc, img: torch.Tensor | None = None) -> None:
        """Stabilize g_ndc attributes in-place (z, opacity, scale).

        Args:
            g_ndc: Gaussians3D with mean_vectors (N, 3) in NDC space.
            img: (1, 3, H, W) float tensor [0,1] — required for flow mode.
        """
        if self.mode == "off":
            return

        if self.mode == "flow":
            scene_cut = self._stabilize_flow(g_ndc, img)
        else:
            scene_cut = self._stabilize_ema(g_ndc)

        # ── Attribute smoothing (opacity + scale) ────────────────────────
        # Gaussians have fixed grid correspondence between frames (same pixel
        # × layer index), so per-index EMA directly reduces edge flickering.
        if scene_cut:
            # Reset attribute state on scene cut to avoid 1-frame ghosting.
            self._prev_opacities = None
            self._prev_scales = None
        self._smooth_attributes(g_ndc)

    def _smooth_attributes(self, g_ndc) -> None:
        """Recursive EMA-smooth opacities and singular_values."""
        a = self._attr_alpha

        # Opacities: (N, 1) or (N,)
        opac = g_ndc.opacities.float()
        if self._prev_opacities is not None:
            smoothed = a * opac + (1 - a) * self._prev_opacities
            g_ndc.opacities = smoothed.to(g_ndc.opacities.dtype)
            self._prev_opacities = smoothed  # store smoothed (recursive EMA)
        else:
            self._prev_opacities = opac

        # Scale (singular_values): (N, 3)
        scales = g_ndc.singular_values.float()
        if self._prev_scales is not None:
            smoothed = a * scales + (1 - a) * self._prev_scales
            g_ndc.singular_values = smoothed.to(g_ndc.singular_values.dtype)
            self._prev_scales = smoothed  # store smoothed (recursive EMA)
        else:
            self._prev_scales = scales

    # ─── EMA-based methods (global / adaptive) ────────────────────────────

    def _stabilize_ema(self, g_ndc) -> bool:
        """Global or adaptive EMA stabilization. Returns True on scene cut."""
        # Force float32 — N can exceed FP16 max (65504), causing inf/nan.
        z = g_ndc.mean_vectors[:, 2].float()  # (N,) depth in NDC
        N = z.numel()

        # First frame: just store and return.
        if self._prev_z is None:
            self._prev_z = z.clone()
            self._infer_shape(N)
            return True  # treat first frame as cut (reset attributes)

        # ── Scale-shift alignment ────────────────────────────────────────
        prev = self._prev_z
        x = z
        y = prev

        sx = x.sum()
        sxx = (x * x).sum()
        sxy = (x * y).sum()
        sy = y.sum()
        n = torch.tensor(float(N), device=z.device, dtype=torch.float32)

        det = sxx * n - sx * sx
        if not torch.isfinite(det) or det.abs() < 1e-12:
            self._prev_z = z.clone()
            return True

        s = (sxy * n - sx * sy) / det
        t = (sxx * sy - sx * sxy) / det
        z_aligned = s * z + t

        # ── Scene cut detection ──────────────────────────────────────────
        residual = (z_aligned - prev).abs()
        mean_residual = residual.mean()

        if not torch.isfinite(mean_residual) or mean_residual > self.cut_threshold:
            self._prev_z = z.clone()
            return True

        # ── Blend ────────────────────────────────────────────────────────
        if self.mode == "global":
            z_out = self.alpha * z_aligned + (1.0 - self.alpha) * prev
        else:
            # Adaptive: per-pixel confidence weighting.
            confidence = torch.exp(-residual / max(self.sigma, 1e-6))
            z_smooth = self.alpha * z_aligned + (1.0 - self.alpha) * prev
            z_out = confidence * z_smooth + (1.0 - confidence) * z

        # Write back in original dtype; keep prev in float32.
        g_ndc.mean_vectors[:, 2] = z_out.to(g_ndc.mean_vectors.dtype)
        self._prev_z = z_out
        return False

    # ─── Optical flow warp method ─────────────────────────────────────────

    def _stabilize_flow(self, g_ndc, img: torch.Tensor | None) -> bool:
        """Flow-based stabilization with occlusion-aware blending. Returns True on scene cut."""
        if img is None:
            # Fallback to global EMA if no image provided.
            return self._stabilize_ema(g_ndc)

        z = g_ndc.mean_vectors[:, 2].float()  # (N,) — float32 to avoid FP16 overflow
        N = z.numel()

        # Prepare current frame at flow resolution.
        curr_img = F.interpolate(img, size=(self.flow_resolution,
                                            self.flow_resolution),
                                 mode="bilinear", align_corners=False)

        # First frame: store and return.
        if self._prev_z is None or self._prev_img is None:
            self._prev_z = z.clone()
            self._prev_img = curr_img
            self._infer_shape(N)
            return True

        self._ensure_flow_model()

        # ── Compute bidirectional flow ───────────────────────────────────
        # RAFT expects [0, 255] range.
        prev_255 = self._prev_img * 255.0
        curr_255 = curr_img * 255.0

        flow_fwd = self._flow_model(prev_255, curr_255)[-1]   # (1, 2, Hf, Wf)
        flow_bwd = self._flow_model(curr_255, prev_255)[-1]   # (1, 2, Hf, Wf)

        # ── Occlusion detection (forward-backward consistency) ───────────
        # Warp backward flow to forward frame's coordinate system.
        flow_bwd_warped = self._warp_flow(flow_bwd, flow_fwd)
        cycle_err = (flow_fwd + flow_bwd_warped).norm(dim=1, keepdim=True)
        # Also check if warped coordinates fall outside the image.
        occ_mask = cycle_err > 2.0  # (1, 1, Hf, Wf) — True = occluded

        # ── Upsample flow + occlusion to full z resolution ───────────────
        L, H, W = self._shape  # e.g. (2, 1536, 1536)
        scale_h = H / self.flow_resolution
        scale_w = W / self.flow_resolution

        # Scale flow values to full resolution.
        flow_full = F.interpolate(flow_fwd, size=(H, W), mode="bilinear",
                                  align_corners=False)
        flow_full[:, 0] *= scale_w  # x displacement
        flow_full[:, 1] *= scale_h  # y displacement

        occ_full = F.interpolate(occ_mask.float(), size=(H, W),
                                 mode="nearest").bool()  # (1, 1, H, W)

        # ── Warp previous z using flow ───────────────────────────────────
        prev_z_spatial = self._prev_z.reshape(L, H, W)  # (L, H, W)
        z_spatial = z.reshape(L, H, W)

        # Build sampling grid.
        gy, gx = torch.meshgrid(
            torch.arange(H, device=z.device, dtype=torch.float32),
            torch.arange(W, device=z.device, dtype=torch.float32),
            indexing="ij",
        )
        # flow_full is (1, 2, H, W): channel 0 = dx, channel 1 = dy.
        sample_x = gx[None] + flow_full[:, 0]  # (1, H, W)
        sample_y = gy[None] + flow_full[:, 1]  # (1, H, W)

        # Normalize to [-1, 1] for grid_sample.
        sample_x = 2.0 * sample_x / (W - 1) - 1.0
        sample_y = 2.0 * sample_y / (H - 1) - 1.0
        grid = torch.stack([sample_x, sample_y], dim=-1)  # (1, H, W, 2)

        # Warp each layer.
        warped_z = torch.empty_like(z_spatial)
        for layer in range(L):
            src = prev_z_spatial[layer].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            warped = F.grid_sample(src, grid, mode="bilinear",
                                   padding_mode="border", align_corners=True)
            warped_z[layer] = warped.squeeze(0).squeeze(0)

        # ── Scale-shift alignment on warped result ───────────────────────
        # Align current z to warped prev (removes residual global drift).
        # valid mask is per-pixel (H, W), expand to (L*H*W) for layer-major z.
        valid = ~occ_full.squeeze(0).squeeze(0)  # (H, W)
        valid_flat = valid.reshape(-1).repeat(L)  # (L*H*W,)

        z_flat = z_spatial.reshape(-1)
        warped_flat = warped_z.reshape(-1)

        if valid_flat.sum() > 100:
            xv = z_flat[valid_flat]
            yv = warped_flat[valid_flat]
            sx = xv.sum()
            sxx = (xv * xv).sum()
            sxy = (xv * yv).sum()
            sy = yv.sum()
            nv = torch.tensor(float(xv.numel()), device=z.device, dtype=torch.float32)
            det = sxx * nv - sx * sx
            if det.abs() > 1e-12:
                s = (sxy * nv - sx * sy) / det
                t = (sxx * sy - sx * sxy) / det
                z_aligned = s * z_flat + t
            else:
                z_aligned = z_flat
        else:
            z_aligned = z_flat

        # ── Scene cut detection ──────────────────────────────────────────
        if valid_flat.sum() > 100:
            res = (z_aligned[valid_flat] - warped_flat[valid_flat]).abs().mean()
            if res > self.cut_threshold:
                self._prev_z = z.clone()
                self._prev_img = curr_img
                return True

        # ── Occlusion-aware blend ────────────────────────────────────────
        # Non-occluded: blend aligned current with warped prev.
        # Occluded: use raw current (no valid correspondence).
        # occ_full is (1,1,H,W) → flatten to (H*W,), then expand to (L*H*W,)
        # since z layout is layer-major: [layer0_all_pixels, layer1_all_pixels].
        occ_pixel = occ_full.squeeze().reshape(-1)  # (H*W,)
        occ_flat = occ_pixel.repeat(L)  # (L*H*W,)

        z_out = torch.empty_like(z_flat)
        # Where occluded: raw prediction.
        z_out[occ_flat] = z_flat[occ_flat]
        # Where visible: EMA blend.
        vis = ~occ_flat
        z_out[vis] = (self.alpha * z_aligned[vis]
                      + (1.0 - self.alpha) * warped_flat[vis])

        g_ndc.mean_vectors[:, 2] = z_out.to(g_ndc.mean_vectors.dtype)
        self._prev_z = z_out
        self._prev_img = curr_img
        return False

    @staticmethod
    def _warp_flow(flow: torch.Tensor, flow_ref: torch.Tensor) -> torch.Tensor:
        """Warp flow field using another flow field (for fb-consistency).

        Args:
            flow: (1, 2, H, W) flow to warp.
            flow_ref: (1, 2, H, W) reference flow defining sampling locations.
        Returns:
            Warped flow (1, 2, H, W).
        """
        _, _, H, W = flow.shape
        gy, gx = torch.meshgrid(
            torch.arange(H, device=flow.device, dtype=torch.float32),
            torch.arange(W, device=flow.device, dtype=torch.float32),
            indexing="ij",
        )
        sample_x = gx[None] + flow_ref[:, 0]
        sample_y = gy[None] + flow_ref[:, 1]
        sample_x = 2.0 * sample_x / (W - 1) - 1.0
        sample_y = 2.0 * sample_y / (H - 1) - 1.0
        grid = torch.stack([sample_x, sample_y], dim=-1)  # (1, H, W, 2)
        warped = F.grid_sample(flow, grid, mode="bilinear",
                               padding_mode="border", align_corners=True)
        return warped

    def _infer_shape(self, N: int) -> None:
        """Infer spatial shape (L, H, W) from total element count."""
        if N == 1536 * 1536 * 2:
            self._shape = (2, 1536, 1536)
        elif N == 1536 * 1536:
            self._shape = (1, 1536, 1536)
        else:
            self._shape = (1, 1, N)
