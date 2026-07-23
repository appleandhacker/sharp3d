"""Shared video conversion engine (eliminates pipeline.py / worker.py duplication).

Both the CLI pipeline and GUI worker delegate their per-frame processing to
this module, providing their own predict function and I/O callbacks.
"""

from __future__ import annotations

import time
from typing import Callable

import torch

from .temporal import TemporalStabilizer, KalmanScalar
from .render import _compute_focus_depth_gpu, render_sbs
from .unproject import fast_unproject, INTERNAL_SHAPE
from .formats import pack as pack_stereo


class VideoConversionEngine:
    """Per-frame stereo conversion with temporal stabilization.

    Encapsulates: predict → stabilize → unproject → convergence Kalman → render → pack.
    Callers handle I/O (decode, encode) and pass prepared frames in.

    Usage:
        engine = VideoConversionEngine(predict_fn, device, f_px, ...)
        for each frame:
            packed_np = engine.process_frame(img_r, df, ir, (w, h))
            writer.append_frame(packed_np)
    """

    def __init__(
        self,
        predict_fn: Callable,
        device: torch.device,
        f_px: float,
        fmt: str = "full_sbs",
        ipd: float = 0.063,
        convergence_q: float | None = None,
        decompose_method: str = "analytical",
        stabilize_mode: str = "adaptive",
        render_width: int | None = None,
        edge_soften: bool = False,
    ):
        """
        Args:
            predict_fn: Callable(img_r, df) → Gaussians3D in NDC.
            device: CUDA device.
            f_px: Focal length in pixels (source image space).
            fmt: Stereo packing format key.
            ipd: Inter-pupillary distance in scene units.
            convergence_q: Quantile for convergence plane (0-1). None or 0 = auto (0.50).
                           E.g., 0.3 = 30% of geometry pops out.
            decompose_method: "analytical" or "svd".
            stabilize_mode: "off", "global", "adaptive", or "flow".
            render_width: Per-eye render width (None = source resolution).
        """
        self._predict = predict_fn
        self._device = device
        self._f_px = f_px
        self._fmt = fmt
        self._ipd = ipd
        self._convergence_q = convergence_q if convergence_q else 0.50
        self._decompose = decompose_method
        self._render_width = render_width
        self._edge_soften = edge_soften

        self._stab = TemporalStabilizer(mode=stabilize_mode, device=device)
        self._conv_kf = KalmanScalar(q_pos=0.05, q_vel=0.02, r=0.15)
        self._eye4 = torch.eye(4, device=device)

    @torch.no_grad()
    def process_frame(self, img_r, df, ir, orig_size: tuple[int, int],
                      return_depth: bool = False,
                      return_gaussians: bool = False):
        """Process one prepared frame → stereo-packed numpy array.

        Args:
            img_r: (1, 3, 1536, 1536) float tensor.
            df: (1,) disparity factor tensor.
            ir: (4, 4) intrinsics scaled to internal resolution.
            orig_size: (W, H) of original frame.
            return_depth: Also return a depth map (H, W, 3) uint8 numpy array.
            return_gaussians: Also return the world-space Gaussians3D object.

        Returns:
            packed_np: (H_out, W_out, 3) uint8 numpy array.
            depth_np: (H, W, 3) uint8 depth map (only if return_depth=True).
            gaussians: Gaussians3D (only if return_gaussians=True).
        """
        w, h = orig_size

        # Predict + temporal stabilize (z + opacity + scale)
        with torch.autocast("cuda", dtype=torch.float16):
            g_ndc = self._predict(img_r, df)
        scene_cut = self._stab.stabilize(g_ndc, img=img_r)

        # Optional: soften depth edges to reduce disocclusion stretching
        if self._edge_soften:
            self._soften_depth_edges(g_ndc)

        # Unproject NDC → world
        g = fast_unproject(g_ndc, self._eye4, ir,
                           INTERNAL_SHAPE, decompose_method=self._decompose)
        del g_ndc

        # Convergence: compute focus depth at specified quantile + Kalman smooth
        focus = _compute_focus_depth_gpu(g.mean_vectors,
                                         q_focus=self._convergence_q)
        if scene_cut:
            self._conv_kf.reset()
        frame_conv = self._conv_kf.update(focus)

        # Render stereo pair + pack format
        sbs, _ = render_sbs(g, self._f_px, w, h,
                            ipd=self._ipd, convergence=frame_conv,
                            render_width=self._render_width)
        packed = pack_stereo(self._fmt, sbs)

        # Optional depth map
        depth_np = None
        if return_depth:
            from .render import render_depth_map
            depth = render_depth_map(g, self._f_px, w, h)
            depth_np = depth.cpu().numpy()

        torch.cuda.synchronize()
        result = packed.cpu().numpy()
        del sbs, packed

        if return_gaussians:
            return result, depth_np, g
        del g
        if return_depth:
            return result, depth_np
        return result

    def _soften_depth_edges(self, g_ndc) -> None:
        """Edge-aware depth smoothing to reduce disocclusion artifacts.

        Applies Gaussian blur to z only at depth discontinuities (high gradient),
        preserving flat regions unchanged. This softens the hard depth jump at
        object boundaries, reducing stretching/fringing in stereo rendering.
        """
        import torch.nn.functional as F

        z = g_ndc.mean_vectors[:, 2].float()
        N = z.numel()

        # Infer spatial layout (L, H, W)
        if N == 1536 * 1536 * 2:
            L, H, W = 2, 1536, 1536
        elif N == 1536 * 1536:
            L, H, W = 1, 1536, 1536
        else:
            return  # unknown layout, skip

        z_map = z.reshape(L, H, W)

        # Compute gradient magnitude (Sobel-like)
        z_4d = z_map.unsqueeze(0)  # (1, L, H, W)
        z_pad = F.pad(z_4d, [1, 1, 1, 1], mode="replicate")
        gx = z_pad[:, :, 1:-1, 2:] - z_pad[:, :, 1:-1, :-2]  # horizontal
        gy = z_pad[:, :, 2:, 1:-1] - z_pad[:, :, :-2, 1:-1]  # vertical
        grad_mag = (gx.pow(2) + gy.pow(2)).sqrt().squeeze(0)  # (L, H, W)

        # Edge weight: sigmoid ramp around gradient threshold
        edge_weight = torch.sigmoid((grad_mag - 0.02) * 200.0)  # soft mask

        # Gaussian blur (5x5, sigma=2) — depthwise conv (groups=L)
        kernel_size = 5
        sigma = 2.0
        coords = torch.arange(kernel_size, device=z.device, dtype=torch.float32) - kernel_size // 2
        kernel_1d = torch.exp(-coords.pow(2) / (2 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]  # (5, 5)
        kernel_2d = kernel_2d.expand(L, 1, -1, -1)  # (L, 1, 5, 5)

        z_blur = F.conv2d(z_4d, kernel_2d, padding=2,
                          groups=L).squeeze(0)  # (L, H, W)

        # Blend: at edges use blurred, elsewhere keep original
        z_out = edge_weight * z_blur + (1.0 - edge_weight) * z_map
        g_ndc.mean_vectors[:, 2] = z_out.reshape(-1).to(g_ndc.mean_vectors.dtype)

    def reset(self) -> None:
        """Reset temporal state (call between videos in batch mode)."""
        self._stab.reset()
        self._conv_kf.reset()
