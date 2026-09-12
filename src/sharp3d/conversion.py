"""Shared video conversion engine (eliminates pipeline.py / worker.py duplication).

Both the CLI pipeline and GUI worker delegate their per-frame processing to
this module, providing their own predict function and I/O callbacks.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import torch
import torch.nn.functional as F

from .temporal import TemporalStabilizer, KalmanScalar
from .render import _compute_focus_depth_gpu, render_sbs
from .unproject import fast_unproject, INTERNAL_SHAPE
from .formats import pack as pack_stereo
from . import profiling


def _gaussian_grid(n: int) -> tuple[int, int] | None:
    """Infer (num_layers, side) of the pixel-aligned gaussian grid from N.

    SHARP's output is layer-major: [layer0 (side×side), layer1 (side×side)].
    The grid side is internal_resolution / 2 (768 for the released model),
    but we infer it from N so other checkpoints keep working.
    """
    for layers in (2, 1):
        if n % layers:
            continue
        side = math.isqrt(n // layers)
        if side * side * layers == n:
            return layers, side
    return None


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
        keyframe_interval: int = 1,
        renderer: str = "standard",
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
            keyframe_interval: Run the full SHARP prediction only every Nth
                frame; in-between frames reuse the keyframe geometry and only
                refresh gaussian colors from the current frame (~N× cheaper).
                1 = every frame (default, no reuse). A scene cut always
                forces a fresh keyframe.
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
        self._renderer = renderer

        self._stab = TemporalStabilizer(mode=stabilize_mode, device=device)
        self._conv_kf = KalmanScalar(q_pos=0.05, q_vel=0.02, r=0.15)
        self._eye4 = torch.eye(4, device=device)
        self._unproj = None  # lazily cached (4, 4) NDC→world matrix

        # ── Keyframe geometry reuse state ────────────────────────────
        self._kf_interval = max(1, int(keyframe_interval))
        self._kf_cut_threshold = 0.08  # mean |ΔsRGB| on pooled grid → cut
        self._kf_g = None            # Gaussians3D of the last keyframe (NDC)
        self._kf_age = 0             # frames rendered since last keyframe
        self._kf_focus = None        # keyframe focus depth (float)
        self._kf_pix = None          # (HW, 3) f32 sRGB pooled keyframe image
        self._kf_colors0 = None      # (N, 3) f32 sRGB keyframe colors (all layers)
        self._kf_grid = None         # (num_layers, side) of gaussian grid

    def _get_unprojection(self, ir) -> torch.Tensor:
        """(4, 4) NDC→world matrix; cached — intrinsics are fixed per video."""
        if self._unproj is None:
            from sharp.utils.gaussians import get_unprojection_matrix
            self._unproj = get_unprojection_matrix(
                self._eye4, ir, INTERNAL_SHAPE).detach()
        return self._unproj

    @torch.no_grad()
    def process_frame(self, img_r, df, ir, orig_size: tuple[int, int],
                      return_depth: bool = False,
                      return_gaussians: bool = False,
                      download: bool = True):
        """Process one prepared frame → stereo-packed frame.

        Args:
            img_r: (1, 3, 1536, 1536) float tensor.
            df: (1,) disparity factor tensor.
            ir: (4, 4) intrinsics scaled to internal resolution.
            orig_size: (W, H) of original frame.
            return_depth: Also return a depth map (H, W, 3) uint8 numpy array.
            return_gaussians: Also return the world-space Gaussians3D object.
            download: True → return a CPU numpy frame (blocking D2H copy).
                False → return the GPU uint8 tensor instead; the caller owns
                the D2H copy (e.g. pinned + async) and can overlap it with
                the next frame's compute.

        Returns:
            packed: (H_out, W_out, 3) uint8 numpy array (download=True) or
                    GPU uint8 tensor (download=False).
            depth_np: (H, W, 3) uint8 depth map (only if return_depth=True).
            gaussians: Gaussians3D (only if return_gaussians=True).
        """
        w, h = orig_size
        need_world = return_depth or return_gaussians
        prof = profiling.get_timer() if profiling.ENABLED else None

        # ── Keyframe reuse fast path ───────────────────────────────
        # Skips the whole SHARP prediction: reuse keyframe geometry, refresh
        # only the gaussian colors from the current frame. Falls through
        # to a full predict on interval expiry or scene cut.
        if self._kf_interval > 1 and not need_world and self._kf_g is not None:
            if prof:
                prof.frame_start()
                prof.mark("predict")
                prof.mark("stabilize")
            packed = self._try_reuse_keyframe(img_r, w, h)
            if packed is not None:
                if prof:
                    prof.mark("render+pack")
                if download:
                    result = packed.cpu().numpy()
                    del packed
                    return result
                return packed
            if prof:  # fell through to a full predict — discard the marks
                prof._frames.pop()

        # Predict + temporal stabilize (z + opacity + scale)
        if prof:
            prof.frame_start()
        with torch.autocast("cuda", dtype=torch.float16):
            g_ndc = self._predict(img_r, df)
        if prof:
            prof.mark("predict")
        scene_cut = self._stab.stabilize(g_ndc, img=img_r)
        if prof:
            prof.mark("stabilize")

        # Optional: soften depth edges to reduce disocclusion stretching
        if self._edge_soften:
            self._soften_depth_edges(g_ndc)

        # World-space gaussians are only needed for the optional depth/PLY
        # outputs. The main render folds the unprojection into the view
        # matrices instead (see render_sbs ndc_transform), skipping the
        # per-frame compose→transform→eigendecompose round-trip entirely.
        if need_world:
            g = fast_unproject(g_ndc, self._eye4, ir,
                               INTERNAL_SHAPE, decompose_method=self._decompose)
            focus = _compute_focus_depth_gpu(g.mean_vectors,
                                             q_focus=self._convergence_q)
        else:
            g = None
            U = self._get_unprojection(ir)
            from .temporal import _mean_view
            mv = _mean_view(g_ndc).float()
            # z_world = row 2 of the unprojection applied to NDC means
            z_world = mv @ U[2, :3] + U[2, 3]
            z_pos = z_world[z_world > 0]
            # Stride-sample before quantile: torch.quantile sorts every
            # element (1.18M per frame) just to pick one percentile, and the
            # Kalman filter below smooths the estimate anyway. A 1/16 sample
            # is statistically indistinguishable here and saves a few ms.
            if z_pos.numel() > 262_144:
                z_pos = z_pos[::16]
            focus = (max(2.0, float(torch.quantile(z_pos, self._convergence_q)))
                     if z_pos.numel() else 2.0)

        if scene_cut:
            self._conv_kf.reset()
        frame_conv = self._conv_kf.update(focus)

        # Cache this frame as the new keyframe for geometry reuse.
        if self._kf_interval > 1 and not need_world:
            self._store_keyframe(g_ndc, img_r, focus)

        # Render stereo pair + pack format
        if g is not None:
            sbs, _ = render_sbs(g, self._f_px, w, h,
                                ipd=self._ipd, convergence=frame_conv,
                                render_width=self._render_width,
                                renderer=self._renderer)
        else:
            sbs, _ = render_sbs(g_ndc, self._f_px, w, h,
                                ipd=self._ipd, convergence=frame_conv,
                                render_width=self._render_width,
                                ndc_transform=self._unproj,
                                renderer=self._renderer)
            if self._kf_g is not g_ndc:
                del g_ndc
        packed = pack_stereo(self._fmt, sbs)
        if prof:
            prof.mark("render+pack")

        # Optional depth map
        depth_np = None
        if return_depth:
            from .render import render_depth_map
            depth = render_depth_map(g, self._f_px, w, h)
            depth_np = depth.cpu().numpy()

        if download:
            # No torch.cuda.synchronize() here: packed.cpu() is a blocking D2H
            # copy ordered on the current stream, which already guarantees the
            # render finished. A device-wide sync would additionally wait for
            # the side-stream prefetch of the next frame and break the overlap.
            result = packed.cpu().numpy()
            del packed
        else:
            result = packed  # GPU tensor; caller handles the D2H copy
        del sbs

        if return_gaussians:
            return result, depth_np, g
        del g
        if return_depth:
            return result, depth_np
        return result

    # ── Keyframe geometry reuse ──────────────────────────────────

    def _pool_to_grid(self, img_r: torch.Tensor) -> torch.Tensor:
        """Average-pool the (1,3,1536,1536) input to the gaussian grid.

        Returns (side*side, 3) float32 sRGB, row-major — the same layout the
        initializer uses for the layer-0 base colors (avg_pool2d + flatten).
        """
        _, side = self._kf_grid
        k = INTERNAL_SHAPE[0] // side
        pix = img_r[0].float()
        if k > 1:
            pix = F.avg_pool2d(pix, k, k)
        return pix.permute(1, 2, 0).reshape(-1, 3)

    def _store_keyframe(self, g_ndc, img_r, focus: float) -> None:
        """Cache the freshly predicted frame as the reuse keyframe."""
        from sharp.utils.color_space import linearRGB2sRGB

        colors = g_ndc.colors
        cview = colors[0] if colors.ndim == 3 else colors  # (N, 3)
        grid = _gaussian_grid(cview.shape[0])
        if grid is None:  # unknown layout — disable reuse
            self._kf_g = None
            return
        self._kf_grid = grid
        self._kf_g = g_ndc
        self._kf_age = 0
        self._kf_focus = float(focus)
        # Gaussian colors live in linearRGB (composer color_space); keep the
        # keyframe snapshot in sRGB so the per-frame delta is perceptual.
        # All layers are cached: the initializer seeds every layer's base
        # color from the same pooled image (color_option="all_layers"), so
        # the pixel delta applies to the occluded layer too. Refreshing only
        # layer 0 leaves stale keyframe colors on layer 1, which shows
        # through the semi-transparent front layer as ghosting when the
        # camera moves.
        self._kf_colors0 = linearRGB2sRGB(cview.float().clamp(0.0, 1.0))
        self._kf_pix = self._pool_to_grid(img_r)

    def _try_reuse_keyframe(self, img_r, w: int, h: int):
        """Render the current frame from keyframe geometry + fresh colors.

        Returns the packed GPU tensor, or None when a full predict is due
        (interval expired or scene cut detected).
        """
        from sharp.utils.color_space import sRGB2linearRGB

        if self._kf_age >= self._kf_interval - 1 or self._kf_grid is None:
            return None

        pix = self._pool_to_grid(img_r)
        # Scene-cut guard: one tiny host sync on the pooled-grid residual.
        if float((pix - self._kf_pix).abs().mean()) > self._kf_cut_threshold:
            return None
        self._kf_age += 1

        # Refresh colors on every layer: apply the sRGB pixel delta on top
        # of the keyframe colors (preserves the model's learned color
        # offsets, including the occluded layer's inpainted content).
        layers = self._kf_grid[0]
        delta = pix - self._kf_pix
        if layers > 1:
            delta = delta.repeat(layers, 1)
        new_srgb = (self._kf_colors0 + delta).clamp_(0.0, 1.0)
        colors = self._kf_g.colors
        cview = colors[0] if colors.ndim == 3 else colors
        cview.copy_(sRGB2linearRGB(new_srgb).to(colors.dtype))

        frame_conv = self._conv_kf.update(self._kf_focus)
        sbs, _ = render_sbs(self._kf_g, self._f_px, w, h,
                            ipd=self._ipd, convergence=frame_conv,
                            render_width=self._render_width,
                            ndc_transform=self._unproj,
                            renderer=self._renderer)
        packed = pack_stereo(self._fmt, sbs)
        del sbs
        return packed

    def _soften_depth_edges(self, g_ndc) -> None:
        """Edge-aware depth smoothing to reduce disocclusion artifacts.

        Applies Gaussian blur to z only at depth discontinuities (high gradient),
        preserving flat regions unchanged. This softens the hard depth jump at
        object boundaries, reducing stretching/fringing in stereo rendering.
        """
        import torch.nn.functional as F

        from .temporal import _mean_view  # BUG#9: squeeze [1, N, 3] batch dim
        mv = _mean_view(g_ndc)
        z = mv[:, 2].float()
        N = z.numel()

        # Infer spatial layout (L, H, W). The gaussian grid side is
        # internal_resolution / 2 (768), not 1536 — the old hardcoded
        # 1536² check never matched, silently turning this into a no-op.
        grid = _gaussian_grid(N)
        if grid is None:
            return  # unknown layout, skip
        L, side = grid
        H = W = side

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
        mv[:, 2] = z_out.reshape(-1).to(mv.dtype)

    def reset(self) -> None:
        """Reset temporal state (call between videos in batch mode)."""
        self._stab.reset()
        self._conv_kf.reset()
        self._kf_g = None
        self._kf_age = 0
        self._kf_focus = None
        self._kf_pix = None
        self._kf_colors0 = None
        self._kf_grid = None
